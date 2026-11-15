//! Real voice-note audio for HEY•AI: WAV validation, audio-note
//! manifests, and playback through the Kernelodys engine.
//!
//! Supported audio: 16-bit PCM WAV, mono or stereo, sample rates
//! 8–48 kHz. The WAV parsing itself lives in Kernelodys
//! ([`kernelodys::wav::decode_wav`]); this module adds the sample
//! rate range check, the declared-duration check, and the
//! language-tagged manifest.
//!
//! A note's declared whole-second duration must equal the audio's
//! true duration (frames / sample rate) rounded to the nearest
//! second (halves round up). A note whose declared duration does
//! not match its audio is refused at ingest and again on read.
//!
//! Playback runs through the Kernelodys [`PlaybackEngine`], so the
//! rendered audio is the 432 Hz retune of the note: pitch about
//! 1.85% lower and duration about 1.85% longer than the stored
//! audio. The stored bytes are never changed. Playback renders to
//! WAV bytes in memory; no audio device is opened, and nothing here
//! records audio or transcribes speech.

use kernel_crypto::Ed25519Signature;
use kernel_identity::{Identity, PeerId};
use kernel_storage::Cid;
use kernelodys::playback::{AudioSink, PlaybackEngine, PlaybackError};
use kernelodys::wav::decode_wav;

use crate::{fingerprint_name, validate_duration, validate_name, VoiceError};

/// Lowest supported sample rate, in Hz.
pub const MIN_SAMPLE_RATE: u32 = 8_000;
/// Highest supported sample rate, in Hz.
pub const MAX_SAMPLE_RATE: u32 = 48_000;
/// Longest language tag, in characters.
pub const MAX_LANGUAGE_LEN: usize = 16;

/// Magic bytes at the start of every audio note manifest.
const AUDIO_NOTE_MAGIC: &[u8; 7] = b"KVOICE2";
/// Domain separator for audio note manifest signatures.
const AUDIO_NOTE_SIGN_DOMAIN: &[u8] = b"KVOICE2";

/// True if a sample rate is inside the supported 8–48 kHz range.
fn sample_rate_supported(sample_rate: u32) -> bool {
    (MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&sample_rate)
}

/// Rules for language tags: a BCP-47-style tag like `en` or `en-US`
/// — 1–16 characters, ASCII letters, digits, and `-`, starting with
/// a letter, no leading/trailing `-`, no `--`.
pub fn validate_language(language: &str) -> Result<(), VoiceError> {
    if language.is_empty() {
        return Err(VoiceError::InvalidAudio(
            "language tag must not be empty".to_string(),
        ));
    }
    if language.len() > MAX_LANGUAGE_LEN {
        return Err(VoiceError::InvalidAudio(format!(
            "language tag must be at most {MAX_LANGUAGE_LEN} characters"
        )));
    }
    if !language
        .chars()
        .next()
        .map(|c| c.is_ascii_alphabetic())
        .unwrap_or(false)
    {
        return Err(VoiceError::InvalidAudio(
            "language tag must start with a letter".to_string(),
        ));
    }
    if !language
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-')
    {
        return Err(VoiceError::InvalidAudio(
            "language tag may only contain letters, digits, and '-'".to_string(),
        ));
    }
    if language.ends_with('-') || language.contains("--") {
        return Err(VoiceError::InvalidAudio(
            "language tag must not end with '-' or contain '--'".to_string(),
        ));
    }
    Ok(())
}

/// What validation learned about a WAV: the facts a manifest
/// records, and what the duration check compares against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WavInfo {
    /// Samples per second per channel.
    pub sample_rate: u32,
    /// 1 for mono, 2 for stereo.
    pub channels: u16,
    /// Number of frames (one frame is one sample per channel).
    pub frames: usize,
}

impl WavInfo {
    /// True duration in seconds, from the frame count and rate.
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.frames as f64 / self.sample_rate as f64
        }
    }

    /// True duration rounded to the nearest whole second (halves
    /// round up). This is what a manifest's declared duration must
    /// equal.
    pub fn duration_rounded_secs(&self) -> u64 {
        self.duration_secs().round() as u64
    }
}

/// Validate WAV bytes: parse the header with the Kernelodys decoder
/// (RIFF/WAVE, PCM, 16-bit, mono/stereo, whole frames, no
/// truncation) and refuse sample rates outside 8–48 kHz. Returns
/// the audio's facts. Every failure is a [`VoiceError::InvalidAudio`],
/// never a panic.
pub fn validate_wav(bytes: &[u8]) -> Result<WavInfo, VoiceError> {
    let audio = decode_wav(bytes).map_err(|e| VoiceError::InvalidAudio(e.to_string()))?;
    if !sample_rate_supported(audio.sample_rate) {
        return Err(VoiceError::InvalidAudio(format!(
            "sample rate {} Hz is outside the supported 8-48 kHz range",
            audio.sample_rate
        )));
    }
    Ok(WavInfo {
        sample_rate: audio.sample_rate,
        channels: audio.channels,
        frames: audio.frames(),
    })
}

/// Check a manifest's declared whole-second duration against the
/// audio it claims to describe: it must equal the true duration
/// rounded to the nearest second. Anything else is refused.
pub fn check_duration_match(declared_secs: u64, info: &WavInfo) -> Result<(), VoiceError> {
    let actual = info.duration_rounded_secs();
    if declared_secs != actual {
        return Err(VoiceError::InvalidDuration(format!(
            "declared duration {declared_secs}s does not match the audio ({:.3}s, {actual}s rounded)",
            info.duration_secs()
        )));
    }
    Ok(())
}

/// Encode interleaved 16-bit PCM samples as a 44-byte-header WAV.
/// Mono or stereo only; the byte count must fit a RIFF header.
pub fn encode_wav(samples: &[i16], sample_rate: u32, channels: u16) -> Result<Vec<u8>, VoiceError> {
    if sample_rate == 0 {
        return Err(VoiceError::InvalidAudio(
            "sample rate must not be zero".to_string(),
        ));
    }
    if channels != 1 && channels != 2 {
        return Err(VoiceError::InvalidAudio(format!(
            "only mono and stereo are supported, got {channels} channels"
        )));
    }
    if !samples.len().is_multiple_of(channels as usize) {
        return Err(VoiceError::InvalidAudio(format!(
            "{} samples are not a whole number of {channels}-channel frames",
            samples.len()
        )));
    }
    let data_bytes = samples.len() as u64 * 2;
    if data_bytes > (u32::MAX - 36) as u64 {
        return Err(VoiceError::InvalidAudio(
            "WAV data too large for a RIFF header".to_string(),
        ));
    }
    let byte_rate = sample_rate as u64 * channels as u64 * 2;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36u32 + data_bytes as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(byte_rate as u32).to_le_bytes());
    out.extend_from_slice(&(channels * 2).to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    Ok(out)
}

/// An [`AudioSink`] that collects samples in memory instead of
/// writing a file, so playback renders to bytes the caller can hold.
struct MemSink {
    rate: u32,
    channels: u16,
    samples: Vec<i16>,
}

impl AudioSink for MemSink {
    fn write(&mut self, samples: &[i16]) -> Result<(), PlaybackError> {
        if !samples.len().is_multiple_of(self.channels as usize) {
            return Err(PlaybackError::InvalidState(format!(
                "{} samples are not a whole number of {}-channel frames",
                samples.len(),
                self.channels
            )));
        }
        self.samples.extend_from_slice(samples);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), PlaybackError> {
        Ok(())
    }

    fn sample_rate(&self) -> u32 {
        self.rate
    }
}

/// Play WAV bytes through the Kernelodys engine and return the
/// rendered WAV bytes. The bytes are validated first (see
/// [`validate_wav`]); playback is offline (no real-time pacing) and
/// the output is the engine's 432 Hz retune of the note at the same
/// sample rate and channel count.
pub fn play_wav_bytes(wav: &[u8]) -> Result<Vec<u8>, VoiceError> {
    let info = validate_wav(wav)?;
    let mut engine =
        PlaybackEngine::from_wav(wav).map_err(|e| VoiceError::Playback(e.to_string()))?;
    engine.set_realtime(false);
    let mut sink = MemSink {
        rate: info.sample_rate,
        channels: info.channels,
        samples: Vec::new(),
    };
    let report = engine
        .play(&mut sink)
        .map_err(|e| VoiceError::Playback(e.to_string()))?;
    if !report.finished {
        return Err(VoiceError::Playback(
            "playback stopped before the note finished".to_string(),
        ));
    }
    encode_wav(&sink.samples, info.sample_rate, info.channels)
}

/// The message an owner signs for an audio note manifest.
fn audio_note_sign_message(
    name: &str,
    duration_secs: u64,
    sample_rate: u32,
    channels: u16,
    language: &str,
    root: &Cid,
    owner: &PeerId,
) -> Vec<u8> {
    let mut msg = Vec::with_capacity(
        AUDIO_NOTE_SIGN_DOMAIN.len() + name.len() + 8 + 4 + 2 + 2 + language.len() + 32 + 32,
    );
    msg.extend_from_slice(AUDIO_NOTE_SIGN_DOMAIN);
    msg.extend_from_slice(name.as_bytes());
    msg.extend_from_slice(&duration_secs.to_le_bytes());
    msg.extend_from_slice(&sample_rate.to_le_bytes());
    msg.extend_from_slice(&channels.to_le_bytes());
    msg.extend_from_slice(&(language.len() as u16).to_le_bytes());
    msg.extend_from_slice(language.as_bytes());
    msg.extend_from_slice(root.as_bytes());
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg
}

/// A signed audio note manifest: name, duration, sample rate,
/// channels, language tag, chunked-content root, owner, signature.
/// The signature covers every field, so the metadata is verified
/// whenever the manifest is verified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioNoteManifest {
    /// Human-readable voice note name.
    pub name: String,
    /// Voice note duration in seconds (must match the audio, see
    /// [`check_duration_match`]).
    pub duration_secs: u64,
    /// Sample rate in Hz (8–48 kHz).
    pub sample_rate: u32,
    /// Channel count (1 mono, 2 stereo).
    pub channels: u16,
    /// Language tag, e.g. `en` or `en-US`.
    pub language: String,
    /// The owner who signed this manifest.
    pub owner: PeerId,
    /// Root CID of the chunked audio content (a `KCHUNK1` manifest block).
    pub root: Cid,
    /// The owner's signature over name, duration, sample rate,
    /// channels, language, root, and owner key.
    pub signature: Ed25519Signature,
}

impl AudioNoteManifest {
    /// Build and sign a manifest as `identity`. Validates the name,
    /// duration, language, sample rate, and channel count — but not
    /// the audio itself: the caller (see `VoiceStore`) checks the
    /// declared duration against the WAV bytes before calling this.
    pub fn new(
        name: &str,
        duration_secs: u64,
        sample_rate: u32,
        channels: u16,
        language: &str,
        root: Cid,
        identity: &Identity,
    ) -> Result<Self, VoiceError> {
        validate_name(name)?;
        validate_duration(duration_secs)?;
        validate_language(language)?;
        if !sample_rate_supported(sample_rate) {
            return Err(VoiceError::InvalidAudio(format!(
                "sample rate {sample_rate} Hz is outside the supported 8-48 kHz range"
            )));
        }
        if channels != 1 && channels != 2 {
            return Err(VoiceError::InvalidAudio(format!(
                "only mono and stereo are supported, got {channels} channels"
            )));
        }
        let owner = identity.peer_id().clone();
        let msg = audio_note_sign_message(
            name,
            duration_secs,
            sample_rate,
            channels,
            language,
            &root,
            &owner,
        );
        let signature = identity.sign(&msg);
        Ok(Self {
            name: name.to_string(),
            duration_secs,
            sample_rate,
            channels,
            language: language.to_string(),
            owner,
            root,
            signature,
        })
    }

    /// True if the signature verifies against the owner for these
    /// exact fields. Any tampering — name, duration, rate, channels,
    /// language, root — fails this check.
    pub fn verify(&self) -> bool {
        let msg = audio_note_sign_message(
            &self.name,
            self.duration_secs,
            self.sample_rate,
            self.channels,
            &self.language,
            &self.root,
            &self.owner,
        );
        self.owner.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary manifest format.
    pub fn encode(&self) -> Vec<u8> {
        let name_bytes = self.name.as_bytes();
        let lang_bytes = self.language.as_bytes();
        let mut out = Vec::with_capacity(
            7 + 2 + name_bytes.len() + 8 + 4 + 2 + 2 + lang_bytes.len() + 32 + 32 + 64,
        );
        out.extend_from_slice(AUDIO_NOTE_MAGIC);
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(&self.duration_secs.to_le_bytes());
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&self.channels.to_le_bytes());
        out.extend_from_slice(&(lang_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(lang_bytes);
        out.extend_from_slice(self.root.as_bytes());
        out.extend_from_slice(self.owner.public_key().as_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a manifest produced by [`AudioNoteManifest::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, VoiceError> {
        if bytes.len() < 7 + 2 + 8 + 4 + 2 + 2 + 32 + 32 + 64 || &bytes[..7] != AUDIO_NOTE_MAGIC {
            return Err(VoiceError::InvalidManifest(
                "missing KVOICE2 magic".to_string(),
            ));
        }
        let name_len = u16::from_le_bytes(bytes[7..9].try_into().expect("2-byte slice")) as usize;
        let mut off = 9 + name_len;
        // duration (8) + sample rate (4) + channels (2) + lang len (2)
        if bytes.len() < off + 16 {
            return Err(VoiceError::InvalidManifest(
                "audio note manifest is truncated".to_string(),
            ));
        }
        let duration_secs =
            u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let sample_rate = u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4-byte slice"));
        off += 4;
        let channels = u16::from_le_bytes(bytes[off..off + 2].try_into().expect("2-byte slice"));
        off += 2;
        let lang_len =
            u16::from_le_bytes(bytes[off..off + 2].try_into().expect("2-byte slice")) as usize;
        off += 2;
        let expected = off + lang_len + 32 + 32 + 64;
        if bytes.len() != expected {
            return Err(VoiceError::InvalidManifest(format!(
                "manifest is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let name = String::from_utf8(bytes[9..9 + name_len].to_vec()).map_err(|_| {
            VoiceError::InvalidManifest("manifest name is not valid UTF-8".to_string())
        })?;
        validate_name(&name)?;
        validate_duration(duration_secs)?;
        if !sample_rate_supported(sample_rate) {
            return Err(VoiceError::InvalidManifest(format!(
                "manifest sample rate {sample_rate} Hz is outside 8-48 kHz"
            )));
        }
        if channels != 1 && channels != 2 {
            return Err(VoiceError::InvalidManifest(format!(
                "manifest channel count {channels} is not 1 or 2"
            )));
        }
        let language = String::from_utf8(bytes[off..off + lang_len].to_vec()).map_err(|_| {
            VoiceError::InvalidManifest("manifest language is not valid UTF-8".to_string())
        })?;
        validate_language(&language)?;
        off += lang_len;
        let mut root_bytes = [0u8; 32];
        root_bytes.copy_from_slice(&bytes[off..off + 32]);
        let root = Cid::from_bytes(root_bytes);
        off += 32;
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let owner_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| VoiceError::InvalidManifest(e.to_string()))?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let owner = PeerId::new(owner_key, &fingerprint_name(&owner_key))?;
        Ok(Self {
            name,
            duration_secs,
            sample_rate,
            channels,
            language,
            owner,
            root,
            signature,
        })
    }
}

/// The metadata of an audio note, as a human sees it — no CIDs, no
/// hashes. Every field is carried on the signed manifest and
/// verified on read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioNoteMetadata {
    /// Human-readable voice note name.
    pub name: String,
    /// Voice note duration in seconds.
    pub duration_secs: u64,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel count (1 mono, 2 stereo).
    pub channels: u16,
    /// Language tag, e.g. `en`.
    pub language: String,
    /// Who owns (recorded) this voice note.
    pub owner: PeerId,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kernelodys::wav::play_wav_432;

    /// Build a minimal 16-bit PCM WAV in memory with an explicit
    /// format tag / bit depth / channel count, so unsupported formats
    /// can be constructed too.
    fn build_wav_ex(
        channels: u16,
        sample_rate: u32,
        format_tag: u16,
        bits: u16,
        samples: &[i16],
    ) -> Vec<u8> {
        let block_align = channels * (bits / 8);
        let byte_rate = sample_rate * block_align as u32;
        let mut fmt = Vec::with_capacity(16);
        fmt.extend_from_slice(&format_tag.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&sample_rate.to_le_bytes());
        fmt.extend_from_slice(&byte_rate.to_le_bytes());
        fmt.extend_from_slice(&block_align.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        let mut data = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        let mut chunks = Vec::new();
        let push_chunk = |id: &[u8; 4], payload: &[u8], chunks: &mut Vec<u8>| {
            chunks.extend_from_slice(id);
            chunks.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            chunks.extend_from_slice(payload);
            if payload.len() % 2 == 1 {
                chunks.push(0);
            }
        };
        push_chunk(b"fmt ", &fmt, &mut chunks);
        push_chunk(b"data", &data, &mut chunks);
        let mut out = Vec::with_capacity(12 + chunks.len());
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&((4 + chunks.len()) as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(&chunks);
        out
    }

    fn build_wav(channels: u16, sample_rate: u32, samples: &[i16]) -> Vec<u8> {
        build_wav_ex(channels, sample_rate, 1, 16, samples)
    }

    fn sine_i16(freq_hz: f64, sample_rate: u32, secs: f64) -> Vec<i16> {
        let n = (sample_rate as f64 * secs) as usize;
        (0..n)
            .map(|i| {
                (20000.0
                    * (2.0 * std::f64::consts::PI * freq_hz * i as f64 / sample_rate as f64).sin())
                    as i16
            })
            .collect()
    }

    /// Dominant frequency of a mono i16 sine, from zero crossings.
    fn measured_freq(samples: &[i16], sample_rate: u32) -> f64 {
        let mut crossings = 0usize;
        for w in samples.windows(2) {
            if (w[0] < 0 && w[1] >= 0) || (w[0] >= 0 && w[1] < 0) {
                crossings += 1;
            }
        }
        let seconds = samples.len() as f64 / sample_rate as f64;
        crossings as f64 / 2.0 / seconds
    }

    #[test]
    fn wav_validation_accepts_supported_notes() {
        // Mono, lowest rate, exactly 1 second.
        let one_sec = vec![0i16; 8000];
        let info = validate_wav(&build_wav(1, 8000, &one_sec)).unwrap();
        assert_eq!(info.sample_rate, 8000);
        assert_eq!(info.channels, 1);
        assert_eq!(info.frames, 8000);
        assert!((info.duration_secs() - 1.0).abs() < 1e-9);
        assert_eq!(info.duration_rounded_secs(), 1);
        // Stereo, highest rate, 2 seconds.
        let two_sec = vec![0i16; 48_000 * 2 * 2];
        let info = validate_wav(&build_wav(2, 48_000, &two_sec)).unwrap();
        assert_eq!(info.sample_rate, 48_000);
        assert_eq!(info.channels, 2);
        assert_eq!(info.frames, 96_000);
        assert_eq!(info.duration_rounded_secs(), 2);
        // A 1.5 s note rounds to 2.
        let half = sine_i16(440.0, 8000, 1.5);
        let info = validate_wav(&build_wav(1, 8000, &half)).unwrap();
        assert_eq!(info.duration_rounded_secs(), 2);
    }

    #[test]
    fn wav_validation_refuses_non_wav() {
        assert!(matches!(
            validate_wav(b""),
            Err(VoiceError::InvalidAudio(_))
        ));
        assert!(matches!(
            validate_wav(b"this is not a wav file at all"),
            Err(VoiceError::InvalidAudio(_))
        ));
        assert!(matches!(
            validate_wav(b"fake voice bytes"),
            Err(VoiceError::InvalidAudio(_))
        ));
        // Truncated: a real header cut short.
        let good = build_wav(1, 8000, &[1, 2, 3, 4]);
        assert!(matches!(
            validate_wav(&good[..20]),
            Err(VoiceError::InvalidAudio(_))
        ));
    }

    #[test]
    fn wav_validation_refuses_bad_sample_rates() {
        let samples = vec![0i16; 8000];
        for rate in [4000, 7999, 48_001, 96_000] {
            assert!(
                matches!(
                    validate_wav(&build_wav(1, rate, &samples)),
                    Err(VoiceError::InvalidAudio(_))
                ),
                "rate {rate} should be refused"
            );
        }
        // The bounds themselves are accepted.
        assert!(validate_wav(&build_wav(1, 8000, &samples)).is_ok());
        assert!(validate_wav(&build_wav(1, 48_000, &samples)).is_ok());
    }

    #[test]
    fn wav_validation_refuses_unsupported_formats() {
        let samples = vec![0i16; 100];
        // Float (format tag 3).
        assert!(matches!(
            validate_wav(&build_wav_ex(1, 8000, 3, 16, &samples)),
            Err(VoiceError::InvalidAudio(_))
        ));
        // 8-bit samples.
        assert!(matches!(
            validate_wav(&build_wav_ex(1, 8000, 1, 8, &samples)),
            Err(VoiceError::InvalidAudio(_))
        ));
        // 24-bit samples.
        assert!(matches!(
            validate_wav(&build_wav_ex(1, 8000, 1, 24, &samples)),
            Err(VoiceError::InvalidAudio(_))
        ));
        // Three channels.
        assert!(matches!(
            validate_wav(&build_wav_ex(3, 8000, 1, 16, &samples)),
            Err(VoiceError::InvalidAudio(_))
        ));
    }

    #[test]
    fn language_tag_rules() {
        for good in ["en", "en-US", "zh-Hans", "pt-BR", "a1"] {
            assert!(validate_language(good).is_ok(), "{good} should be accepted");
        }
        for bad in [
            "",
            "e n",
            "1en",
            "-en",
            "en-",
            "en--US",
            "toolonglanguage01",
        ] {
            assert!(validate_language(bad).is_err(), "{bad} should be refused");
        }
    }

    #[test]
    fn duration_match_rule() {
        let two_sec = sine_i16(440.0, 8000, 2.0);
        let info = validate_wav(&build_wav(1, 8000, &two_sec)).unwrap();
        assert!(check_duration_match(2, &info).is_ok());
        assert!(matches!(
            check_duration_match(3, &info),
            Err(VoiceError::InvalidDuration(_))
        ));
        assert!(matches!(
            check_duration_match(1, &info),
            Err(VoiceError::InvalidDuration(_))
        ));
        // 1.5 s rounds to 2, not 1.
        let half = sine_i16(440.0, 8000, 1.5);
        let info = validate_wav(&build_wav(1, 8000, &half)).unwrap();
        assert!(check_duration_match(2, &info).is_ok());
        assert!(check_duration_match(1, &info).is_err());
    }

    #[test]
    fn audio_manifest_round_trip_and_tamper() {
        let alice = Identity::generate("alice").unwrap();
        let root = Cid::from_bytes(kernel_crypto::sha256(b"audio"));
        let manifest =
            AudioNoteManifest::new("memo", 12, 44_100, 2, "en-US", root, &alice).unwrap();
        assert!(manifest.verify());
        let bytes = manifest.encode();
        let decoded = AudioNoteManifest::decode(&bytes).unwrap();
        // Every signed field survives the round trip. The owner's
        // human display name does not: decode recovers the key, not
        // the label, so the fingerprint name replaces "alice" (the
        // same property the KVOICE1 manifest has).
        assert_eq!(decoded.name, manifest.name);
        assert_eq!(decoded.duration_secs, manifest.duration_secs);
        assert_eq!(decoded.sample_rate, manifest.sample_rate);
        assert_eq!(decoded.channels, manifest.channels);
        assert_eq!(decoded.language, manifest.language);
        assert_eq!(decoded.root, manifest.root);
        assert_eq!(decoded.signature, manifest.signature);
        assert_eq!(decoded.owner.public_key(), manifest.owner.public_key());
        let expected_name = format!(
            "peer-{}",
            &kernel_crypto::to_hex(manifest.owner.public_key().as_bytes())[..8]
        );
        assert_eq!(decoded.owner.display_name(), expected_name);
        assert!(decoded.verify());
        // Tamper with the language: signature must fail.
        let mut tampered = manifest.clone();
        tampered.language = "fr".to_string();
        assert!(!tampered.verify());
        // Tamper with the sample rate: signature must fail.
        let mut tampered = manifest.clone();
        tampered.sample_rate = 8000;
        assert!(!tampered.verify());
        // Truncated and garbage encodings are refused.
        assert!(AudioNoteManifest::decode(&bytes[..bytes.len() - 1]).is_err());
        assert!(AudioNoteManifest::decode(b"garbage").is_err());
        // Bad manifest fields are refused at construction.
        assert!(AudioNoteManifest::new("memo", 12, 4000, 1, "en", root, &alice).is_err());
        assert!(AudioNoteManifest::new("memo", 12, 8000, 3, "en", root, &alice).is_err());
        assert!(AudioNoteManifest::new("memo", 12, 8000, 1, "", root, &alice).is_err());
    }

    #[test]
    fn encode_wav_round_trips_through_kernelodys_decoder() {
        let samples = [0i16, 1000, -1000, 32767, -32768, 12345];
        let bytes = encode_wav(&samples, 44_100, 1).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        let audio = decode_wav(&bytes).unwrap();
        assert_eq!(audio.samples, samples);
        assert_eq!(audio.sample_rate, 44_100);
        assert_eq!(audio.channels, 1);
        // Stereo.
        let stereo = [100i16, -100, 200, -200];
        let bytes = encode_wav(&stereo, 48_000, 2).unwrap();
        let audio = decode_wav(&bytes).unwrap();
        assert_eq!(audio.samples, stereo);
        assert_eq!(audio.channels, 2);
        // Bad inputs are refused.
        assert!(encode_wav(&[1, 2, 3], 8000, 2).is_err());
        assert!(encode_wav(&samples, 0, 1).is_err());
        assert!(encode_wav(&samples, 8000, 3).is_err());
    }

    #[test]
    fn play_wav_bytes_renders_the_432_retune() {
        let sample_rate = 44_100u32;
        let samples = sine_i16(440.0, sample_rate, 1.0);
        let wav = build_wav(1, sample_rate, &samples);
        let played = play_wav_bytes(&wav).unwrap();
        let audio = decode_wav(&played).unwrap();
        assert_eq!(audio.sample_rate, sample_rate);
        assert_eq!(audio.channels, 1);
        // Byte-identical to the direct retune, retune aside: the
        // engine output matches play_wav_432 sample for sample.
        let direct = play_wav_432(&wav).unwrap().retuned.samples;
        assert_eq!(audio.samples, direct);
        // The rendered file measures ~432 Hz, not 440.
        let got = measured_freq(&audio.samples, sample_rate);
        assert!(
            (got - 432.0).abs() < 2.0,
            "rendered sine measured at {got} Hz, expected ~432 Hz"
        );
        // Stereo stays stereo.
        let left = sine_i16(440.0, sample_rate, 0.5);
        let right = sine_i16(220.0, sample_rate, 0.5);
        let mut interleaved = Vec::with_capacity(left.len() * 2);
        for i in 0..left.len() {
            interleaved.push(left[i]);
            interleaved.push(right[i]);
        }
        let stereo_wav = build_wav(2, sample_rate, &interleaved);
        let played = play_wav_bytes(&stereo_wav).unwrap();
        let audio = decode_wav(&played).unwrap();
        assert_eq!(audio.channels, 2);
        let direct = play_wav_432(&stereo_wav).unwrap().retuned.samples;
        assert_eq!(audio.samples, direct);
    }

    #[test]
    fn play_wav_bytes_refuses_non_wav() {
        assert!(matches!(
            play_wav_bytes(b"not audio at all"),
            Err(VoiceError::InvalidAudio(_))
        ));
        assert!(matches!(
            play_wav_bytes(b""),
            Err(VoiceError::InvalidAudio(_))
        ));
    }
}
