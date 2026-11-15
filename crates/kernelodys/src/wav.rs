//! Decode 16-bit PCM WAV bytes held in memory, and play them at 432 Hz.
//!
//! The parser is written by hand — no WAV crate, no new dependencies.
//! It reads the RIFF/WAVE header, the `fmt ` chunk (PCM only), and the
//! `data` chunk, skipping unknown chunks (e.g. `LIST`) on the way.
//! Anything malformed is refused with a [`WavError`]; nothing here
//! panics on bad input.
//!
//! Decoded audio is routed through [`crate::tuning`]: [`play_wav_432`]
//! decodes a WAV and retunes it to A = 432 Hz, channel by channel, so
//! stereo stays stereo. The same honest limit as the tuning module
//! applies: the retune slows the audio ~1.85% and does **not** preserve
//! tempo. Nothing here touches an audio device; the caller sends the
//! returned samples to one.
//!
//! Supported: RIFF/WAVE, PCM (format tag 1), 16-bit samples, mono and
//! stereo. Not supported: float (tag 3), 8-bit or 24-bit samples,
//! more than two channels, compressed formats.

use std::fmt;

use crate::tuning;

/// Errors returned by WAV decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WavError {
    /// The bytes are not a RIFF/WAVE file.
    NotWav(String),
    /// The WAV is well-formed but uses a format this crate does not
    /// support (non-PCM, wrong bit depth, wrong channel count).
    UnsupportedFormat(String),
    /// The bytes end before a declared header or chunk does.
    Truncated(String),
}

impl fmt::Display for WavError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WavError::NotWav(msg) => write!(f, "not a WAV file: {msg}"),
            WavError::UnsupportedFormat(msg) => write!(f, "unsupported WAV format: {msg}"),
            WavError::Truncated(msg) => write!(f, "truncated WAV: {msg}"),
        }
    }
}

impl std::error::Error for WavError {}

/// Decoded PCM audio: interleaved `i16` samples (L, R, L, R, ... for
/// stereo), the sample rate, and the channel count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavAudio {
    /// Interleaved samples, one `i16` per channel per frame.
    pub samples: Vec<i16>,
    /// Samples per second per channel.
    pub sample_rate: u32,
    /// 1 for mono, 2 for stereo.
    pub channels: u16,
}

impl WavAudio {
    /// Number of frames (one frame is one sample per channel).
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }

    /// The samples of one channel, deinterleaved. Channel 0 is left
    /// (or the only channel for mono).
    pub fn channel(&self, index: u16) -> Vec<i16> {
        let n = self.channels as usize;
        if n == 0 || index as usize >= n {
            return Vec::new();
        }
        (0..self.frames())
            .map(|f| self.samples[f * n + index as usize])
            .collect()
    }

    /// The samples of one channel as `f32` in [-1.0, 1.0], for tuning.
    fn channel_f32(&self, index: u16) -> Vec<f32> {
        self.channel(index)
            .iter()
            .map(|&s| s as f32 / 32768.0)
            .collect()
    }

    /// Build interleaved audio from one `f32` vec per channel. Values
    /// are clamped to [-1.0, 1.0] and rounded to the nearest `i16`.
    /// Channels must all have the same length; a shorter channel is
    /// padded with silence, a longer one is cut.
    fn from_channels_f32(channels: &[Vec<f32>], sample_rate: u32) -> WavAudio {
        let n_channels = channels.len() as u16;
        let frames = channels.iter().map(|c| c.len()).max().unwrap_or(0);
        let mut samples = Vec::with_capacity(frames * channels.len());
        for f in 0..frames {
            for ch in channels {
                let v = ch.get(f).copied().unwrap_or(0.0);
                let clamped = v.clamp(-1.0, 1.0);
                samples.push((clamped * 32768.0).round().clamp(-32768.0, 32767.0) as i16);
            }
        }
        WavAudio {
            samples,
            sample_rate,
            channels: n_channels,
        }
    }
}

/// A decoded WAV plus its 432 Hz retune, both as [`WavAudio`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayedWav {
    /// The audio exactly as decoded from the WAV bytes.
    pub decoded: WavAudio,
    /// The same audio retuned to A = 432 Hz (runs ~1.85% longer;
    /// tempo is not preserved — see [`crate::tuning`]).
    pub retuned: WavAudio,
}

fn u16le(bytes: &[u8], off: usize) -> Option<u16> {
    bytes.get(off..off + 2).map(|s| u16::from_le_bytes([s[0], s[1]]))
}

fn u32le(bytes: &[u8], off: usize) -> Option<u32> {
    bytes
        .get(off..off + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// Decode 16-bit PCM WAV bytes held in memory.
///
/// Refuses, with a clean error: non-RIFF/non-WAVE input, non-PCM
/// format, bit depth other than 16, mono/stereo violations, a
/// truncated header, truncated chunk data, or a declared size that
/// runs past the end of the buffer.
pub fn decode_wav(bytes: &[u8]) -> Result<WavAudio, WavError> {
    if bytes.len() < 12 {
        return Err(WavError::Truncated(format!(
            "WAV header needs 12 bytes, got {}",
            bytes.len()
        )));
    }
    if &bytes[0..4] != b"RIFF" {
        return Err(WavError::NotWav(
            "missing RIFF magic".to_string(),
        ));
    }
    // A declared RIFF size that runs past the buffer means the file
    // was cut short. A size of zero means "unknown"; accept it.
    if let Some(riff_size) = u32le(bytes, 4) {
        if riff_size != 0 && 8usize.saturating_add(riff_size as usize) > bytes.len() {
            return Err(WavError::Truncated(format!(
                "RIFF chunk declares {} bytes, buffer holds {}",
                riff_size,
                bytes.len()
            )));
        }
    }
    if &bytes[8..12] != b"WAVE" {
        return Err(WavError::NotWav("missing WAVE magic".to_string()));
    }

    // Scan chunks, recording where the fmt and data chunks are.
    // Unknown chunks are skipped; chunk payloads are word-aligned,
    // so an odd-sized chunk is followed by one pad byte.
    let mut fmt: Option<(usize, usize)> = None;
    let mut data: Option<(usize, usize)> = None;
    let mut off = 12usize;
    while off + 8 <= bytes.len() {
        let id = &bytes[off..off + 4];
        let size = match u32le(bytes, off + 4) {
            Some(s) => s as usize,
            None => break,
        };
        let payload = off + 8;
        let end = payload.saturating_add(size);
        if end > bytes.len() {
            return Err(WavError::Truncated(format!(
                "chunk {} declares {} bytes, only {} remain",
                String::from_utf8_lossy(id),
                size,
                bytes.len() - payload
            )));
        }
        match id {
            b"fmt " => fmt = Some((payload, size)),
            b"data" => data = Some((payload, size)),
            _ => {}
        }
        // Step over the payload plus its pad byte, if any. A missing
        // pad byte at the very end of the file is tolerated.
        off = end + (size & 1);
        if off > bytes.len() {
            off = bytes.len();
        }
    }

    let (fmt_off, fmt_len) = fmt.ok_or_else(|| {
        WavError::Truncated("missing fmt chunk".to_string())
    })?;
    let (data_off, data_len) = data.ok_or_else(|| {
        WavError::Truncated("missing data chunk".to_string())
    })?;
    if fmt_len < 16 {
        return Err(WavError::Truncated(format!(
            "fmt chunk is {fmt_len} bytes, needs at least 16"
        )));
    }
    let format_tag = u16le(bytes, fmt_off).ok_or_else(|| {
        WavError::Truncated("fmt chunk too short for format tag".to_string())
    })?;
    if format_tag != 1 {
        return Err(WavError::UnsupportedFormat(format!(
            "only PCM (format tag 1) is supported, got tag {format_tag}"
        )));
    }
    let channels = u16le(bytes, fmt_off + 2).ok_or_else(|| {
        WavError::Truncated("fmt chunk too short for channel count".to_string())
    })?;
    if channels != 1 && channels != 2 {
        return Err(WavError::UnsupportedFormat(format!(
            "only mono and stereo are supported, got {channels} channels"
        )));
    }
    let sample_rate = u32le(bytes, fmt_off + 4).ok_or_else(|| {
        WavError::Truncated("fmt chunk too short for sample rate".to_string())
    })?;
    if sample_rate == 0 {
        return Err(WavError::UnsupportedFormat(
            "sample rate must not be zero".to_string(),
        ));
    }
    let bits = u16le(bytes, fmt_off + 14).ok_or_else(|| {
        WavError::Truncated("fmt chunk too short for bit depth".to_string())
    })?;
    if bits != 16 {
        return Err(WavError::UnsupportedFormat(format!(
            "only 16-bit samples are supported, got {bits}-bit"
        )));
    }
    let block_align = u16le(bytes, fmt_off + 12).ok_or_else(|| {
        WavError::Truncated("fmt chunk too short for block align".to_string())
    })?;
    if block_align as usize != channels as usize * 2 {
        return Err(WavError::UnsupportedFormat(format!(
            "block align {block_align} does not match {channels} channels of 16-bit samples"
        )));
    }

    let frame_bytes = block_align as usize;
    if data_len % frame_bytes != 0 {
        return Err(WavError::Truncated(format!(
            "data chunk is {data_len} bytes, not a whole number of {frame_bytes}-byte frames"
        )));
    }
    let mut samples = Vec::with_capacity(data_len / 2);
    let mut i = data_off;
    let data_end = data_off + data_len;
    while i + 2 <= data_end {
        samples.push(i16::from_le_bytes([bytes[i], bytes[i + 1]]));
        i += 2;
    }
    Ok(WavAudio {
        samples,
        sample_rate,
        channels,
    })
}

/// Decode a WAV and retune it to A = 432 Hz, channel by channel, so
/// stereo stays stereo. Returns both the decoded audio and the
/// retuned audio.
pub fn play_wav_432(bytes: &[u8]) -> Result<PlayedWav, WavError> {
    let decoded = decode_wav(bytes)?;
    let mut retuned_channels = Vec::with_capacity(decoded.channels as usize);
    for c in 0..decoded.channels {
        let mono = decoded.channel_f32(c);
        retuned_channels.push(tuning::play_432(&mono));
    }
    let retuned = WavAudio::from_channels_f32(&retuned_channels, decoded.sample_rate);
    Ok(PlayedWav { decoded, retuned })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal WAV in memory: RIFF/WAVE, one fmt chunk, one
    /// data chunk, with an optional extra chunk (e.g. LIST) inserted
    /// between fmt and data.
    fn build_wav(
        format_tag: u16,
        channels: u16,
        sample_rate: u32,
        bits: u16,
        samples: &[i16],
        extra_chunk: Option<(&[u8; 4], &[u8])>,
    ) -> Vec<u8> {
        let bytes_per_sample = bits.div_ceil(8) as usize;
        let block_align = channels as usize * bytes_per_sample;
        let byte_rate = sample_rate as usize * block_align;
        let mut fmt = Vec::with_capacity(16);
        fmt.extend_from_slice(&format_tag.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&sample_rate.to_le_bytes());
        fmt.extend_from_slice(&(byte_rate as u32).to_le_bytes());
        fmt.extend_from_slice(&(block_align as u16).to_le_bytes());
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
        if let Some((id, payload)) = extra_chunk {
            push_chunk(id, payload, &mut chunks);
        }
        push_chunk(b"data", &data, &mut chunks);

        let mut out = Vec::with_capacity(12 + chunks.len());
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&((4 + chunks.len()) as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(&chunks);
        out
    }

    fn sine_i16(freq_hz: f64, sample_rate: u32, secs: f64) -> Vec<i16> {
        let n = (sample_rate as f64 * secs) as usize;
        (0..n)
            .map(|i| {
                (20000.0
                    * (2.0 * std::f64::consts::PI * freq_hz * i as f64 / sample_rate as f64)
                        .sin()) as i16
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
    fn mono_16bit_decodes_to_exact_samples() {
        let samples = [0i16, 1000, -1000, 32767, -32768, 12345, -22222];
        let bytes = build_wav(1, 1, 44_100, 16, &samples, None);
        let audio = decode_wav(&bytes).unwrap();
        assert_eq!(audio.samples, samples);
        assert_eq!(audio.sample_rate, 44_100);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.frames(), samples.len());
    }

    #[test]
    fn stereo_decodes_interleaved() {
        // L0 R0 L1 R1 L2 R2
        let samples = [100i16, -100, 200, -200, 300, -300];
        let bytes = build_wav(1, 2, 48_000, 16, &samples, None);
        let audio = decode_wav(&bytes).unwrap();
        assert_eq!(audio.samples, samples);
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.frames(), 3);
        assert_eq!(audio.channel(0), vec![100, 200, 300]);
        assert_eq!(audio.channel(1), vec![-100, -200, -300]);
    }

    #[test]
    fn unknown_chunks_are_skipped() {
        let samples = [7i16, 8, 9];
        // A LIST chunk with an odd-sized payload (5 bytes + pad).
        let bytes = build_wav(1, 1, 8000, 16, &samples, Some((b"LIST", b"hello")));
        let audio = decode_wav(&bytes).unwrap();
        assert_eq!(audio.samples, samples);
        assert_eq!(audio.sample_rate, 8000);
    }

    #[test]
    fn truncated_header_refused() {
        assert!(matches!(decode_wav(&[]), Err(WavError::Truncated(_))));
        assert!(matches!(
            decode_wav(b"RIFF"),
            Err(WavError::Truncated(_))
        ));
        assert!(matches!(
            decode_wav(b"RIFF\x24\x00\x00\x00WAV"),
            Err(WavError::Truncated(_))
        ));
        // A 12-byte header with no chunks at all.
        let mut bare = Vec::new();
        bare.extend_from_slice(b"RIFF");
        bare.extend_from_slice(&4u32.to_le_bytes());
        bare.extend_from_slice(b"WAVE");
        assert!(matches!(
            decode_wav(&bare),
            Err(WavError::Truncated(_))
        ));
        // RIFF size declares far more than the buffer holds.
        let mut lying = Vec::new();
        lying.extend_from_slice(b"RIFF");
        lying.extend_from_slice(&10_000u32.to_le_bytes());
        lying.extend_from_slice(b"WAVE");
        assert!(matches!(
            decode_wav(&lying),
            Err(WavError::Truncated(_))
        ));
    }

    #[test]
    fn truncated_data_refused() {
        let samples = [1i16, 2, 3, 4];
        let mut bytes = build_wav(1, 1, 8000, 16, &samples, None);
        // Cut the last two sample bytes off the data chunk, and lie
        // about the RIFF size to match the short buffer.
        bytes.truncate(bytes.len() - 4);
        let riff_len = (bytes.len() - 8) as u32;
        bytes[4..8].copy_from_slice(&riff_len.to_le_bytes());
        assert!(matches!(
            decode_wav(&bytes),
            Err(WavError::Truncated(_))
        ));
    }

    #[test]
    fn non_pcm_refused() {
        // Format tag 3 is IEEE float.
        let bytes = build_wav(3, 1, 44_100, 32, &[0i16, 0, 0, 0], None);
        assert!(matches!(
            decode_wav(&bytes),
            Err(WavError::UnsupportedFormat(_))
        ));
    }

    #[test]
    fn eight_bit_refused() {
        let bytes = build_wav(1, 1, 8000, 8, &[0i16, 0, 0, 0], None);
        assert!(matches!(
            decode_wav(&bytes),
            Err(WavError::UnsupportedFormat(_))
        ));
    }

    #[test]
    fn non_riff_and_non_wave_refused() {
        let mut bytes = build_wav(1, 1, 8000, 16, &[1i16, 2], None);
        bytes[0] = b'X';
        assert!(matches!(decode_wav(&bytes), Err(WavError::NotWav(_))));
        let mut bytes = build_wav(1, 1, 8000, 16, &[1i16, 2], None);
        bytes[8] = b'X';
        assert!(matches!(decode_wav(&bytes), Err(WavError::NotWav(_))));
    }

    #[test]
    fn empty_data_decodes_to_empty() {
        let bytes = build_wav(1, 1, 8000, 16, &[], None);
        let audio = decode_wav(&bytes).unwrap();
        assert!(audio.samples.is_empty());
        assert_eq!(audio.frames(), 0);
        let played = play_wav_432(&bytes).unwrap();
        assert!(played.retuned.samples.is_empty());
    }

    #[test]
    fn decoded_sine_retunes_to_432() {
        let sample_rate = 44_100u32;
        let samples = sine_i16(440.0, sample_rate, 1.0);
        let bytes = build_wav(1, 1, sample_rate, 16, &samples, None);
        let played = play_wav_432(&bytes).unwrap();
        assert_eq!(played.decoded.samples, samples);
        let before = measured_freq(&played.decoded.samples, sample_rate);
        assert!((before - 440.0).abs() < 1.0, "decoded sine at {before} Hz");
        let got = measured_freq(&played.retuned.samples, sample_rate);
        assert!(
            (got - 432.0).abs() < 2.0,
            "retuned sine measured at {got} Hz, expected ~432 Hz"
        );
        assert!(played.retuned.samples.len() > played.decoded.samples.len());
    }

    #[test]
    fn stereo_retunes_both_channels() {
        let sample_rate = 44_100u32;
        let left = sine_i16(440.0, sample_rate, 0.5);
        let right = sine_i16(440.0, sample_rate, 0.5);
        let mut interleaved = Vec::with_capacity(left.len() * 2);
        for i in 0..left.len() {
            interleaved.push(left[i]);
            interleaved.push(right[i]);
        }
        let bytes = build_wav(1, 2, sample_rate, 16, &interleaved, None);
        let played = play_wav_432(&bytes).unwrap();
        assert_eq!(played.retuned.channels, 2);
        let expected_frames =
            (played.decoded.frames() as f64 * 440.0 / 432.0).ceil() as usize;
        assert_eq!(played.retuned.frames(), expected_frames);
        let got_l = measured_freq(&played.retuned.channel(0), sample_rate);
        let got_r = measured_freq(&played.retuned.channel(1), sample_rate);
        assert!((got_l - 432.0).abs() < 2.0, "left at {got_l} Hz");
        assert!((got_r - 432.0).abs() < 2.0, "right at {got_r} Hz");
    }
}
