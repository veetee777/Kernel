//! Real playback for Kernelodys: a sample-accurate engine that streams
//! 432 Hz-retuned audio to a sink in fixed-size chunks, with a sample
//! clock, pause / resume / stop, and underrun counting.
//!
//! The engine never assumes a sound card exists. A sink is anything
//! that accepts interleaved `i16` samples ([`AudioSink`]):
//! [`WavFileSink`] renders a track to a real `.wav` file on disk, so
//! playback output is verifiable byte-for-byte. A device sink (ALSA on
//! Linux, AAudio on Android via JNI) is the next step and is NOT in
//! this build — nothing here opens an audio device.
//!
//! How it works: the engine takes decoded audio ([`crate::wav`]), runs
//! it through the 432 Hz retune ([`crate::tuning::play_432`], channel
//! by channel, so stereo stays stereo), and holds the retuned samples
//! in memory. [`PlaybackEngine::pump`] writes one fixed-size chunk to
//! a sink per call; [`PlaybackEngine::play`] pumps until the track is
//! done, pacing itself to real time unless told otherwise.
//!
//! Sample clock: the engine tracks its position in frames and derives
//! elapsed time from the sample rate (`elapsed_secs = position /
//! rate`). An underrun is counted when a chunk is delivered after the
//! moment its last frame was due to be heard — i.e. the sink is slower
//! than real time. The engine tolerates up to one chunk of slack, the
//! way a device with a one-chunk buffer would. Underruns are counted,
//! never fatal: the samples still go out exactly once, in order.
//!
//! Honest limits:
//! - The retune slows audio ~1.85% and does **not** preserve tempo
//!   (see [`crate::tuning`]).
//! - The retuned track is held in memory in full; a streaming retune
//!   (retune-while-playing) is future work.
//! - WAV output over ~4 GiB is refused: the RIFF header cannot count
//!   more than `u32::MAX` bytes.
//! - [`WavFileSink`] writes mono or stereo 16-bit PCM only, matching
//!   what the decoder accepts.

use std::fmt;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use crate::normalize;
use crate::tuning;
use crate::wav::{decode_wav, WavAudio, WavError};

/// Frames per chunk when streaming: 4096 frames ≈ 93 ms at 44.1 kHz.
pub const DEFAULT_CHUNK_FRAMES: usize = 4096;

/// Errors returned by playback operations.
#[derive(Debug)]
pub enum PlaybackError {
    /// Wraps a [`WavError`] from decoding.
    Wav(WavError),
    /// A filesystem / I/O operation failed.
    Io(std::io::Error),
    /// The operation does not make sense in the current state (e.g.
    /// playing a paused engine, writing to a finalized sink, or audio
    /// that is not a whole number of frames).
    InvalidState(String),
    /// The sink's sample rate does not match the audio's. Playing
    /// 44.1 kHz audio into a 48 kHz sink without resampling would
    /// play it at the wrong pitch, so the engine refuses instead.
    SampleRateMismatch {
        /// The sample rate the audio needs.
        expected: u32,
        /// The sample rate the sink declares.
        got: u32,
    },
}

impl fmt::Display for PlaybackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlaybackError::Wav(e) => write!(f, "WAV error: {e}"),
            PlaybackError::Io(e) => write!(f, "I/O error: {e}"),
            PlaybackError::InvalidState(msg) => write!(f, "invalid playback state: {msg}"),
            PlaybackError::SampleRateMismatch { expected, got } => write!(
                f,
                "sample rate mismatch: audio needs {expected} Hz, sink runs at {got} Hz"
            ),
        }
    }
}

impl std::error::Error for PlaybackError {}

impl From<WavError> for PlaybackError {
    fn from(e: WavError) -> Self {
        PlaybackError::Wav(e)
    }
}

impl From<std::io::Error> for PlaybackError {
    fn from(e: std::io::Error) -> Self {
        PlaybackError::Io(e)
    }
}

/// Anything that accepts interleaved `i16` audio samples. The engine
/// never assumes a sound card: a file sink, a test sink, and (later)
/// a device sink all speak this trait.
pub trait AudioSink {
    /// Write interleaved samples (one `i16` per channel per frame).
    /// The sink must accept every sample; a sink that cannot keep up
    /// with real time is counted as an underrun by the engine, not
    /// asked to drop samples.
    fn write(&mut self, samples: &[i16]) -> Result<(), PlaybackError>;
    /// Finish the stream: flush buffered data and finalize the output
    /// (for a file sink, write the final header).
    fn flush(&mut self) -> Result<(), PlaybackError>;
    /// The sample rate this sink expects, in samples per second per
    /// channel. The engine refuses to play audio at any other rate.
    fn sample_rate(&self) -> u32;
}

/// Where playback stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackState {
    /// Built, nothing played yet.
    Ready,
    /// Streaming.
    Playing,
    /// Paused; [`PlaybackEngine::resume`] picks up where it left off.
    Paused,
    /// Stopped; cannot be restarted.
    Stopped,
    /// The whole track was streamed.
    Finished,
}

/// What one playback run produced.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaybackReport {
    /// Frames written to the sink.
    pub frames_written: usize,
    /// Samples written (frames × channels).
    pub samples_written: usize,
    /// Playback time in seconds, from the sample rate.
    pub duration_secs: f64,
    /// Chunks delivered after their last frame was due — the sink
    /// was slower than real time. Never fatal; samples still went
    /// out exactly once, in order.
    pub underruns: usize,
    /// True if the whole track was streamed.
    pub finished: bool,
    /// Sample rate of the audio played.
    pub sample_rate: u32,
    /// Channel count of the audio played.
    pub channels: u16,
}

/// An [`AudioSink`] that writes 16-bit PCM to a real `.wav` file on
/// disk: a 44-byte RIFF header plus the samples, with correct sizes
/// once [`AudioSink::flush`] is called. The file decodes back to the
/// exact samples written, so playback output is verifiable
/// byte-for-byte with [`crate::wav::decode_wav`]. Mono or stereo only,
/// matching what the decoder accepts.
pub struct WavFileSink {
    file: File,
    sample_rate: u32,
    channels: u16,
    byte_rate: u32,
    frames_written: usize,
    data_bytes: u64,
    finalized: bool,
}

impl WavFileSink {
    /// Create a sink writing to `path` (created or truncated). The
    /// file starts with a placeholder header; [`AudioSink::flush`]
    /// rewrites it with the real sizes.
    pub fn create(path: &Path, sample_rate: u32, channels: u16) -> Result<Self, PlaybackError> {
        if sample_rate == 0 {
            return Err(PlaybackError::InvalidState(
                "sample rate must not be zero".to_string(),
            ));
        }
        if channels != 1 && channels != 2 {
            return Err(PlaybackError::InvalidState(format!(
                "WavFileSink writes mono or stereo only, got {channels} channels"
            )));
        }
        let byte_rate = sample_rate as u64 * channels as u64 * 2;
        if byte_rate > u32::MAX as u64 {
            return Err(PlaybackError::InvalidState(
                "sample rate too high for a WAV byte rate".to_string(),
            ));
        }
        let file = File::create(path)?;
        let mut sink = Self {
            file,
            sample_rate,
            channels,
            byte_rate: byte_rate as u32,
            frames_written: 0,
            data_bytes: 0,
            finalized: false,
        };
        sink.write_header(0)?;
        Ok(sink)
    }

    /// Number of channels this sink writes.
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Frames written so far.
    pub fn frames_written(&self) -> usize {
        self.frames_written
    }

    fn write_header(&mut self, data_bytes: u32) -> Result<(), PlaybackError> {
        let riff_size = 36u32.checked_add(data_bytes).ok_or_else(|| {
            PlaybackError::InvalidState("WAV data too large for RIFF header".to_string())
        })?;
        let mut header = Vec::with_capacity(44);
        header.extend_from_slice(b"RIFF");
        header.extend_from_slice(&riff_size.to_le_bytes());
        header.extend_from_slice(b"WAVE");
        header.extend_from_slice(b"fmt ");
        header.extend_from_slice(&16u32.to_le_bytes());
        header.extend_from_slice(&1u16.to_le_bytes()); // PCM
        header.extend_from_slice(&self.channels.to_le_bytes());
        header.extend_from_slice(&self.sample_rate.to_le_bytes());
        header.extend_from_slice(&self.byte_rate.to_le_bytes());
        header.extend_from_slice(&(self.channels * 2).to_le_bytes()); // block align
        header.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        header.extend_from_slice(b"data");
        header.extend_from_slice(&data_bytes.to_le_bytes());
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&header)?;
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), PlaybackError> {
        if self.finalized {
            return Ok(());
        }
        if self.data_bytes > (u32::MAX - 36) as u64 {
            return Err(PlaybackError::InvalidState(
                "WAV data too large for RIFF header".to_string(),
            ));
        }
        self.write_header(self.data_bytes as u32)?;
        self.file.flush()?;
        self.finalized = true;
        Ok(())
    }
}

impl AudioSink for WavFileSink {
    fn write(&mut self, samples: &[i16]) -> Result<(), PlaybackError> {
        if self.finalized {
            return Err(PlaybackError::InvalidState(
                "cannot write to a finalized WavFileSink".to_string(),
            ));
        }
        if !samples.len().is_multiple_of(self.channels as usize) {
            return Err(PlaybackError::InvalidState(format!(
                "{} samples are not a whole number of {}-channel frames",
                samples.len(),
                self.channels
            )));
        }
        let bytes = samples.len() as u64 * 2;
        if self.data_bytes + bytes > u32::MAX as u64 {
            return Err(PlaybackError::InvalidState(
                "WAV data would exceed 4 GiB".to_string(),
            ));
        }
        let mut buf = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        self.file.write_all(&buf)?;
        self.frames_written += samples.len() / self.channels as usize;
        self.data_bytes += bytes;
        Ok(())
    }

    fn flush(&mut self) -> Result<(), PlaybackError> {
        self.finalize()
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

impl Drop for WavFileSink {
    fn drop(&mut self) {
        // Best effort: a sink dropped without flush() still leaves a
        // valid WAV of whatever was written, not a zero-size header.
        let _ = self.finalize();
    }
}

/// Retune decoded audio to 432 Hz, channel by channel, exactly as
/// [`crate::wav::play_wav_432`] does, so engine output matches a
/// direct retune sample for sample.
fn retune_audio(decoded: &WavAudio) -> WavAudio {
    let mut retuned_channels: Vec<Vec<f32>> = Vec::with_capacity(decoded.channels as usize);
    for c in 0..decoded.channels {
        let mono: Vec<f32> = decoded
            .channel(c)
            .iter()
            .map(|&s| s as f32 / 32768.0)
            .collect();
        retuned_channels.push(tuning::play_432(&mono));
    }
    let frames = retuned_channels.iter().map(|c| c.len()).max().unwrap_or(0);
    let mut samples = Vec::with_capacity(frames * retuned_channels.len());
    for f in 0..frames {
        for ch in &retuned_channels {
            let v = ch.get(f).copied().unwrap_or(0.0);
            let clamped = v.clamp(-1.0, 1.0);
            samples.push((clamped * 32768.0).round().clamp(-32768.0, 32767.0) as i16);
        }
    }
    WavAudio {
        samples,
        sample_rate: decoded.sample_rate,
        channels: decoded.channels,
    }
}

/// The playback engine: decoded audio, retuned to 432 Hz, streamed to
/// a sink in fixed-size chunks with a sample clock.
///
/// The engine holds the retuned track in memory in full. It never
/// drops or duplicates a sample: every frame is written exactly once,
/// in order, and a slow sink is counted as an underrun rather than
/// asked to skip anything.
pub struct PlaybackEngine {
    /// The retuned audio, interleaved `i16`, held in memory.
    retuned: WavAudio,
    /// Frames per chunk.
    chunk_frames: usize,
    /// Frames already written to the sink.
    position_frames: usize,
    /// Underruns counted so far.
    underruns: usize,
    /// Chunks written so far.
    chunks_written: usize,
    state: PlaybackState,
    /// When `true`, [`PlaybackEngine::play`] paces itself to real
    /// time (sleeps between chunks). Off for offline rendering.
    realtime: bool,
    /// Normalization gain applied to every chunk as it is written
    /// (see [`PlaybackEngine::set_gain`]). `1.0` means no change.
    gain: f32,
    /// Wall-clock origin of the sample clock; set on the first chunk.
    origin: Option<Instant>,
    /// When the current pause started, for shifting the origin on resume.
    paused_at: Option<Instant>,
}

impl PlaybackEngine {
    /// Build an engine from decoded audio: retune it to 432 Hz and
    /// stream it in [`DEFAULT_CHUNK_FRAMES`]-frame chunks.
    pub fn from_decoded(decoded: WavAudio) -> Result<Self, PlaybackError> {
        Self::with_chunk_frames(decoded, DEFAULT_CHUNK_FRAMES)
    }

    /// Build an engine from decoded audio with an explicit chunk size
    /// (in frames). Chunk size must be at least 1 frame.
    pub fn with_chunk_frames(
        decoded: WavAudio,
        chunk_frames: usize,
    ) -> Result<Self, PlaybackError> {
        if decoded.sample_rate == 0 {
            return Err(PlaybackError::InvalidState(
                "sample rate must not be zero".to_string(),
            ));
        }
        if decoded.channels == 0 {
            return Err(PlaybackError::InvalidState(
                "channel count must not be zero".to_string(),
            ));
        }
        if !decoded
            .samples
            .len()
            .is_multiple_of(decoded.channels as usize)
        {
            return Err(PlaybackError::InvalidState(
                "samples are not a whole number of frames".to_string(),
            ));
        }
        if chunk_frames == 0 {
            return Err(PlaybackError::InvalidState(
                "chunk size must be at least 1 frame".to_string(),
            ));
        }
        Ok(Self {
            retuned: retune_audio(&decoded),
            chunk_frames,
            position_frames: 0,
            underruns: 0,
            chunks_written: 0,
            state: PlaybackState::Ready,
            realtime: true,
            gain: 1.0,
            origin: None,
            paused_at: None,
        })
    }

    /// Build an engine straight from WAV bytes: decode, then retune.
    pub fn from_wav(bytes: &[u8]) -> Result<Self, PlaybackError> {
        let decoded = decode_wav(bytes)?;
        Self::from_decoded(decoded)
    }

    /// The retuned audio this engine streams.
    pub fn retuned(&self) -> &WavAudio {
        &self.retuned
    }

    /// Sample rate of the audio, in samples per second per channel.
    pub fn sample_rate(&self) -> u32 {
        self.retuned.sample_rate
    }

    /// Channel count of the audio (1 mono, 2 stereo).
    pub fn channels(&self) -> u16 {
        self.retuned.channels
    }

    /// Total frames in the retuned track.
    pub fn total_frames(&self) -> usize {
        self.retuned.frames()
    }

    /// Frames already written to the sink.
    pub fn position_frames(&self) -> usize {
        self.position_frames
    }

    /// Frames not yet written.
    pub fn remaining_frames(&self) -> usize {
        self.total_frames().saturating_sub(self.position_frames)
    }

    /// Playback time so far in seconds, from the sample rate — the
    /// sample clock. Pauses do not count as playing time.
    pub fn elapsed_secs(&self) -> f64 {
        if self.sample_rate() == 0 {
            0.0
        } else {
            self.position_frames as f64 / self.sample_rate() as f64
        }
    }

    /// Underruns counted so far.
    pub fn underruns(&self) -> usize {
        self.underruns
    }

    /// Chunks written so far.
    pub fn chunks_written(&self) -> usize {
        self.chunks_written
    }

    /// Where playback stands.
    pub fn state(&self) -> PlaybackState {
        self.state
    }

    /// True if the whole track was streamed.
    pub fn is_finished(&self) -> bool {
        self.state == PlaybackState::Finished || self.position_frames >= self.total_frames()
    }

    /// True if playback is paused.
    pub fn is_paused(&self) -> bool {
        self.state == PlaybackState::Paused
    }

    /// Whether [`PlaybackEngine::play`] paces itself to real time.
    pub fn realtime(&self) -> bool {
        self.realtime
    }

    /// Turn real-time pacing on or off. Off for offline rendering
    /// (e.g. writing to a file); on for anything a listener hears.
    pub fn set_realtime(&mut self, realtime: bool) {
        self.realtime = realtime;
    }

    /// Normalization gain applied to the retuned samples as they are
    /// written to the sink. Use [`crate::normalize::normalize_gain`]
    /// to compute it from a track's peak so tracks play at the same
    /// level. The gain is clamped to [`crate::normalize::MAX_GAIN`]
    /// and every scaled sample is clamped to the `i16` range, so the
    /// output can never clip past full scale. A non-finite or
    /// negative gain is refused. May be set at any time; it applies
    /// to chunks written after the call.
    pub fn set_gain(&mut self, gain: f32) -> Result<(), PlaybackError> {
        if !gain.is_finite() || gain < 0.0 {
            return Err(PlaybackError::InvalidState(format!(
                "gain must be finite and non-negative, got {gain}"
            )));
        }
        self.gain = gain.min(normalize::MAX_GAIN);
        Ok(())
    }

    /// The normalization gain currently applied to output chunks.
    pub fn gain(&self) -> f32 {
        self.gain
    }

    /// Stream one chunk (up to `chunk_frames` frames) to `sink`.
    /// Returns the frames written: 0 when paused, stopped, finished,
    /// or when the audio is empty. Never drops or duplicates a sample.
    pub fn pump(&mut self, sink: &mut dyn AudioSink) -> Result<usize, PlaybackError> {
        if self.state == PlaybackState::Paused
            || self.state == PlaybackState::Stopped
            || self.state == PlaybackState::Finished
        {
            return Ok(0);
        }
        let total = self.total_frames();
        if self.position_frames >= total {
            self.state = PlaybackState::Finished;
            return Ok(0);
        }
        if sink.sample_rate() != self.sample_rate() {
            return Err(PlaybackError::SampleRateMismatch {
                expected: self.sample_rate(),
                got: sink.sample_rate(),
            });
        }
        if self.origin.is_none() {
            self.origin = Some(Instant::now());
        }
        self.state = PlaybackState::Playing;
        let frames = (total - self.position_frames).min(self.chunk_frames);
        // Underrun check: this chunk's last frame was due to be heard
        // at origin + (position + frames) / rate. If the wall clock is
        // already past that moment, the sink starved — count one
        // underrun. The chunk still goes out in full, in order.
        if let Some(origin) = self.origin {
            let due = origin
                + Duration::from_secs_f64(
                    (self.position_frames + frames) as f64 / self.sample_rate() as f64,
                );
            if Instant::now() > due {
                self.underruns += 1;
            }
        }
        let n = self.channels() as usize;
        let start = self.position_frames * n;
        let end = (self.position_frames + frames) * n;
        let chunk = &self.retuned.samples[start..end];
        if self.gain == 1.0 {
            sink.write(chunk)?;
        } else {
            let scaled = normalize::apply_gain(chunk, self.gain);
            sink.write(&scaled)?;
        }
        self.position_frames += frames;
        self.chunks_written += 1;
        if self.position_frames >= total {
            self.state = PlaybackState::Finished;
        }
        Ok(frames)
    }

    /// Stream the whole track to `sink`, then return a report. When
    /// real-time pacing is on, this takes about as long as the track;
    /// with pacing off it runs as fast as the sink accepts samples.
    /// Refuses a paused or stopped engine (resume it first).
    pub fn play(&mut self, sink: &mut dyn AudioSink) -> Result<PlaybackReport, PlaybackError> {
        match self.state {
            PlaybackState::Paused => {
                return Err(PlaybackError::InvalidState(
                    "playback is paused; call resume() first".to_string(),
                ));
            }
            PlaybackState::Stopped => {
                return Err(PlaybackError::InvalidState(
                    "playback was stopped".to_string(),
                ));
            }
            PlaybackState::Finished => return Ok(self.report()),
            _ => {}
        }
        self.state = PlaybackState::Playing;
        while !self.is_finished() {
            self.pump(sink)?;
            if self.realtime {
                if let Some(origin) = self.origin {
                    let target = origin
                        + Duration::from_secs_f64(
                            self.position_frames as f64 / self.sample_rate() as f64,
                        );
                    let now = Instant::now();
                    if target > now {
                        std::thread::sleep(target - now);
                    }
                }
            }
        }
        Ok(self.report())
    }

    /// Pause playback. A no-op unless currently playing. Pumping
    /// while paused writes nothing; [`PlaybackEngine::resume`] picks
    /// up exactly where playback left off, and the sample clock is
    /// shifted so the pause does not count as playing time.
    pub fn pause(&mut self) {
        if self.state == PlaybackState::Playing {
            self.state = PlaybackState::Paused;
            self.paused_at = Some(Instant::now());
        }
    }

    /// Resume after [`PlaybackEngine::pause`]. A no-op unless paused.
    pub fn resume(&mut self) {
        if self.state == PlaybackState::Paused {
            if let (Some(origin), Some(paused_at)) = (self.origin, self.paused_at) {
                self.origin = Some(origin + paused_at.elapsed());
            }
            self.paused_at = None;
            self.state = PlaybackState::Playing;
        }
    }

    /// Stop playback. Pumping after a stop writes nothing, and
    /// [`PlaybackEngine::play`] refuses a stopped engine. The
    /// position is kept, so [`PlaybackEngine::report`] still shows
    /// how much was played.
    pub fn stop(&mut self) {
        if self.state != PlaybackState::Finished {
            self.state = PlaybackState::Stopped;
        }
    }

    /// A report of playback so far: frames written, duration from the
    /// sample rate, underruns counted, and whether the track finished.
    pub fn report(&self) -> PlaybackReport {
        PlaybackReport {
            frames_written: self.position_frames,
            samples_written: self.position_frames * self.channels() as usize,
            duration_secs: self.elapsed_secs(),
            underruns: self.underruns,
            finished: self.is_finished(),
            sample_rate: self.sample_rate(),
            channels: self.channels(),
        }
    }
}

/// Stream an engine to a `.wav` file on disk, offline (no real-time
/// pacing). Flush runs even if playback failed, so the file still
/// holds a valid WAV of whatever was written. A playback error wins
/// over a flush error.
fn render_engine_to_file(
    engine: &mut PlaybackEngine,
    path: &Path,
) -> Result<PlaybackReport, PlaybackError> {
    let mut sink = WavFileSink::create(path, engine.sample_rate(), engine.channels())?;
    let result = engine.play(&mut sink);
    let flush_result = sink.flush();
    match result {
        Ok(report) => flush_result.map(|()| report),
        Err(e) => Err(e),
    }
}

/// Decode a WAV, retune it to 432 Hz, and render it to a `.wav` file
/// on disk through the playback engine. Returns a report of what was
/// written. Rendering to a file is offline: the engine does not pace
/// itself to real time. The output file decodes back (see
/// [`crate::wav::decode_wav`]) to the exact retuned samples.
pub fn play_wav_to_file(bytes: &[u8], path: &Path) -> Result<PlaybackReport, PlaybackError> {
    let mut engine = PlaybackEngine::from_wav(bytes)?;
    engine.set_realtime(false);
    render_engine_to_file(&mut engine, path)
}

/// Like [`play_wav_to_file`], but the engine applies `gain` to the
/// retuned samples as they are written. The gain is clamped to
/// [`crate::normalize::MAX_GAIN`] and samples are clamped to the
/// `i16` range, so the output cannot clip.
pub fn play_wav_to_file_with_gain(
    bytes: &[u8],
    path: &Path,
    gain: f32,
) -> Result<PlaybackReport, PlaybackError> {
    let mut engine = PlaybackEngine::from_wav(bytes)?;
    engine.set_realtime(false);
    engine.set_gain(gain)?;
    render_engine_to_file(&mut engine, path)
}

/// Like [`play_wav_to_file`], but the gain is computed from the
/// track's own peak ([`crate::normalize::normalize_gain`]), so the
/// rendered file plays at full-scale peak. The gain is computed on
/// the decoded samples; the retune cannot raise a peak (linear
/// interpolation stays between neighbouring samples), so the output
/// peak never exceeds full scale.
pub fn play_wav_to_file_normalized(
    bytes: &[u8],
    path: &Path,
) -> Result<PlaybackReport, PlaybackError> {
    let decoded = decode_wav(bytes)?;
    let gain = normalize::normalize_gain(&decoded.samples);
    let mut engine = PlaybackEngine::from_decoded(decoded)?;
    engine.set_realtime(false);
    engine.set_gain(gain)?;
    render_engine_to_file(&mut engine, path)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Build a minimal 16-bit PCM WAV in memory: RIFF/WAVE, one fmt
    /// chunk, one data chunk.
    fn build_wav(channels: u16, sample_rate: u32, samples: &[i16]) -> Vec<u8> {
        let block_align = channels * 2;
        let byte_rate = sample_rate * block_align as u32;
        let mut fmt = Vec::with_capacity(16);
        fmt.extend_from_slice(&1u16.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&sample_rate.to_le_bytes());
        fmt.extend_from_slice(&byte_rate.to_le_bytes());
        fmt.extend_from_slice(&block_align.to_le_bytes());
        fmt.extend_from_slice(&16u16.to_le_bytes());

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

    /// A distinctive, deterministic sample pattern: no two frames alike.
    fn pattern_samples(frames: usize) -> Vec<i16> {
        (0..frames)
            .map(|i| (((i * 7919) % 32768) as i32 - 16384) as i16)
            .collect()
    }

    fn test_path(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kernelodys-playback-{}-{}",
            std::process::id(),
            name
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);
        dir.join("out.wav")
    }

    /// A sink that collects samples in memory, optionally sleeping on
    /// each write to simulate a device slower than real time.
    struct VecSink {
        rate: u32,
        channels: u16,
        samples: Vec<i16>,
        flushed: bool,
        delay: Duration,
    }

    impl VecSink {
        fn new(rate: u32, channels: u16) -> Self {
            Self {
                rate,
                channels,
                samples: Vec::new(),
                flushed: false,
                delay: Duration::ZERO,
            }
        }

        fn slow(rate: u32, channels: u16, delay: Duration) -> Self {
            let mut sink = Self::new(rate, channels);
            sink.delay = delay;
            sink
        }
    }

    impl AudioSink for VecSink {
        fn write(&mut self, samples: &[i16]) -> Result<(), PlaybackError> {
            if !samples.len().is_multiple_of(self.channels as usize) {
                return Err(PlaybackError::InvalidState(format!(
                    "{} samples are not a whole number of {}-channel frames",
                    samples.len(),
                    self.channels
                )));
            }
            if !self.delay.is_zero() {
                std::thread::sleep(self.delay);
            }
            self.samples.extend_from_slice(samples);
            Ok(())
        }

        fn flush(&mut self) -> Result<(), PlaybackError> {
            self.flushed = true;
            Ok(())
        }

        fn sample_rate(&self) -> u32 {
            self.rate
        }
    }

    #[test]
    fn file_sink_round_trips_exact_samples() {
        // Mono.
        let samples = [0i16, 1000, -1000, 32767, -32768, 12345, -22222];
        let path = test_path("mono");
        let mut sink = WavFileSink::create(&path, 44_100, 1).unwrap();
        sink.write(&samples).unwrap();
        assert_eq!(sink.frames_written(), samples.len());
        sink.flush().unwrap();
        // Flush is idempotent.
        sink.flush().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        // Header checks, straight from the bytes.
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        assert_eq!(u16::from_le_bytes([bytes[20], bytes[21]]), 1); // PCM
        assert_eq!(u16::from_le_bytes([bytes[22], bytes[23]]), 1); // channels
        assert_eq!(
            u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            44_100
        );
        assert_eq!(
            u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]),
            88_200
        ); // byte rate
        assert_eq!(u16::from_le_bytes([bytes[32], bytes[33]]), 2); // block align
        assert_eq!(u16::from_le_bytes([bytes[34], bytes[35]]), 16); // bits
        assert_eq!(&bytes[36..40], b"data");
        let data_bytes = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
        assert_eq!(data_bytes as usize, samples.len() * 2);
        let riff_size = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(riff_size, 36 + data_bytes);
        assert_eq!(bytes.len(), 44 + samples.len() * 2);
        // And the crate's own decoder reads it back exactly.
        let audio = decode_wav(&bytes).unwrap();
        assert_eq!(audio.samples, samples);
        assert_eq!(audio.sample_rate, 44_100);
        assert_eq!(audio.channels, 1);

        // Stereo.
        let stereo = [100i16, -100, 200, -200, 300, -300];
        let path = test_path("stereo");
        let mut sink = WavFileSink::create(&path, 48_000, 2).unwrap();
        sink.write(&stereo).unwrap();
        sink.flush().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let audio = decode_wav(&bytes).unwrap();
        assert_eq!(audio.samples, stereo);
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.sample_rate, 48_000);
    }

    #[test]
    fn file_sink_rejects_partial_frame_and_write_after_flush() {
        let path = test_path("badwrite");
        let mut sink = WavFileSink::create(&path, 8000, 2).unwrap();
        // 3 samples are not a whole number of stereo frames.
        assert!(sink.write(&[1, 2, 3]).is_err());
        assert!(sink.write(&[1, 2, 3, 4]).is_ok());
        assert_eq!(sink.frames_written(), 2);
        sink.flush().unwrap();
        assert!(sink.write(&[1, 2]).is_err());
        // Flushing twice is fine.
        assert!(sink.flush().is_ok());
        // Zero channels / zero rate are refused at creation.
        assert!(WavFileSink::create(&path, 0, 1).is_err());
        assert!(WavFileSink::create(&path, 8000, 0).is_err());
        assert!(WavFileSink::create(&path, 8000, 3).is_err());
    }

    #[test]
    fn engine_streams_sine_through_retune_at_432() {
        let sample_rate = 44_100u32;
        let samples = sine_i16(440.0, sample_rate, 1.0);
        let bytes = build_wav(1, sample_rate, &samples);
        let path = test_path("sine432");
        let report = play_wav_to_file(&bytes, &path).unwrap();
        assert!(report.finished);
        assert_eq!(report.underruns, 0);
        assert_eq!(report.sample_rate, sample_rate);
        assert_eq!(report.channels, 1);
        let expected_frames = (sample_rate as f64 * 440.0 / 432.0).ceil() as usize;
        assert_eq!(report.frames_written, expected_frames);
        assert_eq!(report.samples_written, expected_frames);
        assert!((report.duration_secs - expected_frames as f64 / sample_rate as f64).abs() < 1e-9);
        // The rendered file measures ~432 Hz, not 440.
        let out = std::fs::read(&path).unwrap();
        let audio = decode_wav(&out).unwrap();
        assert_eq!(audio.samples.len(), expected_frames);
        let got = measured_freq(&audio.samples, sample_rate);
        assert!(
            (got - 432.0).abs() < 2.0,
            "rendered sine measured at {got} Hz, expected ~432 Hz"
        );
    }

    #[test]
    fn chunked_streaming_delivers_every_sample_exactly_once() {
        let sample_rate = 44_100u32;
        let samples = pattern_samples(1050);
        let bytes = build_wav(1, sample_rate, &samples);
        // The direct retune, computed without the engine.
        let direct = crate::wav::play_wav_432(&bytes).unwrap().retuned.samples;
        let mut engine =
            PlaybackEngine::with_chunk_frames(decode_wav(&bytes).unwrap(), 100).unwrap();
        let mut sink = VecSink::new(sample_rate, 1);
        let mut chunk_sizes = Vec::new();
        while !engine.is_finished() {
            chunk_sizes.push(engine.pump(&mut sink).unwrap());
        }
        // The retuned track is longer than the decoded one: 1050
        // frames become ceil(1050 * 440/432) = 1070, i.e. ten
        // 100-frame chunks and one partial chunk of 70.
        let mut expected: Vec<usize> = vec![100; direct.len() / 100];
        if !direct.len().is_multiple_of(100) {
            expected.push(direct.len() % 100);
        }
        assert_eq!(chunk_sizes, expected);
        assert_eq!(engine.chunks_written(), expected.len());
        assert_eq!(engine.position_frames(), direct.len());
        assert_eq!(engine.underruns(), 0);
        // Sample-accurate: exactly the direct retune, in order.
        assert_eq!(sink.samples, direct);
        // The engine's own retuned audio is the same reference.
        assert_eq!(engine.retuned().samples, direct);
    }

    #[test]
    fn pause_resume_keeps_stream_continuous() {
        let sample_rate = 44_100u32;
        let samples = pattern_samples(1050);
        let bytes = build_wav(1, sample_rate, &samples);
        let direct = crate::wav::play_wav_432(&bytes).unwrap().retuned.samples;
        let mut engine =
            PlaybackEngine::with_chunk_frames(decode_wav(&bytes).unwrap(), 100).unwrap();
        let mut sink = VecSink::new(sample_rate, 1);
        assert_eq!(engine.pump(&mut sink).unwrap(), 100);
        assert_eq!(engine.pump(&mut sink).unwrap(), 100);
        engine.pause();
        assert!(engine.is_paused());
        // Pumping while paused writes nothing.
        assert_eq!(engine.pump(&mut sink).unwrap(), 0);
        assert_eq!(sink.samples.len(), 200);
        assert_eq!(engine.position_frames(), 200);
        engine.resume();
        assert!(!engine.is_paused());
        while !engine.is_finished() {
            engine.pump(&mut sink).unwrap();
        }
        assert_eq!(sink.samples, direct);
        assert_eq!(engine.underruns(), 0);
        // play() on a finished engine just returns its report.
        assert!(engine.play(&mut sink).unwrap().finished);
    }

    #[test]
    fn stereo_stays_stereo() {
        let sample_rate = 44_100u32;
        let left = sine_i16(440.0, sample_rate, 0.5);
        let right = sine_i16(220.0, sample_rate, 0.5);
        let mut interleaved = Vec::with_capacity(left.len() * 2);
        for i in 0..left.len() {
            interleaved.push(left[i]);
            interleaved.push(right[i]);
        }
        let bytes = build_wav(2, sample_rate, &interleaved);
        let direct = crate::wav::play_wav_432(&bytes).unwrap().retuned;
        let mut engine =
            PlaybackEngine::with_chunk_frames(decode_wav(&bytes).unwrap(), 256).unwrap();
        let mut sink = VecSink::new(sample_rate, 2);
        engine.set_realtime(false);
        let report = engine.play(&mut sink).unwrap();
        assert!(report.finished);
        assert_eq!(report.channels, 2);
        assert_eq!(sink.samples, direct.samples);
        // Channels were never mixed: left is the 440 Hz retune (~432),
        // right is the 220 Hz retune (~216).
        let out_left: Vec<i16> = sink.samples.iter().step_by(2).copied().collect();
        let out_right: Vec<i16> = sink.samples.iter().skip(1).step_by(2).copied().collect();
        assert_eq!(out_left, direct.channel(0));
        assert_eq!(out_right, direct.channel(1));
        assert_ne!(out_left, out_right);
        let got_l = measured_freq(&out_left, sample_rate);
        let got_r = measured_freq(&out_right, sample_rate);
        assert!((got_l - 432.0).abs() < 2.0, "left at {got_l} Hz");
        assert!((got_r - 216.0).abs() < 2.0, "right at {got_r} Hz");
    }

    #[test]
    fn empty_and_very_short_audio() {
        // Empty: a WAV with no samples at all.
        let bytes = build_wav(1, 8000, &[]);
        let path = test_path("empty");
        let report = play_wav_to_file(&bytes, &path).unwrap();
        assert_eq!(report.frames_written, 0);
        assert_eq!(report.samples_written, 0);
        assert_eq!(report.duration_secs, 0.0);
        assert_eq!(report.underruns, 0);
        assert!(report.finished);
        let out = std::fs::read(&path).unwrap();
        assert_eq!(decode_wav(&out).unwrap().frames(), 0);

        // One frame, mono.
        let bytes = build_wav(1, 8000, &[12345]);
        let direct = crate::wav::play_wav_432(&bytes).unwrap().retuned.samples;
        let path = test_path("oneframe");
        let report = play_wav_to_file(&bytes, &path).unwrap();
        assert_eq!(report.frames_written, direct.len());
        assert!(report.finished);
        let out = std::fs::read(&path).unwrap();
        assert_eq!(decode_wav(&out).unwrap().samples, direct);

        // Three frames, stereo.
        let bytes = build_wav(2, 8000, &[1, -1, 2, -2, 3, -3]);
        let direct = crate::wav::play_wav_432(&bytes).unwrap().retuned.samples;
        let path = test_path("threeframes");
        let report = play_wav_to_file(&bytes, &path).unwrap();
        assert!(report.finished);
        let out = std::fs::read(&path).unwrap();
        assert_eq!(decode_wav(&out).unwrap().samples, direct);
    }

    #[test]
    fn underrun_counting_with_slow_sink() {
        let sample_rate = 44_100u32;
        // Half a second of audio, 4096-frame chunks (~93 ms each),
        // sink sleeps 200 ms per write: it falls behind by more than
        // one chunk almost immediately, so underruns must pile up —
        // and the samples must still all arrive, exactly once.
        let samples = sine_i16(440.0, sample_rate, 0.5);
        let bytes = build_wav(1, sample_rate, &samples);
        let direct = crate::wav::play_wav_432(&bytes).unwrap().retuned.samples;
        let mut engine = PlaybackEngine::from_decoded(decode_wav(&bytes).unwrap()).unwrap();
        let mut sink = VecSink::slow(sample_rate, 1, Duration::from_millis(200));
        let report = engine.play(&mut sink).unwrap();
        assert!(report.finished);
        assert!(
            report.underruns >= 3,
            "slow sink should underrun several times, got {}",
            report.underruns
        );
        assert_eq!(sink.samples, direct);
    }

    #[test]
    fn fast_sink_counts_no_underruns() {
        let sample_rate = 44_100u32;
        let samples = sine_i16(440.0, sample_rate, 0.5);
        let bytes = build_wav(1, sample_rate, &samples);
        let direct = crate::wav::play_wav_432(&bytes).unwrap().retuned.samples;
        let mut engine = PlaybackEngine::from_decoded(decode_wav(&bytes).unwrap()).unwrap();
        let mut sink = VecSink::new(sample_rate, 1);
        let report = engine.play(&mut sink).unwrap();
        assert!(report.finished);
        assert_eq!(report.underruns, 0);
        assert_eq!(sink.samples, direct);
    }

    #[test]
    fn stop_halts_playback() {
        let sample_rate = 44_100u32;
        let samples = pattern_samples(1050);
        let bytes = build_wav(1, sample_rate, &samples);
        let mut engine =
            PlaybackEngine::with_chunk_frames(decode_wav(&bytes).unwrap(), 100).unwrap();
        let mut sink = VecSink::new(sample_rate, 1);
        assert_eq!(engine.pump(&mut sink).unwrap(), 100);
        engine.stop();
        assert_eq!(engine.state(), PlaybackState::Stopped);
        // Pumping after a stop writes nothing.
        assert_eq!(engine.pump(&mut sink).unwrap(), 0);
        assert_eq!(sink.samples.len(), 100);
        let report = engine.report();
        assert!(!report.finished);
        assert_eq!(report.frames_written, 100);
        // play() refuses a stopped engine.
        assert!(matches!(
            engine.play(&mut sink),
            Err(PlaybackError::InvalidState(_))
        ));
    }

    #[test]
    fn sample_rate_mismatch_refused() {
        let sample_rate = 44_100u32;
        let samples = pattern_samples(200);
        let bytes = build_wav(1, sample_rate, &samples);
        let mut engine = PlaybackEngine::from_decoded(decode_wav(&bytes).unwrap()).unwrap();
        // A sink running at 48 kHz cannot take 44.1 kHz audio.
        let mut sink = VecSink::new(48_000, 1);
        assert!(matches!(
            engine.pump(&mut sink),
            Err(PlaybackError::SampleRateMismatch {
                expected: 44_100,
                got: 48_000
            })
        ));
        assert_eq!(engine.position_frames(), 0);
    }

    #[test]
    fn engine_validates_audio() {
        // Zero sample rate.
        let bad = WavAudio {
            samples: vec![1, 2, 3],
            sample_rate: 0,
            channels: 1,
        };
        assert!(PlaybackEngine::from_decoded(bad).is_err());
        // Zero channels.
        let bad = WavAudio {
            samples: vec![1, 2, 3],
            sample_rate: 8000,
            channels: 0,
        };
        assert!(PlaybackEngine::from_decoded(bad).is_err());
        // Samples that are not a whole number of stereo frames.
        let bad = WavAudio {
            samples: vec![1, 2, 3],
            sample_rate: 8000,
            channels: 2,
        };
        assert!(PlaybackEngine::from_decoded(bad).is_err());
        // Zero chunk size.
        let ok = WavAudio {
            samples: vec![1, 2],
            sample_rate: 8000,
            channels: 1,
        };
        assert!(PlaybackEngine::with_chunk_frames(ok, 0).is_err());
    }

    #[test]
    fn play_wav_to_file_rejects_bad_wav() {
        let path = test_path("badwav");
        assert!(matches!(
            play_wav_to_file(b"this is not a wav file at all", &path),
            Err(PlaybackError::Wav(_))
        ));
        assert!(matches!(
            play_wav_to_file(b"", &path),
            Err(PlaybackError::Wav(_))
        ));
    }

    #[test]
    fn engine_applies_gain_during_playback() {
        let sample_rate = 44_100u32;
        let samples = pattern_samples(1050);
        let bytes = build_wav(1, sample_rate, &samples);
        let direct = crate::wav::play_wav_432(&bytes).unwrap().retuned.samples;
        let mut engine =
            PlaybackEngine::with_chunk_frames(decode_wav(&bytes).unwrap(), 100).unwrap();
        assert_eq!(engine.gain(), 1.0);
        engine.set_gain(2.0).unwrap();
        assert_eq!(engine.gain(), 2.0);
        let mut sink = VecSink::new(sample_rate, 1);
        engine.set_realtime(false);
        let report = engine.play(&mut sink).unwrap();
        assert!(report.finished);
        // The output is the direct retune, scaled by the gain, with
        // every sample clamped to the i16 range.
        let expected = crate::normalize::apply_gain(&direct, 2.0);
        assert_eq!(sink.samples, expected);
        // Gain above the cap is clamped, not refused.
        let mut engine = PlaybackEngine::from_decoded(decode_wav(&bytes).unwrap()).unwrap();
        engine.set_gain(100.0).unwrap();
        assert_eq!(engine.gain(), crate::normalize::MAX_GAIN);
        // Non-finite and negative gains are refused.
        assert!(engine.set_gain(f32::NAN).is_err());
        assert!(engine.set_gain(-1.0).is_err());
        assert!(engine.set_gain(f32::INFINITY).is_err());
    }

    #[test]
    fn normalized_render_brings_quiet_track_to_full_scale() {
        let sample_rate = 8000u32;
        // A quiet 440 Hz sine at amplitude 5000.
        let samples: Vec<i16> = (0..8000)
            .map(|i| {
                (5000.0
                    * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sample_rate as f64).sin())
                    as i16
            })
            .collect();
        let bytes = build_wav(1, sample_rate, &samples);
        let path = test_path("normalized");
        let report = play_wav_to_file_normalized(&bytes, &path).unwrap();
        assert!(report.finished);
        let out = std::fs::read(&path).unwrap();
        let audio = decode_wav(&out).unwrap();
        let peak = crate::normalize::peak_amplitude(&audio.samples);
        // Full scale, within interpolation rounding of it.
        assert!(
            peak >= 32_700,
            "normalized render peaked at {peak}, expected near 32767"
        );
        assert!(peak <= 32767);
    }
}
