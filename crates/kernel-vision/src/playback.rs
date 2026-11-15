//! Real video playback for Kernel•Vision: a raw-frame pipeline.
//!
//! This module decodes, converts, scales, and presents video frames.
//! It works on **raw YUV420 planar frames only** — the `KVFR` stream
//! format defined below. It does NOT decode H.264, AV1, VP9, or any
//! other compressed codec; a real compressed-codec decoder is a much
//! bigger project (and, for the formats that need it, a patent minefield
//! a from-scratch Rust decoder should not walk into lightly).
//!
//! There is no display or GPU sink here: this VM has no screen to
//! present to, so output is proven to files. [`PpmSink`] writes one
//! PPM (P6) file per presented frame, which any image viewer can open
//! and which tests can decode back and check pixel-for-pixel.
//!
//! # KVFR stream format (explicit binary, no JSON)
//!
//! A stream is a sequence of frames. Each frame is:
//!
//! ```text
//! magic "KVFR" (4) + width u16 LE + height u16 LE + pts u64 LE
//! + payload length u32 LE + raw YUV420 bytes
//! ```
//!
//! The payload is the Y plane (`width * height` bytes) followed by the
//! U plane and the V plane, each `ceil(width / 2) * ceil(height / 2)`
//! bytes. A frame whose payload length does not match its declared
//! dimensions is refused, as is any frame with bad magic or a truncated
//! header or payload.
//!
//! # Odd dimensions
//!
//! Odd widths and heights are **supported**, not refused: the chroma
//! planes are `ceil(width / 2)` by `ceil(height / 2)`, which is the
//! standard YUV420 rule (the last chroma sample covers the final odd
//! pixel). Conversion samples chroma with `(x / 2, y / 2)`, which stays
//! in bounds for every pixel.
//!
//! # Colour
//!
//! YUV420 to RGB24 uses BT.601 limited-range conversion with integer
//! math (the same constants as the reference C implementation):
//!
//! ```text
//! C = Y - 16, D = U - 128, E = V - 128
//! R = (298*C + 409*E + 128) >> 8
//! G = (298*C - 100*D - 208*E + 128) >> 8
//! B = (298*C + 516*D + 128) >> 8
//! ```
//!
//! Each channel is clamped to 0..=255. So Y=16 is black, Y=235 is
//! white, and (Y=76, U=85, V=255) is pure red.
//!
//! # Timing
//!
//! [`VideoPlayer`] paces frames on a real clock: a frame's `pts_ms` is
//! its due time relative to the first frame's pts. A frame that is more
//! than one frame-duration late when its turn comes is dropped, never
//! shown late, and frames are never presented out of pts order. There
//! is **no audio sync** yet — there is no audio track in a KVFR stream
//! to sync to.

use std::fmt;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::MAX_DIMENSION;

/// Magic bytes at the start of every KVFR frame header.
pub const KVFR_MAGIC: &[u8; 4] = b"KVFR";
/// Length of a KVFR frame header in bytes: magic 4 + width 2 + height 2
/// + pts 8 + payload length 4.
pub const KVFR_HEADER_LEN: usize = 20;
/// Largest width or height a KVFR frame can declare (u16 field).
pub const MAX_KVFR_DIMENSION: u32 = u16::MAX as u32;

/// Errors returned by playback operations.
#[derive(Debug)]
pub enum PlaybackError {
    /// A frame's dimensions or planes were invalid.
    InvalidFrame(String),
    /// A KVFR frame header did not start with `KVFR`.
    BadMagic,
    /// A stream ended in the middle of a header or a payload.
    Truncated(String),
    /// A frame's declared payload length did not match its dimensions.
    SizeMismatch(String),
    /// The configured fps was zero or above the supported maximum.
    InvalidFps(u32),
    /// A scaler target dimension was zero or above the supported maximum.
    InvalidTargetSize(String),
    /// A filesystem operation failed.
    Io(std::io::Error),
}

impl fmt::Display for PlaybackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlaybackError::InvalidFrame(msg) => write!(f, "invalid frame: {msg}"),
            PlaybackError::BadMagic => f.write_str("bad KVFR magic"),
            PlaybackError::Truncated(msg) => write!(f, "truncated stream: {msg}"),
            PlaybackError::SizeMismatch(msg) => write!(f, "size mismatch: {msg}"),
            PlaybackError::InvalidFps(fps) => write!(f, "invalid fps: {fps}"),
            PlaybackError::InvalidTargetSize(msg) => write!(f, "invalid target size: {msg}"),
            PlaybackError::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for PlaybackError {}

impl From<std::io::Error> for PlaybackError {
    fn from(e: std::io::Error) -> Self {
        PlaybackError::Io(e)
    }
}

/// One video frame: YUV420 planar data plus a presentation timestamp.
///
/// The Y plane is `width * height` bytes. The U and V planes are each
/// `ceil(width / 2) * ceil(height / 2)` bytes, so odd dimensions work:
/// the final chroma sample covers the last odd pixel in its row/column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    width: u32,
    height: u32,
    pts_ms: u64,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Frame {
    /// Width of one chroma plane for a frame of this width.
    pub fn chroma_width(width: u32) -> u32 {
        width.div_ceil(2)
    }

    /// Height of one chroma plane for a frame of this height.
    pub fn chroma_height(height: u32) -> u32 {
        height.div_ceil(2)
    }

    /// Expected payload length in bytes for a YUV420 frame of these
    /// dimensions: Y plane plus both chroma planes.
    pub fn expected_payload_len(width: u32, height: u32) -> u64 {
        let y = width as u64 * height as u64;
        let c = Self::chroma_width(width) as u64 * Self::chroma_height(height) as u64;
        y + 2 * c
    }

    /// Build a frame, validating dimensions and plane sizes. Width and
    /// height must be at least 1 and at most [`MAX_DIMENSION`]; the
    /// planes must be exactly the sizes the dimensions imply.
    pub fn new(
        width: u32,
        height: u32,
        pts_ms: u64,
        y: Vec<u8>,
        u: Vec<u8>,
        v: Vec<u8>,
    ) -> Result<Self, PlaybackError> {
        if width == 0 || height == 0 {
            return Err(PlaybackError::InvalidFrame(
                "width and height must be at least 1 pixel".to_string(),
            ));
        }
        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err(PlaybackError::InvalidFrame(format!(
                "dimensions must be at most {MAX_DIMENSION} pixels"
            )));
        }
        let y_len = width as usize * height as usize;
        let c_len = Self::chroma_width(width) as usize * Self::chroma_height(height) as usize;
        if y.len() != y_len {
            return Err(PlaybackError::InvalidFrame(format!(
                "Y plane is {} bytes, expected {y_len}",
                y.len()
            )));
        }
        if u.len() != c_len {
            return Err(PlaybackError::InvalidFrame(format!(
                "U plane is {} bytes, expected {c_len}",
                u.len()
            )));
        }
        if v.len() != c_len {
            return Err(PlaybackError::InvalidFrame(format!(
                "V plane is {} bytes, expected {c_len}",
                v.len()
            )));
        }
        Ok(Self {
            width,
            height,
            pts_ms,
            y,
            u,
            v,
        })
    }

    /// Frame width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Presentation timestamp in milliseconds.
    pub fn pts_ms(&self) -> u64 {
        self.pts_ms
    }

    /// The Y (luma) plane, `width * height` bytes, row-major.
    pub fn y_plane(&self) -> &[u8] {
        &self.y
    }

    /// The U (chroma) plane, `ceil(width / 2) * ceil(height / 2)` bytes.
    pub fn u_plane(&self) -> &[u8] {
        &self.u
    }

    /// The V (chroma) plane, `ceil(width / 2) * ceil(height / 2)` bytes.
    pub fn v_plane(&self) -> &[u8] {
        &self.v
    }
}

/// One decoded frame in RGB24: `width * height * 3` bytes, row-major,
/// three bytes per pixel (R, G, B). Carries the source frame's pts so a
/// sink can order and time what it presents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbFrame {
    width: u32,
    height: u32,
    pts_ms: u64,
    data: Vec<u8>,
}

impl RgbFrame {
    /// Build an RGB frame, validating dimensions and data length.
    pub fn new(width: u32, height: u32, pts_ms: u64, data: Vec<u8>) -> Result<Self, PlaybackError> {
        if width == 0 || height == 0 {
            return Err(PlaybackError::InvalidFrame(
                "width and height must be at least 1 pixel".to_string(),
            ));
        }
        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err(PlaybackError::InvalidFrame(format!(
                "dimensions must be at most {MAX_DIMENSION} pixels"
            )));
        }
        let expected = width as usize * height as usize * 3;
        if data.len() != expected {
            return Err(PlaybackError::InvalidFrame(format!(
                "RGB data is {} bytes, expected {expected}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            pts_ms,
            data,
        })
    }

    /// Frame width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Presentation timestamp in milliseconds.
    pub fn pts_ms(&self) -> u64 {
        self.pts_ms
    }

    /// RGB24 bytes, row-major, three bytes per pixel.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// The pixel at (`x`, `y`) as [R, G, B]. Panics on out-of-range
    /// coordinates, as an index would.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 3] {
        let off = (y as usize * self.width as usize + x as usize) * 3;
        [self.data[off], self.data[off + 1], self.data[off + 2]]
    }
}

/// Clamp an i32 colour channel to 0..=255.
fn clamp_channel(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Convert one YUV420 frame to RGB24 (BT.601 limited range, integer
/// math). Chroma is sampled nearest-neighbour: pixel (`x`, `y`) uses
/// chroma sample (`x / 2`, `y / 2`).
pub fn yuv420_to_rgb(frame: &Frame) -> RgbFrame {
    let w = frame.width() as usize;
    let h = frame.height() as usize;
    let cw = Frame::chroma_width(frame.width()) as usize;
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let yi = frame.y_plane()[y * w + x] as i32;
            let ci = (y / 2) * cw + (x / 2);
            let u = frame.u_plane()[ci] as i32;
            let v = frame.v_plane()[ci] as i32;
            let c = yi - 16;
            let d = u - 128;
            let e = v - 128;
            let r = (298 * c + 409 * e + 128) >> 8;
            let g = (298 * c - 100 * d - 208 * e + 128) >> 8;
            let b = (298 * c + 516 * d + 128) >> 8;
            let off = (y * w + x) * 3;
            out[off] = clamp_channel(r);
            out[off + 1] = clamp_channel(g);
            out[off + 2] = clamp_channel(b);
        }
    }
    RgbFrame {
        width: frame.width(),
        height: frame.height(),
        pts_ms: frame.pts_ms(),
        data: out,
    }
}

/// Scale an RGB frame to an exact target size with nearest-neighbour
/// sampling: target pixel (`x`, `y`) takes source pixel
/// (`x * src_w / target_w`, `y * src_h / target_h`). Refuses zero
/// dimensions. The pts is carried through unchanged.
pub fn scale_rgb(src: &RgbFrame, target_w: u32, target_h: u32) -> Result<RgbFrame, PlaybackError> {
    if target_w == 0 || target_h == 0 {
        return Err(PlaybackError::InvalidTargetSize(
            "target width and height must be at least 1 pixel".to_string(),
        ));
    }
    if target_w > MAX_DIMENSION || target_h > MAX_DIMENSION {
        return Err(PlaybackError::InvalidTargetSize(format!(
            "target dimensions must be at most {MAX_DIMENSION} pixels"
        )));
    }
    let out_len = (target_w as usize)
        .checked_mul(target_h as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| PlaybackError::InvalidTargetSize("target size overflows".to_string()))?;
    let mut out = vec![0u8; out_len];
    for y in 0..target_h as usize {
        let sy = y * src.height() as usize / target_h as usize;
        for x in 0..target_w as usize {
            let sx = x * src.width() as usize / target_w as usize;
            let src_off = (sy * src.width() as usize + sx) * 3;
            let dst_off = (y * target_w as usize + x) * 3;
            out[dst_off..dst_off + 3].copy_from_slice(&src.data()[src_off..src_off + 3]);
        }
    }
    Ok(RgbFrame {
        width: target_w,
        height: target_h,
        pts_ms: src.pts_ms(),
        data: out,
    })
}

/// Encode one frame as KVFR bytes (header + raw YUV420 payload).
/// Refuses frames wider or taller than u16::MAX, which the header
/// field cannot represent.
pub fn encode_frame(frame: &Frame) -> Result<Vec<u8>, PlaybackError> {
    if frame.width() > MAX_KVFR_DIMENSION || frame.height() > MAX_KVFR_DIMENSION {
        return Err(PlaybackError::InvalidFrame(format!(
            "KVFR dimensions must fit in u16, got {}x{}",
            frame.width(),
            frame.height()
        )));
    }
    let payload_len = frame.y_plane().len() + frame.u_plane().len() + frame.v_plane().len();
    let mut out = Vec::with_capacity(KVFR_HEADER_LEN + payload_len);
    out.extend_from_slice(KVFR_MAGIC);
    out.extend_from_slice(&(frame.width() as u16).to_le_bytes());
    out.extend_from_slice(&(frame.height() as u16).to_le_bytes());
    out.extend_from_slice(&frame.pts_ms().to_le_bytes());
    out.extend_from_slice(&(payload_len as u32).to_le_bytes());
    out.extend_from_slice(frame.y_plane());
    out.extend_from_slice(frame.u_plane());
    out.extend_from_slice(frame.v_plane());
    Ok(out)
}

/// A decoder for KVFR streams: raw YUV420 frames with a small header.
/// No compressed codecs — the payload is already pixels.
pub struct YuvDecoder;

impl YuvDecoder {
    /// Decode a whole stream into frames, in stream order. An empty
    /// stream decodes to an empty list. Refuses bad magic, truncated
    /// headers or payloads, zero dimensions, and payload lengths that
    /// do not match the declared dimensions.
    pub fn decode_all(bytes: &[u8]) -> Result<Vec<Frame>, PlaybackError> {
        let mut frames = Vec::new();
        let mut off = 0usize;
        while off < bytes.len() {
            if bytes.len() - off < KVFR_HEADER_LEN {
                return Err(PlaybackError::Truncated(format!(
                    "frame header needs {KVFR_HEADER_LEN} bytes, only {} remain",
                    bytes.len() - off
                )));
            }
            if &bytes[off..off + 4] != KVFR_MAGIC {
                return Err(PlaybackError::BadMagic);
            }
            let width = u16::from_le_bytes(bytes[off + 4..off + 6].try_into().expect("2")) as u32;
            let height = u16::from_le_bytes(bytes[off + 6..off + 8].try_into().expect("2")) as u32;
            let pts = u64::from_le_bytes(bytes[off + 8..off + 16].try_into().expect("8"));
            let payload_len =
                u32::from_le_bytes(bytes[off + 16..off + 20].try_into().expect("4")) as usize;
            off += KVFR_HEADER_LEN;
            if width == 0 || height == 0 {
                return Err(PlaybackError::InvalidFrame(
                    "frame width and height must be at least 1 pixel".to_string(),
                ));
            }
            if width > MAX_DIMENSION || height > MAX_DIMENSION {
                return Err(PlaybackError::InvalidFrame(format!(
                    "frame dimensions must be at most {MAX_DIMENSION} pixels"
                )));
            }
            let expected = Frame::expected_payload_len(width, height);
            if payload_len as u64 != expected {
                return Err(PlaybackError::SizeMismatch(format!(
                    "frame {width}x{height} needs a {expected}-byte payload, header says {payload_len}"
                )));
            }
            if bytes.len() - off < payload_len {
                return Err(PlaybackError::Truncated(format!(
                    "frame payload needs {payload_len} bytes, only {} remain",
                    bytes.len() - off
                )));
            }
            let payload = &bytes[off..off + payload_len];
            let y_len = width as usize * height as usize;
            let c_len = Frame::chroma_width(width) as usize * Frame::chroma_height(height) as usize;
            let frame = Frame::new(
                width,
                height,
                pts,
                payload[..y_len].to_vec(),
                payload[y_len..y_len + c_len].to_vec(),
                payload[y_len + c_len..].to_vec(),
            )?;
            frames.push(frame);
            off += payload_len;
        }
        Ok(frames)
    }
}

/// Something that presents decoded frames: a display, a file writer,
/// a test recorder. `present` receives frames in pts order; a sink
/// that is slow may cause the player to drop later frames instead.
pub trait FrameSink {
    /// Present one RGB frame. The frame carries its pts.
    fn present(&mut self, frame: &RgbFrame) -> Result<(), PlaybackError>;
}

/// A sink that writes each presented frame as a PPM (P6) file in a
/// directory: `frame_000000.ppm`, `frame_000001.ppm`, and so on, in
/// presentation order. The directory is created if it does not exist.
pub struct PpmSink {
    dir: PathBuf,
    count: usize,
}

impl PpmSink {
    /// Create a sink writing into `dir`, creating it if needed.
    pub fn new(dir: &Path) -> Result<Self, PlaybackError> {
        std::fs::create_dir_all(dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            count: 0,
        })
    }

    /// How many frames have been written so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The directory frames are written into.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl FrameSink for PpmSink {
    fn present(&mut self, frame: &RgbFrame) -> Result<(), PlaybackError> {
        let path = self.dir.join(format!("frame_{:06}.ppm", self.count));
        let mut file = File::create(&path)?;
        write!(file, "P6\n{} {}\n255\n", frame.width(), frame.height())?;
        file.write_all(frame.data())?;
        self.count += 1;
        Ok(())
    }
}

/// A handle for controlling a running [`VideoPlayer`] from another
/// thread: pause, resume, and stop. Cheap to clone; all state is
/// shared atomics.
#[derive(Clone, Debug)]
pub struct PlayerControl {
    paused: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
}

impl PlayerControl {
    /// Pause playback. The frame clock freezes while paused, so
    /// resuming does not jump the timing.
    pub fn pause(&self) {
        self.paused.store(true, Ordering::SeqCst);
    }

    /// Resume playback after a pause.
    pub fn resume(&self) {
        self.paused.store(false, Ordering::SeqCst);
    }

    /// Stop playback. The running `play_frames` call returns with a
    /// report of what it presented so far.
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
    }

    /// True while playback is paused.
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    /// True once playback has been stopped.
    pub fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }
}

/// The outcome of playing a stream: how many frames were shown, how
/// many were dropped, how many were seen in total, and the span of
/// presented pts values in milliseconds (0 when nothing was shown).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaybackReport {
    /// Frames presented to the sink.
    pub presented: usize,
    /// Frames skipped: too late, or out of pts order.
    pub dropped: usize,
    /// Frames seen in total (`presented + dropped`).
    pub total: usize,
    /// `last presented pts - first presented pts`, in milliseconds.
    pub duration_ms: u64,
}

/// Plays decoded frames to a sink on a frame clock.
///
/// Frame durations come from `fps`: at 30 fps a frame is due every
/// ~33 ms. A frame's pts is its due time relative to the first frame's
/// pts. A frame that is more than one frame-duration late when its
/// turn comes is dropped, never shown late. A frame whose pts is lower
/// than the last presented frame's pts is also dropped, so frames are
/// never presented out of order.
///
/// Pause freezes the clock; stop ends playback with a partial report.
/// Both are driven through [`PlayerControl`], which works from another
/// thread while `play_frames` runs.
pub struct VideoPlayer<S: FrameSink> {
    sink: S,
    fps: u32,
    target_w: u32,
    target_h: u32,
    control: PlayerControl,
}

impl<S: FrameSink> VideoPlayer<S> {
    /// Build a player. `fps` must be 1..=1000; the target size must be
    /// at least 1 pixel in each direction and at most
    /// [`MAX_DIMENSION`]. Frames are scaled to the target size before
    /// presentation (a no-op when the target matches the source).
    pub fn new(sink: S, fps: u32, target_w: u32, target_h: u32) -> Result<Self, PlaybackError> {
        if fps == 0 || fps > 1000 {
            return Err(PlaybackError::InvalidFps(fps));
        }
        if target_w == 0 || target_h == 0 {
            return Err(PlaybackError::InvalidTargetSize(
                "target width and height must be at least 1 pixel".to_string(),
            ));
        }
        if target_w > MAX_DIMENSION || target_h > MAX_DIMENSION {
            return Err(PlaybackError::InvalidTargetSize(format!(
                "target dimensions must be at most {MAX_DIMENSION} pixels"
            )));
        }
        Ok(Self {
            sink,
            fps,
            target_w,
            target_h,
            control: PlayerControl {
                paused: Arc::new(AtomicBool::new(false)),
                stopped: Arc::new(AtomicBool::new(false)),
            },
        })
    }

    /// A control handle for pausing, resuming, and stopping this player
    /// from another thread.
    pub fn control(&self) -> PlayerControl {
        self.control.clone()
    }

    /// Play `frames` in order. Returns a report when the stream ends
    /// or playback is stopped. A sink error aborts playback with an
    /// error instead of a report.
    pub fn play_frames(&mut self, frames: &[Frame]) -> Result<PlaybackReport, PlaybackError> {
        let frame_duration = Duration::from_millis(1000 / self.fps as u64);
        let start = Instant::now();
        let mut pause_offset = Duration::ZERO;
        let mut presented = 0usize;
        let mut dropped = 0usize;
        let mut base_pts: Option<u64> = None;
        let mut first_presented: Option<u64> = None;
        let mut last_presented: Option<u64> = None;

        'frames: for frame in frames {
            if self.control.is_stopped() {
                break;
            }
            let base = *base_pts.get_or_insert(frame.pts_ms());
            let pts = frame.pts_ms();
            // Out of order: never present a frame older than the last one shown.
            if let Some(last) = last_presented {
                if pts < last {
                    dropped += 1;
                    continue;
                }
            }
            let pts_offset = pts.saturating_sub(base);
            // Wait until the frame is due, honouring pause and stop.
            loop {
                if self.control.is_stopped() {
                    break 'frames;
                }
                if self.control.is_paused() {
                    let paused_since = Instant::now();
                    loop {
                        if self.control.is_stopped() {
                            break 'frames;
                        }
                        if !self.control.is_paused() {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    pause_offset += paused_since.elapsed();
                    continue;
                }
                let due = start + pause_offset + Duration::from_millis(pts_offset);
                let now = Instant::now();
                if now >= due {
                    break;
                }
                let remaining = due - now;
                std::thread::sleep(std::cmp::min(remaining, Duration::from_millis(2)));
            }
            // Too late: more than one frame-duration past its due time.
            let due = start + pause_offset + Duration::from_millis(pts_offset);
            if Instant::now() > due + frame_duration {
                dropped += 1;
                continue;
            }
            let rgb = yuv420_to_rgb(frame);
            let rgb = if rgb.width() == self.target_w && rgb.height() == self.target_h {
                rgb
            } else {
                scale_rgb(&rgb, self.target_w, self.target_h)?
            };
            self.sink.present(&rgb)?;
            presented += 1;
            if first_presented.is_none() {
                first_presented = Some(pts);
            }
            last_presented = Some(pts);
        }

        Ok(PlaybackReport {
            presented,
            dropped,
            total: frames.len(),
            duration_ms: match (first_presented, last_presented) {
                (Some(first), Some(last)) => last - first,
                _ => 0,
            },
        })
    }
}

/// Decode a KVFR stream, convert to RGB, and present every frame to
/// PPM files in `dir` at the stream's own size. Returns a report of
/// what was presented and dropped. An empty stream is a valid empty
/// playback: a report of all zeros.
pub fn play_stream_to_dir(
    bytes: &[u8],
    fps: u32,
    dir: &Path,
) -> Result<PlaybackReport, PlaybackError> {
    let frames = YuvDecoder::decode_all(bytes)?;
    let (target_w, target_h) = match frames.first() {
        Some(first) => (first.width(), first.height()),
        None => (1, 1),
    };
    play_stream_to_dir_scaled(bytes, fps, dir, target_w, target_h)
}

/// Decode a KVFR stream, convert to RGB, scale every frame to
/// `target_w` x `target_h`, and present to PPM files in `dir`.
pub fn play_stream_to_dir_scaled(
    bytes: &[u8],
    fps: u32,
    dir: &Path,
    target_w: u32,
    target_h: u32,
) -> Result<PlaybackReport, PlaybackError> {
    let frames = YuvDecoder::decode_all(bytes)?;
    let sink = PpmSink::new(dir)?;
    let mut player = VideoPlayer::new(sink, fps, target_w, target_h)?;
    player.play_frames(&frames)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A solid-colour frame: every Y sample `y`, every U sample `u`,
    /// every V sample `v`.
    fn solid_frame(w: u32, h: u32, pts: u64, y: u8, u: u8, v: u8) -> Frame {
        let y_len = w as usize * h as usize;
        let c_len = Frame::chroma_width(w) as usize * Frame::chroma_height(h) as usize;
        Frame::new(w, h, pts, vec![y; y_len], vec![u; c_len], vec![v; c_len]).unwrap()
    }

    /// Encode several frames into one KVFR stream.
    fn encode_stream(frames: &[Frame]) -> Vec<u8> {
        let mut out = Vec::new();
        for f in frames {
            out.extend_from_slice(&encode_frame(f).unwrap());
        }
        out
    }

    /// Parse a PPM (P6) file back into (width, height, rgb bytes).
    /// A tiny parser for test verification only.
    fn read_ppm(path: &Path) -> (u32, u32, Vec<u8>) {
        let bytes = std::fs::read(path).unwrap();
        // Header: "P6\n<w> <h>\n255\n" then raw bytes.
        let mut parts = bytes.splitn(4, |b| *b == b'\n');
        assert_eq!(parts.next().unwrap(), b"P6");
        let dims = parts.next().unwrap();
        let mut it = dims.split(|b| *b == b' ');
        let w: u32 = std::str::from_utf8(it.next().unwrap())
            .unwrap()
            .parse()
            .unwrap();
        let h: u32 = std::str::from_utf8(it.next().unwrap())
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(parts.next().unwrap(), b"255");
        let body = parts.next().unwrap();
        assert_eq!(body.len(), w as usize * h as usize * 3);
        (w, h, body.to_vec())
    }

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("kernel-vision-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// A sink that records what it was given, in order.
    struct RecordingSink {
        presented: Vec<RgbFrame>,
    }

    impl RecordingSink {
        fn new() -> Self {
            Self { presented: vec![] }
        }
    }

    impl FrameSink for RecordingSink {
        fn present(&mut self, frame: &RgbFrame) -> Result<(), PlaybackError> {
            self.presented.push(frame.clone());
            Ok(())
        }
    }

    /// A sink that is deliberately slow: every present sleeps, so the
    /// player falls behind and must drop frames.
    struct SlowSink {
        presented: Vec<u64>,
        delay: Duration,
    }

    impl SlowSink {
        fn new(delay_ms: u64) -> Self {
            Self {
                presented: vec![],
                delay: Duration::from_millis(delay_ms),
            }
        }
    }

    impl FrameSink for SlowSink {
        fn present(&mut self, frame: &RgbFrame) -> Result<(), PlaybackError> {
            std::thread::sleep(self.delay);
            self.presented.push(frame.pts_ms());
            Ok(())
        }
    }

    #[test]
    fn decoder_accepts_a_well_formed_stream() {
        let f0 = solid_frame(4, 2, 0, 16, 128, 128);
        let f1 = solid_frame(4, 2, 33, 235, 128, 128);
        let stream = encode_stream(&[f0.clone(), f1.clone()]);
        let frames = YuvDecoder::decode_all(&stream).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], f0);
        assert_eq!(frames[1], f1);
        assert_eq!(frames[0].width(), 4);
        assert_eq!(frames[0].height(), 2);
        assert_eq!(frames[1].pts_ms(), 33);
    }

    #[test]
    fn decoder_refuses_bad_magic() {
        let mut stream = encode_stream(&[solid_frame(4, 2, 0, 16, 128, 128)]);
        stream[0] = b'X';
        assert!(matches!(
            YuvDecoder::decode_all(&stream),
            Err(PlaybackError::BadMagic)
        ));
        // Bad magic in the second frame also fails.
        let mut two = encode_stream(&[
            solid_frame(4, 2, 0, 16, 128, 128),
            solid_frame(4, 2, 33, 16, 128, 128),
        ]);
        let first_len = encode_frame(&solid_frame(4, 2, 0, 16, 128, 128))
            .unwrap()
            .len();
        two[first_len] = b'X';
        assert!(matches!(
            YuvDecoder::decode_all(&two),
            Err(PlaybackError::BadMagic)
        ));
    }

    #[test]
    fn decoder_refuses_truncated_header_and_payload() {
        let stream = encode_stream(&[solid_frame(4, 2, 0, 16, 128, 128)]);
        // Only 10 bytes of the 20-byte header.
        assert!(matches!(
            YuvDecoder::decode_all(&stream[..10]),
            Err(PlaybackError::Truncated(_))
        ));
        // Full header, half the payload.
        assert!(matches!(
            YuvDecoder::decode_all(&stream[..KVFR_HEADER_LEN + 5]),
            Err(PlaybackError::Truncated(_))
        ));
        // Header plus payload, then a dangling partial second header.
        let mut two = stream.clone();
        two.extend_from_slice(b"KVFR\x04\x00");
        assert!(matches!(
            YuvDecoder::decode_all(&two),
            Err(PlaybackError::Truncated(_))
        ));
    }

    #[test]
    fn decoder_refuses_payload_size_mismatch() {
        let mut frame = encode_stream(&[solid_frame(4, 2, 0, 16, 128, 128)]);
        // A 4x2 frame needs 8 + 2 + 2 = 12 payload bytes; claim 11.
        frame[16..20].copy_from_slice(&11u32.to_le_bytes());
        assert!(matches!(
            YuvDecoder::decode_all(&frame),
            Err(PlaybackError::SizeMismatch(_))
        ));
        // Claim 13 instead.
        let mut frame = encode_stream(&[solid_frame(4, 2, 0, 16, 128, 128)]);
        frame[16..20].copy_from_slice(&13u32.to_le_bytes());
        assert!(matches!(
            YuvDecoder::decode_all(&frame),
            Err(PlaybackError::SizeMismatch(_))
        ));
    }

    #[test]
    fn decoder_empty_stream_is_empty() {
        assert_eq!(YuvDecoder::decode_all(b"").unwrap(), Vec::<Frame>::new());
    }

    #[test]
    fn decoder_refuses_zero_dimensions() {
        // Hand-build a header with width 0.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"KVFR");
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            YuvDecoder::decode_all(&bytes),
            Err(PlaybackError::InvalidFrame(_))
        ));
    }

    #[test]
    fn odd_dimensions_are_supported() {
        // 3x3: Y is 9 bytes, each chroma plane is 2x2 = 4 bytes.
        let frame = solid_frame(3, 3, 0, 100, 128, 128);
        assert_eq!(frame.y_plane().len(), 9);
        assert_eq!(frame.u_plane().len(), 4);
        assert_eq!(frame.v_plane().len(), 4);
        let stream = encode_frame(&frame).unwrap();
        let decoded = YuvDecoder::decode_all(&stream).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0], frame);
        // Conversion covers all 9 pixels, including the odd row/column.
        let rgb = yuv420_to_rgb(&decoded[0]);
        assert_eq!(rgb.width(), 3);
        assert_eq!(rgb.height(), 3);
        assert_eq!(rgb.data().len(), 27);
        // Scale 3x3 -> 6x6 exactly.
        let scaled = scale_rgb(&rgb, 6, 6).unwrap();
        assert_eq!(scaled.width(), 6);
        assert_eq!(scaled.height(), 6);
        assert_eq!(scaled.data().len(), 108);
    }

    #[test]
    fn frame_rejects_plane_mismatch() {
        // 4x2 needs Y=8, U=2, V=2.
        assert!(Frame::new(4, 2, 0, vec![0; 7], vec![0; 2], vec![0; 2]).is_err());
        assert!(Frame::new(4, 2, 0, vec![0; 8], vec![0; 1], vec![0; 2]).is_err());
        assert!(Frame::new(4, 2, 0, vec![0; 8], vec![0; 2], vec![0; 3]).is_err());
        assert!(Frame::new(0, 2, 0, vec![], vec![], vec![]).is_err());
        assert!(Frame::new(4, 2, 0, vec![0; 8], vec![0; 2], vec![0; 2]).is_ok());
    }

    #[test]
    fn yuv_to_rgb_known_values() {
        // Black: Y=16, neutral chroma.
        let black = yuv420_to_rgb(&solid_frame(2, 2, 0, 16, 128, 128));
        assert_eq!(black.pixel(0, 0), [0, 0, 0]);
        assert_eq!(black.pixel(1, 1), [0, 0, 0]);
        // White: Y=235, neutral chroma.
        let white = yuv420_to_rgb(&solid_frame(2, 2, 0, 235, 128, 128));
        assert_eq!(white.pixel(0, 0), [255, 255, 255]);
        // Pure red: Y=76, U=85, V=255.
        let red = yuv420_to_rgb(&solid_frame(2, 2, 0, 76, 85, 255));
        assert_eq!(red.pixel(0, 0), [255, 0, 0]);
        // Pure green: Y=150, U=44, V=21.
        let green = yuv420_to_rgb(&solid_frame(2, 2, 0, 150, 44, 21));
        assert_eq!(green.pixel(0, 0), [0, 255, 0]);
        // Mid grey stays grey: Y=128, neutral chroma -> R=G=B.
        let grey = yuv420_to_rgb(&solid_frame(2, 2, 0, 128, 128, 128));
        let [r, g, b] = grey.pixel(0, 0);
        assert_eq!(r, g);
        assert_eq!(g, b);
        assert!((100..=140).contains(&r));
    }

    #[test]
    fn scaler_output_dimensions_exact() {
        // 2x2 with distinct quadrants: red, green / blue, white.
        let mut rgb = RgbFrame::new(
            2,
            2,
            0,
            vec![
                255, 0, 0, 0, 255, 0, // row 0: red, green
                0, 0, 255, 255, 255, 255, // row 1: blue, white
            ],
        )
        .unwrap();
        let up = scale_rgb(&rgb, 4, 4).unwrap();
        assert_eq!(up.width(), 4);
        assert_eq!(up.height(), 4);
        // Nearest neighbour: each source pixel becomes a 2x2 block.
        assert_eq!(up.pixel(0, 0), [255, 0, 0]);
        assert_eq!(up.pixel(1, 1), [255, 0, 0]);
        assert_eq!(up.pixel(2, 0), [0, 255, 0]);
        assert_eq!(up.pixel(0, 2), [0, 0, 255]);
        assert_eq!(up.pixel(3, 3), [255, 255, 255]);
        // Scale down 4x4 -> 2x2 picks the even rows/columns.
        let down = scale_rgb(&up, 2, 2).unwrap();
        assert_eq!(down, rgb);
        // Identity scale is exact.
        let same = scale_rgb(&rgb, 2, 2).unwrap();
        assert_eq!(same, rgb);
        // Non-square target.
        let wide = scale_rgb(&rgb, 4, 2).unwrap();
        assert_eq!((wide.width(), wide.height()), (4, 2));
        // Zero target refused.
        assert!(scale_rgb(&rgb, 0, 2).is_err());
        assert!(scale_rgb(&rgb, 2, 0).is_err());
        // pts is carried through.
        rgb = RgbFrame::new(2, 2, 777, vec![0u8; 12]).unwrap();
        assert_eq!(scale_rgb(&rgb, 4, 4).unwrap().pts_ms(), 777);
    }

    #[test]
    fn player_presents_in_order_with_no_drops() {
        let frames: Vec<Frame> = (0..5)
            .map(|i| solid_frame(4, 2, i * 33, 100 + i as u8, 128, 128))
            .collect();
        let sink = RecordingSink::new();
        let mut player = VideoPlayer::new(sink, 30, 4, 2).unwrap();
        let report = player.play_frames(&frames).unwrap();
        assert_eq!(report.total, 5);
        assert_eq!(report.presented, 5);
        assert_eq!(report.dropped, 0);
        assert_eq!(report.duration_ms, 4 * 33);
        // The sink saw them in pts order.
        let pts: Vec<u64> = player.sink.presented.iter().map(|f| f.pts_ms()).collect();
        assert_eq!(pts, vec![0, 33, 66, 99, 132]);
    }

    #[test]
    fn player_drops_frames_when_the_sink_is_slow() {
        // 20 fps: a frame is due every 50 ms. The sink takes 120 ms
        // per present, so the player must fall behind and drop.
        let frames: Vec<Frame> = (0..6)
            .map(|i| solid_frame(4, 2, i * 50, 100, 128, 128))
            .collect();
        let mut player = VideoPlayer::new(SlowSink::new(120), 20, 4, 2).unwrap();
        let report = player.play_frames(&frames).unwrap();
        assert_eq!(report.total, 6);
        assert_eq!(report.presented + report.dropped, 6);
        assert!(report.dropped >= 1, "slow sink must force drops");
        assert!(report.presented >= 1, "at least the first frame shows");
        // Presented pts are strictly increasing: never out of order.
        let pts = &player.sink.presented;
        assert!(pts.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn player_drops_out_of_order_frames() {
        // pts 25 arrives after pts 50: it is older than what was shown.
        let frames = vec![
            solid_frame(4, 2, 0, 100, 128, 128),
            solid_frame(4, 2, 50, 100, 128, 128),
            solid_frame(4, 2, 25, 100, 128, 128),
        ];
        let mut player = VideoPlayer::new(RecordingSink::new(), 20, 4, 2).unwrap();
        let report = player.play_frames(&frames).unwrap();
        assert_eq!(report.presented, 2);
        assert_eq!(report.dropped, 1);
        let pts: Vec<u64> = player.sink.presented.iter().map(|f| f.pts_ms()).collect();
        assert_eq!(pts, vec![0, 50]);
    }

    #[test]
    fn player_pause_freezes_the_clock_and_resume_continues() {
        let frames: Vec<Frame> = (0..3)
            .map(|i| solid_frame(4, 2, i * 33, 100, 128, 128))
            .collect();
        let mut player = VideoPlayer::new(RecordingSink::new(), 30, 4, 2).unwrap();
        let control = player.control();
        control.pause();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            control.resume();
        });
        let started = Instant::now();
        let report = player.play_frames(&frames).unwrap();
        let elapsed = started.elapsed();
        handle.join().unwrap();
        assert_eq!(report.presented, 3);
        assert_eq!(report.dropped, 0);
        // Playback could not finish before the pause lifted.
        assert!(elapsed >= Duration::from_millis(90));
    }

    #[test]
    fn player_stop_before_play_shows_nothing() {
        let frames: Vec<Frame> = (0..5)
            .map(|i| solid_frame(4, 2, i * 33, 100, 128, 128))
            .collect();
        let mut player = VideoPlayer::new(RecordingSink::new(), 30, 4, 2).unwrap();
        player.control().stop();
        let report = player.play_frames(&frames).unwrap();
        assert_eq!(report.presented, 0);
        assert_eq!(report.dropped, 0);
        assert_eq!(report.total, 5);
        assert_eq!(report.duration_ms, 0);
    }

    #[test]
    fn player_rejects_bad_config() {
        assert!(VideoPlayer::new(RecordingSink::new(), 0, 4, 2).is_err());
        assert!(VideoPlayer::new(RecordingSink::new(), 1001, 4, 2).is_err());
        assert!(VideoPlayer::new(RecordingSink::new(), 30, 0, 2).is_err());
        assert!(VideoPlayer::new(RecordingSink::new(), 30, 4, 0).is_err());
        assert!(VideoPlayer::new(RecordingSink::new(), 30, 4, 2).is_ok());
    }

    #[test]
    fn ppm_sink_writes_valid_files_that_decode_back() {
        let dir = test_dir("ppm");
        let black = solid_frame(4, 2, 0, 16, 128, 128);
        let white = solid_frame(4, 2, 33, 235, 128, 128);
        let stream = encode_stream(&[black.clone(), white.clone()]);
        let report = play_stream_to_dir(&stream, 30, &dir).unwrap();
        assert_eq!(report.presented, 2);
        assert_eq!(report.dropped, 0);
        // Two files, in presentation order.
        let (w0, h0, rgb0) = read_ppm(&dir.join("frame_000000.ppm"));
        let (w1, h1, rgb1) = read_ppm(&dir.join("frame_000001.ppm"));
        assert_eq!((w0, h0), (4, 2));
        assert_eq!((w1, h1), (4, 2));
        // Pixels match the conversion exactly.
        assert_eq!(rgb0, yuv420_to_rgb(&black).data().to_vec());
        assert_eq!(rgb1, yuv420_to_rgb(&white).data().to_vec());
        assert!(rgb0.iter().all(|b| *b == 0));
        assert!(rgb1.iter().all(|b| *b == 255));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn play_stream_to_dir_scales_frames() {
        let dir = test_dir("scaled");
        let frame = solid_frame(4, 4, 0, 76, 85, 255); // pure red
        let stream = encode_stream(&[frame]);
        let report = play_stream_to_dir_scaled(&stream, 30, &dir, 2, 2).unwrap();
        assert_eq!(report.presented, 1);
        let (w, h, rgb) = read_ppm(&dir.join("frame_000000.ppm"));
        assert_eq!((w, h), (2, 2));
        // Still red after scaling.
        assert_eq!(rgb, vec![255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn play_stream_to_dir_empty_stream_is_an_empty_report() {
        let dir = test_dir("empty");
        let report = play_stream_to_dir(b"", 30, &dir).unwrap();
        assert_eq!(report.presented, 0);
        assert_eq!(report.dropped, 0);
        assert_eq!(report.total, 0);
        assert_eq!(report.duration_ms, 0);
        // The directory exists but holds no frames.
        assert!(dir.is_dir());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn play_stream_to_dir_refuses_a_bad_stream() {
        let dir = test_dir("bad");
        let err = play_stream_to_dir(b"KVFR", 30, &dir).unwrap_err();
        assert!(matches!(err, PlaybackError::Truncated(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
