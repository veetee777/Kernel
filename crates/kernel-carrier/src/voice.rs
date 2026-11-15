//! Voice over a carrier session (Phase 2).
//!
//! CODEC: MockCodec only. The lockfile has no pure-Rust Opus, so none was
//! added, and no C dependency either. MockCodec sends raw 16-bit PCM: 20 ms at
//! 8 kHz, 160 samples, 320 bytes. It has no compression and no audio quality.
//! It exercises framing, timing, and call setup. The real codec goes behind the
//! Codec trait later, under a new codec id.
//!
//! LATENCY: ITU-T G.114 sets 150 ms one way for the whole path: network, jitter
//! buffer, and codec together. A 150 ms jitter buffer alone would use the whole
//! budget. The default target is 60 ms, sized to absorb jitter. Choose the
//! target from measured jitter, not from the budget.
//!
//! WIRE: a voice message is a Data payload whose first byte is a voice tag.
//! Other traffic on the same session must not start with 0xA1 to 0xA4.
//!
//! ```text
//! 0xA1  Invite  codec(1)
//! 0xA2  Accept  codec(1)
//! 0xA3  Audio   seq(4, BE) ts_ms(4, BE) codec frame
//! 0xA4  End
//! ```
//!
//! Timestamps are u32 milliseconds, and the jitter arithmetic treats them as
//! plain differences, so one call must stay under about 49 days.

use std::collections::BTreeMap;
use std::fmt;

use crate::transport::{SecureChannel, TransportError};

pub const TAG_INVITE: u8 = 0xA1;
pub const TAG_ACCEPT: u8 = 0xA2;
pub const TAG_AUDIO: u8 = 0xA3;
pub const TAG_END: u8 = 0xA4;

pub const CODEC_MOCK: u8 = 0x00;
pub const FRAME_MS: u32 = 20;
pub const DEFAULT_JITTER_TARGET_MS: u64 = 60;
pub const MAX_JITTER_FRAMES: usize = 64;

const MOCK_SAMPLES: usize = 160;
const AUDIO_HEADER: usize = 9;

#[derive(Debug)]
pub enum VoiceError {
    Transport(TransportError),
    /// A payload did not parse as a voice message.
    Malformed,
    /// A voice tag arrived where another was required.
    UnexpectedTag(u8),
    /// The codec id offered and the codec id supported do not match.
    CodecMismatch(u8),
    /// The peer declined the call.
    Declined,
    /// A codec frame had the wrong length (samples or bytes, per the codec).
    BadFrameLength { expected: usize, got: usize },
}

impl From<TransportError> for VoiceError {
    fn from(e: TransportError) -> Self {
        VoiceError::Transport(e)
    }
}

impl fmt::Display for VoiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VoiceError::Transport(e) => write!(f, "transport: {}", e),
            VoiceError::Malformed => f.write_str("malformed voice payload"),
            VoiceError::UnexpectedTag(t) => write!(f, "unexpected voice tag {:#04x}", t),
            VoiceError::CodecMismatch(c) => write!(f, "codec {:#04x} not supported", c),
            VoiceError::Declined => f.write_str("call declined"),
            VoiceError::BadFrameLength { expected, got } => {
                write!(f, "frame length {} (expected {})", got, expected)
            }
        }
    }
}

impl std::error::Error for VoiceError {}

/// A codec turns PCM frames into bytes and back.
pub trait Codec {
    fn id(&self) -> u8;
    /// PCM samples per frame. Fixed for the codec.
    fn frame_len(&self) -> usize;
    fn encode(&mut self, pcm: &[i16]) -> Result<Vec<u8>, VoiceError>;
    fn decode(&mut self, bytes: &[u8]) -> Result<Vec<i16>, VoiceError>;
}

/// Raw 16-bit little-endian PCM, one 20 ms frame of 160 samples. No compression
/// and no audio quality. Exists to exercise framing, timing, and call setup.
pub struct MockCodec;

impl Codec for MockCodec {
    fn id(&self) -> u8 {
        CODEC_MOCK
    }

    fn frame_len(&self) -> usize {
        MOCK_SAMPLES
    }

    fn encode(&mut self, pcm: &[i16]) -> Result<Vec<u8>, VoiceError> {
        if pcm.len() != MOCK_SAMPLES {
            return Err(VoiceError::BadFrameLength {
                expected: MOCK_SAMPLES,
                got: pcm.len(),
            });
        }
        let mut out = Vec::with_capacity(MOCK_SAMPLES * 2);
        for s in pcm {
            out.extend_from_slice(&s.to_le_bytes());
        }
        Ok(out)
    }

    fn decode(&mut self, bytes: &[u8]) -> Result<Vec<i16>, VoiceError> {
        if bytes.len() != MOCK_SAMPLES * 2 {
            return Err(VoiceError::BadFrameLength {
                expected: MOCK_SAMPLES * 2,
                got: bytes.len(),
            });
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect())
    }
}

/// What `JitterBuffer::push` did with a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushOutcome {
    Accepted,
    /// Arrived after its playout time, or behind the playout cursor.
    Late,
    Duplicate,
    /// The buffer is full.
    Overflow,
}

/// One step of playout.
#[derive(Debug, PartialEq, Eq)]
pub enum Playout {
    Frame { seq: u32, data: Vec<u8> },
    /// The frame never arrived in time.
    Lost { seq: u32 },
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct JitterStats {
    pub late: u64,
    pub duplicate: u64,
    pub overflow: u64,
    pub lost: u64,
}

/// Reorders frames and releases them on a fixed schedule.
///
/// The playout time of a frame is the arrival time of the first frame, plus
/// the difference of timestamps, plus the target delay. Time is an explicit
/// argument, so the buffer is deterministic under test.
pub struct JitterBuffer {
    target_ms: u64,
    anchor: Option<(u32, u64)>,
    next_seq: Option<u32>,
    pending: BTreeMap<u32, (u32, Vec<u8>)>,
    stats: JitterStats,
}

impl JitterBuffer {
    pub fn new(target_ms: u64) -> Self {
        Self {
            target_ms,
            anchor: None,
            next_seq: None,
            pending: BTreeMap::new(),
            stats: JitterStats::default(),
        }
    }

    pub fn stats(&self) -> JitterStats {
        self.stats
    }

    /// Playout time, on the local clock, of a frame with timestamp `ts`.
    fn play_at(&self, ts: u32) -> i64 {
        let (base_ts, base_arrival) = self.anchor.unwrap_or((ts, 0));
        base_arrival as i64 + (ts as i64 - base_ts as i64) + self.target_ms as i64
    }

    pub fn push(&mut self, seq: u32, ts: u32, data: Vec<u8>, arrival_ms: u64) -> PushOutcome {
        if self.anchor.is_none() {
            self.anchor = Some((ts, arrival_ms));
        }
        if let Some(next) = self.next_seq {
            if seq < next {
                self.stats.late += 1;
                return PushOutcome::Late;
            }
        }
        if arrival_ms as i64 > self.play_at(ts) {
            self.stats.late += 1;
            return PushOutcome::Late;
        }
        if self.pending.contains_key(&seq) {
            self.stats.duplicate += 1;
            return PushOutcome::Duplicate;
        }
        if self.pending.len() >= MAX_JITTER_FRAMES {
            self.stats.overflow += 1;
            return PushOutcome::Overflow;
        }
        self.pending.insert(seq, (ts, data));
        PushOutcome::Accepted
    }

    /// Release every frame whose playout time has come, in sequence order.
    pub fn pop_ready(&mut self, now_ms: u64) -> Vec<Playout> {
        let now = now_ms as i64;
        let mut out = Vec::new();
        loop {
            let next = match self.next_seq {
                Some(n) => n,
                None => match self.pending.keys().next() {
                    Some(&first) => {
                        self.next_seq = Some(first);
                        first
                    }
                    None => break,
                },
            };
            match self.pending.get(&next).map(|(ts, _)| *ts) {
                Some(ts) => {
                    if self.play_at(ts) > now {
                        break;
                    }
                    if let Some((_, data)) = self.pending.remove(&next) {
                        out.push(Playout::Frame { seq: next, data });
                    }
                    self.next_seq = Some(next.wrapping_add(1));
                }
                None => {
                    // The head frame is missing. It is lost once a later frame is
                    // due: that frame has a later timestamp, so the missing frame's
                    // playout time has already passed.
                    let later = self.pending.iter().next().map(|(s, v)| (*s, v.0));
                    match later {
                        Some((s, ts)) if self.play_at(ts) <= now => {
                            let gap = s.wrapping_sub(next);
                            self.stats.lost += u64::from(gap);
                            for i in 0..gap.min(MAX_JITTER_FRAMES as u32) {
                                out.push(Playout::Lost {
                                    seq: next.wrapping_add(i),
                                });
                            }
                            self.next_seq = Some(s);
                        }
                        _ => break,
                    }
                }
            }
        }
        out
    }
}

/// What a voice payload meant.
#[derive(Debug, PartialEq, Eq)]
pub enum Signal {
    Audio,
    Ended,
}

/// Process one voice payload. Audio goes into the jitter buffer. `arrival_ms`
/// is the local clock at arrival.
pub fn handle_payload(
    payload: &[u8],
    jitter: &mut JitterBuffer,
    arrival_ms: u64,
) -> Result<Signal, VoiceError> {
    match payload.first() {
        Some(&TAG_AUDIO) => {
            if payload.len() < AUDIO_HEADER {
                return Err(VoiceError::Malformed);
            }
            let seq = u32::from_be_bytes(
                payload[1..5].try_into().map_err(|_| VoiceError::Malformed)?,
            );
            let ts = u32::from_be_bytes(
                payload[5..9].try_into().map_err(|_| VoiceError::Malformed)?,
            );
            let _ = jitter.push(seq, ts, payload[AUDIO_HEADER..].to_vec(), arrival_ms);
            Ok(Signal::Audio)
        }
        Some(&TAG_END) => Ok(Signal::Ended),
        Some(&t) => Err(VoiceError::UnexpectedTag(t)),
        None => Err(VoiceError::Malformed),
    }
}

/// Receive one voice payload from the channel and process it.
pub fn receive(
    chan: &mut SecureChannel,
    jitter: &mut JitterBuffer,
    arrival_ms: u64,
) -> Result<Signal, VoiceError> {
    let payload = chan.recv()?;
    handle_payload(&payload, jitter, arrival_ms)
}

/// Caller side of call setup: offer a codec and wait for the answer.
pub fn start_call(chan: &mut SecureChannel, codec_id: u8) -> Result<(), VoiceError> {
    chan.send(&[TAG_INVITE, codec_id])?;
    let reply = chan.recv()?;
    match reply.as_slice() {
        [TAG_ACCEPT, c] if *c == codec_id => Ok(()),
        [TAG_ACCEPT, c] => Err(VoiceError::CodecMismatch(*c)),
        [TAG_END] => Err(VoiceError::Declined),
        [t, ..] => Err(VoiceError::UnexpectedTag(*t)),
        [] => Err(VoiceError::Malformed),
    }
}

/// Callee side of call setup: take an invite and answer it if the codec matches.
pub fn accept_call(chan: &mut SecureChannel, supported: u8) -> Result<(), VoiceError> {
    let invite = chan.recv()?;
    match invite.as_slice() {
        [TAG_INVITE, c] if *c == supported => {
            chan.send(&[TAG_ACCEPT, supported])?;
            Ok(())
        }
        [TAG_INVITE, c] => {
            let theirs = *c;
            chan.send(&[TAG_END])?;
            Err(VoiceError::CodecMismatch(theirs))
        }
        [t, ..] => Err(VoiceError::UnexpectedTag(*t)),
        [] => Err(VoiceError::Malformed),
    }
}

/// Send one audio frame.
pub fn send_frame(
    chan: &mut SecureChannel,
    seq: u32,
    ts_ms: u32,
    codec_frame: &[u8],
) -> Result<(), VoiceError> {
    let mut p = Vec::with_capacity(AUDIO_HEADER + codec_frame.len());
    p.push(TAG_AUDIO);
    p.extend_from_slice(&seq.to_be_bytes());
    p.extend_from_slice(&ts_ms.to_be_bytes());
    p.extend_from_slice(codec_frame);
    chan.send(&p)?;
    Ok(())
}

/// End the call. The session stays open for other traffic.
pub fn end_call(chan: &mut SecureChannel) -> Result<(), VoiceError> {
    chan.send(&[TAG_END])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pins::PinSet;
    use crate::transport::{accept_channel, connect_channel};
    use ed25519_dalek::SigningKey;
    use std::net::TcpListener;
    use std::thread;
    use std::time::Duration;

    const T: Duration = Duration::from_secs(5);

    fn pcm(v: i16) -> Vec<i16> {
        vec![v; MOCK_SAMPLES]
    }

    fn f(seq: u32, b: u8) -> Playout {
        Playout::Frame {
            seq,
            data: vec![b],
        }
    }

    #[test]
    fn mock_codec_round_trips_a_frame() {
        let mut c = MockCodec;
        let bytes = c.encode(&pcm(7)).expect("encode");
        assert_eq!(bytes.len(), MOCK_SAMPLES * 2);
        assert_eq!(c.decode(&bytes).expect("decode"), pcm(7));
    }

    #[test]
    fn mock_codec_refuses_a_wrong_length() {
        let mut c = MockCodec;
        assert!(matches!(
            c.encode(&[0; 10]),
            Err(VoiceError::BadFrameLength { .. })
        ));
    }

    #[test]
    fn in_order_frames_play_in_order() {
        let mut j = JitterBuffer::new(60);
        for seq in 0..3u32 {
            let arrival = 100 + 20 * u64::from(seq);
            assert_eq!(
                j.push(seq, seq * 20, vec![seq as u8], arrival),
                PushOutcome::Accepted
            );
        }
        assert_eq!(j.pop_ready(1000), vec![f(0, 0), f(1, 1), f(2, 2)]);
    }

    #[test]
    fn reordered_frames_within_target_play_in_order() {
        let mut j = JitterBuffer::new(60);
        j.push(0, 0, vec![0], 100); // plays at 160
        j.push(2, 40, vec![2], 150); // plays at 200
        j.push(1, 20, vec![1], 160); // plays at 180
        assert_eq!(j.pop_ready(170), vec![f(0, 0)]);
        assert_eq!(j.pop_ready(250), vec![f(1, 1), f(2, 2)]);
    }

    #[test]
    fn frame_arriving_after_its_play_time_is_dropped() {
        let mut j = JitterBuffer::new(60);
        j.push(0, 0, vec![0], 100); // plays at 160
        assert_eq!(j.push(1, 20, vec![1], 200), PushOutcome::Late); // plays at 180
        j.push(2, 40, vec![2], 150); // plays at 200
        assert_eq!(
            j.pop_ready(300),
            vec![f(0, 0), Playout::Lost { seq: 1 }, f(2, 2)]
        );
        assert_eq!(j.stats().late, 1);
        assert_eq!(j.stats().lost, 1);
    }

    #[test]
    fn duplicate_frames_are_dropped() {
        let mut j = JitterBuffer::new(60);
        j.push(0, 0, vec![0], 100);
        assert_eq!(j.push(0, 0, vec![0], 101), PushOutcome::Duplicate);
        assert_eq!(j.stats().duplicate, 1);
        assert_eq!(j.pop_ready(200), vec![f(0, 0)]);
    }

    #[test]
    fn a_missing_frame_is_lost_only_once_a_later_frame_is_due() {
        let mut j = JitterBuffer::new(60);
        j.push(0, 0, vec![0], 100); // plays at 160
        j.push(2, 40, vec![2], 140); // plays at 200
        // At 190 the gap is not yet known to be a loss.
        assert_eq!(j.pop_ready(190), vec![f(0, 0)]);
        assert_eq!(j.pop_ready(210), vec![Playout::Lost { seq: 1 }, f(2, 2)]);
    }

    #[test]
    fn buffer_is_bounded() {
        let mut j = JitterBuffer::new(60);
        for seq in 0..MAX_JITTER_FRAMES as u32 {
            // Arrives on time, and frame 0 holds the head so nothing plays.
            assert_eq!(
                j.push(seq, seq * 20, vec![0], 100 + 20 * u64::from(seq)),
                PushOutcome::Accepted
            );
        }
        let n = MAX_JITTER_FRAMES as u32;
        assert_eq!(
            j.push(n, n * 20, vec![0], 100 + 20 * u64::from(n)),
            PushOutcome::Overflow
        );
        assert_eq!(j.stats().overflow, 1);
    }

    #[test]
    fn voice_payloads_are_parsed() {
        let mut j = JitterBuffer::new(60);
        let mut p = vec![TAG_AUDIO];
        p.extend_from_slice(&5u32.to_be_bytes());
        p.extend_from_slice(&100u32.to_be_bytes());
        p.extend_from_slice(&[9, 9]);
        assert_eq!(handle_payload(&p, &mut j, 200).expect("audio"), Signal::Audio);
        assert_eq!(
            handle_payload(&[TAG_END], &mut j, 200).expect("end"),
            Signal::Ended
        );
        assert!(matches!(
            handle_payload(&[0x42], &mut j, 0),
            Err(VoiceError::UnexpectedTag(0x42))
        ));
        assert!(matches!(
            handle_payload(&[], &mut j, 0),
            Err(VoiceError::Malformed)
        ));
        assert!(matches!(
            handle_payload(&[TAG_AUDIO, 0], &mut j, 0),
            Err(VoiceError::Malformed)
        ));
    }

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn public(k: &SigningKey) -> [u8; 32] {
        k.verifying_key().to_bytes()
    }

    /// A connected pair over real loopback TCP: (caller, callee).
    fn pair() -> (SecureChannel, SecureChannel) {
        let a = key(1);
        let b = key(2);
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let mut pins_b = PinSet::new();
        pins_b.pin(public(&a)).expect("pin");
        let server = thread::spawn(move || accept_channel(&listener, &b, &pins_b, T));
        let mut pins_a = PinSet::new();
        pins_a.pin(public(&key(2))).expect("pin");
        let client = connect_channel(addr, &a, public(&key(2)), &pins_a, T).expect("connect");
        let server = server.join().expect("thread").expect("accept");
        (client, server)
    }

    #[test]
    fn a_call_sets_up_streams_audio_and_ends_over_tcp() {
        let (mut a, mut b) = pair();
        let callee = thread::spawn(move || {
            accept_call(&mut b, CODEC_MOCK).expect("accept_call");
            let mut jitter = JitterBuffer::new(DEFAULT_JITTER_TARGET_MS);
            let mut n: u64 = 0;
            loop {
                let arrival = 1000 + 20 * n;
                match receive(&mut b, &mut jitter, arrival).expect("receive") {
                    Signal::Audio => n += 1,
                    Signal::Ended => break,
                }
            }
            (b, jitter.pop_ready(1_000_000))
        });

        start_call(&mut a, CODEC_MOCK).expect("start_call");
        let mut codec = MockCodec;
        for seq in 0..10u32 {
            let frame = codec.encode(&pcm(seq as i16)).expect("encode");
            send_frame(&mut a, seq, seq * 20, &frame).expect("send");
        }
        end_call(&mut a).expect("end_call");

        let (mut b, played) = callee.join().expect("callee");
        let mut seen = Vec::new();
        for p in played {
            match p {
                Playout::Frame { seq, data } => {
                    seen.push(seq);
                    assert_eq!(codec.decode(&data).expect("decode"), pcm(seq as i16));
                }
                Playout::Lost { seq } => panic!("frame {seq} was lost"),
            }
        }
        assert_eq!(seen, (0..10u32).collect::<Vec<u32>>());

        // The session stays open after the call ends.
        a.send(b"text after the call").expect("send");
        assert_eq!(b.recv().expect("recv"), b"text after the call".to_vec());
    }

    #[test]
    fn a_codec_mismatch_is_refused_at_setup() {
        let (mut a, mut b) = pair();
        let callee = thread::spawn(move || accept_call(&mut b, 0x7F));
        let err = start_call(&mut a, CODEC_MOCK).err().expect("declined");
        assert!(matches!(err, VoiceError::Declined));
        let callee_err = callee.join().expect("callee").err().expect("refused");
        assert!(matches!(callee_err, VoiceError::CodecMismatch(CODEC_MOCK)));
    }

    #[test]
    fn voice_continues_across_a_rekey() {
        let (mut a, mut b) = pair();
        let callee = thread::spawn(move || {
            accept_call(&mut b, CODEC_MOCK).expect("accept_call");
            let mut jitter = JitterBuffer::new(DEFAULT_JITTER_TARGET_MS);
            let mut n: u64 = 0;
            loop {
                let arrival = 1000 + 20 * n;
                match receive(&mut b, &mut jitter, arrival).expect("receive") {
                    Signal::Audio => n += 1,
                    Signal::Ended => break,
                }
            }
            (jitter.pop_ready(1_000_000), n)
        });

        start_call(&mut a, CODEC_MOCK).expect("start_call");
        let mut codec = MockCodec;
        for seq in 0..3u32 {
            let frame = codec.encode(&pcm(seq as i16)).expect("encode");
            send_frame(&mut a, seq, seq * 20, &frame).expect("send");
        }
        a.start_rekey().expect("start_rekey");
        // Sent while the rekey is pending: queued, then sent under the new keys.
        for seq in 3..6u32 {
            let frame = codec.encode(&pcm(seq as i16)).expect("encode");
            send_frame(&mut a, seq, seq * 20, &frame).expect("queued send");
        }
        a.complete_rekey().expect("complete_rekey");
        end_call(&mut a).expect("end_call");

        let (played, n) = callee.join().expect("callee");
        assert_eq!(n, 6);
        let seqs: Vec<u32> = played
            .iter()
            .filter_map(|p| match p {
                Playout::Frame { seq, .. } => Some(*seq),
                Playout::Lost { .. } => None,
            })
            .collect();
        assert_eq!(seqs, (0..6u32).collect::<Vec<u32>>());
    }
}
