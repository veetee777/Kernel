//! Sealed sessions after a completed handshake, with rekeying (Phase 2).
//!
//! A Data frame body is nonce (24) || XChaCha20-Poly1305(seq (8, BE) || payload)
//! with a 16-byte tag. The 12-byte frame header is the associated data, so the
//! frame kind and length are authenticated too.
//!
//! REKEY (initiator only). Two Rekey frames, both sealed under the keys in
//! force when they are sent:
//!
//!   I -> R  request:  tag 1 || eph_i   (under the old keys)
//!   R -> I  response: tag 2 || eph_r   (under the old keys)
//!
//! Both sides then compute shared = X25519(eph_i, eph_r) and
//!   th      = SHA-256("KCAR-REKEY-v1\0" || request_frame || eph_r)
//!   i2r, r2i = HKDF-SHA256(salt = th, ikm = shared, info = label)
//!
//! The responder switches keys right after sending its response. The initiator
//! switches when it reads the response. While its rekey is pending, the
//! initiator sends nothing new: payloads are queued, then sealed under the new
//! keys, in order, when the rekey completes. TCP keeps each direction in order,
//! so no frame is sealed under keys the peer has already dropped.
//!
//! Why the rekey messages are not signed: they travel inside the session, so
//! the current keys already authenticate them. Only a holder of the current keys
//! can send a valid request or response. The ephemeral Diffie-Hellman keeps the
//! new keys secret from anyone who later learns the old ones. Signing would need
//! the identity key inside Session, which this design avoids.
//!
//! Sequence numbers and the replay window continue across rekeys. The old keys
//! are dropped, and zeroized, at the switch.

use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use hkdf::Hkdf;
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::error::CarrierError;
use crate::frame::{self, Kind, HEADER_LEN};
use crate::kex::{self, Secret};

const NONCE_LEN: usize = 24;
const SEQ_LEN: usize = 8;
const TAG_LEN: usize = 16;
const WINDOW: u64 = 64;

/// Data frames per key epoch before a rekey is due, by default.
pub const REKEY_AFTER_FRAMES: u64 = 1 << 20;
/// Payloads queued while a rekey is pending.
pub const MAX_QUEUED: usize = 256;

const TAG_REKEY_REQ: u8 = 1;
const TAG_REKEY_RESP: u8 = 2;
const REKEY_BODY: usize = 33;
const LABEL_REKEY_TH: &[u8] = b"KCAR-REKEY-v1\0";
const LABEL_REKEY_I2R: &[u8] = b"KCAR-REKEY-v1 key i2r";
const LABEL_REKEY_R2I: &[u8] = b"KCAR-REKEY-v1 key r2i";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Initiator,
    Responder,
}

/// What `handle_incoming` produced.
pub enum Event {
    /// An application payload.
    Data(Vec<u8>),
    /// Frames the caller must write to the wire, in order. May be empty.
    Write(Vec<Vec<u8>>),
}

struct Pending {
    eph: Secret,
    request: Vec<u8>,
}

/// One established session. Not Debug: it holds key material.
pub struct Session {
    role: Role,
    send_key: Secret,
    recv_key: Secret,
    send_seq: u64,
    window: ReplayWindow,
    peer: [u8; 32],
    frames_this_epoch: u64,
    max_frames: u64,
    pending: Option<Pending>,
    queued: Vec<Vec<u8>>,
}

impl Session {
    pub(crate) fn new(role: Role, i2r: Secret, r2i: Secret, peer: [u8; 32]) -> Session {
        let (send_key, recv_key) = directional(role, i2r, r2i);
        Session {
            role,
            send_key,
            recv_key,
            send_seq: 0,
            window: ReplayWindow::default(),
            peer,
            frames_this_epoch: 0,
            max_frames: REKEY_AFTER_FRAMES,
            pending: None,
            queued: Vec::new(),
        }
    }

    /// The Ed25519 public key the handshake authenticated for the other side.
    pub fn peer(&self) -> [u8; 32] {
        self.peer
    }

    pub fn is_initiator(&self) -> bool {
        self.role == Role::Initiator
    }

    pub fn rekey_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Frames per epoch before a rekey is due. Owner-tunable.
    pub fn set_rekey_policy(&mut self, max_frames: u64) {
        self.max_frames = max_frames.max(1);
    }

    pub fn needs_rekey(&self) -> bool {
        self.frames_this_epoch >= self.max_frames
    }

    /// Seal one Data frame directly. Refused while a rekey is pending: use `send`.
    pub fn seal(&mut self, payload: &[u8]) -> Result<Vec<u8>, CarrierError> {
        if self.pending.is_some() {
            return Err(CarrierError::RekeyPending);
        }
        self.seal_kind(Kind::Data, payload)
    }

    /// Send one payload. Returns the frames to write, in order. While a rekey is
    /// pending, the payload is queued and no frame is returned.
    pub fn send(&mut self, payload: &[u8]) -> Result<Vec<Vec<u8>>, CarrierError> {
        if self.pending.is_some() {
            if self.queued.len() >= MAX_QUEUED {
                return Err(CarrierError::QueueFull);
            }
            self.queued.push(payload.to_vec());
            return Ok(Vec::new());
        }
        Ok(vec![self.seal_kind(Kind::Data, payload)?])
    }

    /// Open one Data frame. Refuses any other kind before decrypting.
    pub fn open(&mut self, frame_bytes: &[u8]) -> Result<Vec<u8>, CarrierError> {
        let (header, _) = frame::decode(frame_bytes)?;
        if header.kind != Kind::Data {
            return Err(CarrierError::UnexpectedKind);
        }
        self.decrypt(frame_bytes)
    }

    /// Start a rekey. Initiator only. Returns the request frame to write.
    pub fn begin_rekey(&mut self) -> Result<Vec<u8>, CarrierError> {
        if self.role != Role::Initiator {
            return Err(CarrierError::UnexpectedKind);
        }
        if self.pending.is_some() {
            return Err(CarrierError::RekeyPending);
        }
        let (eph, eph_pub) = kex::ephemeral()?;
        let request = self.seal_kind(Kind::Rekey, &rekey_body(TAG_REKEY_REQ, &eph_pub))?;
        self.pending = Some(Pending {
            eph,
            request: request.clone(),
        });
        Ok(request)
    }

    /// Process one incoming frame: a Data payload, or a rekey step that may
    /// require frames to be written back.
    pub fn handle_incoming(&mut self, frame_bytes: &[u8]) -> Result<Event, CarrierError> {
        let (header, _) = frame::decode(frame_bytes)?;
        match header.kind {
            Kind::Data => Ok(Event::Data(self.decrypt(frame_bytes)?)),
            Kind::Rekey => {
                // An initiator accepts a Rekey frame only as the answer to its own
                // request. Refused before decrypting, so nothing advances.
                if self.role == Role::Initiator && self.pending.is_none() {
                    return Err(CarrierError::UnexpectedKind);
                }
                let payload = self.decrypt(frame_bytes)?;
                if self.pending.is_some() {
                    self.finish_rekey(&payload)
                } else {
                    self.answer_rekey(frame_bytes, &payload)
                }
            }
            _ => Err(CarrierError::UnexpectedKind),
        }
    }

    /// Responder: answer a rekey request. The response is sealed under the old keys,
    /// and the keys switch only after it is sealed.
    fn answer_rekey(&mut self, request_frame: &[u8], payload: &[u8]) -> Result<Event, CarrierError> {
        let initiator_eph = rekey_key(payload, TAG_REKEY_REQ)?;
        let (eph, eph_pub) = kex::ephemeral()?;
        let shared = kex::shared(&eph, &initiator_eph)?;
        let th = rekey_transcript(request_frame, &eph_pub);
        let (i2r, r2i) = derive_rekey(&shared, &th)?;
        let response = self.seal_kind(Kind::Rekey, &rekey_body(TAG_REKEY_RESP, &eph_pub))?;
        self.switch(i2r, r2i);
        Ok(Event::Write(vec![response]))
    }

    /// Initiator: complete a rekey from the responder's answer, then flush the
    /// payloads queued while it was pending, sealed under the new keys.
    fn finish_rekey(&mut self, payload: &[u8]) -> Result<Event, CarrierError> {
        let responder_eph = rekey_key(payload, TAG_REKEY_RESP)?;
        let (shared, request) = {
            let p = self.pending.as_ref().ok_or(CarrierError::Internal)?;
            (kex::shared(&p.eph, &responder_eph)?, p.request.clone())
        };
        let th = rekey_transcript(&request, &responder_eph);
        let (i2r, r2i) = derive_rekey(&shared, &th)?;
        self.pending = None;
        self.switch(i2r, r2i);
        let queued = std::mem::take(&mut self.queued);
        let mut frames = Vec::with_capacity(queued.len());
        for p in queued {
            frames.push(self.seal_kind(Kind::Data, &p)?);
        }
        Ok(Event::Write(frames))
    }

    fn switch(&mut self, i2r: Secret, r2i: Secret) {
        let (send_key, recv_key) = directional(self.role, i2r, r2i);
        self.send_key = send_key;
        self.recv_key = recv_key;
        self.frames_this_epoch = 0;
    }

    fn seal_kind(&mut self, kind: Kind, payload: &[u8]) -> Result<Vec<u8>, CarrierError> {
        let seq = self.send_seq;
        let next = seq.checked_add(1).ok_or(CarrierError::SequenceExhausted)?;
        let body_len = NONCE_LEN + SEQ_LEN + payload.len() + TAG_LEN;
        let header = frame::header_bytes(kind, body_len)?;

        let mut nonce = [0u8; NONCE_LEN];
        getrandom::getrandom(&mut nonce).map_err(|_| CarrierError::Random)?;

        let mut plain = Zeroizing::new(Vec::with_capacity(SEQ_LEN + payload.len()));
        plain.extend_from_slice(&seq.to_be_bytes());
        plain.extend_from_slice(payload);

        let cipher = XChaCha20Poly1305::new_from_slice(&self.send_key[..])
            .map_err(|_| CarrierError::Internal)?;
        let ct = cipher
            .encrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: &plain[..],
                    aad: &header[..],
                },
            )
            .map_err(|_| CarrierError::Internal)?;

        let mut out = Vec::with_capacity(HEADER_LEN + body_len);
        out.extend_from_slice(&header);
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&ct);
        self.send_seq = next;
        if kind == Kind::Data {
            self.frames_this_epoch += 1;
        }
        Ok(out)
    }

    /// Authenticate and decrypt one frame under the current receive key, then check
    /// the replay window. The window moves only after the tag verifies.
    fn decrypt(&mut self, frame_bytes: &[u8]) -> Result<Vec<u8>, CarrierError> {
        let (header, body) = frame::decode(frame_bytes)?;
        if body.len() < NONCE_LEN + SEQ_LEN + TAG_LEN {
            return Err(CarrierError::MalformedFrame);
        }
        let (nonce, ct) = body.split_at(NONCE_LEN);
        let cipher = XChaCha20Poly1305::new_from_slice(&self.recv_key[..])
            .map_err(|_| CarrierError::Internal)?;
        let plain = Zeroizing::new(
            cipher
                .decrypt(
                    XNonce::from_slice(nonce),
                    Payload {
                        msg: ct,
                        aad: &header.raw()[..],
                    },
                )
                .map_err(|_| CarrierError::AuthFailed)?,
        );
        let seq_bytes: [u8; SEQ_LEN] = plain[..SEQ_LEN]
            .try_into()
            .map_err(|_| CarrierError::MalformedFrame)?;
        self.window.accept(u64::from_be_bytes(seq_bytes))?;
        Ok(plain[SEQ_LEN..].to_vec())
    }
}

fn directional(role: Role, i2r: Secret, r2i: Secret) -> (Secret, Secret) {
    match role {
        Role::Initiator => (i2r, r2i),
        Role::Responder => (r2i, i2r),
    }
}

fn arr32(b: &[u8]) -> Result<[u8; 32], CarrierError> {
    b.try_into().map_err(|_| CarrierError::MalformedFrame)
}

/// Extract the ephemeral key from a rekey payload with the expected tag.
fn rekey_key(payload: &[u8], tag: u8) -> Result<[u8; 32], CarrierError> {
    if payload.len() != REKEY_BODY || payload[0] != tag {
        return Err(CarrierError::UnexpectedKind);
    }
    arr32(&payload[1..])
}

fn rekey_body(tag: u8, key: &[u8; 32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(REKEY_BODY);
    v.push(tag);
    v.extend_from_slice(key);
    v
}

fn rekey_transcript(request: &[u8], responder_eph: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(LABEL_REKEY_TH);
    h.update(request);
    h.update(responder_eph);
    h.finalize().into()
}

fn derive_rekey(shared: &[u8; 32], th: &[u8; 32]) -> Result<(Secret, Secret), CarrierError> {
    let hk = Hkdf::<Sha256>::new(Some(&th[..]), &shared[..]);
    let mut i2r = Zeroizing::new([0u8; 32]);
    let mut r2i = Zeroizing::new([0u8; 32]);
    hk.expand(LABEL_REKEY_I2R, &mut i2r[..])
        .map_err(|_| CarrierError::Internal)?;
    hk.expand(LABEL_REKEY_R2I, &mut r2i[..])
        .map_err(|_| CarrierError::Internal)?;
    Ok((i2r, r2i))
}

/// Sliding window of 64 sequence numbers. Accepts each number at most once.
#[derive(Debug, Default)]
struct ReplayWindow {
    seen_any: bool,
    highest: u64,
    /// Bit i set means sequence (highest - i) was accepted.
    bitmap: u64,
}

impl ReplayWindow {
    fn accept(&mut self, seq: u64) -> Result<(), CarrierError> {
        if !self.seen_any {
            self.seen_any = true;
            self.highest = seq;
            self.bitmap = 1;
            return Ok(());
        }
        if seq > self.highest {
            let shift = seq - self.highest;
            self.bitmap = if shift >= WINDOW { 0 } else { self.bitmap << shift };
            self.bitmap |= 1;
            self.highest = seq;
            return Ok(());
        }
        let behind = self.highest - seq;
        if behind >= WINDOW {
            return Err(CarrierError::TooOld);
        }
        let bit = 1u64 << behind;
        if self.bitmap & bit != 0 {
            return Err(CarrierError::Replay);
        }
        self.bitmap |= bit;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair() -> (Session, Session) {
        let a: Secret = Zeroizing::new([7u8; 32]);
        let b: Secret = Zeroizing::new([9u8; 32]);
        (
            Session::new(Role::Initiator, a.clone(), b.clone(), [1u8; 32]),
            Session::new(Role::Responder, a, b, [2u8; 32]),
        )
    }

    fn one(v: Vec<Vec<u8>>) -> Vec<u8> {
        assert_eq!(v.len(), 1);
        v.into_iter().next().expect("one frame")
    }

    fn data(e: Event) -> Vec<u8> {
        match e {
            Event::Data(p) => p,
            Event::Write(_) => panic!("expected data"),
        }
    }

    fn writes(e: Event) -> Vec<Vec<u8>> {
        match e {
            Event::Write(v) => v,
            Event::Data(_) => panic!("expected frames to write"),
        }
    }

    #[test]
    fn round_trip_both_directions() {
        let (mut i, mut r) = pair();
        let f = i.seal(b"ping").expect("seal");
        assert_eq!(r.open(&f).expect("open"), b"ping".to_vec());
        let g = r.seal(b"pong").expect("seal");
        assert_eq!(i.open(&g).expect("open"), b"pong".to_vec());
    }

    #[test]
    fn empty_payload_round_trips() {
        let (mut i, mut r) = pair();
        let f = i.seal(b"").expect("seal");
        assert_eq!(r.open(&f).expect("open"), Vec::<u8>::new());
    }

    #[test]
    fn peer_identity_is_reported() {
        let (i, r) = pair();
        assert_eq!(i.peer(), [1u8; 32]);
        assert_eq!(r.peer(), [2u8; 32]);
    }

    #[test]
    fn tampered_ciphertext_is_refused() {
        let (mut i, mut r) = pair();
        let mut f = i.seal(b"secret").expect("seal");
        let last = f.len() - 1;
        f[last] ^= 0x01;
        assert_eq!(r.open(&f).err().expect("refuse"), CarrierError::AuthFailed);
    }

    #[test]
    fn altered_kind_is_refused_before_decryption() {
        let (mut i, mut r) = pair();
        let mut f = i.seal(b"x").expect("seal");
        f[5] = Kind::Hello as u8;
        assert_eq!(r.open(&f).err().expect("refuse"), CarrierError::UnexpectedKind);
    }

    #[test]
    fn session_with_other_keys_cannot_open() {
        let (mut i, _) = pair();
        let x: Secret = Zeroizing::new([3u8; 32]);
        let y: Secret = Zeroizing::new([4u8; 32]);
        let mut other = Session::new(Role::Responder, x, y, [0u8; 32]);
        let f = i.seal(b"x").expect("seal");
        assert_eq!(other.open(&f).err().expect("refuse"), CarrierError::AuthFailed);
    }

    #[test]
    fn replayed_frame_is_refused() {
        let (mut i, mut r) = pair();
        let f = i.seal(b"once").expect("seal");
        r.open(&f).expect("first open");
        assert_eq!(r.open(&f).err().expect("refuse"), CarrierError::Replay);
    }

    #[test]
    fn reordering_within_window_is_accepted() {
        let (mut i, mut r) = pair();
        let f0 = i.seal(b"0").expect("seal");
        let f1 = i.seal(b"1").expect("seal");
        let f2 = i.seal(b"2").expect("seal");
        r.open(&f2).expect("2 first");
        r.open(&f0).expect("0 late, inside window");
        r.open(&f1).expect("1 late, inside window");
    }

    #[test]
    fn frame_older_than_window_is_refused() {
        let (mut i, mut r) = pair();
        let first = i.seal(b"first").expect("seal");
        for _ in 0..70 {
            let f = i.seal(b"filler").expect("seal");
            r.open(&f).expect("open");
        }
        assert_eq!(r.open(&first).err().expect("refuse"), CarrierError::TooOld);
    }

    #[test]
    fn oversize_payload_is_refused_before_encryption() {
        let (mut i, _) = pair();
        let big = vec![0u8; frame::MAX_BODY];
        assert_eq!(i.seal(&big).err().expect("refuse"), CarrierError::Oversize);
    }

    #[test]
    fn replay_window_accepts_once_and_tolerates_reorder() {
        let mut w = ReplayWindow::default();
        w.accept(5).expect("first");
        w.accept(3).expect("older, inside window");
        w.accept(7).expect("newer");
        assert_eq!(w.accept(3).err().expect("replay"), CarrierError::Replay);
        assert_eq!(w.accept(5).err().expect("replay"), CarrierError::Replay);
    }

    #[test]
    fn replay_window_edges() {
        let mut w = ReplayWindow::default();
        w.accept(100).expect("first");
        // 100 - 36 = 64: one past the 64-entry window.
        assert_eq!(w.accept(36).err().expect("too old"), CarrierError::TooOld);
        // 100 - 37 = 63: the last slot inside the window.
        w.accept(37).expect("last slot inside window");
    }

    #[test]
    fn rekey_completes_and_data_flows_both_ways() {
        let (mut i, mut r) = pair();
        let before = one(i.send(b"before").expect("send"));
        assert_eq!(data(r.handle_incoming(&before).expect("recv")), b"before".to_vec());

        let request = i.begin_rekey().expect("begin");
        let response = writes(r.handle_incoming(&request).expect("answer"));
        assert_eq!(response.len(), 1);
        assert!(writes(i.handle_incoming(&response[0]).expect("complete")).is_empty());
        assert!(!i.rekey_pending());

        let after = one(i.send(b"after").expect("send"));
        assert_eq!(data(r.handle_incoming(&after).expect("recv")), b"after".to_vec());
        let back = one(r.send(b"back").expect("send"));
        assert_eq!(data(i.handle_incoming(&back).expect("recv")), b"back".to_vec());
    }

    #[test]
    fn payloads_sent_during_rekey_are_queued_and_arrive_once_in_order() {
        let (mut i, mut r) = pair();
        let request = i.begin_rekey().expect("begin");
        assert!(i.send(b"q1").expect("queue").is_empty());
        assert!(i.send(b"q2").expect("queue").is_empty());
        let response = writes(r.handle_incoming(&request).expect("answer"));
        let flushed = writes(i.handle_incoming(&response[0]).expect("complete"));
        assert_eq!(flushed.len(), 2);
        assert_eq!(data(r.handle_incoming(&flushed[0]).expect("q1")), b"q1".to_vec());
        assert_eq!(data(r.handle_incoming(&flushed[1]).expect("q2")), b"q2".to_vec());
        // A flushed frame delivered again is a replay.
        assert_eq!(
            r.handle_incoming(&flushed[0]).err().expect("replay"),
            CarrierError::Replay
        );
    }

    #[test]
    fn frames_from_the_old_epoch_do_not_open_after_the_rekey() {
        let (mut i, mut r) = pair();
        // Sealed under the old keys, never delivered before the rekey.
        let old = one(i.send(b"old").expect("send"));
        let request = i.begin_rekey().expect("begin");
        let response = writes(r.handle_incoming(&request).expect("answer"));
        i.handle_incoming(&response[0]).expect("complete");
        assert_eq!(
            r.handle_incoming(&old).err().expect("old keys"),
            CarrierError::AuthFailed
        );
    }

    #[test]
    fn replayed_rekey_request_is_refused() {
        let (mut i, mut r) = pair();
        let request = i.begin_rekey().expect("begin");
        r.handle_incoming(&request).expect("first answer");
        // The responder has switched keys, so the old request no longer authenticates.
        assert_eq!(
            r.handle_incoming(&request).err().expect("replay"),
            CarrierError::AuthFailed
        );
    }

    #[test]
    fn tampered_rekey_response_is_refused_and_leaves_rekey_pending() {
        let (mut i, mut r) = pair();
        let request = i.begin_rekey().expect("begin");
        let mut response = one(writes(r.handle_incoming(&request).expect("answer")));
        let last = response.len() - 1;
        response[last] ^= 1;
        assert_eq!(
            i.handle_incoming(&response).err().expect("refuse"),
            CarrierError::AuthFailed
        );
        assert!(i.rekey_pending());
    }

    #[test]
    fn responder_cannot_begin_rekey() {
        let (_, mut r) = pair();
        assert_eq!(r.begin_rekey().err().expect("refuse"), CarrierError::UnexpectedKind);
    }

    #[test]
    fn initiator_refuses_an_unsolicited_rekey_frame() {
        let (mut i, mut r) = pair();
        let forged = r
            .seal_kind(Kind::Rekey, &rekey_body(TAG_REKEY_REQ, &[9u8; 32]))
            .expect("seal");
        assert_eq!(
            i.handle_incoming(&forged).err().expect("refuse"),
            CarrierError::UnexpectedKind
        );
    }

    #[test]
    fn relabelled_frame_kind_fails_authentication() {
        let (mut i, mut r) = pair();
        let mut f = one(i.send(b"x").expect("send"));
        f[5] = Kind::Rekey as u8;
        assert_eq!(
            r.handle_incoming(&f).err().expect("refuse"),
            CarrierError::AuthFailed
        );
    }

    #[test]
    fn needs_rekey_follows_the_policy_and_resets_after_a_rekey() {
        let (mut i, mut r) = pair();
        i.set_rekey_policy(3);
        for _ in 0..2 {
            i.send(b"x").expect("send");
        }
        assert!(!i.needs_rekey());
        i.send(b"x").expect("send");
        assert!(i.needs_rekey());

        let request = i.begin_rekey().expect("begin");
        let response = writes(r.handle_incoming(&request).expect("answer"));
        i.handle_incoming(&response[0]).expect("complete");
        assert!(!i.needs_rekey());
    }

    #[test]
    fn seal_is_refused_while_a_rekey_is_pending() {
        let (mut i, _) = pair();
        i.begin_rekey().expect("begin");
        assert_eq!(i.seal(b"x").err().expect("refuse"), CarrierError::RekeyPending);
    }

    #[test]
    fn the_rekey_queue_is_bounded() {
        let (mut i, _) = pair();
        i.begin_rekey().expect("begin");
        for _ in 0..MAX_QUEUED {
            i.send(b"x").expect("queued");
        }
        assert_eq!(i.send(b"x").err().expect("full"), CarrierError::QueueFull);
    }
}
