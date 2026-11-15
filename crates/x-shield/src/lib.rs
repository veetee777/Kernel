//! X•Shield — node security for KERNEL.
//!
//! A shield decides, for every incoming peer, whether that peer may
//! talk to this node. The policy is default-deny and fully
//! deterministic: the same shield state and the same inputs always
//! produce the same decision, so two nodes holding the same lists
//! agree about who is welcome.
//!
//! * **Block list** — peers that are refused outright. A block beats
//!   an allow entry and beats any attestation.
//! * **Allow list** — peers that are welcome, subject to rate limits.
//!   Each entry carries the human name the owner knows the peer by.
//! * **Attestations** — a peer on the allow list can vouch for another
//!   peer by signing a statement about them. An attested peer is
//!   treated like an allowed peer (still rate-limited). An attestation
//!   whose signature does not verify is rejected outright and never
//!   stored.
//! * **Rate limiting** — at most `max_messages` messages per peer per
//!   `window_secs` seconds. The window is a sliding window over
//!   explicit timestamps, so tests are deterministic and do not depend
//!   on wall-clock sleeps.
//!
//! Decisions are [`Decision::Allowed`], [`Decision::Blocked`] (with a
//! human-readable reason), or [`Decision::RateLimited`].
//!
//! Attestation format (explicit binary, no JSON):
//! `XSHIELD1` + voucher public key (32) + subject public key (32) +
//! note length (u16 LE) + note bytes + signature (64).

use std::collections::{HashMap, VecDeque};
use std::fmt;

use kernel_crypto::Ed25519Signature;
use kernel_identity::{Identity, IdentityError, PeerId};
use kernel_node::Node;

/// Magic bytes at the start of every attestation.
const ATTEST_MAGIC: &[u8; 8] = b"XSHIELD1";

/// Domain separator for attestation signatures: a signature covers
/// `b"XSHIELD1" || voucher key || subject key || note`.
const ATTEST_SIGN_DOMAIN: &[u8] = b"XSHIELD1";

/// Default rate limit: 10 messages per peer per 60-second window.
pub const DEFAULT_MAX_MESSAGES: u32 = 10;
/// Default rate-limit window in seconds.
pub const DEFAULT_WINDOW_SECS: u64 = 60;

/// Errors returned by shield operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShieldError {
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// An attestation's signature did not verify against its voucher.
    BadAttestation,
    /// A note broke the naming rules.
    InvalidNote(String),
    /// An attestation was malformed.
    InvalidAttestation(String),
}

impl fmt::Display for ShieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShieldError::Identity(e) => write!(f, "identity error: {e}"),
            ShieldError::BadAttestation => write!(f, "attestation signature is invalid"),
            ShieldError::InvalidNote(msg) => write!(f, "invalid attestation note: {msg}"),
            ShieldError::InvalidAttestation(msg) => {
                write!(f, "invalid attestation: {msg}")
            }
        }
    }
}

impl std::error::Error for ShieldError {}

impl From<IdentityError> for ShieldError {
    fn from(e: IdentityError) -> Self {
        ShieldError::Identity(e)
    }
}

/// Rules for attestation notes: 1-128 printable characters.
fn validate_note(note: &str) -> Result<(), ShieldError> {
    if note.is_empty() {
        return Err(ShieldError::InvalidNote(
            "attestation note must not be empty".to_string(),
        ));
    }
    if note.len() > 128 {
        return Err(ShieldError::InvalidNote(
            "attestation note must be at most 128 characters".to_string(),
        ));
    }
    if !note.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
        return Err(ShieldError::InvalidNote(
            "attestation note must be printable ASCII".to_string(),
        ));
    }
    Ok(())
}

/// The message a voucher signs for an attestation: domain + voucher
/// key + subject key + note.
fn attestation_sign_message(voucher: &PeerId, subject: &PeerId, note: &str) -> Vec<u8> {
    let mut msg = Vec::with_capacity(ATTEST_SIGN_DOMAIN.len() + 32 + 32 + note.len());
    msg.extend_from_slice(ATTEST_SIGN_DOMAIN);
    msg.extend_from_slice(voucher.public_key().as_bytes());
    msg.extend_from_slice(subject.public_key().as_bytes());
    msg.extend_from_slice(note.as_bytes());
    msg
}

/// A signed attestation: one peer vouches for another.
/// The voucher's signature proves they said it; the subject is the
/// peer being vouched for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attestation {
    /// The peer doing the vouching (must be allow-listed to count).
    pub voucher: PeerId,
    /// The peer being vouched for.
    pub subject: PeerId,
    /// A human-readable note, e.g. `"met at the mesh meetup"`.
    pub note: String,
    /// The voucher's signature over voucher, subject, and note.
    pub signature: Ed25519Signature,
}

impl Attestation {
    /// Create and sign an attestation as `voucher` about `subject`.
    pub fn new(
        voucher: &Identity,
        subject: &PeerId,
        note: &str,
    ) -> Result<Self, ShieldError> {
        validate_note(note)?;
        let voucher_id = voucher.peer_id().clone();
        let msg = attestation_sign_message(&voucher_id, subject, note);
        let signature = voucher.sign(&msg);
        Ok(Self {
            voucher: voucher_id,
            subject: subject.clone(),
            note: note.to_string(),
            signature,
        })
    }

    /// True if the signature verifies against the voucher for this
    /// exact subject and note. Any tampering fails this check.
    pub fn verify(&self) -> bool {
        let msg = attestation_sign_message(&self.voucher, &self.subject, &self.note);
        self.voucher.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary attestation format.
    pub fn encode(&self) -> Vec<u8> {
        let note_bytes = self.note.as_bytes();
        let mut out = Vec::with_capacity(8 + 32 + 32 + 2 + note_bytes.len() + 64);
        out.extend_from_slice(ATTEST_MAGIC);
        out.extend_from_slice(self.voucher.public_key().as_bytes());
        out.extend_from_slice(self.subject.public_key().as_bytes());
        out.extend_from_slice(&(note_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(note_bytes);
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode an attestation produced by [`Attestation::encode`].
    /// Rejects anything that is not exactly that format.
    pub fn decode(bytes: &[u8]) -> Result<Self, ShieldError> {
        if bytes.len() < 8 + 32 + 32 + 2 + 64 || &bytes[..8] != ATTEST_MAGIC {
            return Err(ShieldError::InvalidAttestation(
                "missing XSHIELD1 magic".to_string(),
            ));
        }
        let note_len =
            u16::from_le_bytes(bytes[72..74].try_into().expect("2-byte slice")) as usize;
        let expected = 8 + 32 + 32 + 2 + note_len + 64;
        if bytes.len() != expected {
            return Err(ShieldError::InvalidAttestation(format!(
                "attestation is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let mut off = 8;
        let mut voucher_bytes = [0u8; 32];
        voucher_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let voucher_key = kernel_crypto::Ed25519PublicKey::from_bytes(&voucher_bytes)
            .map_err(|e| ShieldError::InvalidAttestation(e.to_string()))?;
        let mut subject_bytes = [0u8; 32];
        subject_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let subject_key = kernel_crypto::Ed25519PublicKey::from_bytes(&subject_bytes)
            .map_err(|e| ShieldError::InvalidAttestation(e.to_string()))?;
        let note_len =
            u16::from_le_bytes(bytes[off..off + 2].try_into().expect("2-byte slice")) as usize;
        off += 2;
        let note = String::from_utf8(bytes[off..off + note_len].to_vec()).map_err(|_| {
            ShieldError::InvalidAttestation("attestation note is not valid UTF-8".to_string())
        })?;
        validate_note(&note)?;
        off += note_len;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        // The attestation stores public keys, not display names:
        // verification needs the keys. Decode rebuilds the PeerIds
        // with deterministic names derived from the keys.
        let voucher = PeerId::new(voucher_key, &fingerprint_name(&voucher_key))?;
        let subject = PeerId::new(subject_key, &fingerprint_name(&subject_key))?;
        Ok(Self {
            voucher,
            subject,
            note,
            signature,
        })
    }
}

/// A display name derived from a public key: the first 8 hex chars of
/// the key. Used when an attestation is decoded and only the key
/// survives (verification needs the key, not the human name).
fn fingerprint_name(key: &kernel_crypto::Ed25519PublicKey) -> String {
    let hex = kernel_crypto::to_hex(key.as_bytes());
    format!("peer-{}", &hex[..8])
}

/// The outcome of a shield check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// The peer may talk to this node.
    Allowed,
    /// The peer is refused, with a human-readable reason.
    Blocked(String),
    /// The peer is welcome but has sent too many messages too fast.
    RateLimited,
}

impl Decision {
    /// True only for [`Decision::Allowed`].
    pub fn is_allowed(&self) -> bool {
        matches!(self, Decision::Allowed)
    }
}

impl fmt::Display for Decision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Decision::Allowed => write!(f, "allowed"),
            Decision::Blocked(reason) => write!(f, "blocked: {reason}"),
            Decision::RateLimited => write!(f, "rate limited"),
        }
    }
}

/// A per-peer sliding-window rate limiter over explicit timestamps.
/// `check` returns true (and records the message) while the peer is
/// under the limit, false once they have sent `max_messages` inside
/// the last `window_secs` seconds. Timestamps are seconds since the
/// Unix epoch, supplied by the caller so behavior is deterministic.
struct RateLimiter {
    max_messages: u32,
    window_secs: u64,
    /// Message timestamps per peer public key, oldest first.
    history: HashMap<[u8; 32], VecDeque<u64>>,
}

impl RateLimiter {
    fn new(max_messages: u32, window_secs: u64) -> Self {
        Self {
            max_messages,
            window_secs,
            history: HashMap::new(),
        }
    }

    fn check(&mut self, key: &[u8; 32], now: u64) -> bool {
        let window_start = now.saturating_sub(self.window_secs);
        let queue = self.history.entry(*key).or_default();
        // Drop timestamps outside the window. A timestamp equal to
        // the window start is outside (the window is (start, now]).
        while let Some(&oldest) = queue.front() {
            if oldest <= window_start {
                queue.pop_front();
            } else {
                break;
            }
        }
        if queue.len() as u32 >= self.max_messages {
            return false;
        }
        queue.push_back(now);
        true
    }
}

/// A shield: allow list, block list, attestations, and a rate limiter.
pub struct Shield {
    /// Allowed peers, by public key, with the human name the owner
    /// knows them by.
    allow: HashMap<[u8; 32], String>,
    /// Blocked peers, by public key, with a human-readable reason.
    block: HashMap<[u8; 32], String>,
    /// Attestations held, in the order they were added.
    attestations: Vec<Attestation>,
    limiter: RateLimiter,
}

impl Shield {
    /// An empty shield with the default rate limit
    /// ([`DEFAULT_MAX_MESSAGES`] per [`DEFAULT_WINDOW_SECS`] seconds).
    pub fn new() -> Self {
        Self::with_limits(DEFAULT_MAX_MESSAGES, DEFAULT_WINDOW_SECS)
    }

    /// An empty shield with a custom rate limit.
    pub fn with_limits(max_messages: u32, window_secs: u64) -> Self {
        Self {
            allow: HashMap::new(),
            block: HashMap::new(),
            attestations: Vec::new(),
            limiter: RateLimiter::new(max_messages, window_secs),
        }
    }

    /// Add a peer to the allow list under the given human name.
    /// A peer already on the block list stays blocked: block beats
    /// allow, so adding them here does not unblock them.
    pub fn allow_peer(&mut self, peer: &PeerId, name: &str) {
        self.allow
            .insert(*peer.public_key().as_bytes(), name.to_string());
    }

    /// Remove a peer from the allow list. Returns true if they were
    /// listed.
    pub fn unallow_peer(&mut self, peer: &PeerId) -> bool {
        self.allow.remove(peer.public_key().as_bytes()).is_some()
    }

    /// Add a peer to the block list with a human-readable reason.
    /// A block beats an allow entry and beats any attestation.
    pub fn block_peer(&mut self, peer: &PeerId, reason: &str) {
        self.block
            .insert(*peer.public_key().as_bytes(), reason.to_string());
    }

    /// Remove a peer from the block list. Returns true if they were
    /// listed.
    pub fn unblock_peer(&mut self, peer: &PeerId) -> bool {
        self.block.remove(peer.public_key().as_bytes()).is_some()
    }

    /// True if the peer is on the allow list.
    pub fn is_allowed_peer(&self, peer: &PeerId) -> bool {
        self.allow.contains_key(peer.public_key().as_bytes())
    }

    /// True if the peer is on the block list.
    pub fn is_blocked_peer(&self, peer: &PeerId) -> bool {
        self.block.contains_key(peer.public_key().as_bytes())
    }

    /// The block reason for a peer, if they are blocked.
    pub fn block_reason(&self, peer: &PeerId) -> Option<&str> {
        self.block
            .get(peer.public_key().as_bytes())
            .map(String::as_str)
    }

    /// Number of peers on the allow list.
    pub fn allow_count(&self) -> usize {
        self.allow.len()
    }

    /// Number of peers on the block list.
    pub fn block_count(&self) -> usize {
        self.block.len()
    }

    /// Number of attestations held.
    pub fn attestation_count(&self) -> usize {
        self.attestations.len()
    }

    /// Store an attestation. Its signature is verified first; a bad
    /// attestation is rejected and never stored. Returns true if the
    /// attestation was new, false if an identical one was already held.
    pub fn add_attestation(&mut self, attestation: Attestation) -> Result<bool, ShieldError> {
        if !attestation.verify() {
            return Err(ShieldError::BadAttestation);
        }
        if self.attestations.contains(&attestation) {
            return Ok(false);
        }
        self.attestations.push(attestation);
        Ok(true)
    }

    /// True if `peer` has a valid attestation from a peer who is
    /// currently on the allow list (and not on the block list).
    /// Attestations from blocked or unknown vouchers do not count.
    fn attested_by_allowed(&self, peer: &PeerId) -> bool {
        let subject_key = peer.public_key().as_bytes();
        self.attestations.iter().any(|att| {
            att.subject.public_key().as_bytes() == subject_key
                && att.verify()
                && self.allow.contains_key(att.voucher.public_key().as_bytes())
                && !self.block.contains_key(att.voucher.public_key().as_bytes())
        })
    }

    /// Decide whether an incoming message from `peer` at time
    /// `now_secs` (seconds since the Unix epoch) may be accepted.
    /// Blocked peers are refused; unknown peers are refused
    /// (default-deny); allowed and attested peers pass subject to the
    /// rate limit. Allowed messages are recorded against the rate
    /// limit; blocked and rate-limited messages are not.
    pub fn check(&mut self, peer: &PeerId, now_secs: u64) -> Decision {
        let key = peer.public_key().as_bytes();
        if let Some(reason) = self.block.get(key) {
            return Decision::Blocked(format!("peer is blocked: {reason}"));
        }
        let trusted = self.allow.contains_key(key) || self.attested_by_allowed(peer);
        if !trusted {
            return Decision::Blocked("unknown peer: not allowed and not attested".to_string());
        }
        if self.limiter.check(key, now_secs) {
            Decision::Allowed
        } else {
            Decision::RateLimited
        }
    }

    /// Decide whether an incoming message from a [`Node`] may be
    /// accepted, using the node's peer id.
    pub fn check_node(&mut self, node: &Node, now_secs: u64) -> Decision {
        self.check(node.peer_id(), now_secs)
    }
}

impl Default for Shield {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attestation_encode_decode_round_trip_and_verify() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let att = Attestation::new(&alice, bob.peer_id(), "met at the mesh meetup").unwrap();
        assert!(att.verify());
        let encoded = att.encode();
        let decoded = Attestation::decode(&encoded).unwrap();
        assert_eq!(decoded.note, "met at the mesh meetup");
        assert_eq!(
            decoded.voucher.public_key(),
            alice.peer_id().public_key()
        );
        assert_eq!(
            decoded.subject.public_key(),
            bob.peer_id().public_key()
        );
        assert_eq!(decoded.signature, att.signature);
        assert!(decoded.verify());
        assert!(Attestation::decode(b"garbage").is_err());
        assert!(Attestation::decode(&encoded[..encoded.len() - 1]).is_err());
    }

    #[test]
    fn allowed_peer_passes_and_unknown_peer_is_blocked() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let carol = Identity::generate("carol").unwrap();
        let mut shield = Shield::new();
        shield.allow_peer(alice.peer_id(), "alice");
        assert!(shield.check(alice.peer_id(), 1000).is_allowed());
        assert_eq!(
            shield.check(bob.peer_id(), 1000),
            Decision::Blocked("unknown peer: not allowed and not attested".to_string())
        );
        assert!(shield.check(carol.peer_id(), 1000).is_allowed() == false);
        // Allow-listing Bob lets him through.
        shield.allow_peer(bob.peer_id(), "bob");
        assert!(shield.check(bob.peer_id(), 1001).is_allowed());
    }

    #[test]
    fn blocked_peer_is_rejected_and_block_beats_allow() {
        let alice = Identity::generate("alice").unwrap();
        let mut shield = Shield::new();
        shield.allow_peer(alice.peer_id(), "alice");
        assert!(shield.check(alice.peer_id(), 1000).is_allowed());
        shield.block_peer(alice.peer_id(), "spamming");
        assert_eq!(
            shield.check(alice.peer_id(), 1001),
            Decision::Blocked("peer is blocked: spamming".to_string())
        );
        // Unblocking restores the allow entry underneath.
        assert!(shield.unblock_peer(alice.peer_id()));
        assert!(shield.check(alice.peer_id(), 1002).is_allowed());
        // Unallowing removes the welcome entirely.
        assert!(shield.unallow_peer(alice.peer_id()));
        assert!(!shield.check(alice.peer_id(), 1003).is_allowed());
    }

    #[test]
    fn attestation_from_allowed_peer_opens_the_door() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let carol = Identity::generate("carol").unwrap();
        let mut shield = Shield::new();
        shield.allow_peer(alice.peer_id(), "alice");
        // Carol is unknown: blocked.
        assert!(!shield.check(carol.peer_id(), 1000).is_allowed());
        // Alice vouches for Carol.
        let att = Attestation::new(&alice, carol.peer_id(), "vouched").unwrap();
        assert!(shield.add_attestation(att).unwrap());
        assert_eq!(shield.attestation_count(), 1);
        assert!(shield.check(carol.peer_id(), 1001).is_allowed());
        // But an attestation from Bob (not allow-listed) does nothing.
        let dave = Identity::generate("dave").unwrap();
        let att2 = Attestation::new(&bob, dave.peer_id(), "bob vouches").unwrap();
        assert!(shield.add_attestation(att2).unwrap());
        assert!(!shield.check(dave.peer_id(), 1002).is_allowed());
    }

    #[test]
    fn bad_attestation_is_rejected_and_never_stored() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut shield = Shield::new();
        shield.allow_peer(alice.peer_id(), "alice");
        let mut att = Attestation::new(&alice, bob.peer_id(), "vouched").unwrap();
        // Tamper with the note after signing.
        att.note = "forged note".to_string();
        assert!(!att.verify());
        assert_eq!(
            shield.add_attestation(att),
            Err(ShieldError::BadAttestation)
        );
        assert_eq!(shield.attestation_count(), 0);
        // Bob is still unknown and blocked.
        assert!(!shield.check(bob.peer_id(), 1000).is_allowed());
        // Tampering with the subject also breaks it.
        let carol = Identity::generate("carol").unwrap();
        let mut att2 = Attestation::new(&alice, bob.peer_id(), "vouched").unwrap();
        att2.subject = carol.peer_id().clone();
        assert!(!att2.verify());
        assert_eq!(
            shield.add_attestation(att2),
            Err(ShieldError::BadAttestation)
        );
    }

    #[test]
    fn rate_limit_triggers_after_configured_count_and_resets_after_window() {
        let alice = Identity::generate("alice").unwrap();
        let mut shield = Shield::with_limits(3, 60);
        shield.allow_peer(alice.peer_id(), "alice");
        // Three messages inside the window pass.
        assert!(shield.check(alice.peer_id(), 1000).is_allowed());
        assert!(shield.check(alice.peer_id(), 1000).is_allowed());
        assert!(shield.check(alice.peer_id(), 1000).is_allowed());
        // The fourth, still inside the window, is rate-limited.
        assert_eq!(
            shield.check(alice.peer_id(), 1000),
            Decision::RateLimited
        );
        // A fifth at the same time is still limited.
        assert_eq!(
            shield.check(alice.peer_id(), 1059),
            Decision::RateLimited
        );
        // After the window slides past (1000 + 60 = 1060), the old
        // messages fall out and the peer may talk again.
        assert!(shield.check(alice.peer_id(), 1060).is_allowed());
    }

    #[test]
    fn rate_limit_is_per_peer() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut shield = Shield::with_limits(2, 60);
        shield.allow_peer(alice.peer_id(), "alice");
        shield.allow_peer(bob.peer_id(), "bob");
        assert!(shield.check(alice.peer_id(), 1000).is_allowed());
        assert!(shield.check(alice.peer_id(), 1000).is_allowed());
        assert_eq!(
            shield.check(alice.peer_id(), 1000),
            Decision::RateLimited
        );
        // Bob has his own budget and is unaffected.
        assert!(shield.check(bob.peer_id(), 1000).is_allowed());
        assert!(shield.check(bob.peer_id(), 1000).is_allowed());
    }

    #[test]
    fn decisions_are_deterministic() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        // Two shields with identical setup must decide identically
        // for the same sequence of inputs.
        let mut a = Shield::with_limits(2, 60);
        let mut b = Shield::with_limits(2, 60);
        for shield in [&mut a, &mut b] {
            shield.allow_peer(alice.peer_id(), "alice");
            shield.block_peer(bob.peer_id(), "bad actor");
        }
        let inputs = [1000, 1000, 1000, 1061, 1061, 2000];
        let mut decisions_a = Vec::new();
        let mut decisions_b = Vec::new();
        for now in inputs {
            decisions_a.push(a.check(alice.peer_id(), now));
            decisions_b.push(b.check(alice.peer_id(), now));
            decisions_a.push(a.check(bob.peer_id(), now));
            decisions_b.push(b.check(bob.peer_id(), now));
        }
        assert_eq!(decisions_a, decisions_b);
        // And the sequence is what the policy predicts: two allowed,
        // one rate-limited, then allowed again after the window.
        assert_eq!(decisions_a[0], Decision::Allowed);
        assert_eq!(decisions_a[2], Decision::Allowed);
        assert_eq!(decisions_a[4], Decision::RateLimited);
        assert_eq!(decisions_a[6], Decision::Allowed);
        assert!(matches!(decisions_a[1], Decision::Blocked(_)));
    }

    #[test]
    fn check_node_uses_the_nodes_peer_id() {
        let node = Node::new("tristan").unwrap();
        let stranger = Node::new("stranger").unwrap();
        let mut shield = Shield::new();
        shield.allow_peer(node.peer_id(), "tristan");
        assert!(shield.check_node(&node, 1000).is_allowed());
        assert!(!shield.check_node(&stranger, 1000).is_allowed());
    }

    #[test]
    fn attestation_note_rules() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        assert!(Attestation::new(&alice, bob.peer_id(), "").is_err());
        assert!(Attestation::new(&alice, bob.peer_id(), "ok note").is_ok());
        assert!(Attestation::new(&alice, bob.peer_id(), "a".repeat(129).as_str()).is_err());
    }
}
