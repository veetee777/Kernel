//! Refusal codes. Every refusal names one code. A refusal never advances
//! sequence numbers, pins, or session state.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarrierError {
    /// Structurally invalid bytes: bad magic, nonzero reserved bytes, or a
    /// length that does not match the buffer.
    MalformedFrame,
    /// Frame version is not one this build speaks.
    UnsupportedVersion,
    /// Unknown frame kind, or a frame that arrived at the wrong step.
    UnexpectedKind,
    /// Body is larger than MAX_BODY.
    Oversize,
    /// Peer key is not pinned.
    UnpinnedPeer,
    /// Peer key was revoked. Revocation is sticky.
    RevokedPeer,
    /// Signature did not verify under the pinned key.
    BadSignature,
    /// Ed25519 public key is not a valid curve point.
    InvalidKey,
    /// Peer's ephemeral key is small-order, so the shared secret is all zero.
    WeakSecret,
    /// The Response came from a key other than the one the initiator expected.
    WrongResponder,
    /// AEAD open failed: tampered bytes, wrong key, or altered header.
    AuthFailed,
    /// Sequence number was already accepted.
    Replay,
    /// Sequence number is older than the replay window.
    TooOld,
    /// Send counter is exhausted. Re-key before sending again.
    SequenceExhausted,
    /// The operating system random source failed.
    Random,
    /// Phase 2: the cookie challenge failed — a cookie the responder never
    /// issued, or a puzzle answer that does not solve.
    BadCookie,
    /// Phase 2: a rekey was requested while one is already pending.
    RekeyPending,
    /// Phase 2: the rekey queue is full; the peer must wait for the rekey
    /// to finish before more payloads are queued.
    QueueFull,
    /// An internal invariant failed. This should never happen; treat it as a bug.
    Internal,
}

impl CarrierError {
    /// Stable snake_case refusal code, for logs and tests.
    pub fn code(self) -> &'static str {
        match self {
            CarrierError::MalformedFrame => "malformed_frame",
            CarrierError::UnsupportedVersion => "unsupported_version",
            CarrierError::UnexpectedKind => "unexpected_kind",
            CarrierError::Oversize => "oversize",
            CarrierError::UnpinnedPeer => "unpinned_peer",
            CarrierError::RevokedPeer => "revoked_peer",
            CarrierError::BadSignature => "bad_signature",
            CarrierError::InvalidKey => "invalid_key",
            CarrierError::WeakSecret => "weak_secret",
            CarrierError::WrongResponder => "wrong_responder",
            CarrierError::AuthFailed => "auth_failed",
            CarrierError::Replay => "replay",
            CarrierError::TooOld => "too_old",
            CarrierError::SequenceExhausted => "sequence_exhausted",
            CarrierError::Random => "random",
            CarrierError::BadCookie => "bad_cookie",
            CarrierError::RekeyPending => "rekey_pending",
            CarrierError::QueueFull => "queue_full",
            CarrierError::Internal => "internal",
        }
    }
}

impl fmt::Display for CarrierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for CarrierError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase2_codes_are_stable_snake_case() {
        assert_eq!(CarrierError::BadCookie.code(), "bad_cookie");
        assert_eq!(CarrierError::RekeyPending.code(), "rekey_pending");
        assert_eq!(CarrierError::QueueFull.code(), "queue_full");
        assert_eq!(CarrierError::BadCookie.to_string(), "bad_cookie");
    }
}
