//! Allowances for X•Koin.
//!
//! An allowance is a signed record: an owner approves a spender to
//! draw up to an amount on their behalf, optionally until an expiry.
//! The spender draws by submitting a signed draw request; the ledger
//! moves the owner's coins to the recipient and decreases the
//! remaining allowance. Drawing more than the remainder is refused,
//! an expired allowance refuses, and the owner can revoke (the
//! remainder drops to zero). A draw can never move more than the
//! owner actually holds at draw time — the balance check runs on
//! every draw, not just at grant time.
//!
//! Allowances are local records, not smart contracts: they live in
//! this ledger, they are enforced by this ledger's code, and they
//! only exist while the ledger holds them.

use kernel_crypto::{sha256, to_hex, Ed25519Signature};
use kernel_identity::{Identity, PeerId};

use crate::{validate_memo, KoinError};

/// Magic bytes at the start of every allowance encoding.
const ALLOWANCE_MAGIC: &[u8; 8] = b"XKALLOW1";

/// Domain separator for allowance signatures: a signature covers
/// `b"XKOIN-ALLOW1" || owner key || spender key || amount || expiry`.
const ALLOWANCE_SIGN_DOMAIN: &[u8] = b"XKOIN-ALLOW1";

/// Domain separator for draw-request signatures: a signature covers
/// `b"XKOIN-DRAW1" || allowance id bytes || recipient key || memo
/// || amount || spender key`.
const DRAW_SIGN_DOMAIN: &[u8] = b"XKOIN-DRAW1";

/// A signed allowance: an owner approves a spender to draw up to
/// `amount` on their behalf, optionally until `expiry` (a
/// caller-supplied unix timestamp; 0 means no expiry).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Allowance {
    /// Who owns the coins.
    pub owner: PeerId,
    /// Who may draw them.
    pub spender: PeerId,
    /// The maximum total the spender may draw.
    pub amount: u64,
    /// Caller-supplied unix timestamp after which draws refuse;
    /// 0 means the allowance never expires.
    pub expiry: u64,
    /// The owner's signature over owner, spender, amount, expiry.
    pub signature: Ed25519Signature,
}

impl Allowance {
    /// Create and sign an allowance as `owner`. A zero amount is
    /// refused: an allowance that permits nothing is not an
    /// allowance.
    pub fn new(
        owner: &Identity,
        spender: &PeerId,
        amount: u64,
        expiry: u64,
    ) -> Result<Self, KoinError> {
        if amount == 0 {
            return Err(KoinError::ZeroAmount);
        }
        let owner_id = owner.peer_id().clone();
        let msg = allowance_sign_message(&owner_id, spender, amount, expiry);
        let signature = owner.sign(&msg);
        Ok(Self {
            owner: owner_id,
            spender: spender.clone(),
            amount,
            expiry,
            signature,
        })
    }

    /// True if the signature verifies against the owner for this
    /// exact spender, amount, and expiry.
    pub fn verify(&self) -> bool {
        let msg = allowance_sign_message(&self.owner, &self.spender, self.amount, self.expiry);
        self.owner.verify(&msg, &self.signature).is_ok()
    }

    /// The allowance's id: the SHA-256 of its canonical encoding,
    /// rendered as hex. A ledger accepts each id exactly once.
    pub fn id(&self) -> String {
        to_hex(&sha256(&self.encode()))
    }

    /// Encode to the explicit binary allowance format.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + 32 + 32 + 8 + 8 + 64);
        out.extend_from_slice(ALLOWANCE_MAGIC);
        out.extend_from_slice(self.owner.public_key().as_bytes());
        out.extend_from_slice(self.spender.public_key().as_bytes());
        out.extend_from_slice(&self.amount.to_le_bytes());
        out.extend_from_slice(&self.expiry.to_le_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode an allowance produced by [`Allowance::encode`].
    /// Rejects anything that is not exactly that format.
    pub fn decode(bytes: &[u8]) -> Result<Self, KoinError> {
        if bytes.len() != 8 + 32 + 32 + 8 + 8 + 64 || &bytes[..8] != ALLOWANCE_MAGIC {
            return Err(KoinError::InvalidAllowance(
                "missing XKALLOW1 magic or wrong length".to_string(),
            ));
        }
        let mut off = 8;
        let mut owner_bytes = [0u8; 32];
        owner_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let mut spender_bytes = [0u8; 32];
        spender_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let amount = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let expiry = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let owner_key = kernel_crypto::Ed25519PublicKey::from_bytes(&owner_bytes)
            .map_err(|e| KoinError::InvalidAllowance(e.to_string()))?;
        let spender_key = kernel_crypto::Ed25519PublicKey::from_bytes(&spender_bytes)
            .map_err(|e| KoinError::InvalidAllowance(e.to_string()))?;
        let owner = PeerId::new(owner_key, &crate::fingerprint_name(&owner_key))?;
        let spender = PeerId::new(spender_key, &crate::fingerprint_name(&spender_key))?;
        Ok(Self {
            owner,
            spender,
            amount,
            expiry,
            signature,
        })
    }
}

/// The message an owner signs for an allowance.
fn allowance_sign_message(owner: &PeerId, spender: &PeerId, amount: u64, expiry: u64) -> Vec<u8> {
    let mut msg = Vec::with_capacity(ALLOWANCE_SIGN_DOMAIN.len() + 32 + 32 + 8 + 8);
    msg.extend_from_slice(ALLOWANCE_SIGN_DOMAIN);
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg.extend_from_slice(spender.public_key().as_bytes());
    msg.extend_from_slice(&amount.to_le_bytes());
    msg.extend_from_slice(&expiry.to_le_bytes());
    msg
}

/// A signed request to draw from an allowance: the spender asks the
/// ledger to move `amount` of the allowance owner's coins to
/// `recipient`, with a memo.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllowanceDrawRequest {
    /// The allowance id to draw from.
    pub allowance_id: String,
    /// Who is drawing (must be the allowance's spender).
    pub spender: PeerId,
    /// Who receives the drawn coins.
    pub recipient: PeerId,
    /// A human-readable note on the draw.
    pub memo: String,
    /// How much to draw.
    pub amount: u64,
    /// The spender's signature over the draw.
    pub signature: Ed25519Signature,
}

impl AllowanceDrawRequest {
    /// Create and sign a draw request as `spender`.
    pub fn new(
        spender: &Identity,
        allowance_id: &str,
        recipient: &PeerId,
        memo: &str,
        amount: u64,
    ) -> Result<Self, KoinError> {
        validate_memo(memo)?;
        let spender_id = spender.peer_id().clone();
        let msg = draw_sign_message(&spender_id, allowance_id, recipient, memo, amount);
        let signature = spender.sign(&msg);
        Ok(Self {
            allowance_id: allowance_id.to_string(),
            spender: spender_id,
            recipient: recipient.clone(),
            memo: memo.to_string(),
            amount,
            signature,
        })
    }

    /// True if the signature verifies against the spender for this
    /// exact allowance, recipient, memo, and amount.
    pub fn verify(&self) -> bool {
        let msg = draw_sign_message(
            &self.spender,
            &self.allowance_id,
            &self.recipient,
            &self.memo,
            self.amount,
        );
        self.spender.verify(&msg, &self.signature).is_ok()
    }
}

/// The message a spender signs for a draw request.
fn draw_sign_message(
    spender: &PeerId,
    allowance_id: &str,
    recipient: &PeerId,
    memo: &str,
    amount: u64,
) -> Vec<u8> {
    let mut msg =
        Vec::with_capacity(DRAW_SIGN_DOMAIN.len() + allowance_id.len() + 32 + memo.len() + 8 + 32);
    msg.extend_from_slice(DRAW_SIGN_DOMAIN);
    msg.extend_from_slice(allowance_id.as_bytes());
    msg.extend_from_slice(recipient.public_key().as_bytes());
    msg.extend_from_slice(memo.as_bytes());
    msg.extend_from_slice(&amount.to_le_bytes());
    msg.extend_from_slice(spender.public_key().as_bytes());
    msg
}

/// A completed allowance draw, recorded by the ledger: which
/// allowance was drawn, who owned and spent, who received, and how
/// much. `seq` is the ledger's acceptance order, shared with
/// ordinary transfers so statements can merge both in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllowanceDraw {
    /// The allowance id that was drawn from.
    pub allowance_id: String,
    /// Who owned the coins.
    pub owner: PeerId,
    /// Who drew them.
    pub spender: PeerId,
    /// Who received them.
    pub recipient: PeerId,
    /// The memo on the draw.
    pub memo: String,
    /// How much moved.
    pub amount: u64,
    /// Acceptance order in the ledger.
    pub seq: u64,
}

impl AllowanceDraw {
    /// The draw's id: the SHA-256 of its canonical encoding,
    /// rendered as hex.
    pub fn id(&self) -> String {
        to_hex(&self.id_bytes())
    }

    /// The raw 32-byte draw id.
    pub fn id_bytes(&self) -> [u8; 32] {
        let mut msg =
            Vec::with_capacity(self.allowance_id.len() + 32 + 32 + 32 + self.memo.len() + 8 + 8);
        msg.extend_from_slice(self.allowance_id.as_bytes());
        msg.extend_from_slice(self.owner.public_key().as_bytes());
        msg.extend_from_slice(self.spender.public_key().as_bytes());
        msg.extend_from_slice(self.recipient.public_key().as_bytes());
        msg.extend_from_slice(self.memo.as_bytes());
        msg.extend_from_slice(&self.amount.to_le_bytes());
        msg.extend_from_slice(&self.seq.to_le_bytes());
        sha256(&msg)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identities() -> (Identity, Identity, Identity) {
        (
            Identity::generate("alice").unwrap(),
            Identity::generate("bob").unwrap(),
            Identity::generate("carol").unwrap(),
        )
    }

    #[test]
    fn allowance_verifies_and_tampering_fails() {
        let (alice, bob, _) = identities();
        let allowance = Allowance::new(&alice, bob.peer_id(), 500, 0).unwrap();
        assert!(allowance.verify());
        assert_eq!(allowance.amount, 500);
        assert_eq!(allowance.expiry, 0);
        // Tamper with the amount after signing.
        let mut forged = allowance.clone();
        forged.amount = 900;
        assert!(!forged.verify());
        // Tamper with the expiry after signing.
        let mut forged = allowance.clone();
        forged.expiry = 1234;
        assert!(!forged.verify());
        // A zero-amount allowance is refused at construction.
        assert!(matches!(
            Allowance::new(&alice, bob.peer_id(), 0, 0),
            Err(KoinError::ZeroAmount)
        ));
    }

    #[test]
    fn allowance_encode_decode_round_trip() {
        let (alice, bob, _) = identities();
        let allowance = Allowance::new(&alice, bob.peer_id(), 250, 9_999).unwrap();
        let encoded = allowance.encode();
        let decoded = Allowance::decode(&encoded).unwrap();
        // Decode rebuilds display names from keys, so compare by
        // key and by every field the encoding carries.
        assert_eq!(decoded.owner.public_key(), allowance.owner.public_key());
        assert_eq!(decoded.spender.public_key(), allowance.spender.public_key());
        assert_eq!(decoded.amount, allowance.amount);
        assert_eq!(decoded.expiry, allowance.expiry);
        assert_eq!(decoded.signature, allowance.signature);
        assert_eq!(decoded.id(), allowance.id());
        assert!(decoded.verify());
        assert!(Allowance::decode(b"garbage").is_err());
        assert!(Allowance::decode(&encoded[..encoded.len() - 1]).is_err());
    }

    #[test]
    fn draw_request_verifies_and_tampering_fails() {
        let (alice, bob, carol) = identities();
        let allowance = Allowance::new(&alice, bob.peer_id(), 500, 0).unwrap();
        let draw =
            AllowanceDrawRequest::new(&bob, &allowance.id(), carol.peer_id(), "rent", 100).unwrap();
        assert!(draw.verify());
        // Tamper with the amount after signing.
        let mut forged = draw.clone();
        forged.amount = 400;
        assert!(!forged.verify());
        // Tamper with the recipient after signing.
        let mut forged = draw.clone();
        forged.recipient = alice.peer_id().clone();
        assert!(!forged.verify());
        // A bad memo is refused at construction.
        assert!(AllowanceDrawRequest::new(
            &bob,
            &allowance.id(),
            carol.peer_id(),
            "a".repeat(129).as_str(),
            1
        )
        .is_err());
    }

    #[test]
    fn draw_id_is_stable_per_sequence() {
        let (alice, bob, carol) = identities();
        let allowance = Allowance::new(&alice, bob.peer_id(), 500, 0).unwrap();
        let draw = AllowanceDraw {
            allowance_id: allowance.id(),
            owner: alice.peer_id().clone(),
            spender: bob.peer_id().clone(),
            recipient: carol.peer_id().clone(),
            memo: "rent".to_string(),
            amount: 100,
            seq: 3,
        };
        let twin = draw.clone();
        assert_eq!(draw.id(), twin.id());
        let mut other_seq = draw.clone();
        other_seq.seq = 4;
        assert_ne!(draw.id(), other_seq.id());
    }
}
