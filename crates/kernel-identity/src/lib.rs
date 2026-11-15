//! Peer identity for KERNEL.
//!
//! An identity is an Ed25519 keypair plus a human-readable display name.
//! Humans see names and short fingerprints, never raw hashes:
//! a peer is "tristan" with fingerprint `a1b2-c3d4`, not a 64-hex string.
//!
//! Only the public half of an identity ever leaves a node. The secret key
//! stays inside [`Identity`] and is zeroized when dropped.

use std::fmt;

use kernel_crypto::{sha256, to_hex, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature};

/// Errors returned by identity operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdentityError {
    /// Wraps a [`kernel_crypto::CryptoError`].
    Crypto(kernel_crypto::CryptoError),
    /// A serialized identity did not match the expected format.
    InvalidFormat(String),
    /// A display name broke the naming rules.
    InvalidName(String),
}

impl fmt::Display for IdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IdentityError::Crypto(e) => write!(f, "crypto error: {e}"),
            IdentityError::InvalidFormat(msg) => write!(f, "invalid identity format: {msg}"),
            IdentityError::InvalidName(msg) => write!(f, "invalid display name: {msg}"),
        }
    }
}

impl std::error::Error for IdentityError {}

impl From<kernel_crypto::CryptoError> for IdentityError {
    fn from(e: kernel_crypto::CryptoError) -> Self {
        IdentityError::Crypto(e)
    }
}

/// Rules for display names: 1-32 characters, ASCII letters, digits, `-`,
/// `_`, `.`. No spaces, no hashes, no look-alike confusion.
fn validate_display_name(name: &str) -> Result<(), IdentityError> {
    if name.is_empty() {
        return Err(IdentityError::InvalidName(
            "display name must not be empty".to_string(),
        ));
    }
    if name.len() > 32 {
        return Err(IdentityError::InvalidName(
            "display name must be at most 32 characters".to_string(),
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Err(IdentityError::InvalidName(
            "display name may only use letters, digits, '-', '_' and '.'".to_string(),
        ));
    }
    Ok(())
}

/// A peer's public identity: their Ed25519 public key plus the name a human
/// calls them. Safe to share, store, and compare.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PeerId {
    public_key: Ed25519PublicKey,
    display_name: String,
}

impl PeerId {
    /// Build a peer id from a public key and a display name.
    pub fn new(public_key: Ed25519PublicKey, display_name: &str) -> Result<Self, IdentityError> {
        validate_display_name(display_name)?;
        Ok(Self {
            public_key,
            display_name: display_name.to_string(),
        })
    }

    /// The peer's Ed25519 public key.
    pub fn public_key(&self) -> &Ed25519PublicKey {
        &self.public_key
    }

    /// The human-readable name for this peer.
    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    /// Short human-friendly fingerprint: the first 4 bytes of the SHA-256
    /// of the public key, rendered as `ab12-cd34`. Enough to recognize a
    /// peer at a glance; the full key is what actually authenticates them.
    pub fn fingerprint(&self) -> String {
        let hash = sha256(self.public_key.as_bytes());
        format!(
            "{:02x}{:02x}-{:02x}{:02x}",
            hash[0], hash[1], hash[2], hash[3]
        )
    }

    /// Verify a signature over a message against this peer's public key.
    pub fn verify(&self, message: &[u8], signature: &Ed25519Signature) -> Result<(), IdentityError> {
        self.public_key
            .verify(message, signature)
            .map_err(IdentityError::Crypto)
    }
}

impl fmt::Display for PeerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}]", self.display_name, self.fingerprint())
    }
}

/// A node's own identity: its secret key plus its public [`PeerId`].
/// The secret key is zeroized when this is dropped. `Identity` is
/// intentionally not `Clone` — there is exactly one owner of a secret key.
pub struct Identity {
    secret: Ed25519SecretKey,
    peer_id: PeerId,
}

impl Identity {
    /// Generate a fresh identity with the given display name.
    pub fn generate(display_name: &str) -> Result<Self, IdentityError> {
        let secret = Ed25519SecretKey::generate();
        let peer_id = PeerId::new(secret.public_key(), display_name)?;
        Ok(Self { secret, peer_id })
    }

    /// This node's public peer id.
    pub fn peer_id(&self) -> &PeerId {
        &self.peer_id
    }

    /// Sign a message as this identity.
    pub fn sign(&self, message: &[u8]) -> Ed25519Signature {
        self.secret.sign(message)
    }

    /// Export the raw 32 secret bytes. Sensitive: the caller must zeroize
    /// the returned bytes after use. This exists only so an identity can
    /// be persisted to owner-only disk storage (see kernel-identity-disk);
    /// nothing else should call it.
    pub fn secret_bytes(&self) -> [u8; 32] {
        self.secret.to_bytes()
    }

    /// Rebuild an identity from raw secret bytes and a display name. The
    /// public key is derived from the secret, so a stored public key can
    /// be checked against it. Inverse of [`Identity::secret_bytes`],
    /// for loading a persisted identity.
    pub fn from_secret_bytes(secret: &[u8; 32], display_name: &str) -> Result<Self, IdentityError> {
        let secret = Ed25519SecretKey::from_bytes(secret);
        let peer_id = PeerId::new(secret.public_key(), display_name)?;
        Ok(Self { secret, peer_id })
    }

    /// The shareable public half of this identity.
    pub fn public_identity(&self) -> PublicIdentity {
        PublicIdentity {
            peer_id: self.peer_id.clone(),
        }
    }
}

/// The shareable public half of an identity: a [`PeerId`] with no secret
/// material. This is what nodes exchange when they meet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicIdentity {
    peer_id: PeerId,
}

impl PublicIdentity {
    /// The peer id this public identity describes.
    pub fn peer_id(&self) -> &PeerId {
        &self.peer_id
    }

    /// Serialize to a simple explicit text format:
    /// `KID1 <display-name> <hex public key>`.
    /// No secret material is ever in the output.
    pub fn encode(&self) -> String {
        format!(
            "KID1 {} {}",
            self.peer_id.display_name(),
            to_hex(self.peer_id.public_key().as_bytes())
        )
    }

    /// Parse a string produced by [`PublicIdentity::encode`].
    /// Rejects anything that is not exactly that format.
    pub fn decode(s: &str) -> Result<Self, IdentityError> {
        let mut parts = s.split_whitespace();
        let magic = parts.next().ok_or_else(|| {
            IdentityError::InvalidFormat("empty identity string".to_string())
        })?;
        if magic != "KID1" {
            return Err(IdentityError::InvalidFormat(format!(
                "expected magic 'KID1', got '{magic}'"
            )));
        }
        let name = parts.next().ok_or_else(|| {
            IdentityError::InvalidFormat("missing display name".to_string())
        })?;
        let key_hex = parts.next().ok_or_else(|| {
            IdentityError::InvalidFormat("missing public key".to_string())
        })?;
        if parts.next().is_some() {
            return Err(IdentityError::InvalidFormat(
                "trailing data after public key".to_string(),
            ));
        }
        let key_bytes = kernel_crypto::from_hex(key_hex).map_err(|e| {
            IdentityError::InvalidFormat(format!("public key is not valid hex: {e}"))
        })?;
        if key_bytes.len() != 32 {
            return Err(IdentityError::InvalidFormat(format!(
                "public key must be 32 bytes, got {}",
                key_bytes.len()
            )));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&key_bytes);
        let public_key = Ed25519PublicKey::from_bytes(&arr)?;
        let peer_id = PeerId::new(public_key, name)?;
        Ok(Self { peer_id })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_and_fingerprint() {
        let id = Identity::generate("tristan").unwrap();
        let peer = id.peer_id();
        assert_eq!(peer.display_name(), "tristan");
        let fp = peer.fingerprint();
        // Format is "xxxx-xxxx": 4 hex, dash, 4 hex.
        assert_eq!(fp.len(), 9);
        assert_eq!(fp.chars().nth(4), Some('-'));
        assert!(fp[..4].chars().all(|c| c.is_ascii_hexdigit()));
        assert!(fp[5..].chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(format!("{peer}"), format!("tristan [{fp}]"));
    }

    #[test]
    fn fingerprint_is_stable_and_key_bound() {
        let a = Identity::generate("alice").unwrap();
        let b = Identity::generate("alice").unwrap();
        // Same name, different keys -> different fingerprints.
        assert_ne!(a.peer_id().fingerprint(), b.peer_id().fingerprint());
        // Same key -> same fingerprint, even under a different name.
        let same_key = PeerId::new(*a.peer_id().public_key(), "alice2").unwrap();
        assert_eq!(same_key.fingerprint(), a.peer_id().fingerprint());
    }

    #[test]
    fn sign_and_verify_as_peer() {
        let alice = Identity::generate("alice").unwrap();
        let msg = b"kernel block 42";
        let sig = alice.sign(msg);
        assert!(alice.peer_id().verify(msg, &sig).is_ok());
        // Verification works from the public identity alone.
        let public = alice.public_identity();
        assert!(public.peer_id().verify(msg, &sig).is_ok());
    }

    #[test]
    fn wrong_peer_cannot_verify() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let sig = alice.sign(b"message");
        assert!(bob.peer_id().verify(b"message", &sig).is_err());
        assert!(alice.peer_id().verify(b"other message", &sig).is_err());
    }

    #[test]
    fn public_identity_encode_decode_round_trip() {
        let id = Identity::generate("tristan").unwrap();
        let public = id.public_identity();
        let encoded = public.encode();
        assert!(encoded.starts_with("KID1 tristan "));
        let decoded = PublicIdentity::decode(&encoded).unwrap();
        assert_eq!(decoded, public);
        assert_eq!(decoded.peer_id().fingerprint(), public.peer_id().fingerprint());
        // The encoded form contains no secret material: it is only the
        // public key hex plus the name.
        assert_eq!(encoded.split_whitespace().count(), 3);
    }

    #[test]
    fn decode_rejects_bad_formats() {
        assert!(PublicIdentity::decode("").is_err());
        assert!(PublicIdentity::decode("KID2 alice aa").is_err());
        assert!(PublicIdentity::decode("KID1").is_err());
        assert!(PublicIdentity::decode("KID1 alice").is_err());
        assert!(PublicIdentity::decode("KID1 alice zz").is_err());
        assert!(PublicIdentity::decode("KID1 alice aabbcc").is_err()); // short key
        let id = Identity::generate("alice").unwrap();
        let encoded = id.public_identity().encode();
        // Trailing junk.
        assert!(PublicIdentity::decode(&format!("{encoded} extra")).is_err());
        // Wrong magic.
        assert!(PublicIdentity::decode(&encoded.replacen("KID1", "KID9", 1)).is_err());
    }

    #[test]
    fn display_name_rules() {
        assert!(Identity::generate("").is_err());
        assert!(Identity::generate("has space").is_err());
        assert!(Identity::generate("has/slash").is_err());
        assert!(Identity::generate("a".repeat(33).as_str()).is_err());
        assert!(Identity::generate("ok-name_1.2").is_ok());
        assert!(Identity::generate("a").is_ok());
    }

    #[test]
    fn peer_ids_compare_by_key_and_name() {
        let id = Identity::generate("alice").unwrap();
        let same = PeerId::new(*id.peer_id().public_key(), "alice").unwrap();
        assert_eq!(same, *id.peer_id());
        let renamed = PeerId::new(*id.peer_id().public_key(), "alice2").unwrap();
        assert_ne!(renamed, *id.peer_id());
    }
}
