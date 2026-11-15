//! Cryptographic primitives for KERNEL.
//!
//! This crate wraps well-audited crates behind a small, explicit API:
//! - Ed25519 signing (identity, authentication)
//! - X25519 key exchange (per-peer shared secrets)
//! - ChaCha20-Poly1305 AEAD (confidential + authenticated encryption)
//! - SHA-256 (content addressing)
//! - HKDF-SHA256 (key derivation)
//!
//! Secret material is held in types that zeroize on drop. Nothing in this
//! crate writes secrets to disk, logs them, or serializes them.

use std::fmt;

pub use zeroize::{Zeroize, ZeroizeOnDrop};

/// Errors returned by cryptographic operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CryptoError {
    /// A key, nonce, or encoded value had the wrong length or form.
    InvalidKey(String),
    /// A signature did not verify.
    InvalidSignature,
    /// AEAD decryption failed (wrong key, wrong nonce, or tampered data).
    DecryptFailed,
    /// The operating system random source failed.
    RandomFailed,
    /// A caller-supplied value was malformed.
    InvalidInput(String),
}

impl fmt::Display for CryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CryptoError::InvalidKey(msg) => write!(f, "invalid key: {msg}"),
            CryptoError::InvalidSignature => write!(f, "invalid signature"),
            CryptoError::DecryptFailed => write!(f, "decryption failed"),
            CryptoError::RandomFailed => write!(f, "random source failed"),
            CryptoError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl std::error::Error for CryptoError {}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Lowercase hex encoding.
pub fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Decode lowercase (or mixed-case) hex. Errors on odd length or bad digits.
pub fn from_hex(hex: &str) -> Result<Vec<u8>, CryptoError> {
    if hex.len() % 2 != 0 {
        return Err(CryptoError::InvalidInput(
            "hex string has odd length".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    for i in (0..hex.len()).step_by(2) {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16)
            .map_err(|_| CryptoError::InvalidInput(format!("bad hex at offset {i}")))?;
        out.push(byte);
    }
    Ok(out)
}

/// Fill a fixed-size buffer with operating-system randomness.
pub fn random_bytes<const N: usize>() -> Result<[u8; N], CryptoError> {
    let mut buf = [0u8; N];
    getrandom::getrandom(&mut buf).map_err(|_| CryptoError::RandomFailed)?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// SHA-256
// ---------------------------------------------------------------------------

/// SHA-256 digest of `data`.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// SHA-256 digest of `data`, lowercase hex encoded.
pub fn sha256_hex(data: &[u8]) -> String {
    to_hex(&sha256(data))
}

// ---------------------------------------------------------------------------
// Ed25519
// ---------------------------------------------------------------------------

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};

/// An Ed25519 secret key, held as raw 32 bytes. Zeroized on drop.
/// The `SigningKey` is constructed on demand, so the secret never sits
/// in a heap structure this crate does not control.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct Ed25519SecretKey([u8; 32]);

impl Ed25519SecretKey {
    /// Generate a fresh random secret key.
    pub fn generate() -> Self {
        Self(random_bytes::<32>().expect("OS random source failed"))
    }

    /// Build from 32 secret bytes.
    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        Self(*bytes)
    }

    /// The public key matching this secret key.
    pub fn public_key(&self) -> Ed25519PublicKey {
        Ed25519PublicKey(SigningKey::from_bytes(&self.0).verifying_key().to_bytes())
    }

    /// Sign a message.
    pub fn sign(&self, message: &[u8]) -> Ed25519Signature {
        Ed25519Signature(SigningKey::from_bytes(&self.0).sign(message).to_bytes())
    }

    /// Export the raw 32 secret bytes. Callers must handle them carefully.
    pub fn to_bytes(&self) -> [u8; 32] {
        self.0
    }
}

/// An Ed25519 public key. Safe to share, store, and compare.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Ed25519PublicKey([u8; 32]);

impl Ed25519PublicKey {
    /// Build from 32 public key bytes. Rejects points that are not valid
    /// Ed25519 public keys.
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self, CryptoError> {
        VerifyingKey::from_bytes(bytes)
            .map_err(|_| CryptoError::InvalidKey("not a valid Ed25519 public key".to_string()))?;
        Ok(Self(*bytes))
    }

    /// The raw 32 public key bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Verify a signature over a message. Fails on any tampering.
    pub fn verify(&self, message: &[u8], signature: &Ed25519Signature) -> Result<(), CryptoError> {
        let vk = VerifyingKey::from_bytes(&self.0)
            .map_err(|_| CryptoError::InvalidKey("not a valid Ed25519 public key".to_string()))?;
        let sig = Signature::from_bytes(&signature.0);
        vk.verify(message, &sig)
            .map_err(|_| CryptoError::InvalidSignature)
    }
}

/// An Ed25519 signature (64 bytes).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Ed25519Signature([u8; 64]);

impl Ed25519Signature {
    /// Build from 64 signature bytes.
    pub fn from_bytes(bytes: &[u8; 64]) -> Self {
        Self(*bytes)
    }

    /// The raw 64 signature bytes.
    pub fn to_bytes(&self) -> [u8; 64] {
        self.0
    }
}

// ---------------------------------------------------------------------------
// X25519
// ---------------------------------------------------------------------------

use curve25519_dalek::MontgomeryPoint;

/// An X25519 secret key, held as raw 32 bytes. Zeroized on drop.
/// The bytes are clamped on every use by the curve operations below
/// (`mul_base_clamped` / `mul_clamped`), exactly as RFC 7748 requires,
/// so callers never have to clamp by hand.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct X25519SecretKey([u8; 32]);

impl X25519SecretKey {
    /// Generate a fresh random secret key.
    pub fn generate() -> Self {
        Self(random_bytes::<32>().expect("OS random source failed"))
    }

    /// Build from 32 secret bytes (clamping is applied on use).
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The public key matching this secret key.
    pub fn public_key(&self) -> X25519PublicKey {
        X25519PublicKey(MontgomeryPoint::mul_base_clamped(self.0).to_bytes())
    }

    /// Diffie-Hellman: combine this secret with a peer's public key.
    /// Both sides derive the same shared secret.
    pub fn diffie_hellman(&self, peer: &X25519PublicKey) -> SharedSecret {
        let peer_point = MontgomeryPoint(peer.0);
        SharedSecret(peer_point.mul_clamped(self.0).to_bytes())
    }
}

/// An X25519 public key (32 bytes). Safe to share.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct X25519PublicKey([u8; 32]);

impl X25519PublicKey {
    /// Build from 32 public key bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The raw 32 public key bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// A Diffie-Hellman shared secret (32 bytes). Zeroized on drop.
/// Feed it through HKDF before using it as an encryption key.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct SharedSecret([u8; 32]);

impl SharedSecret {
    /// The raw 32 shared-secret bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

// ---------------------------------------------------------------------------
// ChaCha20-Poly1305 AEAD
// ---------------------------------------------------------------------------

use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Key, Nonce,
};

/// Length of an AEAD key in bytes.
pub const AEAD_KEY_LEN: usize = 32;
/// Length of an AEAD nonce in bytes.
pub const AEAD_NONCE_LEN: usize = 12;

/// A symmetric key for ChaCha20-Poly1305. Zeroized on drop.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct SymmetricKey([u8; AEAD_KEY_LEN]);

impl SymmetricKey {
    /// Generate a fresh random key.
    pub fn generate() -> Result<Self, CryptoError> {
        Ok(Self(random_bytes::<AEAD_KEY_LEN>()?))
    }

    /// Build from 32 key bytes.
    pub fn from_bytes(bytes: [u8; AEAD_KEY_LEN]) -> Self {
        Self(bytes)
    }

    /// The raw 32 key bytes.
    pub fn as_bytes(&self) -> &[u8; AEAD_KEY_LEN] {
        &self.0
    }
}

/// An encrypted message: a fresh random nonce plus the ciphertext.
/// The nonce is not secret, but it must be stored alongside the ciphertext.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncryptedMessage {
    pub nonce: [u8; AEAD_NONCE_LEN],
    pub ciphertext: Vec<u8>,
}

/// Encrypt `plaintext` under `key` with a fresh random nonce.
pub fn aead_encrypt(key: &SymmetricKey, plaintext: &[u8]) -> Result<EncryptedMessage, CryptoError> {
    let nonce_bytes = random_bytes::<AEAD_NONCE_LEN>()?;
    let cipher = ChaCha20Poly1305::new(Key::from_slice(key.as_bytes()));
    let ciphertext = cipher
        .encrypt(Nonce::from_slice(&nonce_bytes), plaintext)
        .map_err(|_| CryptoError::DecryptFailed)?;
    Ok(EncryptedMessage {
        nonce: nonce_bytes,
        ciphertext,
    })
}

/// Decrypt a ciphertext produced by [`aead_encrypt`].
/// Fails if the key is wrong, the nonce is wrong, or the data was tampered with.
pub fn aead_decrypt(
    key: &SymmetricKey,
    nonce: &[u8; AEAD_NONCE_LEN],
    ciphertext: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    let cipher = ChaCha20Poly1305::new(Key::from_slice(key.as_bytes()));
    cipher
        .decrypt(Nonce::from_slice(nonce), ciphertext)
        .map_err(|_| CryptoError::DecryptFailed)
}

// ---------------------------------------------------------------------------
// HKDF-SHA256
// ---------------------------------------------------------------------------

use hkdf::Hkdf;
use sha2::Sha256;

/// Maximum output length for one HKDF-SHA256 expansion (255 * 32).
pub const HKDF_MAX_OUT_LEN: usize = 255 * 32;

/// Derive `out_len` bytes from input key material using HKDF-SHA256.
/// `salt` may be `None` (zeros are used). `info` binds the output to a
/// purpose string; different `info` values give unrelated outputs.
pub fn hkdf_sha256(
    salt: Option<&[u8]>,
    ikm: &[u8],
    info: &[u8],
    out_len: usize,
) -> Result<Vec<u8>, CryptoError> {
    if out_len > HKDF_MAX_OUT_LEN {
        return Err(CryptoError::InvalidInput(format!(
            "HKDF output length {out_len} exceeds maximum {HKDF_MAX_OUT_LEN}"
        )));
    }
    let hk = Hkdf::<Sha256>::new(salt, ikm);
    let mut out = vec![0u8; out_len];
    hk.expand(info, &mut out)
        .map_err(|_| CryptoError::InvalidInput("HKDF expansion failed".to_string()))?;
    Ok(out)
}

/// Derive a 32-byte key from input key material using HKDF-SHA256.
pub fn derive_key_32(salt: Option<&[u8]>, ikm: &[u8], info: &[u8]) -> Result<[u8; 32], CryptoError> {
    let out = hkdf_sha256(salt, ikm, info, 32)?;
    let mut key = [0u8; 32];
    key.copy_from_slice(&out);
    Ok(key)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(sha256(b"kernel").len(), 32);
        assert_ne!(sha256(b"a"), sha256(b"b"));
    }

    #[test]
    fn hex_round_trip_and_errors() {
        let bytes = [0x00u8, 0x01, 0xab, 0xff];
        assert_eq!(from_hex(&to_hex(&bytes)).unwrap(), bytes);
        assert!(from_hex("zz").is_err());
        assert!(from_hex("abc").is_err()); // odd length
        assert!(from_hex("").unwrap().is_empty());
    }

    #[test]
    fn ed25519_sign_and_verify() {
        let secret = Ed25519SecretKey::generate();
        let public = secret.public_key();
        let msg = b"hello kernel";
        let sig = secret.sign(msg);
        assert!(public.verify(msg, &sig).is_ok());
        // Public key round-trips through bytes.
        let restored = Ed25519PublicKey::from_bytes(public.as_bytes()).unwrap();
        assert_eq!(restored, public);
        assert!(restored.verify(msg, &sig).is_ok());
        // Signature round-trips through bytes.
        let sig2 = Ed25519Signature::from_bytes(&sig.to_bytes());
        assert_eq!(sig2, sig);
    }

    #[test]
    fn ed25519_tampered_message_fails() {
        let secret = Ed25519SecretKey::generate();
        let public = secret.public_key();
        let sig = secret.sign(b"original");
        assert_eq!(
            public.verify(b"tampered", &sig),
            Err(CryptoError::InvalidSignature)
        );
    }

    #[test]
    fn ed25519_wrong_key_fails() {
        let alice = Ed25519SecretKey::generate();
        let bob = Ed25519SecretKey::generate();
        let sig = alice.sign(b"message");
        assert_eq!(
            bob.public_key().verify(b"message", &sig),
            Err(CryptoError::InvalidSignature)
        );
    }

    #[test]
    fn ed25519_empty_message() {
        let secret = Ed25519SecretKey::generate();
        let sig = secret.sign(b"");
        assert!(secret.public_key().verify(b"", &sig).is_ok());
    }

    #[test]
    fn ed25519_secret_round_trip() {
        let secret = Ed25519SecretKey::generate();
        let bytes = secret.to_bytes();
        let restored = Ed25519SecretKey::from_bytes(&bytes);
        assert_eq!(restored.public_key(), secret.public_key());
    }

    #[test]
    fn x25519_shared_secret_matches_both_ways() {
        let alice = X25519SecretKey::generate();
        let bob = X25519SecretKey::generate();
        let alice_shared = alice.diffie_hellman(&bob.public_key());
        let bob_shared = bob.diffie_hellman(&alice.public_key());
        assert_eq!(alice_shared.as_bytes(), bob_shared.as_bytes());
        // A third party gets a different secret.
        let carol = X25519SecretKey::generate();
        let other = alice.diffie_hellman(&carol.public_key());
        assert_ne!(other.as_bytes(), alice_shared.as_bytes());
    }

    #[test]
    fn x25519_public_key_round_trip() {
        let secret = X25519SecretKey::generate();
        let public = secret.public_key();
        let restored = X25519PublicKey::from_bytes(*public.as_bytes());
        assert_eq!(restored, public);
    }

    #[test]
    fn aead_round_trip() {
        let key = SymmetricKey::generate().unwrap();
        let msg = b"secret block for kernel storage";
        let encrypted = aead_encrypt(&key, msg).unwrap();
        assert_eq!(encrypted.nonce.len(), AEAD_NONCE_LEN);
        assert!(!encrypted.ciphertext.is_empty());
        let decrypted = aead_decrypt(&key, &encrypted.nonce, &encrypted.ciphertext).unwrap();
        assert_eq!(decrypted, msg);
    }

    #[test]
    fn aead_empty_plaintext() {
        let key = SymmetricKey::generate().unwrap();
        let encrypted = aead_encrypt(&key, b"").unwrap();
        let decrypted = aead_decrypt(&key, &encrypted.nonce, &encrypted.ciphertext).unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn aead_wrong_key_fails() {
        let key = SymmetricKey::generate().unwrap();
        let wrong = SymmetricKey::generate().unwrap();
        let encrypted = aead_encrypt(&key, b"data").unwrap();
        assert_eq!(
            aead_decrypt(&wrong, &encrypted.nonce, &encrypted.ciphertext),
            Err(CryptoError::DecryptFailed)
        );
    }

    #[test]
    fn aead_tampered_ciphertext_fails() {
        let key = SymmetricKey::generate().unwrap();
        let mut encrypted = aead_encrypt(&key, b"data").unwrap();
        encrypted.ciphertext[0] ^= 0xff;
        assert_eq!(
            aead_decrypt(&key, &encrypted.nonce, &encrypted.ciphertext),
            Err(CryptoError::DecryptFailed)
        );
    }

    #[test]
    fn aead_wrong_nonce_fails() {
        let key = SymmetricKey::generate().unwrap();
        let encrypted = aead_encrypt(&key, b"data").unwrap();
        let mut nonce = encrypted.nonce;
        nonce[0] ^= 0x01;
        assert_eq!(
            aead_decrypt(&key, &nonce, &encrypted.ciphertext),
            Err(CryptoError::DecryptFailed)
        );
    }

    #[test]
    fn aead_two_encryptions_differ() {
        // Fresh random nonces mean identical plaintext encrypts differently.
        let key = SymmetricKey::generate().unwrap();
        let a = aead_encrypt(&key, b"same").unwrap();
        let b = aead_encrypt(&key, b"same").unwrap();
        assert_ne!(a.nonce, b.nonce);
        assert_ne!(a.ciphertext, b.ciphertext);
    }

    #[test]
    fn hkdf_deterministic_and_context_bound() {
        let ikm = b"input key material";
        let a = hkdf_sha256(Some(b"salt"), ikm, b"purpose-a", 32).unwrap();
        let b = hkdf_sha256(Some(b"salt"), ikm, b"purpose-a", 32).unwrap();
        assert_eq!(a, b);
        let c = hkdf_sha256(Some(b"salt"), ikm, b"purpose-b", 32).unwrap();
        assert_ne!(a, c);
        let d = hkdf_sha256(Some(b"other-salt"), ikm, b"purpose-a", 32).unwrap();
        assert_ne!(a, d);
        let e = hkdf_sha256(None, ikm, b"purpose-a", 32).unwrap();
        assert_eq!(e.len(), 32);
        assert_ne!(a, e);
    }

    #[test]
    fn hkdf_output_length_respected() {
        assert_eq!(hkdf_sha256(None, b"ikm", b"info", 64).unwrap().len(), 64);
        assert!(hkdf_sha256(None, b"ikm", b"info", HKDF_MAX_OUT_LEN + 1).is_err());
    }

    #[test]
    fn derive_key_32_matches_hkdf() {
        let k = derive_key_32(Some(b"s"), b"ikm", b"info").unwrap();
        let v = hkdf_sha256(Some(b"s"), b"ikm", b"info", 32).unwrap();
        assert_eq!(&k[..], &v[..]);
    }

    #[test]
    fn random_bytes_are_random() {
        let a = random_bytes::<32>().unwrap();
        let b = random_bytes::<32>().unwrap();
        assert_ne!(a, b);
    }
}
