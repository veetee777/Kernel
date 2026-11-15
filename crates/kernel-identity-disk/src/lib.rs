//! Identity persistence for KERNEL: an Ed25519 identity on disk.
//!
//! One file holds one identity. The format is binary — no JSON anywhere
//! (KERNEL law):
//!
//! ```text
//! magic    "KIDSK1" (6 bytes)
//! version  1 (1 byte)
//! flags    1 byte, bit 0 = wrapped with a passphrase
//! name_len 1 byte (1..=32)
//! name     name_len bytes, UTF-8 display name
//! -- unwrapped --
//! secret   32 bytes
//! public   32 bytes (must match the key derived from the secret)
//! -- wrapped --
//! salt     32 bytes (HKDF salt, random per save)
//! nonce    12 bytes (ChaCha20-Poly1305 nonce, random per save)
//! ct_len   u32 LE, always 80 (64-byte plaintext + 16-byte tag)
//! ct       ct_len bytes: secret (32) || public (32), AEAD-sealed
//! ```
//!
//! The public key is stored alongside the secret so a load can prove the
//! secret was not altered: the key derived from the loaded secret must
//! equal the stored public key. In the wrapped case the AEAD tag already
//! authenticates the plaintext; the public-key check is a second layer.
//!
//! Passphrase wrapping: the wrap key is HKDF-SHA256(salt, passphrase,
//! info `"KERNEL identity-disk wrap v1"`). A wrong passphrase fails the
//! AEAD authentication and is refused — it never yields a garbage
//! identity.
//!
//! Files are owner-only: created with mode 0600 on Unix and set to 0600
//! again after writing. A save writes a sibling `.tmp` file and renames
//! it over the target, so a crash mid-write never leaves a half-written
//! identity file. A load refuses bad magic, a wrong version, unknown
//! flag bits, truncation, or trailing bytes — never silently loading a
//! corrupt file.

use std::fmt;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use kernel_crypto::{
    aead_decrypt, aead_encrypt, derive_key_32, random_bytes, Ed25519SecretKey, SymmetricKey,
    AEAD_NONCE_LEN,
};
use kernel_identity::{Identity, IdentityError};
use zeroize::Zeroize;

/// Magic bytes at the start of every identity file.
pub const MAGIC: &[u8; 6] = b"KIDSK1";
/// The only file version this build reads or writes.
pub const VERSION: u8 = 1;

const FLAG_WRAPPED: u8 = 0x01;
const SALT_LEN: usize = 32;
const SECRET_LEN: usize = 32;
const PUBLIC_LEN: usize = 32;
/// Plaintext sealed by the passphrase wrap: secret || public.
const PLAINTEXT_LEN: usize = SECRET_LEN + PUBLIC_LEN;
const TAG_LEN: usize = 16;
const WRAPPED_CT_LEN: usize = PLAINTEXT_LEN + TAG_LEN;
const HKDF_INFO: &[u8] = b"KERNEL identity-disk wrap v1";

/// Errors returned by identity-disk operations.
#[derive(Debug)]
pub enum IdentityDiskError {
    /// A filesystem operation failed (missing file, bad path, ...).
    Io(std::io::Error),
    /// The bytes are not a valid identity file: bad magic, wrong
    /// version, truncation, trailing bytes, or a public key that does
    /// not match the secret.
    InvalidFormat(String),
    /// The file is passphrase-wrapped but no passphrase was given.
    PassphraseRequired,
    /// The passphrase is wrong. The AEAD authentication failed.
    WrongPassphrase,
    /// A crypto operation failed for another reason.
    Crypto(kernel_crypto::CryptoError),
    /// The identity itself was invalid (e.g. a bad display name).
    Identity(IdentityError),
}

impl fmt::Display for IdentityDiskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IdentityDiskError::Io(e) => write!(f, "io error: {e}"),
            IdentityDiskError::InvalidFormat(msg) => write!(f, "invalid identity file: {msg}"),
            IdentityDiskError::PassphraseRequired => {
                f.write_str("identity file is passphrase-wrapped; a passphrase is required")
            }
            IdentityDiskError::WrongPassphrase => f.write_str("wrong passphrase"),
            IdentityDiskError::Crypto(e) => write!(f, "crypto error: {e}"),
            IdentityDiskError::Identity(e) => write!(f, "identity error: {e}"),
        }
    }
}

impl std::error::Error for IdentityDiskError {}

impl From<std::io::Error> for IdentityDiskError {
    fn from(e: std::io::Error) -> Self {
        IdentityDiskError::Io(e)
    }
}

impl From<kernel_crypto::CryptoError> for IdentityDiskError {
    fn from(e: kernel_crypto::CryptoError) -> Self {
        IdentityDiskError::Crypto(e)
    }
}

impl From<IdentityError> for IdentityDiskError {
    fn from(e: IdentityError) -> Self {
        IdentityDiskError::Identity(e)
    }
}

fn invalid(msg: impl Into<String>) -> IdentityDiskError {
    IdentityDiskError::InvalidFormat(msg.into())
}

/// Encode an identity to its binary file form. `passphrase` wraps the
/// secret with ChaCha20-Poly1305 under an HKDF-SHA256-derived key;
/// `None` stores the secret in the clear (the file itself must stay
/// owner-only, mode 0600).
pub fn encode(identity: &Identity, passphrase: Option<&[u8]>) -> Result<Vec<u8>, IdentityDiskError> {
    let name = identity.peer_id().display_name();
    let name_bytes = name.as_bytes();
    debug_assert!((1..=32).contains(&name_bytes.len()));

    let mut secret = identity.secret_bytes();
    let public = *identity.peer_id().public_key().as_bytes();

    // Plaintext is secret || public, so a load can prove the secret
    // was not altered.
    let mut plaintext = Vec::with_capacity(PLAINTEXT_LEN);
    plaintext.extend_from_slice(&secret);
    plaintext.extend_from_slice(&public);
    secret.zeroize();

    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    match passphrase {
        Some(pass) => {
            out.push(FLAG_WRAPPED);
            out.push(name_bytes.len() as u8);
            out.extend_from_slice(name_bytes);
            let salt = random_bytes::<SALT_LEN>()?;
            let key_bytes = derive_key_32(Some(&salt), pass, HKDF_INFO)?;
            let key = SymmetricKey::from_bytes(key_bytes);
            let enc = aead_encrypt(&key, &plaintext)?;
            plaintext.zeroize();
            out.extend_from_slice(&salt);
            out.extend_from_slice(&enc.nonce);
            out.extend_from_slice(&(enc.ciphertext.len() as u32).to_le_bytes());
            out.extend_from_slice(&enc.ciphertext);
        }
        None => {
            out.push(0);
            out.push(name_bytes.len() as u8);
            out.extend_from_slice(name_bytes);
            out.extend_from_slice(&plaintext);
            plaintext.zeroize();
        }
    }
    Ok(out)
}

fn take<'a>(b: &'a [u8], at: &mut usize, n: usize) -> Result<&'a [u8], IdentityDiskError> {
    if *at + n > b.len() {
        return Err(invalid("truncated identity file"));
    }
    let s = &b[*at..*at + n];
    *at += n;
    Ok(s)
}

/// Decode an identity from its binary file form. `passphrase` must be
/// given for a wrapped file; it is ignored for an unwrapped file.
/// Refuses bad magic, wrong version, truncation, trailing bytes, a
/// wrong passphrase, and a public key that does not match the secret.
pub fn decode(bytes: &[u8], passphrase: Option<&[u8]>) -> Result<Identity, IdentityDiskError> {
    let mut at = 0usize;

    if bytes.len() < MAGIC.len() + 3 {
        return Err(invalid("file too short for an identity header"));
    }
    if take(bytes, &mut at, MAGIC.len())? != MAGIC {
        return Err(invalid("bad magic: not a KIDSK1 identity file"));
    }
    let version = take(bytes, &mut at, 1)?[0];
    if version != VERSION {
        return Err(invalid(format!(
            "unsupported identity file version {version}, expected {VERSION}"
        )));
    }
    let flags = take(bytes, &mut at, 1)?[0];
    if flags & !FLAG_WRAPPED != 0 {
        return Err(invalid(format!("unknown flag bits set: {flags:#04x}")));
    }
    let wrapped = flags & FLAG_WRAPPED != 0;
    let name_len = take(bytes, &mut at, 1)?[0] as usize;
    if name_len == 0 || name_len > 32 {
        return Err(invalid(format!("bad display name length {name_len}")));
    }
    let name_bytes = take(bytes, &mut at, name_len)?;
    let name = std::str::from_utf8(name_bytes)
        .map_err(|_| invalid("display name is not valid UTF-8"))?
        .to_string();

    let mut plaintext: Vec<u8> = if wrapped {
        let pass = passphrase.ok_or(IdentityDiskError::PassphraseRequired)?;
        let salt = take(bytes, &mut at, SALT_LEN)?;
        let nonce = take(bytes, &mut at, AEAD_NONCE_LEN)?;
        let ct_len_bytes = take(bytes, &mut at, 4)?;
        let ct_len = u32::from_le_bytes(ct_len_bytes.try_into().unwrap()) as usize;
        if ct_len != WRAPPED_CT_LEN {
            return Err(invalid(format!(
                "wrapped payload length {ct_len}, expected {WRAPPED_CT_LEN}"
            )));
        }
        let ct = take(bytes, &mut at, ct_len)?;
        let key_bytes = derive_key_32(Some(salt), pass, HKDF_INFO)?;
        let key = SymmetricKey::from_bytes(key_bytes);
        let nonce_arr: [u8; AEAD_NONCE_LEN] = nonce.try_into().unwrap();
        aead_decrypt(&key, &nonce_arr, ct).map_err(|e| match e {
            kernel_crypto::CryptoError::DecryptFailed => {
                IdentityDiskError::WrongPassphrase
            }
            other => IdentityDiskError::Crypto(other),
        })?
    } else {
        take(bytes, &mut at, PLAINTEXT_LEN)?.to_vec()
    };

    if at != bytes.len() {
        plaintext.zeroize();
        return Err(invalid("trailing bytes after identity record"));
    }
    if plaintext.len() != PLAINTEXT_LEN {
        plaintext.zeroize();
        return Err(invalid("identity payload has the wrong length"));
    }
    let mut secret = [0u8; SECRET_LEN];
    secret.copy_from_slice(&plaintext[..SECRET_LEN]);
    let stored_public = &plaintext[SECRET_LEN..];
    // The stored public key must match the key derived from the secret.
    // A flipped bit in either one is caught here.
    let derived_public = Ed25519SecretKey::from_bytes(&secret).public_key();
    if derived_public.as_bytes() != stored_public {
        secret.zeroize();
        plaintext.zeroize();
        return Err(invalid("stored public key does not match the secret key"));
    }
    plaintext.zeroize();
    let identity = Identity::from_secret_bytes(&secret, &name)?;
    secret.zeroize();
    Ok(identity)
}

/// Save an identity to `path`, unwrapped. The file is created
/// owner-only (0600) and written atomically via a sibling `.tmp` file.
pub fn save(identity: &Identity, path: &Path) -> Result<(), IdentityDiskError> {
    save_inner(identity, path, None)
}

/// Save an identity to `path`, wrapped with `passphrase`. Same
/// owner-only, atomic write as [`save`].
pub fn save_with_passphrase(
    identity: &Identity,
    path: &Path,
    passphrase: &[u8],
) -> Result<(), IdentityDiskError> {
    save_inner(identity, path, Some(passphrase))
}

fn save_inner(
    identity: &Identity,
    path: &Path,
    passphrase: Option<&[u8]>,
) -> Result<(), IdentityDiskError> {
    let bytes = encode(identity, passphrase)?;
    let tmp = path.with_file_name(format!(
        "{}.tmp",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("identity")
    ));
    write_owner_only(&tmp, &bytes)?;
    fs::rename(&tmp, path)?;
    // Rename keeps the temp file's mode; set it explicitly so the
    // target is 0600 even on a filesystem that would not.
    set_owner_only(path)?;
    Ok(())
}

#[cfg(unix)]
fn write_owner_only(path: &Path, bytes: &[u8]) -> Result<(), IdentityDiskError> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn write_owner_only(path: &Path, bytes: &[u8]) -> Result<(), IdentityDiskError> {
    // No Unix permission bits here; the file is still written atomically
    // by the caller. Owner-only enforcement is a Unix property.
    let mut f = File::create(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}

#[cfg(unix)]
fn set_owner_only(path: &Path) -> Result<(), IdentityDiskError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_owner_only(_path: &Path) -> Result<(), IdentityDiskError> {
    Ok(())
}

/// Load an identity from `path`. Refuses a wrapped file without a
/// passphrase — use [`load_with_passphrase`] for those.
pub fn load(path: &Path) -> Result<Identity, IdentityDiskError> {
    let bytes = read_file(path)?;
    decode(&bytes, None)
}

/// Load a passphrase-wrapped identity from `path`.
pub fn load_with_passphrase(path: &Path, passphrase: &[u8]) -> Result<Identity, IdentityDiskError> {
    let bytes = read_file(path)?;
    decode(&bytes, Some(passphrase))
}

fn read_file(path: &Path) -> Result<Vec<u8>, IdentityDiskError> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    Ok(bytes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn test_path(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kidentity-disk-{}-{}",
            std::process::id(),
            name
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create test dir");
        dir.join("identity.kidsk")
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_dir_all(path.parent().expect("parent"));
    }

    #[test]
    fn save_load_round_trip_without_passphrase() {
        let path = test_path("plain");
        let id = Identity::generate("tristan").expect("generate");
        save(&id, &path).expect("save");
        let back = load(&path).expect("load");
        assert_eq!(back.peer_id(), id.peer_id());
        assert_eq!(back.peer_id().fingerprint(), id.peer_id().fingerprint());
        // The loaded identity really is the same key: it signs, and the
        // original peer id verifies the signature.
        let sig = back.sign(b"prove it");
        assert!(id.peer_id().verify(b"prove it", &sig).is_ok());
        cleanup(&path);
    }

    #[test]
    fn save_load_round_trip_with_passphrase() {
        let path = test_path("wrapped");
        let id = Identity::generate("tristan").expect("generate");
        save_with_passphrase(&id, &path, b"correct horse").expect("save");
        let back = load_with_passphrase(&path, b"correct horse").expect("load");
        assert_eq!(back.peer_id(), id.peer_id());
        let sig = back.sign(b"prove it");
        assert!(id.peer_id().verify(b"prove it", &sig).is_ok());
        cleanup(&path);
    }

    #[test]
    fn wrong_passphrase_is_refused() {
        let path = test_path("wrong-pass");
        let id = Identity::generate("tristan").expect("generate");
        save_with_passphrase(&id, &path, b"correct horse").expect("save");
        let err = load_with_passphrase(&path, b"wrong horse")
            .err()
            .expect("refuse");
        assert!(matches!(err, IdentityDiskError::WrongPassphrase));
        cleanup(&path);
    }

    #[test]
    fn wrapped_file_without_a_passphrase_is_refused() {
        let path = test_path("needs-pass");
        let id = Identity::generate("tristan").expect("generate");
        save_with_passphrase(&id, &path, b"correct horse").expect("save");
        let err = load(&path).err().expect("refuse");
        assert!(matches!(err, IdentityDiskError::PassphraseRequired));
        cleanup(&path);
    }

    #[test]
    fn tampered_file_bytes_are_refused() {
        let id = Identity::generate("tristan").expect("generate");
        // Tamper with the magic.
        let mut bytes = encode(&id, None).expect("encode");
        bytes[0] = b'X';
        assert!(matches!(
            decode(&bytes, None).err().expect("refuse"),
            IdentityDiskError::InvalidFormat(_)
        ));
        // Tamper with a secret byte: the public-key check must catch it.
        let mut bytes = encode(&id, None).expect("encode");
        let secret_at = MAGIC.len() + 3 + "tristan".len();
        bytes[secret_at] ^= 0x01;
        assert!(matches!(
            decode(&bytes, None).err().expect("refuse"),
            IdentityDiskError::InvalidFormat(_)
        ));
        // Tamper with a ciphertext byte in a wrapped file: the AEAD tag
        // must catch it, reported as a wrong passphrase.
        let mut bytes = encode(&id, Some(b"pw")).expect("encode");
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        assert!(matches!(
            decode(&bytes, Some(b"pw")).err().expect("refuse"),
            IdentityDiskError::WrongPassphrase
        ));
    }

    #[test]
    fn truncated_file_is_refused() {
        let id = Identity::generate("tristan").expect("generate");
        let bytes = encode(&id, None).expect("encode");
        for cut in [1, MAGIC.len() + 1, bytes.len() - 1] {
            assert!(
                decode(&bytes[..cut], None).is_err(),
                "cut at {cut} must be refused"
            );
        }
        let wrapped = encode(&id, Some(b"pw")).expect("encode");
        for cut in [MAGIC.len() + 3, wrapped.len() / 2, wrapped.len() - 1] {
            assert!(
                decode(&wrapped[..cut], Some(b"pw")).is_err(),
                "wrapped cut at {cut} must be refused"
            );
        }
    }

    #[test]
    fn trailing_bytes_are_refused() {
        let id = Identity::generate("tristan").expect("generate");
        let mut bytes = encode(&id, None).expect("encode");
        bytes.push(0);
        assert!(matches!(
            decode(&bytes, None).err().expect("refuse"),
            IdentityDiskError::InvalidFormat(_)
        ));
    }

    #[test]
    fn saved_file_has_0600_permissions() {
        let path = test_path("perms");
        let id = Identity::generate("tristan").expect("generate");
        save(&id, &path).expect("save");
        let mode = fs::metadata(&path).expect("metadata").permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "identity file must be owner-only, got {mode:o}");
        // The wrapped save goes through the same path.
        save_with_passphrase(&id, &path, b"pw").expect("save wrapped");
        let mode = fs::metadata(&path).expect("metadata").permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "wrapped identity file must be owner-only, got {mode:o}");
        cleanup(&path);
    }

    #[test]
    fn loading_a_nonexistent_file_is_a_clean_error() {
        let path = test_path("missing").join("no-such-file.kidsk");
        let err = load(&path).err().expect("refuse");
        assert!(matches!(err, IdentityDiskError::Io(_)));
    }

    #[test]
    fn wrong_version_and_unknown_flags_are_refused() {
        let id = Identity::generate("tristan").expect("generate");
        let mut bytes = encode(&id, None).expect("encode");
        bytes[MAGIC.len()] = 2; // version
        assert!(decode(&bytes, None).is_err());
        let mut bytes = encode(&id, None).expect("encode");
        bytes[MAGIC.len() + 1] = 0x80; // unknown flag bit
        assert!(decode(&bytes, None).is_err());
    }

    #[test]
    fn two_saves_of_one_identity_differ_only_by_salt_and_nonce() {
        // Wrapped saves must not be byte-identical: fresh salt and nonce
        // each time, so two files cannot be told apart by content.
        let id = Identity::generate("tristan").expect("generate");
        let a = encode(&id, Some(b"pw")).expect("encode a");
        let b = encode(&id, Some(b"pw")).expect("encode b");
        assert_ne!(a, b);
        assert_eq!(decode(&a, Some(b"pw")).expect("decode a").peer_id(), id.peer_id());
        assert_eq!(decode(&b, Some(b"pw")).expect("decode b").peer_id(), id.peer_id());
    }
}
