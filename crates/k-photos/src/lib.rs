//! K•Photos — photo storage on top of KERNEL content-addressed storage.
//!
//! A photo is a human name plus bytes. The bytes are stored chunked in
//! the content store (see `kernel_storage`), and the photo is described
//! by a signed manifest: the name, the chunked-content root CID, the
//! owner, the owner's signature, plus photo metadata — width, height,
//! format (e.g. `"jpeg"`), and the timestamp the photo was taken.
//!
//! The manifest is itself a block in the content store, addressed by
//! its own CID. Anyone holding the manifest CID can fetch the photo —
//! and the signature plus per-chunk hash verification mean a tampered
//! manifest or a tampered chunk fails loudly instead of silently
//! returning wrong bytes.
//!
//! Manifest format (explicit binary, no JSON):
//! `KPHOTO1` + name length (u16 LE) + name bytes + owner public key
//! (32) + chunk root CID (32) + width (u32 LE) + height (u32 LE) +
//! format length (u8) + format bytes + taken timestamp (u64 LE) +
//! signature (64).

use std::collections::{HashMap, HashSet};
use std::fmt;

use kernel_crypto::Ed25519Signature;
use kernel_identity::{Identity, IdentityError, PeerId};
use kernel_storage::{decode_manifest, Cid, ContentStore, StorageError};

/// Magic bytes at the start of every photo manifest.
const MANIFEST_MAGIC: &[u8; 7] = b"KPHOTO1";

/// Domain separator for manifest signatures: a signature covers
/// `b"KPHOTO1" || name || root || width || height || format || taken
/// || owner public key`.
const MANIFEST_SIGN_DOMAIN: &[u8] = b"KPHOTO1";

/// Maximum photo dimension in either direction (pixels).
/// Rejects absurd metadata before it is signed.
pub const MAX_DIMENSION: u32 = 100_000;

/// Errors returned by photo operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhotoError {
    /// Wraps a [`StorageError`].
    Storage(StorageError),
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// A photo name broke the naming rules.
    InvalidName(String),
    /// A format string broke the rules.
    InvalidFormat(String),
    /// A width or height was out of range.
    InvalidDimensions(String),
    /// A manifest was malformed.
    InvalidManifest(String),
    /// A manifest's signature did not verify against its owner.
    BadSignature,
    /// No photo with this manifest CID is known here.
    NotFound(Cid),
}

impl fmt::Display for PhotoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhotoError::Storage(e) => write!(f, "storage error: {e}"),
            PhotoError::Identity(e) => write!(f, "identity error: {e}"),
            PhotoError::InvalidName(msg) => write!(f, "invalid photo name: {msg}"),
            PhotoError::InvalidFormat(msg) => write!(f, "invalid photo format: {msg}"),
            PhotoError::InvalidDimensions(msg) => write!(f, "invalid dimensions: {msg}"),
            PhotoError::InvalidManifest(msg) => write!(f, "invalid manifest: {msg}"),
            PhotoError::BadSignature => write!(f, "manifest signature is invalid"),
            PhotoError::NotFound(cid) => write!(f, "photo not found: {cid}"),
        }
    }
}

impl std::error::Error for PhotoError {}

impl From<StorageError> for PhotoError {
    fn from(e: StorageError) -> Self {
        PhotoError::Storage(e)
    }
}

impl From<IdentityError> for PhotoError {
    fn from(e: IdentityError) -> Self {
        PhotoError::Identity(e)
    }
}

/// Rules for photo names: 1-128 printable characters, no path
/// separators. Names are human labels (`"sunset.jpg"`), not hashes,
/// and a name must never be able to escape into a path.
fn validate_name(name: &str) -> Result<(), PhotoError> {
    if name.is_empty() {
        return Err(PhotoError::InvalidName(
            "photo name must not be empty".to_string(),
        ));
    }
    if name.len() > 128 {
        return Err(PhotoError::InvalidName(
            "photo name must be at most 128 characters".to_string(),
        ));
    }
    if name.contains('/') || name.contains('\\') {
        return Err(PhotoError::InvalidName(
            "photo name must not contain a path separator".to_string(),
        ));
    }
    if !name.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
        return Err(PhotoError::InvalidName(
            "photo name must be printable ASCII".to_string(),
        ));
    }
    Ok(())
}

/// Rules for format strings: 1-16 characters, ASCII letters, digits,
/// `-` and `+` only (`"jpeg"`, `"png"`, `"webp"`, `"heic"`).
fn validate_format(format: &str) -> Result<(), PhotoError> {
    if format.is_empty() {
        return Err(PhotoError::InvalidFormat(
            "photo format must not be empty".to_string(),
        ));
    }
    if format.len() > 16 {
        return Err(PhotoError::InvalidFormat(
            "photo format must be at most 16 characters".to_string(),
        ));
    }
    if !format
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '+')
    {
        return Err(PhotoError::InvalidFormat(
            "photo format may only use letters, digits, '-' and '+'".to_string(),
        ));
    }
    Ok(())
}

/// Rules for dimensions: both must be at least 1 pixel and at most
/// [`MAX_DIMENSION`].
fn validate_dimensions(width: u32, height: u32) -> Result<(), PhotoError> {
    if width == 0 || height == 0 {
        return Err(PhotoError::InvalidDimensions(
            "width and height must be at least 1 pixel".to_string(),
        ));
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(PhotoError::InvalidDimensions(format!(
            "dimensions must be at most {MAX_DIMENSION} pixels"
        )));
    }
    Ok(())
}

/// The message an owner signs for a manifest: domain + name + root +
/// width + height + format + taken + owner key.
fn manifest_sign_message(
    name: &str,
    root: &Cid,
    width: u32,
    height: u32,
    format: &str,
    taken: u64,
    owner: &PeerId,
) -> Vec<u8> {
    let mut msg = Vec::with_capacity(
        MANIFEST_SIGN_DOMAIN.len() + name.len() + 32 + 4 + 4 + format.len() + 8 + 32,
    );
    msg.extend_from_slice(MANIFEST_SIGN_DOMAIN);
    msg.extend_from_slice(name.as_bytes());
    msg.extend_from_slice(root.as_bytes());
    msg.extend_from_slice(&width.to_le_bytes());
    msg.extend_from_slice(&height.to_le_bytes());
    msg.extend_from_slice(format.as_bytes());
    msg.extend_from_slice(&taken.to_le_bytes());
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg
}

/// A signed photo manifest: name, chunked-content root, owner,
/// signature, plus photo metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhotoManifest {
    /// Human-readable photo name.
    pub name: String,
    /// The owner who signed this manifest.
    pub owner: PeerId,
    /// Root CID of the chunked content (a `KCHUNK1` manifest block).
    pub root: Cid,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Image format, e.g. `"jpeg"`.
    pub format: String,
    /// Unix timestamp (seconds) the photo was taken.
    pub taken: u64,
    /// The owner's signature over name, root, metadata, and owner key.
    pub signature: Ed25519Signature,
}

impl PhotoManifest {
    /// Build and sign a manifest as `identity`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &str,
        root: Cid,
        width: u32,
        height: u32,
        format: &str,
        taken: u64,
        identity: &Identity,
    ) -> Result<Self, PhotoError> {
        validate_name(name)?;
        validate_format(format)?;
        validate_dimensions(width, height)?;
        let owner = identity.peer_id().clone();
        let msg = manifest_sign_message(name, &root, width, height, format, taken, &owner);
        let signature = identity.sign(&msg);
        Ok(Self {
            name: name.to_string(),
            owner,
            root,
            width,
            height,
            format: format.to_string(),
            taken,
            signature,
        })
    }

    /// True if the signature verifies against the owner for this exact
    /// name, root, and metadata. Any tampering fails this check.
    pub fn verify(&self) -> bool {
        let msg = manifest_sign_message(
            &self.name,
            &self.root,
            self.width,
            self.height,
            &self.format,
            self.taken,
            &self.owner,
        );
        self.owner.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary manifest format.
    pub fn encode(&self) -> Vec<u8> {
        let name_bytes = self.name.as_bytes();
        let format_bytes = self.format.as_bytes();
        let mut out = Vec::with_capacity(
            7 + 2 + name_bytes.len() + 32 + 32 + 4 + 4 + 1 + format_bytes.len() + 8 + 64,
        );
        out.extend_from_slice(MANIFEST_MAGIC);
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(self.owner.public_key().as_bytes());
        out.extend_from_slice(self.root.as_bytes());
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());
        out.push(format_bytes.len() as u8);
        out.extend_from_slice(format_bytes);
        out.extend_from_slice(&self.taken.to_le_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a manifest produced by [`PhotoManifest::encode`].
    /// Rejects anything that is not exactly that format.
    pub fn decode(bytes: &[u8]) -> Result<Self, PhotoError> {
        if bytes.len() < 7 + 2 + 32 + 32 + 4 + 4 + 1 + 8 + 64 || &bytes[..7] != MANIFEST_MAGIC {
            return Err(PhotoError::InvalidManifest(
                "missing KPHOTO1 magic".to_string(),
            ));
        }
        let name_len =
            u16::from_le_bytes(bytes[7..9].try_into().expect("2-byte slice")) as usize;
        // The format length sits after the fixed fields; read it once
        // the name length is known.
        let format_len_off = 9 + name_len + 32 + 32 + 4 + 4;
        if bytes.len() < format_len_off + 1 {
            return Err(PhotoError::InvalidManifest(
                "manifest too short for format length".to_string(),
            ));
        }
        let format_len = bytes[format_len_off] as usize;
        let expected = format_len_off + 1 + format_len + 8 + 64;
        if bytes.len() != expected {
            return Err(PhotoError::InvalidManifest(format!(
                "manifest is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let mut off = 9;
        let name_bytes = &bytes[off..off + name_len];
        let name = String::from_utf8(name_bytes.to_vec()).map_err(|_| {
            PhotoError::InvalidManifest("manifest name is not valid UTF-8".to_string())
        })?;
        validate_name(&name)?;
        off += name_len;
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let owner_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| PhotoError::InvalidManifest(e.to_string()))?;
        let mut root_bytes = [0u8; 32];
        root_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let root = Cid::from_bytes(root_bytes);
        let width = u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4-byte slice"));
        off += 4;
        let height = u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4-byte slice"));
        off += 4;
        let format_len = bytes[off] as usize;
        off += 1;
        let format_bytes = &bytes[off..off + format_len];
        let format = String::from_utf8(format_bytes.to_vec()).map_err(|_| {
            PhotoError::InvalidManifest("manifest format is not valid UTF-8".to_string())
        })?;
        validate_format(&format)?;
        off += format_len;
        let taken = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        validate_dimensions(width, height)?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        // The manifest stores the owner's public key, not their
        // display name: verification needs the key, and the key is
        // what authenticates the owner. Decode rebuilds the PeerId
        // with a deterministic name derived from the key, so a
        // decoded manifest verifies exactly like the original.
        let owner = PeerId::new(owner_key, &fingerprint_name(&owner_key))?;
        Ok(Self {
            name,
            owner,
            root,
            width,
            height,
            format,
            taken,
            signature,
        })
    }
}

/// A display name derived from a public key: the first 8 hex chars of
/// the key. Used when a manifest is decoded and only the key survives
/// (verification needs the key, not the human name).
fn fingerprint_name(key: &kernel_crypto::Ed25519PublicKey) -> String {
    let hex = kernel_crypto::to_hex(key.as_bytes());
    format!("peer-{}", &hex[..8])
}

/// The metadata of a photo, as a human sees it — no CIDs, no hashes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhotoMetadata {
    /// Human-readable photo name.
    pub name: String,
    /// Who took (owns) this photo.
    pub owner: PeerId,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Image format, e.g. `"jpeg"`.
    pub format: String,
    /// Unix timestamp (seconds) the photo was taken.
    pub taken: u64,
}

/// A photo store: a content store plus an index of manifests.
/// Photos are addressed by manifest CID; humans see photo names.
pub struct PhotoStore {
    store: ContentStore,
    /// Manifests this node holds, by manifest CID.
    manifests: HashMap<Cid, PhotoManifest>,
}

impl PhotoStore {
    /// An empty photo store.
    pub fn new() -> Self {
        Self {
            store: ContentStore::new(),
            manifests: HashMap::new(),
        }
    }

    /// Number of photos held.
    pub fn len(&self) -> usize {
        self.manifests.len()
    }

    /// True if no photos are held.
    pub fn is_empty(&self) -> bool {
        self.manifests.is_empty()
    }

    /// The manifest for a photo, if this node holds it.
    pub fn get_manifest(&self, manifest_cid: &Cid) -> Option<&PhotoManifest> {
        self.manifests.get(manifest_cid)
    }

    /// Store a photo: chunk the bytes, build and sign the manifest,
    /// and store the manifest as a block. Returns the manifest CID —
    /// the photo's address. The same name + content + metadata stored
    /// twice returns the same manifest CID (content addressing).
    #[allow(clippy::too_many_arguments)]
    pub fn put_photo(
        &mut self,
        name: &str,
        data: &[u8],
        width: u32,
        height: u32,
        format: &str,
        taken: u64,
        identity: &Identity,
    ) -> Result<Cid, PhotoError> {
        validate_name(name)?;
        validate_format(format)?;
        validate_dimensions(width, height)?;
        let root = self.store.put_chunked(data);
        let manifest = PhotoManifest::new(name, root, width, height, format, taken, identity)?;
        let manifest_bytes = manifest.encode();
        let manifest_cid = self.store.put(&manifest_bytes);
        self.manifests.insert(manifest_cid, manifest);
        Ok(manifest_cid)
    }

    /// Fetch a photo by manifest CID: verify the manifest signature,
    /// then reassemble the chunked content (every chunk hash-verified).
    /// Returns the photo bytes and its human-readable metadata.
    /// A tampered manifest or a tampered chunk fails loudly.
    pub fn get_photo(
        &self,
        manifest_cid: &Cid,
    ) -> Result<(Vec<u8>, PhotoMetadata), PhotoError> {
        let manifest = match self.manifests.get(manifest_cid) {
            Some(m) => m.clone(),
            None => {
                // Not in the index: maybe the manifest block arrived
                // without being indexed. Decode it from the store.
                let bytes = self.store.get(manifest_cid).map_err(|e| match e {
                    StorageError::NotFound(cid) => PhotoError::NotFound(cid),
                    other => PhotoError::Storage(other),
                })?;
                PhotoManifest::decode(&bytes)?
            }
        };
        if !manifest.verify() {
            return Err(PhotoError::BadSignature);
        }
        let data = self
            .store
            .get_chunked(&manifest.root)
            .map_err(PhotoError::Storage)?;
        let metadata = PhotoMetadata {
            name: manifest.name,
            owner: manifest.owner,
            width: manifest.width,
            height: manifest.height,
            format: manifest.format,
            taken: manifest.taken,
        };
        Ok((data, metadata))
    }

    /// All photos held, as (manifest CID, photo name), sorted by name
    /// for a stable, human-readable listing.
    pub fn list_photos(&self) -> Vec<(Cid, String)> {
        let mut out: Vec<(Cid, String)> = self
            .manifests
            .iter()
            .map(|(cid, m)| (*cid, m.name.clone()))
            .collect();
        out.sort_by(|a, b| a.1.cmp(&b.1));
        out
    }

    /// Delete a photo: remove its manifest and any chunks no other
    /// photo references. Chunks shared with another photo (identical
    /// content stored under a different name) are kept. Returns `true`
    /// if a photo was deleted, `false` if no such photo was held.
    pub fn delete_photo(&mut self, manifest_cid: &Cid) -> Result<bool, PhotoError> {
        let manifest = match self.manifests.remove(manifest_cid) {
            Some(m) => m,
            None => return Ok(false),
        };
        // Chunks still referenced by the photos that remain.
        let mut referenced: HashSet<Cid> = HashSet::new();
        for m in self.manifests.values() {
            referenced.insert(m.root);
            if let Ok(bytes) = self.store.get(&m.root) {
                if let Ok(chunk_cids) = decode_manifest(&bytes) {
                    referenced.extend(chunk_cids);
                }
            }
        }
        // The deleted photo's own chunk manifest and chunks.
        let mut doomed: Vec<Cid> = vec![manifest.root];
        if let Ok(bytes) = self.store.get(&manifest.root) {
            if let Ok(chunk_cids) = decode_manifest(&bytes) {
                doomed.extend(chunk_cids);
            }
        }
        for cid in doomed {
            if !referenced.contains(&cid) {
                self.store.remove(&cid);
            }
        }
        self.store.remove(manifest_cid);
        Ok(true)
    }

    /// Mutable access to the underlying content store. Test-only:
    /// lets tests simulate corruption (see `put_unchecked`).
    #[cfg(test)]
    pub fn store_mut(&mut self) -> &mut ContentStore {
        &mut self.store
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data(n: usize) -> Vec<u8> {
        (0..n).map(|i| ((i * 37 + 11) % 251) as u8).collect()
    }

    #[test]
    fn manifest_encode_decode_round_trip_and_verify() {
        let alice = Identity::generate("alice").unwrap();
        let root = Cid::from_bytes(kernel_crypto::sha256(b"photo chunks"));
        let manifest =
            PhotoManifest::new("sunset.jpg", root, 4032, 3024, "jpeg", 1_700_000_000, &alice)
                .unwrap();
        assert!(manifest.verify());
        let encoded = manifest.encode();
        let decoded = PhotoManifest::decode(&encoded).unwrap();
        assert_eq!(decoded.name, "sunset.jpg");
        assert_eq!(decoded.root, root);
        assert_eq!(decoded.width, 4032);
        assert_eq!(decoded.height, 3024);
        assert_eq!(decoded.format, "jpeg");
        assert_eq!(decoded.taken, 1_700_000_000);
        assert_eq!(decoded.signature, manifest.signature);
        assert_eq!(
            decoded.owner.public_key(),
            manifest.owner.public_key()
        );
        assert!(decoded.verify());
        assert!(PhotoManifest::decode(b"garbage").is_err());
        assert!(PhotoManifest::decode(&encoded[..encoded.len() - 1]).is_err());
    }

    #[test]
    fn photo_round_trip_small_and_large() {
        let alice = Identity::generate("alice").unwrap();
        let mut photos = PhotoStore::new();
        let small = b"fake jpeg bytes";
        let cid = photos
            .put_photo("thumb.jpg", small, 320, 240, "jpeg", 1_700_000_000, &alice)
            .unwrap();
        let (data, meta) = photos.get_photo(&cid).unwrap();
        assert_eq!(data, small);
        assert_eq!(meta.name, "thumb.jpg");
        assert_eq!(meta.width, 320);
        assert_eq!(meta.height, 240);
        assert_eq!(meta.format, "jpeg");
        assert_eq!(meta.taken, 1_700_000_000);
        assert_eq!(meta.owner, *alice.peer_id());
        assert_eq!(photos.len(), 1);

        // 600 KiB: spans 3 chunks.
        let large = test_data(600 * 1024);
        let big_cid = photos
            .put_photo("big.png", &large, 8000, 6000, "png", 1_700_000_001, &alice)
            .unwrap();
        let (data, meta) = photos.get_photo(&big_cid).unwrap();
        assert_eq!(data, large);
        assert_eq!(meta.format, "png");
        assert_eq!(photos.len(), 2);
        // Same name + content + metadata -> same manifest CID.
        assert_eq!(
            photos
                .put_photo("thumb.jpg", small, 320, 240, "jpeg", 1_700_000_000, &alice)
                .unwrap(),
            cid
        );
    }

    #[test]
    fn tampered_chunk_fails() {
        let alice = Identity::generate("alice").unwrap();
        let mut photos = PhotoStore::new();
        let data = test_data(600 * 1024);
        let cid = photos
            .put_photo("big.png", &data, 8000, 6000, "png", 1_700_000_000, &alice)
            .unwrap();
        let manifest = photos.get_manifest(&cid).unwrap().clone();
        let manifest_bytes = photos.store_mut().get(&manifest.root).unwrap();
        let chunk_cids = decode_manifest(&manifest_bytes).unwrap();
        let mut bad = photos.store_mut().get(&chunk_cids[1]).unwrap();
        bad[0] ^= 0xff;
        photos.store_mut().put_unchecked(&chunk_cids[1], bad);
        match photos.get_photo(&cid) {
            Err(PhotoError::Storage(StorageError::CorruptChunk { cid: got, .. })) => {
                assert_eq!(got, chunk_cids[1]);
            }
            other => panic!("expected CorruptChunk, got {other:?}"),
        }
    }

    #[test]
    fn tampered_manifest_signature_fails() {
        let alice = Identity::generate("alice").unwrap();
        let mut photos = PhotoStore::new();
        let cid = photos
            .put_photo("doc.jpg", b"important", 100, 100, "jpeg", 1_700_000_000, &alice)
            .unwrap();
        // Take the manifest bytes, flip a signature byte, and store the
        // tampered manifest as a new block.
        let mut bytes = photos.store_mut().get(&cid).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let tampered_cid = photos.store_mut().put(&bytes);
        assert_ne!(tampered_cid, cid);
        assert_eq!(
            photos.get_photo(&tampered_cid),
            Err(PhotoError::BadSignature)
        );
        // The original photo still reads fine.
        assert_eq!(photos.get_photo(&cid).unwrap().0, b"important");
    }

    #[test]
    fn tampered_metadata_breaks_signature() {
        let alice = Identity::generate("alice").unwrap();
        let root = Cid::from_bytes(kernel_crypto::sha256(b"chunks"));
        let mut manifest =
            PhotoManifest::new("a.jpg", root, 100, 100, "jpeg", 1_700_000_000, &alice).unwrap();
        assert!(manifest.verify());
        // Changing metadata after signing breaks verification.
        manifest.width = 200;
        assert!(!manifest.verify());
        manifest.width = 100;
        manifest.taken = 1_700_000_001;
        assert!(!manifest.verify());
    }

    #[test]
    fn list_shows_what_was_put_sorted_by_name() {
        let alice = Identity::generate("alice").unwrap();
        let mut photos = PhotoStore::new();
        assert!(photos.list_photos().is_empty());
        photos
            .put_photo("b.jpg", b"2", 10, 10, "jpeg", 1, &alice)
            .unwrap();
        photos
            .put_photo("a.png", b"1", 10, 10, "png", 1, &alice)
            .unwrap();
        photos
            .put_photo("c.webp", b"3", 10, 10, "webp", 1, &alice)
            .unwrap();
        let listed = photos.list_photos();
        assert_eq!(listed.len(), 3);
        let names: Vec<&str> = listed.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, vec!["a.png", "b.jpg", "c.webp"]);
    }

    #[test]
    fn delete_removes_photo_and_unreferenced_chunks() {
        let alice = Identity::generate("alice").unwrap();
        let mut photos = PhotoStore::new();
        let data = test_data(600 * 1024);
        let cid = photos
            .put_photo("big.png", &data, 8000, 6000, "png", 1_700_000_000, &alice)
            .unwrap();
        let manifest = photos.get_manifest(&cid).unwrap().clone();
        let manifest_bytes = photos.store_mut().get(&manifest.root).unwrap();
        let chunk_cids = decode_manifest(&manifest_bytes).unwrap();
        assert_eq!(chunk_cids.len(), 3);
        assert!(photos.delete_photo(&cid).unwrap());
        assert!(!photos.delete_photo(&cid).unwrap());
        assert_eq!(photos.get_photo(&cid), Err(PhotoError::NotFound(cid)));
        assert!(photos.list_photos().is_empty());
        assert!(!photos.store_mut().has(&cid));
        assert!(!photos.store_mut().has(&manifest.root));
        for chunk in &chunk_cids {
            assert!(!photos.store_mut().has(chunk));
        }
    }

    #[test]
    fn delete_keeps_chunks_shared_with_another_photo() {
        let alice = Identity::generate("alice").unwrap();
        let mut photos = PhotoStore::new();
        let data = b"shared photo bytes";
        let cid_a = photos
            .put_photo("a.jpg", data, 10, 10, "jpeg", 1, &alice)
            .unwrap();
        let cid_b = photos
            .put_photo("b.jpg", data, 10, 10, "jpeg", 1, &alice)
            .unwrap();
        assert_ne!(cid_a, cid_b); // different names -> different manifests
        assert!(photos.delete_photo(&cid_a).unwrap());
        assert_eq!(photos.get_photo(&cid_b).unwrap().0, data);
        assert_eq!(photos.list_photos().len(), 1);
    }

    #[test]
    fn photo_name_format_and_dimension_rules() {
        let alice = Identity::generate("alice").unwrap();
        let mut photos = PhotoStore::new();
        assert!(photos
            .put_photo("", b"x", 10, 10, "jpeg", 1, &alice)
            .is_err());
        assert!(photos
            .put_photo("a/b.jpg", b"x", 10, 10, "jpeg", 1, &alice)
            .is_err());
        assert!(photos
            .put_photo("ok.jpg", b"x", 10, 10, "", 1, &alice)
            .is_err());
        assert!(photos
            .put_photo("ok.jpg", b"x", 10, 10, "j peg", 1, &alice)
            .is_err());
        assert!(photos
            .put_photo("ok.jpg", b"x", 10, 10, "jpeg", 1, &alice)
            .is_ok());
        assert!(photos
            .put_photo("ok.jpg", b"x", 0, 10, "jpeg", 1, &alice)
            .is_err());
        assert!(photos
            .put_photo("ok.jpg", b"x", 10, 0, "jpeg", 1, &alice)
            .is_err());
        assert!(photos
            .put_photo("ok.jpg", b"x", MAX_DIMENSION + 1, 10, "jpeg", 1, &alice)
            .is_err());
    }

    #[test]
    fn get_unknown_photo_is_not_found() {
        let photos = PhotoStore::new();
        let cid = Cid::from_bytes(kernel_crypto::sha256(b"never stored"));
        assert_eq!(photos.get_photo(&cid), Err(PhotoError::NotFound(cid)));
    }
}
