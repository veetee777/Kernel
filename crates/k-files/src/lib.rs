//! K•Files — file storage on top of KERNEL content-addressed storage.
//!
//! A file is a human name plus bytes. The bytes are stored chunked in
//! the content store (see `kernel_storage`), and the file is described
//! by a signed manifest: the name, the chunked-content root CID, the
//! owner, and the owner's signature over all of it.
//!
//! The manifest is itself a block in the content store, addressed by
//! its own CID. Anyone holding the manifest CID can fetch the file —
//! and the signature plus per-chunk hash verification mean a tampered
//! manifest or a tampered chunk fails loudly instead of silently
//! returning wrong bytes.
//!
//! Manifest format (explicit binary, no JSON):
//! `KFILE1` + name length (u16 LE) + name bytes + owner public key
//! (32) + chunk root CID (32) + signature (64).

use std::collections::{HashMap, HashSet};
use std::fmt;

use kernel_crypto::Ed25519Signature;
use kernel_identity::{Identity, IdentityError, PeerId};
use kernel_storage::{decode_manifest, Cid, ContentStore, StorageError};

/// Magic bytes at the start of every file manifest.
const MANIFEST_MAGIC: &[u8; 6] = b"KFILE1";

/// Domain separator for manifest signatures: a signature covers
/// `b"KFILE1" || name || root CID bytes || owner public key bytes`.
const MANIFEST_SIGN_DOMAIN: &[u8] = b"KFILE1";

/// Errors returned by file operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileError {
    /// Wraps a [`StorageError`].
    Storage(StorageError),
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// A file name broke the naming rules.
    InvalidName(String),
    /// A manifest was malformed.
    InvalidManifest(String),
    /// A manifest's signature did not verify against its owner.
    BadSignature,
    /// No file with this manifest CID is known here.
    NotFound(Cid),
}

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileError::Storage(e) => write!(f, "storage error: {e}"),
            FileError::Identity(e) => write!(f, "identity error: {e}"),
            FileError::InvalidName(msg) => write!(f, "invalid file name: {msg}"),
            FileError::InvalidManifest(msg) => write!(f, "invalid manifest: {msg}"),
            FileError::BadSignature => write!(f, "manifest signature is invalid"),
            FileError::NotFound(cid) => write!(f, "file not found: {cid}"),
        }
    }
}

impl std::error::Error for FileError {}

impl From<StorageError> for FileError {
    fn from(e: StorageError) -> Self {
        FileError::Storage(e)
    }
}

impl From<IdentityError> for FileError {
    fn from(e: IdentityError) -> Self {
        FileError::Identity(e)
    }
}

/// Rules for file names: 1-128 printable characters, no path
/// separators. Names are human labels (`"notes.txt"`), not hashes,
/// and a name must never be able to escape into a path.
fn validate_name(name: &str) -> Result<(), FileError> {
    if name.is_empty() {
        return Err(FileError::InvalidName("file name must not be empty".to_string()));
    }
    if name.len() > 128 {
        return Err(FileError::InvalidName(
            "file name must be at most 128 characters".to_string(),
        ));
    }
    if name.contains('/') || name.contains('\\') {
        return Err(FileError::InvalidName(
            "file name must not contain a path separator".to_string(),
        ));
    }
    if !name.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
        return Err(FileError::InvalidName(
            "file name must be printable ASCII".to_string(),
        ));
    }
    Ok(())
}

/// The message an owner signs for a manifest: domain + name + root + key.
fn manifest_sign_message(name: &str, root: &Cid, owner: &PeerId) -> Vec<u8> {
    let mut msg = Vec::with_capacity(
        MANIFEST_SIGN_DOMAIN.len() + name.len() + 32 + 32,
    );
    msg.extend_from_slice(MANIFEST_SIGN_DOMAIN);
    msg.extend_from_slice(name.as_bytes());
    msg.extend_from_slice(root.as_bytes());
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg
}

/// A signed file manifest: name, chunked-content root, owner, signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileManifest {
    /// Human-readable file name.
    pub name: String,
    /// The owner who signed this manifest.
    pub owner: PeerId,
    /// Root CID of the chunked content (a `KCHUNK1` manifest block).
    pub root: Cid,
    /// The owner's signature over name, root, and owner key.
    pub signature: Ed25519Signature,
}

impl FileManifest {
    /// Build and sign a manifest for `name` + `root` as `identity`.
    pub fn new(name: &str, root: Cid, identity: &Identity) -> Result<Self, FileError> {
        validate_name(name)?;
        let owner = identity.peer_id().clone();
        let msg = manifest_sign_message(name, &root, &owner);
        let signature = identity.sign(&msg);
        Ok(Self {
            name: name.to_string(),
            owner,
            root,
            signature,
        })
    }

    /// True if the signature verifies against the owner for this exact
    /// name and root. Any tampering fails this check.
    pub fn verify(&self) -> bool {
        let msg = manifest_sign_message(&self.name, &self.root, &self.owner);
        self.owner.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary manifest format.
    pub fn encode(&self) -> Vec<u8> {
        let name_bytes = self.name.as_bytes();
        let mut out = Vec::with_capacity(6 + 2 + name_bytes.len() + 32 + 32 + 64);
        out.extend_from_slice(MANIFEST_MAGIC);
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(self.owner.public_key().as_bytes());
        out.extend_from_slice(self.root.as_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a manifest produced by [`FileManifest::encode`].
    /// Rejects anything that is not exactly that format.
    pub fn decode(bytes: &[u8]) -> Result<Self, FileError> {
        if bytes.len() < 6 + 2 + 32 + 32 + 64 || &bytes[..6] != MANIFEST_MAGIC {
            return Err(FileError::InvalidManifest(
                "missing KFILE1 magic".to_string(),
            ));
        }
        let name_len = u16::from_le_bytes(bytes[6..8].try_into().expect("2-byte slice")) as usize;
        let expected = 6 + 2 + name_len + 32 + 32 + 64;
        if bytes.len() != expected {
            return Err(FileError::InvalidManifest(format!(
                "manifest is {} bytes, expected {expected} for a {name_len}-byte name",
                bytes.len()
            )));
        }
        let mut off = 8;
        let name_bytes = &bytes[off..off + name_len];
        let name = String::from_utf8(name_bytes.to_vec()).map_err(|_| {
            FileError::InvalidManifest("manifest name is not valid UTF-8".to_string())
        })?;
        validate_name(&name)?;
        off += name_len;
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let owner_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| FileError::InvalidManifest(e.to_string()))?;
        let mut root_bytes = [0u8; 32];
        root_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let root = Cid::from_bytes(root_bytes);
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

/// A file store: a content store plus an index of manifests.
/// Files are addressed by manifest CID; humans see file names.
pub struct FileStore {
    store: ContentStore,
    /// Manifests this node holds, by manifest CID.
    manifests: HashMap<Cid, FileManifest>,
}

impl FileStore {
    /// An empty file store.
    pub fn new() -> Self {
        Self {
            store: ContentStore::new(),
            manifests: HashMap::new(),
        }
    }

    /// Number of files held.
    pub fn len(&self) -> usize {
        self.manifests.len()
    }

    /// True if no files are held.
    pub fn is_empty(&self) -> bool {
        self.manifests.is_empty()
    }

    /// The manifest for a file, if this node holds it.
    pub fn get_manifest(&self, manifest_cid: &Cid) -> Option<&FileManifest> {
        self.manifests.get(manifest_cid)
    }

    /// Store a file: chunk the bytes, build and sign the manifest, and
    /// store the manifest as a block. Returns the manifest CID — the
    /// file's address. The same name + content stored twice returns the
    /// same manifest CID (content addressing).
    pub fn put_file(
        &mut self,
        name: &str,
        data: &[u8],
        identity: &Identity,
    ) -> Result<Cid, FileError> {
        validate_name(name)?;
        let root = self.store.put_chunked(data);
        let manifest = FileManifest::new(name, root, identity)?;
        let manifest_bytes = manifest.encode();
        let manifest_cid = self.store.put(&manifest_bytes);
        self.manifests.insert(manifest_cid, manifest);
        Ok(manifest_cid)
    }

    /// Fetch a file by manifest CID: verify the manifest signature,
    /// then reassemble the chunked content (every chunk hash-verified).
    /// A tampered manifest or a tampered chunk fails loudly.
    pub fn get_file(&self, manifest_cid: &Cid) -> Result<Vec<u8>, FileError> {
        let manifest = match self.manifests.get(manifest_cid) {
            Some(m) => m.clone(),
            None => {
                // Not in the index: maybe the manifest block arrived
                // without being indexed. Decode it from the store.
                let bytes = self.store.get(manifest_cid).map_err(|e| match e {
                    StorageError::NotFound(cid) => FileError::NotFound(cid),
                    other => FileError::Storage(other),
                })?;
                FileManifest::decode(&bytes)?
            }
        };
        if !manifest.verify() {
            return Err(FileError::BadSignature);
        }
        self.store
            .get_chunked(&manifest.root)
            .map_err(FileError::Storage)
    }

    /// All files held, as (manifest CID, file name), sorted by name
    /// for a stable, human-readable listing.
    pub fn list_files(&self) -> Vec<(Cid, String)> {
        let mut out: Vec<(Cid, String)> = self
            .manifests
            .iter()
            .map(|(cid, m)| (*cid, m.name.clone()))
            .collect();
        out.sort_by(|a, b| a.1.cmp(&b.1));
        out
    }

    /// Delete a file: remove its manifest and any chunks no other file
    /// references. Chunks shared with another file (identical content
    /// stored under a different name) are kept. Returns `true` if a
    /// file was deleted, `false` if no such file was held.
    pub fn delete_file(&mut self, manifest_cid: &Cid) -> Result<bool, FileError> {
        let manifest = match self.manifests.remove(manifest_cid) {
            Some(m) => m,
            None => return Ok(false),
        };
        // Chunks still referenced by the files that remain.
        let mut referenced: HashSet<Cid> = HashSet::new();
        for m in self.manifests.values() {
            referenced.insert(m.root);
            if let Ok(bytes) = self.store.get(&m.root) {
                if let Ok(chunk_cids) = decode_manifest(&bytes) {
                    referenced.extend(chunk_cids);
                }
            }
        }
        // The deleted file's own chunk manifest and chunks.
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
        let root = Cid::from_bytes(kernel_crypto::sha256(b"chunk manifest"));
        let manifest = FileManifest::new("notes.txt", root, &alice).unwrap();
        assert!(manifest.verify());
        let encoded = manifest.encode();
        let decoded = FileManifest::decode(&encoded).unwrap();
        // The decoded owner key matches (the display name is rebuilt
        // from the key, so compare keys, not the whole PeerId).
        assert_eq!(decoded.name, "notes.txt");
        assert_eq!(decoded.root, root);
        assert_eq!(decoded.signature, manifest.signature);
        assert_eq!(
            decoded.owner.public_key(),
            manifest.owner.public_key()
        );
        assert!(decoded.verify());
        assert!(FileManifest::decode(b"garbage").is_err());
        assert!(FileManifest::decode(&encoded[..encoded.len() - 1]).is_err());
    }

    #[test]
    fn file_round_trip_small_and_large() {
        let alice = Identity::generate("alice").unwrap();
        let mut files = FileStore::new();
        let small = b"hello files";
        let cid = files.put_file("hello.txt", small, &alice).unwrap();
        assert_eq!(files.get_file(&cid).unwrap(), small);
        assert_eq!(files.len(), 1);

        // 600 KiB: spans 3 chunks.
        let large = test_data(600 * 1024);
        let big_cid = files.put_file("big.bin", &large, &alice).unwrap();
        assert_eq!(files.get_file(&big_cid).unwrap(), large);
        assert_eq!(files.len(), 2);
        // Same name + content -> same manifest CID.
        assert_eq!(files.put_file("hello.txt", small, &alice).unwrap(), cid);
    }

    #[test]
    fn tampered_chunk_fails() {
        let alice = Identity::generate("alice").unwrap();
        let mut files = FileStore::new();
        let data = test_data(600 * 1024);
        let cid = files.put_file("big.bin", &data, &alice).unwrap();
        // Find a chunk CID and corrupt its bytes in place.
        let manifest = files.get_manifest(&cid).unwrap().clone();
        let manifest_bytes = files.store_mut().get(&manifest.root).unwrap();
        let chunk_cids = decode_manifest(&manifest_bytes).unwrap();
        let mut bad = files.store_mut().get(&chunk_cids[1]).unwrap();
        bad[0] ^= 0xff;
        files
            .store_mut()
            .put_unchecked(&chunk_cids[1], bad);
        match files.get_file(&cid) {
            Err(FileError::Storage(StorageError::CorruptChunk { cid: got, .. })) => {
                assert_eq!(got, chunk_cids[1]);
            }
            other => panic!("expected CorruptChunk, got {other:?}"),
        }
    }

    #[test]
    fn tampered_manifest_signature_fails() {
        let alice = Identity::generate("alice").unwrap();
        let mut files = FileStore::new();
        let cid = files.put_file("doc.txt", b"important", &alice).unwrap();
        // Take the manifest bytes, flip a signature byte, and store the
        // tampered manifest as a new block.
        let mut bytes = files.store_mut().get(&cid).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let tampered_cid = files.store_mut().put(&bytes);
        assert_ne!(tampered_cid, cid);
        assert_eq!(
            files.get_file(&tampered_cid),
            Err(FileError::BadSignature)
        );
        // The original file still reads fine.
        assert_eq!(files.get_file(&cid).unwrap(), b"important");
    }

    #[test]
    fn list_shows_what_was_put() {
        let alice = Identity::generate("alice").unwrap();
        let mut files = FileStore::new();
        assert!(files.list_files().is_empty());
        files.put_file("b.txt", b"2", &alice).unwrap();
        files.put_file("a.txt", b"1", &alice).unwrap();
        files.put_file("c.txt", b"3", &alice).unwrap();
        let listed = files.list_files();
        assert_eq!(listed.len(), 3);
        // Sorted by name for a stable listing.
        let names: Vec<&str> = listed.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, vec!["a.txt", "b.txt", "c.txt"]);
    }

    #[test]
    fn delete_removes_file_and_unreferenced_chunks() {
        let alice = Identity::generate("alice").unwrap();
        let mut files = FileStore::new();
        let data = test_data(600 * 1024);
        let cid = files.put_file("big.bin", &data, &alice).unwrap();
        let manifest = files.get_manifest(&cid).unwrap().clone();
        let manifest_bytes = files.store_mut().get(&manifest.root).unwrap();
        let chunk_cids = decode_manifest(&manifest_bytes).unwrap();
        assert_eq!(chunk_cids.len(), 3);
        assert!(files.delete_file(&cid).unwrap());
        assert!(!files.delete_file(&cid).unwrap());
        assert_eq!(files.get_file(&cid), Err(FileError::NotFound(cid)));
        assert!(files.list_files().is_empty());
        // The manifest block and all three chunks are gone.
        assert!(!files.store_mut().has(&cid));
        assert!(!files.store_mut().has(&manifest.root));
        for chunk in &chunk_cids {
            assert!(!files.store_mut().has(chunk));
        }
    }

    #[test]
    fn delete_keeps_chunks_shared_with_another_file() {
        let alice = Identity::generate("alice").unwrap();
        let mut files = FileStore::new();
        let data = b"shared content";
        let cid_a = files.put_file("a.txt", data, &alice).unwrap();
        let cid_b = files.put_file("b.txt", data, &alice).unwrap();
        assert_ne!(cid_a, cid_b); // different names -> different manifests
        assert!(files.delete_file(&cid_a).unwrap());
        // b.txt still reads: its chunks were shared and kept.
        assert_eq!(files.get_file(&cid_b).unwrap(), data);
        assert_eq!(files.list_files().len(), 1);
    }

    #[test]
    fn file_name_rules() {
        let alice = Identity::generate("alice").unwrap();
        let mut files = FileStore::new();
        assert!(files.put_file("", b"x", &alice).is_err());
        assert!(files.put_file("a/b.txt", b"x", &alice).is_err());
        assert!(files.put_file("a\\b.txt", b"x", &alice).is_err());
        assert!(files.put_file("ok name-1.txt", b"x", &alice).is_ok());
    }

    #[test]
    fn get_unknown_file_is_not_found() {
        let files = FileStore::new();
        let cid = Cid::from_bytes(kernel_crypto::sha256(b"never stored"));
        assert_eq!(files.get_file(&cid), Err(FileError::NotFound(cid)));
    }
}
