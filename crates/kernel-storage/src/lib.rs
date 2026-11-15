//! Content-addressed storage for KERNEL.
//!
//! A block's address is the SHA-256 of its bytes, so content verifies
//! itself: if the bytes don't hash to the CID you asked for, they are not
//! the content you asked for.
//!
//! This store is **in-memory only**. Nothing here is a system of record
//! on disk — KERNEL storage is peer-backed, and persistence is a separate
//! design decision, not something smuggled in through a local database.
//! No JSON-on-disk, no SQLite, no Postgres, no S3.

use std::collections::HashMap;
use std::fmt;

use kernel_crypto::{from_hex, sha256, to_hex, CryptoError};

/// Errors returned by storage operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    /// No block with this CID is in the store.
    NotFound(Cid),
    /// A chunk's bytes did not hash to the CID the manifest claimed.
    CorruptChunk { cid: Cid, actual: Cid },
    /// A chunk manifest was malformed.
    InvalidManifest(String),
    /// A CID string did not parse.
    InvalidCid(String),
    /// Wraps a [`CryptoError`].
    Crypto(CryptoError),
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::NotFound(cid) => write!(f, "block not found: {cid}"),
            StorageError::CorruptChunk { cid, actual } => write!(
                f,
                "chunk {cid} is corrupt: bytes hash to {actual} instead"
            ),
            StorageError::InvalidManifest(msg) => write!(f, "invalid chunk manifest: {msg}"),
            StorageError::InvalidCid(msg) => write!(f, "invalid CID: {msg}"),
            StorageError::Crypto(e) => write!(f, "crypto error: {e}"),
        }
    }
}

impl std::error::Error for StorageError {}

impl From<CryptoError> for StorageError {
    fn from(e: CryptoError) -> Self {
        StorageError::Crypto(e)
    }
}

/// A content identifier: the SHA-256 of the block's bytes.
/// Displayed as lowercase hex. Humans should normally see names, not CIDs,
/// but when a CID must be shown, hex is the readable form.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Cid([u8; 32]);

impl Cid {
    /// Build a CID from 32 digest bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The raw 32 digest bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase hex form.
    pub fn to_hex(&self) -> String {
        to_hex(&self.0)
    }

    /// Parse a lowercase hex CID.
    pub fn from_hex(s: &str) -> Result<Self, StorageError> {
        let bytes = from_hex(s).map_err(|e| StorageError::InvalidCid(e.to_string()))?;
        if bytes.len() != 32 {
            return Err(StorageError::InvalidCid(format!(
                "CID must be 32 bytes, got {}",
                bytes.len()
            )));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(Self(arr))
    }
}

impl fmt::Display for Cid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Size of one chunk for chunked storage (256 KiB).
pub const CHUNK_SIZE: usize = 256 * 1024;

/// Magic bytes at the start of every chunk manifest.
const MANIFEST_MAGIC: &[u8; 7] = b"KCHUNK1";

/// Encode a chunk manifest: magic + u32 little-endian chunk count +
/// 32 bytes per chunk CID. Explicit binary format, no JSON.
pub fn encode_manifest(chunk_cids: &[Cid]) -> Vec<u8> {
    let mut out = Vec::with_capacity(7 + 4 + 32 * chunk_cids.len());
    out.extend_from_slice(MANIFEST_MAGIC);
    out.extend_from_slice(&(chunk_cids.len() as u32).to_le_bytes());
    for cid in chunk_cids {
        out.extend_from_slice(cid.as_bytes());
    }
    out
}

/// Decode a chunk manifest produced by [`encode_manifest`].
pub fn decode_manifest(bytes: &[u8]) -> Result<Vec<Cid>, StorageError> {
    if bytes.len() < 11 || &bytes[..7] != MANIFEST_MAGIC {
        return Err(StorageError::InvalidManifest(
            "missing KCHUNK1 magic".to_string(),
        ));
    }
    let count = u32::from_le_bytes(bytes[7..11].try_into().expect("4-byte slice")) as usize;
    if bytes.len() != 11 + 32 * count {
        return Err(StorageError::InvalidManifest(format!(
            "manifest claims {count} chunks but is {} bytes",
            bytes.len()
        )));
    }
    let mut cids = Vec::with_capacity(count);
    for i in 0..count {
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes[11 + 32 * i..11 + 32 * (i + 1)]);
        cids.push(Cid(arr));
    }
    Ok(cids)
}

/// An in-memory content-addressed store.
///
/// `put` returns the CID (SHA-256) of the stored bytes. `get` returns the
/// bytes for a CID, or [`StorageError::NotFound`]. The same bytes always
/// produce the same CID, so storing the same content twice is a no-op.
#[derive(Default)]
pub struct ContentStore {
    blocks: HashMap<Cid, Vec<u8>>,
}

impl ContentStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store bytes, returning their CID.
    pub fn put(&mut self, data: &[u8]) -> Cid {
        let cid = Cid(sha256(data));
        self.blocks.insert(cid, data.to_vec());
        cid
    }

    /// Fetch the bytes for a CID.
    pub fn get(&self, cid: &Cid) -> Result<Vec<u8>, StorageError> {
        self.blocks
            .get(cid)
            .cloned()
            .ok_or(StorageError::NotFound(*cid))
    }

    /// True if the store holds a block with this CID.
    pub fn has(&self, cid: &Cid) -> bool {
        self.blocks.contains_key(cid)
    }

    /// Number of blocks in the store.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// True if the store holds no blocks.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Remove a block. Returns true if a block was present.
    pub fn remove(&mut self, cid: &Cid) -> bool {
        self.blocks.remove(cid).is_some()
    }

    /// Store large content split into fixed-size chunks.
    ///
    /// Each chunk is stored by its own CID, then a manifest listing the
    /// chunk CIDs is stored as a block of its own. Returns the root CID —
    /// the CID of the manifest. Empty content stores a manifest with zero
    /// chunks.
    pub fn put_chunked(&mut self, data: &[u8]) -> Cid {
        let mut chunk_cids = Vec::new();
        for chunk in data.chunks(CHUNK_SIZE) {
            chunk_cids.push(self.put(chunk));
        }
        let manifest = encode_manifest(&chunk_cids);
        self.put(&manifest)
    }

    /// Reassemble chunked content stored by [`ContentStore::put_chunked`].
    /// Every chunk is hash-verified against the manifest before it is
    /// accepted, so a corrupted or swapped chunk fails loudly instead of
    /// silently returning wrong bytes.
    pub fn get_chunked(&self, root: &Cid) -> Result<Vec<u8>, StorageError> {
        let manifest_bytes = self.get(root)?;
        let chunk_cids = decode_manifest(&manifest_bytes)?;
        let mut out = Vec::new();
        for cid in &chunk_cids {
            let chunk = self.get(cid)?;
            let actual = Cid(sha256(&chunk));
            if actual != *cid {
                return Err(StorageError::CorruptChunk {
                    cid: *cid,
                    actual,
                });
            }
            out.extend_from_slice(&chunk);
        }
        Ok(out)
    }

    /// Store bytes under a caller-supplied CID without verifying that
    /// the bytes hash to it. This breaks the content-addressing
    /// invariant and must only be used by tests (to simulate on-disk
    /// corruption) or by recovery tooling that is deliberately
    /// re-importing damaged data. Normal code must use [`ContentStore::put`].
    pub fn put_unchecked(&mut self, cid: &Cid, data: Vec<u8>) {
        self.blocks.insert(*cid, data);
    }

    /// All blocks held, as (CID, bytes). Lets a node enumerate what
    /// it holds — for example, to offer its blocks to a peer during
    /// sync or replication. The returned bytes hash to their CID
    /// (unless a block was deliberately imported unchecked).
    pub fn blocks(&self) -> Vec<(Cid, Vec<u8>)> {
        self.blocks
            .iter()
            .map(|(cid, data)| (*cid, data.clone()))
            .collect()
    }

    /// Test-only hook: overwrite the bytes stored under a CID without
    /// recomputing its hash. Used to simulate on-disk corruption.
    /// Does not exist in release builds.
    #[cfg(test)]
    pub fn corrupt(&mut self, cid: &Cid, data: Vec<u8>) {
        self.blocks.insert(*cid, data);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random test data (no RNG needed).
    fn test_data(n: usize) -> Vec<u8> {
        (0..n).map(|i| ((i * 37 + 11) % 251) as u8).collect()
    }

    #[test]
    fn put_get_round_trip() {
        let mut store = ContentStore::new();
        let data = b"kernel block one";
        let cid = store.put(data);
        assert_eq!(store.get(&cid).unwrap(), data);
        assert!(store.has(&cid));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn same_content_same_cid() {
        let mut store = ContentStore::new();
        let a = store.put(b"same");
        let b = store.put(b"same");
        assert_eq!(a, b);
        assert_eq!(store.len(), 1);
        assert_ne!(a, store.put(b"different"));
    }

    #[test]
    fn cid_is_sha256_hex() {
        let mut store = ContentStore::new();
        let cid = store.put(b"abc");
        assert_eq!(
            cid.to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(Cid::from_hex(&cid.to_hex()).unwrap(), cid);
        assert!(Cid::from_hex("zz").is_err());
        assert!(Cid::from_hex("aabb").is_err());
    }

    #[test]
    fn missing_cid_errors() {
        let store = ContentStore::new();
        let cid = Cid(sha256(b"never stored"));
        assert_eq!(store.get(&cid), Err(StorageError::NotFound(cid)));
        assert!(!store.has(&cid));
    }

    #[test]
    fn empty_block() {
        let mut store = ContentStore::new();
        let cid = store.put(b"");
        assert_eq!(store.get(&cid).unwrap(), b"");
    }

    #[test]
    fn remove_works() {
        let mut store = ContentStore::new();
        let cid = store.put(b"temp");
        assert!(store.remove(&cid));
        assert!(!store.has(&cid));
        assert!(!store.remove(&cid));
    }

    #[test]
    fn manifest_round_trip() {
        let cids = vec![Cid(sha256(b"a")), Cid(sha256(b"b"))];
        let encoded = encode_manifest(&cids);
        assert_eq!(decode_manifest(&encoded).unwrap(), cids);
        assert!(decode_manifest(b"garbage").is_err());
        assert!(decode_manifest(&encoded[..10]).is_err());
        let mut bad_count = encoded.clone();
        bad_count[7] = 9; // claim 9 chunks, only 2 present
        assert!(decode_manifest(&bad_count).is_err());
    }

    #[test]
    fn chunked_small_content_round_trip() {
        let mut store = ContentStore::new();
        let data = test_data(100);
        let root = store.put_chunked(&data);
        assert_eq!(store.get_chunked(&root).unwrap(), data);
    }

    #[test]
    fn chunked_empty_content_round_trip() {
        let mut store = ContentStore::new();
        let root = store.put_chunked(b"");
        assert_eq!(store.get_chunked(&root).unwrap(), b"");
    }

    #[test]
    fn chunked_large_content_round_trip() {
        let mut store = ContentStore::new();
        // 600 KiB: spans 3 chunks at 256 KiB.
        let data = test_data(600 * 1024);
        let root = store.put_chunked(&data);
        // 3 chunks + 1 manifest block.
        assert_eq!(store.len(), 4);
        assert_eq!(store.get_chunked(&root).unwrap(), data);
    }

    #[test]
    fn chunked_exact_boundary() {
        let mut store = ContentStore::new();
        let data = test_data(CHUNK_SIZE);
        let root = store.put_chunked(&data);
        assert_eq!(store.get_chunked(&root).unwrap(), data);
    }

    #[test]
    fn chunked_missing_chunk_fails() {
        let mut store = ContentStore::new();
        let data = test_data(600 * 1024);
        let root = store.put_chunked(&data);
        let manifest = store.get(&root).unwrap();
        let chunk_cids = decode_manifest(&manifest).unwrap();
        assert_eq!(chunk_cids.len(), 3);
        store.remove(&chunk_cids[1]);
        assert_eq!(
            store.get_chunked(&root),
            Err(StorageError::NotFound(chunk_cids[1]))
        );
    }

    #[test]
    fn chunked_corrupted_chunk_fails() {
        let mut store = ContentStore::new();
        let data = test_data(600 * 1024);
        let root = store.put_chunked(&data);
        let manifest = store.get(&root).unwrap();
        let chunk_cids = decode_manifest(&manifest).unwrap();
        // Corrupt the second chunk's bytes in place.
        let mut bad = store.get(&chunk_cids[1]).unwrap();
        bad[0] ^= 0xff;
        store.corrupt(&chunk_cids[1], bad);
        match store.get_chunked(&root) {
            Err(StorageError::CorruptChunk { cid, .. }) => assert_eq!(cid, chunk_cids[1]),
            other => panic!("expected CorruptChunk, got {other:?}"),
        }
    }

    #[test]
    fn chunked_missing_root_fails() {
        let store = ContentStore::new();
        let root = Cid(sha256(b"no manifest here"));
        assert_eq!(store.get_chunked(&root), Err(StorageError::NotFound(root)));
    }

    #[test]
    fn chunked_root_is_not_content() {
        // The root CID addresses the manifest, not the content itself,
        // so get() on the root returns manifest bytes, not the data.
        let mut store = ContentStore::new();
        let data = test_data(10);
        let root = store.put_chunked(&data);
        let manifest = store.get(&root).unwrap();
        assert_ne!(manifest, data);
        assert_eq!(decode_manifest(&manifest).unwrap().len(), 1);
    }
}
