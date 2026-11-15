//! Kernel•Vision — video on top of KERNEL content-addressed storage.
//!
//! A video is a human title plus video bytes. The bytes are stored
//! chunked in the content store, and the video is described by a signed
//! manifest: title, duration in seconds, width, height, the chunked
//! content root CID, the owner, and the owner's signature.
//!
//! Besides whole-video fetch, this crate offers per-chunk access:
//! [`VideoStore::get_chunk`] returns one chunk of a video by index,
//! hash-verified against the chunk manifest. A player can fetch the
//! chunk it needs without reassembling the whole video first.
//!
//! The manifest is itself a block in the content store, addressed by its
//! own CID. Anyone holding the manifest CID can fetch the video — and the
//! signature plus per-chunk hash verification mean a tampered manifest
//! or a tampered chunk fails loudly instead of silently returning wrong
//! bytes.
//!
//! Nothing here decodes or plays video. This crate stores, verifies,
//! and lists videos and their chunks. Playback lives in [`playback`]:
//! raw YUV420 frame streams (`KVFR`), conversion, scaling, and a
//! frame-clocked player that presents to sinks such as
//! [`playback::PpmSink`].
//!
//! Manifest format (explicit binary, no JSON):
//! `KVIDEO1` + title length (u16 LE) + title bytes + duration (u64 LE)
//! + width (u32 LE) + height (u32 LE) + chunk root CID (32) + owner
//! public key (32) + signature (64).

// The playback pipeline: KVFR decoding, YUV->RGB, scaling, player.
pub mod playback;

use std::collections::HashMap;
use std::fmt;

use kernel_crypto::{sha256, Ed25519Signature};
use kernel_identity::{Identity, IdentityError, PeerId};
use kernel_storage::{decode_manifest, Cid, ContentStore, StorageError};

/// Magic bytes at the start of every video manifest.
const MANIFEST_MAGIC: &[u8; 7] = b"KVIDEO1";
/// Domain separator for manifest signatures.
const MANIFEST_SIGN_DOMAIN: &[u8] = b"KVIDEO1";

/// Maximum length of a video title.
pub const MAX_TITLE_LEN: usize = 128;
/// Maximum video duration: 24 hours, in seconds.
pub const MAX_DURATION_SECS: u64 = 86_400;
/// Maximum video dimension in either direction (pixels).
pub const MAX_DIMENSION: u32 = 100_000;

/// Errors returned by video operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VideoError {
    /// Wraps a [`StorageError`].
    Storage(StorageError),
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// A title broke the naming rules.
    InvalidTitle(String),
    /// A duration was out of range.
    InvalidDuration(String),
    /// A width or height was out of range.
    InvalidDimensions(String),
    /// A manifest was malformed.
    InvalidManifest(String),
    /// A chunk index was out of range for this video.
    InvalidChunkIndex(usize),
    /// A manifest's signature did not verify against its owner.
    BadSignature,
    /// No video with this manifest CID is known here.
    NotFound(Cid),
}

impl fmt::Display for VideoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoError::Storage(e) => write!(f, "storage error: {e}"),
            VideoError::Identity(e) => write!(f, "identity error: {e}"),
            VideoError::InvalidTitle(msg) => write!(f, "invalid video title: {msg}"),
            VideoError::InvalidDuration(msg) => write!(f, "invalid duration: {msg}"),
            VideoError::InvalidDimensions(msg) => write!(f, "invalid dimensions: {msg}"),
            VideoError::InvalidManifest(msg) => write!(f, "invalid manifest: {msg}"),
            VideoError::InvalidChunkIndex(i) => write!(f, "chunk index {i} out of range"),
            VideoError::BadSignature => write!(f, "manifest signature is invalid"),
            VideoError::NotFound(cid) => write!(f, "video not found: {cid}"),
        }
    }
}

impl std::error::Error for VideoError {}

impl From<StorageError> for VideoError {
    fn from(e: StorageError) -> Self {
        VideoError::Storage(e)
    }
}

impl From<IdentityError> for VideoError {
    fn from(e: IdentityError) -> Self {
        VideoError::Identity(e)
    }
}

/// Rules for video titles: 1-128 printable ASCII characters, no path
/// separators. Titles are human labels, not hashes, and must never be
/// able to escape into a path.
fn validate_title(title: &str) -> Result<(), VideoError> {
    if title.is_empty() {
        return Err(VideoError::InvalidTitle(
            "video title must not be empty".to_string(),
        ));
    }
    if title.len() > MAX_TITLE_LEN {
        return Err(VideoError::InvalidTitle(format!(
            "video title must be at most {MAX_TITLE_LEN} characters"
        )));
    }
    if title.contains('/') || title.contains('\\') {
        return Err(VideoError::InvalidTitle(
            "video title must not contain a path separator".to_string(),
        ));
    }
    if !title.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
        return Err(VideoError::InvalidTitle(
            "video title must be printable ASCII".to_string(),
        ));
    }
    Ok(())
}

/// Rules for durations: at least 1 second, at most [`MAX_DURATION_SECS`].
fn validate_duration(duration_secs: u64) -> Result<(), VideoError> {
    if duration_secs == 0 {
        return Err(VideoError::InvalidDuration(
            "duration must be at least 1 second".to_string(),
        ));
    }
    if duration_secs > MAX_DURATION_SECS {
        return Err(VideoError::InvalidDuration(format!(
            "duration must be at most {MAX_DURATION_SECS} seconds"
        )));
    }
    Ok(())
}

/// Rules for dimensions: both at least 1 pixel, at most [`MAX_DIMENSION`].
fn validate_dimensions(width: u32, height: u32) -> Result<(), VideoError> {
    if width == 0 || height == 0 {
        return Err(VideoError::InvalidDimensions(
            "width and height must be at least 1 pixel".to_string(),
        ));
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(VideoError::InvalidDimensions(format!(
            "dimensions must be at most {MAX_DIMENSION} pixels"
        )));
    }
    Ok(())
}

/// The message an owner signs for a manifest.
fn manifest_sign_message(
    title: &str,
    duration_secs: u64,
    width: u32,
    height: u32,
    root: &Cid,
    owner: &PeerId,
) -> Vec<u8> {
    let mut msg = Vec::with_capacity(
        MANIFEST_SIGN_DOMAIN.len() + title.len() + 8 + 4 + 4 + 32 + 32,
    );
    msg.extend_from_slice(MANIFEST_SIGN_DOMAIN);
    msg.extend_from_slice(title.as_bytes());
    msg.extend_from_slice(&duration_secs.to_le_bytes());
    msg.extend_from_slice(&width.to_le_bytes());
    msg.extend_from_slice(&height.to_le_bytes());
    msg.extend_from_slice(root.as_bytes());
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg
}

/// A display name derived from a public key: the first 8 hex chars of
/// the key. Used when a manifest is decoded and only the key survives
/// (verification needs the key, not the human name).
fn fingerprint_name(key: &kernel_crypto::Ed25519PublicKey) -> String {
    let hex = kernel_crypto::to_hex(key.as_bytes());
    format!("peer-{}", &hex[..8])
}

/// A signed video manifest: title, duration, dimensions,
/// chunked-content root, owner, signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoManifest {
    /// Human-readable video title.
    pub title: String,
    /// Video duration in seconds.
    pub duration_secs: u64,
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// The owner who signed this manifest.
    pub owner: PeerId,
    /// Root CID of the chunked video content (a `KCHUNK1` manifest block).
    pub root: Cid,
    /// The owner's signature over title, duration, dimensions, root, owner key.
    pub signature: Ed25519Signature,
}

impl VideoManifest {
    /// Build and sign a manifest as `identity`.
    pub fn new(
        title: &str,
        duration_secs: u64,
        width: u32,
        height: u32,
        root: Cid,
        identity: &Identity,
    ) -> Result<Self, VideoError> {
        validate_title(title)?;
        validate_duration(duration_secs)?;
        validate_dimensions(width, height)?;
        let owner = identity.peer_id().clone();
        let msg = manifest_sign_message(title, duration_secs, width, height, &root, &owner);
        let signature = identity.sign(&msg);
        Ok(Self {
            title: title.to_string(),
            duration_secs,
            width,
            height,
            owner,
            root,
            signature,
        })
    }

    /// True if the signature verifies against the owner for this exact
    /// title, duration, dimensions, and root. Any tampering fails this check.
    pub fn verify(&self) -> bool {
        let msg = manifest_sign_message(
            &self.title,
            self.duration_secs,
            self.width,
            self.height,
            &self.root,
            &self.owner,
        );
        self.owner.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary manifest format.
    pub fn encode(&self) -> Vec<u8> {
        let title_bytes = self.title.as_bytes();
        let mut out = Vec::with_capacity(7 + 2 + title_bytes.len() + 8 + 4 + 4 + 32 + 32 + 64);
        out.extend_from_slice(MANIFEST_MAGIC);
        out.extend_from_slice(&(title_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(title_bytes);
        out.extend_from_slice(&self.duration_secs.to_le_bytes());
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());
        out.extend_from_slice(self.root.as_bytes());
        out.extend_from_slice(self.owner.public_key().as_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a manifest produced by [`VideoManifest::encode`].
    /// Rejects anything that is not exactly that format.
    pub fn decode(bytes: &[u8]) -> Result<Self, VideoError> {
        if bytes.len() < 7 + 2 + 8 + 4 + 4 + 32 + 32 + 64 || &bytes[..7] != MANIFEST_MAGIC {
            return Err(VideoError::InvalidManifest(
                "missing KVIDEO1 magic".to_string(),
            ));
        }
        let title_len = u16::from_le_bytes(bytes[7..9].try_into().expect("2-byte slice")) as usize;
        let expected = 9 + title_len + 8 + 4 + 4 + 32 + 32 + 64;
        if bytes.len() != expected {
            return Err(VideoError::InvalidManifest(format!(
                "manifest is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let mut off = 9;
        let title = String::from_utf8(bytes[off..off + title_len].to_vec()).map_err(|_| {
            VideoError::InvalidManifest("manifest title is not valid UTF-8".to_string())
        })?;
        validate_title(&title)?;
        off += title_len;
        let duration_secs =
            u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        validate_duration(duration_secs)?;
        let width = u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4-byte slice"));
        off += 4;
        let height = u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4-byte slice"));
        off += 4;
        validate_dimensions(width, height)?;
        let mut root_bytes = [0u8; 32];
        root_bytes.copy_from_slice(&bytes[off..off + 32]);
        let root = Cid::from_bytes(root_bytes);
        off += 32;
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let owner_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| VideoError::InvalidManifest(e.to_string()))?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let owner = PeerId::new(owner_key, &fingerprint_name(&owner_key))?;
        Ok(Self {
            title,
            duration_secs,
            width,
            height,
            owner,
            root,
            signature,
        })
    }
}

/// The metadata of a video, as a human sees it — no CIDs, no hashes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoMetadata {
    /// Human-readable video title.
    pub title: String,
    /// Video duration in seconds.
    pub duration_secs: u64,
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Who owns (published) this video.
    pub owner: PeerId,
    /// Number of chunks the video content spans.
    pub chunk_count: usize,
}

/// A video store: a content store plus an index of manifests.
/// Videos are addressed by manifest CID; humans see titles.
pub struct VideoStore {
    store: ContentStore,
    /// Manifests this node holds, by manifest CID.
    manifests: HashMap<Cid, VideoManifest>,
}

impl VideoStore {
    /// An empty video store.
    pub fn new() -> Self {
        Self {
            store: ContentStore::new(),
            manifests: HashMap::new(),
        }
    }

    /// Number of videos held.
    pub fn len(&self) -> usize {
        self.manifests.len()
    }

    /// True if no videos are held.
    pub fn is_empty(&self) -> bool {
        self.manifests.is_empty()
    }

    /// The manifest for a video, if this node holds it.
    pub fn get_manifest(&self, manifest_cid: &Cid) -> Option<&VideoManifest> {
        self.manifests.get(manifest_cid)
    }

    /// Load and verify a manifest by CID: from the index if held, else
    /// decoded from the content store. A bad signature fails here.
    fn load_manifest(&self, manifest_cid: &Cid) -> Result<VideoManifest, VideoError> {
        let manifest = match self.manifests.get(manifest_cid) {
            Some(m) => m.clone(),
            None => {
                let bytes = self.store.get(manifest_cid).map_err(|e| match e {
                    StorageError::NotFound(cid) => VideoError::NotFound(cid),
                    other => VideoError::Storage(other),
                })?;
                VideoManifest::decode(&bytes)?
            }
        };
        if !manifest.verify() {
            return Err(VideoError::BadSignature);
        }
        Ok(manifest)
    }

    /// Store a video: chunk the bytes, build and sign the manifest,
    /// and store the manifest as a block. Returns the manifest CID —
    /// the video's address.
    pub fn put_video(
        &mut self,
        title: &str,
        duration_secs: u64,
        width: u32,
        height: u32,
        data: &[u8],
        identity: &Identity,
    ) -> Result<Cid, VideoError> {
        validate_title(title)?;
        validate_duration(duration_secs)?;
        validate_dimensions(width, height)?;
        let root = self.store.put_chunked(data);
        let manifest = VideoManifest::new(title, duration_secs, width, height, root, identity)?;
        let manifest_cid = self.store.put(&manifest.encode());
        self.manifests.insert(manifest_cid, manifest);
        Ok(manifest_cid)
    }

    /// Fetch a video by manifest CID: verify the manifest signature,
    /// then reassemble the chunked content (every chunk hash-verified).
    /// Returns the video bytes and human-readable metadata.
    pub fn get_video(
        &self,
        manifest_cid: &Cid,
    ) -> Result<(Vec<u8>, VideoMetadata), VideoError> {
        let manifest = self.load_manifest(manifest_cid)?;
        let chunk_cids = decode_manifest(&self.store.get(&manifest.root)?)?;
        let data = self
            .store
            .get_chunked(&manifest.root)
            .map_err(VideoError::Storage)?;
        let metadata = VideoMetadata {
            title: manifest.title,
            duration_secs: manifest.duration_secs,
            width: manifest.width,
            height: manifest.height,
            owner: manifest.owner,
            chunk_count: chunk_cids.len(),
        };
        Ok((data, metadata))
    }

    /// Fetch one chunk of a video by index, hash-verified against the
    /// chunk manifest. The manifest signature is verified first, so a
    /// tampered manifest cannot serve chunks. An out-of-range index is
    /// a clean error, not a panic.
    pub fn get_chunk(
        &self,
        manifest_cid: &Cid,
        index: usize,
    ) -> Result<Vec<u8>, VideoError> {
        let manifest = self.load_manifest(manifest_cid)?;
        let chunk_cids = decode_manifest(&self.store.get(&manifest.root)?)?;
        let chunk_cid = chunk_cids
            .get(index)
            .ok_or(VideoError::InvalidChunkIndex(index))?;
        let chunk = self.store.get(chunk_cid).map_err(VideoError::Storage)?;
        let actual = Cid::from_bytes(sha256(&chunk));
        if actual != *chunk_cid {
            return Err(VideoError::Storage(StorageError::CorruptChunk {
                cid: *chunk_cid,
                actual,
            }));
        }
        Ok(chunk)
    }

    /// All videos held, as (manifest CID, title), sorted by title
    /// for a stable, human-readable listing.
    pub fn list_videos(&self) -> Vec<(Cid, String)> {
        let mut out: Vec<(Cid, String)> = self
            .manifests
            .iter()
            .map(|(cid, m)| (*cid, m.title.clone()))
            .collect();
        out.sort_by(|a, b| a.1.cmp(&b.1));
        out
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
        let root = Cid::from_bytes(kernel_crypto::sha256(b"video chunks"));
        let manifest =
            VideoManifest::new("Launch", 3600, 1920, 1080, root, &alice).unwrap();
        assert!(manifest.verify());
        let encoded = manifest.encode();
        let decoded = VideoManifest::decode(&encoded).unwrap();
        assert_eq!(decoded.title, "Launch");
        assert_eq!(decoded.duration_secs, 3600);
        assert_eq!(decoded.width, 1920);
        assert_eq!(decoded.height, 1080);
        assert_eq!(decoded.root, root);
        assert_eq!(decoded.signature, manifest.signature);
        assert_eq!(decoded.owner.public_key(), manifest.owner.public_key());
        assert!(decoded.verify());
        assert!(VideoManifest::decode(b"garbage").is_err());
        assert!(VideoManifest::decode(&encoded[..encoded.len() - 1]).is_err());
    }

    #[test]
    fn video_round_trip_small_and_large() {
        let alice = Identity::generate("alice").unwrap();
        let mut videos = VideoStore::new();
        let small = b"fake mp4 bytes";
        let cid = videos
            .put_video("Clip", 30, 1280, 720, small, &alice)
            .unwrap();
        let (data, meta) = videos.get_video(&cid).unwrap();
        assert_eq!(data, small);
        assert_eq!(meta.title, "Clip");
        assert_eq!(meta.duration_secs, 30);
        assert_eq!(meta.width, 1280);
        assert_eq!(meta.height, 720);
        assert_eq!(meta.chunk_count, 1);
        assert_eq!(meta.owner, *alice.peer_id());
        assert_eq!(videos.len(), 1);

        let large = test_data(600 * 1024);
        let big_cid = videos
            .put_video("Feature", 7200, 3840, 2160, &large, &alice)
            .unwrap();
        let (data, meta) = videos.get_video(&big_cid).unwrap();
        assert_eq!(data, large);
        assert_eq!(meta.chunk_count, 3);
        assert_eq!(videos.len(), 2);
    }

    #[test]
    fn get_chunk_returns_the_right_chunk_bytes() {
        let alice = Identity::generate("alice").unwrap();
        let mut videos = VideoStore::new();
        // 600 KiB spans 3 chunks at 256 KiB.
        let data = test_data(600 * 1024);
        let cid = videos
            .put_video("Feature", 7200, 1920, 1080, &data, &alice)
            .unwrap();
        let chunk0 = videos.get_chunk(&cid, 0).unwrap();
        let chunk1 = videos.get_chunk(&cid, 1).unwrap();
        let chunk2 = videos.get_chunk(&cid, 2).unwrap();
        assert_eq!(chunk0, &data[..256 * 1024]);
        assert_eq!(chunk1, &data[256 * 1024..512 * 1024]);
        assert_eq!(chunk2, &data[512 * 1024..]);
        assert_eq!(videos.get_chunk(&cid, 3), Err(VideoError::InvalidChunkIndex(3)));
        // The chunks reassemble to the whole video.
        let mut reassembled = Vec::new();
        reassembled.extend_from_slice(&chunk0);
        reassembled.extend_from_slice(&chunk1);
        reassembled.extend_from_slice(&chunk2);
        assert_eq!(reassembled, data);
    }

    #[test]
    fn tampered_chunk_fails() {
        let alice = Identity::generate("alice").unwrap();
        let mut videos = VideoStore::new();
        let data = test_data(600 * 1024);
        let cid = videos
            .put_video("Feature", 7200, 1920, 1080, &data, &alice)
            .unwrap();
        let manifest = videos.get_manifest(&cid).unwrap().clone();
        let manifest_bytes = videos.store_mut().get(&manifest.root).unwrap();
        let chunk_cids = decode_manifest(&manifest_bytes).unwrap();
        let mut bad = videos.store_mut().get(&chunk_cids[1]).unwrap();
        bad[0] ^= 0xff;
        videos.store_mut().put_unchecked(&chunk_cids[1], bad);
        // Whole-video fetch fails on the corrupt chunk.
        match videos.get_video(&cid) {
            Err(VideoError::Storage(StorageError::CorruptChunk { cid: got, .. })) => {
                assert_eq!(got, chunk_cids[1]);
            }
            other => panic!("expected CorruptChunk, got {other:?}"),
        }
        // Single-chunk fetch fails the same way.
        match videos.get_chunk(&cid, 1) {
            Err(VideoError::Storage(StorageError::CorruptChunk { cid: got, .. })) => {
                assert_eq!(got, chunk_cids[1]);
            }
            other => panic!("expected CorruptChunk, got {other:?}"),
        }
        // The untouched chunk still reads fine.
        assert_eq!(
            videos.get_chunk(&cid, 0).unwrap(),
            &data[..256 * 1024]
        );
    }

    #[test]
    fn tampered_manifest_signature_fails() {
        let alice = Identity::generate("alice").unwrap();
        let mut videos = VideoStore::new();
        let cid = videos
            .put_video("Secret", 60, 640, 480, b"video", &alice)
            .unwrap();
        let mut bytes = videos.store_mut().get(&cid).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let tampered_cid = videos.store_mut().put(&bytes);
        assert_ne!(tampered_cid, cid);
        assert_eq!(
            videos.get_video(&tampered_cid),
            Err(VideoError::BadSignature)
        );
        assert_eq!(
            videos.get_chunk(&tampered_cid, 0),
            Err(VideoError::BadSignature)
        );
        assert_eq!(videos.get_video(&cid).unwrap().0, b"video");
    }

    #[test]
    fn tampered_metadata_breaks_signature() {
        let alice = Identity::generate("alice").unwrap();
        let root = Cid::from_bytes(kernel_crypto::sha256(b"chunks"));
        let mut manifest =
            VideoManifest::new("A", 60, 640, 480, root, &alice).unwrap();
        assert!(manifest.verify());
        manifest.width = 1280;
        assert!(!manifest.verify());
        manifest.width = 640;
        manifest.duration_secs = 61;
        assert!(!manifest.verify());
    }

    #[test]
    fn list_videos_sorted_by_title() {
        let alice = Identity::generate("alice").unwrap();
        let mut videos = VideoStore::new();
        assert!(videos.list_videos().is_empty());
        videos.put_video("Zulu", 60, 640, 480, b"1", &alice).unwrap();
        videos.put_video("Alpha", 60, 640, 480, b"2", &alice).unwrap();
        videos.put_video("Mike", 60, 640, 480, b"3", &alice).unwrap();
        let listed = videos.list_videos();
        let titles: Vec<&str> = listed.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(titles, vec!["Alpha", "Mike", "Zulu"]);
    }

    #[test]
    fn title_duration_and_dimension_rules() {
        let alice = Identity::generate("alice").unwrap();
        let mut videos = VideoStore::new();
        assert!(videos
            .put_video("", 60, 640, 480, b"x", &alice)
            .is_err());
        assert!(videos
            .put_video("a/b", 60, 640, 480, b"x", &alice)
            .is_err());
        assert!(videos
            .put_video("Ok", 0, 640, 480, b"x", &alice)
            .is_err());
        assert!(videos
            .put_video("Ok", 60, 0, 480, b"x", &alice)
            .is_err());
        assert!(videos
            .put_video("Ok", 60, MAX_DIMENSION + 1, 480, b"x", &alice)
            .is_err());
        assert!(videos
            .put_video("Ok", 60, 640, 480, b"x", &alice)
            .is_ok());
    }

    #[test]
    fn get_unknown_video_is_not_found() {
        let videos = VideoStore::new();
        let cid = Cid::from_bytes(kernel_crypto::sha256(b"never stored"));
        assert_eq!(videos.get_video(&cid), Err(VideoError::NotFound(cid)));
        assert_eq!(
            videos.get_chunk(&cid, 0),
            Err(VideoError::NotFound(cid))
        );
    }
}
