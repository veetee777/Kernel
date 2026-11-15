//! Planet•K — social on KERNEL.
//!
//! A **post** is a signed record: the author, text, a timestamp, an
//! optional media attachment (chunk root CID plus kind and size), and
//! the author's signature. A **follow** is a signed record: follower,
//! followed, and the follower's signature. The **feed** is the set of
//! posts from peers you follow, newest first — and every post's
//! signature is verified on read, so a post whose signature does not
//! check out never reaches the feed.
//!
//! Comments, likes, reposts, and blocks are signed records too:
//! a **comment** is attached to a post, a **like** marks a post, a
//! **repost** shares a post (optionally with a comment of the
//! reposter's own) into the reposter's followers' feeds, and a
//! **block** hides an author's posts, comments, likes, and reposts
//! from the blocker — locally only. A blocked author can still be
//! seen by everyone else.
//!
//! Posts are addressed by record CID; humans see author names and
//! text, never hashes. Timestamps are caller-supplied (seconds since
//! the Unix epoch) so ordering is explicit and testable; a node that
//! wants wall-clock time supplies its own clock.
//!
//! Record formats (explicit binary, no JSON):
//!
//! ```text
//! KPOST1 + author public key (32) + text length (u32 LE) + text
//! bytes + timestamp (u64 LE) + media flag (u8: 0 = none, 1 = present)
//! + [media chunk root CID (32) + kind (u8: 1 = image, 2 = audio,
//! 3 = video) + size in bytes (u64 LE)] + signature (64).
//! KFOL1 + follower public key (32) + followed public key (32) +
//! signature (64).
//! KCOM1 + post CID (32) + author public key (32) + text length
//! (u32 LE) + text bytes + timestamp (u64 LE) + signature (64).
//! KLIKE1 + post CID (32) + liker public key (32) + signature (64).
//! KRPT1 + original CID (32) + reposter public key (32) + comment
//! flag (u8: 0 = none, 1 = present) + [comment length (u32 LE) +
//! comment bytes] + timestamp (u64 LE) + signature (64).
//! KBLK1 + blocker public key (32) + blocked public key (32) +
//! signature (64).
//! ```
//!
//! Media is capped at 64 MiB (`MAX_MEDIA_BYTES`). Media is a content
//! hash plus metadata, not a hosted file: the bytes live in chunked
//! storage under the root CID, and this crate only records the root,
//! the kind, and the size. A post whose media metadata is missing,
//! oversized, or of an unknown kind is refused.
//!
//! The Korn world is wired in through [`sim_bridge`]: sim events are
//! published as ordinary signed posts (timestamped with the sim tick,
//! never the wall clock), so the feed, blocking, and reposts all
//! apply to them exactly as to any other post.

pub mod sim_bridge;

use std::collections::{HashMap, HashSet};
use std::fmt;

use kernel_crypto::Ed25519Signature;
use kernel_identity::{Identity, IdentityError, PeerId};
use kernel_storage::{Cid, ContentStore, StorageError};

/// Magic bytes at the start of every post record.
const POST_MAGIC: &[u8; 6] = b"KPOST1";
/// Domain separator for post signatures.
const POST_SIGN_DOMAIN: &[u8] = b"KPOST1";
/// Magic bytes at the start of every follow record.
const FOLLOW_MAGIC: &[u8; 5] = b"KFOL1";
/// Domain separator for follow signatures.
const FOLLOW_SIGN_DOMAIN: &[u8] = b"KFOL1";
/// Magic bytes at the start of every comment record.
const COMMENT_MAGIC: &[u8; 5] = b"KCOM1";
/// Domain separator for comment signatures.
const COMMENT_SIGN_DOMAIN: &[u8] = b"KCOM1";
/// Magic bytes at the start of every like record.
const LIKE_MAGIC: &[u8; 6] = b"KLIKE1";
/// Domain separator for like signatures.
const LIKE_SIGN_DOMAIN: &[u8] = b"KLIKE1";
/// Magic bytes at the start of every repost record.
const REPOST_MAGIC: &[u8; 5] = b"KRPT1";
/// Domain separator for repost signatures.
const REPOST_SIGN_DOMAIN: &[u8] = b"KRPT1";
/// Magic bytes at the start of every block record.
const BLOCK_MAGIC: &[u8; 5] = b"KBLK1";
/// Domain separator for block signatures.
const BLOCK_SIGN_DOMAIN: &[u8] = b"KBLK1";

/// Maximum post text length in characters.
pub const MAX_POST_LEN: usize = 10_000;
/// Maximum media attachment size in bytes (64 MiB). A post whose
/// media is larger than this is refused at post time. The cap keeps
/// one attachment from swallowing a node's storage; chunked media is
/// still fetched by CID, so the cap is about the single attachment,
/// not the whole library.
pub const MAX_MEDIA_BYTES: u64 = 64 * 1024 * 1024;
/// Maximum hops when resolving a repost chain to its original post.
/// A chain can only grow by reposting a repost, and a record's CID
/// is the hash of its own bytes, so a real cycle cannot be built —
/// the cap is a defensive backstop, not a reachable limit.
const MAX_REPOST_HOPS: usize = 32;

/// Errors returned by social operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SocialError {
    /// Wraps a [`StorageError`].
    Storage(StorageError),
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// Post text broke the rules.
    InvalidText(String),
    /// A follow was invalid (e.g. following yourself).
    InvalidFollow(String),
    /// Media metadata broke the rules (bad kind, zero size, over the cap).
    InvalidMedia(String),
    /// A block was invalid (e.g. blocking yourself).
    InvalidBlock(String),
    /// A post or follow record was malformed.
    InvalidRecord(String),
    /// A record's signature did not verify.
    BadSignature,
    /// No post with this record CID is known here.
    NotFound(Cid),
}

impl fmt::Display for SocialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SocialError::Storage(e) => write!(f, "storage error: {e}"),
            SocialError::Identity(e) => write!(f, "identity error: {e}"),
            SocialError::InvalidText(msg) => write!(f, "invalid post text: {msg}"),
            SocialError::InvalidFollow(msg) => write!(f, "invalid follow: {msg}"),
            SocialError::InvalidMedia(msg) => write!(f, "invalid media: {msg}"),
            SocialError::InvalidBlock(msg) => write!(f, "invalid block: {msg}"),
            SocialError::InvalidRecord(msg) => write!(f, "invalid record: {msg}"),
            SocialError::BadSignature => write!(f, "record signature is invalid"),
            SocialError::NotFound(cid) => write!(f, "post not found: {cid}"),
        }
    }
}

impl std::error::Error for SocialError {}

impl From<StorageError> for SocialError {
    fn from(e: StorageError) -> Self {
        SocialError::Storage(e)
    }
}

impl From<IdentityError> for SocialError {
    fn from(e: IdentityError) -> Self {
        SocialError::Identity(e)
    }
}

/// Rules for post text: non-empty, at most [`MAX_POST_LEN`] characters,
/// no control characters except newline and tab. Shared by posts,
/// comments, and repost comments.
fn validate_text(text: &str) -> Result<(), SocialError> {
    if text.is_empty() {
        return Err(SocialError::InvalidText(
            "post text must not be empty".to_string(),
        ));
    }
    if text.chars().count() > MAX_POST_LEN {
        return Err(SocialError::InvalidText(format!(
            "post text must be at most {MAX_POST_LEN} characters"
        )));
    }
    if text
        .chars()
        .any(|c| c.is_control() && c != '\n' && c != '\t')
    {
        return Err(SocialError::InvalidText(
            "post text must not contain control characters".to_string(),
        ));
    }
    Ok(())
}

/// A display name derived from a public key: the first 8 hex chars of
/// the key. Used when a record is decoded and only the key survives
/// (verification needs the key, not the human name).
fn fingerprint_name(key: &kernel_crypto::Ed25519PublicKey) -> String {
    let hex = kernel_crypto::to_hex(key.as_bytes());
    format!("peer-{}", &hex[..8])
}

/// The record CID for encoded record bytes: a CID is the SHA-256 of
/// the bytes, the same address the content store assigns on `put`.
fn record_cid(bytes: &[u8]) -> Cid {
    Cid::from_bytes(kernel_crypto::sha256(bytes))
}

/// The kind of a media attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaKind {
    /// A still image.
    Image = 1,
    /// Audio (a track, a voice note, a song).
    Audio = 2,
    /// A video.
    Video = 3,
}

impl MediaKind {
    /// Decode a kind byte, or `None` for an unknown kind.
    pub fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            1 => Some(MediaKind::Image),
            2 => Some(MediaKind::Audio),
            3 => Some(MediaKind::Video),
            _ => None,
        }
    }

    /// The kind as a single byte for the record format.
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Media metadata: what kind of attachment a post carries, how big
/// it is, and the chunk root CID its bytes live under. The bytes
/// themselves are not held here — only the hash and the metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaMeta {
    /// The kind of media.
    pub kind: MediaKind,
    /// Size of the media in bytes. Must be greater than zero and at
    /// most [`MAX_MEDIA_BYTES`].
    pub size_bytes: u64,
    /// Chunk root CID (a `KCHUNK1` manifest block) for the media bytes.
    pub root: Cid,
}

impl MediaMeta {
    /// Build media metadata, validating the size: zero is refused
    /// (an attachment with no bytes is not an attachment) and anything
    /// over [`MAX_MEDIA_BYTES`] is refused.
    pub fn new(kind: MediaKind, size_bytes: u64, root: Cid) -> Result<Self, SocialError> {
        if size_bytes == 0 {
            return Err(SocialError::InvalidMedia(
                "media size must be greater than zero".to_string(),
            ));
        }
        if size_bytes > MAX_MEDIA_BYTES {
            return Err(SocialError::InvalidMedia(format!(
                "media size {size_bytes} exceeds the {MAX_MEDIA_BYTES} byte cap"
            )));
        }
        Ok(Self {
            kind,
            size_bytes,
            root,
        })
    }
}

/// The message an author signs for a post.
fn post_sign_message(
    author: &PeerId,
    text: &str,
    timestamp: u64,
    media: Option<&MediaMeta>,
) -> Vec<u8> {
    let mut msg = Vec::with_capacity(POST_SIGN_DOMAIN.len() + 32 + text.len() + 8 + 1 + 32 + 1 + 8);
    msg.extend_from_slice(POST_SIGN_DOMAIN);
    msg.extend_from_slice(author.public_key().as_bytes());
    msg.extend_from_slice(text.as_bytes());
    msg.extend_from_slice(&timestamp.to_le_bytes());
    match media {
        Some(meta) => {
            msg.push(1);
            msg.extend_from_slice(meta.root.as_bytes());
            msg.push(meta.kind.as_u8());
            msg.extend_from_slice(&meta.size_bytes.to_le_bytes());
        }
        None => msg.push(0),
    }
    msg
}

/// The message a follower signs for a follow record.
fn follow_sign_message(follower: &PeerId, followed: &PeerId) -> Vec<u8> {
    let mut msg = Vec::with_capacity(FOLLOW_SIGN_DOMAIN.len() + 32 + 32);
    msg.extend_from_slice(FOLLOW_SIGN_DOMAIN);
    msg.extend_from_slice(follower.public_key().as_bytes());
    msg.extend_from_slice(followed.public_key().as_bytes());
    msg
}

/// The message an author signs for a comment.
fn comment_sign_message(post_cid: &Cid, author: &PeerId, text: &str, timestamp: u64) -> Vec<u8> {
    let mut msg = Vec::with_capacity(COMMENT_SIGN_DOMAIN.len() + 32 + 32 + text.len() + 8);
    msg.extend_from_slice(COMMENT_SIGN_DOMAIN);
    msg.extend_from_slice(post_cid.as_bytes());
    msg.extend_from_slice(author.public_key().as_bytes());
    msg.extend_from_slice(text.as_bytes());
    msg.extend_from_slice(&timestamp.to_le_bytes());
    msg
}

/// The message a liker signs for a like.
fn like_sign_message(post_cid: &Cid, liker: &PeerId) -> Vec<u8> {
    let mut msg = Vec::with_capacity(LIKE_SIGN_DOMAIN.len() + 32 + 32);
    msg.extend_from_slice(LIKE_SIGN_DOMAIN);
    msg.extend_from_slice(post_cid.as_bytes());
    msg.extend_from_slice(liker.public_key().as_bytes());
    msg
}

/// The message a reposter signs for a repost.
fn repost_sign_message(
    original: &Cid,
    reposter: &PeerId,
    comment: Option<&str>,
    timestamp: u64,
) -> Vec<u8> {
    let mut msg = Vec::with_capacity(
        REPOST_SIGN_DOMAIN.len() + 32 + 32 + 1 + 4 + comment.map_or(0, str::len) + 8,
    );
    msg.extend_from_slice(REPOST_SIGN_DOMAIN);
    msg.extend_from_slice(original.as_bytes());
    msg.extend_from_slice(reposter.public_key().as_bytes());
    match comment {
        Some(text) => {
            msg.push(1);
            msg.extend_from_slice(&(text.len() as u32).to_le_bytes());
            msg.extend_from_slice(text.as_bytes());
        }
        None => msg.push(0),
    }
    msg.extend_from_slice(&timestamp.to_le_bytes());
    msg
}

/// The message a blocker signs for a block.
fn block_sign_message(blocker: &PeerId, blocked: &PeerId) -> Vec<u8> {
    let mut msg = Vec::with_capacity(BLOCK_SIGN_DOMAIN.len() + 32 + 32);
    msg.extend_from_slice(BLOCK_SIGN_DOMAIN);
    msg.extend_from_slice(blocker.public_key().as_bytes());
    msg.extend_from_slice(blocked.public_key().as_bytes());
    msg
}

/// A signed post: author, text, timestamp, optional media, signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Post {
    /// The peer who wrote this post.
    pub author: PeerId,
    /// The post text.
    pub text: String,
    /// Unix timestamp (seconds) the post was written.
    pub timestamp: u64,
    /// Optional media metadata (kind, size, chunk root CID).
    pub media: Option<MediaMeta>,
    /// The author's signature over author, text, timestamp, and media.
    pub signature: Ed25519Signature,
}

impl Post {
    /// Build and sign a post as `identity`. Media metadata is
    /// validated at post time: a post with oversized, zero-size, or
    /// otherwise invalid media is refused here, never stored.
    pub fn new(
        text: &str,
        timestamp: u64,
        media: Option<MediaMeta>,
        identity: &Identity,
    ) -> Result<Self, SocialError> {
        validate_text(text)?;
        let author = identity.peer_id().clone();
        let msg = post_sign_message(&author, text, timestamp, media.as_ref());
        let signature = identity.sign(&msg);
        Ok(Self {
            author,
            text: text.to_string(),
            timestamp,
            media,
            signature,
        })
    }

    /// True if the signature verifies against the author for this
    /// exact text, timestamp, and media. Any tampering — text, media
    /// kind, or media size — fails this check.
    pub fn verify(&self) -> bool {
        let msg = post_sign_message(
            &self.author,
            &self.text,
            self.timestamp,
            self.media.as_ref(),
        );
        self.author.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary post format.
    pub fn encode(&self) -> Vec<u8> {
        let text_bytes = self.text.as_bytes();
        let mut out = Vec::with_capacity(6 + 32 + 4 + text_bytes.len() + 8 + 1 + 32 + 1 + 8 + 64);
        out.extend_from_slice(POST_MAGIC);
        out.extend_from_slice(self.author.public_key().as_bytes());
        out.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(text_bytes);
        out.extend_from_slice(&self.timestamp.to_le_bytes());
        match &self.media {
            Some(meta) => {
                out.push(1);
                out.extend_from_slice(meta.root.as_bytes());
                out.push(meta.kind.as_u8());
                out.extend_from_slice(&meta.size_bytes.to_le_bytes());
            }
            None => out.push(0),
        }
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a post produced by [`Post::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, SocialError> {
        if bytes.len() < 6 + 32 + 4 + 8 + 1 + 64 || &bytes[..6] != POST_MAGIC {
            return Err(SocialError::InvalidRecord(
                "missing KPOST1 magic".to_string(),
            ));
        }
        let text_len = u32::from_le_bytes(bytes[38..42].try_into().expect("4-byte slice")) as usize;
        if text_len > MAX_POST_LEN {
            return Err(SocialError::InvalidRecord(format!(
                "post claims {text_len} chars, max is {MAX_POST_LEN}"
            )));
        }
        let flag_off = 42 + text_len + 8;
        if bytes.len() < flag_off + 1 + 64 {
            return Err(SocialError::InvalidRecord(
                "post too short for flag and signature".to_string(),
            ));
        }
        let flag = bytes[flag_off];
        if flag > 1 {
            return Err(SocialError::InvalidRecord(format!(
                "media flag must be 0 or 1, got {flag}"
            )));
        }
        let expected = flag_off + 1 + if flag == 1 { 32 + 1 + 8 } else { 0 } + 64;
        if bytes.len() != expected {
            return Err(SocialError::InvalidRecord(format!(
                "post is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[6..38]);
        let author_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| SocialError::InvalidRecord(e.to_string()))?;
        let text = String::from_utf8(bytes[42..42 + text_len].to_vec())
            .map_err(|_| SocialError::InvalidRecord("post text is not valid UTF-8".to_string()))?;
        validate_text(&text)?;
        let mut off = 42 + text_len;
        let timestamp = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let flag = bytes[off];
        off += 1;
        let media = if flag == 1 {
            let mut root_bytes = [0u8; 32];
            root_bytes.copy_from_slice(&bytes[off..off + 32]);
            off += 32;
            let kind = MediaKind::from_u8(bytes[off]).ok_or_else(|| {
                SocialError::InvalidRecord(format!("unknown media kind {}", bytes[off]))
            })?;
            off += 1;
            let size_bytes =
                u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
            off += 8;
            Some(MediaMeta::new(
                kind,
                size_bytes,
                Cid::from_bytes(root_bytes),
            )?)
        } else {
            None
        };
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let author = PeerId::new(author_key, &fingerprint_name(&author_key))?;
        Ok(Self {
            author,
            text,
            timestamp,
            media,
            signature,
        })
    }
}

/// A signed follow record: follower, followed, follower's signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Follow {
    /// The peer doing the following.
    pub follower: PeerId,
    /// The peer being followed.
    pub followed: PeerId,
    /// The follower's signature over follower and followed keys.
    pub signature: Ed25519Signature,
}

impl Follow {
    /// Build and sign a follow record as `follower_identity`.
    pub fn new(follower_identity: &Identity, followed: &PeerId) -> Result<Self, SocialError> {
        if follower_identity.peer_id() == followed {
            return Err(SocialError::InvalidFollow(
                "a peer cannot follow itself".to_string(),
            ));
        }
        let follower = follower_identity.peer_id().clone();
        let msg = follow_sign_message(&follower, followed);
        let signature = follower_identity.sign(&msg);
        Ok(Self {
            follower,
            followed: followed.clone(),
            signature,
        })
    }

    /// True if the signature verifies against the follower.
    pub fn verify(&self) -> bool {
        let msg = follow_sign_message(&self.follower, &self.followed);
        self.follower.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary follow format.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(5 + 32 + 32 + 64);
        out.extend_from_slice(FOLLOW_MAGIC);
        out.extend_from_slice(self.follower.public_key().as_bytes());
        out.extend_from_slice(self.followed.public_key().as_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a follow produced by [`Follow::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, SocialError> {
        if bytes.len() != 5 + 32 + 32 + 64 || &bytes[..5] != FOLLOW_MAGIC {
            return Err(SocialError::InvalidRecord(
                "missing KFOL1 magic or wrong length".to_string(),
            ));
        }
        let mut follower_bytes = [0u8; 32];
        follower_bytes.copy_from_slice(&bytes[5..37]);
        let follower_key = kernel_crypto::Ed25519PublicKey::from_bytes(&follower_bytes)
            .map_err(|e| SocialError::InvalidRecord(e.to_string()))?;
        let mut followed_bytes = [0u8; 32];
        followed_bytes.copy_from_slice(&bytes[37..69]);
        let followed_key = kernel_crypto::Ed25519PublicKey::from_bytes(&followed_bytes)
            .map_err(|e| SocialError::InvalidRecord(e.to_string()))?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[69..133]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let follower = PeerId::new(follower_key, &fingerprint_name(&follower_key))?;
        let followed = PeerId::new(followed_key, &fingerprint_name(&followed_key))?;
        Ok(Self {
            follower,
            followed,
            signature,
        })
    }
}

/// A signed comment: attached to one post, by one author.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Comment {
    /// The record CID of the post this comment is attached to.
    pub post: Cid,
    /// The peer who wrote the comment.
    pub author: PeerId,
    /// The comment text.
    pub text: String,
    /// Unix timestamp (seconds) the comment was written.
    pub timestamp: u64,
    /// The author's signature over post, author, text, and timestamp.
    pub signature: Ed25519Signature,
}

impl Comment {
    /// Build and sign a comment as `identity`.
    pub fn new(
        post_cid: &Cid,
        text: &str,
        timestamp: u64,
        identity: &Identity,
    ) -> Result<Self, SocialError> {
        validate_text(text)?;
        let author = identity.peer_id().clone();
        let msg = comment_sign_message(post_cid, &author, text, timestamp);
        let signature = identity.sign(&msg);
        Ok(Self {
            post: *post_cid,
            author,
            text: text.to_string(),
            timestamp,
            signature,
        })
    }

    /// True if the signature verifies against the author for this
    /// exact post, text, and timestamp.
    pub fn verify(&self) -> bool {
        let msg = comment_sign_message(&self.post, &self.author, &self.text, self.timestamp);
        self.author.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary comment format.
    pub fn encode(&self) -> Vec<u8> {
        let text_bytes = self.text.as_bytes();
        let mut out = Vec::with_capacity(5 + 32 + 32 + 4 + text_bytes.len() + 8 + 64);
        out.extend_from_slice(COMMENT_MAGIC);
        out.extend_from_slice(self.post.as_bytes());
        out.extend_from_slice(self.author.public_key().as_bytes());
        out.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(text_bytes);
        out.extend_from_slice(&self.timestamp.to_le_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a comment produced by [`Comment::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, SocialError> {
        if bytes.len() < 5 + 32 + 32 + 4 + 8 + 64 || &bytes[..5] != COMMENT_MAGIC {
            return Err(SocialError::InvalidRecord(
                "missing KCOM1 magic".to_string(),
            ));
        }
        let mut post_bytes = [0u8; 32];
        post_bytes.copy_from_slice(&bytes[5..37]);
        let post = Cid::from_bytes(post_bytes);
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[37..69]);
        let author_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| SocialError::InvalidRecord(e.to_string()))?;
        let text_len = u32::from_le_bytes(bytes[69..73].try_into().expect("4-byte slice")) as usize;
        let expected = 5 + 32 + 32 + 4 + text_len + 8 + 64;
        if bytes.len() != expected {
            return Err(SocialError::InvalidRecord(format!(
                "comment is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let text = String::from_utf8(bytes[73..73 + text_len].to_vec()).map_err(|_| {
            SocialError::InvalidRecord("comment text is not valid UTF-8".to_string())
        })?;
        validate_text(&text)?;
        let mut off = 73 + text_len;
        let timestamp = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let author = PeerId::new(author_key, &fingerprint_name(&author_key))?;
        Ok(Self {
            post,
            author,
            text,
            timestamp,
            signature,
        })
    }
}

/// A signed like: one peer marking one post.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Like {
    /// The record CID of the post being liked.
    pub post: Cid,
    /// The peer who liked it.
    pub liker: PeerId,
    /// The liker's signature over post and liker.
    pub signature: Ed25519Signature,
}

impl Like {
    /// Build and sign a like as `identity`.
    pub fn new(post_cid: &Cid, identity: &Identity) -> Result<Self, SocialError> {
        let liker = identity.peer_id().clone();
        let msg = like_sign_message(post_cid, &liker);
        let signature = identity.sign(&msg);
        Ok(Self {
            post: *post_cid,
            liker,
            signature,
        })
    }

    /// True if the signature verifies against the liker for this post.
    pub fn verify(&self) -> bool {
        let msg = like_sign_message(&self.post, &self.liker);
        self.liker.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary like format.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(6 + 32 + 32 + 64);
        out.extend_from_slice(LIKE_MAGIC);
        out.extend_from_slice(self.post.as_bytes());
        out.extend_from_slice(self.liker.public_key().as_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a like produced by [`Like::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, SocialError> {
        if bytes.len() != 6 + 32 + 32 + 64 || &bytes[..6] != LIKE_MAGIC {
            return Err(SocialError::InvalidRecord(
                "missing KLIKE1 magic or wrong length".to_string(),
            ));
        }
        let mut post_bytes = [0u8; 32];
        post_bytes.copy_from_slice(&bytes[6..38]);
        let post = Cid::from_bytes(post_bytes);
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[38..70]);
        let liker_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| SocialError::InvalidRecord(e.to_string()))?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[70..134]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let liker = PeerId::new(liker_key, &fingerprint_name(&liker_key))?;
        Ok(Self {
            post,
            liker,
            signature,
        })
    }
}

/// A signed repost: one peer sharing a post, optionally with a
/// comment of their own. `original` is the record CID of the thing
/// being reposted — a post CID, or a repost record CID, which makes
/// repost chains possible. The chain always resolves to one original
/// post (see [`SocialStore::resolve_original`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Repost {
    /// The record CID being reposted (a post CID or a repost CID).
    pub original: Cid,
    /// The peer doing the reposting.
    pub reposter: PeerId,
    /// The reposter's own comment, if any.
    pub comment: Option<String>,
    /// Unix timestamp (seconds) the repost was made.
    pub timestamp: u64,
    /// The reposter's signature over original, reposter, comment,
    /// and timestamp.
    pub signature: Ed25519Signature,
}

impl Repost {
    /// Build and sign a repost as `identity`.
    pub fn new(
        original_cid: &Cid,
        comment: Option<&str>,
        timestamp: u64,
        identity: &Identity,
    ) -> Result<Self, SocialError> {
        if let Some(text) = comment {
            validate_text(text)?;
        }
        let reposter = identity.peer_id().clone();
        let msg = repost_sign_message(original_cid, &reposter, comment, timestamp);
        let signature = identity.sign(&msg);
        Ok(Self {
            original: *original_cid,
            reposter,
            comment: comment.map(str::to_string),
            timestamp,
            signature,
        })
    }

    /// True if the signature verifies against the reposter for this
    /// exact original, comment, and timestamp.
    pub fn verify(&self) -> bool {
        let msg = repost_sign_message(
            &self.original,
            &self.reposter,
            self.comment.as_deref(),
            self.timestamp,
        );
        self.reposter.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary repost format.
    pub fn encode(&self) -> Vec<u8> {
        let comment_bytes = self.comment.as_ref().map(|c| c.as_bytes());
        let mut out =
            Vec::with_capacity(5 + 32 + 32 + 1 + 4 + comment_bytes.map_or(0, |b| b.len()) + 8 + 64);
        out.extend_from_slice(REPOST_MAGIC);
        out.extend_from_slice(self.original.as_bytes());
        out.extend_from_slice(self.reposter.public_key().as_bytes());
        match comment_bytes {
            Some(bytes) => {
                out.push(1);
                out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                out.extend_from_slice(bytes);
            }
            None => out.push(0),
        }
        out.extend_from_slice(&self.timestamp.to_le_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a repost produced by [`Repost::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, SocialError> {
        if bytes.len() < 5 + 32 + 32 + 1 + 8 + 64 || &bytes[..5] != REPOST_MAGIC {
            return Err(SocialError::InvalidRecord(
                "missing KRPT1 magic".to_string(),
            ));
        }
        let mut original_bytes = [0u8; 32];
        original_bytes.copy_from_slice(&bytes[5..37]);
        let original = Cid::from_bytes(original_bytes);
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[37..69]);
        let reposter_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| SocialError::InvalidRecord(e.to_string()))?;
        let flag = bytes[69];
        if flag > 1 {
            return Err(SocialError::InvalidRecord(format!(
                "repost comment flag must be 0 or 1, got {flag}"
            )));
        }
        let mut off = 70;
        let comment = if flag == 1 {
            if bytes.len() < off + 4 {
                return Err(SocialError::InvalidRecord(
                    "repost too short for comment length".to_string(),
                ));
            }
            let comment_len =
                u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4-byte slice")) as usize;
            off += 4;
            let expected = off + comment_len + 8 + 64;
            if bytes.len() != expected {
                return Err(SocialError::InvalidRecord(format!(
                    "repost is {} bytes, expected {expected}",
                    bytes.len()
                )));
            }
            let text = String::from_utf8(bytes[off..off + comment_len].to_vec()).map_err(|_| {
                SocialError::InvalidRecord("repost comment is not valid UTF-8".to_string())
            })?;
            validate_text(&text)?;
            off += comment_len;
            Some(text)
        } else {
            let expected = off + 8 + 64;
            if bytes.len() != expected {
                return Err(SocialError::InvalidRecord(format!(
                    "repost is {} bytes, expected {expected}",
                    bytes.len()
                )));
            }
            None
        };
        let timestamp = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let reposter = PeerId::new(reposter_key, &fingerprint_name(&reposter_key))?;
        Ok(Self {
            original,
            reposter,
            comment,
            timestamp,
            signature,
        })
    }
}

/// A signed block record: one peer hiding another, locally.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// The peer doing the blocking.
    pub blocker: PeerId,
    /// The peer being blocked.
    pub blocked: PeerId,
    /// The blocker's signature over blocker and blocked keys.
    pub signature: Ed25519Signature,
}

impl Block {
    /// Build and sign a block record as `blocker_identity`.
    pub fn new(blocker_identity: &Identity, blocked: &PeerId) -> Result<Self, SocialError> {
        if blocker_identity.peer_id() == blocked {
            return Err(SocialError::InvalidBlock(
                "a peer cannot block itself".to_string(),
            ));
        }
        let blocker = blocker_identity.peer_id().clone();
        let msg = block_sign_message(&blocker, blocked);
        let signature = blocker_identity.sign(&msg);
        Ok(Self {
            blocker,
            blocked: blocked.clone(),
            signature,
        })
    }

    /// True if the signature verifies against the blocker.
    pub fn verify(&self) -> bool {
        let msg = block_sign_message(&self.blocker, &self.blocked);
        self.blocker.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary block format.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(5 + 32 + 32 + 64);
        out.extend_from_slice(BLOCK_MAGIC);
        out.extend_from_slice(self.blocker.public_key().as_bytes());
        out.extend_from_slice(self.blocked.public_key().as_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a block produced by [`Block::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, SocialError> {
        if bytes.len() != 5 + 32 + 32 + 64 || &bytes[..5] != BLOCK_MAGIC {
            return Err(SocialError::InvalidRecord(
                "missing KBLK1 magic or wrong length".to_string(),
            ));
        }
        let mut blocker_bytes = [0u8; 32];
        blocker_bytes.copy_from_slice(&bytes[5..37]);
        let blocker_key = kernel_crypto::Ed25519PublicKey::from_bytes(&blocker_bytes)
            .map_err(|e| SocialError::InvalidRecord(e.to_string()))?;
        let mut blocked_bytes = [0u8; 32];
        blocked_bytes.copy_from_slice(&bytes[37..69]);
        let blocked_key = kernel_crypto::Ed25519PublicKey::from_bytes(&blocked_bytes)
            .map_err(|e| SocialError::InvalidRecord(e.to_string()))?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[69..133]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let blocker = PeerId::new(blocker_key, &fingerprint_name(&blocker_key))?;
        let blocked = PeerId::new(blocked_key, &fingerprint_name(&blocked_key))?;
        Ok(Self {
            blocker,
            blocked,
            signature,
        })
    }
}

/// One entry in a feed: either a post, or a repost of a post. A
/// repost is never presented as the reposter's own post — the
/// original post travels with it, so a reader always sees whose
/// words they are and who shared them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FeedItem {
    /// A post by someone the viewer follows.
    Post(Post),
    /// A repost by someone the viewer follows, with the original
    /// post it resolves to.
    Repost {
        /// The repost record.
        repost: Repost,
        /// The original post at the end of the repost chain.
        original: Post,
    },
}

impl FeedItem {
    /// True if this entry is a repost rather than a post.
    pub fn is_repost(&self) -> bool {
        matches!(self, FeedItem::Repost { .. })
    }

    /// The timestamp that orders this entry in the feed: the post
    /// timestamp for a post, the repost timestamp for a repost.
    pub fn timestamp(&self) -> u64 {
        match self {
            FeedItem::Post(post) => post.timestamp,
            FeedItem::Repost { repost, .. } => repost.timestamp,
        }
    }

    /// The peer this entry comes from: the author for a post, the
    /// reposter for a repost.
    pub fn author(&self) -> &PeerId {
        match self {
            FeedItem::Post(post) => &post.author,
            FeedItem::Repost { repost, .. } => &repost.reposter,
        }
    }

    /// The original post behind this entry: the post itself, or the
    /// post a repost resolves to.
    pub fn original_post(&self) -> &Post {
        match self {
            FeedItem::Post(post) => post,
            FeedItem::Repost { original, .. } => original,
        }
    }

    /// The text of the original post behind this entry.
    pub fn text(&self) -> &str {
        &self.original_post().text
    }

    /// The repost record, if this entry is a repost.
    pub fn repost(&self) -> Option<&Repost> {
        match self {
            FeedItem::Repost { repost, .. } => Some(repost),
            FeedItem::Post(_) => None,
        }
    }
}

/// The social store: a content store holding signed posts, follows,
/// comments, likes, reposts, and blocks, plus the follow graph, the
/// like counts, and the block lists. Posts are addressed by record
/// CID; humans see author names and text.
pub struct SocialStore {
    store: ContentStore,
    /// Posts this node holds, by record CID.
    posts: HashMap<Cid, Post>,
    /// Follow graph: follower -> set of followed peers.
    follows: HashMap<PeerId, HashSet<PeerId>>,
    /// Follow record CIDs, by (follower, followed).
    follow_records: HashMap<(PeerId, PeerId), Cid>,
    /// Comments this node holds, by the post CID they attach to.
    comments: HashMap<Cid, Vec<Comment>>,
    /// Comment records, by comment record CID.
    comment_records: HashMap<Cid, Comment>,
    /// Likes this node holds: post CID -> set of likers.
    likes: HashMap<Cid, HashSet<PeerId>>,
    /// Like record CIDs, by (post CID, liker).
    like_records: HashMap<(Cid, PeerId), Cid>,
    /// Reposts this node holds, by repost record CID.
    reposts: HashMap<Cid, Repost>,
    /// Block lists: blocker -> set of blocked peers.
    blocks: HashMap<PeerId, HashSet<PeerId>>,
    /// Block record CIDs, by (blocker, blocked).
    block_records: HashMap<(PeerId, PeerId), Cid>,
}

impl Default for SocialStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SocialStore {
    /// An empty social store.
    pub fn new() -> Self {
        Self {
            store: ContentStore::new(),
            posts: HashMap::new(),
            follows: HashMap::new(),
            follow_records: HashMap::new(),
            comments: HashMap::new(),
            comment_records: HashMap::new(),
            likes: HashMap::new(),
            like_records: HashMap::new(),
            reposts: HashMap::new(),
            blocks: HashMap::new(),
            block_records: HashMap::new(),
        }
    }

    /// Number of posts held.
    pub fn len(&self) -> usize {
        self.posts.len()
    }

    /// True if no posts are held.
    pub fn is_empty(&self) -> bool {
        self.posts.is_empty()
    }

    /// The post record for a CID, if this node holds it.
    pub fn get_post_record(&self, post_cid: &Cid) -> Option<&Post> {
        self.posts.get(post_cid)
    }

    /// Create a post: build and sign the record and store it as a
    /// block. Returns the post record CID. Media metadata is
    /// validated at post time; invalid media is refused here.
    pub fn create_post(
        &mut self,
        text: &str,
        timestamp: u64,
        media: Option<MediaMeta>,
        identity: &Identity,
    ) -> Result<Cid, SocialError> {
        validate_text(text)?;
        let post = Post::new(text, timestamp, media, identity)?;
        let cid = self.store.put(&post.encode());
        self.posts.insert(cid, post);
        Ok(cid)
    }

    /// Fetch a post by record CID, verifying its signature.
    pub fn get_post(&self, post_cid: &Cid) -> Result<Post, SocialError> {
        let post = match self.posts.get(post_cid) {
            Some(p) => p.clone(),
            None => {
                let bytes = self.store.get(post_cid).map_err(|e| match e {
                    StorageError::NotFound(cid) => SocialError::NotFound(cid),
                    other => SocialError::Storage(other),
                })?;
                Post::decode(&bytes)?
            }
        };
        if !post.verify() {
            return Err(SocialError::BadSignature);
        }
        Ok(post)
    }

    /// Follow a peer: build and sign the follow record, store it, and
    /// add the edge to the follow graph. Returns the follow record CID.
    /// Following yourself is refused. Following someone you already
    /// follow returns the existing record CID.
    pub fn follow(
        &mut self,
        follower_identity: &Identity,
        followed: &PeerId,
    ) -> Result<Cid, SocialError> {
        let follower = follower_identity.peer_id().clone();
        if follower == *followed {
            return Err(SocialError::InvalidFollow(
                "a peer cannot follow itself".to_string(),
            ));
        }
        if let Some(cid) = self
            .follow_records
            .get(&(follower.clone(), followed.clone()))
        {
            return Ok(*cid);
        }
        let follow = Follow::new(follower_identity, followed)?;
        let cid = self.store.put(&follow.encode());
        self.follows
            .entry(follower.clone())
            .or_default()
            .insert(followed.clone());
        self.follow_records
            .insert((follower, followed.clone()), cid);
        Ok(cid)
    }

    /// Unfollow a peer: remove the edge from the follow graph and drop
    /// the follow record. Returns `true` if the peer was followed,
    /// `false` if there was nothing to remove. Only the follower can
    /// unfollow for themselves.
    pub fn unfollow(
        &mut self,
        follower_identity: &Identity,
        followed: &PeerId,
    ) -> Result<bool, SocialError> {
        let follower = follower_identity.peer_id().clone();
        let removed = match self.follows.get_mut(&follower) {
            Some(set) => set.remove(followed),
            None => false,
        };
        if removed {
            if let Some(cid) = self.follow_records.remove(&(follower, followed.clone())) {
                self.store.remove(&cid);
            }
        }
        Ok(removed)
    }

    /// Add a comment to a post: build and sign the comment record and
    /// store it. Returns the comment record CID. A comment on a post
    /// this node does not hold is refused, as is an empty comment.
    pub fn add_comment(
        &mut self,
        text: &str,
        timestamp: u64,
        post_cid: &Cid,
        identity: &Identity,
    ) -> Result<Cid, SocialError> {
        let comment = Comment::new(post_cid, text, timestamp, identity)?;
        self.ingest_comment(comment)
    }

    /// Ingest a comment record built elsewhere: its signature is
    /// verified, the post it attaches to must be held here, and the
    /// same record ingested twice returns the same CID without
    /// duplicating the comment. A bad signature or an unknown post is
    /// refused.
    pub fn ingest_comment(&mut self, comment: Comment) -> Result<Cid, SocialError> {
        if !comment.verify() {
            return Err(SocialError::BadSignature);
        }
        if !self.posts.contains_key(&comment.post) {
            return Err(SocialError::NotFound(comment.post));
        }
        let cid = self.store.put(&comment.encode());
        if self.comment_records.contains_key(&cid) {
            return Ok(cid);
        }
        self.comment_records.insert(cid, comment.clone());
        self.comments.entry(comment.post).or_default().push(comment);
        Ok(cid)
    }

    /// The comments under a post, as seen by `viewer`: oldest first
    /// (timestamp ascending, ties broken by record CID hex so the
    /// order is deterministic). Comments with a bad signature never
    /// show up. If `viewer` blocked the post's author, nothing shows
    /// up — the post is hidden from them entirely — and comments by
    /// authors `viewer` blocked are left out too. An unknown post
    /// has no comments.
    pub fn comments_for(&self, viewer: &PeerId, post_cid: &Cid) -> Vec<Comment> {
        let Some(post) = self.posts.get(post_cid) else {
            return Vec::new();
        };
        if self.is_blocked(viewer, &post.author) {
            return Vec::new();
        }
        let Some(list) = self.comments.get(post_cid) else {
            return Vec::new();
        };
        let mut out: Vec<(Cid, Comment)> = list
            .iter()
            .filter(|c| c.verify() && !self.is_blocked(viewer, &c.author))
            .map(|c| (record_cid(&c.encode()), c.clone()))
            .collect();
        out.sort_by(|a, b| {
            a.1.timestamp
                .cmp(&b.1.timestamp)
                .then_with(|| a.0.to_hex().cmp(&b.0.to_hex()))
        });
        out.into_iter().map(|(_, c)| c).collect()
    }

    /// Like a post: build and sign the like record and store it.
    /// Returns the like record CID. Liking is idempotent — liking a
    /// post you already liked returns the existing record CID and
    /// counts once. Liking your own post is allowed. A like on a
    /// post this node does not hold is refused.
    pub fn like(&mut self, identity: &Identity, post_cid: &Cid) -> Result<Cid, SocialError> {
        let like = Like::new(post_cid, identity)?;
        self.ingest_like(like)
    }

    /// Ingest a like record built elsewhere: its signature is
    /// verified, the post must be held here, and a duplicate like
    /// returns the existing record CID. A bad signature or an
    /// unknown post is refused.
    pub fn ingest_like(&mut self, like: Like) -> Result<Cid, SocialError> {
        if !like.verify() {
            return Err(SocialError::BadSignature);
        }
        if !self.posts.contains_key(&like.post) {
            return Err(SocialError::NotFound(like.post));
        }
        if let Some(cid) = self.like_records.get(&(like.post, like.liker.clone())) {
            return Ok(*cid);
        }
        let cid = self.store.put(&like.encode());
        self.likes
            .entry(like.post)
            .or_default()
            .insert(like.liker.clone());
        self.like_records.insert((like.post, like.liker), cid);
        Ok(cid)
    }

    /// Unlike a post: remove the like and drop the like record.
    /// Returns `true` if a like was removed, `false` if there was
    /// nothing to remove. A like on a post this node does not hold
    /// is refused.
    pub fn unlike(&mut self, identity: &Identity, post_cid: &Cid) -> Result<bool, SocialError> {
        if !self.posts.contains_key(post_cid) {
            return Err(SocialError::NotFound(*post_cid));
        }
        let liker = identity.peer_id().clone();
        let removed = match self.likes.get_mut(post_cid) {
            Some(set) => set.remove(&liker),
            None => false,
        };
        if removed {
            if let Some(cid) = self.like_records.remove(&(*post_cid, liker)) {
                self.store.remove(&cid);
            }
        }
        Ok(removed)
    }

    /// The total number of likes on a post. Unknown posts have none.
    pub fn like_count(&self, post_cid: &Cid) -> usize {
        self.likes.get(post_cid).map_or(0, HashSet::len)
    }

    /// The number of likes on a post as seen by `viewer`: likes from
    /// authors `viewer` blocked are not counted. Unknown posts have
    /// none.
    pub fn like_count_for(&self, viewer: &PeerId, post_cid: &Cid) -> usize {
        match self.likes.get(post_cid) {
            Some(set) => set
                .iter()
                .filter(|liker| !self.is_blocked(viewer, liker))
                .count(),
            None => 0,
        }
    }

    /// Repost a post: build and sign the repost record and store it.
    /// Returns the repost record CID. `target_cid` is the record CID
    /// of the thing being reposted — a post CID, or a repost record
    /// CID, which makes a repost chain that resolves to the original
    /// post. A repost of a record this node does not hold is refused.
    pub fn repost(
        &mut self,
        identity: &Identity,
        target_cid: &Cid,
        comment: Option<&str>,
        timestamp: u64,
    ) -> Result<Cid, SocialError> {
        let repost = Repost::new(target_cid, comment, timestamp, identity)?;
        self.ingest_repost(repost)
    }

    /// Ingest a repost record built elsewhere: its signature is
    /// verified, and the thing it reposts must resolve to a post this
    /// node holds. The same record ingested twice returns the same
    /// CID. A bad signature or an unresolvable target is refused.
    pub fn ingest_repost(&mut self, repost: Repost) -> Result<Cid, SocialError> {
        if !repost.verify() {
            return Err(SocialError::BadSignature);
        }
        if self.resolve_original(&repost.original).is_none() {
            return Err(SocialError::NotFound(repost.original));
        }
        let cid = self.store.put(&repost.encode());
        if self.reposts.contains_key(&cid) {
            return Ok(cid);
        }
        self.reposts.insert(cid, repost);
        Ok(cid)
    }

    /// The repost record for a CID, if this node holds it.
    pub fn get_repost(&self, repost_cid: &Cid) -> Option<&Repost> {
        self.reposts.get(repost_cid)
    }

    /// Resolve a record CID to the original post it points at: a post
    /// CID resolves to itself; a repost record CID follows the chain
    /// until it reaches a post. Returns `None` if the CID is unknown
    /// or the chain does not end at a post held here (a repost whose
    /// original was never received, or a chain deeper than the cap).
    pub fn resolve_original(&self, cid: &Cid) -> Option<Cid> {
        let mut current = *cid;
        let mut hops = 0;
        loop {
            if self.posts.contains_key(&current) {
                return Some(current);
            }
            let repost = self.reposts.get(&current)?;
            hops += 1;
            if hops > MAX_REPOST_HOPS {
                return None;
            }
            current = repost.original;
        }
    }

    /// Block a peer: build and sign the block record, store it, and
    /// add the peer to the block list. Returns the block record CID.
    /// Blocking is idempotent — blocking someone you already blocked
    /// returns the existing record CID. Blocking yourself is refused.
    /// Blocking is local: it hides the author's posts, comments,
    /// likes, and reposts from you, and from no one else.
    pub fn block(&mut self, identity: &Identity, blocked: &PeerId) -> Result<Cid, SocialError> {
        let block = Block::new(identity, blocked)?;
        self.ingest_block(block)
    }

    /// Ingest a block record built elsewhere: its signature is
    /// verified, and a duplicate block returns the existing record
    /// CID. A bad signature or a self-block is refused.
    pub fn ingest_block(&mut self, block: Block) -> Result<Cid, SocialError> {
        if !block.verify() {
            return Err(SocialError::BadSignature);
        }
        if block.blocker == block.blocked {
            return Err(SocialError::InvalidBlock(
                "a peer cannot block itself".to_string(),
            ));
        }
        if let Some(cid) = self
            .block_records
            .get(&(block.blocker.clone(), block.blocked.clone()))
        {
            return Ok(*cid);
        }
        let cid = self.store.put(&block.encode());
        self.blocks
            .entry(block.blocker.clone())
            .or_default()
            .insert(block.blocked.clone());
        self.block_records
            .insert((block.blocker, block.blocked), cid);
        Ok(cid)
    }

    /// Unblock a peer: remove them from the block list and drop the
    /// block record. Returns `true` if the peer was blocked, `false`
    /// if there was nothing to remove. Visibility is restored at once.
    pub fn unblock(&mut self, identity: &Identity, blocked: &PeerId) -> Result<bool, SocialError> {
        let blocker = identity.peer_id().clone();
        let removed = match self.blocks.get_mut(&blocker) {
            Some(set) => set.remove(blocked),
            None => false,
        };
        if removed {
            if let Some(cid) = self.block_records.remove(&(blocker, blocked.clone())) {
                self.store.remove(&cid);
            }
        }
        Ok(removed)
    }

    /// True if `viewer` blocked `author`. Blocking is local to the
    /// viewer: it says nothing about what anyone else sees.
    pub fn is_blocked(&self, viewer: &PeerId, author: &PeerId) -> bool {
        self.blocks
            .get(viewer)
            .is_some_and(|set| set.contains(author))
    }

    /// The peers a viewer blocked, sorted by display name.
    pub fn list_blocks(&self, viewer: &PeerId) -> Vec<PeerId> {
        let mut out: Vec<PeerId> = self
            .blocks
            .get(viewer)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default();
        out.sort_by(|a, b| a.display_name().cmp(b.display_name()));
        out
    }

    /// The feed for a peer: posts from followed peers plus reposts by
    /// followed peers, newest first (timestamp descending, ties broken
    /// by record CID hex so the order is deterministic). Every post's
    /// and repost's signature is verified on read; a record that fails
    /// verification is left out. A repost appears marked as a repost,
    /// with its original post — never as the reposter's own post.
    /// Anything from an author the viewer blocked is left out: their
    /// posts, their reposts, and reposts of their posts by anyone.
    pub fn feed(&self, viewer: &PeerId) -> Vec<FeedItem> {
        let followed = match self.follows.get(viewer) {
            Some(set) => set,
            None => return Vec::new(),
        };
        let mut items: Vec<(u64, String, FeedItem)> = Vec::new();
        for (cid, post) in &self.posts {
            if !followed.contains(&post.author) {
                continue;
            }
            if !post.verify() || self.is_blocked(viewer, &post.author) {
                continue;
            }
            items.push((post.timestamp, cid.to_hex(), FeedItem::Post(post.clone())));
        }
        for (cid, repost) in &self.reposts {
            if !followed.contains(&repost.reposter) {
                continue;
            }
            if !repost.verify() || self.is_blocked(viewer, &repost.reposter) {
                continue;
            }
            let Some(original_cid) = self.resolve_original(&repost.original) else {
                continue;
            };
            let Some(original) = self.posts.get(&original_cid) else {
                continue;
            };
            if !original.verify() || self.is_blocked(viewer, &original.author) {
                continue;
            }
            items.push((
                repost.timestamp,
                cid.to_hex(),
                FeedItem::Repost {
                    repost: repost.clone(),
                    original: original.clone(),
                },
            ));
        }
        items.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        items.into_iter().map(|(_, _, item)| item).collect()
    }

    /// The peers a follower follows, sorted by display name.
    pub fn list_follows(&self, follower: &PeerId) -> Vec<PeerId> {
        let mut out: Vec<PeerId> = self
            .follows
            .get(follower)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default();
        out.sort_by(|a, b| a.display_name().cmp(b.display_name()));
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

    fn image_meta(root: Cid) -> MediaMeta {
        MediaMeta::new(MediaKind::Image, 1_024, root).unwrap()
    }

    #[test]
    fn post_encode_decode_round_trip_and_verify() {
        let alice = Identity::generate("alice").unwrap();
        let root = Cid::from_bytes(kernel_crypto::sha256(b"photo chunks"));
        let meta = image_meta(root);
        let post = Post::new("hello world", 1_700_000_000, Some(meta.clone()), &alice).unwrap();
        assert!(post.verify());
        let encoded = post.encode();
        let decoded = Post::decode(&encoded).unwrap();
        assert_eq!(decoded.text, "hello world");
        assert_eq!(decoded.timestamp, 1_700_000_000);
        assert_eq!(decoded.media, Some(meta));
        assert_eq!(decoded.media.as_ref().unwrap().root, root);
        assert_eq!(decoded.media.as_ref().unwrap().kind, MediaKind::Image);
        assert_eq!(decoded.media.as_ref().unwrap().size_bytes, 1_024);
        assert_eq!(decoded.signature, post.signature);
        assert_eq!(decoded.author.public_key(), post.author.public_key());
        assert!(decoded.verify());
        // Without media.
        let plain = Post::new("plain", 5, None, &alice).unwrap();
        let decoded = Post::decode(&plain.encode()).unwrap();
        assert_eq!(decoded.media, None);
        assert!(decoded.verify());
        assert!(Post::decode(b"garbage").is_err());
        assert!(Post::decode(&encoded[..encoded.len() - 1]).is_err());
    }

    #[test]
    fn post_round_trip_through_store() {
        let alice = Identity::generate("alice").unwrap();
        let mut social = SocialStore::new();
        let cid = social
            .create_post("first post", 1_700_000_000, None, &alice)
            .unwrap();
        let post = social.get_post(&cid).unwrap();
        assert_eq!(post.text, "first post");
        assert_eq!(post.timestamp, 1_700_000_000);
        assert_eq!(post.author, *alice.peer_id());
        assert_eq!(social.len(), 1);
        assert!(social.create_post("", 1, None, &alice).is_err());
    }

    #[test]
    fn tampered_post_rejected() {
        let alice = Identity::generate("alice").unwrap();
        let mut social = SocialStore::new();
        let cid = social
            .create_post("important", 1_700_000_000, None, &alice)
            .unwrap();
        // Flip a signature byte and store the tampered record.
        let mut bytes = social.store_mut().get(&cid).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let tampered_cid = social.store_mut().put(&bytes);
        assert_ne!(tampered_cid, cid);
        assert_eq!(
            social.get_post(&tampered_cid),
            Err(SocialError::BadSignature)
        );
        // The original post still reads fine.
        assert_eq!(social.get_post(&cid).unwrap().text, "important");
        // Tampering with the text after signing breaks verification.
        let mut post = Post::new("hello", 1, None, &alice).unwrap();
        assert!(post.verify());
        post.text = "forged".to_string();
        assert!(!post.verify());
    }

    #[test]
    fn follow_then_feed_shows_followed_posts() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let carol = Identity::generate("carol").unwrap();
        let mut social = SocialStore::new();
        let bob_post = social.create_post("bob says hi", 100, None, &bob).unwrap();
        let carol_post = social
            .create_post("carol says hi", 101, None, &carol)
            .unwrap();
        // Alice follows nobody yet: empty feed.
        assert!(social.feed(alice.peer_id()).is_empty());
        // Alice follows Bob.
        social.follow(&alice, bob.peer_id()).unwrap();
        let feed = social.feed(alice.peer_id());
        assert_eq!(feed.len(), 1);
        assert_eq!(feed[0].text(), "bob says hi");
        assert_eq!(*feed[0].author(), *bob.peer_id());
        assert!(!feed[0].is_repost());
        // Carol's post is not in Alice's feed.
        assert!(!feed.iter().any(|item| item.text() == "carol says hi"));
        // Bob's own feed does not include his own post (he follows nobody).
        assert!(social.feed(bob.peer_id()).is_empty());
        // Follow is idempotent: same record CID.
        let again = social.follow(&alice, bob.peer_id()).unwrap();
        let first = social.follow(&alice, bob.peer_id()).unwrap();
        assert_eq!(again, first);
        let _ = (bob_post, carol_post);
    }

    #[test]
    fn unfollow_removes_from_feed() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut social = SocialStore::new();
        social.create_post("bob post", 100, None, &bob).unwrap();
        social.follow(&alice, bob.peer_id()).unwrap();
        assert_eq!(social.feed(alice.peer_id()).len(), 1);
        assert_eq!(social.list_follows(alice.peer_id()).len(), 1);
        assert!(social.unfollow(&alice, bob.peer_id()).unwrap());
        assert!(social.feed(alice.peer_id()).is_empty());
        assert!(social.list_follows(alice.peer_id()).is_empty());
        // Unfollowing again is a clean no-op.
        assert!(!social.unfollow(&alice, bob.peer_id()).unwrap());
    }

    #[test]
    fn feed_is_newest_first() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut social = SocialStore::new();
        social.create_post("oldest", 100, None, &bob).unwrap();
        social.create_post("newest", 300, None, &bob).unwrap();
        social.create_post("middle", 200, None, &bob).unwrap();
        social.follow(&alice, bob.peer_id()).unwrap();
        let feed = social.feed(alice.peer_id());
        let texts: Vec<&str> = feed.iter().map(|item| item.text()).collect();
        assert_eq!(texts, vec!["newest", "middle", "oldest"]);
    }

    #[test]
    fn feed_includes_media_posts() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut social = SocialStore::new();
        let root = Cid::from_bytes(kernel_crypto::sha256(b"photo chunks"));
        social
            .create_post("look at this", 100, Some(image_meta(root)), &bob)
            .unwrap();
        social.follow(&alice, bob.peer_id()).unwrap();
        let feed = social.feed(alice.peer_id());
        assert_eq!(feed.len(), 1);
        assert_eq!(feed[0].text(), "look at this");
        assert_eq!(feed[0].original_post().media.as_ref().unwrap().root, root);
    }

    #[test]
    fn self_follow_rejected_and_list_follows_sorted() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let carol = Identity::generate("carol").unwrap();
        let mut social = SocialStore::new();
        assert_eq!(
            social.follow(&alice, alice.peer_id()),
            Err(SocialError::InvalidFollow(
                "a peer cannot follow itself".to_string()
            ))
        );
        social.follow(&alice, carol.peer_id()).unwrap();
        social.follow(&alice, bob.peer_id()).unwrap();
        let follows = social.list_follows(alice.peer_id());
        let names: Vec<&str> = follows.iter().map(|p| p.display_name()).collect();
        assert_eq!(names, vec!["bob", "carol"]);
    }

    #[test]
    fn get_unknown_post_is_not_found() {
        let social = SocialStore::new();
        let cid = Cid::from_bytes(kernel_crypto::sha256(b"never stored"));
        assert_eq!(social.get_post(&cid), Err(SocialError::NotFound(cid)));
    }

    #[test]
    fn comment_attaches_and_bad_comments_refused() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let carol = Identity::generate("carol").unwrap();
        let mut social = SocialStore::new();
        let post = social.create_post("bob post", 100, None, &bob).unwrap();
        // Comment on an unknown post is refused.
        let unknown = Cid::from_bytes(kernel_crypto::sha256(b"no such post"));
        assert_eq!(
            social.add_comment("hi", 101, &unknown, &alice),
            Err(SocialError::NotFound(unknown))
        );
        // Empty comment refused.
        assert!(social.add_comment("", 101, &post, &alice).is_err());
        // Two comments, listed oldest first.
        social.add_comment("first", 101, &post, &alice).unwrap();
        social.add_comment("second", 102, &post, &bob).unwrap();
        let comments = social.comments_for(alice.peer_id(), &post);
        assert_eq!(comments.len(), 2);
        assert_eq!(comments[0].text, "first");
        assert_eq!(comments[0].author, *alice.peer_id());
        assert_eq!(comments[1].text, "second");
        // A comment with a forged signature is refused on ingest.
        let mut forged = Comment::new(&post, "forged", 103, &alice).unwrap();
        forged.signature = carol.sign(b"not the comment");
        assert_eq!(
            social.ingest_comment(forged),
            Err(SocialError::BadSignature)
        );
        // The same comment ingested twice is stored once.
        let dup = Comment::new(&post, "dup", 105, &alice).unwrap();
        let cid1 = social.ingest_comment(dup.clone()).unwrap();
        let cid2 = social.ingest_comment(dup).unwrap();
        assert_eq!(cid1, cid2);
        let all = social.comments_for(alice.peer_id(), &post);
        let dups: Vec<&Comment> = all.iter().filter(|c| c.text == "dup").collect();
        assert_eq!(dups.len(), 1);
        // Unknown posts have no comments.
        assert!(social.comments_for(alice.peer_id(), &unknown).is_empty());
    }

    #[test]
    fn like_idempotent_unlike_and_count() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let carol = Identity::generate("carol").unwrap();
        let mut social = SocialStore::new();
        let post = social.create_post("post", 100, None, &bob).unwrap();
        // Like on an unknown post is refused.
        let unknown = Cid::from_bytes(kernel_crypto::sha256(b"no such post"));
        assert_eq!(
            social.like(&alice, &unknown),
            Err(SocialError::NotFound(unknown))
        );
        // Liking your own post is allowed.
        let bob_like = social.like(&bob, &post).unwrap();
        assert_eq!(social.like_count(&post), 1);
        // Liking twice is one like: same record CID, count unchanged.
        let again = social.like(&bob, &post).unwrap();
        assert_eq!(again, bob_like);
        assert_eq!(social.like_count(&post), 1);
        social.like(&alice, &post).unwrap();
        assert_eq!(social.like_count(&post), 2);
        // Unlike works; a second unlike is a clean no-op.
        assert!(social.unlike(&alice, &post).unwrap());
        assert_eq!(social.like_count(&post), 1);
        assert!(!social.unlike(&alice, &post).unwrap());
        assert_eq!(social.like_count(&post), 1);
        // A like with a forged signature is refused on ingest.
        let mut forged = Like::new(&post, &alice).unwrap();
        forged.signature = carol.sign(b"not the like");
        assert_eq!(social.ingest_like(forged), Err(SocialError::BadSignature));
        // Unknown posts have no likes.
        assert_eq!(social.like_count(&unknown), 0);
    }

    #[test]
    fn repost_in_feed_and_chain_resolves() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let carol = Identity::generate("carol").unwrap();
        let dave = Identity::generate("dave").unwrap();
        let mut social = SocialStore::new();
        let bob_post = social.create_post("bob wisdom", 100, None, &bob).unwrap();
        // Repost of an unknown record is refused.
        let unknown = Cid::from_bytes(kernel_crypto::sha256(b"no such record"));
        assert_eq!(
            social.repost(&carol, &unknown, None, 110),
            Err(SocialError::NotFound(unknown))
        );
        // Carol reposts Bob's post with a comment of her own.
        let carol_repost = social
            .repost(&carol, &bob_post, Some("so true"), 110)
            .unwrap();
        // Alice follows Carol, not Bob: she sees the repost, marked as
        // a repost, with Bob's words and Carol as the sharer.
        social.follow(&alice, carol.peer_id()).unwrap();
        let feed = social.feed(alice.peer_id());
        assert_eq!(feed.len(), 1);
        assert!(feed[0].is_repost());
        assert_eq!(feed[0].text(), "bob wisdom");
        assert_eq!(*feed[0].author(), *carol.peer_id());
        assert_eq!(feed[0].original_post().author, *bob.peer_id());
        assert_eq!(
            feed[0].repost().unwrap().comment.as_deref(),
            Some("so true")
        );
        // Dave reposts Carol's repost: the chain resolves to Bob's post.
        let dave_repost = social.repost(&dave, &carol_repost, None, 120).unwrap();
        assert_eq!(social.resolve_original(&dave_repost), Some(bob_post));
        assert_eq!(
            social.get_repost(&dave_repost).unwrap().original,
            carol_repost
        );
        // Alice follows Dave too: his repost also shows Bob's post.
        social.follow(&alice, dave.peer_id()).unwrap();
        let feed = social.feed(alice.peer_id());
        assert_eq!(feed.len(), 2);
        assert!(feed.iter().all(|item| item.is_repost()));
        assert!(feed.iter().all(|item| item.text() == "bob wisdom"));
        // Newest first: Dave's repost (120) before Carol's (110).
        assert_eq!(feed[0].timestamp(), 120);
        assert_eq!(feed[1].timestamp(), 110);
        // A repost with a forged signature is refused on ingest.
        let mut forged = Repost::new(&bob_post, None, 130, &carol).unwrap();
        forged.signature = dave.sign(b"not the repost");
        assert_eq!(social.ingest_repost(forged), Err(SocialError::BadSignature));
    }

    #[test]
    fn media_metadata_validated() {
        let alice = Identity::generate("alice").unwrap();
        let root = Cid::from_bytes(kernel_crypto::sha256(b"video chunks"));
        // Zero size and oversize are refused at metadata time.
        assert!(MediaMeta::new(MediaKind::Image, 0, root).is_err());
        assert!(MediaMeta::new(MediaKind::Video, MAX_MEDIA_BYTES + 1, root).is_err());
        // Exactly at the cap is fine.
        let meta = MediaMeta::new(MediaKind::Video, MAX_MEDIA_BYTES, root).unwrap();
        let post = Post::new("video", 1, Some(meta.clone()), &alice).unwrap();
        assert!(post.verify());
        let decoded = Post::decode(&post.encode()).unwrap();
        assert_eq!(decoded.media, Some(meta));
        assert_eq!(decoded.media.as_ref().unwrap().kind, MediaKind::Video);
        assert_eq!(decoded.media.as_ref().unwrap().size_bytes, MAX_MEDIA_BYTES);
        assert!(decoded.verify());
        // Tampering with the kind after signing breaks verification.
        let mut tampered = decoded.clone();
        tampered.media.as_mut().unwrap().kind = MediaKind::Audio;
        assert!(!tampered.verify());
        // Tampering with the size after signing breaks verification.
        let mut tampered = decoded.clone();
        tampered.media.as_mut().unwrap().size_bytes = 42;
        assert!(!tampered.verify());
        // A bad kind byte in the encoding is refused at decode.
        // Offset of the kind byte: magic 6 + author 32 + text len 4 +
        // text 5 ("video") + timestamp 8 + flag 1 + root 32.
        let kind_off = 6 + 32 + 4 + 5 + 8 + 1 + 32;
        let mut bytes = post.encode();
        bytes[kind_off] = 9;
        assert!(Post::decode(&bytes).is_err());
        // A size over the cap in the encoding is refused at decode.
        let mut bytes = post.encode();
        bytes[kind_off + 1 + 7] = 0xff;
        assert!(Post::decode(&bytes).is_err());
        // A post created with invalid media never reaches the store.
        assert!(social_create_with_bad_media(&alice).is_err());
    }

    fn social_create_with_bad_media(alice: &Identity) -> Result<Cid, SocialError> {
        // MediaMeta::new refuses the oversize metadata before any
        // post is built, so this can only fail at metadata time.
        let root = Cid::from_bytes(kernel_crypto::sha256(b"huge"));
        let meta = MediaMeta::new(MediaKind::Video, MAX_MEDIA_BYTES + 1, root)?;
        let mut social = SocialStore::new();
        social.create_post("huge video", 1, Some(meta), alice)
    }

    #[test]
    fn blocking_hides_content_and_unblock_restores() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let carol = Identity::generate("carol").unwrap();
        let mut social = SocialStore::new();
        let bob_post = social.create_post("bob post", 100, None, &bob).unwrap();
        social.create_post("carol post", 101, None, &carol).unwrap();
        social
            .add_comment("carol comments", 102, &bob_post, &carol)
            .unwrap();
        social.like(&carol, &bob_post).unwrap();
        social.like(&bob, &bob_post).unwrap();
        social.repost(&carol, &bob_post, None, 103).unwrap();
        social.follow(&alice, bob.peer_id()).unwrap();
        social.follow(&alice, carol.peer_id()).unwrap();
        // Before blocking: Bob's post, Carol's post, Carol's repost.
        assert_eq!(social.feed(alice.peer_id()).len(), 3);
        assert_eq!(social.comments_for(alice.peer_id(), &bob_post).len(), 1);
        assert_eq!(social.like_count(&bob_post), 2);
        assert_eq!(social.like_count_for(alice.peer_id(), &bob_post), 2);
        // Blocking yourself is refused; blocking is idempotent.
        assert_eq!(
            social.block(&alice, alice.peer_id()),
            Err(SocialError::InvalidBlock(
                "a peer cannot block itself".to_string()
            ))
        );
        let first = social.block(&alice, carol.peer_id()).unwrap();
        let again = social.block(&alice, carol.peer_id()).unwrap();
        assert_eq!(first, again);
        assert!(social.is_blocked(alice.peer_id(), carol.peer_id()));
        assert_eq!(
            social.list_blocks(alice.peer_id()),
            vec![carol.peer_id().clone()]
        );
        // After blocking Carol: her post, her comment, her like, and
        // her repost are all hidden from Alice. Bob's post stays.
        let feed = social.feed(alice.peer_id());
        assert_eq!(feed.len(), 1);
        assert_eq!(feed[0].text(), "bob post");
        assert!(social.comments_for(alice.peer_id(), &bob_post).is_empty());
        // The total like count is unchanged; Alice's view drops Carol.
        assert_eq!(social.like_count(&bob_post), 2);
        assert_eq!(social.like_count_for(alice.peer_id(), &bob_post), 1);
        // Unblocking restores everything at once.
        assert!(social.unblock(&alice, carol.peer_id()).unwrap());
        assert!(!social.is_blocked(alice.peer_id(), carol.peer_id()));
        assert_eq!(social.feed(alice.peer_id()).len(), 3);
        assert_eq!(social.comments_for(alice.peer_id(), &bob_post).len(), 1);
        assert_eq!(social.like_count_for(alice.peer_id(), &bob_post), 2);
        // Unblocking again is a clean no-op.
        assert!(!social.unblock(&alice, carol.peer_id()).unwrap());
    }

    #[test]
    fn repost_of_blocked_author_hidden() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let carol = Identity::generate("carol").unwrap();
        let mut social = SocialStore::new();
        let bob_post = social.create_post("bob wisdom", 100, None, &bob).unwrap();
        social.repost(&carol, &bob_post, None, 110).unwrap();
        social.follow(&alice, carol.peer_id()).unwrap();
        // Before blocking Bob, Carol's repost of his post is visible.
        assert_eq!(social.feed(alice.peer_id()).len(), 1);
        // Blocking the original author hides the repost too, even
        // though the reposter is not blocked.
        social.block(&alice, bob.peer_id()).unwrap();
        assert!(social.feed(alice.peer_id()).is_empty());
        // And a comment under a blocked author's post is hidden, even
        // though the commenter is not blocked.
        social
            .add_comment("carol comments", 111, &bob_post, &carol)
            .unwrap();
        assert!(social.comments_for(alice.peer_id(), &bob_post).is_empty());
        // Unblocking Bob restores both.
        social.unblock(&alice, bob.peer_id()).unwrap();
        assert_eq!(social.feed(alice.peer_id()).len(), 1);
        assert_eq!(social.comments_for(alice.peer_id(), &bob_post).len(), 1);
    }

    #[test]
    fn block_record_round_trip_and_bad_signature_refused() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let carol = Identity::generate("carol").unwrap();
        let block = Block::new(&alice, bob.peer_id()).unwrap();
        assert!(block.verify());
        let decoded = Block::decode(&block.encode()).unwrap();
        assert_eq!(decoded.blocker.public_key(), alice.peer_id().public_key());
        assert_eq!(decoded.blocked.public_key(), bob.peer_id().public_key());
        assert!(decoded.verify());
        assert!(Block::decode(b"garbage").is_err());
        // A block with a forged signature is refused on ingest.
        let mut social = SocialStore::new();
        let mut forged = Block::new(&alice, bob.peer_id()).unwrap();
        forged.signature = carol.sign(b"not the block");
        assert_eq!(social.ingest_block(forged), Err(SocialError::BadSignature));
        // Ingesting the real record works and is idempotent.
        let cid1 = social.ingest_block(block.clone()).unwrap();
        let cid2 = social.ingest_block(block).unwrap();
        assert_eq!(cid1, cid2);
        assert!(social.is_blocked(alice.peer_id(), bob.peer_id()));
    }

    #[test]
    fn like_and_comment_record_round_trips() {
        let alice = Identity::generate("alice").unwrap();
        let post = Cid::from_bytes(kernel_crypto::sha256(b"some post"));
        let like = Like::new(&post, &alice).unwrap();
        assert!(like.verify());
        let decoded = Like::decode(&like.encode()).unwrap();
        assert_eq!(decoded.post, post);
        assert_eq!(decoded.liker.public_key(), alice.peer_id().public_key());
        assert!(decoded.verify());
        assert!(Like::decode(b"garbage").is_err());
        let comment = Comment::new(&post, "nice", 7, &alice).unwrap();
        assert!(comment.verify());
        let decoded = Comment::decode(&comment.encode()).unwrap();
        assert_eq!(decoded.post, post);
        assert_eq!(decoded.text, "nice");
        assert!(decoded.verify());
        assert!(Comment::decode(b"garbage").is_err());
    }
}
