//! Planet•K — social on KERNEL.
//!
//! A **post** is a signed record: the author, text, a timestamp, an
//! optional media chunk root CID, and the author's signature. A
//! **follow** is a signed record: follower, followed, and the
//! follower's signature. The **feed** is the set of posts from peers
//! you follow, newest first — and every post's signature is verified
//! on read, so a post whose signature does not check out never reaches
//! the feed.
//!
//! Posts are addressed by record CID; humans see author names and
//! text, never hashes. Timestamps are caller-supplied (seconds since
//! the Unix epoch) so ordering is explicit and testable; a node that
//! wants wall-clock time supplies its own clock.
//!
//! Record formats (explicit binary, no JSON):
//! `KPOST1` + author public key (32) + text length (u32 LE) + text
//! bytes + timestamp (u64 LE) + media flag (u8: 0 = none, 1 = present)
//! + media chunk root CID (32, only if flag is 1) + signature (64).
//! `KFOL1` + follower public key (32) + followed public key (32) +
//! signature (64).

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

/// Maximum post text length in characters.
pub const MAX_POST_LEN: usize = 10_000;

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
/// no control characters except newline and tab.
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
    if text.chars().any(|c| c.is_control() && c != '\n' && c != '\t') {
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

/// The message an author signs for a post.
fn post_sign_message(
    author: &PeerId,
    text: &str,
    timestamp: u64,
    media: Option<&Cid>,
) -> Vec<u8> {
    let mut msg =
        Vec::with_capacity(POST_SIGN_DOMAIN.len() + 32 + text.len() + 8 + 1 + 32);
    msg.extend_from_slice(POST_SIGN_DOMAIN);
    msg.extend_from_slice(author.public_key().as_bytes());
    msg.extend_from_slice(text.as_bytes());
    msg.extend_from_slice(&timestamp.to_le_bytes());
    match media {
        Some(root) => {
            msg.push(1);
            msg.extend_from_slice(root.as_bytes());
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

/// A signed post: author, text, timestamp, optional media root, signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Post {
    /// The peer who wrote this post.
    pub author: PeerId,
    /// The post text.
    pub text: String,
    /// Unix timestamp (seconds) the post was written.
    pub timestamp: u64,
    /// Optional media chunk root CID (a `KCHUNK1` manifest block).
    pub media: Option<Cid>,
    /// The author's signature over author, text, timestamp, and media.
    pub signature: Ed25519Signature,
}

impl Post {
    /// Build and sign a post as `identity`.
    pub fn new(
        text: &str,
        timestamp: u64,
        media: Option<Cid>,
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
    /// exact text, timestamp, and media. Any tampering fails this check.
    pub fn verify(&self) -> bool {
        let msg = post_sign_message(&self.author, &self.text, self.timestamp, self.media.as_ref());
        self.author.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary post format.
    pub fn encode(&self) -> Vec<u8> {
        let text_bytes = self.text.as_bytes();
        let mut out =
            Vec::with_capacity(6 + 32 + 4 + text_bytes.len() + 8 + 1 + 32 + 64);
        out.extend_from_slice(POST_MAGIC);
        out.extend_from_slice(self.author.public_key().as_bytes());
        out.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(text_bytes);
        out.extend_from_slice(&self.timestamp.to_le_bytes());
        match &self.media {
            Some(root) => {
                out.push(1);
                out.extend_from_slice(root.as_bytes());
            }
            None => out.push(0),
        }
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a post produced by [`Post::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, SocialError> {
        if bytes.len() < 6 + 32 + 4 + 8 + 1 + 64 || &bytes[..6] != POST_MAGIC {
            return Err(SocialError::InvalidRecord("missing KPOST1 magic".to_string()));
        }
        let text_len = u32::from_le_bytes(
            bytes[38..42].try_into().expect("4-byte slice"),
        ) as usize;
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
        let expected = flag_off + 1 + if flag == 1 { 32 } else { 0 } + 64;
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
            Some(Cid::from_bytes(root_bytes))
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

/// The social store: a content store holding signed posts and follow
/// records, plus the follow graph. Posts are addressed by record CID;
/// humans see author names and text.
pub struct SocialStore {
    store: ContentStore,
    /// Posts this node holds, by record CID.
    posts: HashMap<Cid, Post>,
    /// Follow graph: follower -> set of followed peers.
    follows: HashMap<PeerId, HashSet<PeerId>>,
    /// Follow record CIDs, by (follower, followed).
    follow_records: HashMap<(PeerId, PeerId), Cid>,
}

impl SocialStore {
    /// An empty social store.
    pub fn new() -> Self {
        Self {
            store: ContentStore::new(),
            posts: HashMap::new(),
            follows: HashMap::new(),
            follow_records: HashMap::new(),
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
    /// block. Returns the post record CID.
    pub fn create_post(
        &mut self,
        text: &str,
        timestamp: u64,
        media: Option<Cid>,
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
        if let Some(cid) = self.follow_records.get(&(follower.clone(), followed.clone())) {
            return Ok(*cid);
        }
        let follow = Follow::new(follower_identity, followed)?;
        let cid = self.store.put(&follow.encode());
        self.follows
            .entry(follower.clone())
            .or_default()
            .insert(followed.clone());
        self.follow_records.insert((follower, followed.clone()), cid);
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

    /// The feed for a peer: posts from followed peers, newest first
    /// (timestamp descending, ties broken by record CID hex so the
    /// order is deterministic). Every post's signature is verified on
    /// read; a post that fails verification is left out of the feed.
    pub fn feed(&self, follower: &PeerId) -> Vec<Post> {
        let followed = match self.follows.get(follower) {
            Some(set) => set,
            None => return Vec::new(),
        };
        let mut out: Vec<(Cid, Post)> = self
            .posts
            .iter()
            .filter(|(_, p)| followed.contains(&p.author) && p.verify())
            .map(|(cid, p)| (*cid, p.clone()))
            .collect();
        out.sort_by(|a, b| {
            b.1.timestamp
                .cmp(&a.1.timestamp)
                .then_with(|| a.0.to_hex().cmp(&b.0.to_hex()))
        });
        out.into_iter().map(|(_, p)| p).collect()
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

    #[test]
    fn post_encode_decode_round_trip_and_verify() {
        let alice = Identity::generate("alice").unwrap();
        let media = Cid::from_bytes(kernel_crypto::sha256(b"photo chunks"));
        let post = Post::new("hello world", 1_700_000_000, Some(media), &alice).unwrap();
        assert!(post.verify());
        let encoded = post.encode();
        let decoded = Post::decode(&encoded).unwrap();
        assert_eq!(decoded.text, "hello world");
        assert_eq!(decoded.timestamp, 1_700_000_000);
        assert_eq!(decoded.media, Some(media));
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
        let bob_post = social
            .create_post("bob says hi", 100, None, &bob)
            .unwrap();
        let carol_post = social
            .create_post("carol says hi", 101, None, &carol)
            .unwrap();
        // Alice follows nobody yet: empty feed.
        assert!(social.feed(alice.peer_id()).is_empty());
        // Alice follows Bob.
        social.follow(&alice, bob.peer_id()).unwrap();
        let feed = social.feed(alice.peer_id());
        assert_eq!(feed.len(), 1);
        assert_eq!(feed[0].text, "bob says hi");
        assert_eq!(feed[0].author, *bob.peer_id());
        // Carol's post is not in Alice's feed.
        assert!(!feed.iter().any(|p| p.text == "carol says hi"));
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
        let texts: Vec<&str> = feed.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(texts, vec!["newest", "middle", "oldest"]);
    }

    #[test]
    fn feed_includes_media_posts() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut social = SocialStore::new();
        let media = Cid::from_bytes(kernel_crypto::sha256(b"photo chunks"));
        social
            .create_post("look at this", 100, Some(media), &bob)
            .unwrap();
        social.follow(&alice, bob.peer_id()).unwrap();
        let feed = social.feed(alice.peer_id());
        assert_eq!(feed.len(), 1);
        assert_eq!(feed[0].text, "look at this");
        assert_eq!(feed[0].media, Some(media));
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
        let names: Vec<&str> = follows
            .iter()
            .map(|p| p.display_name())
            .collect();
        assert_eq!(names, vec!["bob", "carol"]);
    }

    #[test]
    fn get_unknown_post_is_not_found() {
        let social = SocialStore::new();
        let cid = Cid::from_bytes(kernel_crypto::sha256(b"never stored"));
        assert_eq!(social.get_post(&cid), Err(SocialError::NotFound(cid)));
    }
}
