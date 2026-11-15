//! K•Docs — documents with version history on top of KERNEL storage.
//!
//! A document has a stable human-readable id, a title, and a chain of
//! versions. Each version is a signed record: the doc id, the title,
//! the version number, the chunked-content root CID of that version's
//! content, the owner, and the owner's signature over all of it.
//!
//! The id is derived from the title and the first version's content
//! root, so the same owner creating the same title with the same
//! initial bytes gets the same id (idempotent create); different
//! content gives a different id. Versions are numbered from 1 and
//! increase monotonically; only the doc's owner may add a version.
//! Older versions stay retrievable — history is never rewritten.
//!
//! A version record is itself a block in the content store, addressed
//! by its own CID, so versions can be received from peers and verified
//! before they are accepted.
//!
//! Version record format (explicit binary, no JSON):
//! `KDOCV1` + id length (u16 LE) + id bytes + title length (u16 LE) +
//! title bytes + version (u64 LE) + chunk root CID (32) + owner public
//! key (32) + signature (64).

use std::collections::HashMap;
use std::fmt;

use kernel_crypto::{sha256, to_hex, Ed25519Signature};
use kernel_identity::{Identity, IdentityError, PeerId};
use kernel_storage::{Cid, ContentStore, StorageError};

/// Magic bytes at the start of every version record.
const RECORD_MAGIC: &[u8; 6] = b"KDOCV1";

/// Domain separator for version signatures: a signature covers
/// `b"KDOCV1" || id || title || version || root || owner key`.
const RECORD_SIGN_DOMAIN: &[u8] = b"KDOCV1";

/// Errors returned by document operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocError {
    /// Wraps a [`StorageError`].
    Storage(StorageError),
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// A title broke the naming rules.
    InvalidTitle(String),
    /// A version record was malformed.
    InvalidRecord(String),
    /// A version record's signature did not verify against its owner.
    BadSignature,
    /// No document with this id is known here.
    NotFound(String),
    /// No such version of the document exists.
    VersionNotFound { id: String, version: u64 },
    /// Only the document's owner may add a version.
    NotOwner,
    /// A version record's title does not match the document's title.
    TitleMismatch,
}

impl fmt::Display for DocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DocError::Storage(e) => write!(f, "storage error: {e}"),
            DocError::Identity(e) => write!(f, "identity error: {e}"),
            DocError::InvalidTitle(msg) => write!(f, "invalid title: {msg}"),
            DocError::InvalidRecord(msg) => write!(f, "invalid version record: {msg}"),
            DocError::BadSignature => write!(f, "version signature is invalid"),
            DocError::NotFound(id) => write!(f, "document not found: {id}"),
            DocError::VersionNotFound { id, version } => {
                write!(f, "document {id} has no version {version}")
            }
            DocError::NotOwner => write!(f, "only the document owner may add a version"),
            DocError::TitleMismatch => {
                write!(f, "version record title does not match the document")
            }
        }
    }
}

impl std::error::Error for DocError {}

impl From<StorageError> for DocError {
    fn from(e: StorageError) -> Self {
        DocError::Storage(e)
    }
}

impl From<IdentityError> for DocError {
    fn from(e: IdentityError) -> Self {
        DocError::Identity(e)
    }
}

/// Rules for titles: 1-128 printable characters. Titles are human
/// labels (`"Meeting notes"`), not hashes.
fn validate_title(title: &str) -> Result<(), DocError> {
    if title.is_empty() {
        return Err(DocError::InvalidTitle(
            "title must not be empty".to_string(),
        ));
    }
    if title.len() > 128 {
        return Err(DocError::InvalidTitle(
            "title must be at most 128 characters".to_string(),
        ));
    }
    if !title.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
        return Err(DocError::InvalidTitle(
            "title must be printable ASCII".to_string(),
        ));
    }
    Ok(())
}

/// Slugify a title into a URL-safe id fragment: lowercase, runs of
/// non-alphanumerics become one `-`, at most 32 characters.
/// A title with no alphanumerics becomes `"doc"`.
fn slug(title: &str) -> String {
    let mut out = String::new();
    let mut last_dash = true; // trims leading dashes
    for c in title.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
        if out.len() >= 32 {
            break;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        out.push_str("doc");
    }
    out
}

/// Derive a document id from its title and its first version's
/// content root, bound to the owner: `slug(title)-<8 hex>`.
/// Deterministic for (owner, title, first-version content), so
/// creating the same document twice returns the same id.
fn derive_doc_id(title: &str, root: &Cid, owner: &PeerId) -> String {
    let mut msg = Vec::new();
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg.extend_from_slice(title.as_bytes());
    msg.extend_from_slice(root.as_bytes());
    let hash = sha256(&msg);
    format!("{}-{}", slug(title), &to_hex(&hash)[..8])
}

/// The message an owner signs for a version record: domain + id +
/// title + version + root + owner key.
fn record_sign_message(
    id: &str,
    title: &str,
    version: u64,
    root: &Cid,
    owner: &PeerId,
) -> Vec<u8> {
    let mut msg = Vec::with_capacity(
        RECORD_SIGN_DOMAIN.len() + id.len() + title.len() + 8 + 32 + 32,
    );
    msg.extend_from_slice(RECORD_SIGN_DOMAIN);
    msg.extend_from_slice(id.as_bytes());
    msg.extend_from_slice(title.as_bytes());
    msg.extend_from_slice(&version.to_le_bytes());
    msg.extend_from_slice(root.as_bytes());
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg
}

/// One signed version of a document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VersionRecord {
    /// The document's stable id.
    pub id: String,
    /// Human-readable title (immutable across versions).
    pub title: String,
    /// Version number, starting at 1, increasing by 1 per update.
    pub version: u64,
    /// Root CID of the chunked content of this version.
    pub root: Cid,
    /// The owner who signed this version.
    pub owner: PeerId,
    /// The owner's signature over id, title, version, root, and key.
    pub signature: Ed25519Signature,
}

impl VersionRecord {
    /// Build and sign a version record as `identity`.
    pub fn new(
        id: &str,
        title: &str,
        version: u64,
        root: Cid,
        identity: &Identity,
    ) -> Result<Self, DocError> {
        validate_title(title)?;
        let owner = identity.peer_id().clone();
        let msg = record_sign_message(id, title, version, &root, &owner);
        let signature = identity.sign(&msg);
        Ok(Self {
            id: id.to_string(),
            title: title.to_string(),
            version,
            root,
            owner,
            signature,
        })
    }

    /// True if the signature verifies against the owner for this exact
    /// id, title, version, and root. Any tampering fails this check.
    pub fn verify(&self) -> bool {
        let msg = record_sign_message(&self.id, &self.title, self.version, &self.root, &self.owner);
        self.owner.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary record format.
    pub fn encode(&self) -> Vec<u8> {
        let id_bytes = self.id.as_bytes();
        let title_bytes = self.title.as_bytes();
        let mut out = Vec::with_capacity(
            6 + 2 + id_bytes.len() + 2 + title_bytes.len() + 8 + 32 + 32 + 64,
        );
        out.extend_from_slice(RECORD_MAGIC);
        out.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(id_bytes);
        out.extend_from_slice(&(title_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(title_bytes);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(self.root.as_bytes());
        out.extend_from_slice(self.owner.public_key().as_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a record produced by [`VersionRecord::encode`].
    /// Rejects anything that is not exactly that format.
    pub fn decode(bytes: &[u8]) -> Result<Self, DocError> {
        if bytes.len() < 6 + 2 + 2 + 8 + 32 + 32 + 64 || &bytes[..6] != RECORD_MAGIC {
            return Err(DocError::InvalidRecord("missing KDOCV1 magic".to_string()));
        }
        let id_len = u16::from_le_bytes(bytes[6..8].try_into().expect("2-byte slice")) as usize;
        let title_len_off = 8 + id_len;
        if bytes.len() < title_len_off + 2 {
            return Err(DocError::InvalidRecord(
                "record too short for title length".to_string(),
            ));
        }
        let title_len =
            u16::from_le_bytes(bytes[title_len_off..title_len_off + 2].try_into().expect("2-byte slice"))
                as usize;
        let expected = title_len_off + 2 + title_len + 8 + 32 + 32 + 64;
        if bytes.len() != expected {
            return Err(DocError::InvalidRecord(format!(
                "record is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let mut off = 8;
        let id = String::from_utf8(bytes[off..off + id_len].to_vec()).map_err(|_| {
            DocError::InvalidRecord("record id is not valid UTF-8".to_string())
        })?;
        off += id_len;
        let title_len =
            u16::from_le_bytes(bytes[off..off + 2].try_into().expect("2-byte slice")) as usize;
        off += 2;
        let title = String::from_utf8(bytes[off..off + title_len].to_vec()).map_err(|_| {
            DocError::InvalidRecord("record title is not valid UTF-8".to_string())
        })?;
        validate_title(&title)?;
        off += title_len;
        let version = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let mut root_bytes = [0u8; 32];
        root_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let root = Cid::from_bytes(root_bytes);
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let owner_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| DocError::InvalidRecord(e.to_string()))?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        // The record stores the owner's public key, not their display
        // name: verification needs the key, and the key is what
        // authenticates the owner. Decode rebuilds the PeerId with a
        // deterministic name derived from the key, so a decoded record
        // verifies exactly like the original.
        let owner = PeerId::new(owner_key, &fingerprint_name(&owner_key))?;
        Ok(Self {
            id,
            title,
            version,
            root,
            owner,
            signature,
        })
    }
}

/// A display name derived from a public key: the first 8 hex chars of
/// the key. Used when a record is decoded and only the key survives
/// (verification needs the key, not the human name).
fn fingerprint_name(key: &kernel_crypto::Ed25519PublicKey) -> String {
    let hex = kernel_crypto::to_hex(key.as_bytes());
    format!("peer-{}", &hex[..8])
}

/// The metadata of a document version, as a human sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocMetadata {
    /// The document's stable id.
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// Version number this metadata describes.
    pub version: u64,
    /// Who owns this document.
    pub owner: PeerId,
}

/// One document's entry: its title, owner, and its version chain
/// (index `version - 1` holds version `version`).
struct DocEntry {
    title: String,
    owner: PeerId,
    versions: Vec<VersionRecord>,
}

/// A document store: a content store plus an index of documents.
/// Documents are addressed by human-readable id; humans see titles.
pub struct DocStore {
    store: ContentStore,
    /// Documents held, by doc id.
    docs: HashMap<String, DocEntry>,
}

impl DocStore {
    /// An empty document store.
    pub fn new() -> Self {
        Self {
            store: ContentStore::new(),
            docs: HashMap::new(),
        }
    }

    /// Number of documents held.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// True if no documents are held.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// The latest version number of a document, if it is held.
    pub fn latest_version(&self, doc_id: &str) -> Option<u64> {
        self.docs.get(doc_id).map(|e| e.versions.len() as u64)
    }

    /// The version record for a document version, if held.
    pub fn get_version_record(&self, doc_id: &str, version: u64) -> Option<&VersionRecord> {
        let entry = self.docs.get(doc_id)?;
        if version == 0 || version as usize > entry.versions.len() {
            return None;
        }
        Some(&entry.versions[version as usize - 1])
    }

    /// Create a document: store the content chunked, sign version 1,
    /// and store the version record as a block. Returns the document
    /// id. Creating the same title with the same initial bytes as the
    /// same owner returns the same id (idempotent create).
    pub fn create_doc(
        &mut self,
        title: &str,
        data: &[u8],
        identity: &Identity,
    ) -> Result<String, DocError> {
        validate_title(title)?;
        let root = self.store.put_chunked(data);
        let id = derive_doc_id(title, &root, identity.peer_id());
        if self.docs.contains_key(&id) {
            // Same owner, title, and first-version content: the same
            // document. Idempotent — no duplicate version is made.
            return Ok(id);
        }
        let record = VersionRecord::new(&id, title, 1, root, identity)?;
        let record_bytes = record.encode();
        self.store.put(&record_bytes);
        self.docs.insert(
            id.clone(),
            DocEntry {
                title: title.to_string(),
                owner: identity.peer_id().clone(),
                versions: vec![record],
            },
        );
        Ok(id)
    }

    /// Add a new version to a document. Only the document's owner may
    /// do this. Returns the new version number.
    pub fn update_doc(
        &mut self,
        doc_id: &str,
        data: &[u8],
        identity: &Identity,
    ) -> Result<u64, DocError> {
        let entry = self
            .docs
            .get_mut(doc_id)
            .ok_or_else(|| DocError::NotFound(doc_id.to_string()))?;
        if entry.owner.public_key() != identity.peer_id().public_key() {
            return Err(DocError::NotOwner);
        }
        let version = entry.versions.len() as u64 + 1;
        let root = self.store.put_chunked(data);
        let record = VersionRecord::new(doc_id, &entry.title, version, root, identity)?;
        let record_bytes = record.encode();
        self.store.put(&record_bytes);
        entry.versions.push(record);
        Ok(version)
    }

    /// Fetch a document's latest version: verify the version record's
    /// signature, then reassemble the chunked content (every chunk
    /// hash-verified). Returns the bytes and the version's metadata.
    pub fn get_doc(&self, doc_id: &str) -> Result<(Vec<u8>, DocMetadata), DocError> {
        let entry = self
            .docs
            .get(doc_id)
            .ok_or_else(|| DocError::NotFound(doc_id.to_string()))?;
        let record = entry
            .versions
            .last()
            .expect("a document entry always holds at least version 1");
        self.read_version(record)
    }

    /// Fetch one specific version's bytes, verifying its signature
    /// and every chunk hash. History is never rewritten, so an old
    /// version reads exactly what was written for it.
    pub fn get_version(&self, doc_id: &str, version: u64) -> Result<Vec<u8>, DocError> {
        let record = self.get_version_record(doc_id, version).ok_or_else(|| {
            if self.docs.contains_key(doc_id) {
                DocError::VersionNotFound {
                    id: doc_id.to_string(),
                    version,
                }
            } else {
                DocError::NotFound(doc_id.to_string())
            }
        })?;
        Ok(self.read_version(record)?.0)
    }

    /// Read one version record's content: verify its signature first,
    /// then reassemble its chunked content.
    fn read_version(&self, record: &VersionRecord) -> Result<(Vec<u8>, DocMetadata), DocError> {
        if !record.verify() {
            return Err(DocError::BadSignature);
        }
        let data = self
            .store
            .get_chunked(&record.root)
            .map_err(DocError::Storage)?;
        let metadata = DocMetadata {
            id: record.id.clone(),
            title: record.title.clone(),
            version: record.version,
            owner: record.owner.clone(),
        };
        Ok((data, metadata))
    }

    /// Accept a version record received from a peer (or from the
    /// content store). The signature is verified first; a bad record
    /// is never stored. Returns `Ok(true)` if the record was accepted
    /// as a new version, `Ok(false)` if it was a duplicate or an old
    /// version that loses to what is already held.
    pub fn apply_version(&mut self, record: VersionRecord) -> Result<bool, DocError> {
        if !record.verify() {
            return Err(DocError::BadSignature);
        }
        match self.docs.get_mut(&record.id) {
            None => {
                // A new document: only version 1 can start it.
                if record.version != 1 {
                    return Ok(false);
                }
                let record_bytes = record.encode();
                self.store.put(&record_bytes);
                self.docs.insert(
                    record.id.clone(),
                    DocEntry {
                        title: record.title.clone(),
                        owner: record.owner.clone(),
                        versions: vec![record],
                    },
                );
                Ok(true)
            }
            Some(entry) => {
                if entry.title != record.title {
                    return Err(DocError::TitleMismatch);
                }
                if entry.owner.public_key() != record.owner.public_key() {
                    return Err(DocError::NotOwner);
                }
                let expected = entry.versions.len() as u64 + 1;
                if record.version != expected {
                    // Duplicate or out-of-order version: ignored, not
                    // stored. Versions arrive in order; gaps are not
                    // filled by guessing.
                    return Ok(false);
                }
                let record_bytes = record.encode();
                self.store.put(&record_bytes);
                entry.versions.push(record);
                Ok(true)
            }
        }
    }

    /// All documents held, as (doc id, title, latest version),
    /// sorted by title for a stable, human-readable listing.
    pub fn list_docs(&self) -> Vec<(String, String, u64)> {
        let mut out: Vec<(String, String, u64)> = self
            .docs
            .iter()
            .map(|(id, e)| (id.clone(), e.title.clone(), e.versions.len() as u64))
            .collect();
        out.sort_by(|a, b| a.1.cmp(&b.1));
        out
    }

    /// All content blocks held, as (CID, bytes). Lets a node offer
    /// its blocks to a peer during sync.
    pub fn export_blocks(&self) -> Vec<(Cid, Vec<u8>)> {
        self.store.blocks()
    }

    /// Store content blocks received from a peer. Bytes are stored
    /// content-addressed, so a block lands under the same CID it had
    /// on the sending node. Version records arrive separately via
    /// [`DocStore::apply_version`]; content and records travel apart
    /// because records are small metadata and content is fetched by
    /// CID, on demand.
    pub fn import_blocks(&mut self, blocks: &[(Cid, Vec<u8>)]) {
        for (cid, bytes) in blocks {
            // Content addressing: the CID must match the bytes.
            // A block that does not hash to its claimed CID is
            // refused rather than stored under a lie.
            if Cid::from_bytes(kernel_crypto::sha256(bytes)) == *cid {
                self.store.put(bytes);
            }
        }
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
    fn record_encode_decode_round_trip_and_verify() {
        let alice = Identity::generate("alice").unwrap();
        let root = Cid::from_bytes(kernel_crypto::sha256(b"doc chunks"));
        let record = VersionRecord::new("notes-1a2b3c4d", "Meeting notes", 1, root, &alice)
            .unwrap();
        assert!(record.verify());
        let encoded = record.encode();
        let decoded = VersionRecord::decode(&encoded).unwrap();
        assert_eq!(decoded.id, "notes-1a2b3c4d");
        assert_eq!(decoded.title, "Meeting notes");
        assert_eq!(decoded.version, 1);
        assert_eq!(decoded.root, root);
        assert_eq!(decoded.signature, record.signature);
        assert_eq!(
            decoded.owner.public_key(),
            record.owner.public_key()
        );
        assert!(decoded.verify());
        assert!(VersionRecord::decode(b"garbage").is_err());
        assert!(VersionRecord::decode(&encoded[..encoded.len() - 1]).is_err());
    }

    #[test]
    fn create_get_round_trip() {
        let alice = Identity::generate("alice").unwrap();
        let mut docs = DocStore::new();
        let id = docs
            .create_doc("Meeting notes", b"agenda: launch", &alice)
            .unwrap();
        assert!(id.starts_with("meeting-notes-"));
        assert_eq!(docs.len(), 1);
        assert_eq!(docs.latest_version(&id), Some(1));
        let (data, meta) = docs.get_doc(&id).unwrap();
        assert_eq!(data, b"agenda: launch");
        assert_eq!(meta.id, id);
        assert_eq!(meta.title, "Meeting notes");
        assert_eq!(meta.version, 1);
        assert_eq!(meta.owner, *alice.peer_id());
    }

    #[test]
    fn create_is_idempotent_for_same_content() {
        let alice = Identity::generate("alice").unwrap();
        let mut docs = DocStore::new();
        let a = docs.create_doc("Notes", b"same", &alice).unwrap();
        let b = docs.create_doc("Notes", b"same", &alice).unwrap();
        assert_eq!(a, b);
        assert_eq!(docs.len(), 1);
        assert_eq!(docs.latest_version(&a), Some(1));
        // Different initial content -> a different document.
        let c = docs.create_doc("Notes", b"different", &alice).unwrap();
        assert_ne!(c, a);
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn update_creates_new_versions_and_history_survives() {
        let alice = Identity::generate("alice").unwrap();
        let mut docs = DocStore::new();
        let id = docs.create_doc("Draft", b"v1 text", &alice).unwrap();
        let v2 = docs.update_doc(&id, b"v2 text", &alice).unwrap();
        assert_eq!(v2, 2);
        let v3 = docs.update_doc(&id, b"v3 text", &alice).unwrap();
        assert_eq!(v3, 3);
        // Latest reads v3.
        let (data, meta) = docs.get_doc(&id).unwrap();
        assert_eq!(data, b"v3 text");
        assert_eq!(meta.version, 3);
        // Older versions still read exactly what was written.
        assert_eq!(docs.get_version(&id, 1).unwrap(), b"v1 text");
        assert_eq!(docs.get_version(&id, 2).unwrap(), b"v2 text");
        assert_eq!(docs.get_version(&id, 3).unwrap(), b"v3 text");
        // Version numbers increase monotonically.
        assert_eq!(docs.latest_version(&id), Some(3));
    }

    #[test]
    fn tampered_version_signature_fails() {
        let alice = Identity::generate("alice").unwrap();
        let mut docs = DocStore::new();
        let id = docs.create_doc("Secret", b"real", &alice).unwrap();
        docs.update_doc(&id, b"newer", &alice).unwrap();
        // Tamper with version 2's root after signing, then try to
        // apply it as a received record: the signature must fail.
        let mut record = docs.get_version_record(&id, 2).unwrap().clone();
        record.root = Cid::from_bytes(kernel_crypto::sha256(b"forged chunks"));
        assert!(!record.verify());
        assert_eq!(docs.apply_version(record), Err(DocError::BadSignature));
        // The stored document is untouched.
        assert_eq!(docs.get_doc(&id).unwrap().0, b"newer");
        assert_eq!(docs.latest_version(&id), Some(2));
    }

    #[test]
    fn tampered_title_breaks_signature() {
        let alice = Identity::generate("alice").unwrap();
        let root = Cid::from_bytes(kernel_crypto::sha256(b"chunks"));
        let mut record =
            VersionRecord::new("doc-12345678", "Real title", 1, root, &alice).unwrap();
        assert!(record.verify());
        record.title = "Forged title".to_string();
        assert!(!record.verify());
    }

    #[test]
    fn apply_version_from_a_peer() {
        let alice = Identity::generate("alice").unwrap();
        let mut a = DocStore::new();
        let mut b = DocStore::new();
        let id = a.create_doc("Shared", b"v1", &alice).unwrap();
        a.update_doc(&id, b"v2", &alice).unwrap();
        // B receives version 1, then version 2, as encoded records.
        let v1 = a.get_version_record(&id, 1).unwrap().clone();
        let v2 = a.get_version_record(&id, 2).unwrap().clone();
        let v2_root = v2.root;
        assert!(b.apply_version(v1).unwrap());
        assert_eq!(b.latest_version(&id), Some(1));
        assert!(b.apply_version(v2).unwrap());
        // The records alone are not enough to read: B also needs the
        // content blocks (in the real network these arrive by
        // replication/fetch by CID). Transfer them explicitly.
        let blocks = a.export_blocks();
        b.import_blocks(&blocks);
        assert_eq!(b.get_doc(&id).unwrap().0, b"v2");
        // Re-applying the same versions is a no-op, not an error.
        assert!(!b
            .apply_version(a.get_version_record(&id, 2).unwrap().clone())
            .unwrap());
        // A version 4 (gap) is ignored, not stored.
        let v4 = VersionRecord::new(&id, "Shared", 4, v2_root, &alice).unwrap();
        assert!(!b.apply_version(v4).unwrap());
        assert_eq!(b.latest_version(&id), Some(2));
    }

    #[test]
    fn only_the_owner_may_update() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut docs = DocStore::new();
        let id = docs.create_doc("Mine", b"v1", &alice).unwrap();
        assert_eq!(
            docs.update_doc(&id, b"hijack", &bob),
            Err(DocError::NotOwner)
        );
        assert_eq!(docs.latest_version(&id), Some(1));
        assert_eq!(docs.get_doc(&id).unwrap().0, b"v1");
    }

    #[test]
    fn list_docs_sorted_by_title() {
        let alice = Identity::generate("alice").unwrap();
        let mut docs = DocStore::new();
        assert!(docs.list_docs().is_empty());
        docs.create_doc("Zebra", b"z", &alice).unwrap();
        docs.create_doc("Apple", b"a", &alice).unwrap();
        docs.create_doc("Mango", b"m", &alice).unwrap();
        let listed = docs.list_docs();
        assert_eq!(listed.len(), 3);
        let titles: Vec<&str> = listed.iter().map(|(_, t, _)| t.as_str()).collect();
        assert_eq!(titles, vec!["Apple", "Mango", "Zebra"]);
        assert!(listed.iter().all(|(_, _, v)| *v == 1));
    }

    #[test]
    fn unknown_doc_and_version_errors() {
        let alice = Identity::generate("alice").unwrap();
        let mut docs = DocStore::new();
        assert_eq!(
            docs.get_doc("nope-12345678"),
            Err(DocError::NotFound("nope-12345678".to_string()))
        );
        assert_eq!(
            docs.update_doc("nope-12345678", b"x", &alice),
            Err(DocError::NotFound("nope-12345678".to_string()))
        );
        let id = docs.create_doc("One", b"v1", &alice).unwrap();
        assert_eq!(
            docs.get_version(&id, 2),
            Err(DocError::VersionNotFound {
                id: id.clone(),
                version: 2
            })
        );
        assert_eq!(
            docs.get_version(&id, 0),
            Err(DocError::VersionNotFound { id, version: 0 })
        );
    }

    #[test]
    fn title_rules() {
        let alice = Identity::generate("alice").unwrap();
        let mut docs = DocStore::new();
        assert!(docs.create_doc("", b"x", &alice).is_err());
        assert!(docs.create_doc("a".repeat(129).as_str(), b"x", &alice).is_err());
        assert!(docs.create_doc("Q3 Report!", b"x", &alice).is_ok());
        // A title with no alphanumerics still gets a usable id.
        let id = docs.create_doc("!!!", b"x", &alice).unwrap();
        assert!(id.starts_with("doc-"));
    }
}
