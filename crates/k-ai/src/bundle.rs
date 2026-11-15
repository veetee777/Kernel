//! Creator bundles: signed manifests for a bundle of content.
//!
//! A **bundle** is a creator's package: a human-readable id, a title,
//! a description, a price in X•Koin, and a list of items. Each item
//! is a name, a kind (like `track`, `video`, `art`, `text`), and the
//! content CID of the bytes in kernel-storage. The manifest itself is
//! signed by the creator and content-addressed: its CID is the
//! SHA-256 of its encoded bytes, so the bundle's address is the
//! bundle.
//!
//! The creator supplies the content. This module assembles, signs,
//! verifies, and decodes manifests — it does not create the content
//! inside them.
//!
//! Record format (explicit binary, no JSON):
//! ```text
//! KBNDL1 + id length (u16 LE) + id bytes + creator public key (32)
//!   + title length (u16 LE) + title bytes + description length (u32 LE)
//!   + description bytes + price (u64 LE) + item count (u32 LE)
//!   + per item: name length (u16 LE) + name bytes + kind length (u16 LE)
//!   + kind bytes + content CID (32) + signature (64)
//! ```

use std::fmt;

use kernel_crypto::{sha256, Ed25519Signature};
use kernel_identity::{Identity, IdentityError, PeerId};
use kernel_storage::Cid;

/// Magic bytes at the start of every bundle manifest.
pub const BUNDLE_MAGIC: &[u8; 6] = b"KBNDL1";
/// Domain separator for bundle signatures.
pub const BUNDLE_SIGN_DOMAIN: &[u8; 6] = b"KBNDL1";

/// Maximum length of a bundle id.
pub const MAX_BUNDLE_ID_LEN: usize = 64;
/// Maximum title length.
pub const MAX_TITLE_LEN: usize = 128;
/// Maximum description length in bytes.
pub const MAX_DESCRIPTION_LEN: usize = 10_000;
/// Maximum items in one bundle.
pub const MAX_ITEMS: usize = 256;
/// Maximum item name length.
pub const MAX_ITEM_NAME_LEN: usize = 128;
/// Maximum item kind length.
pub const MAX_KIND_LEN: usize = 32;

/// Errors returned by bundle operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BundleError {
    /// A bundle id broke the naming rules.
    InvalidId(String),
    /// A title broke the rules.
    InvalidTitle(String),
    /// A description broke the rules.
    InvalidDescription(String),
    /// An item broke the rules.
    InvalidItem(String),
    /// A manifest's bytes were malformed.
    InvalidManifest(String),
    /// A manifest's signature did not verify against its creator.
    BadSignature,
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
}

impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BundleError::InvalidId(msg) => write!(f, "invalid bundle id: {msg}"),
            BundleError::InvalidTitle(msg) => write!(f, "invalid title: {msg}"),
            BundleError::InvalidDescription(msg) => {
                write!(f, "invalid description: {msg}")
            }
            BundleError::InvalidItem(msg) => write!(f, "invalid item: {msg}"),
            BundleError::InvalidManifest(msg) => write!(f, "invalid manifest: {msg}"),
            BundleError::BadSignature => write!(f, "bundle signature is invalid"),
            BundleError::Identity(e) => write!(f, "identity error: {e}"),
        }
    }
}

impl std::error::Error for BundleError {}

impl From<IdentityError> for BundleError {
    fn from(e: IdentityError) -> Self {
        BundleError::Identity(e)
    }
}

/// Rules for bundle ids: 1-64 characters, ASCII letters, digits, `-`,
/// `_`, `.`. Ids are human labels (`"summer-tracks-001"`), not hashes.
fn validate_bundle_id(id: &str) -> Result<(), BundleError> {
    if id.is_empty() {
        return Err(BundleError::InvalidId(
            "bundle id must not be empty".to_string(),
        ));
    }
    if id.len() > MAX_BUNDLE_ID_LEN {
        return Err(BundleError::InvalidId(format!(
            "bundle id must be at most {MAX_BUNDLE_ID_LEN} characters"
        )));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Err(BundleError::InvalidId(
            "bundle id may only use letters, digits, '-', '_' and '.'".to_string(),
        ));
    }
    Ok(())
}

/// Rules for item kinds: 1-32 characters, ASCII letters, digits, `-`,
/// `_`. Common kinds are `track`, `video`, `art`, `text`; anything
/// matching the rules is accepted.
fn validate_kind(kind: &str) -> Result<(), BundleError> {
    if kind.is_empty() {
        return Err(BundleError::InvalidItem(
            "item kind must not be empty".to_string(),
        ));
    }
    if kind.len() > MAX_KIND_LEN {
        return Err(BundleError::InvalidItem(format!(
            "item kind must be at most {MAX_KIND_LEN} characters"
        )));
    }
    if !kind
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(BundleError::InvalidItem(
            "item kind may only use letters, digits, '-' and '_'".to_string(),
        ));
    }
    Ok(())
}

/// One item in a bundle: a name, a kind, and the content CID of the
/// bytes in kernel-storage. The item never holds the bytes itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleItem {
    /// Human-readable item name, e.g. `"opening-track"`.
    pub name: String,
    /// Item kind, e.g. `"track"`, `"video"`, `"art"`, `"text"`.
    pub kind: String,
    /// Content CID of the item's bytes.
    pub cid: Cid,
}

impl BundleItem {
    /// Build an item. The name must be 1-128 characters; the kind
    /// must match the kind rules; the CID is taken as given — whether
    /// the bytes exist is checked by the assembler, not here.
    pub fn new(name: &str, kind: &str, cid: Cid) -> Result<Self, BundleError> {
        if name.is_empty() {
            return Err(BundleError::InvalidItem(
                "item name must not be empty".to_string(),
            ));
        }
        if name.len() > MAX_ITEM_NAME_LEN {
            return Err(BundleError::InvalidItem(format!(
                "item name must be at most {MAX_ITEM_NAME_LEN} characters"
            )));
        }
        validate_kind(kind)?;
        Ok(Self {
            name: name.to_string(),
            kind: kind.to_string(),
            cid,
        })
    }
}

/// The message a creator signs for a bundle manifest.
fn manifest_sign_message(
    id: &str,
    creator: &PeerId,
    title: &str,
    description: &str,
    price: u64,
    items: &[BundleItem],
) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend_from_slice(BUNDLE_SIGN_DOMAIN);
    msg.extend_from_slice(id.as_bytes());
    msg.extend_from_slice(creator.public_key().as_bytes());
    msg.extend_from_slice(title.as_bytes());
    msg.extend_from_slice(description.as_bytes());
    msg.extend_from_slice(&price.to_le_bytes());
    msg.extend_from_slice(&(items.len() as u32).to_le_bytes());
    for item in items {
        msg.extend_from_slice(item.name.as_bytes());
        msg.extend_from_slice(item.kind.as_bytes());
        msg.extend_from_slice(item.cid.as_bytes());
    }
    msg
}

/// A signed creator bundle manifest: id, creator, title, description,
/// price in X•Koin, items, and the creator's signature over all of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleManifest {
    /// Human-readable bundle id, e.g. `"summer-tracks-001"`.
    pub id: String,
    /// The creator who signed this manifest.
    pub creator: PeerId,
    /// Bundle title.
    pub title: String,
    /// Bundle description.
    pub description: String,
    /// Price in X•Koin, in the coin's smallest unit. Zero is free.
    pub price: u64,
    /// The bundle's items, in the order given.
    pub items: Vec<BundleItem>,
    /// The creator's signature over id, creator, title, description,
    /// price, and items.
    pub signature: Ed25519Signature,
}

impl BundleManifest {
    /// Build and sign a manifest as `creator`. A bundle must have at
    /// least one item and at most [`MAX_ITEMS`]; the price is any
    /// `u64` (zero is free).
    pub fn new(
        id: &str,
        title: &str,
        description: &str,
        price: u64,
        items: Vec<BundleItem>,
        creator: &Identity,
    ) -> Result<Self, BundleError> {
        validate_bundle_id(id)?;
        if title.is_empty() {
            return Err(BundleError::InvalidTitle(
                "title must not be empty".to_string(),
            ));
        }
        if title.len() > MAX_TITLE_LEN {
            return Err(BundleError::InvalidTitle(format!(
                "title must be at most {MAX_TITLE_LEN} characters"
            )));
        }
        if description.len() > MAX_DESCRIPTION_LEN {
            return Err(BundleError::InvalidDescription(format!(
                "description must be at most {MAX_DESCRIPTION_LEN} bytes"
            )));
        }
        if items.is_empty() {
            return Err(BundleError::InvalidItem(
                "bundle must have at least one item".to_string(),
            ));
        }
        if items.len() > MAX_ITEMS {
            return Err(BundleError::InvalidItem(format!(
                "bundle must have at most {MAX_ITEMS} items"
            )));
        }
        let creator_id = creator.peer_id().clone();
        let msg = manifest_sign_message(id, &creator_id, title, description, price, &items);
        let signature = creator.sign(&msg);
        Ok(Self {
            id: id.to_string(),
            creator: creator_id,
            title: title.to_string(),
            description: description.to_string(),
            price,
            items,
            signature,
        })
    }

    /// True if the signature verifies against the creator for this
    /// exact id, title, description, price, and items.
    pub fn verify(&self) -> bool {
        let msg = manifest_sign_message(
            &self.id,
            &self.creator,
            &self.title,
            &self.description,
            self.price,
            &self.items,
        );
        self.creator.verify(&msg, &self.signature).is_ok()
    }

    /// The manifest's content address: the SHA-256 of its encoded
    /// bytes. The bundle's address is the bundle.
    pub fn cid(&self) -> Cid {
        Cid::from_bytes(sha256(&self.encode()))
    }

    /// Encode to the explicit binary manifest format.
    pub fn encode(&self) -> Vec<u8> {
        let id_bytes = self.id.as_bytes();
        let title_bytes = self.title.as_bytes();
        let desc_bytes = self.description.as_bytes();
        let mut out = Vec::with_capacity(
            6 + 2
                + id_bytes.len()
                + 32
                + 2
                + title_bytes.len()
                + 4
                + desc_bytes.len()
                + 8
                + 4
                + self.items.len() * (2 + 2 + 32)
                + self
                    .items
                    .iter()
                    .map(|i| i.name.len() + i.kind.len())
                    .sum::<usize>()
                + 64,
        );
        out.extend_from_slice(BUNDLE_MAGIC);
        out.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(id_bytes);
        out.extend_from_slice(self.creator.public_key().as_bytes());
        out.extend_from_slice(&(title_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(title_bytes);
        out.extend_from_slice(&(desc_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(desc_bytes);
        out.extend_from_slice(&self.price.to_le_bytes());
        out.extend_from_slice(&(self.items.len() as u32).to_le_bytes());
        for item in &self.items {
            out.extend_from_slice(&(item.name.len() as u16).to_le_bytes());
            out.extend_from_slice(item.name.as_bytes());
            out.extend_from_slice(&(item.kind.len() as u16).to_le_bytes());
            out.extend_from_slice(item.kind.as_bytes());
            out.extend_from_slice(item.cid.as_bytes());
        }
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a manifest produced by [`BundleManifest::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, BundleError> {
        let bad = |msg: &str| BundleError::InvalidManifest(msg.to_string());
        if bytes.len() < 6 + 2 + 32 + 2 + 4 + 8 + 4 + 64 || &bytes[..6] != BUNDLE_MAGIC {
            return Err(bad("missing KBNDL1 magic"));
        }
        let mut off = 6;
        let id_len =
            u16::from_le_bytes(bytes[off..off + 2].try_into().expect("2-byte slice")) as usize;
        off += 2;
        if bytes.len() < off + id_len + 32 + 2 {
            return Err(bad("manifest too short for id and creator"));
        }
        let id = String::from_utf8(bytes[off..off + id_len].to_vec())
            .map_err(|_| bad("bundle id is not valid UTF-8"))?;
        validate_bundle_id(&id)?;
        off += id_len;
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let creator_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| bad(&e.to_string()))?;
        let title_len =
            u16::from_le_bytes(bytes[off..off + 2].try_into().expect("2-byte slice")) as usize;
        off += 2;
        if bytes.len() < off + title_len + 4 {
            return Err(bad("manifest too short for title"));
        }
        let title = String::from_utf8(bytes[off..off + title_len].to_vec())
            .map_err(|_| bad("title is not valid UTF-8"))?;
        if title.is_empty() || title.len() > MAX_TITLE_LEN {
            return Err(bad("title breaks the rules"));
        }
        off += title_len;
        let desc_len =
            u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4-byte slice")) as usize;
        off += 4;
        if desc_len > MAX_DESCRIPTION_LEN || bytes.len() < off + desc_len + 8 + 4 {
            return Err(bad("manifest too short for description"));
        }
        let description = String::from_utf8(bytes[off..off + desc_len].to_vec())
            .map_err(|_| bad("description is not valid UTF-8"))?;
        off += desc_len;
        let price = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let count =
            u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4-byte slice")) as usize;
        off += 4;
        if count == 0 || count > MAX_ITEMS {
            return Err(bad("item count breaks the rules"));
        }
        let mut items = Vec::with_capacity(count);
        for _ in 0..count {
            if bytes.len() < off + 2 {
                return Err(bad("manifest too short for an item name"));
            }
            let name_len =
                u16::from_le_bytes(bytes[off..off + 2].try_into().expect("2-byte slice")) as usize;
            off += 2;
            if bytes.len() < off + name_len + 2 {
                return Err(bad("manifest too short for an item name"));
            }
            let name = String::from_utf8(bytes[off..off + name_len].to_vec())
                .map_err(|_| bad("item name is not valid UTF-8"))?;
            off += name_len;
            let kind_len =
                u16::from_le_bytes(bytes[off..off + 2].try_into().expect("2-byte slice")) as usize;
            off += 2;
            if bytes.len() < off + kind_len + 32 {
                return Err(bad("manifest too short for an item kind"));
            }
            let kind = String::from_utf8(bytes[off..off + kind_len].to_vec())
                .map_err(|_| bad("item kind is not valid UTF-8"))?;
            off += kind_len;
            let mut cid_bytes = [0u8; 32];
            cid_bytes.copy_from_slice(&bytes[off..off + 32]);
            off += 32;
            items.push(BundleItem::new(&name, &kind, Cid::from_bytes(cid_bytes))?);
        }
        if bytes.len() != off + 64 {
            return Err(bad("manifest has trailing bytes"));
        }
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let creator = PeerId::new(creator_key, &fingerprint_name(&creator_key))?;
        Ok(Self {
            id,
            creator,
            title,
            description,
            price,
            items,
            signature,
        })
    }
}

/// A display name derived from a public key: the first 8 hex chars of
/// the key. Used when a manifest is decoded and only the key survives.
fn fingerprint_name(key: &kernel_crypto::Ed25519PublicKey) -> String {
    let hex = kernel_crypto::to_hex(key.as_bytes());
    format!("peer-{}", &hex[..8])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_storage::ContentStore;

    fn test_manifest() -> BundleManifest {
        let creator = Identity::generate("creator").unwrap();
        let mut store = ContentStore::new();
        let cid1 = store.put(b"track one audio");
        let cid2 = store.put(b"cover art bytes");
        BundleManifest::new(
            "summer-tracks-001",
            "Summer Tracks",
            "Two songs and a cover.",
            500,
            vec![
                BundleItem::new("opening-track", "track", cid1).unwrap(),
                BundleItem::new("cover", "art", cid2).unwrap(),
            ],
            &creator,
        )
        .unwrap()
    }

    #[test]
    fn manifest_encode_decode_round_trip_and_verify() {
        let manifest = test_manifest();
        assert!(manifest.verify());
        let encoded = manifest.encode();
        let decoded = BundleManifest::decode(&encoded).unwrap();
        // Decode rebuilds the creator PeerId with the key-derived
        // display name (peer-xxxxxxxx), so compare keys, not names —
        // the same behavior as AiTask::decode.
        assert_eq!(decoded.encode(), encoded);
        assert_eq!(decoded.creator.public_key(), manifest.creator.public_key());
        assert_eq!(decoded.id, manifest.id);
        assert_eq!(decoded.title, manifest.title);
        assert_eq!(decoded.description, manifest.description);
        assert_eq!(decoded.price, manifest.price);
        assert_eq!(decoded.items, manifest.items);
        assert_eq!(decoded.signature, manifest.signature);
        assert!(decoded.verify());
        assert_eq!(decoded.cid(), manifest.cid());
        assert_eq!(
            manifest.cid().to_hex(),
            kernel_crypto::to_hex(&sha256(&encoded))
        );
        assert!(BundleManifest::decode(b"garbage").is_err());
        assert!(BundleManifest::decode(&encoded[..encoded.len() - 1]).is_err());
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(BundleManifest::decode(&trailing).is_err());
    }

    #[test]
    fn tampered_manifest_fails_verify() {
        let mut manifest = test_manifest();
        assert!(manifest.verify());
        manifest.price = 999;
        assert!(!manifest.verify());
        let mut manifest = test_manifest();
        manifest.title = "Forged Title".to_string();
        assert!(!manifest.verify());
        let mut manifest = test_manifest();
        manifest.items[0].name = "renamed".to_string();
        assert!(!manifest.verify());
    }

    #[test]
    fn bundle_rules_refused() {
        let creator = Identity::generate("creator").unwrap();
        let cid = ContentStore::new().put(b"x");
        let item = BundleItem::new("track", "track", cid).unwrap();
        assert!(BundleManifest::new("", "T", "D", 1, vec![item.clone()], &creator).is_err());
        assert!(
            BundleManifest::new("has space", "T", "D", 1, vec![item.clone()], &creator).is_err()
        );
        assert!(BundleManifest::new("ok-id", "", "D", 1, vec![item.clone()], &creator).is_err());
        assert!(
            BundleManifest::new("ok-id", "T", "D", 1, vec![], &creator).is_err(),
            "a bundle needs at least one item"
        );
        assert!(BundleItem::new("", "track", cid).is_err());
        assert!(BundleItem::new("name", "", cid).is_err());
        assert!(BundleItem::new("name", "has space", cid).is_err());
    }
}
