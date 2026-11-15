//! Offline-first sync for KERNEL.
//!
//! Two nodes that have been offline diverge: each edits its own copy of
//! a record. When they meet again, they exchange records and converge —
//! both end up with the same newest version of every record, without a
//! server and without asking a human to resolve anything.
//!
//! A record is a human-named entry (`"notes/todo"`, `"contacts/tristan"`)
//! with a version counter, a payload, an owner, and the owner's
//! signature over all of it. The conflict rule is simple and
//! deterministic, so every node that sees the same records reaches the
//! same result:
//!
//! * Highest version wins.
//! * On equal versions, the record whose signature is lexicographically
//!   greater wins (a deterministic tie-break, not a coin flip).
//! * A record whose signature does not verify is rejected outright —
//!   it never overwrites anything.
//!
//! Sync is idempotent: syncing two stores that already agree changes
//! nothing, so it is safe to sync as often as the nodes meet.

use std::collections::HashMap;
use std::fmt;

use kernel_crypto::Ed25519Signature;
use kernel_identity::{Identity, PeerId};

/// Domain separator for record signatures: a signature covers
/// `b"KSYNC1" || id || version (u64 LE) || payload`.
const RECORD_SIGN_DOMAIN: &[u8] = b"KSYNC1";

/// Errors returned by sync operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncError {
    /// A record's signature did not verify against its owner.
    InvalidSignature,
    /// A record id broke the naming rules.
    InvalidId(String),
}

impl fmt::Display for SyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SyncError::InvalidSignature => write!(f, "record signature is invalid"),
            SyncError::InvalidId(msg) => write!(f, "invalid record id: {msg}"),
        }
    }
}

impl std::error::Error for SyncError {}

/// Rules for record ids: 1-128 characters, printable ASCII without
/// control characters. Ids are human names (`"notes/todo"`), not hashes.
fn validate_id(id: &str) -> Result<(), SyncError> {
    if id.is_empty() {
        return Err(SyncError::InvalidId("record id must not be empty".to_string()));
    }
    if id.len() > 128 {
        return Err(SyncError::InvalidId(
            "record id must be at most 128 characters".to_string(),
        ));
    }
    if !id.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
        return Err(SyncError::InvalidId(
            "record id must be printable ASCII".to_string(),
        ));
    }
    Ok(())
}

/// The message an owner signs for a record: domain + id + version + payload.
pub fn record_sign_message(id: &str, version: u64, payload: &[u8]) -> Vec<u8> {
    let mut msg = Vec::with_capacity(RECORD_SIGN_DOMAIN.len() + id.len() + 8 + payload.len());
    msg.extend_from_slice(RECORD_SIGN_DOMAIN);
    msg.extend_from_slice(id.as_bytes());
    msg.extend_from_slice(&version.to_le_bytes());
    msg.extend_from_slice(payload);
    msg
}

/// A versioned, signed record. The newest version wins; the signature
/// proves the owner wrote this version.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    /// Human-readable id, e.g. `"notes/todo"`.
    pub id: String,
    /// Version counter. Increases on every edit; highest wins.
    pub version: u64,
    /// The record's content bytes.
    pub payload: Vec<u8>,
    /// Who wrote this version.
    pub owner: PeerId,
    /// The owner's signature over id, version, and payload.
    pub signature: Ed25519Signature,
}

impl Record {
    /// Create and sign a new record version as `identity`.
    pub fn new(
        id: &str,
        version: u64,
        payload: &[u8],
        identity: &Identity,
    ) -> Result<Self, SyncError> {
        validate_id(id)?;
        let msg = record_sign_message(id, version, payload);
        let signature = identity.sign(&msg);
        Ok(Self {
            id: id.to_string(),
            version,
            payload: payload.to_vec(),
            owner: identity.peer_id().clone(),
            signature,
        })
    }

    /// True if the signature verifies against the owner for this exact
    /// id, version, and payload. Any tampering fails this check.
    pub fn verify(&self) -> bool {
        let msg = record_sign_message(&self.id, self.version, &self.payload);
        self.owner.verify(&msg, &self.signature).is_ok()
    }

    /// Does `incoming` beat `current` under the conflict rule?
    /// Higher version wins; on a tie, the greater signature wins.
    fn beats(incoming: &Record, current: &Record) -> bool {
        if incoming.version != current.version {
            return incoming.version > current.version;
        }
        incoming.signature.to_bytes() > current.signature.to_bytes()
    }
}

/// A store of records for one node. Holds at most one version of each
/// id — the winning one under the conflict rule.
#[derive(Default)]
pub struct SyncStore {
    records: HashMap<String, Record>,
}

impl SyncStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of records held.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True if no records are held.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The winning record for `id`, if any.
    pub fn get(&self, id: &str) -> Option<&Record> {
        self.records.get(id)
    }

    /// All records held, in no particular order.
    pub fn records(&self) -> Vec<Record> {
        self.records.values().cloned().collect()
    }

    /// Apply one record. Returns `Ok(true)` if it was accepted (new
    /// record, newer version, or tie-break winner), `Ok(false)` if it
    /// lost to the record already held, and `Err` if its signature does
    /// not verify — a bad record never overwrites anything.
    pub fn put(&mut self, record: Record) -> Result<bool, SyncError> {
        if !record.verify() {
            return Err(SyncError::InvalidSignature);
        }
        match self.records.get(&record.id) {
            None => {
                self.records.insert(record.id.clone(), record);
                Ok(true)
            }
            Some(current) => {
                if Record::beats(&record, current) {
                    self.records.insert(record.id.clone(), record);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }
    }

    /// Apply many records. Returns `(accepted, rejected)`: how many were
    /// accepted (new or winning) and how many lost to a record already
    /// held. A record with a bad signature counts as rejected and is
    /// never stored.
    pub fn merge(&mut self, records: &[Record]) -> (usize, usize) {
        let mut accepted = 0;
        let mut rejected = 0;
        for record in records {
            match self.put(record.clone()) {
                Ok(true) => accepted += 1,
                Ok(false) => rejected += 1,
                Err(_) => rejected += 1,
            }
        }
        (accepted, rejected)
    }
}

/// Sync two stores: exchange records both ways until they agree.
/// Returns `(changed_in_a, changed_in_b)`: how many records each store
/// accepted during the sync. A second sync of already-agreeing stores
/// returns `(0, 0)` — sync is idempotent.
pub fn sync(a: &mut SyncStore, b: &mut SyncStore) -> (usize, usize) {
    let a_records = a.records();
    let (changed_b, _) = b.merge(&a_records);
    let b_records = b.records();
    let (changed_a, _) = a.merge(&b_records);
    (changed_a, changed_b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_sign_and_verify() {
        let alice = Identity::generate("alice").unwrap();
        let record = Record::new("notes/todo", 1, b"buy milk", &alice).unwrap();
        assert!(record.verify());
        assert_eq!(record.owner, *alice.peer_id());
        // Tampering with the payload breaks the signature.
        let mut tampered = record.clone();
        tampered.payload = b"buy diamonds".to_vec();
        assert!(!tampered.verify());
        // Tampering with the version breaks it too.
        let mut tampered = record.clone();
        tampered.version = 99;
        assert!(!tampered.verify());
    }

    #[test]
    fn record_id_rules() {
        let alice = Identity::generate("alice").unwrap();
        assert!(Record::new("", 1, b"x", &alice).is_err());
        assert!(Record::new("notes/todo", 1, b"x", &alice).is_ok());
        assert!(Record::new("a".repeat(129).as_str(), 1, b"x", &alice).is_err());
    }

    #[test]
    fn divergent_records_converge_after_sync() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut a = SyncStore::new();
        let mut b = SyncStore::new();
        // Alice has v1, Bob has v2 (newer, different payload).
        a.put(Record::new("notes/todo", 1, b"buy milk", &alice).unwrap())
            .unwrap();
        b.put(Record::new("notes/todo", 2, b"buy milk and eggs", &bob).unwrap())
            .unwrap();
        let (changed_a, changed_b) = sync(&mut a, &mut b);
        // A accepted Bob's v2; B already had v2.
        assert_eq!(changed_a, 1);
        assert_eq!(changed_b, 0);
        // Both now hold v2 with Bob's payload.
        assert_eq!(a.get("notes/todo").unwrap().version, 2);
        assert_eq!(b.get("notes/todo").unwrap().version, 2);
        assert_eq!(a.get("notes/todo").unwrap().payload, b"buy milk and eggs");
        assert_eq!(a.get("notes/todo"), b.get("notes/todo"));
    }

    #[test]
    fn older_version_never_overwrites_newer() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut a = SyncStore::new();
        let mut b = SyncStore::new();
        a.put(Record::new("doc", 3, b"newest", &alice).unwrap())
            .unwrap();
        b.put(Record::new("doc", 1, b"old", &bob).unwrap())
            .unwrap();
        let (changed_a, changed_b) = sync(&mut a, &mut b);
        assert_eq!(changed_a, 0);
        assert_eq!(changed_b, 1);
        assert_eq!(a.get("doc").unwrap().payload, b"newest");
        assert_eq!(b.get("doc").unwrap().payload, b"newest");
        assert_eq!(b.get("doc").unwrap().version, 3);
        // Direct put of an older version loses too.
        assert_eq!(
            a.put(Record::new("doc", 2, b"older", &bob).unwrap()).unwrap(),
            false
        );
        assert_eq!(a.get("doc").unwrap().payload, b"newest");
    }

    #[test]
    fn sync_is_idempotent() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut a = SyncStore::new();
        let mut b = SyncStore::new();
        a.put(Record::new("x", 1, b"one", &alice).unwrap()).unwrap();
        b.put(Record::new("y", 1, b"two", &bob).unwrap()).unwrap();
        let first = sync(&mut a, &mut b);
        assert_eq!(first, (1, 1));
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
        // Syncing again changes nothing.
        let second = sync(&mut a, &mut b);
        assert_eq!(second, (0, 0));
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
        let third = sync(&mut a, &mut b);
        assert_eq!(third, (0, 0));
    }

    #[test]
    fn equal_version_tie_break_is_deterministic() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        // Same id, same version, different payloads and owners.
        let ra = Record::new("doc", 1, b"alice version", &alice).unwrap();
        let rb = Record::new("doc", 1, b"bob version", &bob).unwrap();
        // The winner is deterministic: greater signature bytes win.
        let winner_is_a = ra.signature.to_bytes() > rb.signature.to_bytes();

        let mut a = SyncStore::new();
        let mut b = SyncStore::new();
        a.put(ra.clone()).unwrap();
        b.put(rb.clone()).unwrap();
        sync(&mut a, &mut b);
        // Both converged to the same record.
        assert_eq!(a.get("doc"), b.get("doc"));
        let winner = a.get("doc").unwrap();
        if winner_is_a {
            assert_eq!(winner.payload, b"alice version");
        } else {
            assert_eq!(winner.payload, b"bob version");
        }
        // And syncing again does not flip it.
        sync(&mut a, &mut b);
        assert_eq!(a.get("doc"), b.get("doc"));
    }

    #[test]
    fn bad_signature_is_rejected_and_stores_nothing() {
        let alice = Identity::generate("alice").unwrap();
        let mut store = SyncStore::new();
        let mut record = Record::new("doc", 1, b"real", &alice).unwrap();
        // Tamper the payload after signing.
        record.payload = b"forged".to_vec();
        assert_eq!(store.put(record), Err(SyncError::InvalidSignature));
        assert!(store.is_empty());
        // A bad record in a merge is rejected, not stored.
        let mut bad = Record::new("doc", 1, b"real", &alice).unwrap();
        bad.version = 5; // version changed after signing
        let (accepted, rejected) = store.merge(&[bad]);
        assert_eq!((accepted, rejected), (0, 1));
        assert!(store.is_empty());
    }

    #[test]
    fn new_records_propagate_both_ways() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut a = SyncStore::new();
        let mut b = SyncStore::new();
        a.put(Record::new("a-only", 1, b"from a", &alice).unwrap())
            .unwrap();
        b.put(Record::new("b-only", 1, b"from b", &bob).unwrap())
            .unwrap();
        sync(&mut a, &mut b);
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
        assert_eq!(a.get("b-only").unwrap().payload, b"from b");
        assert_eq!(b.get("a-only").unwrap().payload, b"from a");
    }

    #[test]
    fn merge_counts_accepted_and_rejected() {
        let alice = Identity::generate("alice").unwrap();
        let mut store = SyncStore::new();
        store
            .put(Record::new("x", 2, b"v2", &alice).unwrap())
            .unwrap();
        let records = vec![
            Record::new("x", 1, b"v1", &alice).unwrap(), // older: rejected
            Record::new("y", 1, b"new", &alice).unwrap(), // new: accepted
            Record::new("x", 3, b"v3", &alice).unwrap(), // newer: accepted
        ];
        assert_eq!(store.merge(&records), (2, 1));
        assert_eq!(store.get("x").unwrap().payload, b"v3");
        assert_eq!(store.get("y").unwrap().payload, b"new");
    }
}
