//! Religious text manuscripts and artifacts pipeline for KERNEL.
//!
//! A **record** is a manuscript, a fragment, or an artifact. It carries a
//! human-readable id (e.g. `ms-codex-001`), a kind, a title, a description,
//! and a **content CID** — the actual bytes live in [`kernel_storage`]
//! (addressed by CID), so a record never holds the bytes itself.
//!
//! A record also carries a **provenance chain**: an ordered list of events
//! (holder identity, caller-supplied timestamp, location text, note), each
//! signed by the holder. The chain is verified in order on read: a bad
//! signature or an out-of-order timestamp is refused.
//!
//! A record can link **variant** copies to a parent record (e.g. a fragment
//! that is a variant of a codex). Variants never form cycles — linking a
//! variant that would close a cycle is refused.
//!
//! Pipeline stages, enforced in order:
//! `Intake -> Verified -> Catalogued -> Published`.
//! A record cannot skip a stage. Verification checks that the stored
//! content hashes to the record's CID and that every provenance signature
//! checks out; a record that fails verification moves to the terminal
//! `Failed` stage and can never be published.
//!
//! ## What this does NOT do
//!
//! No real manuscripts are bundled — the caller supplies the content and
//! the claims. Provenance is caller-supplied signed claims, not
//! independently verified history: a signed lie still passes verification.
//! This pipeline proves integrity and chain order, not truth.

use std::collections::HashMap;
use std::fmt;

use kernel_crypto::{sha256, Ed25519PublicKey, Ed25519Signature};
use kernel_identity::{Identity, PeerId};
use kernel_storage::{Cid, ContentStore};

/// Domain separator for provenance signatures.
const PROVENANCE_DOMAIN: &[u8] = b"KART-PROV1";

/// Errors returned by the artifacts pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactError {
    /// No record with this id exists.
    NotFound(String),
    /// A record with this id already exists.
    DuplicateId(String),
    /// An argument was empty or otherwise invalid.
    InvalidInput(String),
    /// The record is in the wrong stage for the requested transition.
    WrongStage {
        id: String,
        expected: &'static str,
        found: Stage,
    },
    /// The record failed verification and is terminal; it can never
    /// advance again.
    FailedVerification(String),
    /// The content for a CID is not in the store.
    ContentMissing(Cid),
    /// The stored bytes do not hash to the CID the record claims.
    ContentMismatch { cid: Cid, actual: Cid },
    /// A provenance event's signature does not check out, or the chain
    /// is out of order.
    BadProvenance(String),
    /// Linking this variant would close a cycle in the variant tree.
    Cycle { parent: String, child: String },
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArtifactError::NotFound(id) => write!(f, "no record with id {id}"),
            ArtifactError::DuplicateId(id) => write!(f, "record {id} already exists"),
            ArtifactError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            ArtifactError::WrongStage {
                id,
                expected,
                found,
            } => write!(f, "record {id} is {found}, expected {expected}",),
            ArtifactError::FailedVerification(msg) => {
                write!(f, "record failed verification and is terminal: {msg}")
            }
            ArtifactError::ContentMissing(cid) => {
                write!(f, "content for CID {cid} is not in the store")
            }
            ArtifactError::ContentMismatch { cid, actual } => write!(
                f,
                "stored bytes hash to {actual}, not to the claimed CID {cid}"
            ),
            ArtifactError::BadProvenance(msg) => write!(f, "bad provenance: {msg}"),
            ArtifactError::Cycle { parent, child } => write!(
                f,
                "linking {child} as a variant of {parent} would close a cycle"
            ),
        }
    }
}

impl std::error::Error for ArtifactError {}

/// What kind of record this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecordKind {
    /// A complete manuscript.
    Manuscript,
    /// A fragment of a manuscript — often a variant of a larger record.
    Fragment,
    /// A physical artifact (tablet, scroll case, seal, and so on).
    Artifact,
}

impl fmt::Display for RecordKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            RecordKind::Manuscript => "manuscript",
            RecordKind::Fragment => "fragment",
            RecordKind::Artifact => "artifact",
        };
        write!(f, "{s}")
    }
}

/// Where a record sits in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Taken in; content stored, not yet checked.
    Intake,
    /// Content hash matches the CID and the provenance chain checks out.
    Verified,
    /// Verified and entered into the catalogue.
    Catalogued,
    /// Catalogued and published.
    Published,
    /// Failed verification. Terminal: a failed record can never advance.
    Failed,
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Stage::Intake => "intake",
            Stage::Verified => "verified",
            Stage::Catalogued => "catalogued",
            Stage::Published => "published",
            Stage::Failed => "failed",
        };
        write!(f, "{s}")
    }
}

/// One event in a record's provenance chain: who held it, when, where,
/// and a note. Signed by the holder over the record id, so an event
/// cannot be moved to a different record.
#[derive(Debug, Clone)]
pub struct ProvenanceEvent {
    /// The holder who signed this event.
    pub holder: PeerId,
    /// Caller-supplied timestamp (seconds since the Unix epoch).
    /// Must be strictly increasing along the chain.
    pub timestamp: u64,
    /// Where the record was held (free text, e.g. a city or archive).
    pub location: String,
    /// What happened (free text).
    pub note: String,
    /// The holder's signature over the event's contents.
    pub signature: Ed25519Signature,
}

impl ProvenanceEvent {
    /// Create and sign a provenance event for `record_id`.
    pub fn new(
        holder: &Identity,
        record_id: &str,
        timestamp: u64,
        location: &str,
        note: &str,
    ) -> Self {
        let message = provenance_message(
            record_id,
            holder.peer_id().public_key(),
            timestamp,
            location,
            note,
        );
        Self {
            holder: holder.peer_id().clone(),
            timestamp,
            location: location.to_string(),
            note: note.to_string(),
            signature: holder.sign(&message),
        }
    }
}

/// The exact bytes a provenance signature covers.
fn provenance_message(
    record_id: &str,
    holder_key: &Ed25519PublicKey,
    timestamp: u64,
    location: &str,
    note: &str,
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(PROVENANCE_DOMAIN);
    out.extend_from_slice(&(record_id.len() as u32).to_le_bytes());
    out.extend_from_slice(record_id.as_bytes());
    out.extend_from_slice(holder_key.as_bytes());
    out.extend_from_slice(&timestamp.to_le_bytes());
    out.extend_from_slice(&(location.len() as u32).to_le_bytes());
    out.extend_from_slice(location.as_bytes());
    out.extend_from_slice(&(note.len() as u32).to_le_bytes());
    out.extend_from_slice(note.as_bytes());
    out
}

/// Verify a provenance chain in order: every signature must check out
/// against its holder, and timestamps must be strictly increasing.
/// An empty chain verifies trivially.
fn verify_chain(record_id: &str, chain: &[ProvenanceEvent]) -> Result<(), ArtifactError> {
    let mut prev: Option<u64> = None;
    for event in chain {
        if let Some(prev_ts) = prev {
            if event.timestamp <= prev_ts {
                return Err(ArtifactError::BadProvenance(format!(
                    "timestamp {} does not follow {}",
                    event.timestamp, prev_ts
                )));
            }
        }
        prev = Some(event.timestamp);
        let message = provenance_message(
            record_id,
            event.holder.public_key(),
            event.timestamp,
            &event.location,
            &event.note,
        );
        event
            .holder
            .verify(&message, &event.signature)
            .map_err(|_| {
                ArtifactError::BadProvenance(format!(
                    "signature does not check out for holder {}",
                    event.holder
                ))
            })?;
    }
    Ok(())
}

/// A manuscript or artifact record. The record never holds the content
/// bytes — only their CID; the bytes live in the [`ContentStore`].
#[derive(Debug, Clone)]
pub struct Record {
    /// Human-readable id, e.g. `ms-codex-001`. Unique in a catalog.
    pub id: String,
    /// Manuscript, fragment, or artifact.
    pub kind: RecordKind,
    /// Human-readable title.
    pub title: String,
    /// Human-readable description.
    pub description: String,
    /// CID of the content bytes in the store.
    pub content_cid: Cid,
    /// Ordered provenance chain, verified in order on read.
    pub provenance: Vec<ProvenanceEvent>,
    /// Parent record id, if this record is a variant of another record.
    pub parent_id: Option<String>,
    /// Pipeline stage.
    pub stage: Stage,
    /// Why verification failed, if the record is in the `Failed` stage.
    pub verification_error: Option<String>,
}

/// The pipeline: a catalogue of records plus the content store their
/// CIDs point into.
#[derive(Default)]
pub struct Catalog {
    records: HashMap<String, Record>,
    store: ContentStore,
}

impl Catalog {
    /// An empty catalog with an empty content store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of records in the catalog.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True if the catalog holds no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Read access to the content store (e.g. to check a CID is held).
    pub fn store(&self) -> &ContentStore {
        &self.store
    }

    /// Mutable access to the content store. This is how content is
    /// repaired or replaced — and exactly what `verify` catches: if the
    /// bytes no longer hash to the record's CID, verification fails and
    /// the record can never be published.
    pub fn store_mut(&mut self) -> &mut ContentStore {
        &mut self.store
    }

    /// Take content in: store the bytes, record their CID, and create
    /// the record at the `Intake` stage. The provenance chain is stored
    /// as supplied — it is checked by `verify`, not here.
    pub fn intake(
        &mut self,
        id: &str,
        kind: RecordKind,
        title: &str,
        description: &str,
        content: &[u8],
        provenance: Vec<ProvenanceEvent>,
    ) -> Result<(), ArtifactError> {
        if id.is_empty() {
            return Err(ArtifactError::InvalidInput(
                "record id must not be empty".to_string(),
            ));
        }
        if title.is_empty() {
            return Err(ArtifactError::InvalidInput(
                "title must not be empty".to_string(),
            ));
        }
        if self.records.contains_key(id) {
            return Err(ArtifactError::DuplicateId(id.to_string()));
        }
        let content_cid = self.store.put(content);
        self.records.insert(
            id.to_string(),
            Record {
                id: id.to_string(),
                kind,
                title: title.to_string(),
                description: description.to_string(),
                content_cid,
                provenance,
                parent_id: None,
                stage: Stage::Intake,
                verification_error: None,
            },
        );
        Ok(())
    }

    /// Read a record. The provenance chain is verified in order first;
    /// a broken chain is refused rather than returned.
    pub fn get(&self, id: &str) -> Result<&Record, ArtifactError> {
        let record = self
            .records
            .get(id)
            .ok_or_else(|| ArtifactError::NotFound(id.to_string()))?;
        verify_chain(&record.id, &record.provenance)?;
        Ok(record)
    }

    /// Append a provenance event to a record. Only allowed while the
    /// record is at `Intake` or `Verified` — once catalogued, the
    /// chain is closed. The event's signature is checked here, and its
    /// timestamp must be strictly later than the last event's.
    pub fn add_provenance(
        &mut self,
        id: &str,
        event: ProvenanceEvent,
    ) -> Result<(), ArtifactError> {
        let record = self
            .records
            .get_mut(id)
            .ok_or_else(|| ArtifactError::NotFound(id.to_string()))?;
        match record.stage {
            Stage::Intake | Stage::Verified => {}
            stage => {
                return Err(ArtifactError::WrongStage {
                    id: id.to_string(),
                    expected: "intake or verified",
                    found: stage,
                });
            }
        }
        if let Some(last) = record.provenance.last() {
            if event.timestamp <= last.timestamp {
                return Err(ArtifactError::BadProvenance(format!(
                    "timestamp {} does not follow {}",
                    event.timestamp, last.timestamp
                )));
            }
        }
        let message = provenance_message(
            &record.id,
            event.holder.public_key(),
            event.timestamp,
            &event.location,
            &event.note,
        );
        event
            .holder
            .verify(&message, &event.signature)
            .map_err(|_| {
                ArtifactError::BadProvenance(format!(
                    "signature does not check out for holder {}",
                    event.holder
                ))
            })?;
        record.provenance.push(event);
        Ok(())
    }

    /// Verify a record: the stored bytes must hash to the record's CID
    /// and the provenance chain must check out. Only from `Intake`.
    /// On success the record moves to `Verified`. On failure it moves
    /// to the terminal `Failed` stage and can never be published.
    pub fn verify(&mut self, id: &str) -> Result<(), ArtifactError> {
        let record = self
            .records
            .get_mut(id)
            .ok_or_else(|| ArtifactError::NotFound(id.to_string()))?;
        if record.stage == Stage::Failed {
            return Err(ArtifactError::FailedVerification(
                record
                    .verification_error
                    .clone()
                    .unwrap_or_else(|| "verification failed".to_string()),
            ));
        }
        if record.stage != Stage::Intake {
            return Err(ArtifactError::WrongStage {
                id: id.to_string(),
                expected: "intake",
                found: record.stage,
            });
        }
        let failure = match self.store.get(&record.content_cid) {
            Err(_) => Some(ArtifactError::ContentMissing(record.content_cid)),
            Ok(bytes) => {
                let actual = Cid::from_bytes(sha256(&bytes));
                if actual != record.content_cid {
                    Some(ArtifactError::ContentMismatch {
                        cid: record.content_cid,
                        actual,
                    })
                } else {
                    verify_chain(&record.id, &record.provenance).err()
                }
            }
        };
        match failure {
            None => {
                record.stage = Stage::Verified;
                Ok(())
            }
            Some(e) => {
                record.verification_error = Some(e.to_string());
                record.stage = Stage::Failed;
                Err(e)
            }
        }
    }

    /// Move a verified record into the catalogue. Only from `Verified`.
    pub fn catalog(&mut self, id: &str) -> Result<(), ArtifactError> {
        let record = self
            .records
            .get_mut(id)
            .ok_or_else(|| ArtifactError::NotFound(id.to_string()))?;
        if record.stage == Stage::Failed {
            return Err(ArtifactError::FailedVerification(
                record
                    .verification_error
                    .clone()
                    .unwrap_or_else(|| "verification failed".to_string()),
            ));
        }
        if record.stage != Stage::Verified {
            return Err(ArtifactError::WrongStage {
                id: id.to_string(),
                expected: "verified",
                found: record.stage,
            });
        }
        record.stage = Stage::Catalogued;
        Ok(())
    }

    /// Publish a catalogued record. Only from `Catalogued` — an
    /// unverified or failed record can never reach this stage.
    pub fn publish(&mut self, id: &str) -> Result<(), ArtifactError> {
        let record = self
            .records
            .get_mut(id)
            .ok_or_else(|| ArtifactError::NotFound(id.to_string()))?;
        if record.stage == Stage::Failed {
            return Err(ArtifactError::FailedVerification(
                record
                    .verification_error
                    .clone()
                    .unwrap_or_else(|| "verification failed".to_string()),
            ));
        }
        if record.stage != Stage::Catalogued {
            return Err(ArtifactError::WrongStage {
                id: id.to_string(),
                expected: "catalogued",
                found: record.stage,
            });
        }
        record.stage = Stage::Published;
        Ok(())
    }

    /// Link `child_id` as a variant of `parent_id`. Both records must
    /// exist, a record cannot be a variant of itself, and the link must
    /// not close a cycle in the variant tree. Re-linking a variant to
    /// a different parent is allowed, as long as no cycle results.
    pub fn link_variant(&mut self, parent_id: &str, child_id: &str) -> Result<(), ArtifactError> {
        if !self.records.contains_key(parent_id) {
            return Err(ArtifactError::NotFound(parent_id.to_string()));
        }
        if !self.records.contains_key(child_id) {
            return Err(ArtifactError::NotFound(child_id.to_string()));
        }
        if parent_id == child_id {
            return Err(ArtifactError::Cycle {
                parent: parent_id.to_string(),
                child: child_id.to_string(),
            });
        }
        // Walk up from the parent: if we reach the child, linking the
        // child under the parent would close a cycle.
        let mut cursor = Some(parent_id.to_string());
        while let Some(current) = cursor {
            if current == child_id {
                return Err(ArtifactError::Cycle {
                    parent: parent_id.to_string(),
                    child: child_id.to_string(),
                });
            }
            cursor = self.records.get(&current).and_then(|r| r.parent_id.clone());
        }
        self.records.get_mut(child_id).unwrap().parent_id = Some(parent_id.to_string());
        Ok(())
    }

    /// All records of a kind, ordered by id.
    pub fn search_by_kind(&self, kind: RecordKind) -> Vec<&Record> {
        let mut out: Vec<&Record> = self.records.values().filter(|r| r.kind == kind).collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    /// All records whose title contains `substring` (case-insensitive),
    /// ordered by id.
    pub fn search_title(&self, substring: &str) -> Vec<&Record> {
        let needle = substring.to_lowercase();
        let mut out: Vec<&Record> = self
            .records
            .values()
            .filter(|r| r.title.to_lowercase().contains(&needle))
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    /// All records whose provenance chain names `holder` as a holder,
    /// ordered by id.
    pub fn search_by_holder(&self, holder: &PeerId) -> Vec<&Record> {
        let mut out: Vec<&Record> = self
            .records
            .values()
            .filter(|r| r.provenance.iter().any(|e| &e.holder == holder))
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn holder(name: &str) -> Identity {
        Identity::generate(name).expect("generate identity")
    }

    fn intake_codex(catalog: &mut Catalog) {
        let scribe = holder("scribe");
        let events = vec![
            ProvenanceEvent::new(&scribe, "ms-codex-001", 1000, "Cairo", "copied"),
            ProvenanceEvent::new(&scribe, "ms-codex-001", 2000, "Alexandria", "bound"),
        ];
        catalog
            .intake(
                "ms-codex-001",
                RecordKind::Manuscript,
                "Codex of the Dawn",
                "A complete codex, 214 leaves.",
                b"codex bytes: in the beginning",
                events,
            )
            .expect("intake codex");
    }

    #[test]
    fn full_flow_intake_verify_catalog_publish() {
        let mut catalog = Catalog::new();
        intake_codex(&mut catalog);
        assert_eq!(catalog.get("ms-codex-001").unwrap().stage, Stage::Intake);
        catalog.verify("ms-codex-001").expect("verify");
        assert_eq!(catalog.get("ms-codex-001").unwrap().stage, Stage::Verified);
        catalog.catalog("ms-codex-001").expect("catalog");
        assert_eq!(
            catalog.get("ms-codex-001").unwrap().stage,
            Stage::Catalogued
        );
        catalog.publish("ms-codex-001").expect("publish");
        let record = catalog.get("ms-codex-001").expect("read published");
        assert_eq!(record.stage, Stage::Published);
        assert_eq!(record.title, "Codex of the Dawn");
        assert!(catalog.store().has(&record.content_cid));
    }

    #[test]
    fn content_mismatch_is_refused_at_verify() {
        let mut catalog = Catalog::new();
        intake_codex(&mut catalog);
        let cid = catalog.get("ms-codex-001").unwrap().content_cid;
        // Swap the stored bytes for forged ones under the same CID.
        // (`corrupt` is cfg(test) on the store itself, so the public
        // recovery hook `put_unchecked` is the way in from here.)
        catalog
            .store_mut()
            .put_unchecked(&cid, b"forged text".to_vec());
        let err = catalog.verify("ms-codex-001").unwrap_err();
        assert!(
            matches!(err, ArtifactError::ContentMismatch { .. }),
            "expected ContentMismatch, got {err}"
        );
        // A failed record is terminal and can never be published.
        assert_eq!(catalog.records["ms-codex-001"].stage, Stage::Failed);
        assert!(catalog.publish("ms-codex-001").is_err());
        assert!(catalog.verify("ms-codex-001").is_err());
    }

    #[test]
    fn tampered_provenance_event_is_refused() {
        let mut catalog = Catalog::new();
        intake_codex(&mut catalog);
        // Tamper with an event after intake: change the note, breaking
        // the signature over it.
        catalog.records.get_mut("ms-codex-001").unwrap().provenance[0].note =
            "forged note".to_string();
        let err = catalog.verify("ms-codex-001").unwrap_err();
        assert!(
            matches!(err, ArtifactError::BadProvenance(_)),
            "expected BadProvenance, got {err}"
        );
        assert_eq!(catalog.records["ms-codex-001"].stage, Stage::Failed);
    }

    #[test]
    fn tampered_chain_is_refused_on_read() {
        let mut catalog = Catalog::new();
        intake_codex(&mut catalog);
        catalog.verify("ms-codex-001").expect("verify");
        // Tamper after verification: a later read must still refuse it.
        catalog.records.get_mut("ms-codex-001").unwrap().provenance[1].location =
            "Nowhere".to_string();
        let err = catalog.get("ms-codex-001").unwrap_err();
        assert!(
            matches!(err, ArtifactError::BadProvenance(_)),
            "expected BadProvenance on read, got {err}"
        );
    }

    #[test]
    fn out_of_order_provenance_is_refused() {
        let mut catalog = Catalog::new();
        let scribe = holder("scribe");
        let events = vec![
            ProvenanceEvent::new(&scribe, "ms-frag-001", 2000, "Cairo", "found"),
            ProvenanceEvent::new(&scribe, "ms-frag-001", 1000, "Alexandria", "moved"),
        ];
        catalog
            .intake(
                "ms-frag-001",
                RecordKind::Fragment,
                "Dawn Fragment",
                "A single leaf.",
                b"fragment bytes",
                events,
            )
            .expect("intake fragment");
        let err = catalog.verify("ms-frag-001").unwrap_err();
        assert!(
            matches!(err, ArtifactError::BadProvenance(_)),
            "expected BadProvenance, got {err}"
        );
    }

    #[test]
    fn equal_timestamps_are_refused() {
        let mut catalog = Catalog::new();
        let scribe = holder("scribe");
        let events = vec![
            ProvenanceEvent::new(&scribe, "ms-frag-002", 1000, "Cairo", "found"),
            ProvenanceEvent::new(&scribe, "ms-frag-002", 1000, "Cairo", "moved"),
        ];
        catalog
            .intake(
                "ms-frag-002",
                RecordKind::Fragment,
                "Twin Fragment",
                "A single leaf.",
                b"fragment bytes",
                events,
            )
            .expect("intake fragment");
        assert!(catalog.verify("ms-frag-002").is_err());
    }

    #[test]
    fn variant_links_to_parent_and_cycle_is_refused() {
        let mut catalog = Catalog::new();
        intake_codex(&mut catalog);
        catalog
            .intake(
                "ms-frag-001",
                RecordKind::Fragment,
                "Dawn Fragment",
                "A leaf from the codex.",
                b"fragment bytes",
                vec![],
            )
            .expect("intake fragment");
        catalog
            .link_variant("ms-codex-001", "ms-frag-001")
            .expect("link variant");
        assert_eq!(
            catalog.get("ms-frag-001").unwrap().parent_id,
            Some("ms-codex-001".to_string())
        );
        // Linking the parent under its own variant would close a cycle.
        let err = catalog
            .link_variant("ms-frag-001", "ms-codex-001")
            .unwrap_err();
        assert!(
            matches!(err, ArtifactError::Cycle { .. }),
            "expected Cycle, got {err}"
        );
        // A record cannot be a variant of itself.
        assert!(catalog
            .link_variant("ms-codex-001", "ms-codex-001")
            .is_err());
        // The refused links changed nothing.
        assert_eq!(catalog.get("ms-codex-001").unwrap().parent_id, None);
    }

    #[test]
    fn longer_cycle_is_refused() {
        let mut catalog = Catalog::new();
        for (id, title) in [("ms-a", "Leaf A"), ("ms-b", "Leaf B"), ("ms-c", "Leaf C")] {
            catalog
                .intake(id, RecordKind::Fragment, title, "", b"bytes", vec![])
                .expect("intake");
        }
        catalog.link_variant("ms-a", "ms-b").expect("a->b");
        catalog.link_variant("ms-b", "ms-c").expect("b->c");
        // c -> a would close a -> b -> c -> a.
        assert!(catalog.link_variant("ms-c", "ms-a").is_err());
    }

    #[test]
    fn records_cannot_skip_stages() {
        let mut catalog = Catalog::new();
        intake_codex(&mut catalog);
        // Intake -> catalogued skips verify.
        assert!(catalog.catalog("ms-codex-001").is_err());
        // Intake -> published skips verify and catalog.
        assert!(catalog.publish("ms-codex-001").is_err());
        catalog.verify("ms-codex-001").expect("verify");
        // Verified -> published skips catalog.
        assert!(catalog.publish("ms-codex-001").is_err());
        // Verify twice is also refused.
        assert!(catalog.verify("ms-codex-001").is_err());
    }

    #[test]
    fn publishing_an_unverified_record_is_refused() {
        let mut catalog = Catalog::new();
        intake_codex(&mut catalog);
        assert!(catalog.publish("ms-codex-001").is_err());
        assert_eq!(catalog.records["ms-codex-001"].stage, Stage::Intake);
    }

    #[test]
    fn search_by_kind_title_and_holder() {
        let mut catalog = Catalog::new();
        let scribe = holder("scribe");
        let keeper = holder("keeper");
        catalog
            .intake(
                "ms-codex-001",
                RecordKind::Manuscript,
                "Codex of the Dawn",
                "complete",
                b"codex",
                vec![ProvenanceEvent::new(
                    &scribe,
                    "ms-codex-001",
                    1000,
                    "Cairo",
                    "copied",
                )],
            )
            .expect("intake codex");
        catalog
            .intake(
                "ms-frag-001",
                RecordKind::Fragment,
                "Dawn Fragment",
                "a leaf",
                b"fragment",
                vec![ProvenanceEvent::new(
                    &scribe,
                    "ms-frag-001",
                    1000,
                    "Cairo",
                    "found",
                )],
            )
            .expect("intake fragment");
        catalog
            .intake(
                "art-seal-001",
                RecordKind::Artifact,
                "Clay Seal",
                "a seal",
                b"seal",
                vec![ProvenanceEvent::new(
                    &keeper,
                    "art-seal-001",
                    1000,
                    "Ur",
                    "excavated",
                )],
            )
            .expect("intake seal");

        let manuscripts = catalog.search_by_kind(RecordKind::Manuscript);
        assert_eq!(manuscripts.len(), 1);
        assert_eq!(manuscripts[0].id, "ms-codex-001");

        let fragments = catalog.search_by_kind(RecordKind::Fragment);
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].id, "ms-frag-001");

        // Title substring, case-insensitive.
        let dawn = catalog.search_title("dawn");
        let dawn_ids: Vec<&str> = dawn.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(dawn_ids, vec!["ms-codex-001", "ms-frag-001"]);
        let seal = catalog.search_title("SEAL");
        assert_eq!(seal.len(), 1);
        assert_eq!(seal[0].id, "art-seal-001");
        assert!(catalog.search_title("no such title").is_empty());

        // By holder: scribe held the codex and the fragment.
        let scribe_records = catalog.search_by_holder(scribe.peer_id());
        let scribe_ids: Vec<&str> = scribe_records.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(scribe_ids, vec!["ms-codex-001", "ms-frag-001"]);
        let keeper_records = catalog.search_by_holder(keeper.peer_id());
        assert_eq!(keeper_records.len(), 1);
        assert_eq!(keeper_records[0].id, "art-seal-001");
    }

    #[test]
    fn duplicate_id_is_refused() {
        let mut catalog = Catalog::new();
        intake_codex(&mut catalog);
        let err = catalog
            .intake(
                "ms-codex-001",
                RecordKind::Manuscript,
                "Another Codex",
                "",
                b"other bytes",
                vec![],
            )
            .unwrap_err();
        assert!(
            matches!(err, ArtifactError::DuplicateId(_)),
            "expected DuplicateId, got {err}"
        );
    }

    #[test]
    fn empty_id_or_title_is_refused() {
        let mut catalog = Catalog::new();
        assert!(catalog
            .intake("", RecordKind::Manuscript, "Title", "", b"x", vec![])
            .is_err());
        assert!(catalog
            .intake("ms-x", RecordKind::Manuscript, "", "", b"x", vec![])
            .is_err());
    }

    #[test]
    fn add_provenance_enforces_order_and_signature() {
        let mut catalog = Catalog::new();
        let scribe = holder("scribe");
        let other = holder("other");
        catalog
            .intake(
                "ms-codex-001",
                RecordKind::Manuscript,
                "Codex of the Dawn",
                "",
                b"codex",
                vec![ProvenanceEvent::new(
                    &scribe,
                    "ms-codex-001",
                    1000,
                    "Cairo",
                    "copied",
                )],
            )
            .expect("intake");
        // A later event appends fine.
        catalog
            .add_provenance(
                "ms-codex-001",
                ProvenanceEvent::new(&scribe, "ms-codex-001", 2000, "Alexandria", "bound"),
            )
            .expect("append event");
        // An earlier timestamp is refused.
        assert!(catalog
            .add_provenance(
                "ms-codex-001",
                ProvenanceEvent::new(&scribe, "ms-codex-001", 1500, "Cairo", "late note"),
            )
            .is_err());
        // An event signed for a different record is refused: its
        // signature does not cover this record's id.
        let foreign = ProvenanceEvent::new(&other, "ms-other", 3000, "Ur", "held");
        assert!(catalog.add_provenance("ms-codex-001", foreign).is_err());
        // The chain still verifies end to end.
        catalog.verify("ms-codex-001").expect("verify");
    }

    #[test]
    fn provenance_is_closed_once_catalogued() {
        let mut catalog = Catalog::new();
        intake_codex(&mut catalog);
        catalog.verify("ms-codex-001").expect("verify");
        catalog.catalog("ms-codex-001").expect("catalog");
        let scribe = holder("scribe");
        assert!(catalog
            .add_provenance(
                "ms-codex-001",
                ProvenanceEvent::new(&scribe, "ms-codex-001", 3000, "Cairo", "late"),
            )
            .is_err());
    }

    #[test]
    fn missing_content_is_refused_at_verify() {
        let mut catalog = Catalog::new();
        intake_codex(&mut catalog);
        let cid = catalog.get("ms-codex-001").unwrap().content_cid;
        catalog.store_mut().remove(&cid);
        let err = catalog.verify("ms-codex-001").unwrap_err();
        assert!(
            matches!(err, ArtifactError::ContentMissing(_)),
            "expected ContentMissing, got {err}"
        );
    }
}
