//! Durable binary persistence for the artifacts catalog.
//!
//! One append-only binary log file, `artifacts.klog`, in the store
//! directory. The format follows the `kernel-store-disk` `KBLK1`
//! pattern: every record is framed as
//!
//! ```text
//! KART1 (5 bytes) + kind (1 byte) + payload length (u32 LE)
//!       + payload bytes + SHA-256(kind || payload) (32 bytes)
//! ```
//!
//! Record kinds, in the order a save writes them:
//!
//! - `1` intake: a whole record — id, kind, title, description,
//!   content CID, content bytes, and the provenance chain as it
//!   stood at intake. The content bytes are stored inline, so the
//!   log alone rebuilds the catalog exactly, with no second file.
//!   The record is created at the `Intake` stage with no parent;
//!   later records advance it.
//! - `2` provenance: one [`ProvenanceEvent`] appended to a record
//!   (its `KPEVT1` encoding, plus the record id).
//! - `3` stage: a record's stage transition, with the verification
//!   error when the stage is `Failed`.
//! - `4` variant: a parent/child variant link.
//!
//! A [`ArtifactStore`] replays the log on open, rebuilding the
//! catalog. [`ArtifactStore::save`] appends only what is not
//! durable yet, in four phases — all new intakes first, then new
//! provenance events, then stage transitions, then variant links —
//! so a record always exists before anything cites it, and both
//! ends of a variant link exist before the link. The batch is
//! fsync'd once; a save that returns has its records on disk.
//! Replay is idempotent for the records a retried save can
//! duplicate (a repeated intake with identical fields, a repeated
//! provenance event, a repeated stage event, a repeated variant
//! link are all skipped), but a save that fails partway leaves the
//! store's persisted tracking stale: **drop the store and re-open
//! it before saving again** — reopening replays the log (cutting
//! any torn tail) and rebuilds exact tracking.
//!
//! Corruption behavior, by design:
//!
//! - A log whose **first** record is corrupt (bad magic, unknown
//!   kind, oversize length, failed hash, undecodable payload, or an
//!   intake whose id is a duplicate) is **refused** with a clean
//!   [`ArtifactError::Persistence`] error — the file is not an
//!   artifacts log at all, and silently treating it as empty would
//!   hide the damage.
//! - Corruption **later** in the log cuts the tail: replay stops at
//!   the first bad record and the file is truncated back to the
//!   last good record, exactly like `kernel-store-disk`. A crash
//!   mid-write leaves a torn final record, which is cut the same
//!   way; everything before it survives.
//! - Load re-verifies every record, in order: the stored bytes must
//!   hash to the record's CID and the provenance chain must check
//!   out, signature by signature, in timestamp order. A record
//!   that fails either check is **not** dropped and **not** treated
//!   as a corrupt log — it loads at the terminal `Failed` stage
//!   with the failure recorded, the same outcome as calling
//!   `verify` on it. That is how a record that failed verification
//!   before a restart keeps its `Failed` stage: it fails
//!   verification again on load, instead of being silently
//!   resurrected as `Intake`.
//!
//! No JSON on disk, no SQLite, no Postgres — binary records only.
//! Time is never read here: provenance timestamps are
//! caller-supplied, and replay uses none of its own.

use std::collections::HashMap;
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use kernel_crypto::sha256;
use kernel_storage::Cid;

use crate::{
    ArtifactError, Catalog, ProvenanceEvent, Record, RecordKind, Stage,
};

/// Magic bytes at the start of every log record.
const RECORD_MAGIC: &[u8; 5] = b"KART1";
/// Name of the log file inside the store directory.
const LOG_FILE_NAME: &str = "artifacts.klog";
/// Largest payload a record may carry (64 MiB). A length above this
/// is corruption, not a record — it also bounds the replay allocation.
const MAX_PAYLOAD: u32 = 64 * 1024 * 1024;
/// Record kind: intake of a whole record.
const KIND_INTAKE: u8 = 1;
/// Record kind: one provenance event appended to a record.
const KIND_PROVENANCE: u8 = 2;
/// Record kind: a stage transition.
const KIND_STAGE: u8 = 3;
/// Record kind: a variant link.
const KIND_VARIANT: u8 = 4;

/// What of one record is already durable in the log.
#[derive(Clone)]
struct Persisted {
    /// The intake record has been written.
    intake: bool,
    /// Provenance events durable, as a prefix count of the record's
    /// chain (events are append-only; the intake record carried the
    /// chain as it stood at first save).
    prov_count: usize,
    /// Stage durable in the log.
    stage: Stage,
    /// Variant parent durable in the log.
    parent: Option<String>,
}

impl Default for Persisted {
    fn default() -> Self {
        Self {
            intake: false,
            prov_count: 0,
            stage: Stage::Intake,
            parent: None,
        }
    }
}

/// A catalog with a durable append-only log behind it.
///
/// Open replays the log into a [`Catalog`]; mutate the catalog
/// through [`ArtifactStore::catalog_mut`], then
/// [`ArtifactStore::save`] to make the new records, events, stages,
/// and links durable. Only what is not already in the log is
/// appended.
pub struct ArtifactStore {
    catalog: Catalog,
    file: File,
    /// Full path of the log file.
    path: PathBuf,
    /// Persisted state per record id.
    persisted: HashMap<String, Persisted>,
}

impl ArtifactStore {
    /// Open the store in `dir`, creating the directory and the log
    /// file if they do not exist. Replays the log to rebuild the
    /// catalog; a torn or corrupt tail is cut, a corrupt first
    /// record is refused with an error.
    pub fn open(dir: &Path) -> Result<Self, ArtifactError> {
        create_dir_all(dir)?;
        let path = dir.join(LOG_FILE_NAME);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        let (catalog, persisted) = replay(&mut file)?;
        Ok(Self {
            catalog,
            file,
            path,
            persisted,
        })
    }

    /// Load the catalog stored in `dir`, without keeping the store
    /// open. This is [`ArtifactStore::open`] followed by
    /// [`ArtifactStore::into_catalog`].
    pub fn load(dir: &Path) -> Result<Catalog, ArtifactError> {
        Ok(Self::open(dir)?.into_catalog())
    }

    /// The catalog, rebuilt from the log.
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// The catalog, mutably. Intake records, append provenance,
    /// verify, catalog, publish, and link variants here, then call
    /// [`ArtifactStore::save`] to make them durable.
    pub fn catalog_mut(&mut self) -> &mut Catalog {
        &mut self.catalog
    }

    /// Path of the log file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Take the catalog out of the store, dropping the log handle.
    pub fn into_catalog(self) -> Catalog {
        self.catalog
    }

    /// Append everything that is not durable yet, in four phases
    /// (intakes, provenance events, stage transitions, variant
    /// links), then fsync. A save that returns has its records on
    /// disk. A save with nothing new to write touches nothing and
    /// returns immediately. If a save fails partway, the store's
    /// persisted tracking is stale: drop the store and re-open it
    /// before saving again.
    pub fn save(&mut self) -> Result<(), ArtifactError> {
        let mut buf: Vec<u8> = Vec::new();
        // Working copy of the persisted tracking. Phases read and
        // update this as they build the batch; it replaces
        // `self.persisted` only after the batch is durable. A failed
        // write discards it, so the next save retries the same
        // records — and replay skips duplicates.
        let mut next_states: HashMap<String, Persisted> = self.persisted.clone();
        let ids = self.catalog.record_ids();

        // Phase 1: intakes for records never written. The intake
        // record carries the record's content bytes and its chain
        // as it stands now.
        for id in &ids {
            let record = self.catalog.record_raw(id).unwrap();
            let state = next_states.entry(id.clone()).or_default();
            if state.intake {
                continue;
            }
            let content = self
                .catalog
                .store()
                .get(&record.content_cid)
                .map_err(|_| ArtifactError::ContentMissing(record.content_cid))?;
            buf.extend_from_slice(&frame_record(
                KIND_INTAKE,
                &encode_intake_payload(record, &content),
            ));
            state.intake = true;
            state.prov_count = record.provenance.len();
            state.stage = Stage::Intake;
            state.parent = None;
        }
        // Phase 2: provenance events appended since the last save.
        for id in &ids {
            let record = self.catalog.record_raw(id).unwrap();
            let state = next_states.entry(id.clone()).or_default();
            if !state.intake {
                continue;
            }
            let from = state.prov_count;
            if record.provenance.len() > from {
                for event in &record.provenance[from..] {
                    buf.extend_from_slice(&frame_record(
                        KIND_PROVENANCE,
                        &encode_provenance_payload(id, event),
                    ));
                }
                state.prov_count = record.provenance.len();
            }
        }
        // Phase 3: stage transitions, as the full path from the
        // durable stage to the current stage (a record can advance
        // several stages between saves).
        for id in &ids {
            let record = self.catalog.record_raw(id).unwrap();
            let state = next_states.entry(id.clone()).or_default();
            if !state.intake {
                continue;
            }
            let path = stage_path(state.stage, record.stage).ok_or_else(|| {
                ArtifactError::Persistence(format!(
                    "no stage path from {} to {} for record {id}",
                    state.stage, record.stage
                ))
            })?;
            if path.is_empty() {
                continue;
            }
            // A Failed stage carries the verification error; other
            // stages carry none.
            for stage in &path {
                let error = if *stage == Stage::Failed {
                    record.verification_error.clone()
                } else {
                    None
                };
                buf.extend_from_slice(&frame_record(
                    KIND_STAGE,
                    &encode_stage_payload(id, *stage, error.as_deref()),
                ));
            }
            state.stage = record.stage;
        }
        // Phase 4: variant links. Both ends exist by now: every
        // intake in the log precedes every link.
        for id in &ids {
            let record = self.catalog.record_raw(id).unwrap();
            let state = next_states.entry(id.clone()).or_default();
            if !state.intake {
                continue;
            }
            if state.parent == record.parent_id {
                continue;
            }
            let Some(parent) = &record.parent_id else {
                continue;
            };
            buf.extend_from_slice(&frame_record(
                KIND_VARIANT,
                &encode_variant_payload(parent, id),
            ));
            state.parent = Some(parent.clone());
        }

        if buf.is_empty() {
            return Ok(());
        }
        self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(&buf)?;
        self.file.sync_all()?;
        // Only now, with the batch durable, is the state "persisted".
        self.persisted = next_states;
        Ok(())
    }
}

/// The stages a record passes through to get from `from` to `to`,
/// in order, not including `from`. `Failed` is reachable only from
/// `Intake` (verification only runs there); the linear order is
/// `Intake -> Verified -> Catalogued -> Published`. Returns `None`
/// for a transition the public pipeline cannot produce.
fn stage_path(from: Stage, to: Stage) -> Option<Vec<Stage>> {
    if from == to {
        return Some(Vec::new());
    }
    if to == Stage::Failed {
        if from == Stage::Intake {
            return Some(vec![Stage::Failed]);
        }
        return None;
    }
    if from == Stage::Failed {
        return None;
    }
    const ORDER: [Stage; 4] = [
        Stage::Intake,
        Stage::Verified,
        Stage::Catalogued,
        Stage::Published,
    ];
    let fi = ORDER.iter().position(|s| *s == from)?;
    let ti = ORDER.iter().position(|s| *s == to)?;
    if fi < ti {
        Some(ORDER[fi + 1..=ti].to_vec())
    } else {
        None
    }
}

/// Frame one record: magic + kind + length + payload + hash.
fn frame_record(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(10 + payload.len() + 32);
    out.extend_from_slice(RECORD_MAGIC);
    out.push(kind);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    let mut hashed = vec![kind];
    hashed.extend_from_slice(payload);
    out.extend_from_slice(&sha256(&hashed));
    out
}

/// Encode an intake payload: id, kind, title, description, content
/// CID, content bytes, and the provenance chain.
fn encode_intake_payload(record: &Record, content: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    push_len_bytes(&mut out, record.id.as_bytes());
    out.push(kind_byte(record.kind));
    push_len_bytes(&mut out, record.title.as_bytes());
    push_len_bytes(&mut out, record.description.as_bytes());
    out.extend_from_slice(record.content_cid.as_bytes());
    push_len_bytes(&mut out, content);
    out.extend_from_slice(&(record.provenance.len() as u32).to_le_bytes());
    for event in &record.provenance {
        let encoded = event.encode();
        push_len_bytes(&mut out, &encoded);
    }
    out
}

/// Encode a provenance payload: record id + one event.
fn encode_provenance_payload(id: &str, event: &ProvenanceEvent) -> Vec<u8> {
    let mut out = Vec::new();
    push_len_bytes(&mut out, id.as_bytes());
    let encoded = event.encode();
    push_len_bytes(&mut out, &encoded);
    out
}

/// Encode a stage payload: record id + stage + optional error.
fn encode_stage_payload(id: &str, stage: Stage, error: Option<&str>) -> Vec<u8> {
    let mut out = Vec::new();
    push_len_bytes(&mut out, id.as_bytes());
    out.push(stage_byte(stage));
    match error {
        Some(error) => {
            out.push(1);
            push_len_bytes(&mut out, error.as_bytes());
        }
        None => out.push(0),
    }
    out
}

/// Encode a variant payload: parent id + child id.
fn encode_variant_payload(parent: &str, child: &str) -> Vec<u8> {
    let mut out = Vec::new();
    push_len_bytes(&mut out, parent.as_bytes());
    push_len_bytes(&mut out, child.as_bytes());
    out
}

fn kind_byte(kind: RecordKind) -> u8 {
    match kind {
        RecordKind::Manuscript => 1,
        RecordKind::Fragment => 2,
        RecordKind::Artifact => 3,
    }
}

fn stage_byte(stage: Stage) -> u8 {
    match stage {
        Stage::Intake => 1,
        Stage::Verified => 2,
        Stage::Catalogued => 3,
        Stage::Published => 4,
        Stage::Failed => 5,
    }
}

fn kind_from_byte(byte: u8) -> Result<RecordKind, ArtifactError> {
    match byte {
        1 => Ok(RecordKind::Manuscript),
        2 => Ok(RecordKind::Fragment),
        3 => Ok(RecordKind::Artifact),
        _ => Err(ArtifactError::Persistence(format!(
            "unknown record kind {byte}"
        ))),
    }
}

fn stage_from_byte(byte: u8) -> Result<Stage, ArtifactError> {
    match byte {
        1 => Ok(Stage::Intake),
        2 => Ok(Stage::Verified),
        3 => Ok(Stage::Catalogued),
        4 => Ok(Stage::Published),
        5 => Ok(Stage::Failed),
        _ => Err(ArtifactError::Persistence(format!(
            "unknown stage {byte}"
        ))),
    }
}

/// Append a u32-LE length prefix plus the bytes.
fn push_len_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// Replay the log, rebuilding the catalog and the persisted-state
/// tracking. Stops at the first truncated or corrupt record and
/// truncates the file to the last good record end, so the log is
/// always appendable. A corrupt *first* record is an error, not a
/// cut: the file is not an artifacts log at all.
fn replay(file: &mut File) -> Result<(Catalog, HashMap<String, Persisted>), ArtifactError> {
    let mut catalog = Catalog::new();
    let mut persisted: HashMap<String, Persisted> = HashMap::new();
    let mut reader = BufReader::new(file.try_clone()?);
    let mut offset: u64 = 0;
    loop {
        let record_start = offset;
        let first = record_start == 0;
        let mut magic = [0u8; 5];
        match reader.read_exact(&mut magic) {
            Ok(()) => {}
            // A short read at a record boundary is a torn write:
            // stop, keep everything before it.
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        if &magic != RECORD_MAGIC {
            if first {
                return Err(ArtifactError::Persistence(
                    "not an artifacts log (bad magic)".to_string(),
                ));
            }
            break;
        }
        let mut kind_buf = [0u8; 1];
        let mut len_buf = [0u8; 4];
        if reader.read_exact(&mut kind_buf).is_err()
            || reader.read_exact(&mut len_buf).is_err()
        {
            break;
        }
        let kind = kind_buf[0];
        if kind != KIND_INTAKE
            && kind != KIND_PROVENANCE
            && kind != KIND_STAGE
            && kind != KIND_VARIANT
        {
            if first {
                return Err(ArtifactError::Persistence(format!(
                    "unknown record kind {kind} in artifacts log"
                )));
            }
            break;
        }
        let len = u32::from_le_bytes(len_buf);
        if len > MAX_PAYLOAD {
            if first {
                return Err(ArtifactError::Persistence(
                    "record length exceeds the 64 MiB limit".to_string(),
                ));
            }
            break;
        }
        let mut payload = vec![0u8; len as usize];
        if reader.read_exact(&mut payload).is_err() {
            break;
        }
        let mut hash = [0u8; 32];
        if reader.read_exact(&mut hash).is_err() {
            break;
        }
        let mut hashed = vec![kind];
        hashed.extend_from_slice(&payload);
        if sha256(&hashed) != hash {
            if first {
                return Err(ArtifactError::Persistence(
                    "first record failed hash verification".to_string(),
                ));
            }
            break;
        }
        // The framing is sound; now decode and apply. A failure
        // here is a corrupt record, handled like a bad hash.
        let applied = match kind {
            KIND_INTAKE => apply_intake(&mut catalog, &persisted, &payload),
            KIND_PROVENANCE => apply_provenance(&mut catalog, &persisted, &payload),
            KIND_STAGE => apply_stage(&mut catalog, &persisted, &payload),
            KIND_VARIANT => apply_variant(&mut catalog, &persisted, &payload),
            _ => unreachable!("kind checked above"),
        };
        match applied {
            Ok(Some((id, state))) => {
                persisted.insert(id, state);
            }
            Ok(None) => {}
            Err(e) => {
                if first {
                    return Err(e);
                }
                break;
            }
        }
        offset = record_start + 10 + len as u64 + 32;
    }
    // Cut any torn or corrupt tail so later appends have no hole.
    let file_len = reader.seek(SeekFrom::End(0))?;
    if file_len != offset {
        file.set_len(offset)?;
    }
    file.seek(SeekFrom::End(0))?;
    Ok((catalog, persisted))
}

/// Decode an intake payload and restore the record. Returns the
/// updated persisted state for the record.
fn apply_intake(
    catalog: &mut Catalog,
    persisted: &HashMap<String, Persisted>,
    payload: &[u8],
) -> Result<Option<(String, Persisted)>, ArtifactError> {
    let mut cursor = Cursor::new(payload);
    let id = cursor.str_bytes()?;
    let kind = kind_from_byte(cursor.byte()?)?;
    let title = cursor.str_bytes()?;
    let description = cursor.str_bytes()?;
    let cid_bytes: [u8; 32] = cursor.take(32)?.try_into().unwrap();
    let content_cid = Cid::from_bytes(cid_bytes);
    let content = cursor.len_bytes()?.to_vec();
    let prov_count = cursor.u32()? as usize;
    let mut provenance = Vec::with_capacity(prov_count);
    for _ in 0..prov_count {
        let event_bytes = cursor.len_bytes()?;
        provenance.push(ProvenanceEvent::decode(event_bytes)?);
    }
    cursor.done()?;
    let record = Record {
        id: id.clone(),
        kind,
        title,
        description,
        content_cid,
        provenance,
        parent_id: None,
        stage: Stage::Intake,
        verification_error: None,
    };
    catalog.restore_record(record, &content)?;
    let mut state = persisted.get(&id).cloned().unwrap_or_default();
    state.intake = true;
    state.prov_count = catalog.record_raw(&id).unwrap().provenance.len();
    state.stage = catalog.record_raw(&id).unwrap().stage;
    state.parent = None;
    Ok(Some((id, state)))
}

/// Decode a provenance payload and append the event. A repeated
/// event (identical encoding already in the chain) is a no-op.
/// Returns the updated persisted state for the record.
fn apply_provenance(
    catalog: &mut Catalog,
    persisted: &HashMap<String, Persisted>,
    payload: &[u8],
) -> Result<Option<(String, Persisted)>, ArtifactError> {
    let mut cursor = Cursor::new(payload);
    let id = cursor.str_bytes()?;
    let event_bytes = cursor.len_bytes()?;
    cursor.done()?;
    let event = ProvenanceEvent::decode(event_bytes)?;
    let record = catalog
        .record_raw(&id)
        .ok_or_else(|| ArtifactError::NotFound(id.clone()))?;
    if record
        .provenance
        .iter()
        .any(|e| e.encode() == event_bytes)
    {
        let mut state = persisted.get(&id).cloned().unwrap_or_default();
        state.prov_count = record.provenance.len();
        return Ok(Some((id, state)));
    }
    catalog.add_provenance(&id, event)?;
    let mut state = persisted.get(&id).cloned().unwrap_or_default();
    state.prov_count = catalog.record_raw(&id).unwrap().provenance.len();
    Ok(Some((id, state)))
}

/// Decode a stage payload and apply the transition. Returns the
/// updated persisted state for the record.
fn apply_stage(
    catalog: &mut Catalog,
    persisted: &HashMap<String, Persisted>,
    payload: &[u8],
) -> Result<Option<(String, Persisted)>, ArtifactError> {
    let mut cursor = Cursor::new(payload);
    let id = cursor.str_bytes()?;
    let stage = stage_from_byte(cursor.byte()?)?;
    let has_error = cursor.byte()?;
    let error = match has_error {
        0 => None,
        1 => Some(cursor.str_bytes()?),
        _ => {
            return Err(ArtifactError::Persistence(
                "stage record has an invalid error flag".to_string(),
            ));
        }
    };
    cursor.done()?;
    if (stage == Stage::Failed) != error.is_some() {
        return Err(ArtifactError::Persistence(
            "stage record carries an error without the Failed stage".to_string(),
        ));
    }
    catalog.restore_stage(&id, stage, error)?;
    let mut state = persisted.get(&id).cloned().unwrap_or_default();
    state.stage = catalog.record_raw(&id).unwrap().stage;
    Ok(Some((id, state)))
}

/// Decode a variant payload and link it. A link that is already in
/// place is a no-op. Returns the updated persisted state for the
/// child record.
fn apply_variant(
    catalog: &mut Catalog,
    persisted: &HashMap<String, Persisted>,
    payload: &[u8],
) -> Result<Option<(String, Persisted)>, ArtifactError> {
    let mut cursor = Cursor::new(payload);
    let parent = cursor.str_bytes()?;
    let child = cursor.str_bytes()?;
    cursor.done()?;
    let record = catalog
        .record_raw(&child)
        .ok_or_else(|| ArtifactError::NotFound(child.clone()))?;
    if record.parent_id.as_deref() == Some(parent.as_str()) {
        let mut state = persisted.get(&child).cloned().unwrap_or_default();
        state.parent = Some(parent);
        return Ok(Some((child, state)));
    }
    catalog.link_variant(&parent, &child)?;
    let mut state = persisted.get(&child).cloned().unwrap_or_default();
    state.parent = Some(parent);
    Ok(Some((child, state)))
}

/// A bounds-checked cursor over a record payload.
struct Cursor<'a> {
    bytes: &'a [u8],
    off: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, off: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], ArtifactError> {
        if self.bytes.len() - self.off < n {
            return Err(ArtifactError::Persistence(
                "record payload is truncated".to_string(),
            ));
        }
        let slice = &self.bytes[self.off..self.off + n];
        self.off += n;
        Ok(slice)
    }

    fn byte(&mut self) -> Result<u8, ArtifactError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, ArtifactError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn len_bytes(&mut self) -> Result<&'a [u8], ArtifactError> {
        let len = self.u32()? as usize;
        self.take(len)
    }

    fn str_bytes(&mut self) -> Result<String, ArtifactError> {
        let bytes = self.len_bytes()?;
        String::from_utf8(bytes.to_vec()).map_err(|_| {
            ArtifactError::Persistence("record text is not UTF-8".to_string())
        })
    }

    fn done(&self) -> Result<(), ArtifactError> {
        if self.off != self.bytes.len() {
            return Err(ArtifactError::Persistence(
                "trailing bytes in record payload".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_identity::Identity;
    use std::fs;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kernel-artifacts-persist-{}-{}",
            std::process::id(),
            name
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn holder(name: &str) -> Identity {
        Identity::generate(name).expect("generate identity")
    }

    fn sample_catalog() -> Catalog {
        let mut catalog = Catalog::new();
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
            .unwrap();
        catalog
            .intake(
                "ms-frag-001",
                RecordKind::Fragment,
                "Dawn Fragment",
                "A leaf from the codex.",
                b"fragment bytes",
                vec![],
            )
            .unwrap();
        catalog.verify("ms-codex-001").unwrap();
        catalog.catalog("ms-codex-001").unwrap();
        catalog.publish("ms-codex-001").unwrap();
        catalog
            .link_variant("ms-codex-001", "ms-frag-001")
            .unwrap();
        catalog
    }

    fn log_bytes(dir: &Path) -> Vec<u8> {
        fs::read(dir.join(LOG_FILE_NAME)).unwrap()
    }

    /// Length of the first framed record in `bytes`, by its own header.
    fn first_record_len(bytes: &[u8]) -> usize {
        let len = u32::from_le_bytes(bytes[6..10].try_into().unwrap()) as usize;
        10 + len + 32
    }

    #[test]
    fn round_trip_preserves_catalog() {
        let dir = test_dir("round-trip");
        {
            let mut store = ArtifactStore::open(&dir).unwrap();
            *store.catalog_mut() = sample_catalog();
            store.save().unwrap();
        }
        let catalog = ArtifactStore::load(&dir).unwrap();
        assert_eq!(catalog.len(), 2);
        let codex = catalog.get("ms-codex-001").unwrap();
        assert_eq!(codex.stage, Stage::Published);
        assert_eq!(codex.title, "Codex of the Dawn");
        assert_eq!(codex.provenance.len(), 2);
        assert_eq!(codex.provenance[0].location, "Cairo");
        assert_eq!(codex.provenance[1].location, "Alexandria");
        assert_eq!(
            catalog.store().get(&codex.content_cid).unwrap(),
            b"codex bytes: in the beginning"
        );
        let frag = catalog.get("ms-frag-001").unwrap();
        assert_eq!(frag.stage, Stage::Intake);
        assert_eq!(
            frag.parent_id,
            Some("ms-codex-001".to_string())
        );
        assert_eq!(
            catalog.store().get(&frag.content_cid).unwrap(),
            b"fragment bytes"
        );
        // Search still works on the loaded catalog.
        assert_eq!(catalog.search_by_kind(RecordKind::Manuscript).len(), 1);
        assert_eq!(catalog.search_title("dawn").len(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_store_loads_cleanly() {
        let dir = test_dir("empty");
        let catalog = ArtifactStore::load(&dir).unwrap();
        assert!(catalog.is_empty());
        {
            let mut store = ArtifactStore::open(&dir).unwrap();
            store.save().unwrap();
        }
        assert_eq!(log_bytes(&dir).len(), 0);
        assert!(ArtifactStore::load(&dir).unwrap().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_is_idempotent() {
        let dir = test_dir("idempotent");
        {
            let mut store = ArtifactStore::open(&dir).unwrap();
            *store.catalog_mut() = sample_catalog();
            store.save().unwrap();
        }
        let size_after_first = log_bytes(&dir).len();
        {
            let mut store = ArtifactStore::open(&dir).unwrap();
            store.save().unwrap();
        }
        assert_eq!(log_bytes(&dir).len(), size_after_first);
        assert_eq!(ArtifactStore::load(&dir).unwrap().len(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn incremental_save_appends() {
        let dir = test_dir("incremental");
        {
            let mut store = ArtifactStore::open(&dir).unwrap();
            *store.catalog_mut() = sample_catalog();
            store.save().unwrap();
        }
        let size_after_first = log_bytes(&dir).len();
        {
            let mut store = ArtifactStore::open(&dir).unwrap();
            assert_eq!(store.catalog().len(), 2);
            let scribe = holder("scribe");
            store
                .catalog_mut()
                .add_provenance(
                    "ms-frag-001",
                    ProvenanceEvent::new(&scribe, "ms-frag-001", 3000, "Cairo", "found"),
                )
                .unwrap();
            store
                .catalog_mut()
                .intake(
                    "art-seal-001",
                    RecordKind::Artifact,
                    "Clay Seal",
                    "a seal",
                    b"seal bytes",
                    vec![],
                )
                .unwrap();
            store.save().unwrap();
        }
        assert!(log_bytes(&dir).len() > size_after_first);
        let catalog = ArtifactStore::load(&dir).unwrap();
        assert_eq!(catalog.len(), 3);
        assert_eq!(
            catalog.get("ms-frag-001").unwrap().provenance.len(),
            1
        );
        assert_eq!(
            catalog.get("art-seal-001").unwrap().stage,
            Stage::Intake
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn flipped_byte_in_first_record_is_refused() {
        let dir = test_dir("flip-first");
        {
            let mut store = ArtifactStore::open(&dir).unwrap();
            *store.catalog_mut() = sample_catalog();
            store.save().unwrap();
        }
        let mut bytes = log_bytes(&dir);
        bytes[12] ^= 0xff; // inside the first record's payload
        fs::write(dir.join(LOG_FILE_NAME), &bytes).unwrap();
        assert!(ArtifactStore::load(&dir).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn flipped_byte_in_later_record_cuts_tail() {
        let dir = test_dir("flip-later");
        {
            let mut store = ArtifactStore::open(&dir).unwrap();
            *store.catalog_mut() = sample_catalog();
            store.save().unwrap();
        }
        let mut bytes = log_bytes(&dir);
        // Flip a byte inside the second record's payload.
        let second = first_record_len(&bytes);
        bytes[second + 12] ^= 0xff;
        fs::write(dir.join(LOG_FILE_NAME), &bytes).unwrap();
        let catalog = ArtifactStore::load(&dir).unwrap();
        // The first record survived; everything from the bad record
        // on was cut.
        assert_eq!(catalog.len(), 1);
        assert!(catalog.get("ms-codex-001").is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncated_file_cuts_torn_tail() {
        let dir = test_dir("truncated");
        {
            let mut store = ArtifactStore::open(&dir).unwrap();
            *store.catalog_mut() = sample_catalog();
            store.save().unwrap();
        }
        let bytes = log_bytes(&dir);
        let first_len = first_record_len(&bytes);
        // Cut the file in the middle of the second record.
        let cut = first_len + 5;
        fs::write(dir.join(LOG_FILE_NAME), &bytes[..cut]).unwrap();
        let catalog = ArtifactStore::load(&dir).unwrap();
        assert_eq!(catalog.len(), 1);
        assert!(catalog.get("ms-codex-001").is_ok());
        // The store is usable after repair: new saves append cleanly.
        let mut store = ArtifactStore::open(&dir).unwrap();
        store
            .catalog_mut()
            .intake(
                "art-seal-001",
                RecordKind::Artifact,
                "Clay Seal",
                "",
                b"seal bytes",
                vec![],
            )
            .unwrap();
        store.save().unwrap();
        let catalog = ArtifactStore::load(&dir).unwrap();
        assert_eq!(catalog.len(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn garbage_file_is_refused() {
        let dir = test_dir("garbage");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(LOG_FILE_NAME), b"this is not an artifacts log").unwrap();
        assert!(ArtifactStore::load(&dir).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn tampered_provenance_signature_is_refused_on_load() {
        let dir = test_dir("bad-prov-sig");
        // A catalog whose only record carries a provenance event
        // with a bad signature. It can only exist by hand-crafting
        // the log: intake refuses nothing about the chain, but
        // verify() would fail it — so write the intake record
        // directly with a tampered event and a correct framing hash.
        let scribe = holder("scribe");
        let mut event = ProvenanceEvent::new(&scribe, "ms-codex-001", 1000, "Cairo", "copied");
        event.note = "forged note".to_string();
        let record = Record {
            id: "ms-codex-001".to_string(),
            kind: RecordKind::Manuscript,
            title: "Codex of the Dawn".to_string(),
            description: String::new(),
            content_cid: Cid::from_bytes(sha256(b"codex bytes")),
            provenance: vec![event],
            parent_id: None,
            stage: Stage::Intake,
            verification_error: None,
        };
        let payload = encode_intake_payload(&record, b"codex bytes");
        let framed = frame_record(KIND_INTAKE, &payload);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(LOG_FILE_NAME), &framed).unwrap();
        // The record fails verification on load, so it loads as
        // Failed — it is not dropped, and it can never publish.
        let mut catalog = ArtifactStore::load(&dir).unwrap();
        let loaded = catalog.record_raw("ms-codex-001").unwrap();
        assert_eq!(loaded.stage, Stage::Failed);
        assert!(loaded.verification_error.is_some());
        assert!(catalog.publish("ms-codex-001").is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn failed_record_round_trips_as_failed() {
        let dir = test_dir("failed-round-trip");
        {
            let mut store = ArtifactStore::open(&dir).unwrap();
            let scribe = holder("scribe");
            // Out-of-order chain: verification must fail it.
            let events = vec![
                ProvenanceEvent::new(&scribe, "ms-frag-001", 2000, "Cairo", "found"),
                ProvenanceEvent::new(&scribe, "ms-frag-001", 1000, "Alexandria", "moved"),
            ];
            store
                .catalog_mut()
                .intake(
                    "ms-frag-001",
                    RecordKind::Fragment,
                    "Dawn Fragment",
                    "A single leaf.",
                    b"fragment bytes",
                    events,
                )
                .unwrap();
            let err = store.catalog_mut().verify("ms-frag-001").unwrap_err();
            assert!(matches!(err, ArtifactError::BadProvenance(_)));
            store.save().unwrap();
        }
        let mut catalog = ArtifactStore::load(&dir).unwrap();
        let loaded = catalog.record_raw("ms-frag-001").unwrap();
        assert_eq!(loaded.stage, Stage::Failed);
        assert!(loaded.verification_error.is_some());
        assert!(loaded.verification_error.as_ref().unwrap().contains("BadProvenance")
            || loaded.verification_error.as_ref().unwrap().contains("bad provenance"));
        // Terminal: it can never advance or publish.
        assert!(catalog.publish("ms-frag-001").is_err());
        assert!(catalog.verify("ms-frag-001").is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn duplicate_record_is_skipped() {
        let dir = test_dir("duplicate");
        {
            let mut store = ArtifactStore::open(&dir).unwrap();
            *store.catalog_mut() = sample_catalog();
            store.save().unwrap();
        }
        // Hand-append a byte-identical copy of the first intake
        // record, as a retried save would after an interrupted write.
        let bytes = log_bytes(&dir);
        let first_len = first_record_len(&bytes);
        let duplicate = bytes[..first_len].to_vec();
        {
            let mut f = OpenOptions::new()
                .append(true)
                .open(dir.join(LOG_FILE_NAME))
                .unwrap();
            f.write_all(&duplicate).unwrap();
            f.sync_all().unwrap();
        }
        let catalog = ArtifactStore::load(&dir).unwrap();
        assert_eq!(catalog.len(), 2);
        assert_eq!(
            catalog.get("ms-codex-001").unwrap().stage,
            Stage::Published
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
