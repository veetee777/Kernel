//! Persistent content-addressed storage for KERNEL: blocks on disk.
//!
//! The law for KERNEL storage is binary on disk — no JSON, no SQLite,
//! no Postgres, no hosted database. This crate is the persistent side
//! of [`kernel_storage::ContentStore`]: an append-only binary log file,
//! one record per block:
//!
//! ```text
//! KBLK1 (5 bytes) + CID (32 bytes, SHA-256 of the payload)
//!       + length (u32 little-endian) + payload bytes
//! ```
//!
//! The CID *is* the hash of the payload, so a record is self-verifying:
//! a payload that does not hash to its recorded CID is corrupt and is
//! never served. On open, the log is replayed from the start and an
//! in-memory index (CID -> file offset, length) is rebuilt. A crash
//! mid-write leaves a truncated final record; the replay stops there
//! and the file is truncated back to the last good record, so later
//! appends never leave a hole. A corrupt record in the middle ends the
//! replay the same way — everything after it is suspect, so it is cut.
//!
//! Writes are fsync'd per put: a block that `put` returned is on disk,
//! not just in the OS page cache. Duplicate puts of the same bytes
//! write nothing — the CID is already in the index.
//!
//! Over time the log can still hold duplicate records (for example,
//! written by an older writer before dedup landed, or appended by
//! hand). [`DiskStore::compact`] rewrites the log with exactly one
//! record per block, crash-safely: temp file, fsync, atomic rename,
//! directory fsync. [`DiskStore::maybe_compact`] runs that rewrite
//! only when the log exceeds a size threshold, and
//! [`DiskStore::set_auto_compact`] arms the same check after every
//! `put`, so a store can keep itself bounded without the caller
//! watching it. Auto-compaction is best-effort: the block is fsync'd
//! first, and a failed compaction never fails the put (see
//! [`DiskStore::auto_compact_error`]).
//!
//! Reads seek to the indexed offset and re-verify the hash before
//! returning bytes, so a block that rotted on disk after open is still
//! caught instead of silently served.
//!
//! This crate stores blocks only. It does not store identities — a
//! node's private key on disk is a separate concern (encryption at
//! rest, owner-only permissions) and is deliberately not done here.

use std::collections::HashMap;
use std::fmt;
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use kernel_crypto::sha256;
use kernel_storage::Cid;

/// Magic bytes at the start of every log record.
const RECORD_MAGIC: &[u8; 5] = b"KBLK1";
/// Name of the log file inside the store directory.
const LOG_FILE_NAME: &str = "blocks.klog";

/// Errors returned by disk store operations.
#[derive(Debug)]
pub enum DiskError {
    /// An underlying I/O failure.
    Io(std::io::Error),
    /// A record failed verification (bad magic, wrong CID, hash mismatch).
    Corrupt(String),
}

impl fmt::Display for DiskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiskError::Io(e) => write!(f, "I/O error: {e}"),
            DiskError::Corrupt(msg) => write!(f, "corrupt log: {msg}"),
        }
    }
}

impl std::error::Error for DiskError {}

impl From<std::io::Error> for DiskError {
    fn from(e: std::io::Error) -> Self {
        DiskError::Io(e)
    }
}

/// A persistent content-addressed store backed by one append-only
/// binary log file in `dir`.
#[derive(Debug)]
pub struct DiskStore {
    file: File,
    /// Full path of the log file, kept so [`DiskStore::compact`] can
    /// write its sibling temp file and rename it over the log.
    path: PathBuf,
    index: HashMap<Cid, (u64, u32)>,
    /// Size threshold for automatic compaction, in bytes. `None`
    /// (the default) disables it.
    auto_compact: Option<u64>,
    /// Outcome of the most recent auto-compaction that ran.
    last_auto_compact: Option<Compaction>,
    /// Error from the most recent auto-compaction attempt that
    /// failed, if any. A failed auto-compaction never fails the
    /// `put` that triggered it; the failure is recorded here.
    auto_compact_error: Option<String>,
}

/// What [`DiskStore::compact`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Compaction {
    /// Records removed from the log (duplicates of a CID already held).
    pub dropped: u64,
    /// Bytes the log file shrank by.
    pub bytes_saved: u64,
}

impl DiskStore {
    /// Open the store in `dir`, creating the directory and the log file
    /// if they do not exist. Replays the log to rebuild the index.
    pub fn open(dir: &Path) -> Result<Self, DiskError> {
        create_dir_all(dir)?;
        let path = dir.join(LOG_FILE_NAME);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        let index = Self::replay(&mut file)?;
        Ok(Self {
            file,
            path: path.clone(),
            index,
            auto_compact: None,
            last_auto_compact: None,
            auto_compact_error: None,
        })
    }

    /// Replay the log, building the index. Stops at the first truncated
    /// or corrupt record and truncates the file to the last good record
    /// end, so the log is always appendable.
    fn replay(file: &mut File) -> Result<HashMap<Cid, (u64, u32)>, DiskError> {
        let mut index = HashMap::new();
        let mut reader = BufReader::new(file.try_clone()?);
        let mut offset: u64 = 0;
        loop {
            let mut magic = [0u8; 5];
            match reader.read_exact(&mut magic) {
                Ok(()) => {}
                // A short read at a record boundary is a torn write:
                // stop, keep everything before it.
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(DiskError::Io(e)),
            }
            if &magic != RECORD_MAGIC {
                break;
            }
            let mut cid_bytes = [0u8; 32];
            let mut len_bytes = [0u8; 4];
            if reader.read_exact(&mut cid_bytes).is_err()
                || reader.read_exact(&mut len_bytes).is_err()
            {
                break;
            }
            let len = u32::from_le_bytes(len_bytes) as usize;
            let mut payload = vec![0u8; len];
            if reader.read_exact(&mut payload).is_err() {
                break;
            }
            let cid = Cid::from_bytes(cid_bytes);
            if Cid::from_bytes(sha256(&payload)) != cid {
                break;
            }
            index.insert(cid, (offset, len as u32));
            offset += 5 + 32 + 4 + len as u64;
        }
        // Cut any torn or corrupt tail so later appends have no hole.
        let file_len = reader.seek(SeekFrom::End(0))?;
        if file_len != offset {
            file.set_len(offset)?;
        }
        file.seek(SeekFrom::End(0))?;
        Ok(index)
    }

    /// Store bytes, returning their CID. Already-stored bytes are not
    /// written again. The record is fsync'd before this returns.
    /// If auto-compaction is armed (see [`DiskStore::set_auto_compact`]),
    /// the size check runs after the put — including after a
    /// duplicate put that wrote nothing.
    pub fn put(&mut self, data: &[u8]) -> Result<Cid, DiskError> {
        let cid = Cid::from_bytes(sha256(data));
        if self.index.contains_key(&cid) {
            self.auto_compact_check();
            return Ok(cid);
        }
        let mut record = Vec::with_capacity(5 + 32 + 4 + data.len());
        record.extend_from_slice(RECORD_MAGIC);
        record.extend_from_slice(cid.as_bytes());
        record.extend_from_slice(&(data.len() as u32).to_le_bytes());
        record.extend_from_slice(data);
        let offset = self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(&record)?;
        self.file.sync_all()?;
        self.index.insert(cid, (offset, data.len() as u32));
        // The block is durably stored before any compaction is
        // attempted, so a failed auto-compaction can never lose it.
        self.auto_compact_check();
        Ok(cid)
    }

    /// Current size of the log file in bytes, from the file's actual
    /// metadata — so it stays true after a reopen, and reflects
    /// compaction as well as puts. Returns 0 if the metadata cannot
    /// be read (which does not happen for an open file handle in
    /// practice).
    pub fn log_len(&self) -> u64 {
        self.file.metadata().map(|m| m.len()).unwrap_or(0)
    }

    /// Arm (or disarm) automatic compaction: after every `put`, the
    /// store compacts itself when the log exceeds `max_log_bytes`.
    /// Pass `None` to turn it off again (the default). Auto-compaction
    /// is best-effort — see [`DiskStore::auto_compact_error`]. The
    /// threshold is not persisted: a reopened store starts with
    /// auto-compaction off, and the caller must arm it again.
    pub fn set_auto_compact(&mut self, max_log_bytes: Option<u64>) {
        self.auto_compact = max_log_bytes;
    }

    /// The currently armed auto-compaction threshold, if any.
    pub fn auto_compact_threshold(&self) -> Option<u64> {
        self.auto_compact
    }

    /// Outcome of the most recent auto-compaction that ran, if any.
    pub fn last_auto_compaction(&self) -> Option<Compaction> {
        self.last_auto_compact
    }

    /// Error from the most recent auto-compaction attempt that
    /// failed, if any. Cleared by the next successful auto-compaction.
    /// Callers that need a guarantee should call
    /// [`DiskStore::maybe_compact`] directly instead of relying on
    /// the trigger.
    pub fn auto_compact_error(&self) -> Option<&str> {
        self.auto_compact_error.as_deref()
    }

    /// Run the auto-compaction check once, if armed. Best-effort by
    /// design: errors are recorded on the store, never returned, so a
    /// failed compaction can never fail the `put` that triggered it.
    fn auto_compact_check(&mut self) {
        let Some(max) = self.auto_compact else {
            return;
        };
        match self.maybe_compact(max) {
            Ok(Some(done)) => {
                self.last_auto_compact = Some(done);
                self.auto_compact_error = None;
            }
            Ok(None) => {}
            Err(e) => {
                self.auto_compact_error = Some(e.to_string());
            }
        }
    }

    /// Compact only when the log exceeds `max_log_bytes`, returning
    /// `Some(compaction)` when it ran and `None` when it did not.
    ///
    /// The size condition is necessary but not sufficient: the rewrite
    /// is skipped when the log holds no duplicate records, because
    /// rewriting a duplicate-free log saves 0 bytes and only costs a
    /// full rewrite. A log with no duplicates cannot shrink, so it is
    /// left alone even over threshold.
    ///
    /// A threshold of 0 means "always compact when there is anything
    /// to drop": any non-empty log exceeds 0 bytes, so the only gate
    /// left is whether a duplicate exists. On an empty store this
    /// returns `None` — compacting an empty store is never an error.
    pub fn maybe_compact(&mut self, max_log_bytes: u64) -> Result<Option<Compaction>, DiskError> {
        if self.log_len() <= max_log_bytes {
            return Ok(None);
        }
        // More records than blocks means at least one duplicate.
        let records = Self::count_records(&mut self.file)?;
        if records <= self.index.len() as u64 {
            return Ok(None);
        }
        Ok(Some(self.compact()?))
    }

    /// Fetch bytes by CID. `Ok(None)` if the block is not held.
    /// A block whose bytes no longer hash to its CID is an error, not
    /// silently served bytes.
    pub fn get(&mut self, cid: &Cid) -> Result<Option<Vec<u8>>, DiskError> {
        let Some(&(offset, len)) = self.index.get(cid) else {
            return Ok(None);
        };
        self.file.seek(SeekFrom::Start(offset))?;
        let mut magic = [0u8; 5];
        let mut cid_bytes = [0u8; 32];
        let mut len_bytes = [0u8; 4];
        self.file.read_exact(&mut magic)?;
        self.file.read_exact(&mut cid_bytes)?;
        self.file.read_exact(&mut len_bytes)?;
        if &magic != RECORD_MAGIC || &cid_bytes != cid.as_bytes() {
            return Err(DiskError::Corrupt(format!(
                "record header at offset {offset} does not match CID {cid}"
            )));
        }
        let mut payload = vec![0u8; len as usize];
        self.file.read_exact(&mut payload)?;
        if Cid::from_bytes(sha256(&payload)) != *cid {
            return Err(DiskError::Corrupt(format!(
                "block {cid} failed hash verification on read"
            )));
        }
        Ok(Some(payload))
    }

    /// True if the store holds a block with this CID.
    pub fn has(&self, cid: &Cid) -> bool {
        self.index.contains_key(cid)
    }

    /// Number of blocks held.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// True if the store holds no blocks.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Read every block held, in index order. Used to load a store's
    /// contents into memory (for example, when a node starts up).
    pub fn all(&mut self) -> Result<Vec<(Cid, Vec<u8>)>, DiskError> {
        let cids: Vec<Cid> = self.index.keys().copied().collect();
        let mut out = Vec::with_capacity(cids.len());
        for cid in cids {
            if let Some(data) = self.get(&cid)? {
                out.push((cid, data));
            }
        }
        Ok(out)
    }

    /// Path of the sibling temp file compaction writes to.
    fn tmp_path(&self) -> PathBuf {
        let mut tmp = self.path.clone().into_os_string();
        tmp.push(".tmp");
        PathBuf::from(tmp)
    }

    /// Count the structurally complete records in the log, stopping at
    /// the first truncated or corrupt record. The replay at open time
    /// already cut any torn tail, so in normal use this counts every
    /// record in the file; it is only a count, not a verification pass.
    fn count_records(file: &mut File) -> Result<u64, DiskError> {
        let file_len = file.seek(SeekFrom::End(0))?;
        file.seek(SeekFrom::Start(0))?;
        let mut offset: u64 = 0;
        let mut count: u64 = 0;
        while offset + 41 <= file_len {
            let mut header = [0u8; 41];
            file.read_exact(&mut header)?;
            if &header[..5] != RECORD_MAGIC {
                break;
            }
            let len = u32::from_le_bytes(header[37..41].try_into().unwrap()) as u64;
            if offset + 41 + len > file_len {
                break;
            }
            offset += 41 + len;
            file.seek(SeekFrom::Start(offset))?;
            count += 1;
        }
        Ok(count)
    }

    /// Rewrite the log so it holds exactly one record per block in the
    /// index — the latest bytes for each CID — and nothing else.
    ///
    /// Crash safety: the compacted log is written to a sibling temp
    /// file (`blocks.klog.tmp`) and fsync'd before it is atomically
    /// renamed over `blocks.klog`, so a crash before the rename leaves
    /// the old log intact and a crash after it finds the new log
    /// complete and fsync'd; the directory is fsync'd after the rename
    /// so the rename itself survives a power loss.
    ///
    /// This needs exclusive access: no other reader or writer may use
    /// this store (or the same directory) while it runs, because the
    /// file handle and the index are swapped for the new log. Callers
    /// that want this to happen on its own should use
    /// [`DiskStore::maybe_compact`] or [`DiskStore::set_auto_compact`];
    /// this method itself never runs unless called.
    pub fn compact(&mut self) -> Result<Compaction, DiskError> {
        let result = self.compact_inner();
        if result.is_err() {
            // A failed compaction must not leave a stale temp file
            // behind for the next attempt to trip over.
            let _ = std::fs::remove_file(self.tmp_path());
        }
        result
    }

    fn compact_inner(&mut self) -> Result<Compaction, DiskError> {
        let records_before = Self::count_records(&mut self.file)?;
        let size_before = self.file.metadata()?.len();
        // Every block, hash-verified on read: exactly one entry per
        // CID in the index, which is exactly what the new log holds.
        let blocks = self.all()?;

        let tmp_path = self.tmp_path();
        let mut tmp = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        let mut new_index = HashMap::with_capacity(blocks.len());
        let mut offset: u64 = 0;
        for (cid, data) in &blocks {
            let mut record = Vec::with_capacity(5 + 32 + 4 + data.len());
            record.extend_from_slice(RECORD_MAGIC);
            record.extend_from_slice(cid.as_bytes());
            record.extend_from_slice(&(data.len() as u32).to_le_bytes());
            record.extend_from_slice(data);
            tmp.write_all(&record)?;
            new_index.insert(*cid, (offset, data.len() as u32));
            offset += record.len() as u64;
        }
        // The new log is complete on disk before the old name moves.
        tmp.sync_all()?;
        drop(tmp);
        let size_after = offset;

        std::fs::rename(&tmp_path, &self.path)?;
        // Reopen on the new log before touching the old handle: if
        // this fails, the store keeps working on the old (unlinked)
        // log, whose contents are a superset of the new one.
        let mut new_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)?;
        new_file.seek(SeekFrom::End(0))?;
        self.file = new_file;
        self.index = new_index;
        // Make the rename durable.
        let dir = self.path.parent().unwrap_or(Path::new("."));
        File::open(dir)?.sync_all()?;

        Ok(Compaction {
            dropped: records_before.saturating_sub(blocks.len() as u64),
            bytes_saved: size_before.saturating_sub(size_after),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kernel-store-disk-{}-{}",
            std::process::id(),
            name
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn put_and_get_survive_reopen() {
        let dir = test_dir("reopen");
        let cid;
        {
            let mut store = DiskStore::open(&dir).unwrap();
            cid = store.put(b"hello disk").unwrap();
            assert_eq!(store.get(&cid).unwrap().unwrap(), b"hello disk");
        }
        let mut reopened = DiskStore::open(&dir).unwrap();
        assert!(reopened.has(&cid));
        assert_eq!(reopened.get(&cid).unwrap().unwrap(), b"hello disk");
        assert_eq!(reopened.len(), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn duplicate_put_writes_nothing() {
        let dir = test_dir("dup");
        let mut store = DiskStore::open(&dir).unwrap();
        let cid = store.put(b"same bytes").unwrap();
        let size_after_first = fs::metadata(dir.join(LOG_FILE_NAME)).unwrap().len();
        let cid2 = store.put(b"same bytes").unwrap();
        assert_eq!(cid, cid2);
        let size_after_second = fs::metadata(dir.join(LOG_FILE_NAME)).unwrap().len();
        assert_eq!(size_after_first, size_after_second);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn large_block_round_trips() {
        let dir = test_dir("large");
        let data: Vec<u8> = (0..700 * 1024).map(|i| (i % 251) as u8).collect();
        let cid;
        {
            let mut store = DiskStore::open(&dir).unwrap();
            cid = store.put(&data).unwrap();
        }
        let mut reopened = DiskStore::open(&dir).unwrap();
        assert_eq!(reopened.get(&cid).unwrap().unwrap(), data);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn torn_tail_is_cut_and_earlier_blocks_survive() {
        let dir = test_dir("torn");
        let cid_a;
        let cid_b;
        {
            let mut store = DiskStore::open(&dir).unwrap();
            cid_a = store.put(b"first block").unwrap();
            cid_b = store.put(b"second block").unwrap();
        }
        // Simulate a crash mid-write: garbage appended to the log.
        {
            let mut f = OpenOptions::new()
                .append(true)
                .open(dir.join(LOG_FILE_NAME))
                .unwrap();
            f.write_all(b"KBLK1\x00\x01\x02").unwrap();
            f.sync_all().unwrap();
        }
        let mut reopened = DiskStore::open(&dir).unwrap();
        assert_eq!(reopened.get(&cid_a).unwrap().unwrap(), b"first block");
        assert_eq!(reopened.get(&cid_b).unwrap().unwrap(), b"second block");
        assert_eq!(reopened.len(), 2);
        // The store is usable after repair: new puts append cleanly.
        let cid_c = reopened.put(b"third block").unwrap();
        assert_eq!(reopened.get(&cid_c).unwrap().unwrap(), b"third block");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_payload_is_cut_at_replay_and_not_served() {
        let dir = test_dir("corrupt");
        let cid_a;
        let cid_b;
        {
            let mut store = DiskStore::open(&dir).unwrap();
            cid_a = store.put(b"good block").unwrap();
            cid_b = store.put(b"doomed block").unwrap();
        }
        // Flip one byte inside the second block's payload on disk.
        {
            let path = dir.join(LOG_FILE_NAME);
            let mut bytes = fs::read(&path).unwrap();
            // Second record starts after the first: 5+32+4+10 = 51 bytes in.
            let payload_start = 51 + 5 + 32 + 4;
            bytes[payload_start] ^= 0xff;
            fs::write(&path, &bytes).unwrap();
        }
        let mut reopened = DiskStore::open(&dir).unwrap();
        // The good block before the corruption still reads.
        assert_eq!(reopened.get(&cid_a).unwrap().unwrap(), b"good block");
        // The corrupted block is gone from the index (replay cut there).
        assert!(!reopened.has(&cid_b));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_cid_is_none_not_an_error() {
        let dir = test_dir("unknown");
        let mut store = DiskStore::open(&dir).unwrap();
        let missing = Cid::from_bytes(sha256(b"never stored"));
        assert_eq!(store.get(&missing).unwrap(), None);
        assert!(!store.has(&missing));
        let _ = fs::remove_dir_all(&dir);
    }

    /// Append a well-formed record for `data` directly to the log file,
    /// bypassing `put`'s dedup: this is how a log accumulates duplicate
    /// records of one CID (an older writer, or a hand-appended copy).
    fn append_raw_record(dir: &Path, data: &[u8]) {
        let cid = Cid::from_bytes(sha256(data));
        let mut record = Vec::with_capacity(5 + 32 + 4 + data.len());
        record.extend_from_slice(RECORD_MAGIC);
        record.extend_from_slice(cid.as_bytes());
        record.extend_from_slice(&(data.len() as u32).to_le_bytes());
        record.extend_from_slice(data);
        let mut f = OpenOptions::new()
            .append(true)
            .open(dir.join(LOG_FILE_NAME))
            .unwrap();
        f.write_all(&record).unwrap();
        f.sync_all().unwrap();
    }

    fn log_len(dir: &Path) -> u64 {
        fs::metadata(dir.join(LOG_FILE_NAME)).unwrap().len()
    }

    #[test]
    fn compact_drops_duplicate_records_and_keeps_every_block() {
        let dir = test_dir("compact-dup");
        let cid_a;
        let cid_b;
        let cid_c;
        {
            let mut store = DiskStore::open(&dir).unwrap();
            cid_a = store.put(b"alpha block").unwrap();
            cid_b = store.put(b"beta block").unwrap();
            cid_c = store.put(b"gamma block").unwrap();
        }
        // Two extra records for CIDs already in the log.
        append_raw_record(&dir, b"alpha block");
        append_raw_record(&dir, b"beta block");
        let mut store = DiskStore::open(&dir).unwrap();
        assert_eq!(store.len(), 3);
        let size_before = log_len(&dir);

        let done = store.compact().unwrap();
        // Two duplicate records dropped: 41+11 for alpha, 41+10 for beta.
        assert_eq!(done.dropped, 2);
        assert_eq!(done.bytes_saved, 52 + 51);
        assert_eq!(log_len(&dir), size_before - 103);
        // No temp file left behind.
        assert!(!dir.join("blocks.klog.tmp").exists());
        // Every block still reads back byte-identical.
        assert_eq!(store.get(&cid_a).unwrap().unwrap(), b"alpha block");
        assert_eq!(store.get(&cid_b).unwrap().unwrap(), b"beta block");
        assert_eq!(store.get(&cid_c).unwrap().unwrap(), b"gamma block");
        assert_eq!(store.len(), 3);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn compacted_store_survives_restart() {
        let dir = test_dir("compact-restart");
        let cid_a;
        let cid_b;
        {
            let mut store = DiskStore::open(&dir).unwrap();
            cid_a = store.put(b"alpha block").unwrap();
            cid_b = store.put(b"beta block").unwrap();
        }
        append_raw_record(&dir, b"alpha block");
        {
            let mut store = DiskStore::open(&dir).unwrap();
            let done = store.compact().unwrap();
            assert_eq!(done.dropped, 1);
        }
        let mut reopened = DiskStore::open(&dir).unwrap();
        assert_eq!(reopened.len(), 2);
        assert_eq!(reopened.get(&cid_a).unwrap().unwrap(), b"alpha block");
        assert_eq!(reopened.get(&cid_b).unwrap().unwrap(), b"beta block");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn compact_empty_store_is_a_noop() {
        let dir = test_dir("compact-empty");
        let mut store = DiskStore::open(&dir).unwrap();
        assert!(store.is_empty());
        let done = store.compact().unwrap();
        assert_eq!(done.dropped, 0);
        assert_eq!(done.bytes_saved, 0);
        assert_eq!(log_len(&dir), 0);
        assert!(store.is_empty());
        // Still usable after: a put lands and reads back.
        let cid = store.put(b"first after compact").unwrap();
        assert_eq!(
            store.get(&cid).unwrap().unwrap(),
            b"first after compact"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn compact_twice_in_a_row_drops_nothing_the_second_time() {
        let dir = test_dir("compact-twice");
        let cid_a;
        {
            let mut store = DiskStore::open(&dir).unwrap();
            cid_a = store.put(b"alpha block").unwrap();
            store.put(b"beta block").unwrap();
        }
        append_raw_record(&dir, b"alpha block");
        let mut store = DiskStore::open(&dir).unwrap();
        let first = store.compact().unwrap();
        assert_eq!(first.dropped, 1);
        let size_after_first = log_len(&dir);
        let second = store.compact().unwrap();
        assert_eq!(second.dropped, 0);
        assert_eq!(second.bytes_saved, 0);
        assert_eq!(log_len(&dir), size_after_first);
        assert_eq!(store.get(&cid_a).unwrap().unwrap(), b"alpha block");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn compact_does_not_resurrect_a_torn_tail() {
        let dir = test_dir("compact-torn");
        let cid_a;
        let cid_b;
        {
            let mut store = DiskStore::open(&dir).unwrap();
            cid_a = store.put(b"first block").unwrap();
            cid_b = store.put(b"second block").unwrap();
        }
        {
            let mut f = OpenOptions::new()
                .append(true)
                .open(dir.join(LOG_FILE_NAME))
                .unwrap();
            f.write_all(b"KBLK1\x00\x01\x02").unwrap();
            f.sync_all().unwrap();
        }
        // Reopen cuts the torn tail (replay); compact then sees a clean log.
        let mut store = DiskStore::open(&dir).unwrap();
        assert_eq!(store.len(), 2);
        let done = store.compact().unwrap();
        assert_eq!(done.dropped, 0);
        assert_eq!(done.bytes_saved, 0);
        assert_eq!(store.get(&cid_a).unwrap().unwrap(), b"first block");
        assert_eq!(store.get(&cid_b).unwrap().unwrap(), b"second block");
        // And a restart still shows exactly the two good blocks.
        drop(store);
        let mut reopened = DiskStore::open(&dir).unwrap();
        assert_eq!(reopened.len(), 2);
        assert_eq!(reopened.get(&cid_b).unwrap().unwrap(), b"second block");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn puts_after_compact_survive_a_restart() {
        let dir = test_dir("compact-puts");
        let cid_a;
        let cid_b;
        let cid_c;
        {
            let mut store = DiskStore::open(&dir).unwrap();
            cid_a = store.put(b"alpha block").unwrap();
            cid_b = store.put(b"beta block").unwrap();
            let done = store.compact().unwrap();
            assert_eq!(done.dropped, 0);
            cid_c = store.put(b"gamma block").unwrap();
        }
        let mut reopened = DiskStore::open(&dir).unwrap();
        assert_eq!(reopened.len(), 3);
        assert_eq!(reopened.get(&cid_a).unwrap().unwrap(), b"alpha block");
        assert_eq!(reopened.get(&cid_b).unwrap().unwrap(), b"beta block");
        assert_eq!(reopened.get(&cid_c).unwrap().unwrap(), b"gamma block");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn log_len_tracks_puts_and_compaction() {
        let dir = test_dir("log-len");
        let mut store = DiskStore::open(&dir).unwrap();
        assert_eq!(store.log_len(), 0);
        store.put(b"alpha block").unwrap(); // 41 + 11 = 52 bytes
        assert_eq!(store.log_len(), 52);
        store.put(b"beta block").unwrap(); // 41 + 10 = 51 bytes
        assert_eq!(store.log_len(), 103);
        // A duplicate put writes nothing, so the length does not move.
        store.put(b"alpha block").unwrap();
        assert_eq!(store.log_len(), 103);
        // Hand-appended duplicates grow the log; compaction shrinks it.
        append_raw_record(&dir, b"alpha block");
        append_raw_record(&dir, b"beta block");
        let mut store = DiskStore::open(&dir).unwrap();
        assert_eq!(store.log_len(), 103 + 52 + 51);
        store.compact().unwrap();
        assert_eq!(store.log_len(), 103);
        // The length stays true after a reopen.
        drop(store);
        let reopened = DiskStore::open(&dir).unwrap();
        assert_eq!(reopened.log_len(), 103);
        assert_eq!(reopened.log_len(), log_len(&dir));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn maybe_compact_under_threshold_returns_none_and_changes_nothing() {
        let dir = test_dir("maybe-under");
        let mut store = DiskStore::open(&dir).unwrap();
        let cid_a = store.put(b"alpha block").unwrap();
        let cid_b = store.put(b"beta block").unwrap();
        let size_before = store.log_len();
        assert_eq!(store.maybe_compact(1_000_000).unwrap(), None);
        assert_eq!(store.log_len(), size_before);
        assert_eq!(store.get(&cid_a).unwrap().unwrap(), b"alpha block");
        assert_eq!(store.get(&cid_b).unwrap().unwrap(), b"beta block");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn maybe_compact_over_threshold_compacts_and_keeps_every_block() {
        let dir = test_dir("maybe-over");
        let cid_a;
        let cid_b;
        {
            let mut store = DiskStore::open(&dir).unwrap();
            cid_a = store.put(b"alpha block").unwrap();
            cid_b = store.put(b"beta block").unwrap();
        }
        append_raw_record(&dir, b"alpha block");
        let mut store = DiskStore::open(&dir).unwrap();
        assert_eq!(store.log_len(), 103 + 52);
        let done = store.maybe_compact(10).unwrap();
        let done = done.expect("over threshold with a duplicate must run");
        assert_eq!(done.dropped, 1);
        assert_eq!(done.bytes_saved, 52);
        assert_eq!(store.log_len(), 103);
        assert_eq!(store.get(&cid_a).unwrap().unwrap(), b"alpha block");
        assert_eq!(store.get(&cid_b).unwrap().unwrap(), b"beta block");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn maybe_compact_threshold_zero_means_compact_when_anything_to_drop() {
        let dir = test_dir("maybe-zero");
        // Empty store: never an error, never a run.
        let mut store = DiskStore::open(&dir).unwrap();
        assert_eq!(store.maybe_compact(0).unwrap(), None);
        // With a duplicate present, threshold 0 always fires.
        store.put(b"alpha block").unwrap();
        store.put(b"beta block").unwrap();
        append_raw_record(&dir, b"alpha block");
        drop(store);
        let mut store = DiskStore::open(&dir).unwrap();
        let done = store.maybe_compact(0).unwrap();
        assert_eq!(done.expect("duplicate present").dropped, 1);
        // Nothing left to drop: threshold 0 now returns None instead
        // of rewriting a duplicate-free log on every call.
        assert_eq!(store.maybe_compact(0).unwrap(), None);
        assert_eq!(store.log_len(), 103);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn maybe_compact_over_threshold_without_duplicates_leaves_log_alone() {
        let dir = test_dir("maybe-nodup");
        let mut store = DiskStore::open(&dir).unwrap();
        let cid_a = store.put(b"alpha block").unwrap();
        store.put(b"beta block").unwrap();
        // Over threshold, but rewriting would save 0 bytes: skipped.
        assert_eq!(store.maybe_compact(10).unwrap(), None);
        assert_eq!(store.log_len(), 103);
        assert_eq!(store.get(&cid_a).unwrap().unwrap(), b"alpha block");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn auto_compact_keeps_log_bounded_and_survives_restart() {
        let dir = test_dir("auto-on");
        let mut store = DiskStore::open(&dir).unwrap();
        assert_eq!(store.auto_compact_threshold(), None);
        assert_eq!(store.last_auto_compaction(), None);
        store.set_auto_compact(Some(120));
        assert_eq!(store.auto_compact_threshold(), Some(120));
        let cid_a = store.put(b"alpha block").unwrap();
        let cid_b = store.put(b"beta block").unwrap();
        // Log is now 52 + 51 = 103 bytes, under the 120 threshold.
        // Duplicates from an older writer, then one more put pushes
        // the log over the threshold and the trigger fires.
        append_raw_record(&dir, b"alpha block");
        append_raw_record(&dir, b"beta block");
        // Log is now 103 + 52 + 51 = 206, over the 120 threshold, but
        // nothing has triggered yet: only a put runs the check.
        assert_eq!(store.log_len(), 206);
        assert_eq!(store.last_auto_compaction(), None);
        let cid_c = store.put(b"gamma block").unwrap();
        let done = store
            .last_auto_compaction()
            .expect("auto-compact must have run on the over-threshold put");
        assert_eq!(done.dropped, 2);
        assert_eq!(done.bytes_saved, 103);
        // One record per block: 52 + 51 + 52 = 155, bounded by the
        // number of unique blocks, not by accumulated duplicates.
        assert_eq!(store.log_len(), 155);
        assert_eq!(store.auto_compact_error(), None);
        assert_eq!(store.get(&cid_a).unwrap().unwrap(), b"alpha block");
        assert_eq!(store.get(&cid_b).unwrap().unwrap(), b"beta block");
        assert_eq!(store.get(&cid_c).unwrap().unwrap(), b"gamma block");
        drop(store);
        let mut reopened = DiskStore::open(&dir).unwrap();
        assert_eq!(reopened.len(), 3);
        assert_eq!(reopened.log_len(), 155);
        assert_eq!(reopened.get(&cid_a).unwrap().unwrap(), b"alpha block");
        assert_eq!(reopened.get(&cid_b).unwrap().unwrap(), b"beta block");
        assert_eq!(reopened.get(&cid_c).unwrap().unwrap(), b"gamma block");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn auto_compact_off_by_default_and_when_disarmed() {
        let dir = test_dir("auto-off");
        let mut store = DiskStore::open(&dir).unwrap();
        store.put(b"alpha block").unwrap();
        store.put(b"beta block").unwrap();
        append_raw_record(&dir, b"alpha block");
        append_raw_record(&dir, b"beta block");
        // No threshold armed: puts never compact, duplicates stay.
        store.put(b"gamma block").unwrap();
        assert_eq!(store.log_len(), 103 + 52 + 51 + 52);
        assert_eq!(store.last_auto_compaction(), None);
        // Arm it, then disarm it again: the disarm sticks.
        store.set_auto_compact(Some(10));
        store.set_auto_compact(None);
        assert_eq!(store.auto_compact_threshold(), None);
        store.put(b"delta block").unwrap();
        assert_eq!(store.last_auto_compaction(), None);
        assert_eq!(store.log_len(), 103 + 52 + 51 + 52 + 52);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn tiny_threshold_compacting_often_never_loses_a_put() {
        let dir = test_dir("auto-tiny");
        let mut store = DiskStore::open(&dir).unwrap();
        // A 1-byte threshold: every put runs the check, and the log
        // is over threshold from the first record on.
        store.set_auto_compact(Some(1));
        let mut cids = Vec::new();
        for i in 0..20u32 {
            let data = format!("block number {i:02} payload");
            let cid = store.put(data.as_bytes()).unwrap();
            cids.push((cid, data));
        }
        // Hand-append duplicates mid-stream, then keep putting: the
        // trigger must drop them without losing any real block.
        append_raw_record(&dir, b"block number 00 payload");
        append_raw_record(&dir, b"block number 05 payload");
        for i in 20..30u32 {
            let data = format!("block number {i:02} payload");
            let cid = store.put(data.as_bytes()).unwrap();
            cids.push((cid, data));
        }
        assert_eq!(store.len(), 30);
        for (cid, data) in &cids {
            assert_eq!(store.get(cid).unwrap().unwrap().as_slice(), data.as_bytes());
        }
        // The trigger ran (duplicates were dropped at some point).
        assert!(store.last_auto_compaction().is_some());
        assert_eq!(store.auto_compact_error(), None);
        // And everything survives a restart.
        drop(store);
        let mut reopened = DiskStore::open(&dir).unwrap();
        assert_eq!(reopened.len(), 30);
        for (cid, data) in &cids {
            assert_eq!(
                reopened.get(cid).unwrap().unwrap().as_slice(),
                data.as_bytes()
            );
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
