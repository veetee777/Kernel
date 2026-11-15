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
    path: PathBuf,
    index: HashMap<Cid, (u64, u32)>,
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
        Ok(Self { file, path, index })
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
    pub fn put(&mut self, data: &[u8]) -> Result<Cid, DiskError> {
        let cid = Cid::from_bytes(sha256(data));
        if self.index.contains_key(&cid) {
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
        Ok(cid)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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
}
