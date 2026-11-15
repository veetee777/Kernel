//! The blocked-requests log: a hash-chained, tamper-evident record
//! of every allow/block decision the shield made — which host, which
//! app, allowed or blocked, why, and when (caller-supplied time).
//! Each entry's hash covers everything in the entry plus the
//! previous entry's hash, so changing, reordering, dropping, or
//! inserting an entry breaks the chain. [`BlockLog::verify`]
//! recomputes the whole chain; [`BlockLog::load`] refuses a
//! tampered file, never half-loads it. The log saves/loads as
//! explicit binary (no JSON).

use crate::ShieldError;

/// Magic bytes at the start of a saved blocked-requests log.
const BLOCK_MAGIC: &[u8; 7] = b"XBLOCK1";

/// One entry in the blocked-requests log: a single allow/block
/// decision, chained to the entry before it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockEntry {
    /// Position in the log, starting at 0.
    pub seq: u64,
    /// Caller-supplied time of the decision.
    pub timestamp: u64,
    /// The host the request was for, e.g. `google-analytics.com`
    /// (or `peer:<hex key>` for a peer check).
    pub host: String,
    /// The app that made the request, e.g. `k-talk` (or
    /// `kernel-node` for a peer check).
    pub app: String,
    /// True for allowed, false for blocked.
    pub allowed: bool,
    /// Human-readable reason for the decision.
    pub reason: String,
    /// Hash of the previous entry (all zeros for the first entry).
    pub prev_hash: [u8; 32],
    /// Hash of this entry, covering seq, timestamp, host, app,
    /// allowed, reason, and `prev_hash`.
    pub hash: [u8; 32],
}

impl BlockEntry {
    /// The hash of an entry's contents chained to `prev_hash`.
    fn compute_hash(
        seq: u64,
        timestamp: u64,
        host: &str,
        app: &str,
        allowed: bool,
        reason: &str,
        prev_hash: &[u8; 32],
    ) -> [u8; 32] {
        let mut msg =
            Vec::with_capacity(32 + 8 + 8 + 2 + host.len() + 2 + app.len() + 1 + 2 + reason.len());
        msg.extend_from_slice(prev_hash);
        msg.extend_from_slice(&seq.to_le_bytes());
        msg.extend_from_slice(&timestamp.to_le_bytes());
        msg.extend_from_slice(&(host.len() as u16).to_le_bytes());
        msg.extend_from_slice(host.as_bytes());
        msg.extend_from_slice(&(app.len() as u16).to_le_bytes());
        msg.extend_from_slice(app.as_bytes());
        msg.push(allowed as u8);
        msg.extend_from_slice(&(reason.len() as u16).to_le_bytes());
        msg.extend_from_slice(reason.as_bytes());
        kernel_crypto::sha256(&msg)
    }

    /// Encode one entry: seq (u64 LE) + timestamp (u64 LE) + host
    /// length (u16 LE) + host + app length (u16 LE) + app + allowed
    /// (u8) + reason length (u16 LE) + reason + prev_hash (32) +
    /// hash (32).
    fn encode(&self) -> Vec<u8> {
        let host_bytes = self.host.as_bytes();
        let app_bytes = self.app.as_bytes();
        let reason_bytes = self.reason.as_bytes();
        let mut out = Vec::with_capacity(
            8 + 8
                + 2
                + host_bytes.len()
                + 2
                + app_bytes.len()
                + 1
                + 2
                + reason_bytes.len()
                + 32
                + 32,
        );
        out.extend_from_slice(&self.seq.to_le_bytes());
        out.extend_from_slice(&self.timestamp.to_le_bytes());
        out.extend_from_slice(&(host_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(host_bytes);
        out.extend_from_slice(&(app_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(app_bytes);
        out.push(self.allowed as u8);
        out.extend_from_slice(&(reason_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(reason_bytes);
        out.extend_from_slice(&self.prev_hash);
        out.extend_from_slice(&self.hash);
        out
    }

    /// Decode one entry from `bytes` starting at `off`, returning
    /// the entry and the offset just past it.
    fn decode_at(bytes: &[u8], off: usize) -> Result<(Self, usize), ShieldError> {
        let err = |msg: &str| ShieldError::InvalidBlockLog(msg.to_string());
        if bytes.len() < off + 8 + 8 + 2 {
            return Err(err("block entry is truncated"));
        }
        let mut o = off;
        let seq = u64::from_le_bytes(bytes[o..o + 8].try_into().expect("8-byte slice"));
        o += 8;
        let timestamp = u64::from_le_bytes(bytes[o..o + 8].try_into().expect("8-byte slice"));
        o += 8;
        let host_len =
            u16::from_le_bytes(bytes[o..o + 2].try_into().expect("2-byte slice")) as usize;
        o += 2;
        if bytes.len() < o + host_len + 2 + 1 + 2 + 32 + 32 {
            return Err(err("block entry is truncated"));
        }
        let host = String::from_utf8(bytes[o..o + host_len].to_vec())
            .map_err(|_| err("block host is not valid UTF-8"))?;
        o += host_len;
        let app_len =
            u16::from_le_bytes(bytes[o..o + 2].try_into().expect("2-byte slice")) as usize;
        o += 2;
        if bytes.len() < o + app_len + 1 + 2 + 32 + 32 {
            return Err(err("block entry is truncated"));
        }
        let app = String::from_utf8(bytes[o..o + app_len].to_vec())
            .map_err(|_| err("block app is not valid UTF-8"))?;
        o += app_len;
        let allowed = match bytes[o] {
            0 => false,
            1 => true,
            other => return Err(err(&format!("allowed byte must be 0 or 1, got {other}"))),
        };
        o += 1;
        let reason_len =
            u16::from_le_bytes(bytes[o..o + 2].try_into().expect("2-byte slice")) as usize;
        o += 2;
        if bytes.len() < o + reason_len + 32 + 32 {
            return Err(err("block entry is truncated"));
        }
        let reason = String::from_utf8(bytes[o..o + reason_len].to_vec())
            .map_err(|_| err("block reason is not valid UTF-8"))?;
        o += reason_len;
        let mut prev_hash = [0u8; 32];
        prev_hash.copy_from_slice(&bytes[o..o + 32]);
        o += 32;
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&bytes[o..o + 32]);
        o += 32;
        Ok((
            Self {
                seq,
                timestamp,
                host,
                app,
                allowed,
                reason,
                prev_hash,
                hash,
            },
            o,
        ))
    }
}

/// An append-only, hash-chained log of shield decisions. Entries are
/// only ever appended; the chain makes tampering detectable:
/// [`BlockLog::verify`] recomputes every hash and the links between
/// entries, and refuses a log that does not check out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockLog {
    entries: Vec<BlockEntry>,
}

impl BlockLog {
    /// An empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a decision. The entry's sequence number, previous hash,
    /// and hash are computed here, never supplied by the caller.
    pub fn record(
        &mut self,
        host: &str,
        app: &str,
        allowed: bool,
        reason: &str,
        timestamp: u64,
    ) -> &BlockEntry {
        let seq = self.entries.len() as u64;
        let prev_hash = self.entries.last().map(|e| e.hash).unwrap_or([0u8; 32]);
        let hash = BlockEntry::compute_hash(seq, timestamp, host, app, allowed, reason, &prev_hash);
        self.entries.push(BlockEntry {
            seq,
            timestamp,
            host: host.to_string(),
            app: app.to_string(),
            allowed,
            reason: reason.to_string(),
            prev_hash,
            hash,
        });
        self.entries.last().expect("just pushed")
    }

    /// Number of entries in the log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if the log holds no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All entries, oldest first.
    pub fn entries(&self) -> &[BlockEntry] {
        &self.entries
    }

    /// The hash of the latest entry, or all zeros for an empty log.
    pub fn last_hash(&self) -> [u8; 32] {
        self.entries.last().map(|e| e.hash).unwrap_or([0u8; 32])
    }

    /// Verify the whole chain: every entry's hash must recompute
    /// from its contents and its predecessor's hash, sequence
    /// numbers must run 0, 1, 2, ... with no gaps, and the first
    /// entry's previous hash must be all zeros. Any failure is a
    /// clean error naming the first bad entry.
    pub fn verify(&self) -> Result<(), ShieldError> {
        let mut prev_hash = [0u8; 32];
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.seq != i as u64 {
                return Err(ShieldError::InvalidBlockLog(format!(
                    "entry {} has seq {}, expected {i}",
                    i, entry.seq
                )));
            }
            if entry.prev_hash != prev_hash {
                return Err(ShieldError::InvalidBlockLog(format!(
                    "entry {i} does not chain to the previous entry"
                )));
            }
            let expected = BlockEntry::compute_hash(
                entry.seq,
                entry.timestamp,
                &entry.host,
                &entry.app,
                entry.allowed,
                &entry.reason,
                &entry.prev_hash,
            );
            if entry.hash != expected {
                return Err(ShieldError::InvalidBlockLog(format!(
                    "entry {i} hash does not match its contents"
                )));
            }
            prev_hash = entry.hash;
        }
        Ok(())
    }

    /// Encode the whole log: magic + entry count (u32 LE) + entries.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(7 + 4);
        out.extend_from_slice(BLOCK_MAGIC);
        out.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for entry in &self.entries {
            out.extend_from_slice(&entry.encode());
        }
        out
    }

    /// Decode a log produced by [`BlockLog::encode`]. Structural
    /// problems are refused here; chain integrity is checked by
    /// [`BlockLog::verify`], which the caller should run after
    /// loading.
    pub fn decode(bytes: &[u8]) -> Result<Self, ShieldError> {
        if bytes.len() < 7 + 4 || &bytes[..7] != BLOCK_MAGIC {
            return Err(ShieldError::InvalidBlockLog(
                "missing XBLOCK1 magic".to_string(),
            ));
        }
        let count = u32::from_le_bytes(bytes[7..11].try_into().expect("4-byte slice")) as usize;
        let mut entries = Vec::with_capacity(count.min(1_000_000));
        let mut off = 11;
        for _ in 0..count {
            let (entry, next) = BlockEntry::decode_at(bytes, off)?;
            entries.push(entry);
            off = next;
        }
        if off != bytes.len() {
            return Err(ShieldError::InvalidBlockLog(format!(
                "block log has {} trailing bytes",
                bytes.len() - off
            )));
        }
        Ok(Self { entries })
    }

    /// Save the log to `path` atomically: write a sibling temp file,
    /// fsync it, then rename over the destination.
    pub fn save(&self, path: &std::path::Path) -> Result<(), ShieldError> {
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, self.encode())
            .map_err(|e| ShieldError::InvalidBlockLog(format!("block save failed: {e}")))?;
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&tmp)
            .map_err(|e| ShieldError::InvalidBlockLog(format!("block save failed: {e}")))?;
        file.sync_all()
            .map_err(|e| ShieldError::InvalidBlockLog(format!("block save failed: {e}")))?;
        drop(file);
        std::fs::rename(&tmp, path)
            .map_err(|e| ShieldError::InvalidBlockLog(format!("block save failed: {e}")))?;
        Ok(())
    }

    /// Load a log saved by [`BlockLog::save`]. The loaded log is
    /// verified end-to-end before it is returned: a tampered file
    /// is refused, never half-loaded.
    pub fn load(path: &std::path::Path) -> Result<Self, ShieldError> {
        let bytes = std::fs::read(path)
            .map_err(|e| ShieldError::InvalidBlockLog(format!("block load failed: {e}")))?;
        let log = Self::decode(&bytes)?;
        log.verify()?;
        Ok(log)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_records_in_order_and_verifies() {
        let mut log = BlockLog::new();
        assert!(log.is_empty());
        assert_eq!(log.last_hash(), [0u8; 32]);
        log.record(
            "google-analytics.com",
            "k-talk",
            false,
            "on the blocklist",
            1000,
        );
        log.record("example.com", "k-talk", true, "allowed", 1001);
        log.record(
            "doubleclick.net",
            "kernel-vision",
            false,
            "on the blocklist",
            1002,
        );
        assert_eq!(log.len(), 3);
        assert!(log.verify().is_ok());
        let entries = log.entries();
        assert_eq!(entries[0].seq, 0);
        assert_eq!(entries[2].seq, 2);
        assert_eq!(entries[0].host, "google-analytics.com");
        assert_eq!(entries[0].app, "k-talk");
        assert!(!entries[0].allowed);
        assert!(entries[1].allowed);
        assert_eq!(entries[0].prev_hash, [0u8; 32]);
        assert_eq!(entries[1].prev_hash, entries[0].hash);
        assert_eq!(entries[2].prev_hash, entries[1].hash);
        assert_eq!(log.last_hash(), entries[2].hash);
    }

    #[test]
    fn log_tamper_is_caught_on_verify() {
        let mut log = BlockLog::new();
        log.record(
            "google-analytics.com",
            "k-talk",
            false,
            "on the blocklist",
            1000,
        );
        log.record("example.com", "k-talk", true, "allowed", 1001);
        let encoded = log.encode();
        // Flip the last byte: the last entry's hash. The log still
        // decodes structurally, but the chain no longer verifies.
        let mut tampered = encoded.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0xff;
        let decoded = BlockLog::decode(&tampered).unwrap();
        assert!(decoded.verify().is_err());
        // Flip a byte inside the first entry's reason: same result.
        let mut tampered2 = encoded.clone();
        tampered2[40] ^= 0x01;
        let decoded2 = BlockLog::decode(&tampered2).unwrap();
        assert!(decoded2.verify().is_err());
        // The untampered log verifies.
        assert!(BlockLog::decode(&encoded).unwrap().verify().is_ok());
        assert!(BlockLog::decode(b"garbage").is_err());
    }

    #[test]
    fn log_save_load_round_trip() {
        let mut log = BlockLog::new();
        log.record(
            "google-analytics.com",
            "k-talk",
            false,
            "on the blocklist",
            1000,
        );
        log.record("example.com", "k-talk", true, "allowed", 1001);
        let path = std::env::temp_dir().join("x-shield-block-test.klog");
        log.save(&path).unwrap();
        let loaded = BlockLog::load(&path).unwrap();
        assert_eq!(loaded, log);
        assert!(loaded.verify().is_ok());
        // Tamper with the file on disk: load refuses it.
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        assert!(BlockLog::load(&path).is_err());
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("tmp"));
    }
}
