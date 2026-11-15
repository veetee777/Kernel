//! Persistent history and delivery receipts for K•Talk.
//!
//! Every message (direct and group) is appended to a binary history
//! log. The log is per-conversation — a direct conversation with one
//! peer, or one group — newest last, retrievable by conversation id
//! with a limit. The log survives a restart through an exact binary
//! save/load: a corrupted log is a clean error, never a half-loaded
//! history.
//!
//! History log format (explicit binary, no JSON):
//! `KTHIST1` (7) + version (u8, 1) + owner key (32) + owner name
//! length (u16 LE) + owner name + entry count (u32 LE) + entries +
//! SHA-256 checksum (32) over everything before it. Each entry is a
//! kind byte (1 = direct message, 2 = group message, 3 = receipt) +
//! payload length (u32 LE) + payload bytes (the message's or
//! receipt's own encoding).
//!
//! A delivery receipt is a signed record: a message id, a status
//! (Sent / Delivered / Read), the peer asserting it, and a
//! caller-supplied timestamp. Receipts are stored alongside history.
//! A receipt is only valid for a message this node holds, asserted by
//! someone who was a recipient of that message — a receipt for an
//! unknown message, or from someone who was never a recipient, is
//! refused by [`crate::TalkNode::record_receipt`].

use std::fmt;
use std::path::Path;

use kernel_crypto::{sha256, Ed25519Signature};
use kernel_identity::PeerId;

use crate::group::GroupMessage;
use crate::{Message, TalkError};

/// Magic bytes at the start of every saved history log.
const HISTORY_MAGIC: &[u8; 7] = b"KTHIST1";
/// History log format version.
const HISTORY_VERSION: u8 = 1;

/// Magic bytes at the start of every encoded receipt.
const RECEIPT_MAGIC: &[u8; 9] = b"KRECEIPT1";
/// Domain separator for receipt signatures.
const RECEIPT_SIGN_DOMAIN: &[u8; 9] = b"KRECEIPT1";

/// Which conversation a history entry belongs to.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConversationId {
    /// A direct conversation with one peer.
    Direct(PeerId),
    /// One group, by its human-readable id.
    Group(String),
}

/// One entry in the history log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HistoryEntry {
    /// A direct message.
    Direct(Message),
    /// A group message.
    Group(GroupMessage),
    /// A delivery receipt.
    Receipt(Receipt),
}

impl HistoryEntry {
    fn kind_byte(&self) -> u8 {
        match self {
            HistoryEntry::Direct(_) => 1,
            HistoryEntry::Group(_) => 2,
            HistoryEntry::Receipt(_) => 3,
        }
    }

    fn payload(&self) -> Vec<u8> {
        match self {
            HistoryEntry::Direct(m) => m.encode(),
            HistoryEntry::Group(m) => m.encode(),
            HistoryEntry::Receipt(r) => r.encode(),
        }
    }

    fn decode(kind: u8, payload: &[u8]) -> Result<Self, TalkError> {
        match kind {
            1 => Ok(HistoryEntry::Direct(Message::decode(payload)?)),
            2 => Ok(HistoryEntry::Group(GroupMessage::decode(payload)?)),
            3 => Ok(HistoryEntry::Receipt(Receipt::decode(payload)?)),
            other => Err(TalkError::CorruptHistory(format!(
                "unknown history entry kind {other}"
            ))),
        }
    }
}

/// What a receipt asserts about a message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReceiptStatus {
    /// The sender recorded that the message was sent.
    Sent = 1,
    /// A recipient recorded that it arrived.
    Delivered = 2,
    /// A recipient recorded that it was read.
    Read = 3,
}

impl ReceiptStatus {
    fn from_byte(b: u8) -> Result<Self, TalkError> {
        match b {
            1 => Ok(ReceiptStatus::Sent),
            2 => Ok(ReceiptStatus::Delivered),
            3 => Ok(ReceiptStatus::Read),
            other => Err(TalkError::BadReceipt(format!(
                "unknown receipt status byte {other}"
            ))),
        }
    }
}

/// A signed delivery receipt: a message id, a status, the peer
/// asserting it, and a caller-supplied timestamp.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Receipt {
    /// SHA-256 of the message's exact encoding.
    pub message_id: [u8; 32],
    /// What is asserted about the message.
    pub status: ReceiptStatus,
    /// Who asserts it (signs the receipt).
    pub by: PeerId,
    /// Caller-supplied unix seconds.
    pub timestamp: u64,
    /// The asserting peer's signature over message id, status,
    /// timestamp.
    pub signature: Ed25519Signature,
}

fn receipt_sign_bytes(message_id: &[u8; 32], status: ReceiptStatus, timestamp: u64) -> Vec<u8> {
    let mut msg = Vec::with_capacity(RECEIPT_SIGN_DOMAIN.len() + 32 + 1 + 8);
    msg.extend_from_slice(RECEIPT_SIGN_DOMAIN);
    msg.extend_from_slice(message_id);
    msg.push(status as u8);
    msg.extend_from_slice(&timestamp.to_le_bytes());
    msg
}

impl Receipt {
    /// Create and sign a receipt. `timestamp` is caller-supplied:
    /// this crate has no clock for receipts, so the caller states
    /// when the status was observed.
    pub fn new(
        by: &kernel_identity::Identity,
        message_id: [u8; 32],
        status: ReceiptStatus,
        timestamp: u64,
    ) -> Self {
        let peer_id = by.peer_id().clone();
        let signature = by.sign(&receipt_sign_bytes(&message_id, status, timestamp));
        Self {
            message_id,
            status,
            by: peer_id,
            timestamp,
            signature,
        }
    }

    /// True if the signature verifies against `by` for this exact
    /// message id, status, and timestamp.
    pub fn verify(&self) -> bool {
        let msg = receipt_sign_bytes(&self.message_id, self.status, self.timestamp);
        self.by.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary wire format.
    pub fn encode(&self) -> Vec<u8> {
        let name = self.by.display_name().as_bytes();
        let mut out = Vec::with_capacity(9 + 32 + 1 + 32 + 2 + name.len() + 8 + 64);
        out.extend_from_slice(RECEIPT_MAGIC);
        out.extend_from_slice(&self.message_id);
        out.push(self.status as u8);
        out.extend_from_slice(self.by.public_key().as_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&self.timestamp.to_le_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a receipt produced by [`Receipt::encode`]. Decoding does
    /// not verify the signature — call [`Receipt::verify`] for that.
    pub fn decode(bytes: &[u8]) -> Result<Self, TalkError> {
        if bytes.len() < 9 + 32 + 1 + 32 + 2 + 8 + 64 || &bytes[..9] != RECEIPT_MAGIC {
            return Err(TalkError::BadReceipt("missing KRECEIPT1 magic".to_string()));
        }
        let mut off = 9;
        let mut message_id = [0u8; 32];
        message_id.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let status = ReceiptStatus::from_byte(bytes[off])?;
        off += 1;
        let mut key = [0u8; 32];
        key.copy_from_slice(&bytes[off..off + 32]);
        let key = kernel_crypto::Ed25519PublicKey::from_bytes(&key)
            .map_err(|e| TalkError::BadReceipt(e.to_string()))?;
        off += 32;
        let name_len =
            u16::from_le_bytes(bytes[off..off + 2].try_into().expect("2-byte slice")) as usize;
        off += 2;
        if bytes.len() < off + name_len + 8 + 64 {
            return Err(TalkError::BadReceipt(
                "receipt truncates the peer name or timestamp".to_string(),
            ));
        }
        let name = String::from_utf8(bytes[off..off + name_len].to_vec())
            .map_err(|_| TalkError::BadReceipt("peer name is not UTF-8".to_string()))?;
        off += name_len;
        let by = PeerId::new(key, &name).map_err(|e| TalkError::BadReceipt(e.to_string()))?;
        let timestamp = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        if bytes.len() != off + 64 {
            return Err(TalkError::BadReceipt(format!(
                "receipt is {} bytes, expected {}",
                bytes.len(),
                off + 64
            )));
        }
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        Ok(Self {
            message_id,
            status,
            by,
            timestamp,
            signature,
        })
    }
}

/// A message found in history, by id.
pub enum FoundMessage<'a> {
    /// A direct message.
    Direct(&'a Message),
    /// A group message.
    Group(&'a GroupMessage),
}

/// The persistent history log: every message and receipt this node
/// has seen, in arrival order (newest last).
#[derive(Clone, Debug)]
pub struct History {
    /// Whose history this is. Direct conversations are matched
    /// against this peer id.
    owner: PeerId,
    entries: Vec<HistoryEntry>,
}

impl History {
    /// An empty history for `owner`.
    pub fn new(owner: PeerId) -> Self {
        Self {
            owner,
            entries: Vec::new(),
        }
    }

    /// Whose history this is.
    pub fn owner(&self) -> &PeerId {
        &self.owner
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if there are no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Append a direct message.
    pub fn append_direct(&mut self, message: Message) {
        self.entries.push(HistoryEntry::Direct(message));
    }

    /// Append a group message.
    pub fn append_group(&mut self, message: GroupMessage) {
        self.entries.push(HistoryEntry::Group(message));
    }

    /// Append a delivery receipt.
    pub fn append_receipt(&mut self, receipt: Receipt) {
        self.entries.push(HistoryEntry::Receipt(receipt));
    }

    /// All entries, in order (newest last).
    pub fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    /// Find a message by its id (SHA-256 of its encoding).
    pub fn find_message(&self, id: &[u8; 32]) -> Option<FoundMessage<'_>> {
        for entry in &self.entries {
            match entry {
                HistoryEntry::Direct(m) if m.id() == *id => {
                    return Some(FoundMessage::Direct(m));
                }
                HistoryEntry::Group(m) if m.id() == *id => {
                    return Some(FoundMessage::Group(m));
                }
                _ => {}
            }
        }
        None
    }

    /// Entries for one conversation, in order (newest last). `limit`
    /// caps how many are returned (the newest `limit`); 0 means no
    /// limit. Receipts are included when the message they name belongs
    /// to the conversation.
    ///
    /// Ordering is by timestamp, then by the sender's sequence for
    /// messages (so messages sent in the same second keep send order
    /// even if they arrived out of order), with receipts after
    /// same-timestamp messages. Arrival order breaks any remaining
    /// tie (the sort is stable).
    pub fn entries_for(&self, conv: &ConversationId, limit: usize) -> Vec<HistoryEntry> {
        let matches = |entry: &HistoryEntry| -> bool {
            match (conv, entry) {
                (ConversationId::Direct(peer), HistoryEntry::Direct(m)) => {
                    (m.sender == *peer && m.recipient == self.owner)
                        || (m.sender == self.owner && m.recipient == *peer)
                }
                (ConversationId::Group(id), HistoryEntry::Group(m)) => m.group_id == *id,
                (_, HistoryEntry::Receipt(r)) => match self.find_message(&r.message_id) {
                    Some(FoundMessage::Direct(m)) => match conv {
                        ConversationId::Direct(peer) => {
                            (m.sender == *peer && m.recipient == self.owner)
                                || (m.sender == self.owner && m.recipient == *peer)
                        }
                        ConversationId::Group(_) => false,
                    },
                    Some(FoundMessage::Group(m)) => match conv {
                        ConversationId::Group(id) => m.group_id == *id,
                        ConversationId::Direct(_) => false,
                    },
                    None => false,
                },
                _ => false,
            }
        };
        let mut out: Vec<HistoryEntry> = self
            .entries
            .iter()
            .filter(|e| matches(e))
            .cloned()
            .collect();
        out.sort_by_key(|entry| match entry {
            HistoryEntry::Direct(m) => (m.timestamp, m.seq, 0u8),
            HistoryEntry::Group(m) => (m.timestamp, m.seq, 0u8),
            HistoryEntry::Receipt(r) => (r.timestamp, u64::MAX, 1u8),
        });
        if limit > 0 && out.len() > limit {
            out.drain(..out.len() - limit);
        }
        out
    }

    /// Serialize the whole log to bytes. The same history always
    /// serializes to the same bytes.
    pub fn save(&self) -> Vec<u8> {
        let owner_name = self.owner.display_name().as_bytes();
        let mut out = Vec::with_capacity(7 + 1 + 32 + 2 + owner_name.len() + 4 + 32);
        out.extend_from_slice(HISTORY_MAGIC);
        out.push(HISTORY_VERSION);
        out.extend_from_slice(self.owner.public_key().as_bytes());
        out.extend_from_slice(&(owner_name.len() as u16).to_le_bytes());
        out.extend_from_slice(owner_name);
        out.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for entry in &self.entries {
            let payload = entry.payload();
            out.push(entry.kind_byte());
            out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            out.extend_from_slice(&payload);
        }
        let checksum = sha256(&out);
        out.extend_from_slice(&checksum);
        out
    }

    /// Load a log produced by [`History::save`]. Any corruption —
    /// bad magic, wrong version, truncated entries, a bad checksum,
    /// trailing bytes, an entry that does not decode — is a clean
    /// [`TalkError::CorruptHistory`], never a half-loaded history.
    pub fn load(bytes: &[u8]) -> Result<Self, TalkError> {
        let corrupt = |msg: String| TalkError::CorruptHistory(msg);
        if bytes.len() < 7 + 1 + 32 + 2 + 4 + 32 || &bytes[..7] != HISTORY_MAGIC {
            return Err(corrupt("missing KTHIST1 magic".to_string()));
        }
        if bytes[7] != HISTORY_VERSION {
            return Err(corrupt(format!("unsupported history version {}", bytes[7])));
        }
        let body = &bytes[..bytes.len() - 32];
        let checksum = sha256(body);
        if checksum != bytes[bytes.len() - 32..] {
            return Err(corrupt("history checksum mismatch".to_string()));
        }
        let mut off = 8;
        let mut key = [0u8; 32];
        key.copy_from_slice(&body[off..off + 32]);
        let key = kernel_crypto::Ed25519PublicKey::from_bytes(&key)
            .map_err(|e| corrupt(e.to_string()))?;
        off += 32;
        let name_len =
            u16::from_le_bytes(body[off..off + 2].try_into().expect("2-byte slice")) as usize;
        off += 2;
        if body.len() < off + name_len + 4 {
            return Err(corrupt("history truncates the owner name".to_string()));
        }
        let name = String::from_utf8(body[off..off + name_len].to_vec())
            .map_err(|_| corrupt("owner name is not UTF-8".to_string()))?;
        off += name_len;
        let owner = PeerId::new(key, &name).map_err(|e| corrupt(e.to_string()))?;
        let count =
            u32::from_le_bytes(body[off..off + 4].try_into().expect("4-byte slice")) as usize;
        off += 4;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            if body.len() < off + 1 + 4 {
                return Err(corrupt("history truncates an entry header".to_string()));
            }
            let kind = body[off];
            let len = u32::from_le_bytes(body[off + 1..off + 5].try_into().expect("4-byte slice"))
                as usize;
            off += 5;
            if body.len() < off + len {
                return Err(corrupt("history truncates an entry payload".to_string()));
            }
            let entry = HistoryEntry::decode(kind, &body[off..off + len])?;
            entries.push(entry);
            off += len;
        }
        if off != body.len() {
            return Err(corrupt(format!(
                "history has {} trailing bytes after {count} entries",
                body.len() - off
            )));
        }
        Ok(Self { owner, entries })
    }

    /// Save the log to a file at `path`.
    pub fn save_to_file(&self, path: &Path) -> Result<(), TalkError> {
        std::fs::write(path, self.save())
            .map_err(|e| TalkError::CorruptHistory(format!("cannot write history: {e}")))?;
        Ok(())
    }

    /// Load a log from a file at `path`.
    pub fn load_from_file(path: &Path) -> Result<Self, TalkError> {
        let bytes = std::fs::read(path)
            .map_err(|e| TalkError::CorruptHistory(format!("cannot read history: {e}")))?;
        Self::load(&bytes)
    }
}

impl fmt::Display for ReceiptStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ReceiptStatus::Sent => "sent",
            ReceiptStatus::Delivered => "delivered",
            ReceiptStatus::Read => "read",
        };
        write!(f, "{s}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::GroupMessage;
    use kernel_identity::Identity;

    fn sample_history() -> (Identity, Identity, History) {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut history = History::new(alice.peer_id().clone());
        let m1 = Message::new(&alice, bob.peer_id(), "first", 0);
        let m2 = Message::new(&bob, alice.peer_id(), "second", 0);
        let m3 = Message::new(&alice, bob.peer_id(), "third", 1);
        history.append_direct(m1);
        history.append_direct(m2);
        history.append_direct(m3);
        (alice, bob, history)
    }

    #[test]
    fn history_save_load_round_trip_is_exact() {
        let (alice, bob, history) = sample_history();
        let bytes = history.save();
        let loaded = History::load(&bytes).unwrap();
        assert_eq!(loaded.owner(), alice.peer_id());
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.entries(), history.entries());
        // Saving again gives the same bytes.
        assert_eq!(loaded.save(), bytes);
        // Per-conversation retrieval: newest last, limit works.
        let conv = ConversationId::Direct(bob.peer_id().clone());
        let all = loaded.entries_for(&conv, 0);
        assert_eq!(all.len(), 3);
        let last_two = loaded.entries_for(&conv, 2);
        assert_eq!(last_two.len(), 2);
        assert_eq!(last_two, all[1..]);
        // A conversation with someone else is empty.
        let carol = Identity::generate("carol").unwrap();
        assert!(loaded
            .entries_for(&ConversationId::Direct(carol.peer_id().clone()), 0)
            .is_empty());
    }

    #[test]
    fn history_group_entries_are_separate_conversations() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut history = History::new(alice.peer_id().clone());
        let direct = Message::new(&alice, bob.peer_id(), "direct", 0);
        let group = GroupMessage::new(&alice, "korn-builders", "group", 0).unwrap();
        history.append_direct(direct);
        history.append_group(group);
        assert_eq!(
            history
                .entries_for(&ConversationId::Direct(bob.peer_id().clone()), 0)
                .len(),
            1
        );
        assert_eq!(
            history
                .entries_for(&ConversationId::Group("korn-builders".to_string()), 0)
                .len(),
            1
        );
        assert!(history
            .entries_for(&ConversationId::Group("other-group".to_string()), 0)
            .is_empty());
    }

    #[test]
    fn history_corruption_is_a_clean_error() {
        let (_, _, history) = sample_history();
        let bytes = history.save();
        // Truncation.
        assert!(History::load(&bytes[..bytes.len() - 5]).is_err());
        assert!(History::load(&bytes[..20]).is_err());
        // A flipped byte in the body breaks the checksum.
        let mut flipped = bytes.clone();
        flipped[30] ^= 0xff;
        assert!(matches!(
            History::load(&flipped),
            Err(TalkError::CorruptHistory(_))
        ));
        // A flipped byte in the checksum itself.
        let mut bad_checksum = bytes.clone();
        let last = bad_checksum.len() - 1;
        bad_checksum[last] ^= 0xff;
        assert!(matches!(
            History::load(&bad_checksum),
            Err(TalkError::CorruptHistory(_))
        ));
        // Trailing bytes.
        let mut trailing = bytes.clone();
        trailing.extend_from_slice(b"junk");
        assert!(History::load(&trailing).is_err());
        // Garbage.
        assert!(History::load(b"not a history log").is_err());
        assert!(History::load(b"").is_err());
        // An entry with an unknown kind: build a valid log, then flip
        // the first entry's kind byte and fix the checksum so the only
        // failure is the unknown kind.
        let mut tampered = bytes.clone();
        // Locate the first entry kind byte: header is 7 + 1 + 32 + 2 +
        // name len (5 for "alice") + 4 = 51.
        tampered[51] = 9;
        let fixed = {
            let body = &tampered[..tampered.len() - 32];
            let sum = sha256(body);
            let mut t = tampered[..tampered.len() - 32].to_vec();
            t.extend_from_slice(&sum);
            t
        };
        assert!(matches!(
            History::load(&fixed),
            Err(TalkError::CorruptHistory(_))
        ));
    }

    #[test]
    fn history_file_round_trip() {
        let (_, _, history) = sample_history();
        let dir = std::env::temp_dir().join("k-talk-history-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("history.bin");
        history.save_to_file(&path).unwrap();
        let loaded = History::load_from_file(&path).unwrap();
        assert_eq!(loaded.entries(), history.entries());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn receipt_encode_decode_verify_and_tamper() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let msg = Message::new(&alice, bob.peer_id(), "hello", 0);
        let receipt = Receipt::new(&bob, msg.id(), ReceiptStatus::Read, 1_700_000_000);
        assert!(receipt.verify());
        let decoded = Receipt::decode(&receipt.encode()).unwrap();
        assert_eq!(decoded, receipt);
        assert!(decoded.verify());
        assert!(Receipt::decode(b"garbage").is_err());
        // A tampered status fails verification.
        let mut tampered = receipt.clone();
        tampered.status = ReceiptStatus::Delivered;
        assert!(!tampered.verify());
        // A tampered message id fails verification.
        let mut tampered2 = receipt.clone();
        tampered2.message_id[0] ^= 0xff;
        assert!(!tampered2.verify());
    }

    #[test]
    fn receipts_follow_their_message_conversation() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut history = History::new(alice.peer_id().clone());
        let msg = Message::new(&alice, bob.peer_id(), "hello", 0);
        let id = msg.id();
        let msg_ts = msg.timestamp;
        history.append_direct(msg);
        let receipt = Receipt::new(&bob, id, ReceiptStatus::Delivered, msg_ts + 1);
        history.append_receipt(receipt);
        let conv = ConversationId::Direct(bob.peer_id().clone());
        let entries = history.entries_for(&conv, 0);
        assert_eq!(entries.len(), 2);
        assert!(matches!(entries[1], HistoryEntry::Receipt(_)));
        // find_message resolves the receipt's message.
        assert!(matches!(
            history.find_message(&id),
            Some(FoundMessage::Direct(_))
        ));
        assert!(history.find_message(&[9u8; 32]).is_none());
    }
}
