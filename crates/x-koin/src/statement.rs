//! Account statements for X•Koin.
//!
//! A statement is a peer's history with the ledger: every
//! transaction in or out, in acceptance order, with the running
//! balance after each entry, plus totals in and out. Statements
//! cover ordinary transfers and allowance draws alike, so a peer
//! who spent from someone else's allowance sees those draws as
//! entries too.
//!
//! A statement exports to an explicit binary format (no JSON) and
//! round-trips exactly, so a peer can hand a statement to anyone
//! and the reader can check it byte for byte.

use kernel_crypto::Ed25519PublicKey;
use kernel_identity::PeerId;

use crate::{fingerprint_name, KoinError};

/// Magic bytes at the start of every statement encoding.
const STATEMENT_MAGIC: &[u8; 11] = b"XKOIN-STMT1";

/// Minimum bytes of one encoded entry: seq (8) + id (32) +
/// counterparty (32) + amount (8) + balance_after (8) + kind (1) +
/// memo length (2). Used to bound the entry count on decode.
const MIN_ENTRY_BYTES: usize = 91;

/// What kind of ledger movement an entry records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryKind {
    /// An ordinary signed transfer.
    Transfer = 0,
    /// A draw from an allowance.
    AllowanceDraw = 1,
}

impl EntryKind {
    fn from_byte(b: u8) -> Result<Self, KoinError> {
        match b {
            0 => Ok(EntryKind::Transfer),
            1 => Ok(EntryKind::AllowanceDraw),
            other => Err(KoinError::InvalidStatement(format!(
                "unknown entry kind byte {other}"
            ))),
        }
    }
}

/// One line of a statement: a single movement in or out, with the
/// balance after it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatementEntry {
    /// Ledger acceptance order, shared with transfers and draws.
    pub seq: u64,
    /// The id of the transfer (SHA-256 of its encoding) or draw.
    pub id: [u8; 32],
    /// The other party to the movement.
    pub counterparty: PeerId,
    /// The memo on the transfer or draw.
    pub memo: String,
    /// Signed amount: positive means coins came in, negative means
    /// coins went out.
    pub amount: i64,
    /// The peer's balance after this entry.
    pub balance_after: u64,
    /// What kind of movement this was.
    pub kind: EntryKind,
}

/// A peer's statement: entries in acceptance order, totals, and
/// the final balance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement {
    /// Whose statement this is.
    pub peer: PeerId,
    /// Every movement, in acceptance order, with running balances.
    pub entries: Vec<StatementEntry>,
    /// Total coins received across all entries.
    pub total_in: u64,
    /// Total coins sent across all entries.
    pub total_out: u64,
    /// The peer's balance after the last entry.
    pub final_balance: u64,
}

impl Statement {
    /// Build a statement from entries whose `balance_after` fields
    /// are already filled in, plus the peer's opening balance (the
    /// genesis allocation, or 0). Totals are computed here; the
    /// final balance is the last entry's balance, or the opening
    /// balance when there are no entries.
    pub fn build(peer: PeerId, entries: Vec<StatementEntry>, opening_balance: u64) -> Self {
        let mut total_in = 0u64;
        let mut total_out = 0u64;
        for entry in &entries {
            if entry.amount >= 0 {
                total_in = total_in.saturating_add(entry.amount as u64);
            } else {
                total_out = total_out.saturating_add(entry.amount.unsigned_abs());
            }
        }
        let final_balance = entries
            .last()
            .map(|e| e.balance_after)
            .unwrap_or(opening_balance);
        Self {
            peer,
            entries,
            total_in,
            total_out,
            final_balance,
        }
    }

    /// Encode to the explicit binary statement format.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(71);
        out.extend_from_slice(STATEMENT_MAGIC);
        out.extend_from_slice(self.peer.public_key().as_bytes());
        out.extend_from_slice(&self.total_in.to_le_bytes());
        out.extend_from_slice(&self.total_out.to_le_bytes());
        out.extend_from_slice(&self.final_balance.to_le_bytes());
        out.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for entry in &self.entries {
            out.extend_from_slice(&entry.seq.to_le_bytes());
            out.extend_from_slice(&entry.id);
            out.extend_from_slice(entry.counterparty.public_key().as_bytes());
            out.extend_from_slice(&entry.amount.to_le_bytes());
            out.extend_from_slice(&entry.balance_after.to_le_bytes());
            out.push(entry.kind as u8);
            let memo_bytes = entry.memo.as_bytes();
            out.extend_from_slice(&(memo_bytes.len() as u16).to_le_bytes());
            out.extend_from_slice(memo_bytes);
        }
        out
    }

    /// Decode a statement produced by [`Statement::encode`].
    /// Rejects anything that is not exactly that format.
    pub fn decode(bytes: &[u8]) -> Result<Self, KoinError> {
        let invalid = |msg: String| KoinError::InvalidStatement(msg);
        if bytes.len() < 71 || &bytes[..11] != STATEMENT_MAGIC {
            return Err(invalid("missing XKOIN-STMT1 magic".to_string()));
        }
        let mut off = 11;
        let mut peer_bytes = [0u8; 32];
        peer_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let total_in = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let total_out = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let final_balance =
            u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let count =
            u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4-byte slice")) as usize;
        off += 4;
        // Bound the count by what the remaining bytes could hold, so
        // a corrupt count cannot force a giant allocation.
        if count > (bytes.len() - off) / MIN_ENTRY_BYTES {
            return Err(invalid(format!(
                "statement claims {count} entries but is only {} bytes",
                bytes.len()
            )));
        }
        let peer_key =
            Ed25519PublicKey::from_bytes(&peer_bytes).map_err(|e| invalid(e.to_string()))?;
        let peer = PeerId::new(peer_key, &fingerprint_name(&peer_key))?;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            if bytes.len() - off < MIN_ENTRY_BYTES {
                return Err(invalid("truncated statement entry".to_string()));
            }
            let seq = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
            off += 8;
            let mut id = [0u8; 32];
            id.copy_from_slice(&bytes[off..off + 32]);
            off += 32;
            let mut counterparty_bytes = [0u8; 32];
            counterparty_bytes.copy_from_slice(&bytes[off..off + 32]);
            off += 32;
            let amount = i64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
            off += 8;
            let balance_after =
                u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
            off += 8;
            let kind = EntryKind::from_byte(bytes[off])?;
            off += 1;
            let memo_len =
                u16::from_le_bytes(bytes[off..off + 2].try_into().expect("2-byte slice")) as usize;
            off += 2;
            if bytes.len() - off < memo_len {
                return Err(invalid("truncated statement memo".to_string()));
            }
            let memo = String::from_utf8(bytes[off..off + memo_len].to_vec())
                .map_err(|_| invalid("statement memo is not valid UTF-8".to_string()))?;
            off += memo_len;
            let counterparty_key = Ed25519PublicKey::from_bytes(&counterparty_bytes)
                .map_err(|e| invalid(e.to_string()))?;
            let counterparty = PeerId::new(counterparty_key, &fingerprint_name(&counterparty_key))?;
            entries.push(StatementEntry {
                seq,
                id,
                counterparty,
                memo,
                amount,
                balance_after,
                kind,
            });
        }
        if off != bytes.len() {
            return Err(invalid(format!(
                "statement has {} trailing bytes",
                bytes.len() - off
            )));
        }
        Ok(Self {
            peer,
            entries,
            total_in,
            total_out,
            final_balance,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(name: &str) -> PeerId {
        kernel_identity::Identity::generate(name)
            .unwrap()
            .peer_id()
            .clone()
    }

    fn entry(seq: u64, amount: i64, balance_after: u64) -> StatementEntry {
        StatementEntry {
            seq,
            id: [seq as u8; 32],
            counterparty: peer("counter"),
            memo: "test".to_string(),
            amount,
            balance_after,
            kind: EntryKind::Transfer,
        }
    }

    /// Compare two statements by everything the encoding carries:
    /// keys (not display names — decode rebuilds names from keys),
    /// entries, totals, and final balance.
    fn assert_same_statement(decoded: &Statement, expected: &Statement) {
        assert_eq!(decoded.peer.public_key(), expected.peer.public_key());
        assert_eq!(decoded.total_in, expected.total_in);
        assert_eq!(decoded.total_out, expected.total_out);
        assert_eq!(decoded.final_balance, expected.final_balance);
        assert_eq!(decoded.entries.len(), expected.entries.len());
        for (d, e) in decoded.entries.iter().zip(expected.entries.iter()) {
            assert_eq!(d.seq, e.seq);
            assert_eq!(d.id, e.id);
            assert_eq!(d.counterparty.public_key(), e.counterparty.public_key());
            assert_eq!(d.memo, e.memo);
            assert_eq!(d.amount, e.amount);
            assert_eq!(d.balance_after, e.balance_after);
            assert_eq!(d.kind, e.kind);
        }
    }

    #[test]
    fn build_computes_totals_and_final_balance() {
        let me = peer("me");
        let statement = Statement::build(
            me.clone(),
            vec![entry(0, -250, 750), entry(1, 50, 800), entry(2, -100, 700)],
            1_000,
        );
        assert_eq!(statement.total_in, 50);
        assert_eq!(statement.total_out, 350);
        assert_eq!(statement.final_balance, 700);
        // No entries: totals are zero and the final balance is the
        // opening balance.
        let empty = Statement::build(me, vec![], 42);
        assert_eq!(empty.total_in, 0);
        assert_eq!(empty.total_out, 0);
        assert_eq!(empty.final_balance, 42);
    }

    #[test]
    fn encode_decode_round_trip() {
        let me = peer("me");
        let statement = Statement::build(me, vec![entry(0, 100, 100), entry(5, -30, 70)], 0);
        let encoded = statement.encode();
        let decoded = Statement::decode(&encoded).unwrap();
        assert_same_statement(&decoded, &statement);
        assert!(Statement::decode(b"garbage").is_err());
        assert!(Statement::decode(&encoded[..encoded.len() - 1]).is_err());
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(Statement::decode(&trailing).is_err());
        // A bad kind byte is refused.
        let mut bad_kind = encoded.clone();
        // Kind byte of the first entry: header (71) + seq (8) + id
        // (32) + counterparty (32) + amount (8) + balance (8).
        bad_kind[71 + 8 + 32 + 32 + 8 + 8] = 9;
        assert!(Statement::decode(&bad_kind).is_err());
        // A huge claimed count is refused without allocating.
        let mut bad_count = encoded.clone();
        bad_count[67..71].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Statement::decode(&bad_count).is_err());
    }

    #[test]
    fn empty_statement_round_trips() {
        let me = peer("me");
        let statement = Statement::build(me, vec![], 0);
        assert_same_statement(&Statement::decode(&statement.encode()).unwrap(), &statement);
    }
}
