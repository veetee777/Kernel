//! K•Chain — a minimal blockchain on top of KERNEL crypto and storage.
//!
//! A chain is a list of blocks. A block holds an index, the hash of
//! the previous block, a timestamp, and a list of signed transactions.
//! A transaction is a signed payload: sender, recipient, a
//! human-readable memo, and an amount.
//!
//! The rules are simple and deterministic:
//!
//! * A block's hash is the SHA-256 of its contents (index, previous
//!   hash, timestamp, transaction hashes). Any change to a block
//!   changes its hash, so tampering is detected by recomputation.
//! * A block links to its predecessor: its `prev_hash` must equal the
//!   previous block's hash. A broken link fails validation.
//! * Every transaction's signature must verify against its sender.
//!   A transaction with a bad signature is refused before it can
//!   enter a block.
//! * Fork choice is longest valid chain wins. Ties go to the first
//!   chain seen, so the choice is deterministic.
//!
//! Blocks are stored as blocks in the content store (see
//! `kernel_storage`), addressed by CID, so a block can be fetched and
//! verified independently of the chain that holds it.
//!
//! Transaction format (explicit binary, no JSON):
//! `KTX1` + sender public key (32) + recipient public key (32) +
//! memo length (u16 LE) + memo bytes + amount (u64 LE) +
//! signature (64).
//!
//! Block format (explicit binary, no JSON):
//! `KBLOCK1` + index (u64 LE) + previous hash (32) + timestamp
//! (u64 LE) + block hash (32) + transaction count (u32 LE) + for each
//! transaction: length (u32 LE) + transaction bytes.

use std::fmt;

use kernel_crypto::{sha256, Ed25519Signature};
use kernel_identity::{Identity, IdentityError, PeerId};
use kernel_storage::{Cid, ContentStore, StorageError};

/// Magic bytes at the start of every transaction.
const TX_MAGIC: &[u8; 4] = b"KTX1";

/// Magic bytes at the start of every block.
const BLOCK_MAGIC: &[u8; 7] = b"KBLOCK1";

/// Domain separator for transaction signatures: a signature covers
/// `b"KTX1" || sender key || recipient key || memo || amount`.
const TX_SIGN_DOMAIN: &[u8] = b"KTX1";

/// Domain separator for block hashes: a block hash covers
/// `b"KBLOCK1" || index || prev hash || timestamp || tx hashes`.
const BLOCK_HASH_DOMAIN: &[u8] = b"KBLOCK1";

/// The previous hash of the genesis block: all zeros, because there
/// is no block before it.
pub const GENESIS_PREV_HASH: [u8; 32] = [0u8; 32];

/// Errors returned by chain operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainError {
    /// Wraps a [`StorageError`].
    Storage(StorageError),
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// A transaction memo broke the rules.
    InvalidMemo(String),
    /// A transaction was malformed.
    InvalidTransaction(String),
    /// A block was malformed.
    InvalidBlock(String),
    /// A transaction's signature did not verify against its sender.
    BadTransactionSignature,
    /// A block's stored hash does not match its recomputed hash.
    BadBlockHash {
        /// The block's index.
        index: u64,
    },
    /// A block's previous hash does not match the previous block.
    BrokenLink {
        /// The block's index.
        index: u64,
        /// The previous hash the block claims.
        expected: [u8; 32],
        /// The previous block's actual hash.
        actual: [u8; 32],
    },
    /// The genesis block is malformed (wrong index or prev hash).
    BadGenesis,
    /// No block with this CID is known here.
    BlockNotFound(Cid),
}

impl fmt::Display for ChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChainError::Storage(e) => write!(f, "storage error: {e}"),
            ChainError::Identity(e) => write!(f, "identity error: {e}"),
            ChainError::InvalidMemo(msg) => write!(f, "invalid memo: {msg}"),
            ChainError::InvalidTransaction(msg) => write!(f, "invalid transaction: {msg}"),
            ChainError::InvalidBlock(msg) => write!(f, "invalid block: {msg}"),
            ChainError::BadTransactionSignature => {
                write!(f, "transaction signature is invalid")
            }
            ChainError::BadBlockHash { index } => {
                write!(f, "block {index} hash does not match its contents")
            }
            ChainError::BrokenLink {
                index,
                expected,
                actual,
            } => write!(
                f,
                "block {index} prev hash {} does not match previous block hash {}",
                kernel_crypto::to_hex(expected),
                kernel_crypto::to_hex(actual)
            ),
            ChainError::BadGenesis => write!(f, "genesis block is malformed"),
            ChainError::BlockNotFound(cid) => write!(f, "block not found: {cid}"),
        }
    }
}

impl std::error::Error for ChainError {}

impl From<StorageError> for ChainError {
    fn from(e: StorageError) -> Self {
        ChainError::Storage(e)
    }
}

impl From<IdentityError> for ChainError {
    fn from(e: IdentityError) -> Self {
        ChainError::Identity(e)
    }
}

/// Rules for memos: at most 128 printable characters. A memo may be
/// empty (a plain payment needs no note).
fn validate_memo(memo: &str) -> Result<(), ChainError> {
    if memo.len() > 128 {
        return Err(ChainError::InvalidMemo(
            "memo must be at most 128 characters".to_string(),
        ));
    }
    if !memo.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
        return Err(ChainError::InvalidMemo(
            "memo must be printable ASCII".to_string(),
        ));
    }
    Ok(())
}

/// The message a sender signs for a transaction: domain + sender key
/// + recipient key + memo + amount.
fn tx_sign_message(sender: &PeerId, recipient: &PeerId, memo: &str, amount: u64) -> Vec<u8> {
    let mut msg = Vec::with_capacity(TX_SIGN_DOMAIN.len() + 32 + 32 + memo.len() + 8);
    msg.extend_from_slice(TX_SIGN_DOMAIN);
    msg.extend_from_slice(sender.public_key().as_bytes());
    msg.extend_from_slice(recipient.public_key().as_bytes());
    msg.extend_from_slice(memo.as_bytes());
    msg.extend_from_slice(&amount.to_le_bytes());
    msg
}

/// A signed transaction: sender, recipient, memo, amount.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transaction {
    /// Who is sending.
    pub sender: PeerId,
    /// Who is receiving.
    pub recipient: PeerId,
    /// A human-readable note, e.g. `"coffee"`.
    pub memo: String,
    /// The amount, in the chain's smallest unit.
    pub amount: u64,
    /// The sender's signature over sender, recipient, memo, amount.
    pub signature: Ed25519Signature,
}

impl Transaction {
    /// Create and sign a transaction as `sender`.
    pub fn new(
        sender: &Identity,
        recipient: &PeerId,
        memo: &str,
        amount: u64,
    ) -> Result<Self, ChainError> {
        validate_memo(memo)?;
        let sender_id = sender.peer_id().clone();
        let msg = tx_sign_message(&sender_id, recipient, memo, amount);
        let signature = sender.sign(&msg);
        Ok(Self {
            sender: sender_id,
            recipient: recipient.clone(),
            memo: memo.to_string(),
            amount,
            signature,
        })
    }

    /// True if the signature verifies against the sender for this
    /// exact recipient, memo, and amount. Any tampering fails this.
    pub fn verify(&self) -> bool {
        let msg = tx_sign_message(&self.sender, &self.recipient, &self.memo, self.amount);
        self.sender.verify(&msg, &self.signature).is_ok()
    }

    /// The transaction's hash: the SHA-256 of its canonical encoding.
    /// Used to bind transactions into a block hash.
    pub fn hash(&self) -> [u8; 32] {
        sha256(&self.encode())
    }

    /// Encode to the explicit binary transaction format.
    pub fn encode(&self) -> Vec<u8> {
        let memo_bytes = self.memo.as_bytes();
        let mut out = Vec::with_capacity(4 + 32 + 32 + 2 + memo_bytes.len() + 8 + 64);
        out.extend_from_slice(TX_MAGIC);
        out.extend_from_slice(self.sender.public_key().as_bytes());
        out.extend_from_slice(self.recipient.public_key().as_bytes());
        out.extend_from_slice(&(memo_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(memo_bytes);
        out.extend_from_slice(&self.amount.to_le_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a transaction produced by [`Transaction::encode`].
    /// Rejects anything that is not exactly that format.
    pub fn decode(bytes: &[u8]) -> Result<Self, ChainError> {
        if bytes.len() < 4 + 32 + 32 + 2 + 8 + 64 || &bytes[..4] != TX_MAGIC {
            return Err(ChainError::InvalidTransaction(
                "missing KTX1 magic".to_string(),
            ));
        }
        let memo_len =
            u16::from_le_bytes(bytes[68..70].try_into().expect("2-byte slice")) as usize;
        let expected = 4 + 32 + 32 + 2 + memo_len + 8 + 64;
        if bytes.len() != expected {
            return Err(ChainError::InvalidTransaction(format!(
                "transaction is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let mut off = 4;
        let mut sender_bytes = [0u8; 32];
        sender_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let sender_key = kernel_crypto::Ed25519PublicKey::from_bytes(&sender_bytes)
            .map_err(|e| ChainError::InvalidTransaction(e.to_string()))?;
        let mut recipient_bytes = [0u8; 32];
        recipient_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let recipient_key = kernel_crypto::Ed25519PublicKey::from_bytes(&recipient_bytes)
            .map_err(|e| ChainError::InvalidTransaction(e.to_string()))?;
        let memo_len =
            u16::from_le_bytes(bytes[off..off + 2].try_into().expect("2-byte slice")) as usize;
        off += 2;
        let memo = String::from_utf8(bytes[off..off + memo_len].to_vec()).map_err(|_| {
            ChainError::InvalidTransaction("transaction memo is not valid UTF-8".to_string())
        })?;
        validate_memo(&memo)?;
        off += memo_len;
        let amount = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        // Transactions store public keys, not display names:
        // verification needs the keys. Decode rebuilds the PeerIds
        // with deterministic names derived from the keys.
        let sender = PeerId::new(sender_key, &fingerprint_name(&sender_key))?;
        let recipient = PeerId::new(recipient_key, &fingerprint_name(&recipient_key))?;
        Ok(Self {
            sender,
            recipient,
            memo,
            amount,
            signature,
        })
    }
}

/// A display name derived from a public key: the first 8 hex chars of
/// the key. Used when a transaction is decoded and only the key
/// survives (verification needs the key, not the human name).
fn fingerprint_name(key: &kernel_crypto::Ed25519PublicKey) -> String {
    let hex = kernel_crypto::to_hex(key.as_bytes());
    format!("peer-{}", &hex[..8])
}

/// Compute a block's hash: SHA-256 over domain + index + prev hash +
/// timestamp + the hash of every transaction, in order.
fn compute_block_hash(
    index: u64,
    prev_hash: &[u8; 32],
    timestamp: u64,
    transactions: &[Transaction],
) -> [u8; 32] {
    let mut msg = Vec::new();
    msg.extend_from_slice(BLOCK_HASH_DOMAIN);
    msg.extend_from_slice(&index.to_le_bytes());
    msg.extend_from_slice(prev_hash);
    msg.extend_from_slice(&timestamp.to_le_bytes());
    for tx in transactions {
        msg.extend_from_slice(&tx.hash());
    }
    sha256(&msg)
}

/// One block in the chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// Block height, starting at 0 for genesis.
    pub index: u64,
    /// Hash of the previous block (`GENESIS_PREV_HASH` for genesis).
    pub prev_hash: [u8; 32],
    /// Unix timestamp (seconds) the block was made.
    pub timestamp: u64,
    /// The block's transactions, in order.
    pub transactions: Vec<Transaction>,
    /// The block's hash, computed over its contents.
    pub hash: [u8; 32],
}

impl Block {
    /// Build a block and compute its hash. Every transaction's
    /// signature is verified first; a bad transaction is refused.
    pub fn new(
        index: u64,
        prev_hash: [u8; 32],
        timestamp: u64,
        transactions: Vec<Transaction>,
    ) -> Result<Self, ChainError> {
        for tx in &transactions {
            if !tx.verify() {
                return Err(ChainError::BadTransactionSignature);
            }
        }
        let hash = compute_block_hash(index, &prev_hash, timestamp, &transactions);
        Ok(Self {
            index,
            prev_hash,
            timestamp,
            transactions,
            hash,
        })
    }

    /// True if the stored hash matches a recomputation over the
    /// block's contents and every transaction signature verifies.
    /// Any tampering — to the index, prev hash, timestamp,
    /// transactions, or the stored hash itself — fails this check.
    pub fn verify(&self) -> bool {
        if compute_block_hash(self.index, &self.prev_hash, self.timestamp, &self.transactions)
            != self.hash
        {
            return false;
        }
        self.transactions.iter().all(|tx| tx.verify())
    }

    /// Encode to the explicit binary block format.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(BLOCK_MAGIC);
        out.extend_from_slice(&self.index.to_le_bytes());
        out.extend_from_slice(&self.prev_hash);
        out.extend_from_slice(&self.timestamp.to_le_bytes());
        out.extend_from_slice(&self.hash);
        out.extend_from_slice(&(self.transactions.len() as u32).to_le_bytes());
        for tx in &self.transactions {
            let tx_bytes = tx.encode();
            out.extend_from_slice(&(tx_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&tx_bytes);
        }
        out
    }

    /// Decode a block produced by [`Block::encode`].
    /// Rejects anything that is not exactly that format. Decoding
    /// does not verify the block — call [`Block::verify`] for that.
    pub fn decode(bytes: &[u8]) -> Result<Self, ChainError> {
        if bytes.len() < 7 + 8 + 32 + 8 + 32 + 4 || &bytes[..7] != BLOCK_MAGIC {
            return Err(ChainError::InvalidBlock("missing KBLOCK1 magic".to_string()));
        }
        let mut off = 7;
        let index = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let mut prev_hash = [0u8; 32];
        prev_hash.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let timestamp =
            u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let count = u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4-byte slice")) as usize;
        off += 4;
        let mut transactions = Vec::with_capacity(count);
        for _ in 0..count {
            if bytes.len() < off + 4 {
                return Err(ChainError::InvalidBlock(
                    "block truncated inside transactions".to_string(),
                ));
            }
            let tx_len =
                u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4-byte slice")) as usize;
            off += 4;
            if bytes.len() < off + tx_len {
                return Err(ChainError::InvalidBlock(
                    "block truncated inside a transaction".to_string(),
                ));
            }
            transactions.push(Transaction::decode(&bytes[off..off + tx_len])?);
            off += tx_len;
        }
        if off != bytes.len() {
            return Err(ChainError::InvalidBlock(
                "trailing bytes after last transaction".to_string(),
            ));
        }
        Ok(Self {
            index,
            prev_hash,
            timestamp,
            transactions,
            hash,
        })
    }
}

/// A chain: an ordered list of blocks plus the content store holding
/// each block's encoded bytes, addressed by CID.
pub struct Chain {
    store: ContentStore,
    blocks: Vec<Block>,
    /// The content-store CID of each block's encoded bytes,
    /// in the same order as `blocks`.
    block_cids: Vec<Cid>,
}

impl Chain {
    /// A new chain holding only its genesis block: index 0, no
    /// transactions, zero previous hash, the given timestamp.
    pub fn genesis(timestamp: u64) -> Self {
        let genesis = Block::new(0, GENESIS_PREV_HASH, timestamp, Vec::new())
            .expect("genesis block has no transactions to reject");
        let mut chain = Self {
            store: ContentStore::new(),
            blocks: Vec::new(),
            block_cids: Vec::new(),
        };
        let cid = chain.store.put(&genesis.encode());
        chain.blocks.push(genesis);
        chain.block_cids.push(cid);
        chain
    }

    /// Number of blocks in the chain (genesis counts as 1).
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// True if the chain holds only its genesis block.
    pub fn is_genesis_only(&self) -> bool {
        self.blocks.len() == 1
    }

    /// The hash of the latest block.
    pub fn latest_hash(&self) -> [u8; 32] {
        self.blocks
            .last()
            .expect("a chain always holds at least its genesis block")
            .hash
    }

    /// The block at an index, if it exists.
    pub fn get_block(&self, index: u64) -> Option<&Block> {
        self.blocks.get(index as usize)
    }

    /// The content-store CID of the block at an index, if it exists.
    pub fn block_cid(&self, index: u64) -> Option<Cid> {
        self.block_cids.get(index as usize).copied()
    }

    /// Fetch a block by its content-store CID and verify it. The
    /// decoded block's hash and transaction signatures are checked,
    /// so a corrupted stored block fails loudly.
    pub fn get_block_by_cid(&self, cid: &Cid) -> Result<Block, ChainError> {
        let bytes = self.store.get(cid).map_err(|e| match e {
            StorageError::NotFound(c) => ChainError::BlockNotFound(c),
            other => ChainError::Storage(other),
        })?;
        let block = Block::decode(&bytes)?;
        if !block.verify() {
            return Err(ChainError::BadBlockHash { index: block.index });
        }
        Ok(block)
    }

    /// Append a block with the given transactions and timestamp.
    /// The block links to the latest block's hash; every transaction
    /// is verified before the block is built, so a bad transaction
    /// is refused and the chain is unchanged. Returns the new block's
    /// content-store CID.
    pub fn add_block(
        &mut self,
        transactions: Vec<Transaction>,
        timestamp: u64,
    ) -> Result<Cid, ChainError> {
        let index = self.blocks.len() as u64;
        let prev_hash = self.latest_hash();
        let block = Block::new(index, prev_hash, timestamp, transactions)?;
        let cid = self.store.put(&block.encode());
        self.blocks.push(block);
        self.block_cids.push(cid);
        Ok(cid)
    }

    /// Verify the whole chain: the genesis block must be index 0
    /// with a zero previous hash; every later block must link to its
    /// predecessor and must verify (hash and transaction signatures).
    /// Returns `Ok(())` for a valid chain, or the first problem found.
    pub fn validate_chain(&self) -> Result<(), ChainError> {
        let first = self
            .blocks
            .first()
            .expect("a chain always holds at least its genesis block");
        if first.index != 0 || first.prev_hash != GENESIS_PREV_HASH {
            return Err(ChainError::BadGenesis);
        }
        if !first.verify() {
            return Err(ChainError::BadBlockHash { index: 0 });
        }
        for (i, block) in self.blocks.iter().enumerate().skip(1) {
            let prev = &self.blocks[i - 1];
            if block.prev_hash != prev.hash {
                return Err(ChainError::BrokenLink {
                    index: block.index,
                    expected: block.prev_hash,
                    actual: prev.hash,
                });
            }
            if !block.verify() {
                // A bad transaction signature is the root cause and
                // is reported first; if every signature verifies but
                // the hash still mismatches, the stored hash itself
                // was tampered with.
                if block.transactions.iter().any(|tx| !tx.verify()) {
                    return Err(ChainError::BadTransactionSignature);
                }
                return Err(ChainError::BadBlockHash { index: block.index });
            }
        }
        Ok(())
    }

    /// True if [`Chain::validate_chain`] succeeds.
    pub fn is_valid(&self) -> bool {
        self.validate_chain().is_ok()
    }

    /// Mutable access to the block list. Test-only: lets tests
    /// simulate tampering after a block was added.
    #[cfg(test)]
    pub fn blocks_mut(&mut self) -> &mut Vec<Block> {
        &mut self.blocks
    }

    /// Mutable access to the underlying content store. Test-only.
    #[cfg(test)]
    pub fn store_mut(&mut self) -> &mut ContentStore {
        &mut self.store
    }
}

/// Fork choice: the longest valid chain wins. Invalid chains are
/// ignored entirely, however long they are. On a tie, the first
/// chain in the list wins, so the choice is deterministic. Returns
/// `None` if no chain in the list is valid.
pub fn longest_valid<'a>(chains: &[&'a Chain]) -> Option<&'a Chain> {
    chains
        .iter()
        .copied()
        .filter(|c| c.is_valid())
        .max_by_key(|c| c.len())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tx(alice: &Identity, bob: &PeerId, memo: &str, amount: u64) -> Transaction {
        Transaction::new(alice, bob, memo, amount).unwrap()
    }

    #[test]
    fn transaction_encode_decode_round_trip_and_verify() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let t = tx(&alice, bob.peer_id(), "coffee", 50);
        assert!(t.verify());
        let encoded = t.encode();
        let decoded = Transaction::decode(&encoded).unwrap();
        assert_eq!(decoded.memo, "coffee");
        assert_eq!(decoded.amount, 50);
        assert_eq!(
            decoded.sender.public_key(),
            alice.peer_id().public_key()
        );
        assert_eq!(
            decoded.recipient.public_key(),
            bob.peer_id().public_key()
        );
        assert_eq!(decoded.signature, t.signature);
        assert!(decoded.verify());
        assert!(Transaction::decode(b"garbage").is_err());
        assert!(Transaction::decode(&encoded[..encoded.len() - 1]).is_err());
        // Tampering with the memo breaks the signature.
        let mut tampered = t.clone();
        tampered.memo = "diamonds".to_string();
        assert!(!tampered.verify());
        // Tampering with the amount breaks it too.
        let mut tampered = t.clone();
        tampered.amount = 5_000;
        assert!(!tampered.verify());
    }

    #[test]
    fn genesis_and_blocks_link_correctly() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut chain = Chain::genesis(1_700_000_000);
        assert_eq!(chain.len(), 1);
        assert!(chain.is_genesis_only());
        assert_eq!(chain.get_block(0).unwrap().index, 0);
        assert_eq!(chain.get_block(0).unwrap().prev_hash, GENESIS_PREV_HASH);
        assert!(chain.get_block(0).unwrap().transactions.is_empty());
        assert!(chain.is_valid());

        chain
            .add_block(vec![tx(&alice, bob.peer_id(), "one", 10)], 1_700_000_001)
            .unwrap();
        chain
            .add_block(vec![tx(&alice, bob.peer_id(), "two", 20)], 1_700_000_002)
            .unwrap();
        assert_eq!(chain.len(), 3);
        assert!(!chain.is_genesis_only());
        // Each block links to the previous block's hash.
        assert_eq!(
            chain.get_block(1).unwrap().prev_hash,
            chain.get_block(0).unwrap().hash
        );
        assert_eq!(
            chain.get_block(2).unwrap().prev_hash,
            chain.get_block(1).unwrap().hash
        );
        assert_eq!(chain.get_block(2).unwrap().index, 2);
        assert!(chain.validate_chain().is_ok());
    }

    #[test]
    fn empty_transaction_block_is_valid() {
        let mut chain = Chain::genesis(100);
        chain.add_block(Vec::new(), 101).unwrap();
        chain.add_block(Vec::new(), 102).unwrap();
        assert_eq!(chain.len(), 3);
        assert!(chain.is_valid());
        assert!(chain.get_block(1).unwrap().transactions.is_empty());
    }

    #[test]
    fn tampered_transaction_breaks_validation() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut chain = Chain::genesis(100);
        chain
            .add_block(vec![tx(&alice, bob.peer_id(), "real", 10)], 101)
            .unwrap();
        assert!(chain.is_valid());
        // Tamper with the transaction's memo after the block was made.
        chain.blocks_mut()[1].transactions[0].memo = "forged".to_string();
        assert!(!chain.get_block(1).unwrap().verify());
        assert_eq!(
            chain.validate_chain(),
            Err(ChainError::BadTransactionSignature)
        );
    }

    #[test]
    fn tampered_prev_hash_breaks_validation() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut chain = Chain::genesis(100);
        chain
            .add_block(vec![tx(&alice, bob.peer_id(), "one", 10)], 101)
            .unwrap();
        chain
            .add_block(vec![tx(&alice, bob.peer_id(), "two", 20)], 102)
            .unwrap();
        assert!(chain.is_valid());
        // Break block 2's link to block 1.
        chain.blocks_mut()[2].prev_hash = [9u8; 32];
        match chain.validate_chain() {
            Err(ChainError::BrokenLink { index, .. }) => assert_eq!(index, 2),
            other => panic!("expected BrokenLink, got {other:?}"),
        }
        // And a tampered stored hash is caught too.
        chain.blocks_mut()[2].prev_hash = chain.get_block(1).unwrap().hash;
        chain.blocks_mut()[2].hash = [7u8; 32];
        assert_eq!(
            chain.validate_chain(),
            Err(ChainError::BadBlockHash { index: 2 })
        );
    }

    #[test]
    fn bad_signature_transaction_is_refused_on_add() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut chain = Chain::genesis(100);
        let mut bad = tx(&alice, bob.peer_id(), "real", 10);
        bad.amount = 10_000; // changed after signing
        assert!(!bad.verify());
        assert_eq!(
            chain.add_block(vec![bad], 101),
            Err(ChainError::BadTransactionSignature)
        );
        // The chain is unchanged: still genesis only.
        assert_eq!(chain.len(), 1);
        assert!(chain.is_valid());
    }

    #[test]
    fn longest_valid_chain_wins_a_fork() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        // Chain A: genesis + 2 blocks.
        let mut a = Chain::genesis(100);
        a.add_block(vec![tx(&alice, bob.peer_id(), "a1", 1)], 101)
            .unwrap();
        a.add_block(vec![tx(&alice, bob.peer_id(), "a2", 2)], 102)
            .unwrap();
        // Chain B: genesis + 3 blocks — the longest valid chain.
        let mut b = Chain::genesis(100);
        b.add_block(vec![tx(&alice, bob.peer_id(), "b1", 1)], 101)
            .unwrap();
        b.add_block(vec![tx(&alice, bob.peer_id(), "b2", 2)], 102)
            .unwrap();
        b.add_block(vec![tx(&alice, bob.peer_id(), "b3", 3)], 103)
            .unwrap();
        // Chain C: genesis + 4 blocks, but the last one is tampered —
        // longer, yet invalid, so it must lose.
        let mut c = Chain::genesis(100);
        c.add_block(vec![tx(&alice, bob.peer_id(), "c1", 1)], 101)
            .unwrap();
        c.add_block(vec![tx(&alice, bob.peer_id(), "c2", 2)], 102)
            .unwrap();
        c.add_block(vec![tx(&alice, bob.peer_id(), "c3", 3)], 103)
            .unwrap();
        c.add_block(vec![tx(&alice, bob.peer_id(), "c4", 4)], 104)
            .unwrap();
        c.blocks_mut()[4].transactions[0].memo = "forged".to_string();
        assert!(!c.is_valid());
        assert_eq!(longest_valid(&[&a, &b, &c]).map(|ch| ch.len()), Some(4));
        // The winner is B, not A and not the longer-but-invalid C.
        let winner = longest_valid(&[&a, &b, &c]).unwrap();
        assert_eq!(winner.get_block(3).unwrap().transactions[0].memo, "b3");
        // A tie goes to the first chain in the list.
        let mut a2 = Chain::genesis(100);
        a2.add_block(vec![tx(&alice, bob.peer_id(), "a1", 1)], 101)
            .unwrap();
        a2.add_block(vec![tx(&alice, bob.peer_id(), "a2", 2)], 102)
            .unwrap();
        let winner = longest_valid(&[&a, &a2]).unwrap();
        assert_eq!(
            winner.get_block(1).unwrap().transactions[0].memo,
            "a1"
        );
        // No valid chains -> no winner.
        assert!(longest_valid(&[&c]).is_none());
        assert!(longest_valid(&[]).is_none());
    }

    #[test]
    fn blocks_are_stored_and_retrievable_by_cid() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut chain = Chain::genesis(100);
        let cid1 = chain
            .add_block(vec![tx(&alice, bob.peer_id(), "one", 10)], 101)
            .unwrap();
        let cid2 = chain
            .add_block(vec![tx(&alice, bob.peer_id(), "two", 20)], 102)
            .unwrap();
        assert_ne!(cid1, cid2);
        assert_eq!(chain.block_cid(1), Some(cid1));
        let fetched = chain.get_block_by_cid(&cid2).unwrap();
        assert_eq!(fetched.index, 2);
        assert_eq!(fetched.transactions[0].memo, "two");
        assert!(fetched.verify());
        // An unknown CID is a clean error.
        let missing = Cid::from_bytes(kernel_crypto::sha256(b"no block here"));
        assert_eq!(
            chain.get_block_by_cid(&missing),
            Err(ChainError::BlockNotFound(missing))
        );
    }

    #[test]
    fn block_encode_decode_round_trip() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let block = Block::new(
            5,
            [3u8; 32],
            1_700_000_000,
            vec![tx(&alice, bob.peer_id(), "a", 1), tx(&alice, bob.peer_id(), "b", 2)],
        )
        .unwrap();
        assert!(block.verify());
        let encoded = block.encode();
        let decoded = Block::decode(&encoded).unwrap();
        // Whole-struct equality would fail on display names: decoding
        // rebuilds PeerIds with deterministic key-derived names
        // ("peer-..."), while the originals are "alice"/"bob". The
        // keys, signatures, and hashes are what authenticate a block,
        // so compare those.
        assert_eq!(decoded.index, block.index);
        assert_eq!(decoded.prev_hash, block.prev_hash);
        assert_eq!(decoded.timestamp, block.timestamp);
        assert_eq!(decoded.hash, block.hash);
        assert_eq!(decoded.transactions.len(), block.transactions.len());
        for (d, o) in decoded.transactions.iter().zip(block.transactions.iter()) {
            assert_eq!(d.memo, o.memo);
            assert_eq!(d.amount, o.amount);
            assert_eq!(d.sender.public_key(), o.sender.public_key());
            assert_eq!(d.recipient.public_key(), o.recipient.public_key());
            assert_eq!(d.signature, o.signature);
        }
        assert!(decoded.verify());
        assert!(Block::decode(b"garbage").is_err());
        assert!(Block::decode(&encoded[..encoded.len() - 1]).is_err());
        // An empty-transaction block round-trips too (no PeerIds, so
        // whole-struct equality holds here).
        let empty = Block::new(0, GENESIS_PREV_HASH, 0, Vec::new()).unwrap();
        assert_eq!(Block::decode(&empty.encode()).unwrap(), empty);
    }

    #[test]
    fn memo_rules() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        assert!(Transaction::new(&alice, bob.peer_id(), "", 1).is_ok()); // empty memo ok
        assert!(Transaction::new(&alice, bob.peer_id(), "coffee", 1).is_ok());
        assert!(Transaction::new(&alice, bob.peer_id(), "a".repeat(129).as_str(), 1).is_err());
    }
}
