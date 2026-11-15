//! K•Chain — a blockchain on top of KERNEL crypto and storage.
//!
//! A chain is a list of blocks. A block holds an index, the hash of
//! the previous block, a timestamp, a nonce, and a list of signed
//! transactions. A transaction is a signed payload: sender,
//! recipient, a human-readable memo, and an amount.
//!
//! The rules are simple and deterministic:
//!
//! * A block's hash is the SHA-256 of its contents (index, previous
//!   hash, timestamp, nonce, transaction root). Any change to a block
//!   changes its hash, so tampering is detected by recomputation.
//!   The transaction root is the SHA-256 of the concatenated hashes
//!   of the block's transactions, in order.
//! * A block links to its predecessor: its `prev_hash` must equal the
//!   previous block's hash. A broken link fails validation.
//! * Every transaction's signature must verify against its sender.
//!   A transaction with a bad signature is refused before it can
//!   enter a block.
//! * Proof of work: a block's hash must have at least `difficulty`
//!   leading zero bits. Difficulty is caller-set on the chain
//!   (default 0, meaning no work required); it never auto-adjusts.
//!   Mining searches nonces until the hash meets the target.
//! * Timestamps never go backwards along a chain, and a transaction
//!   may appear exactly once in the whole chain.
//! * Fork choice is longest valid chain wins. Ties go to the first
//!   chain seen, so the choice is deterministic.
//!
//! Blocks are stored as blocks in the content store (see
//! `kernel_storage`), addressed by CID, so a block can be fetched and
//! verified independently of the chain that holds it.
//!
//! A [`Mempool`] holds pending transactions: deduped by transaction
//! hash, capped (oldest evicted first), handed out in insertion order
//! for mining. A [`ChainSet`] holds the current best chain plus
//! competing forks: a submitted block extends the tip it links to, or
//! forks from an earlier block; when a fork grows longer than the
//! current best, a reorg swaps it in and the orphaned transactions
//! return to the mempool unless they are already confirmed on the
//! new tip.
//!
//! A whole chain can be saved to disk as one binary file
//! ([`Chain::save`]) and loaded back exactly ([`Chain::load`]).
//! No JSON anywhere: the save format is `KCHAIN1` + difficulty +
//! block count + length-prefixed encoded blocks. A corrupt save is a
//! clean error, never a panic.
//!
//! Transaction format (explicit binary, no JSON):
//! `KTX1` + sender public key (32) + recipient public key (32) +
//! memo length (u16 LE) + memo bytes + amount (u64 LE) +
//! signature (64).
//!
//! Block format (explicit binary, no JSON):
//! `KBLOCK1` + index (u64 LE) + previous hash (32) + timestamp
//! (u64 LE) + nonce (u64 LE) + block hash (32) + transaction count
//! (u32 LE) + for each transaction: length (u32 LE) + transaction
//! bytes.
//!
//! Chain save format (explicit binary, no JSON):
//! `KCHAIN1` + difficulty (u8) + block count (u32 LE) + for each
//! block: length (u32 LE) + block bytes.

use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::path::Path;

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
    /// A block's proof of work does not meet the chain's difficulty.
    InsufficientPow {
        /// The block's index.
        index: u64,
        /// The difficulty the block failed to meet.
        difficulty: u8,
    },
    /// A block's timestamp goes backwards from its predecessor.
    NonMonotonicTimestamp {
        /// The block's index.
        index: u64,
        /// The block's timestamp.
        timestamp: u64,
        /// Its predecessor's timestamp.
        prev_timestamp: u64,
    },
    /// A transaction appears twice — twice in one block, or once in
    /// a block and once earlier in the chain.
    DuplicateTransaction {
        /// The block's index.
        index: u64,
        /// The repeated transaction's hash.
        tx_hash: [u8; 32],
    },
    /// A submitted block's parent is unknown to this chain set: it
    /// links to no known tip and no known block.
    UnknownParent {
        /// The previous hash the block claims.
        prev_hash: [u8; 32],
    },
    /// Mining searched the whole nonce space without finding a hash
    /// that meets the difficulty.
    MiningFailed,
    /// A saved chain file is corrupt: bad magic, truncation,
    /// trailing bytes, or a block that fails to decode.
    CorruptSave(String),
    /// A filesystem error while saving or loading a chain.
    Io(String),
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
            ChainError::InsufficientPow { index, difficulty } => write!(
                f,
                "block {index} proof of work does not meet difficulty {difficulty}"
            ),
            ChainError::NonMonotonicTimestamp {
                index,
                timestamp,
                prev_timestamp,
            } => write!(
                f,
                "block {index} timestamp {timestamp} goes backwards from {prev_timestamp}"
            ),
            ChainError::DuplicateTransaction { index, tx_hash } => write!(
                f,
                "block {index} repeats transaction {}",
                kernel_crypto::to_hex(tx_hash)
            ),
            ChainError::UnknownParent { prev_hash } => write!(
                f,
                "block parent {} is unknown",
                kernel_crypto::to_hex(prev_hash)
            ),
            ChainError::MiningFailed => {
                write!(f, "mining exhausted the nonce space without a valid nonce")
            }
            ChainError::CorruptSave(msg) => write!(f, "corrupt chain save: {msg}"),
            ChainError::Io(msg) => write!(f, "chain file I/O error: {msg}"),
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
        let memo_len = u16::from_le_bytes(bytes[68..70].try_into().expect("2-byte slice")) as usize;
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

/// The transaction root of a block: the SHA-256 of the concatenated
/// hashes of its transactions, in order. An empty block's root is
/// the SHA-256 of nothing. Any change to any transaction — or to
/// their order — changes the root, and therefore the block hash.
pub fn tx_root(transactions: &[Transaction]) -> [u8; 32] {
    let mut msg = Vec::with_capacity(32 * transactions.len());
    for tx in transactions {
        msg.extend_from_slice(&tx.hash());
    }
    sha256(&msg)
}

/// Count the leading zero bits of a hash. Used to check proof of
/// work: a block meets difficulty `d` when this is at least `d`.
fn leading_zero_bits(hash: &[u8; 32]) -> u32 {
    let mut count = 0u32;
    for byte in hash {
        if *byte == 0 {
            count += 8;
        } else {
            count += byte.leading_zeros();
            break;
        }
    }
    count
}

/// Compute a block's hash: SHA-256 over domain + index + prev hash +
/// timestamp + nonce + transaction root.
fn compute_block_hash(
    index: u64,
    prev_hash: &[u8; 32],
    timestamp: u64,
    nonce: u64,
    transactions: &[Transaction],
) -> [u8; 32] {
    let mut msg = Vec::new();
    msg.extend_from_slice(BLOCK_HASH_DOMAIN);
    msg.extend_from_slice(&index.to_le_bytes());
    msg.extend_from_slice(prev_hash);
    msg.extend_from_slice(&timestamp.to_le_bytes());
    msg.extend_from_slice(&nonce.to_le_bytes());
    msg.extend_from_slice(&tx_root(transactions));
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
    /// Proof-of-work nonce. The block hash covers this, so mining
    /// searches nonces until the hash meets the difficulty.
    pub nonce: u64,
    /// The block's transactions, in order.
    pub transactions: Vec<Transaction>,
    /// The block's hash, computed over its contents.
    pub hash: [u8; 32],
}

impl Block {
    /// Build a block and compute its hash. Every transaction's
    /// signature is verified first; a bad transaction is refused.
    /// The nonce is 0: this block carries no proof of work, so it
    /// only passes a chain whose difficulty is 0. Use [`Block::mine`]
    /// to produce a block that meets a difficulty.
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
        let hash = compute_block_hash(index, &prev_hash, timestamp, 0, &transactions);
        Ok(Self {
            index,
            prev_hash,
            timestamp,
            nonce: 0,
            transactions,
            hash,
        })
    }

    /// Mine a block: search nonces from 0 upward until the block
    /// hash has at least `difficulty` leading zero bits. Returns the
    /// mined block with its nonce and hash set. Mining is CPU-bound
    /// and slow by design at high difficulty — tests should use a
    /// low difficulty (8 or 12 bits). The result is deterministic in
    /// the sense that any block this returns is valid for the given
    /// inputs and difficulty; the exact nonce found is not promised
    /// to be the same across runs or machines, because it depends on
    /// nothing but the search order — in practice the search is a
    /// plain sequential scan from 0, so the same inputs do yield the
    /// same first nonce. Difficulty is caller-set and never adjusts.
    pub fn mine(
        index: u64,
        prev_hash: [u8; 32],
        timestamp: u64,
        transactions: Vec<Transaction>,
        difficulty: u8,
    ) -> Result<Self, ChainError> {
        for tx in &transactions {
            if !tx.verify() {
                return Err(ChainError::BadTransactionSignature);
            }
        }
        let mut nonce = 0u64;
        loop {
            let hash = compute_block_hash(index, &prev_hash, timestamp, nonce, &transactions);
            if leading_zero_bits(&hash) >= difficulty as u32 {
                return Ok(Self {
                    index,
                    prev_hash,
                    timestamp,
                    nonce,
                    transactions,
                    hash,
                });
            }
            nonce = nonce.checked_add(1).ok_or(ChainError::MiningFailed)?;
        }
    }

    /// True if the stored hash matches a recomputation over the
    /// block's contents and every transaction signature verifies.
    /// Any tampering — to the index, prev hash, timestamp, nonce,
    /// transactions, or the stored hash itself — fails this check.
    pub fn verify(&self) -> bool {
        if compute_block_hash(
            self.index,
            &self.prev_hash,
            self.timestamp,
            self.nonce,
            &self.transactions,
        ) != self.hash
        {
            return false;
        }
        self.transactions.iter().all(|tx| tx.verify())
    }

    /// True if this block's hash has at least `difficulty` leading
    /// zero bits. This checks the stored hash only — call
    /// [`Block::verify`] as well to be sure the hash is the honest
    /// hash of the block's contents.
    pub fn verify_pow(&self, difficulty: u8) -> bool {
        leading_zero_bits(&self.hash) >= difficulty as u32
    }

    /// This block's transaction root (see [`tx_root`]).
    pub fn tx_root(&self) -> [u8; 32] {
        tx_root(&self.transactions)
    }

    /// Encode to the explicit binary block format.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(BLOCK_MAGIC);
        out.extend_from_slice(&self.index.to_le_bytes());
        out.extend_from_slice(&self.prev_hash);
        out.extend_from_slice(&self.timestamp.to_le_bytes());
        out.extend_from_slice(&self.nonce.to_le_bytes());
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
        if bytes.len() < 7 + 8 + 32 + 8 + 8 + 32 + 4 || &bytes[..7] != BLOCK_MAGIC {
            return Err(ChainError::InvalidBlock(
                "missing KBLOCK1 magic".to_string(),
            ));
        }
        let mut off = 7;
        let index = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let mut prev_hash = [0u8; 32];
        prev_hash.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let timestamp = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let nonce = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let count =
            u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4-byte slice")) as usize;
        off += 4;
        // A transaction is at least 142 bytes on the wire
        // (4 magic + 32 + 32 + 2 length + 0 memo + 8 amount + 64
        // signature), so the remaining bytes bound how many there
        // can possibly be. Refuse an absurd count before allocating
        // for it — a corrupt file must be a clean error, not an
        // allocation failure.
        const MIN_TX_BYTES: usize = 4 + 32 + 32 + 2 + 8 + 64;
        if count > (bytes.len() - off) / MIN_TX_BYTES {
            return Err(ChainError::InvalidBlock(
                "transaction count exceeds possible transactions".to_string(),
            ));
        }
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
            nonce,
            transactions,
            hash,
        })
    }
}

/// Magic bytes at the start of every saved chain file.
const CHAIN_SAVE_MAGIC: &[u8; 7] = b"KCHAIN1";

/// A chain: an ordered list of blocks plus the content store holding
/// each block's encoded bytes, addressed by CID. The chain also
/// carries its proof-of-work difficulty: a non-genesis block must
/// have at least that many leading zero bits in its hash. Difficulty
/// is caller-set (default 0, meaning no work required) and never
/// auto-adjusts.
pub struct Chain {
    store: ContentStore,
    blocks: Vec<Block>,
    /// The content-store CID of each block's encoded bytes,
    /// in the same order as `blocks`.
    block_cids: Vec<Cid>,
    /// Proof-of-work difficulty: leading zero bits required in a
    /// non-genesis block's hash.
    difficulty: u8,
}

impl Chain {
    /// A new chain holding only its genesis block: index 0, no
    /// transactions, zero previous hash, the given timestamp.
    /// Difficulty is 0 — use [`Chain::set_difficulty`] or
    /// [`Chain::genesis_with_difficulty`] to require proof of work.
    pub fn genesis(timestamp: u64) -> Self {
        Self::genesis_with_difficulty(timestamp, 0)
    }

    /// A new chain holding only its genesis block, with the given
    /// proof-of-work difficulty for every later block. The genesis
    /// block itself is trusted by construction and is never required
    /// to meet the difficulty.
    pub fn genesis_with_difficulty(timestamp: u64, difficulty: u8) -> Self {
        let genesis = Block::new(0, GENESIS_PREV_HASH, timestamp, Vec::new())
            .expect("genesis block has no transactions to reject");
        let mut chain = Self {
            store: ContentStore::new(),
            blocks: Vec::new(),
            block_cids: Vec::new(),
            difficulty,
        };
        let cid = chain.store.put(&genesis.encode());
        chain.blocks.push(genesis);
        chain.block_cids.push(cid);
        chain
    }

    /// The chain's proof-of-work difficulty.
    pub fn difficulty(&self) -> u8 {
        self.difficulty
    }

    /// Set the proof-of-work difficulty for blocks appended from now
    /// on. Existing blocks are not re-checked against the new value
    /// by this call — [`Chain::validate_chain`] always checks every
    /// block against the current difficulty, so raising it after the
    /// fact can make a previously valid chain invalid, which is the
    /// honest outcome: the chain no longer meets its own rules.
    pub fn set_difficulty(&mut self, difficulty: u8) {
        self.difficulty = difficulty;
    }

    /// Number of blocks in the chain (genesis counts as 1).
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// True if the chain holds no blocks. A chain always holds at
    /// least its genesis block, so this is always false in practice.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
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

    /// The height (index) of the block with this hash, if the chain
    /// holds it. Used to find a fork point.
    pub fn find_height(&self, hash: &[u8; 32]) -> Option<usize> {
        self.blocks.iter().position(|b| &b.hash == hash)
    }

    /// Every transaction hash in the chain, as a set. Used to refuse
    /// a transaction that is already confirmed.
    pub fn tx_hash_set(&self) -> HashSet<[u8; 32]> {
        self.blocks
            .iter()
            .flat_map(|b| b.transactions.iter().map(|tx| tx.hash()))
            .collect()
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

    /// Validate a block as the next block of this chain, without
    /// appending it. Checks, in order: the index follows the tip,
    /// the previous hash links to the tip, the timestamp does not go
    /// backwards, the block verifies (hash recomputation and
    /// transaction signatures), the proof of work meets this chain's
    /// difficulty, and no transaction repeats — neither twice in the
    /// block nor once already in the chain. A bad block is a clean
    /// error naming the failure.
    pub fn validate_block(&self, block: &Block) -> Result<(), ChainError> {
        let tip = self
            .blocks
            .last()
            .expect("a chain always holds at least its genesis block");
        if block.index != tip.index + 1 {
            return Err(ChainError::InvalidBlock(format!(
                "expected index {}, got {}",
                tip.index + 1,
                block.index
            )));
        }
        if block.prev_hash != tip.hash {
            return Err(ChainError::BrokenLink {
                index: block.index,
                expected: block.prev_hash,
                actual: tip.hash,
            });
        }
        if block.timestamp < tip.timestamp {
            return Err(ChainError::NonMonotonicTimestamp {
                index: block.index,
                timestamp: block.timestamp,
                prev_timestamp: tip.timestamp,
            });
        }
        if !block.verify() {
            // A bad transaction signature is the root cause and is
            // reported first; if every signature verifies but the
            // hash still mismatches, the stored hash itself was
            // tampered with — which is also how a wrong transaction
            // root shows up, since the root feeds the hash.
            if block.transactions.iter().any(|tx| !tx.verify()) {
                return Err(ChainError::BadTransactionSignature);
            }
            return Err(ChainError::BadBlockHash { index: block.index });
        }
        if !block.verify_pow(self.difficulty) {
            return Err(ChainError::InsufficientPow {
                index: block.index,
                difficulty: self.difficulty,
            });
        }
        let mut seen = self.tx_hash_set();
        for tx in &block.transactions {
            let hash = tx.hash();
            if !seen.insert(hash) {
                return Err(ChainError::DuplicateTransaction {
                    index: block.index,
                    tx_hash: hash,
                });
            }
        }
        Ok(())
    }

    /// Append a pre-built block to the chain after validating it as
    /// the next block (see [`Chain::validate_block`]). Returns the
    /// new block's content-store CID. A refused block leaves the
    /// chain unchanged.
    pub fn append_block(&mut self, block: Block) -> Result<Cid, ChainError> {
        self.validate_block(&block)?;
        let cid = self.store.put(&block.encode());
        self.blocks.push(block);
        self.block_cids.push(cid);
        Ok(cid)
    }

    /// Append a block with the given transactions and timestamp.
    /// The block links to the latest block's hash; every transaction
    /// is verified before the block is built, so a bad transaction
    /// is refused and the chain is unchanged. The block carries no
    /// proof of work, so this only succeeds while the chain's
    /// difficulty is 0 — use [`Chain::mine_block`] otherwise.
    /// Returns the new block's content-store CID.
    pub fn add_block(
        &mut self,
        transactions: Vec<Transaction>,
        timestamp: u64,
    ) -> Result<Cid, ChainError> {
        let index = self.blocks.len() as u64;
        let prev_hash = self.latest_hash();
        let block = Block::new(index, prev_hash, timestamp, transactions)?;
        self.append_block(block)
    }

    /// Mine a block with the given transactions and timestamp and
    /// append it. The block is mined at this chain's difficulty and
    /// validated before it is appended, so a failed mine leaves the
    /// chain unchanged. Returns the new block's content-store CID.
    pub fn mine_block(
        &mut self,
        transactions: Vec<Transaction>,
        timestamp: u64,
    ) -> Result<Cid, ChainError> {
        let index = self.blocks.len() as u64;
        let prev_hash = self.latest_hash();
        let block = Block::mine(index, prev_hash, timestamp, transactions, self.difficulty)?;
        self.append_block(block)
    }

    /// Build a new chain holding the first `len` blocks of this one
    /// (the prefix up to and including the block at height `len -
    /// 1`). Used to fork from an earlier block: the fork shares the
    /// prefix and then grows its own way. `len` must be between 1
    /// and this chain's length. The fork inherits this chain's
    /// difficulty.
    pub fn fork_from(&self, len: usize) -> Result<Chain, ChainError> {
        if len == 0 || len > self.blocks.len() {
            return Err(ChainError::InvalidBlock(format!(
                "cannot fork at length {len}: chain holds {} blocks",
                self.blocks.len()
            )));
        }
        let mut fork = Chain {
            store: ContentStore::new(),
            blocks: Vec::with_capacity(len),
            block_cids: Vec::with_capacity(len),
            difficulty: self.difficulty,
        };
        for block in &self.blocks[..len] {
            let cid = fork.store.put(&block.encode());
            fork.blocks.push(block.clone());
            fork.block_cids.push(cid);
        }
        Ok(fork)
    }

    /// Verify the whole chain: the genesis block must be index 0
    /// with a zero previous hash; every later block must link to its
    /// predecessor, must not go backwards in time, must verify (hash
    /// and transaction signatures), must meet the chain's proof-of-
    /// work difficulty, and must introduce no transaction the chain
    /// already holds. Returns `Ok(())` for a valid chain, or the
    /// first problem found.
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
        let mut seen_tx = HashSet::new();
        for tx in &first.transactions {
            seen_tx.insert(tx.hash());
        }
        for (i, block) in self.blocks.iter().enumerate().skip(1) {
            let prev = &self.blocks[i - 1];
            if block.index != i as u64 {
                return Err(ChainError::InvalidBlock(format!(
                    "block at position {i} claims index {}",
                    block.index
                )));
            }
            if block.prev_hash != prev.hash {
                return Err(ChainError::BrokenLink {
                    index: block.index,
                    expected: block.prev_hash,
                    actual: prev.hash,
                });
            }
            if block.timestamp < prev.timestamp {
                return Err(ChainError::NonMonotonicTimestamp {
                    index: block.index,
                    timestamp: block.timestamp,
                    prev_timestamp: prev.timestamp,
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
            if !block.verify_pow(self.difficulty) {
                return Err(ChainError::InsufficientPow {
                    index: block.index,
                    difficulty: self.difficulty,
                });
            }
            for tx in &block.transactions {
                if !seen_tx.insert(tx.hash()) {
                    return Err(ChainError::DuplicateTransaction {
                        index: block.index,
                        tx_hash: tx.hash(),
                    });
                }
            }
        }
        Ok(())
    }

    /// True if [`Chain::validate_chain`] succeeds.
    pub fn is_valid(&self) -> bool {
        self.validate_chain().is_ok()
    }

    /// Save the whole chain to a binary file: `KCHAIN1`, then the
    /// difficulty, then the block count, then length-prefixed encoded
    /// blocks. The write goes to a sibling `.tmp` file that is
    /// fsync'd and then atomically renamed over the target, so a
    /// crash mid-save never leaves a half-written chain file behind.
    /// Saving does not change the chain. This saves the chain only,
    /// not any mempool or fork state held elsewhere.
    pub fn save(&self, path: &Path) -> Result<(), ChainError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(CHAIN_SAVE_MAGIC);
        bytes.push(self.difficulty);
        bytes.extend_from_slice(&(self.blocks.len() as u32).to_le_bytes());
        for block in &self.blocks {
            let encoded = block.encode();
            bytes.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&encoded);
        }
        let tmp_path = path.with_extension("tmp");
        {
            use std::io::Write;
            let mut file =
                std::fs::File::create(&tmp_path).map_err(|e| ChainError::Io(e.to_string()))?;
            file.write_all(&bytes)
                .map_err(|e| ChainError::Io(e.to_string()))?;
            file.sync_all().map_err(|e| ChainError::Io(e.to_string()))?;
        }
        std::fs::rename(&tmp_path, path).map_err(|e| ChainError::Io(e.to_string()))?;
        Ok(())
    }

    /// Load a chain saved by [`Chain::save`]. Refuses a bad magic,
    /// a truncation, trailing bytes, a block that fails to decode,
    /// and a chain that fails [`Chain::validate_chain`] — all as
    /// clean errors, never a panic. The loaded chain is re-validated
    /// in full, so a save whose blocks were tampered with on disk is
    /// refused rather than trusted.
    pub fn load(path: &Path) -> Result<Self, ChainError> {
        let bytes = std::fs::read(path).map_err(|e| ChainError::Io(e.to_string()))?;
        Self::decode_save(&bytes)
    }

    /// Decode a chain save produced by [`Chain::save`].
    fn decode_save(bytes: &[u8]) -> Result<Self, ChainError> {
        if bytes.len() < 7 + 1 + 4 || &bytes[..7] != CHAIN_SAVE_MAGIC {
            return Err(ChainError::CorruptSave("missing KCHAIN1 magic".to_string()));
        }
        let difficulty = bytes[7];
        let count = u32::from_le_bytes(bytes[8..12].try_into().expect("4-byte slice")) as usize;
        let mut off = 12;
        let mut blocks = Vec::new();
        for _ in 0..count {
            if bytes.len() < off + 4 {
                return Err(ChainError::CorruptSave(
                    "truncated block length".to_string(),
                ));
            }
            let len =
                u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4-byte slice")) as usize;
            off += 4;
            if bytes.len() < off + len {
                return Err(ChainError::CorruptSave("truncated block".to_string()));
            }
            let block = Block::decode(&bytes[off..off + len])
                .map_err(|e| ChainError::CorruptSave(format!("block decode failed: {e}")))?;
            blocks.push(block);
            off += len;
        }
        if off != bytes.len() {
            return Err(ChainError::CorruptSave(
                "trailing bytes after last block".to_string(),
            ));
        }
        if blocks.is_empty() {
            return Err(ChainError::CorruptSave(
                "chain save holds no blocks".to_string(),
            ));
        }
        let mut chain = Chain {
            store: ContentStore::new(),
            blocks: Vec::with_capacity(blocks.len()),
            block_cids: Vec::with_capacity(blocks.len()),
            difficulty,
        };
        for block in blocks {
            let cid = chain.store.put(&block.encode());
            chain.blocks.push(block);
            chain.block_cids.push(cid);
        }
        chain.validate_chain()?;
        Ok(chain)
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

/// Pending transactions waiting to be mined into a block.
///
/// * Dedupe: a transaction is identified by its hash, so submitting
///   the same signed transaction twice holds it once.
/// * Cap: at most `max_pending` transactions. When full, adding one
///   more evicts the oldest pending transaction first (FIFO), so the
///   pool never grows without bound. A cap of 0 means the pool holds
///   nothing — every add is refused.
/// * Order: transactions are handed out in the order they arrived
///   ([`Mempool::take_for_block`]).
/// * Only transactions whose signature verifies are accepted; a bad
///   one is refused and the pool is unchanged.
pub struct Mempool {
    pending: VecDeque<Transaction>,
    ids: HashSet<[u8; 32]>,
    max_pending: usize,
}

impl Mempool {
    /// An empty mempool with the given cap.
    pub fn new(max_pending: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            ids: HashSet::new(),
            max_pending,
        }
    }

    /// The cap on pending transactions.
    pub fn max_pending(&self) -> usize {
        self.max_pending
    }

    /// Number of pending transactions.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// True if no transaction is pending.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// True if the transaction with this hash is pending.
    pub fn contains(&self, tx_hash: &[u8; 32]) -> bool {
        self.ids.contains(tx_hash)
    }

    /// Add a transaction. Returns `Ok(true)` if it was added,
    /// `Ok(false)` if it was already pending (dedupe) or the cap is
    /// 0. A bad signature is an error and nothing is added. If the
    /// pool is full, the oldest pending transaction is evicted first.
    pub fn add(&mut self, tx: Transaction) -> Result<bool, ChainError> {
        if !tx.verify() {
            return Err(ChainError::BadTransactionSignature);
        }
        if self.max_pending == 0 {
            return Ok(false);
        }
        let id = tx.hash();
        if self.ids.contains(&id) {
            return Ok(false);
        }
        if self.pending.len() >= self.max_pending {
            if let Some(old) = self.pending.pop_front() {
                self.ids.remove(&old.hash());
            }
        }
        self.ids.insert(id);
        self.pending.push_back(tx);
        Ok(true)
    }

    /// Take up to `n` transactions for a block, in arrival order.
    /// Taken transactions leave the pool. Fewer than `n` are
    /// returned if fewer are pending.
    pub fn take_for_block(&mut self, n: usize) -> Vec<Transaction> {
        let mut out = Vec::new();
        for _ in 0..n {
            match self.pending.pop_front() {
                Some(tx) => {
                    self.ids.remove(&tx.hash());
                    out.push(tx);
                }
                None => break,
            }
        }
        out
    }

    /// Remove the pending transaction with this hash. Returns true
    /// if one was present.
    pub fn remove(&mut self, tx_hash: &[u8; 32]) -> bool {
        if !self.ids.remove(tx_hash) {
            return false;
        }
        if let Some(pos) = self.pending.iter().position(|tx| &tx.hash() == tx_hash) {
            self.pending.remove(pos);
        }
        true
    }

    /// Iterate the pending transactions in arrival order.
    pub fn iter(&self) -> impl Iterator<Item = &Transaction> {
        self.pending.iter()
    }
}

/// The length of the longest common prefix of two chains, counted in
/// blocks: the number of leading blocks whose hashes agree. Two
/// chains that share a genesis always share at least one block.
fn common_prefix_len(a: &Chain, b: &Chain) -> usize {
    let mut len = 0;
    while len < a.len() && len < b.len() {
        if a.get_block(len as u64).unwrap().hash != b.get_block(len as u64).unwrap().hash {
            break;
        }
        len += 1;
    }
    len
}

/// A chain set: the current best chain, competing forks, and the
/// mempool that feeds them.
///
/// Submitting a block extends whichever chain it links to — the main
/// chain's tip, a fork's tip, or an earlier block in either, in
/// which case a new fork is branched from that height. When a fork
/// grows strictly longer than the current best, a reorg swaps it in:
/// the old best becomes a fork, and the transactions in the abandoned
/// blocks that are not confirmed on the new best return to the
/// mempool. A tie keeps the current best, so the choice stays
/// deterministic. Only valid chains can become the best: a submitted
/// block is validated before it is appended anywhere, and
/// [`ChainSet::best_chain`] only ever considers valid chains.
pub struct ChainSet {
    main: Chain,
    forks: Vec<Chain>,
    mempool: Mempool,
}

impl ChainSet {
    /// A new chain set: one chain holding only its genesis block at
    /// the given timestamp and difficulty, no forks, an empty
    /// mempool with the given cap.
    pub fn new(timestamp: u64, difficulty: u8, max_pending: usize) -> Self {
        Self {
            main: Chain::genesis_with_difficulty(timestamp, difficulty),
            forks: Vec::new(),
            mempool: Mempool::new(max_pending),
        }
    }

    /// The current best chain.
    pub fn main(&self) -> &Chain {
        &self.main
    }

    /// The competing forks, if any.
    pub fn forks(&self) -> &[Chain] {
        &self.forks
    }

    /// The longest valid chain among the main chain and the forks.
    /// Ties go to the main chain. Since a reorg runs after every
    /// accepted block, this is normally the main chain itself.
    pub fn best_chain(&self) -> &Chain {
        let mut best = &self.main;
        for fork in &self.forks {
            if fork.is_valid() && fork.len() > best.len() {
                best = fork;
            }
        }
        best
    }

    /// The mempool feeding this chain set.
    pub fn mempool(&self) -> &Mempool {
        &self.mempool
    }

    /// Mutable access to the mempool.
    pub fn mempool_mut(&mut self) -> &mut Mempool {
        &mut self.mempool
    }

    /// Submit a transaction to the mempool. Returns `Ok(true)` if it
    /// was added, `Ok(false)` if it was already pending.
    pub fn submit_transaction(&mut self, tx: Transaction) -> Result<bool, ChainError> {
        self.mempool.add(tx)
    }

    /// Mine a block from the mempool onto the main chain: take up to
    /// `n` pending transactions, mine at the chain's difficulty, and
    /// append. If mining or appending fails, the taken transactions
    /// go back into the mempool. Returns the new block's CID.
    pub fn mine_from_mempool(&mut self, n: usize, timestamp: u64) -> Result<Cid, ChainError> {
        let txs = self.mempool.take_for_block(n);
        match self.main.mine_block(txs.clone(), timestamp) {
            Ok(cid) => Ok(cid),
            Err(e) => {
                for tx in txs {
                    let _ = self.mempool.add(tx);
                }
                Err(e)
            }
        }
    }

    /// Submit a block to the chain set. The block extends whichever
    /// chain it links to — the main tip, a fork tip, or an earlier
    /// block, which branches a new fork from that height. The block
    /// is validated before it is appended anywhere, so a bad block
    /// is refused and nothing changes. A block whose parent is
    /// unknown is refused with [`ChainError::UnknownParent`]. After
    /// a fork grows longer than the main chain, a reorg swaps it in
    /// and orphaned transactions return to the mempool.
    pub fn submit_block(&mut self, block: Block) -> Result<(), ChainError> {
        // Extends the main tip.
        if block.prev_hash == self.main.latest_hash() {
            self.main.append_block(block)?;
            return Ok(());
        }
        // Extends a fork tip.
        for fork in &mut self.forks {
            if block.prev_hash == fork.latest_hash() {
                fork.append_block(block)?;
                self.maybe_reorg();
                return Ok(());
            }
        }
        // Forks from an earlier block of the main chain.
        if let Some(height) = self.main.find_height(&block.prev_hash) {
            let mut fork = self.main.fork_from(height + 1)?;
            fork.append_block(block)?;
            self.forks.push(fork);
            self.maybe_reorg();
            return Ok(());
        }
        // Forks from an earlier block of an existing fork.
        for i in 0..self.forks.len() {
            if let Some(height) = self.forks[i].find_height(&block.prev_hash) {
                let mut fork = self.forks[i].fork_from(height + 1)?;
                fork.append_block(block)?;
                self.forks.push(fork);
                self.maybe_reorg();
                return Ok(());
            }
        }
        Err(ChainError::UnknownParent {
            prev_hash: block.prev_hash,
        })
    }

    /// If some fork is strictly longer than the main chain and valid,
    /// swap it in as the new main chain. The old main becomes a fork.
    /// Transactions in the abandoned blocks (after the common
    /// prefix) that are not confirmed on the new best return to the
    /// mempool; transactions already confirmed on the new best stay
    /// confirmed and are not re-queued.
    fn maybe_reorg(&mut self) {
        let mut best_idx: Option<usize> = None;
        let mut best_len = self.main.len();
        for (i, fork) in self.forks.iter().enumerate() {
            if fork.is_valid() && fork.len() > best_len {
                best_len = fork.len();
                best_idx = Some(i);
            }
        }
        let Some(idx) = best_idx else {
            return;
        };
        let mut new_main = self.forks.remove(idx);
        let prefix = common_prefix_len(&self.main, &new_main);
        let new_hashes = new_main.tx_hash_set();
        let mut orphaned: Vec<Transaction> = Vec::new();
        for i in prefix..self.main.len() {
            for tx in &self
                .main
                .get_block(i as u64)
                .expect("index within main chain")
                .transactions
            {
                if !new_hashes.contains(&tx.hash()) && tx.verify() {
                    orphaned.push(tx.clone());
                }
            }
        }
        std::mem::swap(&mut self.main, &mut new_main);
        self.forks.push(new_main);
        for tx in orphaned {
            let _ = self.mempool.add(tx);
        }
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
        assert_eq!(decoded.sender.public_key(), alice.peer_id().public_key());
        assert_eq!(decoded.recipient.public_key(), bob.peer_id().public_key());
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
        assert_eq!(winner.get_block(1).unwrap().transactions[0].memo, "a1");
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
            vec![
                tx(&alice, bob.peer_id(), "a", 1),
                tx(&alice, bob.peer_id(), "b", 2),
            ],
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

    #[test]
    fn mined_block_meets_difficulty_and_validates() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut chain = Chain::genesis_with_difficulty(100, 8);
        let cid = chain
            .mine_block(vec![tx(&alice, bob.peer_id(), "mined", 5)], 101)
            .unwrap();
        let block = chain.get_block(1).unwrap();
        assert!(block.verify_pow(8));
        assert!(block.verify());
        assert!(chain.is_valid());
        assert_eq!(chain.block_cid(1), Some(cid));
        // Mining the same inputs again finds a valid block too.
        let mut chain2 = Chain::genesis_with_difficulty(100, 8);
        chain2
            .mine_block(vec![tx(&alice, bob.peer_id(), "mined", 5)], 101)
            .unwrap();
        assert!(chain2.get_block(1).unwrap().verify_pow(8));
        assert!(chain2.is_valid());
    }

    #[test]
    fn unmined_block_is_refused_at_difficulty() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        // Find a timestamp whose nonce-0 block definitely fails
        // difficulty 16, so the refusal is deterministic, not luck.
        let mut ts = 101u64;
        loop {
            let probe = Block::new(
                1,
                Chain::genesis(100).latest_hash(),
                ts,
                vec![tx(&alice, bob.peer_id(), "one", 10)],
            )
            .unwrap();
            if !probe.verify_pow(16) {
                break;
            }
            ts += 1;
        }
        let mut chain = Chain::genesis(100);
        chain.set_difficulty(16);
        match chain.add_block(vec![tx(&alice, bob.peer_id(), "one", 10)], ts) {
            Err(ChainError::InsufficientPow { index, difficulty }) => {
                assert_eq!(index, 1);
                assert_eq!(difficulty, 16);
            }
            other => panic!("expected InsufficientPow, got {other:?}"),
        }
        // The chain is unchanged: still genesis only.
        assert_eq!(chain.len(), 1);
        assert!(chain.is_valid());
        // A block mined at difficulty 8 also fails a chain at 16.
        // Mine until one is found that definitely fails 16.
        let mut ts2 = 201u64;
        let mined8 = loop {
            let b = Block::mine(
                1,
                chain.latest_hash(),
                ts2,
                vec![tx(&alice, bob.peer_id(), "x", 1)],
                8,
            )
            .unwrap();
            if !b.verify_pow(16) {
                break b;
            }
            ts2 += 1;
        };
        assert!(mined8.verify_pow(8));
        match chain.append_block(mined8) {
            Err(ChainError::InsufficientPow { index, difficulty }) => {
                assert_eq!(index, 1);
                assert_eq!(difficulty, 16);
            }
            other => panic!("expected InsufficientPow, got {other:?}"),
        }
    }

    #[test]
    fn raising_difficulty_invalidates_unmined_blocks() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut chain = Chain::genesis(100);
        chain
            .add_block(vec![tx(&alice, bob.peer_id(), "one", 10)], 101)
            .unwrap();
        assert!(chain.is_valid());
        // Find the first difficulty the existing block fails.
        let block = chain.get_block(1).unwrap();
        let mut difficulty = 1u8;
        while difficulty <= 64 && block.verify_pow(difficulty) {
            difficulty += 1;
        }
        assert!(
            difficulty <= 64,
            "test block unexpectedly meets difficulty 64"
        );
        chain.set_difficulty(difficulty);
        match chain.validate_chain() {
            Err(ChainError::InsufficientPow {
                index,
                difficulty: d,
            }) => {
                assert_eq!(index, 1);
                assert_eq!(d, difficulty);
            }
            other => panic!("expected InsufficientPow, got {other:?}"),
        }
    }

    #[test]
    fn mempool_dedupes_caps_and_hands_out_in_order() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut pool = Mempool::new(2);
        let t1 = tx(&alice, bob.peer_id(), "one", 1);
        let t2 = tx(&alice, bob.peer_id(), "two", 2);
        let t3 = tx(&alice, bob.peer_id(), "three", 3);
        assert!(pool.add(t1.clone()).unwrap());
        assert!(!pool.add(t1.clone()).unwrap()); // dedupe
        assert_eq!(pool.len(), 1);
        assert!(pool.add(t2.clone()).unwrap());
        assert_eq!(pool.len(), 2);
        // Full: adding a third evicts the oldest (t1).
        assert!(pool.add(t3.clone()).unwrap());
        assert_eq!(pool.len(), 2);
        assert!(!pool.contains(&t1.hash()));
        assert!(pool.contains(&t2.hash()));
        assert!(pool.contains(&t3.hash()));
        // take_for_block hands out in arrival order.
        let taken = pool.take_for_block(1);
        assert_eq!(taken.len(), 1);
        assert_eq!(taken[0].memo, "two");
        assert_eq!(pool.len(), 1);
        let taken = pool.take_for_block(10);
        assert_eq!(taken.len(), 1);
        assert_eq!(taken[0].memo, "three");
        assert!(pool.is_empty());
        // remove() works, and removing twice is false the second time.
        pool.add(t1.clone()).unwrap();
        assert!(pool.remove(&t1.hash()));
        assert!(!pool.remove(&t1.hash()));
    }

    #[test]
    fn mempool_rejects_bad_signature_and_zero_cap_holds_nothing() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut bad = tx(&alice, bob.peer_id(), "real", 10);
        bad.amount = 999; // changed after signing
        let mut pool = Mempool::new(10);
        assert_eq!(pool.add(bad), Err(ChainError::BadTransactionSignature));
        assert!(pool.is_empty());
        let mut zero = Mempool::new(0);
        assert!(!zero.add(tx(&alice, bob.peer_id(), "one", 1)).unwrap());
        assert!(zero.is_empty());
    }

    #[test]
    fn validate_block_catches_broken_link_wrong_root_duplicates_and_time() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut chain = Chain::genesis(100);
        let t1 = tx(&alice, bob.peer_id(), "one", 10);
        chain.add_block(vec![t1.clone()], 101).unwrap();
        // Broken link.
        let bad_link = Block::new(2, [9u8; 32], 102, vec![]).unwrap();
        match chain.validate_block(&bad_link) {
            Err(ChainError::BrokenLink { index, .. }) => assert_eq!(index, 2),
            other => panic!("expected BrokenLink, got {other:?}"),
        }
        // Wrong transaction root: a different validly-signed
        // transaction, with the original block's hash pasted on top,
        // so the root no longer matches the hash.
        let forged = tx(&alice, bob.peer_id(), "forged", 10);
        let mut wrong_root = Block::new(2, chain.latest_hash(), 102, vec![forged]).unwrap();
        wrong_root.hash = chain.get_block(1).unwrap().hash;
        assert_eq!(
            chain.validate_block(&wrong_root),
            Err(ChainError::BadBlockHash { index: 2 })
        );
        // Duplicate across the chain.
        let dup = Block::new(2, chain.latest_hash(), 102, vec![t1.clone()]).unwrap();
        match chain.validate_block(&dup) {
            Err(ChainError::DuplicateTransaction { index, tx_hash }) => {
                assert_eq!(index, 2);
                assert_eq!(tx_hash, t1.hash());
            }
            other => panic!("expected DuplicateTransaction, got {other:?}"),
        }
        // Duplicate inside one block.
        let t2 = tx(&alice, bob.peer_id(), "two", 20);
        let dup_in_block =
            Block::new(2, chain.latest_hash(), 102, vec![t2.clone(), t2.clone()]).unwrap();
        assert!(matches!(
            chain.validate_block(&dup_in_block),
            Err(ChainError::DuplicateTransaction { index: 2, .. })
        ));
        // Backwards timestamp.
        let old = Block::new(2, chain.latest_hash(), 100, vec![]).unwrap();
        match chain.validate_block(&old) {
            Err(ChainError::NonMonotonicTimestamp {
                index,
                timestamp,
                prev_timestamp,
            }) => {
                assert_eq!(index, 2);
                assert_eq!(timestamp, 100);
                assert_eq!(prev_timestamp, 101);
            }
            other => panic!("expected NonMonotonicTimestamp, got {other:?}"),
        }
        // A good block validates.
        let good = Block::new(2, chain.latest_hash(), 102, vec![t2]).unwrap();
        assert!(chain.validate_block(&good).is_ok());
    }

    #[test]
    fn fork_choice_reorg_returns_orphans_to_mempool() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut set = ChainSet::new(100, 0, 100);
        assert!(set.main().fork_from(0).is_err());
        assert!(set.main().fork_from(99).is_err());
        let a1 = tx(&alice, bob.peer_id(), "a1", 1);
        let a2 = tx(&alice, bob.peer_id(), "a2", 2);
        // Main chain: genesis, B1(a1), B2(a2).
        let b1 = Block::new(
            1,
            set.main().get_block(0).unwrap().hash,
            101,
            vec![a1.clone()],
        )
        .unwrap();
        set.submit_block(b1).unwrap();
        let b2 = Block::new(2, set.main().latest_hash(), 102, vec![a2.clone()]).unwrap();
        set.submit_block(b2).unwrap();
        assert_eq!(set.main().len(), 3);
        assert!(set.mempool().is_empty());
        // Fork from genesis: F1 carries a1 again (so it is confirmed
        // on both chains), F2 carries f2, F3 carries f3.
        let genesis_hash = set.main().get_block(0).unwrap().hash;
        let f1 = Block::new(1, genesis_hash, 101, vec![a1.clone()]).unwrap();
        set.submit_block(f1).unwrap();
        assert_eq!(set.forks().len(), 1);
        assert_eq!(set.main().len(), 3); // shorter fork: no reorg yet
        let f2 = Block::new(
            2,
            set.forks()[0].latest_hash(),
            102,
            vec![tx(&alice, bob.peer_id(), "f2", 2)],
        )
        .unwrap();
        set.submit_block(f2).unwrap();
        assert_eq!(set.main().len(), 3); // tie: no reorg yet
        assert_eq!(set.main().get_block(2).unwrap().transactions[0].memo, "a2");
        let f3 = Block::new(
            3,
            set.forks()[0].latest_hash(),
            103,
            vec![tx(&alice, bob.peer_id(), "f3", 3)],
        )
        .unwrap();
        set.submit_block(f3).unwrap();
        // Reorg: the fork is now the main chain.
        assert_eq!(set.main().len(), 4);
        assert_eq!(set.best_chain().len(), 4);
        assert_eq!(set.main().get_block(3).unwrap().transactions[0].memo, "f3");
        assert!(set.main().is_valid());
        // a2 was orphaned and returns to the mempool; a1 is confirmed
        // on the new best, so it does not.
        assert_eq!(set.mempool().len(), 1);
        assert!(set.mempool().contains(&a2.hash()));
        assert!(!set.mempool().contains(&a1.hash()));
    }

    #[test]
    fn submit_block_with_unknown_parent_is_refused() {
        let mut set = ChainSet::new(100, 0, 10);
        let orphan = Block::new(5, [7u8; 32], 200, vec![]).unwrap();
        match set.submit_block(orphan) {
            Err(ChainError::UnknownParent { prev_hash }) => {
                assert_eq!(prev_hash, [7u8; 32]);
            }
            other => panic!("expected UnknownParent, got {other:?}"),
        }
        assert_eq!(set.main().len(), 1);
        assert!(set.forks().is_empty());
    }

    #[test]
    fn mine_from_mempool_works() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut set = ChainSet::new(100, 0, 10);
        set.submit_transaction(tx(&alice, bob.peer_id(), "a", 1))
            .unwrap();
        set.submit_transaction(tx(&alice, bob.peer_id(), "b", 2))
            .unwrap();
        assert_eq!(set.mempool().len(), 2);
        let cid = set.mine_from_mempool(1, 101).unwrap();
        assert_eq!(set.main().len(), 2);
        assert_eq!(set.main().get_block(1).unwrap().transactions[0].memo, "a");
        assert_eq!(set.mempool().len(), 1);
        assert_eq!(set.main().block_cid(1), Some(cid));
        assert!(set.main().is_valid());
    }

    #[test]
    fn tx_root_changes_with_transactions() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let t1 = tx(&alice, bob.peer_id(), "one", 1);
        let t2 = tx(&alice, bob.peer_id(), "two", 2);
        assert_eq!(tx_root(&[]), tx_root(&[]));
        assert_ne!(tx_root(std::slice::from_ref(&t1)), tx_root(&[]));
        assert_ne!(
            tx_root(&[t1.clone(), t2.clone()]),
            tx_root(std::slice::from_ref(&t1))
        );
        // Order matters: the same transactions in a different order
        // give a different root.
        assert_ne!(
            tx_root(&[t1.clone(), t2.clone()]),
            tx_root(&[t2.clone(), t1.clone()])
        );
    }

    fn save_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("k-chain-test-{}-{name}", std::process::id()))
    }

    fn cleanup_save(path: &std::path::PathBuf) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("tmp"));
    }

    #[test]
    fn persistence_round_trip_is_exact() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut chain = Chain::genesis_with_difficulty(100, 8);
        chain
            .mine_block(vec![tx(&alice, bob.peer_id(), "one", 10)], 101)
            .unwrap();
        chain
            .mine_block(vec![tx(&alice, bob.peer_id(), "two", 20)], 102)
            .unwrap();
        let path = save_path("round-trip");
        chain.save(&path).unwrap();
        let loaded = Chain::load(&path).unwrap();
        assert_eq!(loaded.len(), chain.len());
        assert_eq!(loaded.difficulty(), 8);
        for i in 0..chain.len() {
            let orig = chain.get_block(i as u64).unwrap();
            let back = loaded.get_block(i as u64).unwrap();
            assert_eq!(back.hash, orig.hash);
            assert_eq!(back.nonce, orig.nonce);
            assert_eq!(back.timestamp, orig.timestamp);
            assert_eq!(back.transactions.len(), orig.transactions.len());
            assert_eq!(loaded.block_cid(i as u64), chain.block_cid(i as u64));
        }
        assert!(loaded.is_valid());
        // Saving the loaded chain again produces identical bytes.
        let path2 = save_path("round-trip-2");
        loaded.save(&path2).unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap(),
            std::fs::read(&path2).unwrap()
        );
        cleanup_save(&path);
        cleanup_save(&path2);
    }

    #[test]
    fn corrupted_save_is_refused() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut chain = Chain::genesis_with_difficulty(100, 8);
        chain
            .mine_block(vec![tx(&alice, bob.peer_id(), "one", 10)], 101)
            .unwrap();
        let path = save_path("corrupt");
        chain.save(&path).unwrap();
        let good = std::fs::read(&path).unwrap();
        // Bad magic.
        let mut bad = good.clone();
        bad[0] = b'X';
        std::fs::write(&path, &bad).unwrap();
        assert!(Chain::load(&path).is_err());
        // Truncation (last byte missing).
        std::fs::write(&path, &good[..good.len() - 1]).unwrap();
        assert!(Chain::load(&path).is_err());
        // Trailing bytes.
        let mut trailing = good.clone();
        trailing.push(0);
        std::fs::write(&path, &trailing).unwrap();
        assert!(Chain::load(&path).is_err());
        // Flipped byte inside the first block's magic: the block no
        // longer decodes.
        let mut flipped = good.clone();
        flipped[20] ^= 0xff;
        std::fs::write(&path, &flipped).unwrap();
        assert!(Chain::load(&path).is_err());
        // A missing file is a clean error too.
        cleanup_save(&path);
        assert!(Chain::load(&path).is_err());
    }
}
