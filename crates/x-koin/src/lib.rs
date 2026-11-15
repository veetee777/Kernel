//! X•Koin — the coin. An account-based ledger on top of KERNEL
//! crypto and identity.
//!
//! A transaction is a signed payload: sender, recipient, a
//! human-readable memo, and an amount. The ledger holds balances
//! derived only from validated transactions, starting from a genesis
//! allocation — a fixed initial supply assigned to named genesis
//! holders. There is no mining and no inflation: every transfer
//! conserves the supply, so `total_supply` always equals the genesis
//! total.
//!
//! The rules are simple and deterministic:
//!
//! * A transaction's signature must verify against its sender.
//!   A tampered transaction (memo, amount, or either peer changed
//!   after signing) is refused.
//! * The amount must be greater than zero.
//! * The sender must hold at least the amount. No credit, no
//!   overdraft — insufficient funds are refused and the ledger is
//!   unchanged.
//! * Every transaction has an id (the SHA-256 of its canonical
//!   encoding). A transaction id is accepted exactly once, so
//!   replaying the same signed transaction cannot double-spend.
//! * Balances are keyed by public key, not by display name, so a
//!   peer keeps their balance even if their name changes. The
//!   first-seen display name is kept so humans still see names,
//!   not hashes.
//!
//! Transaction format (explicit binary, no JSON):
//! `XKOIN1` + sender public key (32) + recipient public key (32) +
//! memo length (u16 LE) + memo bytes + amount (u64 LE) +
//! signature (64).

use std::collections::{HashMap, HashSet};
use std::fmt;

use kernel_crypto::{sha256, Ed25519Signature};
use kernel_identity::{Identity, IdentityError, PeerId};

pub mod allowance;
pub mod multisig;
pub mod statement;

pub use allowance::{Allowance, AllowanceDraw, AllowanceDrawRequest};
pub use multisig::{MultisigApproval, MultisigTransaction};
pub use statement::{EntryKind, Statement, StatementEntry};

/// Magic bytes at the start of every transaction.
const TX_MAGIC: &[u8; 6] = b"XKOIN1";

/// Domain separator for transaction signatures: a signature covers
/// `b"XKOIN1" || sender key || recipient key || memo || amount`.
const TX_SIGN_DOMAIN: &[u8] = b"XKOIN1";

/// Errors returned by coin operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KoinError {
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// A transaction memo broke the rules.
    InvalidMemo(String),
    /// A transaction was malformed.
    InvalidTransaction(String),
    /// A transaction's signature did not verify against its sender.
    BadSignature,
    /// The transaction amount was zero.
    ZeroAmount,
    /// The sender did not hold enough to cover the amount.
    InsufficientFunds {
        /// The amount the transaction tried to move.
        needed: u64,
        /// The sender's actual balance.
        available: u64,
    },
    /// The transaction was already accepted once. Its id is shown
    /// as hex so a human can recognize the replay.
    Replay(String),
    /// A balance update would overflow a u64.
    Overflow,
    /// A multisig proposal id is not known to the ledger.
    UnknownProposal,
    /// An approver is not in the proposal's approver set.
    NotApprover,
    /// An approver already approved this proposal.
    DuplicateApproval,
    /// A multisig proposal has fewer approvals than its threshold.
    BelowThreshold {
        /// Approvals collected so far.
        have: usize,
        /// Approvals required.
        need: usize,
    },
    /// A multisig proposal broke the construction rules.
    InvalidMultisig(String),
    /// An allowance id is not known to the ledger.
    UnknownAllowance,
    /// A draw was attempted on a revoked allowance.
    AllowanceRevoked,
    /// A draw was attempted after the allowance's expiry.
    ExpiredAllowance {
        /// The allowance's expiry timestamp.
        expiry: u64,
    },
    /// A draw exceeded the allowance's remaining amount.
    AllowanceExceeded {
        /// The amount the draw tried to take.
        needed: u64,
        /// What was left in the allowance.
        remaining: u64,
    },
    /// An allowance broke the rules (bad owner, wrong spender, or
    /// a malformed encoding).
    InvalidAllowance(String),
    /// A statement broke the decoding rules.
    InvalidStatement(String),
}

impl fmt::Display for KoinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KoinError::Identity(e) => write!(f, "identity error: {e}"),
            KoinError::InvalidMemo(msg) => write!(f, "invalid memo: {msg}"),
            KoinError::InvalidTransaction(msg) => write!(f, "invalid transaction: {msg}"),
            KoinError::BadSignature => write!(f, "transaction signature is invalid"),
            KoinError::ZeroAmount => write!(f, "transaction amount must be greater than zero"),
            KoinError::InsufficientFunds { needed, available } => write!(
                f,
                "insufficient funds: needed {needed}, available {available}"
            ),
            KoinError::Replay(id) => write!(f, "transaction {id} was already accepted"),
            KoinError::Overflow => write!(f, "balance update would overflow"),
            KoinError::UnknownProposal => write!(f, "unknown multisig proposal"),
            KoinError::NotApprover => write!(f, "approver is not in the proposal's approver set"),
            KoinError::DuplicateApproval => {
                write!(f, "approver already approved this proposal")
            }
            KoinError::BelowThreshold { have, need } => {
                write!(f, "multisig proposal has {have} approvals, needs {need}")
            }
            KoinError::InvalidMultisig(msg) => write!(f, "invalid multisig proposal: {msg}"),
            KoinError::UnknownAllowance => write!(f, "unknown allowance"),
            KoinError::AllowanceRevoked => write!(f, "allowance has been revoked"),
            KoinError::ExpiredAllowance { expiry } => {
                write!(f, "allowance expired at {expiry}")
            }
            KoinError::AllowanceExceeded { needed, remaining } => write!(
                f,
                "allowance draw of {needed} exceeds remaining {remaining}"
            ),
            KoinError::InvalidAllowance(msg) => write!(f, "invalid allowance: {msg}"),
            KoinError::InvalidStatement(msg) => write!(f, "invalid statement: {msg}"),
        }
    }
}

impl std::error::Error for KoinError {}

impl From<IdentityError> for KoinError {
    fn from(e: IdentityError) -> Self {
        KoinError::Identity(e)
    }
}

/// Rules for memos: at most 128 printable characters. A memo may be
/// empty (a plain payment needs no note).
pub(crate) fn validate_memo(memo: &str) -> Result<(), KoinError> {
    if memo.len() > 128 {
        return Err(KoinError::InvalidMemo(
            "memo must be at most 128 characters".to_string(),
        ));
    }
    if !memo.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
        return Err(KoinError::InvalidMemo(
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

/// A display name derived from a public key: the first 8 hex chars of
/// the key. Used when a transaction is decoded and only the key
/// survives (verification needs the key, not the human name).
pub(crate) fn fingerprint_name(key: &kernel_crypto::Ed25519PublicKey) -> String {
    let hex = kernel_crypto::to_hex(key.as_bytes());
    format!("peer-{}", &hex[..8])
}

/// A signed coin transaction: sender, recipient, memo, amount.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoinTransaction {
    /// Who is sending.
    pub sender: PeerId,
    /// Who is receiving.
    pub recipient: PeerId,
    /// A human-readable note, e.g. `"coffee"`.
    pub memo: String,
    /// The amount, in the coin's smallest unit.
    pub amount: u64,
    /// The sender's signature over sender, recipient, memo, amount.
    pub signature: Ed25519Signature,
}

impl CoinTransaction {
    /// Create and sign a transaction as `sender`. Only the memo is
    /// validated here; the amount, balance, and replay rules are
    /// enforced by [`Ledger::submit`], which sees the whole ledger.
    pub fn new(
        sender: &Identity,
        recipient: &PeerId,
        memo: &str,
        amount: u64,
    ) -> Result<Self, KoinError> {
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

    /// The transaction's id: the SHA-256 of its canonical encoding,
    /// rendered as hex. A ledger accepts each id exactly once.
    pub fn id(&self) -> String {
        kernel_crypto::to_hex(&sha256(&self.encode()))
    }

    /// Encode to the explicit binary transaction format.
    pub fn encode(&self) -> Vec<u8> {
        let memo_bytes = self.memo.as_bytes();
        let mut out = Vec::with_capacity(6 + 32 + 32 + 2 + memo_bytes.len() + 8 + 64);
        out.extend_from_slice(TX_MAGIC);
        out.extend_from_slice(self.sender.public_key().as_bytes());
        out.extend_from_slice(self.recipient.public_key().as_bytes());
        out.extend_from_slice(&(memo_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(memo_bytes);
        out.extend_from_slice(&self.amount.to_le_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a transaction produced by [`CoinTransaction::encode`].
    /// Rejects anything that is not exactly that format.
    pub fn decode(bytes: &[u8]) -> Result<Self, KoinError> {
        if bytes.len() < 6 + 32 + 32 + 2 + 8 + 64 || &bytes[..6] != TX_MAGIC {
            return Err(KoinError::InvalidTransaction(
                "missing XKOIN1 magic".to_string(),
            ));
        }
        let memo_len = u16::from_le_bytes(bytes[70..72].try_into().expect("2-byte slice")) as usize;
        let expected = 6 + 32 + 32 + 2 + memo_len + 8 + 64;
        if bytes.len() != expected {
            return Err(KoinError::InvalidTransaction(format!(
                "transaction is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let mut off = 6;
        let mut sender_bytes = [0u8; 32];
        sender_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let sender_key = kernel_crypto::Ed25519PublicKey::from_bytes(&sender_bytes)
            .map_err(|e| KoinError::InvalidTransaction(e.to_string()))?;
        let mut recipient_bytes = [0u8; 32];
        recipient_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let recipient_key = kernel_crypto::Ed25519PublicKey::from_bytes(&recipient_bytes)
            .map_err(|e| KoinError::InvalidTransaction(e.to_string()))?;
        let memo_len =
            u16::from_le_bytes(bytes[off..off + 2].try_into().expect("2-byte slice")) as usize;
        off += 2;
        let memo = String::from_utf8(bytes[off..off + memo_len].to_vec()).map_err(|_| {
            KoinError::InvalidTransaction("transaction memo is not valid UTF-8".to_string())
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

/// A multisig proposal held by the ledger, with the approvals
/// collected for it so far (approver public keys).
struct MultisigState {
    proposal: MultisigTransaction,
    approvals: HashSet<[u8; 32]>,
}

/// An allowance held by the ledger, with what is left to draw and
/// whether the owner revoked it.
struct AllowanceState {
    allowance: Allowance,
    remaining: u64,
    revoked: bool,
}

/// An account-based coin ledger. Balances start at the genesis
/// allocation and change only through validated transactions, so
/// the total supply never changes after genesis.
pub struct Ledger {
    /// Balances keyed by public key bytes.
    balances: HashMap<[u8; 32], u64>,
    /// First-seen display name per public key, so humans see names.
    names: HashMap<[u8; 32], String>,
    /// Transaction ids already accepted, for replay rejection.
    seen: HashSet<String>,
    /// Accepted transactions, in acceptance order.
    history: Vec<CoinTransaction>,
    /// The fixed total supply, set by the genesis allocation.
    genesis_supply: u64,
    /// The genesis allocation, kept so statements can start a
    /// peer's running balance at what they were given.
    genesis: HashMap<[u8; 32], u64>,
    /// Next acceptance sequence number, shared by transfers and
    /// allowance draws so statements merge both in order.
    seq: u64,
    /// Acceptance sequence of each entry in `history`, parallel
    /// to it.
    history_seqs: Vec<u64>,
    /// Multisig proposals awaiting approval, keyed by proposal id.
    multisig: HashMap<String, MultisigState>,
    /// Granted allowances, keyed by allowance id.
    allowances: HashMap<String, AllowanceState>,
    /// Completed allowance draws, in acceptance order, each with
    /// its sequence number.
    draws: Vec<(u64, AllowanceDraw)>,
}

impl Ledger {
    /// Create a ledger with a genesis allocation: a fixed initial
    /// supply assigned to the named holders. The allocation is the
    /// only way coins ever enter the ledger.
    pub fn new(genesis: &[(PeerId, u64)]) -> Result<Self, KoinError> {
        let mut balances: HashMap<[u8; 32], u64> = HashMap::new();
        let mut names: HashMap<[u8; 32], String> = HashMap::new();
        let mut genesis_supply: u64 = 0;
        let mut genesis_map: HashMap<[u8; 32], u64> = HashMap::new();
        for (peer, amount) in genesis {
            let key = *peer.public_key().as_bytes();
            let entry = balances.entry(key).or_insert(0);
            *entry = entry.checked_add(*amount).ok_or(KoinError::Overflow)?;
            genesis_supply = genesis_supply
                .checked_add(*amount)
                .ok_or(KoinError::Overflow)?;
            let gentry = genesis_map.entry(key).or_insert(0);
            *gentry = gentry.checked_add(*amount).ok_or(KoinError::Overflow)?;
            names
                .entry(key)
                .or_insert_with(|| peer.display_name().to_string());
        }
        Ok(Self {
            balances,
            names,
            seen: HashSet::new(),
            history: Vec::new(),
            genesis_supply,
            genesis: genesis_map,
            seq: 0,
            history_seqs: Vec::new(),
            multisig: HashMap::new(),
            allowances: HashMap::new(),
            draws: Vec::new(),
        })
    }

    /// The balance of a peer. Unknown peers hold zero.
    pub fn balance_of(&self, peer: &PeerId) -> u64 {
        self.balances
            .get(peer.public_key().as_bytes())
            .copied()
            .unwrap_or(0)
    }

    /// The current total supply: the sum of all balances. Transfers
    /// conserve it, so this always equals the genesis supply.
    pub fn total_supply(&self) -> u64 {
        self.balances
            .values()
            .fold(0u64, |acc, b| acc.saturating_add(*b))
    }

    /// The fixed total supply set by the genesis allocation.
    pub fn genesis_supply(&self) -> u64 {
        self.genesis_supply
    }

    /// True if this transaction id was already accepted.
    pub fn has_seen(&self, tx: &CoinTransaction) -> bool {
        self.seen.contains(&tx.id())
    }

    /// Accepted transactions, in acceptance order.
    pub fn transactions(&self) -> &[CoinTransaction] {
        &self.history
    }

    /// Every holder with a non-zero balance, as `(peer, balance)`
    /// pairs sorted by display name so the list is deterministic.
    /// Peers are rebuilt with their first-seen display names.
    pub fn holders(&self) -> Vec<(PeerId, u64)> {
        let mut out: Vec<(PeerId, u64)> = self
            .balances
            .iter()
            .filter(|(_, amount)| **amount > 0)
            .map(|(key, amount)| {
                let name = self.names.get(key).cloned().unwrap_or_else(|| {
                    fingerprint_name(
                        &kernel_crypto::Ed25519PublicKey::from_bytes(key)
                            .expect("stored keys are valid"),
                    )
                });
                let peer = PeerId::new(
                    kernel_crypto::Ed25519PublicKey::from_bytes(key)
                        .expect("stored keys are valid"),
                    &name,
                )
                .expect("stored names are valid");
                (peer, *amount)
            })
            .collect();
        out.sort_by(|a, b| a.0.display_name().cmp(b.0.display_name()));
        out
    }

    /// Submit a signed transaction. Validates, in order: the amount
    /// is greater than zero, the signature verifies against the
    /// sender, the transaction was not accepted before, and the
    /// sender holds enough. A rejected transaction changes nothing.
    /// On success the balances move and the transaction is recorded.
    pub fn submit(&mut self, tx: &CoinTransaction) -> Result<(), KoinError> {
        if tx.amount == 0 {
            return Err(KoinError::ZeroAmount);
        }
        if !tx.verify() {
            return Err(KoinError::BadSignature);
        }
        let id = tx.id();
        if self.seen.contains(&id) {
            return Err(KoinError::Replay(id));
        }
        let sender_key = *tx.sender.public_key().as_bytes();
        let recipient_key = *tx.recipient.public_key().as_bytes();
        let sender_balance = self.balances.get(&sender_key).copied().unwrap_or(0);
        if sender_balance < tx.amount {
            return Err(KoinError::InsufficientFunds {
                needed: tx.amount,
                available: sender_balance,
            });
        }
        let recipient_balance = self.balances.get(&recipient_key).copied().unwrap_or(0);
        let new_recipient_balance = recipient_balance
            .checked_add(tx.amount)
            .ok_or(KoinError::Overflow)?;
        // All checks passed: apply.
        self.balances.insert(sender_key, sender_balance - tx.amount);
        self.balances.insert(recipient_key, new_recipient_balance);
        self.names
            .entry(sender_key)
            .or_insert_with(|| tx.sender.display_name().to_string());
        self.names
            .entry(recipient_key)
            .or_insert_with(|| tx.recipient.display_name().to_string());
        self.seen.insert(id);
        let seq = self.seq;
        self.seq += 1;
        self.history.push(tx.clone());
        self.history_seqs.push(seq);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Multi-signature transactions
    // ------------------------------------------------------------------

    /// Propose a multi-signature transfer. Validates the underlying
    /// transaction's signature and amount, and refuses a proposal
    /// whose id is already pending or whose transaction was already
    /// accepted. The transfer does not move until the threshold is
    /// reached and [`Ledger::submit_multisig`] is called.
    pub fn propose_multisig(&mut self, proposal: &MultisigTransaction) -> Result<(), KoinError> {
        let id = proposal.id();
        if self.multisig.contains_key(&id) {
            return Err(KoinError::Replay(id));
        }
        let tx_id = proposal.tx.id();
        if self.seen.contains(&tx_id) {
            return Err(KoinError::Replay(tx_id));
        }
        if proposal.tx.amount == 0 {
            return Err(KoinError::ZeroAmount);
        }
        if !proposal.tx.verify() {
            return Err(KoinError::BadSignature);
        }
        self.multisig.insert(
            id,
            MultisigState {
                proposal: proposal.clone(),
                approvals: HashSet::new(),
            },
        );
        Ok(())
    }

    /// Record a signed approval for a pending proposal. Returns the
    /// approval count after this approval. Refuses unknown
    /// proposals, approvals from non-approvers, bad signatures, and
    /// a second approval from the same approver.
    pub fn approve_multisig(&mut self, approval: &MultisigApproval) -> Result<usize, KoinError> {
        let state = self
            .multisig
            .get_mut(&approval.proposal_id)
            .ok_or(KoinError::UnknownProposal)?;
        if !state.proposal.is_approver(&approval.approver) {
            return Err(KoinError::NotApprover);
        }
        if !approval.verify() {
            return Err(KoinError::BadSignature);
        }
        let key = *approval.approver.public_key().as_bytes();
        if !state.approvals.insert(key) {
            return Err(KoinError::DuplicateApproval);
        }
        Ok(state.approvals.len())
    }

    /// Submit a pending proposal whose threshold is reached. The
    /// underlying transfer submits exactly like a normal transfer:
    /// insufficient funds still refuse it, and the proposal stays
    /// pending so it can be retried after the sender is funded.
    /// On success the proposal is removed; submitting again is an
    /// unknown proposal.
    pub fn submit_multisig(&mut self, proposal_id: &str) -> Result<(), KoinError> {
        let (threshold, have, tx) = {
            let state = self
                .multisig
                .get(proposal_id)
                .ok_or(KoinError::UnknownProposal)?;
            (
                state.proposal.threshold as usize,
                state.approvals.len(),
                state.proposal.tx.clone(),
            )
        };
        if have < threshold {
            return Err(KoinError::BelowThreshold {
                have,
                need: threshold,
            });
        }
        self.submit(&tx)?;
        self.multisig.remove(proposal_id);
        Ok(())
    }

    /// Approvals collected for a pending proposal, or `None` if the
    /// proposal is unknown (or already submitted).
    pub fn multisig_approval_count(&self, proposal_id: &str) -> Option<usize> {
        self.multisig.get(proposal_id).map(|s| s.approvals.len())
    }

    /// The pending proposal with this id, or `None` if unknown (or
    /// already submitted).
    pub fn multisig_proposal(&self, proposal_id: &str) -> Option<&MultisigTransaction> {
        self.multisig.get(proposal_id).map(|s| &s.proposal)
    }

    // ------------------------------------------------------------------
    // Allowances
    // ------------------------------------------------------------------

    /// Grant an allowance: record the owner's signed approval for
    /// a spender. Refuses a bad signature and a grant whose id is
    /// already held (granting the same allowance twice is a replay).
    pub fn grant_allowance(&mut self, allowance: &Allowance) -> Result<(), KoinError> {
        if !allowance.verify() {
            return Err(KoinError::BadSignature);
        }
        let id = allowance.id();
        if self.allowances.contains_key(&id) {
            return Err(KoinError::Replay(id));
        }
        self.allowances.insert(
            id,
            AllowanceState {
                remaining: allowance.amount,
                allowance: allowance.clone(),
                revoked: false,
            },
        );
        Ok(())
    }

    /// Revoke an allowance. Only the owner may revoke; the
    /// remainder drops to zero and later draws refuse. Revoking an
    /// already-revoked allowance is a no-op success.
    pub fn revoke_allowance(
        &mut self,
        owner: &PeerId,
        allowance_id: &str,
    ) -> Result<(), KoinError> {
        let state = self
            .allowances
            .get_mut(allowance_id)
            .ok_or(KoinError::UnknownAllowance)?;
        if state.allowance.owner.public_key() != owner.public_key() {
            return Err(KoinError::InvalidAllowance(
                "only the allowance owner can revoke it".to_string(),
            ));
        }
        state.revoked = true;
        state.remaining = 0;
        Ok(())
    }

    /// What is left to draw on an allowance, or `None` if the
    /// allowance is unknown. A revoked allowance has zero left.
    pub fn allowance_remaining(&self, allowance_id: &str) -> Option<u64> {
        self.allowances.get(allowance_id).map(|s| s.remaining)
    }

    /// The granted allowance with this id, or `None` if unknown.
    pub fn allowance(&self, allowance_id: &str) -> Option<&Allowance> {
        self.allowances.get(allowance_id).map(|s| &s.allowance)
    }

    /// Draw from an allowance. The caller supplies `now` (a unix
    /// timestamp); the ledger never reads the system clock. Checks,
    /// in order: the allowance exists, it is not revoked, it has
    /// not expired, the draw request's signature verifies against
    /// the spender, the spender matches the allowance, the amount
    /// is within the remainder, and the owner actually holds the
    /// amount. A refused draw changes nothing.
    pub fn spend_from_allowance(
        &mut self,
        draw: &AllowanceDrawRequest,
        now: u64,
    ) -> Result<AllowanceDraw, KoinError> {
        let (owner, spender) = {
            let state = self
                .allowances
                .get(&draw.allowance_id)
                .ok_or(KoinError::UnknownAllowance)?;
            if state.revoked {
                return Err(KoinError::AllowanceRevoked);
            }
            if state.allowance.expiry != 0 && now > state.allowance.expiry {
                return Err(KoinError::ExpiredAllowance {
                    expiry: state.allowance.expiry,
                });
            }
            (
                state.allowance.owner.clone(),
                state.allowance.spender.clone(),
            )
        };
        if !draw.verify() {
            return Err(KoinError::BadSignature);
        }
        if draw.spender.public_key() != spender.public_key() {
            return Err(KoinError::InvalidAllowance(
                "draw spender does not match the allowance spender".to_string(),
            ));
        }
        if draw.amount == 0 {
            return Err(KoinError::ZeroAmount);
        }
        let state = self
            .allowances
            .get_mut(&draw.allowance_id)
            .ok_or(KoinError::UnknownAllowance)?;
        if draw.amount > state.remaining {
            return Err(KoinError::AllowanceExceeded {
                needed: draw.amount,
                remaining: state.remaining,
            });
        }
        let owner_key = *owner.public_key().as_bytes();
        let recipient_key = *draw.recipient.public_key().as_bytes();
        let owner_balance = self.balances.get(&owner_key).copied().unwrap_or(0);
        if owner_balance < draw.amount {
            return Err(KoinError::InsufficientFunds {
                needed: draw.amount,
                available: owner_balance,
            });
        }
        let recipient_balance = self.balances.get(&recipient_key).copied().unwrap_or(0);
        let new_recipient_balance = recipient_balance
            .checked_add(draw.amount)
            .ok_or(KoinError::Overflow)?;
        // All checks passed: apply.
        self.balances.insert(owner_key, owner_balance - draw.amount);
        self.balances.insert(recipient_key, new_recipient_balance);
        self.names
            .entry(owner_key)
            .or_insert_with(|| owner.display_name().to_string());
        self.names
            .entry(recipient_key)
            .or_insert_with(|| draw.recipient.display_name().to_string());
        state.remaining -= draw.amount;
        let seq = self.seq;
        self.seq += 1;
        let completed = AllowanceDraw {
            allowance_id: draw.allowance_id.clone(),
            owner,
            spender,
            recipient: draw.recipient.clone(),
            memo: draw.memo.clone(),
            amount: draw.amount,
            seq,
        };
        self.draws.push((seq, completed.clone()));
        Ok(completed)
    }

    // ------------------------------------------------------------------
    // Account statements
    // ------------------------------------------------------------------

    /// A peer's statement: every transfer and allowance draw in or
    /// out, in acceptance order, with the running balance after each
    /// entry. A peer with no activity gets an empty statement whose
    /// final balance is their genesis allocation.
    pub fn statement(&self, peer: &PeerId) -> Statement {
        let peer_key = *peer.public_key().as_bytes();
        let mut entries: Vec<StatementEntry> = Vec::new();
        // Merge transfers and draws by acceptance sequence.
        let mut transfer_idx = 0usize;
        let mut draw_idx = 0usize;
        let mut balance: i128 = self.genesis.get(&peer_key).copied().unwrap_or(0) as i128;
        loop {
            let next_transfer_seq = self.history_seqs.get(transfer_idx).copied();
            let next_draw_seq = self.draws.get(draw_idx).map(|(s, _)| *s);
            let take_transfer = match (next_transfer_seq, next_draw_seq) {
                (Some(t), Some(d)) => t <= d,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            };
            if take_transfer {
                let seq = next_transfer_seq.expect("transfer seq present");
                let tx = &self.history[transfer_idx];
                transfer_idx += 1;
                let is_sender = *tx.sender.public_key().as_bytes() == peer_key;
                let is_recipient = *tx.recipient.public_key().as_bytes() == peer_key;
                if is_sender {
                    balance -= tx.amount as i128;
                    entries.push(StatementEntry {
                        seq,
                        id: sha256(&tx.encode()),
                        counterparty: tx.recipient.clone(),
                        memo: tx.memo.clone(),
                        amount: -(tx.amount as i64),
                        balance_after: u64::try_from(balance)
                            .expect("ledger invariant violated: negative balance in statement"),
                        kind: EntryKind::Transfer,
                    });
                }
                if is_recipient {
                    balance += tx.amount as i128;
                    entries.push(StatementEntry {
                        seq,
                        id: sha256(&tx.encode()),
                        counterparty: tx.sender.clone(),
                        memo: tx.memo.clone(),
                        amount: tx.amount as i64,
                        balance_after: u64::try_from(balance)
                            .expect("ledger invariant violated: negative balance in statement"),
                        kind: EntryKind::Transfer,
                    });
                }
            } else {
                let (seq, draw) = &self.draws[draw_idx];
                let seq = *seq;
                draw_idx += 1;
                let is_owner = *draw.owner.public_key().as_bytes() == peer_key;
                let is_recipient = *draw.recipient.public_key().as_bytes() == peer_key;
                if is_owner {
                    balance -= draw.amount as i128;
                    entries.push(StatementEntry {
                        seq,
                        id: draw.id_bytes(),
                        counterparty: draw.recipient.clone(),
                        memo: draw.memo.clone(),
                        amount: -(draw.amount as i64),
                        balance_after: u64::try_from(balance)
                            .expect("ledger invariant violated: negative balance in statement"),
                        kind: EntryKind::AllowanceDraw,
                    });
                }
                if is_recipient {
                    balance += draw.amount as i128;
                    entries.push(StatementEntry {
                        seq,
                        id: draw.id_bytes(),
                        counterparty: draw.owner.clone(),
                        memo: draw.memo.clone(),
                        amount: draw.amount as i64,
                        balance_after: u64::try_from(balance)
                            .expect("ledger invariant violated: negative balance in statement"),
                        kind: EntryKind::AllowanceDraw,
                    });
                }
            }
        }
        let opening = self.genesis.get(&peer_key).copied().unwrap_or(0);
        Statement::build(peer.clone(), entries, opening)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identities() -> (Identity, Identity, Identity) {
        (
            Identity::generate("alice").unwrap(),
            Identity::generate("bob").unwrap(),
            Identity::generate("carol").unwrap(),
        )
    }

    fn funded() -> (Ledger, Identity, Identity, Identity) {
        let (alice, bob, carol) = identities();
        let ledger = Ledger::new(&[
            (alice.peer_id().clone(), 1_000),
            (bob.peer_id().clone(), 500),
        ])
        .unwrap();
        (ledger, alice, bob, carol)
    }

    #[test]
    fn genesis_supply_matches_allocation() {
        let (alice, bob, _) = identities();
        let ledger = Ledger::new(&[
            (alice.peer_id().clone(), 1_000),
            (bob.peer_id().clone(), 500),
        ])
        .unwrap();
        assert_eq!(ledger.genesis_supply(), 1_500);
        assert_eq!(ledger.total_supply(), 1_500);
        assert_eq!(ledger.balance_of(alice.peer_id()), 1_000);
        assert_eq!(ledger.balance_of(bob.peer_id()), 500);
        // An empty genesis is a valid, empty ledger.
        let empty = Ledger::new(&[]).unwrap();
        assert_eq!(empty.genesis_supply(), 0);
        assert_eq!(empty.total_supply(), 0);
    }

    #[test]
    fn transfer_moves_balances_and_conserves_supply() {
        let (mut ledger, alice, bob, _) = funded();
        let tx = CoinTransaction::new(&alice, bob.peer_id(), "coffee", 250).unwrap();
        ledger.submit(&tx).unwrap();
        assert_eq!(ledger.balance_of(alice.peer_id()), 750);
        assert_eq!(ledger.balance_of(bob.peer_id()), 750);
        assert_eq!(ledger.total_supply(), 1_500);
        assert_eq!(ledger.genesis_supply(), 1_500);
        assert_eq!(ledger.transactions().len(), 1);
        // A second transfer keeps conserving.
        let tx2 = CoinTransaction::new(&bob, alice.peer_id(), "refund", 50).unwrap();
        ledger.submit(&tx2).unwrap();
        assert_eq!(ledger.balance_of(alice.peer_id()), 800);
        assert_eq!(ledger.balance_of(bob.peer_id()), 700);
        assert_eq!(ledger.total_supply(), 1_500);
    }

    #[test]
    fn insufficient_funds_rejected_and_ledger_unchanged() {
        let (mut ledger, alice, bob, _) = funded();
        // Alice holds 1,000; 1,001 is too much.
        let tx = CoinTransaction::new(&alice, bob.peer_id(), "too much", 1_001).unwrap();
        assert_eq!(
            ledger.submit(&tx),
            Err(KoinError::InsufficientFunds {
                needed: 1_001,
                available: 1_000
            })
        );
        assert_eq!(ledger.balance_of(alice.peer_id()), 1_000);
        assert_eq!(ledger.balance_of(bob.peer_id()), 500);
        assert!(ledger.transactions().is_empty());
        // Spending exactly the balance works.
        let exact = CoinTransaction::new(&alice, bob.peer_id(), "all of it", 1_000).unwrap();
        ledger.submit(&exact).unwrap();
        assert_eq!(ledger.balance_of(alice.peer_id()), 0);
        assert_eq!(ledger.balance_of(bob.peer_id()), 1_500);
    }

    #[test]
    fn unknown_peer_holds_zero_and_cannot_spend() {
        let (mut ledger, alice, _, carol) = funded();
        assert_eq!(ledger.balance_of(carol.peer_id()), 0);
        let tx = CoinTransaction::new(&carol, alice.peer_id(), "nothing", 1).unwrap();
        assert_eq!(
            ledger.submit(&tx),
            Err(KoinError::InsufficientFunds {
                needed: 1,
                available: 0
            })
        );
    }

    #[test]
    fn tampered_transaction_rejected() {
        let (mut ledger, alice, bob, _) = funded();
        let tx = CoinTransaction::new(&alice, bob.peer_id(), "coffee", 100).unwrap();
        // Tamper with the amount after signing.
        let mut forged = tx.clone();
        forged.amount = 900;
        assert!(!forged.verify());
        assert_eq!(ledger.submit(&forged), Err(KoinError::BadSignature));
        // Tamper with the memo after signing.
        let mut forged = tx.clone();
        forged.memo = "diamonds".to_string();
        assert!(!forged.verify());
        assert_eq!(ledger.submit(&forged), Err(KoinError::BadSignature));
        // Tamper with the recipient after signing.
        let carol = Identity::generate("carol").unwrap();
        let mut forged = tx.clone();
        forged.recipient = carol.peer_id().clone();
        assert!(!forged.verify());
        assert_eq!(ledger.submit(&forged), Err(KoinError::BadSignature));
        // Nothing moved.
        assert_eq!(ledger.balance_of(alice.peer_id()), 1_000);
        assert_eq!(ledger.balance_of(bob.peer_id()), 500);
    }

    #[test]
    fn replaying_the_same_transaction_is_rejected_no_double_spend() {
        let (mut ledger, alice, bob, _) = funded();
        let tx = CoinTransaction::new(&alice, bob.peer_id(), "coffee", 400).unwrap();
        ledger.submit(&tx).unwrap();
        assert_eq!(ledger.balance_of(alice.peer_id()), 600);
        assert_eq!(ledger.balance_of(bob.peer_id()), 900);
        // The exact same signed transaction again: replay.
        let err = ledger.submit(&tx).unwrap_err();
        assert!(matches!(err, KoinError::Replay(_)));
        // Balances unchanged: no double-spend.
        assert_eq!(ledger.balance_of(alice.peer_id()), 600);
        assert_eq!(ledger.balance_of(bob.peer_id()), 900);
        assert_eq!(ledger.transactions().len(), 1);
        // A fresh transaction for the same amount and memo is a
        // different transaction (same fields, same signature bytes —
        // Ed25519 is deterministic, so the id matches!). The ledger
        // must still refuse it: identical bytes, identical id.
        let twin = CoinTransaction::new(&alice, bob.peer_id(), "coffee", 400).unwrap();
        assert_eq!(twin.id(), tx.id());
        assert!(matches!(ledger.submit(&twin), Err(KoinError::Replay(_))));
        // A different amount is a different transaction and goes
        // through.
        let other = CoinTransaction::new(&alice, bob.peer_id(), "coffee", 401).unwrap();
        assert_ne!(other.id(), tx.id());
        ledger.submit(&other).unwrap();
        assert_eq!(ledger.balance_of(alice.peer_id()), 199);
    }

    #[test]
    fn zero_amount_rejected() {
        let (mut ledger, alice, bob, _) = funded();
        let tx = CoinTransaction::new(&alice, bob.peer_id(), "nothing", 0).unwrap();
        assert_eq!(ledger.submit(&tx), Err(KoinError::ZeroAmount));
        assert_eq!(ledger.balance_of(alice.peer_id()), 1_000);
    }

    #[test]
    fn transaction_encode_decode_round_trip() {
        let (alice, bob, _) = identities();
        let tx = CoinTransaction::new(&alice, bob.peer_id(), "coffee", 50).unwrap();
        let encoded = tx.encode();
        let decoded = CoinTransaction::decode(&encoded).unwrap();
        assert_eq!(decoded.memo, "coffee");
        assert_eq!(decoded.amount, 50);
        assert_eq!(decoded.sender.public_key(), alice.peer_id().public_key());
        assert_eq!(decoded.recipient.public_key(), bob.peer_id().public_key());
        assert_eq!(decoded.signature, tx.signature);
        assert_eq!(decoded.id(), tx.id());
        assert!(decoded.verify());
        assert!(CoinTransaction::decode(b"garbage").is_err());
        assert!(CoinTransaction::decode(&encoded[..encoded.len() - 1]).is_err());
        // An empty-memo transaction round-trips too.
        let plain = CoinTransaction::new(&alice, bob.peer_id(), "", 1).unwrap();
        assert_eq!(
            CoinTransaction::decode(&plain.encode()).unwrap().id(),
            plain.id()
        );
    }

    #[test]
    fn holders_list_is_human_readable_and_sorted() {
        let (mut ledger, alice, _bob, carol) = funded();
        // Carol holds nothing yet, so she is not listed.
        let names: Vec<String> = ledger
            .holders()
            .iter()
            .map(|(p, _)| p.display_name().to_string())
            .collect();
        assert_eq!(names, vec!["alice".to_string(), "bob".to_string()]);
        // After a transfer to Carol she appears, sorted by name.
        let tx = CoinTransaction::new(&alice, carol.peer_id(), "gift", 10).unwrap();
        ledger.submit(&tx).unwrap();
        let holders = ledger.holders();
        assert_eq!(holders.len(), 3);
        assert_eq!(holders[0].0.display_name(), "alice");
        assert_eq!(holders[0].1, 990);
        assert_eq!(holders[1].0.display_name(), "bob");
        assert_eq!(holders[1].1, 500);
        assert_eq!(holders[2].0.display_name(), "carol");
        assert_eq!(holders[2].1, 10);
    }

    #[test]
    fn memo_rules() {
        let (alice, bob, _) = identities();
        assert!(CoinTransaction::new(&alice, bob.peer_id(), "", 1).is_ok());
        assert!(CoinTransaction::new(&alice, bob.peer_id(), "ok memo", 1).is_ok());
        assert!(CoinTransaction::new(&alice, bob.peer_id(), "a".repeat(129).as_str(), 1).is_err());
        assert!(CoinTransaction::new(&alice, bob.peer_id(), "tab\there", 1).is_err());
    }

    // ------------------------------------------------------------------
    // Multi-signature tests
    // ------------------------------------------------------------------

    /// A ledger funding Alice with 1,000, plus a 2-of-3 proposal
    /// for Alice to pay Bob 100, approved by Bob, Carol, Dave.
    /// Ledger and proposal share the same identities.
    fn multisig_setup() -> (
        Ledger,
        MultisigTransaction,
        Identity,
        Identity,
        Identity,
        Identity,
    ) {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let carol = Identity::generate("carol").unwrap();
        let dave = Identity::generate("dave").unwrap();
        let ledger = Ledger::new(&[(alice.peer_id().clone(), 1_000)]).unwrap();
        let tx = CoinTransaction::new(&alice, bob.peer_id(), "shared pot", 100).unwrap();
        let proposal = MultisigTransaction::new(
            tx,
            2,
            vec![
                bob.peer_id().clone(),
                carol.peer_id().clone(),
                dave.peer_id().clone(),
            ],
        )
        .unwrap();
        (ledger, proposal, alice, bob, carol, dave)
    }

    #[test]
    fn multisig_submits_only_at_threshold() {
        let (mut ledger, proposal, alice, bob, carol, _) = multisig_setup();
        let id = proposal.id();
        ledger.propose_multisig(&proposal).unwrap();
        // One approval is not enough.
        assert_eq!(
            ledger.approve_multisig(&MultisigApproval::new(&bob, &id)),
            Ok(1)
        );
        assert_eq!(ledger.multisig_approval_count(&id), Some(1));
        assert_eq!(
            ledger.submit_multisig(&id),
            Err(KoinError::BelowThreshold { have: 1, need: 2 })
        );
        // Nothing moved yet.
        assert_eq!(ledger.balance_of(alice.peer_id()), 1_000);
        assert_eq!(ledger.balance_of(bob.peer_id()), 0);
        // Second distinct approver reaches the threshold.
        assert_eq!(
            ledger.approve_multisig(&MultisigApproval::new(&carol, &id)),
            Ok(2)
        );
        ledger.submit_multisig(&id).unwrap();
        assert_eq!(ledger.balance_of(alice.peer_id()), 900);
        assert_eq!(ledger.balance_of(bob.peer_id()), 100);
        // The proposal is gone; submitting again is unknown.
        assert_eq!(ledger.submit_multisig(&id), Err(KoinError::UnknownProposal));
        assert_eq!(ledger.multisig_approval_count(&id), None);
    }

    #[test]
    fn multisig_duplicate_approval_refused() {
        let (mut ledger, proposal, _, bob, _, _) = multisig_setup();
        let id = proposal.id();
        ledger.propose_multisig(&proposal).unwrap();
        ledger
            .approve_multisig(&MultisigApproval::new(&bob, &id))
            .unwrap();
        assert_eq!(
            ledger.approve_multisig(&MultisigApproval::new(&bob, &id)),
            Err(KoinError::DuplicateApproval)
        );
        assert_eq!(ledger.multisig_approval_count(&id), Some(1));
    }

    #[test]
    fn multisig_non_approver_refused() {
        let (mut ledger, proposal, alice, bob, _, _) = multisig_setup();
        let id = proposal.id();
        ledger.propose_multisig(&proposal).unwrap();
        // Alice is the sender, not an approver.
        assert_eq!(
            ledger.approve_multisig(&MultisigApproval::new(&alice, &id)),
            Err(KoinError::NotApprover)
        );
        // A stranger with no role at all is refused too.
        let stranger = Identity::generate("stranger").unwrap();
        assert_eq!(
            ledger.approve_multisig(&MultisigApproval::new(&stranger, &id)),
            Err(KoinError::NotApprover)
        );
        // An approval whose proposal id was tampered after signing
        // names a proposal that does not exist.
        let mut forged = MultisigApproval::new(&bob, &id);
        forged.proposal_id = "00".repeat(32);
        assert!(!forged.verify());
        assert_eq!(
            ledger.approve_multisig(&forged),
            Err(KoinError::UnknownProposal)
        );
    }

    #[test]
    fn multisig_bad_signature_approval_refused() {
        let (mut ledger, proposal, _, bob, _, _) = multisig_setup();
        let id = proposal.id();
        ledger.propose_multisig(&proposal).unwrap();
        // Sign for a different proposal id, then present it for this one.
        let wrong = MultisigApproval::new(&bob, "deadbeef");
        let mut forged = wrong.clone();
        forged.proposal_id = id.clone();
        assert!(!forged.verify());
        assert_eq!(
            ledger.approve_multisig(&forged),
            Err(KoinError::BadSignature)
        );
    }

    #[test]
    fn multisig_proposal_rules_enforced() {
        let (mut ledger, proposal, alice, bob, _, _) = multisig_setup();
        // Proposing the same proposal twice is a replay.
        ledger.propose_multisig(&proposal).unwrap();
        assert!(matches!(
            ledger.propose_multisig(&proposal),
            Err(KoinError::Replay(_))
        ));
        // A proposal whose underlying transaction is tampered (bad
        // signature) is refused at propose time.
        let mut bad_tx = CoinTransaction::new(&alice, bob.peer_id(), "shared pot", 100).unwrap();
        bad_tx.amount = 200;
        let bad = MultisigTransaction::new(bad_tx, 1, vec![bob.peer_id().clone()]).unwrap();
        assert_eq!(ledger.propose_multisig(&bad), Err(KoinError::BadSignature));
        // A zero-amount proposal is refused.
        let zero = CoinTransaction::new(&alice, bob.peer_id(), "nothing", 0).unwrap();
        let zero_proposal = MultisigTransaction::new(zero, 1, vec![bob.peer_id().clone()]).unwrap();
        assert_eq!(
            ledger.propose_multisig(&zero_proposal),
            Err(KoinError::ZeroAmount)
        );
    }

    #[test]
    fn multisig_submit_keeps_proposal_when_funds_short() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let carol = Identity::generate("carol").unwrap();
        let mut ledger = Ledger::new(&[(alice.peer_id().clone(), 1_000)]).unwrap();
        // Alice holds 1,000; propose a 900 transfer, then drain her
        // down to 100 with a normal transfer first.
        let tx = CoinTransaction::new(&alice, bob.peer_id(), "big pot", 900).unwrap();
        let proposal = MultisigTransaction::new(tx, 1, vec![bob.peer_id().clone()]).unwrap();
        let id = proposal.id();
        ledger.propose_multisig(&proposal).unwrap();
        ledger
            .approve_multisig(&MultisigApproval::new(&bob, &id))
            .unwrap();
        let drain = CoinTransaction::new(&alice, carol.peer_id(), "drain", 900).unwrap();
        ledger.submit(&drain).unwrap();
        assert_eq!(
            ledger.submit_multisig(&id),
            Err(KoinError::InsufficientFunds {
                needed: 900,
                available: 100
            })
        );
        // The proposal is still pending and can succeed once Alice
        // is funded again.
        assert_eq!(ledger.multisig_approval_count(&id), Some(1));
        let top_up = CoinTransaction::new(&carol, alice.peer_id(), "back", 800).unwrap();
        ledger.submit(&top_up).unwrap();
        ledger.submit_multisig(&id).unwrap();
        assert_eq!(ledger.balance_of(alice.peer_id()), 0);
        assert_eq!(ledger.balance_of(bob.peer_id()), 900);
    }

    // ------------------------------------------------------------------
    // Allowance tests
    // ------------------------------------------------------------------

    fn allowance_funded() -> (Ledger, Identity, Identity, Identity) {
        let (alice, bob, carol) = identities();
        let ledger = Ledger::new(&[(alice.peer_id().clone(), 1_000)]).unwrap();
        (ledger, alice, bob, carol)
    }

    #[test]
    fn allowance_draws_decrease_remaining() {
        let (mut ledger, alice, bob, carol) = allowance_funded();
        let allowance = Allowance::new(&alice, bob.peer_id(), 500, 0).unwrap();
        let id = allowance.id();
        ledger.grant_allowance(&allowance).unwrap();
        assert_eq!(ledger.allowance_remaining(&id), Some(500));
        let draw = AllowanceDrawRequest::new(&bob, &id, carol.peer_id(), "rent", 200).unwrap();
        let completed = ledger.spend_from_allowance(&draw, 0).unwrap();
        assert_eq!(completed.amount, 200);
        assert_eq!(ledger.allowance_remaining(&id), Some(300));
        assert_eq!(ledger.balance_of(alice.peer_id()), 800);
        assert_eq!(ledger.balance_of(carol.peer_id()), 200);
        assert_eq!(ledger.total_supply(), 1_000);
        // A second draw takes from what is left.
        let draw2 = AllowanceDrawRequest::new(&bob, &id, carol.peer_id(), "more", 300).unwrap();
        ledger.spend_from_allowance(&draw2, 0).unwrap();
        assert_eq!(ledger.allowance_remaining(&id), Some(0));
        assert_eq!(ledger.balance_of(alice.peer_id()), 500);
        assert_eq!(ledger.balance_of(carol.peer_id()), 500);
    }

    #[test]
    fn allowance_overdraw_refused() {
        let (mut ledger, alice, bob, carol) = allowance_funded();
        let allowance = Allowance::new(&alice, bob.peer_id(), 100, 0).unwrap();
        let id = allowance.id();
        ledger.grant_allowance(&allowance).unwrap();
        let too_much =
            AllowanceDrawRequest::new(&bob, &id, carol.peer_id(), "too much", 101).unwrap();
        assert_eq!(
            ledger.spend_from_allowance(&too_much, 0),
            Err(KoinError::AllowanceExceeded {
                needed: 101,
                remaining: 100
            })
        );
        // Nothing moved and the remainder is untouched.
        assert_eq!(ledger.allowance_remaining(&id), Some(100));
        assert_eq!(ledger.balance_of(alice.peer_id()), 1_000);
        assert_eq!(ledger.balance_of(carol.peer_id()), 0);
    }

    #[test]
    fn allowance_expired_refused_at_boundary() {
        let (mut ledger, alice, bob, carol) = allowance_funded();
        let allowance = Allowance::new(&alice, bob.peer_id(), 100, 1_000).unwrap();
        let id = allowance.id();
        ledger.grant_allowance(&allowance).unwrap();
        // At the expiry timestamp itself the allowance still works.
        let draw = AllowanceDrawRequest::new(&bob, &id, carol.peer_id(), "on time", 10).unwrap();
        ledger.spend_from_allowance(&draw, 1_000).unwrap();
        // One second past expiry it refuses.
        let late = AllowanceDrawRequest::new(&bob, &id, carol.peer_id(), "late", 10).unwrap();
        assert_eq!(
            ledger.spend_from_allowance(&late, 1_001),
            Err(KoinError::ExpiredAllowance { expiry: 1_000 })
        );
        assert_eq!(ledger.allowance_remaining(&id), Some(90));
    }

    #[test]
    fn allowance_revoked_refused_and_owner_only() {
        let (mut ledger, alice, bob, carol) = allowance_funded();
        let allowance = Allowance::new(&alice, bob.peer_id(), 100, 0).unwrap();
        let id = allowance.id();
        ledger.grant_allowance(&allowance).unwrap();
        // Only the owner can revoke: Bob revoking Alice's allowance
        // is refused.
        assert!(matches!(
            ledger.revoke_allowance(bob.peer_id(), &id),
            Err(KoinError::InvalidAllowance(_))
        ));
        // Revoking an unknown allowance is refused.
        assert_eq!(
            ledger.revoke_allowance(alice.peer_id(), "00".repeat(32).as_str()),
            Err(KoinError::UnknownAllowance)
        );
        ledger.revoke_allowance(alice.peer_id(), &id).unwrap();
        assert_eq!(ledger.allowance_remaining(&id), Some(0));
        let draw = AllowanceDrawRequest::new(&bob, &id, carol.peer_id(), "too late", 10).unwrap();
        assert_eq!(
            ledger.spend_from_allowance(&draw, 0),
            Err(KoinError::AllowanceRevoked)
        );
        // Revoking again is a harmless no-op.
        ledger.revoke_allowance(alice.peer_id(), &id).unwrap();
    }

    #[test]
    fn allowance_draw_beyond_owner_balance_refused() {
        let (mut ledger, alice, bob, carol) = allowance_funded();
        // Allowance of 500, but Alice only holds 1,000 and then
        // spends 900 herself, leaving 100.
        let allowance = Allowance::new(&alice, bob.peer_id(), 500, 0).unwrap();
        let id = allowance.id();
        ledger.grant_allowance(&allowance).unwrap();
        let spend = CoinTransaction::new(&alice, carol.peer_id(), "spree", 900).unwrap();
        ledger.submit(&spend).unwrap();
        let draw = AllowanceDrawRequest::new(&bob, &id, carol.peer_id(), "too big", 200).unwrap();
        assert_eq!(
            ledger.spend_from_allowance(&draw, 0),
            Err(KoinError::InsufficientFunds {
                needed: 200,
                available: 100
            })
        );
        // Remainder untouched, and a draw within the balance works.
        assert_eq!(ledger.allowance_remaining(&id), Some(500));
        let small = AllowanceDrawRequest::new(&bob, &id, carol.peer_id(), "fits", 100).unwrap();
        ledger.spend_from_allowance(&small, 0).unwrap();
        assert_eq!(ledger.allowance_remaining(&id), Some(400));
    }

    #[test]
    fn allowance_draw_by_wrong_spender_refused() {
        let (mut ledger, alice, bob, carol) = allowance_funded();
        let allowance = Allowance::new(&alice, bob.peer_id(), 100, 0).unwrap();
        let id = allowance.id();
        ledger.grant_allowance(&allowance).unwrap();
        // Carol is not the spender on this allowance.
        let draw = AllowanceDrawRequest::new(&carol, &id, carol.peer_id(), "mine", 10).unwrap();
        assert!(matches!(
            ledger.spend_from_allowance(&draw, 0),
            Err(KoinError::InvalidAllowance(_))
        ));
        // A draw request with a tampered signature is refused.
        let mut forged = AllowanceDrawRequest::new(&bob, &id, carol.peer_id(), "rent", 10).unwrap();
        forged.amount = 50;
        assert_eq!(
            ledger.spend_from_allowance(&forged, 0),
            Err(KoinError::BadSignature)
        );
        // An unknown allowance id is refused.
        let unknown =
            AllowanceDrawRequest::new(&bob, "00".repeat(32).as_str(), carol.peer_id(), "x", 1)
                .unwrap();
        assert_eq!(
            ledger.spend_from_allowance(&unknown, 0),
            Err(KoinError::UnknownAllowance)
        );
    }

    #[test]
    fn allowance_duplicate_grant_is_replay() {
        let (mut ledger, alice, bob, _) = allowance_funded();
        let allowance = Allowance::new(&alice, bob.peer_id(), 100, 0).unwrap();
        ledger.grant_allowance(&allowance).unwrap();
        assert!(matches!(
            ledger.grant_allowance(&allowance),
            Err(KoinError::Replay(_))
        ));
        // A forged grant (bad signature) is refused.
        let mut forged = allowance.clone();
        forged.amount = 200;
        assert_eq!(
            ledger.grant_allowance(&forged),
            Err(KoinError::BadSignature)
        );
    }

    // ------------------------------------------------------------------
    // Statement tests
    // ------------------------------------------------------------------

    #[test]
    fn statement_lists_in_and_out_with_running_balances() {
        let (mut ledger, alice, bob, carol) = funded();
        // Alice: 1,000 genesis. Sends 250 to Bob, receives 50 from
        // Bob, sends 100 to Carol.
        let tx1 = CoinTransaction::new(&alice, bob.peer_id(), "coffee", 250).unwrap();
        ledger.submit(&tx1).unwrap();
        let tx2 = CoinTransaction::new(&bob, alice.peer_id(), "refund", 50).unwrap();
        ledger.submit(&tx2).unwrap();
        let tx3 = CoinTransaction::new(&alice, carol.peer_id(), "gift", 100).unwrap();
        ledger.submit(&tx3).unwrap();
        let statement = ledger.statement(alice.peer_id());
        assert_eq!(statement.entries.len(), 3);
        assert_eq!(statement.entries[0].amount, -250);
        assert_eq!(statement.entries[0].balance_after, 750);
        assert_eq!(
            statement.entries[0].counterparty.public_key(),
            bob.peer_id().public_key()
        );
        assert_eq!(statement.entries[0].kind, EntryKind::Transfer);
        assert_eq!(statement.entries[1].amount, 50);
        assert_eq!(statement.entries[1].balance_after, 800);
        assert_eq!(statement.entries[2].amount, -100);
        assert_eq!(statement.entries[2].balance_after, 700);
        assert_eq!(statement.total_in, 50);
        assert_eq!(statement.total_out, 350);
        assert_eq!(statement.final_balance, 700);
        assert_eq!(statement.final_balance, ledger.balance_of(alice.peer_id()));
        // Bob's statement: +250 in, -50 out.
        let bob_statement = ledger.statement(bob.peer_id());
        assert_eq!(bob_statement.entries.len(), 2);
        assert_eq!(bob_statement.entries[0].amount, 250);
        assert_eq!(bob_statement.entries[0].balance_after, 750);
        assert_eq!(bob_statement.entries[1].amount, -50);
        assert_eq!(bob_statement.entries[1].balance_after, 700);
        assert_eq!(bob_statement.total_in, 250);
        assert_eq!(bob_statement.total_out, 50);
        // Carol: one entry, +100.
        let carol_statement = ledger.statement(carol.peer_id());
        assert_eq!(carol_statement.entries.len(), 1);
        assert_eq!(carol_statement.entries[0].amount, 100);
        assert_eq!(carol_statement.entries[0].balance_after, 100);
        // A stranger with no activity: empty, zero final balance.
        let stranger = Identity::generate("stranger").unwrap();
        let empty = ledger.statement(stranger.peer_id());
        assert!(empty.entries.is_empty());
        assert_eq!(empty.total_in, 0);
        assert_eq!(empty.total_out, 0);
        assert_eq!(empty.final_balance, 0);
    }

    #[test]
    fn statement_includes_allowance_draws_in_order() {
        let (mut ledger, alice, bob, carol) = allowance_funded();
        let allowance = Allowance::new(&alice, bob.peer_id(), 500, 0).unwrap();
        let id = allowance.id();
        ledger.grant_allowance(&allowance).unwrap();
        // A normal transfer first, then a draw, then another
        // transfer: the statement must interleave by acceptance.
        let tx1 = CoinTransaction::new(&alice, carol.peer_id(), "gift", 100).unwrap();
        ledger.submit(&tx1).unwrap();
        let draw = AllowanceDrawRequest::new(&bob, &id, carol.peer_id(), "rent", 200).unwrap();
        ledger.spend_from_allowance(&draw, 0).unwrap();
        let tx2 = CoinTransaction::new(&carol, alice.peer_id(), "thanks", 50).unwrap();
        ledger.submit(&tx2).unwrap();
        let statement = ledger.statement(alice.peer_id());
        assert_eq!(statement.entries.len(), 3);
        assert_eq!(statement.entries[0].kind, EntryKind::Transfer);
        assert_eq!(statement.entries[0].amount, -100);
        assert_eq!(statement.entries[0].balance_after, 900);
        assert_eq!(statement.entries[1].kind, EntryKind::AllowanceDraw);
        assert_eq!(statement.entries[1].amount, -200);
        assert_eq!(statement.entries[1].balance_after, 700);
        assert_eq!(
            statement.entries[1].counterparty.public_key(),
            carol.peer_id().public_key()
        );
        assert_eq!(statement.entries[2].amount, 50);
        assert_eq!(statement.entries[2].balance_after, 750);
        assert_eq!(statement.total_out, 300);
        assert_eq!(statement.total_in, 50);
        // Carol received both the gift and the draw.
        let carol_statement = ledger.statement(carol.peer_id());
        assert_eq!(carol_statement.entries.len(), 3);
        assert_eq!(carol_statement.entries[1].kind, EntryKind::AllowanceDraw);
        assert_eq!(carol_statement.entries[1].amount, 200);
    }

    #[test]
    fn statement_self_transfer_is_out_then_in() {
        let (mut ledger, alice, _, _) = funded();
        let tx = CoinTransaction::new(&alice, alice.peer_id(), "to self", 100).unwrap();
        ledger.submit(&tx).unwrap();
        let statement = ledger.statement(alice.peer_id());
        assert_eq!(statement.entries.len(), 2);
        assert_eq!(statement.entries[0].amount, -100);
        assert_eq!(statement.entries[0].balance_after, 900);
        assert_eq!(statement.entries[1].amount, 100);
        assert_eq!(statement.entries[1].balance_after, 1_000);
        assert_eq!(statement.total_in, 100);
        assert_eq!(statement.total_out, 100);
        assert_eq!(statement.final_balance, 1_000);
    }

    #[test]
    fn statement_export_import_round_trips() {
        let (mut ledger, alice, bob, carol) = allowance_funded();
        let allowance = Allowance::new(&alice, bob.peer_id(), 500, 0).unwrap();
        let id = allowance.id();
        ledger.grant_allowance(&allowance).unwrap();
        let tx = CoinTransaction::new(&alice, carol.peer_id(), "gift", 100).unwrap();
        ledger.submit(&tx).unwrap();
        let draw = AllowanceDrawRequest::new(&bob, &id, carol.peer_id(), "rent", 200).unwrap();
        ledger.spend_from_allowance(&draw, 0).unwrap();
        let statement = ledger.statement(alice.peer_id());
        let encoded = statement.encode();
        let decoded = Statement::decode(&encoded).unwrap();
        // Decode rebuilds display names from keys, so compare by
        // key and by every field the encoding carries.
        assert_eq!(decoded.peer.public_key(), statement.peer.public_key());
        assert_eq!(decoded.total_in, statement.total_in);
        assert_eq!(decoded.total_out, statement.total_out);
        assert_eq!(decoded.final_balance, statement.final_balance);
        assert_eq!(decoded.entries.len(), statement.entries.len());
        for (d, e) in decoded.entries.iter().zip(statement.entries.iter()) {
            assert_eq!(d.seq, e.seq);
            assert_eq!(d.id, e.id);
            assert_eq!(d.counterparty.public_key(), e.counterparty.public_key());
            assert_eq!(d.memo, e.memo);
            assert_eq!(d.amount, e.amount);
            assert_eq!(d.balance_after, e.balance_after);
            assert_eq!(d.kind, e.kind);
        }
        // An empty statement round-trips too.
        let stranger = Identity::generate("stranger").unwrap();
        let empty = ledger.statement(stranger.peer_id());
        let decoded_empty = Statement::decode(&empty.encode()).unwrap();
        assert_eq!(decoded_empty.peer.public_key(), empty.peer.public_key());
        assert!(decoded_empty.entries.is_empty());
        assert_eq!(decoded_empty.final_balance, empty.final_balance);
    }
}
