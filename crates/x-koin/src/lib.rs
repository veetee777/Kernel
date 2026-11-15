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
fn validate_memo(memo: &str) -> Result<(), KoinError> {
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
fn fingerprint_name(key: &kernel_crypto::Ed25519PublicKey) -> String {
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
        let memo_len =
            u16::from_le_bytes(bytes[70..72].try_into().expect("2-byte slice")) as usize;
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
}

impl Ledger {
    /// Create a ledger with a genesis allocation: a fixed initial
    /// supply assigned to the named holders. The allocation is the
    /// only way coins ever enter the ledger.
    pub fn new(genesis: &[(PeerId, u64)]) -> Result<Self, KoinError> {
        let mut balances: HashMap<[u8; 32], u64> = HashMap::new();
        let mut names: HashMap<[u8; 32], String> = HashMap::new();
        let mut genesis_supply: u64 = 0;
        for (peer, amount) in genesis {
            let key = *peer.public_key().as_bytes();
            let entry = balances.entry(key).or_insert(0);
            *entry = entry
                .checked_add(*amount)
                .ok_or(KoinError::Overflow)?;
            genesis_supply = genesis_supply
                .checked_add(*amount)
                .ok_or(KoinError::Overflow)?;
            names.entry(key).or_insert_with(|| peer.display_name().to_string());
        }
        Ok(Self {
            balances,
            names,
            seen: HashSet::new(),
            history: Vec::new(),
            genesis_supply,
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
        self.balances.values().fold(0u64, |acc, b| acc.saturating_add(*b))
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
                let name = self
                    .names
                    .get(key)
                    .cloned()
                    .unwrap_or_else(|| fingerprint_name(
                        &kernel_crypto::Ed25519PublicKey::from_bytes(key)
                            .expect("stored keys are valid"),
                    ));
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
        self.history.push(tx.clone());
        Ok(())
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
}
