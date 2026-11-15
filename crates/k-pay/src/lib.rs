//! K•PAY — payments on top of X•Koin.
//!
//! A payment starts as an invoice: a human-readable id, a
//! recipient, an amount, a memo, and an expiry — a simple
//! monotonically increasing timestamp (or sequence) supplied by
//! the caller. The recipient signs the invoice, so anyone can
//! verify who is asking to be paid and for what.
//!
//! Paying an invoice settles it by submitting a signed coin
//! transaction from the payer to the recipient, and returns a
//! signed receipt: invoice id, payer, recipient, amount, and the
//! payer's signature. The receipt is proof of payment that anyone
//! can verify without seeing the ledger.
//!
//! The rules are simple and deterministic:
//!
//! * An invoice's signature must verify against its recipient.
//!   An invoice with a bad signature is refused.
//! * An invoice can only be paid before its expiry: paying at a
//!   time after the expiry is rejected. The caller supplies the
//!   current time, so no wall clock is needed.
//! * An invoice can only be paid once. Paying twice is rejected,
//!   and the ledger is unchanged by the second attempt.
//! * Payment settles through the coin ledger, so all coin rules
//!   still apply: the payer must hold enough, and the signed
//!   coin transaction cannot be replayed.
//!
//! Invoice and receipt ids are human-readable (`inv-coffee-001`),
//! never hashes.

use std::collections::{HashMap, HashSet};
use std::fmt;

use kernel_crypto::Ed25519Signature;
use kernel_identity::{Identity, IdentityError, PeerId};
use x_koin::{CoinTransaction, KoinError, Ledger};

/// Domain separator for invoice signatures: a signature covers
/// `b"KPAY-INV1" || id || recipient key || amount || memo || expiry`.
const INVOICE_SIGN_DOMAIN: &[u8] = b"KPAY-INV1";

/// Domain separator for receipt signatures: a signature covers
/// `b"KPAY-RCPT1" || invoice id || payer key || recipient key ||
/// amount`.
const RECEIPT_SIGN_DOMAIN: &[u8] = b"KPAY-RCPT1";

/// Errors returned by payment operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PayError {
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// Wraps a [`KoinError`] from settlement.
    Koin(KoinError),
    /// An invoice id, amount, or memo broke the rules.
    InvalidInvoice(String),
    /// An invoice's signature did not verify against its recipient.
    BadInvoiceSignature,
    /// An invoice with this id already exists.
    DuplicateInvoice(String),
    /// The invoice's expiry has passed.
    Expired {
        /// The expiry on the invoice.
        expiry: u64,
        /// The time payment was attempted.
        now: u64,
    },
    /// The invoice was already paid.
    AlreadyPaid(String),
}

impl fmt::Display for PayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PayError::Identity(e) => write!(f, "identity error: {e}"),
            PayError::Koin(e) => write!(f, "coin error: {e}"),
            PayError::InvalidInvoice(msg) => write!(f, "invalid invoice: {msg}"),
            PayError::BadInvoiceSignature => {
                write!(f, "invoice signature is invalid")
            }
            PayError::DuplicateInvoice(id) => {
                write!(f, "an invoice with id '{id}' already exists")
            }
            PayError::Expired { expiry, now } => write!(
                f,
                "invoice expired at {expiry}, payment attempted at {now}"
            ),
            PayError::AlreadyPaid(id) => write!(f, "invoice '{id}' was already paid"),
        }
    }
}

impl std::error::Error for PayError {}

impl From<IdentityError> for PayError {
    fn from(e: IdentityError) -> Self {
        PayError::Identity(e)
    }
}

impl From<KoinError> for PayError {
    fn from(e: KoinError) -> Self {
        PayError::Koin(e)
    }
}

/// Rules for human-readable ids: 1-64 characters, ASCII letters,
/// digits, `-`, `_`, `.`. No spaces, no hashes.
fn validate_id(id: &str, what: &str) -> Result<(), PayError> {
    if id.is_empty() {
        return Err(PayError::InvalidInvoice(format!(
            "{what} id must not be empty"
        )));
    }
    if id.len() > 64 {
        return Err(PayError::InvalidInvoice(format!(
            "{what} id must be at most 64 characters"
        )));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Err(PayError::InvalidInvoice(format!(
            "{what} id may only use letters, digits, '-', '_' and '.'"
        )));
    }
    Ok(())
}

/// Rules for memos: at most 128 printable characters.
fn validate_memo(memo: &str) -> Result<(), PayError> {
    if memo.len() > 128 {
        return Err(PayError::InvalidInvoice(
            "memo must be at most 128 characters".to_string(),
        ));
    }
    if !memo.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
        return Err(PayError::InvalidInvoice(
            "memo must be printable ASCII".to_string(),
        ));
    }
    Ok(())
}

/// The message a recipient signs for an invoice.
fn invoice_sign_message(
    id: &str,
    recipient: &PeerId,
    amount: u64,
    memo: &str,
    expiry: u64,
) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend_from_slice(INVOICE_SIGN_DOMAIN);
    msg.extend_from_slice(&(id.len() as u16).to_le_bytes());
    msg.extend_from_slice(id.as_bytes());
    msg.extend_from_slice(recipient.public_key().as_bytes());
    msg.extend_from_slice(&amount.to_le_bytes());
    msg.extend_from_slice(&(memo.len() as u16).to_le_bytes());
    msg.extend_from_slice(memo.as_bytes());
    msg.extend_from_slice(&expiry.to_le_bytes());
    msg
}

/// The message a payer signs for a receipt.
fn receipt_sign_message(
    invoice_id: &str,
    payer: &PeerId,
    recipient: &PeerId,
    amount: u64,
) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend_from_slice(RECEIPT_SIGN_DOMAIN);
    msg.extend_from_slice(&(invoice_id.len() as u16).to_le_bytes());
    msg.extend_from_slice(invoice_id.as_bytes());
    msg.extend_from_slice(payer.public_key().as_bytes());
    msg.extend_from_slice(recipient.public_key().as_bytes());
    msg.extend_from_slice(&amount.to_le_bytes());
    msg
}

/// An invoice: a signed request to be paid. Anyone holding it can
/// verify the recipient's signature before paying.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Invoice {
    /// Human-readable id, e.g. `"inv-coffee-001"`.
    pub id: String,
    /// Who is to be paid.
    pub recipient: PeerId,
    /// The amount, in the coin's smallest unit.
    pub amount: u64,
    /// A human-readable note, e.g. `"coffee"`.
    pub memo: String,
    /// The last time (or sequence) at which payment is accepted.
    pub expiry: u64,
    /// The recipient's signature over id, recipient, amount,
    /// memo, and expiry.
    pub signature: Ed25519Signature,
}

impl Invoice {
    /// Create and sign an invoice as `recipient`. The amount must
    /// be greater than zero; the id, memo, and expiry are validated
    /// but the expiry itself is just a number the caller supplies.
    pub fn new(
        recipient: &Identity,
        id: &str,
        amount: u64,
        memo: &str,
        expiry: u64,
    ) -> Result<Self, PayError> {
        validate_id(id, "invoice")?;
        validate_memo(memo)?;
        if amount == 0 {
            return Err(PayError::InvalidInvoice(
                "invoice amount must be greater than zero".to_string(),
            ));
        }
        let recipient_id = recipient.peer_id().clone();
        let msg = invoice_sign_message(id, &recipient_id, amount, memo, expiry);
        let signature = recipient.sign(&msg);
        Ok(Self {
            id: id.to_string(),
            recipient: recipient_id,
            amount,
            memo: memo.to_string(),
            expiry,
            signature,
        })
    }

    /// True if the signature verifies against the recipient for
    /// this exact id, amount, memo, and expiry. Any tampering
    /// fails this.
    pub fn verify(&self) -> bool {
        let msg = invoice_sign_message(
            &self.id,
            &self.recipient,
            self.amount,
            &self.memo,
            self.expiry,
        );
        self.recipient.verify(&msg, &self.signature).is_ok()
    }
}

/// A receipt: signed proof that an invoice was paid. Anyone holding
/// it can verify the payer's signature without seeing the ledger.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Receipt {
    /// The id of the invoice that was paid.
    pub invoice_id: String,
    /// Who paid.
    pub payer: PeerId,
    /// Who was paid.
    pub recipient: PeerId,
    /// The amount that moved.
    pub amount: u64,
    /// The payer's signature over invoice id, payer, recipient,
    /// and amount.
    pub signature: Ed25519Signature,
}

impl Receipt {
    /// True if the signature verifies against the payer for this
    /// exact invoice id, recipient, and amount. Any tampering
    /// fails this.
    pub fn verify(&self) -> bool {
        let msg = receipt_sign_message(&self.invoice_id, &self.payer, &self.recipient, self.amount);
        self.payer.verify(&msg, &self.signature).is_ok()
    }
}

/// A payment service: a coin ledger plus the invoices issued
/// against it. Invoices are created by recipients and can arrive
/// from any peer; only their signatures authenticate them, so an
/// invoice does not need to have been created through this
/// service to be payable here.
pub struct PaymentService {
    ledger: Ledger,
    /// Invoices registered through [`PaymentService::create_invoice`].
    invoices: HashMap<String, Invoice>,
    /// Ids of invoices already paid.
    paid: HashSet<String>,
}

impl PaymentService {
    /// Create a payment service whose ledger starts at the given
    /// genesis allocation.
    pub fn new(genesis: &[(PeerId, u64)]) -> Result<Self, PayError> {
        Ok(Self {
            ledger: Ledger::new(genesis)?,
            invoices: HashMap::new(),
            paid: HashSet::new(),
        })
    }

    /// The underlying coin ledger.
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// The balance of a peer in the underlying ledger.
    pub fn balance_of(&self, peer: &PeerId) -> u64 {
        self.ledger.balance_of(peer)
    }

    /// True if the invoice with this id was already paid.
    pub fn is_paid(&self, id: &str) -> bool {
        self.paid.contains(id)
    }

    /// The registered invoice with this id, if any.
    pub fn get_invoice(&self, id: &str) -> Option<&Invoice> {
        self.invoices.get(id)
    }

    /// Create and register an invoice as `recipient`. The id must
    /// not already be registered.
    pub fn create_invoice(
        &mut self,
        recipient: &Identity,
        id: &str,
        amount: u64,
        memo: &str,
        expiry: u64,
    ) -> Result<Invoice, PayError> {
        if self.invoices.contains_key(id) {
            return Err(PayError::DuplicateInvoice(id.to_string()));
        }
        let invoice = Invoice::new(recipient, id, amount, memo, expiry)?;
        self.invoices.insert(id.to_string(), invoice.clone());
        Ok(invoice)
    }

    /// Pay an invoice as `payer` at time `now`. Validates, in
    /// order: the invoice signature, that it has not been paid,
    /// and that it has not expired; then settles through the coin
    /// ledger and returns a receipt signed by the payer. A
    /// rejected payment changes nothing — no coins move and the
    /// invoice stays unpaid.
    pub fn pay_invoice(
        &mut self,
        invoice: &Invoice,
        payer: &Identity,
        now: u64,
    ) -> Result<Receipt, PayError> {
        if !invoice.verify() {
            return Err(PayError::BadInvoiceSignature);
        }
        if self.paid.contains(&invoice.id) {
            return Err(PayError::AlreadyPaid(invoice.id.clone()));
        }
        if now > invoice.expiry {
            return Err(PayError::Expired {
                expiry: invoice.expiry,
                now,
            });
        }
        // Settle through the ledger: a signed coin transaction
        // from the payer to the recipient, memoed with the
        // invoice id so the payment is recognizable on the ledger.
        let memo = format!("pay:{}", invoice.id);
        let tx = CoinTransaction::new(payer, &invoice.recipient, &memo, invoice.amount)?;
        self.ledger.submit(&tx)?;
        self.paid.insert(invoice.id.clone());
        let payer_id = payer.peer_id().clone();
        let msg = receipt_sign_message(
            &invoice.id,
            &payer_id,
            &invoice.recipient,
            invoice.amount,
        );
        let signature = payer.sign(&msg);
        Ok(Receipt {
            invoice_id: invoice.id.clone(),
            payer: payer_id,
            recipient: invoice.recipient.clone(),
            amount: invoice.amount,
            signature,
        })
    }

    /// Verify a receipt's payer signature. This is a plain
    /// function, not a method, because a receipt is verified
    /// without any service state.
    pub fn verify_receipt(receipt: &Receipt) -> bool {
        receipt.verify()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identities() -> (Identity, Identity) {
        (
            Identity::generate("alice").unwrap(),
            Identity::generate("bob").unwrap(),
        )
    }

    fn funded() -> (PaymentService, Identity, Identity) {
        let (alice, bob) = identities();
        let service = PaymentService::new(&[
            (alice.peer_id().clone(), 1_000),
            (bob.peer_id().clone(), 100),
        ])
        .unwrap();
        (service, alice, bob)
    }

    #[test]
    fn invoice_and_payment_settles_and_balances_move() {
        let (mut service, alice, bob) = funded();
        // Bob invoices Alice for 250.
        let invoice = service
            .create_invoice(&bob, "inv-coffee-001", 250, "coffee", 1_000)
            .unwrap();
        assert!(invoice.verify());
        assert_eq!(invoice.recipient.display_name(), "bob");
        assert!(!service.is_paid("inv-coffee-001"));
        // Alice pays before expiry.
        let receipt = service.pay_invoice(&invoice, &alice, 500).unwrap();
        assert_eq!(receipt.invoice_id, "inv-coffee-001");
        assert_eq!(receipt.amount, 250);
        assert_eq!(receipt.payer.display_name(), "alice");
        assert_eq!(receipt.recipient.display_name(), "bob");
        assert!(receipt.verify());
        assert!(PaymentService::verify_receipt(&receipt));
        assert!(service.is_paid("inv-coffee-001"));
        // Balances moved: Alice 1,000 -> 750, Bob 100 -> 350.
        assert_eq!(service.balance_of(alice.peer_id()), 750);
        assert_eq!(service.balance_of(bob.peer_id()), 350);
        // The settlement is on the ledger as a memoed transaction.
        assert_eq!(service.ledger().transactions().len(), 1);
        assert_eq!(
            service.ledger().transactions()[0].memo,
            "pay:inv-coffee-001"
        );
    }

    #[test]
    fn paying_an_expired_invoice_is_rejected() {
        let (mut service, alice, bob) = funded();
        let invoice = service
            .create_invoice(&bob, "inv-late-001", 100, "late", 500)
            .unwrap();
        // At the expiry itself, payment is still accepted.
        let receipt = service
            .pay_invoice(&invoice, &alice, 500)
            .expect("paying at expiry should work");
        assert!(receipt.verify());
        assert_eq!(service.balance_of(alice.peer_id()), 900);
        // A second invoice, paid one tick past expiry, is refused.
        let late = service
            .create_invoice(&bob, "inv-late-002", 100, "late", 500)
            .unwrap();
        assert_eq!(
            service.pay_invoice(&late, &alice, 501),
            Err(PayError::Expired {
                expiry: 500,
                now: 501
            })
        );
        // Nothing moved for the refused payment.
        assert_eq!(service.balance_of(alice.peer_id()), 900);
        assert_eq!(service.balance_of(bob.peer_id()), 200);
        assert!(!service.is_paid("inv-late-002"));
    }

    #[test]
    fn paying_twice_is_rejected() {
        let (mut service, alice, bob) = funded();
        let invoice = service
            .create_invoice(&bob, "inv-once-001", 100, "once", 1_000)
            .unwrap();
        service.pay_invoice(&invoice, &alice, 10).unwrap();
        assert_eq!(service.balance_of(alice.peer_id()), 900);
        assert_eq!(service.balance_of(bob.peer_id()), 200);
        // The second attempt is refused and moves nothing.
        assert_eq!(
            service.pay_invoice(&invoice, &alice, 11),
            Err(PayError::AlreadyPaid("inv-once-001".to_string()))
        );
        assert_eq!(service.balance_of(alice.peer_id()), 900);
        assert_eq!(service.balance_of(bob.peer_id()), 200);
        assert_eq!(service.ledger().transactions().len(), 1);
    }

    #[test]
    fn receipt_verifies_and_tampered_receipt_fails() {
        let (mut service, alice, bob) = funded();
        let invoice = service
            .create_invoice(&bob, "inv-rcpt-001", 50, "tea", 1_000)
            .unwrap();
        let receipt = service.pay_invoice(&invoice, &alice, 10).unwrap();
        assert!(receipt.verify());
        // Tamper with the amount: the payer's signature no longer
        // covers it.
        let mut forged = receipt.clone();
        forged.amount = 5_000;
        assert!(!forged.verify());
        assert!(!PaymentService::verify_receipt(&forged));
        // Tamper with the invoice id.
        let mut forged = receipt.clone();
        forged.invoice_id = "inv-rcpt-999".to_string();
        assert!(!forged.verify());
        // Tamper with the payer.
        let mut forged = receipt.clone();
        forged.payer = bob.peer_id().clone();
        assert!(!forged.verify());
    }

    #[test]
    fn invoice_with_bad_signature_is_rejected() {
        let (mut service, alice, bob) = funded();
        let invoice = service
            .create_invoice(&bob, "inv-bad-001", 100, "bad", 1_000)
            .unwrap();
        // Change the amount after the recipient signed it.
        let mut forged = invoice.clone();
        forged.amount = 999;
        assert!(!forged.verify());
        assert_eq!(
            service.pay_invoice(&forged, &alice, 10),
            Err(PayError::BadInvoiceSignature)
        );
        assert_eq!(service.balance_of(alice.peer_id()), 1_000);
        assert!(!service.is_paid("inv-bad-001"));
    }

    #[test]
    fn paying_with_insufficient_funds_is_rejected() {
        let (mut service, alice, bob) = funded();
        // Bob holds 100; Alice invoices him for 500.
        let invoice = service
            .create_invoice(&alice, "inv-poor-001", 500, "too much", 1_000)
            .unwrap();
        let err = service.pay_invoice(&invoice, &bob, 10).unwrap_err();
        assert!(matches!(err, PayError::Koin(KoinError::InsufficientFunds { .. })));
        assert!(!service.is_paid("inv-poor-001"));
        assert_eq!(service.balance_of(bob.peer_id()), 100);
        assert_eq!(service.balance_of(alice.peer_id()), 1_000);
    }

    #[test]
    fn invoice_rules() {
        let (mut service, alice, bob) = funded();
        // Duplicate ids are refused.
        service
            .create_invoice(&bob, "inv-dup-001", 10, "one", 1_000)
            .unwrap();
        assert_eq!(
            service.create_invoice(&bob, "inv-dup-001", 10, "two", 1_000),
            Err(PayError::DuplicateInvoice("inv-dup-001".to_string()))
        );
        // Bad ids, zero amounts, and over-long memos are refused.
        assert!(service
            .create_invoice(&bob, "", 10, "x", 1_000)
            .is_err());
        assert!(service
            .create_invoice(&bob, "has space", 10, "x", 1_000)
            .is_err());
        assert!(service
            .create_invoice(&bob, "inv-zero-001", 0, "x", 1_000)
            .is_err());
        assert!(service
            .create_invoice(&bob, "inv-long-001", 10, "a".repeat(129).as_str(), 1_000)
            .is_err());
        // An invoice that was never registered can still be paid:
        // only its signature authenticates it.
        let unregistered = Invoice::new(&alice, "inv-stranger-001", 25, "from afar", 1_000)
            .unwrap();
        let receipt = service.pay_invoice(&unregistered, &bob, 10).unwrap();
        assert!(receipt.verify());
        assert_eq!(service.balance_of(bob.peer_id()), 75);
        assert_eq!(service.balance_of(alice.peer_id()), 1_025);
    }
}
