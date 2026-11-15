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
//!
//! On top of one-off invoices, the service also supports:
//!
//! * **Recurring payments**: a signed subscription (payer, payee,
//!   amount, interval, total count, first due time). The payer
//!   pre-signs one coin transaction and one receipt per occurrence
//!   at creation — the service never holds a private key, so it
//!   can only submit what the payer already authorized. `tick(now)`
//!   submits every occurrence due at `now` through the normal
//!   ledger path; a failed occurrence (e.g. insufficient funds)
//!   is recorded as failed and the subscription moves on to the
//!   next due time. After the total count of occurrences the
//!   subscription is finished; it can be cancelled before that.
//! * **Refunds**: a signed refund of a paid payment, authorized
//!   by the original recipient, for exactly the original amount.
//!   A payment can be refunded exactly once; the coins move back
//!   through the same ledger path.
//! * **Payment history**: every settled payment is recorded with
//!   its receipt, timestamps, and status (`Paid` / `Refunded`).
//!   History is listed by payer or by payee, newest first, with
//!   a limit. History is in-memory in this crate.
//!
//! All time is caller-supplied (unix seconds). This crate never
//! reads the system clock, so a wrong clock means wrong expiries
//! and wrong due times — that is the caller's responsibility.
//!
//! ## OWNER-FREE LAW
//!
//! Everything inside KERNEL is 100% free for the owner (Tristan):
//! he pays nothing, ever. [`PaymentService::set_owner`] registers
//! the owner's [`PeerId`] on a service; from then on, paying an
//! invoice as the owner satisfies it without moving any coins — a
//! valid owner-free [`Receipt`] (amount 0, `owner_free` set, signed
//! by the payer) is still issued, the invoice is marked paid, and
//! the payment is recorded with amount 0. Subscription occurrences
//! for the owner are pre-signed owner-free at registration and
//! execute free on `tick`. The exemption is local: a service that
//! does not know the owner charges normally, and this crate does
//! not decide who the owner is — that is the node's job. The owner
//! must be registered before a subscription is registered for its
//! occurrences to be owner-free: the prepaid receipts are signed
//! at authorization time and cannot be rewritten later.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use kernel_crypto::Ed25519Signature;
use kernel_identity::{Identity, IdentityError, PeerId};
use x_koin::{CoinTransaction, KoinError, Ledger};

/// Domain separator for invoice signatures: a signature covers
/// `b"KPAY-INV1" || id || recipient key || amount || memo || expiry`.
const INVOICE_SIGN_DOMAIN: &[u8] = b"KPAY-INV1";

/// Domain separator for receipt signatures: a signature covers
/// `b"KPAY-RCPT1" || invoice id || payer key || recipient key ||
/// amount || owner-free flag (0 or 1)`. The flag is covered so an
/// owner-free receipt cannot be turned into a paid one (or the
/// reverse) without breaking the signature.
const RECEIPT_SIGN_DOMAIN: &[u8] = b"KPAY-RCPT1";

/// Domain separator for subscription signatures: a signature
/// covers `b"KPAY-SUB1" || id || payer key || payee key ||
/// amount || interval || total count || first due`.
const SUBSCRIPTION_SIGN_DOMAIN: &[u8] = b"KPAY-SUB1";

/// Domain separator for refund signatures: a signature covers
/// `b"KPAY-RFD1" || id || payment id || refunder key ||
/// beneficiary key || amount`.
const REFUND_SIGN_DOMAIN: &[u8] = b"KPAY-RFD1";

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
    /// A subscription id, amount, interval, count, or first due
    /// time broke the rules.
    InvalidSubscription(String),
    /// A subscription with this id already exists.
    DuplicateSubscription(String),
    /// A subscription's signature did not verify against its payer.
    BadSubscriptionSignature,
    /// The subscription is not active (finished or cancelled), so
    /// the requested operation does not apply.
    SubscriptionNotActive(String),
    /// A refund id, payment id, or amount broke the rules.
    InvalidRefund(String),
    /// A refund with this id already exists.
    DuplicateRefund(String),
    /// A refund's signature did not verify against its refunder.
    BadRefundSignature,
    /// A refund's amount differs from the original payment.
    RefundAmountMismatch {
        /// The amount of the original payment.
        expected: u64,
        /// The amount the refund tried to return.
        got: u64,
    },
    /// No paid payment exists with this id, so there is nothing
    /// to refund.
    NotPaid(String),
    /// The payment was already refunded once.
    AlreadyRefunded(String),
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
            PayError::Expired { expiry, now } => {
                write!(f, "invoice expired at {expiry}, payment attempted at {now}")
            }
            PayError::AlreadyPaid(id) => write!(f, "invoice '{id}' was already paid"),
            PayError::InvalidSubscription(msg) => {
                write!(f, "invalid subscription: {msg}")
            }
            PayError::DuplicateSubscription(id) => {
                write!(f, "a subscription with id '{id}' already exists")
            }
            PayError::BadSubscriptionSignature => {
                write!(f, "subscription signature is invalid")
            }
            PayError::SubscriptionNotActive(id) => {
                write!(f, "subscription '{id}' is not active")
            }
            PayError::InvalidRefund(msg) => write!(f, "invalid refund: {msg}"),
            PayError::DuplicateRefund(id) => {
                write!(f, "a refund with id '{id}' already exists")
            }
            PayError::BadRefundSignature => {
                write!(f, "refund signature is invalid")
            }
            PayError::RefundAmountMismatch { expected, got } => write!(
                f,
                "refund amount {got} does not match the original payment {expected}"
            ),
            PayError::NotPaid(id) => {
                write!(f, "no paid payment with id '{id}' exists")
            }
            PayError::AlreadyRefunded(id) => {
                write!(f, "payment '{id}' was already refunded")
            }
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
    owner_free: bool,
) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend_from_slice(RECEIPT_SIGN_DOMAIN);
    msg.extend_from_slice(&(invoice_id.len() as u16).to_le_bytes());
    msg.extend_from_slice(invoice_id.as_bytes());
    msg.extend_from_slice(payer.public_key().as_bytes());
    msg.extend_from_slice(recipient.public_key().as_bytes());
    msg.extend_from_slice(&amount.to_le_bytes());
    msg.push(owner_free as u8);
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
    /// The amount that moved. For an owner-free receipt this is
    /// 0 — nothing moved — and `owner_free` is set.
    pub amount: u64,
    /// True when this receipt is owner-free: the registered owner
    /// acquired the thing without any coins moving. Covered by
    /// the payer's signature, so the flag cannot be flipped
    /// after the fact.
    pub owner_free: bool,
    /// The payer's signature over invoice id, payer, recipient,
    /// amount, and the owner-free flag.
    pub signature: Ed25519Signature,
}

impl Receipt {
    /// Build an owner-free receipt: proof that the registered
    /// owner acquired something without any coins moving. The
    /// amount is 0 and `owner_free` is set; the payer signs the
    /// flag along with the rest. This is only meaningful when
    /// the payer really is the registered owner — a
    /// [`PaymentService`] checks that before issuing one. A
    /// receipt built by hand proves nothing on its own; what
    /// makes an owner-free purchase real is the service's
    /// payment record behind it.
    pub fn owner_free(invoice_id: &str, payer: &Identity, recipient: &PeerId) -> Self {
        let payer_id = payer.peer_id().clone();
        let msg = receipt_sign_message(invoice_id, &payer_id, recipient, 0, true);
        let signature = payer.sign(&msg);
        Self {
            invoice_id: invoice_id.to_string(),
            payer: payer_id,
            recipient: recipient.clone(),
            amount: 0,
            owner_free: true,
            signature,
        }
    }

    /// True if the signature verifies against the payer for this
    /// exact invoice id, recipient, amount, and owner-free flag.
    /// Any tampering fails this.
    pub fn verify(&self) -> bool {
        let msg = receipt_sign_message(
            &self.invoice_id,
            &self.payer,
            &self.recipient,
            self.amount,
            self.owner_free,
        );
        self.payer.verify(&msg, &self.signature).is_ok()
    }
}

/// The message a payer signs for a subscription.
fn subscription_sign_message(
    id: &str,
    payer: &PeerId,
    payee: &PeerId,
    amount: u64,
    interval_secs: u64,
    total_count: u32,
    first_due: u64,
) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend_from_slice(SUBSCRIPTION_SIGN_DOMAIN);
    msg.extend_from_slice(&(id.len() as u16).to_le_bytes());
    msg.extend_from_slice(id.as_bytes());
    msg.extend_from_slice(payer.public_key().as_bytes());
    msg.extend_from_slice(payee.public_key().as_bytes());
    msg.extend_from_slice(&amount.to_le_bytes());
    msg.extend_from_slice(&interval_secs.to_le_bytes());
    msg.extend_from_slice(&total_count.to_le_bytes());
    msg.extend_from_slice(&first_due.to_le_bytes());
    msg
}

/// The message a refunder signs for a refund.
fn refund_sign_message(
    id: &str,
    payment_id: &str,
    refunder: &PeerId,
    beneficiary: &PeerId,
    amount: u64,
) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend_from_slice(REFUND_SIGN_DOMAIN);
    msg.extend_from_slice(&(id.len() as u16).to_le_bytes());
    msg.extend_from_slice(id.as_bytes());
    msg.extend_from_slice(&(payment_id.len() as u16).to_le_bytes());
    msg.extend_from_slice(payment_id.as_bytes());
    msg.extend_from_slice(refunder.public_key().as_bytes());
    msg.extend_from_slice(beneficiary.public_key().as_bytes());
    msg.extend_from_slice(&amount.to_le_bytes());
    msg
}

/// The lifecycle state of a subscription.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubscriptionStatus {
    /// Due occurrences are still being processed by `tick`.
    Active,
    /// All `total_count` occurrences have been processed.
    Finished,
    /// Cancelled by the payer; no further payments.
    Cancelled,
}

/// A recurring payment agreement: the payer authorizes `total_count`
/// payments of `amount` to `payee`, the first due at `first_due`,
/// then one every `interval_secs`. Signed by the payer, so anyone
/// can verify who authorized it and for what.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Subscription {
    /// Human-readable id, e.g. `"sub-rent-001"`.
    pub id: String,
    /// Who pays.
    pub payer: PeerId,
    /// Who is paid.
    pub payee: PeerId,
    /// The amount of each occurrence, in the coin's smallest unit.
    pub amount: u64,
    /// Seconds between occurrences.
    pub interval_secs: u64,
    /// How many occurrences the subscription runs for.
    pub total_count: u32,
    /// Caller-supplied time of the first due occurrence.
    pub first_due: u64,
    /// The payer's signature over id, payer, payee, amount,
    /// interval, total count, and first due.
    pub signature: Ed25519Signature,
}

impl Subscription {
    /// Create and sign a subscription as `payer`. The amount and
    /// interval must be greater than zero and the count must be
    /// at least one.
    pub fn new(
        payer: &Identity,
        id: &str,
        payee: &PeerId,
        amount: u64,
        interval_secs: u64,
        total_count: u32,
        first_due: u64,
    ) -> Result<Self, PayError> {
        validate_id(id, "subscription")?;
        if amount == 0 {
            return Err(PayError::InvalidSubscription(
                "subscription amount must be greater than zero".to_string(),
            ));
        }
        if interval_secs == 0 {
            return Err(PayError::InvalidSubscription(
                "subscription interval must be greater than zero".to_string(),
            ));
        }
        if total_count == 0 {
            return Err(PayError::InvalidSubscription(
                "subscription total count must be at least one".to_string(),
            ));
        }
        let payer_id = payer.peer_id().clone();
        let msg = subscription_sign_message(
            id,
            &payer_id,
            payee,
            amount,
            interval_secs,
            total_count,
            first_due,
        );
        let signature = payer.sign(&msg);
        Ok(Self {
            id: id.to_string(),
            payer: payer_id,
            payee: payee.clone(),
            amount,
            interval_secs,
            total_count,
            first_due,
            signature,
        })
    }

    /// True if the signature verifies against the payer for this
    /// exact id, payee, amount, interval, count, and first due.
    /// Any tampering fails this.
    pub fn verify(&self) -> bool {
        let msg = subscription_sign_message(
            &self.id,
            &self.payer,
            &self.payee,
            self.amount,
            self.interval_secs,
            self.total_count,
            self.first_due,
        );
        self.payer.verify(&msg, &self.signature).is_ok()
    }

    /// The payment id of occurrence `n` (0-based): `{id}:{n}`.
    /// This is the id the occurrence's receipt and history entry
    /// carry, so a subscription payment is recognizable and a
    /// refund can name it.
    pub fn occurrence_payment_id(&self, n: u32) -> String {
        format!("{}:{n}", self.id)
    }
}

/// The outcome of one due subscription occurrence processed by
/// `tick`. A failed occurrence is consumed — it is not retried —
/// and the subscription moves on to its next due time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OccurrenceOutcome {
    /// The occurrence settled; carries the payer's receipt.
    Paid {
        /// The subscription id.
        subscription_id: String,
        /// The 0-based occurrence index.
        occurrence: u32,
        /// The pre-signed receipt for this occurrence.
        receipt: Receipt,
    },
    /// The occurrence was due but settlement failed (for example
    /// insufficient funds). No coins moved for it.
    Failed {
        /// The subscription id.
        subscription_id: String,
        /// The 0-based occurrence index.
        occurrence: u32,
        /// Why it failed.
        error: PayError,
    },
}

/// A refund: a signed instruction to return a payment in full.
/// Signed by the original recipient (the refunder), who is
/// returning the coins to the original payer (the beneficiary).
/// The amount must equal the original payment exactly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refund {
    /// Human-readable id, e.g. `"rfd-coffee-001"`.
    pub id: String,
    /// The payment id being refunded (an invoice id, or a
    /// subscription occurrence id like `"sub-rent-001:2"`).
    pub payment_id: String,
    /// Who is returning the coins (the original recipient).
    pub refunder: PeerId,
    /// Who receives the coins back (the original payer).
    pub beneficiary: PeerId,
    /// The amount returned; must equal the original payment.
    pub amount: u64,
    /// The refunder's signature over id, payment id, refunder,
    /// beneficiary, and amount.
    pub signature: Ed25519Signature,
}

impl Refund {
    /// Create and sign a refund as `refunder`. The amount must be
    /// greater than zero; whether it matches the original payment
    /// is checked when the refund is processed, not here, because
    /// this record can be built before the service sees it.
    pub fn new(
        refunder: &Identity,
        id: &str,
        payment_id: &str,
        beneficiary: &PeerId,
        amount: u64,
    ) -> Result<Self, PayError> {
        validate_id(id, "refund")?;
        // The payment id is not validated with the strict id
        // rules: subscription occurrence ids contain ':' (e.g.
        // "sub-rent-001:2"). It only needs to be a non-empty,
        // reasonably short printable string; whether a payment
        // with this id exists is checked at processing time.
        if payment_id.is_empty() || payment_id.len() > 128 {
            return Err(PayError::InvalidRefund(
                "refund payment id must be 1-128 characters".to_string(),
            ));
        }
        if !payment_id.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
            return Err(PayError::InvalidRefund(
                "refund payment id must be printable ASCII".to_string(),
            ));
        }
        if amount == 0 {
            return Err(PayError::InvalidRefund(
                "refund amount must be greater than zero".to_string(),
            ));
        }
        let refunder_id = refunder.peer_id().clone();
        let msg = refund_sign_message(id, payment_id, &refunder_id, beneficiary, amount);
        let signature = refunder.sign(&msg);
        Ok(Self {
            id: id.to_string(),
            payment_id: payment_id.to_string(),
            refunder: refunder_id,
            beneficiary: beneficiary.clone(),
            amount,
            signature,
        })
    }

    /// True if the signature verifies against the refunder for
    /// this exact id, payment id, beneficiary, and amount.
    pub fn verify(&self) -> bool {
        let msg = refund_sign_message(
            &self.id,
            &self.payment_id,
            &self.refunder,
            &self.beneficiary,
            self.amount,
        );
        self.refunder.verify(&msg, &self.signature).is_ok()
    }
}

/// Where a recorded payment came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaymentKind {
    /// A one-off invoice payment.
    Invoice,
    /// One occurrence of a subscription.
    Subscription,
}

/// The status of a recorded payment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaymentStatus {
    /// Settled and still standing.
    Paid,
    /// Settled, then refunded in full.
    Refunded,
}

/// One settled payment, as recorded for history. The receipt is
/// the signed proof; `status` tracks whether it was later
/// refunded. For subscription occurrences the `invoice_id` is
/// the occurrence payment id (there is no invoice).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaymentRecord {
    /// The payment id: an invoice id, or a subscription
    /// occurrence id like `"sub-rent-001:2"`.
    pub payment_id: String,
    /// The invoice id, or the occurrence id for subscriptions.
    pub invoice_id: String,
    /// Who paid.
    pub payer: PeerId,
    /// Who was paid.
    pub payee: PeerId,
    /// The amount that moved.
    pub amount: u64,
    /// Caller-supplied time the payment settled.
    pub paid_at: u64,
    /// Caller-supplied time the payment was refunded, if any.
    pub refunded_at: Option<u64>,
    /// `Paid`, or `Refunded` after a successful refund.
    pub status: PaymentStatus,
    /// Whether this was an invoice or a subscription occurrence.
    pub kind: PaymentKind,
    /// The payer's signed receipt for this payment.
    pub receipt: Receipt,
}

/// The stored state of one subscription. The payer pre-signs one
/// coin transaction and one receipt per occurrence at creation,
/// so `tick` only ever submits what the payer already authorized;
/// the service never holds a private key.
struct SubscriptionState {
    subscription: Subscription,
    /// The caller-supplied time the next occurrence is due.
    next_due: u64,
    /// How many occurrences have been processed (paid or failed).
    occurrences_done: u32,
    /// How many occurrences actually settled.
    payments_made: u32,
    /// Lifecycle state.
    status: SubscriptionStatus,
    /// Pre-signed (transaction, receipt) per occurrence, in order.
    prepaid: Vec<(CoinTransaction, Receipt)>,
}

/// A payment service: a coin ledger plus the invoices issued
/// against it. Invoices are created by recipients and can arrive
/// from any peer; only their signatures authenticate them, so an
/// invoice does not need to have been created through this
/// service to be payable here.
pub struct PaymentService {
    ledger: Ledger,
    /// The registered owner, if any: the peer for whom
    /// everything inside KERNEL is free (the OWNER-FREE LAW).
    /// `None` means no owner is registered and everyone pays.
    /// This crate does not decide who the owner is — the node
    /// registers them.
    owner: Option<PeerId>,
    /// Invoices registered through [`PaymentService::create_invoice`].
    invoices: HashMap<String, Invoice>,
    /// Ids of invoices already paid.
    paid: HashSet<String>,
    /// Every settled payment, in settlement order (oldest first).
    /// History queries read this newest-first.
    payments: Vec<PaymentRecord>,
    /// Subscriptions by id. A `BTreeMap` so `tick` processes them
    /// in a deterministic order.
    subscriptions: BTreeMap<String, SubscriptionState>,
    /// Ids of refunds already processed.
    refund_ids: HashSet<String>,
}

impl PaymentService {
    /// Create a payment service whose ledger starts at the given
    /// genesis allocation. No owner is registered: everyone pays
    /// until [`PaymentService::set_owner`] is called.
    pub fn new(genesis: &[(PeerId, u64)]) -> Result<Self, PayError> {
        Ok(Self {
            ledger: Ledger::new(genesis)?,
            owner: None,
            invoices: HashMap::new(),
            paid: HashSet::new(),
            payments: Vec::new(),
            subscriptions: BTreeMap::new(),
            refund_ids: HashSet::new(),
        })
    }

    /// Register the owner of this node: the peer for whom
    /// everything inside KERNEL is free (the OWNER-FREE LAW).
    /// From then on, paying an invoice as the owner satisfies it
    /// without moving any coins, and subscription occurrences
    /// for the owner execute free. Register the owner before
    /// registering a subscription for its occurrences to be
    /// owner-free: the prepaid receipts are signed at
    /// registration time and cannot be rewritten later.
    pub fn set_owner(&mut self, owner: PeerId) {
        self.owner = Some(owner);
    }

    /// The registered owner, if any.
    pub fn owner(&self) -> Option<&PeerId> {
        self.owner.as_ref()
    }

    /// True if `peer` is the registered owner. Compares public
    /// keys, not display names, so a peer id rebuilt from a
    /// decoded record (which carries a fingerprint name instead
    /// of the original display name) still matches.
    pub fn is_owner(&self, peer: &PeerId) -> bool {
        self.owner
            .as_ref()
            .is_some_and(|o| o.public_key() == peer.public_key())
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
    /// invoice stays unpaid. When the payer is the registered
    /// owner (see the OWNER-FREE LAW), the invoice is satisfied
    /// without touching the ledger: no coin transaction is
    /// submitted, an owner-free receipt is issued, and the
    /// payment is recorded with amount 0.
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
        if self.is_owner(payer.peer_id()) {
            // OWNER-FREE LAW: the owner pays nothing. The
            // invoice is satisfied without touching the ledger:
            // no coin transaction is submitted and no coins move.
            // A valid owner-free receipt is still issued, the
            // invoice is marked paid, and the payment is
            // recorded with amount 0.
            self.paid.insert(invoice.id.clone());
            let receipt = Receipt::owner_free(&invoice.id, payer, &invoice.recipient);
            self.payments.push(PaymentRecord {
                payment_id: invoice.id.clone(),
                invoice_id: invoice.id.clone(),
                payer: receipt.payer.clone(),
                payee: receipt.recipient.clone(),
                amount: 0,
                paid_at: now,
                refunded_at: None,
                status: PaymentStatus::Paid,
                kind: PaymentKind::Invoice,
                receipt: receipt.clone(),
            });
            return Ok(receipt);
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
            false,
        );
        let signature = payer.sign(&msg);
        let receipt = Receipt {
            invoice_id: invoice.id.clone(),
            payer: payer_id,
            recipient: invoice.recipient.clone(),
            amount: invoice.amount,
            owner_free: false,
            signature,
        };
        self.payments.push(PaymentRecord {
            payment_id: invoice.id.clone(),
            invoice_id: invoice.id.clone(),
            payer: receipt.payer.clone(),
            payee: receipt.recipient.clone(),
            amount: receipt.amount,
            paid_at: now,
            refunded_at: None,
            status: PaymentStatus::Paid,
            kind: PaymentKind::Invoice,
            receipt: receipt.clone(),
        });
        Ok(receipt)
    }

    /// Verify a receipt's payer signature. This is a plain
    /// function, not a method, because a receipt is verified
    /// without any service state.
    pub fn verify_receipt(receipt: &Receipt) -> bool {
        receipt.verify()
    }

    // -- Recurring payments -------------------------------------

    /// Register a signed subscription as `payer` for execution.
    /// The subscription is a signed record that can arrive from
    /// any peer (like an invoice); registration verifies its
    /// signature and that the presented identity is the payer,
    /// then the payer pre-signs one coin transaction and one
    /// receipt per occurrence here, at authorization time, so
    /// `tick` later only submits what was already authorized.
    /// The id must not already be registered.
    pub fn register_subscription(
        &mut self,
        payer: &Identity,
        subscription: &Subscription,
    ) -> Result<(), PayError> {
        if self.subscriptions.contains_key(&subscription.id) {
            return Err(PayError::DuplicateSubscription(subscription.id.clone()));
        }
        if !subscription.verify() {
            return Err(PayError::BadSubscriptionSignature);
        }
        if payer.peer_id() != &subscription.payer {
            return Err(PayError::InvalidSubscription(
                "the presented identity is not the subscription's payer".to_string(),
            ));
        }
        let payer_id = payer.peer_id().clone();
        let payee = subscription.payee.clone();
        let mut prepaid = Vec::with_capacity(subscription.total_count as usize);
        for n in 0..subscription.total_count {
            let payment_id = subscription.occurrence_payment_id(n);
            let memo = format!("sub:{payment_id}");
            let tx = CoinTransaction::new(payer, &payee, &memo, subscription.amount)?;
            // An owner subscription's occurrences are owner-free:
            // the prepaid receipt carries amount 0 and the flag,
            // so `tick` settles them without moving coins. The
            // owner must be registered before the subscription is
            // registered — the receipt is signed here, at
            // authorization time, and cannot be rewritten later.
            // The transaction is still built (it is never
            // submitted for an owner-free occurrence).
            let receipt = if self.is_owner(&payer_id) {
                Receipt::owner_free(&payment_id, payer, &payee)
            } else {
                let msg = receipt_sign_message(
                    &payment_id,
                    &payer_id,
                    &payee,
                    subscription.amount,
                    false,
                );
                let signature = payer.sign(&msg);
                Receipt {
                    invoice_id: payment_id,
                    payer: payer_id.clone(),
                    recipient: payee.clone(),
                    amount: subscription.amount,
                    owner_free: false,
                    signature,
                }
            };
            prepaid.push((tx, receipt));
        }
        self.subscriptions.insert(
            subscription.id.clone(),
            SubscriptionState {
                next_due: subscription.first_due,
                occurrences_done: 0,
                payments_made: 0,
                status: SubscriptionStatus::Active,
                prepaid,
                subscription: subscription.clone(),
            },
        );
        Ok(())
    }

    /// The registered subscription with this id, if any.
    pub fn get_subscription(&self, id: &str) -> Option<&Subscription> {
        self.subscriptions.get(id).map(|s| &s.subscription)
    }

    /// The lifecycle state of the subscription with this id, if any.
    pub fn subscription_status(&self, id: &str) -> Option<SubscriptionStatus> {
        self.subscriptions.get(id).map(|s| s.status)
    }

    /// `(payments_made, occurrences_done)` for the subscription
    /// with this id, if any. Occurrences done includes failed
    /// occurrences; payments made counts only settled ones.
    pub fn subscription_counts(&self, id: &str) -> Option<(u32, u32)> {
        self.subscriptions
            .get(id)
            .map(|s| (s.payments_made, s.occurrences_done))
    }

    /// Process every subscription occurrence due at `now`, in
    /// subscription-id order and occurrence order. Each due
    /// occurrence is submitted through the normal ledger path
    /// using the transaction the payer pre-signed at creation.
    /// A failed occurrence (for example insufficient funds) is
    /// reported as [`OccurrenceOutcome::Failed`] and consumed —
    /// it is not retried, and the subscription moves on to its
    /// next due time. A subscription whose total count of
    /// occurrences has been processed becomes `Finished`.
    /// Nothing runs in the background: recurring payments only
    /// happen when the caller drives `tick`. Times must be
    /// non-decreasing across calls; this is the caller's job.
    pub fn tick(&mut self, now: u64) -> Vec<OccurrenceOutcome> {
        let mut outcomes = Vec::new();
        // Collect the ids first: processing mutates the states.
        let ids: Vec<String> = self.subscriptions.keys().cloned().collect();
        for id in ids {
            // The borrow must end before the next iteration, so
            // work on one subscription at a time and drop it.
            loop {
                let (due, tx, receipt) = {
                    let Some(state) = self.subscriptions.get(&id) else {
                        break;
                    };
                    if state.status != SubscriptionStatus::Active {
                        break;
                    }
                    if state.occurrences_done >= state.subscription.total_count
                        || state.next_due > now
                    {
                        break;
                    }
                    let n = state.occurrences_done;
                    let (tx, receipt) = state.prepaid[n as usize].clone();
                    (n, tx, receipt)
                };
                let n = due;
                // An owner-free occurrence never touches the
                // ledger: its prepaid receipt was authorized as
                // free at registration, so the occurrence is
                // settled by recording that receipt. The flag on
                // the receipt is authoritative here — it was
                // built under the rules in force at authorization
                // time, and `tick` holds no key to re-sign one.
                let submit = if receipt.owner_free {
                    Ok(())
                } else {
                    self.ledger.submit(&tx)
                };
                match submit {
                    Ok(()) => {
                        let state = self
                            .subscriptions
                            .get_mut(&id)
                            .expect("subscription still present");
                        state.payments_made += 1;
                        state.occurrences_done += 1;
                        state.next_due = state
                            .next_due
                            .saturating_add(state.subscription.interval_secs);
                        if state.occurrences_done >= state.subscription.total_count {
                            state.status = SubscriptionStatus::Finished;
                        }
                        self.payments.push(PaymentRecord {
                            payment_id: receipt.invoice_id.clone(),
                            invoice_id: receipt.invoice_id.clone(),
                            payer: receipt.payer.clone(),
                            payee: receipt.recipient.clone(),
                            amount: receipt.amount,
                            paid_at: now,
                            refunded_at: None,
                            status: PaymentStatus::Paid,
                            kind: PaymentKind::Subscription,
                            receipt: receipt.clone(),
                        });
                        outcomes.push(OccurrenceOutcome::Paid {
                            subscription_id: id.clone(),
                            occurrence: n,
                            receipt,
                        });
                    }
                    Err(e) => {
                        let state = self
                            .subscriptions
                            .get_mut(&id)
                            .expect("subscription still present");
                        state.occurrences_done += 1;
                        state.next_due = state
                            .next_due
                            .saturating_add(state.subscription.interval_secs);
                        if state.occurrences_done >= state.subscription.total_count {
                            state.status = SubscriptionStatus::Finished;
                        }
                        outcomes.push(OccurrenceOutcome::Failed {
                            subscription_id: id.clone(),
                            occurrence: n,
                            error: PayError::Koin(e),
                        });
                    }
                }
            }
        }
        outcomes
    }

    /// Cancel an active subscription. No further occurrences are
    /// processed after this. Cancelling a finished or already
    /// cancelled subscription is a clean error, as is cancelling
    /// one that does not exist.
    pub fn cancel_subscription(&mut self, id: &str) -> Result<(), PayError> {
        let Some(state) = self.subscriptions.get_mut(id) else {
            return Err(PayError::InvalidSubscription(format!(
                "no subscription with id '{id}'"
            )));
        };
        if state.status != SubscriptionStatus::Active {
            return Err(PayError::SubscriptionNotActive(id.to_string()));
        }
        state.status = SubscriptionStatus::Cancelled;
        Ok(())
    }

    // -- Refunds ---------------------------------------------------

    /// Process a signed refund. The `refunder` identity (the
    /// original recipient) must be presented so the service can
    /// submit the returning coin transaction; the refund record's
    /// own signature is verified first, and the record is the
    /// independently verifiable authorization. Checks, in order:
    /// the refund signature, that the refund id is unused, that
    /// a paid payment with this payment id exists, that it has
    /// not already been refunded, that the roles match the
    /// original payment, and that the amount equals the original
    /// exactly. On success the coins move back through the ledger
    /// (memo `refund:{payment_id}`) and the payment's history
    /// entry becomes `Refunded`. A rejected refund changes
    /// nothing. Returns the updated payment record.
    pub fn process_refund(
        &mut self,
        refund: &Refund,
        refunder: &Identity,
        now: u64,
    ) -> Result<PaymentRecord, PayError> {
        if !refund.verify() {
            return Err(PayError::BadRefundSignature);
        }
        if self.refund_ids.contains(&refund.id) {
            return Err(PayError::DuplicateRefund(refund.id.clone()));
        }
        if refunder.peer_id() != &refund.refunder {
            return Err(PayError::InvalidRefund(
                "the presented identity is not the refund's refunder".to_string(),
            ));
        }
        let Some(record) = self
            .payments
            .iter()
            .find(|r| r.payment_id == refund.payment_id)
        else {
            return Err(PayError::NotPaid(refund.payment_id.clone()));
        };
        if record.status == PaymentStatus::Refunded {
            return Err(PayError::AlreadyRefunded(refund.payment_id.clone()));
        }
        if refund.refunder != record.payee || refund.beneficiary != record.payer {
            return Err(PayError::InvalidRefund(
                "refund roles do not match the original payment".to_string(),
            ));
        }
        if refund.amount != record.amount {
            return Err(PayError::RefundAmountMismatch {
                expected: record.amount,
                got: refund.amount,
            });
        }
        let memo = format!("refund:{}", refund.payment_id);
        let tx = CoinTransaction::new(refunder, &refund.beneficiary, &memo, refund.amount)?;
        self.ledger.submit(&tx)?;
        self.refund_ids.insert(refund.id.clone());
        let record = self
            .payments
            .iter_mut()
            .find(|r| r.payment_id == refund.payment_id)
            .expect("payment still present");
        record.status = PaymentStatus::Refunded;
        record.refunded_at = Some(now);
        Ok(record.clone())
    }

    /// Void a zero-amount (owner-free) payment: mark its history
    /// record `Refunded` without moving any coins, because none
    /// moved in the first place. This is how an owner-free
    /// escrow dispute is settled — there is nothing to return,
    /// so there is nothing to refund. Refuses an unknown payment
    /// id, an already-refunded payment, and any payment whose
    /// amount is not zero (those must go through
    /// [`PaymentService::process_refund`]).
    pub fn void_payment(&mut self, payment_id: &str, now: u64) -> Result<PaymentRecord, PayError> {
        let Some(record) = self.payments.iter().find(|r| r.payment_id == payment_id) else {
            return Err(PayError::NotPaid(payment_id.to_string()));
        };
        if record.status == PaymentStatus::Refunded {
            return Err(PayError::AlreadyRefunded(payment_id.to_string()));
        }
        if record.amount != 0 {
            return Err(PayError::InvalidRefund(format!(
                "payment '{payment_id}' moved {} coins; voiding is only for zero-amount owner-free payments",
                record.amount
            )));
        }
        let record = self
            .payments
            .iter_mut()
            .find(|r| r.payment_id == payment_id)
            .expect("payment still present");
        record.status = PaymentStatus::Refunded;
        record.refunded_at = Some(now);
        Ok(record.clone())
    }

    // -- Payment history -------------------------------------------

    /// All settled payments, newest first, up to `limit`.
    pub fn history(&self, limit: usize) -> Vec<PaymentRecord> {
        self.payments.iter().rev().take(limit).cloned().collect()
    }

    /// Settled payments made by `payer`, newest first, up to
    /// `limit`.
    pub fn history_for_payer(&self, payer: &PeerId, limit: usize) -> Vec<PaymentRecord> {
        self.payments
            .iter()
            .rev()
            .filter(|r| &r.payer == payer)
            .take(limit)
            .cloned()
            .collect()
    }

    /// Settled payments received by `payee`, newest first, up to
    /// `limit`.
    pub fn history_for_payee(&self, payee: &PeerId, limit: usize) -> Vec<PaymentRecord> {
        self.payments
            .iter()
            .rev()
            .filter(|r| &r.payee == payee)
            .take(limit)
            .cloned()
            .collect()
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

    /// Build a subscription, panicking on any error — for tests
    /// only. Registration is a separate explicit step so tests
    /// can exercise duplicate and bad-signature paths.
    fn make_sub(
        payer: &Identity,
        id: &str,
        payee: &PeerId,
        amount: u64,
        interval: u64,
        count: u32,
        first_due: u64,
    ) -> Subscription {
        Subscription::new(payer, id, payee, amount, interval, count, first_due).unwrap()
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
        assert!(matches!(
            err,
            PayError::Koin(KoinError::InsufficientFunds { .. })
        ));
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
        assert!(service.create_invoice(&bob, "", 10, "x", 1_000).is_err());
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
        let unregistered =
            Invoice::new(&alice, "inv-stranger-001", 25, "from afar", 1_000).unwrap();
        let receipt = service.pay_invoice(&unregistered, &bob, 10).unwrap();
        assert!(receipt.verify());
        assert_eq!(service.balance_of(bob.peer_id()), 75);
        assert_eq!(service.balance_of(alice.peer_id()), 1_025);
    }

    #[test]
    fn subscription_pays_exactly_its_count_then_stops() {
        let (mut service, alice, bob) = funded();
        let sub = make_sub(&alice, "sub-rent-001", bob.peer_id(), 100, 10, 3, 100);
        service.register_subscription(&alice, &sub).unwrap();
        assert!(sub.verify());
        assert_eq!(
            service.subscription_status("sub-rent-001"),
            Some(SubscriptionStatus::Active)
        );
        // Nothing due before the first due time.
        assert!(service.tick(99).is_empty());
        // One occurrence per due time.
        let first = service.tick(100);
        assert_eq!(first.len(), 1);
        assert!(matches!(
            first[0],
            OccurrenceOutcome::Paid { occurrence: 0, .. }
        ));
        if let OccurrenceOutcome::Paid { receipt, .. } = &first[0] {
            assert!(receipt.verify());
            assert_eq!(receipt.invoice_id, "sub-rent-001:0");
        }
        assert!(service.tick(109).is_empty());
        assert_eq!(service.tick(110).len(), 1);
        assert_eq!(service.tick(115).len(), 0);
        let third = service.tick(120);
        assert_eq!(third.len(), 1);
        assert!(matches!(
            third[0],
            OccurrenceOutcome::Paid { occurrence: 2, .. }
        ));
        // Exactly three payments, then finished: nothing more.
        assert_eq!(
            service.subscription_status("sub-rent-001"),
            Some(SubscriptionStatus::Finished)
        );
        assert_eq!(service.subscription_counts("sub-rent-001"), Some((3, 3)));
        assert!(service.tick(130).is_empty());
        assert!(service.tick(10_000).is_empty());
        assert_eq!(service.balance_of(alice.peer_id()), 700);
        assert_eq!(service.balance_of(bob.peer_id()), 400);
        // History holds the three subscription payments.
        let history = service.history_for_payer(alice.peer_id(), 10);
        assert_eq!(history.len(), 3);
        assert!(history.iter().all(|r| r.kind == PaymentKind::Subscription));
        assert_eq!(history[0].payment_id, "sub-rent-001:2");
        assert_eq!(history[2].payment_id, "sub-rent-001:0");
    }

    #[test]
    fn subscription_failed_occurrence_does_not_stop_it() {
        let (mut service, alice, bob) = funded();
        // Alice holds 1,000; three occurrences of 400: the third
        // fails for insufficient funds.
        let sub = make_sub(&alice, "sub-big-001", bob.peer_id(), 400, 10, 3, 100);
        service.register_subscription(&alice, &sub).unwrap();
        let first = service.tick(100);
        assert_eq!(first.len(), 1);
        assert!(matches!(first[0], OccurrenceOutcome::Paid { .. }));
        let second = service.tick(110);
        assert_eq!(second.len(), 1);
        assert!(matches!(second[0], OccurrenceOutcome::Paid { .. }));
        let third = service.tick(120);
        assert_eq!(third.len(), 1);
        match &third[0] {
            OccurrenceOutcome::Failed {
                occurrence, error, ..
            } => {
                assert_eq!(*occurrence, 2);
                assert!(matches!(
                    error,
                    PayError::Koin(KoinError::InsufficientFunds { .. })
                ));
            }
            other => panic!("expected a failed occurrence, got {other:?}"),
        }
        // The failed occurrence did not stop the subscription:
        // it finished after its total count, and the failed
        // occurrence left no payment in history.
        assert_eq!(
            service.subscription_status("sub-big-001"),
            Some(SubscriptionStatus::Finished)
        );
        assert_eq!(service.subscription_counts("sub-big-001"), Some((2, 3)));
        assert_eq!(service.balance_of(alice.peer_id()), 200);
        assert_eq!(service.balance_of(bob.peer_id()), 900);
        assert_eq!(service.history_for_payer(alice.peer_id(), 10).len(), 2);
        // A late tick processes every occurrence due at once.
        let (mut service, alice, bob) = funded();
        let sub = make_sub(&alice, "sub-late-001", bob.peer_id(), 100, 10, 3, 100);
        service.register_subscription(&alice, &sub).unwrap();
        let all = service.tick(125);
        assert_eq!(all.len(), 3);
        assert!(all
            .iter()
            .all(|o| matches!(o, OccurrenceOutcome::Paid { .. })));
        assert_eq!(service.balance_of(alice.peer_id()), 700);
    }

    #[test]
    fn subscription_validation_and_bad_signature() {
        let (mut service, alice, bob) = funded();
        assert!(matches!(
            Subscription::new(&alice, "sub-z-001", bob.peer_id(), 0, 10, 3, 100),
            Err(PayError::InvalidSubscription(_))
        ));
        assert!(matches!(
            Subscription::new(&alice, "sub-z-002", bob.peer_id(), 10, 0, 3, 100),
            Err(PayError::InvalidSubscription(_))
        ));
        assert!(matches!(
            Subscription::new(&alice, "sub-z-003", bob.peer_id(), 10, 10, 0, 100),
            Err(PayError::InvalidSubscription(_))
        ));
        let sub = make_sub(&alice, "sub-dup-001", bob.peer_id(), 10, 10, 2, 100);
        service.register_subscription(&alice, &sub).unwrap();
        assert_eq!(
            service.register_subscription(&alice, &sub),
            Err(PayError::DuplicateSubscription("sub-dup-001".to_string()))
        );
        // Registering a forged subscription is refused: build
        // one under a fresh id so the signature check is what
        // fires, not the duplicate check.
        let mut forged = make_sub(&alice, "sub-forged-001", bob.peer_id(), 10, 10, 2, 100);
        forged.amount = 999;
        assert!(!forged.verify());
        assert_eq!(
            service.register_subscription(&alice, &forged),
            Err(PayError::BadSubscriptionSignature)
        );
    }

    #[test]
    fn cancel_stops_future_payments_and_re_cancel_errors() {
        let (mut service, alice, bob) = funded();
        let sub = make_sub(&alice, "sub-can-001", bob.peer_id(), 50, 10, 5, 100);
        service.register_subscription(&alice, &sub).unwrap();
        assert_eq!(service.tick(100).len(), 1);
        assert_eq!(service.balance_of(alice.peer_id()), 950);
        service.cancel_subscription("sub-can-001").unwrap();
        assert_eq!(
            service.subscription_status("sub-can-001"),
            Some(SubscriptionStatus::Cancelled)
        );
        // No further occurrences are processed.
        assert!(service.tick(200).is_empty());
        assert_eq!(service.balance_of(alice.peer_id()), 950);
        assert_eq!(service.balance_of(bob.peer_id()), 150);
        // Cancelling again is a clean error.
        assert_eq!(
            service.cancel_subscription("sub-can-001"),
            Err(PayError::SubscriptionNotActive("sub-can-001".to_string()))
        );
        // Cancelling a finished subscription is also an error.
        let sub = make_sub(&alice, "sub-can-002", bob.peer_id(), 10, 10, 1, 100);
        service.register_subscription(&alice, &sub).unwrap();
        assert_eq!(service.tick(100).len(), 1);
        assert_eq!(
            service.subscription_status("sub-can-002"),
            Some(SubscriptionStatus::Finished)
        );
        assert_eq!(
            service.cancel_subscription("sub-can-002"),
            Err(PayError::SubscriptionNotActive("sub-can-002".to_string()))
        );
        // Cancelling one that does not exist is an error.
        assert!(matches!(
            service.cancel_subscription("sub-nope-001"),
            Err(PayError::InvalidSubscription(_))
        ));
    }

    #[test]
    fn refund_moves_coins_back_exactly_once() {
        let (mut service, alice, bob) = funded();
        let invoice = service
            .create_invoice(&bob, "inv-coffee-001", 250, "coffee", 1_000)
            .unwrap();
        service.pay_invoice(&invoice, &alice, 500).unwrap();
        assert_eq!(service.balance_of(alice.peer_id()), 750);
        assert_eq!(service.balance_of(bob.peer_id()), 350);
        // Bob, the original recipient, refunds Alice in full.
        let refund = Refund::new(&bob, "rfd-001", "inv-coffee-001", alice.peer_id(), 250).unwrap();
        assert!(refund.verify());
        let record = service.process_refund(&refund, &bob, 600).unwrap();
        assert_eq!(record.status, PaymentStatus::Refunded);
        assert_eq!(record.refunded_at, Some(600));
        assert_eq!(record.amount, 250);
        // The coins moved back, and the refund is on the ledger.
        assert_eq!(service.balance_of(alice.peer_id()), 1_000);
        assert_eq!(service.balance_of(bob.peer_id()), 100);
        assert_eq!(service.ledger().transactions().len(), 2);
        assert_eq!(
            service.ledger().transactions()[1].memo,
            "refund:inv-coffee-001"
        );
        // A second refund of the same payment is refused, with a
        // new refund id or the same one.
        let again = Refund::new(&bob, "rfd-002", "inv-coffee-001", alice.peer_id(), 250).unwrap();
        assert_eq!(
            service.process_refund(&again, &bob, 601),
            Err(PayError::AlreadyRefunded("inv-coffee-001".to_string()))
        );
        assert_eq!(
            service.process_refund(&refund, &bob, 602),
            Err(PayError::DuplicateRefund("rfd-001".to_string()))
        );
        // Nothing moved for the refused refunds.
        assert_eq!(service.balance_of(alice.peer_id()), 1_000);
        assert_eq!(service.balance_of(bob.peer_id()), 100);
        // History shows the payment as refunded.
        let history = service.history_for_payer(alice.peer_id(), 10);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].status, PaymentStatus::Refunded);
    }

    #[test]
    fn refund_of_unpaid_and_wrong_amount_and_bad_signature() {
        let (mut service, alice, bob) = funded();
        // Unpaid invoice: nothing to refund.
        service
            .create_invoice(&bob, "inv-unpaid-001", 100, "unpaid", 1_000)
            .unwrap();
        let refund =
            Refund::new(&bob, "rfd-u-001", "inv-unpaid-001", alice.peer_id(), 100).unwrap();
        assert_eq!(
            service.process_refund(&refund, &bob, 10),
            Err(PayError::NotPaid("inv-unpaid-001".to_string()))
        );
        // Unknown payment id: also nothing to refund.
        let refund = Refund::new(&bob, "rfd-u-002", "inv-ghost-001", alice.peer_id(), 100).unwrap();
        assert_eq!(
            service.process_refund(&refund, &bob, 10),
            Err(PayError::NotPaid("inv-ghost-001".to_string()))
        );
        // Wrong amount: refused, coins untouched.
        let invoice = service
            .create_invoice(&bob, "inv-amt-001", 250, "amt", 1_000)
            .unwrap();
        service.pay_invoice(&invoice, &alice, 10).unwrap();
        let short = Refund::new(&bob, "rfd-a-001", "inv-amt-001", alice.peer_id(), 100).unwrap();
        assert_eq!(
            service.process_refund(&short, &bob, 11),
            Err(PayError::RefundAmountMismatch {
                expected: 250,
                got: 100
            })
        );
        assert_eq!(service.balance_of(alice.peer_id()), 750);
        assert_eq!(service.balance_of(bob.peer_id()), 350);
        // Tampered refund: the signature no longer covers it.
        let mut forged =
            Refund::new(&bob, "rfd-a-002", "inv-amt-001", alice.peer_id(), 250).unwrap();
        forged.amount = 1;
        assert!(!forged.verify());
        assert_eq!(
            service.process_refund(&forged, &bob, 12),
            Err(PayError::BadRefundSignature)
        );
        // Wrong roles: the beneficiary must be the original payer.
        let wrong_roles =
            Refund::new(&bob, "rfd-a-003", "inv-amt-001", bob.peer_id(), 250).unwrap();
        assert!(matches!(
            service.process_refund(&wrong_roles, &bob, 13),
            Err(PayError::InvalidRefund(_))
        ));
        // Wrong presenter: Alice cannot execute Bob's refund.
        let refund = Refund::new(&bob, "rfd-a-004", "inv-amt-001", alice.peer_id(), 250).unwrap();
        assert!(matches!(
            service.process_refund(&refund, &alice, 14),
            Err(PayError::InvalidRefund(_))
        ));
        assert_eq!(service.balance_of(alice.peer_id()), 750);
    }

    #[test]
    fn subscription_payment_can_be_refunded() {
        let (mut service, alice, bob) = funded();
        let sub = make_sub(&alice, "sub-rfd-001", bob.peer_id(), 100, 10, 1, 100);
        service.register_subscription(&alice, &sub).unwrap();
        let outcomes = service.tick(100);
        assert_eq!(outcomes.len(), 1);
        assert!(matches!(outcomes[0], OccurrenceOutcome::Paid { .. }));
        assert_eq!(service.balance_of(alice.peer_id()), 900);
        // The occurrence id is the payment id for the refund.
        let refund = Refund::new(&bob, "rfd-s-001", "sub-rfd-001:0", alice.peer_id(), 100).unwrap();
        let record = service.process_refund(&refund, &bob, 110).unwrap();
        assert_eq!(record.status, PaymentStatus::Refunded);
        assert_eq!(record.kind, PaymentKind::Subscription);
        assert_eq!(service.balance_of(alice.peer_id()), 1_000);
        assert_eq!(service.balance_of(bob.peer_id()), 100);
    }

    #[test]
    fn history_lists_by_payer_and_payee_newest_first() {
        let (mut service, alice, bob) = funded();
        let first = service
            .create_invoice(&bob, "inv-h-001", 100, "first", 1_000)
            .unwrap();
        service.pay_invoice(&first, &alice, 10).unwrap();
        let second = service
            .create_invoice(&bob, "inv-h-002", 50, "second", 1_000)
            .unwrap();
        service.pay_invoice(&second, &alice, 20).unwrap();
        let third = service
            .create_invoice(&alice, "inv-h-003", 25, "third", 1_000)
            .unwrap();
        service.pay_invoice(&third, &bob, 30).unwrap();
        // Alice paid twice, newest first.
        let alice_paid = service.history_for_payer(alice.peer_id(), 10);
        assert_eq!(alice_paid.len(), 2);
        assert_eq!(alice_paid[0].payment_id, "inv-h-002");
        assert_eq!(alice_paid[1].payment_id, "inv-h-001");
        assert!(alice_paid.iter().all(|r| r.status == PaymentStatus::Paid));
        // Limit applies.
        let limited = service.history_for_payer(alice.peer_id(), 1);
        assert_eq!(limited.len(), 1);
        assert_eq!(limited[0].payment_id, "inv-h-002");
        assert!(service.history_for_payer(alice.peer_id(), 0).is_empty());
        // Bob received twice and paid once.
        let bob_got = service.history_for_payee(bob.peer_id(), 10);
        assert_eq!(bob_got.len(), 2);
        assert_eq!(bob_got[0].payment_id, "inv-h-002");
        let bob_paid = service.history_for_payer(bob.peer_id(), 10);
        assert_eq!(bob_paid.len(), 1);
        assert_eq!(bob_paid[0].payment_id, "inv-h-003");
        // Whole history, newest first, with a limit.
        let all = service.history(2);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].payment_id, "inv-h-003");
        assert_eq!(all[1].payment_id, "inv-h-002");
        // Every entry carries its receipt.
        for record in service.history(10) {
            assert!(record.receipt.verify());
            assert_eq!(record.receipt.amount, record.amount);
        }
        // After a refund, the entry's status flips in place.
        let refund = Refund::new(&bob, "rfd-h-001", "inv-h-002", alice.peer_id(), 50).unwrap();
        service.process_refund(&refund, &bob, 40).unwrap();
        let alice_paid = service.history_for_payer(alice.peer_id(), 10);
        assert_eq!(alice_paid[0].status, PaymentStatus::Refunded);
        assert_eq!(alice_paid[0].refunded_at, Some(40));
        assert_eq!(alice_paid[1].status, PaymentStatus::Paid);
    }

    // -- OWNER-FREE LAW ------------------------------------------

    /// A fixture with the owner registered: Alice is the owner.
    fn funded_with_owner() -> (PaymentService, Identity, Identity) {
        let (mut service, alice, bob) = funded();
        service.set_owner(alice.peer_id().clone());
        (service, alice, bob)
    }

    #[test]
    fn owner_pays_invoice_free_and_repay_is_still_refused() {
        let (mut service, alice, bob) = funded_with_owner();
        assert!(service.is_owner(alice.peer_id()));
        assert!(!service.is_owner(bob.peer_id()));
        // Bob invoices the owner for 250.
        let invoice = service
            .create_invoice(&bob, "inv-free-001", 250, "coffee", 1_000)
            .unwrap();
        let receipt = service.pay_invoice(&invoice, &alice, 500).unwrap();
        // A valid receipt is still issued, marked owner-free,
        // with amount moved 0.
        assert!(receipt.verify());
        assert!(receipt.owner_free);
        assert_eq!(receipt.amount, 0);
        assert_eq!(receipt.payer, *alice.peer_id());
        assert_eq!(receipt.recipient, *bob.peer_id());
        assert!(service.is_paid("inv-free-001"));
        // No coins moved for anyone, and the ledger has no
        // transaction for it.
        assert_eq!(service.balance_of(alice.peer_id()), 1_000);
        assert_eq!(service.balance_of(bob.peer_id()), 100);
        assert!(service.ledger().transactions().is_empty());
        // History records the payment with amount 0.
        let history = service.history_for_payer(alice.peer_id(), 10);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].amount, 0);
        assert_eq!(history[0].receipt.amount, 0);
        assert!(history[0].receipt.owner_free);
        // Paying the same invoice again is still a replay
        // refusal, and still moves nothing.
        assert_eq!(
            service.pay_invoice(&invoice, &alice, 501),
            Err(PayError::AlreadyPaid("inv-free-001".to_string()))
        );
        assert_eq!(service.balance_of(alice.peer_id()), 1_000);
        assert_eq!(service.balance_of(bob.peer_id()), 100);
    }

    #[test]
    fn owner_exemption_only_applies_when_registered() {
        // No owner registered: the same payer is charged.
        let (mut service, alice, bob) = funded();
        assert!(service.owner().is_none());
        assert!(!service.is_owner(alice.peer_id()));
        let invoice = service
            .create_invoice(&bob, "inv-chg-001", 250, "coffee", 1_000)
            .unwrap();
        let receipt = service.pay_invoice(&invoice, &alice, 500).unwrap();
        assert!(!receipt.owner_free);
        assert_eq!(receipt.amount, 250);
        assert_eq!(service.balance_of(alice.peer_id()), 750);
        assert_eq!(service.balance_of(bob.peer_id()), 350);
        // An owner registered for someone else does not exempt
        // this payer either.
        let (mut service, alice, bob) = funded();
        service.set_owner(bob.peer_id().clone());
        let invoice = service
            .create_invoice(&bob, "inv-chg-002", 250, "coffee", 1_000)
            .unwrap();
        let receipt = service.pay_invoice(&invoice, &alice, 500).unwrap();
        assert!(!receipt.owner_free);
        assert_eq!(service.balance_of(alice.peer_id()), 750);
        assert_eq!(service.balance_of(bob.peer_id()), 350);
    }

    #[test]
    fn owner_subscription_occurrences_execute_free() {
        let (mut service, alice, bob) = funded_with_owner();
        let sub = make_sub(&alice, "sub-free-001", bob.peer_id(), 100, 10, 3, 100);
        service.register_subscription(&alice, &sub).unwrap();
        let outcomes = service.tick(125);
        assert_eq!(outcomes.len(), 3);
        for (n, outcome) in outcomes.iter().enumerate() {
            match outcome {
                OccurrenceOutcome::Paid {
                    occurrence,
                    receipt,
                    ..
                } => {
                    assert_eq!(*occurrence, n as u32);
                    assert!(receipt.verify());
                    assert!(receipt.owner_free);
                    assert_eq!(receipt.amount, 0);
                }
                other => panic!("expected a paid occurrence, got {other:?}"),
            }
        }
        assert_eq!(
            service.subscription_status("sub-free-001"),
            Some(SubscriptionStatus::Finished)
        );
        assert_eq!(service.subscription_counts("sub-free-001"), Some((3, 3)));
        // No coins moved for any occurrence.
        assert_eq!(service.balance_of(alice.peer_id()), 1_000);
        assert_eq!(service.balance_of(bob.peer_id()), 100);
        assert!(service.ledger().transactions().is_empty());
        let history = service.history_for_payer(alice.peer_id(), 10);
        assert_eq!(history.len(), 3);
        assert!(history.iter().all(|r| r.amount == 0));
        assert!(history.iter().all(|r| r.receipt.owner_free));
    }

    #[test]
    fn void_payment_voids_owner_free_only() {
        let (mut service, alice, bob) = funded_with_owner();
        // An owner-free payment can be voided: the record flips
        // to Refunded and no coins move.
        let invoice = service
            .create_invoice(&bob, "inv-void-001", 250, "coffee", 1_000)
            .unwrap();
        service.pay_invoice(&invoice, &alice, 500).unwrap();
        let record = service.void_payment("inv-void-001", 600).unwrap();
        assert_eq!(record.status, PaymentStatus::Refunded);
        assert_eq!(record.refunded_at, Some(600));
        assert_eq!(record.amount, 0);
        assert_eq!(service.balance_of(alice.peer_id()), 1_000);
        assert_eq!(service.balance_of(bob.peer_id()), 100);
        // Voiding again is refused.
        assert_eq!(
            service.void_payment("inv-void-001", 601),
            Err(PayError::AlreadyRefunded("inv-void-001".to_string()))
        );
        // A normal (non-owner) payment cannot be voided: it
        // moved real coins, so only a real refund settles it.
        let invoice = service
            .create_invoice(&alice, "inv-void-002", 100, "tea", 1_000)
            .unwrap();
        service.pay_invoice(&invoice, &bob, 500).unwrap();
        assert!(matches!(
            service.void_payment("inv-void-002", 600),
            Err(PayError::InvalidRefund(_))
        ));
        // An unknown payment id is a clean error.
        assert_eq!(
            service.void_payment("inv-nope-001", 600),
            Err(PayError::NotPaid("inv-nope-001".to_string()))
        );
    }
}
