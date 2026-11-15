//! Kash•Korner — a simple store / marketplace settled on X•Koin.
//!
//! A listing is a signed offer to sell: a human-readable id, a
//! seller, an item name, a description, a category, and a price.
//! The seller's signature covers the whole listing, so a forged or
//! tampered listing is refused before it can be listed.
//!
//! Buying a listing settles through K•Pay on the coin ledger: the
//! buyer pays the listing's invoice and gets a signed receipt, and
//! the listing is marked sold. A sold listing cannot be bought
//! again, and a buyer who does not hold enough is refused before
//! any coins move.
//!
//! A purchase can also go through **escrow** instead of paying the
//! seller directly: the buyer pays into an escrow account held by
//! the store; the seller marks the item delivered; the buyer
//! confirms receipt and the escrow releases to the seller — or the
//! buyer disputes and the escrow refunds the buyer. Escrow states
//! are `Held -> Delivered -> Released`, with `Held | Delivered ->
//! Refunded`; `Released` and `Refunded` are terminal.
//!
//! Listings are identified by human-readable ids
//! (`listing-lamp-001`) and item names (`"brass lamp"`), never
//! hashes. The store holds no inventory itself — it is a record
//! of offers and sales, and settlement happens on the ledger.
//!
//! Reviews are proof of purchase, not proof of quality: a review
//! is only accepted from a buyer holding the completed purchase
//! receipt for that listing, one review per purchase.

use std::collections::{HashMap, HashSet};
use std::fmt;

use k_pay::{Invoice, PayError, PaymentRecord, PaymentService, Receipt, Refund};
use kernel_crypto::{sha256, to_hex, Ed25519Signature};
use kernel_identity::{Identity, IdentityError, PeerId};
use x_koin::{KoinError, Ledger};

/// Domain separator for listing signatures: a signature covers
/// `b"KASH-LIST1" || id || seller key || name || description ||
/// category || price`.
const LISTING_SIGN_DOMAIN: &[u8] = b"KASH-LIST1";
/// Domain separator for review signatures: a signature covers
/// `b"KASH-REVIEW1" || listing id || reviewer key || stars || text`.
const REVIEW_SIGN_DOMAIN: &[u8; 12] = b"KASH-REVIEW1";
/// Direct-purchase invoices never expire.
const NO_EXPIRY: u64 = u64::MAX;

/// Errors returned by store operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KashError {
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// Wraps a [`KoinError`] from settlement.
    Koin(KoinError),
    /// Wraps a [`PayError`] from K•Pay settlement.
    Pay(PayError),
    /// A listing id, name, description, category, or price broke
    /// the rules.
    InvalidListing(String),
    /// A listing's signature did not verify against its seller.
    BadSignature,
    /// A listing with this id already exists.
    DuplicateListing(String),
    /// No listing has this id.
    ListingNotFound(String),
    /// The listing was already sold.
    AlreadySold(String),
    /// A review's stars or text broke the rules.
    InvalidReview(String),
    /// The reviewer holds no completed purchase receipt for this
    /// listing.
    NoPurchaseReceipt(String),
    /// This purchase was already reviewed.
    DuplicateReview(String),
    /// The review's receipt signature did not verify.
    BadReceipt,
    /// No escrow exists for this listing.
    EscrowNotFound(String),
    /// The escrow is in the wrong state for this operation.
    WrongEscrowState(String),
    /// The caller is not the escrow's buyer.
    NotBuyer(String),
    /// The caller is not the listing's seller.
    NotSeller(String),
    /// No invoice is registered for this listing.
    InvoiceNotFound(String),
}

impl fmt::Display for KashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KashError::Identity(e) => write!(f, "identity error: {e}"),
            KashError::Koin(e) => write!(f, "coin error: {e}"),
            KashError::Pay(e) => write!(f, "payment error: {e}"),
            KashError::InvalidListing(msg) => write!(f, "invalid listing: {msg}"),
            KashError::BadSignature => write!(f, "listing signature is invalid"),
            KashError::DuplicateListing(id) => {
                write!(f, "a listing with id '{id}' already exists")
            }
            KashError::ListingNotFound(id) => write!(f, "no listing with id '{id}'"),
            KashError::AlreadySold(id) => write!(f, "listing '{id}' was already sold"),
            KashError::InvalidReview(msg) => write!(f, "invalid review: {msg}"),
            KashError::NoPurchaseReceipt(msg) => write!(f, "no purchase receipt: {msg}"),
            KashError::DuplicateReview(id) => {
                write!(f, "listing '{id}' was already reviewed by this buyer")
            }
            KashError::BadReceipt => write!(f, "receipt signature is invalid"),
            KashError::EscrowNotFound(id) => write!(f, "no escrow for listing '{id}'"),
            KashError::WrongEscrowState(msg) => write!(f, "wrong escrow state: {msg}"),
            KashError::NotBuyer(id) => write!(f, "caller is not the buyer of '{id}'"),
            KashError::NotSeller(id) => write!(f, "caller is not the seller of '{id}'"),
            KashError::InvoiceNotFound(id) => write!(f, "no invoice for listing '{id}'"),
        }
    }
}

impl std::error::Error for KashError {}

impl From<IdentityError> for KashError {
    fn from(e: IdentityError) -> Self {
        KashError::Identity(e)
    }
}

impl From<KoinError> for KashError {
    fn from(e: KoinError) -> Self {
        KashError::Koin(e)
    }
}

impl From<PayError> for KashError {
    fn from(e: PayError) -> Self {
        KashError::Pay(e)
    }
}

/// Rules for human-readable ids: 1-64 characters, ASCII letters,
/// digits, `-`, `_`, `.`. No spaces, no hashes.
fn validate_id(id: &str) -> Result<(), KashError> {
    if id.is_empty() {
        return Err(KashError::InvalidListing(
            "listing id must not be empty".to_string(),
        ));
    }
    if id.len() > 64 {
        return Err(KashError::InvalidListing(
            "listing id must be at most 64 characters".to_string(),
        ));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Err(KashError::InvalidListing(
            "listing id may only use letters, digits, '-', '_' and '.'".to_string(),
        ));
    }
    Ok(())
}

/// Rules for item names and descriptions: non-empty, at most 128
/// printable characters.
fn validate_text(text: &str, what: &str) -> Result<(), KashError> {
    if text.is_empty() {
        return Err(KashError::InvalidListing(format!(
            "{what} must not be empty"
        )));
    }
    if text.len() > 128 {
        return Err(KashError::InvalidListing(format!(
            "{what} must be at most 128 characters"
        )));
    }
    if !text.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
        return Err(KashError::InvalidListing(format!(
            "{what} must be printable ASCII"
        )));
    }
    Ok(())
}

/// Normalize a category: lowercase, trimmed; 1-32 characters of
/// lowercase letters, digits, `-`, `_`. Returns `None` when the
/// category breaks the rules.
fn normalize_category(category: &str) -> Option<String> {
    let lower = category.trim().to_lowercase();
    if lower.is_empty() || lower.len() > 32 {
        return None;
    }
    if !lower
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
    {
        return None;
    }
    Some(lower)
}

/// Derive a k-pay id for one step of an escrow flow: readable
/// (`<prefix>-<listing id>-<seq>`) while it fits k-pay's
/// 64-character id limit, otherwise a deterministic SHA-256 hex
/// fallback (still unique in practice).
fn derive_id(prefix: &str, listing_id: &str, seq: u64) -> String {
    let candidate = format!("{prefix}-{listing_id}-{seq}");
    if candidate.len() <= 64 {
        return candidate;
    }
    let digest = sha256(candidate.as_bytes());
    let hex = to_hex(&digest);
    format!("{prefix}-{}", &hex[..64 - prefix.len() - 1])
}

/// The message a seller signs for a listing.
fn listing_sign_message(
    id: &str,
    seller: &PeerId,
    name: &str,
    description: &str,
    category: &str,
    price: u64,
) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend_from_slice(LISTING_SIGN_DOMAIN);
    msg.extend_from_slice(&(id.len() as u16).to_le_bytes());
    msg.extend_from_slice(id.as_bytes());
    msg.extend_from_slice(seller.public_key().as_bytes());
    msg.extend_from_slice(&(name.len() as u16).to_le_bytes());
    msg.extend_from_slice(name.as_bytes());
    msg.extend_from_slice(&(description.len() as u16).to_le_bytes());
    msg.extend_from_slice(description.as_bytes());
    msg.extend_from_slice(&(category.len() as u16).to_le_bytes());
    msg.extend_from_slice(category.as_bytes());
    msg.extend_from_slice(&price.to_le_bytes());
    msg
}

/// The message a reviewer signs for a review.
fn review_sign_message(listing_id: &str, reviewer: &PeerId, stars: u8, text: &str) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend_from_slice(REVIEW_SIGN_DOMAIN);
    msg.extend_from_slice(&(listing_id.len() as u16).to_le_bytes());
    msg.extend_from_slice(listing_id.as_bytes());
    msg.extend_from_slice(reviewer.public_key().as_bytes());
    msg.push(stars);
    msg.extend_from_slice(&(text.len() as u16).to_le_bytes());
    msg.extend_from_slice(text.as_bytes());
    msg
}

/// A listing: a signed offer to sell one item at a price.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Listing {
    /// Human-readable id, e.g. `"listing-lamp-001"`.
    pub id: String,
    /// Who is selling.
    pub seller: PeerId,
    /// The item's human-readable name, e.g. `"brass lamp"`.
    pub name: String,
    /// A human-readable description of the item.
    pub description: String,
    /// The listing's category, e.g. `"home"`, normalized to
    /// lowercase.
    pub category: String,
    /// The asking price, in the coin's smallest unit.
    pub price: u64,
    /// The seller's signature over id, seller, name,
    /// description, category, and price.
    pub signature: Ed25519Signature,
}

impl Listing {
    /// Create and sign a listing as `seller`. The price must be
    /// greater than zero; the id, name, description, and category
    /// are validated (the category is normalized to lowercase).
    pub fn new(
        seller: &Identity,
        id: &str,
        name: &str,
        description: &str,
        price: u64,
        category: &str,
    ) -> Result<Self, KashError> {
        validate_id(id)?;
        validate_text(name, "item name")?;
        validate_text(description, "description")?;
        let category = normalize_category(category).ok_or_else(|| {
            KashError::InvalidListing(
                "category must be 1-32 lowercase letters, digits, '-' or '_'".to_string(),
            )
        })?;
        if price == 0 {
            return Err(KashError::InvalidListing(
                "listing price must be greater than zero".to_string(),
            ));
        }
        let seller_id = seller.peer_id().clone();
        let msg = listing_sign_message(id, &seller_id, name, description, &category, price);
        let signature = seller.sign(&msg);
        Ok(Self {
            id: id.to_string(),
            seller: seller_id,
            name: name.to_string(),
            description: description.to_string(),
            category,
            price,
            signature,
        })
    }

    /// True if the signature verifies against the seller for this
    /// exact id, name, description, category, and price. Any
    /// tampering fails this.
    pub fn verify(&self) -> bool {
        let msg = listing_sign_message(
            &self.id,
            &self.seller,
            &self.name,
            &self.description,
            &self.category,
            self.price,
        );
        self.seller.verify(&msg, &self.signature).is_ok()
    }
}

/// A review: a signed rating from a buyer who completed a
/// purchase of the listing. Reviews prove a purchase happened,
/// not that the item was good.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Review {
    /// The id of the listing being reviewed.
    pub listing_id: String,
    /// Who wrote the review (must be the buyer).
    pub reviewer: PeerId,
    /// Stars, 1-5.
    pub stars: u8,
    /// Optional review text (here required, 1-128 printable
    /// characters).
    pub text: String,
    /// The reviewer's signature over listing id, reviewer,
    /// stars, and text.
    pub signature: Ed25519Signature,
}

impl Review {
    /// Create and sign a review as `reviewer`. Stars must be
    /// 1-5 and the text 1-128 printable characters. Whether the
    /// reviewer actually bought the listing is checked by the
    /// store, not here.
    pub fn new(
        reviewer: &Identity,
        listing_id: &str,
        stars: u8,
        text: &str,
    ) -> Result<Self, KashError> {
        if !(1..=5).contains(&stars) {
            return Err(KashError::InvalidReview(format!(
                "stars must be 1-5, got {stars}"
            )));
        }
        if text.is_empty()
            || text.len() > 128
            || !text.chars().all(|c| c.is_ascii_graphic() || c == ' ')
        {
            return Err(KashError::InvalidReview(
                "review text must be 1-128 printable characters".to_string(),
            ));
        }
        let reviewer_id = reviewer.peer_id().clone();
        let msg = review_sign_message(listing_id, &reviewer_id, stars, text);
        let signature = reviewer.sign(&msg);
        Ok(Self {
            listing_id: listing_id.to_string(),
            reviewer: reviewer_id,
            stars,
            text: text.to_string(),
            signature,
        })
    }

    /// True if the signature verifies against the reviewer for
    /// this exact listing id, stars, and text.
    pub fn verify(&self) -> bool {
        let msg = review_sign_message(&self.listing_id, &self.reviewer, self.stars, &self.text);
        self.reviewer.verify(&msg, &self.signature).is_ok()
    }
}

/// The state of one escrow purchase. `Held -> Delivered ->
/// Released`, with `Held | Delivered -> Refunded`. `Released`
/// and `Refunded` are terminal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EscrowState {
    /// The buyer paid; coins sit in the escrow account.
    Held,
    /// The seller marked the item delivered; coins still held.
    Delivered,
    /// The buyer confirmed receipt; coins paid to the seller.
    Released,
    /// The buyer disputed; coins returned to the buyer.
    Refunded,
}

impl fmt::Display for EscrowState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EscrowState::Held => write!(f, "held"),
            EscrowState::Delivered => write!(f, "delivered"),
            EscrowState::Released => write!(f, "released"),
            EscrowState::Refunded => write!(f, "refunded"),
        }
    }
}

/// One escrow purchase: the coins for one listing, held by the
/// store until release or refund.
#[derive(Clone, Debug)]
pub struct Escrow {
    /// The id of the listing being bought.
    pub listing_id: String,
    /// Who paid into escrow.
    pub buyer: PeerId,
    /// Who is selling.
    pub seller: PeerId,
    /// The amount held, in the coin's smallest unit.
    pub amount: u64,
    /// True when the buyer is the registered owner and the
    /// purchase was owner-free: nothing was ever held, so the
    /// release and any dispute move no coins.
    pub owner_free: bool,
    /// Where the escrow is in its lifecycle.
    pub state: EscrowState,
    /// The k-pay invoice id the buyer paid into escrow.
    pub escrow_invoice_id: String,
    /// The k-pay invoice id the seller is paid from on release,
    /// once the seller has marked the item delivered.
    pub release_invoice_id: Option<String>,
    /// The purchase attempt number for this listing; keeps
    /// invoice ids unique across re-sales after a refund.
    pub seq: u64,
}

/// A store: a K•Pay payment service plus the listings offered
/// against it. Listings are created by sellers and can arrive
/// from any peer; only their signatures authenticate them, so a
/// listing does not need to have been created through this store
/// to be listed here — but a listing bought here is marked sold
/// here. (A listing bought here must have its direct-purchase
/// invoice registered here, which happens when the seller creates
/// it through this store.)
pub struct Store {
    payments: PaymentService,
    /// The store's own escrow identity: a fresh random key held
    /// only by this store. Coins paid into escrow sit in this
    /// account until release or refund.
    escrow: Identity,
    /// Listings by id, including sold ones.
    listings: HashMap<String, Listing>,
    /// Ids of listings already sold.
    sold: HashSet<String>,
    /// Buyer of each sold listing, by listing id.
    buyers: HashMap<String, PeerId>,
    /// Current escrow purchase per listing, by listing id.
    escrows: HashMap<String, Escrow>,
    /// Escrow purchase attempts per listing, for unique
    /// invoice ids across re-sales after a refund.
    purchase_seq: HashMap<String, u64>,
    /// Accepted reviews, in submission order.
    reviews: Vec<Review>,
    /// (listing id, buyer key) pairs already reviewed, so one
    /// purchase gets at most one review.
    reviewed: HashSet<(String, [u8; 32])>,
}

impl Store {
    /// Create a store whose ledger starts at the given genesis
    /// allocation.
    pub fn new(genesis: &[(PeerId, u64)]) -> Result<Self, KashError> {
        Ok(Self {
            payments: PaymentService::new(genesis)?,
            escrow: Identity::generate("kash-escrow")?,
            listings: HashMap::new(),
            sold: HashSet::new(),
            buyers: HashMap::new(),
            escrows: HashMap::new(),
            purchase_seq: HashMap::new(),
            reviews: Vec::new(),
            reviewed: HashSet::new(),
        })
    }

    /// The underlying K•Pay payment service.
    pub fn payments(&self) -> &PaymentService {
        &self.payments
    }

    /// Register the owner of this node on the store's payment
    /// service: the peer for whom everything inside KERNEL is
    /// free (the OWNER-FREE LAW). Owner purchases — direct and
    /// escrow — then move no coins: a direct buy is satisfied
    /// without payment, and an escrow buy holds nothing, so its
    /// release and any dispute move nothing either.
    pub fn set_owner(&mut self, owner: PeerId) {
        self.payments.set_owner(owner);
    }

    /// The underlying coin ledger.
    pub fn ledger(&self) -> &Ledger {
        self.payments.ledger()
    }

    /// The balance of a peer in the underlying ledger.
    pub fn balance_of(&self, peer: &PeerId) -> u64 {
        self.payments.balance_of(peer)
    }

    /// The store's escrow account: coins paid into escrow sit
    /// here until release or refund.
    pub fn escrow_peer_id(&self) -> &PeerId {
        self.escrow.peer_id()
    }

    /// True if the listing with this id was already sold.
    pub fn is_sold(&self, id: &str) -> bool {
        self.sold.contains(id)
    }

    /// The listing with this id, if any — sold or not.
    pub fn get_listing(&self, id: &str) -> Option<&Listing> {
        self.listings.get(id)
    }

    /// The current escrow purchase for a listing, if any.
    pub fn get_escrow(&self, id: &str) -> Option<&Escrow> {
        self.escrows.get(id)
    }

    /// The state of the escrow purchase for a listing, if any.
    pub fn escrow_state(&self, id: &str) -> Option<EscrowState> {
        self.escrows.get(id).map(|e| e.state)
    }

    /// Listings still for sale, sorted by id so the list is
    /// deterministic.
    pub fn list_open(&self) -> Vec<&Listing> {
        let mut open: Vec<&Listing> = self
            .listings
            .values()
            .filter(|l| !self.sold.contains(&l.id))
            .collect();
        open.sort_by(|a, b| a.id.cmp(&b.id));
        open
    }

    /// Listings still for sale in one category, in listing
    /// (id) order. An unknown or empty category is a clean
    /// empty list, not an error. The category is normalized
    /// the same way as at listing creation.
    pub fn listings_by_category(&self, category: &str) -> Vec<&Listing> {
        let Some(want) = normalize_category(category) else {
            return Vec::new();
        };
        let mut found: Vec<&Listing> = self
            .listings
            .values()
            .filter(|l| !self.sold.contains(&l.id) && l.category == want)
            .collect();
        found.sort_by(|a, b| a.id.cmp(&b.id));
        found
    }

    /// Search listings by name/description substring,
    /// case-insensitive, optionally scoped to one category.
    /// Results are in listing (id) order. Sold listings are
    /// hidden unless `include_sold` is set.
    pub fn search(&self, query: &str, category: Option<&str>, include_sold: bool) -> Vec<&Listing> {
        let q = query.to_lowercase();
        let scope = category.and_then(normalize_category);
        if category.is_some() && scope.is_none() {
            return Vec::new();
        }
        let mut found: Vec<&Listing> = self
            .listings
            .values()
            .filter(|l| {
                (include_sold || !self.sold.contains(&l.id))
                    && scope.as_ref().is_none_or(|c| &l.category == c)
                    && (l.name.to_lowercase().contains(&q)
                        || l.description.to_lowercase().contains(&q))
            })
            .collect();
        found.sort_by(|a, b| a.id.cmp(&b.id));
        found
    }

    /// Create and register a listing as `seller`. The id must
    /// not already be registered. Registers the listing's
    /// direct-purchase invoice with K•Pay (recipient is the
    /// seller, expiry never) before the listing goes live, so
    /// a listing can never exist here without a way to pay
    /// for it.
    pub fn create_listing(
        &mut self,
        seller: &Identity,
        id: &str,
        name: &str,
        description: &str,
        price: u64,
        category: &str,
    ) -> Result<Listing, KashError> {
        if self.listings.contains_key(id) {
            return Err(KashError::DuplicateListing(id.to_string()));
        }
        let listing = Listing::new(seller, id, name, description, price, category)?;
        let memo = format!("kash:{}", listing.id);
        self.payments
            .create_invoice(seller, &listing.id, listing.price, &memo, NO_EXPIRY)?;
        self.listings.insert(id.to_string(), listing.clone());
        Ok(listing)
    }

    /// Buy a listing as `buyer` at time `now`, paying the
    /// seller directly. Validates, in order: the listing
    /// exists, the listing's signature verifies against its
    /// seller, and it has not been sold; then pays the
    /// listing's K•Pay invoice through the coin ledger and
    /// returns the signed receipt, and marks the listing sold.
    /// A rejected purchase changes nothing — no coins move
    /// and the listing stays for sale.
    pub fn buy_listing(
        &mut self,
        id: &str,
        buyer: &Identity,
        now: u64,
    ) -> Result<Receipt, KashError> {
        let listing = self
            .listings
            .get(id)
            .ok_or_else(|| KashError::ListingNotFound(id.to_string()))?
            .clone();
        if !listing.verify() {
            return Err(KashError::BadSignature);
        }
        if self.sold.contains(id) {
            return Err(KashError::AlreadySold(id.to_string()));
        }
        let invoice = self
            .payments
            .get_invoice(id)
            .ok_or_else(|| KashError::InvoiceNotFound(id.to_string()))?
            .clone();
        let receipt = self.payments.pay_invoice(&invoice, buyer, now)?;
        self.sold.insert(id.to_string());
        self.buyers.insert(id.to_string(), buyer.peer_id().clone());
        Ok(receipt)
    }

    /// Buy a listing as `buyer` at time `now` through escrow:
    /// the buyer pays the listing's price into the store's
    /// escrow account (coins leave the buyer, held by the
    /// store, not the seller) and the listing is marked sold.
    /// A rejected purchase changes nothing — no coins move,
    /// no escrow is created, and the listing stays for sale.
    pub fn buy_listing_escrow(
        &mut self,
        id: &str,
        buyer: &Identity,
        now: u64,
    ) -> Result<Receipt, KashError> {
        let listing = self
            .listings
            .get(id)
            .ok_or_else(|| KashError::ListingNotFound(id.to_string()))?
            .clone();
        if !listing.verify() {
            return Err(KashError::BadSignature);
        }
        if self.sold.contains(id) {
            return Err(KashError::AlreadySold(id.to_string()));
        }
        let seq = {
            let entry = self.purchase_seq.entry(id.to_string()).or_insert(0);
            *entry += 1;
            *entry
        };
        let invoice_id = derive_id("escrow", id, seq);
        let memo = format!("escrow:{}", id);
        let invoice = self.payments.create_invoice(
            &self.escrow,
            &invoice_id,
            listing.price,
            &memo,
            NO_EXPIRY,
        )?;
        let receipt = self.payments.pay_invoice(&invoice, buyer, now)?;
        self.sold.insert(id.to_string());
        self.buyers.insert(id.to_string(), buyer.peer_id().clone());
        self.escrows.insert(
            id.to_string(),
            Escrow {
                listing_id: id.to_string(),
                buyer: buyer.peer_id().clone(),
                seller: listing.seller.clone(),
                amount: listing.price,
                owner_free: receipt.owner_free,
                state: EscrowState::Held,
                escrow_invoice_id: invoice_id,
                release_invoice_id: None,
                seq,
            },
        );
        Ok(receipt)
    }

    /// Mark an escrowed item delivered, as the seller. Only the
    /// listing's seller may do this, and only while the escrow
    /// is `Held`. Registers the release invoice (recipient is
    /// the seller) that the buyer later confirms, and moves
    /// the escrow to `Delivered`. Returns the release invoice.
    pub fn mark_delivered(&mut self, id: &str, seller: &Identity) -> Result<Invoice, KashError> {
        let escrow = self
            .escrows
            .get(id)
            .ok_or_else(|| KashError::EscrowNotFound(id.to_string()))?;
        if escrow.state != EscrowState::Held {
            return Err(KashError::WrongEscrowState(format!(
                "listing '{id}' escrow is {}, only a held escrow can be marked delivered",
                escrow.state
            )));
        }
        if &escrow.seller != seller.peer_id() {
            return Err(KashError::NotSeller(id.to_string()));
        }
        let seq = escrow.seq;
        let amount = escrow.amount;
        let release_id = derive_id("release", id, seq);
        let memo = format!("release:{}", id);
        let invoice =
            self.payments
                .create_invoice(seller, &release_id, amount, &memo, NO_EXPIRY)?;
        let escrow = self.escrows.get_mut(id).expect("escrow still present");
        escrow.release_invoice_id = Some(release_id);
        escrow.state = EscrowState::Delivered;
        Ok(invoice)
    }

    /// Confirm receipt of an escrowed item, as the buyer, and
    /// release the held coins to the seller. Only the escrow's
    /// buyer may do this, and only once the seller has marked
    /// the item delivered. Returns the release receipt (payer
    /// is the escrow account). The escrow moves to `Released`.
    pub fn confirm_receipt(
        &mut self,
        id: &str,
        buyer: &Identity,
        now: u64,
    ) -> Result<Receipt, KashError> {
        let escrow = self
            .escrows
            .get(id)
            .ok_or_else(|| KashError::EscrowNotFound(id.to_string()))?
            .clone();
        if escrow.state != EscrowState::Delivered {
            return Err(KashError::WrongEscrowState(format!(
                "listing '{id}' escrow is {}, only a delivered escrow can be confirmed",
                escrow.state
            )));
        }
        if &escrow.buyer != buyer.peer_id() {
            return Err(KashError::NotBuyer(id.to_string()));
        }
        let release_id = escrow
            .release_invoice_id
            .clone()
            .ok_or_else(|| KashError::InvoiceNotFound(id.to_string()))?;
        if escrow.owner_free {
            // OWNER-FREE LAW: nothing was ever held, so the
            // release moves nothing. The release receipt is
            // owner-free (amount 0), signed by the escrow
            // account that would have paid it.
            let receipt = Receipt::owner_free(&release_id, &self.escrow, &escrow.seller);
            self.escrows
                .get_mut(id)
                .expect("escrow still present")
                .state = EscrowState::Released;
            return Ok(receipt);
        }
        let invoice = self
            .payments
            .get_invoice(&release_id)
            .ok_or_else(|| KashError::InvoiceNotFound(release_id.clone()))?
            .clone();
        let receipt = self.payments.pay_invoice(&invoice, &self.escrow, now)?;
        self.escrows
            .get_mut(id)
            .expect("escrow still present")
            .state = EscrowState::Released;
        Ok(receipt)
    }

    /// Dispute an escrowed purchase, as the buyer, and refund
    /// the held coins to the buyer. Only the escrow's buyer
    /// may do this, and only while the escrow is `Held` or
    /// `Delivered`. The escrow moves to `Refunded` and the
    /// listing goes back on sale. Returns the refund record.
    pub fn dispute_escrow(
        &mut self,
        id: &str,
        buyer: &Identity,
        now: u64,
    ) -> Result<PaymentRecord, KashError> {
        let escrow = self
            .escrows
            .get(id)
            .ok_or_else(|| KashError::EscrowNotFound(id.to_string()))?
            .clone();
        if escrow.state != EscrowState::Held && escrow.state != EscrowState::Delivered {
            return Err(KashError::WrongEscrowState(format!(
                "listing '{id}' escrow is {}, only a held or delivered escrow can be disputed",
                escrow.state
            )));
        }
        if &escrow.buyer != buyer.peer_id() {
            return Err(KashError::NotBuyer(id.to_string()));
        }
        if escrow.owner_free {
            // OWNER-FREE LAW: nothing was ever held, so the
            // dispute refunds nothing. The owner-free payment
            // record is voided instead of refunded, and the
            // listing goes back on sale.
            let record = self.payments.void_payment(&escrow.escrow_invoice_id, now)?;
            self.escrows
                .get_mut(id)
                .expect("escrow still present")
                .state = EscrowState::Refunded;
            self.sold.remove(id);
            self.buyers.remove(id);
            return Ok(record);
        }
        let refund_id = derive_id("refund", id, escrow.seq);
        let refund = Refund::new(
            &self.escrow,
            &refund_id,
            &escrow.escrow_invoice_id,
            buyer.peer_id(),
            escrow.amount,
        )?;
        let record = self.payments.process_refund(&refund, &self.escrow, now)?;
        self.escrows
            .get_mut(id)
            .expect("escrow still present")
            .state = EscrowState::Refunded;
        self.sold.remove(id);
        self.buyers.remove(id);
        Ok(record)
    }

    /// Submit a review for a listing, as `reviewer`, presenting
    /// the completed purchase receipt for that listing. The
    /// receipt must verify, name the reviewer as payer, match
    /// the listing's price, and belong to a completed purchase:
    /// a direct purchase, or an escrow purchase that was
    /// released. One review per purchase; a second review of
    /// the same purchase is refused.
    pub fn submit_review(
        &mut self,
        listing_id: &str,
        reviewer: &Identity,
        stars: u8,
        text: &str,
        receipt: &Receipt,
    ) -> Result<Review, KashError> {
        let listing = self
            .listings
            .get(listing_id)
            .ok_or_else(|| KashError::ListingNotFound(listing_id.to_string()))?
            .clone();
        if !receipt.verify() {
            return Err(KashError::BadReceipt);
        }
        let reviewer_id = reviewer.peer_id();
        if &receipt.payer != reviewer_id {
            return Err(KashError::NoPurchaseReceipt(format!(
                "receipt for listing '{listing_id}' was paid by someone else"
            )));
        }
        // An owner-free receipt carries amount 0 (nothing
        // moved) and is still proof of a completed purchase:
        // the purchase record below is what proves the buyer
        // bought it. A normal receipt must match the price.
        if !receipt.owner_free && receipt.amount != listing.price {
            return Err(KashError::NoPurchaseReceipt(format!(
                "receipt for listing '{listing_id}' is for the wrong amount"
            )));
        }
        let direct_ok =
            receipt.invoice_id == listing.id && self.buyers.get(listing_id) == Some(reviewer_id);
        let escrow_ok = match self.escrows.get(listing_id) {
            Some(e) => {
                e.state == EscrowState::Released
                    && e.escrow_invoice_id == receipt.invoice_id
                    && &e.buyer == reviewer_id
            }
            None => false,
        };
        if !direct_ok && !escrow_ok {
            return Err(KashError::NoPurchaseReceipt(format!(
                "no completed purchase of listing '{listing_id}' by this buyer"
            )));
        }
        let key = (listing_id.to_string(), *reviewer_id.public_key().as_bytes());
        if self.reviewed.contains(&key) {
            return Err(KashError::DuplicateReview(listing_id.to_string()));
        }
        let review = Review::new(reviewer, listing_id, stars, text)?;
        if !review.verify() {
            return Err(KashError::BadSignature);
        }
        self.reviewed.insert(key);
        self.reviews.push(review.clone());
        Ok(review)
    }

    /// Reviews for one listing, in submission order.
    pub fn reviews_for_listing(&self, listing_id: &str) -> Vec<&Review> {
        self.reviews
            .iter()
            .filter(|r| r.listing_id == listing_id)
            .collect()
    }

    /// A seller's rating: (average stars, review count) over
    /// all reviews of listings they sold. (0.0, 0) when the
    /// seller has no reviews.
    pub fn seller_rating(&self, seller: &PeerId) -> (f64, usize) {
        let mut total: u64 = 0;
        let mut count: usize = 0;
        for review in &self.reviews {
            if let Some(listing) = self.listings.get(&review.listing_id) {
                if &listing.seller == seller {
                    total += review.stars as u64;
                    count += 1;
                }
            }
        }
        if count == 0 {
            (0.0, 0)
        } else {
            (total as f64 / count as f64, count)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use k_pay::PayError;
    use k_pay::PaymentStatus;

    const NOW: u64 = 1_000;

    fn identities() -> (Identity, Identity, Identity) {
        (
            Identity::generate("alice").unwrap(),
            Identity::generate("bob").unwrap(),
            Identity::generate("carol").unwrap(),
        )
    }

    fn funded() -> (Store, Identity, Identity, Identity) {
        let (alice, bob, carol) = identities();
        let store = Store::new(&[
            (alice.peer_id().clone(), 1_000),
            (bob.peer_id().clone(), 100),
            (carol.peer_id().clone(), 50),
        ])
        .unwrap();
        (store, alice, bob, carol)
    }

    fn lamp(store: &mut Store, seller: &Identity) -> Listing {
        store
            .create_listing(
                seller,
                "listing-lamp-001",
                "brass lamp",
                "a sturdy brass lamp, works",
                250,
                "home",
            )
            .unwrap()
    }

    #[test]
    fn create_and_list_works() {
        let (mut store, alice, bob, _) = funded();
        lamp(&mut store, &bob);
        store
            .create_listing(
                &alice,
                "listing-book-001",
                "old book",
                "a worn paperback",
                40,
                "books",
            )
            .unwrap();
        let open = store.list_open();
        assert_eq!(open.len(), 2);
        // Sorted by id.
        assert_eq!(open[0].id, "listing-book-001");
        assert_eq!(open[0].name, "old book");
        assert_eq!(open[0].category, "books");
        assert_eq!(open[0].seller.display_name(), "alice");
        assert_eq!(open[1].id, "listing-lamp-001");
        assert_eq!(open[1].name, "brass lamp");
        assert_eq!(open[1].category, "home");
        assert_eq!(open[1].price, 250);
        assert!(open[1].verify());
        // get_listing finds both.
        assert_eq!(
            store.get_listing("listing-lamp-001").unwrap().name,
            "brass lamp"
        );
        assert!(store.get_listing("listing-nope-001").is_none());
        assert!(!store.is_sold("listing-lamp-001"));
    }

    #[test]
    fn buying_moves_payment_and_marks_sold() {
        let (mut store, alice, bob, _) = funded();
        lamp(&mut store, &bob);
        // Alice buys Bob's lamp for 250.
        let receipt = store.buy_listing("listing-lamp-001", &alice, NOW).unwrap();
        assert!(receipt.verify());
        assert_eq!(receipt.invoice_id, "listing-lamp-001");
        assert_eq!(receipt.payer, *alice.peer_id());
        assert_eq!(receipt.recipient, *bob.peer_id());
        assert_eq!(receipt.amount, 250);
        assert!(store.is_sold("listing-lamp-001"));
        assert_eq!(store.balance_of(alice.peer_id()), 750);
        assert_eq!(store.balance_of(bob.peer_id()), 350);
        // The lamp is no longer listed as open, but the record
        // stays retrievable.
        assert!(store.list_open().is_empty());
        assert!(store.get_listing("listing-lamp-001").is_some());
        // The payment is on the ledger as a memoed transaction.
        assert_eq!(store.ledger().transactions().len(), 1);
        assert_eq!(
            store.ledger().transactions()[0].memo,
            "pay:listing-lamp-001"
        );
        assert_eq!(store.ledger().transactions()[0].amount, 250);
    }

    #[test]
    fn buying_an_already_sold_listing_fails() {
        let (mut store, alice, bob, carol) = funded();
        lamp(&mut store, &bob);
        store.buy_listing("listing-lamp-001", &alice, NOW).unwrap();
        // Carol tries to buy the same lamp: refused.
        assert_eq!(
            store.buy_listing("listing-lamp-001", &carol, NOW),
            Err(KashError::AlreadySold("listing-lamp-001".to_string()))
        );
        // Nothing moved for the refused purchase.
        assert_eq!(store.balance_of(carol.peer_id()), 50);
        assert_eq!(store.balance_of(bob.peer_id()), 350);
        // Buying a listing that was never listed is a clean error.
        assert_eq!(
            store.buy_listing("listing-ghost-001", &carol, NOW),
            Err(KashError::ListingNotFound("listing-ghost-001".to_string()))
        );
    }

    #[test]
    fn buying_with_insufficient_funds_fails() {
        let (mut store, alice, bob, carol) = funded();
        lamp(&mut store, &bob);
        // Carol holds 50; the lamp costs 250. The failure now
        // surfaces through K•Pay.
        let err = store
            .buy_listing("listing-lamp-001", &carol, NOW)
            .unwrap_err();
        assert!(matches!(
            err,
            KashError::Pay(PayError::Koin(KoinError::InsufficientFunds { .. }))
        ));
        // The lamp is still for sale and nobody's balance moved.
        assert!(!store.is_sold("listing-lamp-001"));
        assert_eq!(store.list_open().len(), 1);
        assert_eq!(store.balance_of(carol.peer_id()), 50);
        assert_eq!(store.balance_of(bob.peer_id()), 100);
        // Alice, who can afford it, buys it afterwards.
        store.buy_listing("listing-lamp-001", &alice, NOW).unwrap();
        assert!(store.is_sold("listing-lamp-001"));
    }

    #[test]
    fn listing_with_bad_signature_is_rejected() {
        let (mut store, alice, bob, _) = funded();
        let listing = lamp(&mut store, &bob);
        // Change the price after the seller signed it.
        let mut forged = listing.clone();
        forged.price = 1;
        assert!(!forged.verify());
        // Swap the forged listing in place of the real one: the
        // store must refuse the purchase on the bad signature.
        store.listings.insert(forged.id.clone(), forged);
        assert_eq!(
            store.buy_listing("listing-lamp-001", &alice, NOW),
            Err(KashError::BadSignature)
        );
        assert!(!store.is_sold("listing-lamp-001"));
        assert_eq!(store.balance_of(alice.peer_id()), 1_000);
    }

    #[test]
    fn listing_rules() {
        let (mut store, alice, bob, _) = funded();
        // Duplicate ids are refused.
        lamp(&mut store, &bob);
        assert_eq!(
            store.create_listing(&bob, "listing-lamp-001", "other", "other", 10, "home"),
            Err(KashError::DuplicateListing("listing-lamp-001".to_string()))
        );
        // Bad ids, empty names, empty descriptions, and zero
        // prices are refused.
        assert!(store
            .create_listing(&alice, "", "x", "y", 10, "home")
            .is_err());
        assert!(store
            .create_listing(&alice, "has space", "x", "y", 10, "home")
            .is_err());
        assert!(store
            .create_listing(&alice, "listing-a", "", "y", 10, "home")
            .is_err());
        assert!(store
            .create_listing(&alice, "listing-a", "x", "", 10, "home")
            .is_err());
        assert!(store
            .create_listing(&alice, "listing-a", "x", "y", 0, "home")
            .is_err());
        // Bad categories are refused, and a mixed-case category
        // is normalized to lowercase.
        assert!(store
            .create_listing(&alice, "listing-a", "x", "y", 10, "")
            .is_err());
        assert!(store
            .create_listing(&alice, "listing-a", "x", "y", 10, "has space")
            .is_err());
        assert!(store
            .create_listing(&alice, "listing-a", "x", "y", 10, "a".repeat(33).as_str())
            .is_err());
        let listing = store
            .create_listing(&alice, "listing-a", "x", "y", 10, "Music")
            .unwrap();
        assert_eq!(listing.category, "music");
        assert!(store
            .create_listing(&alice, "listing-b", "x", "y", 10, "home")
            .is_ok());
    }

    #[test]
    fn categories_browse() {
        let (mut store, alice, bob, _) = funded();
        store
            .create_listing(
                &bob,
                "listing-track-001",
                "lofi track",
                "a chill beat",
                30,
                "music",
            )
            .unwrap();
        store
            .create_listing(
                &bob,
                "listing-paint-001",
                "oil painting",
                "a small canvas",
                90,
                "art",
            )
            .unwrap();
        store
            .create_listing(
                &alice,
                "listing-track-002",
                "drum loop",
                "a punchy loop",
                20,
                "music",
            )
            .unwrap();
        store
            .create_listing(
                &alice,
                "listing-cable-001",
                "usb cable",
                "a long cable",
                5,
                "gear",
            )
            .unwrap();
        let music = store.listings_by_category("music");
        assert_eq!(music.len(), 2);
        // Listing (id) order.
        assert_eq!(music[0].id, "listing-track-001");
        assert_eq!(music[1].id, "listing-track-002");
        let art = store.listings_by_category("art");
        assert_eq!(art.len(), 1);
        assert_eq!(art[0].id, "listing-paint-001");
        // A mixed-case query is normalized; an unknown or
        // invalid category is a clean empty list.
        assert_eq!(store.listings_by_category("MUSIC").len(), 2);
        assert!(store.listings_by_category("nope").is_empty());
        assert!(store.listings_by_category("").is_empty());
        // A sold listing leaves its category browse.
        store.buy_listing("listing-track-001", &alice, NOW).unwrap();
        let music = store.listings_by_category("music");
        assert_eq!(music.len(), 1);
        assert_eq!(music[0].id, "listing-track-002");
    }

    #[test]
    fn search_finds_by_name_description_and_scope() {
        let (mut store, alice, bob, _) = funded();
        lamp(&mut store, &bob);
        store
            .create_listing(
                &alice,
                "listing-book-001",
                "old book",
                "a worn paperback about brass",
                40,
                "books",
            )
            .unwrap();
        // By name, case-insensitive.
        let found = store.search("BRASS", None, false);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].id, "listing-book-001");
        assert_eq!(found[1].id, "listing-lamp-001");
        // By description only.
        let found = store.search("sturdy", None, false);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, "listing-lamp-001");
        // Category scope.
        let found = store.search("brass", Some("books"), false);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, "listing-book-001");
        let found = store.search("brass", Some("music"), false);
        assert!(found.is_empty());
        // No match at all.
        assert!(store.search("zeppelin", None, false).is_empty());
        // Sold listings are hidden by default...
        store.buy_listing("listing-lamp-001", &alice, NOW).unwrap();
        let found = store.search("brass", None, false);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, "listing-book-001");
        // ...unless the caller asks for them.
        let found = store.search("brass", None, true);
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn reviews_require_a_completed_purchase() {
        let (mut store, alice, bob, carol) = funded();
        lamp(&mut store, &bob);
        // Carol never bought the lamp: a review from her is
        // refused, even with a receipt she signed herself for
        // the listing's invoice id — the store's purchase
        // record says Alice bought it, not Carol.
        let forged = Receipt {
            invoice_id: "listing-lamp-001".to_string(),
            payer: carol.peer_id().clone(),
            recipient: bob.peer_id().clone(),
            amount: 250,
            owner_free: false,
            signature: carol.sign(&k_pay_receipt_message(
                "listing-lamp-001",
                carol.peer_id(),
                bob.peer_id(),
                250,
            )),
        };
        assert!(forged.verify());
        assert!(matches!(
            store.submit_review("listing-lamp-001", &carol, 5, "great lamp", &forged),
            Err(KashError::NoPurchaseReceipt(_))
        ));
        // Alice buys, then reviews: accepted.
        let receipt = store.buy_listing("listing-lamp-001", &alice, NOW).unwrap();
        let review = store
            .submit_review("listing-lamp-001", &alice, 5, "great lamp", &receipt)
            .unwrap();
        assert!(review.verify());
        assert_eq!(review.stars, 5);
        assert_eq!(store.reviews_for_listing("listing-lamp-001").len(), 1);
        // One review per purchase: a second review is refused.
        assert_eq!(
            store.submit_review("listing-lamp-001", &alice, 4, "still great", &receipt),
            Err(KashError::DuplicateReview("listing-lamp-001".to_string()))
        );
    }

    #[test]
    fn review_rules() {
        let (mut store, alice, bob, _) = funded();
        lamp(&mut store, &bob);
        let receipt = store.buy_listing("listing-lamp-001", &alice, NOW).unwrap();
        // Stars outside 1-5 are refused.
        assert!(matches!(
            store.submit_review("listing-lamp-001", &alice, 0, "bad", &receipt),
            Err(KashError::InvalidReview(_))
        ));
        assert!(matches!(
            store.submit_review("listing-lamp-001", &alice, 6, "bad", &receipt),
            Err(KashError::InvalidReview(_))
        ));
        // Empty review text is refused.
        assert!(matches!(
            store.submit_review("listing-lamp-001", &alice, 5, "", &receipt),
            Err(KashError::InvalidReview(_))
        ));
        // A receipt for a different listing does not prove this
        // purchase.
        store
            .create_listing(
                &bob,
                "listing-book-001",
                "old book",
                "a worn book",
                40,
                "books",
            )
            .unwrap();
        assert!(matches!(
            store.submit_review("listing-book-001", &alice, 5, "great", &receipt),
            Err(KashError::NoPurchaseReceipt(_))
        ));
        // A tampered receipt is refused outright.
        let mut bad = receipt.clone();
        bad.amount = 1;
        assert!(!bad.verify());
        assert_eq!(
            store.submit_review("listing-lamp-001", &alice, 5, "great", &bad),
            Err(KashError::BadReceipt)
        );
    }

    #[test]
    fn seller_ratings_average() {
        let (mut store, alice, bob, carol) = funded();
        lamp(&mut store, &bob);
        store
            .create_listing(
                &bob,
                "listing-book-001",
                "old book",
                "a worn book",
                40,
                "books",
            )
            .unwrap();
        // Alice reviews the lamp 5 stars; Carol reviews the
        // book 3 stars. Bob's average is 4.0 over 2 reviews.
        let lamp_receipt = store.buy_listing("listing-lamp-001", &alice, NOW).unwrap();
        store
            .submit_review("listing-lamp-001", &alice, 5, "great lamp", &lamp_receipt)
            .unwrap();
        let book_receipt = store.buy_listing("listing-book-001", &carol, NOW).unwrap();
        store
            .submit_review("listing-book-001", &carol, 3, "ok book", &book_receipt)
            .unwrap();
        assert_eq!(store.seller_rating(bob.peer_id()), (4.0, 2));
        // A seller with no reviews rates zero.
        assert_eq!(store.seller_rating(alice.peer_id()), (0.0, 0));
    }

    #[test]
    fn escrow_full_flow_releases_to_seller() {
        let (mut store, alice, bob, _) = funded();
        lamp(&mut store, &bob);
        let escrow_peer = store.escrow_peer_id().clone();
        // Alice pays into escrow: her coins leave, the escrow
        // holds them, the seller gets nothing yet.
        let receipt = store
            .buy_listing_escrow("listing-lamp-001", &alice, NOW)
            .unwrap();
        assert!(receipt.verify());
        assert_eq!(receipt.recipient, escrow_peer);
        assert_eq!(receipt.amount, 250);
        assert_eq!(store.balance_of(alice.peer_id()), 750);
        assert_eq!(store.balance_of(&escrow_peer), 250);
        assert_eq!(store.balance_of(bob.peer_id()), 100);
        assert!(store.is_sold("listing-lamp-001"));
        assert_eq!(
            store.escrow_state("listing-lamp-001"),
            Some(EscrowState::Held)
        );
        // Confirming before delivery is refused.
        assert!(matches!(
            store.confirm_receipt("listing-lamp-001", &alice, NOW),
            Err(KashError::WrongEscrowState(_))
        ));
        // Only the seller can mark delivered.
        assert_eq!(
            store.mark_delivered("listing-lamp-001", &alice),
            Err(KashError::NotSeller("listing-lamp-001".to_string()))
        );
        store.mark_delivered("listing-lamp-001", &bob).unwrap();
        assert_eq!(
            store.escrow_state("listing-lamp-001"),
            Some(EscrowState::Delivered)
        );
        // Only the buyer can confirm.
        assert_eq!(
            store.confirm_receipt("listing-lamp-001", &bob, NOW),
            Err(KashError::NotBuyer("listing-lamp-001".to_string()))
        );
        // Alice confirms: the escrow pays Bob, the escrow
        // account empties, the escrow is released.
        let release = store
            .confirm_receipt("listing-lamp-001", &alice, NOW)
            .unwrap();
        assert!(release.verify());
        assert_eq!(release.payer, escrow_peer);
        assert_eq!(release.recipient, *bob.peer_id());
        assert_eq!(store.balance_of(alice.peer_id()), 750);
        assert_eq!(store.balance_of(&escrow_peer), 0);
        assert_eq!(store.balance_of(bob.peer_id()), 350);
        assert_eq!(
            store.escrow_state("listing-lamp-001"),
            Some(EscrowState::Released)
        );
        // Terminal: confirming or disputing again is refused.
        assert!(matches!(
            store.confirm_receipt("listing-lamp-001", &alice, NOW),
            Err(KashError::WrongEscrowState(_))
        ));
        assert!(matches!(
            store.dispute_escrow("listing-lamp-001", &alice, NOW),
            Err(KashError::WrongEscrowState(_))
        ));
        // The buyer can now review with the escrow receipt.
        let review = store
            .submit_review("listing-lamp-001", &alice, 5, "great lamp", &receipt)
            .unwrap();
        assert!(review.verify());
    }

    #[test]
    fn escrow_dispute_refunds_buyer_and_relists() {
        let (mut store, alice, bob, carol) = funded();
        lamp(&mut store, &bob);
        store
            .buy_listing_escrow("listing-lamp-001", &alice, NOW)
            .unwrap();
        assert_eq!(store.balance_of(alice.peer_id()), 750);
        // Only the buyer can dispute.
        assert_eq!(
            store.dispute_escrow("listing-lamp-001", &bob, NOW),
            Err(KashError::NotBuyer("listing-lamp-001".to_string()))
        );
        let record = store
            .dispute_escrow("listing-lamp-001", &alice, NOW)
            .unwrap();
        assert_eq!(record.status, PaymentStatus::Refunded);
        assert_eq!(record.amount, 250);
        // Buyer whole again, escrow empty, seller got nothing.
        assert_eq!(store.balance_of(alice.peer_id()), 1_000);
        assert_eq!(store.balance_of(store.escrow_peer_id()), 0);
        assert_eq!(store.balance_of(bob.peer_id()), 100);
        assert_eq!(
            store.escrow_state("listing-lamp-001"),
            Some(EscrowState::Refunded)
        );
        // The listing goes back on sale and can be bought again
        // — this time through escrow to a full release.
        assert!(!store.is_sold("listing-lamp-001"));
        assert_eq!(store.list_open().len(), 1);
        store
            .buy_listing_escrow("listing-lamp-001", &alice, NOW)
            .unwrap();
        assert_eq!(
            store.escrow_state("listing-lamp-001"),
            Some(EscrowState::Held)
        );
        store.mark_delivered("listing-lamp-001", &bob).unwrap();
        store
            .confirm_receipt("listing-lamp-001", &alice, NOW)
            .unwrap();
        assert_eq!(store.balance_of(bob.peer_id()), 350);
        // Disputing a refunded escrow is refused — but this
        // escrow is Released now, so it is refused as terminal.
        assert!(matches!(
            store.dispute_escrow("listing-lamp-001", &alice, NOW),
            Err(KashError::WrongEscrowState(_))
        ));
        // Carol cannot buy the sold lamp through escrow either.
        assert_eq!(
            store.buy_listing_escrow("listing-lamp-001", &carol, NOW),
            Err(KashError::AlreadySold("listing-lamp-001".to_string()))
        );
    }

    #[test]
    fn escrow_dispute_after_delivery_refunds() {
        let (mut store, alice, bob, _) = funded();
        lamp(&mut store, &bob);
        store
            .buy_listing_escrow("listing-lamp-001", &alice, NOW)
            .unwrap();
        store.mark_delivered("listing-lamp-001", &bob).unwrap();
        // The buyer disputes instead of confirming.
        store
            .dispute_escrow("listing-lamp-001", &alice, NOW)
            .unwrap();
        assert_eq!(store.balance_of(alice.peer_id()), 1_000);
        assert_eq!(store.balance_of(bob.peer_id()), 100);
        assert_eq!(
            store.escrow_state("listing-lamp-001"),
            Some(EscrowState::Refunded)
        );
        assert!(!store.is_sold("listing-lamp-001"));
    }

    #[test]
    fn escrow_insufficient_funds_changes_nothing() {
        let (mut store, alice, bob, carol) = funded();
        lamp(&mut store, &bob);
        // Carol holds 50; the lamp costs 250.
        let err = store
            .buy_listing_escrow("listing-lamp-001", &carol, NOW)
            .unwrap_err();
        assert!(matches!(
            err,
            KashError::Pay(PayError::Koin(KoinError::InsufficientFunds { .. }))
        ));
        assert!(!store.is_sold("listing-lamp-001"));
        assert_eq!(store.escrow_state("listing-lamp-001"), None);
        assert_eq!(store.balance_of(carol.peer_id()), 50);
        assert_eq!(store.balance_of(bob.peer_id()), 100);
        // Alice, who can afford it, buys through escrow after.
        store
            .buy_listing_escrow("listing-lamp-001", &alice, NOW)
            .unwrap();
        assert_eq!(
            store.escrow_state("listing-lamp-001"),
            Some(EscrowState::Held)
        );
    }

    #[test]
    fn escrow_with_max_length_listing_id() {
        let (mut store, alice, bob, _) = funded();
        // A 64-character listing id: the derived escrow,
        // release, and refund invoice ids would overflow
        // k-pay's 64-character limit, so the store must fall
        // back to hashed ids — the whole flow still works.
        let long_id = "listing-".repeat(8); // 64 chars
        assert_eq!(long_id.len(), 64);
        store
            .create_listing(
                &bob,
                &long_id,
                "long lamp",
                "a lamp with a long id",
                250,
                "home",
            )
            .unwrap();
        store.buy_listing_escrow(&long_id, &alice, NOW).unwrap();
        store.mark_delivered(&long_id, &bob).unwrap();
        store.confirm_receipt(&long_id, &alice, NOW).unwrap();
        assert_eq!(store.balance_of(bob.peer_id()), 350);
        assert_eq!(store.escrow_state(&long_id), Some(EscrowState::Released));
    }

    #[test]
    fn review_before_escrow_release_is_refused() {
        let (mut store, alice, bob, _) = funded();
        lamp(&mut store, &bob);
        let receipt = store
            .buy_listing_escrow("listing-lamp-001", &alice, NOW)
            .unwrap();
        // The purchase is paid but not completed: the seller
        // has not been paid yet, so no review.
        assert!(matches!(
            store.submit_review("listing-lamp-001", &alice, 5, "great", &receipt),
            Err(KashError::NoPurchaseReceipt(_))
        ));
        // After release the same receipt proves the purchase.
        store.mark_delivered("listing-lamp-001", &bob).unwrap();
        store
            .confirm_receipt("listing-lamp-001", &alice, NOW)
            .unwrap();
        store
            .submit_review("listing-lamp-001", &alice, 5, "great", &receipt)
            .unwrap();
    }

    /// Helper: rebuild the exact message a payer signs for a
    /// receipt, so tests can fabricate a receipt the buyer
    /// signed herself. Mirrors k-pay's receipt signing,
    /// including the owner-free flag byte (0 here: the forged
    /// receipt is a normal paid one).
    fn k_pay_receipt_message(
        invoice_id: &str,
        payer: &PeerId,
        recipient: &PeerId,
        amount: u64,
    ) -> Vec<u8> {
        let mut msg = Vec::new();
        msg.extend_from_slice(b"KPAY-RCPT1");
        msg.extend_from_slice(&(invoice_id.len() as u16).to_le_bytes());
        msg.extend_from_slice(invoice_id.as_bytes());
        msg.extend_from_slice(payer.public_key().as_bytes());
        msg.extend_from_slice(recipient.public_key().as_bytes());
        msg.extend_from_slice(&amount.to_le_bytes());
        msg.push(0);
        msg
    }

    // -- OWNER-FREE LAW ------------------------------------------

    /// A fixture with the owner registered: Alice is the owner.
    fn funded_with_owner() -> (Store, Identity, Identity, Identity) {
        let (mut store, alice, bob, carol) = funded();
        store.set_owner(alice.peer_id().clone());
        (store, alice, bob, carol)
    }

    #[test]
    fn owner_buys_direct_free_and_can_review() {
        let (mut store, alice, bob, _) = funded_with_owner();
        lamp(&mut store, &bob);
        let receipt = store.buy_listing("listing-lamp-001", &alice, NOW).unwrap();
        // A valid owner-free receipt, and the listing is sold.
        assert!(receipt.verify());
        assert!(receipt.owner_free);
        assert_eq!(receipt.amount, 0);
        assert!(store.is_sold("listing-lamp-001"));
        // No coins moved for anyone.
        assert_eq!(store.balance_of(alice.peer_id()), 1_000);
        assert_eq!(store.balance_of(bob.peer_id()), 100);
        assert!(store.ledger().transactions().is_empty());
        // The owner can still review the purchase: the receipt
        // proves the buy even though nothing moved.
        let review = store
            .submit_review("listing-lamp-001", &alice, 5, "great lamp", &receipt)
            .unwrap();
        assert!(review.verify());
        assert_eq!(store.seller_rating(bob.peer_id()), (5.0, 1));
    }

    #[test]
    fn owner_escrow_buy_confirm_releases_nothing() {
        let (mut store, alice, bob, _) = funded_with_owner();
        lamp(&mut store, &bob);
        let escrow_peer = store.escrow_peer_id().clone();
        // The owner pays into escrow: nothing is held.
        let receipt = store
            .buy_listing_escrow("listing-lamp-001", &alice, NOW)
            .unwrap();
        assert!(receipt.verify());
        assert!(receipt.owner_free);
        assert_eq!(receipt.amount, 0);
        assert_eq!(receipt.recipient, escrow_peer);
        assert!(store.is_sold("listing-lamp-001"));
        assert_eq!(
            store.escrow_state("listing-lamp-001"),
            Some(EscrowState::Held)
        );
        assert_eq!(store.balance_of(alice.peer_id()), 1_000);
        assert_eq!(store.balance_of(&escrow_peer), 0);
        assert_eq!(store.balance_of(bob.peer_id()), 100);
        // Delivery and confirmation move nothing either.
        store.mark_delivered("listing-lamp-001", &bob).unwrap();
        let release = store
            .confirm_receipt("listing-lamp-001", &alice, NOW)
            .unwrap();
        assert!(release.verify());
        assert!(release.owner_free);
        assert_eq!(release.amount, 0);
        assert_eq!(
            store.escrow_state("listing-lamp-001"),
            Some(EscrowState::Released)
        );
        assert_eq!(store.balance_of(alice.peer_id()), 1_000);
        assert_eq!(store.balance_of(&escrow_peer), 0);
        assert_eq!(store.balance_of(bob.peer_id()), 100);
        // The released escrow purchase still proves the buy for
        // a review.
        store
            .submit_review("listing-lamp-001", &alice, 5, "great", &receipt)
            .unwrap();
    }

    #[test]
    fn owner_escrow_buy_dispute_refunds_nothing_and_relists() {
        let (mut store, alice, bob, _) = funded_with_owner();
        lamp(&mut store, &bob);
        let escrow_peer = store.escrow_peer_id().clone();
        store
            .buy_listing_escrow("listing-lamp-001", &alice, NOW)
            .unwrap();
        assert_eq!(store.balance_of(alice.peer_id()), 1_000);
        assert_eq!(store.balance_of(&escrow_peer), 0);
        // Dispute: the escrow is refunded, the listing goes back
        // on sale, and no coins move anywhere.
        let record = store
            .dispute_escrow("listing-lamp-001", &alice, NOW)
            .unwrap();
        assert_eq!(record.status, PaymentStatus::Refunded);
        assert_eq!(record.amount, 0);
        assert_eq!(
            store.escrow_state("listing-lamp-001"),
            Some(EscrowState::Refunded)
        );
        assert!(!store.is_sold("listing-lamp-001"));
        assert_eq!(store.list_open().len(), 1);
        assert_eq!(store.balance_of(alice.peer_id()), 1_000);
        assert_eq!(store.balance_of(&escrow_peer), 0);
        assert_eq!(store.balance_of(bob.peer_id()), 100);
        // The relisted lamp can be bought again through escrow.
        store
            .buy_listing_escrow("listing-lamp-001", &alice, NOW)
            .unwrap();
        assert!(store.is_sold("listing-lamp-001"));
        assert_eq!(store.balance_of(alice.peer_id()), 1_000);
    }
}
