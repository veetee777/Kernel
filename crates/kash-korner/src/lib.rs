//! Kash•Korner — a simple store / marketplace settled on X•Koin.
//!
//! A listing is a signed offer to sell: a human-readable id, a
//! seller, an item name, a description, and a price. The seller's
//! signature covers the whole listing, so a forged or tampered
//! listing is refused before it can be listed.
//!
//! Buying a listing settles through the coin ledger: the buyer
//! submits a signed coin transaction to the seller, and the
//! listing is marked sold. A sold listing cannot be bought again,
//! and a buyer who does not hold enough is refused before any
//! coins move.
//!
//! Listings are identified by human-readable ids
//! (`listing-lamp-001`) and item names (`"brass lamp"`), never
//! hashes. The store holds no inventory itself — it is a record
//! of offers and sales, and settlement happens on the ledger.

use std::collections::{HashMap, HashSet};
use std::fmt;

use kernel_crypto::Ed25519Signature;
use kernel_identity::{Identity, IdentityError, PeerId};
use x_koin::{CoinTransaction, KoinError, Ledger};

/// Domain separator for listing signatures: a signature covers
/// `b"KASH-LIST1" || id || seller key || name || description ||
/// price`.
const LISTING_SIGN_DOMAIN: &[u8] = b"KASH-LIST1";

/// Errors returned by store operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KashError {
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// Wraps a [`KoinError`] from settlement.
    Koin(KoinError),
    /// A listing id, name, description, or price broke the rules.
    InvalidListing(String),
    /// A listing's signature did not verify against its seller.
    BadSignature,
    /// A listing with this id already exists.
    DuplicateListing(String),
    /// No listing has this id.
    ListingNotFound(String),
    /// The listing was already sold.
    AlreadySold(String),
}

impl fmt::Display for KashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KashError::Identity(e) => write!(f, "identity error: {e}"),
            KashError::Koin(e) => write!(f, "coin error: {e}"),
            KashError::InvalidListing(msg) => write!(f, "invalid listing: {msg}"),
            KashError::BadSignature => write!(f, "listing signature is invalid"),
            KashError::DuplicateListing(id) => {
                write!(f, "a listing with id '{id}' already exists")
            }
            KashError::ListingNotFound(id) => write!(f, "no listing with id '{id}'"),
            KashError::AlreadySold(id) => write!(f, "listing '{id}' was already sold"),
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

/// The message a seller signs for a listing.
fn listing_sign_message(
    id: &str,
    seller: &PeerId,
    name: &str,
    description: &str,
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
    msg.extend_from_slice(&price.to_le_bytes());
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
    /// The asking price, in the coin's smallest unit.
    pub price: u64,
    /// The seller's signature over id, seller, name,
    /// description, and price.
    pub signature: Ed25519Signature,
}

impl Listing {
    /// Create and sign a listing as `seller`. The price must be
    /// greater than zero; the id, name, and description are
    /// validated.
    pub fn new(
        seller: &Identity,
        id: &str,
        name: &str,
        description: &str,
        price: u64,
    ) -> Result<Self, KashError> {
        validate_id(id)?;
        validate_text(name, "item name")?;
        validate_text(description, "description")?;
        if price == 0 {
            return Err(KashError::InvalidListing(
                "listing price must be greater than zero".to_string(),
            ));
        }
        let seller_id = seller.peer_id().clone();
        let msg = listing_sign_message(id, &seller_id, name, description, price);
        let signature = seller.sign(&msg);
        Ok(Self {
            id: id.to_string(),
            seller: seller_id,
            name: name.to_string(),
            description: description.to_string(),
            price,
            signature,
        })
    }

    /// True if the signature verifies against the seller for this
    /// exact id, name, description, and price. Any tampering
    /// fails this.
    pub fn verify(&self) -> bool {
        let msg = listing_sign_message(
            &self.id,
            &self.seller,
            &self.name,
            &self.description,
            self.price,
        );
        self.seller.verify(&msg, &self.signature).is_ok()
    }
}

/// A store: a coin ledger plus the listings offered against it.
/// Listings are created by sellers and can arrive from any peer;
/// only their signatures authenticate them, so a listing does
/// not need to have been created through this store to be bought
/// here — but a listing bought here is marked sold here.
pub struct Store {
    ledger: Ledger,
    /// Listings by id, including sold ones.
    listings: HashMap<String, Listing>,
    /// Ids of listings already sold.
    sold: HashSet<String>,
}

impl Store {
    /// Create a store whose ledger starts at the given genesis
    /// allocation.
    pub fn new(genesis: &[(PeerId, u64)]) -> Result<Self, KashError> {
        Ok(Self {
            ledger: Ledger::new(genesis)?,
            listings: HashMap::new(),
            sold: HashSet::new(),
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

    /// True if the listing with this id was already sold.
    pub fn is_sold(&self, id: &str) -> bool {
        self.sold.contains(id)
    }

    /// The listing with this id, if any — sold or not.
    pub fn get_listing(&self, id: &str) -> Option<&Listing> {
        self.listings.get(id)
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

    /// Create and register a listing as `seller`. The id must
    /// not already be registered.
    pub fn create_listing(
        &mut self,
        seller: &Identity,
        id: &str,
        name: &str,
        description: &str,
        price: u64,
    ) -> Result<Listing, KashError> {
        if self.listings.contains_key(id) {
            return Err(KashError::DuplicateListing(id.to_string()));
        }
        let listing = Listing::new(seller, id, name, description, price)?;
        self.listings.insert(id.to_string(), listing.clone());
        Ok(listing)
    }

    /// Buy a listing as `buyer`. Validates, in order: the listing
    /// exists, the listing's signature verifies against its
    /// seller, and it has not been sold; then settles through
    /// the coin ledger with a signed transaction from the buyer
    /// to the seller, and marks the listing sold. A rejected
    /// purchase changes nothing — no coins move and the listing
    /// stays for sale.
    pub fn buy_listing(&mut self, id: &str, buyer: &Identity) -> Result<(), KashError> {
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
        let memo = format!("kash:{}", listing.id);
        let tx = CoinTransaction::new(buyer, &listing.seller, &memo, listing.price)?;
        self.ledger.submit(&tx)?;
        self.sold.insert(id.to_string());
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
            )
            .unwrap();
        let open = store.list_open();
        assert_eq!(open.len(), 2);
        // Sorted by id.
        assert_eq!(open[0].id, "listing-book-001");
        assert_eq!(open[0].name, "old book");
        assert_eq!(open[0].seller.display_name(), "alice");
        assert_eq!(open[1].id, "listing-lamp-001");
        assert_eq!(open[1].name, "brass lamp");
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
        store.buy_listing("listing-lamp-001", &alice).unwrap();
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
            "kash:listing-lamp-001"
        );
        assert_eq!(store.ledger().transactions()[0].amount, 250);
    }

    #[test]
    fn buying_an_already_sold_listing_fails() {
        let (mut store, alice, bob, carol) = funded();
        lamp(&mut store, &bob);
        store.buy_listing("listing-lamp-001", &alice).unwrap();
        // Carol tries to buy the same lamp: refused.
        assert_eq!(
            store.buy_listing("listing-lamp-001", &carol),
            Err(KashError::AlreadySold("listing-lamp-001".to_string()))
        );
        // Nothing moved for the refused purchase.
        assert_eq!(store.balance_of(carol.peer_id()), 50);
        assert_eq!(store.balance_of(bob.peer_id()), 350);
        // Buying a listing that was never listed is a clean error.
        assert_eq!(
            store.buy_listing("listing-ghost-001", &carol),
            Err(KashError::ListingNotFound("listing-ghost-001".to_string()))
        );
    }

    #[test]
    fn buying_with_insufficient_funds_fails() {
        let (mut store, alice, bob, carol) = funded();
        lamp(&mut store, &bob);
        // Carol holds 50; the lamp costs 250.
        let err = store.buy_listing("listing-lamp-001", &carol).unwrap_err();
        assert!(matches!(err, KashError::Koin(KoinError::InsufficientFunds { .. })));
        // The lamp is still for sale and nobody's balance moved.
        assert!(!store.is_sold("listing-lamp-001"));
        assert_eq!(store.list_open().len(), 1);
        assert_eq!(store.balance_of(carol.peer_id()), 50);
        assert_eq!(store.balance_of(bob.peer_id()), 100);
        // Alice, who can afford it, buys it afterwards.
        store.buy_listing("listing-lamp-001", &alice).unwrap();
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
            store.buy_listing("listing-lamp-001", &alice),
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
            store.create_listing(&bob, "listing-lamp-001", "other", "other", 10),
            Err(KashError::DuplicateListing("listing-lamp-001".to_string()))
        );
        // Bad ids, empty names, empty descriptions, and zero
        // prices are refused.
        assert!(store
            .create_listing(&alice, "", "x", "y", 10)
            .is_err());
        assert!(store
            .create_listing(&alice, "has space", "x", "y", 10)
            .is_err());
        assert!(store
            .create_listing(&alice, "listing-a", "", "y", 10)
            .is_err());
        assert!(store
            .create_listing(&alice, "listing-a", "x", "", 10)
            .is_err());
        assert!(store
            .create_listing(&alice, "listing-a", "x", "y", 0)
            .is_err());
        assert!(store
            .create_listing(&alice, "listing-a", "x", "y", 10)
            .is_ok());
    }
}
