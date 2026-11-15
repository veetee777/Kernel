//! Kernel•XChange — an exchange order book.
//!
//! An order is a signed intent to trade: an owner, a side (buy or
//! sell), a price per unit, and an amount. The owner's signature
//! covers the whole order, so a forged or tampered order is
//! refused before it can enter the book.
//!
//! Matching is deterministic and simple:
//!
//! * A buy matches a sell when the buy price is at least the sell
//!   price. The trade executes at the sell price — the buyer
//!   never pays more than they offered, and the seller always
//!   gets exactly what they asked.
//! * When several sells could match a buy, the lowest sell price
//!   wins; ties go to the order placed first. When several buys
//!   could match a sell, the highest buy price wins; ties go to
//!   the order placed first.
//! * Fills can be partial: a trade moves the smaller of the two
//!   remaining amounts, and whatever is left of either order stays
//!   open in the book.
//! * Only the owner can cancel an order. Cancelling removes it
//!   from the book; it can never match afterwards.
//!
//! Orders are identified by human-readable ids (`ord-001`), never
//! hashes. This crate moves no coins itself: it records trades,
//! and settlement happens on the coin ledger by the caller.

use std::collections::HashMap;
use std::fmt;

use kernel_crypto::Ed25519Signature;
use kernel_identity::{Identity, IdentityError, PeerId};

/// Domain separator for order signatures: a signature covers
/// `b"XCHG-ORD1" || id || owner key || side || price || amount`.
const ORDER_SIGN_DOMAIN: &[u8] = b"XCHG-ORD1";

/// Errors returned by exchange operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum XChangeError {
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// An order id, price, or amount broke the rules.
    InvalidOrder(String),
    /// An order's signature did not verify against its owner.
    BadSignature,
    /// An order with this id is already open.
    DuplicateOrder(String),
    /// No open order has this id.
    OrderNotFound(String),
    /// The requester does not own the order.
    NotOwner {
        /// The id of the order.
        id: String,
        /// The peer who tried to cancel it.
        requester: String,
    },
}

impl fmt::Display for XChangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            XChangeError::Identity(e) => write!(f, "identity error: {e}"),
            XChangeError::InvalidOrder(msg) => write!(f, "invalid order: {msg}"),
            XChangeError::BadSignature => write!(f, "order signature is invalid"),
            XChangeError::DuplicateOrder(id) => {
                write!(f, "an order with id '{id}' is already open")
            }
            XChangeError::OrderNotFound(id) => write!(f, "no open order with id '{id}'"),
            XChangeError::NotOwner { id, requester } => write!(
                f,
                "peer '{requester}' does not own order '{id}'"
            ),
        }
    }
}

impl std::error::Error for XChangeError {}

impl From<IdentityError> for XChangeError {
    fn from(e: IdentityError) -> Self {
        XChangeError::Identity(e)
    }
}

/// Rules for human-readable ids: 1-64 characters, ASCII letters,
/// digits, `-`, `_`, `.`. No spaces, no hashes.
fn validate_id(id: &str) -> Result<(), XChangeError> {
    if id.is_empty() {
        return Err(XChangeError::InvalidOrder(
            "order id must not be empty".to_string(),
        ));
    }
    if id.len() > 64 {
        return Err(XChangeError::InvalidOrder(
            "order id must be at most 64 characters".to_string(),
        ));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Err(XChangeError::InvalidOrder(
            "order id may only use letters, digits, '-', '_' and '.'".to_string(),
        ));
    }
    Ok(())
}

/// Which side of a trade an order is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// Wants to buy at up to `price` per unit.
    Buy,
    /// Wants to sell at no less than `price` per unit.
    Sell,
}

impl Side {
    /// The byte used for this side in the signed message.
    fn as_byte(self) -> u8 {
        match self {
            Side::Buy => 0,
            Side::Sell => 1,
        }
    }
}

/// The message an owner signs for an order.
fn order_sign_message(
    id: &str,
    owner: &PeerId,
    side: Side,
    price: u64,
    amount: u64,
) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend_from_slice(ORDER_SIGN_DOMAIN);
    msg.extend_from_slice(&(id.len() as u16).to_le_bytes());
    msg.extend_from_slice(id.as_bytes());
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg.push(side.as_byte());
    msg.extend_from_slice(&price.to_le_bytes());
    msg.extend_from_slice(&amount.to_le_bytes());
    msg
}

/// A signed order: an intent to buy or sell at a price.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Order {
    /// Human-readable id, e.g. `"ord-001"`.
    pub id: String,
    /// Who placed the order.
    pub owner: PeerId,
    /// Buy or sell.
    pub side: Side,
    /// Price per unit, in the coin's smallest unit.
    pub price: u64,
    /// How many units are offered or wanted.
    pub amount: u64,
    /// The owner's signature over id, owner, side, price, amount.
    pub signature: Ed25519Signature,
}

impl Order {
    /// Create and sign an order as `owner`. The price and amount
    /// must both be greater than zero.
    pub fn new(
        owner: &Identity,
        id: &str,
        side: Side,
        price: u64,
        amount: u64,
    ) -> Result<Self, XChangeError> {
        validate_id(id)?;
        if price == 0 {
            return Err(XChangeError::InvalidOrder(
                "order price must be greater than zero".to_string(),
            ));
        }
        if amount == 0 {
            return Err(XChangeError::InvalidOrder(
                "order amount must be greater than zero".to_string(),
            ));
        }
        let owner_id = owner.peer_id().clone();
        let msg = order_sign_message(id, &owner_id, side, price, amount);
        let signature = owner.sign(&msg);
        Ok(Self {
            id: id.to_string(),
            owner: owner_id,
            side,
            price,
            amount,
            signature,
        })
    }

    /// True if the signature verifies against the owner for this
    /// exact id, side, price, and amount. Any tampering fails this.
    pub fn verify(&self) -> bool {
        let msg = order_sign_message(&self.id, &self.owner, self.side, self.price, self.amount);
        self.owner.verify(&msg, &self.signature).is_ok()
    }
}

/// A completed trade: a buy and a sell that matched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trade {
    /// The id of the buy order.
    pub buy_order_id: String,
    /// The id of the sell order.
    pub sell_order_id: String,
    /// Who bought.
    pub buyer: PeerId,
    /// Who sold.
    pub seller: PeerId,
    /// The execution price: always the sell price.
    pub price: u64,
    /// How many units changed hands.
    pub amount: u64,
}

/// An open order plus its insertion sequence, used to break price
/// ties deterministically: the order placed first matches first.
#[derive(Clone, Debug)]
struct OpenOrder {
    order: Order,
    seq: u64,
}

/// An order book: open orders plus the trades they produced.
pub struct OrderBook {
    /// Open orders by id.
    orders: HashMap<String, OpenOrder>,
    /// Executed trades, in execution order.
    trades: Vec<Trade>,
    /// Insertion sequence counter.
    next_seq: u64,
}

impl OrderBook {
    /// An empty order book.
    pub fn new() -> Self {
        Self {
            orders: HashMap::new(),
            trades: Vec::new(),
            next_seq: 0,
        }
    }

    /// Open orders, sorted by the order they were placed, so the
    /// list is deterministic.
    pub fn open_orders(&self) -> Vec<&Order> {
        let mut open: Vec<&OpenOrder> = self.orders.values().collect();
        open.sort_by_key(|o| o.seq);
        open.iter().map(|o| &o.order).collect()
    }

    /// The open order with this id, if any.
    pub fn get_order(&self, id: &str) -> Option<&Order> {
        self.orders.get(id).map(|o| &o.order)
    }

    /// Executed trades, in execution order.
    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }

    /// Place an order. Validates the signature, the price and
    /// amount, and that no open order already has the id; then
    /// tries to match it against the book. Returns the trades
    /// this placement executed — possibly none, if nothing
    /// matched. Partially filled orders stay open with their
    /// remaining amount.
    pub fn place_order(&mut self, order: Order) -> Result<Vec<Trade>, XChangeError> {
        if !order.verify() {
            return Err(XChangeError::BadSignature);
        }
        if order.price == 0 || order.amount == 0 {
            return Err(XChangeError::InvalidOrder(
                "order price and amount must be greater than zero".to_string(),
            ));
        }
        if self.orders.contains_key(&order.id) {
            return Err(XChangeError::DuplicateOrder(order.id.clone()));
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        self.orders.insert(
            order.id.clone(),
            OpenOrder {
                order: order.clone(),
                seq,
            },
        );
        let executed = self.match_for(&order.id);
        Ok(executed)
    }

    /// Cancel an open order. Only the owner can cancel: anyone
    /// else gets [`XChangeError::NotOwner`] and the order stays
    /// open. Returns the cancelled order.
    pub fn cancel_order(&mut self, id: &str, requester: &PeerId) -> Result<Order, XChangeError> {
        let open = self
            .orders
            .get(id)
            .ok_or_else(|| XChangeError::OrderNotFound(id.to_string()))?;
        if open.order.owner.public_key() != requester.public_key() {
            return Err(XChangeError::NotOwner {
                id: id.to_string(),
                requester: requester.display_name().to_string(),
            });
        }
        Ok(self
            .orders
            .remove(id)
            .expect("order was present")
            .order)
    }

    /// Match the order with `id` against the book until it is
    /// filled or no counterparty matches. Returns the trades
    /// executed. The order itself is updated in place (its
    /// remaining amount shrinks) and removed when filled.
    fn match_for(&mut self, id: &str) -> Vec<Trade> {
        let mut executed = Vec::new();
        loop {
            // Read the taker's current state.
            let Some(taker) = self.orders.get(id) else {
                break;
            };
            let taker = taker.order.clone();
            // Find the best counterparty: the cheapest sell for a
            // buy (highest buy for a sell), ties to the earliest
            // placed order.
            let best = self.best_counterparty(&taker);
            let Some((maker_id, maker)) = best else {
                break;
            };
            let maker = maker.clone();
            let price = match taker.side {
                // The execution price is always the sell price:
                // for a buy taker it is the maker's price, for a
                // sell taker it is the taker's own price.
                Side::Buy => maker.price,
                Side::Sell => taker.price,
            };
            let amount = taker.amount.min(maker.amount);
            let (buy, sell) = match taker.side {
                Side::Buy => (taker.clone(), maker.clone()),
                Side::Sell => (maker.clone(), taker.clone()),
            };
            executed.push(Trade {
                buy_order_id: buy.id.clone(),
                sell_order_id: sell.id.clone(),
                buyer: buy.owner.clone(),
                seller: sell.owner.clone(),
                price,
                amount,
            });
            self.trades.push(executed.last().unwrap().clone());
            // Shrink both orders by the traded amount, removing
            // whichever filled completely.
            self.apply_fill(&taker.id, amount);
            self.apply_fill(&maker_id, amount);
            // If the taker filled, stop: it is gone from the book.
            if !self.orders.contains_key(id) {
                break;
            }
        }
        executed
    }

    /// The best open counterparty for `taker`, if any: a sell at
    /// or below the buy price (cheapest first), or a buy at or
    /// above the sell price (highest first). Returns the maker's
    /// id and order.
    fn best_counterparty(&self, taker: &Order) -> Option<(String, Order)> {
        let mut candidates: Vec<&OpenOrder> = self
            .orders
            .values()
            .filter(|o| o.order.id != taker.id)
            .filter(|o| match (taker.side, o.order.side) {
                (Side::Buy, Side::Sell) => o.order.price <= taker.price,
                (Side::Sell, Side::Buy) => o.order.price >= taker.price,
                _ => false,
            })
            .collect();
        match taker.side {
            // Cheapest sell first, then earliest placed.
            Side::Buy => candidates.sort_by(|a, b| {
                a.order
                    .price
                    .cmp(&b.order.price)
                    .then(a.seq.cmp(&b.seq))
            }),
            // Highest buy first, then earliest placed.
            Side::Sell => candidates.sort_by(|a, b| {
                b.order
                    .price
                    .cmp(&a.order.price)
                    .then(a.seq.cmp(&b.seq))
            }),
        }
        candidates
            .first()
            .map(|o| (o.order.id.clone(), o.order.clone()))
    }

    /// Reduce an open order's remaining amount by `amount`,
    /// removing it from the book if it is now filled.
    fn apply_fill(&mut self, id: &str, amount: u64) {
        if let Some(open) = self.orders.get_mut(id) {
            open.order.amount -= amount;
            if open.order.amount == 0 {
                self.orders.remove(id);
            }
        }
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

    fn buy(owner: &Identity, id: &str, price: u64, amount: u64) -> Order {
        Order::new(owner, id, Side::Buy, price, amount).unwrap()
    }

    fn sell(owner: &Identity, id: &str, price: u64, amount: u64) -> Order {
        Order::new(owner, id, Side::Sell, price, amount).unwrap()
    }

    #[test]
    fn matching_buy_and_sell_executes_at_sell_price() {
        let (alice, bob, _) = identities();
        let mut book = OrderBook::new();
        // Bob sells 10 units at 100 each.
        let trades = book.place_order(sell(&bob, "ord-sell-001", 100, 10)).unwrap();
        assert!(trades.is_empty());
        assert_eq!(book.open_orders().len(), 1);
        // Alice buys 10 units at up to 120 each: matches at the
        // sell price of 100, not her 120.
        let trades = book.place_order(buy(&alice, "ord-buy-001", 120, 10)).unwrap();
        assert_eq!(trades.len(), 1);
        let trade = &trades[0];
        assert_eq!(trade.buy_order_id, "ord-buy-001");
        assert_eq!(trade.sell_order_id, "ord-sell-001");
        assert_eq!(trade.buyer.display_name(), "alice");
        assert_eq!(trade.seller.display_name(), "bob");
        assert_eq!(trade.price, 100);
        assert_eq!(trade.amount, 10);
        // Both orders filled: the book is empty.
        assert!(book.open_orders().is_empty());
        assert_eq!(book.trades().len(), 1);
        // The same match works in the other order: a resting buy
        // matched by a new sell still executes at the sell price.
        let mut book = OrderBook::new();
        book.place_order(buy(&alice, "ord-buy-002", 120, 5)).unwrap();
        let trades = book.place_order(sell(&bob, "ord-sell-002", 100, 5)).unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, 100);
        assert_eq!(trades[0].amount, 5);
        assert!(book.open_orders().is_empty());
    }

    #[test]
    fn no_match_leaves_both_orders_open() {
        let (alice, bob, _) = identities();
        let mut book = OrderBook::new();
        // Bob sells at 150; Alice will only pay 100.
        book.place_order(sell(&bob, "ord-sell-001", 150, 10)).unwrap();
        let trades = book.place_order(buy(&alice, "ord-buy-001", 100, 10)).unwrap();
        assert!(trades.is_empty());
        assert_eq!(book.open_orders().len(), 2);
        // Equal prices do match.
        let mut book = OrderBook::new();
        book.place_order(sell(&bob, "ord-sell-002", 100, 4)).unwrap();
        let trades = book.place_order(buy(&alice, "ord-buy-002", 100, 4)).unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, 100);
    }

    #[test]
    fn partial_fill_leaves_remainder_open() {
        let (alice, bob, carol) = identities();
        let mut book = OrderBook::new();
        // Bob sells 10 at 100.
        book.place_order(sell(&bob, "ord-sell-001", 100, 10)).unwrap();
        // Alice buys only 4: the sell keeps 6 open.
        let trades = book.place_order(buy(&alice, "ord-buy-001", 100, 4)).unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].amount, 4);
        let open = book.open_orders();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].id, "ord-sell-001");
        assert_eq!(open[0].amount, 6);
        // Carol buys 10 at 100: fills the remaining 6, and her
        // own order keeps 4 open.
        let trades = book.place_order(buy(&carol, "ord-buy-002", 100, 10)).unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].amount, 6);
        assert_eq!(trades[0].buy_order_id, "ord-buy-002");
        let open = book.open_orders();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].id, "ord-buy-002");
        assert_eq!(open[0].amount, 4);
    }

    #[test]
    fn cancel_removes_order_and_only_owner_can_cancel() {
        let (alice, bob, _) = identities();
        let mut book = OrderBook::new();
        book.place_order(sell(&bob, "ord-sell-001", 100, 10)).unwrap();
        // Alice is not the owner: refused, order stays open.
        let err = book
            .cancel_order("ord-sell-001", alice.peer_id())
            .unwrap_err();
        assert_eq!(
            err,
            XChangeError::NotOwner {
                id: "ord-sell-001".to_string(),
                requester: "alice".to_string(),
            }
        );
        assert_eq!(book.open_orders().len(), 1);
        // The owner cancels: the order is gone.
        let cancelled = book.cancel_order("ord-sell-001", bob.peer_id()).unwrap();
        assert_eq!(cancelled.id, "ord-sell-001");
        assert!(book.open_orders().is_empty());
        // Cancelling again: nothing to cancel.
        assert_eq!(
            book.cancel_order("ord-sell-001", bob.peer_id()),
            Err(XChangeError::OrderNotFound("ord-sell-001".to_string()))
        );
        // A cancelled order can never match afterwards.
        book.place_order(buy(&alice, "ord-buy-001", 200, 10)).unwrap();
        assert!(book.trades().is_empty());
    }

    #[test]
    fn order_with_bad_signature_is_rejected() {
        let (alice, bob, _) = identities();
        let mut book = OrderBook::new();
        let mut bad = buy(&alice, "ord-buy-001", 100, 10);
        bad.price = 1; // changed after signing
        assert!(!bad.verify());
        assert_eq!(book.place_order(bad), Err(XChangeError::BadSignature));
        assert!(book.open_orders().is_empty());
        // A tampered amount is refused too.
        let mut bad = sell(&bob, "ord-sell-001", 100, 10);
        bad.amount = 1_000;
        assert!(!bad.verify());
        assert_eq!(book.place_order(bad), Err(XChangeError::BadSignature));
    }

    #[test]
    fn duplicate_order_id_is_rejected() {
        let (alice, bob, _) = identities();
        let mut book = OrderBook::new();
        book.place_order(buy(&alice, "ord-001", 100, 5)).unwrap();
        assert_eq!(
            book.place_order(sell(&bob, "ord-001", 100, 5)),
            Err(XChangeError::DuplicateOrder("ord-001".to_string()))
        );
        // After the first order fills and leaves, the id is free.
        book.place_order(sell(&bob, "ord-002", 100, 5)).unwrap();
        assert!(book.open_orders().is_empty());
        assert!(book.place_order(buy(&alice, "ord-001", 100, 5)).is_ok());
    }

    #[test]
    fn cheapest_sell_matches_first() {
        let (alice, bob, carol) = identities();
        let mut book = OrderBook::new();
        // Two sells: Carol at 120, Bob at 90. Alice buys 5 at 130.
        book.place_order(sell(&carol, "ord-sell-expensive", 120, 5))
            .unwrap();
        book.place_order(sell(&bob, "ord-sell-cheap", 90, 5)).unwrap();
        let trades = book.place_order(buy(&alice, "ord-buy-001", 130, 5)).unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].sell_order_id, "ord-sell-cheap");
        assert_eq!(trades[0].price, 90);
        // Carol's more expensive sell is still open.
        let open = book.open_orders();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].id, "ord-sell-expensive");
    }

    #[test]
    fn one_buy_sweeps_multiple_sells() {
        let (alice, bob, carol) = identities();
        let mut book = OrderBook::new();
        book.place_order(sell(&bob, "ord-sell-001", 90, 3)).unwrap();
        book.place_order(sell(&carol, "ord-sell-002", 100, 4)).unwrap();
        // Alice buys 6 at 110: 3 from Bob at 90, 3 from Carol
        // at 100 (partial fill of Carol's 4).
        let trades = book.place_order(buy(&alice, "ord-buy-001", 110, 6)).unwrap();
        assert_eq!(trades.len(), 2);
        assert_eq!(trades[0].sell_order_id, "ord-sell-001");
        assert_eq!(trades[0].price, 90);
        assert_eq!(trades[0].amount, 3);
        assert_eq!(trades[1].sell_order_id, "ord-sell-002");
        assert_eq!(trades[1].price, 100);
        assert_eq!(trades[1].amount, 3);
        // Carol's sell keeps 1 open; Alice's buy filled.
        let open = book.open_orders();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].id, "ord-sell-002");
        assert_eq!(open[0].amount, 1);
    }

    #[test]
    fn order_rules() {
        let (alice, _, _) = identities();
        assert!(Order::new(&alice, "", Side::Buy, 100, 1).is_err());
        assert!(Order::new(&alice, "has space", Side::Buy, 100, 1).is_err());
        assert!(Order::new(&alice, "ord-zero-price", Side::Buy, 0, 1).is_err());
        assert!(Order::new(&alice, "ord-zero-amount", Side::Buy, 100, 0).is_err());
        assert!(Order::new(&alice, "ord-ok_1.2", Side::Sell, 100, 1).is_ok());
    }
}
