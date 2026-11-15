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
//! * A market order has no price: it takes the best available
//!   price(s) until it is filled or the book is empty. A market
//!   order that cannot fill at all is refused with
//!   [`XChangeError::NoLiquidity`] — never a silent no-op — and
//!   any unfilled remainder is discarded, never posted.
//!
//! Orders are identified by human-readable ids (`ord-001`), never
//! hashes.
//!
//! ## Two ways to place an order
//!
//! * `place_order` / `place_order_at` match and record trades but
//!   move no coins: settlement is the caller's job. They exist so
//!   a book can be used as a pure matching engine.
//! * `place_order_settled` / `place_market_order` settle every
//!   fill through the X•Koin ledger: the buyer pays the seller
//!   for each fill. Settlement is atomic — the whole match is
//!   planned first, every buyer's balance is checked, every
//!   transfer is signed and verified, and only then do any coins
//!   move. A fill that cannot settle fails the whole match: no
//!   coins move, the orders stay exactly as they were, and no
//!   trade is recorded. After the pre-checks, `Ledger::submit`
//!   cannot fail (amounts are non-zero, signatures verify, ids
//!   are fresh, funds were checked, balances cannot overflow),
//!   which is what makes the atomicity real instead of hopeful.
//!
//! The book never holds a trader's secret key, so it cannot sign
//! a coin transaction itself. Settlement takes a
//! [`SettlementSigner`] — supplied by the caller, usually the
//! node, which holds the keys — that signs each transfer. The
//! book checks every returned transaction (sender, recipient,
//! amount, memo, signature, and that the ledger has not seen it)
//! before submitting any of them.
//!
//! ## Trade history
//!
//! Every fill is recorded with a caller-supplied timestamp
//! (seconds since the Unix epoch, or a sequence; `0` when the
//! caller did not supply one). The log is bounded:
//! [`MAX_TRADES`] trades are kept and the oldest are dropped.
//! `last_price` is the most recent fill price, and `volume`
//! sums the units traded in a time window.

use std::collections::HashMap;
use std::fmt;

use kernel_crypto::{sha256, to_hex, Ed25519Signature};
use kernel_identity::{Identity, IdentityError, PeerId};
use x_koin::{CoinTransaction, KoinError, Ledger};

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
    /// A market order found no counterparty at all.
    NoLiquidity,
    /// A settlement could not complete. No coins moved and the
    /// book is unchanged — the whole match failed atomically.
    SettlementFailed(String),
    /// Wraps a [`KoinError`] from settlement.
    Koin(KoinError),
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
            XChangeError::NotOwner { id, requester } => {
                write!(f, "peer '{requester}' does not own order '{id}'")
            }
            XChangeError::NoLiquidity => {
                write!(f, "market order found no counterparty to fill against")
            }
            XChangeError::SettlementFailed(msg) => {
                write!(f, "settlement failed, nothing moved: {msg}")
            }
            XChangeError::Koin(e) => write!(f, "coin error: {e}"),
        }
    }
}

impl std::error::Error for XChangeError {}

impl From<IdentityError> for XChangeError {
    fn from(e: IdentityError) -> Self {
        XChangeError::Identity(e)
    }
}

impl From<KoinError> for XChangeError {
    fn from(e: KoinError) -> Self {
        XChangeError::Koin(e)
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
fn order_sign_message(id: &str, owner: &PeerId, side: Side, price: u64, amount: u64) -> Vec<u8> {
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
    /// Caller-supplied timestamp of the fill (seconds since the
    /// Unix epoch, or a sequence); `0` when the caller did not
    /// supply one.
    pub timestamp: u64,
}

/// How many trades the book keeps: the trade log is bounded and
/// the oldest trades are dropped once it is full.
pub const MAX_TRADES: usize = 1_000;

/// A market order: an intent to buy or sell at whatever the best
/// available price is. Unlike a limit [`Order`] it has no price,
/// so it can never rest on the book — it fills against the best
/// counterparties until it is filled or the book is empty, and
/// any unfilled remainder is discarded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarketOrder {
    /// Human-readable id, e.g. `"mkt-001"`.
    pub id: String,
    /// Who placed the order.
    pub owner: PeerId,
    /// Buy or sell.
    pub side: Side,
    /// How many units are wanted.
    pub amount: u64,
    /// The owner's signature over id, owner, side, amount.
    pub signature: Ed25519Signature,
}

/// Domain separator for market order signatures: a signature
/// covers `b"XCHG-MKT1" || id || owner key || side || amount`.
const MARKET_SIGN_DOMAIN: &[u8] = b"XCHG-MKT1";

/// The message an owner signs for a market order.
fn market_sign_message(id: &str, owner: &PeerId, side: Side, amount: u64) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend_from_slice(MARKET_SIGN_DOMAIN);
    msg.extend_from_slice(&(id.len() as u16).to_le_bytes());
    msg.extend_from_slice(id.as_bytes());
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg.push(side.as_byte());
    msg.extend_from_slice(&amount.to_le_bytes());
    msg
}

impl MarketOrder {
    /// Create and sign a market order as `owner`. The amount must
    /// be greater than zero.
    pub fn new(owner: &Identity, id: &str, side: Side, amount: u64) -> Result<Self, XChangeError> {
        validate_id(id)?;
        if amount == 0 {
            return Err(XChangeError::InvalidOrder(
                "market order amount must be greater than zero".to_string(),
            ));
        }
        let owner_id = owner.peer_id().clone();
        let msg = market_sign_message(id, &owner_id, side, amount);
        let signature = owner.sign(&msg);
        Ok(Self {
            id: id.to_string(),
            owner: owner_id,
            side,
            amount,
            signature,
        })
    }

    /// True if the signature verifies against the owner for this
    /// exact id, side, and amount. Any tampering fails this.
    pub fn verify(&self) -> bool {
        let msg = market_sign_message(&self.id, &self.owner, self.side, self.amount);
        self.owner.verify(&msg, &self.signature).is_ok()
    }
}

/// Signs coin transfers on a trader's behalf so the exchange can
/// settle fills. The book never holds a secret key, so settlement
/// needs a signer supplied by the caller — usually the node,
/// which holds the keys. The signer must return a transaction
/// for exactly the transfer asked for (sender, recipient, amount,
/// memo); the book verifies every returned transaction before
/// submitting any of them, and a signer that returns anything
/// else fails the whole match.
pub trait SettlementSigner {
    /// Sign a transfer of `amount` coins from `sender` to
    /// `recipient`, memoed `memo`. The memo is chosen by the book
    /// (unique per settlement) so the ledger never sees a replay.
    fn sign_transfer(
        &self,
        sender: &PeerId,
        recipient: &PeerId,
        amount: u64,
        memo: &str,
    ) -> Result<CoinTransaction, XChangeError>;
}

/// An open order plus its insertion sequence, used to break price
/// ties deterministically: the order placed first matches first.
#[derive(Clone, Debug)]
struct OpenOrder {
    order: Order,
    seq: u64,
}

/// A completed fill that matching plans before it is applied:
/// which maker it fills against, how much, and at what price.
/// Planning is read-only — the book is only mutated once a plan
/// is fully settled (or, for the unsettled path, once the plan
/// is known to be complete).
#[derive(Clone, Debug)]
struct PlannedFill {
    /// The id of the resting (maker) order.
    maker_id: String,
    /// How many units change hands.
    amount: u64,
    /// The execution price: always the sell price.
    price: u64,
    /// The id of the buy order in the fill.
    buy_order_id: String,
    /// The id of the sell order in the fill.
    sell_order_id: String,
    /// Who bought.
    buyer: PeerId,
    /// Who sold.
    seller: PeerId,
}

/// One coin transfer a settlement needs: the buyer pays the
/// seller. Carries the fill's identity so the settlement memo
/// can be made unique to this fill.
struct PlannedTransfer {
    /// Who pays.
    buyer: PeerId,
    /// Who is paid.
    seller: PeerId,
    /// Coins to move: price * amount, fits in a u64.
    cost: u64,
    /// Buy order id, for the memo.
    buy_order_id: String,
    /// Sell order id, for the memo.
    sell_order_id: String,
    /// Fill price, for the memo.
    price: u64,
    /// Fill amount, for the memo.
    amount: u64,
}

/// A memo unique to this fill: the settlement sequence plus the
/// first 12 hex chars of the SHA-256 of the fill's identity
/// (buy id, sell id, price, amount). Two different fills never
/// share a memo, so the ledger never mistakes one settlement
/// for a replay of another — even when two books share one
/// ledger. The only way two settlements collide is the same
/// fill content at the same sequence in two books, which the
/// ledger then refuses loudly as a replay before any coin
/// moves.
fn settlement_memo(seq: u64, buy_id: &str, sell_id: &str, price: u64, amount: u64) -> String {
    let mut input = Vec::new();
    input.extend_from_slice(buy_id.as_bytes());
    input.extend_from_slice(sell_id.as_bytes());
    input.extend_from_slice(&price.to_le_bytes());
    input.extend_from_slice(&amount.to_le_bytes());
    let digest = sha256(&input);
    format!("xchg-{}-{}", seq, &to_hex(&digest)[..12])
}

/// An order book: open orders plus the trades they produced.
#[derive(Default)]
pub struct OrderBook {
    /// Open orders by id.
    orders: HashMap<String, OpenOrder>,
    /// Executed trades, in execution order, bounded to
    /// [`MAX_TRADES`] — the oldest are dropped when full.
    trades: Vec<Trade>,
    /// Insertion sequence counter.
    next_seq: u64,
    /// Settlement nonce counter: each settlement transaction is
    /// memoed with a fresh number from this counter, so the
    /// ledger never sees a replayed settlement id.
    settlement_seq: u64,
}

impl OrderBook {
    /// An empty order book.
    pub fn new() -> Self {
        Self {
            orders: HashMap::new(),
            trades: Vec::new(),
            next_seq: 0,
            settlement_seq: 0,
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

    /// Executed trades, in execution order. The log is bounded
    /// to [`MAX_TRADES`]: once it is full the oldest trade is
    /// dropped for each new one.
    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }

    /// Open buy orders, best first: highest price first, ties to
    /// the order placed first.
    pub fn bids(&self) -> Vec<&Order> {
        let mut bids: Vec<&OpenOrder> = self
            .orders
            .values()
            .filter(|o| o.order.side == Side::Buy)
            .collect();
        bids.sort_by(|a, b| b.order.price.cmp(&a.order.price).then(a.seq.cmp(&b.seq)));
        bids.iter().map(|o| &o.order).collect()
    }

    /// Open sell orders, best first: lowest price first, ties to
    /// the order placed first.
    pub fn asks(&self) -> Vec<&Order> {
        let mut asks: Vec<&OpenOrder> = self
            .orders
            .values()
            .filter(|o| o.order.side == Side::Sell)
            .collect();
        asks.sort_by(|a, b| a.order.price.cmp(&b.order.price).then(a.seq.cmp(&b.seq)));
        asks.iter().map(|o| &o.order).collect()
    }

    /// The best open bid price, if any.
    pub fn best_bid(&self) -> Option<u64> {
        self.bids().first().map(|o| o.price)
    }

    /// The best open ask price, if any.
    pub fn best_ask(&self) -> Option<u64> {
        self.asks().first().map(|o| o.price)
    }

    /// The price of the most recent fill, if any trade has
    /// executed.
    pub fn last_price(&self) -> Option<u64> {
        self.trades.last().map(|t| t.price)
    }

    /// Total units traded in the last `window_secs` seconds as of
    /// `now`: every recorded trade with a timestamp in
    /// `[now - window_secs, now]` counts. Amounts are units, not
    /// coins. Trades recorded without a timestamp (timestamp `0`)
    /// only count in a window that reaches back to `0`.
    pub fn volume(&self, now: u64, window_secs: u64) -> u64 {
        let start = now.saturating_sub(window_secs);
        self.trades
            .iter()
            .filter(|t| t.timestamp >= start && t.timestamp <= now)
            .fold(0u64, |acc, t| acc.saturating_add(t.amount))
    }

    /// Place an order, matching without settlement: validates the
    /// signature, the price and amount, and that no open order
    /// already has the id; then tries to match it against the
    /// book. Returns the trades this placement executed —
    /// possibly none, if nothing matched. Partially filled orders
    /// stay open with their remaining amount. Trades are stamped
    /// with timestamp `0`; use [`place_order_at`](Self::place_order_at)
    /// to supply one. No coins move on this path — for settlement
    /// use [`place_order_settled`](Self::place_order_settled).
    pub fn place_order(&mut self, order: Order) -> Result<Vec<Trade>, XChangeError> {
        self.place_order_at(order, 0)
    }

    /// Place an order as [`place_order`](Self::place_order), but
    /// stamp any executed trades with `now`. No coins move on
    /// this path.
    pub fn place_order_at(&mut self, order: Order, now: u64) -> Result<Vec<Trade>, XChangeError> {
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
        // Plan the match before the taker enters the book, so the
        // plan only ever sees resting orders.
        let plan = self.plan_limit_match(&order);
        let seq = self.next_seq;
        self.next_seq += 1;
        self.orders.insert(
            order.id.clone(),
            OpenOrder {
                order: order.clone(),
                seq,
            },
        );
        Ok(self.apply_plan(&order.id, &plan, now))
    }

    /// Place an order with settlement: like
    /// [`place_order`](Self::place_order), but every fill moves
    /// coins through `ledger` — the buyer pays the seller for
    /// each fill — and the whole match is atomic. The match is
    /// planned first; if any fill cannot settle (a buyer cannot
    /// cover it at fill time, a transfer would overflow, or the
    /// signer fails), the whole placement is refused with
    /// [`XChangeError::SettlementFailed`]: no coins move, the
    /// orders stay exactly as they were, and no trade is
    /// recorded. Trades are stamped with timestamp `0`; use
    /// [`place_order_settled_at`](Self::place_order_settled_at)
    /// to supply one.
    pub fn place_order_settled(
        &mut self,
        order: Order,
        ledger: &mut Ledger,
        signer: &dyn SettlementSigner,
    ) -> Result<Vec<Trade>, XChangeError> {
        self.place_order_settled_at(order, 0, ledger, signer)
    }

    /// Place an order with settlement as
    /// [`place_order_settled`](Self::place_order_settled), but
    /// stamp any executed trades with `now`.
    pub fn place_order_settled_at(
        &mut self,
        order: Order,
        now: u64,
        ledger: &mut Ledger,
        signer: &dyn SettlementSigner,
    ) -> Result<Vec<Trade>, XChangeError> {
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
        let plan = self.plan_limit_match(&order);
        // Settle first: nothing is mutated until every fill is
        // paid for. If this fails, the book and the ledger are
        // exactly as they were.
        self.settle_plan(&plan, ledger, signer)?;
        let seq = self.next_seq;
        self.next_seq += 1;
        self.orders.insert(
            order.id.clone(),
            OpenOrder {
                order: order.clone(),
                seq,
            },
        );
        Ok(self.apply_plan(&order.id, &plan, now))
    }

    /// Place a market order with settlement: it takes the best
    /// available price(s) until it is filled or the book is
    /// empty, settling every fill through `ledger` atomically, as
    /// in [`place_order_settled`](Self::place_order_settled).
    /// A market order that cannot fill at all is refused with
    /// [`XChangeError::NoLiquidity`] and changes nothing; any
    /// unfilled remainder is discarded, never posted. Trades are
    /// stamped with timestamp `0`; use
    /// [`place_market_order_at`](Self::place_market_order_at) to
    /// supply one.
    pub fn place_market_order(
        &mut self,
        order: MarketOrder,
        ledger: &mut Ledger,
        signer: &dyn SettlementSigner,
    ) -> Result<Vec<Trade>, XChangeError> {
        self.place_market_order_at(order, 0, ledger, signer)
    }

    /// Place a market order with settlement as
    /// [`place_market_order`](Self::place_market_order), but stamp
    /// any executed trades with `now`.
    pub fn place_market_order_at(
        &mut self,
        order: MarketOrder,
        now: u64,
        ledger: &mut Ledger,
        signer: &dyn SettlementSigner,
    ) -> Result<Vec<Trade>, XChangeError> {
        if !order.verify() {
            return Err(XChangeError::BadSignature);
        }
        if order.amount == 0 {
            return Err(XChangeError::InvalidOrder(
                "market order amount must be greater than zero".to_string(),
            ));
        }
        if self.orders.contains_key(&order.id) {
            return Err(XChangeError::DuplicateOrder(order.id.clone()));
        }
        let plan = self.plan_against(order.side, None, &order.id, order.amount, &order.owner);
        if plan.is_empty() {
            return Err(XChangeError::NoLiquidity);
        }
        self.settle_plan(&plan, ledger, signer)?;
        // A market order never rests: apply the fills to the
        // makers only. `apply_fill` on the absent taker id is a
        // no-op.
        Ok(self.apply_plan(&order.id, &plan, now))
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
        Ok(self.orders.remove(id).expect("order was present").order)
    }

    /// Plan the fills a limit order would execute, in
    /// price-time order, without touching the book: the cheapest
    /// sell for a buy (highest buy for a sell), ties to the
    /// earliest placed order. Each maker is visited at most once
    /// — a fill takes the smaller of the two remaining amounts,
    /// so after a maker is touched either it is fully consumed
    /// or the taker is filled.
    fn plan_limit_match(&self, taker: &Order) -> Vec<PlannedFill> {
        self.plan_against(
            taker.side,
            Some(taker.price),
            &taker.id,
            taker.amount,
            &taker.owner,
        )
    }

    /// Plan fills for a taker of `taker_side` wanting
    /// `taker_amount` units at up to `taker_price` per unit
    /// (`None` for a market order: no price limit). Read-only:
    /// the book is untouched, so a plan can be settled — or
    /// rejected — before anything is applied.
    fn plan_against(
        &self,
        taker_side: Side,
        taker_price: Option<u64>,
        taker_id: &str,
        taker_amount: u64,
        taker_owner: &PeerId,
    ) -> Vec<PlannedFill> {
        let mut makers: Vec<&OpenOrder> = self
            .orders
            .values()
            .filter(|o| o.order.id != taker_id)
            .filter(|o| match (taker_side, o.order.side) {
                (Side::Buy, Side::Sell) => taker_price.is_none_or(|p| o.order.price <= p),
                (Side::Sell, Side::Buy) => taker_price.is_none_or(|p| o.order.price >= p),
                _ => false,
            })
            .collect();
        match taker_side {
            // Cheapest sell first, then earliest placed.
            Side::Buy => {
                makers.sort_by(|a, b| a.order.price.cmp(&b.order.price).then(a.seq.cmp(&b.seq)))
            }
            // Highest buy first, then earliest placed.
            Side::Sell => {
                makers.sort_by(|a, b| b.order.price.cmp(&a.order.price).then(a.seq.cmp(&b.seq)))
            }
        }
        let mut remaining = taker_amount;
        let mut fills = Vec::new();
        for maker in makers {
            if remaining == 0 {
                break;
            }
            let amount = remaining.min(maker.order.amount);
            // The execution price is always the sell price: for
            // a buy taker it is the maker's price; for a sell
            // taker it is the taker's own limit price — or, for
            // a market sell, the resting buy's price.
            let price = match taker_side {
                Side::Buy => maker.order.price,
                Side::Sell => taker_price.unwrap_or(maker.order.price),
            };
            let (buy_order_id, sell_order_id, buyer, seller) = match taker_side {
                Side::Buy => (
                    taker_id.to_string(),
                    maker.order.id.clone(),
                    taker_owner.clone(),
                    maker.order.owner.clone(),
                ),
                Side::Sell => (
                    maker.order.id.clone(),
                    taker_id.to_string(),
                    maker.order.owner.clone(),
                    taker_owner.clone(),
                ),
            };
            fills.push(PlannedFill {
                maker_id: maker.order.id.clone(),
                amount,
                price,
                buy_order_id,
                sell_order_id,
                buyer,
                seller,
            });
            remaining -= amount;
        }
        fills
    }

    /// Apply a plan: record each fill as a trade (stamped `now`),
    /// shrink the taker and the maker by the filled amount, and
    /// remove whichever filled completely. The taker id may be
    /// absent from the book — a market order never rests — and
    /// [`apply_fill`](Self::apply_fill) on an absent id is a
    /// no-op. Returns the trades executed, in plan order.
    fn apply_plan(&mut self, taker_id: &str, plan: &[PlannedFill], now: u64) -> Vec<Trade> {
        let mut executed = Vec::new();
        for fill in plan {
            let trade = Trade {
                buy_order_id: fill.buy_order_id.clone(),
                sell_order_id: fill.sell_order_id.clone(),
                buyer: fill.buyer.clone(),
                seller: fill.seller.clone(),
                price: fill.price,
                amount: fill.amount,
                timestamp: now,
            };
            self.record_trade(trade.clone());
            executed.push(trade);
            self.apply_fill(taker_id, fill.amount);
            self.apply_fill(&fill.maker_id, fill.amount);
        }
        executed
    }

    /// Record a trade in the bounded log, dropping the oldest
    /// trade when the log is full.
    fn record_trade(&mut self, trade: Trade) {
        self.trades.push(trade);
        if self.trades.len() > MAX_TRADES {
            self.trades.remove(0);
        }
    }

    /// Settle every fill in `plan` through `ledger`, atomically.
    /// A fill where buyer and seller are the same peer moves no
    /// coins and needs no transfer. For the rest, the whole match
    /// is checked before a single coin moves:
    ///
    /// 1. Per-peer net movement is tallied (a peer may both buy
    ///    and sell in one match, e.g. a self-match against their
    ///    own resting order): no peer may end negative, and no
    ///    balance may overflow.
    /// 2. Every transfer is signed by `signer` first. Each
    ///    returned transaction is checked — sender, recipient,
    ///    amount, memo, signature, and that the ledger has not
    ///    seen it — before any submission.
    /// 3. Only then are the transactions submitted. After the
    ///    checks above, submission cannot fail: amounts are
    ///    non-zero, signatures verify, ids are fresh, funds were
    ///    checked, and balances cannot overflow.
    ///
    /// Any failure in steps 1–2 returns an error with the ledger
    /// and the book untouched.
    fn settle_plan(
        &mut self,
        plan: &[PlannedFill],
        ledger: &mut Ledger,
        signer: &dyn SettlementSigner,
    ) -> Result<(), XChangeError> {
        // Tally net movement per peer, in u128 so price * amount
        // can never overflow the tally.
        let mut spent: HashMap<PeerId, u128> = HashMap::new();
        let mut received: HashMap<PeerId, u128> = HashMap::new();
        let mut transfers: Vec<PlannedTransfer> = Vec::new();
        for fill in plan {
            if fill.buyer.public_key() == fill.seller.public_key() {
                // Self-trade: the coins would go to the same
                // peer, so no transfer is needed.
                continue;
            }
            let cost128 = fill.price as u128 * fill.amount as u128;
            let cost = u64::try_from(cost128).map_err(|_| {
                XChangeError::SettlementFailed(format!(
                    "trade value overflows: {} x {} units",
                    fill.price, fill.amount
                ))
            })?;
            *spent.entry(fill.buyer.clone()).or_insert(0) += cost128;
            *received.entry(fill.seller.clone()).or_insert(0) += cost128;
            transfers.push(PlannedTransfer {
                buyer: fill.buyer.clone(),
                seller: fill.seller.clone(),
                cost,
                buy_order_id: fill.buy_order_id.clone(),
                sell_order_id: fill.sell_order_id.clone(),
                price: fill.price,
                amount: fill.amount,
            });
        }
        // No peer may end negative: spend must be covered by the
        // balance plus what the peer receives in this same match.
        for (peer, spend) in &spent {
            let balance = ledger.balance_of(peer) as u128;
            let incoming = received.get(peer).copied().unwrap_or(0);
            if *spend > balance + incoming {
                return Err(XChangeError::SettlementFailed(format!(
                    "{} needs {} but holds {} (plus {} incoming in this match)",
                    peer.display_name(),
                    spend,
                    balance,
                    incoming
                )));
            }
        }
        // No balance may overflow from what a peer receives.
        for (peer, incoming) in &received {
            let balance = ledger.balance_of(peer) as u128;
            if balance + incoming > u64::MAX as u128 {
                return Err(XChangeError::SettlementFailed(format!(
                    "paying {} to {} would overflow their balance",
                    incoming,
                    peer.display_name()
                )));
            }
        }
        // Sign every transfer first. Any signing failure aborts
        // before a single coin moves. Each memo is unique to its
        // fill, so the ledger never sees a replay.
        let mut txs: Vec<CoinTransaction> = Vec::new();
        for (i, transfer) in transfers.iter().enumerate() {
            let memo = settlement_memo(
                self.settlement_seq.wrapping_add(i as u64),
                &transfer.buy_order_id,
                &transfer.sell_order_id,
                transfer.price,
                transfer.amount,
            );
            let tx =
                signer.sign_transfer(&transfer.buyer, &transfer.seller, transfer.cost, &memo)?;
            if tx.sender.public_key() != transfer.buyer.public_key()
                || tx.recipient.public_key() != transfer.seller.public_key()
                || tx.amount != transfer.cost
                || tx.memo != memo
                || !tx.verify()
            {
                return Err(XChangeError::SettlementFailed(format!(
                    "signer returned a transaction that does not match the requested transfer to {}",
                    transfer.seller.display_name()
                )));
            }
            if ledger.has_seen(&tx) {
                return Err(XChangeError::SettlementFailed(
                    "settlement transaction was already accepted by the ledger".to_string(),
                ));
            }
            txs.push(tx);
        }
        // Submit. After the checks above this cannot fail.
        for tx in &txs {
            ledger.submit(tx)?;
        }
        self.settlement_seq = self.settlement_seq.wrapping_add(txs.len() as u64);
        Ok(())
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
        let trades = book
            .place_order(sell(&bob, "ord-sell-001", 100, 10))
            .unwrap();
        assert!(trades.is_empty());
        assert_eq!(book.open_orders().len(), 1);
        // Alice buys 10 units at up to 120 each: matches at the
        // sell price of 100, not her 120.
        let trades = book
            .place_order(buy(&alice, "ord-buy-001", 120, 10))
            .unwrap();
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
        book.place_order(buy(&alice, "ord-buy-002", 120, 5))
            .unwrap();
        let trades = book
            .place_order(sell(&bob, "ord-sell-002", 100, 5))
            .unwrap();
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
        book.place_order(sell(&bob, "ord-sell-001", 150, 10))
            .unwrap();
        let trades = book
            .place_order(buy(&alice, "ord-buy-001", 100, 10))
            .unwrap();
        assert!(trades.is_empty());
        assert_eq!(book.open_orders().len(), 2);
        // Equal prices do match.
        let mut book = OrderBook::new();
        book.place_order(sell(&bob, "ord-sell-002", 100, 4))
            .unwrap();
        let trades = book
            .place_order(buy(&alice, "ord-buy-002", 100, 4))
            .unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, 100);
    }

    #[test]
    fn partial_fill_leaves_remainder_open() {
        let (alice, bob, carol) = identities();
        let mut book = OrderBook::new();
        // Bob sells 10 at 100.
        book.place_order(sell(&bob, "ord-sell-001", 100, 10))
            .unwrap();
        // Alice buys only 4: the sell keeps 6 open.
        let trades = book
            .place_order(buy(&alice, "ord-buy-001", 100, 4))
            .unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].amount, 4);
        let open = book.open_orders();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].id, "ord-sell-001");
        assert_eq!(open[0].amount, 6);
        // Carol buys 10 at 100: fills the remaining 6, and her
        // own order keeps 4 open.
        let trades = book
            .place_order(buy(&carol, "ord-buy-002", 100, 10))
            .unwrap();
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
        book.place_order(sell(&bob, "ord-sell-001", 100, 10))
            .unwrap();
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
        book.place_order(buy(&alice, "ord-buy-001", 200, 10))
            .unwrap();
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
        book.place_order(sell(&bob, "ord-sell-cheap", 90, 5))
            .unwrap();
        let trades = book
            .place_order(buy(&alice, "ord-buy-001", 130, 5))
            .unwrap();
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
        book.place_order(sell(&carol, "ord-sell-002", 100, 4))
            .unwrap();
        // Alice buys 6 at 110: 3 from Bob at 90, 3 from Carol
        // at 100 (partial fill of Carol's 4).
        let trades = book
            .place_order(buy(&alice, "ord-buy-001", 110, 6))
            .unwrap();
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

    // -- Settlement test helpers -------------------------------------

    /// Signs transfers with whatever keys it was given, standing
    /// in for the node keyring.
    struct TestSigner<'a> {
        keys: Vec<&'a Identity>,
    }

    impl SettlementSigner for TestSigner<'_> {
        fn sign_transfer(
            &self,
            sender: &PeerId,
            recipient: &PeerId,
            amount: u64,
            memo: &str,
        ) -> Result<CoinTransaction, XChangeError> {
            let key = self
                .keys
                .iter()
                .find(|k| k.peer_id().public_key() == sender.public_key())
                .ok_or_else(|| {
                    XChangeError::SettlementFailed(format!("no key for {}", sender.display_name()))
                })?;
            CoinTransaction::new(key, recipient, memo, amount)
                .map_err(|e| XChangeError::SettlementFailed(e.to_string()))
        }
    }

    fn funded_ledger(alice: &Identity, bob: &Identity, carol: &Identity) -> Ledger {
        Ledger::new(&[
            (alice.peer_id().clone(), 10_000),
            (bob.peer_id().clone(), 10_000),
            (carol.peer_id().clone(), 10_000),
        ])
        .unwrap()
    }

    fn market_buy(owner: &Identity, id: &str, amount: u64) -> MarketOrder {
        MarketOrder::new(owner, id, Side::Buy, amount).unwrap()
    }

    fn market_sell(owner: &Identity, id: &str, amount: u64) -> MarketOrder {
        MarketOrder::new(owner, id, Side::Sell, amount).unwrap()
    }

    #[test]
    fn bids_and_asks_sorted_best_first() {
        let (alice, bob, carol) = identities();
        let mut book = OrderBook::new();
        // Bids: 100, 120, 100 (the two 100s in placement order).
        book.place_order(buy(&alice, "b-100-a", 100, 1)).unwrap();
        book.place_order(buy(&bob, "b-120", 120, 1)).unwrap();
        book.place_order(buy(&carol, "b-100-b", 100, 1)).unwrap();
        // Asks: 150, 130, 130 (the two 130s in placement order).
        book.place_order(sell(&alice, "s-150", 150, 1)).unwrap();
        book.place_order(sell(&bob, "s-130-a", 130, 1)).unwrap();
        book.place_order(sell(&carol, "s-130-b", 130, 1)).unwrap();
        // Nothing crossed (best bid 120 < best ask 130), so all
        // six orders are still open.
        assert_eq!(book.open_orders().len(), 6);
        let bid_ids: Vec<&str> = book.bids().iter().map(|o| o.id.as_str()).collect();
        assert_eq!(bid_ids, vec!["b-120", "b-100-a", "b-100-b"]);
        let ask_ids: Vec<&str> = book.asks().iter().map(|o| o.id.as_str()).collect();
        assert_eq!(ask_ids, vec!["s-130-a", "s-130-b", "s-150"]);
        assert_eq!(book.best_bid(), Some(120));
        assert_eq!(book.best_ask(), Some(130));
    }

    #[test]
    fn settled_buy_fills_at_ask_price_and_moves_exact_coins() {
        let (alice, bob, carol) = identities();
        let mut ledger = funded_ledger(&alice, &bob, &carol);
        let signer = TestSigner {
            keys: vec![&alice, &bob, &carol],
        };
        let mut book = OrderBook::new();
        // Bob sells 10 at 100.
        book.place_order(sell(&bob, "ord-sell-001", 100, 10))
            .unwrap();
        // Alice buys 10 at up to 120, settled: fills at the ask
        // price of 100, so exactly 1,000 coins move.
        let trades = book
            .place_order_settled(buy(&alice, "ord-buy-001", 120, 10), &mut ledger, &signer)
            .unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, 100);
        assert_eq!(trades[0].amount, 10);
        assert_eq!(ledger.balance_of(alice.peer_id()), 9_000);
        assert_eq!(ledger.balance_of(bob.peer_id()), 11_000);
        assert_eq!(ledger.total_supply(), 30_000);
        assert!(book.open_orders().is_empty());
        // The other direction settles too: Alice rests a buy at
        // 110, Bob sells 5 at 100 as the taker. The trade
        // executes at the sell price (100) and Alice pays.
        book.place_order(buy(&alice, "ord-buy-002", 110, 5))
            .unwrap();
        let trades = book
            .place_order_settled(sell(&bob, "ord-sell-002", 100, 5), &mut ledger, &signer)
            .unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, 100);
        assert_eq!(trades[0].amount, 5);
        assert_eq!(ledger.balance_of(alice.peer_id()), 8_500);
        assert_eq!(ledger.balance_of(bob.peer_id()), 11_500);
    }

    #[test]
    fn partial_fill_leaves_remainder_in_position() {
        let (alice, bob, carol) = identities();
        let mut ledger = funded_ledger(&alice, &bob, &carol);
        let signer = TestSigner {
            keys: vec![&alice, &bob, &carol],
        };
        let mut book = OrderBook::new();
        book.place_order(sell(&bob, "ord-sell-001", 100, 10))
            .unwrap();
        // Alice buys 4, settled: the sell keeps 6 open as the
        // only ask.
        let trades = book
            .place_order_settled(buy(&alice, "ord-buy-001", 100, 4), &mut ledger, &signer)
            .unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].amount, 4);
        assert_eq!(book.asks().len(), 1);
        assert_eq!(book.asks()[0].id, "ord-sell-001");
        assert_eq!(book.asks()[0].amount, 6);
        assert_eq!(ledger.balance_of(alice.peer_id()), 9_600);
        assert_eq!(ledger.balance_of(bob.peer_id()), 10_400);
        // Carol buys 10: fills the remaining 6, and her own order
        // keeps 4 open as the only bid.
        let trades = book
            .place_order_settled(buy(&carol, "ord-buy-002", 100, 10), &mut ledger, &signer)
            .unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].amount, 6);
        assert_eq!(book.bids().len(), 1);
        assert_eq!(book.bids()[0].id, "ord-buy-002");
        assert_eq!(book.bids()[0].amount, 4);
        assert_eq!(ledger.balance_of(carol.peer_id()), 9_400);
        assert_eq!(ledger.balance_of(bob.peer_id()), 11_000);
    }

    #[test]
    fn price_time_priority_earlier_order_fills_first() {
        let (alice, bob, carol) = identities();
        let mut ledger = funded_ledger(&alice, &bob, &carol);
        let signer = TestSigner {
            keys: vec![&alice, &bob, &carol],
        };
        // Two sells at the same price: Bob's placed first.
        let mut book = OrderBook::new();
        book.place_order(sell(&bob, "ord-sell-first", 100, 5))
            .unwrap();
        book.place_order(sell(&carol, "ord-sell-second", 100, 5))
            .unwrap();
        let trades = book
            .place_order_settled(buy(&alice, "ord-buy-001", 100, 5), &mut ledger, &signer)
            .unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].sell_order_id, "ord-sell-first");
        assert_eq!(book.open_orders().len(), 1);
        assert_eq!(book.open_orders()[0].id, "ord-sell-second");
        // And the buy side: two buys at the same price, Alice's
        // placed first; Bob's sell fills hers first.
        let mut book = OrderBook::new();
        book.place_order(buy(&alice, "ord-buy-first", 100, 5))
            .unwrap();
        book.place_order(buy(&carol, "ord-buy-second", 100, 5))
            .unwrap();
        let trades = book
            .place_order_settled(sell(&bob, "ord-sell-001", 100, 5), &mut ledger, &signer)
            .unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].buy_order_id, "ord-buy-first");
        assert_eq!(book.open_orders().len(), 1);
        assert_eq!(book.open_orders()[0].id, "ord-buy-second");
    }

    #[test]
    fn market_order_sweeps_multiple_price_levels() {
        let (alice, bob, carol) = identities();
        let mut ledger = funded_ledger(&alice, &bob, &carol);
        let signer = TestSigner {
            keys: vec![&alice, &bob, &carol],
        };
        let mut book = OrderBook::new();
        book.place_order(sell(&bob, "ord-sell-001", 90, 3)).unwrap();
        book.place_order(sell(&carol, "ord-sell-002", 100, 4))
            .unwrap();
        // A market buy for 6: 3 from Bob at 90, 3 from Carol at
        // 100. Exactly 3*90 + 3*100 = 570 coins move.
        let trades = book
            .place_market_order(market_buy(&alice, "mkt-buy-001", 6), &mut ledger, &signer)
            .unwrap();
        assert_eq!(trades.len(), 2);
        assert_eq!(trades[0].sell_order_id, "ord-sell-001");
        assert_eq!(trades[0].price, 90);
        assert_eq!(trades[0].amount, 3);
        assert_eq!(trades[1].sell_order_id, "ord-sell-002");
        assert_eq!(trades[1].price, 100);
        assert_eq!(trades[1].amount, 3);
        assert_eq!(ledger.balance_of(alice.peer_id()), 10_000 - 570);
        assert_eq!(ledger.balance_of(bob.peer_id()), 10_000 + 270);
        assert_eq!(ledger.balance_of(carol.peer_id()), 10_000 + 300);
        // Carol's sell keeps 1 open; the market order never
        // rests on the book.
        let open = book.open_orders();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].id, "ord-sell-002");
        assert_eq!(open[0].amount, 1);
        // A market sell the other way: Alice rests a buy, Bob
        // market-sells into it at the buy's price.
        let mut book = OrderBook::new();
        book.place_order(buy(&alice, "ord-buy-001", 110, 4))
            .unwrap();
        let trades = book
            .place_market_order(market_sell(&bob, "mkt-sell-001", 2), &mut ledger, &signer)
            .unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, 110);
        assert_eq!(trades[0].amount, 2);
        assert_eq!(ledger.balance_of(bob.peer_id()), 10_000 + 270 + 220);
        assert_eq!(ledger.balance_of(alice.peer_id()), 10_000 - 570 - 220);
    }

    #[test]
    fn market_order_with_empty_book_errors() {
        let (alice, bob, carol) = identities();
        let mut ledger = funded_ledger(&alice, &bob, &carol);
        let signer = TestSigner {
            keys: vec![&alice, &bob, &carol],
        };
        let mut book = OrderBook::new();
        assert_eq!(
            book.place_market_order(market_buy(&alice, "mkt-buy-001", 5), &mut ledger, &signer),
            Err(XChangeError::NoLiquidity)
        );
        assert_eq!(
            book.place_market_order(market_sell(&bob, "mkt-sell-001", 5), &mut ledger, &signer),
            Err(XChangeError::NoLiquidity)
        );
        // Nothing changed: no orders, no trades, no coins moved.
        assert!(book.open_orders().is_empty());
        assert!(book.trades().is_empty());
        assert_eq!(ledger.balance_of(alice.peer_id()), 10_000);
        assert_eq!(ledger.balance_of(bob.peer_id()), 10_000);
        // A market order for less than the book holds fills
        // partially and the remainder is discarded, not posted.
        book.place_order(sell(&bob, "ord-sell-001", 100, 2))
            .unwrap();
        let trades = book
            .place_market_order(market_buy(&alice, "mkt-buy-002", 5), &mut ledger, &signer)
            .unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].amount, 2);
        assert!(book.open_orders().is_empty());
    }

    #[test]
    fn cancel_filled_order_errors() {
        let (alice, bob, _) = identities();
        let mut book = OrderBook::new();
        book.place_order(sell(&bob, "ord-sell-001", 100, 5))
            .unwrap();
        book.place_order(buy(&alice, "ord-buy-001", 100, 5))
            .unwrap();
        // Both filled: neither can be cancelled any more.
        assert_eq!(
            book.cancel_order("ord-buy-001", alice.peer_id()),
            Err(XChangeError::OrderNotFound("ord-buy-001".to_string()))
        );
        assert_eq!(
            book.cancel_order("ord-sell-001", bob.peer_id()),
            Err(XChangeError::OrderNotFound("ord-sell-001".to_string()))
        );
        // A bad market order signature is refused too.
        let mut bad = market_buy(&alice, "mkt-bad-001", 5);
        bad.amount = 500;
        assert!(!bad.verify());
    }

    #[test]
    fn failed_settlement_changes_nothing() {
        let (alice, bob, carol) = identities();
        // Alice holds only 50 coins; Bob and Carol are funded.
        let mut ledger = Ledger::new(&[
            (alice.peer_id().clone(), 50),
            (bob.peer_id().clone(), 10_000),
            (carol.peer_id().clone(), 10_000),
        ])
        .unwrap();
        let signer = TestSigner {
            keys: vec![&alice, &bob, &carol],
        };
        let mut book = OrderBook::new();
        book.place_order(sell(&bob, "ord-sell-001", 100, 10))
            .unwrap();
        // Alice tries to buy 10 at 100: 1,000 coins needed, 50
        // held. The whole placement is refused.
        let err = book
            .place_order_settled(buy(&alice, "ord-buy-001", 100, 10), &mut ledger, &signer)
            .unwrap_err();
        assert!(matches!(err, XChangeError::SettlementFailed(_)));
        // The book is exactly as it was: Bob's sell still open
        // with 10, Alice's buy never entered, no trade recorded.
        assert_eq!(book.open_orders().len(), 1);
        assert_eq!(book.open_orders()[0].id, "ord-sell-001");
        assert_eq!(book.open_orders()[0].amount, 10);
        assert!(book.trades().is_empty());
        assert_eq!(ledger.balance_of(alice.peer_id()), 50);
        assert_eq!(ledger.balance_of(bob.peer_id()), 10_000);
        // Atomicity across fills: Alice holds 500 and buys 6 at
        // 110. The first fill (3 at 90 = 270) is affordable, the
        // second (3 at 100 = 300) is not — 570 total. The whole
        // match fails, not just the second fill.
        let mut ledger = Ledger::new(&[
            (alice.peer_id().clone(), 500),
            (bob.peer_id().clone(), 10_000),
            (carol.peer_id().clone(), 10_000),
        ])
        .unwrap();
        let mut book = OrderBook::new();
        book.place_order(sell(&bob, "ord-sell-001", 90, 3)).unwrap();
        book.place_order(sell(&carol, "ord-sell-002", 100, 4))
            .unwrap();
        let err = book
            .place_order_settled(buy(&alice, "ord-buy-001", 110, 6), &mut ledger, &signer)
            .unwrap_err();
        assert!(matches!(err, XChangeError::SettlementFailed(_)));
        assert_eq!(book.open_orders().len(), 2);
        assert_eq!(book.open_orders()[0].amount, 3);
        assert_eq!(book.open_orders()[1].amount, 4);
        assert!(book.trades().is_empty());
        assert_eq!(ledger.balance_of(alice.peer_id()), 500);
        assert_eq!(ledger.balance_of(bob.peer_id()), 10_000);
        assert_eq!(ledger.balance_of(carol.peer_id()), 10_000);
        assert!(ledger.transactions().is_empty());
    }

    #[test]
    fn trade_history_last_price_and_volume() {
        let (alice, bob, carol) = identities();
        let mut book = OrderBook::new();
        // Three fills at three timestamps.
        book.place_order(sell(&bob, "ord-sell-001", 100, 5))
            .unwrap();
        book.place_order_at(buy(&alice, "ord-buy-001", 100, 5), 100)
            .unwrap();
        book.place_order(sell(&carol, "ord-sell-002", 110, 3))
            .unwrap();
        book.place_order_at(buy(&alice, "ord-buy-002", 110, 3), 150)
            .unwrap();
        book.place_order(sell(&bob, "ord-sell-003", 90, 7)).unwrap();
        book.place_order_at(buy(&alice, "ord-buy-003", 90, 7), 200)
            .unwrap();
        assert_eq!(book.trades().len(), 3);
        assert_eq!(book.trades()[0].timestamp, 100);
        assert_eq!(book.trades()[1].timestamp, 150);
        assert_eq!(book.trades()[2].timestamp, 200);
        assert_eq!(book.last_price(), Some(90));
        // Volume over windows ending at 200.
        assert_eq!(book.volume(200, 100), 15); // [100, 200]: all
        assert_eq!(book.volume(200, 60), 10); // [140, 200]: 3 + 7
        assert_eq!(book.volume(200, 10), 7); // [190, 200]: 7
        assert_eq!(book.volume(150, 100), 8); // [50, 150]: 5 + 3
        assert_eq!(book.volume(50, 100), 0); // before any trade
                                             // An empty book has no last price and no volume.
        let empty = OrderBook::new();
        assert_eq!(empty.last_price(), None);
        assert_eq!(empty.volume(1000, 100), 0);
    }

    #[test]
    fn trade_log_is_bounded() {
        let (alice, bob, _) = identities();
        let mut book = OrderBook::new();
        // Push 1,005 trades straight into the log — no orders,
        // no signatures — so the test exercises the cap itself
        // instead of paying debug-mode Ed25519 costs per fill.
        for i in 0..(MAX_TRADES + 5) {
            book.record_trade(Trade {
                buy_order_id: format!("b-{i:04}"),
                sell_order_id: format!("s-{i:04}"),
                buyer: alice.peer_id().clone(),
                seller: bob.peer_id().clone(),
                price: 100,
                amount: 1,
                timestamp: i as u64,
            });
        }
        // Only the newest 1,000 are kept: the first five dropped
        // off the front.
        assert_eq!(book.trades().len(), MAX_TRADES);
        assert_eq!(book.trades()[0].buy_order_id, "b-0005");
        assert_eq!(book.trades()[MAX_TRADES - 1].buy_order_id, "b-1004");
        assert_eq!(book.last_price(), Some(100));
        // Fills still append to the log end to end (covered in
        // detail by trade_history_last_price_and_volume); one
        // fill here proves the path records.
        book.place_order(sell(&bob, "ord-sell-001", 100, 1))
            .unwrap();
        book.place_order(buy(&alice, "ord-buy-001", 100, 1))
            .unwrap();
        assert_eq!(book.trades().len(), MAX_TRADES);
        assert_eq!(book.trades()[MAX_TRADES - 1].buy_order_id, "ord-buy-001");
    }

    #[test]
    fn unsettled_placement_moves_no_coins() {
        let (alice, bob, carol) = identities();
        let ledger = funded_ledger(&alice, &bob, &carol);
        let mut book = OrderBook::new();
        book.place_order(sell(&bob, "ord-sell-001", 100, 10))
            .unwrap();
        let trades = book
            .place_order(buy(&alice, "ord-buy-001", 100, 10))
            .unwrap();
        assert_eq!(trades.len(), 1);
        // The trade is recorded, but the ledger is untouched:
        // settlement is the settled path's job, not this one.
        assert_eq!(ledger.balance_of(alice.peer_id()), 10_000);
        assert_eq!(ledger.balance_of(bob.peer_id()), 10_000);
        assert!(ledger.transactions().is_empty());
    }

    #[test]
    fn self_trade_settles_without_moving_coins() {
        let (alice, bob, carol) = identities();
        let mut ledger = funded_ledger(&alice, &bob, &carol);
        let signer = TestSigner {
            keys: vec![&alice, &bob, &carol],
        };
        let mut book = OrderBook::new();
        // Alice sells 5 at 100, then buys 5 at 100: she fills
        // against her own resting order.
        book.place_order(sell(&alice, "ord-sell-001", 100, 5))
            .unwrap();
        let trades = book
            .place_order_settled(buy(&alice, "ord-buy-001", 100, 5), &mut ledger, &signer)
            .unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].buyer.display_name(), "alice");
        assert_eq!(trades[0].seller.display_name(), "alice");
        // No coins moved: buyer and seller are the same peer.
        assert_eq!(ledger.balance_of(alice.peer_id()), 10_000);
        assert!(ledger.transactions().is_empty());
        assert!(book.open_orders().is_empty());
    }
}
