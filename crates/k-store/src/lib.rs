//! K•Store — a KERNEL-native storefront.
//!
//! Products are signed listings: a human-readable id, a name, a
//! description, a price in X•Koin, and a stock count. The seller's
//! signature covers the offer terms (id, seller, name, description,
//! price); stock is the store's own count, not part of the signed
//! offer, so selling one does not invalidate the listing. A forged
//! listing — tampered terms, or a signature from anyone but the
//! seller — is refused before it can be listed.
//!
//! Product pages are **rendered** through the HEY•AI engine: the
//! store builds a template plus product data and hands it to
//! [`hey_ai::render_addressed`]. The same product plus the same
//! template always renders the same bytes, addressed by their
//! SHA-256. Rendering is text/HTML bytes only — there is no
//! browser, no image pipeline, and no Shopify API here. This is a
//! KERNEL-native store, not Shopify itself.
//!
//! Orders settle through K•Pay (which settles on the X•Koin
//! ledger). The flow mirrors a real marketplace: the buyer places
//! an order, the seller issues a signed invoice for that order's
//! id and amount, and the buyer pays the invoice through the
//! store. Only a successful payment marks the order paid and
//! decrements stock; a failed payment changes nothing.
//!
//! Nothing here moves real money. Balances are ledger entries in
//! the coin's smallest unit, and the store holds no inventory —
//! it is a record of offers, orders, and sales.

use std::collections::HashMap;
use std::fmt;

use hey_ai::{render_addressed, verify_signature, RenderError, Value};
use k_pay::{Invoice, PayError, PaymentService, Receipt};
use kernel_identity::{Identity, PeerId};
use kernel_storage::Cid;
use x_koin::Ledger;

/// Domain separator for listing signatures: a signature covers
/// `b"KSTORE1" || id || seller key || name || description || price`.
const LISTING_SIGN_DOMAIN: &[u8; 7] = b"KSTORE1";

/// The default product page template, rendered by HEY•AI.
pub const DEFAULT_PRODUCT_TEMPLATE: &str = "<h1>{{name}}</h1>\n<p>{{description}}</p>\n<p>Price: {{price}} X•Koin</p>\n<p>Stock: {{stock}}</p>\n<p>Seller: {{seller}}</p>\n";

/// Errors returned by store operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    /// A listing id, name, description, or price broke the rules.
    InvalidListing(String),
    /// A product with this id is already listed.
    DuplicateProduct(String),
    /// No product with this id is listed.
    ProductNotFound(String),
    /// The product is out of stock.
    OutOfStock(String),
    /// No order with this id exists.
    OrderNotFound(String),
    /// The order was already paid.
    AlreadyPaid(String),
    /// The invoice does not match the order (wrong id, recipient,
    /// amount, or payer).
    InvoiceMismatch(String),
    /// A listing's signature did not verify against its seller.
    BadListingSignature,
    /// Wraps a [`RenderError`] from the HEY•AI engine.
    Render(RenderError),
    /// Wraps a [`PayError`] from settlement.
    Pay(PayError),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::InvalidListing(msg) => write!(f, "invalid listing: {msg}"),
            StoreError::DuplicateProduct(id) => {
                write!(f, "a product with id '{id}' is already listed")
            }
            StoreError::ProductNotFound(id) => write!(f, "no product with id '{id}'"),
            StoreError::OutOfStock(id) => write!(f, "product '{id}' is out of stock"),
            StoreError::OrderNotFound(id) => write!(f, "no order with id '{id}'"),
            StoreError::AlreadyPaid(id) => write!(f, "order '{id}' was already paid"),
            StoreError::InvoiceMismatch(msg) => write!(f, "invoice mismatch: {msg}"),
            StoreError::BadListingSignature => {
                write!(f, "listing signature is invalid")
            }
            StoreError::Render(e) => write!(f, "render error: {e}"),
            StoreError::Pay(e) => write!(f, "payment error: {e}"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<RenderError> for StoreError {
    fn from(e: RenderError) -> Self {
        StoreError::Render(e)
    }
}

impl From<PayError> for StoreError {
    fn from(e: PayError) -> Self {
        StoreError::Pay(e)
    }
}

/// Rules for product ids: 1-64 characters, ASCII letters, digits,
/// `-`, `_`, `.`. No spaces, no hashes — the same rule K•Pay uses
/// for invoice ids.
fn validate_id(id: &str) -> Result<(), StoreError> {
    if id.is_empty() {
        return Err(StoreError::InvalidListing(
            "product id must not be empty".to_string(),
        ));
    }
    if id.len() > 64 {
        return Err(StoreError::InvalidListing(
            "product id must be at most 64 characters".to_string(),
        ));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Err(StoreError::InvalidListing(
            "product id may only use letters, digits, '-', '_' and '.'".to_string(),
        ));
    }
    Ok(())
}

/// Rules for listing text: a name is 1-128 printable characters; a
/// description is at most 1000 printable characters (and may be
/// empty). A price must be greater than zero — a zero-price product
/// could never be paid through K•Pay, whose invoices require a
/// positive amount.
fn validate_listing(id: &str, name: &str, description: &str, price: u64) -> Result<(), StoreError> {
    validate_id(id)?;
    if name.is_empty() || name.len() > 128 {
        return Err(StoreError::InvalidListing(
            "product name must be 1-128 characters".to_string(),
        ));
    }
    if description.len() > 1000 {
        return Err(StoreError::InvalidListing(
            "product description must be at most 1000 characters".to_string(),
        ));
    }
    if !name.chars().all(|c| c.is_ascii_graphic() || c == ' ')
        || !description
            .chars()
            .all(|c| c.is_ascii_graphic() || c == ' ')
    {
        return Err(StoreError::InvalidListing(
            "product name and description must be printable ASCII".to_string(),
        ));
    }
    if price == 0 {
        return Err(StoreError::InvalidListing(
            "product price must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

/// The message a seller signs for a listing: domain + id + seller
/// key + name + description + price. Stock is deliberately not
/// signed — it is the store's own count and changes as units sell.
fn listing_sign_message(
    id: &str,
    seller: &PeerId,
    name: &str,
    description: &str,
    price: u64,
) -> Vec<u8> {
    let mut msg = Vec::with_capacity(7 + 32 + id.len() + name.len() + description.len() + 8);
    msg.extend_from_slice(LISTING_SIGN_DOMAIN);
    msg.extend_from_slice(id.as_bytes());
    msg.extend_from_slice(seller.public_key().as_bytes());
    msg.extend_from_slice(name.as_bytes());
    msg.extend_from_slice(description.as_bytes());
    msg.extend_from_slice(&price.to_le_bytes());
    msg
}

/// A signed product listing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Product {
    /// Human-readable id, e.g. `"lamp-001"`.
    pub id: String,
    /// Human-readable name, e.g. `"Brass Lamp"`.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Price, in the coin's smallest unit.
    pub price: u64,
    /// Units the store has left. Store state, not signed.
    pub stock: u32,
    /// Who sells this product.
    pub seller: PeerId,
    /// The seller's signature over id, seller, name, description,
    /// and price, as raw bytes. Stored raw so this crate does not
    /// need the crypto crate; verification goes through
    /// [`hey_ai::verify_signature`].
    pub signature: [u8; 64],
}

impl Product {
    /// Build and sign a listing as `seller`.
    pub fn new(
        id: &str,
        name: &str,
        description: &str,
        price: u64,
        stock: u32,
        seller: &Identity,
    ) -> Result<Self, StoreError> {
        validate_listing(id, name, description, price)?;
        let seller_id = seller.peer_id().clone();
        let msg = listing_sign_message(id, &seller_id, name, description, price);
        let signature = seller.sign(&msg).to_bytes();
        Ok(Self {
            id: id.to_string(),
            name: name.to_string(),
            description: description.to_string(),
            price,
            stock,
            seller: seller_id,
            signature,
        })
    }

    /// True if the signature verifies against the seller for these
    /// exact listing terms. Any tampering fails this check.
    pub fn verify(&self) -> bool {
        let msg = listing_sign_message(
            &self.id,
            &self.seller,
            &self.name,
            &self.description,
            self.price,
        );
        verify_signature(&self.seller, &msg, &self.signature)
    }
}

/// The status of an order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderStatus {
    /// Placed, not paid yet.
    Pending,
    /// Paid; the coins moved and stock was decremented.
    Paid,
}

/// An order: one buyer buying one unit of one product.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Order {
    /// Human-readable id, e.g. `"order-1"`, assigned in order.
    pub id: String,
    /// The product being bought.
    pub product_id: String,
    /// Who placed the order.
    pub buyer: PeerId,
    /// The amount to pay, in the coin's smallest unit — the
    /// product's price at order time.
    pub amount: u64,
    /// Where the order stands.
    pub status: OrderStatus,
    /// The payment receipt, once paid.
    pub receipt: Option<Receipt>,
}

/// A rendered product page: the bytes plus their content address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedPage {
    /// The product the page is for.
    pub product_id: String,
    /// SHA-256 of the rendered bytes, as a content address. The
    /// same product plus the same template always yields the same
    /// address.
    pub address: Cid,
    /// The rendered page bytes (text / HTML).
    pub bytes: Vec<u8>,
}

/// A store: signed listings, orders settled through K•Pay, and
/// product pages rendered by HEY•AI.
pub struct Store {
    payments: PaymentService,
    products: HashMap<String, Product>,
    orders: HashMap<String, Order>,
    next_order: u64,
}

impl Store {
    /// A store over an existing payment service (which owns the
    /// coin ledger and its genesis allocation).
    pub fn new(payments: PaymentService) -> Self {
        Self {
            payments,
            products: HashMap::new(),
            orders: HashMap::new(),
            next_order: 1,
        }
    }

    /// A store whose payment service starts at the given genesis
    /// allocation.
    pub fn new_with_genesis(genesis: &[(PeerId, u64)]) -> Result<Self, StoreError> {
        Ok(Self::new(PaymentService::new(genesis)?))
    }

    /// The underlying payment service.
    pub fn payments(&self) -> &PaymentService {
        &self.payments
    }

    /// The underlying coin ledger, for read access.
    pub fn ledger(&self) -> &Ledger {
        self.payments.ledger()
    }

    /// A peer's balance in the underlying ledger.
    pub fn balance_of(&self, peer: &PeerId) -> u64 {
        self.payments.balance_of(peer)
    }

    /// Register a signed listing. The listing is authenticated
    /// first — the signature must verify against the listing's
    /// seller — and only then checked against the id registry, so
    /// a forged listing is refused as a forgery even when it copies
    /// an existing id.
    pub fn register_product(&mut self, product: Product) -> Result<(), StoreError> {
        validate_listing(
            &product.id,
            &product.name,
            &product.description,
            product.price,
        )?;
        if !product.verify() {
            return Err(StoreError::BadListingSignature);
        }
        if self.products.contains_key(&product.id) {
            return Err(StoreError::DuplicateProduct(product.id.clone()));
        }
        self.products.insert(product.id.clone(), product);
        Ok(())
    }

    /// List a product: build and sign the listing as `seller` and
    /// register it. Returns the stored listing.
    pub fn list_product(
        &mut self,
        id: &str,
        name: &str,
        description: &str,
        price: u64,
        stock: u32,
        seller: &Identity,
    ) -> Result<Product, StoreError> {
        let product = Product::new(id, name, description, price, stock, seller)?;
        self.register_product(product.clone())?;
        Ok(product)
    }

    /// The listed product with this id, if any.
    pub fn get_product(&self, id: &str) -> Option<&Product> {
        self.products.get(id)
    }

    /// All listed products, sorted by id for a stable listing.
    pub fn list_products(&self) -> Vec<&Product> {
        let mut out: Vec<&Product> = self.products.values().collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    /// Render a product's page with the default template, through
    /// the HEY•AI engine. The same product plus the same template
    /// always renders the same bytes and the same address.
    pub fn render_product_page(&self, product_id: &str) -> Result<RenderedPage, StoreError> {
        self.render_product_page_with(product_id, DEFAULT_PRODUCT_TEMPLATE)
    }

    /// Render a product's page with a caller-supplied template,
    /// through the HEY•AI engine.
    pub fn render_product_page_with(
        &self,
        product_id: &str,
        template: &str,
    ) -> Result<RenderedPage, StoreError> {
        let product = self
            .products
            .get(product_id)
            .ok_or_else(|| StoreError::ProductNotFound(product_id.to_string()))?;
        let data = Value::map(vec![
            ("id".to_string(), Value::text(&product.id)),
            ("name".to_string(), Value::text(&product.name)),
            ("description".to_string(), Value::text(&product.description)),
            ("price".to_string(), Value::number(product.price as i64)),
            ("stock".to_string(), Value::number(product.stock as i64)),
            (
                "seller".to_string(),
                Value::text(product.seller.display_name()),
            ),
        ]);
        let rendered = render_addressed(template, &data)?;
        Ok(RenderedPage {
            product_id: product_id.to_string(),
            address: rendered.address,
            bytes: rendered.bytes,
        })
    }

    /// Place an order for one unit of a product, as `buyer`. The
    /// product must exist and have stock; nothing moves yet — no
    /// coins, no stock. Returns the pending order. The seller
    /// issues a signed invoice for this order's id and amount, and
    /// the buyer pays it with [`Store::pay_order`].
    pub fn place_order(&mut self, product_id: &str, buyer: &PeerId) -> Result<Order, StoreError> {
        let product = self
            .products
            .get(product_id)
            .ok_or_else(|| StoreError::ProductNotFound(product_id.to_string()))?;
        if product.stock == 0 {
            return Err(StoreError::OutOfStock(product_id.to_string()));
        }
        let id = format!("order-{}", self.next_order);
        self.next_order += 1;
        let order = Order {
            id: id.clone(),
            product_id: product_id.to_string(),
            buyer: buyer.clone(),
            amount: product.price,
            status: OrderStatus::Pending,
            receipt: None,
        };
        self.orders.insert(id, order.clone());
        Ok(order)
    }

    /// Pay an order with the seller's signed invoice, as `buyer` at
    /// time `now`. The invoice must match the order exactly (same
    /// id, the product's seller as recipient, the order's amount)
    /// and the payer must be the order's buyer. On success the
    /// order is marked paid, the receipt is kept, and stock drops
    /// by one. On any failure — bad invoice, already paid, out of
    /// stock, insufficient funds — nothing changes: no coins move
    /// and stock is untouched.
    pub fn pay_order(
        &mut self,
        order_id: &str,
        invoice: &Invoice,
        buyer: &Identity,
        now: u64,
    ) -> Result<Receipt, StoreError> {
        let order = self
            .orders
            .get(order_id)
            .ok_or_else(|| StoreError::OrderNotFound(order_id.to_string()))?
            .clone();
        if order.status != OrderStatus::Pending {
            return Err(StoreError::AlreadyPaid(order_id.to_string()));
        }
        if invoice.id != order.id {
            return Err(StoreError::InvoiceMismatch(format!(
                "invoice '{}' does not match order '{}'",
                invoice.id, order.id
            )));
        }
        if buyer.peer_id() != &order.buyer {
            return Err(StoreError::InvoiceMismatch(
                "the payer is not the order's buyer".to_string(),
            ));
        }
        let product = self
            .products
            .get(&order.product_id)
            .ok_or_else(|| StoreError::ProductNotFound(order.product_id.clone()))?
            .clone();
        if invoice.recipient != product.seller {
            return Err(StoreError::InvoiceMismatch(
                "invoice recipient is not the product's seller".to_string(),
            ));
        }
        if invoice.amount != order.amount {
            return Err(StoreError::InvoiceMismatch(format!(
                "invoice amount {} does not match order amount {}",
                invoice.amount, order.amount
            )));
        }
        if product.stock == 0 {
            return Err(StoreError::OutOfStock(product.id.clone()));
        }
        // Settlement happens here. Only on success does anything
        // else change.
        let receipt = self.payments.pay_invoice(invoice, buyer, now)?;
        if let Some(order) = self.orders.get_mut(order_id) {
            order.status = OrderStatus::Paid;
            order.receipt = Some(receipt.clone());
        }
        if let Some(product) = self.products.get_mut(&order.product_id) {
            product.stock -= 1;
        }
        Ok(receipt)
    }

    /// The order with this id, if any.
    pub fn get_order(&self, id: &str) -> Option<&Order> {
        self.orders.get(id)
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
            Identity::generate("seller").unwrap(),
            Identity::generate("buyer").unwrap(),
        )
    }

    fn funded_store() -> (Store, Identity, Identity) {
        let (seller, buyer) = identities();
        let payments = PaymentService::new(&[
            (seller.peer_id().clone(), 1_000),
            (buyer.peer_id().clone(), 1_000),
        ])
        .unwrap();
        (Store::new(payments), seller, buyer)
    }

    fn list_lamp(store: &mut Store, seller: &Identity, stock: u32) -> Product {
        store
            .list_product(
                "lamp-001",
                "Brass Lamp",
                "A heavy brass lamp.",
                250,
                stock,
                seller,
            )
            .unwrap()
    }

    /// The seller signs the invoice for an order; the buyer pays it
    /// through the store.
    fn pay_the_order(
        store: &mut Store,
        order: &Order,
        seller: &Identity,
        buyer: &Identity,
    ) -> Result<Receipt, StoreError> {
        let invoice = Invoice::new(seller, &order.id, order.amount, "lamp-001", u64::MAX).unwrap();
        store.pay_order(&order.id, &invoice, buyer, 0)
    }

    #[test]
    fn list_render_buy_flow() {
        let (mut store, seller, buyer) = funded_store();
        let product = list_lamp(&mut store, &seller, 2);
        assert!(product.verify());
        assert_eq!(product.stock, 2);

        // The rendered page names the product and its price, and
        // rendering the same product twice is byte-identical.
        let page1 = store.render_product_page("lamp-001").unwrap();
        let page2 = store.render_product_page("lamp-001").unwrap();
        assert_eq!(page1.bytes, page2.bytes);
        assert_eq!(page1.address, page2.address);
        let text = String::from_utf8(page1.bytes).unwrap();
        assert!(text.contains("Brass Lamp"));
        assert!(text.contains("A heavy brass lamp."));
        assert!(text.contains("250"));
        assert!(text.contains("seller"));

        // First unit.
        let order = store.place_order("lamp-001", buyer.peer_id()).unwrap();
        assert_eq!(order.status, OrderStatus::Pending);
        assert_eq!(order.amount, 250);
        let receipt = pay_the_order(&mut store, &order, &seller, &buyer).unwrap();
        assert!(receipt.verify());
        assert_eq!(receipt.amount, 250);
        assert_eq!(
            store.get_order(&order.id).unwrap().status,
            OrderStatus::Paid
        );
        assert!(store
            .get_order(&order.id)
            .unwrap()
            .receipt
            .as_ref()
            .unwrap()
            .verify());
        assert_eq!(store.get_product("lamp-001").unwrap().stock, 1);
        assert_eq!(store.balance_of(seller.peer_id()), 1_250);
        assert_eq!(store.balance_of(buyer.peer_id()), 750);
        assert!(store.payments().is_paid(&order.id));

        // Second unit sells out the listing.
        let order2 = store.place_order("lamp-001", buyer.peer_id()).unwrap();
        pay_the_order(&mut store, &order2, &seller, &buyer).unwrap();
        assert_eq!(store.get_product("lamp-001").unwrap().stock, 0);
        assert_eq!(store.balance_of(seller.peer_id()), 1_500);
        assert_eq!(store.balance_of(buyer.peer_id()), 500);
    }

    #[test]
    fn custom_template_renders_product() {
        let (mut store, seller, _buyer) = funded_store();
        list_lamp(&mut store, &seller, 1);
        let page = store
            .render_product_page_with("lamp-001", "{{name}} costs {{price}} X-Koin.")
            .unwrap();
        assert_eq!(page.bytes, b"Brass Lamp costs 250 X-Koin.");
        assert_eq!(
            store.render_product_page("nope"),
            Err(StoreError::ProductNotFound("nope".to_string()))
        );
    }

    #[test]
    fn insufficient_funds_refused_with_stock_untouched() {
        let (seller, buyer) = identities();
        let payments = PaymentService::new(&[
            (seller.peer_id().clone(), 1_000),
            (buyer.peer_id().clone(), 100),
        ])
        .unwrap();
        let mut store = Store::new(payments);
        list_lamp(&mut store, &seller, 1);
        let order = store.place_order("lamp-001", buyer.peer_id()).unwrap();
        let result = pay_the_order(&mut store, &order, &seller, &buyer);
        assert!(result.is_err());
        // The order is still pending, stock is untouched, and no
        // coins moved.
        assert_eq!(
            store.get_order(&order.id).unwrap().status,
            OrderStatus::Pending
        );
        assert_eq!(store.get_product("lamp-001").unwrap().stock, 1);
        assert_eq!(store.balance_of(buyer.peer_id()), 100);
        assert_eq!(store.balance_of(seller.peer_id()), 1_000);
        assert!(!store.payments().is_paid(&order.id));
    }

    #[test]
    fn out_of_stock_refused() {
        let (mut store, seller, buyer) = funded_store();
        list_lamp(&mut store, &seller, 1);
        let order = store.place_order("lamp-001", buyer.peer_id()).unwrap();
        pay_the_order(&mut store, &order, &seller, &buyer).unwrap();
        assert_eq!(store.get_product("lamp-001").unwrap().stock, 0);
        assert_eq!(
            store.place_order("lamp-001", buyer.peer_id()),
            Err(StoreError::OutOfStock("lamp-001".to_string()))
        );
    }

    #[test]
    fn forged_listing_refused() {
        let (mut store, seller, _buyer) = funded_store();
        let product = list_lamp(&mut store, &seller, 1);
        assert!(product.verify());

        // Tampered terms: the price was changed after signing.
        let mut forged = product.clone();
        forged.price = 1;
        assert!(!forged.verify());
        assert_eq!(
            store.register_product(forged),
            Err(StoreError::BadListingSignature)
        );

        // A signature from anyone but the seller.
        let mallory = Identity::generate("mallory").unwrap();
        let mut forged = product.clone();
        forged.signature = mallory.sign(b"not the listing").to_bytes();
        assert!(!forged.verify());
        assert_eq!(
            store.register_product(forged),
            Err(StoreError::BadListingSignature)
        );

        // The real listing is still there, untouched.
        assert_eq!(store.get_product("lamp-001").unwrap().stock, 1);
    }

    #[test]
    fn invoice_mismatch_refused() {
        let (mut store, seller, buyer) = funded_store();
        list_lamp(&mut store, &seller, 1);
        let order = store.place_order("lamp-001", buyer.peer_id()).unwrap();

        // Wrong amount.
        let wrong_amount = Invoice::new(&seller, &order.id, 100, "lamp-001", u64::MAX).unwrap();
        assert_eq!(
            store.pay_order(&order.id, &wrong_amount, &buyer, 0),
            Err(StoreError::InvoiceMismatch(
                "invoice amount 100 does not match order amount 250".to_string()
            ))
        );
        // Wrong order id.
        let wrong_id = Invoice::new(&seller, "order-999", 250, "lamp-001", u64::MAX).unwrap();
        assert!(matches!(
            store.pay_order(&order.id, &wrong_id, &buyer, 0),
            Err(StoreError::InvoiceMismatch(_))
        ));
        // Wrong recipient: someone else's invoice.
        let other = Identity::generate("other").unwrap();
        let wrong_seller = Invoice::new(&other, &order.id, 250, "lamp-001", u64::MAX).unwrap();
        assert!(matches!(
            store.pay_order(&order.id, &wrong_seller, &buyer, 0),
            Err(StoreError::InvoiceMismatch(_))
        ));
        // Nothing moved and the order is still pending.
        assert_eq!(
            store.get_order(&order.id).unwrap().status,
            OrderStatus::Pending
        );
        assert_eq!(store.get_product("lamp-001").unwrap().stock, 1);
        assert_eq!(store.balance_of(buyer.peer_id()), 1_000);
    }

    #[test]
    fn already_paid_order_refused() {
        let (mut store, seller, buyer) = funded_store();
        list_lamp(&mut store, &seller, 2);
        let order = store.place_order("lamp-001", buyer.peer_id()).unwrap();
        pay_the_order(&mut store, &order, &seller, &buyer).unwrap();
        // Paying the same order again is refused — and the invoice
        // itself is already spent, so the payment service refuses
        // too. Either way, stock does not move twice.
        let invoice = Invoice::new(&seller, &order.id, 250, "lamp-001", u64::MAX).unwrap();
        assert_eq!(
            store.pay_order(&order.id, &invoice, &buyer, 0),
            Err(StoreError::AlreadyPaid("order-1".to_string()))
        );
        assert_eq!(store.get_product("lamp-001").unwrap().stock, 1);
        assert_eq!(store.balance_of(buyer.peer_id()), 750);
    }

    #[test]
    fn listing_rules_enforced() {
        let (mut store, seller, _buyer) = funded_store();
        // Duplicate id.
        list_lamp(&mut store, &seller, 1);
        assert_eq!(
            store.list_product("lamp-001", "Other Lamp", "Another lamp.", 100, 1, &seller),
            Err(StoreError::DuplicateProduct("lamp-001".to_string()))
        );
        // Zero price.
        assert!(matches!(
            store.list_product("free-001", "Free", "Costs nothing.", 0, 1, &seller),
            Err(StoreError::InvalidListing(_))
        ));
        // Bad id characters.
        assert!(matches!(
            store.list_product("bad id!", "Bad", "Bad id.", 10, 1, &seller),
            Err(StoreError::InvalidListing(_))
        ));
        // Unknown product for order and render.
        assert_eq!(
            store.place_order("nope", seller.peer_id()),
            Err(StoreError::ProductNotFound("nope".to_string()))
        );
        assert_eq!(
            store.pay_order(
                "order-999",
                &Invoice::new(&seller, "order-999", 10, "x", u64::MAX).unwrap(),
                &seller,
                0
            ),
            Err(StoreError::OrderNotFound("order-999".to_string()))
        );
    }

    #[test]
    fn list_products_sorted_by_id() {
        let (mut store, seller, _buyer) = funded_store();
        store
            .list_product("zebra-001", "Zebra", "Striped.", 10, 1, &seller)
            .unwrap();
        store
            .list_product("apple-001", "Apple", "Round.", 10, 1, &seller)
            .unwrap();
        let ids: Vec<&str> = store
            .list_products()
            .iter()
            .map(|p| p.id.as_str())
            .collect();
        assert_eq!(ids, vec!["apple-001", "zebra-001"]);
    }
}
