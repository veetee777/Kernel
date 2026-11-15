//! Kai — a real local agent provider for K•AI, built for creator bundles.
//!
//! Kai is a **tool-using agent, not a model**. It routes a task to one
//! of four tools by deterministic keyword intent, runs the tool —
//! which does real work against other workspace crates — and composes
//! the signed response from the tool's actual output. It never invents
//! an answer: an intent it cannot route is refused.
//!
//! ## Routing rules (exact)
//!
//! The prompt is trimmed; the first whitespace-separated word,
//! lowercased, is the intent:
//!
//! * `search <needle>` — storage search. The rest is a
//!   case-insensitive substring; blocks whose UTF-8 text contains it
//!   are listed with their CIDs and excerpts.
//! * `bundle assemble` — assemble a bundle (see
//!   [`AssembleBundleTool`]). The task owner is the creator: the
//!   manifest is signed as them.
//! * `bundle buy <bundle-cid-hex>` — buy a bundle (see
//!   [`BuyBundleTool`]): the creator's invoice is paid through
//!   K•Pay/X•Koin and the buyer receives the manifest and its item
//!   CIDs. A free bundle (price zero) is delivered without payment.
//! * `bundle show <bundle-cid-hex>` — show a bundle manifest without
//!   paying.
//! * `render <template> --- <data>` — render through the HEY•AI
//!   engine (see [`RenderTool`]).
//! * Anything else — refused with a clean error. No tool runs, no
//!   answer is invented.
//!
//! One tool runs per task. Every tool call — success or failure — is
//! recorded on the agent's audit trail ([`ToolCall`]), and the signed
//! response text carries a `tools: <name>` header naming the tool
//! that produced it, so the answer is auditable from the record
//! alone.
//!
//! ## Signing keys
//!
//! Kai is a **local** agent: it signs only for identities registered
//! with it ([`LocalAgent::register_signer`]). A bundle is signed as
//! its task owner, and a purchase is paid as its task owner; if the
//! owner never registered a key with this agent, the tool refuses
//! with [`ToolError::NoSigner`]. Kai never holds a key it was not
//! given, and never signs for a stranger.
//!
//! ## What Kai does not do
//!
//! Kai assembles and sells bundles. It does not create the content
//! inside them — the creator supplies the content, Kai finds it,
//! verifies it, wraps it, renders its page, and settles its sale.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Mutex, MutexGuard};

use hey_ai::{render_addressed, RenderError, Value as HeyValue};
use k_pay::{PayError, PaymentService};
use kernel_crypto::sha256;
use kernel_identity::{Identity, PeerId};
use kernel_storage::{Cid, ContentStore, StorageError};

use crate::bundle::{BundleError, BundleItem, BundleManifest};
use crate::{AiError, AiProvider, AiTask};

/// Maximum storage-search hits reported in one tool output.
pub const MAX_SEARCH_RESULTS: usize = 20;
/// Characters of tool output kept in a trail entry's detail.
const TRAIL_DETAIL_LEN: usize = 200;
/// Characters of a stored block shown as a search excerpt.
const EXCERPT_LEN: usize = 80;

/// Default bundle page template, used when the render input carries
/// no template of its own. It reads the bundle fields `title`,
/// `description`, `creator`, `price`, and `items` (each with `name`,
/// `kind`, `cid`).
pub const DEFAULT_BUNDLE_TEMPLATE: &str = "<h1>{{title}}</h1>\n<p>{{description}}</p>\n<p>By {{creator}} — {{price}} X•Koin</p>\n<ul>\n{{#each items}}<li>{{name}} ({{kind}}) — {{cid}}</li>\n{{/each}}</ul>";

/// Lock a mutex, recovering from poisoning instead of panicking: a
/// poisoned lock means a previous holder panicked, and the state it
/// guarded is still the state to keep working with.
fn lock<'a, T>(mutex: &'a Mutex<T>) -> MutexGuard<'a, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Errors a tool can return. Every failure is a value, never a panic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolError {
    /// The tool's input was empty or malformed.
    InvalidInput(String),
    /// No signing key is registered for the peer the tool must act as.
    NoSigner(String),
    /// Rendering failed.
    Render(RenderError),
    /// A bundle operation failed.
    Bundle(BundleError),
    /// A payment operation failed.
    Pay(PayError),
    /// Storage failed.
    Storage(StorageError),
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            ToolError::NoSigner(msg) => write!(f, "no signing key: {msg}"),
            ToolError::Render(e) => write!(f, "render error: {e}"),
            ToolError::Bundle(e) => write!(f, "bundle error: {e}"),
            ToolError::Pay(e) => write!(f, "payment error: {e}"),
            ToolError::Storage(e) => write!(f, "storage error: {e}"),
        }
    }
}

impl std::error::Error for ToolError {}

impl From<RenderError> for ToolError {
    fn from(e: RenderError) -> Self {
        ToolError::Render(e)
    }
}

impl From<BundleError> for ToolError {
    fn from(e: BundleError) -> Self {
        ToolError::Bundle(e)
    }
}

impl From<PayError> for ToolError {
    fn from(e: PayError) -> Self {
        ToolError::Pay(e)
    }
}

impl From<StorageError> for ToolError {
    fn from(e: StorageError) -> Self {
        ToolError::Storage(e)
    }
}

/// A tool Kai can run: a name, a description, and a `run` that does
/// real work and returns real output. Tools never invent answers — a
/// tool that cannot do what was asked returns an error.
pub trait Tool {
    /// The tool's name, as it appears in responses and on the trail.
    fn name(&self) -> &'static str;
    /// One line saying what the tool does.
    fn description(&self) -> &str;
    /// Run the tool on `input`. The input format is documented on
    /// each tool. Takes `&mut self` because tools that assemble and
    /// sell bundles mutate the store, the bundle registry, and the
    /// payment service.
    fn run(&mut self, input: &str) -> Result<String, ToolError>;
}

/// Load a bundle manifest by CID: from the agent's registry first,
/// then from the content store (decoding and verifying the stored
/// manifest). The returned manifest always verifies, and its bytes
/// always hash to the CID asked for.
fn load_manifest(
    store: &ContentStore,
    bundles: &HashMap<Cid, BundleManifest>,
    cid: &Cid,
) -> Result<BundleManifest, ToolError> {
    if let Some(manifest) = bundles.get(cid) {
        return Ok(manifest.clone());
    }
    let bytes = store.get(cid).map_err(|e| match e {
        StorageError::NotFound(_) => ToolError::InvalidInput(format!("no bundle at {cid}")),
        other => ToolError::Storage(other),
    })?;
    let manifest = BundleManifest::decode(&bytes)?;
    if !manifest.verify() {
        return Err(ToolError::Bundle(BundleError::BadSignature));
    }
    if manifest.cid() != *cid {
        return Err(ToolError::InvalidInput(format!(
            "manifest bytes do not hash to {cid}"
        )));
    }
    Ok(manifest)
}

/// The human-readable form of a manifest, shared by the show and buy
/// tools.
fn format_manifest(manifest: &BundleManifest) -> String {
    let mut out = format!(
        "bundle \"{}\" — {}\n{}\nprice: {} X•Koin\ncreator: {}\nmanifest: {}\nitems ({}):",
        manifest.id,
        manifest.title,
        manifest.description,
        manifest.price,
        manifest.creator.display_name(),
        manifest.cid().to_hex(),
        manifest.items.len()
    );
    for item in &manifest.items {
        out.push_str(&format!(
            "\n- {} ({}) {}",
            item.name,
            item.kind,
            item.cid.to_hex()
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// Storage search
// ---------------------------------------------------------------------------

/// Search content-addressed storage.
///
/// Input is a needle: a case-insensitive substring. The tool scans
/// every block in the store (the store has no search, so the tool
/// does the scan), skips blocks that are not valid UTF-8, and lists
/// every block whose text contains the needle, with its CID and an
/// 80-character excerpt. Hits are ordered by CID hex, so the output
/// is deterministic. At most [`MAX_SEARCH_RESULTS`] hits are listed.
pub struct StorageSearchTool<'a> {
    store: &'a ContentStore,
}

impl<'a> StorageSearchTool<'a> {
    /// Build the tool over a store.
    pub fn new(store: &'a ContentStore) -> Self {
        Self { store }
    }
}

impl<'a> Tool for StorageSearchTool<'a> {
    fn name(&self) -> &'static str {
        "storage-search"
    }

    fn description(&self) -> &str {
        "Search the agent's content store for blocks whose UTF-8 text contains a case-insensitive substring, returning matching CIDs with excerpts."
    }

    fn run(&mut self, input: &str) -> Result<String, ToolError> {
        let needle = input.trim();
        if needle.is_empty() {
            return Err(ToolError::InvalidInput(
                "search needle must not be empty".to_string(),
            ));
        }
        let needle_lower = needle.to_lowercase();
        let mut hits: Vec<(Cid, String)> = Vec::new();
        for (cid, bytes) in self.store.blocks() {
            let Ok(text) = std::str::from_utf8(&bytes) else {
                continue;
            };
            if text.to_lowercase().contains(&needle_lower) {
                hits.push((cid, excerpt(text)));
            }
        }
        hits.sort_by_key(|(cid, _)| *cid.as_bytes());
        if hits.is_empty() {
            return Ok(format!("no blocks contain \"{needle}\""));
        }
        let mut out = format!(
            "{} block{} contain \"{}\":",
            hits.len(),
            if hits.len() == 1 { "" } else { "s" },
            needle
        );
        for (cid, ex) in hits.iter().take(MAX_SEARCH_RESULTS) {
            out.push_str(&format!("\n{}: {}", cid.to_hex(), ex));
        }
        if hits.len() > MAX_SEARCH_RESULTS {
            out.push_str(&format!(
                "\n... and {} more",
                hits.len() - MAX_SEARCH_RESULTS
            ));
        }
        Ok(out)
    }
}

/// The first 80 characters of a block's text, newlines flattened to
/// spaces.
fn excerpt(text: &str) -> String {
    let flat: String = text
        .chars()
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .collect();
    flat.chars().take(EXCERPT_LEN).collect()
}

// ---------------------------------------------------------------------------
// Assemble bundle
// ---------------------------------------------------------------------------

/// Assemble a creator bundle.
///
/// Input is a bundle spec, one item per line:
///
/// ```text
/// <id>
/// <title>
/// <description>
/// <price>
/// <name>|<kind>|<cid-hex>
/// <name>|<kind>|<cid-hex>
/// ...
/// ```
///
/// Every CID is verified before the manifest is built: the block
/// must exist in the store, and its bytes must hash to the CID. A
/// missing block or a corrupt item (bytes that do not hash to their
/// CID) refuses the whole bundle — nothing is assembled. The
/// manifest is signed as the task owner, stored as a block, and
/// registered; its CID is the bundle's address.
pub struct AssembleBundleTool<'a> {
    store: &'a mut ContentStore,
    signers: &'a HashMap<[u8; 32], Identity>,
    owner: &'a PeerId,
    bundles: &'a mut HashMap<Cid, BundleManifest>,
}

impl<'a> AssembleBundleTool<'a> {
    /// Build the tool. The manifest is signed as `owner`, who must
    /// be registered in `signers`.
    pub fn new(
        store: &'a mut ContentStore,
        signers: &'a HashMap<[u8; 32], Identity>,
        owner: &'a PeerId,
        bundles: &'a mut HashMap<Cid, BundleManifest>,
    ) -> Self {
        Self {
            store,
            signers,
            owner,
            bundles,
        }
    }
}

impl<'a> Tool for AssembleBundleTool<'a> {
    fn name(&self) -> &'static str {
        "assemble-bundle"
    }

    fn description(&self) -> &str {
        "Assemble a signed creator bundle manifest from content CIDs, verifying every item exists and hash-checks first."
    }

    fn run(&mut self, input: &str) -> Result<String, ToolError> {
        let mut lines = input.lines();
        let spec = |lines: &mut std::str::Lines<'_>, what: &str| -> Result<String, ToolError> {
            lines
                .next()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .ok_or_else(|| {
                    ToolError::InvalidInput(format!("bundle spec is missing its {what} line"))
                })
        };
        let id = spec(&mut lines, "id")?;
        let title = spec(&mut lines, "title")?;
        let description = spec(&mut lines, "description")?;
        let price: u64 = spec(&mut lines, "price")?
            .parse()
            .map_err(|_| ToolError::InvalidInput("bundle price must be a number".to_string()))?;
        let mut items = Vec::new();
        for line in lines {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split('|').collect();
            if parts.len() != 3 {
                return Err(ToolError::InvalidInput(format!(
                    "bundle item line must be name|kind|cid-hex, got: {line}"
                )));
            }
            let name = parts[0].trim();
            let kind = parts[1].trim();
            let cid = Cid::from_hex(parts[2].trim())
                .map_err(|e| ToolError::InvalidInput(format!("bad item CID: {e}")))?;
            // Verify the item before the manifest is built: the block
            // must exist, and its bytes must hash to the CID.
            let bytes = self.store.get(&cid).map_err(|e| match e {
                StorageError::NotFound(_) => ToolError::InvalidInput(format!(
                    "item \"{name}\": block {cid} not found in the store"
                )),
                other => ToolError::Storage(other),
            })?;
            if Cid::from_bytes(sha256(&bytes)) != cid {
                return Err(ToolError::InvalidInput(format!(
                    "item \"{name}\": bytes do not hash to {cid}"
                )));
            }
            items.push(BundleItem::new(name, kind, cid)?);
        }
        let signer = self
            .signers
            .get(self.owner.public_key().as_bytes())
            .ok_or_else(|| {
                ToolError::NoSigner(format!(
                    "no signing key registered for {}",
                    self.owner.display_name()
                ))
            })?;
        let manifest = BundleManifest::new(&id, &title, &description, price, items, signer)?;
        let cid = manifest.cid();
        let stored = self.store.put(&manifest.encode());
        assert_eq!(stored, cid, "manifest bytes must hash to the manifest CID");
        self.bundles.insert(cid, manifest.clone());
        Ok(format!(
            "assembled {}:\n{}",
            cid.to_hex(),
            format_manifest(&manifest)
        ))
    }
}

// ---------------------------------------------------------------------------
// Show bundle
// ---------------------------------------------------------------------------

/// Show a bundle manifest without paying.
///
/// Input is a bundle CID in hex. The manifest is loaded from the
/// agent's registry or the store and displayed: id, title,
/// description, price, creator, manifest CID, and every item with
/// its kind and CID.
pub struct ShowBundleTool<'a> {
    store: &'a ContentStore,
    bundles: &'a HashMap<Cid, BundleManifest>,
}

impl<'a> ShowBundleTool<'a> {
    /// Build the tool.
    pub fn new(store: &'a ContentStore, bundles: &'a HashMap<Cid, BundleManifest>) -> Self {
        Self { store, bundles }
    }
}

impl<'a> Tool for ShowBundleTool<'a> {
    fn name(&self) -> &'static str {
        "show-bundle"
    }

    fn description(&self) -> &str {
        "Show a bundle manifest by CID, without paying."
    }

    fn run(&mut self, input: &str) -> Result<String, ToolError> {
        let hex = input.trim();
        if hex.is_empty() {
            return Err(ToolError::InvalidInput(
                "show needs a bundle CID".to_string(),
            ));
        }
        let cid = Cid::from_hex(hex)
            .map_err(|e| ToolError::InvalidInput(format!("bad bundle CID: {e}")))?;
        let manifest = load_manifest(self.store, self.bundles, &cid)?;
        Ok(format_manifest(&manifest))
    }
}

// ---------------------------------------------------------------------------
// Buy bundle
// ---------------------------------------------------------------------------

/// Buy a bundle.
///
/// Input is a bundle CID in hex. The buyer is the task owner: the
/// creator's invoice for the bundle's price is paid through K•Pay /
/// X•Koin from the buyer's balance, and the buyer receives the
/// manifest and its item CIDs. A free bundle (price zero) is
/// delivered without payment. When the buyer is the registered
/// owner of the agent's payment service (see
/// [`LocalAgent::set_owner`]), the purchase is owner-free under
/// the OWNER-FREE LAW: the invoice is satisfied without moving
/// any coins. The invoice id is
/// `buy-<bundle-id>-<buyer-fingerprint>`; an already-paid invoice
/// means this buyer already bought this bundle, and the purchase is
/// refused instead of charging twice. A failed payment — bad
/// invoice, insufficient funds — moves nothing: K•Pay settles
/// atomically, and the invoice stays unpaid.
///
/// Invoices never expire here: the agent has no wall clock, so the
/// invoice expiry and the payment time are both `u64::MAX`.
pub struct BuyBundleTool<'a> {
    store: &'a ContentStore,
    pay: &'a mut PaymentService,
    signers: &'a HashMap<[u8; 32], Identity>,
    bundles: &'a HashMap<Cid, BundleManifest>,
    buyer: &'a PeerId,
}

impl<'a> BuyBundleTool<'a> {
    /// Build the tool. The purchase is paid as `buyer`, who must be
    /// registered in `signers`; the invoice is signed by the
    /// bundle's creator, who must be registered too.
    pub fn new(
        store: &'a ContentStore,
        pay: &'a mut PaymentService,
        signers: &'a HashMap<[u8; 32], Identity>,
        bundles: &'a HashMap<Cid, BundleManifest>,
        buyer: &'a PeerId,
    ) -> Self {
        Self {
            store,
            pay,
            signers,
            bundles,
            buyer,
        }
    }
}

impl<'a> Tool for BuyBundleTool<'a> {
    fn name(&self) -> &'static str {
        "buy-bundle"
    }

    fn description(&self) -> &str {
        "Buy a bundle by CID: pay the creator's invoice through K-Pay/X-Koin and receive the manifest and its item CIDs."
    }

    fn run(&mut self, input: &str) -> Result<String, ToolError> {
        let hex = input.trim();
        if hex.is_empty() {
            return Err(ToolError::InvalidInput(
                "buy needs a bundle CID".to_string(),
            ));
        }
        let cid = Cid::from_hex(hex)
            .map_err(|e| ToolError::InvalidInput(format!("bad bundle CID: {e}")))?;
        let manifest = load_manifest(self.store, self.bundles, &cid)?;
        if manifest.price == 0 {
            return Ok(format!(
                "bought {}:\n{}\nreceipt: free — no payment needed",
                cid.to_hex(),
                format_manifest(&manifest)
            ));
        }
        let buyer_signer = self
            .signers
            .get(self.buyer.public_key().as_bytes())
            .ok_or_else(|| {
                ToolError::NoSigner(format!(
                    "no signing key registered for buyer {}",
                    self.buyer.display_name()
                ))
            })?;
        let invoice_id = format!("buy-{}-{}", manifest.id, self.buyer.fingerprint());
        if self.pay.is_paid(&invoice_id) {
            return Err(ToolError::InvalidInput(format!(
                "bundle \"{}\" already purchased by this buyer",
                manifest.id
            )));
        }
        let invoice = match self.pay.get_invoice(&invoice_id) {
            Some(invoice) => invoice.clone(),
            None => {
                let creator_signer = self
                    .signers
                    .get(manifest.creator.public_key().as_bytes())
                    .ok_or_else(|| {
                        ToolError::NoSigner(format!(
                            "no signing key registered for creator {}",
                            manifest.creator.display_name()
                        ))
                    })?;
                self.pay.create_invoice(
                    creator_signer,
                    &invoice_id,
                    manifest.price,
                    &format!("bundle:{}", manifest.id),
                    u64::MAX,
                )?
            }
        };
        let receipt = self.pay.pay_invoice(&invoice, buyer_signer, u64::MAX)?;
        // An owner-free purchase is marked as such in the output:
        // the receipt says so, and no coins moved.
        let free_note = if receipt.owner_free {
            " (owner-free — no coins moved)"
        } else {
            ""
        };
        Ok(format!(
            "bought {}:\n{}\nreceipt: {} paid by {} to {}{}",
            cid.to_hex(),
            format_manifest(&manifest),
            receipt.invoice_id,
            receipt.payer.display_name(),
            receipt.recipient.display_name(),
            free_note
        ))
    }
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

/// Render through the HEY•AI engine.
///
/// Input is a template, then a line that is exactly `---`, then data
/// lines. If there is no `---` line, the whole input is the template
/// and the data is empty.
///
/// ```text
/// <template lines...>
/// ---
/// bundle=<bundle-cid-hex>
/// key=value
/// key=value
/// ```
///
/// The template language is HEY•AI's: `{{name}}` substitution,
/// `{{#each items}}...{{/each}}` loops, `{{#if flag}}...{{/if}}`
/// conditionals. If the template part is empty, the default bundle
/// page template ([`DEFAULT_BUNDLE_TEMPLATE`]) is used, which
/// requires a `bundle=` line. A `bundle=` line loads the manifest
/// and adds its fields (`id`, `title`, `description`, `price`,
/// `creator`, `items` — each item a map of `name`, `kind`, `cid`);
/// bundle fields win over `key=value` extras with the same name.
/// Extra values are `true`/`false` for booleans, integers for
/// numbers, anything else for text.
///
/// The output is the rendered text plus its content address:
/// `rendered <address>:\n<text>`. Rendering is deterministic — the
/// same template plus the same data always yields the same bytes and
/// the same address.
pub struct RenderTool<'a> {
    store: &'a ContentStore,
    bundles: &'a HashMap<Cid, BundleManifest>,
}

impl<'a> RenderTool<'a> {
    /// Build the tool. The store and registry are only used to
    /// resolve `bundle=` lines.
    pub fn new(store: &'a ContentStore, bundles: &'a HashMap<Cid, BundleManifest>) -> Self {
        Self { store, bundles }
    }
}

impl<'a> Tool for RenderTool<'a> {
    fn name(&self) -> &'static str {
        "render"
    }

    fn description(&self) -> &str {
        "Render a template through the HEY-AI engine, optionally with a bundle's fields as data, returning the rendered text and its content address."
    }

    fn run(&mut self, input: &str) -> Result<String, ToolError> {
        let mut template_lines: Vec<&str> = Vec::new();
        let mut data_lines: Vec<&str> = Vec::new();
        let mut seen_sep = false;
        for line in input.lines() {
            if !seen_sep && line == "---" {
                seen_sep = true;
                continue;
            }
            if seen_sep {
                data_lines.push(line);
            } else {
                template_lines.push(line);
            }
        }
        let mut bundle_cid: Option<Cid> = None;
        let mut extras: Vec<(String, HeyValue)> = Vec::new();
        for line in data_lines {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(hex) = line.strip_prefix("bundle=") {
                bundle_cid = Some(
                    Cid::from_hex(hex.trim())
                        .map_err(|e| ToolError::InvalidInput(format!("bad bundle CID: {e}")))?,
                );
                continue;
            }
            let (key, value) = line.split_once('=').ok_or_else(|| {
                ToolError::InvalidInput(format!("render data line has no '=': {line}"))
            })?;
            let key = key.trim();
            if key.is_empty() {
                return Err(ToolError::InvalidInput(
                    "render data line has an empty key".to_string(),
                ));
            }
            extras.push((key.to_string(), parse_value(value.trim())));
        }
        let template_is_empty = template_lines.iter().all(|l| l.trim().is_empty());
        let template =
            if template_is_empty {
                match bundle_cid {
                    Some(_) => DEFAULT_BUNDLE_TEMPLATE.to_string(),
                    None => return Err(ToolError::InvalidInput(
                        "render needs a template, or bundle=<cid> to use the default bundle page"
                            .to_string(),
                    )),
                }
            } else {
                template_lines.join("\n")
            };
        let mut pairs: Vec<(String, HeyValue)> = Vec::new();
        if let Some(cid) = bundle_cid {
            let manifest = load_manifest(self.store, self.bundles, &cid)?;
            pairs = bundle_data(&manifest);
        }
        // Extras go after the bundle fields, so a bundle field wins
        // on lookup when a name collides.
        pairs.extend(extras);
        let data = HeyValue::map(pairs);
        let out = render_addressed(&template, &data)?;
        let text = String::from_utf8(out.bytes).map_err(|_| {
            ToolError::InvalidInput("rendered output is not valid UTF-8".to_string())
        })?;
        Ok(format!("rendered {}:\n{}", out.address.to_hex(), text))
    }
}

/// Parse one `key=value` data value: `true`/`false` are booleans,
/// integers are numbers, anything else is text.
fn parse_value(s: &str) -> HeyValue {
    match s {
        "true" => HeyValue::bool(true),
        "false" => HeyValue::bool(false),
        _ => match s.parse::<i64>() {
            Ok(n) => HeyValue::number(n),
            Err(_) => HeyValue::text(s),
        },
    }
}

/// A bundle manifest as template data: scalar fields plus `items` as
/// a list of maps with `name`, `kind`, and `cid`.
fn bundle_data(manifest: &BundleManifest) -> Vec<(String, HeyValue)> {
    let items: Vec<HeyValue> = manifest
        .items
        .iter()
        .map(|item| {
            HeyValue::map(vec![
                ("name".to_string(), HeyValue::text(&item.name)),
                ("kind".to_string(), HeyValue::text(&item.kind)),
                ("cid".to_string(), HeyValue::text(&item.cid.to_hex())),
            ])
        })
        .collect();
    vec![
        ("id".to_string(), HeyValue::text(&manifest.id)),
        ("title".to_string(), HeyValue::text(&manifest.title)),
        (
            "description".to_string(),
            HeyValue::text(&manifest.description),
        ),
        (
            "price".to_string(),
            HeyValue::number((manifest.price.min(i64::MAX as u64)) as i64),
        ),
        (
            "creator".to_string(),
            HeyValue::text(manifest.creator.display_name()),
        ),
        ("items".to_string(), HeyValue::list(items)),
    ]
}

// ---------------------------------------------------------------------------
// The agent
// ---------------------------------------------------------------------------

/// One recorded tool call on the audit trail: which task, which tool,
/// whether it worked, and what it produced (or why it failed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    /// The task id the tool ran for.
    pub task_id: String,
    /// The tool's name.
    pub tool: String,
    /// True if the tool succeeded.
    pub success: bool,
    /// The first 200 characters of the tool's output, or the error
    /// message if it failed.
    pub detail: String,
}

/// Kai, the creator-bundle agent: a real local provider that routes
/// tasks to tools by deterministic keyword intent.
///
/// Kai owns a content store (the creator's content lives here), a
/// K•Pay service (its ledger starts at the genesis allocation given
/// to [`LocalAgent::new`]), a registry of signing identities, and a
/// registry of assembled bundles. Each piece is behind its own lock;
/// tools borrow what they need while they run, in the fixed order
/// signers, store, bundles, payment service.
///
/// [`LocalAgent::new`] takes the genesis allocation: the only way
/// coins enter the ledger. A buyer with no genesis funds cannot buy
/// anything — fund buyers at construction.
pub struct LocalAgent {
    store: Mutex<ContentStore>,
    pay: Mutex<PaymentService>,
    signers: Mutex<HashMap<[u8; 32], Identity>>,
    bundles: Mutex<HashMap<Cid, BundleManifest>>,
    trail: Mutex<Vec<ToolCall>>,
}

impl LocalAgent {
    /// A new agent. `genesis` funds the payment ledger, e.g.
    /// `&[(creator_peer, 0), (buyer_peer, 10_000)]`. Fails only if
    /// the genesis allocation overflows.
    pub fn new(genesis: &[(PeerId, u64)]) -> Result<Self, PayError> {
        Ok(Self {
            store: Mutex::new(ContentStore::new()),
            pay: Mutex::new(PaymentService::new(genesis)?),
            signers: Mutex::new(HashMap::new()),
            bundles: Mutex::new(HashMap::new()),
            trail: Mutex::new(Vec::new()),
        })
    }

    /// Register a signing identity: Kai may now sign and pay as this
    /// peer. Returns the peer id. Registering is the only way a key
    /// enters the agent — Kai never holds a key it was not given.
    pub fn register_signer(&self, identity: Identity) -> PeerId {
        let peer = identity.peer_id().clone();
        lock(&self.signers).insert(*peer.public_key().as_bytes(), identity);
        peer
    }

    /// Register the owner of this node on the agent's payment
    /// service: the peer for whom everything inside KERNEL is free
    /// (the OWNER-FREE LAW). Owner bundle purchases then go through
    /// the exempt path in K•Pay — no coins move. Must be set before
    /// the owner registers a subscription for its occurrences to
    /// be owner-free (see K•Pay).
    pub fn set_owner(&self, owner: PeerId) {
        lock(&self.pay).set_owner(owner);
    }

    /// Store bytes in the agent's content store, returning their CID.
    /// This is how a creator's content gets in: Kai never creates
    /// content, it only stores what it is given.
    pub fn put_block(&self, data: &[u8]) -> Cid {
        lock(&self.store).put(data)
    }

    /// The balance of a peer in the agent's payment ledger.
    pub fn balance_of(&self, peer: &PeerId) -> u64 {
        lock(&self.pay).balance_of(peer)
    }

    /// The assembled bundle with this CID, if this agent assembled
    /// it (or loaded it).
    pub fn bundle(&self, cid: &Cid) -> Option<BundleManifest> {
        lock(&self.bundles).get(cid).cloned()
    }

    /// The full audit trail, in the order tool calls happened.
    pub fn trail(&self) -> Vec<ToolCall> {
        lock(&self.trail).clone()
    }

    /// The trail entries for one task, in order.
    pub fn trail_for(&self, task_id: &str) -> Vec<ToolCall> {
        lock(&self.trail)
            .iter()
            .filter(|c| c.task_id == task_id)
            .cloned()
            .collect()
    }

    /// Record one tool call on the trail.
    fn record(&self, task_id: &str, tool: &str, result: &Result<String, ToolError>) {
        let (success, detail) = match result {
            Ok(out) => (true, out.chars().take(TRAIL_DETAIL_LEN).collect()),
            Err(e) => (false, e.to_string()),
        };
        lock(&self.trail).push(ToolCall {
            task_id: task_id.to_string(),
            tool: tool.to_string(),
            success,
            detail,
        });
    }

    /// Run one tool for a task and turn its result into a response:
    /// success composes `tools: <name>\n---\n<output>`; failure is a
    /// clean provider error. The trail is recorded either way.
    fn finish(
        &self,
        task: &AiTask,
        tool_name: &str,
        result: Result<String, ToolError>,
    ) -> Result<String, AiError> {
        self.record(&task.id, tool_name, &result);
        match result {
            Ok(output) => Ok(format!("tools: {tool_name}\n---\n{output}")),
            Err(e) => Err(AiError::Provider(format!(
                "local-agent: {tool_name} failed: {e}"
            ))),
        }
    }

    /// Test-only: store bytes under a caller-supplied CID without
    /// verifying the hash, to simulate a corrupt item.
    #[cfg(test)]
    pub fn put_unchecked_block(&self, cid: &Cid, data: Vec<u8>) {
        lock(&self.store).put_unchecked(cid, data);
    }
}

impl AiProvider for LocalAgent {
    fn name(&self) -> &str {
        "local-agent"
    }

    fn run(&self, task: &AiTask) -> Result<String, AiError> {
        let prompt = task.prompt.trim();
        let first = prompt.split_whitespace().next().unwrap_or("");
        let rest = prompt[first.len()..].trim();
        match first.to_lowercase().as_str() {
            "search" => {
                let store = lock(&self.store);
                let mut tool = StorageSearchTool::new(&store);
                self.finish(task, "storage-search", tool.run(rest))
            }
            "render" => {
                let store = lock(&self.store);
                let bundles = lock(&self.bundles);
                let mut tool = RenderTool::new(&store, &bundles);
                self.finish(task, "render", tool.run(rest))
            }
            "bundle" => {
                let second = rest.split_whitespace().next().unwrap_or("");
                let sub_rest = rest[second.len()..].trim();
                match second.to_lowercase().as_str() {
                    "assemble" => {
                        let signers = lock(&self.signers);
                        let mut store = lock(&self.store);
                        let mut bundles = lock(&self.bundles);
                        let mut tool = AssembleBundleTool::new(
                            &mut store,
                            &signers,
                            &task.owner,
                            &mut bundles,
                        );
                        self.finish(task, "assemble-bundle", tool.run(sub_rest))
                    }
                    "buy" => {
                        let signers = lock(&self.signers);
                        let store = lock(&self.store);
                        let bundles = lock(&self.bundles);
                        let mut pay = lock(&self.pay);
                        let mut tool = BuyBundleTool::new(
                            &store,
                            &mut pay,
                            &signers,
                            &bundles,
                            &task.owner,
                        );
                        self.finish(task, "buy-bundle", tool.run(sub_rest))
                    }
                    "show" => {
                        let store = lock(&self.store);
                        let bundles = lock(&self.bundles);
                        let mut tool = ShowBundleTool::new(&store, &bundles);
                        self.finish(task, "show-bundle", tool.run(sub_rest))
                    }
                    _ => Err(AiError::Provider(
                        "local-agent: unknown bundle command; use 'bundle assemble', 'bundle buy', or 'bundle show'".to_string(),
                    )),
                }
            }
            _ => Err(AiError::Provider(
                "local-agent: no tool for this task; start the prompt with 'search', 'bundle', or 'render'".to_string(),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AiService, TestProvider};

    const TRACK: &[u8] = b"track one audio bytes";
    const ART: &[u8] = b"cover art bytes";

    /// A fixture: an agent with a funded buyer, a creator and a buyer
    /// registered as signers, and three blocks stored. The original
    /// identities are kept for signing tasks; the agent holds copies
    /// rebuilt from the same secret bytes.
    struct Fixture {
        agent: LocalAgent,
        creator: Identity,
        buyer: Identity,
        creator_peer: PeerId,
        buyer_peer: PeerId,
        track_cid: Cid,
        art_cid: Cid,
    }

    fn fixture() -> Fixture {
        fixture_with_funds(10_000)
    }

    fn fixture_with_funds(buyer_funds: u64) -> Fixture {
        let creator = Identity::generate("creator").unwrap();
        let buyer = Identity::generate("buyer").unwrap();
        let creator_peer = creator.peer_id().clone();
        let buyer_peer = buyer.peer_id().clone();
        let agent =
            LocalAgent::new(&[(creator_peer.clone(), 0), (buyer_peer.clone(), buyer_funds)])
                .unwrap();
        let creator_secret = creator.secret_bytes();
        let buyer_secret = buyer.secret_bytes();
        agent.register_signer(Identity::from_secret_bytes(&creator_secret, "creator").unwrap());
        agent.register_signer(Identity::from_secret_bytes(&buyer_secret, "buyer").unwrap());
        let track_cid = agent.put_block(TRACK);
        let art_cid = agent.put_block(ART);
        agent.put_block(b"nothing relevant here");
        Fixture {
            agent,
            creator,
            buyer,
            creator_peer,
            buyer_peer,
            track_cid,
            art_cid,
        }
    }

    /// Run one task through the service with the agent and return the
    /// signed response text.
    fn run_task_text(
        service: &mut AiService,
        agent: &LocalAgent,
        provider_id: &Identity,
        task_id: &str,
    ) -> Result<String, AiError> {
        let cid = service.run_task(task_id, agent, provider_id)?;
        Ok(service.get_response(&cid)?.text)
    }

    fn assemble_prompt(track: &Cid, art: &Cid) -> String {
        format!(
            "bundle assemble\nsummer-tracks-001\nSummer Tracks\nTwo songs and a cover.\n500\nopening-track|track|{}\ncover|art|{}",
            track.to_hex(),
            art.to_hex()
        )
    }

    /// Assemble a bundle in the fixture and return its manifest CID.
    fn assemble_in(f: &Fixture, service: &mut AiService, provider_id: &Identity) -> Cid {
        service
            .submit_task(
                "assemble-1",
                &assemble_prompt(&f.track_cid, &f.art_cid),
                &f.creator,
            )
            .unwrap();
        let text = run_task_text(service, &f.agent, provider_id, "assemble-1").unwrap();
        assert!(text.starts_with("tools: assemble-bundle\n---\nassembled "));
        let hex = text
            .lines()
            .nth(2)
            .unwrap()
            .strip_prefix("assembled ")
            .unwrap()
            .trim_end_matches(':');
        let cid = Cid::from_hex(hex).unwrap();
        assert!(f.agent.bundle(&cid).is_some());
        cid
    }

    #[test]
    fn routing_picks_the_right_tool_for_each_intent() {
        let f = fixture();
        let provider_id = Identity::generate("provider").unwrap();
        let mut service = AiService::new();
        // search
        service
            .submit_task("t-search", "search track one", &f.creator)
            .unwrap();
        let text = run_task_text(&mut service, &f.agent, &provider_id, "t-search").unwrap();
        assert!(text.starts_with("tools: storage-search\n---\n"));
        // assemble + show + render + buy, in order
        let bundle_cid = assemble_in(&f, &mut service, &provider_id);
        service
            .submit_task(
                "t-show",
                &format!("bundle show {}", bundle_cid.to_hex()),
                &f.creator,
            )
            .unwrap();
        let text = run_task_text(&mut service, &f.agent, &provider_id, "t-show").unwrap();
        assert!(text.starts_with("tools: show-bundle\n---\n"));
        assert!(text.contains("Summer Tracks"));
        service
            .submit_task(
                "t-render",
                &format!("render ---\nbundle={}", bundle_cid.to_hex()),
                &f.creator,
            )
            .unwrap();
        let text = run_task_text(&mut service, &f.agent, &provider_id, "t-render").unwrap();
        assert!(text.starts_with("tools: render\n---\nrendered "));
        service
            .submit_task(
                "t-buy",
                &format!("bundle buy {}", bundle_cid.to_hex()),
                &f.buyer,
            )
            .unwrap();
        let text = run_task_text(&mut service, &f.agent, &provider_id, "t-buy").unwrap();
        assert!(text.starts_with("tools: buy-bundle\n---\nbought "));
        // The trail records every tool, in order.
        let trail = f.agent.trail();
        let names: Vec<&str> = trail.iter().map(|c| c.tool.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "storage-search",
                "assemble-bundle",
                "show-bundle",
                "render",
                "buy-bundle"
            ]
        );
        assert!(trail.iter().all(|c| c.success));
    }

    #[test]
    fn storage_search_finds_stored_blocks_and_misses_nothing() {
        let f = fixture();
        let provider_id = Identity::generate("provider").unwrap();
        let mut service = AiService::new();
        service
            .submit_task("t1", "search bytes", &f.creator)
            .unwrap();
        let text = run_task_text(&mut service, &f.agent, &provider_id, "t1").unwrap();
        // "bytes" is in the track and art blocks, not in the third.
        assert!(text.contains("2 blocks contain"));
        assert!(text.contains(&f.track_cid.to_hex()));
        assert!(text.contains(&f.art_cid.to_hex()));
        assert!(text.contains("track one audio bytes"));
        // Search is case-insensitive.
        service
            .submit_task("t2", "SEARCH TRACK ONE", &f.creator)
            .unwrap();
        let text = run_task_text(&mut service, &f.agent, &provider_id, "t2").unwrap();
        assert!(text.contains(&f.track_cid.to_hex()));
        // A needle that matches nothing is a successful empty result.
        service
            .submit_task("t3", "search zebra", &f.creator)
            .unwrap();
        let text = run_task_text(&mut service, &f.agent, &provider_id, "t3").unwrap();
        assert!(text.contains("no blocks contain \"zebra\""));
    }

    #[test]
    fn assemble_refuses_a_missing_cid() {
        let f = fixture();
        let provider_id = Identity::generate("provider").unwrap();
        let mut service = AiService::new();
        let missing = Cid::from_bytes(sha256(b"never stored"));
        let prompt = format!(
            "bundle assemble\nghost-bundle\nGhost\nNo such content.\n100\ntrack|track|{}",
            missing.to_hex()
        );
        service.submit_task("t1", &prompt, &f.creator).unwrap();
        let err = service.run_task("t1", &f.agent, &provider_id).unwrap_err();
        assert!(err.to_string().contains("not found in the store"));
        assert!(f.agent.trail_for("t1").iter().all(|c| !c.success));
        assert!(f.agent.trail_for("t1").len() == 1);
    }

    #[test]
    fn assemble_refuses_a_corrupt_item() {
        let f = fixture();
        let provider_id = Identity::generate("provider").unwrap();
        let mut service = AiService::new();
        // A CID whose stored bytes do not hash to it.
        let cid = Cid::from_bytes(sha256(b"real bytes"));
        f.agent
            .put_unchecked_block(&cid, b"tampered bytes".to_vec());
        let prompt = format!(
            "bundle assemble\nbad-bundle\nBad\nCorrupt item.\n100\ntrack|track|{}",
            cid.to_hex()
        );
        service.submit_task("t1", &prompt, &f.creator).unwrap();
        let err = service.run_task("t1", &f.agent, &provider_id).unwrap_err();
        assert!(err.to_string().contains("do not hash"));
        assert!(f.agent.trail_for("t1").iter().all(|c| !c.success));
    }

    #[test]
    fn valid_bundle_assembles_and_manifest_verifies() {
        let f = fixture();
        let provider_id = Identity::generate("provider").unwrap();
        let mut service = AiService::new();
        let cid = assemble_in(&f, &mut service, &provider_id);
        let manifest = f.agent.bundle(&cid).unwrap();
        assert!(manifest.verify());
        assert_eq!(manifest.id, "summer-tracks-001");
        assert_eq!(manifest.title, "Summer Tracks");
        assert_eq!(manifest.price, 500);
        assert_eq!(manifest.creator, f.creator_peer);
        assert_eq!(manifest.items.len(), 2);
        assert_eq!(manifest.items[0].name, "opening-track");
        assert_eq!(manifest.items[0].kind, "track");
        assert_eq!(manifest.items[0].cid, f.track_cid);
        assert_eq!(manifest.cid(), cid);
        // The manifest bytes are in the store under the same CID.
        let stored = f.agent.bundle(&cid).map(|m| m.encode()).unwrap();
        assert_eq!(Cid::from_bytes(sha256(&stored)), cid);
    }

    #[test]
    fn render_returns_real_rendered_bytes_from_the_engine() {
        let f = fixture();
        let provider_id = Identity::generate("provider").unwrap();
        let mut service = AiService::new();
        let cid = assemble_in(&f, &mut service, &provider_id);
        // Default bundle page.
        service
            .submit_task(
                "t1",
                &format!("render ---\nbundle={}", cid.to_hex()),
                &f.creator,
            )
            .unwrap();
        let text = run_task_text(&mut service, &f.agent, &provider_id, "t1").unwrap();
        assert!(text.contains("<h1>Summer Tracks</h1>"));
        assert!(text.contains("opening-track (track)"));
        assert!(text.contains("By creator — 500 X•Koin"));
        // Custom one-line template with bundle data. The `---`
        // separator goes on its own line.
        service
            .submit_task(
                "t2",
                &format!(
                    "render {{{{title}}}} costs {{{{price}}}}\n---\nbundle={}",
                    cid.to_hex()
                ),
                &f.creator,
            )
            .unwrap();
        let text = run_task_text(&mut service, &f.agent, &provider_id, "t2").unwrap();
        assert!(text.contains("Summer Tracks costs 500"));
        // A render whose output address is the real hash of the bytes.
        service
            .submit_task("t3", "render Hello, {{name}}!\n---\nname=World", &f.creator)
            .unwrap();
        let text = run_task_text(&mut service, &f.agent, &provider_id, "t3").unwrap();
        let expected_addr = Cid::from_bytes(sha256(b"Hello, World!")).to_hex();
        assert!(text.contains(&format!("rendered {expected_addr}:")));
        assert!(text.contains("Hello, World!"));
        // An unknown key is a clean failure, recorded on the trail.
        service
            .submit_task("t4", "render Hello {{missing}}", &f.creator)
            .unwrap();
        assert!(service.run_task("t4", &f.agent, &provider_id).is_err());
        let trail = f.agent.trail_for("t4");
        assert_eq!(trail.len(), 1);
        assert_eq!(trail[0].tool, "render");
        assert!(!trail[0].success);
        assert!(trail[0].detail.contains("unknown key"));
    }

    #[test]
    fn buy_settles_the_exact_price_and_delivers() {
        let f = fixture();
        let provider_id = Identity::generate("provider").unwrap();
        let mut service = AiService::new();
        let cid = assemble_in(&f, &mut service, &provider_id);
        assert_eq!(f.agent.balance_of(&f.buyer_peer), 10_000);
        assert_eq!(f.agent.balance_of(&f.creator_peer), 0);
        service
            .submit_task("t1", &format!("bundle buy {}", cid.to_hex()), &f.buyer)
            .unwrap();
        let text = run_task_text(&mut service, &f.agent, &provider_id, "t1").unwrap();
        assert!(text.contains("bought"));
        assert!(text.contains(&cid.to_hex()));
        assert!(text.contains("opening-track (track)"));
        assert!(text.contains(&f.track_cid.to_hex()));
        assert!(text.contains(&f.art_cid.to_hex()));
        assert!(text.contains("receipt: buy-summer-tracks-001-"));
        // Exact price moved, nothing else.
        assert_eq!(f.agent.balance_of(&f.buyer_peer), 9_500);
        assert_eq!(f.agent.balance_of(&f.creator_peer), 500);
        // Buying the same bundle twice is refused, not charged twice.
        service
            .submit_task("t2", &format!("bundle buy {}", cid.to_hex()), &f.buyer)
            .unwrap();
        assert!(service.run_task("t2", &f.agent, &provider_id).is_err());
        assert_eq!(f.agent.balance_of(&f.buyer_peer), 9_500);
        assert_eq!(f.agent.balance_of(&f.creator_peer), 500);
    }

    #[test]
    fn owner_buys_paid_bundle_free() {
        let f = fixture();
        let provider_id = Identity::generate("provider").unwrap();
        let mut service = AiService::new();
        let cid = assemble_in(&f, &mut service, &provider_id);
        // The buyer is the registered owner: the paid bundle is
        // free for them under the OWNER-FREE LAW.
        f.agent.set_owner(f.buyer_peer.clone());
        assert_eq!(f.agent.balance_of(&f.buyer_peer), 10_000);
        assert_eq!(f.agent.balance_of(&f.creator_peer), 0);
        service
            .submit_task("t1", &format!("bundle buy {}", cid.to_hex()), &f.buyer)
            .unwrap();
        let text = run_task_text(&mut service, &f.agent, &provider_id, "t1").unwrap();
        assert!(text.contains("bought"));
        assert!(text.contains(&cid.to_hex()));
        assert!(text.contains("opening-track (track)"));
        assert!(text.contains("owner-free"));
        // No coins moved for anyone.
        assert_eq!(f.agent.balance_of(&f.buyer_peer), 10_000);
        assert_eq!(f.agent.balance_of(&f.creator_peer), 0);
        // Buying the same bundle twice is still refused, not
        // delivered twice.
        service
            .submit_task("t2", &format!("bundle buy {}", cid.to_hex()), &f.buyer)
            .unwrap();
        assert!(service.run_task("t2", &f.agent, &provider_id).is_err());
        assert_eq!(f.agent.balance_of(&f.buyer_peer), 10_000);
        assert_eq!(f.agent.balance_of(&f.creator_peer), 0);
    }

    #[test]
    fn buy_with_insufficient_funds_refuses_with_nothing_moved() {
        let f = fixture_with_funds(100);
        let provider_id = Identity::generate("provider").unwrap();
        let mut service = AiService::new();
        let cid = assemble_in(&f, &mut service, &provider_id);
        service
            .submit_task("t1", &format!("bundle buy {}", cid.to_hex()), &f.buyer)
            .unwrap();
        let err = service.run_task("t1", &f.agent, &provider_id).unwrap_err();
        assert!(err
            .to_string()
            .to_lowercase()
            .contains("insufficient funds"));
        assert_eq!(f.agent.balance_of(&f.buyer_peer), 100);
        assert_eq!(f.agent.balance_of(&f.creator_peer), 0);
        let trail = f.agent.trail_for("t1");
        assert_eq!(trail.len(), 1);
        assert!(!trail[0].success);
    }

    #[test]
    fn free_bundle_is_delivered_without_payment() {
        let f = fixture();
        let provider_id = Identity::generate("provider").unwrap();
        let mut service = AiService::new();
        let prompt = format!(
            "bundle assemble\nfree-stuff\nFree Stuff\nCosts nothing.\n0\ntrack|track|{}",
            f.track_cid.to_hex()
        );
        service.submit_task("a1", &prompt, &f.creator).unwrap();
        let text = run_task_text(&mut service, &f.agent, &provider_id, "a1").unwrap();
        let hex = text
            .lines()
            .nth(2)
            .unwrap()
            .strip_prefix("assembled ")
            .unwrap()
            .trim_end_matches(':');
        service
            .submit_task("t1", &format!("bundle buy {hex}"), &f.buyer)
            .unwrap();
        let text = run_task_text(&mut service, &f.agent, &provider_id, "t1").unwrap();
        assert!(text.contains("receipt: free — no payment needed"));
        assert_eq!(f.agent.balance_of(&f.buyer_peer), 10_000);
        assert_eq!(f.agent.balance_of(&f.creator_peer), 0);
    }

    #[test]
    fn unroutable_task_is_refused_not_answered() {
        let f = fixture();
        let provider_id = Identity::generate("provider").unwrap();
        let mut service = AiService::new();
        service
            .submit_task("t1", "summarize this for me", &f.creator)
            .unwrap();
        let err = service.run_task("t1", &f.agent, &provider_id).unwrap_err();
        assert!(err.to_string().contains("no tool for this task"));
        assert!(service.responses_for("t1").is_empty());
        assert!(f.agent.trail_for("t1").is_empty());
        // An unknown bundle subcommand is refused the same way.
        service
            .submit_task("t2", "bundle explode", &f.creator)
            .unwrap();
        assert!(service.run_task("t2", &f.agent, &provider_id).is_err());
        assert!(f.agent.trail_for("t2").is_empty());
    }

    #[test]
    fn assemble_without_a_registered_signer_is_refused() {
        let creator = Identity::generate("creator").unwrap();
        let creator_peer = creator.peer_id().clone();
        let agent = LocalAgent::new(&[(creator_peer, 0)]).unwrap();
        // Deliberately NOT registering the creator's key.
        let track_cid = agent.put_block(TRACK);
        let provider_id = Identity::generate("provider").unwrap();
        let mut service = AiService::new();
        let prompt = format!(
            "bundle assemble\nno-key-bundle\nNo Key\nNobody signed this.\n10\ntrack|track|{}",
            track_cid.to_hex()
        );
        service.submit_task("t1", &prompt, &creator).unwrap();
        let err = service.run_task("t1", &agent, &provider_id).unwrap_err();
        assert!(err.to_string().contains("no signing key"));
    }

    #[test]
    fn signed_responses_verify_and_tampered_ones_are_refused() {
        let f = fixture();
        let provider_id = Identity::generate("provider").unwrap();
        let mut service = AiService::new();
        service
            .submit_task("t1", "search track", &f.creator)
            .unwrap();
        let cid = service.run_task("t1", &f.agent, &provider_id).unwrap();
        let response = service.get_response(&cid).unwrap();
        assert_eq!(response.provider_name, "local-agent");
        assert!(response.verify());
        assert!(response.text.contains("track one audio bytes"));
        // Tamper with the stored response record bytes.
        let mut bytes = service.store_mut().get(&cid).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let tampered_cid = service.store_mut().put(&bytes);
        assert_eq!(
            service.get_response(&tampered_cid),
            Err(AiError::BadResponseSignature)
        );
        // The original response still reads fine.
        assert!(service.get_response(&cid).unwrap().verify());
    }

    #[test]
    fn test_provider_still_works_unchanged() {
        let alice = Identity::generate("alice").unwrap();
        let task = AiTask::new("t", "Mixed Case 123!", &alice).unwrap();
        assert_eq!(TestProvider.name(), "test-uppercase");
        assert_eq!(
            TestProvider.run(&task).unwrap(),
            "KAI-TEST: MIXED CASE 123!"
        );
    }
}
