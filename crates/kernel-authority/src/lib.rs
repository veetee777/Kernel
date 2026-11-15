//! Owner override authority for KERNEL (packet law).
//!
//! The registered owner of a KERNEL node can override ANY hard rule in
//! KERNEL, including security hard rules that currently say even the
//! owner cannot allow. This crate is that mechanism, and nothing else:
//!
//! * [`RuleRegistry`] — named hard rules. Every rule refuses everyone
//!   by default; a rule is lifted only by a recorded owner override.
//! * [`OwnerRegistry`] — the node's registered owner, identified by
//!   public key ([`PeerId`](kernel_identity::PeerId)). First
//!   registration wins; a different owner cannot be registered over
//!   the first one.
//! * [`Authority`] — the override flow. The owner signs an override
//!   for a named rule with a reason and a caller-supplied timestamp.
//!   Non-owners, unknown rules, missing reasons, and bad signatures
//!   are refused, and nothing is logged for a refused override.
//! * [`OverrideLog`] — the append-only audit log. Binary records with
//!   magic `KOVR1`, hash-chained, written via temp file + fsync +
//!   atomic rename. Loading verifies every record's self-hash, the
//!   chain, and the owner's signature, and refuses bad magic,
//!   truncation, or any tampering.
//!
//! This is NOT the owner-free payment exemption — that already exists
//! in k-pay. This crate touches no payment code, no network, and no
//! randomness. Time is always caller-supplied (`now: u64`); this crate
//! never reads the system clock.

use std::collections::HashMap;
use std::fmt;
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use kernel_crypto::{sha256, Ed25519Signature};
use kernel_identity::{Identity, IdentityError, PeerId};

/// Magic bytes at the start of every audit-log record.
const RECORD_MAGIC: &[u8; 5] = b"KOVR1";
/// Name of the audit log file inside the authority directory.
const LOG_FILE_NAME: &str = "overrides.kovr";
/// Longest reason an override may carry, in characters.
const MAX_REASON_LEN: usize = 512;
/// Longest rule name, in characters.
const MAX_RULE_NAME_LEN: usize = 64;

/// Errors returned by authority operations.
#[derive(Debug)]
pub enum AuthorityError {
    /// An underlying I/O failure.
    Io(std::io::Error),
    /// The audit log failed verification (bad magic, truncation,
    /// broken hash chain, bad record hash, or bad signature).
    Corrupt(String),
    /// An identity operation failed (bad key, bad name, bad signature
    /// format) while checking an override.
    Identity(IdentityError),
    /// No owner has been registered yet.
    NoOwnerRegistered,
    /// An owner is already registered; a different owner cannot
    /// replace them.
    OwnerAlreadySet,
    /// The caller is not the registered owner.
    NotOwner,
    /// The named rule does not exist in the registry.
    UnknownRule(String),
    /// A rule name broke the naming rules.
    InvalidRuleName(String),
    /// The override carried no reason (empty or whitespace-only).
    MissingReason,
    /// The override reason exceeded [`MAX_REASON_LEN`] characters.
    ReasonTooLong,
    /// The override signature did not verify against the registered
    /// owner's public key.
    BadSignature,
}

impl fmt::Display for AuthorityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuthorityError::Io(e) => write!(f, "I/O error: {e}"),
            AuthorityError::Corrupt(msg) => write!(f, "corrupt audit log: {msg}"),
            AuthorityError::Identity(e) => write!(f, "identity error: {e}"),
            AuthorityError::NoOwnerRegistered => write!(f, "no owner is registered"),
            AuthorityError::OwnerAlreadySet => {
                write!(f, "an owner is already registered")
            }
            AuthorityError::NotOwner => write!(f, "caller is not the registered owner"),
            AuthorityError::UnknownRule(name) => {
                write!(f, "unknown hard rule '{name}'")
            }
            AuthorityError::InvalidRuleName(name) => {
                write!(f, "invalid rule name '{name}'")
            }
            AuthorityError::MissingReason => {
                write!(f, "an override requires a reason")
            }
            AuthorityError::ReasonTooLong => {
                write!(f, "override reason exceeds {MAX_REASON_LEN} characters")
            }
            AuthorityError::BadSignature => {
                write!(f, "override signature does not verify")
            }
        }
    }
}

impl std::error::Error for AuthorityError {}

impl From<std::io::Error> for AuthorityError {
    fn from(e: std::io::Error) -> Self {
        AuthorityError::Io(e)
    }
}

impl From<IdentityError> for AuthorityError {
    fn from(e: IdentityError) -> Self {
        AuthorityError::Identity(e)
    }
}

/// What a hard-rule check returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// The rule does not block this action.
    Allowed,
    /// The rule blocks this action. This is the default for every
    /// hard rule until the registered owner overrides it.
    Denied,
}

/// Rules for rule names: 1-64 characters, ASCII letters, digits,
/// `-`, `_`, `.`. Same shape as display names, so a rule name is
/// always safe to print, log, and put in a file name.
fn validate_rule_name(name: &str) -> Result<(), AuthorityError> {
    if name.is_empty() || name.len() > MAX_RULE_NAME_LEN {
        return Err(AuthorityError::InvalidRuleName(name.to_string()));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Err(AuthorityError::InvalidRuleName(name.to_string()));
    }
    Ok(())
}

/// Validate an override reason: non-empty after trimming, at most
/// [`MAX_REASON_LEN`] characters.
fn validate_reason(reason: &str) -> Result<(), AuthorityError> {
    if reason.trim().is_empty() {
        return Err(AuthorityError::MissingReason);
    }
    if reason.chars().count() > MAX_REASON_LEN {
        return Err(AuthorityError::ReasonTooLong);
    }
    Ok(())
}

/// The exact bytes an owner signs for an override. Domain-separated
/// so an override signature can never be mistaken for a signature
/// over anything else in KERNEL.
fn override_message(rule: &str, reason: &str, timestamp: u64) -> Vec<u8> {
    let mut msg = Vec::with_capacity(17 + rule.len() + reason.len() + 8);
    msg.extend_from_slice(b"KOVR-OVERRIDE-v1");
    msg.push(0);
    msg.extend_from_slice(rule.as_bytes());
    msg.push(0);
    msg.extend_from_slice(reason.as_bytes());
    msg.push(0);
    msg.extend_from_slice(&timestamp.to_le_bytes());
    msg
}

// ---------------------------------------------------------------------------
// Hard rules
// ---------------------------------------------------------------------------

/// One hard rule: a name and what it forbids. A hard rule refuses
/// everyone by default — including the owner — until the registered
/// owner overrides it through [`Authority`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HardRule {
    /// The rule's name, e.g. `no-unsigned-code`.
    pub name: String,
    /// A human sentence describing what the rule forbids.
    pub description: String,
}

/// The registry of named hard rules. Rules are registered by name;
/// an override for a name that is not registered is refused as
/// [`AuthorityError::UnknownRule`].
#[derive(Clone, Debug, Default)]
pub struct RuleRegistry {
    rules: HashMap<String, HardRule>,
}

impl RuleRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// The registry every node starts with: the sample hard rules
    /// that ship with KERNEL. Each one refuses everyone by default.
    pub fn default_registry() -> Self {
        let mut registry = Self::new();
        registry
            .register(
                "no-unsigned-code",
                "Refuse to execute code that is not signed by a trusted key.",
            )
            .expect("sample rule names are valid");
        registry
            .register(
                "no-remote-admin",
                "Refuse remote administration sessions from any peer.",
            )
            .expect("sample rule names are valid");
        registry
            .register(
                "no-unmetered-export",
                "Refuse to export node data without metering.",
            )
            .expect("sample rule names are valid");
        registry
    }

    /// Register a hard rule. Re-registering the same name replaces
    /// its description.
    pub fn register(&mut self, name: &str, description: &str) -> Result<(), AuthorityError> {
        validate_rule_name(name)?;
        self.rules.insert(
            name.to_string(),
            HardRule {
                name: name.to_string(),
                description: description.to_string(),
            },
        );
        Ok(())
    }

    /// Look a rule up by name.
    pub fn get(&self, name: &str) -> Option<&HardRule> {
        self.rules.get(name)
    }

    /// True if the registry holds a rule with this name.
    pub fn contains(&self, name: &str) -> bool {
        self.rules.contains_key(name)
    }

    /// Number of registered rules.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// True if no rules are registered.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Owner registry
// ---------------------------------------------------------------------------

/// The node's registered owner, identified by public key. There is
/// exactly one owner per node: the first registration wins, and a
/// different owner cannot be registered over it.
#[derive(Clone, Debug, Default)]
pub struct OwnerRegistry {
    owner: Option<PeerId>,
}

impl OwnerRegistry {
    /// An empty registry (no owner yet).
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the owner. Registering the same owner again is a
    /// no-op. Registering a different owner is refused.
    pub fn register_owner(&mut self, owner: PeerId) -> Result<(), AuthorityError> {
        match &self.owner {
            None => {
                self.owner = Some(owner);
                Ok(())
            }
            Some(current) if *current == owner => Ok(()),
            Some(_) => Err(AuthorityError::OwnerAlreadySet),
        }
    }

    /// The registered owner, if any.
    pub fn owner(&self) -> Option<&PeerId> {
        self.owner.as_ref()
    }

    /// True if `peer` is the registered owner (same public key and
    /// same display name).
    pub fn is_owner(&self, peer: &PeerId) -> bool {
        self.owner.as_ref() == Some(peer)
    }
}

// ---------------------------------------------------------------------------
// Audit-log records
// ---------------------------------------------------------------------------

/// One recorded owner override: which rule, which owner, when
/// (caller-supplied), why, and the owner's signature over all of it.
///
/// Binary layout, all integers little-endian:
///
/// ```text
/// KOVR1 (5) | prev_hash (32) | rule_len u16 | rule |
/// owner_key (32) | owner_name_len u8 | owner_name |
/// reason_len u16 | reason | timestamp u64 | signature (64) |
/// record_hash (32)
/// ```
///
/// `record_hash` is the SHA-256 of every byte before it in the
/// record. `prev_hash` is the previous record's `record_hash`, or
/// all zeros for the first record. The chain makes silent edits
/// impossible: changing any byte breaks either that record's own
/// hash, the next record's chain link, or the signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OverrideRecord {
    /// The overridden rule's name.
    pub rule: String,
    /// The owner who overrode it.
    pub owner: PeerId,
    /// Why the owner overrode it. Never empty.
    pub reason: String,
    /// Caller-supplied time of the override (seconds since epoch,
    /// by the caller's clock — this crate never reads the clock).
    pub timestamp: u64,
    /// The owner's signature over [`override_message`].
    pub signature: Ed25519Signature,
    /// Hash of the previous record, or all zeros for the first.
    pub prev_hash: [u8; 32],
    /// SHA-256 of every byte of this record before this field.
    pub record_hash: [u8; 32],
}

impl OverrideRecord {
    /// Build a record from its fields, computing `record_hash`.
    /// The signature is NOT verified here — verification happens in
    /// [`Authority::apply_override`] and on log load.
    fn new(
        rule: String,
        owner: PeerId,
        reason: String,
        timestamp: u64,
        signature: Ed25519Signature,
        prev_hash: [u8; 32],
    ) -> Self {
        let mut record = Self {
            rule,
            owner,
            reason,
            timestamp,
            signature,
            prev_hash,
            record_hash: [0u8; 32],
        };
        record.record_hash = sha256(&record.encode_without_hash());
        record
    }

    /// Serialize the record without its trailing `record_hash`
    /// (the bytes the hash covers).
    fn encode_without_hash(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(RECORD_MAGIC);
        out.extend_from_slice(&self.prev_hash);
        out.extend_from_slice(&(self.rule.len() as u16).to_le_bytes());
        out.extend_from_slice(self.rule.as_bytes());
        out.extend_from_slice(self.owner.public_key().as_bytes());
        out.extend_from_slice(&[self.owner.display_name().len() as u8]);
        out.extend_from_slice(self.owner.display_name().as_bytes());
        out.extend_from_slice(&(self.reason.len() as u16).to_le_bytes());
        out.extend_from_slice(self.reason.as_bytes());
        out.extend_from_slice(&self.timestamp.to_le_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Serialize the full record, including `record_hash`.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = self.encode_without_hash();
        out.extend_from_slice(&self.record_hash);
        out
    }

    /// Parse one record from the front of `bytes`, returning the
    /// record and the number of bytes consumed. Verifies the magic,
    /// the record hash, the reason, and the rule name — but not the
    /// chain link or the signature; those are checked by the log
    /// loader, which has the context for them.
    pub fn decode(bytes: &[u8]) -> Result<(Self, usize), AuthorityError> {
        let corrupt = |msg: &str| AuthorityError::Corrupt(msg.to_string());
        let mut off = 0usize;
        let need = |off: usize, n: usize| -> Result<(), AuthorityError> {
            if bytes.len() < off + n {
                Err(corrupt("truncated override record"))
            } else {
                Ok(())
            }
        };
        need(off, 5)?;
        if &bytes[off..off + 5] != RECORD_MAGIC {
            return Err(corrupt("bad record magic"));
        }
        off += 5;
        need(off, 32)?;
        let mut prev_hash = [0u8; 32];
        prev_hash.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        need(off, 2)?;
        let rule_len = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap()) as usize;
        off += 2;
        need(off, rule_len)?;
        let rule = String::from_utf8(bytes[off..off + rule_len].to_vec())
            .map_err(|_| corrupt("rule name is not valid UTF-8"))?;
        off += rule_len;
        need(off, 32)?;
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        need(off, 1)?;
        let name_len = bytes[off] as usize;
        off += 1;
        need(off, name_len)?;
        let owner_name = String::from_utf8(bytes[off..off + name_len].to_vec())
            .map_err(|_| corrupt("owner name is not valid UTF-8"))?;
        off += name_len;
        need(off, 2)?;
        let reason_len = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap()) as usize;
        off += 2;
        need(off, reason_len)?;
        let reason = String::from_utf8(bytes[off..off + reason_len].to_vec())
            .map_err(|_| corrupt("reason is not valid UTF-8"))?;
        off += reason_len;
        need(off, 8)?;
        let timestamp = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        off += 8;
        need(off, 64)?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        off += 64;
        need(off, 32)?;
        let mut record_hash = [0u8; 32];
        record_hash.copy_from_slice(&bytes[off..off + 32]);
        off += 32;

        validate_rule_name(&rule).map_err(|_| corrupt("record holds an invalid rule name"))?;
        if reason.trim().is_empty() {
            return Err(corrupt("record holds an empty reason"));
        }
        let owner_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|_| corrupt("record holds an invalid owner key"))?;
        let owner = PeerId::new(owner_key, &owner_name)
            .map_err(|_| corrupt("record holds an invalid owner name"))?;
        let record = Self {
            rule,
            owner,
            reason,
            timestamp,
            signature: Ed25519Signature::from_bytes(&sig_bytes),
            prev_hash,
            record_hash,
        };
        if sha256(&record.encode_without_hash()) != record_hash {
            return Err(corrupt("record hash mismatch"));
        }
        Ok((record, off))
    }
}

// ---------------------------------------------------------------------------
// Audit log
// ---------------------------------------------------------------------------

/// The append-only audit log for owner overrides, kept in one binary
/// file (`overrides.kovr`) inside a directory.
///
/// Records are only ever added. [`OverrideLog::save`] writes the whole
/// log to a sibling temp file, fsyncs it, atomically renames it over
/// the log, and fsyncs the directory — so a crash before the rename
/// leaves the old log intact and a crash after it finds the new log
/// complete. Loading replays the file and verifies every record's
/// self-hash, the hash chain, and the owner's signature; any bad
/// magic, truncation, or tampering is refused.
#[derive(Debug)]
pub struct OverrideLog {
    /// Full path of the log file.
    path: PathBuf,
    /// Records in file order. Only grows.
    records: Vec<OverrideRecord>,
}

impl OverrideLog {
    /// Open the log in `dir`, creating the directory and the file if
    /// they do not exist. An existing log is replayed and fully
    /// verified; a corrupt one refuses to open.
    pub fn open(dir: &Path) -> Result<Self, AuthorityError> {
        create_dir_all(dir)?;
        let path = dir.join(LOG_FILE_NAME);
        let records = if path.exists() {
            let bytes = std::fs::read(&path)?;
            Self::parse_and_verify(&bytes)?
        } else {
            Vec::new()
        };
        Ok(Self { path, records })
    }

    /// Parse and fully verify a log's bytes: every record's magic and
    /// self-hash, the hash chain from the genesis record on, and every
    /// record's owner signature over its canonical message. Any
    /// failure — including trailing bytes after the last complete
    /// record — is corruption.
    fn parse_and_verify(bytes: &[u8]) -> Result<Vec<OverrideRecord>, AuthorityError> {
        let mut records = Vec::new();
        let mut off = 0usize;
        let mut expected_prev = [0u8; 32];
        while off < bytes.len() {
            let (record, consumed) = OverrideRecord::decode(&bytes[off..])?;
            if record.prev_hash != expected_prev {
                return Err(AuthorityError::Corrupt(
                    "audit log hash chain is broken".to_string(),
                ));
            }
            // A record whose signature does not verify against its own
            // owner key was not made by that owner, even if its hashes
            // were recomputed after tampering.
            let message = override_message(&record.rule, &record.reason, record.timestamp);
            record
                .owner
                .verify(&message, &record.signature)
                .map_err(|_| {
                    AuthorityError::Corrupt(
                        "audit log record signature does not verify".to_string(),
                    )
                })?;
            expected_prev = record.record_hash;
            records.push(record);
            off += consumed;
        }
        Ok(records)
    }

    /// Append a record and durably save the log. The record is only
    /// returned as stored once the save has succeeded.
    pub fn append(&mut self, record: OverrideRecord) -> Result<(), AuthorityError> {
        self.records.push(record);
        if let Err(e) = self.save() {
            self.records.pop();
            return Err(e);
        }
        Ok(())
    }

    /// Durably write the whole log: temp file, fsync, atomic rename
    /// over the log, directory fsync. A failure removes the temp
    /// file and leaves the previous log untouched.
    pub fn save(&self) -> Result<(), AuthorityError> {
        let result = self.save_inner();
        if result.is_err() {
            let _ = std::fs::remove_file(self.tmp_path());
        }
        result
    }

    fn save_inner(&self) -> Result<(), AuthorityError> {
        let tmp_path = self.tmp_path();
        let mut tmp = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        for record in &self.records {
            tmp.write_all(&record.encode())?;
        }
        tmp.sync_all()?;
        drop(tmp);
        std::fs::rename(&tmp_path, &self.path)?;
        // Make the rename itself durable.
        let dir = self.path.parent().unwrap_or(Path::new("."));
        File::open(dir)?.sync_all()?;
        Ok(())
    }

    /// Path of the sibling temp file saves write to.
    fn tmp_path(&self) -> PathBuf {
        let mut tmp = self.path.clone().into_os_string();
        tmp.push(".tmp");
        PathBuf::from(tmp)
    }

    /// Every record in the log, in file order.
    pub fn records(&self) -> &[OverrideRecord] {
        &self.records
    }

    /// Number of records in the log.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True if the log holds no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Hash of the most recent record, or all zeros when the log is
    /// empty. The next record chains from this.
    pub fn head_hash(&self) -> [u8; 32] {
        self.records
            .last()
            .map(|r| r.record_hash)
            .unwrap_or([0u8; 32])
    }
}

// ---------------------------------------------------------------------------
// Authority
// ---------------------------------------------------------------------------

/// The owner-override authority for one node.
///
/// A hard rule refuses everyone until the registered owner overrides
/// it, and every override is signed and written to the audit log
/// before the rule is lifted. Checking a rule consults only recorded
/// overrides — there is no back door, no unsigned override, and no
/// override without a reason.
#[derive(Debug)]
pub struct Authority {
    rules: RuleRegistry,
    owners: OwnerRegistry,
    log: OverrideLog,
    /// Rules lifted by a recorded override, with the latest record
    /// for each. Rebuilt from the log at open time.
    overridden: HashMap<String, OverrideRecord>,
}

impl Authority {
    /// Open the authority in `dir`, starting from the default rule
    /// registry and no registered owner. An existing audit log in
    /// `dir` is loaded and verified, and its overrides are restored.
    pub fn new(dir: &Path) -> Result<Self, AuthorityError> {
        let log = OverrideLog::open(dir)?;
        let mut overridden = HashMap::new();
        for record in log.records() {
            overridden.insert(record.rule.clone(), record.clone());
        }
        Ok(Self {
            rules: RuleRegistry::default_registry(),
            owners: OwnerRegistry::new(),
            log,
            overridden,
        })
    }

    /// The hard-rule registry.
    pub fn rules(&self) -> &RuleRegistry {
        &self.rules
    }

    /// The owner registry.
    pub fn owners(&self) -> &OwnerRegistry {
        &self.owners
    }

    /// Register the node's owner. First registration wins; a
    /// different owner cannot replace the first.
    pub fn register_owner(&mut self, owner: PeerId) -> Result<(), AuthorityError> {
        self.owners.register_owner(owner)
    }

    /// Check a hard rule: [`Decision::Allowed`] only when the
    /// registered owner has overridden it; [`Decision::Denied`]
    /// otherwise. An unknown rule is an error, not a denial — callers
    /// must not silently treat a typo as a refusal.
    pub fn check(&self, rule: &str) -> Result<Decision, AuthorityError> {
        if !self.rules.contains(rule) {
            return Err(AuthorityError::UnknownRule(rule.to_string()));
        }
        if self.overridden.contains_key(rule) {
            Ok(Decision::Allowed)
        } else {
            Ok(Decision::Denied)
        }
    }

    /// True if the rule has a recorded owner override.
    pub fn is_overridden(&self, rule: &str) -> bool {
        self.overridden.contains_key(rule)
    }

    /// The latest recorded override for a rule, if any.
    pub fn override_for(&self, rule: &str) -> Option<&OverrideRecord> {
        self.overridden.get(rule)
    }

    /// Every recorded override, in log order.
    pub fn overrides(&self) -> &[OverrideRecord] {
        self.log.records()
    }

    /// Number of recorded overrides.
    pub fn override_count(&self) -> usize {
        self.log.len()
    }

    /// Override a hard rule as the owner, signing with `identity`.
    /// The caller supplies the time (`now`); this crate never reads
    /// the system clock. On success the override is in the audit log
    /// before this returns, and the rule is lifted.
    pub fn request_override(
        &mut self,
        identity: &Identity,
        rule: &str,
        reason: &str,
        now: u64,
    ) -> Result<OverrideRecord, AuthorityError> {
        let message = override_message(rule, reason, now);
        let signature = identity.sign(&message);
        self.apply_override(identity.peer_id(), rule, reason, now, &signature)
    }

    /// Apply an override signed elsewhere by the owner (for example,
    /// signed on a different device and delivered to this node).
    /// Verifies the rule exists, the reason is present, the signer is
    /// the registered owner, and the signature verifies — then logs
    /// the override and lifts the rule. Refused overrides are never
    /// logged.
    pub fn apply_override(
        &mut self,
        owner: &PeerId,
        rule: &str,
        reason: &str,
        timestamp: u64,
        signature: &Ed25519Signature,
    ) -> Result<OverrideRecord, AuthorityError> {
        validate_rule_name(rule)?;
        if !self.rules.contains(rule) {
            return Err(AuthorityError::UnknownRule(rule.to_string()));
        }
        validate_reason(reason)?;
        let registered = self
            .owners
            .owner()
            .ok_or(AuthorityError::NoOwnerRegistered)?;
        if registered != owner {
            return Err(AuthorityError::NotOwner);
        }
        let message = override_message(rule, reason, timestamp);
        owner
            .verify(&message, signature)
            .map_err(|_| AuthorityError::BadSignature)?;
        let record = OverrideRecord::new(
            rule.to_string(),
            owner.clone(),
            reason.to_string(),
            timestamp,
            *signature,
            self.log.head_hash(),
        );
        self.log.append(record.clone())?;
        self.overridden.insert(rule.to_string(), record.clone());
        Ok(record)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kernel-authority-{}-{}",
            std::process::id(),
            name
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn owner_and_other() -> (Identity, Identity) {
        (
            Identity::generate("tristan").unwrap(),
            Identity::generate("mallory").unwrap(),
        )
    }

    /// An authority with `owner` registered, ready for overrides.
    fn authority_with_owner(dir: &Path, owner: &Identity) -> Authority {
        let mut auth = Authority::new(dir).unwrap();
        auth.register_owner(owner.peer_id().clone()).unwrap();
        auth
    }

    #[test]
    fn default_rules_deny_everyone() {
        let dir = test_dir("default-deny");
        let (owner, other) = owner_and_other();
        let auth = authority_with_owner(&dir, &owner);
        assert_eq!(auth.rules().len(), 3);
        for rule in ["no-unsigned-code", "no-remote-admin", "no-unmetered-export"] {
            assert_eq!(auth.check(rule).unwrap(), Decision::Denied);
        }
        // Denial does not depend on who asks: the rule refuses the
        // owner too until they override it.
        let _ = other;
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_rule_check_is_an_error_not_a_denial() {
        let dir = test_dir("unknown-check");
        let auth = Authority::new(&dir).unwrap();
        assert!(matches!(
            auth.check("no-such-rule"),
            Err(AuthorityError::UnknownRule(_))
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn override_without_registered_owner_is_refused() {
        let dir = test_dir("no-owner");
        let (owner, _) = owner_and_other();
        let mut auth = Authority::new(&dir).unwrap();
        assert!(matches!(
            auth.request_override(&owner, "no-unsigned-code", "need to run it", 1000),
            Err(AuthorityError::NoOwnerRegistered)
        ));
        assert_eq!(auth.override_count(), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn non_owner_override_is_refused_and_not_logged() {
        let dir = test_dir("non-owner");
        let (owner, other) = owner_and_other();
        let mut auth = authority_with_owner(&dir, &owner);
        assert!(matches!(
            auth.request_override(&other, "no-unsigned-code", "let me in", 1000),
            Err(AuthorityError::NotOwner)
        ));
        assert_eq!(auth.override_count(), 0);
        assert_eq!(auth.check("no-unsigned-code").unwrap(), Decision::Denied);
        assert!(!auth.is_overridden("no-unsigned-code"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn owner_override_succeeds_lifts_rule_and_is_logged() {
        let dir = test_dir("owner-ok");
        let (owner, _) = owner_and_other();
        let mut auth = authority_with_owner(&dir, &owner);
        let record = auth
            .request_override(&owner, "no-remote-admin", "debugging my own node", 1_700_000_000)
            .unwrap();
        assert_eq!(record.rule, "no-remote-admin");
        assert_eq!(record.owner, *owner.peer_id());
        assert_eq!(record.reason, "debugging my own node");
        assert_eq!(record.timestamp, 1_700_000_000);
        assert_eq!(record.prev_hash, [0u8; 32]);
        assert_ne!(record.record_hash, [0u8; 32]);
        assert_eq!(auth.check("no-remote-admin").unwrap(), Decision::Allowed);
        assert!(auth.is_overridden("no-remote-admin"));
        // The other rules are still denied.
        assert_eq!(auth.check("no-unsigned-code").unwrap(), Decision::Denied);
        assert_eq!(auth.override_count(), 1);
        assert_eq!(auth.overrides()[0], record);
        // The record's signature verifies against the owner's key.
        let message = override_message(&record.rule, &record.reason, record.timestamp);
        assert!(record.owner.verify(&message, &record.signature).is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn override_without_reason_is_refused() {
        let dir = test_dir("no-reason");
        let (owner, _) = owner_and_other();
        let mut auth = authority_with_owner(&dir, &owner);
        assert!(matches!(
            auth.request_override(&owner, "no-unsigned-code", "", 1000),
            Err(AuthorityError::MissingReason)
        ));
        assert!(matches!(
            auth.request_override(&owner, "no-unsigned-code", "   ", 1000),
            Err(AuthorityError::MissingReason)
        ));
        assert_eq!(auth.override_count(), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn override_reason_too_long_is_refused() {
        let dir = test_dir("long-reason");
        let (owner, _) = owner_and_other();
        let mut auth = authority_with_owner(&dir, &owner);
        let long = "x".repeat(MAX_REASON_LEN + 1);
        assert!(matches!(
            auth.request_override(&owner, "no-unsigned-code", &long, 1000),
            Err(AuthorityError::ReasonTooLong)
        ));
        // Exactly the limit is fine.
        let exact = "x".repeat(MAX_REASON_LEN);
        assert!(auth
            .request_override(&owner, "no-unsigned-code", &exact, 1000)
            .is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_rule_override_is_refused() {
        let dir = test_dir("unknown-override");
        let (owner, _) = owner_and_other();
        let mut auth = authority_with_owner(&dir, &owner);
        assert!(matches!(
            auth.request_override(&owner, "no-such-rule", "please", 1000),
            Err(AuthorityError::UnknownRule(_))
        ));
        assert_eq!(auth.override_count(), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn bad_signature_is_refused_and_not_logged() {
        let dir = test_dir("bad-sig");
        let (owner, other) = owner_and_other();
        let mut auth = authority_with_owner(&dir, &owner);
        // Mallory signs the exact message, but presents it as the
        // owner: the key does not match the registered owner.
        let message = override_message("no-unsigned-code", "trust me", 1000);
        let mallory_sig = other.sign(&message);
        assert!(matches!(
            auth.apply_override(
                owner.peer_id(),
                "no-unsigned-code",
                "trust me",
                1000,
                &mallory_sig
            ),
            Err(AuthorityError::BadSignature)
        ));
        // And a signature over a different reason does not verify
        // for the reason submitted either.
        let other_message = override_message("no-unsigned-code", "different reason", 1000);
        let wrong_reason_sig = owner.sign(&other_message);
        assert!(matches!(
            auth.apply_override(
                owner.peer_id(),
                "no-unsigned-code",
                "trust me",
                1000,
                &wrong_reason_sig
            ),
            Err(AuthorityError::BadSignature)
        ));
        assert_eq!(auth.override_count(), 0);
        assert_eq!(auth.check("no-unsigned-code").unwrap(), Decision::Denied);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn owner_re_registration_is_refused_but_same_owner_is_fine() {
        let dir = test_dir("re-register");
        let (owner, other) = owner_and_other();
        let mut auth = Authority::new(&dir).unwrap();
        auth.register_owner(owner.peer_id().clone()).unwrap();
        assert!(auth.register_owner(owner.peer_id().clone()).is_ok());
        assert!(matches!(
            auth.register_owner(other.peer_id().clone()),
            Err(AuthorityError::OwnerAlreadySet)
        ));
        assert_eq!(auth.owners().owner(), Some(owner.peer_id()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn log_round_trips_across_reopen() {
        let dir = test_dir("round-trip");
        let (owner, _) = owner_and_other();
        {
            let mut auth = authority_with_owner(&dir, &owner);
            auth.request_override(&owner, "no-unsigned-code", "first", 1000)
                .unwrap();
            auth.request_override(&owner, "no-remote-admin", "second", 2000)
                .unwrap();
            assert_eq!(auth.override_count(), 2);
        }
        // Reopen: the log must verify, and both overrides must still
        // be in force — but the owner registry is not persisted, so
        // the caller re-registers the owner (as a node does at boot
        // from its own identity).
        let mut reopened = Authority::new(&dir).unwrap();
        assert_eq!(reopened.override_count(), 2);
        assert_eq!(reopened.check("no-unsigned-code").unwrap(), Decision::Allowed);
        assert_eq!(reopened.check("no-remote-admin").unwrap(), Decision::Allowed);
        assert_eq!(
            reopened.overrides()[0].reason,
            "first",
        );
        assert_eq!(reopened.overrides()[1].reason, "second");
        // Chain links line up: record 2 chains from record 1.
        assert_eq!(
            reopened.overrides()[1].prev_hash,
            reopened.overrides()[0].record_hash
        );
        // A fresh override still appends after the loaded records.
        reopened.register_owner(owner.peer_id().clone()).unwrap();
        reopened
            .request_override(&owner, "no-unmetered-export", "third", 3000)
            .unwrap();
        assert_eq!(reopened.override_count(), 3);
        assert_eq!(
            reopened.overrides()[2].prev_hash,
            reopened.overrides()[1].record_hash
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn log_refuses_tampered_payload() {
        let dir = test_dir("tamper-payload");
        let (owner, _) = owner_and_other();
        {
            let mut auth = authority_with_owner(&dir, &owner);
            auth.request_override(&owner, "no-unsigned-code", "legit reason", 1000)
                .unwrap();
        }
        // Flip one byte inside the reason on disk.
        let path = dir.join(LOG_FILE_NAME);
        let mut bytes = fs::read(&path).unwrap();
        let at = bytes
            .windows(b"legit reason".len())
            .position(|w| w == b"legit reason")
            .expect("reason must be in the log bytes");
        bytes[at] ^= 0xff;
        fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            Authority::new(&dir),
            Err(AuthorityError::Corrupt(_))
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn log_refuses_bad_magic() {
        let dir = test_dir("bad-magic");
        let (owner, _) = owner_and_other();
        {
            let mut auth = authority_with_owner(&dir, &owner);
            auth.request_override(&owner, "no-unsigned-code", "reason", 1000)
                .unwrap();
        }
        let path = dir.join(LOG_FILE_NAME);
        let mut bytes = fs::read(&path).unwrap();
        bytes[0] = b'X';
        fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            Authority::new(&dir),
            Err(AuthorityError::Corrupt(_))
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn log_refuses_truncation() {
        let dir = test_dir("truncated");
        let (owner, _) = owner_and_other();
        {
            let mut auth = authority_with_owner(&dir, &owner);
            auth.request_override(&owner, "no-unsigned-code", "reason", 1000)
                .unwrap();
            auth.request_override(&owner, "no-remote-admin", "reason two", 2000)
                .unwrap();
        }
        let path = dir.join(LOG_FILE_NAME);
        let bytes = fs::read(&path).unwrap();
        // Cut the file in the middle of the second record.
        fs::write(&path, &bytes[..bytes.len() - 20]).unwrap();
        assert!(matches!(
            Authority::new(&dir),
            Err(AuthorityError::Corrupt(_))
        ));
        // Trailing garbage after the last complete record is
        // corruption too, not a torn tail to silently ignore.
        fs::write(&path, {
            let mut b = bytes.clone();
            b.extend_from_slice(b"junk");
            b
        })
        .unwrap();
        assert!(matches!(
            Authority::new(&dir),
            Err(AuthorityError::Corrupt(_))
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn log_refuses_broken_chain_even_when_hashes_are_recomputed() {
        let dir = test_dir("broken-chain");
        let (owner, _) = owner_and_other();
        {
            let mut auth = authority_with_owner(&dir, &owner);
            auth.request_override(&owner, "no-unsigned-code", "first", 1000)
                .unwrap();
            auth.request_override(&owner, "no-remote-admin", "second", 2000)
                .unwrap();
        }
        let path = dir.join(LOG_FILE_NAME);
        let bytes = fs::read(&path).unwrap();
        // Decode both records, change the first record's reason, and
        // re-encode it with a fresh self-hash — so its own hash
        // verifies, but the second record's chain link no longer
        // matches the first record's new hash.
        let (mut first, first_len) = OverrideRecord::decode(&bytes).unwrap();
        let (second, _) = OverrideRecord::decode(&bytes[first_len..]).unwrap();
        first.reason = "changed".to_string();
        first.record_hash = sha256(&first.encode_without_hash());
        let mut forged = first.encode();
        forged.extend_from_slice(&second.encode());
        fs::write(&path, &forged).unwrap();
        assert!(matches!(
            Authority::new(&dir),
            Err(AuthorityError::Corrupt(_))
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn log_refuses_record_with_forged_signature() {
        let dir = test_dir("forged-sig");
        let (owner, other) = owner_and_other();
        {
            let mut auth = authority_with_owner(&dir, &owner);
            auth.request_override(&owner, "no-unsigned-code", "real", 1000)
                .unwrap();
        }
        let path = dir.join(LOG_FILE_NAME);
        let bytes = fs::read(&path).unwrap();
        // Take the valid record, swap in Mallory's signature over the
        // same message, and recompute the record hash so the hash
        // checks pass — the signature check on load must still catch
        // it, because Mallory's key is not the owner's key.
        let (mut record, _) = OverrideRecord::decode(&bytes).unwrap();
        let message = override_message(&record.rule, &record.reason, record.timestamp);
        record.signature = other.sign(&message);
        record.record_hash = sha256(&record.encode_without_hash());
        fs::write(&path, record.encode()).unwrap();
        assert!(matches!(
            Authority::new(&dir),
            Err(AuthorityError::Corrupt(_))
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn re_override_of_an_overridden_rule_is_logged_too() {
        let dir = test_dir("re-override");
        let (owner, _) = owner_and_other();
        let mut auth = authority_with_owner(&dir, &owner);
        auth.request_override(&owner, "no-unsigned-code", "first time", 1000)
            .unwrap();
        auth.request_override(&owner, "no-unsigned-code", "second time", 2000)
            .unwrap();
        assert_eq!(auth.override_count(), 2);
        assert_eq!(auth.override_for("no-unsigned-code").unwrap().reason, "second time");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn invalid_rule_names_are_refused() {
        let dir = test_dir("bad-names");
        let (owner, _) = owner_and_other();
        let mut auth = authority_with_owner(&dir, &owner);
        assert!(matches!(
            auth.request_override(&owner, "has space", "reason", 1000),
            Err(AuthorityError::InvalidRuleName(_))
        ));
        assert!(matches!(
            auth.request_override(&owner, "", "reason", 1000),
            Err(AuthorityError::InvalidRuleName(_))
        ));
        assert_eq!(auth.override_count(), 0);
        let _ = fs::remove_dir_all(&dir);
    }
}
