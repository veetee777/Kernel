//! X•Kom's — the owner's personal tool on KERNEL.
//!
//! Three personal pieces, all signed:
//!
//! * A **note** is a signed manifest: a human title, a body stored
//!   chunked in the content store, the owner, and the owner's
//!   signature. Notes read exactly like files: signature plus
//!   per-chunk hash verification on read.
//! * A **task** is a signed record: a stable human id, a title, a done
//!   flag, a priority, an optional due date, the owner, and the owner's
//!   signature. Toggling a task flips the flag and re-signs, producing
//!   a new record (content addressing: the task's address changes when
//!   its state changes). Tasks can depend on other tasks (completing
//!   is refused while a dependency is incomplete; cycles are refused),
//!   and recurring tasks create dated occurrences through `tick`.
//!   Notes are searchable by substring, and `due_soon` lists what is
//!   due within a window, overdue included.
//! * A **quick command** is a signed record: a name, a command string,
//!   the owner, and the owner's signature. "Running" a command only
//!   appends an execution entry (name, timestamp) to a log. **It does
//!   not execute any shell command, and it never will.** X•Kom's
//!   records what the owner asked for; running the thing itself is a
//!   separate, explicit act by the owner outside this crate.
//!
//! Manifest/record formats (explicit binary, no JSON):
//!
//! ```text
//! KNOTE1 + title length (u16 LE) + title bytes + body chunk root
//! CID (32) + owner public key (32) + signature (64).
//! XKTASK1 + id length (u16 LE) + id bytes + title length (u16 LE)
//! + title bytes + done flag (u8) + priority (u8: Low=0, Medium=1,
//! High=2) + due flag (u8) + due date (u64 LE, only when the flag
//! is 1) + owner public key (32) + signature (64).
//! XKOMCMD + name length (u16 LE) + name bytes + command length
//! (u32 LE) + command bytes + owner public key (32) + signature (64).
//! ```

use std::collections::HashMap;
use std::fmt;

use kernel_crypto::Ed25519Signature;
use kernel_identity::{Identity, IdentityError, PeerId};
use kernel_storage::{Cid, ContentStore, StorageError};

/// Magic bytes at the start of every note manifest.
const NOTE_MAGIC: &[u8; 6] = b"KNOTE1";
/// Domain separator for note manifest signatures.
const NOTE_SIGN_DOMAIN: &[u8] = b"KNOTE1";
/// Magic bytes at the start of every task record.
const TASK_MAGIC: &[u8; 7] = b"XKTASK1";
/// Domain separator for task signatures.
const TASK_SIGN_DOMAIN: &[u8] = b"XKTASK1";
/// Magic bytes at the start of every command record.
const COMMAND_MAGIC: &[u8; 7] = b"XKOMCMD";
/// Domain separator for command signatures.
const COMMAND_SIGN_DOMAIN: &[u8] = b"XKOMCMD";

/// Maximum length of a note title or task title.
pub const MAX_TITLE_LEN: usize = 128;
/// Maximum length of a task id or command name.
pub const MAX_ID_LEN: usize = 64;
/// Maximum command string length in characters.
pub const MAX_COMMAND_LEN: usize = 1_000;

/// Errors returned by personal-tool operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KomError {
    /// Wraps a [`StorageError`].
    Storage(StorageError),
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// A title broke the naming rules.
    InvalidTitle(String),
    /// A task id or command name broke the naming rules.
    InvalidId(String),
    /// A command string broke the rules.
    InvalidCommand(String),
    /// A manifest or record was malformed.
    InvalidManifest(String),
    /// A manifest's or record's signature did not verify.
    BadSignature,
    /// No note, task, or command with this CID is known here.
    NotFound(Cid),
    /// A task id or command name is already in use.
    AlreadyExists(String),
    /// The caller is not the owner of this task or command.
    NotOwner,
    /// Completing a task was refused because a dependency is not done.
    DependencyIncomplete(String),
    /// A dependency would create a cycle (or a task depends on itself).
    DependencyCycle(String),
    /// A task id named as a dependency is not held here.
    UnknownDependency(String),
    /// A task id is not held here.
    UnknownTask(String),
    /// A task cannot be deleted because another task depends on it.
    DependedOn(String),
    /// A recurring task's interval or count was invalid.
    InvalidRecurrence(String),
}

impl fmt::Display for KomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KomError::Storage(e) => write!(f, "storage error: {e}"),
            KomError::Identity(e) => write!(f, "identity error: {e}"),
            KomError::InvalidTitle(msg) => write!(f, "invalid title: {msg}"),
            KomError::InvalidId(msg) => write!(f, "invalid id: {msg}"),
            KomError::InvalidCommand(msg) => write!(f, "invalid command: {msg}"),
            KomError::InvalidManifest(msg) => write!(f, "invalid manifest: {msg}"),
            KomError::BadSignature => write!(f, "signature is invalid"),
            KomError::NotFound(cid) => write!(f, "not found: {cid}"),
            KomError::AlreadyExists(id) => write!(f, "already exists: {id}"),
            KomError::NotOwner => write!(f, "only the owner can do that"),
            KomError::DependencyIncomplete(id) => {
                write!(f, "dependency {id} is not done yet")
            }
            KomError::DependencyCycle(msg) => write!(f, "dependency cycle: {msg}"),
            KomError::UnknownDependency(id) => {
                write!(f, "unknown dependency task: {id}")
            }
            KomError::UnknownTask(id) => write!(f, "unknown task: {id}"),
            KomError::DependedOn(id) => {
                write!(f, "task {id} cannot be deleted: another task depends on it")
            }
            KomError::InvalidRecurrence(msg) => write!(f, "invalid recurrence: {msg}"),
        }
    }
}

impl std::error::Error for KomError {}

impl From<StorageError> for KomError {
    fn from(e: StorageError) -> Self {
        KomError::Storage(e)
    }
}

impl From<IdentityError> for KomError {
    fn from(e: IdentityError) -> Self {
        KomError::Identity(e)
    }
}

/// Rules for titles (notes, tasks): 1-128 printable ASCII characters,
/// no path separators.
fn validate_title(title: &str) -> Result<(), KomError> {
    if title.is_empty() {
        return Err(KomError::InvalidTitle(
            "title must not be empty".to_string(),
        ));
    }
    if title.len() > MAX_TITLE_LEN {
        return Err(KomError::InvalidTitle(format!(
            "title must be at most {MAX_TITLE_LEN} characters"
        )));
    }
    if title.contains('/') || title.contains('\\') {
        return Err(KomError::InvalidTitle(
            "title must not contain a path separator".to_string(),
        ));
    }
    if !title.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
        return Err(KomError::InvalidTitle(
            "title must be printable ASCII".to_string(),
        ));
    }
    Ok(())
}

/// Rules for ids (task ids) and command names: 1-64 characters, ASCII
/// letters, digits, `-`, `_`, `.`. Ids are human labels, not hashes.
fn validate_id(id: &str, kind: &str) -> Result<(), KomError> {
    if id.is_empty() {
        return Err(KomError::InvalidId(format!("{kind} must not be empty")));
    }
    if id.len() > MAX_ID_LEN {
        return Err(KomError::InvalidId(format!(
            "{kind} must be at most {MAX_ID_LEN} characters"
        )));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Err(KomError::InvalidId(format!(
            "{kind} may only use letters, digits, '-', '_' and '.'"
        )));
    }
    Ok(())
}

/// Rules for command strings: non-empty, at most [`MAX_COMMAND_LEN`] chars.
fn validate_command_str(command: &str) -> Result<(), KomError> {
    if command.trim().is_empty() {
        return Err(KomError::InvalidCommand(
            "command must not be empty".to_string(),
        ));
    }
    if command.chars().count() > MAX_COMMAND_LEN {
        return Err(KomError::InvalidCommand(format!(
            "command must be at most {MAX_COMMAND_LEN} characters"
        )));
    }
    Ok(())
}

/// A display name derived from a public key: the first 8 hex chars of
/// the key. Used when a manifest is decoded and only the key survives
/// (verification needs the key, not the human name).
fn fingerprint_name(key: &kernel_crypto::Ed25519PublicKey) -> String {
    let hex = kernel_crypto::to_hex(key.as_bytes());
    format!("peer-{}", &hex[..8])
}

/// The message an owner signs for a note manifest.
fn note_sign_message(title: &str, root: &Cid, owner: &PeerId) -> Vec<u8> {
    let mut msg = Vec::with_capacity(NOTE_SIGN_DOMAIN.len() + title.len() + 32 + 32);
    msg.extend_from_slice(NOTE_SIGN_DOMAIN);
    msg.extend_from_slice(title.as_bytes());
    msg.extend_from_slice(root.as_bytes());
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg
}

/// Task priority. Ordering is Low < Medium < High; listings sort
/// High first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Medium = 1,
    High = 2,
}

impl Priority {
    /// Decode a priority byte from a task record.
    fn from_byte(b: u8) -> Result<Self, KomError> {
        match b {
            0 => Ok(Priority::Low),
            1 => Ok(Priority::Medium),
            2 => Ok(Priority::High),
            other => Err(KomError::InvalidManifest(format!(
                "priority byte must be 0, 1, or 2, got {other}"
            ))),
        }
    }
}

/// The message an owner signs for a task record.
fn task_sign_message(
    id: &str,
    title: &str,
    done: bool,
    priority: Priority,
    due: Option<u64>,
    owner: &PeerId,
) -> Vec<u8> {
    let mut msg =
        Vec::with_capacity(TASK_SIGN_DOMAIN.len() + id.len() + title.len() + 1 + 1 + 1 + 8 + 32);
    msg.extend_from_slice(TASK_SIGN_DOMAIN);
    msg.extend_from_slice(id.as_bytes());
    msg.extend_from_slice(title.as_bytes());
    msg.push(done as u8);
    msg.push(priority as u8);
    match due {
        Some(d) => {
            msg.push(1);
            msg.extend_from_slice(&d.to_le_bytes());
        }
        None => msg.push(0),
    }
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg
}

/// The message an owner signs for a command record.
fn command_sign_message(name: &str, command: &str, owner: &PeerId) -> Vec<u8> {
    let mut msg = Vec::with_capacity(COMMAND_SIGN_DOMAIN.len() + name.len() + command.len() + 32);
    msg.extend_from_slice(COMMAND_SIGN_DOMAIN);
    msg.extend_from_slice(name.as_bytes());
    msg.extend_from_slice(command.as_bytes());
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg
}

/// A signed note manifest: title, chunked body root, owner, signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoteManifest {
    /// Human-readable note title.
    pub title: String,
    /// The owner who signed this manifest.
    pub owner: PeerId,
    /// Root CID of the chunked body content (a `KCHUNK1` manifest block).
    pub root: Cid,
    /// The owner's signature over title, root, and owner key.
    pub signature: Ed25519Signature,
}

impl NoteManifest {
    /// Build and sign a manifest as `identity`.
    pub fn new(title: &str, root: Cid, identity: &Identity) -> Result<Self, KomError> {
        validate_title(title)?;
        let owner = identity.peer_id().clone();
        let msg = note_sign_message(title, &root, &owner);
        let signature = identity.sign(&msg);
        Ok(Self {
            title: title.to_string(),
            owner,
            root,
            signature,
        })
    }

    /// True if the signature verifies against the owner for this exact
    /// title and root.
    pub fn verify(&self) -> bool {
        let msg = note_sign_message(&self.title, &self.root, &self.owner);
        self.owner.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary manifest format.
    pub fn encode(&self) -> Vec<u8> {
        let title_bytes = self.title.as_bytes();
        let mut out = Vec::with_capacity(6 + 2 + title_bytes.len() + 32 + 32 + 64);
        out.extend_from_slice(NOTE_MAGIC);
        out.extend_from_slice(&(title_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(title_bytes);
        out.extend_from_slice(self.root.as_bytes());
        out.extend_from_slice(self.owner.public_key().as_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a manifest produced by [`NoteManifest::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, KomError> {
        if bytes.len() < 6 + 2 + 32 + 32 + 64 || &bytes[..6] != NOTE_MAGIC {
            return Err(KomError::InvalidManifest(
                "missing KNOTE1 magic".to_string(),
            ));
        }
        let title_len = u16::from_le_bytes(bytes[6..8].try_into().expect("2-byte slice")) as usize;
        let expected = 8 + title_len + 32 + 32 + 64;
        if bytes.len() != expected {
            return Err(KomError::InvalidManifest(format!(
                "manifest is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let mut off = 8;
        let title = String::from_utf8(bytes[off..off + title_len].to_vec()).map_err(|_| {
            KomError::InvalidManifest("manifest title is not valid UTF-8".to_string())
        })?;
        validate_title(&title)?;
        off += title_len;
        let mut root_bytes = [0u8; 32];
        root_bytes.copy_from_slice(&bytes[off..off + 32]);
        let root = Cid::from_bytes(root_bytes);
        off += 32;
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let owner_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| KomError::InvalidManifest(e.to_string()))?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let owner = PeerId::new(owner_key, &fingerprint_name(&owner_key))?;
        Ok(Self {
            title,
            owner,
            root,
            signature,
        })
    }
}

/// A signed task record: stable id, title, done flag, priority, due
/// date, owner, signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KomTask {
    /// Stable human id, e.g. `"buy-milk"`.
    pub id: String,
    /// Human-readable task title.
    pub title: String,
    /// Whether the task is done.
    pub done: bool,
    /// Task priority.
    pub priority: Priority,
    /// Optional due date (unix timestamp, seconds), caller-supplied.
    pub due: Option<u64>,
    /// The owner who signed this record.
    pub owner: PeerId,
    /// The owner's signature over id, title, done flag, priority,
    /// due date, and owner key.
    pub signature: Ed25519Signature,
}

impl KomTask {
    /// Build and sign a task record as `identity`, with default
    /// priority (Medium) and no due date.
    pub fn new(id: &str, title: &str, done: bool, identity: &Identity) -> Result<Self, KomError> {
        Self::new_full(id, title, done, Priority::Medium, None, identity)
    }

    /// Build and sign a task record as `identity`, with an explicit
    /// priority and optional due date.
    pub fn new_full(
        id: &str,
        title: &str,
        done: bool,
        priority: Priority,
        due: Option<u64>,
        identity: &Identity,
    ) -> Result<Self, KomError> {
        validate_id(id, "task id")?;
        validate_title(title)?;
        let owner = identity.peer_id().clone();
        let msg = task_sign_message(id, title, done, priority, due, &owner);
        let signature = identity.sign(&msg);
        Ok(Self {
            id: id.to_string(),
            title: title.to_string(),
            done,
            priority,
            due,
            owner,
            signature,
        })
    }

    /// True if the signature verifies against the owner for this exact
    /// id, title, done flag, priority, and due date.
    pub fn verify(&self) -> bool {
        let msg = task_sign_message(
            &self.id,
            &self.title,
            self.done,
            self.priority,
            self.due,
            &self.owner,
        );
        self.owner.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary task format.
    pub fn encode(&self) -> Vec<u8> {
        let id_bytes = self.id.as_bytes();
        let title_bytes = self.title.as_bytes();
        let mut out = Vec::with_capacity(
            7 + 2 + id_bytes.len() + 2 + title_bytes.len() + 1 + 1 + 1 + 8 + 32 + 64,
        );
        out.extend_from_slice(TASK_MAGIC);
        out.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(id_bytes);
        out.extend_from_slice(&(title_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(title_bytes);
        out.push(self.done as u8);
        out.push(self.priority as u8);
        match self.due {
            Some(d) => {
                out.push(1);
                out.extend_from_slice(&d.to_le_bytes());
            }
            None => out.push(0),
        }
        out.extend_from_slice(self.owner.public_key().as_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a task produced by [`KomTask::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, KomError> {
        if bytes.len() < 7 + 2 + 2 + 1 + 1 + 1 + 32 + 64 || &bytes[..7] != TASK_MAGIC {
            return Err(KomError::InvalidManifest(
                "missing XKTASK1 magic".to_string(),
            ));
        }
        let id_len = u16::from_le_bytes(bytes[7..9].try_into().expect("2-byte slice")) as usize;
        let title_len_off = 9 + id_len;
        if bytes.len() < title_len_off + 2 {
            return Err(KomError::InvalidManifest(
                "task too short for title length".to_string(),
            ));
        }
        let title_len = u16::from_le_bytes(
            bytes[title_len_off..title_len_off + 2]
                .try_into()
                .expect("2-byte slice"),
        ) as usize;
        let after_title = title_len_off + 2 + title_len;
        // Fixed tail after the title: done(1) + priority(1) +
        // due flag(1) + owner(32) + signature(64), plus an optional
        // 8-byte due date.
        if bytes.len() < after_title + 1 + 1 + 1 + 32 + 64 {
            return Err(KomError::InvalidManifest(
                "task too short for flags and owner".to_string(),
            ));
        }
        let id = String::from_utf8(bytes[9..9 + id_len].to_vec())
            .map_err(|_| KomError::InvalidManifest("task id is not valid UTF-8".to_string()))?;
        validate_id(&id, "task id")?;
        let title = String::from_utf8(bytes[title_len_off + 2..after_title].to_vec())
            .map_err(|_| KomError::InvalidManifest("task title is not valid UTF-8".to_string()))?;
        validate_title(&title)?;
        let mut off = after_title;
        let done_byte = bytes[off];
        if done_byte > 1 {
            return Err(KomError::InvalidManifest(format!(
                "done flag must be 0 or 1, got {done_byte}"
            )));
        }
        let done = done_byte == 1;
        off += 1;
        let priority = Priority::from_byte(bytes[off])?;
        off += 1;
        let due_flag = bytes[off];
        if due_flag > 1 {
            return Err(KomError::InvalidManifest(format!(
                "due flag must be 0 or 1, got {due_flag}"
            )));
        }
        off += 1;
        let due = if due_flag == 1 {
            if bytes.len() < off + 8 + 32 + 64 {
                return Err(KomError::InvalidManifest(
                    "task too short for due date".to_string(),
                ));
            }
            let d = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
            off += 8;
            Some(d)
        } else {
            None
        };
        let expected = off + 32 + 64;
        if bytes.len() != expected {
            return Err(KomError::InvalidManifest(format!(
                "task is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let owner_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| KomError::InvalidManifest(e.to_string()))?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let owner = PeerId::new(owner_key, &fingerprint_name(&owner_key))?;
        Ok(Self {
            id,
            title,
            done,
            priority,
            due,
            owner,
            signature,
        })
    }
}

/// A signed quick-command record: name, command string, owner, signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KomCommand {
    /// Human-readable command name, e.g. `"backup"`.
    pub name: String,
    /// The command string (recorded, never executed by this crate).
    pub command: String,
    /// The owner who signed this record.
    pub owner: PeerId,
    /// The owner's signature over name, command, and owner key.
    pub signature: Ed25519Signature,
}

impl KomCommand {
    /// Build and sign a command record as `identity`.
    pub fn new(name: &str, command: &str, identity: &Identity) -> Result<Self, KomError> {
        validate_id(name, "command name")?;
        validate_command_str(command)?;
        let owner = identity.peer_id().clone();
        let msg = command_sign_message(name, command, &owner);
        let signature = identity.sign(&msg);
        Ok(Self {
            name: name.to_string(),
            command: command.to_string(),
            owner,
            signature,
        })
    }

    /// True if the signature verifies against the owner for this exact
    /// name and command string.
    pub fn verify(&self) -> bool {
        let msg = command_sign_message(&self.name, &self.command, &self.owner);
        self.owner.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary command format.
    pub fn encode(&self) -> Vec<u8> {
        let name_bytes = self.name.as_bytes();
        let command_bytes = self.command.as_bytes();
        let mut out =
            Vec::with_capacity(7 + 2 + name_bytes.len() + 4 + command_bytes.len() + 32 + 64);
        out.extend_from_slice(COMMAND_MAGIC);
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(&(command_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(command_bytes);
        out.extend_from_slice(self.owner.public_key().as_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a command produced by [`KomCommand::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, KomError> {
        if bytes.len() < 7 + 2 + 4 + 32 + 64 || &bytes[..7] != COMMAND_MAGIC {
            return Err(KomError::InvalidManifest(
                "missing XKOMCMD magic".to_string(),
            ));
        }
        let name_len = u16::from_le_bytes(bytes[7..9].try_into().expect("2-byte slice")) as usize;
        let cmd_len_off = 9 + name_len;
        if bytes.len() < cmd_len_off + 4 {
            return Err(KomError::InvalidManifest(
                "command too short for command length".to_string(),
            ));
        }
        let cmd_len = u32::from_le_bytes(
            bytes[cmd_len_off..cmd_len_off + 4]
                .try_into()
                .expect("4-byte slice"),
        ) as usize;
        if cmd_len > MAX_COMMAND_LEN {
            return Err(KomError::InvalidManifest(format!(
                "command claims {cmd_len} chars, max is {MAX_COMMAND_LEN}"
            )));
        }
        let expected = cmd_len_off + 4 + cmd_len + 32 + 64;
        if bytes.len() != expected {
            return Err(KomError::InvalidManifest(format!(
                "command is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let name = String::from_utf8(bytes[9..9 + name_len].to_vec()).map_err(|_| {
            KomError::InvalidManifest("command name is not valid UTF-8".to_string())
        })?;
        validate_id(&name, "command name")?;
        let mut off = cmd_len_off + 4;
        let command = String::from_utf8(bytes[off..off + cmd_len].to_vec()).map_err(|_| {
            KomError::InvalidManifest("command string is not valid UTF-8".to_string())
        })?;
        validate_command_str(&command)?;
        off += cmd_len;
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let owner_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| KomError::InvalidManifest(e.to_string()))?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let owner = PeerId::new(owner_key, &fingerprint_name(&owner_key))?;
        Ok(Self {
            name,
            command,
            owner,
            signature,
        })
    }
}

/// One recorded execution of a quick command: which command, under
/// what name, and when. This is a log entry only — recording it runs
/// nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandExecution {
    /// The record CID of the command that was "run".
    pub command_cid: Cid,
    /// The command name, for human-readable logs.
    pub name: String,
    /// Unix timestamp (seconds) the execution was recorded.
    pub timestamp: u64,
}

/// The metadata of a note, as a human sees it — no CIDs, no hashes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoteMetadata {
    /// Human-readable note title.
    pub title: String,
    /// Who owns (wrote) this note.
    pub owner: PeerId,
}

/// Store-side metadata for a task: creation order, dependencies, and
/// recurring-series linkage. Not part of the signed record — the
/// record carries id, title, done, priority, and due date.
#[derive(Clone, Debug)]
struct TaskMeta {
    /// Creation sequence number; orders tasks inside the same sort tier.
    seq: u64,
    /// Task ids this task depends on.
    deps: Vec<String>,
    /// (series base id, occurrence number) for recurring occurrences.
    series: Option<(String, u32)>,
}

/// State of a recurring task series.
#[derive(Clone, Debug)]
struct SeriesState {
    /// Human title shared by all occurrences.
    title: String,
    /// Priority shared by all occurrences.
    priority: Priority,
    /// Seconds between occurrences.
    interval: u64,
    /// Total occurrences to create, including the first.
    total: u32,
    /// Occurrences created so far.
    created: u32,
    /// Due timestamp of the next occurrence to create.
    next_due: u64,
    /// The owner who created the series.
    owner: PeerId,
}

/// The owner's personal store: a content store holding signed notes,
/// tasks, and commands, plus indexes and the execution log.
pub struct KomStore {
    store: ContentStore,
    /// Note manifests this node holds, by manifest CID.
    notes: HashMap<Cid, NoteManifest>,
    /// Note manifest CIDs in insertion order (newest last).
    note_order: Vec<Cid>,
    /// Task records this node holds, by record CID.
    tasks: HashMap<Cid, KomTask>,
    /// Task record CIDs, by task id.
    task_ids: HashMap<String, Cid>,
    /// Per-task metadata, by task id.
    task_meta: HashMap<String, TaskMeta>,
    /// Recurring series, by base task id.
    series: HashMap<String, SeriesState>,
    /// Next creation sequence number.
    next_seq: u64,
    /// Command records this node holds, by record CID.
    commands: HashMap<Cid, KomCommand>,
    /// Command record CIDs, by command name.
    command_names: HashMap<String, Cid>,
    /// Execution log, in the order commands were "run".
    executions: Vec<CommandExecution>,
}

impl Default for KomStore {
    /// An empty personal store.
    fn default() -> Self {
        Self::new()
    }
}

impl KomStore {
    /// An empty personal store.
    pub fn new() -> Self {
        Self {
            store: ContentStore::new(),
            notes: HashMap::new(),
            note_order: Vec::new(),
            tasks: HashMap::new(),
            task_ids: HashMap::new(),
            task_meta: HashMap::new(),
            series: HashMap::new(),
            next_seq: 0,
            commands: HashMap::new(),
            command_names: HashMap::new(),
            executions: Vec::new(),
        }
    }

    /// Number of notes held.
    pub fn len(&self) -> usize {
        self.notes.len()
    }

    /// True if no notes are held.
    pub fn is_empty(&self) -> bool {
        self.notes.is_empty()
    }

    /// The manifest for a note, if this node holds it.
    pub fn get_note_manifest(&self, manifest_cid: &Cid) -> Option<&NoteManifest> {
        self.notes.get(manifest_cid)
    }

    /// Store a note: chunk the body, build and sign the manifest, and
    /// store the manifest as a block. Returns the manifest CID.
    pub fn put_note(
        &mut self,
        title: &str,
        body: &[u8],
        identity: &Identity,
    ) -> Result<Cid, KomError> {
        validate_title(title)?;
        let root = self.store.put_chunked(body);
        let manifest = NoteManifest::new(title, root, identity)?;
        let manifest_cid = self.store.put(&manifest.encode());
        self.notes.insert(manifest_cid, manifest);
        self.note_order.push(manifest_cid);
        Ok(manifest_cid)
    }

    /// Fetch a note by manifest CID: verify the manifest signature,
    /// then reassemble the chunked body (every chunk hash-verified).
    pub fn get_note(&self, manifest_cid: &Cid) -> Result<(Vec<u8>, NoteMetadata), KomError> {
        let manifest = match self.notes.get(manifest_cid) {
            Some(m) => m.clone(),
            None => {
                let bytes = self.store.get(manifest_cid).map_err(|e| match e {
                    StorageError::NotFound(cid) => KomError::NotFound(cid),
                    other => KomError::Storage(other),
                })?;
                NoteManifest::decode(&bytes)?
            }
        };
        if !manifest.verify() {
            return Err(KomError::BadSignature);
        }
        let body = self
            .store
            .get_chunked(&manifest.root)
            .map_err(KomError::Storage)?;
        let metadata = NoteMetadata {
            title: manifest.title,
            owner: manifest.owner,
        };
        Ok((body, metadata))
    }

    /// All notes held, as (manifest CID, title), sorted by title.
    pub fn list_notes(&self) -> Vec<(Cid, String)> {
        let mut out: Vec<(Cid, String)> = self
            .notes
            .iter()
            .map(|(cid, m)| (*cid, m.title.clone()))
            .collect();
        out.sort_by(|a, b| a.1.cmp(&b.1));
        out
    }

    /// Create a task: build and sign the record (initially not done)
    /// and store it. Returns the task record CID. A duplicate id is refused.
    /// Priority defaults to Medium and there is no due date; use
    /// [`KomStore::create_task_full`] for priority, due date, and
    /// dependencies.
    pub fn create_task(
        &mut self,
        id: &str,
        title: &str,
        identity: &Identity,
    ) -> Result<Cid, KomError> {
        self.create_task_full(id, title, Priority::Medium, None, &[], identity)
    }

    /// Create a task with a priority, an optional due date, and
    /// dependencies on other tasks (by id). Dependencies must already
    /// exist; a dependency cycle (including depending on yourself) is
    /// refused. Returns the task record CID.
    pub fn create_task_full(
        &mut self,
        id: &str,
        title: &str,
        priority: Priority,
        due: Option<u64>,
        deps: &[&str],
        identity: &Identity,
    ) -> Result<Cid, KomError> {
        validate_id(id, "task id")?;
        validate_title(title)?;
        if self.task_ids.contains_key(id) {
            return Err(KomError::AlreadyExists(id.to_string()));
        }
        let dep_ids: Vec<String> = deps.iter().map(|d| d.to_string()).collect();
        for dep in &dep_ids {
            if dep == id {
                return Err(KomError::DependencyCycle(format!(
                    "task {id} cannot depend on itself"
                )));
            }
            if !self.task_ids.contains_key(dep.as_str()) {
                return Err(KomError::UnknownDependency(dep.clone()));
            }
        }
        if self.would_cycle(id, &dep_ids) {
            return Err(KomError::DependencyCycle(format!(
                "dependencies of {id} would create a cycle"
            )));
        }
        let task = KomTask::new_full(id, title, false, priority, due, identity)?;
        let cid = self.store.put(&task.encode());
        self.tasks.insert(cid, task);
        self.task_ids.insert(id.to_string(), cid);
        self.task_meta.insert(
            id.to_string(),
            TaskMeta {
                seq: self.next_seq,
                deps: dep_ids,
                series: None,
            },
        );
        self.next_seq += 1;
        Ok(cid)
    }

    /// True if adding edges from `task_id` to `new_deps` would close a
    /// dependency cycle. Walks the existing dependency graph from each
    /// new dependency; reaching `task_id` means a cycle.
    fn would_cycle(&self, task_id: &str, new_deps: &[String]) -> bool {
        let mut stack: Vec<String> = new_deps.to_vec();
        let mut seen: Vec<String> = Vec::new();
        while let Some(cur) = stack.pop() {
            if cur == task_id {
                return true;
            }
            if seen.contains(&cur) {
                continue;
            }
            seen.push(cur.clone());
            if let Some(meta) = self.task_meta.get(&cur) {
                stack.extend(meta.deps.iter().cloned());
            }
        }
        false
    }

    /// Refuse completing task `id` while any dependency is incomplete.
    /// A dependency that cannot be found is treated as incomplete.
    fn check_dependencies_met(&self, id: &str) -> Result<(), KomError> {
        let deps = match self.task_meta.get(id) {
            Some(m) => m.deps.clone(),
            None => return Ok(()),
        };
        for dep in deps {
            let done = self
                .task_ids
                .get(&dep)
                .and_then(|cid| self.tasks.get(cid))
                .map(|t| t.done)
                .unwrap_or(false);
            if !done {
                return Err(KomError::DependencyIncomplete(dep));
            }
        }
        Ok(())
    }

    /// The task ids a task depends on, in the order they were added.
    pub fn dependencies_of(&self, task_id: &str) -> Vec<String> {
        self.task_meta
            .get(task_id)
            .map(|m| m.deps.clone())
            .unwrap_or_default()
    }

    /// Add a dependency to an existing task: only the task's owner can.
    /// Refused when the task or the dependency does not exist, when
    /// the dependency is the task itself, or when it would close a
    /// cycle. Adding a dependency that already exists is a no-op
    /// success.
    pub fn add_dependency(
        &mut self,
        task_id: &str,
        dep_id: &str,
        identity: &Identity,
    ) -> Result<(), KomError> {
        let cid = match self.task_ids.get(task_id) {
            Some(c) => *c,
            None => return Err(KomError::UnknownTask(task_id.to_string())),
        };
        let task = self.get_task(&cid)?;
        if task.owner != *identity.peer_id() {
            return Err(KomError::NotOwner);
        }
        if dep_id == task_id {
            return Err(KomError::DependencyCycle(format!(
                "task {task_id} cannot depend on itself"
            )));
        }
        if !self.task_ids.contains_key(dep_id) {
            return Err(KomError::UnknownDependency(dep_id.to_string()));
        }
        if self.would_cycle(task_id, &[dep_id.to_string()]) {
            return Err(KomError::DependencyCycle(format!(
                "depending on {dep_id} would create a cycle"
            )));
        }
        if let Some(meta) = self.task_meta.get_mut(task_id) {
            if !meta.deps.iter().any(|d| d == dep_id) {
                meta.deps.push(dep_id.to_string());
            }
        }
        Ok(())
    }

    /// Fetch a task by record CID, verifying its signature.
    pub fn get_task(&self, task_cid: &Cid) -> Result<KomTask, KomError> {
        let task = match self.tasks.get(task_cid) {
            Some(t) => t.clone(),
            None => {
                let bytes = self.store.get(task_cid).map_err(|e| match e {
                    StorageError::NotFound(cid) => KomError::NotFound(cid),
                    other => KomError::Storage(other),
                })?;
                KomTask::decode(&bytes)?
            }
        };
        if !task.verify() {
            return Err(KomError::BadSignature);
        }
        Ok(task)
    }

    /// The record CID for a task id, if held.
    pub fn task_cid(&self, id: &str) -> Option<Cid> {
        self.task_ids.get(id).copied()
    }

    /// Set a task's done flag (owner only). Completing (done=true) is
    /// refused while any dependency is incomplete; un-completing is
    /// always allowed. The new record is signed and stored as a new
    /// block, preserving priority and due date, and the index moves to
    /// the new record CID (the old record stays in the content store
    /// as history). Returns the new record CID and the new done state.
    fn set_done(
        &mut self,
        task_cid: &Cid,
        done: bool,
        identity: &Identity,
    ) -> Result<(Cid, bool), KomError> {
        let task = self.get_task(task_cid)?;
        if task.owner != *identity.peer_id() {
            return Err(KomError::NotOwner);
        }
        if done && !task.done {
            self.check_dependencies_met(&task.id)?;
        }
        let flipped = KomTask::new_full(
            &task.id,
            &task.title,
            done,
            task.priority,
            task.due,
            identity,
        )?;
        let new_cid = self.store.put(&flipped.encode());
        self.tasks.remove(task_cid);
        self.tasks.insert(new_cid, flipped);
        self.task_ids.insert(task.id, new_cid);
        Ok((new_cid, done))
    }

    /// Toggle a task's done flag: only the task's owner can toggle.
    /// Completing is refused while a dependency is incomplete. The
    /// flipped record is signed and stored as a new block, and the
    /// index moves to the new record CID (the old record stays in the
    /// content store as history). Returns the new record CID and the
    /// new done state.
    pub fn toggle_task(
        &mut self,
        task_cid: &Cid,
        identity: &Identity,
    ) -> Result<(Cid, bool), KomError> {
        let task = self.get_task(task_cid)?;
        self.set_done(task_cid, !task.done, identity)
    }

    /// Mark a task done (owner only). Completing is refused while any
    /// dependency is incomplete. An already-done task succeeds
    /// without creating a new record.
    pub fn complete_task(
        &mut self,
        task_cid: &Cid,
        identity: &Identity,
    ) -> Result<(Cid, bool), KomError> {
        let task = self.get_task(task_cid)?;
        if task.done {
            return Ok((*task_cid, true));
        }
        self.set_done(task_cid, true, identity)
    }

    /// Delete a task (owner only). Refused when another task depends
    /// on it. The record block stays in the content store as history;
    /// the indexes forget it. Deleting a recurring series' base task
    /// also cancels the series (no further occurrences); occurrences
    /// already created stay as ordinary tasks.
    pub fn delete_task(&mut self, task_id: &str, identity: &Identity) -> Result<(), KomError> {
        let cid = match self.task_ids.get(task_id) {
            Some(c) => *c,
            None => return Err(KomError::UnknownTask(task_id.to_string())),
        };
        let task = self.get_task(&cid)?;
        if task.owner != *identity.peer_id() {
            return Err(KomError::NotOwner);
        }
        for (other_id, meta) in &self.task_meta {
            if other_id != task_id && meta.deps.iter().any(|d| d == task_id) {
                return Err(KomError::DependedOn(task_id.to_string()));
            }
        }
        self.tasks.remove(&cid);
        self.task_ids.remove(task_id);
        self.task_meta.remove(task_id);
        self.series.remove(task_id);
        Ok(())
    }

    /// All tasks held, as (record CID, title, done), sorted by title.
    pub fn list_tasks(&self) -> Vec<(Cid, String, bool)> {
        let mut out: Vec<(Cid, String, bool)> = self
            .tasks
            .iter()
            .map(|(cid, t)| (*cid, t.title.clone(), t.done))
            .collect();
        out.sort_by(|a, b| a.1.cmp(&b.1));
        out
    }

    /// All tasks sorted by priority (High first), then due date
    /// (earliest first; tasks with no due date last), then creation
    /// order. The sort is exactly: priority descending, due ascending
    /// with None last, creation sequence ascending.
    pub fn tasks_by_priority(&self) -> Vec<(Cid, KomTask)> {
        let mut out: Vec<(Cid, KomTask, u64)> = self
            .tasks
            .iter()
            .map(|(cid, t)| {
                let seq = self.task_meta.get(&t.id).map(|m| m.seq).unwrap_or(u64::MAX);
                (*cid, t.clone(), seq)
            })
            .collect();
        out.sort_by(|a, b| {
            b.1.priority
                .cmp(&a.1.priority)
                .then_with(|| match (a.1.due, b.1.due) {
                    (Some(x), Some(y)) => x.cmp(&y),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                })
                .then_with(|| a.2.cmp(&b.2))
        });
        out.into_iter().map(|(cid, t, _)| (cid, t)).collect()
    }

    /// Tasks due strictly before `ts`, earliest due first (ties by
    /// creation order). Tasks with no due date are never included.
    pub fn tasks_due_before(&self, ts: u64) -> Vec<(Cid, KomTask)> {
        let mut out: Vec<(Cid, KomTask, u64)> = self
            .tasks
            .iter()
            .filter(|(_, t)| matches!(t.due, Some(d) if d < ts))
            .map(|(cid, t)| {
                let seq = self.task_meta.get(&t.id).map(|m| m.seq).unwrap_or(u64::MAX);
                (*cid, t.clone(), seq)
            })
            .collect();
        out.sort_by(|a, b| {
            a.1.due
                .unwrap()
                .cmp(&b.1.due.unwrap())
                .then_with(|| a.2.cmp(&b.2))
        });
        out.into_iter().map(|(cid, t, _)| (cid, t)).collect()
    }

    /// Tasks due within the window ending at `now + window_secs`,
    /// including overdue tasks of any age, sorted by due date
    /// (earliest first, ties by creation order). Tasks with no due
    /// date are never included. Time is caller-supplied: a wrong
    /// clock means wrong reminders.
    pub fn due_soon(&self, now: u64, window_secs: u64) -> Vec<(Cid, KomTask)> {
        let horizon = now.saturating_add(window_secs);
        let mut out: Vec<(Cid, KomTask, u64)> = self
            .tasks
            .iter()
            .filter(|(_, t)| matches!(t.due, Some(d) if d <= horizon))
            .map(|(cid, t)| {
                let seq = self.task_meta.get(&t.id).map(|m| m.seq).unwrap_or(u64::MAX);
                (*cid, t.clone(), seq)
            })
            .collect();
        out.sort_by(|a, b| {
            a.1.due
                .unwrap()
                .cmp(&b.1.due.unwrap())
                .then_with(|| a.2.cmp(&b.2))
        });
        out.into_iter().map(|(cid, t, _)| (cid, t)).collect()
    }

    /// Create a recurring task: occurrence #1 is created immediately
    /// with due date `first_due`; `tick` creates the rest. `interval`
    /// is seconds between occurrences; `count` is the total number of
    /// occurrences including the first. Occurrences after the first
    /// get ids `{id}-002`, `{id}-003`, ... . Interval 0 or count 0 is
    /// refused.
    // Eight parameters, each a distinct caller-supplied field of the
    // series; grouping them would hide what a recurring task needs.
    #[allow(clippy::too_many_arguments)]
    pub fn create_recurring_task(
        &mut self,
        id: &str,
        title: &str,
        priority: Priority,
        first_due: u64,
        interval: u64,
        count: u32,
        identity: &Identity,
    ) -> Result<Cid, KomError> {
        if interval == 0 || count == 0 {
            return Err(KomError::InvalidRecurrence(format!(
                "interval must be > 0 and count must be >= 1, got interval {interval}, count {count}"
            )));
        }
        let cid = self.create_task_full(id, title, priority, Some(first_due), &[], identity)?;
        if let Some(meta) = self.task_meta.get_mut(id) {
            meta.series = Some((id.to_string(), 1));
        }
        self.series.insert(
            id.to_string(),
            SeriesState {
                title: title.to_string(),
                priority,
                interval,
                total: count,
                created: 1,
                next_due: first_due.saturating_add(interval),
                owner: identity.peer_id().clone(),
            },
        );
        Ok(cid)
    }

    /// Advance recurring series to `now`: for every series owned by
    /// `identity`, create each occurrence whose due time has arrived,
    /// in order, until the series is complete. Returns the record CIDs
    /// of the occurrences created, in creation order. A late tick
    /// creates all due occurrences at once. Nothing runs in the
    /// background — `tick` is the only thing that advances a series.
    /// Stops at the first error (e.g. an occurrence id already taken
    /// by a manually created task); occurrences created before the
    /// error are kept.
    pub fn tick(&mut self, now: u64, identity: &Identity) -> Result<Vec<Cid>, KomError> {
        let mut bases: Vec<String> = self.series.keys().cloned().collect();
        bases.sort();
        let mut created = Vec::new();
        for base in bases {
            loop {
                let (due, next_n, priority, title) = {
                    let Some(state) = self.series.get(&base) else {
                        break;
                    };
                    if state.owner != *identity.peer_id() {
                        break;
                    }
                    if state.created >= state.total || state.next_due > now {
                        break;
                    }
                    (
                        state.next_due,
                        state.created + 1,
                        state.priority,
                        state.title.clone(),
                    )
                };
                let occ_id = format!("{base}-{next_n:03}");
                let cid =
                    self.create_task_full(&occ_id, &title, priority, Some(due), &[], identity)?;
                if let Some(meta) = self.task_meta.get_mut(&occ_id) {
                    meta.series = Some((base.clone(), next_n));
                }
                if let Some(state) = self.series.get_mut(&base) {
                    state.created = next_n;
                    state.next_due = due.saturating_add(state.interval);
                }
                created.push(cid);
            }
        }
        Ok(created)
    }

    /// Search notes by substring over titles and bodies,
    /// case-insensitive, newest first. An empty query matches every
    /// note. A note whose body cannot be read is matched on its title
    /// only.
    pub fn search_notes(&self, query: &str) -> Vec<(Cid, String)> {
        let q = query.to_ascii_lowercase();
        let mut out = Vec::new();
        for cid in self.note_order.iter().rev() {
            let Some(manifest) = self.notes.get(cid) else {
                continue;
            };
            let title_lower = manifest.title.to_ascii_lowercase();
            let body_matches = self
                .store
                .get_chunked(&manifest.root)
                .map(|body| {
                    String::from_utf8_lossy(&body)
                        .to_ascii_lowercase()
                        .contains(&q)
                })
                .unwrap_or(false);
            if title_lower.contains(&q) || body_matches {
                out.push((*cid, manifest.title.clone()));
            }
        }
        out
    }

    /// Register a quick command: build and sign the record and store
    /// it. Returns the command record CID. A duplicate name is refused.
    pub fn register_command(
        &mut self,
        name: &str,
        command: &str,
        identity: &Identity,
    ) -> Result<Cid, KomError> {
        validate_id(name, "command name")?;
        validate_command_str(command)?;
        if self.command_names.contains_key(name) {
            return Err(KomError::AlreadyExists(name.to_string()));
        }
        let record = KomCommand::new(name, command, identity)?;
        let cid = self.store.put(&record.encode());
        self.commands.insert(cid, record);
        self.command_names.insert(name.to_string(), cid);
        Ok(cid)
    }

    /// Fetch a command by record CID, verifying its signature.
    pub fn get_command(&self, command_cid: &Cid) -> Result<KomCommand, KomError> {
        let command = match self.commands.get(command_cid) {
            Some(c) => c.clone(),
            None => {
                let bytes = self.store.get(command_cid).map_err(|e| match e {
                    StorageError::NotFound(cid) => KomError::NotFound(cid),
                    other => KomError::Storage(other),
                })?;
                KomCommand::decode(&bytes)?
            }
        };
        if !command.verify() {
            return Err(KomError::BadSignature);
        }
        Ok(command)
    }

    /// All commands held, as (record CID, name, command string),
    /// sorted by name.
    pub fn list_commands(&self) -> Vec<(Cid, String, String)> {
        let mut out: Vec<(Cid, String, String)> = self
            .commands
            .iter()
            .map(|(cid, c)| (*cid, c.name.clone(), c.command.clone()))
            .collect();
        out.sort_by(|a, b| a.1.cmp(&b.1));
        out
    }

    /// "Run" a command: record an execution entry (command, name,
    /// timestamp) in the log and return it. Only the command's owner
    /// can run it. **This executes nothing.** No shell, no subprocess,
    /// no side effect beyond the log entry. Running the thing itself
    /// is the owner's explicit act, outside this crate.
    pub fn run_command(
        &mut self,
        command_cid: &Cid,
        timestamp: u64,
        runner: &Identity,
    ) -> Result<CommandExecution, KomError> {
        let command = self.get_command(command_cid)?;
        if command.owner != *runner.peer_id() {
            return Err(KomError::NotOwner);
        }
        let execution = CommandExecution {
            command_cid: *command_cid,
            name: command.name.clone(),
            timestamp,
        };
        self.executions.push(execution.clone());
        Ok(execution)
    }

    /// The execution log, in the order commands were "run".
    pub fn list_executions(&self) -> Vec<CommandExecution> {
        self.executions.clone()
    }

    /// Mutable access to the underlying content store. Test-only:
    /// lets tests simulate corruption (see `put_unchecked`).
    #[cfg(test)]
    pub fn store_mut(&mut self) -> &mut ContentStore {
        &mut self.store
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_round_trip_small_and_large() {
        let alice = Identity::generate("alice").unwrap();
        let mut kom = KomStore::new();
        let body = b"remember to back up the keys";
        let cid = kom.put_note("reminder", body, &alice).unwrap();
        let (data, meta) = kom.get_note(&cid).unwrap();
        assert_eq!(data, body);
        assert_eq!(meta.title, "reminder");
        assert_eq!(meta.owner, *alice.peer_id());
        assert_eq!(kom.len(), 1);

        let large: Vec<u8> = (0..600 * 1024)
            .map(|i| ((i * 37 + 11) % 251) as u8)
            .collect();
        let big_cid = kom.put_note("big note", &large, &alice).unwrap();
        let (data, _) = kom.get_note(&big_cid).unwrap();
        assert_eq!(data, large);
        assert_eq!(kom.len(), 2);
    }

    #[test]
    fn tampered_note_signature_fails() {
        let alice = Identity::generate("alice").unwrap();
        let mut kom = KomStore::new();
        let cid = kom.put_note("secret", b"do not share", &alice).unwrap();
        let mut bytes = kom.store_mut().get(&cid).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let tampered_cid = kom.store_mut().put(&bytes);
        assert_eq!(kom.get_note(&tampered_cid), Err(KomError::BadSignature));
        assert_eq!(kom.get_note(&cid).unwrap().0, b"do not share");
    }

    #[test]
    fn task_create_toggle_list() {
        let alice = Identity::generate("alice").unwrap();
        let mut kom = KomStore::new();
        let cid = kom.create_task("buy-milk", "Buy milk", &alice).unwrap();
        let task = kom.get_task(&cid).unwrap();
        assert_eq!(task.id, "buy-milk");
        assert_eq!(task.title, "Buy milk");
        assert!(!task.done);
        // Toggle once: done.
        let (cid2, done) = kom.toggle_task(&cid, &alice).unwrap();
        assert!(done);
        assert_ne!(cid2, cid);
        let task = kom.get_task(&cid2).unwrap();
        assert!(task.done);
        assert_eq!(task.id, "buy-milk");
        // The old record CID still resolves as history (done=false);
        // the index now points at the new record.
        let old = kom.get_task(&cid).unwrap();
        assert!(!old.done);
        assert_eq!(kom.task_cid("buy-milk"), Some(cid2));
        // Toggle again: not done.
        let (cid3, done) = kom.toggle_task(&cid2, &alice).unwrap();
        assert!(!done);
        assert!(!kom.get_task(&cid3).unwrap().done);
        // List shows the current state.
        let listed = kom.list_tasks();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].1, "Buy milk");
        assert!(!listed[0].2);
        // Duplicate id refused.
        assert_eq!(
            kom.create_task("buy-milk", "Buy milk again", &alice),
            Err(KomError::AlreadyExists("buy-milk".to_string()))
        );
    }

    #[test]
    fn toggle_requires_owner() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut kom = KomStore::new();
        let cid = kom.create_task("task-1", "Alice's task", &alice).unwrap();
        assert_eq!(kom.toggle_task(&cid, &bob), Err(KomError::NotOwner));
        // Still not done after the refused toggle.
        assert!(!kom.get_task(&cid).unwrap().done);
    }

    #[test]
    fn tampered_task_signature_fails() {
        let alice = Identity::generate("alice").unwrap();
        let mut kom = KomStore::new();
        let cid = kom.create_task("task-1", "Title", &alice).unwrap();
        let mut bytes = kom.store_mut().get(&cid).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let tampered_cid = kom.store_mut().put(&bytes);
        assert_eq!(kom.get_task(&tampered_cid), Err(KomError::BadSignature));
        // Tampering with the done flag after signing breaks it too.
        let mut task = KomTask::new("task-2", "Title", false, &alice).unwrap();
        assert!(task.verify());
        task.done = true;
        assert!(!task.verify());
    }

    #[test]
    fn command_register_list_run_records_execution() {
        let alice = Identity::generate("alice").unwrap();
        let mut kom = KomStore::new();
        let cid = kom
            .register_command("backup", "tar -czf backup.tar.gz ~/notes", &alice)
            .unwrap();
        let command = kom.get_command(&cid).unwrap();
        assert_eq!(command.name, "backup");
        assert_eq!(command.command, "tar -czf backup.tar.gz ~/notes");
        assert_eq!(command.owner, *alice.peer_id());
        let listed = kom.list_commands();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].1, "backup");
        // "Running" records an execution entry — and nothing else.
        assert!(kom.list_executions().is_empty());
        let exec = kom.run_command(&cid, 1_700_000_000, &alice).unwrap();
        assert_eq!(exec.command_cid, cid);
        assert_eq!(exec.name, "backup");
        assert_eq!(exec.timestamp, 1_700_000_000);
        assert_eq!(kom.list_executions().len(), 1);
        kom.run_command(&cid, 1_700_000_001, &alice).unwrap();
        assert_eq!(kom.list_executions().len(), 2);
        // The command record itself is unchanged by running.
        assert_eq!(
            kom.get_command(&cid).unwrap().command,
            "tar -czf backup.tar.gz ~/notes"
        );
        // Duplicate name refused; running requires the owner.
        assert_eq!(
            kom.register_command("backup", "other", &alice),
            Err(KomError::AlreadyExists("backup".to_string()))
        );
        let bob = Identity::generate("bob").unwrap();
        assert_eq!(
            kom.run_command(&cid, 1_700_000_002, &bob),
            Err(KomError::NotOwner)
        );
        assert_eq!(kom.list_executions().len(), 2);
    }

    #[test]
    fn tampered_command_signature_fails() {
        let alice = Identity::generate("alice").unwrap();
        let mut kom = KomStore::new();
        let cid = kom
            .register_command("backup", "do a backup", &alice)
            .unwrap();
        let mut bytes = kom.store_mut().get(&cid).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let tampered_cid = kom.store_mut().put(&bytes);
        assert_eq!(kom.get_command(&tampered_cid), Err(KomError::BadSignature));
        assert!(kom.get_command(&cid).is_ok());
    }

    #[test]
    fn list_notes_sorted_by_title() {
        let alice = Identity::generate("alice").unwrap();
        let mut kom = KomStore::new();
        kom.put_note("zebra", b"1", &alice).unwrap();
        kom.put_note("apple", b"2", &alice).unwrap();
        kom.put_note("mango", b"3", &alice).unwrap();
        let listed = kom.list_notes();
        let titles: Vec<&str> = listed.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(titles, vec!["apple", "mango", "zebra"]);
    }

    #[test]
    fn note_task_command_encode_decode_round_trips() {
        let alice = Identity::generate("alice").unwrap();
        let root = Cid::from_bytes(kernel_crypto::sha256(b"body chunks"));
        let note = NoteManifest::new("title", root, &alice).unwrap();
        let decoded = NoteManifest::decode(&note.encode()).unwrap();
        assert_eq!(decoded.title, "title");
        assert_eq!(decoded.root, root);
        assert!(decoded.verify());
        let task = KomTask::new("id-1", "Title", true, &alice).unwrap();
        let decoded = KomTask::decode(&task.encode()).unwrap();
        assert_eq!(decoded.id, "id-1");
        assert!(decoded.done);
        assert!(decoded.verify());
        let command = KomCommand::new("run", "echo hi", &alice).unwrap();
        let decoded = KomCommand::decode(&command.encode()).unwrap();
        assert_eq!(decoded.name, "run");
        assert_eq!(decoded.command, "echo hi");
        assert!(decoded.verify());
        assert!(NoteManifest::decode(b"garbage").is_err());
        assert!(KomTask::decode(b"garbage").is_err());
        assert!(KomCommand::decode(b"garbage").is_err());
    }

    #[test]
    fn task_record_round_trip_with_priority_and_due() {
        let alice = Identity::generate("alice").unwrap();
        let task = KomTask::new_full("id-9", "Title", false, Priority::High, Some(12_345), &alice)
            .unwrap();
        let decoded = KomTask::decode(&task.encode()).unwrap();
        assert_eq!(decoded.priority, Priority::High);
        assert_eq!(decoded.due, Some(12_345));
        assert!(decoded.verify());
        // A task with no due date round-trips too.
        let plain = KomTask::new("id-10", "Plain", true, &alice).unwrap();
        assert_eq!(plain.priority, Priority::Medium);
        assert_eq!(plain.due, None);
        let decoded = KomTask::decode(&plain.encode()).unwrap();
        assert_eq!(decoded.priority, Priority::Medium);
        assert_eq!(decoded.due, None);
        assert!(decoded.verify());
        // A bad priority byte is refused. Layout: magic 7 + id len 2 +
        // id "id-9" 4 + title len 2 + title "Title" 5 + done 1 = 21.
        let mut bytes = task.encode();
        bytes[21] = 9;
        assert!(matches!(
            KomTask::decode(&bytes),
            Err(KomError::InvalidManifest(_))
        ));
        // Tampering with the priority after signing breaks verification.
        let mut tampered = task.clone();
        tampered.priority = Priority::Low;
        assert!(!tampered.verify());
    }

    #[test]
    fn priority_and_due_sorting() {
        let alice = Identity::generate("alice").unwrap();
        let mut kom = KomStore::new();
        kom.create_task_full(
            "low-late",
            "Low late",
            Priority::Low,
            Some(500),
            &[],
            &alice,
        )
        .unwrap();
        kom.create_task_full(
            "high-nodue",
            "High no due",
            Priority::High,
            None,
            &[],
            &alice,
        )
        .unwrap();
        kom.create_task_full(
            "med-early",
            "Med early",
            Priority::Medium,
            Some(10),
            &[],
            &alice,
        )
        .unwrap();
        kom.create_task_full(
            "high-early",
            "High early",
            Priority::High,
            Some(20),
            &[],
            &alice,
        )
        .unwrap();
        kom.create_task_full(
            "high-late",
            "High late",
            Priority::High,
            Some(90),
            &[],
            &alice,
        )
        .unwrap();
        kom.create_task_full(
            "med-nodue",
            "Med no due",
            Priority::Medium,
            None,
            &[],
            &alice,
        )
        .unwrap();
        // Same priority and due date: creation order wins.
        kom.create_task_full("high-tie-a", "Tie A", Priority::High, Some(20), &[], &alice)
            .unwrap();
        kom.create_task_full("high-tie-b", "Tie B", Priority::High, Some(20), &[], &alice)
            .unwrap();
        let listed = kom.tasks_by_priority();
        let order: Vec<&str> = listed.iter().map(|(_, t)| t.id.as_str()).collect();
        assert_eq!(
            order,
            vec![
                "high-early",
                "high-tie-a",
                "high-tie-b",
                "high-late",
                "high-nodue",
                "med-early",
                "med-nodue",
                "low-late",
            ]
        );
    }

    #[test]
    fn tasks_due_before_is_strict() {
        let alice = Identity::generate("alice").unwrap();
        let mut kom = KomStore::new();
        kom.create_task_full("due-10", "Ten", Priority::Medium, Some(10), &[], &alice)
            .unwrap();
        kom.create_task_full("due-20", "Twenty", Priority::Medium, Some(20), &[], &alice)
            .unwrap();
        kom.create_task_full("due-30", "Thirty", Priority::Medium, Some(30), &[], &alice)
            .unwrap();
        kom.create_task_full("nodue", "No due", Priority::Medium, None, &[], &alice)
            .unwrap();
        let ids =
            |v: Vec<(Cid, KomTask)>| -> Vec<String> { v.into_iter().map(|(_, t)| t.id).collect() };
        assert_eq!(ids(kom.tasks_due_before(20)), vec!["due-10".to_string()]);
        assert_eq!(
            ids(kom.tasks_due_before(21)),
            vec!["due-10".to_string(), "due-20".to_string()]
        );
        assert!(kom.tasks_due_before(10).is_empty());
        assert_eq!(
            ids(kom.tasks_due_before(1_000)),
            vec![
                "due-10".to_string(),
                "due-20".to_string(),
                "due-30".to_string()
            ]
        );
    }

    #[test]
    fn toggle_preserves_priority_and_due() {
        let alice = Identity::generate("alice").unwrap();
        let mut kom = KomStore::new();
        let cid = kom
            .create_task_full("t", "T", Priority::High, Some(42), &[], &alice)
            .unwrap();
        let (cid2, done) = kom.toggle_task(&cid, &alice).unwrap();
        assert!(done);
        let task = kom.get_task(&cid2).unwrap();
        assert_eq!(task.priority, Priority::High);
        assert_eq!(task.due, Some(42));
    }

    #[test]
    fn dependencies_block_and_unblock() {
        let alice = Identity::generate("alice").unwrap();
        let mut kom = KomStore::new();
        kom.create_task("a", "A", &alice).unwrap();
        let b = kom
            .create_task_full("b", "B", Priority::Medium, None, &["a"], &alice)
            .unwrap();
        assert_eq!(
            kom.complete_task(&b, &alice),
            Err(KomError::DependencyIncomplete("a".to_string()))
        );
        // The toggle path refuses the same way.
        assert_eq!(
            kom.toggle_task(&b, &alice),
            Err(KomError::DependencyIncomplete("a".to_string()))
        );
        let a_cid = kom.task_cid("a").unwrap();
        kom.complete_task(&a_cid, &alice).unwrap();
        let (new_b, done) = kom.complete_task(&b, &alice).unwrap();
        assert!(done);
        assert!(kom.get_task(&new_b).unwrap().done);
        // Completing an already-done task succeeds without a new record.
        assert_eq!(kom.complete_task(&new_b, &alice), Ok((new_b, true)));
        assert_eq!(kom.dependencies_of("b"), vec!["a".to_string()]);
    }

    #[test]
    fn dependency_cycles_refused() {
        let alice = Identity::generate("alice").unwrap();
        let mut kom = KomStore::new();
        // Depending on yourself at creation.
        assert!(matches!(
            kom.create_task_full(
                "self-dep",
                "Self",
                Priority::Medium,
                None,
                &["self-dep"],
                &alice
            ),
            Err(KomError::DependencyCycle(_))
        ));
        // Depending on a task that does not exist.
        assert!(matches!(
            kom.create_task_full("x", "X", Priority::Medium, None, &["nope"], &alice),
            Err(KomError::UnknownDependency(_))
        ));
        // A two-node cycle through add_dependency.
        kom.create_task("a", "A", &alice).unwrap();
        kom.create_task("b", "B", &alice).unwrap();
        kom.add_dependency("b", "a", &alice).unwrap();
        assert!(matches!(
            kom.add_dependency("a", "b", &alice),
            Err(KomError::DependencyCycle(_))
        ));
        // Adding the same dependency again is a no-op success.
        kom.add_dependency("b", "a", &alice).unwrap();
        assert_eq!(kom.dependencies_of("b"), vec!["a".to_string()]);
        // Depending on yourself through add_dependency.
        assert!(matches!(
            kom.add_dependency("a", "a", &alice),
            Err(KomError::DependencyCycle(_))
        ));
        // Unknown task or dependency.
        assert!(matches!(
            kom.add_dependency("ghost", "a", &alice),
            Err(KomError::UnknownTask(_))
        ));
        assert!(matches!(
            kom.add_dependency("a", "ghost", &alice),
            Err(KomError::UnknownDependency(_))
        ));
    }

    #[test]
    fn delete_depended_on_task_refused() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut kom = KomStore::new();
        kom.create_task("a", "A", &alice).unwrap();
        kom.create_task_full("b", "B", Priority::Medium, None, &["a"], &alice)
            .unwrap();
        assert!(matches!(
            kom.delete_task("a", &alice),
            Err(KomError::DependedOn(_))
        ));
        assert!(kom.task_cid("a").is_some());
        // Only the owner can delete.
        assert_eq!(kom.delete_task("b", &bob), Err(KomError::NotOwner));
        kom.delete_task("b", &alice).unwrap();
        assert!(kom.task_cid("b").is_none());
        kom.delete_task("a", &alice).unwrap();
        assert!(kom.task_cid("a").is_none());
        // Deleting an unknown task.
        assert!(matches!(
            kom.delete_task("ghost", &alice),
            Err(KomError::UnknownTask(_))
        ));
    }

    #[test]
    fn recurring_creates_exact_count() {
        let alice = Identity::generate("alice").unwrap();
        let mut kom = KomStore::new();
        let base = kom
            .create_recurring_task(
                "water",
                "Water plants",
                Priority::Medium,
                100,
                10,
                3,
                &alice,
            )
            .unwrap();
        assert_eq!(kom.task_cid("water"), Some(base));
        assert_eq!(kom.get_task(&base).unwrap().due, Some(100));
        // Nothing due yet.
        assert!(kom.tick(109, &alice).unwrap().is_empty());
        // First tick: occurrence #2 due at 110.
        let created = kom.tick(110, &alice).unwrap();
        assert_eq!(created.len(), 1);
        assert_eq!(kom.task_cid("water-002"), Some(created[0]));
        assert_eq!(kom.get_task(&created[0]).unwrap().due, Some(110));
        // Late tick: occurrence #3 due at 120, then the count is reached.
        let created = kom.tick(135, &alice).unwrap();
        assert_eq!(created.len(), 1);
        assert_eq!(kom.task_cid("water-003"), Some(created[0]));
        assert_eq!(kom.get_task(&created[0]).unwrap().due, Some(120));
        assert!(kom.tick(1_000_000, &alice).unwrap().is_empty());
    }

    #[test]
    fn recurring_late_tick_batches_in_order() {
        let alice = Identity::generate("alice").unwrap();
        let mut kom = KomStore::new();
        kom.create_recurring_task("gym", "Gym", Priority::High, 100, 10, 4, &alice)
            .unwrap();
        let created = kom.tick(135, &alice).unwrap();
        assert_eq!(created.len(), 3);
        assert_eq!(kom.task_cid("gym-002"), Some(created[0]));
        assert_eq!(kom.task_cid("gym-003"), Some(created[1]));
        assert_eq!(kom.task_cid("gym-004"), Some(created[2]));
        assert_eq!(kom.get_task(&created[0]).unwrap().due, Some(110));
        assert_eq!(kom.get_task(&created[1]).unwrap().due, Some(120));
        assert_eq!(kom.get_task(&created[2]).unwrap().due, Some(130));
        assert!(kom.tick(10_000, &alice).unwrap().is_empty());
    }

    #[test]
    fn recurring_validation() {
        let alice = Identity::generate("alice").unwrap();
        let mut kom = KomStore::new();
        assert!(matches!(
            kom.create_recurring_task("x", "X", Priority::Medium, 100, 0, 3, &alice),
            Err(KomError::InvalidRecurrence(_))
        ));
        assert!(matches!(
            kom.create_recurring_task("x", "X", Priority::Medium, 100, 10, 0, &alice),
            Err(KomError::InvalidRecurrence(_))
        ));
    }

    #[test]
    fn notes_search_case_insensitive_newest_first() {
        let alice = Identity::generate("alice").unwrap();
        let mut kom = KomStore::new();
        kom.put_note("Shopping", b"buy milk and eggs", &alice)
            .unwrap();
        kom.put_note("Milkshake recipe", b"blend ice cream", &alice)
            .unwrap();
        kom.put_note("Work", b"finish the MILK report", &alice)
            .unwrap();
        kom.put_note("Random", b"nothing relevant", &alice).unwrap();
        let found = kom.search_notes("milk");
        let titles: Vec<&str> = found.iter().map(|(_, t)| t.as_str()).collect();
        // Newest first: Work, then Milkshake recipe, then Shopping.
        assert_eq!(titles, vec!["Work", "Milkshake recipe", "Shopping"]);
        assert_eq!(kom.search_notes("MILK").len(), 3);
        assert!(kom.search_notes("zebra").is_empty());
        // An empty query matches every note, newest first.
        let found = kom.search_notes("");
        assert_eq!(found.len(), 4);
        assert_eq!(found[0].1, "Random");
        assert_eq!(found[3].1, "Shopping");
    }

    #[test]
    fn due_soon_includes_overdue_and_sorts_by_due() {
        let alice = Identity::generate("alice").unwrap();
        let mut kom = KomStore::new();
        kom.create_task_full("overdue", "Overdue", Priority::Low, Some(50), &[], &alice)
            .unwrap();
        kom.create_task_full("soon-1", "Soon 1", Priority::High, Some(105), &[], &alice)
            .unwrap();
        kom.create_task_full("soon-2", "Soon 2", Priority::Medium, Some(150), &[], &alice)
            .unwrap();
        kom.create_task_full("later", "Later", Priority::High, Some(1_000), &[], &alice)
            .unwrap();
        kom.create_task_full("nodue", "No due", Priority::High, None, &[], &alice)
            .unwrap();
        let ids =
            |v: Vec<(Cid, KomTask)>| -> Vec<String> { v.into_iter().map(|(_, t)| t.id).collect() };
        // Window [.., 160]: overdue + soon-1 + soon-2, earliest due first.
        assert_eq!(
            ids(kom.due_soon(100, 60)),
            vec![
                "overdue".to_string(),
                "soon-1".to_string(),
                "soon-2".to_string()
            ]
        );
        // Zero window: only what is already due.
        assert_eq!(ids(kom.due_soon(100, 0)), vec!["overdue".to_string()]);
    }
}
