//! X•Kom's — the owner's personal tool on KERNEL.
//!
//! Three personal pieces, all signed:
//!
//! * A **note** is a signed manifest: a human title, a body stored
//!   chunked in the content store, the owner, and the owner's
//!   signature. Notes read exactly like files: signature plus
//!   per-chunk hash verification on read.
//! * A **task** is a signed record: a stable human id, a title, a done
//!   flag, the owner, and the owner's signature. Toggling a task flips
//!   the flag and re-signs, producing a new record (content addressing:
//!   the task's address changes when its state changes).
//! * A **quick command** is a signed record: a name, a command string,
//!   the owner, and the owner's signature. "Running" a command only
//!   appends an execution entry (name, timestamp) to a log. **It does
//!   not execute any shell command, and it never will.** X•Kom's
//!   records what the owner asked for; running the thing itself is a
//!   separate, explicit act by the owner outside this crate.
//!
//! Manifest/record formats (explicit binary, no JSON):
//! `KNOTE1` + title length (u16 LE) + title bytes + body chunk root
//! CID (32) + owner public key (32) + signature (64).
//! `XKTASK1` + id length (u16 LE) + id bytes + title length (u16 LE)
//! + title bytes + done flag (u8) + owner public key (32) +
//! signature (64).
//! `XKOMCMD` + name length (u16 LE) + name bytes + command length
//! (u32 LE) + command bytes + owner public key (32) + signature (64).

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
        return Err(KomError::InvalidTitle("title must not be empty".to_string()));
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
        return Err(KomError::InvalidTitle("title must be printable ASCII".to_string()));
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

/// The message an owner signs for a task record.
fn task_sign_message(id: &str, title: &str, done: bool, owner: &PeerId) -> Vec<u8> {
    let mut msg = Vec::with_capacity(TASK_SIGN_DOMAIN.len() + id.len() + title.len() + 1 + 32);
    msg.extend_from_slice(TASK_SIGN_DOMAIN);
    msg.extend_from_slice(id.as_bytes());
    msg.extend_from_slice(title.as_bytes());
    msg.push(done as u8);
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg
}

/// The message an owner signs for a command record.
fn command_sign_message(name: &str, command: &str, owner: &PeerId) -> Vec<u8> {
    let mut msg =
        Vec::with_capacity(COMMAND_SIGN_DOMAIN.len() + name.len() + command.len() + 32);
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
            return Err(KomError::InvalidManifest("missing KNOTE1 magic".to_string()));
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

/// A signed task record: stable id, title, done flag, owner, signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KomTask {
    /// Stable human id, e.g. `"buy-milk"`.
    pub id: String,
    /// Human-readable task title.
    pub title: String,
    /// Whether the task is done.
    pub done: bool,
    /// The owner who signed this record.
    pub owner: PeerId,
    /// The owner's signature over id, title, done flag, and owner key.
    pub signature: Ed25519Signature,
}

impl KomTask {
    /// Build and sign a task record as `identity`.
    pub fn new(id: &str, title: &str, done: bool, identity: &Identity) -> Result<Self, KomError> {
        validate_id(id, "task id")?;
        validate_title(title)?;
        let owner = identity.peer_id().clone();
        let msg = task_sign_message(id, title, done, &owner);
        let signature = identity.sign(&msg);
        Ok(Self {
            id: id.to_string(),
            title: title.to_string(),
            done,
            owner,
            signature,
        })
    }

    /// True if the signature verifies against the owner for this exact
    /// id, title, and done flag.
    pub fn verify(&self) -> bool {
        let msg = task_sign_message(&self.id, &self.title, self.done, &self.owner);
        self.owner.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary task format.
    pub fn encode(&self) -> Vec<u8> {
        let id_bytes = self.id.as_bytes();
        let title_bytes = self.title.as_bytes();
        let mut out = Vec::with_capacity(
            7 + 2 + id_bytes.len() + 2 + title_bytes.len() + 1 + 32 + 64,
        );
        out.extend_from_slice(TASK_MAGIC);
        out.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(id_bytes);
        out.extend_from_slice(&(title_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(title_bytes);
        out.push(self.done as u8);
        out.extend_from_slice(self.owner.public_key().as_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a task produced by [`KomTask::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, KomError> {
        if bytes.len() < 7 + 2 + 2 + 1 + 32 + 64 || &bytes[..7] != TASK_MAGIC {
            return Err(KomError::InvalidManifest("missing XKTASK1 magic".to_string()));
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
        let expected = title_len_off + 2 + title_len + 1 + 32 + 64;
        if bytes.len() != expected {
            return Err(KomError::InvalidManifest(format!(
                "task is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let id = String::from_utf8(bytes[9..9 + id_len].to_vec())
            .map_err(|_| KomError::InvalidManifest("task id is not valid UTF-8".to_string()))?;
        validate_id(&id, "task id")?;
        let mut off = title_len_off + 2;
        let title = String::from_utf8(bytes[off..off + title_len].to_vec()).map_err(|_| {
            KomError::InvalidManifest("task title is not valid UTF-8".to_string())
        })?;
        validate_title(&title)?;
        off += title_len;
        let done_byte = bytes[off];
        if done_byte > 1 {
            return Err(KomError::InvalidManifest(format!(
                "done flag must be 0 or 1, got {done_byte}"
            )));
        }
        let done = done_byte == 1;
        off += 1;
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
        let mut out = Vec::with_capacity(
            7 + 2 + name_bytes.len() + 4 + command_bytes.len() + 32 + 64,
        );
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
            return Err(KomError::InvalidManifest("missing XKOMCMD magic".to_string()));
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
        let name = String::from_utf8(bytes[9..9 + name_len].to_vec())
            .map_err(|_| KomError::InvalidManifest("command name is not valid UTF-8".to_string()))?;
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

/// The owner's personal store: a content store holding signed notes,
/// tasks, and commands, plus indexes and the execution log.
pub struct KomStore {
    store: ContentStore,
    /// Note manifests this node holds, by manifest CID.
    notes: HashMap<Cid, NoteManifest>,
    /// Task records this node holds, by record CID.
    tasks: HashMap<Cid, KomTask>,
    /// Task record CIDs, by task id.
    task_ids: HashMap<String, Cid>,
    /// Command records this node holds, by record CID.
    commands: HashMap<Cid, KomCommand>,
    /// Command record CIDs, by command name.
    command_names: HashMap<String, Cid>,
    /// Execution log, in the order commands were "run".
    executions: Vec<CommandExecution>,
}

impl KomStore {
    /// An empty personal store.
    pub fn new() -> Self {
        Self {
            store: ContentStore::new(),
            notes: HashMap::new(),
            tasks: HashMap::new(),
            task_ids: HashMap::new(),
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
        Ok(manifest_cid)
    }

    /// Fetch a note by manifest CID: verify the manifest signature,
    /// then reassemble the chunked body (every chunk hash-verified).
    pub fn get_note(
        &self,
        manifest_cid: &Cid,
    ) -> Result<(Vec<u8>, NoteMetadata), KomError> {
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
    pub fn create_task(
        &mut self,
        id: &str,
        title: &str,
        identity: &Identity,
    ) -> Result<Cid, KomError> {
        validate_id(id, "task id")?;
        validate_title(title)?;
        if self.task_ids.contains_key(id) {
            return Err(KomError::AlreadyExists(id.to_string()));
        }
        let task = KomTask::new(id, title, false, identity)?;
        let cid = self.store.put(&task.encode());
        self.tasks.insert(cid, task);
        self.task_ids.insert(id.to_string(), cid);
        Ok(cid)
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

    /// Toggle a task's done flag: only the task's owner can toggle.
    /// The flipped record is signed and stored as a new block, and the
    /// index moves to the new record CID (the old record stays in the
    /// content store as history). Returns the new record CID and the
    /// new done state.
    pub fn toggle_task(
        &mut self,
        task_cid: &Cid,
        identity: &Identity,
    ) -> Result<(Cid, bool), KomError> {
        let task = self.get_task(task_cid)?;
        if task.owner != *identity.peer_id() {
            return Err(KomError::NotOwner);
        }
        let flipped = KomTask::new(&task.id, &task.title, !task.done, identity)?;
        let new_cid = self.store.put(&flipped.encode());
        self.tasks.remove(task_cid);
        self.tasks.insert(new_cid, flipped);
        self.task_ids.insert(task.id, new_cid);
        Ok((new_cid, !task.done))
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

        let large: Vec<u8> = (0..600 * 1024).map(|i| ((i * 37 + 11) % 251) as u8).collect();
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
        assert_eq!(kom.get_task(&cid3).unwrap().done, false);
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
        assert_eq!(kom.get_command(&cid).unwrap().command, "tar -czf backup.tar.gz ~/notes");
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
        let cid = kom.register_command("backup", "do a backup", &alice).unwrap();
        let mut bytes = kom.store_mut().get(&cid).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let tampered_cid = kom.store_mut().put(&bytes);
        assert_eq!(
            kom.get_command(&tampered_cid),
            Err(KomError::BadSignature)
        );
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
}
