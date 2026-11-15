//! HEY•AI — voice on KERNEL.
//!
//! Two pieces, both signed:
//!
//! * A **voice note** is a signed manifest: a human name, a duration in
//!   seconds, the chunked audio root CID, the owner, and the owner's
//!   signature. Voice notes are stored exactly like photos and tracks —
//!   chunked audio in the content store, manifest as a block, signature
//!   plus per-chunk hash verification on read.
//! * A **voice command** is a signed record: command text, the owner,
//!   and the owner's signature. The text is parsed by a small fixed
//!   router ([`parse_command`]) into a [`VoiceAction`]. The router is
//!   deliberately tiny and total: unknown text maps to
//!   [`VoiceAction::Unknown`], never a panic, never an error.
//!
//! Nothing here records audio, transcribes speech, or plays sound.
//! This crate stores signed voice notes and routes command text to
//! actions. Recording, transcription, and playback are separate
//! concerns.
//!
//! The same crate also holds HEY•AI's **render engine**: a small
//! deterministic template language (`{{name}}`, `{{#each}}`,
//! `{{#if}}`) over plain data values — no JSON, no code execution.
//! Rendering is pure data transformation: the same template plus
//! the same data always produces the exact same bytes, addressed
//! by their SHA-256. A [`RenderQueue`] runs jobs in order and
//! records each result; a failed job never blocks the next one.
//!
//! Formats (explicit binary, no JSON):
//! `KVOICE1` + name length (u16 LE) + name bytes + duration (u64 LE) +
//! chunk root CID (32) + owner public key (32) + signature (64).
//! `KCMDVC1` + text length (u32 LE) + text bytes + owner public key
//! (32) + signature (64).

use std::collections::{HashMap, VecDeque};
use std::fmt;

use kernel_crypto::Ed25519Signature;
use kernel_identity::{Identity, IdentityError, PeerId};
use kernel_storage::{Cid, ContentStore, StorageError};

/// Magic bytes at the start of every voice note manifest.
const NOTE_MAGIC: &[u8; 7] = b"KVOICE1";
/// Domain separator for voice note manifest signatures.
const NOTE_SIGN_DOMAIN: &[u8] = b"KVOICE1";
/// Magic bytes at the start of every voice command record.
const COMMAND_MAGIC: &[u8; 7] = b"KCMDVC1";
/// Domain separator for voice command signatures.
const COMMAND_SIGN_DOMAIN: &[u8] = b"KCMDVC1";

/// Maximum length of a voice note name.
pub const MAX_NAME_LEN: usize = 128;
/// Maximum voice note duration: 24 hours, in seconds.
pub const MAX_DURATION_SECS: u64 = 86_400;
/// Maximum voice command text length in characters.
pub const MAX_COMMAND_LEN: usize = 1_000;

/// Errors returned by voice note and command operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiceError {
    /// Wraps a [`StorageError`].
    Storage(StorageError),
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// A voice note name broke the naming rules.
    InvalidName(String),
    /// A duration was out of range.
    InvalidDuration(String),
    /// A command text broke the rules.
    InvalidCommand(String),
    /// A manifest or command record was malformed.
    InvalidManifest(String),
    /// A manifest's or command's signature did not verify.
    BadSignature,
    /// No voice note or command with this CID is known here.
    NotFound(Cid),
}

impl fmt::Display for VoiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VoiceError::Storage(e) => write!(f, "storage error: {e}"),
            VoiceError::Identity(e) => write!(f, "identity error: {e}"),
            VoiceError::InvalidName(msg) => write!(f, "invalid voice note name: {msg}"),
            VoiceError::InvalidDuration(msg) => write!(f, "invalid duration: {msg}"),
            VoiceError::InvalidCommand(msg) => write!(f, "invalid command: {msg}"),
            VoiceError::InvalidManifest(msg) => write!(f, "invalid manifest: {msg}"),
            VoiceError::BadSignature => write!(f, "signature is invalid"),
            VoiceError::NotFound(cid) => write!(f, "voice note not found: {cid}"),
        }
    }
}

impl std::error::Error for VoiceError {}

impl From<StorageError> for VoiceError {
    fn from(e: StorageError) -> Self {
        VoiceError::Storage(e)
    }
}

impl From<IdentityError> for VoiceError {
    fn from(e: IdentityError) -> Self {
        VoiceError::Identity(e)
    }
}

/// Rules for voice note names: 1-128 printable ASCII characters, no
/// path separators. Names are human labels, not hashes.
fn validate_name(name: &str) -> Result<(), VoiceError> {
    if name.is_empty() {
        return Err(VoiceError::InvalidName(
            "voice note name must not be empty".to_string(),
        ));
    }
    if name.len() > MAX_NAME_LEN {
        return Err(VoiceError::InvalidName(format!(
            "voice note name must be at most {MAX_NAME_LEN} characters"
        )));
    }
    if name.contains('/') || name.contains('\\') {
        return Err(VoiceError::InvalidName(
            "voice note name must not contain a path separator".to_string(),
        ));
    }
    if !name.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
        return Err(VoiceError::InvalidName(
            "voice note name must be printable ASCII".to_string(),
        ));
    }
    Ok(())
}

/// Rules for durations: at least 1 second, at most [`MAX_DURATION_SECS`].
fn validate_duration(duration_secs: u64) -> Result<(), VoiceError> {
    if duration_secs == 0 {
        return Err(VoiceError::InvalidDuration(
            "duration must be at least 1 second".to_string(),
        ));
    }
    if duration_secs > MAX_DURATION_SECS {
        return Err(VoiceError::InvalidDuration(format!(
            "duration must be at most {MAX_DURATION_SECS} seconds"
        )));
    }
    Ok(())
}

/// Rules for command text: non-empty, at most [`MAX_COMMAND_LEN`] chars.
fn validate_command_text(text: &str) -> Result<(), VoiceError> {
    if text.trim().is_empty() {
        return Err(VoiceError::InvalidCommand(
            "command text must not be empty".to_string(),
        ));
    }
    if text.chars().count() > MAX_COMMAND_LEN {
        return Err(VoiceError::InvalidCommand(format!(
            "command text must be at most {MAX_COMMAND_LEN} characters"
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

/// The message an owner signs for a voice note manifest.
fn note_sign_message(name: &str, duration_secs: u64, root: &Cid, owner: &PeerId) -> Vec<u8> {
    let mut msg = Vec::with_capacity(NOTE_SIGN_DOMAIN.len() + name.len() + 8 + 32 + 32);
    msg.extend_from_slice(NOTE_SIGN_DOMAIN);
    msg.extend_from_slice(name.as_bytes());
    msg.extend_from_slice(&duration_secs.to_le_bytes());
    msg.extend_from_slice(root.as_bytes());
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg
}

/// The message an owner signs for a voice command record.
fn command_sign_message(text: &str, owner: &PeerId) -> Vec<u8> {
    let mut msg = Vec::with_capacity(COMMAND_SIGN_DOMAIN.len() + text.len() + 32);
    msg.extend_from_slice(COMMAND_SIGN_DOMAIN);
    msg.extend_from_slice(text.as_bytes());
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg
}

/// An action a voice command can request. The router is fixed and
/// small on purpose: these are the only actions a command can name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VoiceAction {
    /// Play the named track (the track name as spoken, trimmed).
    Play {
        /// The track name from the command.
        track_name: String,
    },
    /// Pause playback.
    Pause,
    /// Skip to the next track.
    Next,
    /// Go back to the previous track.
    Previous,
    /// List the available tracks.
    ListTracks,
    /// The text did not match any known command.
    Unknown,
}

/// Parse voice command text into an action.
///
/// The router is case-insensitive on the command word and preserves
/// the track name's original casing. It never panics and never errors:
/// anything it does not recognize is [`VoiceAction::Unknown`].
///
/// Recognized forms (leading/trailing whitespace ignored):
/// * `play <name>` — [`VoiceAction::Play`] with the name trimmed.
/// * `pause`, `next`, `previous` or `prev`, `list`.
pub fn parse_command(text: &str) -> VoiceAction {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return VoiceAction::Unknown;
    }
    let lower = trimmed.to_ascii_lowercase();
    // "play " is 5 bytes; ASCII lowercasing never changes byte length,
    // so the same byte index works on the original text.
    if let Some(_rest) = lower.strip_prefix("play ") {
        let name = trimmed[5..].trim();
        if name.is_empty() {
            return VoiceAction::Unknown;
        }
        return VoiceAction::Play {
            track_name: name.to_string(),
        };
    }
    match lower.as_str() {
        "pause" => VoiceAction::Pause,
        "next" => VoiceAction::Next,
        "previous" | "prev" => VoiceAction::Previous,
        "list" => VoiceAction::ListTracks,
        _ => VoiceAction::Unknown,
    }
}

/// A signed voice note manifest: name, duration, chunked-content
/// root, owner, signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VoiceNoteManifest {
    /// Human-readable voice note name.
    pub name: String,
    /// Voice note duration in seconds.
    pub duration_secs: u64,
    /// The owner who signed this manifest.
    pub owner: PeerId,
    /// Root CID of the chunked audio content (a `KCHUNK1` manifest block).
    pub root: Cid,
    /// The owner's signature over name, duration, root, owner key.
    pub signature: Ed25519Signature,
}

impl VoiceNoteManifest {
    /// Build and sign a manifest as `identity`.
    pub fn new(
        name: &str,
        duration_secs: u64,
        root: Cid,
        identity: &Identity,
    ) -> Result<Self, VoiceError> {
        validate_name(name)?;
        validate_duration(duration_secs)?;
        let owner = identity.peer_id().clone();
        let msg = note_sign_message(name, duration_secs, &root, &owner);
        let signature = identity.sign(&msg);
        Ok(Self {
            name: name.to_string(),
            duration_secs,
            owner,
            root,
            signature,
        })
    }

    /// True if the signature verifies against the owner for this exact
    /// name, duration, and root. Any tampering fails this check.
    pub fn verify(&self) -> bool {
        let msg = note_sign_message(&self.name, self.duration_secs, &self.root, &self.owner);
        self.owner.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary manifest format.
    pub fn encode(&self) -> Vec<u8> {
        let name_bytes = self.name.as_bytes();
        let mut out = Vec::with_capacity(7 + 2 + name_bytes.len() + 8 + 32 + 32 + 64);
        out.extend_from_slice(NOTE_MAGIC);
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(&self.duration_secs.to_le_bytes());
        out.extend_from_slice(self.root.as_bytes());
        out.extend_from_slice(self.owner.public_key().as_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a manifest produced by [`VoiceNoteManifest::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, VoiceError> {
        if bytes.len() < 7 + 2 + 8 + 32 + 32 + 64 || &bytes[..7] != NOTE_MAGIC {
            return Err(VoiceError::InvalidManifest(
                "missing KVOICE1 magic".to_string(),
            ));
        }
        let name_len = u16::from_le_bytes(bytes[7..9].try_into().expect("2-byte slice")) as usize;
        let expected = 9 + name_len + 8 + 32 + 32 + 64;
        if bytes.len() != expected {
            return Err(VoiceError::InvalidManifest(format!(
                "manifest is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let mut off = 9;
        let name = String::from_utf8(bytes[off..off + name_len].to_vec()).map_err(|_| {
            VoiceError::InvalidManifest("manifest name is not valid UTF-8".to_string())
        })?;
        validate_name(&name)?;
        off += name_len;
        let duration_secs =
            u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        validate_duration(duration_secs)?;
        let mut root_bytes = [0u8; 32];
        root_bytes.copy_from_slice(&bytes[off..off + 32]);
        let root = Cid::from_bytes(root_bytes);
        off += 32;
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let owner_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| VoiceError::InvalidManifest(e.to_string()))?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let owner = PeerId::new(owner_key, &fingerprint_name(&owner_key))?;
        Ok(Self {
            name,
            duration_secs,
            owner,
            root,
            signature,
        })
    }
}

/// A signed voice command record: command text, owner, signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VoiceCommand {
    /// The command text as spoken, e.g. `"play Midnight"`.
    pub text: String,
    /// The peer who signed this command.
    pub owner: PeerId,
    /// The owner's signature over text and owner key.
    pub signature: Ed25519Signature,
}

impl VoiceCommand {
    /// Build and sign a command as `identity`.
    pub fn new(text: &str, identity: &Identity) -> Result<Self, VoiceError> {
        validate_command_text(text)?;
        let owner = identity.peer_id().clone();
        let msg = command_sign_message(text, &owner);
        let signature = identity.sign(&msg);
        Ok(Self {
            text: text.to_string(),
            owner,
            signature,
        })
    }

    /// True if the signature verifies against the owner for this exact text.
    pub fn verify(&self) -> bool {
        let msg = command_sign_message(&self.text, &self.owner);
        self.owner.verify(&msg, &self.signature).is_ok()
    }

    /// The action this command requests, via the fixed router.
    pub fn action(&self) -> VoiceAction {
        parse_command(&self.text)
    }

    /// Encode to the explicit binary command format.
    pub fn encode(&self) -> Vec<u8> {
        let text_bytes = self.text.as_bytes();
        let mut out = Vec::with_capacity(7 + 4 + text_bytes.len() + 32 + 64);
        out.extend_from_slice(COMMAND_MAGIC);
        out.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(text_bytes);
        out.extend_from_slice(self.owner.public_key().as_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a command produced by [`VoiceCommand::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, VoiceError> {
        if bytes.len() < 7 + 4 + 32 + 64 || &bytes[..7] != COMMAND_MAGIC {
            return Err(VoiceError::InvalidManifest(
                "missing KCMDVC1 magic".to_string(),
            ));
        }
        let text_len = u32::from_le_bytes(bytes[7..11].try_into().expect("4-byte slice")) as usize;
        if text_len > MAX_COMMAND_LEN {
            return Err(VoiceError::InvalidManifest(format!(
                "command claims {text_len} chars, max is {MAX_COMMAND_LEN}"
            )));
        }
        let expected = 11 + text_len + 32 + 64;
        if bytes.len() != expected {
            return Err(VoiceError::InvalidManifest(format!(
                "command is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let text = String::from_utf8(bytes[11..11 + text_len].to_vec()).map_err(|_| {
            VoiceError::InvalidManifest("command text is not valid UTF-8".to_string())
        })?;
        validate_command_text(&text)?;
        let mut off = 11 + text_len;
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let owner_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| VoiceError::InvalidManifest(e.to_string()))?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let owner = PeerId::new(owner_key, &fingerprint_name(&owner_key))?;
        Ok(Self {
            text,
            owner,
            signature,
        })
    }
}

/// The metadata of a voice note, as a human sees it — no CIDs, no hashes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VoiceNoteMetadata {
    /// Human-readable voice note name.
    pub name: String,
    /// Voice note duration in seconds.
    pub duration_secs: u64,
    /// Who owns (recorded) this voice note.
    pub owner: PeerId,
}

/// A voice store: a content store plus indexes of voice note manifests
/// and voice command records. Voice notes are addressed by manifest
/// CID; humans see names.
pub struct VoiceStore {
    store: ContentStore,
    /// Voice note manifests this node holds, by manifest CID.
    notes: HashMap<Cid, VoiceNoteManifest>,
    /// Voice command records this node holds, by record CID.
    commands: HashMap<Cid, VoiceCommand>,
}

impl Default for VoiceStore {
    /// An empty voice store, the same as [`VoiceStore::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl VoiceStore {
    /// An empty voice store.
    pub fn new() -> Self {
        Self {
            store: ContentStore::new(),
            notes: HashMap::new(),
            commands: HashMap::new(),
        }
    }

    /// Number of voice notes held.
    pub fn len(&self) -> usize {
        self.notes.len()
    }

    /// True if no voice notes are held.
    pub fn is_empty(&self) -> bool {
        self.notes.is_empty()
    }

    /// The manifest for a voice note, if this node holds it.
    pub fn get_manifest(&self, manifest_cid: &Cid) -> Option<&VoiceNoteManifest> {
        self.notes.get(manifest_cid)
    }

    /// Store a voice note: chunk the audio bytes, build and sign the
    /// manifest, and store the manifest as a block. Returns the
    /// manifest CID — the voice note's address.
    pub fn put_voice_note(
        &mut self,
        name: &str,
        duration_secs: u64,
        data: &[u8],
        identity: &Identity,
    ) -> Result<Cid, VoiceError> {
        validate_name(name)?;
        validate_duration(duration_secs)?;
        let root = self.store.put_chunked(data);
        let manifest = VoiceNoteManifest::new(name, duration_secs, root, identity)?;
        let manifest_cid = self.store.put(&manifest.encode());
        self.notes.insert(manifest_cid, manifest);
        Ok(manifest_cid)
    }

    /// Fetch a voice note by manifest CID: verify the manifest
    /// signature, then reassemble the chunked audio (every chunk
    /// hash-verified). Returns the audio bytes and metadata.
    pub fn get_voice_note(
        &self,
        manifest_cid: &Cid,
    ) -> Result<(Vec<u8>, VoiceNoteMetadata), VoiceError> {
        let manifest = match self.notes.get(manifest_cid) {
            Some(m) => m.clone(),
            None => {
                let bytes = self.store.get(manifest_cid).map_err(|e| match e {
                    StorageError::NotFound(cid) => VoiceError::NotFound(cid),
                    other => VoiceError::Storage(other),
                })?;
                VoiceNoteManifest::decode(&bytes)?
            }
        };
        if !manifest.verify() {
            return Err(VoiceError::BadSignature);
        }
        let data = self
            .store
            .get_chunked(&manifest.root)
            .map_err(VoiceError::Storage)?;
        let metadata = VoiceNoteMetadata {
            name: manifest.name,
            owner: manifest.owner,
            duration_secs: manifest.duration_secs,
        };
        Ok((data, metadata))
    }

    /// All voice notes held, as (manifest CID, name), sorted by name
    /// for a stable, human-readable listing.
    pub fn list_voice_notes(&self) -> Vec<(Cid, String)> {
        let mut out: Vec<(Cid, String)> = self
            .notes
            .iter()
            .map(|(cid, m)| (*cid, m.name.clone()))
            .collect();
        out.sort_by(|a, b| a.1.cmp(&b.1));
        out
    }

    /// Submit a voice command: build and sign the record and store it
    /// as a block. Returns the command record CID. Parsing the command
    /// into an action is [`parse_command`] (or [`VoiceCommand::action`]).
    pub fn submit_command(&mut self, text: &str, identity: &Identity) -> Result<Cid, VoiceError> {
        validate_command_text(text)?;
        let command = VoiceCommand::new(text, identity)?;
        let cid = self.store.put(&command.encode());
        self.commands.insert(cid, command);
        Ok(cid)
    }

    /// Fetch a voice command by record CID, verifying its signature.
    pub fn get_command(&self, command_cid: &Cid) -> Result<VoiceCommand, VoiceError> {
        let command = match self.commands.get(command_cid) {
            Some(c) => c.clone(),
            None => {
                let bytes = self.store.get(command_cid).map_err(|e| match e {
                    StorageError::NotFound(cid) => VoiceError::NotFound(cid),
                    other => VoiceError::Storage(other),
                })?;
                VoiceCommand::decode(&bytes)?
            }
        };
        if !command.verify() {
            return Err(VoiceError::BadSignature);
        }
        Ok(command)
    }

    /// Mutable access to the underlying content store. Test-only:
    /// lets tests simulate corruption (see `put_unchecked`).
    #[cfg(test)]
    pub fn store_mut(&mut self) -> &mut ContentStore {
        &mut self.store
    }
}

/// Verify a peer's Ed25519 signature over a message, given the
/// signature as raw bytes. The signature is exactly 64 bytes; any
/// other length cannot be a signature and fails. This is the
/// byte-level entry point for crates that hold signatures as raw
/// bytes (for example, signed listings) without depending on the
/// crypto crate themselves.
pub fn verify_signature(peer: &PeerId, message: &[u8], signature_bytes: &[u8; 64]) -> bool {
    let signature = Ed25519Signature::from_bytes(signature_bytes);
    peer.verify(message, &signature).is_ok()
}

// ---------------------------------------------------------------------------
// Render engine — HEY•AI as a deterministic template renderer
// ---------------------------------------------------------------------------

/// Maximum template size, in bytes. A bigger template is refused
/// rather than parsed.
pub const MAX_TEMPLATE_LEN: usize = 65_536;
/// Maximum rendered output size, in bytes. Rendering stops with an
/// error past this point.
pub const MAX_OUTPUT_LEN: usize = 1_048_576;
/// Maximum nesting depth of `{{#each}}` / `{{#if}}` blocks.
pub const MAX_NESTING_DEPTH: usize = 32;

/// A data value for rendering. Templates never see JSON and never
/// parse it: the caller builds values directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    /// A string.
    Text(String),
    /// An integer.
    Number(i64),
    /// A boolean.
    Bool(bool),
    /// An ordered list.
    List(Vec<Value>),
    /// Named fields, in the order given.
    Map(Vec<(String, Value)>),
}

impl Value {
    /// A text value.
    pub fn text(s: &str) -> Self {
        Value::Text(s.to_string())
    }

    /// A number value.
    pub fn number(n: i64) -> Self {
        Value::Number(n)
    }

    /// A boolean value.
    pub fn bool(b: bool) -> Self {
        Value::Bool(b)
    }

    /// A list value.
    pub fn list(items: Vec<Value>) -> Self {
        Value::List(items)
    }

    /// A map value.
    pub fn map(pairs: Vec<(String, Value)>) -> Self {
        Value::Map(pairs)
    }

    /// Look up a field by name. Only maps have fields; anything
    /// else returns `None`. The first matching pair wins; a later
    /// pair with the same key is ignored on lookup.
    pub fn lookup(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Map(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

/// Errors returned by template parsing and rendering. Every failure
/// is a value, never a panic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderError {
    /// The template was larger than [`MAX_TEMPLATE_LEN`] bytes.
    TemplateTooLarge(usize),
    /// The template was not well-formed (empty tag, unknown block
    /// tag, bad variable name, mismatched close tag).
    Malformed(String),
    /// A tag or block was never closed.
    UnclosedTag(String),
    /// The template named a key the data does not have.
    UnknownKey(String),
    /// A value had the wrong shape for where it was used (a list
    /// substituted as text, a non-list in `{{#each}}`, a non-bool
    /// in `{{#if}}`, non-map data at the top level).
    TypeMismatch(String),
    /// The rendered output passed [`MAX_OUTPUT_LEN`] bytes.
    OutputTooLarge(usize),
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenderError::TemplateTooLarge(n) => {
                write!(f, "template is {n} bytes, max is {MAX_TEMPLATE_LEN}")
            }
            RenderError::Malformed(msg) => write!(f, "malformed template: {msg}"),
            RenderError::UnclosedTag(msg) => write!(f, "unclosed tag: {msg}"),
            RenderError::UnknownKey(key) => write!(f, "unknown key: {key}"),
            RenderError::TypeMismatch(msg) => write!(f, "type mismatch: {msg}"),
            RenderError::OutputTooLarge(n) => {
                write!(f, "rendered output is {n} bytes, max is {MAX_OUTPUT_LEN}")
            }
        }
    }
}

impl std::error::Error for RenderError {}

/// One parsed template node.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Node {
    /// Literal text, copied verbatim.
    Text(String),
    /// `{{name}}` — substitute a scalar value.
    Var(String),
    /// `{{#each key}}...{{/each}}` — repeat the body for each item.
    Each { key: String, body: Vec<Node> },
    /// `{{#if key}}...{{/if}}` — include the body only if the key
    /// is a true boolean.
    If { key: String, body: Vec<Node> },
}

/// A variable name is one word: ASCII letters, digits, `_`, `-`.
/// `this` is the same kind of name; it is bound inside `{{#each}}`
/// to the current item.
fn is_valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Parse nodes until the end of the template or a matching close
/// tag. `closing` is the block we are inside (`"each"` / `"if"`),
/// or `None` at the top level.
fn parse_nodes(
    template: &str,
    mut pos: usize,
    closing: Option<&str>,
    depth: usize,
) -> Result<(Vec<Node>, usize), RenderError> {
    let mut nodes = Vec::new();
    while pos < template.len() {
        let rest = &template[pos..];
        let Some(open) = rest.find("{{") else {
            nodes.push(Node::Text(rest.to_string()));
            pos = template.len();
            break;
        };
        if open > 0 {
            nodes.push(Node::Text(rest[..open].to_string()));
        }
        let tag_start = pos + open;
        let after_open = tag_start + 2;
        let Some(close_rel) = template[after_open..].find("}}") else {
            return Err(RenderError::UnclosedTag(format!(
                "tag starting at byte {tag_start} has no closing '}}}}'"
            )));
        };
        let tag = template[after_open..after_open + close_rel].trim();
        let next_pos = after_open + close_rel + 2;
        if tag.is_empty() {
            return Err(RenderError::Malformed(format!(
                "empty tag at byte {tag_start}"
            )));
        }
        if let Some(close_name) = tag.strip_prefix('/').map(str::trim) {
            match closing {
                Some(expected) if close_name == expected => {
                    return Ok((nodes, next_pos));
                }
                Some(expected) => {
                    return Err(RenderError::Malformed(format!(
                        "closing tag '{{{{/{close_name}}}}}' does not match open '{{{{#{expected}}}}}'"
                    )));
                }
                None => {
                    return Err(RenderError::Malformed(format!(
                        "closing tag '{{{{/{close_name}}}}}' with no open block"
                    )));
                }
            }
        }
        if tag == "#each" || tag.starts_with("#each ") {
            let key = tag[5..].trim();
            if !is_valid_name(key) {
                return Err(RenderError::Malformed(format!(
                    "'{{{{#each}}}}' needs a variable name, got '{{{{ {tag} }}}}'"
                )));
            }
            if depth >= MAX_NESTING_DEPTH {
                return Err(RenderError::Malformed(
                    "template blocks nest too deep".to_string(),
                ));
            }
            let (body, new_pos) = parse_nodes(template, next_pos, Some("each"), depth + 1)?;
            nodes.push(Node::Each {
                key: key.to_string(),
                body,
            });
            pos = new_pos;
            continue;
        }
        if tag == "#if" || tag.starts_with("#if ") {
            let key = tag[3..].trim();
            if !is_valid_name(key) {
                return Err(RenderError::Malformed(format!(
                    "'{{{{#if}}}}' needs a variable name, got '{{{{ {tag} }}}}'"
                )));
            }
            if depth >= MAX_NESTING_DEPTH {
                return Err(RenderError::Malformed(
                    "template blocks nest too deep".to_string(),
                ));
            }
            let (body, new_pos) = parse_nodes(template, next_pos, Some("if"), depth + 1)?;
            nodes.push(Node::If {
                key: key.to_string(),
                body,
            });
            pos = new_pos;
            continue;
        }
        if tag.starts_with('#') {
            return Err(RenderError::Malformed(format!(
                "unknown block tag '{{{{ {tag} }}}}'"
            )));
        }
        if !is_valid_name(tag) {
            return Err(RenderError::Malformed(format!(
                "invalid variable name '{{{{ {tag} }}}}'"
            )));
        }
        nodes.push(Node::Var(tag.to_string()));
        pos = next_pos;
    }
    match closing {
        Some(name) => Err(RenderError::UnclosedTag(format!(
            "'{{{{#{name}}}}}' block is never closed"
        ))),
        None => Ok((nodes, pos)),
    }
}

/// Parse a whole template.
fn parse_template(template: &str) -> Result<Vec<Node>, RenderError> {
    if template.len() > MAX_TEMPLATE_LEN {
        return Err(RenderError::TemplateTooLarge(template.len()));
    }
    let (nodes, _) = parse_nodes(template, 0, None, 0)?;
    Ok(nodes)
}

/// Look a name up through the scope stack, innermost first.
fn lookup<'a>(scopes: &'a [HashMap<String, Value>], name: &str) -> Option<&'a Value> {
    scopes.iter().rev().find_map(|scope| scope.get(name))
}

/// Render a scalar value as text. Lists and maps have no text
/// form — they must be used with `{{#each}}`.
fn format_value(value: &Value) -> Option<String> {
    match value {
        Value::Text(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::List(_) | Value::Map(_) => None,
    }
}

fn render_nodes(
    nodes: &[Node],
    scopes: &mut Vec<HashMap<String, Value>>,
    out: &mut Vec<u8>,
) -> Result<(), RenderError> {
    for node in nodes {
        match node {
            Node::Text(text) => out.extend_from_slice(text.as_bytes()),
            Node::Var(name) => {
                let value =
                    lookup(scopes, name).ok_or_else(|| RenderError::UnknownKey(name.clone()))?;
                let text = format_value(value).ok_or_else(|| {
                    RenderError::TypeMismatch(format!(
                        "key '{name}' is a list or map and cannot be substituted as text"
                    ))
                })?;
                out.extend_from_slice(text.as_bytes());
            }
            Node::Each { key, body } => {
                let items =
                    lookup(scopes, key).ok_or_else(|| RenderError::UnknownKey(key.clone()))?;
                let items = match items {
                    Value::List(items) => items.clone(),
                    _ => {
                        return Err(RenderError::TypeMismatch(format!(
                            "key '{key}' is not a list, so '{{{{#each {key}}}}}' cannot loop over it"
                        )));
                    }
                };
                for item in &items {
                    let mut scope = HashMap::new();
                    scope.insert("this".to_string(), item.clone());
                    if let Value::Map(pairs) = item {
                        for (k, v) in pairs {
                            scope.entry(k.clone()).or_insert_with(|| v.clone());
                        }
                    }
                    scopes.push(scope);
                    render_nodes(body, scopes, out)?;
                    scopes.pop();
                }
            }
            Node::If { key, body } => {
                let flag =
                    lookup(scopes, key).ok_or_else(|| RenderError::UnknownKey(key.clone()))?;
                match flag {
                    Value::Bool(true) => render_nodes(body, scopes, out)?,
                    Value::Bool(false) => {}
                    _ => {
                        return Err(RenderError::TypeMismatch(format!(
                            "key '{key}' is not a boolean, so '{{{{#if {key}}}}}' cannot test it"
                        )));
                    }
                }
            }
        }
        if out.len() > MAX_OUTPUT_LEN {
            return Err(RenderError::OutputTooLarge(out.len()));
        }
    }
    Ok(())
}

/// Render a template against top-level map data.
///
/// The template language is small and deterministic:
/// * `{{name}}` substitutes a text, number, or boolean value.
/// * `{{#each items}}...{{/each}}` repeats the body once per item
///   in a list; inside, `{{this}}` is the item, and a map item's
///   fields are visible by name.
/// * `{{#if flag}}...{{/if}}` includes the body only when `flag`
///   is a true boolean.
/// * Anything else — unclosed tags, unknown keys, malformed
///   templates — is a [`RenderError`], never a panic.
///
/// The same template plus the same data always produces the exact
/// same bytes. Rendering is pure data transformation: no code is
/// executed, no shell is involved.
pub fn render(template: &str, data: &Value) -> Result<Vec<u8>, RenderError> {
    let nodes = parse_template(template)?;
    let mut scopes = match data {
        Value::Map(pairs) => {
            let mut top = HashMap::new();
            for (k, v) in pairs {
                top.insert(k.clone(), v.clone());
            }
            vec![top]
        }
        _ => {
            return Err(RenderError::TypeMismatch(
                "template data must be a map of named fields".to_string(),
            ));
        }
    };
    let mut out = Vec::new();
    render_nodes(&nodes, &mut scopes, &mut out)?;
    Ok(out)
}

/// A rendered output: the bytes plus their content address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedOutput {
    /// The rendered bytes.
    pub bytes: Vec<u8>,
    /// SHA-256 of the bytes, as a content address.
    pub address: Cid,
}

/// Render a template and address the output by its hash. The same
/// template plus the same data always yields the same address.
pub fn render_addressed(template: &str, data: &Value) -> Result<RenderedOutput, RenderError> {
    let bytes = render(template, data)?;
    let address = Cid::from_bytes(kernel_crypto::sha256(&bytes));
    Ok(RenderedOutput { bytes, address })
}

/// A canonical byte encoding of a value, for hashing. Map pairs are
/// sorted by key, so the same data in a different field order hashes
/// the same.
fn canonical_bytes(value: &Value, out: &mut Vec<u8>) {
    match value {
        Value::Text(s) => {
            out.push(b'T');
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        Value::Number(n) => {
            out.push(b'N');
            out.extend_from_slice(&n.to_le_bytes());
        }
        Value::Bool(b) => {
            out.push(if *b { b'B' } else { b'b' });
        }
        Value::List(items) => {
            out.push(b'L');
            out.extend_from_slice(&(items.len() as u32).to_le_bytes());
            for item in items {
                canonical_bytes(item, out);
            }
        }
        Value::Map(pairs) => {
            out.push(b'M');
            let mut sorted: Vec<&(String, Value)> = pairs.iter().collect();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            out.extend_from_slice(&(sorted.len() as u32).to_le_bytes());
            for (k, v) in sorted {
                out.extend_from_slice(&(k.len() as u32).to_le_bytes());
                out.extend_from_slice(k.as_bytes());
                canonical_bytes(v, out);
            }
        }
    }
}

/// The input hash of a render job: SHA-256 over a domain separator,
/// the template bytes, and the canonical data encoding. Two jobs
/// with the same template and the same data have the same input
/// hash, whatever order the data fields were given in.
pub fn input_hash(template: &str, data: &Value) -> Cid {
    let mut bytes = b"HEYAI-RENDER1".to_vec();
    bytes.extend_from_slice(template.as_bytes());
    bytes.push(0);
    canonical_bytes(data, &mut bytes);
    Cid::from_bytes(kernel_crypto::sha256(&bytes))
}

/// A render job: a template plus its data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderJob {
    /// The template to render.
    pub template: String,
    /// The data the template reads.
    pub data: Value,
}

/// The status of a render job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobStatus {
    /// Submitted, not run yet.
    Pending,
    /// Rendered. `output` is the content address of the rendered
    /// bytes.
    Done {
        /// The rendered output's content address.
        output: Cid,
        /// The rendered output's length in bytes.
        output_len: usize,
    },
    /// Rendering failed. The message is the render error.
    Failed(String),
}

/// The recorded result of one render job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobRecord {
    /// The job's id, assigned in submission order.
    pub id: u64,
    /// The hash of the job's template plus data.
    pub input_hash: Cid,
    /// Where the job stands.
    pub status: JobStatus,
}

/// A queue of render jobs, run in submission order. A failed job
/// records its error and never blocks the jobs after it.
pub struct RenderQueue {
    next_id: u64,
    pending: VecDeque<(u64, RenderJob)>,
    results: Vec<JobRecord>,
    outputs: HashMap<u64, Vec<u8>>,
}

impl Default for RenderQueue {
    /// An empty queue, the same as [`RenderQueue::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl RenderQueue {
    /// An empty queue. Job ids start at 1.
    pub fn new() -> Self {
        Self {
            next_id: 1,
            pending: VecDeque::new(),
            results: Vec::new(),
            outputs: HashMap::new(),
        }
    }

    /// Submit a job. Returns its id. Submission never fails: a bad
    /// template is discovered when the job runs, and recorded as a
    /// failed result.
    pub fn submit(&mut self, template: &str, data: Value) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let hash = input_hash(template, &data);
        self.pending.push_back((
            id,
            RenderJob {
                template: template.to_string(),
                data,
            },
        ));
        self.results.push(JobRecord {
            id,
            input_hash: hash,
            status: JobStatus::Pending,
        });
        id
    }

    /// Number of jobs waiting to run.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// All recorded results, in submission order.
    pub fn results(&self) -> &[JobRecord] {
        &self.results
    }

    /// The recorded result for one job, if it was submitted.
    pub fn result(&self, id: u64) -> Option<&JobRecord> {
        self.results.iter().find(|r| r.id == id)
    }

    /// The rendered bytes of a finished job, if it succeeded.
    pub fn output_bytes(&self, id: u64) -> Option<&[u8]> {
        self.outputs.get(&id).map(|b| b.as_slice())
    }

    /// Run the next pending job, if any. Returns its id. A failure
    /// is recorded on the job and does not stop the queue.
    pub fn run_next(&mut self) -> Option<u64> {
        let (id, job) = self.pending.pop_front()?;
        self.run_one(id, job);
        Some(id)
    }

    /// Run every pending job, in submission order.
    pub fn run_all(&mut self) {
        while self.run_next().is_some() {}
    }

    fn run_one(&mut self, id: u64, job: RenderJob) {
        match render_addressed(&job.template, &job.data) {
            Ok(rendered) => {
                let output_len = rendered.bytes.len();
                let output = rendered.address;
                self.outputs.insert(id, rendered.bytes);
                if let Some(record) = self.results.iter_mut().find(|r| r.id == id) {
                    record.status = JobStatus::Done { output, output_len };
                }
            }
            Err(e) => {
                if let Some(record) = self.results.iter_mut().find(|r| r.id == id) {
                    record.status = JobStatus::Failed(e.to_string());
                }
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
    use kernel_storage::decode_manifest;

    fn test_data(n: usize) -> Vec<u8> {
        (0..n).map(|i| ((i * 37 + 11) % 251) as u8).collect()
    }

    #[test]
    fn voice_note_round_trip_small_and_large() {
        let alice = Identity::generate("alice").unwrap();
        let mut voice = VoiceStore::new();
        let small = b"fake voice bytes";
        let cid = voice.put_voice_note("memo", 12, small, &alice).unwrap();
        let (data, meta) = voice.get_voice_note(&cid).unwrap();
        assert_eq!(data, small);
        assert_eq!(meta.name, "memo");
        assert_eq!(meta.duration_secs, 12);
        assert_eq!(meta.owner, *alice.peer_id());
        assert_eq!(voice.len(), 1);

        let large = test_data(600 * 1024);
        let big_cid = voice
            .put_voice_note("long memo", 300, &large, &alice)
            .unwrap();
        let (data, _) = voice.get_voice_note(&big_cid).unwrap();
        assert_eq!(data, large);
        assert_eq!(voice.len(), 2);
    }

    #[test]
    fn tampered_chunk_fails() {
        let alice = Identity::generate("alice").unwrap();
        let mut voice = VoiceStore::new();
        let data = test_data(600 * 1024);
        let cid = voice.put_voice_note("memo", 300, &data, &alice).unwrap();
        let manifest = voice.get_manifest(&cid).unwrap().clone();
        let manifest_bytes = voice.store_mut().get(&manifest.root).unwrap();
        let chunk_cids = decode_manifest(&manifest_bytes).unwrap();
        let mut bad = voice.store_mut().get(&chunk_cids[1]).unwrap();
        bad[0] ^= 0xff;
        voice.store_mut().put_unchecked(&chunk_cids[1], bad);
        match voice.get_voice_note(&cid) {
            Err(VoiceError::Storage(StorageError::CorruptChunk { cid: got, .. })) => {
                assert_eq!(got, chunk_cids[1]);
            }
            other => panic!("expected CorruptChunk, got {other:?}"),
        }
    }

    #[test]
    fn tampered_manifest_signature_fails() {
        let alice = Identity::generate("alice").unwrap();
        let mut voice = VoiceStore::new();
        let cid = voice.put_voice_note("memo", 12, b"audio", &alice).unwrap();
        let mut bytes = voice.store_mut().get(&cid).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let tampered_cid = voice.store_mut().put(&bytes);
        assert_ne!(tampered_cid, cid);
        assert_eq!(
            voice.get_voice_note(&tampered_cid),
            Err(VoiceError::BadSignature)
        );
        assert_eq!(voice.get_voice_note(&cid).unwrap().0, b"audio");
    }

    #[test]
    fn parse_command_maps_known_commands() {
        assert_eq!(
            parse_command("play Midnight"),
            VoiceAction::Play {
                track_name: "Midnight".to_string()
            }
        );
        assert_eq!(
            parse_command("play   Bohemian Rhapsody  "),
            VoiceAction::Play {
                track_name: "Bohemian Rhapsody".to_string()
            }
        );
        assert_eq!(
            parse_command("PLAY Stairway"),
            VoiceAction::Play {
                track_name: "Stairway".to_string()
            }
        );
        assert_eq!(parse_command("pause"), VoiceAction::Pause);
        assert_eq!(parse_command("  PAUSE  "), VoiceAction::Pause);
        assert_eq!(parse_command("next"), VoiceAction::Next);
        assert_eq!(parse_command("previous"), VoiceAction::Previous);
        assert_eq!(parse_command("prev"), VoiceAction::Previous);
        assert_eq!(parse_command("list"), VoiceAction::ListTracks);
    }

    #[test]
    fn parse_command_unknown_never_panics() {
        assert_eq!(parse_command(""), VoiceAction::Unknown);
        assert_eq!(parse_command("   "), VoiceAction::Unknown);
        assert_eq!(parse_command("play"), VoiceAction::Unknown);
        assert_eq!(parse_command("play   "), VoiceAction::Unknown);
        assert_eq!(parse_command("stop the music"), VoiceAction::Unknown);
        assert_eq!(parse_command("volume up"), VoiceAction::Unknown);
        assert_eq!(parse_command("playing games"), VoiceAction::Unknown);
        assert_eq!(parse_command("pause the song please"), VoiceAction::Unknown);
        // Non-ASCII input is handled, not a panic.
        assert_eq!(parse_command("▶ play"), VoiceAction::Unknown);
    }

    #[test]
    fn command_round_trip_and_bad_signature() {
        let alice = Identity::generate("alice").unwrap();
        let mut voice = VoiceStore::new();
        let cid = voice.submit_command("play Midnight", &alice).unwrap();
        let command = voice.get_command(&cid).unwrap();
        assert_eq!(command.text, "play Midnight");
        assert_eq!(command.owner, *alice.peer_id());
        assert!(command.verify());
        assert_eq!(
            command.action(),
            VoiceAction::Play {
                track_name: "Midnight".to_string()
            }
        );
        // Tampered command record fails.
        let mut bytes = voice.store_mut().get(&cid).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let tampered_cid = voice.store_mut().put(&bytes);
        assert_eq!(
            voice.get_command(&tampered_cid),
            Err(VoiceError::BadSignature)
        );
        assert!(voice.submit_command("", &alice).is_err());
        assert!(voice.submit_command("   ", &alice).is_err());
    }

    #[test]
    fn list_voice_notes_sorted_by_name() {
        let alice = Identity::generate("alice").unwrap();
        let mut voice = VoiceStore::new();
        voice.put_voice_note("zebra", 5, b"1", &alice).unwrap();
        voice.put_voice_note("apple", 5, b"2", &alice).unwrap();
        voice.put_voice_note("mango", 5, b"3", &alice).unwrap();
        let listed = voice.list_voice_notes();
        let names: Vec<&str> = listed.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, vec!["apple", "mango", "zebra"]);
    }

    // --- Render engine tests ---

    fn sample_data() -> Value {
        Value::map(vec![
            ("name".to_string(), Value::text("Tristan")),
            ("price".to_string(), Value::number(250)),
            ("in_stock".to_string(), Value::bool(true)),
            (
                "items".to_string(),
                Value::list(vec![Value::text("lamp"), Value::text("chair")]),
            ),
        ])
    }

    fn rendered(template: &str, data: &Value) -> String {
        String::from_utf8(render(template, data).unwrap()).unwrap()
    }

    #[test]
    fn template_substitution() {
        let data = sample_data();
        assert_eq!(
            rendered("Hello {{name}}, {{price}} coins.", &data),
            "Hello Tristan, 250 coins."
        );
        assert_eq!(rendered("In stock: {{in_stock}}", &data), "In stock: true");
        // Spaces inside tags are fine.
        assert_eq!(rendered("Hello {{ name }}!", &data), "Hello Tristan!");
        // A template with no tags is literal text.
        assert_eq!(rendered("plain text", &data), "plain text");
    }

    #[test]
    fn template_loops() {
        let data = sample_data();
        assert_eq!(
            rendered("{{#each items}}[{{this}}]{{/each}}", &data),
            "[lamp][chair]"
        );
        // An empty list renders nothing.
        let empty = Value::map(vec![("items".to_string(), Value::list(vec![]))]);
        assert_eq!(rendered("{{#each items}}x{{/each}}", &empty), "");
        // Looping over maps exposes each item's fields by name.
        let products = Value::map(vec![(
            "products".to_string(),
            Value::list(vec![
                Value::map(vec![
                    ("name".to_string(), Value::text("lamp")),
                    ("price".to_string(), Value::number(250)),
                ]),
                Value::map(vec![
                    ("name".to_string(), Value::text("chair")),
                    ("price".to_string(), Value::number(80)),
                ]),
            ]),
        )]);
        assert_eq!(
            rendered("{{#each products}}{{name}}={{price}};{{/each}}", &products),
            "lamp=250;chair=80;"
        );
    }

    #[test]
    fn template_conditionals() {
        let data = sample_data();
        assert_eq!(rendered("{{#if in_stock}}yes{{/if}}", &data), "yes");
        let out = Value::map(vec![("in_stock".to_string(), Value::bool(false))]);
        assert_eq!(rendered("{{#if in_stock}}yes{{/if}}", &out), "");
        assert_eq!(rendered("{{#if in_stock}}yes{{/if}}no", &out), "no");
    }

    #[test]
    fn template_nested_loops() {
        let groups = Value::map(vec![(
            "groups".to_string(),
            Value::list(vec![
                Value::map(vec![(
                    "members".to_string(),
                    Value::list(vec![Value::text("a"), Value::text("b")]),
                )]),
                Value::map(vec![(
                    "members".to_string(),
                    Value::list(vec![Value::text("c"), Value::text("d")]),
                )]),
            ]),
        )]);
        assert_eq!(
            rendered(
                "{{#each groups}}{{#each members}}{{this}},{{/each}}|{{/each}}",
                &groups
            ),
            "a,b,|c,d,|"
        );
    }

    #[test]
    fn unknown_key_refused() {
        let data = sample_data();
        assert_eq!(
            render("Hello {{missing}}", &data),
            Err(RenderError::UnknownKey("missing".to_string()))
        );
        assert_eq!(
            render("{{#each missing}}x{{/each}}", &data),
            Err(RenderError::UnknownKey("missing".to_string()))
        );
        assert_eq!(
            render("{{#if missing}}x{{/if}}", &data),
            Err(RenderError::UnknownKey("missing".to_string()))
        );
    }

    #[test]
    fn malformed_template_refused() {
        let data = sample_data();
        // Unclosed block.
        assert!(matches!(
            render("{{#each items}}x", &data),
            Err(RenderError::UnclosedTag(_))
        ));
        // Unclosed tag.
        assert!(matches!(
            render("Hello {{name", &data),
            Err(RenderError::UnclosedTag(_))
        ));
        // Empty tag.
        assert!(matches!(
            render("Hello {{}}", &data),
            Err(RenderError::Malformed(_))
        ));
        // Closing tag with no open block.
        assert!(matches!(
            render("x{{/each}}", &data),
            Err(RenderError::Malformed(_))
        ));
        // Mismatched close tag.
        assert!(matches!(
            render("{{#each items}}x{{/if}}", &data),
            Err(RenderError::Malformed(_))
        ));
        // Block tag with no key.
        assert!(matches!(
            render("{{#each}}x{{/each}}", &data),
            Err(RenderError::Malformed(_))
        ));
        assert!(matches!(
            render("{{#if}}x{{/if}}", &data),
            Err(RenderError::Malformed(_))
        ));
        // Unknown block tag.
        assert!(matches!(
            render("{{#foo}}x{{/foo}}", &data),
            Err(RenderError::Malformed(_))
        ));
        // Bad variable name.
        assert!(matches!(
            render("{{na me}}", &data),
            Err(RenderError::Malformed(_))
        ));
    }

    #[test]
    fn type_errors_are_clean() {
        let data = sample_data();
        // A list cannot be substituted as text.
        assert!(matches!(
            render("{{items}}", &data),
            Err(RenderError::TypeMismatch(_))
        ));
        // #each needs a list.
        assert!(matches!(
            render("{{#each name}}x{{/each}}", &data),
            Err(RenderError::TypeMismatch(_))
        ));
        // #if needs a boolean.
        assert!(matches!(
            render("{{#if price}}x{{/if}}", &data),
            Err(RenderError::TypeMismatch(_))
        ));
        // Top-level data must be a map.
        assert!(matches!(
            render("{{this}}", &Value::text("nope")),
            Err(RenderError::TypeMismatch(_))
        ));
    }

    #[test]
    fn deterministic_rendering() {
        let data = sample_data();
        let first = render_addressed("Hello {{name}}, {{price}} coins.", &data).unwrap();
        let second = render_addressed("Hello {{name}}, {{price}} coins.", &data).unwrap();
        assert_eq!(first.bytes, second.bytes);
        assert_eq!(first.address, second.address);
        assert_eq!(
            first.address,
            Cid::from_bytes(kernel_crypto::sha256(&first.bytes))
        );
        // Different data, different address.
        let mut other = sample_data();
        if let Value::Map(pairs) = &mut other {
            pairs[0].1 = Value::text("Someone else");
        }
        let third = render_addressed("Hello {{name}}, {{price}} coins.", &other).unwrap();
        assert_ne!(first.address, third.address);
        // Same data in a different field order: same input hash.
        let reordered = Value::map(vec![
            (
                "items".to_string(),
                Value::list(vec![Value::text("lamp"), Value::text("chair")]),
            ),
            ("in_stock".to_string(), Value::bool(true)),
            ("price".to_string(), Value::number(250)),
            ("name".to_string(), Value::text("Tristan")),
        ]);
        assert_eq!(
            input_hash("Hello {{name}}", &data),
            input_hash("Hello {{name}}", &reordered)
        );
    }

    #[test]
    fn job_queue_runs_in_order_and_failure_does_not_block() {
        let mut queue = RenderQueue::new();
        let good = sample_data();
        let id1 = queue.submit("Hello {{name}}", good.clone());
        let id2 = queue.submit("Hello {{missing}}", good.clone());
        let id3 = queue.submit("{{price}} coins", good.clone());
        assert_eq!((id1, id2, id3), (1, 2, 3));
        assert_eq!(queue.pending_count(), 3);
        queue.run_all();
        assert_eq!(queue.pending_count(), 0);

        let results = queue.results().to_vec();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 1);
        assert_eq!(results[1].id, 2);
        assert_eq!(results[2].id, 3);
        assert!(matches!(results[0].status, JobStatus::Done { .. }));
        assert!(matches!(results[1].status, JobStatus::Failed(_)));
        // The failed job did not block the third one.
        assert!(matches!(results[2].status, JobStatus::Done { .. }));
        assert_eq!(queue.output_bytes(1).unwrap(), b"Hello Tristan");
        assert_eq!(queue.output_bytes(2), None);
        assert_eq!(queue.output_bytes(3).unwrap(), b"250 coins");
        // Identical jobs share an input hash.
        let id4 = queue.submit("Hello {{name}}", good);
        assert_eq!(queue.result(id4).unwrap().input_hash, results[0].input_hash);
        assert_eq!(queue.result(id4).unwrap().status, JobStatus::Pending);
    }

    #[test]
    fn job_queue_run_next_one_at_a_time() {
        let mut queue = RenderQueue::new();
        let data = sample_data();
        let id1 = queue.submit("{{name}}", data.clone());
        let id2 = queue.submit("{{price}}", data);
        assert_eq!(queue.run_next(), Some(id1));
        assert_eq!(queue.pending_count(), 1);
        assert!(matches!(
            queue.result(id1).unwrap().status,
            JobStatus::Done { .. }
        ));
        assert_eq!(queue.result(id2).unwrap().status, JobStatus::Pending);
        assert_eq!(queue.run_next(), Some(id2));
        assert_eq!(queue.run_next(), None);
    }
}
