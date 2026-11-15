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
//! Formats (explicit binary, no JSON):
//! `KVOICE1` + name length (u16 LE) + name bytes + duration (u64 LE) +
//! chunk root CID (32) + owner public key (32) + signature (64).
//! `KCMDVC1` + text length (u32 LE) + text bytes + owner public key
//! (32) + signature (64).

use std::collections::HashMap;
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
        let text = String::from_utf8(bytes[11..11 + text_len].to_vec())
            .map_err(|_| VoiceError::InvalidManifest("command text is not valid UTF-8".to_string()))?;
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
    pub fn submit_command(
        &mut self,
        text: &str,
        identity: &Identity,
    ) -> Result<Cid, VoiceError> {
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
}
