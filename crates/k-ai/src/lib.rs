//! K•AI — an AI task/response framework on KERNEL.
//!
//! This crate is a **framework, not a model**. It defines how AI work is
//! requested and answered on the network, with signatures on both sides
//! so a task and its response are attributable and tamper-evident:
//!
//! * A **task** is a signed record: a human-readable id, the owner's
//!   [`PeerId`], prompt text, and the owner's signature.
//! * A **provider** is a [`AiProvider`] trait: given a task, it returns
//!   response text. Any implementation can be plugged in — a local
//!   model, a remote service, a human in the loop.
//! * A **response** is a signed record: the task id it answers, the
//!   provider's name, the response text, and the provider identity's
//!   signature. A response that names the wrong task, or whose
//!   signature does not verify, is rejected.
//!
//! This crate ships exactly one provider, [`TestProvider`]: a local,
//! deterministic test provider that returns the prompt uppercased with
//! a fixed `KAI-TEST:` prefix. It is clearly labeled a test provider.
//! It is not intelligence, and nothing here pretends otherwise.
//!
//! [`agent::LocalAgent`] is a real local agent provider — Kai, the
//! creator-bundle agent. It routes a task to one of four tools by
//! deterministic keyword intent (storage search, bundle assembly,
//! bundle purchase, template rendering through the HEY•AI engine)
//! and composes the signed response from the tool's actual output.
//! It never invents an answer: an intent it cannot route is refused.
//! It is still not a model — the routing is keywords, not
//! understanding. Bundles are signed manifests over content the
//! creator supplies ([`bundle::BundleManifest`]); Kai assembles and
//! sells them, it does not create the content inside them.
//!
//! Record formats (explicit binary, no JSON):
//! `KAITSK1` + id length (u16 LE) + id bytes + owner public key (32) +
//! prompt length (u32 LE) + prompt bytes + signature (64).
//! `KAIRESP` + task id length (u16 LE) + task id bytes + provider name
//! length (u16 LE) + provider name bytes + response length (u32 LE) +
//! response bytes + provider public key (32) + signature (64).

pub mod agent;
pub mod bundle;

pub use agent::{LocalAgent, Tool, ToolCall, ToolError};
pub use bundle::{BundleItem, BundleManifest};

use std::collections::HashMap;
use std::fmt;

use kernel_crypto::Ed25519Signature;
use kernel_identity::{Identity, IdentityError, PeerId};
use kernel_storage::{Cid, ContentStore, StorageError};

/// Magic bytes at the start of every task record.
const TASK_MAGIC: &[u8; 7] = b"KAITSK1";
/// Domain separator for task signatures.
const TASK_SIGN_DOMAIN: &[u8] = b"KAITSK1";
/// Magic bytes at the start of every response record.
const RESPONSE_MAGIC: &[u8; 7] = b"KAIRESP";
/// Domain separator for response signatures.
const RESPONSE_SIGN_DOMAIN: &[u8] = b"KAIRESP";

/// Maximum length of a task id.
pub const MAX_TASK_ID_LEN: usize = 64;
/// Maximum prompt or response length in bytes.
pub const MAX_TEXT_LEN: usize = 100_000;

/// Errors returned by AI task/response operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AiError {
    /// Wraps a [`StorageError`].
    Storage(StorageError),
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// A task id broke the naming rules.
    InvalidId(String),
    /// A prompt or response text broke the rules.
    InvalidText(String),
    /// A task record was malformed.
    InvalidTask(String),
    /// A response record was malformed.
    InvalidResponse(String),
    /// A task id is already in use.
    AlreadyExists(String),
    /// No task with this id is known here.
    TaskNotFound(String),
    /// No response with this CID is known here.
    ResponseNotFound(Cid),
    /// A task's signature did not verify against its owner.
    BadTaskSignature,
    /// A response's signature did not verify against its provider.
    BadResponseSignature,
    /// A provider failed to produce a response.
    Provider(String),
}

impl fmt::Display for AiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AiError::Storage(e) => write!(f, "storage error: {e}"),
            AiError::Identity(e) => write!(f, "identity error: {e}"),
            AiError::InvalidId(msg) => write!(f, "invalid task id: {msg}"),
            AiError::InvalidText(msg) => write!(f, "invalid text: {msg}"),
            AiError::InvalidTask(msg) => write!(f, "invalid task: {msg}"),
            AiError::InvalidResponse(msg) => write!(f, "invalid response: {msg}"),
            AiError::AlreadyExists(id) => write!(f, "task id already exists: {id}"),
            AiError::TaskNotFound(id) => write!(f, "task not found: {id}"),
            AiError::ResponseNotFound(cid) => write!(f, "response not found: {cid}"),
            AiError::BadTaskSignature => write!(f, "task signature is invalid"),
            AiError::BadResponseSignature => write!(f, "response signature is invalid"),
            AiError::Provider(msg) => write!(f, "provider error: {msg}"),
        }
    }
}

impl std::error::Error for AiError {}

impl From<StorageError> for AiError {
    fn from(e: StorageError) -> Self {
        AiError::Storage(e)
    }
}

impl From<IdentityError> for AiError {
    fn from(e: IdentityError) -> Self {
        AiError::Identity(e)
    }
}

/// Rules for task ids: 1-64 characters, ASCII letters, digits, `-`,
/// `_`, `.`. Ids are human labels (`"summarize-001"`), not hashes.
fn validate_task_id(id: &str) -> Result<(), AiError> {
    if id.is_empty() {
        return Err(AiError::InvalidId("task id must not be empty".to_string()));
    }
    if id.len() > MAX_TASK_ID_LEN {
        return Err(AiError::InvalidId(format!(
            "task id must be at most {MAX_TASK_ID_LEN} characters"
        )));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Err(AiError::InvalidId(
            "task id may only use letters, digits, '-', '_' and '.'".to_string(),
        ));
    }
    Ok(())
}

/// Rules for prompts and responses: non-empty, at most [`MAX_TEXT_LEN`] bytes.
fn validate_text(text: &str) -> Result<(), AiError> {
    if text.is_empty() {
        return Err(AiError::InvalidText("text must not be empty".to_string()));
    }
    if text.len() > MAX_TEXT_LEN {
        return Err(AiError::InvalidText(format!(
            "text must be at most {MAX_TEXT_LEN} bytes"
        )));
    }
    Ok(())
}

/// A display name derived from a public key: the first 8 hex chars of
/// the key. Used when a record is decoded and only the key survives
/// (verification needs the key, not the human name).
fn fingerprint_name(key: &kernel_crypto::Ed25519PublicKey) -> String {
    let hex = kernel_crypto::to_hex(key.as_bytes());
    format!("peer-{}", &hex[..8])
}

/// The message an owner signs for a task.
fn task_sign_message(id: &str, prompt: &str, owner: &PeerId) -> Vec<u8> {
    let mut msg = Vec::with_capacity(TASK_SIGN_DOMAIN.len() + id.len() + prompt.len() + 32);
    msg.extend_from_slice(TASK_SIGN_DOMAIN);
    msg.extend_from_slice(id.as_bytes());
    msg.extend_from_slice(prompt.as_bytes());
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg
}

/// The message a provider signs for a response.
fn response_sign_message(
    task_id: &str,
    provider_name: &str,
    text: &str,
    provider: &PeerId,
) -> Vec<u8> {
    let mut msg = Vec::with_capacity(
        RESPONSE_SIGN_DOMAIN.len() + task_id.len() + provider_name.len() + text.len() + 32,
    );
    msg.extend_from_slice(RESPONSE_SIGN_DOMAIN);
    msg.extend_from_slice(task_id.as_bytes());
    msg.extend_from_slice(provider_name.as_bytes());
    msg.extend_from_slice(text.as_bytes());
    msg.extend_from_slice(provider.public_key().as_bytes());
    msg
}

/// A signed AI task: human-readable id, owner, prompt text, signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AiTask {
    /// Human-readable task id, e.g. `"summarize-001"`.
    pub id: String,
    /// The peer who submitted this task.
    pub owner: PeerId,
    /// The prompt text.
    pub prompt: String,
    /// The owner's signature over id, prompt, and owner key.
    pub signature: Ed25519Signature,
}

impl AiTask {
    /// Build and sign a task as `identity`.
    pub fn new(id: &str, prompt: &str, identity: &Identity) -> Result<Self, AiError> {
        validate_task_id(id)?;
        validate_text(prompt)?;
        let owner = identity.peer_id().clone();
        let msg = task_sign_message(id, prompt, &owner);
        let signature = identity.sign(&msg);
        Ok(Self {
            id: id.to_string(),
            owner,
            prompt: prompt.to_string(),
            signature,
        })
    }

    /// True if the signature verifies against the owner for this exact
    /// id and prompt.
    pub fn verify(&self) -> bool {
        let msg = task_sign_message(&self.id, &self.prompt, &self.owner);
        self.owner.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary task format.
    pub fn encode(&self) -> Vec<u8> {
        let id_bytes = self.id.as_bytes();
        let prompt_bytes = self.prompt.as_bytes();
        let mut out = Vec::with_capacity(7 + 2 + id_bytes.len() + 32 + 4 + prompt_bytes.len() + 64);
        out.extend_from_slice(TASK_MAGIC);
        out.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(id_bytes);
        out.extend_from_slice(self.owner.public_key().as_bytes());
        out.extend_from_slice(&(prompt_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(prompt_bytes);
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a task produced by [`AiTask::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, AiError> {
        if bytes.len() < 7 + 2 + 32 + 4 + 64 || &bytes[..7] != TASK_MAGIC {
            return Err(AiError::InvalidTask("missing KAITSK1 magic".to_string()));
        }
        let id_len = u16::from_le_bytes(bytes[7..9].try_into().expect("2-byte slice")) as usize;
        let prompt_len_off = 9 + id_len + 32;
        if bytes.len() < prompt_len_off + 4 {
            return Err(AiError::InvalidTask(
                "task too short for prompt length".to_string(),
            ));
        }
        let prompt_len = u32::from_le_bytes(
            bytes[prompt_len_off..prompt_len_off + 4]
                .try_into()
                .expect("4-byte slice"),
        ) as usize;
        if prompt_len > MAX_TEXT_LEN {
            return Err(AiError::InvalidTask(format!(
                "prompt claims {prompt_len} bytes, max is {MAX_TEXT_LEN}"
            )));
        }
        let expected = prompt_len_off + 4 + prompt_len + 64;
        if bytes.len() != expected {
            return Err(AiError::InvalidTask(format!(
                "task is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let id = String::from_utf8(bytes[9..9 + id_len].to_vec())
            .map_err(|_| AiError::InvalidTask("task id is not valid UTF-8".to_string()))?;
        validate_task_id(&id)?;
        let mut off = prompt_len_off;
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[9 + id_len..9 + id_len + 32]);
        off += 4; // skip prompt length
        let prompt = String::from_utf8(bytes[off..off + prompt_len].to_vec())
            .map_err(|_| AiError::InvalidTask("prompt is not valid UTF-8".to_string()))?;
        validate_text(&prompt)?;
        off += prompt_len;
        let owner_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| AiError::InvalidTask(e.to_string()))?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let owner = PeerId::new(owner_key, &fingerprint_name(&owner_key))?;
        Ok(Self {
            id,
            owner,
            prompt,
            signature,
        })
    }
}

/// A signed AI response: the task it answers, the provider's name,
/// the response text, and the provider identity's signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AiResponse {
    /// The id of the task this response answers.
    pub task_id: String,
    /// The name of the provider that produced this response.
    pub provider_name: String,
    /// The response text.
    pub text: String,
    /// The provider identity that signed this response.
    pub provider: PeerId,
    /// The provider's signature over task id, provider name, text, provider key.
    pub signature: Ed25519Signature,
}

impl AiResponse {
    /// Build and sign a response as `provider_identity`.
    pub fn new(
        task_id: &str,
        provider_name: &str,
        text: &str,
        provider_identity: &Identity,
    ) -> Result<Self, AiError> {
        validate_task_id(task_id)?;
        if provider_name.is_empty() || provider_name.len() > 64 {
            return Err(AiError::InvalidResponse(
                "provider name must be 1-64 characters".to_string(),
            ));
        }
        validate_text(text)?;
        let provider = provider_identity.peer_id().clone();
        let msg = response_sign_message(task_id, provider_name, text, &provider);
        let signature = provider_identity.sign(&msg);
        Ok(Self {
            task_id: task_id.to_string(),
            provider_name: provider_name.to_string(),
            text: text.to_string(),
            provider,
            signature,
        })
    }

    /// True if the signature verifies against the provider for this
    /// exact task id, provider name, and text.
    pub fn verify(&self) -> bool {
        let msg = response_sign_message(
            &self.task_id,
            &self.provider_name,
            &self.text,
            &self.provider,
        );
        self.provider.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary response format.
    pub fn encode(&self) -> Vec<u8> {
        let task_bytes = self.task_id.as_bytes();
        let provider_bytes = self.provider_name.as_bytes();
        let text_bytes = self.text.as_bytes();
        let mut out = Vec::with_capacity(
            7 + 2 + task_bytes.len() + 2 + provider_bytes.len() + 4 + text_bytes.len() + 32 + 64,
        );
        out.extend_from_slice(RESPONSE_MAGIC);
        out.extend_from_slice(&(task_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(task_bytes);
        out.extend_from_slice(&(provider_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(provider_bytes);
        out.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(text_bytes);
        out.extend_from_slice(self.provider.public_key().as_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a response produced by [`AiResponse::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, AiError> {
        if bytes.len() < 7 + 2 + 2 + 4 + 32 + 64 || &bytes[..7] != RESPONSE_MAGIC {
            return Err(AiError::InvalidResponse(
                "missing KAIRESP magic".to_string(),
            ));
        }
        let task_len = u16::from_le_bytes(bytes[7..9].try_into().expect("2-byte slice")) as usize;
        let provider_len_off = 9 + task_len;
        if bytes.len() < provider_len_off + 2 {
            return Err(AiError::InvalidResponse(
                "response too short for provider length".to_string(),
            ));
        }
        let provider_len = u16::from_le_bytes(
            bytes[provider_len_off..provider_len_off + 2]
                .try_into()
                .expect("2-byte slice"),
        ) as usize;
        let text_len_off = provider_len_off + 2 + provider_len;
        if bytes.len() < text_len_off + 4 {
            return Err(AiError::InvalidResponse(
                "response too short for text length".to_string(),
            ));
        }
        let text_len = u32::from_le_bytes(
            bytes[text_len_off..text_len_off + 4]
                .try_into()
                .expect("4-byte slice"),
        ) as usize;
        if text_len > MAX_TEXT_LEN {
            return Err(AiError::InvalidResponse(format!(
                "response claims {text_len} bytes, max is {MAX_TEXT_LEN}"
            )));
        }
        let expected = text_len_off + 4 + text_len + 32 + 64;
        if bytes.len() != expected {
            return Err(AiError::InvalidResponse(format!(
                "response is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let task_id = String::from_utf8(bytes[9..9 + task_len].to_vec())
            .map_err(|_| AiError::InvalidResponse("task id is not valid UTF-8".to_string()))?;
        validate_task_id(&task_id)?;
        let provider_name = String::from_utf8(
            bytes[provider_len_off + 2..provider_len_off + 2 + provider_len].to_vec(),
        )
        .map_err(|_| AiError::InvalidResponse("provider name is not valid UTF-8".to_string()))?;
        let mut off = text_len_off + 4;
        let text = String::from_utf8(bytes[off..off + text_len].to_vec()).map_err(|_| {
            AiError::InvalidResponse("response text is not valid UTF-8".to_string())
        })?;
        validate_text(&text)?;
        off += text_len;
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let provider_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| AiError::InvalidResponse(e.to_string()))?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let provider = PeerId::new(provider_key, &fingerprint_name(&provider_key))?;
        Ok(Self {
            task_id,
            provider_name,
            text,
            provider,
            signature,
        })
    }
}

/// An AI provider: something that takes a task and returns response
/// text. The framework does not care how the text is produced — a
/// local model, a remote service, a human — only that the response is
/// signed by the provider identity that ran it.
pub trait AiProvider {
    /// The provider's human-readable name, e.g. `"test-uppercase"`.
    /// This name is recorded on every response the provider signs.
    fn name(&self) -> &str;

    /// Produce response text for a task. An error here means the
    /// provider refused or failed; nothing is recorded.
    fn run(&self, task: &AiTask) -> Result<String, AiError>;
}

/// A local, deterministic test provider.
///
/// Its rule is documented and fixed: it returns the prompt uppercased,
/// prefixed with `KAI-TEST: `. It is a test provider for exercising the
/// task/response plumbing. It is not intelligence, and it must never
/// be presented as a model.
pub struct TestProvider;

impl AiProvider for TestProvider {
    fn name(&self) -> &str {
        "test-uppercase"
    }

    fn run(&self, task: &AiTask) -> Result<String, AiError> {
        Ok(format!("KAI-TEST: {}", task.prompt.to_uppercase()))
    }
}

/// The AI service: a content store holding signed tasks and signed
/// responses, plus indexes by task id and by task.
pub struct AiService {
    store: ContentStore,
    /// Task record CIDs, by task id.
    task_cids: HashMap<String, Cid>,
    /// Response records this node holds, by record CID.
    responses: HashMap<Cid, AiResponse>,
    /// Response record CIDs, by task id.
    responses_by_task: HashMap<String, Vec<Cid>>,
}

impl Default for AiService {
    fn default() -> Self {
        Self::new()
    }
}

impl AiService {
    /// An empty AI service.
    pub fn new() -> Self {
        Self {
            store: ContentStore::new(),
            task_cids: HashMap::new(),
            responses: HashMap::new(),
            responses_by_task: HashMap::new(),
        }
    }

    /// Number of tasks held.
    pub fn len(&self) -> usize {
        self.task_cids.len()
    }

    /// True if no tasks are held.
    pub fn is_empty(&self) -> bool {
        self.task_cids.is_empty()
    }

    /// The record CID for a task id, if held.
    pub fn task_cid(&self, task_id: &str) -> Option<Cid> {
        self.task_cids.get(task_id).copied()
    }

    /// Submit a task: build and sign the record and store it as a
    /// block. Returns the task record CID. A duplicate id is refused.
    pub fn submit_task(
        &mut self,
        id: &str,
        prompt: &str,
        identity: &Identity,
    ) -> Result<Cid, AiError> {
        validate_task_id(id)?;
        validate_text(prompt)?;
        if self.task_cids.contains_key(id) {
            return Err(AiError::AlreadyExists(id.to_string()));
        }
        let task = AiTask::new(id, prompt, identity)?;
        let cid = self.store.put(&task.encode());
        self.task_cids.insert(id.to_string(), cid);
        Ok(cid)
    }

    /// Fetch a task by id. The record is re-decoded from the store and
    /// its signature verified on every read, so a tampered record
    /// fails here instead of being trusted from an index.
    pub fn get_task(&self, id: &str) -> Result<AiTask, AiError> {
        let cid = self
            .task_cids
            .get(id)
            .copied()
            .ok_or_else(|| AiError::TaskNotFound(id.to_string()))?;
        let bytes = self.store.get(&cid).map_err(AiError::Storage)?;
        let task = AiTask::decode(&bytes)?;
        if !task.verify() {
            return Err(AiError::BadTaskSignature);
        }
        if task.id != id {
            return Err(AiError::InvalidTask(format!(
                "record id {} does not match requested id {id}",
                task.id
            )));
        }
        Ok(task)
    }

    /// Run a task through a provider: the provider produces response
    /// text, which is signed by the provider identity and stored as a
    /// block. Returns the response record CID.
    pub fn run_task(
        &mut self,
        task_id: &str,
        provider: &dyn AiProvider,
        provider_identity: &Identity,
    ) -> Result<Cid, AiError> {
        let task = self.get_task(task_id)?;
        let text = provider
            .run(&task)
            .map_err(|e| AiError::Provider(e.to_string()))?;
        let response = AiResponse::new(task_id, provider.name(), &text, provider_identity)?;
        let cid = self.store.put(&response.encode());
        self.responses.insert(cid, response);
        self.responses_by_task
            .entry(task_id.to_string())
            .or_default()
            .push(cid);
        Ok(cid)
    }

    /// Fetch a response by record CID: verify its signature, and check
    /// it names a task this node holds. A response for an unknown task
    /// is refused — a response must answer a real task.
    pub fn get_response(&self, response_cid: &Cid) -> Result<AiResponse, AiError> {
        let response = match self.responses.get(response_cid) {
            Some(r) => r.clone(),
            None => {
                let bytes = self.store.get(response_cid).map_err(|e| match e {
                    StorageError::NotFound(cid) => AiError::ResponseNotFound(cid),
                    other => AiError::Storage(other),
                })?;
                AiResponse::decode(&bytes)?
            }
        };
        if !response.verify() {
            return Err(AiError::BadResponseSignature);
        }
        if !self.task_cids.contains_key(&response.task_id) {
            return Err(AiError::TaskNotFound(response.task_id.clone()));
        }
        Ok(response)
    }

    /// Response record CIDs for a task, in the order they were produced.
    pub fn responses_for(&self, task_id: &str) -> Vec<Cid> {
        self.responses_by_task
            .get(task_id)
            .cloned()
            .unwrap_or_default()
    }

    /// All tasks held, as (id, prompt), sorted by id for a stable,
    /// human-readable listing.
    pub fn list_tasks(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        for id in self.task_cids.keys() {
            if let Ok(task) = self.get_task(id) {
                out.push((task.id, task.prompt));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
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
    fn task_encode_decode_round_trip_and_verify() {
        let alice = Identity::generate("alice").unwrap();
        let task = AiTask::new("summarize-001", "hello world", &alice).unwrap();
        assert!(task.verify());
        let encoded = task.encode();
        let decoded = AiTask::decode(&encoded).unwrap();
        assert_eq!(decoded.id, "summarize-001");
        assert_eq!(decoded.prompt, "hello world");
        assert_eq!(decoded.signature, task.signature);
        assert_eq!(decoded.owner.public_key(), task.owner.public_key());
        assert!(decoded.verify());
        assert!(AiTask::decode(b"garbage").is_err());
        assert!(AiTask::decode(&encoded[..encoded.len() - 1]).is_err());
    }

    #[test]
    fn task_round_trip_through_service() {
        let alice = Identity::generate("alice").unwrap();
        let mut service = AiService::new();
        let cid = service
            .submit_task("task-1", "summarize this", &alice)
            .unwrap();
        assert_eq!(service.task_cid("task-1"), Some(cid));
        let task = service.get_task("task-1").unwrap();
        assert_eq!(task.id, "task-1");
        assert_eq!(task.prompt, "summarize this");
        // get_task re-decodes the record from the store, and a decoded
        // record rebuilds its PeerId with the key-derived display name
        // (peer-xxxxxxxx), so compare keys, not display names.
        assert_eq!(task.owner.public_key(), alice.peer_id().public_key());
        assert_eq!(service.len(), 1);
        // Duplicate id refused.
        assert_eq!(
            service.submit_task("task-1", "again", &alice),
            Err(AiError::AlreadyExists("task-1".to_string()))
        );
        assert_eq!(
            service.get_task("nope"),
            Err(AiError::TaskNotFound("nope".to_string()))
        );
    }

    #[test]
    fn response_linked_to_its_task() {
        let alice = Identity::generate("alice").unwrap();
        let provider_id = Identity::generate("provider").unwrap();
        let mut service = AiService::new();
        service
            .submit_task("task-1", "hello world", &alice)
            .unwrap();
        let response_cid = service
            .run_task("task-1", &TestProvider, &provider_id)
            .unwrap();
        let response = service.get_response(&response_cid).unwrap();
        assert_eq!(response.task_id, "task-1");
        assert_eq!(response.provider_name, "test-uppercase");
        assert_eq!(response.text, "KAI-TEST: HELLO WORLD");
        assert_eq!(response.provider, *provider_id.peer_id());
        assert!(response.verify());
        assert_eq!(service.responses_for("task-1"), vec![response_cid]);
        assert!(service.responses_for("task-2").is_empty());
        // Running an unknown task fails.
        assert_eq!(
            service.run_task("nope", &TestProvider, &provider_id),
            Err(AiError::TaskNotFound("nope".to_string()))
        );
    }

    #[test]
    fn test_provider_returns_documented_transform() {
        let alice = Identity::generate("alice").unwrap();
        let task = AiTask::new("t", "Mixed Case 123!", &alice).unwrap();
        assert_eq!(TestProvider.name(), "test-uppercase");
        assert_eq!(
            TestProvider.run(&task).unwrap(),
            "KAI-TEST: MIXED CASE 123!"
        );
    }

    #[test]
    fn bad_task_signature_rejected() {
        let alice = Identity::generate("alice").unwrap();
        let mut service = AiService::new();
        let cid = service.submit_task("task-1", "hello", &alice).unwrap();
        // Tamper with the stored task record bytes (flip a signature byte).
        let mut bytes = service.store_mut().get(&cid).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        service.store_mut().put_unchecked(&cid, bytes);
        assert_eq!(service.get_task("task-1"), Err(AiError::BadTaskSignature));
        // Tampering with the prompt after signing also breaks it.
        let mut task = AiTask::new("task-2", "hello", &alice).unwrap();
        assert!(task.verify());
        task.prompt = "goodbye".to_string();
        assert!(!task.verify());
    }

    #[test]
    fn bad_response_signature_rejected() {
        let alice = Identity::generate("alice").unwrap();
        let provider_id = Identity::generate("provider").unwrap();
        let mut service = AiService::new();
        service.submit_task("task-1", "hello", &alice).unwrap();
        let cid = service
            .run_task("task-1", &TestProvider, &provider_id)
            .unwrap();
        // Tamper with the stored response record bytes.
        let mut bytes = service.store_mut().get(&cid).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let tampered_cid = service.store_mut().put(&bytes);
        assert_ne!(tampered_cid, cid);
        assert_eq!(
            service.get_response(&tampered_cid),
            Err(AiError::BadResponseSignature)
        );
        // The original response still reads fine.
        assert!(service.get_response(&cid).unwrap().verify());
    }

    #[test]
    fn response_for_unknown_task_refused() {
        let provider_id = Identity::generate("provider").unwrap();
        let mut service = AiService::new();
        // A response record naming a task this node never saw.
        let response =
            AiResponse::new("ghost-task", "test-uppercase", "text", &provider_id).unwrap();
        let cid = service.store_mut().put(&response.encode());
        assert_eq!(
            service.get_response(&cid),
            Err(AiError::TaskNotFound("ghost-task".to_string()))
        );
    }

    #[test]
    fn list_tasks_sorted_by_id() {
        let alice = Identity::generate("alice").unwrap();
        let mut service = AiService::new();
        service.submit_task("zeta", "third", &alice).unwrap();
        service.submit_task("alpha", "first", &alice).unwrap();
        service.submit_task("mid", "second", &alice).unwrap();
        let listed = service.list_tasks();
        let ids: Vec<&str> = listed.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, vec!["alpha", "mid", "zeta"]);
    }

    #[test]
    fn task_id_and_text_rules() {
        let alice = Identity::generate("alice").unwrap();
        let mut service = AiService::new();
        assert!(service.submit_task("", "x", &alice).is_err());
        assert!(service.submit_task("has space", "x", &alice).is_err());
        assert!(service.submit_task("ok-id_1.2", "", &alice).is_err());
        assert!(service.submit_task("ok-id_1.2", "fine", &alice).is_ok());
    }
}
