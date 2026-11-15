//! K•Talk — signed direct messages between KERNEL peers.
//!
//! A message is a sender, a recipient, human-readable text, a
//! timestamp, and the sender's signature over all of it. Messages travel
//! as single frames over the TCP transport; the receiver verifies the
//! signature and the recipient before the message ever reaches the
//! inbox, and every message is verified again when it is read back out.
//! A message with a bad signature, or addressed to someone else, is
//! rejected — it is never stored and never shown.
//!
//! Wire format (explicit binary, no JSON):
//! `KTALK1` + sender public key (32) + sender name length (u16 LE) +
//! sender name + recipient public key (32) + recipient name length
//! (u16 LE) + recipient name + timestamp (u64 LE, unix seconds) +
//! sender sequence (u64 LE, per-sender counter) + text length (u32 LE)
//! + text bytes + signature (64).
//!
//! The sender sequence is the sender's own monotonic counter. Messages
//! travel over separate TCP connections, which have no guaranteed
//! arrival order, so the sequence — not arrival time — is what orders a
//! sender's messages.
//!
//! On top of direct messages, K•Talk has groups (see [`group`]),
//! a persistent per-conversation history log with delivery receipts
//! (see [`history`]), and an offline store-and-forward outbox (see
//! [`outbox`]).

pub mod group;
pub mod history;
pub mod outbox;

pub use group::{Group, GroupMessage, MembershipAction, MembershipChange};
pub use history::{ConversationId, FoundMessage, History, HistoryEntry, Receipt, ReceiptStatus};
pub use outbox::{Outbox, QueuedMessage, MAX_QUEUED_PER_PEER};

use std::collections::HashMap;
use std::fmt;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use kernel_crypto::{sha256, Ed25519Signature};
use kernel_identity::{Identity, IdentityError, PeerId};
use kernel_transport::{Connection, Listener, TransportError};

/// Magic bytes at the start of every encoded message.
const MESSAGE_MAGIC: &[u8; 6] = b"KTALK1";

/// Domain separator for message signatures: a signature covers
/// `b"KTALK1" || sender key || recipient key || timestamp (u64 LE) ||
/// sender sequence (u64 LE) || text`.
const MESSAGE_SIGN_DOMAIN: &[u8] = b"KTALK1";

/// Errors returned by messaging operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TalkError {
    /// Wraps a [`TransportError`].
    Transport(String),
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// A message was malformed.
    InvalidMessage(String),
    /// A message's signature did not verify against its sender.
    BadSignature,
    /// A message was addressed to a different peer than this node.
    WrongRecipient,
    /// A group record, membership change, or group message was
    /// malformed or failed a group rule.
    InvalidGroup(String),
    /// A group message or membership change named a group this node
    /// does not know.
    UnknownGroup(String),
    /// A sender who is not a member tried to send to a group.
    NotGroupMember,
    /// A receipt named a message this node does not hold.
    UnknownMessage,
    /// A receipt was malformed, had a bad signature, or came from
    /// someone who was not a recipient of its message.
    BadReceipt(String),
    /// The outbox for a peer is at its cap (see
    /// [`MAX_QUEUED_PER_PEER`]).
    OutboxFull {
        /// The peer whose queue is full.
        peer: PeerId,
        /// The cap that was hit.
        max: usize,
    },
    /// A history log failed to load: bad magic, bad checksum,
    /// truncation, or an entry that does not decode.
    CorruptHistory(String),
}

impl fmt::Display for TalkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TalkError::Transport(e) => write!(f, "transport error: {e}"),
            TalkError::Identity(e) => write!(f, "identity error: {e}"),
            TalkError::InvalidMessage(msg) => write!(f, "invalid message: {msg}"),
            TalkError::BadSignature => write!(f, "message signature is invalid"),
            TalkError::WrongRecipient => write!(f, "message is addressed to another peer"),
            TalkError::InvalidGroup(msg) => write!(f, "invalid group: {msg}"),
            TalkError::UnknownGroup(id) => write!(f, "unknown group: {id}"),
            TalkError::NotGroupMember => {
                write!(f, "sender is not a member of the group")
            }
            TalkError::UnknownMessage => {
                write!(f, "receipt names a message this node does not hold")
            }
            TalkError::BadReceipt(msg) => write!(f, "bad receipt: {msg}"),
            TalkError::OutboxFull { peer, max } => write!(
                f,
                "outbox for {} is full (max {max} queued messages)",
                peer.display_name()
            ),
            TalkError::CorruptHistory(msg) => write!(f, "corrupt history: {msg}"),
        }
    }
}

impl std::error::Error for TalkError {}

impl From<TransportError> for TalkError {
    fn from(e: TransportError) -> Self {
        TalkError::Transport(e.to_string())
    }
}

impl From<IdentityError> for TalkError {
    fn from(e: IdentityError) -> Self {
        TalkError::Identity(e)
    }
}

/// Unix time in seconds. Messages that arrive in the same second are
/// ordered by arrival sequence (see [`StoredMessage::seq`]).
fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The message a sender signs: domain + sender key + recipient key +
/// timestamp + text.
pub fn message_sign_bytes(
    sender: &PeerId,
    recipient: &PeerId,
    timestamp: u64,
    seq: u64,
    text: &str,
) -> Vec<u8> {
    let mut msg = Vec::with_capacity(MESSAGE_SIGN_DOMAIN.len() + 32 + 32 + 8 + 8 + text.len());
    msg.extend_from_slice(MESSAGE_SIGN_DOMAIN);
    msg.extend_from_slice(sender.public_key().as_bytes());
    msg.extend_from_slice(recipient.public_key().as_bytes());
    msg.extend_from_slice(&timestamp.to_le_bytes());
    msg.extend_from_slice(&seq.to_le_bytes());
    msg.extend_from_slice(text.as_bytes());
    msg
}

/// A signed direct message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    /// Who sent it.
    pub sender: PeerId,
    /// Who it is for.
    pub recipient: PeerId,
    /// Human-readable text.
    pub text: String,
    /// Unix seconds when it was sent.
    pub timestamp: u64,
    /// The sender's per-sender sequence number. Orders this sender's
    /// messages (connections do not guarantee arrival order).
    pub seq: u64,
    /// The sender's signature over sender, recipient, timestamp, seq, text.
    pub signature: Ed25519Signature,
}

impl Message {
    /// Create and sign a message from `sender` to `recipient`.
    /// `seq` is the sender's per-sender sequence number (see the
    /// module docs); the caller assigns it from a monotonic counter.
    pub fn new(sender: &Identity, recipient: &PeerId, text: &str, seq: u64) -> Self {
        let timestamp = now_unix_secs();
        let msg = message_sign_bytes(sender.peer_id(), recipient, timestamp, seq, text);
        let signature = sender.sign(&msg);
        Self {
            sender: sender.peer_id().clone(),
            recipient: recipient.clone(),
            text: text.to_string(),
            timestamp,
            seq,
            signature,
        }
    }

    /// True if the signature verifies against the sender for this exact
    /// recipient, timestamp, and text. Any tampering fails this check.
    pub fn verify(&self) -> bool {
        let msg = message_sign_bytes(
            &self.sender,
            &self.recipient,
            self.timestamp,
            self.seq,
            &self.text,
        );
        self.sender.verify(&msg, &self.signature).is_ok()
    }

    /// The message's id: the SHA-256 of its exact encoding. Two nodes
    /// holding the same message compute the same id. Receipts name
    /// messages by this id.
    pub fn id(&self) -> [u8; 32] {
        sha256(&self.encode())
    }

    /// Encode to the explicit binary wire format.
    pub fn encode(&self) -> Vec<u8> {
        let sender_name = self.sender.display_name().as_bytes();
        let recipient_name = self.recipient.display_name().as_bytes();
        let text_bytes = self.text.as_bytes();
        let mut out = Vec::with_capacity(
            6 + 32
                + 2
                + sender_name.len()
                + 32
                + 2
                + recipient_name.len()
                + 8
                + 8
                + 4
                + text_bytes.len()
                + 64,
        );
        out.extend_from_slice(MESSAGE_MAGIC);
        out.extend_from_slice(self.sender.public_key().as_bytes());
        out.extend_from_slice(&(sender_name.len() as u16).to_le_bytes());
        out.extend_from_slice(sender_name);
        out.extend_from_slice(self.recipient.public_key().as_bytes());
        out.extend_from_slice(&(recipient_name.len() as u16).to_le_bytes());
        out.extend_from_slice(recipient_name);
        out.extend_from_slice(&self.timestamp.to_le_bytes());
        out.extend_from_slice(&self.seq.to_le_bytes());
        out.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(text_bytes);
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a message produced by [`Message::encode`].
    /// Rejects anything that is not exactly that format. Decoding does
    /// not verify the signature — callers must call [`Message::verify`]
    /// (or use [`TalkNode::receive_raw`], which verifies for you).
    pub fn decode(bytes: &[u8]) -> Result<Self, TalkError> {
        if bytes.len() < 6 + 32 + 2 + 32 + 2 + 8 + 8 + 4 + 64 || &bytes[..6] != MESSAGE_MAGIC {
            return Err(TalkError::InvalidMessage(
                "missing KTALK1 magic".to_string(),
            ));
        }
        let mut off = 6;
        let mut sender_key = [0u8; 32];
        sender_key.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let sender_key = kernel_crypto::Ed25519PublicKey::from_bytes(&sender_key)
            .map_err(|e| TalkError::InvalidMessage(e.to_string()))?;
        let sender_name_len =
            u16::from_le_bytes(bytes[off..off + 2].try_into().expect("2-byte slice")) as usize;
        off += 2;
        if bytes.len() < off + sender_name_len + 32 + 2 + 8 + 8 + 4 + 64 {
            return Err(TalkError::InvalidMessage(
                "message truncates the sender name".to_string(),
            ));
        }
        let sender_name = String::from_utf8(bytes[off..off + sender_name_len].to_vec())
            .map_err(|_| TalkError::InvalidMessage("sender name is not UTF-8".to_string()))?;
        off += sender_name_len;
        let sender = PeerId::new(sender_key, &sender_name)?;

        let mut recipient_key = [0u8; 32];
        recipient_key.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let recipient_key = kernel_crypto::Ed25519PublicKey::from_bytes(&recipient_key)
            .map_err(|e| TalkError::InvalidMessage(e.to_string()))?;
        let recipient_name_len =
            u16::from_le_bytes(bytes[off..off + 2].try_into().expect("2-byte slice")) as usize;
        off += 2;
        if bytes.len() < off + recipient_name_len + 8 + 8 + 4 + 64 {
            return Err(TalkError::InvalidMessage(
                "message truncates the recipient name".to_string(),
            ));
        }
        let recipient_name = String::from_utf8(bytes[off..off + recipient_name_len].to_vec())
            .map_err(|_| TalkError::InvalidMessage("recipient name is not UTF-8".to_string()))?;
        off += recipient_name_len;
        let recipient = PeerId::new(recipient_key, &recipient_name)?;

        let timestamp = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let seq = u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        let text_len =
            u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4-byte slice")) as usize;
        off += 4;
        if bytes.len() != off + text_len + 64 {
            return Err(TalkError::InvalidMessage(format!(
                "message is {} bytes, expected {} for a {text_len}-byte text",
                bytes.len(),
                off + text_len + 64
            )));
        }
        let text = String::from_utf8(bytes[off..off + text_len].to_vec())
            .map_err(|_| TalkError::InvalidMessage("text is not valid UTF-8".to_string()))?;
        off += text_len;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        Ok(Self {
            sender,
            recipient,
            text,
            timestamp,
            seq,
            signature,
        })
    }
}

/// A message in the inbox, with its arrival sequence number.
/// Messages are listed ordered by (timestamp, seq): same-second
/// messages keep their arrival order.
#[derive(Clone, Debug)]
struct StoredMessage {
    message: Message,
    seq: u64,
}

/// A group message in the group inbox, with its arrival sequence
/// number.
#[derive(Clone, Debug)]
struct StoredGroupMessage {
    message: GroupMessage,
    seq: u64,
}

/// Whether a message went out on the wire or waited in the outbox.
#[derive(Clone, Debug)]
pub enum SendOutcome {
    /// The message was sent to the peer.
    Sent(Message),
    /// The peer was unreachable, so the message was queued in the
    /// outbox instead (see [`TalkNode::peer_online`]).
    Queued(Message),
}

/// One outcome per member a group message was sent to: the member
/// and whether their send succeeded.
pub type GroupSendOutcomes = Vec<(PeerId, Result<(), TalkError>)>;

/// Apply a signed membership change to a group book. If the group
/// is not known yet, the change is held in `pending` and applied when
/// the group's record arrives (membership changes can arrive before
/// the record over separate connections, in either order).
fn apply_membership_change_to(
    groups: &Mutex<HashMap<String, Group>>,
    pending: &Mutex<Vec<MembershipChange>>,
    change: &MembershipChange,
) -> Result<(), TalkError> {
    let mut groups_guard = groups.lock().unwrap();
    match groups_guard.get_mut(&change.group_id) {
        Some(group) => group.apply_change(change),
        None => {
            pending.lock().unwrap().push(change.clone());
            Ok(())
        }
    }
}

/// Register a group record in a group book, then apply any membership
/// changes for it that arrived before the record. A record that
/// conflicts with a known group is refused.
fn register_group_to(
    groups: &Mutex<HashMap<String, Group>>,
    pending: &Mutex<Vec<MembershipChange>>,
    group: Group,
) -> Result<(), TalkError> {
    if !group.verify() {
        return Err(TalkError::BadSignature);
    }
    let id = group.id.clone();
    {
        let mut groups_guard = groups.lock().unwrap();
        match groups_guard.get(&id) {
            Some(existing) if *existing != group => {
                return Err(TalkError::InvalidGroup(format!(
                    "conflicting group record for {id}"
                )));
            }
            Some(_) => return Ok(()),
            None => {
                groups_guard.insert(id.clone(), group);
            }
        }
    }
    // Apply, in arrival order, any membership changes for this group
    // that arrived before the record.
    let mut pending_guard = pending.lock().unwrap();
    let mut remaining = Vec::new();
    for change in pending_guard.drain(..) {
        if change.group_id == id {
            let mut groups_guard = groups.lock().unwrap();
            if let Some(g) = groups_guard.get_mut(&id) {
                let _ = g.apply_change(&change);
            }
        } else {
            remaining.push(change);
        }
    }
    *pending_guard = remaining;
    Ok(())
}

/// A messaging node: an identity, a TCP listener for incoming messages,
/// and an inbox. The server runs on its own thread; [`TalkNode::stop`]
/// shuts it down.
///
/// The node also holds the group book, the persistent history log, the
/// delivery receipts, and the offline outbox. Incoming frames are
/// dispatched by magic: `KTALK1` direct messages, `KTALKG` group
/// messages, `KMEMB1` membership changes, `KGROUP1` group records.
pub struct TalkNode {
    identity: Identity,
    inbox: Arc<Mutex<Vec<StoredMessage>>>,
    seq: Arc<AtomicU64>,
    /// Per-sender sequence counter: every message this node sends gets
    /// the next number, so recipients can order them.
    send_seq: AtomicU64,
    addr: SocketAddr,
    stop_flag: Arc<AtomicBool>,
    server: Option<JoinHandle<()>>,
    history: Arc<Mutex<History>>,
    groups: Arc<Mutex<HashMap<String, Group>>>,
    /// Membership changes for groups not yet known. Applied when the
    /// group's record arrives (see [`apply_membership_change_to`]).
    pending_changes: Arc<Mutex<Vec<MembershipChange>>>,
    group_inbox: Arc<Mutex<Vec<StoredGroupMessage>>>,
    group_seq: Arc<AtomicU64>,
    receipts: Arc<Mutex<Vec<Receipt>>>,
    outbox: Arc<Mutex<Outbox>>,
}

impl TalkNode {
    /// Start a messaging node listening on loopback (ephemeral port).
    pub fn spawn(display_name: &str) -> Result<Self, TalkError> {
        let identity = Identity::generate(display_name)?;
        let peer_id = identity.peer_id().clone();
        let inbox: Arc<Mutex<Vec<StoredMessage>>> = Arc::new(Mutex::new(Vec::new()));
        let seq = Arc::new(AtomicU64::new(0));
        let listener = Listener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        listener.set_nonblocking(true)?;
        let stop_flag = Arc::new(AtomicBool::new(false));
        let history = Arc::new(Mutex::new(History::new(peer_id.clone())));
        let groups: Arc<Mutex<HashMap<String, Group>>> = Arc::new(Mutex::new(HashMap::new()));
        let pending_changes: Arc<Mutex<Vec<MembershipChange>>> = Arc::new(Mutex::new(Vec::new()));
        let group_inbox: Arc<Mutex<Vec<StoredGroupMessage>>> = Arc::new(Mutex::new(Vec::new()));
        let group_seq = Arc::new(AtomicU64::new(0));
        let receipts: Arc<Mutex<Vec<Receipt>>> = Arc::new(Mutex::new(Vec::new()));
        let outbox: Arc<Mutex<Outbox>> = Arc::new(Mutex::new(Outbox::new()));

        let server_stop = Arc::clone(&stop_flag);
        let shared = Arc::new(ServerShared {
            inbox: Arc::clone(&inbox),
            seq: Arc::clone(&seq),
            history: Arc::clone(&history),
            groups: Arc::clone(&groups),
            pending: Arc::clone(&pending_changes),
            group_inbox: Arc::clone(&group_inbox),
            group_seq: Arc::clone(&group_seq),
        });
        let server = thread::spawn(move || {
            serve(listener, peer_id, server_stop, shared);
        });

        Ok(Self {
            identity,
            inbox,
            seq,
            send_seq: AtomicU64::new(0),
            addr,
            stop_flag,
            server: Some(server),
            history,
            groups,
            pending_changes,
            group_inbox,
            group_seq,
            receipts,
            outbox,
        })
    }

    /// The loopback address this node listens on.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// This node's public peer id.
    pub fn peer_id(&self) -> &PeerId {
        self.identity.peer_id()
    }

    /// Send a message to a peer at `to_addr`. Returns the message that
    /// was sent. Fails cleanly if the peer is unreachable. The message
    /// is appended to this node's history.
    pub fn send_message(
        &self,
        to_addr: SocketAddr,
        recipient: &PeerId,
        text: &str,
    ) -> Result<Message, TalkError> {
        let seq = self.send_seq.fetch_add(1, Ordering::SeqCst);
        let message = Message::new(&self.identity, recipient, text, seq);
        let mut conn = Connection::connect(to_addr)?;
        conn.set_read_timeout(Some(Duration::from_secs(5)))?;
        conn.send(&message.encode())?;
        conn.close();
        self.history.lock().unwrap().append_direct(message.clone());
        Ok(message)
    }

    /// Send a message to a peer, or queue it if the peer is
    /// unreachable. Returns [`SendOutcome::Sent`] when the message
    /// went out and [`SendOutcome::Queued`] when it is waiting in the
    /// outbox. Queuing past the per-peer cap is an error.
    pub fn send_message_or_queue(
        &self,
        to_addr: SocketAddr,
        recipient: &PeerId,
        text: &str,
    ) -> Result<SendOutcome, TalkError> {
        match self.send_message(to_addr, recipient, text) {
            Ok(message) => Ok(SendOutcome::Sent(message)),
            Err(TalkError::Transport(_)) => {
                let message = self.queue_message(recipient, text)?;
                Ok(SendOutcome::Queued(message))
            }
            Err(e) => Err(e),
        }
    }

    /// Send an arbitrary frame to a peer at `to_addr` (a group record,
    /// a membership change, or any other frame). Used to share group
    /// state over the network.
    pub fn send_raw(&self, to_addr: SocketAddr, bytes: &[u8]) -> Result<(), TalkError> {
        let mut conn = Connection::connect(to_addr)?;
        conn.set_read_timeout(Some(Duration::from_secs(5)))?;
        conn.send(bytes)?;
        conn.close();
        Ok(())
    }

    /// Accept an encoded message (from the network or from a test):
    /// decode it, verify the signature, check it is addressed to this
    /// node, and store it in the inbox and the history. Anything that
    /// fails any check is rejected and never stored.
    pub fn receive_raw(&self, bytes: &[u8]) -> Result<(), TalkError> {
        let message = Message::decode(bytes)?;
        if !message.verify() {
            return Err(TalkError::BadSignature);
        }
        if message.recipient != *self.identity.peer_id() {
            return Err(TalkError::WrongRecipient);
        }
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        self.inbox.lock().unwrap().push(StoredMessage {
            message: message.clone(),
            seq,
        });
        self.history.lock().unwrap().append_direct(message);
        Ok(())
    }

    /// Messages from `peer`, in (timestamp, arrival) order. Every
    /// message is verified again on read: a message whose signature
    /// fails verification is skipped, never returned.
    pub fn messages_for(&self, peer: &PeerId) -> Vec<Message> {
        // Clone the matching messages out from under the lock, then
        // sort the owned copies: holding references across the lock
        // would borrow a guard that is dropped at the statement end.
        let mut stored: Vec<StoredMessage> = self
            .inbox
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s.message.sender == *peer && s.message.verify())
            .cloned()
            .collect();
        // Order by the sender's sequence (the send order), with the
        // timestamp first for stability and the arrival sequence as a
        // final tie-break for messages received out of band.
        stored.sort_by(|a, b| {
            (a.message.timestamp, a.message.seq, a.seq).cmp(&(
                b.message.timestamp,
                b.message.seq,
                b.seq,
            ))
        });
        stored.into_iter().map(|s| s.message).collect()
    }

    /// Total messages in the inbox (from all peers).
    pub fn inbox_len(&self) -> usize {
        self.inbox.lock().unwrap().len()
    }

    // -- Groups -------------------------------------------------------

    /// Create a group. This node is the creator (and first member).
    /// The group is registered locally; share its encoding with
    /// future members so they can join.
    pub fn create_group(&self, id: &str, name: &str) -> Result<Group, TalkError> {
        let group = Group::create(&self.identity, id, name)?;
        self.groups
            .lock()
            .unwrap()
            .insert(group.id.clone(), group.clone());
        Ok(group)
    }

    /// Groups this node knows about.
    pub fn groups(&self) -> Vec<Group> {
        self.groups.lock().unwrap().values().cloned().collect()
    }

    /// The local copy of a group, if this node knows it.
    pub fn group(&self, group_id: &str) -> Option<Group> {
        self.groups.lock().unwrap().get(group_id).cloned()
    }

    /// Accept an encoded group record (from the network or from a
    /// test): decode it, verify the creator's signature, and register
    /// it locally. A record for a group already known is ignored
    /// unless it is byte-identical to the local copy. Membership
    /// changes for this group that arrived before the record are
    /// applied now.
    pub fn receive_group_raw(&self, bytes: &[u8]) -> Result<(), TalkError> {
        let group = Group::decode(bytes)?;
        register_group_to(&self.groups, &self.pending_changes, group)
    }

    /// Add a member to a group. Only the creator may do this: the
    /// change is signed by this node and applied locally. Returns the
    /// signed change so the caller can send it to the other members.
    pub fn add_member(&self, group_id: &str, peer: &PeerId) -> Result<MembershipChange, TalkError> {
        let change = MembershipChange::new(&self.identity, group_id, peer, MembershipAction::Add)?;
        self.apply_membership_change(&change)?;
        Ok(change)
    }

    /// Remove a member from a group. Only the creator may do this.
    /// Returns the signed change so the caller can send it to the
    /// other members. A removed member stops receiving group messages.
    pub fn remove_member(
        &self,
        group_id: &str,
        peer: &PeerId,
    ) -> Result<MembershipChange, TalkError> {
        let change =
            MembershipChange::new(&self.identity, group_id, peer, MembershipAction::Remove)?;
        self.apply_membership_change(&change)?;
        Ok(change)
    }

    /// Apply a signed membership change to the local group copy. The
    /// change must be signed by the group's creator. If the group is
    /// not known yet, the change is held and applied when the group's
    /// record arrives.
    pub fn apply_membership_change(&self, change: &MembershipChange) -> Result<(), TalkError> {
        apply_membership_change_to(&self.groups, &self.pending_changes, change)
    }

    /// Accept an encoded membership change (from the network or from a
    /// test): decode it and apply it to the local group copy.
    pub fn receive_membership_raw(&self, bytes: &[u8]) -> Result<(), TalkError> {
        let change = MembershipChange::decode(bytes)?;
        self.apply_membership_change(&change)
    }

    /// Send a group message to every current member (except this
    /// node) at the addresses in `addrs`, and store our own copy.
    /// The caller supplies member addresses: this node does not
    /// discover peers. Returns one outcome per member that was sent
    /// to; members with no address in `addrs` are skipped. Fails if
    /// this node does not know the group or is not a member of it.
    pub fn send_group_message(
        &self,
        group_id: &str,
        addrs: &[(PeerId, SocketAddr)],
        text: &str,
    ) -> Result<GroupSendOutcomes, TalkError> {
        let group = self
            .group(group_id)
            .ok_or_else(|| TalkError::UnknownGroup(group_id.to_string()))?;
        if !group.is_member(self.identity.peer_id()) {
            return Err(TalkError::NotGroupMember);
        }
        let seq = self.send_seq.fetch_add(1, Ordering::SeqCst);
        let message = GroupMessage::new(&self.identity, group_id, text, seq)?;
        let encoded = message.encode();
        let mut outcomes = Vec::new();
        for member in group.members() {
            if member == *self.identity.peer_id() {
                continue;
            }
            let Some((_, addr)) = addrs.iter().find(|(p, _)| *p == member) else {
                continue;
            };
            outcomes.push((member, self.send_raw(*addr, &encoded)));
        }
        let gseq = self.group_seq.fetch_add(1, Ordering::SeqCst);
        self.group_inbox.lock().unwrap().push(StoredGroupMessage {
            message: message.clone(),
            seq: gseq,
        });
        self.history.lock().unwrap().append_group(message);
        Ok(outcomes)
    }

    /// Accept an encoded group message (from the network or from a
    /// test): decode it, verify the signature, check this node knows
    /// the group, the sender is a current member, and this node is a
    /// member. Anything failing any check is rejected and never
    /// stored. A removed member's messages are refused here, and a
    /// removed member refuses everyone else's.
    pub fn receive_group_message_raw(&self, bytes: &[u8]) -> Result<(), TalkError> {
        let message = GroupMessage::decode(bytes)?;
        if !message.verify() {
            return Err(TalkError::BadSignature);
        }
        let groups = self.groups.lock().unwrap();
        let group = groups
            .get(&message.group_id)
            .ok_or_else(|| TalkError::UnknownGroup(message.group_id.clone()))?;
        if !group.is_member(&message.sender) {
            return Err(TalkError::NotGroupMember);
        }
        if !group.is_member(self.identity.peer_id()) {
            return Err(TalkError::NotGroupMember);
        }
        drop(groups);
        let seq = self.group_seq.fetch_add(1, Ordering::SeqCst);
        self.group_inbox.lock().unwrap().push(StoredGroupMessage {
            message: message.clone(),
            seq,
        });
        self.history.lock().unwrap().append_group(message);
        Ok(())
    }

    /// Group messages for `group_id`, in (timestamp, sequence,
    /// arrival) order. Every message is verified again on read.
    pub fn group_messages(&self, group_id: &str) -> Vec<GroupMessage> {
        let mut stored: Vec<StoredGroupMessage> = self
            .group_inbox
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s.message.group_id == group_id && s.message.verify())
            .cloned()
            .collect();
        stored.sort_by(|a, b| {
            (a.message.timestamp, a.message.seq, a.seq).cmp(&(
                b.message.timestamp,
                b.message.seq,
                b.seq,
            ))
        });
        stored.into_iter().map(|s| s.message).collect()
    }

    // -- History and receipts ------------------------------------------

    /// History entries for one conversation, newest last. `limit`
    /// caps how many are returned (the newest `limit`); 0 means no
    /// limit.
    pub fn history_for(&self, conv: &ConversationId, limit: usize) -> Vec<HistoryEntry> {
        self.history.lock().unwrap().entries_for(conv, limit)
    }

    /// Total entries in the history log.
    pub fn history_len(&self) -> usize {
        self.history.lock().unwrap().len()
    }

    /// Serialize the whole history log to bytes.
    pub fn save_history(&self) -> Vec<u8> {
        self.history.lock().unwrap().save()
    }

    /// Replace this node's history with a log produced by
    /// [`History::save`]. The log must belong to this node.
    pub fn load_history(&self, bytes: &[u8]) -> Result<(), TalkError> {
        let loaded = History::load(bytes)?;
        if loaded.owner() != self.identity.peer_id() {
            return Err(TalkError::CorruptHistory(
                "history belongs to a different peer".to_string(),
            ));
        }
        *self.history.lock().unwrap() = loaded;
        Ok(())
    }

    /// Save the history log to a file at `path`.
    pub fn save_history_to_file(&self, path: &std::path::Path) -> Result<(), TalkError> {
        self.history.lock().unwrap().save_to_file(path)
    }

    /// Load the history log from a file at `path`, replacing this
    /// node's history.
    pub fn load_history_from_file(&self, path: &std::path::Path) -> Result<(), TalkError> {
        let loaded = History::load_from_file(path)?;
        if loaded.owner() != self.identity.peer_id() {
            return Err(TalkError::CorruptHistory(
                "history belongs to a different peer".to_string(),
            ));
        }
        *self.history.lock().unwrap() = loaded;
        Ok(())
    }

    /// Record a delivery receipt. The receipt's signature is verified,
    /// the message it names must be in this node's history, and the
    /// peer asserting it must be a recipient of that message (the
    /// direct recipient, or a current member of the group). Anything
    /// else is refused and never stored.
    pub fn record_receipt(&self, receipt: Receipt) -> Result<(), TalkError> {
        if !receipt.verify() {
            return Err(TalkError::BadReceipt(
                "receipt signature does not verify".to_string(),
            ));
        }
        let history = self.history.lock().unwrap();
        let found = history
            .find_message(&receipt.message_id)
            .ok_or(TalkError::UnknownMessage)?;
        let is_recipient = match found {
            FoundMessage::Direct(m) => receipt.by == m.recipient,
            FoundMessage::Group(m) => {
                let groups = self.groups.lock().unwrap();
                match groups.get(&m.group_id) {
                    Some(group) => group.is_member(&receipt.by),
                    None => false,
                }
            }
        };
        if !is_recipient {
            return Err(TalkError::BadReceipt(format!(
                "peer {} was never a recipient of that message",
                receipt.by.display_name()
            )));
        }
        drop(history);
        self.receipts.lock().unwrap().push(receipt.clone());
        self.history.lock().unwrap().append_receipt(receipt);
        Ok(())
    }

    /// Receipts recorded for one message id.
    pub fn receipts_for(&self, message_id: &[u8; 32]) -> Vec<Receipt> {
        self.receipts
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.message_id == *message_id)
            .cloned()
            .collect()
    }

    // -- Offline outbox -------------------------------------------------

    /// Create and queue a message for `recipient` without sending it.
    /// Use it when the recipient is offline; see [`TalkNode::peer_online`].
    pub fn queue_message(&self, recipient: &PeerId, text: &str) -> Result<Message, TalkError> {
        let seq = self.send_seq.fetch_add(1, Ordering::SeqCst);
        let message = Message::new(&self.identity, recipient, text, seq);
        self.outbox
            .lock()
            .unwrap()
            .queue(recipient, message.clone())?;
        Ok(message)
    }

    /// Signal that `peer` is back online. Returns their queued
    /// messages in order. The queue entries stay queued until the
    /// caller delivers each message and acknowledges it with
    /// [`TalkNode::acknowledge_delivery`].
    pub fn peer_online(&self, peer: &PeerId) -> Vec<Message> {
        self.outbox.lock().unwrap().peer_online(peer)
    }

    /// Acknowledge that the message with `message_id` was delivered
    /// to `peer`: its queue entry is removed. Returns true if an
    /// entry was removed.
    pub fn acknowledge_delivery(&self, peer: &PeerId, message_id: &[u8; 32]) -> bool {
        self.outbox.lock().unwrap().acknowledge(peer, message_id)
    }

    /// Messages still queued for `peer`, in order.
    pub fn queued_for(&self, peer: &PeerId) -> Vec<Message> {
        self.outbox.lock().unwrap().pending(peer)
    }

    /// Number of messages queued for `peer`.
    pub fn queued_count(&self, peer: &PeerId) -> usize {
        self.outbox.lock().unwrap().pending_count(peer)
    }

    /// Mark `peer` offline. Queued messages are kept.
    pub fn peer_offline(&self, peer: &PeerId) {
        self.outbox.lock().unwrap().mark_offline(peer);
    }

    /// Stop the server thread and shut this node down.
    pub fn stop(mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        if let Some(server) = self.server.take() {
            let _ = server.join();
        }
    }
}

/// State shared between the server thread and the node.
struct ServerShared {
    inbox: Arc<Mutex<Vec<StoredMessage>>>,
    seq: Arc<AtomicU64>,
    history: Arc<Mutex<History>>,
    groups: Arc<Mutex<HashMap<String, Group>>>,
    pending: Arc<Mutex<Vec<MembershipChange>>>,
    group_inbox: Arc<Mutex<Vec<StoredGroupMessage>>>,
    group_seq: Arc<AtomicU64>,
}

/// The server loop: accept connections and handle each on its own
/// thread until the stop flag is set.
fn serve(
    listener: Listener,
    peer_id: PeerId,
    stop_flag: Arc<AtomicBool>,
    shared: Arc<ServerShared>,
) {
    loop {
        if stop_flag.load(Ordering::SeqCst) {
            break;
        }
        match listener.accept() {
            Ok(conn) => {
                let peer_id = peer_id.clone();
                let shared = Arc::clone(&shared);
                thread::spawn(move || handle_connection(conn, peer_id, shared));
            }
            Err(TransportError::Io(ref e)) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(5));
            }
            Err(_) => break,
        }
    }
}

/// Handle one peer connection: read frames until the peer goes away,
/// dispatching each by its magic: `KTALK1` direct messages, `KTALKG`
/// group messages, `KMEMB1` membership changes, `KGROUP1` group
/// records. Anything failing verification is ignored and never
/// stored; the connection stays open for the next frame.
fn handle_connection(mut conn: Connection, peer_id: PeerId, shared: Arc<ServerShared>) {
    loop {
        let frame = match conn.recv() {
            Ok(frame) => frame,
            Err(_) => return, // peer went away; nothing more to do
        };
        if frame.starts_with(b"KTALK1") {
            let message = match Message::decode(&frame) {
                Ok(message) => message,
                Err(_) => continue, // malformed frame: ignore it, keep reading
            };
            if !message.verify() {
                continue; // bad signature: never store it
            }
            if message.recipient != peer_id {
                continue; // not for this node: never store it
            }
            let s = shared.seq.fetch_add(1, Ordering::SeqCst);
            shared.inbox.lock().unwrap().push(StoredMessage {
                message: message.clone(),
                seq: s,
            });
            shared.history.lock().unwrap().append_direct(message);
        } else if frame.starts_with(b"KTALKG") {
            let message = match GroupMessage::decode(&frame) {
                Ok(message) => message,
                Err(_) => continue,
            };
            if !message.verify() {
                continue;
            }
            let groups_guard = shared.groups.lock().unwrap();
            let Some(group) = groups_guard.get(&message.group_id) else {
                continue; // unknown group: never store it
            };
            if !group.is_member(&message.sender) {
                continue; // non-member sender: never store it
            }
            if !group.is_member(&peer_id) {
                continue; // this node is not (or is no longer) a member
            }
            drop(groups_guard);
            let s = shared.group_seq.fetch_add(1, Ordering::SeqCst);
            shared.group_inbox.lock().unwrap().push(StoredGroupMessage {
                message: message.clone(),
                seq: s,
            });
            shared.history.lock().unwrap().append_group(message);
        } else if frame.starts_with(b"KMEMB1") {
            let change = match MembershipChange::decode(&frame) {
                Ok(change) => change,
                Err(_) => continue,
            };
            let _ = apply_membership_change_to(&shared.groups, &shared.pending, &change);
        } else if frame.starts_with(b"KGROUP1") {
            let group = match Group::decode(&frame) {
                Ok(group) => group,
                Err(_) => continue,
            };
            let _ = register_group_to(&shared.groups, &shared.pending, group);
        }
        // Unknown frames are ignored; the connection stays open.
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_encode_decode_round_trip_and_verify() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let msg = Message::new(&alice, bob.peer_id(), "hello bob", 0);
        assert!(msg.verify());
        let encoded = msg.encode();
        let decoded = Message::decode(&encoded).unwrap();
        assert_eq!(decoded, msg);
        assert!(decoded.verify());
        assert!(Message::decode(b"garbage").is_err());
        assert!(Message::decode(&encoded[..encoded.len() - 1]).is_err());
    }

    #[test]
    fn two_peers_exchange_messages() {
        let alice = TalkNode::spawn("alice").unwrap();
        let bob = TalkNode::spawn("bob").unwrap();
        let alice_addr = alice.addr();
        let bob_addr = bob.addr();
        let alice_peer = alice.peer_id().clone();
        let bob_peer = bob.peer_id().clone();

        let sent_a = alice
            .send_message(bob_addr, &bob_peer, "hey bob, it is alice")
            .unwrap();
        let sent_b = bob
            .send_message(alice_addr, &alice_peer, "hey alice, bob here")
            .unwrap();

        // Give the server threads a moment to receive and store.
        assert!(wait_for(
            || bob.messages_for(&alice_peer).len() == 1 && alice.messages_for(&bob_peer).len() == 1,
            2000
        ));

        let bob_inbox = bob.messages_for(&alice_peer);
        assert_eq!(bob_inbox.len(), 1);
        assert_eq!(bob_inbox[0].text, "hey bob, it is alice");
        assert_eq!(bob_inbox[0].sender, alice_peer);
        assert_eq!(bob_inbox[0].recipient, bob_peer);
        assert!(bob_inbox[0].verify());
        assert_eq!(bob_inbox[0].timestamp, sent_a.timestamp);

        let alice_inbox = alice.messages_for(&bob_peer);
        assert_eq!(alice_inbox.len(), 1);
        assert_eq!(alice_inbox[0].text, "hey alice, bob here");
        assert_eq!(alice_inbox[0].timestamp, sent_b.timestamp);
    }

    #[test]
    fn signature_verified_on_receipt() {
        let alice = TalkNode::spawn("alice").unwrap();
        let bob = TalkNode::spawn("bob").unwrap();
        let bob_peer = bob.peer_id().clone();
        // Send through the real network path.
        alice
            .send_message(bob.addr(), &bob_peer, "signed and sealed")
            .unwrap();
        assert!(wait_for(
            || bob.messages_for(alice.peer_id()).len() == 1,
            2000
        ));
        let inbox = bob.messages_for(alice.peer_id());
        assert_eq!(inbox.len(), 1);
        assert!(inbox[0].verify());
        assert_eq!(inbox[0].sender, *alice.peer_id());
        // An impostor with the same display name cannot forge Alice's
        // signature: their key signs the same words, but the signature
        // does not verify against the real Alice's key.
        let impostor = Identity::generate("alice").unwrap();
        assert_ne!(impostor.peer_id(), alice.peer_id());
        let forged_sig = impostor.sign(&message_sign_bytes(
            alice.peer_id(),
            &bob_peer,
            inbox[0].timestamp,
            inbox[0].seq,
            "signed and sealed",
        ));
        assert!(alice
            .peer_id()
            .verify(
                &message_sign_bytes(
                    alice.peer_id(),
                    &bob_peer,
                    inbox[0].timestamp,
                    inbox[0].seq,
                    "signed and sealed",
                ),
                &forged_sig,
            )
            .is_err());
    }

    #[test]
    fn tampered_message_is_rejected() {
        let alice = Identity::generate("alice").unwrap();
        let bob = TalkNode::spawn("bob").unwrap();
        let bob_peer = bob.peer_id().clone();
        let msg = Message::new(&alice, &bob_peer, "original text", 0);
        let mut encoded = msg.encode();
        // Tamper with the text itself: replace one ASCII letter with
        // another, so the text stays valid UTF-8 (decode succeeds) but
        // the signature no longer matches the content.
        let text = b"original text";
        let pos = encoded
            .windows(text.len())
            .position(|w| w == text)
            .expect("text must appear verbatim in the encoding");
        encoded[pos] = b'X'; // 'o' -> 'X': "Xriginal text"
        assert_eq!(bob.receive_raw(&encoded), Err(TalkError::BadSignature));
        assert_eq!(bob.inbox_len(), 0);

        // Tamper with the signature itself.
        let mut encoded2 = msg.encode();
        let last = encoded2.len() - 1;
        encoded2[last] ^= 0xff;
        assert_eq!(bob.receive_raw(&encoded2), Err(TalkError::BadSignature));
        assert_eq!(bob.inbox_len(), 0);

        // A garbage frame is rejected as invalid, not stored.
        assert!(matches!(
            bob.receive_raw(b"not a message"),
            Err(TalkError::InvalidMessage(_))
        ));
        assert_eq!(bob.inbox_len(), 0);
    }

    #[test]
    fn message_to_wrong_recipient_is_rejected() {
        let alice = Identity::generate("alice").unwrap();
        let bob = TalkNode::spawn("bob").unwrap();
        let carol = Identity::generate("carol").unwrap();
        // Alice writes to Carol, but the bytes arrive at Bob.
        let msg = Message::new(&alice, carol.peer_id(), "for carol only", 0);
        assert_eq!(
            bob.receive_raw(&msg.encode()),
            Err(TalkError::WrongRecipient)
        );
        assert_eq!(bob.inbox_len(), 0);
    }

    #[test]
    fn messages_listed_in_order() {
        let alice = TalkNode::spawn("alice").unwrap();
        let bob = TalkNode::spawn("bob").unwrap();
        let bob_peer = bob.peer_id().clone();
        alice.send_message(bob.addr(), &bob_peer, "first").unwrap();
        alice.send_message(bob.addr(), &bob_peer, "second").unwrap();
        alice.send_message(bob.addr(), &bob_peer, "third").unwrap();
        assert!(wait_for(
            || bob.messages_for(alice.peer_id()).len() == 3,
            2000
        ));
        let inbox = bob.messages_for(alice.peer_id());
        assert_eq!(inbox.len(), 3);
        let texts: Vec<&str> = inbox.iter().map(|m| m.text.as_str()).collect();
        assert_eq!(texts, vec!["first", "second", "third"]);
    }

    #[test]
    fn send_to_unreachable_peer_is_a_clean_error() {
        let alice = TalkNode::spawn("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let dead: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let result = alice.send_message(dead, bob.peer_id(), "hello?");
        assert!(matches!(result, Err(TalkError::Transport(_))));
    }

    #[test]
    fn messages_from_other_peers_are_separate() {
        let alice = TalkNode::spawn("alice").unwrap();
        let bob = TalkNode::spawn("bob").unwrap();
        let carol = TalkNode::spawn("carol").unwrap();
        let bob_peer = bob.peer_id().clone();
        let carol_peer = carol.peer_id().clone();
        alice.send_message(bob.addr(), &bob_peer, "to bob").unwrap();
        alice
            .send_message(carol.addr(), &carol_peer, "to carol")
            .unwrap();
        assert!(wait_for(
            || bob.messages_for(alice.peer_id()).len() == 1
                && carol.messages_for(alice.peer_id()).len() == 1,
            2000
        ));
        assert_eq!(bob.messages_for(alice.peer_id()).len(), 1);
        assert_eq!(carol.messages_for(alice.peer_id()).len(), 1);
        // Bob has no messages from Carol.
        assert!(bob.messages_for(&carol_peer).is_empty());
        // Carol has no messages from Bob.
        assert!(carol.messages_for(&bob_peer).is_empty());
    }

    /// Poll `cond` every 10ms for up to `timeout_ms`; true if it ever
    /// held. Network delivery is asynchronous, and a fixed sleep is
    /// flaky under parallel test load.
    fn wait_for<F: Fn() -> bool>(cond: F, timeout_ms: u64) -> bool {
        let deadline = std::time::Instant::now() + Duration::from_millis(timeout_ms);
        while std::time::Instant::now() < deadline {
            if cond() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        cond()
    }

    /// Share a group record and a membership change with a peer over
    /// the network path, then wait until the peer's group has
    /// `expected_members` members.
    fn share_group_with(
        alice: &TalkNode,
        bob: &TalkNode,
        group: &Group,
        change: &MembershipChange,
        expected_members: usize,
    ) {
        alice.send_raw(bob.addr(), &group.encode()).unwrap();
        alice.send_raw(bob.addr(), &change.encode()).unwrap();
        assert!(wait_for(
            || bob
                .group(&group.id)
                .is_some_and(|g| g.member_count() == expected_members),
            2000
        ));
    }

    #[test]
    fn group_message_reaches_members_only() {
        let alice = TalkNode::spawn("alice").unwrap();
        let bob = TalkNode::spawn("bob").unwrap();
        let carol = TalkNode::spawn("carol").unwrap();
        let alice_peer = alice.peer_id().clone();
        let bob_peer = bob.peer_id().clone();
        let carol_peer = carol.peer_id().clone();

        let group = alice
            .create_group("korn-builders", "Korn Builders")
            .unwrap();
        let add_bob = alice.add_member(&group.id, &bob_peer).unwrap();
        // Carol is never added.
        share_group_with(&alice, &bob, &group, &add_bob, 2);

        assert!(bob.group(&group.id).is_some());
        assert!(bob.group(&group.id).unwrap().is_member(&bob_peer));
        assert!(carol.group(&group.id).is_none());

        let addrs = vec![
            (bob_peer.clone(), bob.addr()),
            (carol_peer.clone(), carol.addr()),
        ];
        let outcomes = alice
            .send_group_message(&group.id, &addrs, "hello builders")
            .unwrap();
        // Sent to Bob only — Carol is not a member, so she is not
        // even attempted.
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].0, bob_peer);
        assert!(outcomes[0].1.is_ok());
        assert!(wait_for(|| bob.group_messages(&group.id).len() == 1, 2000));

        let bob_msgs = bob.group_messages(&group.id);
        assert_eq!(bob_msgs.len(), 1);
        assert_eq!(bob_msgs[0].text, "hello builders");
        assert_eq!(bob_msgs[0].sender, alice_peer);
        assert!(bob_msgs[0].verify());
        // Carol received nothing.
        assert!(carol.group_messages(&group.id).is_empty());
        // Alice keeps her own copy.
        assert_eq!(alice.group_messages(&group.id).len(), 1);
    }

    #[test]
    fn non_member_send_is_refused() {
        let alice = TalkNode::spawn("alice").unwrap();
        let bob = TalkNode::spawn("bob").unwrap();
        let carol = TalkNode::spawn("carol").unwrap();
        let alice_peer = alice.peer_id().clone();
        let bob_peer = bob.peer_id().clone();
        let carol_peer = carol.peer_id().clone();

        let group = alice
            .create_group("korn-builders", "Korn Builders")
            .unwrap();
        let add_bob = alice.add_member(&group.id, &bob_peer).unwrap();
        share_group_with(&alice, &bob, &group, &add_bob, 2);

        // Carol is not a member: a group message she crafts is
        // refused by Alice, even though the signature is valid.
        let carol_identity = Identity::generate("carol").unwrap();
        // Carol's own node does not know the group at all.
        assert_eq!(
            carol.send_group_message(&group.id, &[], "gatecrash"),
            Err(TalkError::UnknownGroup(group.id.clone()))
        );
        let forged = GroupMessage::new(&carol_identity, &group.id, "gatecrash", 0).unwrap();
        assert_eq!(
            alice.receive_group_message_raw(&forged.encode()),
            Err(TalkError::NotGroupMember)
        );
        assert!(alice.group_messages(&group.id).is_empty());
        // An impostor with Bob's display name but a different key is
        // refused too: membership is by key, not by name.
        let impostor_bob = Identity::generate("bob").unwrap();
        let legit_looking = GroupMessage::new(&impostor_bob, &group.id, "hi all", 0).unwrap();
        assert_eq!(
            alice.receive_group_message_raw(&legit_looking.encode()),
            Err(TalkError::NotGroupMember)
        );
        // A forged membership change (signed by Bob, not the creator)
        // is refused by Alice.
        let bad_change =
            MembershipChange::new(&impostor_bob, &group.id, &carol_peer, MembershipAction::Add)
                .unwrap();
        assert!(alice.receive_membership_raw(&bad_change.encode()).is_err());
        assert!(!alice.group(&group.id).unwrap().is_member(&carol_peer));
        // The real Bob, a member, sends a group message through his
        // node and Alice receives it.
        bob.send_group_message(&group.id, &[(alice_peer.clone(), alice.addr())], "hi all")
            .unwrap();
        assert!(wait_for(
            || alice.group_messages(&group.id).len() == 1,
            2000
        ));
        let alice_msgs = alice.group_messages(&group.id);
        assert_eq!(alice_msgs.len(), 1);
        assert_eq!(alice_msgs[0].text, "hi all");
        assert_eq!(alice_msgs[0].sender, bob_peer);
    }

    #[test]
    fn removed_member_stops_receiving() {
        let alice = TalkNode::spawn("alice").unwrap();
        let bob = TalkNode::spawn("bob").unwrap();
        let bob_peer = bob.peer_id().clone();

        let group = alice
            .create_group("korn-builders", "Korn Builders")
            .unwrap();
        let add_bob = alice.add_member(&group.id, &bob_peer).unwrap();
        share_group_with(&alice, &bob, &group, &add_bob, 2);

        let addrs = vec![(bob_peer.clone(), bob.addr())];
        alice
            .send_group_message(&group.id, &addrs, "before removal")
            .unwrap();
        assert!(wait_for(|| bob.group_messages(&group.id).len() == 1, 2000));
        assert_eq!(bob.group_messages(&group.id).len(), 1);

        // Alice removes Bob and tells him over the network.
        let remove_bob = alice.remove_member(&group.id, &bob_peer).unwrap();
        alice.send_raw(bob.addr(), &remove_bob.encode()).unwrap();
        assert!(wait_for(
            || bob
                .group(&group.id)
                .is_some_and(|g| !g.is_member(&bob_peer)),
            2000
        ));
        assert!(!bob.group(&group.id).unwrap().is_member(&bob_peer));

        // Alice's next group message no longer goes to Bob at all.
        let outcomes = alice
            .send_group_message(&group.id, &addrs, "after removal")
            .unwrap();
        assert!(outcomes.is_empty());
        assert_eq!(bob.group_messages(&group.id).len(), 1);

        // And if a message signed by Alice's real key is handed to
        // Bob directly, his node refuses it: he is no longer a
        // member. (Tests can reach the node's identity: the tests
        // module is a child of the module that defines TalkNode.)
        let seq = alice.send_seq.load(Ordering::SeqCst);
        let direct = GroupMessage::new(&alice.identity, &group.id, "sneaky", seq).unwrap();
        assert_eq!(
            bob.receive_group_message_raw(&direct.encode()),
            Err(TalkError::NotGroupMember)
        );
        assert_eq!(bob.group_messages(&group.id).len(), 1);
    }

    #[test]
    fn node_history_survives_save_load_round_trip() {
        let alice = TalkNode::spawn("alice").unwrap();
        let bob = TalkNode::spawn("bob").unwrap();
        let bob_peer = bob.peer_id().clone();
        alice.send_message(bob.addr(), &bob_peer, "one").unwrap();
        alice.send_message(bob.addr(), &bob_peer, "two").unwrap();
        assert!(wait_for(|| bob.history_len() == 2, 2000));
        assert_eq!(alice.history_len(), 2);
        assert_eq!(bob.history_len(), 2);

        // Save Bob's history, corrupt a copy, then load the good one
        // back into the same node.
        let bytes = bob.save_history();
        let mut corrupted = bytes.clone();
        corrupted[40] ^= 0xff;
        assert!(matches!(
            bob.load_history(&corrupted),
            Err(TalkError::CorruptHistory(_))
        ));
        // A failed load leaves the history untouched.
        assert_eq!(bob.history_len(), 2);
        assert!(bob.load_history(&bytes).is_ok());
        assert_eq!(bob.history_len(), 2);
        let conv = ConversationId::Direct(alice.peer_id().clone());
        let entries = bob.history_for(&conv, 1);
        assert_eq!(entries.len(), 1);
        assert!(matches!(&entries[0], HistoryEntry::Direct(m) if m.text == "two"));
        // File round-trip.
        let dir = std::env::temp_dir().join("k-talk-node-history-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("history.bin");
        bob.save_history_to_file(&path).unwrap();
        assert!(bob.load_history_from_file(&path).is_ok());
        assert_eq!(bob.history_len(), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn receipts_recorded_and_bad_receipts_refused() {
        let alice = TalkNode::spawn("alice").unwrap();
        let bob = TalkNode::spawn("bob").unwrap();
        let bob_peer = bob.peer_id().clone();

        let sent = alice
            .send_message(bob.addr(), &bob_peer, "receipt me")
            .unwrap();
        assert!(wait_for(|| bob.history_len() == 1, 2000));
        assert_eq!(bob.history_len(), 1);
        let id = sent.id();

        // Genuine receipt: Bob's node key signs Delivered. (Tests can
        // reach the node's identity: the tests module is a child of
        // the module that defines TalkNode.)
        let receipt = Receipt::new(&bob.identity, id, ReceiptStatus::Delivered, 1_700_000_000);
        assert!(receipt.verify());
        assert!(bob.record_receipt(receipt).is_ok());
        assert_eq!(bob.receipts_for(&id).len(), 1);
        assert_eq!(bob.receipts_for(&id)[0].status, ReceiptStatus::Delivered);
        // History now holds the message plus the receipt.
        let conv = ConversationId::Direct(alice.peer_id().clone());
        assert_eq!(bob.history_for(&conv, 0).len(), 2);

        // Unknown message id.
        let unknown = Receipt::new(&bob.identity, [3u8; 32], ReceiptStatus::Read, 1_700_000_001);
        assert_eq!(bob.record_receipt(unknown), Err(TalkError::UnknownMessage));

        // A receipt from Carol, who was never a recipient.
        let carol = Identity::generate("carol").unwrap();
        let carol_receipt = Receipt::new(&carol, id, ReceiptStatus::Read, 1_700_000_002);
        assert!(matches!(
            bob.record_receipt(carol_receipt),
            Err(TalkError::BadReceipt(_))
        ));
        assert_eq!(bob.receipts_for(&id).len(), 1);

        // A tampered receipt (status changed after signing) fails
        // verification.
        let mut tampered = Receipt::new(&bob.identity, id, ReceiptStatus::Read, 1_700_000_003);
        tampered.status = ReceiptStatus::Delivered;
        assert!(matches!(
            bob.record_receipt(tampered),
            Err(TalkError::BadReceipt(_))
        ));
        assert_eq!(bob.receipts_for(&id).len(), 1);
    }

    #[test]
    fn receipt_from_real_recipient_is_recorded() {
        // Receipts are signed by the asserting peer's identity, so
        // the genuine path needs the node's own key. Build the
        // receipt directly from the identities used to create the
        // message: Alice sends to Bob, Bob's node holds the message,
        // and the receipt is signed by a Bob identity created in
        // this test — but Bob's node key is generated inside spawn,
        // so instead verify the pure path: construct the message and
        // receipt from the same test identities and check them
        // against a History directly.
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let msg = Message::new(&alice, bob.peer_id(), "hello", 0);
        let msg_ts = msg.timestamp;
        let mut history = History::new(bob.peer_id().clone());
        history.append_direct(msg.clone());
        let receipt = Receipt::new(&bob, msg.id(), ReceiptStatus::Read, msg_ts + 1);
        assert!(receipt.verify());
        assert!(history.find_message(&msg.id()).is_some());
        history.append_receipt(receipt.clone());
        let conv = ConversationId::Direct(alice.peer_id().clone());
        let entries = history.entries_for(&conv, 0);
        assert_eq!(entries.len(), 2);
        assert!(matches!(entries[1], HistoryEntry::Receipt(_)));
    }

    #[test]
    fn offline_queue_delivers_in_order_on_reconnect() {
        let alice = TalkNode::spawn("alice").unwrap();
        let bob = TalkNode::spawn("bob").unwrap();
        let bob_peer = bob.peer_id().clone();

        // Bob is "offline": queue three messages without sending.
        let m1 = alice.queue_message(&bob_peer, "first").unwrap();
        let m2 = alice.queue_message(&bob_peer, "second").unwrap();
        let m3 = alice.queue_message(&bob_peer, "third").unwrap();
        assert_eq!(alice.queued_count(&bob_peer), 3);
        assert_eq!(bob.inbox_len(), 0);

        // Bob comes back: the queue is offered in order, but not
        // cleared.
        let offered = alice.peer_online(&bob_peer);
        assert_eq!(offered.len(), 3);
        assert_eq!(offered[0].text, "first");
        assert_eq!(offered[1].text, "second");
        assert_eq!(offered[2].text, "third");
        assert_eq!(alice.queued_count(&bob_peer), 3);

        // Deliver each one and acknowledge it; the queue drains.
        for m in offered {
            alice.send_message(bob.addr(), &bob_peer, &m.text).unwrap();
            assert!(alice.acknowledge_delivery(&bob_peer, &m.id()));
        }
        assert_eq!(alice.queued_count(&bob_peer), 0);
        assert!(wait_for(
            || bob.messages_for(alice.peer_id()).len() == 3,
            2000
        ));
        let inbox = bob.messages_for(alice.peer_id());
        assert_eq!(inbox.len(), 3);
        let texts: Vec<&str> = inbox.iter().map(|m| m.text.as_str()).collect();
        assert_eq!(texts, vec!["first", "second", "third"]);
        let _ = (m1, m2, m3);
    }

    #[test]
    fn send_or_queue_falls_back_when_peer_is_down() {
        let alice = TalkNode::spawn("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let dead: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let outcome = alice
            .send_message_or_queue(dead, bob.peer_id(), "queued not lost")
            .unwrap();
        let queued = match outcome {
            SendOutcome::Queued(m) => m,
            SendOutcome::Sent(_) => panic!("expected the message to be queued"),
        };
        assert_eq!(queued.text, "queued not lost");
        assert_eq!(alice.queued_count(bob.peer_id()), 1);
        // The queue entry survives until acknowledged.
        let offered = alice.peer_online(bob.peer_id());
        assert_eq!(offered.len(), 1);
        assert_eq!(alice.queued_count(bob.peer_id()), 1);
        assert!(alice.acknowledge_delivery(bob.peer_id(), &queued.id()));
        assert_eq!(alice.queued_count(bob.peer_id()), 0);
    }

    #[test]
    fn node_queue_cap_refuses_past_the_cap() {
        let alice = TalkNode::spawn("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        for i in 0..MAX_QUEUED_PER_PEER {
            alice
                .queue_message(bob.peer_id(), &format!("msg {i}"))
                .unwrap();
        }
        assert_eq!(alice.queued_count(bob.peer_id()), MAX_QUEUED_PER_PEER);
        assert!(matches!(
            alice.queue_message(bob.peer_id(), "one too many"),
            Err(TalkError::OutboxFull { .. })
        ));
    }
}
