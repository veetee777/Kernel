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

use std::fmt;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use kernel_crypto::Ed25519Signature;
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
}

impl fmt::Display for TalkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TalkError::Transport(e) => write!(f, "transport error: {e}"),
            TalkError::Identity(e) => write!(f, "identity error: {e}"),
            TalkError::InvalidMessage(msg) => write!(f, "invalid message: {msg}"),
            TalkError::BadSignature => write!(f, "message signature is invalid"),
            TalkError::WrongRecipient => write!(f, "message is addressed to another peer"),
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
    let mut msg = Vec::with_capacity(
        MESSAGE_SIGN_DOMAIN.len() + 32 + 32 + 8 + 8 + text.len(),
    );
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

    /// Encode to the explicit binary wire format.
    pub fn encode(&self) -> Vec<u8> {
        let sender_name = self.sender.display_name().as_bytes();
        let recipient_name = self.recipient.display_name().as_bytes();
        let text_bytes = self.text.as_bytes();
        let mut out = Vec::with_capacity(
            6 + 32 + 2 + sender_name.len() + 32 + 2 + recipient_name.len() + 8 + 8 + 4
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

/// A messaging node: an identity, a TCP listener for incoming messages,
/// and an inbox. The server runs on its own thread; [`TalkNode::stop`]
/// shuts it down.
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

        let server_inbox = Arc::clone(&inbox);
        let server_seq = Arc::clone(&seq);
        let server_stop = Arc::clone(&stop_flag);
        let server = thread::spawn(move || {
            serve(listener, peer_id, server_inbox, server_seq, server_stop);
        });

        Ok(Self {
            identity,
            inbox,
            seq,
            send_seq: AtomicU64::new(0),
            addr,
            stop_flag,
            server: Some(server),
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
    /// was sent. Fails cleanly if the peer is unreachable.
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
        Ok(message)
    }

    /// Accept an encoded message (from the network or from a test):
    /// decode it, verify the signature, check it is addressed to this
    /// node, and store it in the inbox. Anything that fails any check
    /// is rejected and never stored.
    pub fn receive_raw(&self, bytes: &[u8]) -> Result<(), TalkError> {
        let message = Message::decode(bytes)?;
        if !message.verify() {
            return Err(TalkError::BadSignature);
        }
        if message.recipient != *self.identity.peer_id() {
            return Err(TalkError::WrongRecipient);
        }
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        self.inbox
            .lock()
            .unwrap()
            .push(StoredMessage { message, seq });
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
            (a.message.timestamp, a.message.seq, a.seq)
                .cmp(&(b.message.timestamp, b.message.seq, b.seq))
        });
        stored.into_iter().map(|s| s.message).collect()
    }

    /// Total messages in the inbox (from all peers).
    pub fn inbox_len(&self) -> usize {
        self.inbox.lock().unwrap().len()
    }

    /// Stop the server thread and shut this node down.
    pub fn stop(mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        if let Some(server) = self.server.take() {
            let _ = server.join();
        }
    }
}

/// The server loop: accept connections and handle each on its own
/// thread until the stop flag is set.
fn serve(
    listener: Listener,
    peer_id: PeerId,
    inbox: Arc<Mutex<Vec<StoredMessage>>>,
    seq: Arc<AtomicU64>,
    stop_flag: Arc<AtomicBool>,
) {
    loop {
        if stop_flag.load(Ordering::SeqCst) {
            break;
        }
        match listener.accept() {
            Ok(conn) => {
                let inbox = Arc::clone(&inbox);
                let seq = Arc::clone(&seq);
                let peer_id = peer_id.clone();
                thread::spawn(move || handle_connection(conn, peer_id, inbox, seq));
            }
            Err(TransportError::Io(ref e)) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(5));
            }
            Err(_) => break,
        }
    }
}

/// Handle one peer connection: read frames until the peer goes away,
/// verifying and storing each message.
fn handle_connection(
    mut conn: Connection,
    peer_id: PeerId,
    inbox: Arc<Mutex<Vec<StoredMessage>>>,
    seq: Arc<AtomicU64>,
) {
    loop {
        let frame = match conn.recv() {
            Ok(frame) => frame,
            Err(_) => return, // peer went away; nothing more to do
        };
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
        let s = seq.fetch_add(1, Ordering::SeqCst);
        inbox.lock().unwrap().push(StoredMessage { message, seq: s });
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
        std::thread::sleep(Duration::from_millis(200));

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
        std::thread::sleep(Duration::from_millis(200));
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
        std::thread::sleep(Duration::from_millis(300));
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
        alice
            .send_message(bob.addr(), &bob_peer, "to bob")
            .unwrap();
        alice
            .send_message(carol.addr(), &carol_peer, "to carol")
            .unwrap();
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(bob.messages_for(alice.peer_id()).len(), 1);
        assert_eq!(carol.messages_for(alice.peer_id()).len(), 1);
        // Bob has no messages from Carol.
        assert!(bob.messages_for(&carol_peer).is_empty());
        // Carol has no messages from Bob.
        assert!(carol.messages_for(&bob_peer).is_empty());
    }
}
