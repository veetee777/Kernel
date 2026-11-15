//! Authenticated encrypted sessions over KERNEL TCP transport.
//!
//! This crate is a thin integration layer, not a second crypto
//! implementation. All cryptography lives in `kernel-carrier`: the
//! SIGMA handshake (`Initiator` / `Responder`), sealed sessions
//! (`Session`: XChaCha20-Poly1305, 64-entry replay window, rekeying),
//! and pinned peer keys (`PinSet`). This crate only carries those
//! frames over a [`kernel_transport::Connection`] (length-prefixed
//! TCP frames) and presents a small session API on top.
//!
//! One [`kernel_transport::Connection`] message carries exactly one
//! KCAR frame. KCAR frames are at most 65,548 bytes, far under the
//! transport's 64 MiB cap, so no chunking is needed.
//!
//! The integration point is the handshake, not raw session
//! construction: `Session::new` is `pub(crate)` in the carrier, so
//! sessions are created only by a completed handshake. `connect`
//! runs the initiator side, `accept` the responder side, and both
//! return a [`SecureSession`] that seals and opens payloads.
//!
//! Key ownership: `connect` and `accept` take a reference to the
//! caller's Ed25519 `SigningKey`, exactly as the carrier's own
//! transport does. The key is used only during the handshake and is
//! never stored. `kernel_identity::Identity` deliberately does not
//! export its secret key, so a node that holds its identity only as
//! an `Identity` cannot use this layer yet. Bridging that (an export
//! API on `Identity`, or a handshake that signs through a callback)
//! is future work, not a reason to copy the handshake here.
//!
//! Known limit, inherited from the carrier handshake: there is no
//! cookie challenge or proof-of-work before the responder's
//! Diffie-Hellman, so Hello floods cost the responder one DH and one
//! signature each. The carrier's own `transport` module adds that
//! protection; this layer, riding on `kernel-transport`, does not.
//! The handshake itself is still authenticated: both sides must have
//! the other's key pinned, and the signatures bind the full
//! transcript, so a stranger cannot complete a handshake.

#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::fmt;

use ed25519_dalek::SigningKey;
use kernel_carrier::{CarrierError, Event, Initiator, PinSet, Responder, Session};
use kernel_transport::{Connection, TransportError};

/// Anything that can go wrong establishing or using a session.
#[derive(Debug)]
pub enum SessionError {
    /// A carrier-layer refusal: bad frame, unpinned peer, bad
    /// signature, replay, tampered bytes, bad rekey.
    Carrier(CarrierError),
    /// A transport failure: refused connection, closed peer, frame
    /// too large.
    Transport(TransportError),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SessionError::Carrier(e) => write!(f, "carrier: {e}"),
            SessionError::Transport(e) => write!(f, "transport: {e}"),
        }
    }
}

impl std::error::Error for SessionError {}

impl From<CarrierError> for SessionError {
    fn from(e: CarrierError) -> Self {
        SessionError::Carrier(e)
    }
}

impl From<TransportError> for SessionError {
    fn from(e: TransportError) -> Self {
        SessionError::Transport(e)
    }
}

/// An established, authenticated, encrypted session on one TCP
/// connection. Payloads are sealed with the carrier session keys;
/// the connection carries one KCAR frame per message.
pub struct SecureSession {
    conn: Connection,
    session: Session,
    inbox: VecDeque<Vec<u8>>,
}

impl fmt::Debug for SecureSession {
    /// Shows the peer and the role only. Key material is never
    /// printed, and the underlying carrier session is not Debug.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SecureSession")
            .field("peer", &self.session.peer())
            .field("is_initiator", &self.session.is_initiator())
            .finish_non_exhaustive()
    }
}

impl SecureSession {
    fn new(conn: Connection, session: Session) -> Self {
        Self {
            conn,
            session,
            inbox: VecDeque::new(),
        }
    }

    /// The Ed25519 public key the handshake authenticated for the
    /// other side.
    pub fn peer(&self) -> [u8; 32] {
        self.session.peer()
    }

    /// True on the side that initiated the handshake.
    pub fn is_initiator(&self) -> bool {
        self.session.is_initiator()
    }

    /// Frames per key epoch before a rekey is due. Owner-tunable.
    pub fn set_rekey_policy(&mut self, max_frames: u64) {
        self.session.set_rekey_policy(max_frames);
    }

    /// True once a rekey is due under the current policy.
    pub fn needs_rekey(&self) -> bool {
        self.session.needs_rekey()
    }

    /// True while a rekey started by [`SecureSession::start_rekey`]
    /// has not completed.
    pub fn rekey_pending(&self) -> bool {
        self.session.rekey_pending()
    }

    /// Send one payload, sealed. While a rekey is pending the
    /// payload is queued in the session and sent under the new keys
    /// when the rekey completes.
    pub fn send(&mut self, payload: &[u8]) -> Result<(), SessionError> {
        let frames = self.session.send(payload)?;
        for f in frames {
            self.conn.send(&f)?;
        }
        Ok(())
    }

    /// Receive one payload. Rekey steps are handled here, and any
    /// frames they require are written back before this returns.
    pub fn recv(&mut self) -> Result<Vec<u8>, SessionError> {
        if let Some(p) = self.inbox.pop_front() {
            return Ok(p);
        }
        loop {
            let frame = self.conn.recv()?;
            match self.session.handle_incoming(&frame)? {
                Event::Data(p) => return Ok(p),
                Event::Write(frames) => {
                    for f in frames {
                        self.conn.send(&f)?;
                    }
                }
            }
        }
    }

    /// Initiator: request a rekey. The caller keeps using `send`,
    /// which queues while the rekey is pending.
    pub fn start_rekey(&mut self) -> Result<(), SessionError> {
        let request = self.session.begin_rekey()?;
        self.conn.send(&request)?;
        Ok(())
    }

    /// Read until a pending rekey completes. Data that arrives
    /// meanwhile is kept, in order, for `recv`.
    pub fn complete_rekey(&mut self) -> Result<(), SessionError> {
        while self.session.rekey_pending() {
            let frame = self.conn.recv()?;
            match self.session.handle_incoming(&frame)? {
                Event::Data(p) => self.inbox.push_back(p),
                Event::Write(frames) => {
                    for f in frames {
                        self.conn.send(&f)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Initiator: start a rekey if the session says one is due and
    /// none is pending. Responders never start rekeys.
    pub fn rekey_if_due(&mut self) -> Result<(), SessionError> {
        if self.session.is_initiator()
            && self.session.needs_rekey()
            && !self.session.rekey_pending()
        {
            self.start_rekey()?;
        }
        Ok(())
    }
}

/// Connect to a peer and run the initiator side of the carrier
/// handshake over the open connection: Hello, Response, Confirm.
/// The pin check runs before any byte is written, so an unpinned
/// peer costs no handshake at all.
pub fn connect(
    mut conn: Connection,
    my_key: &SigningKey,
    peer_pub: [u8; 32],
    pins: &PinSet,
) -> Result<SecureSession, SessionError> {
    let (initiator, hello) = Initiator::start(my_key, peer_pub, pins)?;
    conn.send(&hello)?;
    let response = conn.recv()?;
    let (session, confirm) = initiator.finish(&response, pins)?;
    conn.send(&confirm)?;
    Ok(SecureSession::new(conn, session))
}

/// Accept one connection and run the responder side of the carrier
/// handshake over it: read the Hello, answer with a Response, verify
/// the Confirm.
pub fn accept(
    mut conn: Connection,
    my_key: &SigningKey,
    pins: &PinSet,
) -> Result<SecureSession, SessionError> {
    let hello = conn.recv()?;
    let (responder, response) = Responder::accept(my_key, pins, &hello)?;
    conn.send(&response)?;
    let confirm = conn.recv()?;
    let session = responder.finish(&confirm, pins)?;
    Ok(SecureSession::new(conn, session))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_transport::Listener;
    use std::thread;
    use std::time::Duration;

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn public(k: &SigningKey) -> [u8; 32] {
        k.verifying_key().to_bytes()
    }

    fn pins_with(keys: &[&SigningKey]) -> PinSet {
        let mut p = PinSet::new();
        for k in keys {
            p.pin(public(k)).expect("pin");
        }
        p
    }

    /// A connected pair over real loopback TCP: (initiator, responder).
    fn pair() -> (SecureSession, SecureSession) {
        let a = key(1);
        let b = key(2);
        let listener = Listener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let b_srv = b.clone();
        let server = thread::spawn(move || {
            let conn = listener.accept().expect("accept");
            accept(conn, &b_srv, &pins_with(&[&key(1)])).expect("server handshake")
        });
        let conn = Connection::connect(addr).expect("connect");
        let client = connect(conn, &a, public(&b), &pins_with(&[&b])).expect("client handshake");
        let server = server.join().expect("server thread");
        (client, server)
    }

    #[test]
    fn two_nodes_establish_a_session_and_exchange_payloads_both_ways() {
        let (mut a, mut b) = pair();
        assert_eq!(a.peer(), public(&key(2)));
        assert_eq!(b.peer(), public(&key(1)));
        assert!(a.is_initiator());
        assert!(!b.is_initiator());

        a.send(b"ping").expect("send");
        assert_eq!(b.recv().expect("recv"), b"ping".to_vec());
        b.send(b"pong").expect("send");
        assert_eq!(a.recv().expect("recv"), b"pong".to_vec());
    }

    #[test]
    fn tampered_bytes_on_the_wire_are_refused() {
        let a = key(1);
        let b = key(2);
        let listener = Listener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");

        let b_srv = b.clone();
        let server = thread::spawn(move || {
            let conn = listener.accept().expect("accept");
            let mut s = accept(conn, &b_srv, &pins_with(&[&key(1)])).expect("server handshake");
            s.recv()
        });

        // Man in the middle: forward the handshake untouched, then flip
        // one byte in the first Data frame (its AEAD tag).
        let (tx, rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
        let middle = thread::spawn(move || {
            let mitm = Listener::bind("127.0.0.1:0").expect("bind middle");
            tx.send(mitm.local_addr().expect("middle addr"))
                .expect("send addr");
            let mut from_client = mitm.accept().expect("accept client");
            let mut to_server = Connection::connect(addr).expect("connect server");
            from_client
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("timeout");
            to_server
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("timeout");
            // 1. Hello, client -> server.
            let f = from_client.recv().expect("hello");
            to_server.send(&f).expect("forward hello");
            // 2. Response, server -> client.
            let f = to_server.recv().expect("response");
            from_client.send(&f).expect("forward response");
            // 3. Confirm, client -> server.
            let f = from_client.recv().expect("confirm");
            to_server.send(&f).expect("forward confirm");
            // 4. First Data frame, client -> server, tampered.
            let mut f = from_client.recv().expect("data");
            let last = f.len() - 1;
            f[last] ^= 0x01;
            to_server.send(&f).expect("forward tampered");
        });

        let middle_addr = rx.recv().expect("middle addr");
        let conn = Connection::connect(middle_addr).expect("connect middle");
        let mut client = connect(conn, &a, public(&b), &pins_with(&[&b])).expect("client handshake");
        client.send(b"secret").expect("send");
        middle.join().expect("middle thread");

        let err = server.join().expect("server thread").expect_err("refused");
        assert!(
            matches!(err, SessionError::Carrier(CarrierError::AuthFailed)),
            "expected AuthFailed, got {err:?}"
        );
    }

    #[test]
    fn session_for_the_wrong_peer_is_refused() {
        let a = key(1);
        let b = key(2);
        let c = key(3);
        let listener = Listener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let a_srv = a.clone();
        let server = thread::spawn(move || {
            let conn = listener.accept().expect("accept");
            // The server holds key c. The client expects key b.
            accept(conn, &c, &pins_with(&[&a_srv]))
        });
        let conn = Connection::connect(addr).expect("connect");
        let client = connect(conn, &a, public(&b), &pins_with(&[&b]));
        // The client cannot learn why: the server closes the connection
        // after refusing the Hello. Any error here is the refusal.
        assert!(client.is_err());
        let server_err = server.join().expect("server thread").expect_err("server refused");
        assert!(
            matches!(
                server_err,
                SessionError::Carrier(CarrierError::WrongResponder)
            ),
            "expected WrongResponder, got {server_err:?}"
        );
    }

    #[test]
    fn unpinned_peer_is_refused_before_any_byte_is_sent() {
        let a = key(1);
        let b = key(2);
        let listener = Listener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let server = thread::spawn(move || {
            let mut conn = listener.accept().expect("accept");
            conn.set_read_timeout(Some(Duration::from_secs(5)))
                .expect("timeout");
            // If the Hello had been sent, this would return its bytes.
            // It must return a clean close instead.
            conn.recv()
        });
        let conn = Connection::connect(addr).expect("connect");
        let err = connect(conn, &a, public(&b), &PinSet::new()).expect_err("refuse");
        assert!(
            matches!(err, SessionError::Carrier(CarrierError::UnpinnedPeer)),
            "expected UnpinnedPeer, got {err:?}"
        );
        let server_result = server.join().expect("server thread");
        assert!(
            matches!(server_result, Err(TransportError::ConnectionClosed)),
            "expected ConnectionClosed, got {server_result:?}"
        );
    }

    #[test]
    fn data_flows_across_a_rekey() {
        let (mut a, mut b) = pair();
        a.set_rekey_policy(2);
        let reader = thread::spawn(move || {
            // The responder answers the rekey request inside these recv calls.
            let got: Vec<Vec<u8>> = (0..4).map(|_| b.recv().expect("recv")).collect();
            got
        });
        a.send(b"one").expect("send");
        a.send(b"two").expect("send");
        assert!(a.needs_rekey());
        a.start_rekey().expect("start rekey");
        a.send(b"three").expect("queued during rekey");
        a.complete_rekey().expect("complete rekey");
        a.send(b"four").expect("send");
        let got = reader.join().expect("reader");
        assert_eq!(
            got,
            vec![
                b"one".to_vec(),
                b"two".to_vec(),
                b"three".to_vec(),
                b"four".to_vec()
            ]
        );
    }
}
