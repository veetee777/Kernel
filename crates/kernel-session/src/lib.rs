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
//! Hello floods: this layer rate-limits handshake attempts per peer
//! IP before any Diffie-Hellman work happens (see [`HandshakeLimiter`]).
//! `accept` applies a process-wide default limit; servers that expect
//! floods should use [`accept_with_limiter`] with their own limiter.
//! This is throttling, not the carrier's cookie challenge or
//! proof-of-work — the carrier's own `transport` module has the
//! stronger protection, and this layer, riding on `kernel-transport`,
//! does not. A distributed flood from many addresses still costs the
//! responder, and an attacker behind the same NAT as a legitimate peer
//! shares that peer's budget. The handshake itself is still
//! authenticated: both sides must have the other's key pinned, and the
//! signatures bind the full transcript, so a stranger cannot complete
//! a handshake.

#![forbid(unsafe_code)]

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::net::IpAddr;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

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
    /// The handshake was refused by the rate limiter before any
    /// Diffie-Hellman work: too many attempts from this address in
    /// the current window.
    RateLimited,
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SessionError::Carrier(e) => write!(f, "carrier: {e}"),
            SessionError::Transport(e) => write!(f, "transport: {e}"),
            SessionError::RateLimited => {
                f.write_str("too many handshake attempts from this address")
            }
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

/// Default handshake attempts allowed per peer IP per window for the
/// process-wide limiter used by [`accept`]. Generous on purpose: it
/// is a safety net against floods, not a quota on legitimate use.
pub const DEFAULT_MAX_ATTEMPTS: u32 = 100;
/// Default window for [`DEFAULT_MAX_ATTEMPTS`].
pub const DEFAULT_WINDOW: Duration = Duration::from_secs(60);
/// Default number of peer IPs tracked at once by a limiter.
pub const DEFAULT_MAX_TRACKED: usize = 1024;

#[derive(Clone, Copy)]
struct Window {
    started: Instant,
    count: u32,
}

/// Per-IP handshake rate limiting for the responder side.
///
/// The limiter is keyed by the peer's IP address, not the full socket
/// address: the source port is ephemeral and attacker-controlled, so
/// keying by `SocketAddr` would hand every new connection a fresh
/// budget and the limit would do nothing against a flood.
///
/// Each handshake attempt consumes one slot when it is admitted. A
/// completed handshake refunds its slot (see
/// [`HandshakeLimiter::note_success`]): a stranger cannot complete a
/// handshake — the carrier handshake is authenticated — so floods burn
/// budget while legitimate peers that finish handshakes do not. A
/// failed or abandoned attempt keeps its slot until the window resets.
///
/// The table is bounded: at most `max_tracked` addresses are kept.
/// When it is full, expired entries are pruned first, then the oldest
/// tracked address is evicted to make room. The limiter never grows
/// unbounded from attacker-controlled addresses.
pub struct HandshakeLimiter {
    max_attempts: u32,
    window: Duration,
    max_tracked: usize,
    table: HashMap<IpAddr, Window>,
    /// Insertion order of tracked addresses, oldest first; used only
    /// for eviction.
    order: VecDeque<IpAddr>,
}

impl HandshakeLimiter {
    /// A limiter allowing `max_attempts` handshakes per peer IP per
    /// `window`, tracking at most [`DEFAULT_MAX_TRACKED`] addresses.
    /// `max_attempts` of 0 refuses every attempt.
    pub fn new(max_attempts: u32, window: Duration) -> Self {
        Self::with_capacity(max_attempts, window, DEFAULT_MAX_TRACKED)
    }

    /// A limiter with an explicit cap on tracked addresses.
    pub fn with_capacity(max_attempts: u32, window: Duration, max_tracked: usize) -> Self {
        Self {
            max_attempts,
            window,
            max_tracked,
            table: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// Attempts allowed per peer IP per window.
    pub fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    /// Length of one counting window.
    pub fn window(&self) -> Duration {
        self.window
    }

    /// Maximum number of peer IPs tracked at once.
    pub fn max_tracked(&self) -> usize {
        self.max_tracked
    }

    /// Number of peer IPs currently tracked.
    pub fn tracked(&self) -> usize {
        self.table.len()
    }

    /// Count one handshake attempt from `ip` at the current time.
    /// True means the attempt may proceed; false means refuse it
    /// before doing any handshake work.
    pub fn admit(&mut self, ip: IpAddr) -> bool {
        self.admit_at(ip, Instant::now())
    }

    /// Count one handshake attempt from `ip` at `now`. Exposed so
    /// tests can move time without sleeping; production callers use
    /// [`HandshakeLimiter::admit`].
    pub fn admit_at(&mut self, ip: IpAddr, now: Instant) -> bool {
        if let Some(w) = self.table.get_mut(&ip) {
            if now.saturating_duration_since(w.started) >= self.window {
                w.started = now;
                w.count = 0;
            }
            if w.count >= self.max_attempts {
                return false;
            }
            w.count += 1;
            return true;
        }
        // A new address. Make room first: prune expired entries,
        // then evict the oldest tracked address.
        if self.table.len() >= self.max_tracked {
            let window = self.window;
            self.table
                .retain(|_, w| now.saturating_duration_since(w.started) < window);
            self.order.retain(|k| self.table.contains_key(k));
            if self.table.len() >= self.max_tracked {
                match self.order.pop_front() {
                    Some(oldest) => {
                        self.table.remove(&oldest);
                    }
                    None => return false, // max_tracked == 0
                }
            }
        }
        self.table.insert(
            ip,
            Window {
                started: now,
                count: 1,
            },
        );
        self.order.push_back(ip);
        true
    }

    /// Refund one attempt for `ip` after a handshake completed.
    /// A completed handshake costs the responder its DH work once,
    /// but it is a legitimate peer, not a flood, so it does not keep
    /// consuming budget. If the refund empties the address's entry
    /// the entry is dropped. Calling this for an address with no
    /// entry is a no-op.
    pub fn note_success(&mut self, ip: IpAddr) {
        let emptied = match self.table.get_mut(&ip) {
            Some(w) => {
                w.count = w.count.saturating_sub(1);
                w.count == 0
            }
            None => false,
        };
        if emptied {
            self.table.remove(&ip);
            self.order.retain(|k| *k != ip);
        }
    }
}

impl Default for HandshakeLimiter {
    /// [`DEFAULT_MAX_ATTEMPTS`] per [`DEFAULT_WINDOW`] per IP,
    /// [`DEFAULT_MAX_TRACKED`] addresses tracked.
    fn default() -> Self {
        Self::new(DEFAULT_MAX_ATTEMPTS, DEFAULT_WINDOW)
    }
}

/// The process-wide default limiter behind [`accept`]. The lock is
/// held only for the admit/refund check — never across network I/O —
/// so concurrent handshakes do not serialize on it.
fn default_limiter() -> &'static Mutex<HandshakeLimiter> {
    static LIMITER: OnceLock<Mutex<HandshakeLimiter>> = OnceLock::new();
    LIMITER.get_or_init(|| Mutex::new(HandshakeLimiter::default()))
}

/// Accept one connection and run the responder side of the carrier
/// handshake over it: read the Hello, answer with a Response, verify
/// the Confirm.
///
/// The attempt is rate-limited by a process-wide default limiter
/// ([`DEFAULT_MAX_ATTEMPTS`] attempts per peer IP per
/// [`DEFAULT_WINDOW`], [`DEFAULT_MAX_TRACKED`] addresses tracked).
/// A refused attempt returns [`SessionError::RateLimited`] before any
/// byte is read and before any Diffie-Hellman work. Servers that
/// expect floods, or that want their own limits, should use
/// [`accept_with_limiter`] instead.
pub fn accept(
    conn: Connection,
    my_key: &SigningKey,
    pins: &PinSet,
) -> Result<SecureSession, SessionError> {
    let ip = conn.peer_addr()?.ip();
    {
        let mut limiter = default_limiter()
            .lock()
            .expect("default handshake limiter lock");
        if !limiter.admit(ip) {
            return Err(SessionError::RateLimited);
        }
    }
    let result = accept_inner(conn, my_key, pins);
    if result.is_ok() {
        let mut limiter = default_limiter()
            .lock()
            .expect("default handshake limiter lock");
        limiter.note_success(ip);
    }
    result
}

/// Accept one connection with an explicit rate limiter. The limiter
/// is checked before any byte is read from the connection and before
/// any Diffie-Hellman work: a refused attempt costs the responder
/// nothing but the dropped socket. A completed handshake refunds its
/// slot in the limiter; a failed or abandoned one keeps it until the
/// window resets.
///
/// A server handling connections on several threads should hold its
/// limiter behind a `Mutex` (or use the process-wide default via
/// [`accept`]); this function takes `&mut` and does not lock.
pub fn accept_with_limiter(
    conn: Connection,
    my_key: &SigningKey,
    pins: &PinSet,
    limiter: &mut HandshakeLimiter,
) -> Result<SecureSession, SessionError> {
    let ip = conn.peer_addr()?.ip();
    if !limiter.admit(ip) {
        return Err(SessionError::RateLimited);
    }
    let result = accept_inner(conn, my_key, pins);
    if result.is_ok() {
        limiter.note_success(ip);
    }
    result
}

fn accept_inner(
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
        let mut client =
            connect(conn, &a, public(&b), &pins_with(&[&b])).expect("client handshake");
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
        let server_err = server
            .join()
            .expect("server thread")
            .expect_err("server refused");
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

    fn ip(a: u8, b: u8, c: u8, d: u8) -> IpAddr {
        IpAddr::from([a, b, c, d])
    }

    #[test]
    fn limiter_admits_up_to_the_limit_then_refuses() {
        let mut l = HandshakeLimiter::new(3, Duration::from_secs(60));
        let now = Instant::now();
        let addr = ip(10, 0, 0, 1);
        assert!(l.admit_at(addr, now));
        assert!(l.admit_at(addr, now));
        assert!(l.admit_at(addr, now));
        assert!(!l.admit_at(addr, now));
    }

    #[test]
    fn limiter_counts_each_address_separately() {
        let mut l = HandshakeLimiter::new(1, Duration::from_secs(60));
        let now = Instant::now();
        assert!(l.admit_at(ip(10, 0, 0, 1), now));
        assert!(!l.admit_at(ip(10, 0, 0, 1), now));
        assert!(l.admit_at(ip(10, 0, 0, 2), now));
    }

    #[test]
    fn limiter_window_resets() {
        let mut l = HandshakeLimiter::new(1, Duration::from_millis(50));
        let now = Instant::now();
        let addr = ip(10, 0, 0, 1);
        assert!(l.admit_at(addr, now));
        assert!(!l.admit_at(addr, now));
        assert!(l.admit_at(addr, now + Duration::from_millis(60)));
    }

    #[test]
    fn limiter_success_refunds_the_attempt() {
        let mut l = HandshakeLimiter::new(1, Duration::from_secs(60));
        let now = Instant::now();
        let addr = ip(10, 0, 0, 1);
        assert!(l.admit_at(addr, now));
        assert!(!l.admit_at(addr, now));
        l.note_success(addr);
        assert!(l.admit_at(addr, now));
        assert_eq!(l.tracked(), 1);
        // note_success on an unknown address is a no-op.
        l.note_success(ip(10, 0, 0, 9));
        assert_eq!(l.tracked(), 1);
        // A refund that empties the entry drops it from the table.
        l.note_success(addr);
        assert_eq!(l.tracked(), 0);
    }

    #[test]
    fn limiter_table_is_bounded_and_evicts_the_oldest() {
        let mut l = HandshakeLimiter::with_capacity(2, Duration::from_secs(60), 2);
        let now = Instant::now();
        assert!(l.admit_at(ip(10, 0, 0, 1), now));
        assert!(l.admit_at(ip(10, 0, 0, 1), now));
        assert!(l.admit_at(ip(10, 0, 0, 2), now));
        assert_eq!(l.tracked(), 2);
        // Table full: admitting a third address evicts the oldest.
        assert!(l.admit_at(ip(10, 0, 0, 3), now));
        assert_eq!(l.tracked(), 2);
        // The evicted address starts a fresh window.
        assert!(l.admit_at(ip(10, 0, 0, 1), now));
        assert!(l.admit_at(ip(10, 0, 0, 1), now));
        assert!(!l.admit_at(ip(10, 0, 0, 1), now));
    }

    /// Outcome of one responder-side attempt, for tests.
    #[derive(Debug, PartialEq, Eq)]
    enum Attempt {
        Ok,
        RateLimited,
        Unpinned,
        Other(String),
    }

    fn classify(result: Result<SecureSession, SessionError>) -> Attempt {
        match result {
            Ok(_) => Attempt::Ok,
            Err(SessionError::RateLimited) => Attempt::RateLimited,
            Err(SessionError::Carrier(CarrierError::UnpinnedPeer)) => Attempt::Unpinned,
            Err(e) => Attempt::Other(format!("{e:?}")),
        }
    }

    /// Spawn a server that accepts `count` connections on `listener`,
    /// running each through `accept_with_limiter`, and returns the
    /// outcomes. The limiter is owned by the server thread.
    fn run_attempts(
        listener: Listener,
        server_key: SigningKey,
        pins: PinSet,
        limiter: HandshakeLimiter,
        count: usize,
    ) -> thread::JoinHandle<Vec<Attempt>> {
        thread::spawn(move || {
            let mut limiter = limiter;
            let mut outcomes = Vec::new();
            for _ in 0..count {
                let conn = listener.accept().expect("accept");
                outcomes.push(classify(accept_with_limiter(
                    conn,
                    &server_key,
                    &pins,
                    &mut limiter,
                )));
            }
            outcomes
        })
    }

    #[test]
    fn rapid_failed_handshakes_from_one_address_are_refused_before_dh() {
        let a = key(1); // client key, NOT pinned by the server
        let b = key(2); // server key
        let listener = Listener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        // The server pins a key nobody holds, so every attempt fails.
        let limiter = HandshakeLimiter::new(3, Duration::from_secs(60));
        let server = run_attempts(listener, b.clone(), pins_with(&[&key(9)]), limiter, 4);
        for _ in 0..4 {
            let conn = Connection::connect(addr).expect("connect");
            // The client's own pin check passes (it pins b), so it
            // sends a Hello and waits for a Response.
            assert!(connect(conn, &a, public(&b), &pins_with(&[&b])).is_err());
        }
        let outcomes = server.join().expect("server thread");
        assert_eq!(
            outcomes,
            vec![
                Attempt::Unpinned,
                Attempt::Unpinned,
                Attempt::Unpinned,
                Attempt::RateLimited
            ]
        );
    }

    #[test]
    fn refused_attempt_sends_no_response_bytes() {
        let a = key(1);
        let b = key(2);
        let listener = Listener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let limiter = HandshakeLimiter::new(3, Duration::from_secs(60));
        let server = run_attempts(listener, b.clone(), pins_with(&[&key(9)]), limiter, 4);
        // Exhaust the budget with three failed attempts.
        for _ in 0..3 {
            let conn = Connection::connect(addr).expect("connect");
            assert!(connect(conn, &a, public(&b), &pins_with(&[&b])).is_err());
        }
        // The fourth attempt: send a valid Hello by hand. If the
        // responder had read it and done DH work, it would answer
        // with a Response; a refusal must close the connection
        // without sending anything. Timing varies: the responder
        // may already have closed by the time we write (BrokenPipe
        // on send), or the close may land during the read
        // (ConnectionClosed or ConnectionReset). Every one of those
        // proves no Response was sent; only actual bytes would
        // disprove it.
        let mut raw = Connection::connect(addr).expect("connect");
        let (_initiator, hello) =
            Initiator::start(&a, public(&b), &pins_with(&[&b])).expect("hello");
        match raw.send(&hello) {
            Ok(()) => {
                raw.set_read_timeout(Some(Duration::from_secs(2)))
                    .expect("timeout");
                match raw.recv() {
                    Err(TransportError::ConnectionClosed) => {}
                    Err(TransportError::Io(e))
                        if e.kind() == std::io::ErrorKind::ConnectionReset => {}
                    other => panic!(
                        "expected the refused responder to close without sending a Response, got {other:?}"
                    ),
                }
            }
            Err(TransportError::Io(e)) if e.kind() == std::io::ErrorKind::BrokenPipe => {}
            Err(e) => panic!("unexpected error sending Hello to a refused responder: {e:?}"),
        }
        let outcomes = server.join().expect("server thread");
        assert_eq!(
            outcomes,
            vec![
                Attempt::Unpinned,
                Attempt::Unpinned,
                Attempt::Unpinned,
                Attempt::RateLimited
            ]
        );
    }

    #[test]
    fn legitimate_handshakes_from_different_addresses_still_succeed() {
        let a = key(1); // flood client, unpinned by the server
        let c = key(3); // legitimate client, pinned by the server
        let b = key(2); // server key
                        // Bound dual-stack so clients can reach it via 127.0.0.1
                        // (seen as ::ffff:127.0.0.1) and via ::1. Note 127.0.0.2 does
                        // NOT work as a second address here: this VM's loopback
                        // reports it to the server as 127.0.0.1, so it would share
                        // the flooded budget.
        let listener = Listener::bind("[::]:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let limiter = HandshakeLimiter::new(2, Duration::from_secs(60));
        let server = run_attempts(listener, b.clone(), pins_with(&[&c]), limiter, 3);
        // Flood from 127.0.0.1: two failed attempts exhaust its budget.
        for _ in 0..2 {
            let conn = Connection::connect(("127.0.0.1", port)).expect("connect");
            assert!(connect(conn, &a, public(&b), &pins_with(&[&b])).is_err());
        }
        // A peer from ::1 has its own budget and completes.
        let conn = Connection::connect(("::1", port)).expect("connect ::1");
        let session = connect(conn, &c, public(&b), &pins_with(&[&b])).expect("legit handshake");
        assert_eq!(session.peer(), public(&b));
        let outcomes = server.join().expect("server thread");
        assert_eq!(
            outcomes,
            vec![Attempt::Unpinned, Attempt::Unpinned, Attempt::Ok]
        );
    }

    #[test]
    fn a_single_failed_handshake_does_not_lock_out_a_later_legitimate_one() {
        let a = key(1); // unpinned
        let c = key(3); // pinned
        let b = key(2);
        let listener = Listener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let limiter = HandshakeLimiter::new(3, Duration::from_secs(60));
        let server = run_attempts(listener, b.clone(), pins_with(&[&c]), limiter, 2);
        let conn = Connection::connect(addr).expect("connect");
        assert!(connect(conn, &a, public(&b), &pins_with(&[&b])).is_err());
        let conn = Connection::connect(addr).expect("connect");
        let session = connect(conn, &c, public(&b), &pins_with(&[&b])).expect("legit handshake");
        assert_eq!(session.peer(), public(&b));
        let outcomes = server.join().expect("server thread");
        assert_eq!(outcomes, vec![Attempt::Unpinned, Attempt::Ok]);
    }

    #[test]
    fn successful_handshakes_do_not_consume_budget() {
        let c = key(3);
        let b = key(2);
        let listener = Listener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        // Only 2 attempts per window — but three successes in a row
        // must all pass, because each completed handshake refunds
        // its slot.
        let limiter = HandshakeLimiter::new(2, Duration::from_secs(60));
        let server = run_attempts(listener, b.clone(), pins_with(&[&c]), limiter, 3);
        for _ in 0..3 {
            let conn = Connection::connect(addr).expect("connect");
            let session = connect(conn, &c, public(&b), &pins_with(&[&b])).expect("handshake");
            assert_eq!(session.peer(), public(&b));
        }
        let outcomes = server.join().expect("server thread");
        assert_eq!(outcomes, vec![Attempt::Ok, Attempt::Ok, Attempt::Ok]);
    }

    #[test]
    fn limiter_resets_after_the_window() {
        let a = key(1); // unpinned
        let b = key(2);
        let listener = Listener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let limiter = HandshakeLimiter::new(1, Duration::from_millis(100));
        let server = run_attempts(listener, b.clone(), pins_with(&[&key(9)]), limiter, 3);
        let conn = Connection::connect(addr).expect("connect");
        assert!(connect(conn, &a, public(&b), &pins_with(&[&b])).is_err());
        let conn = Connection::connect(addr).expect("connect");
        assert!(connect(conn, &a, public(&b), &pins_with(&[&b])).is_err());
        thread::sleep(Duration::from_millis(150));
        let conn = Connection::connect(addr).expect("connect");
        assert!(connect(conn, &a, public(&b), &pins_with(&[&b])).is_err());
        let outcomes = server.join().expect("server thread");
        // Third attempt is admitted again after the window (and fails
        // on the pin check, not on the limiter).
        assert_eq!(
            outcomes,
            vec![Attempt::Unpinned, Attempt::RateLimited, Attempt::Unpinned]
        );
    }
}
