//! Live TCP transport for carrier frames (Phases 1 and 2).
//!
//! Std only: blocking sockets, one thread per connection, no async runtime, no
//! new crates.
//!
//! One TCP connection carries one handshake, then Data frames for that session.
//! The handshake is preceded by a cookie challenge (flood.rs), so the responder
//! does no Diffie-Hellman for an address that has not echoed a cookie and solved
//! the puzzle. Every handshake read shares one deadline for the whole handshake.
//!
//! Rekeying is driven by the session (session.rs). `recv` answers rekey requests
//! itself. `complete_rekey` waits for a pending rekey to finish, keeping any data
//! that arrives meanwhile in order for `recv`.
//!
//! Known limits:
//! - `accept` has no timeout in std, so `Listener::accept` blocks until a peer
//!   connects. Run it on a thread you can abandon.
//! - One thread per connection. The guard caps connections per address, not in total.
//! - A refused frame does not close the channel. Drop the channel after any error.
//! - Data reads after the handshake use per-operation timeouts. A slow-drip peer
//!   after the handshake can hold a read.
//! - The guard counts connections, not Hellos. A connection costs one accept and
//!   one challenge, so counting connections bounds the same work.

use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use ed25519_dalek::SigningKey;

use crate::error::CarrierError;
use crate::flood::{
    self, CookieState, FloodGuard, COOKIE_LEN, DEFAULT_DIFFICULTY, DEFAULT_MAX_TRACKED, NONCE_LEN,
};
use crate::frame::{self, Kind, HEADER_LEN};
use crate::gateway::GatewayTable;
use crate::handshake::{Initiator, Responder, CONFIRM_BODY, HELLO_BODY, RESPONSE_BODY};
use crate::pins::PinSet;
use crate::session::{Event, Session};

/// Default timeout for each socket connect, read, and write.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// Cookie challenge body: the cookie, then the difficulty byte.
const COOKIE_CHALLENGE_LEN: usize = COOKIE_LEN + 1;
/// Cookie echo body: the cookie, then the puzzle nonce.
const COOKIE_ECHO_LEN: usize = COOKIE_LEN + NONCE_LEN;

/// Anything that can go wrong on the wire, in one type.
#[derive(Debug)]
pub enum TransportError {
    /// Socket error: a timeout, a refused connection, or a peer that closed early.
    Io(io::Error),
    /// Carrier-layer refusal: malformed frame, unpinned peer, bad signature, replay, bad cookie.
    Carrier(CarrierError),
    /// The gateway table has no address for this peer.
    NoRoute,
    /// The address has exceeded its connection limit. Refused before any bytes are written.
    RateLimited,
}

impl From<io::Error> for TransportError {
    fn from(e: io::Error) -> Self {
        TransportError::Io(e)
    }
}

impl From<CarrierError> for TransportError {
    fn from(e: CarrierError) -> Self {
        TransportError::Carrier(e)
    }
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportError::Io(e) => write!(f, "io: {}", e),
            TransportError::Carrier(e) => write!(f, "carrier: {}", e),
            TransportError::NoRoute => f.write_str("no route"),
            TransportError::RateLimited => f.write_str("rate limited"),
        }
    }
}

impl std::error::Error for TransportError {}

fn arr32(b: &[u8]) -> Result<[u8; 32], CarrierError> {
    b.try_into().map_err(|_| CarrierError::MalformedFrame)
}

/// Set per-operation timeouts and disable Nagle, so small frames go out at once.
fn configure(stream: &TcpStream, timeout: Duration) -> io::Result<()> {
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    stream.set_nodelay(true)
}

/// Read one carrier frame of any kind. Used for Data frames, whose body may be up
/// to MAX_BODY. The header is validated before the body is allocated.
pub fn read_frame(stream: &mut TcpStream) -> Result<Vec<u8>, TransportError> {
    let mut hbuf = [0u8; HEADER_LEN];
    stream.read_exact(&mut hbuf)?;
    let header = frame::decode_header(&hbuf)?;
    let mut out = vec![0u8; HEADER_LEN + header.body_len];
    out[..HEADER_LEN].copy_from_slice(&hbuf);
    stream.read_exact(&mut out[HEADER_LEN..])?;
    Ok(out)
}

/// Read exactly `buf.len()` bytes, but never past `deadline`. Each read gets the
/// time remaining, so a peer that trickles bytes cannot extend the handshake.
fn read_exact_by(stream: &mut TcpStream, buf: &mut [u8], deadline: Instant) -> io::Result<()> {
    let mut filled = 0;
    while filled < buf.len() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "handshake deadline passed",
            ));
        }
        stream.set_read_timeout(Some(remaining))?;
        match stream.read(&mut buf[filled..]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "peer closed during handshake",
                ))
            }
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Read one handshake frame of a known kind and exact body length, before the
/// deadline. The kind and length are checked from the header before any body byte
/// is allocated.
fn read_handshake(
    stream: &mut TcpStream,
    kind: Kind,
    body_len: usize,
    deadline: Instant,
) -> Result<Vec<u8>, TransportError> {
    let mut hbuf = [0u8; HEADER_LEN];
    read_exact_by(stream, &mut hbuf, deadline)?;
    let header = frame::decode_header(&hbuf)?;
    if header.kind != kind {
        return Err(TransportError::Carrier(CarrierError::UnexpectedKind));
    }
    if header.body_len != body_len {
        return Err(TransportError::Carrier(CarrierError::MalformedFrame));
    }
    let mut out = vec![0u8; HEADER_LEN + body_len];
    out[..HEADER_LEN].copy_from_slice(&hbuf);
    read_exact_by(stream, &mut out[HEADER_LEN..], deadline)?;
    Ok(out)
}

/// Write one carrier frame. Refuses bytes that are not a well-formed frame, so this
/// layer never sends garbage.
pub fn write_frame(stream: &mut TcpStream, frame_bytes: &[u8]) -> Result<(), TransportError> {
    let _ = frame::decode(frame_bytes)?;
    stream.write_all(frame_bytes)?;
    stream.flush()?;
    Ok(())
}

fn write_all_frames(stream: &mut TcpStream, frames: &[Vec<u8>]) -> Result<(), TransportError> {
    for f in frames {
        write_frame(stream, f)?;
    }
    Ok(())
}

/// An established session on one TCP connection.
pub struct SecureChannel {
    stream: TcpStream,
    session: Session,
    inbox: VecDeque<Vec<u8>>,
}

impl SecureChannel {
    fn new(stream: TcpStream, session: Session) -> Self {
        Self {
            stream,
            session,
            inbox: VecDeque::new(),
        }
    }

    /// The Ed25519 public key the handshake authenticated for the other side.
    pub fn peer(&self) -> [u8; 32] {
        self.session.peer()
    }

    pub fn is_initiator(&self) -> bool {
        self.session.is_initiator()
    }

    pub fn set_rekey_policy(&mut self, max_frames: u64) {
        self.session.set_rekey_policy(max_frames);
    }

    /// Time per epoch before a rekey is due. See `Session::set_rekey_interval`.
    pub fn set_rekey_interval(&mut self, interval: std::time::Duration) {
        self.session.set_rekey_interval(interval);
    }

    pub fn needs_rekey(&self) -> bool {
        self.session.needs_rekey()
    }

    /// Send one payload. While a rekey is pending, the payload is queued instead.
    pub fn send(&mut self, payload: &[u8]) -> Result<(), TransportError> {
        let frames = self.session.send(payload)?;
        write_all_frames(&mut self.stream, &frames)
    }

    /// Receive one payload. Rekey steps are handled here, and any frames they require
    /// are written back before this returns.
    pub fn recv(&mut self) -> Result<Vec<u8>, TransportError> {
        if let Some(p) = self.inbox.pop_front() {
            return Ok(p);
        }
        loop {
            let f = read_frame(&mut self.stream)?;
            match self.session.handle_incoming(&f)? {
                Event::Data(p) => return Ok(p),
                Event::Write(frames) => write_all_frames(&mut self.stream, &frames)?,
            }
        }
    }

    /// Initiator: request a rekey. The caller keeps using `send`, which queues while
    /// the rekey is pending.
    pub fn start_rekey(&mut self) -> Result<(), TransportError> {
        let request = self.session.begin_rekey()?;
        write_frame(&mut self.stream, &request)
    }

    /// Read until a pending rekey completes. Data that arrives meanwhile is kept, in
    /// order, for `recv`.
    pub fn complete_rekey(&mut self) -> Result<(), TransportError> {
        while self.session.rekey_pending() {
            let f = read_frame(&mut self.stream)?;
            match self.session.handle_incoming(&f)? {
                Event::Data(p) => self.inbox.push_back(p),
                Event::Write(frames) => write_all_frames(&mut self.stream, &frames)?,
            }
        }
        Ok(())
    }

    /// Initiator: start a rekey if the session says one is due and none is pending.
    /// Responders never start rekeys.
    pub fn rekey_if_due(&mut self) -> Result<(), TransportError> {
        if self.session.is_initiator() && self.session.needs_rekey() && !self.session.rekey_pending() {
            self.start_rekey()?;
        }
        Ok(())
    }
}

/// Connect to a peer and run the initiator side: cookie, puzzle, Hello, Response,
/// Confirm, all before one deadline.
///
/// The pin check runs before any network activity, so an unpinned peer costs no
/// connection at all.
pub fn connect_channel(
    addr: SocketAddr,
    my_key: &SigningKey,
    peer_pub: [u8; 32],
    pins: &PinSet,
    timeout: Duration,
) -> Result<SecureChannel, TransportError> {
    let (initiator, hello) = Initiator::start(my_key, peer_pub, pins)?;
    let deadline = Instant::now() + timeout;
    let mut stream = TcpStream::connect_timeout(&addr, timeout)?;
    configure(&stream, timeout)?;

    let challenge = read_handshake(&mut stream, Kind::Cookie, COOKIE_CHALLENGE_LEN, deadline)?;
    let cookie = arr32(&challenge[HEADER_LEN..HEADER_LEN + COOKIE_LEN])?;
    let difficulty = challenge[HEADER_LEN + COOKIE_LEN];
    let nonce = flood::solve_pow(&cookie, difficulty)?;
    let mut echo_body = cookie.to_vec();
    echo_body.extend_from_slice(&nonce.to_be_bytes());
    write_frame(&mut stream, &frame::encode(Kind::Cookie, &echo_body)?)?;

    write_frame(&mut stream, &hello)?;
    let response = read_handshake(&mut stream, Kind::Response, RESPONSE_BODY, deadline)?;
    let (session, confirm) = initiator.finish(&response, pins)?;
    write_frame(&mut stream, &confirm)?;
    Ok(SecureChannel::new(stream, session))
}

/// Responder-side state that outlives one connection: the cookie secret and the
/// per-address guard.
pub struct Listener {
    cookies: CookieState,
    guard: FloodGuard,
    handshake_timeout: Duration,
}

impl Listener {
    /// `difficulty`: puzzle bits. `max_per_window`: connections per address per
    /// `window`. `handshake_timeout`: the deadline for one whole handshake.
    pub fn new(
        difficulty: u8,
        max_per_window: u32,
        window: Duration,
        handshake_timeout: Duration,
    ) -> Result<Self, CarrierError> {
        Ok(Self {
            cookies: CookieState::new(difficulty)?,
            guard: FloodGuard::new(max_per_window, window, DEFAULT_MAX_TRACKED),
            handshake_timeout,
        })
    }

    /// Accept one connection and run the responder side.
    ///
    /// Order: the guard, then the cookie challenge, then the echo check, and only
    /// then the Hello. Diffie-Hellman happens after all of those.
    pub fn accept(
        &mut self,
        listener: &TcpListener,
        my_key: &SigningKey,
        pins: &PinSet,
    ) -> Result<SecureChannel, TransportError> {
        let (mut stream, peer) = listener.accept()?;
        let now = Instant::now();
        if !self.guard.admit(peer.ip(), now) {
            return Err(TransportError::RateLimited);
        }
        configure(&stream, self.handshake_timeout)?;
        let deadline = now + self.handshake_timeout;

        let cookie = self.cookies.issue(peer.ip(), now)?;
        let mut challenge = cookie.to_vec();
        challenge.push(self.cookies.difficulty());
        write_frame(&mut stream, &frame::encode(Kind::Cookie, &challenge)?)?;

        let echo = read_handshake(&mut stream, Kind::Cookie, COOKIE_ECHO_LEN, deadline)?;
        let presented = arr32(&echo[HEADER_LEN..HEADER_LEN + COOKIE_LEN])?;
        let nonce = u64::from_be_bytes(
            echo[HEADER_LEN + COOKIE_LEN..]
                .try_into()
                .map_err(|_| CarrierError::MalformedFrame)?,
        );
        if !self.cookies.verify(peer.ip(), Instant::now(), &presented)
            || !flood::pow_ok(&presented, nonce, self.cookies.difficulty())
        {
            return Err(TransportError::Carrier(CarrierError::BadCookie));
        }

        let hello = read_handshake(&mut stream, Kind::Hello, HELLO_BODY, deadline)?;
        let (responder, response) = Responder::accept(my_key, pins, &hello)?;
        write_frame(&mut stream, &response)?;
        let confirm = read_handshake(&mut stream, Kind::Confirm, CONFIRM_BODY, deadline)?;
        let session = responder.finish(&confirm, pins)?;
        Ok(SecureChannel::new(stream, session))
    }
}

/// One-shot responder: a fresh Listener for one connection. Keeps the Phase 1 call
/// shape. A real node keeps one `Listener` and loops.
pub fn accept_channel(
    listener: &TcpListener,
    my_key: &SigningKey,
    pins: &PinSet,
    timeout: Duration,
) -> Result<SecureChannel, TransportError> {
    let mut state = Listener::new(DEFAULT_DIFFICULTY, u32::MAX, Duration::from_secs(60), timeout)?;
    state.accept(listener, my_key, pins)
}

/// Connect to a peer whose address is recorded in the gateway table.
pub fn connect_via(
    table: &GatewayTable,
    my_key: &SigningKey,
    peer_pub: [u8; 32],
    pins: &PinSet,
    timeout: Duration,
) -> Result<SecureChannel, TransportError> {
    let addr = table.lookup(&peer_pub).ok_or(TransportError::NoRoute)?;
    connect_channel(addr, my_key, peer_pub, pins, timeout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    const TEST_TIMEOUT: Duration = Duration::from_secs(5);

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

    fn loopback() -> (TcpListener, SocketAddr) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        (listener, addr)
    }

    fn spawn_responder(
        listener: TcpListener,
        my_key: SigningKey,
        pins: PinSet,
    ) -> thread::JoinHandle<Result<SecureChannel, TransportError>> {
        thread::spawn(move || accept_channel(&listener, &my_key, &pins, TEST_TIMEOUT))
    }

    /// A connected pair over loopback: (initiator, responder).
    fn pair() -> (SecureChannel, SecureChannel) {
        let (listener, addr) = loopback();
        let a = key(1);
        let b = key(2);
        let server = spawn_responder(listener, b.clone(), pins_with(&[&a]));
        let client = connect_channel(addr, &a, public(&b), &pins_with(&[&b]), TEST_TIMEOUT)
            .expect("client handshake");
        let srv = server.join().expect("server thread").expect("server handshake");
        (client, srv)
    }

    #[test]
    fn handshake_and_data_both_ways_over_tcp() {
        let a = key(1);
        let b = key(2);
        let (listener, addr) = loopback();
        let server = spawn_responder(listener, b.clone(), pins_with(&[&a]));

        let mut client = connect_channel(addr, &a, public(&b), &pins_with(&[&b]), TEST_TIMEOUT)
            .expect("client handshake");
        let mut srv = server
            .join()
            .expect("server thread")
            .expect("server handshake");

        assert_eq!(client.peer(), public(&b));
        assert_eq!(srv.peer(), public(&a));

        client.send(b"ping").expect("send");
        assert_eq!(srv.recv().expect("recv"), b"ping".to_vec());
        srv.send(b"pong").expect("send");
        assert_eq!(client.recv().expect("recv"), b"pong".to_vec());
    }

    #[test]
    fn replayed_frame_is_refused_over_the_wire() {
        let (mut client, mut srv) = pair();
        // Put the same sealed frame on the wire twice.
        let sealed = client.session.seal(b"once").expect("seal");
        write_frame(&mut client.stream, &sealed).expect("first write");
        write_frame(&mut client.stream, &sealed).expect("replayed write");

        assert_eq!(srv.recv().expect("first"), b"once".to_vec());
        assert!(matches!(
            srv.recv(),
            Err(TransportError::Carrier(CarrierError::Replay))
        ));
    }

    #[test]
    fn tampered_frame_is_refused_over_the_wire() {
        let (mut client, mut srv) = pair();
        let mut sealed = client.session.seal(b"secret").expect("seal");
        let last = sealed.len() - 1;
        sealed[last] ^= 0x01;
        write_frame(&mut client.stream, &sealed).expect("write");
        assert!(matches!(
            srv.recv(),
            Err(TransportError::Carrier(CarrierError::AuthFailed))
        ));
    }

    #[test]
    fn unpinned_responder_is_refused_before_any_connection() {
        let a = key(1);
        let b = key(2);
        // Nothing listens here. If the pin check ran after connecting, the error
        // would be an Io error (connection refused), not UnpinnedPeer.
        let (listener, addr) = loopback();
        drop(listener);
        let err = connect_channel(addr, &a, public(&b), &PinSet::new(), TEST_TIMEOUT)
            .err()
            .expect("refuse");
        assert!(matches!(
            err,
            TransportError::Carrier(CarrierError::UnpinnedPeer)
        ));
    }

    #[test]
    fn unpinned_initiator_is_refused_by_the_responder() {
        let a = key(1);
        let b = key(2);
        let (listener, addr) = loopback();
        // The responder pins nobody, so it must refuse the initiator.
        let server = spawn_responder(listener, b.clone(), PinSet::new());
        let client = connect_channel(addr, &a, public(&b), &pins_with(&[&b]), TEST_TIMEOUT);
        assert!(client.is_err());
        let server_err = server
            .join()
            .expect("server thread")
            .err()
            .expect("server refused");
        assert!(matches!(
            server_err,
            TransportError::Carrier(CarrierError::UnpinnedPeer)
        ));
    }

    #[test]
    fn client_expecting_another_responder_is_refused() {
        let a = key(1);
        let b = key(2);
        let c = key(3);
        let (listener, addr) = loopback();
        // The server holds key c. The client expects key b.
        let server = spawn_responder(listener, c.clone(), pins_with(&[&a]));
        let client = connect_channel(addr, &a, public(&b), &pins_with(&[&b]), TEST_TIMEOUT);
        assert!(client.is_err());
        let server_err = server
            .join()
            .expect("server thread")
            .err()
            .expect("server refused");
        assert!(matches!(
            server_err,
            TransportError::Carrier(CarrierError::WrongResponder)
        ));
    }

    #[test]
    fn silent_peer_times_out_instead_of_hanging() {
        let a = key(1);
        let b = key(2);
        let (listener, addr) = loopback();
        // Accept the connection, say nothing, hold it open briefly, then close.
        let holder = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept");
            thread::sleep(Duration::from_secs(1));
            drop(stream);
        });
        let short = Duration::from_millis(200);
        let err = connect_channel(addr, &a, public(&b), &pins_with(&[&b]), short)
            .err()
            .expect("timeout");
        assert!(matches!(err, TransportError::Io(_)));
        holder.join().expect("holder thread");
    }

    #[test]
    fn oversize_declared_body_is_refused_before_allocation() {
        let (listener, addr) = loopback();
        let reader = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            read_frame(&mut stream)
        });
        let mut raw = TcpStream::connect(addr).expect("connect");
        // Header claims 65_537 body bytes. Nothing else is sent.
        let header = [b'K', b'C', b'A', b'R', 1, 4, 0, 0, 0x00, 0x01, 0x00, 0x01];
        raw.write_all(&header).expect("write");
        let err = reader
            .join()
            .expect("reader thread")
            .err()
            .expect("refused");
        assert!(matches!(err, TransportError::Carrier(CarrierError::Oversize)));
    }

    #[test]
    fn truncated_frame_is_an_io_error() {
        let (listener, addr) = loopback();
        let reader = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            read_frame(&mut stream)
        });
        let mut raw = TcpStream::connect(addr).expect("connect");
        // Header claims 100 body bytes. Send 10, then close.
        let mut bytes = vec![b'K', b'C', b'A', b'R', 1, 4, 0, 0, 0, 0, 0, 100];
        bytes.extend_from_slice(&[0u8; 10]);
        raw.write_all(&bytes).expect("write");
        drop(raw);
        let err = reader
            .join()
            .expect("reader thread")
            .err()
            .expect("refused");
        assert!(matches!(err, TransportError::Io(_)));
    }

    #[test]
    fn handshake_read_refuses_wrong_kind() {
        let (listener, addr) = loopback();
        let reader = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            read_handshake(&mut stream, Kind::Hello, HELLO_BODY, Instant::now() + TEST_TIMEOUT)
        });
        let mut raw = TcpStream::connect(addr).expect("connect");
        // A Data header with no body, arriving where a Hello belongs.
        raw.write_all(&[b'K', b'C', b'A', b'R', 1, 4, 0, 0, 0, 0, 0, 0])
            .expect("write");
        let err = reader
            .join()
            .expect("reader thread")
            .err()
            .expect("refused");
        assert!(matches!(
            err,
            TransportError::Carrier(CarrierError::UnexpectedKind)
        ));
    }

    #[test]
    fn handshake_read_refuses_wrong_length() {
        let (listener, addr) = loopback();
        let reader = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            read_handshake(&mut stream, Kind::Hello, HELLO_BODY, Instant::now() + TEST_TIMEOUT)
        });
        let mut raw = TcpStream::connect(addr).expect("connect");
        // A Hello header claiming 65_000 body bytes. Right kind, wrong length.
        raw.write_all(&[b'K', b'C', b'A', b'R', 1, 1, 0, 0, 0, 0, 0xFD, 0xE8])
            .expect("write");
        let err = reader
            .join()
            .expect("reader thread")
            .err()
            .expect("refused");
        assert!(matches!(
            err,
            TransportError::Carrier(CarrierError::MalformedFrame)
        ));
    }

    #[test]
    fn handshake_read_refuses_oversize_claim() {
        let (listener, addr) = loopback();
        let reader = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            read_handshake(&mut stream, Kind::Hello, HELLO_BODY, Instant::now() + TEST_TIMEOUT)
        });
        let mut raw = TcpStream::connect(addr).expect("connect");
        raw.write_all(&[b'K', b'C', b'A', b'R', 1, 1, 0, 0, 0x00, 0x01, 0x00, 0x01])
            .expect("write");
        let err = reader
            .join()
            .expect("reader thread")
            .err()
            .expect("refused");
        assert!(matches!(err, TransportError::Carrier(CarrierError::Oversize)));
    }

    #[test]
    fn write_frame_refuses_bytes_that_are_not_a_frame() {
        let (_listener, addr) = loopback();
        let mut stream = TcpStream::connect(addr).expect("connect");
        let err = write_frame(&mut stream, b"not a frame").err().expect("refuse");
        assert!(matches!(
            err,
            TransportError::Carrier(CarrierError::MalformedFrame)
        ));
    }

    #[test]
    fn gateway_routes_to_the_recorded_address() {
        let a = key(1);
        let b = key(2);
        let pins_a = pins_with(&[&b]);
        let (listener, addr) = loopback();
        let server = spawn_responder(listener, b.clone(), pins_with(&[&a]));

        let mut table = GatewayTable::new();
        table.set(&pins_a, public(&b), addr).expect("record route");
        let mut client =
            connect_via(&table, &a, public(&b), &pins_a, TEST_TIMEOUT).expect("routed connect");
        let mut srv = server
            .join()
            .expect("server thread")
            .expect("server handshake");

        // The handshake authenticated b, so the route reached the right peer.
        assert_eq!(srv.peer(), public(&a));
        client.send(b"routed").expect("send");
        assert_eq!(srv.recv().expect("recv"), b"routed".to_vec());
    }

    #[test]
    fn connect_via_without_a_route_is_refused() {
        let a = key(1);
        let b = key(2);
        let table = GatewayTable::new();
        let err = connect_via(&table, &a, public(&b), &pins_with(&[&b]), TEST_TIMEOUT)
            .err()
            .expect("refuse");
        assert!(matches!(err, TransportError::NoRoute));
    }

    #[test]
    fn a_flood_from_one_address_is_rate_limited() {
        let (listener, addr) = loopback();
        let server = thread::spawn(move || {
            let mut st =
                Listener::new(0, 3, Duration::from_secs(60), Duration::from_millis(300))
                    .expect("listener");
            let pins = pins_with(&[&key(1)]);
            let b = key(2);
            (0..10)
                .map(|_| st.accept(&listener, &b, &pins).err().expect("refused"))
                .collect::<Vec<TransportError>>()
        });
        for _ in 0..10 {
            drop(TcpStream::connect(addr).expect("connect"));
        }
        let errs = server.join().expect("server");
        let limited = errs
            .iter()
            .filter(|e| matches!(e, TransportError::RateLimited))
            .count();
        assert_eq!(errs.len(), 10);
        // Three connections were admitted; the other seven were refused at the guard.
        assert_eq!(limited, 7);
    }

    #[test]
    fn a_forged_cookie_is_refused_before_any_handshake() {
        let (listener, addr) = loopback();
        let server = thread::spawn(move || {
            let mut st =
                Listener::new(0, u32::MAX, Duration::from_secs(60), TEST_TIMEOUT).expect("listener");
            st.accept(&listener, &key(2), &pins_with(&[&key(1)]))
        });
        let mut raw = TcpStream::connect(addr).expect("connect");
        let challenge = read_handshake(
            &mut raw,
            Kind::Cookie,
            COOKIE_CHALLENGE_LEN,
            Instant::now() + TEST_TIMEOUT,
        )
        .expect("challenge");
        let difficulty = challenge[HEADER_LEN + COOKIE_LEN];
        // Answer with a cookie the responder never issued, and a correct puzzle for it.
        let forged = [0x55u8; COOKIE_LEN];
        let nonce = flood::solve_pow(&forged, difficulty).expect("solve");
        let mut body = forged.to_vec();
        body.extend_from_slice(&nonce.to_be_bytes());
        write_frame(&mut raw, &frame::encode(Kind::Cookie, &body).expect("encode")).expect("write");
        let err = server.join().expect("server").err().expect("refused");
        assert!(matches!(err, TransportError::Carrier(CarrierError::BadCookie)));
    }

    #[test]
    fn a_valid_cookie_with_a_wrong_puzzle_answer_is_refused() {
        let (listener, addr) = loopback();
        let server = thread::spawn(move || {
            let mut st =
                Listener::new(8, u32::MAX, Duration::from_secs(60), TEST_TIMEOUT).expect("listener");
            st.accept(&listener, &key(2), &pins_with(&[&key(1)]))
        });
        let mut raw = TcpStream::connect(addr).expect("connect");
        let challenge = read_handshake(
            &mut raw,
            Kind::Cookie,
            COOKIE_CHALLENGE_LEN,
            Instant::now() + TEST_TIMEOUT,
        )
        .expect("challenge");
        // The real cookie, with a nonce that fails the puzzle.
        let cookie: [u8; COOKIE_LEN] = challenge[HEADER_LEN..HEADER_LEN + COOKIE_LEN]
            .try_into()
            .expect("cookie");
        let difficulty = challenge[HEADER_LEN + COOKIE_LEN];
        let wrong = (0u64..)
            .find(|n| !flood::pow_ok(&cookie, *n, difficulty))
            .expect("a wrong nonce");
        let mut body = cookie.to_vec();
        body.extend_from_slice(&wrong.to_be_bytes());
        write_frame(&mut raw, &frame::encode(Kind::Cookie, &body).expect("encode")).expect("write");
        let err = server.join().expect("server").err().expect("refused");
        assert!(matches!(err, TransportError::Carrier(CarrierError::BadCookie)));
    }

    #[test]
    fn a_hello_sent_without_the_cookie_echo_is_refused() {
        let (listener, addr) = loopback();
        let server_key = key(2);
        let client_pin = key(1);
        let server = thread::spawn(move || {
            let mut st =
                Listener::new(0, u32::MAX, Duration::from_secs(60), TEST_TIMEOUT).expect("listener");
            st.accept(&listener, &server_key, &pins_with(&[&client_pin]))
        });
        let client_key = key(1);
        let server_pub = public(&key(2));
        let (_ini, hello) =
            Initiator::start(&client_key, server_pub, &pins_with(&[&key(2)])).expect("start");
        let mut raw = TcpStream::connect(addr).expect("connect");
        write_frame(&mut raw, &hello).expect("write hello");
        let err = server.join().expect("server").err().expect("refused");
        assert!(matches!(
            err,
            TransportError::Carrier(CarrierError::UnexpectedKind)
        ));
    }

    #[test]
    fn a_legitimate_client_completes_through_the_listener() {
        let (listener, addr) = loopback();
        let a = key(1);
        let b = key(2);
        let b_srv = b.clone();
        let server = thread::spawn(move || {
            let mut st = Listener::new(DEFAULT_DIFFICULTY, 10, Duration::from_secs(60), TEST_TIMEOUT)
                .expect("listener");
            st.accept(&listener, &b_srv, &pins_with(&[&key(1)]))
        });
        let mut client = connect_channel(addr, &a, public(&b), &pins_with(&[&b]), TEST_TIMEOUT)
            .expect("client");
        let mut srv = server.join().expect("server").expect("handshake");
        client.send(b"through the listener").expect("send");
        assert_eq!(srv.recv().expect("recv"), b"through the listener".to_vec());
    }

    #[test]
    fn a_slow_peer_hits_the_handshake_deadline() {
        let (listener, addr) = loopback();
        let server = thread::spawn(move || {
            let mut st = Listener::new(0, u32::MAX, Duration::from_secs(60), Duration::from_millis(300))
                .expect("listener");
            st.accept(&listener, &key(2), &pins_with(&[&key(1)]))
        });
        let raw = TcpStream::connect(addr).expect("connect");
        let dripper = thread::spawn(move || {
            let mut raw = raw;
            // One byte every 100 ms. Each read returns in time, but the whole handshake
            // runs past its 300 ms deadline.
            for _ in 0..12 {
                if raw.write_all(&[0]).is_err() {
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }
        });
        let err = server.join().expect("server").err().expect("timed out");
        assert!(matches!(err, TransportError::Io(_)));
        dripper.join().expect("dripper");
    }

    #[test]
    fn data_flows_across_a_rekey_over_tcp() {
        let (mut a, mut b) = pair();
        let reader = thread::spawn(move || {
            // The responder answers the rekey request inside these recv calls.
            let got: Vec<Vec<u8>> = (0..3).map(|_| b.recv().expect("recv")).collect();
            got
        });
        a.send(b"one").expect("send");
        a.start_rekey().expect("start rekey");
        a.send(b"two").expect("queued during rekey");
        a.complete_rekey().expect("complete rekey");
        a.send(b"three").expect("send");
        let got = reader.join().expect("reader");
        assert_eq!(
            got,
            vec![b"one".to_vec(), b"two".to_vec(), b"three".to_vec()]
        );
    }

    #[test]
    fn time_triggered_rekey_completes_over_tcp() {
        let (mut a, mut b) = pair();
        a.set_rekey_interval(std::time::Duration::from_millis(50));
        std::thread::sleep(std::time::Duration::from_millis(60));
        assert!(a.needs_rekey(), "time trigger is due");
        let reader = thread::spawn(move || {
            let got: Vec<Vec<u8>> = (0..3).map(|_| b.recv().expect("recv")).collect();
            got
        });
        a.send(b"one").expect("send");
        a.rekey_if_due().expect("rekey if due");
        a.send(b"two").expect("queued during rekey");
        a.complete_rekey().expect("complete rekey");
        a.send(b"three").expect("send");
        let got = reader.join().expect("reader");
        assert_eq!(
            got,
            vec![b"one".to_vec(), b"two".to_vec(), b"three".to_vec()]
        );
        // The rekey reset both triggers.
        assert!(!a.needs_rekey());
    }
}
