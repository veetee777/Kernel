//! A relay fallback for peers that cannot punch through.
//!
//! When hole punching fails — symmetric NATs, strict firewalls —
//! both peers talk to a relay they can both reach, and the relay
//! forwards datagrams between them. It costs a hop and the relay
//! sees the traffic (this crate does not encrypt relayed payloads;
//! end-to-end encryption lives in `kernel-session`), but it works
//! anywhere UDP reaches the relay.
//!
//! Wire format, client → relay: magic (6) + destination public key
//! (32) + payload. The relay strips the header and forwards the
//! payload as-is to the destination's registered address. The
//! destination replies the same way, naming the first peer. A frame
//! naming an unregistered peer is dropped, never forwarded.

use std::collections::HashMap;
use std::fmt;
use std::io;
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use kernel_identity::PeerId;

/// Magic at the start of every relay frame.
pub const RELAY_MAGIC: &[u8; 6] = b"KRLAY1";
/// Header size: magic (6) + destination public key (32).
pub const RELAY_HEADER_LEN: usize = 38;

/// What happened to one datagram the relay received.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelayOutcome {
    /// The payload was forwarded to the destination's address.
    Forwarded {
        /// Where it was sent.
        to: SocketAddr,
        /// Payload bytes forwarded.
        bytes: usize,
    },
    /// The frame named a peer that never registered; nothing was sent.
    DroppedUnknown,
    /// The datagram was not a well-formed relay frame; nothing was sent.
    DroppedMalformed,
}

/// Errors from relay operations.
#[derive(Debug)]
pub enum RelayError {
    /// A socket operation failed.
    Io(io::Error),
    /// No datagram arrived within the timeout.
    Timeout,
}

impl fmt::Display for RelayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RelayError::Io(e) => write!(f, "io error: {e}"),
            RelayError::Timeout => f.write_str("timed out waiting for a relay datagram"),
        }
    }
}

impl std::error::Error for RelayError {}

impl From<io::Error> for RelayError {
    fn from(e: io::Error) -> Self {
        RelayError::Io(e)
    }
}

/// Frame a payload for the relay: magic + destination key + payload.
/// The destination is named by its Ed25519 public key, which is how
/// the relay's registry is keyed (no display name needed).
pub fn frame_for_relay(dest_key: &[u8; 32], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(RELAY_HEADER_LEN + payload.len());
    out.extend_from_slice(RELAY_MAGIC);
    out.extend_from_slice(dest_key);
    out.extend_from_slice(payload);
    out
}

/// Parse a relay frame. Returns the destination key and the payload,
/// or `None` when the magic is wrong or the frame is too short.
pub fn parse_relay_frame(bytes: &[u8]) -> Option<(&[u8; 32], &[u8])> {
    if bytes.len() < RELAY_HEADER_LEN {
        return None;
    }
    if &bytes[..RELAY_MAGIC.len()] != RELAY_MAGIC {
        return None;
    }
    let key: &[u8; 32] = bytes[RELAY_MAGIC.len()..RELAY_HEADER_LEN]
        .try_into()
        .ok()?;
    Some((key, &bytes[RELAY_HEADER_LEN..]))
}

/// A UDP relay. Peers register their address (usually via the
/// rendezvous); the relay forwards framed datagrams between
/// registered peers and drops everything else.
///
/// The listening socket stays unconnected so it can receive from
/// anyone, but a connected socket only receives from its peer — so
/// forwarding goes out on one connected send socket per registered
/// peer, created lazily on the first forward to that peer. Peers
/// therefore need a separate socket for receiving relayed payloads:
/// the socket they send through the relay with becomes connected
/// to the relay and will only hear back from the relay's listening
/// address.
pub struct Relay {
    socket: UdpSocket,
    peers: HashMap<[u8; 32], SocketAddr>,
    /// One connected send socket per registered peer, created on
    /// first forward. Keyed the same way as `peers`.
    send_sockets: HashMap<[u8; 32], UdpSocket>,
    /// Datagrams forwarded so far.
    pub forwarded: u64,
    /// Datagrams dropped (unknown destination or malformed).
    pub refused: u64,
}

impl Relay {
    /// Create a relay on a bound UDP socket.
    pub fn new(socket: UdpSocket) -> Self {
        Self {
            socket,
            peers: HashMap::new(),
            send_sockets: HashMap::new(),
            forwarded: 0,
            refused: 0,
        }
    }

    /// Register a peer's address under its public key. Drops any
    /// cached send socket for the peer, so the next forward uses
    /// the new address.
    pub fn register(&mut self, peer: &PeerId, addr: SocketAddr) {
        let key = *peer.public_key().as_bytes();
        self.peers.insert(key, addr);
        self.send_sockets.remove(&key);
    }

    /// Register an address under a raw public key. Drops any cached
    /// send socket for the key, so the next forward uses the new
    /// address.
    pub fn register_key(&mut self, key: [u8; 32], addr: SocketAddr) {
        self.peers.insert(key, addr);
        self.send_sockets.remove(&key);
    }

    /// Send `payload` to the peer at `addr`, reusing the cached
    /// connected send socket for `key` or creating one. A connected
    /// socket only receives from its peer, so the relay cannot
    /// forward from its listening socket — each destination gets
    /// its own connected send socket.
    fn send_to_peer(
        &mut self,
        key: &[u8; 32],
        addr: SocketAddr,
        payload: &[u8],
    ) -> io::Result<usize> {
        if !self.send_sockets.contains_key(key) {
            let sock = UdpSocket::bind("0.0.0.0:0")?;
            sock.connect(addr)?;
            self.send_sockets.insert(*key, sock);
        }
        let sock = self
            .send_sockets
            .get(key)
            .expect("send socket was just inserted");
        sock.send(payload)
    }

    /// The registered address of a peer, if any.
    pub fn address_of(&self, peer: &PeerId) -> Option<SocketAddr> {
        self.peers.get(peer.public_key().as_bytes()).copied()
    }

    /// Number of registered peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Handle one datagram: wait up to `timeout` for a frame, then
    /// forward it or drop it. Returns what happened to the datagram.
    pub fn handle_one(&mut self, timeout: Duration) -> Result<RelayOutcome, RelayError> {
        self.socket.set_read_timeout(Some(timeout))?;
        let mut buf = [0u8; 2048];
        let result = match self.socket.recv_from(&mut buf) {
            Ok((len, _from)) => {
                let outcome = match parse_relay_frame(&buf[..len]) {
                    Some((dest_key, payload)) => match self.peers.get(dest_key) {
                        Some(&to) => {
                            self.send_to_peer(dest_key, to, payload)?;
                            self.forwarded += 1;
                            RelayOutcome::Forwarded {
                                to,
                                bytes: payload.len(),
                            }
                        }
                        None => {
                            self.refused += 1;
                            RelayOutcome::DroppedUnknown
                        }
                    },
                    None => {
                        self.refused += 1;
                        RelayOutcome::DroppedMalformed
                    }
                };
                Ok(outcome)
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Err(RelayError::Timeout),
            Err(e) => Err(RelayError::Io(e)),
        };
        let _ = self.socket.set_read_timeout(None);
        result
    }

    /// Handle datagrams until `deadline`, returning the number of
    /// datagrams forwarded. Stops early on a socket error.
    pub fn handle_until(&mut self, deadline: Duration) -> Result<u64, RelayError> {
        let start = Instant::now();
        let mut forwarded = 0;
        loop {
            let remaining = deadline.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                return Ok(forwarded);
            }
            match self.handle_one(remaining) {
                Ok(RelayOutcome::Forwarded { .. }) => forwarded += 1,
                Ok(_) => {}
                Err(RelayError::Timeout) => return Ok(forwarded),
                Err(e) => return Err(e),
            }
        }
    }
}

/// Send a payload through a relay to `dest`. Frames it with the
/// relay magic and the destination's public key. The socket is
/// connected to the relay (once) and stays that way — use a
/// dedicated socket for relay traffic, and a separate unconnected
/// socket for receiving relayed payloads, since a connected socket
/// only hears back from the relay's listening address.
pub fn send_via_relay(
    socket: &UdpSocket,
    relay: SocketAddr,
    dest: &PeerId,
    payload: &[u8],
) -> io::Result<usize> {
    let frame = frame_for_relay(dest.public_key().as_bytes(), payload);
    crate::ensure_connected(socket, relay)?;
    socket.send(&frame)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_identity::Identity;
    use std::net::UdpSocket;
    use std::thread;

    fn socket() -> UdpSocket {
        UdpSocket::bind("127.0.0.1:0").unwrap()
    }

    fn recv_plain(socket: &UdpSocket, timeout: Duration) -> Vec<u8> {
        socket.set_read_timeout(Some(timeout)).unwrap();
        let mut buf = [0u8; 2048];
        let (len, _) = socket.recv_from(&mut buf).unwrap();
        let _ = socket.set_read_timeout(None);
        buf[..len].to_vec()
    }

    #[test]
    fn relay_frame_round_trip() {
        let key = [9u8; 32];
        let frame = frame_for_relay(&key, b"payload");
        let (dest, payload) = parse_relay_frame(&frame).unwrap();
        assert_eq!(dest, &key);
        assert_eq!(payload, b"payload");
        assert_eq!(parse_relay_frame(b"short"), None);
        assert_eq!(parse_relay_frame(b"XXXXXXxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxpayload"), None);
    }

    #[test]
    fn relay_forwards_both_ways() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        // One socket per direction per peer: the send socket becomes
        // connected to the relay, so it cannot receive the relay's
        // forwarded payloads (they come from the relay's per-peer
        // send socket, a different address). Receiving happens on
        // separate unconnected sockets, whose addresses are the
        // ones registered with the relay.
        let a_send = socket();
        let a_recv = socket();
        let b_send = socket();
        let b_recv = socket();
        let relay_sock = socket();
        let relay_addr = relay_sock.local_addr().unwrap();
        let mut relay = Relay::new(relay_sock);
        relay.register(alice.peer_id(), a_recv.local_addr().unwrap());
        relay.register(bob.peer_id(), b_recv.local_addr().unwrap());
        assert_eq!(relay.peer_count(), 2);

        // A -> relay -> B, then B -> relay -> A. The relay serves
        // both frames on a thread while the test drives both ends.
        let relay_thread = thread::spawn(move || {
            let mut outcomes = Vec::new();
            for _ in 0..2 {
                outcomes.push(relay.handle_one(Duration::from_secs(5)).unwrap());
            }
            outcomes
        });
        send_via_relay(&a_send, relay_addr, bob.peer_id(), b"hello bob").unwrap();
        assert_eq!(recv_plain(&b_recv, Duration::from_secs(2)), b"hello bob");
        send_via_relay(&b_send, relay_addr, alice.peer_id(), b"hello alice").unwrap();
        assert_eq!(recv_plain(&a_recv, Duration::from_secs(2)), b"hello alice");
        let outcomes = relay_thread.join().unwrap();
        assert_eq!(outcomes.len(), 2);
        assert!(outcomes
            .iter()
            .all(|o| matches!(o, RelayOutcome::Forwarded { .. })));
    }

    #[test]
    fn relay_refuses_unregistered_peer() {
        let alice = Identity::generate("alice").unwrap();
        let stranger = Identity::generate("stranger").unwrap();
        let a = socket();
        let b = socket();
        let relay_sock = socket();
        let relay_addr = relay_sock.local_addr().unwrap();
        let mut relay = Relay::new(relay_sock);
        // Only Alice is registered; the stranger never registered.
        relay.register(alice.peer_id(), a.local_addr().unwrap());
        assert_eq!(relay.peer_count(), 1);

        let relay_thread =
            thread::spawn(move || relay.handle_one(Duration::from_secs(2)).unwrap());
        send_via_relay(&a, relay_addr, stranger.peer_id(), b"nobody home").unwrap();
        assert_eq!(relay_thread.join().unwrap(), RelayOutcome::DroppedUnknown);

        // Nothing arrives at B either.
        b.set_read_timeout(Some(Duration::from_millis(200))).unwrap();
        let mut buf = [0u8; 64];
        assert!(matches!(
            b.recv_from(&mut buf),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock
        ));
    }

    #[test]
    fn relay_drops_malformed_frames() {
        let alice = Identity::generate("alice").unwrap();
        let a = socket();
        let relay_sock = socket();
        let relay_addr = relay_sock.local_addr().unwrap();
        let mut relay = Relay::new(relay_sock);
        relay.register(alice.peer_id(), a.local_addr().unwrap());

        let relay_thread =
            thread::spawn(move || relay.handle_one(Duration::from_secs(2)).unwrap());
        a.connect(relay_addr).unwrap();
        a.send(b"not a relay frame").unwrap();
        assert_eq!(
            relay_thread.join().unwrap(),
            RelayOutcome::DroppedMalformed
        );
    }
}
