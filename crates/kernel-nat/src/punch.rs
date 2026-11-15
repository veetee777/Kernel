//! UDP hole punching, plus the rendezvous that introduces peers.
//!
//! The classic simultaneous-open trick: two peers behind NATs learn
//! each other's public address (see [`crate::stun`]), a rendezvous
//! hands each peer the other's address, and then both peers send UDP
//! packets at each other at the same time. Each NAT sees an outbound
//! packet to the peer's address first, so when the peer's packet
//! arrives it matches an existing mapping and is let through. After
//! one punch packet lands, the hole is open and payloads flow.
//!
//! This module only speaks the exchange. It does not run a public
//! rendezvous server — [`Rendezvous`] is the in-process registry the
//! real server would back with a wire protocol. A deployment needs
//! that server, plus the STUN servers from [`crate::stun`].

use std::collections::HashMap;
use std::fmt;
use std::io;
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use kernel_identity::PeerId;

/// Magic at the start of every punch/data packet.
pub const PUNCH_MAGIC: &[u8; 6] = b"KPUNCH";
/// A hole-punch ping: "I am trying to reach you".
pub const KIND_HELLO: u8 = 1;
/// A payload packet, sent once the hole is open.
pub const KIND_DATA: u8 = 2;
/// Header size: magic (6) + kind (1).
pub const PUNCH_HEADER_LEN: usize = 7;
/// How often a punching peer re-sends its hello while waiting.
const PUNCH_RESEND_INTERVAL: Duration = Duration::from_millis(50);

/// Errors from hole punching and payload exchange.
#[derive(Debug)]
pub enum PunchError {
    /// A socket operation failed.
    Io(io::Error),
    /// No punch packet arrived from the peer within the deadline.
    Timeout,
    /// A payload packet was shorter than the header, or had the
    /// wrong magic or kind.
    Malformed(String),
}

impl fmt::Display for PunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PunchError::Io(e) => write!(f, "io error: {e}"),
            PunchError::Timeout => {
                f.write_str("timed out waiting for the peer's punch packet")
            }
            PunchError::Malformed(msg) => write!(f, "malformed punch packet: {msg}"),
        }
    }
}

impl std::error::Error for PunchError {}

impl From<io::Error> for PunchError {
    fn from(e: io::Error) -> Self {
        PunchError::Io(e)
    }
}

/// Encode one punch packet: magic + kind + payload.
pub fn encode_packet(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(PUNCH_HEADER_LEN + payload.len());
    out.extend_from_slice(PUNCH_MAGIC);
    out.push(kind);
    out.extend_from_slice(payload);
    out
}

/// Decode a punch packet. Returns the kind and payload, or `None`
/// when the magic is wrong or the packet is shorter than the header.
pub fn decode_packet(bytes: &[u8]) -> Option<(u8, &[u8])> {
    if bytes.len() < PUNCH_HEADER_LEN {
        return None;
    }
    if &bytes[..PUNCH_MAGIC.len()] != PUNCH_MAGIC {
        return None;
    }
    Some((bytes[PUNCH_MAGIC.len()], &bytes[PUNCH_HEADER_LEN..]))
}

/// The rendezvous: a registry of who is where. A peer registers its
/// public address; another peer asks for it. In production this is a
/// public server both peers can reach; here it is an in-process map
/// so the exchange can be tested without one.
#[derive(Default)]
pub struct Rendezvous {
    peers: HashMap<PeerId, SocketAddr>,
}

impl Rendezvous {
    /// An empty rendezvous.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register (or re-register) a peer's public address.
    pub fn register(&mut self, peer: PeerId, addr: SocketAddr) {
        self.peers.insert(peer, addr);
    }

    /// The registered address of a peer, if it has registered.
    pub fn address_of(&self, peer: &PeerId) -> Option<SocketAddr> {
        self.peers.get(peer).copied()
    }

    /// Introduce two peers: returns `(a's address, b's address)` so
    /// each side knows where to punch. `None` if either peer has not
    /// registered yet.
    pub fn introduce(&self, a: &PeerId, b: &PeerId) -> Option<(SocketAddr, SocketAddr)> {
        Some((self.address_of(a)?, self.address_of(b)?))
    }

    /// Number of registered peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }
}

/// Punch a hole to `peer`: send hello packets at the peer's public
/// address and wait until one of the peer's hellos arrives, or the
/// deadline passes. Both sides must call this at roughly the same
/// time — that simultaneity is what opens the path through a
/// stateful NAT. A packet from any other address is ignored. The
/// socket is connected to `peer` (once) and stays that way, so one
/// socket serves exactly one peer — use a fresh socket per peer.
/// An unreachable peer (ICMP port-unreachable) is treated as
/// silence: the loop keeps trying until the deadline, then reports
/// [`PunchError::Timeout`].
pub fn punch(socket: &UdpSocket, peer: SocketAddr, deadline: Duration) -> Result<(), PunchError> {
    let hello = encode_packet(KIND_HELLO, &[]);
    let start = Instant::now();
    crate::ensure_connected(socket, peer)?;
    socket.set_read_timeout(Some(PUNCH_RESEND_INTERVAL))?;
    let result = loop {
        if start.elapsed() >= deadline {
            break Err(PunchError::Timeout);
        }
        // Send first, so the NAT mapping exists before the peer's
        // packet arrives; re-send on every iteration while waiting.
        // A refused send just means nobody is listening (yet).
        match socket.send(&hello) {
            Ok(_) => {}
            Err(e) if crate::is_no_answer(&e) => {}
            Err(e) => break Err(PunchError::Io(e)),
        }
        let mut buf = [0u8; 1024];
        match socket.recv_from(&mut buf) {
            Ok((len, from)) => {
                if from == peer {
                    if let Some((KIND_HELLO, _)) = decode_packet(&buf[..len]) {
                        break Ok(());
                    }
                }
                // Anything else (a stray packet, a late data packet)
                // is ignored; keep punching until the deadline.
            }
            Err(e) if crate::is_no_answer(&e) => {
                // No packet this round; loop and re-send.
            }
            Err(e) => break Err(PunchError::Io(e)),
        }
    };
    let _ = socket.set_read_timeout(None);
    result
}

/// Send a payload through the open hole. The payload is framed with
/// the punch magic and the DATA kind so the receiver can tell it
/// apart from hellos and stray traffic. The socket must be the one
/// punched to `peer` (see [`punch`]); a socket connected to anyone
/// else is refused.
pub fn send_data(socket: &UdpSocket, peer: SocketAddr, payload: &[u8]) -> io::Result<usize> {
    let packet = encode_packet(KIND_DATA, payload);
    crate::ensure_connected(socket, peer)?;
    socket.send(&packet)
}

/// Receive one payload packet. Waits up to `timeout` for a DATA
/// packet from `peer`; hellos and packets from anyone else are
/// skipped. Returns the number of payload bytes copied into `buf`
/// (truncated if the payload is larger than `buf`) and the sender.
pub fn recv_data(
    socket: &UdpSocket,
    peer: SocketAddr,
    buf: &mut [u8],
    timeout: Duration,
) -> Result<(usize, SocketAddr), PunchError> {
    let start = Instant::now();
    socket.set_read_timeout(Some(PUNCH_RESEND_INTERVAL))?;
    let result = loop {
        if start.elapsed() >= timeout {
            break Err(PunchError::Timeout);
        }
        let mut tmp = [0u8; 2048];
        match socket.recv_from(&mut tmp) {
            Ok((len, from)) => {
                if from != peer {
                    continue;
                }
                match decode_packet(&tmp[..len]) {
                    Some((KIND_DATA, payload)) => {
                        let n = payload.len().min(buf.len());
                        buf[..n].copy_from_slice(&payload[..n]);
                        break Ok((n, from));
                    }
                    _ => continue, // hello or garbage: keep waiting
                }
            }
            Err(e) if crate::is_no_answer(&e) => {
                // Nothing this round (or the peer's port is closed);
                // loop until the deadline.
            }
            Err(e) => break Err(PunchError::Io(e)),
        }
    };
    let _ = socket.set_read_timeout(None);
    result
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

    #[test]
    fn packet_round_trip() {
        let encoded = encode_packet(KIND_DATA, b"hello");
        let (kind, payload) = decode_packet(&encoded).unwrap();
        assert_eq!(kind, KIND_DATA);
        assert_eq!(payload, b"hello");
        assert_eq!(decode_packet(b"short"), None);
        assert_eq!(decode_packet(b"XXXXXX\x02payload"), None);
    }

    #[test]
    fn rendezvous_introduces_peers() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let mut rz = Rendezvous::new();
        let a_addr: SocketAddr = "203.0.113.1:40001".parse().unwrap();
        let b_addr: SocketAddr = "198.51.100.2:40002".parse().unwrap();
        assert_eq!(rz.introduce(alice.peer_id(), bob.peer_id()), None);
        rz.register(alice.peer_id().clone(), a_addr);
        assert_eq!(rz.introduce(alice.peer_id(), bob.peer_id()), None);
        rz.register(bob.peer_id().clone(), b_addr);
        assert_eq!(
            rz.introduce(alice.peer_id(), bob.peer_id()),
            Some((a_addr, b_addr))
        );
        assert_eq!(rz.peer_count(), 2);
    }

    #[test]
    fn punch_exchange_succeeds_over_loopback() {
        use std::sync::Arc;
        let a = Arc::new(socket());
        let b = Arc::new(socket());
        let a_addr = a.local_addr().unwrap();
        let b_addr = b.local_addr().unwrap();

        // Both sides punch at the same time, on two threads.
        let a_for_thread = Arc::clone(&a);
        let a_handle = thread::spawn(move || punch(&a_for_thread, b_addr, Duration::from_secs(5)));
        let b_result = punch(&b, a_addr, Duration::from_secs(5));
        let a_result = a_handle.join().unwrap();
        assert!(a_result.is_ok(), "a: {a_result:?}");
        assert!(b_result.is_ok(), "b: {b_result:?}");

        // The hole is open: payloads flow both ways.
        send_data(&a, b_addr, b"from a").unwrap();
        let mut buf = [0u8; 64];
        let (n, from) = recv_data(&b, a_addr, &mut buf, Duration::from_secs(2)).unwrap();
        assert_eq!(&buf[..n], b"from a");
        assert_eq!(from, a_addr);

        send_data(&b, a_addr, b"from b").unwrap();
        let (n, from) = recv_data(&a, b_addr, &mut buf, Duration::from_secs(2)).unwrap();
        assert_eq!(&buf[..n], b"from b");
        assert_eq!(from, b_addr);
    }

    #[test]
    fn punch_times_out_when_peer_never_answers() {
        let a = socket();
        let nobody: SocketAddr = "127.0.0.1:1".parse().unwrap();
        assert!(matches!(
            punch(&a, nobody, Duration::from_millis(300)),
            Err(PunchError::Timeout)
        ));
    }
}
