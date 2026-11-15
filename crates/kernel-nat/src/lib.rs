//! KERNEL NAT traversal: STUN-style address discovery, UDP hole
//! punching, and a relay fallback.
//!
//! The order of operations for two peers behind NATs is:
//!
//! 1. Each peer learns its public address with [`stun`] (a minimal
//!    RFC 5389 binding request against a STUN server).
//! 2. A [`punch::Rendezvous`] introduces the peers by address.
//! 3. Both peers [`punch::punch`] at each other at the same time, so
//!    each NAT sees an outbound packet first and lets the reply in.
//! 4. If punching fails (symmetric NATs, strict firewalls), traffic
//!    goes through a [`relay::Relay`] instead — slower, but it works
//!    anywhere UDP reaches the relay.
//!
//! [`learn::learn_nat`] ties step 1 together: it queries one or more
//! STUN servers from one socket and classifies the NAT from whether
//! the observed public address stays the same across destinations,
//! plus an inbound probe that separates full cone from restricted.
//!
//! Honest limits, stated up front: the tests in this crate run over
//! loopback, and loopback has no NAT — they prove the protocol
//! machinery (encoding, parsing, exchange, forwarding), not that a
//! packet crosses a real NAT. A public STUN server, a public
//! rendezvous, and a real-network test are still needed before any
//! of this can be claimed to work across the internet. Symmetric
//! NATs will usually need the relay.
//!
//! Socket model: a UDP socket here is connected to exactly one peer
//! (connect once, then `send`). That is plain UDP and works
//! everywhere, but it means one socket cannot talk to two peers:
//! STUN queries go out on one socket per server, punching uses one
//! socket per peer, and the relay keeps one connected send socket
//! per registered peer (its listening socket stays unconnected so
//! it can receive from anyone). A connected socket only receives
//! from its peer, so a socket used for receiving from several
//! peers must stay unconnected.

#![forbid(unsafe_code)]

pub mod learn;
pub mod punch;
pub mod relay;
pub mod stun;

pub use learn::{learn_nat, NatInfo, NatType};
pub use punch::{punch, Rendezvous};
pub use relay::Relay;
pub use stun::StunClient;

use std::io;
use std::net::{SocketAddr, UdpSocket};

/// Connect `socket` to `peer` unless it is already connected to
/// that peer. A second `connect()` on the same socket is an error
/// here (and is refused by this machine's sandbox), so callers that
/// send repeatedly must go through this helper instead of
/// connecting on every call. Connecting to a *different* peer is
/// also an error — the caller needs a fresh socket for that peer.
pub(crate) fn ensure_connected(socket: &UdpSocket, peer: SocketAddr) -> io::Result<()> {
    match socket.peer_addr() {
        Ok(addr) if addr == peer => Ok(()),
        Ok(addr) => Err(io::Error::other(format!(
            "socket is connected to {addr}, not {peer}"
        ))),
        Err(_) => socket.connect(peer),
    }
}

/// True when a socket error just means "nobody answered": a read
/// timeout, or an ICMP port-unreachable reported on a connected
/// socket (nothing is listening at the other end). Callers treat
/// both as silence and keep waiting until their deadline, instead
/// of failing the whole exchange on the first unreachable packet.
pub(crate) fn is_no_answer(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::ConnectionRefused
    )
}
