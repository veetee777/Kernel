//! Multi-hop source routing for KERNEL.
//!
//! Messages carry their full path: a [`Route`] is an ordered list of
//! peer ids, source first, destination last. Each hop follows the route
//! exactly; TTL is the backstop that kills loops. The route is
//! authoritative — a hop forwards only where the route says, never
//! anywhere else.
//!
//! Forwarding is **plaintext at this layer**. Encryption is the session
//! layer's job, not this crate's. A malicious hop can read or drop the
//! payload; do not send secrets through a route you do not trust.
//! Routes are operator-supplied: there is NO route discovery here (no
//! gossip, no DHT lookup for next hops). A router only knows the
//! next-hop addresses it was explicitly given.
//!
//! Wire format (`KRTE1`, binary, no JSON):
//! ```text
//! magic      5 bytes  "KRTE1"
//! destination  peer id (32-byte key + u8 name length + name bytes)
//! ttl        1 byte
//! hops       1 byte   (number of remaining hops; must be >= 1)
//! remaining  hops × peer id
//! payload_len 4 bytes big-endian
//! payload    payload_len bytes (must run exactly to the end)
//! ```
//! The remaining route's last entry must be the destination. The
//! remaining route never includes the node that is currently holding
//! the message: at origination it is `route[1..]`, and each forwarder
//! drops its own entry before sending on.

use std::fmt;

use kernel_crypto::Ed25519PublicKey;
use kernel_identity::PeerId;
use kernel_transport::MAX_FRAME_SIZE;

pub mod router;

pub use router::{
    send_routed, send_routed_with_ttl, DropReason, ForwardOutcome, Router, SendOutcome,
};

/// Magic bytes at the start of every routed envelope.
pub const MAGIC: &[u8; 5] = b"KRTE1";

/// TTL used by [`send_routed`] when the caller does not pick one.
pub const DEFAULT_TTL: u8 = 32;

/// Maximum hops in a remaining route (it is counted in one byte).
pub const MAX_ROUTE_HOPS: usize = 255;

/// A full source route: ordered peer ids, source first, destination last.
pub type Route = Vec<PeerId>;

/// Errors returned by routing operations.
#[derive(Debug)]
pub enum RoutingError {
    /// The envelope or the route is structurally wrong (bad magic,
    /// truncation, trailing bytes, a route that does not end at its
    /// destination, ...). The message is never forwarded.
    Malformed(String),
    /// No next-hop address is known for this peer.
    NoRoute(PeerId),
    /// The message's next hop is not this node, or a route offered for
    /// origination does not start at the local node.
    NotInRoute,
    /// The route names this node as its own next hop.
    SelfLoop(PeerId),
    /// The payload (with its envelope) would not fit in one transport
    /// frame.
    PayloadTooLarge(usize),
    /// The transport failed (connect refused, peer went away, ...).
    Transport(kernel_transport::TransportError),
}

impl fmt::Display for RoutingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RoutingError::Malformed(msg) => write!(f, "malformed routed message: {msg}"),
            RoutingError::NoRoute(peer) => write!(f, "no route to {peer}"),
            RoutingError::NotInRoute => f.write_str("message's next hop is not this node"),
            RoutingError::SelfLoop(peer) => {
                write!(f, "route names {peer} as its own next hop")
            }
            RoutingError::PayloadTooLarge(n) => {
                write!(f, "payload of {n} bytes exceeds the frame cap")
            }
            RoutingError::Transport(e) => write!(f, "transport error: {e}"),
        }
    }
}

impl std::error::Error for RoutingError {}

impl From<kernel_transport::TransportError> for RoutingError {
    fn from(e: kernel_transport::TransportError) -> Self {
        RoutingError::Transport(e)
    }
}

/// Two peer ids name the same peer when their public keys match. The
/// display name is a human label; it does not affect routing.
pub(crate) fn same_peer(a: &PeerId, b: &PeerId) -> bool {
    a.public_key().as_bytes() == b.public_key().as_bytes()
}

/// A framed, source-routed message in flight.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutedMessage {
    /// The final destination (the last entry of the original route).
    destination: PeerId,
    /// Hops still to go, first entry first. Never includes the node
    /// currently holding the message; the last entry is the destination.
    remaining: Vec<PeerId>,
    /// Forwards still allowed. Delivery to the destination does not
    /// consume TTL; each forward decrements it by one.
    ttl: u8,
    /// The opaque payload. Plaintext at this layer.
    payload: Vec<u8>,
}

impl RoutedMessage {
    /// Build a message. Does not validate the route — [`Router`] does
    /// that when it originates or forwards one.
    pub fn new(destination: PeerId, remaining: Vec<PeerId>, ttl: u8, payload: Vec<u8>) -> Self {
        Self {
            destination,
            remaining,
            ttl,
            payload,
        }
    }

    /// The final destination.
    pub fn destination(&self) -> &PeerId {
        &self.destination
    }

    /// Hops still to go, first entry first.
    pub fn remaining(&self) -> &[PeerId] {
        &self.remaining
    }

    /// Forwards still allowed.
    pub fn ttl(&self) -> u8 {
        self.ttl
    }

    /// The payload.
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// Encode the message to its binary wire form.
    pub fn encode(&self) -> Result<Vec<u8>, RoutingError> {
        if self.remaining.is_empty() {
            return Err(RoutingError::Malformed(
                "routed message has no remaining hops".to_string(),
            ));
        }
        if self.remaining.len() > MAX_ROUTE_HOPS {
            return Err(RoutingError::Malformed(format!(
                "route has {} hops, more than the maximum {MAX_ROUTE_HOPS}",
                self.remaining.len()
            )));
        }
        if self.payload.len() > MAX_FRAME_SIZE {
            return Err(RoutingError::PayloadTooLarge(self.payload.len()));
        }
        let mut out = Vec::with_capacity(
            MAGIC.len() + 33 + 16 + 1 + 1 + self.remaining.len() * 33 + 4 + self.payload.len(),
        );
        out.extend_from_slice(MAGIC);
        encode_peer(&mut out, &self.destination);
        out.push(self.ttl);
        out.push(self.remaining.len() as u8);
        for hop in &self.remaining {
            encode_peer(&mut out, hop);
        }
        out.extend_from_slice(&(self.payload.len() as u32).to_be_bytes());
        out.extend_from_slice(&self.payload);
        if out.len() > MAX_FRAME_SIZE {
            return Err(RoutingError::PayloadTooLarge(out.len()));
        }
        Ok(out)
    }

    /// Decode a message from its binary wire form. Refuses bad magic,
    /// truncation, trailing bytes, an empty remaining route, a route
    /// that does not end at the destination, and invalid peer ids.
    pub fn decode(bytes: &[u8]) -> Result<Self, RoutingError> {
        let mut r = Reader { buf: bytes, pos: 0 };
        let magic = r.take(MAGIC.len())?;
        if magic != MAGIC {
            return Err(RoutingError::Malformed("bad magic".to_string()));
        }
        let destination = r.peer()?;
        let ttl = r.u8()?;
        let hops = r.u8()? as usize;
        if hops == 0 {
            return Err(RoutingError::Malformed("empty remaining route".to_string()));
        }
        let mut remaining = Vec::with_capacity(hops);
        for _ in 0..hops {
            remaining.push(r.peer()?);
        }
        if !same_peer(remaining.last().unwrap(), &destination) {
            return Err(RoutingError::Malformed(
                "route does not end at the destination".to_string(),
            ));
        }
        let payload_len = r.u32_be()? as usize;
        if payload_len > MAX_FRAME_SIZE {
            return Err(RoutingError::Malformed(
                "payload length exceeds the frame cap".to_string(),
            ));
        }
        let payload = r.take(payload_len)?.to_vec();
        if r.left() != 0 {
            return Err(RoutingError::Malformed(
                "trailing bytes after payload".to_string(),
            ));
        }
        Ok(Self {
            destination,
            remaining,
            ttl,
            payload,
        })
    }
}

/// A peer id on the wire: 32-byte public key, u8 name length, name bytes.
fn encode_peer(out: &mut Vec<u8>, peer: &PeerId) {
    out.extend_from_slice(peer.public_key().as_bytes());
    let name = peer.display_name().as_bytes();
    debug_assert!(name.len() <= u8::MAX as usize);
    out.push(name.len() as u8);
    out.extend_from_slice(name);
}

/// A bounds-checked cursor over a byte slice being decoded.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Bytes still unread.
    fn left(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Take `n` bytes, or fail if fewer remain.
    fn take(&mut self, n: usize) -> Result<&'a [u8], RoutingError> {
        if n > self.left() {
            return Err(RoutingError::Malformed("truncated message".to_string()));
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, RoutingError> {
        Ok(self.take(1)?[0])
    }

    fn u32_be(&mut self) -> Result<u32, RoutingError> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn peer(&mut self) -> Result<PeerId, RoutingError> {
        let key_bytes = self.take(32)?;
        let mut key = [0u8; 32];
        key.copy_from_slice(key_bytes);
        let name_len = self.u8()? as usize;
        let name_bytes = self.take(name_len)?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|_| RoutingError::Malformed("peer name is not valid UTF-8".to_string()))?;
        let public_key = Ed25519PublicKey::from_bytes(&key)
            .map_err(|e| RoutingError::Malformed(format!("bad peer key: {e}")))?;
        PeerId::new(public_key, name)
            .map_err(|e| RoutingError::Malformed(format!("bad peer id: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_identity::Identity;

    fn peer(name: &str) -> PeerId {
        Identity::generate(name).unwrap().peer_id().clone()
    }

    fn sample_message() -> RoutedMessage {
        let b = peer("bob");
        let c = peer("carol");
        RoutedMessage::new(c.clone(), vec![b, c], 32, b"hello".to_vec())
    }

    #[test]
    fn envelope_round_trips() {
        let msg = sample_message();
        let bytes = msg.encode().unwrap();
        let back = RoutedMessage::decode(&bytes).unwrap();
        assert_eq!(msg, back);
    }

    #[test]
    fn envelope_round_trips_large_payload() {
        let b = peer("bob");
        let payload: Vec<u8> = (0..700 * 1024).map(|i| (i % 251) as u8).collect();
        let msg = RoutedMessage::new(b.clone(), vec![b.clone()], 5, payload.clone());
        let bytes = msg.encode().unwrap();
        let back = RoutedMessage::decode(&bytes).unwrap();
        assert_eq!(back.payload(), &payload);
        assert_eq!(back.destination(), &b);
        assert_eq!(back.ttl(), 5);
    }

    #[test]
    fn envelope_malformed_inputs_are_refused() {
        let msg = sample_message();
        let bytes = msg.encode().unwrap();

        // Bad magic.
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(matches!(
            RoutedMessage::decode(&bad),
            Err(RoutingError::Malformed(_))
        ));

        // Truncated at several points.
        for cut in [1, 5, 10, 30, bytes.len() / 2, bytes.len() - 1] {
            assert!(
                matches!(
                    RoutedMessage::decode(&bytes[..cut]),
                    Err(RoutingError::Malformed(_))
                ),
                "truncation at {cut} was accepted"
            );
        }

        // Empty input.
        assert!(matches!(
            RoutedMessage::decode(&[]),
            Err(RoutingError::Malformed(_))
        ));

        // Trailing bytes.
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(matches!(
            RoutedMessage::decode(&trailing),
            Err(RoutingError::Malformed(_))
        ));

        // A hand-built envelope with an empty remaining route.
        let mut empty_route = Vec::new();
        empty_route.extend_from_slice(MAGIC);
        encode_peer(&mut empty_route, msg.destination());
        empty_route.push(32); // ttl
        empty_route.push(0); // zero hops
        empty_route.extend_from_slice(&0u32.to_be_bytes());
        assert!(matches!(
            RoutedMessage::decode(&empty_route),
            Err(RoutingError::Malformed(_))
        ));

        // A route that does not end at the destination.
        let stranger = peer("mallory");
        let wrong_end =
            RoutedMessage::new(msg.destination().clone(), vec![stranger], 32, b"x".to_vec());
        let wrong_bytes = wrong_end.encode().unwrap();
        assert!(matches!(
            RoutedMessage::decode(&wrong_bytes),
            Err(RoutingError::Malformed(_))
        ));

        // A peer name length that runs past the end of the buffer.
        let mut overrun = Vec::new();
        overrun.extend_from_slice(MAGIC);
        overrun.extend_from_slice(&[7u8; 32]); // key
        overrun.push(200); // name length 200, but no name follows
        assert!(matches!(
            RoutedMessage::decode(&overrun),
            Err(RoutingError::Malformed(_))
        ));
    }

    #[test]
    fn payload_too_large_is_refused() {
        let b = peer("bob");
        let huge = vec![0u8; MAX_FRAME_SIZE + 1];
        let msg = RoutedMessage::new(b.clone(), vec![b], 1, huge);
        assert!(matches!(
            msg.encode(),
            Err(RoutingError::PayloadTooLarge(_))
        ));
    }
}
