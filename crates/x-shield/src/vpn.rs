//! The VPN tunnel contract.
//!
//! [`VpnTunnel`] is the interface a real VPN implementation must
//! satisfy: connect, disconnect, report status, and route packets.
//! This crate ships only [`MockTunnel`], an in-process stand-in
//! that proves the interface works: it connects, wraps each packet
//! in a small `KVPN1` frame, records what it routed, and
//! disconnects.
//!
//! There is **no real WireGuard here, no network, and nothing here
//! touches the operating system's network layer**. A real tunnel —
//! WireGuard, or the platform VPN service on Android — implements
//! `VpnTunnel` and is plugged in by the caller. Until then, the
//! mock is the contract: any code written against the trait works
//! against the real thing, and the mock's tests prove the shape of
//! that code.

use crate::ShieldError;

/// Largest packet the tunnel accepts, in bytes.
pub const MAX_PACKET_BYTES: usize = 65_535;

/// Magic at the start of every packet frame the mock tunnel emits.
const VPN_MAGIC: &[u8; 5] = b"KVPN1";

/// Whether a tunnel is up or down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VpnStatus {
    /// The tunnel is down; routing is refused.
    Disconnected,
    /// The tunnel is up; packets route through it.
    Connected,
}

/// The contract every VPN tunnel must satisfy, real or mock.
///
/// * `connect` brings the tunnel up (idempotent: connecting an
///   already-connected tunnel is a no-op success).
/// * `disconnect` brings it down (idempotent).
/// * `status` reports the current state.
/// * `route` sends one packet through the tunnel and returns what
///   comes out the other side. Routing while disconnected is an
///   error, as is a packet over [`MAX_PACKET_BYTES`].
pub trait VpnTunnel {
    /// Bring the tunnel up.
    fn connect(&mut self) -> Result<(), ShieldError>;
    /// Bring the tunnel down.
    fn disconnect(&mut self);
    /// The tunnel's current state.
    fn status(&self) -> VpnStatus;
    /// Send one packet through the tunnel.
    fn route(&mut self, packet: &[u8]) -> Result<Vec<u8>, ShieldError>;
    /// A human name for this tunnel implementation, e.g. `"mock"`.
    fn name(&self) -> &str;
}

/// An in-process mock tunnel: it connects, wraps packets in a
/// `KVPN1` frame (`KVPN1` + payload length u32 LE + payload),
/// records the original packets it routed, and disconnects. It
/// sends nothing anywhere — the frame is the proof that routing
/// happened, and [`MockTunnel::unwrap_wrapped`] recovers the
/// original packet from a frame.
#[derive(Clone, Debug, Default)]
pub struct MockTunnel {
    connected: bool,
    /// Original packets routed, in order.
    routed: Vec<Vec<u8>>,
    /// Payload bytes routed in total.
    bytes_routed: u64,
}

impl MockTunnel {
    /// A new, disconnected mock tunnel.
    pub fn new() -> Self {
        Self::default()
    }

    /// The original packets routed through this tunnel, in order.
    pub fn routed_packets(&self) -> &[Vec<u8>] {
        &self.routed
    }

    /// Number of packets routed.
    pub fn routed_count(&self) -> usize {
        self.routed.len()
    }

    /// Payload bytes routed in total.
    pub fn bytes_routed(&self) -> u64 {
        self.bytes_routed
    }

    /// Recover the original packet from a frame produced by
    /// [`MockTunnel::route`]. Refuses anything that is not exactly
    /// that frame format.
    pub fn unwrap_wrapped(wrapped: &[u8]) -> Result<Vec<u8>, ShieldError> {
        let err = |msg: String| ShieldError::Vpn(msg);
        if wrapped.len() < 5 + 4 || &wrapped[..5] != VPN_MAGIC {
            return Err(err("not a KVPN1 frame".to_string()));
        }
        let len = u32::from_le_bytes(wrapped[5..9].try_into().expect("4-byte slice")) as usize;
        if wrapped.len() != 5 + 4 + len {
            return Err(err(format!(
                "frame is {} bytes, expected {}",
                wrapped.len(),
                5 + 4 + len
            )));
        }
        Ok(wrapped[9..].to_vec())
    }
}

impl VpnTunnel for MockTunnel {
    fn connect(&mut self) -> Result<(), ShieldError> {
        self.connected = true;
        Ok(())
    }

    fn disconnect(&mut self) {
        self.connected = false;
    }

    fn status(&self) -> VpnStatus {
        if self.connected {
            VpnStatus::Connected
        } else {
            VpnStatus::Disconnected
        }
    }

    fn route(&mut self, packet: &[u8]) -> Result<Vec<u8>, ShieldError> {
        if !self.connected {
            return Err(ShieldError::Vpn("tunnel is not connected".to_string()));
        }
        if packet.len() > MAX_PACKET_BYTES {
            return Err(ShieldError::Vpn(format!(
                "packet is {} bytes, max is {MAX_PACKET_BYTES}",
                packet.len()
            )));
        }
        let mut out = Vec::with_capacity(5 + 4 + packet.len());
        out.extend_from_slice(VPN_MAGIC);
        out.extend_from_slice(&(packet.len() as u32).to_le_bytes());
        out.extend_from_slice(packet);
        self.bytes_routed += packet.len() as u64;
        self.routed.push(packet.to_vec());
        Ok(out)
    }

    fn name(&self) -> &str {
        "mock"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_tunnel_connects_routes_and_disconnects() {
        let mut tunnel = MockTunnel::new();
        assert_eq!(tunnel.status(), VpnStatus::Disconnected);
        assert_eq!(tunnel.name(), "mock");
        // Routing while down is refused.
        assert!(tunnel.route(b"hello").is_err());
        assert!(tunnel.connect().is_ok());
        assert_eq!(tunnel.status(), VpnStatus::Connected);
        // Connecting again is a no-op success.
        assert!(tunnel.connect().is_ok());
        let wrapped = tunnel.route(b"hello").unwrap();
        assert_eq!(wrapped.len(), 5 + 4 + 5);
        assert_eq!(MockTunnel::unwrap_wrapped(&wrapped).unwrap(), b"hello");
        assert_eq!(tunnel.routed_count(), 1);
        assert_eq!(tunnel.bytes_routed(), 5);
        assert_eq!(tunnel.routed_packets()[0], b"hello");
        tunnel.route(b"second packet").unwrap();
        assert_eq!(tunnel.routed_count(), 2);
        assert_eq!(tunnel.bytes_routed(), 5 + 13);
        tunnel.disconnect();
        assert_eq!(tunnel.status(), VpnStatus::Disconnected);
        assert!(tunnel.route(b"hello").is_err());
        // Disconnecting again is a no-op.
        tunnel.disconnect();
        assert_eq!(tunnel.status(), VpnStatus::Disconnected);
        // Reconnecting works.
        assert!(tunnel.connect().is_ok());
        assert!(tunnel.route(b"back").is_ok());
        assert_eq!(tunnel.routed_count(), 3);
    }

    #[test]
    fn oversized_packet_is_refused_and_empty_packet_routes() {
        let mut tunnel = MockTunnel::new();
        tunnel.connect().unwrap();
        let big = vec![0u8; MAX_PACKET_BYTES + 1];
        assert!(tunnel.route(&big).is_err());
        assert_eq!(tunnel.routed_count(), 0);
        // The maximum size is accepted.
        let max = vec![7u8; MAX_PACKET_BYTES];
        let wrapped = tunnel.route(&max).unwrap();
        assert_eq!(MockTunnel::unwrap_wrapped(&wrapped).unwrap(), max);
        // An empty packet routes (and unwraps to empty).
        let wrapped = tunnel.route(b"").unwrap();
        assert_eq!(MockTunnel::unwrap_wrapped(&wrapped).unwrap(), b"");
    }

    #[test]
    fn unwrap_wrapped_refuses_non_frames() {
        assert!(MockTunnel::unwrap_wrapped(b"garbage").is_err());
        assert!(MockTunnel::unwrap_wrapped(b"KVPN1").is_err());
        let mut tunnel = MockTunnel::new();
        tunnel.connect().unwrap();
        let wrapped = tunnel.route(b"data").unwrap();
        // A truncated frame is refused.
        assert!(MockTunnel::unwrap_wrapped(&wrapped[..wrapped.len() - 1]).is_err());
        // Trailing bytes are refused.
        let mut trailing = wrapped.clone();
        trailing.push(0);
        assert!(MockTunnel::unwrap_wrapped(&trailing).is_err());
        // A frame with a doctored length is refused.
        let mut doctored = wrapped.clone();
        doctored[5] = 99;
        assert!(MockTunnel::unwrap_wrapped(&doctored).is_err());
    }
}
