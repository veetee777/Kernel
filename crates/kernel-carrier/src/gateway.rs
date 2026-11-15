//! Gateway table v1: peer public key to socket address.
//!
//! Populated by the operator, not by the network. `set` requires the peer to
//! be pinned, so an address can only be recorded for a peer the node already
//! trusts. Routes are keyed by the Ed25519 public key, the same key the pins
//! use. The K•Talk PeerId is a hash of that key, so the caller can map one to
//! the other.
//!
//! There is no announce message and no discovery. An unauthenticated announce
//! would let anyone redirect traffic meant for a peer. A safe announce (a peer
//! sends only its own address, over an authenticated session) is a later
//! step. DHT and discovery are Phase 3.

use std::collections::HashMap;
use std::net::SocketAddr;

use crate::error::CarrierError;
use crate::pins::{PinSet, PublicKey};

#[derive(Debug, Default)]
pub struct GatewayTable {
    routes: HashMap<PublicKey, SocketAddr>,
}

impl GatewayTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record where to reach `peer`. Refused unless `peer` is pinned.
    /// Replaces any earlier address for the same peer.
    pub fn set(
        &mut self,
        pins: &PinSet,
        peer: PublicKey,
        addr: SocketAddr,
    ) -> Result<(), CarrierError> {
        pins.check(&peer)?;
        self.routes.insert(peer, addr);
        Ok(())
    }

    /// The recorded address for `peer`, if any.
    pub fn lookup(&self, peer: &PublicKey) -> Option<SocketAddr> {
        self.routes.get(peer).copied()
    }

    /// Forget the address for `peer`. Returns the old address, if any.
    pub fn remove(&mut self, peer: &PublicKey) -> Option<SocketAddr> {
        self.routes.remove(peer)
    }

    /// Drop every route whose peer is no longer pinned (for example, revoked).
    pub fn prune(&mut self, pins: &PinSet) {
        self.routes.retain(|peer, _| pins.check(peer).is_ok());
    }

    pub fn len(&self) -> usize {
        self.routes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: PublicKey = [0x0a; 32];
    const B: PublicKey = [0x0b; 32];

    fn addr(port: u16) -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], port))
    }

    fn pinned(keys: &[PublicKey]) -> PinSet {
        let mut p = PinSet::new();
        for k in keys {
            p.pin(*k).expect("pin");
        }
        p
    }

    #[test]
    fn refuses_an_address_for_an_unpinned_peer() {
        let pins = PinSet::new();
        let mut t = GatewayTable::new();
        assert_eq!(
            t.set(&pins, A, addr(7000)).err().expect("refuse"),
            CarrierError::UnpinnedPeer
        );
        assert!(t.is_empty());
    }

    #[test]
    fn routes_each_peer_to_its_own_address() {
        let pins = pinned(&[A, B]);
        let mut t = GatewayTable::new();
        t.set(&pins, A, addr(7001)).expect("set A");
        t.set(&pins, B, addr(7002)).expect("set B");
        assert_eq!(t.lookup(&A), Some(addr(7001)));
        assert_eq!(t.lookup(&B), Some(addr(7002)));
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn unknown_peer_has_no_route() {
        let t = GatewayTable::new();
        assert_eq!(t.lookup(&A), None);
    }

    #[test]
    fn set_replaces_and_remove_clears() {
        let pins = pinned(&[A]);
        let mut t = GatewayTable::new();
        t.set(&pins, A, addr(7003)).expect("first");
        t.set(&pins, A, addr(7004)).expect("replace");
        assert_eq!(t.lookup(&A), Some(addr(7004)));
        assert_eq!(t.remove(&A), Some(addr(7004)));
        assert_eq!(t.lookup(&A), None);
    }

    #[test]
    fn prune_drops_revoked_peers_only() {
        let mut pins = pinned(&[A, B]);
        let mut t = GatewayTable::new();
        t.set(&pins, A, addr(7005)).expect("A");
        t.set(&pins, B, addr(7006)).expect("B");
        pins.revoke(A);
        t.prune(&pins);
        assert_eq!(t.lookup(&A), None);
        assert_eq!(t.lookup(&B), Some(addr(7006)));
    }
}
