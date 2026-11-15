//! A router: a local peer id plus an operator-populated table of
//! next-hop addresses (peer id -> socket address), like the carrier
//! gateway table. The router follows the route in each message exactly;
//! TTL is the backstop that kills loops. Forwarding is plaintext — see
//! the crate docs.

use std::collections::HashMap;
use std::net::SocketAddr;

use kernel_identity::PeerId;
use kernel_transport::{Connection, MAX_FRAME_SIZE};

use crate::{same_peer, Route, RoutedMessage, RoutingError, DEFAULT_TTL, MAX_ROUTE_HOPS};

/// Why a message was dropped instead of forwarded or delivered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropReason {
    /// TTL reached 0 at a hop that was not the destination.
    TtlExpired,
}

/// What happened to a message a router received.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ForwardOutcome {
    /// The message reached its destination: this node. The payload.
    Delivered(Vec<u8>),
    /// The message was sent on to the next hop, with the TTL it was
    /// sent with.
    Forwarded { to: PeerId, ttl: u8 },
    /// The message was dropped and counted, not forwarded.
    Dropped { reason: DropReason },
}

/// What happened to a message a router originated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SendOutcome {
    /// The destination was the local node; the payload, delivered
    /// without touching the network.
    Delivered(Vec<u8>),
    /// The message was sent to the first hop, with the TTL it was
    /// sent with.
    Sent { to: PeerId, ttl: u8 },
}

/// A node in a source-routed network.
pub struct Router {
    local: PeerId,
    /// Next-hop addresses, keyed by the peer's public key bytes.
    /// Operator-populated; nothing here discovers routes.
    next_hops: HashMap<[u8; 32], SocketAddr>,
    /// Messages dropped for TTL expiry (never forwarded, never counted
    /// as errors).
    dropped: u64,
}

impl Router {
    /// A router for `local`, with an empty next-hop table.
    pub fn new(local: PeerId) -> Self {
        Self {
            local,
            next_hops: HashMap::new(),
            dropped: 0,
        }
    }

    /// This router's local peer id.
    pub fn local(&self) -> &PeerId {
        &self.local
    }

    /// Teach the router where a peer lives. Overwrites any previous
    /// address for that peer.
    pub fn add_next_hop(&mut self, peer: &PeerId, addr: SocketAddr) {
        self.next_hops.insert(*peer.public_key().as_bytes(), addr);
    }

    /// Forget a peer's next-hop address. True if one was stored.
    pub fn remove_next_hop(&mut self, peer: &PeerId) -> bool {
        self.next_hops
            .remove(peer.public_key().as_bytes())
            .is_some()
    }

    /// Number of peers in the next-hop table.
    pub fn next_hop_count(&self) -> usize {
        self.next_hops.len()
    }

    /// Messages dropped for TTL expiry so far.
    pub fn dropped_count(&self) -> u64 {
        self.dropped
    }

    /// Build an envelope for `route` with `ttl`. The route must start
    /// at the local node and have at least one hop after it; the
    /// destination is the route's last entry. The envelope's remaining
    /// route is `route[1..]`.
    pub fn originate(
        &self,
        route: &[PeerId],
        ttl: u8,
        payload: &[u8],
    ) -> Result<RoutedMessage, RoutingError> {
        if route.is_empty() {
            return Err(RoutingError::Malformed("route is empty".to_string()));
        }
        if route.len() == 1 {
            return Err(RoutingError::Malformed(
                "route names only the local node; nothing to send".to_string(),
            ));
        }
        if !same_peer(&route[0], &self.local) {
            return Err(RoutingError::NotInRoute);
        }
        if same_peer(&route[1], &self.local) {
            return Err(RoutingError::SelfLoop(route[1].clone()));
        }
        if route.len() - 1 > MAX_ROUTE_HOPS {
            return Err(RoutingError::Malformed(format!(
                "route has {} hops, more than the maximum {MAX_ROUTE_HOPS}",
                route.len() - 1
            )));
        }
        if payload.len() > MAX_FRAME_SIZE {
            return Err(RoutingError::PayloadTooLarge(payload.len()));
        }
        Ok(RoutedMessage::new(
            route[route.len() - 1].clone(),
            route[1..].to_vec(),
            ttl,
            payload.to_vec(),
        ))
    }

    /// Handle a message that arrived from `from`, deciding delivery,
    /// forwarding, or drop:
    ///
    /// 1. If the message's next hop is not this node, refuse it
    ///    ([`RoutingError::NotInRoute`]) — it was misdelivered, and
    ///    forwarding it would send it somewhere the route never said.
    /// 2. If the destination is this node, deliver the payload.
    /// 3. If TTL is 0, drop the message and count it.
    /// 4. Otherwise advance the route by one hop, decrement TTL, and
    ///    send it to the next hop's address. An unknown next hop is a
    ///    clean [`RoutingError::NoRoute`]; a next hop that is this node
    ///    itself is refused as [`RoutingError::SelfLoop`].
    ///
    /// `from` is where the message arrived from. The route is
    /// authoritative: a hop forwards only where the route says, and
    /// never back to `from` unless the route explicitly names that
    /// peer as the next hop. TTL is the backstop for routes that loop.
    pub fn forward(
        &mut self,
        msg: &RoutedMessage,
        _from: SocketAddr,
    ) -> Result<ForwardOutcome, RoutingError> {
        if msg.remaining().is_empty() {
            return Err(RoutingError::Malformed(
                "routed message has no remaining hops".to_string(),
            ));
        }
        if !same_peer(&msg.remaining()[0], &self.local) {
            return Err(RoutingError::NotInRoute);
        }
        if same_peer(msg.destination(), &self.local) {
            return Ok(ForwardOutcome::Delivered(msg.payload().to_vec()));
        }
        if msg.ttl() == 0 {
            self.dropped += 1;
            return Ok(ForwardOutcome::Dropped {
                reason: DropReason::TtlExpired,
            });
        }
        let remaining = &msg.remaining()[1..];
        if remaining.is_empty() {
            return Err(RoutingError::Malformed(
                "route ends before the destination".to_string(),
            ));
        }
        let next = &remaining[0];
        if same_peer(next, &self.local) {
            return Err(RoutingError::SelfLoop(next.clone()));
        }
        let addr = self
            .next_hops
            .get(next.public_key().as_bytes())
            .copied()
            .ok_or_else(|| RoutingError::NoRoute(next.clone()))?;
        let forwarded = RoutedMessage::new(
            msg.destination().clone(),
            remaining.to_vec(),
            msg.ttl() - 1,
            msg.payload().to_vec(),
        );
        let bytes = forwarded.encode()?;
        let mut conn = Connection::connect(addr)?;
        conn.send(&bytes)?;
        Ok(ForwardOutcome::Forwarded {
            to: next.clone(),
            ttl: msg.ttl() - 1,
        })
    }

    /// Decode raw bytes received on the wire and handle the message.
    /// A malformed envelope is refused, never forwarded.
    pub fn handle_bytes(
        &mut self,
        bytes: &[u8],
        from: SocketAddr,
    ) -> Result<ForwardOutcome, RoutingError> {
        let msg = RoutedMessage::decode(bytes)?;
        self.forward(&msg, from)
    }
}

/// Originate a message from the local node and hand it to the first hop
/// over a real transport connection, with [`DEFAULT_TTL`]. A route whose
/// only entry is the local node is delivered locally without touching
/// the network.
pub fn send_routed(
    router: &Router,
    route: &Route,
    payload: &[u8],
) -> Result<SendOutcome, RoutingError> {
    send_routed_with_ttl(router, route, DEFAULT_TTL, payload)
}

/// Like [`send_routed`], with an explicit TTL.
pub fn send_routed_with_ttl(
    router: &Router,
    route: &Route,
    ttl: u8,
    payload: &[u8],
) -> Result<SendOutcome, RoutingError> {
    if route.is_empty() {
        return Err(RoutingError::Malformed("route is empty".to_string()));
    }
    if !same_peer(&route[0], router.local()) {
        return Err(RoutingError::NotInRoute);
    }
    if route.len() == 1 {
        return Ok(SendOutcome::Delivered(payload.to_vec()));
    }
    let msg = router.originate(route, ttl, payload)?;
    let first = &route[1];
    let addr = router
        .next_hops
        .get(first.public_key().as_bytes())
        .copied()
        .ok_or_else(|| RoutingError::NoRoute(first.clone()))?;
    let bytes = msg.encode()?;
    let mut conn = Connection::connect(addr)?;
    conn.send(&bytes)?;
    Ok(SendOutcome::Sent {
        to: first.clone(),
        ttl,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_identity::Identity;
    use kernel_transport::Listener;
    use std::sync::mpsc;
    use std::time::Duration;

    /// One test node: an identity, a router, and a bound listener.
    fn test_node(name: &str) -> (Identity, Router, Listener, SocketAddr) {
        let identity = Identity::generate(name).unwrap();
        let listener = Listener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let router = Router::new(identity.peer_id().clone());
        (identity, router, listener, addr)
    }

    /// Spawn `f` on a thread; its result arrives on the channel.
    fn spawn_server<F, T>(f: F) -> mpsc::Receiver<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx
    }

    /// Wait for a server thread's result, failing fast instead of
    /// hanging the suite if the thread is stuck in accept().
    fn recv_timeout<T>(rx: mpsc::Receiver<T>) -> T {
        rx.recv_timeout(Duration::from_secs(10))
            .expect("server thread did not finish in time")
    }

    /// Serve `n` incoming connections on `listener` with `router`,
    /// returning each outcome plus the router's drop count.
    fn serve_n(
        mut router: Router,
        listener: Listener,
        n: usize,
    ) -> (Vec<Result<ForwardOutcome, RoutingError>>, u64) {
        let mut outcomes = Vec::new();
        for _ in 0..n {
            let mut conn = match listener.accept() {
                Ok(conn) => conn,
                Err(e) => {
                    outcomes.push(Err(RoutingError::Transport(e)));
                    continue;
                }
            };
            let from = match conn.peer_addr() {
                Ok(addr) => addr,
                Err(e) => {
                    outcomes.push(Err(RoutingError::Transport(e)));
                    continue;
                }
            };
            let bytes = match conn.recv() {
                Ok(bytes) => bytes,
                Err(e) => {
                    outcomes.push(Err(RoutingError::Transport(e)));
                    continue;
                }
            };
            outcomes.push(router.handle_bytes(&bytes, from));
        }
        (outcomes, router.dropped_count())
    }

    #[test]
    fn three_nodes_relay_a_message() {
        let (a_id, a_router, _a_listener, _a_addr) = test_node("alice");
        let (b_id, mut b_router, b_listener, b_addr) = test_node("bob");
        let (c_id, c_router, c_listener, _c_addr) = test_node("carol");
        let (a_peer, b_peer, c_peer) = (
            a_id.peer_id().clone(),
            b_id.peer_id().clone(),
            c_id.peer_id().clone(),
        );
        b_router.add_next_hop(&c_peer, c_listener.local_addr().unwrap());

        let c_rx = spawn_server(move || serve_n(c_router, c_listener, 1));
        let b_rx = spawn_server(move || serve_n(b_router, b_listener, 1));

        let mut sender = a_router;
        sender.add_next_hop(&b_peer, b_addr);
        let route = vec![a_peer, b_peer.clone(), c_peer.clone()];
        let sent = send_routed(&sender, &route, b"hello via bob").unwrap();
        assert!(matches!(sent, SendOutcome::Sent { .. }));

        let (b_outcomes, b_dropped) = recv_timeout(b_rx);
        assert_eq!(b_dropped, 0);
        assert_eq!(b_outcomes.len(), 1);
        assert!(
            matches!(&b_outcomes[0], Ok(ForwardOutcome::Forwarded { to, ttl }) if same_peer(to, &c_peer) && *ttl == DEFAULT_TTL - 1),
            "unexpected B outcome: {:?}",
            b_outcomes[0]
        );

        let (c_outcomes, _c_dropped) = recv_timeout(c_rx);
        assert_eq!(c_outcomes.len(), 1);
        assert!(
            matches!(&c_outcomes[0], Ok(ForwardOutcome::Delivered(payload)) if payload == b"hello via bob"),
            "unexpected C outcome: {:?}",
            c_outcomes[0]
        );
    }

    #[test]
    fn unknown_destination_errors_cleanly() {
        let (a_id, a_router, _l, _a) = test_node("alice");
        let stranger = Identity::generate("stranger").unwrap();
        let route = vec![a_id.peer_id().clone(), stranger.peer_id().clone()];
        let err = send_routed(&a_router, &route, b"hi").unwrap_err();
        assert!(
            matches!(&err, RoutingError::NoRoute(peer) if same_peer(peer, stranger.peer_id())),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn missing_middle_hop_errors() {
        let (a_id, a_router, _l, _a) = test_node("alice");
        let (b_id, b_router, b_listener, b_addr) = test_node("bob");
        let c_id = Identity::generate("carol").unwrap();
        let (a_peer, b_peer, c_peer) = (
            a_id.peer_id().clone(),
            b_id.peer_id().clone(),
            c_id.peer_id().clone(),
        );
        // B knows nobody: no next hop for Carol.
        assert_eq!(b_router.next_hop_count(), 0);

        let b_rx = spawn_server(move || serve_n(b_router, b_listener, 1));

        let mut sender = a_router;
        sender.add_next_hop(&b_peer, b_addr);
        let route = vec![a_peer, b_peer, c_peer.clone()];
        send_routed(&sender, &route, b"never arrives").unwrap();

        let (b_outcomes, b_dropped) = recv_timeout(b_rx);
        assert_eq!(b_dropped, 0);
        assert_eq!(b_outcomes.len(), 1);
        assert!(
            matches!(&b_outcomes[0], Err(RoutingError::NoRoute(peer)) if same_peer(peer, &c_peer)),
            "unexpected B outcome: {:?}",
            b_outcomes[0]
        );
    }

    #[test]
    fn ttl_expiry_drops_a_looping_message() {
        let (a_id, a_router, a_listener, a_addr) = test_node("alice");
        let (b_id, b_router, b_listener, b_addr) = test_node("bob");
        let c_id = Identity::generate("carol").unwrap();
        let (a_peer, b_peer, c_peer) = (
            a_id.peer_id().clone(),
            b_id.peer_id().clone(),
            c_id.peer_id().clone(),
        );

        // A serves one forward (back to B); B serves two (forward, then drop).
        let mut a_server = a_router;
        a_server.add_next_hop(&b_peer, b_addr);
        let mut b_server = b_router;
        b_server.add_next_hop(&a_peer, a_addr);

        let a_rx = spawn_server(move || serve_n(a_server, a_listener, 1));
        let b_rx = spawn_server(move || serve_n(b_server, b_listener, 2));

        let mut sender = Router::new(a_peer.clone());
        sender.add_next_hop(&b_peer, b_addr);
        let route = vec![
            a_peer,
            b_peer,
            a_id.peer_id().clone(),
            b_id.peer_id().clone(),
            c_peer,
        ];
        let sent = send_routed_with_ttl(&sender, &route, 2, b"round and round").unwrap();
        assert!(matches!(sent, SendOutcome::Sent { .. }));

        let (a_outcomes, a_dropped) = recv_timeout(a_rx);
        assert_eq!(a_dropped, 0);
        assert_eq!(a_outcomes.len(), 1);
        assert!(
            matches!(&a_outcomes[0], Ok(ForwardOutcome::Forwarded { ttl, .. }) if *ttl == 0),
            "unexpected A outcome: {:?}",
            a_outcomes[0]
        );

        let (b_outcomes, b_dropped) = recv_timeout(b_rx);
        assert_eq!(b_dropped, 1, "the loop should be dropped exactly once");
        assert_eq!(b_outcomes.len(), 2);
        assert!(
            matches!(&b_outcomes[0], Ok(ForwardOutcome::Forwarded { ttl, .. }) if *ttl == 1),
            "unexpected first B outcome: {:?}",
            b_outcomes[0]
        );
        assert!(
            matches!(
                &b_outcomes[1],
                Ok(ForwardOutcome::Dropped {
                    reason: DropReason::TtlExpired
                })
            ),
            "unexpected second B outcome: {:?}",
            b_outcomes[1]
        );
    }

    #[test]
    fn local_delivery_without_forwarding() {
        let (a_id, a_router, _l, _a) = test_node("alice");
        // No next hops at all: if it tried to send, it would fail.
        assert_eq!(a_router.next_hop_count(), 0);
        let out = send_routed(&a_router, &vec![a_id.peer_id().clone()], b"to myself").unwrap();
        assert!(
            matches!(&out, SendOutcome::Delivered(payload) if payload == b"to myself"),
            "unexpected outcome: {out:?}"
        );
        assert_eq!(a_router.dropped_count(), 0);
    }

    #[test]
    fn route_not_starting_with_local_is_refused() {
        let (_a_id, a_router, _l, _a) = test_node("alice");
        let b_id = Identity::generate("bob").unwrap();
        let c_id = Identity::generate("carol").unwrap();
        let route = vec![b_id.peer_id().clone(), c_id.peer_id().clone()];
        let err = send_routed(&a_router, &route, b"hi").unwrap_err();
        assert!(matches!(err, RoutingError::NotInRoute));

        // And on the receive path: a message whose next hop is Carol,
        // arriving at Alice, is refused rather than forwarded.
        let msg = RoutedMessage::new(
            c_id.peer_id().clone(),
            vec![c_id.peer_id().clone()],
            32,
            b"wrong node".to_vec(),
        );
        let mut router = a_router;
        let err = router
            .handle_bytes(&msg.encode().unwrap(), "127.0.0.1:9".parse().unwrap())
            .unwrap_err();
        assert!(matches!(err, RoutingError::NotInRoute));
    }

    #[test]
    fn self_loop_in_route_is_refused() {
        let (a_id, a_router, _l, _a) = test_node("alice");
        let (b_id, b_router, b_listener, b_addr) = test_node("bob");
        let c_id = Identity::generate("carol").unwrap();
        let (a_peer, b_peer, c_peer) = (
            a_id.peer_id().clone(),
            b_id.peer_id().clone(),
            c_id.peer_id().clone(),
        );

        let b_rx = spawn_server(move || serve_n(b_router, b_listener, 1));

        let mut sender = a_router;
        sender.add_next_hop(&b_peer, b_addr);
        // Bob's next hop after himself is Bob again.
        let route = vec![a_peer, b_peer.clone(), b_peer, c_peer];
        send_routed(&sender, &route, b"around myself").unwrap();

        let (b_outcomes, b_dropped) = recv_timeout(b_rx);
        assert_eq!(b_dropped, 0);
        assert_eq!(b_outcomes.len(), 1);
        assert!(
            matches!(&b_outcomes[0], Err(RoutingError::SelfLoop(peer)) if same_peer(peer, &b_id.peer_id().clone())),
            "unexpected B outcome: {:?}",
            b_outcomes[0]
        );
    }

    #[test]
    fn ttl_zero_still_delivers_to_the_first_hop() {
        let (a_id, a_router, _l, _a) = test_node("alice");
        let (b_id, b_router, b_listener, b_addr) = test_node("bob");
        let (a_peer, b_peer) = (a_id.peer_id().clone(), b_id.peer_id().clone());

        let b_rx = spawn_server(move || serve_n(b_router, b_listener, 1));

        let mut sender = a_router;
        sender.add_next_hop(&b_peer, b_addr);
        let route = vec![a_peer, b_peer];
        let sent = send_routed_with_ttl(&sender, &route, 0, b"direct").unwrap();
        assert!(matches!(sent, SendOutcome::Sent { ttl: 0, .. }));

        let (b_outcomes, b_dropped) = recv_timeout(b_rx);
        assert_eq!(b_dropped, 0);
        assert!(
            matches!(&b_outcomes[0], Ok(ForwardOutcome::Delivered(payload)) if payload == b"direct"),
            "unexpected B outcome: {:?}",
            b_outcomes[0]
        );
    }

    #[test]
    fn ttl_zero_drops_before_the_second_hop() {
        let (a_id, a_router, _l, _a) = test_node("alice");
        let (b_id, b_router, b_listener, b_addr) = test_node("bob");
        let c_id = Identity::generate("carol").unwrap();
        let (a_peer, b_peer, c_peer) = (
            a_id.peer_id().clone(),
            b_id.peer_id().clone(),
            c_id.peer_id().clone(),
        );

        let b_rx = spawn_server(move || serve_n(b_router, b_listener, 1));

        let mut sender = a_router;
        sender.add_next_hop(&b_peer, b_addr);
        let route = vec![a_peer, b_peer, c_peer];
        send_routed_with_ttl(&sender, &route, 0, b"too far").unwrap();

        let (b_outcomes, b_dropped) = recv_timeout(b_rx);
        assert_eq!(b_dropped, 1);
        assert!(
            matches!(
                &b_outcomes[0],
                Ok(ForwardOutcome::Dropped {
                    reason: DropReason::TtlExpired
                })
            ),
            "unexpected B outcome: {:?}",
            b_outcomes[0]
        );
    }

    #[test]
    fn originate_validates_the_route() {
        let (a_id, a_router, _l, _a) = test_node("alice");
        let b_id = Identity::generate("bob").unwrap();
        let c_id = Identity::generate("carol").unwrap();
        let (a_peer, b_peer, c_peer) = (
            a_id.peer_id().clone(),
            b_id.peer_id().clone(),
            c_id.peer_id().clone(),
        );

        assert!(matches!(
            a_router.originate(&[], 32, b"x"),
            Err(RoutingError::Malformed(_))
        ));
        assert!(matches!(
            a_router.originate(&[b_peer.clone(), c_peer.clone()], 32, b"x"),
            Err(RoutingError::NotInRoute)
        ));
        assert!(matches!(
            a_router.originate(std::slice::from_ref(&a_peer), 32, b"x"),
            Err(RoutingError::Malformed(_))
        ));
        assert!(matches!(
            a_router.originate(&[a_peer.clone(), a_peer.clone(), c_peer.clone()], 32, b"x"),
            Err(RoutingError::SelfLoop(_))
        ));
        let ok = a_router
            .originate(&[a_peer, b_peer, c_peer.clone()], 32, b"x")
            .unwrap();
        assert_eq!(ok.destination(), &c_peer);
        assert_eq!(ok.remaining().len(), 2);
    }

    #[test]
    fn next_hop_table_management() {
        let (a_id, mut router, _l, _a) = test_node("alice");
        let b_id = Identity::generate("bob").unwrap();
        let addr: SocketAddr = "127.0.0.1:1234".parse().unwrap();
        assert_eq!(router.next_hop_count(), 0);
        router.add_next_hop(b_id.peer_id(), addr);
        assert_eq!(router.next_hop_count(), 1);
        // Re-adding overwrites, does not duplicate.
        router.add_next_hop(b_id.peer_id(), addr);
        assert_eq!(router.next_hop_count(), 1);
        assert!(router.remove_next_hop(b_id.peer_id()));
        assert_eq!(router.next_hop_count(), 0);
        assert!(!router.remove_next_hop(b_id.peer_id()));
        assert_eq!(router.local(), a_id.peer_id());
    }
}
