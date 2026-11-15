//! Distributed storage replication for KERNEL.
//!
//! A block is content-addressed (CID = SHA-256), so any peer holding the
//! bytes can serve them and any peer receiving them can verify them.
//! Replication copies a block to N peers; fetching tries the peers that
//! are known to hold it and takes the first one that answers, so one
//! peer going offline does not break reads from the rest.
//!
//! Wire protocol (inside the length-prefixed frames of
//! `kernel_transport`):
//!
//! * STORE request: `0x01` + CID (32 bytes) + block bytes.
//!   Response: `0x10` (OK) + the responder's `KID1` public identity,
//!   or `0x13` on failure.
//! * FETCH request: `0x02` + CID (32 bytes).
//!   Response: `0x11` (HIT) + identity length (u16 LE) + the
//!   responder's `KID1` public identity + block bytes, or `0x12` (MISS).
//!   The identity in a HIT lets the fetcher record who holds the block.
//!
//! A node tracks which peers it knows hold which CIDs (learned from
//! successful replications and fetches). That is a hint for choosing
//! whom to ask, not a guarantee — a peer may have dropped a block, and
//! a fetch falls through to the next holder.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use kernel_identity::{PeerId, PublicIdentity};
use kernel_node::{Node, NodeError};
use kernel_storage::Cid;
use kernel_transport::{Connection, Listener, TransportError};

/// Wire opcodes.
const OP_STORE: u8 = 0x01;
const OP_FETCH: u8 = 0x02;
const RESP_OK: u8 = 0x10;
const RESP_HIT: u8 = 0x11;
const RESP_MISS: u8 = 0x12;
const RESP_FAIL: u8 = 0x13;

/// Errors returned by replication operations.
#[derive(Debug)]
pub enum ReplicationError {
    /// Wraps a [`NodeError`].
    Node(NodeError),
    /// Wraps a [`TransportError`].
    Transport(TransportError),
    /// The block is not held locally (for replicate) or by any peer
    /// tried (for fetch).
    BlockNotHeld(Cid),
    /// Replication reached no peer at all.
    NoPeersAccepted,
    /// A peer answered with bytes this node could not parse.
    InvalidResponse(String),
}

impl fmt::Display for ReplicationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReplicationError::Node(e) => write!(f, "node error: {e}"),
            ReplicationError::Transport(e) => write!(f, "transport error: {e}"),
            ReplicationError::BlockNotHeld(cid) => {
                write!(f, "no peer holds block {cid}")
            }
            ReplicationError::NoPeersAccepted => {
                write!(f, "no peer accepted the replication")
            }
            ReplicationError::InvalidResponse(msg) => {
                write!(f, "invalid response from peer: {msg}")
            }
        }
    }
}

impl std::error::Error for ReplicationError {}

impl From<NodeError> for ReplicationError {
    fn from(e: NodeError) -> Self {
        ReplicationError::Node(e)
    }
}

impl From<TransportError> for ReplicationError {
    fn from(e: TransportError) -> Self {
        ReplicationError::Transport(e)
    }
}

/// A replica node: a KERNEL node (identity + content store) plus a TCP
/// server that answers STORE and FETCH requests from other replicas.
/// The server runs on its own thread; [`ReplicaNode::stop`] shuts it
/// down.
pub struct ReplicaNode {
    node: Arc<Mutex<Node>>,
    holders: Arc<Mutex<HashMap<Cid, HashSet<PeerId>>>>,
    addr: SocketAddr,
    stop_flag: Arc<AtomicBool>,
    server: Option<JoinHandle<()>>,
}

impl ReplicaNode {
    /// Start a replica node listening on loopback (ephemeral port).
    pub fn spawn(display_name: &str) -> Result<Self, ReplicationError> {
        let node = Arc::new(Mutex::new(Node::new(display_name)?));
        let holders = Arc::new(Mutex::new(HashMap::new()));
        let listener = Listener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        listener.set_nonblocking(true)?;
        let stop_flag = Arc::new(AtomicBool::new(false));

        let server_node = Arc::clone(&node);
        let server_stop = Arc::clone(&stop_flag);
        let server = thread::spawn(move || {
            serve(listener, server_node, server_stop);
        });

        Ok(Self {
            node,
            holders,
            addr,
            stop_flag,
            server: Some(server),
        })
    }

    /// The loopback address this node listens on.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// This node's public peer id.
    pub fn peer_id(&self) -> PeerId {
        self.node.lock().unwrap().peer_id().clone()
    }

    /// Store bytes locally, returning their CID.
    pub fn store_local(&self, data: &[u8]) -> Cid {
        self.node.lock().unwrap().store_local(data)
    }

    /// Fetch bytes this node already holds.
    pub fn get_local(&self, cid: &Cid) -> Result<Vec<u8>, ReplicationError> {
        Ok(self.node.lock().unwrap().get_local(cid)?)
    }

    /// True if this node holds a block with this CID.
    pub fn has_local(&self, cid: &Cid) -> bool {
        self.node.lock().unwrap().has_local(cid)
    }

    /// Peers this node knows hold `cid` (from successful replications
    /// and fetches). A hint, not a guarantee.
    pub fn holders_of(&self, cid: &Cid) -> Vec<PeerId> {
        self.holders
            .lock()
            .unwrap()
            .get(cid)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Replicate the block `cid` (which must be held locally) to each
    /// peer in `peers`. Returns the peers that accepted it. A peer that
    /// is unreachable or refuses is skipped, not fatal — but if no peer
    /// accepts, the error [`ReplicationError::NoPeersAccepted`] is
    /// returned. Accepted peers are recorded as holders of the block.
    pub fn replicate(
        &self,
        cid: &Cid,
        peers: &[SocketAddr],
    ) -> Result<Vec<PeerId>, ReplicationError> {
        let data = self.get_local(cid).map_err(|_| ReplicationError::BlockNotHeld(*cid))?;
        let mut accepted = Vec::new();
        for peer_addr in peers {
            if let Ok(peer_id) = store_to(*peer_addr, cid, &data) {
                self.record_holder(cid, peer_id.clone());
                accepted.push(peer_id);
            }
        }
        if accepted.is_empty() {
            return Err(ReplicationError::NoPeersAccepted);
        }
        Ok(accepted)
    }

    /// Fetch a block by CID, trying each peer in order. The first peer
    /// that holds it wins; the block is stored locally and that peer is
    /// recorded as a holder. Returns [`ReplicationError::BlockNotHeld`]
    /// if no peer tried holds the block (unreachable peers are skipped,
    /// not fatal).
    pub fn fetch(&self, cid: &Cid, peers: &[SocketAddr]) -> Result<Vec<u8>, ReplicationError> {
        for peer_addr in peers {
            match fetch_from(*peer_addr, cid) {
                Ok(Some((data, holder))) => {
                    // Verify the bytes really are the content asked for:
                    // content addressing means the hash must match.
                    let actual = Cid::from_bytes(kernel_crypto::sha256(&data));
                    if actual != *cid {
                        continue; // a lying peer: skip it, try the next
                    }
                    self.node.lock().unwrap().store_local(&data);
                    self.record_holder(cid, holder);
                    return Ok(data);
                }
                Ok(None) => continue, // MISS: try the next peer
                Err(_) => continue,   // unreachable: try the next peer
            }
        }
        Err(ReplicationError::BlockNotHeld(*cid))
    }

    /// Stop the server thread and shut this node down. In-flight
    /// connections are dropped. The node's stored blocks are gone with
    /// it (in-memory store).
    pub fn stop(mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        if let Some(server) = self.server.take() {
            let _ = server.join();
        }
    }

    fn record_holder(&self, cid: &Cid, peer: PeerId) {
        self.holders
            .lock()
            .unwrap()
            .entry(*cid)
            .or_default()
            .insert(peer);
    }
}

/// Send a STORE frame to one peer. Returns the peer's id on acceptance.
fn store_to(addr: SocketAddr, cid: &Cid, data: &[u8]) -> Result<PeerId, ReplicationError> {
    let mut conn = Connection::connect(addr)?;
    conn.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut frame = Vec::with_capacity(1 + 32 + data.len());
    frame.push(OP_STORE);
    frame.extend_from_slice(cid.as_bytes());
    frame.extend_from_slice(data);
    conn.send(&frame)?;
    let resp = conn.recv()?;
    conn.close();
    if resp.is_empty() || resp[0] != RESP_OK {
        return Err(ReplicationError::InvalidResponse(
            "STORE was not accepted".to_string(),
        ));
    }
    let identity = PublicIdentity::decode(
        std::str::from_utf8(&resp[1..])
            .map_err(|_| ReplicationError::InvalidResponse("bad identity".to_string()))?,
    )
    .map_err(|e| ReplicationError::InvalidResponse(e.to_string()))?;
    Ok(identity.peer_id().clone())
}

/// Send a FETCH frame to one peer. `Ok(Some((data, holder)))` on a hit
/// (with the serving peer's id, so the caller can record a holder),
/// `Ok(None)` on a miss, `Err` if the peer is unreachable.
fn fetch_from(addr: SocketAddr, cid: &Cid) -> Result<Option<(Vec<u8>, PeerId)>, ReplicationError> {
    let mut conn = Connection::connect(addr)?;
    conn.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut frame = Vec::with_capacity(1 + 32);
    frame.push(OP_FETCH);
    frame.extend_from_slice(cid.as_bytes());
    conn.send(&frame)?;
    let resp = conn.recv()?;
    conn.close();
    if resp.is_empty() {
        return Err(ReplicationError::InvalidResponse(
            "empty response".to_string(),
        ));
    }
    match resp[0] {
        RESP_HIT => {
            // HIT layout: opcode + identity length (u16 LE) + KID1
            // identity bytes + block bytes.
            if resp.len() < 3 {
                return Err(ReplicationError::InvalidResponse(
                    "HIT response too short".to_string(),
                ));
            }
            let id_len = u16::from_le_bytes(resp[1..3].try_into().expect("2-byte slice")) as usize;
            if resp.len() < 3 + id_len {
                return Err(ReplicationError::InvalidResponse(
                    "HIT response truncates the identity".to_string(),
                ));
            }
            let identity = PublicIdentity::decode(
                std::str::from_utf8(&resp[3..3 + id_len])
                    .map_err(|_| ReplicationError::InvalidResponse("bad identity".to_string()))?,
            )
            .map_err(|e| ReplicationError::InvalidResponse(e.to_string()))?;
            Ok(Some((
                resp[3 + id_len..].to_vec(),
                identity.peer_id().clone(),
            )))
        }
        RESP_MISS => Ok(None),
        _ => Err(ReplicationError::InvalidResponse(
            "unexpected response opcode".to_string(),
        )),
    }
}

/// The server loop: accept connections and handle each on its own
/// thread until the stop flag is set.
fn serve(listener: Listener, node: Arc<Mutex<Node>>, stop_flag: Arc<AtomicBool>) {
    loop {
        if stop_flag.load(Ordering::SeqCst) {
            break;
        }
        match listener.accept() {
            Ok(conn) => {
                let node = Arc::clone(&node);
                thread::spawn(move || handle_connection(conn, node));
            }
            Err(TransportError::Io(ref e))
                if e.kind() == std::io::ErrorKind::WouldBlock =>
            {
                thread::sleep(Duration::from_millis(5));
            }
            Err(_) => {
                // The listener is gone (or another error): stop serving.
                break;
            }
        }
    }
}

/// Handle one peer connection: read frames until the peer goes away.
fn handle_connection(mut conn: Connection, node: Arc<Mutex<Node>>) {
    loop {
        let frame = match conn.recv() {
            Ok(frame) => frame,
            Err(_) => return, // peer went away; nothing more to do
        };
        if frame.is_empty() {
            return;
        }
        match frame[0] {
            OP_STORE => {
                if frame.len() < 33 {
                    let _ = conn.send(&[RESP_FAIL]);
                    continue;
                }
                let mut cid_bytes = [0u8; 32];
                cid_bytes.copy_from_slice(&frame[1..33]);
                let cid = Cid::from_bytes(cid_bytes);
                let data = &frame[33..];
                // Verify the bytes hash to the CID they claim: a peer
                // must not be able to store lying bytes under a CID.
                let actual = Cid::from_bytes(kernel_crypto::sha256(data));
                if actual != cid {
                    let _ = conn.send(&[RESP_FAIL]);
                    continue;
                }
                node.lock().unwrap().store_local(data);
                let identity = node.lock().unwrap().public_identity().encode();
                let mut resp = vec![RESP_OK];
                resp.extend_from_slice(identity.as_bytes());
                if conn.send(&resp).is_err() {
                    return;
                }
            }
            OP_FETCH => {
                if frame.len() < 33 {
                    let _ = conn.send(&[RESP_FAIL]);
                    continue;
                }
                let mut cid_bytes = [0u8; 32];
                cid_bytes.copy_from_slice(&frame[1..33]);
                let cid = Cid::from_bytes(cid_bytes);
                // Bind the lookup result first: locking inside the
                // match scrutinee would hold the mutex through the arms
                // and deadlock when the arm locks again for the identity.
                let result = node.lock().unwrap().get_local(&cid);
                match result {
                    Ok(data) => {
                        // HIT: opcode + identity length + identity + data.
                        let identity = node.lock().unwrap().public_identity().encode();
                        let mut resp = vec![RESP_HIT];
                        resp.extend_from_slice(&(identity.len() as u16).to_le_bytes());
                        resp.extend_from_slice(identity.as_bytes());
                        resp.extend_from_slice(&data);
                        if conn.send(&resp).is_err() {
                            return;
                        }
                    }
                    Err(_) => {
                        if conn.send(&[RESP_MISS]).is_err() {
                            return;
                        }
                    }
                }
            }
            _ => {
                let _ = conn.send(&[RESP_FAIL]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn spawn_peers(n: usize) -> Vec<ReplicaNode> {
        (0..n)
            .map(|i| ReplicaNode::spawn(&format!("peer{i}")).unwrap())
            .collect()
    }

    fn addrs(peers: &[ReplicaNode]) -> Vec<SocketAddr> {
        peers.iter().map(|p| p.addr()).collect()
    }

    /// Assert a result is `Err(BlockNotHeld(cid))`.
    fn assert_block_not_held<T: std::fmt::Debug>(
        r: Result<T, ReplicationError>,
        cid: Cid,
    ) {
        match r {
            Err(ReplicationError::BlockNotHeld(got)) => assert_eq!(got, cid),
            other => panic!("expected BlockNotHeld({cid}), got {other:?}"),
        }
    }

    #[test]
    fn replicate_to_three_peers() {
        let origin = ReplicaNode::spawn("origin").unwrap();
        let peers = spawn_peers(3);
        let data = b"replicated block";
        let cid = origin.store_local(data);

        let accepted = origin.replicate(&cid, &addrs(&peers)).unwrap();
        assert_eq!(accepted.len(), 3);
        // Every peer now holds the block.
        for peer in &peers {
            assert!(peer.has_local(&cid));
            assert_eq!(peer.get_local(&cid).unwrap(), data);
        }
        // The origin recorded all three as holders.
        let holders = origin.holders_of(&cid);
        assert_eq!(holders.len(), 3);
        for peer in &peers {
            assert!(holders.contains(&peer.peer_id()));
        }
        // The origin still holds it too.
        assert!(origin.has_local(&cid));
    }

    #[test]
    fn fetch_from_a_non_origin_peer() {
        let origin = ReplicaNode::spawn("origin").unwrap();
        let peers = spawn_peers(3);
        let data = b"fetch me from a replica";
        let cid = origin.store_local(data);
        origin.replicate(&cid, &addrs(&peers)).unwrap();

        // A fresh node that never saw the origin fetches from replica 1.
        let newcomer = ReplicaNode::spawn("newcomer").unwrap();
        let fetched = newcomer.fetch(&cid, &[peers[1].addr()]).unwrap();
        assert_eq!(fetched, data);
        // And now the newcomer holds it locally.
        assert!(newcomer.has_local(&cid));
        assert_eq!(newcomer.get_local(&cid).unwrap(), data);
    }

    #[test]
    fn fetch_of_a_cid_nobody_holds_is_a_clean_error() {
        let origin = ReplicaNode::spawn("origin").unwrap();
        let peers = spawn_peers(2);
        let missing = Cid::from_bytes(kernel_crypto::sha256(b"nobody has this"));
        let asker = ReplicaNode::spawn("asker").unwrap();
        assert_block_not_held(asker.fetch(&missing, &addrs(&peers)), missing);
        // The origin itself does not hold it either.
        assert!(!origin.has_local(&missing));
    }

    #[test]
    fn offline_peer_does_not_break_fetches_from_remaining_holders() {
        let origin = ReplicaNode::spawn("origin").unwrap();
        let peers = spawn_peers(2);
        let peer_addrs = addrs(&peers);
        let data = b"skip the dead, use the live";
        let cid = origin.store_local(data);
        origin.replicate(&cid, &peer_addrs).unwrap();

        // Stop peer 0, keep peer 1 live.
        let dead_addr = peer_addrs[0];
        let live_addr = peer_addrs[1];
        // Stop peer 0 by taking it out of the vec.
        let mut peers = peers;
        let dead = peers.remove(0);
        dead.stop();
        std::thread::sleep(Duration::from_millis(150));
        assert!(Connection::connect(dead_addr).is_err());

        // Asker lists the dead peer first, then the live one.
        let asker = ReplicaNode::spawn("asker").unwrap();
        let fetched = asker.fetch(&cid, &[dead_addr, live_addr]).unwrap();
        assert_eq!(fetched, data);
    }

    #[test]
    fn replicate_to_no_reachable_peer_is_a_clean_error() {
        let origin = ReplicaNode::spawn("origin").unwrap();
        let data = b"nowhere to go";
        let cid = origin.store_local(data);
        // Port 1 is never listening.
        let dead: SocketAddr = "127.0.0.1:1".parse().unwrap();
        assert!(matches!(
            origin.replicate(&cid, &[dead]),
            Err(ReplicationError::NoPeersAccepted)
        ));
    }

    #[test]
    fn replicate_requires_holding_the_block_locally() {
        let origin = ReplicaNode::spawn("origin").unwrap();
        let peers = spawn_peers(1);
        let missing = Cid::from_bytes(kernel_crypto::sha256(b"not held locally"));
        assert_block_not_held(origin.replicate(&missing, &addrs(&peers)), missing);
    }

    #[test]
    fn store_rejects_bytes_that_do_not_hash_to_the_cid() {
        // A peer that tries to STORE lying bytes (bytes that do not hash
        // to the CID it claims) is refused by the server.
        let peer = ReplicaNode::spawn("peer").unwrap();
        let data = b"real bytes";
        let real_cid = Cid::from_bytes(kernel_crypto::sha256(data));
        let lying_cid = Cid::from_bytes(kernel_crypto::sha256(b"other bytes"));
        // Send a STORE frame claiming `lying_cid` for `data`.
        let mut conn = Connection::connect(peer.addr()).unwrap();
        let mut frame = vec![OP_STORE];
        frame.extend_from_slice(lying_cid.as_bytes());
        frame.extend_from_slice(data);
        conn.send(&frame).unwrap();
        let resp = conn.recv().unwrap();
        assert_eq!(resp, vec![RESP_FAIL]);
        // The lying bytes were not stored under either CID.
        assert!(!peer.has_local(&lying_cid));
        assert!(!peer.has_local(&real_cid));
        conn.close();
    }
}
