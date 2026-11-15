//! A KERNEL node: a peer identity plus a local content store.
//!
//! Nodes hold blocks and answer block requests from other nodes. The
//! transport in this crate is deliberately in-process (channels), so the
//! node logic can be tested without any real networking. Real networking
//! over TCP lives in the [`net`] module; peer discovery and NAT
//! traversal are still future work.
//!
//! Trust rule: a node only serves blocks it actually holds, and a block
//! is only accepted if its bytes hash to the CID that was requested.

use std::collections::HashMap;
use std::fmt;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::time::Duration;

use kernel_crypto::Ed25519Signature;
use kernel_identity::{Identity, IdentityError, PeerId, PublicIdentity};
use kernel_storage::{Cid, ContentStore, StorageError};

/// Block exchange over real TCP. See [`net`] for the wire protocol.
pub mod net;

/// Errors returned by node operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeError {
    /// Wraps a [`StorageError`].
    Storage(StorageError),
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// The requested peer is not on this network.
    PeerNotFound(PeerId),
    /// The peer does not hold a block with this CID.
    BlockNotFound(Cid),
    /// The in-process transport failed.
    Network(String),
}

impl fmt::Display for NodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeError::Storage(e) => write!(f, "storage error: {e}"),
            NodeError::Identity(e) => write!(f, "identity error: {e}"),
            NodeError::PeerNotFound(peer) => write!(f, "peer not on this network: {peer}"),
            NodeError::BlockNotFound(cid) => write!(f, "peer does not hold block {cid}"),
            NodeError::Network(msg) => write!(f, "network error: {msg}"),
        }
    }
}

impl std::error::Error for NodeError {}

impl From<StorageError> for NodeError {
    fn from(e: StorageError) -> Self {
        NodeError::Storage(e)
    }
}

impl From<IdentityError> for NodeError {
    fn from(e: IdentityError) -> Self {
        NodeError::Identity(e)
    }
}

/// A block request sent to a peer. Carries a reply channel so the
/// answering peer can send the block (or its absence) back directly.
pub struct Request {
    /// Who is asking.
    pub from: PeerId,
    /// The content being asked for.
    pub cid: Cid,
    /// Where the answer goes.
    pub reply: Sender<Response>,
}

/// An answer to a [`Request`]. `data` is `None` when the peer does not
/// hold the requested block.
pub struct Response {
    /// The CID that was asked for.
    pub cid: Cid,
    /// The block bytes, or `None` if the peer does not hold them.
    pub data: Option<Vec<u8>>,
}

/// An in-process network: a directory of peers and their request channels.
/// Used for testing node behavior without real sockets. A real transport
/// replaces this, not the node logic around it.
#[derive(Default)]
pub struct Network {
    peers: HashMap<PeerId, Sender<Request>>,
}

impl Network {
    /// An empty network with no peers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a peer's request channel. The peer must drain the channel
    /// (see [`Node::handle_next`]) or requests will queue up.
    pub fn register(&mut self, peer_id: PeerId, tx: Sender<Request>) {
        self.peers.insert(peer_id, tx);
    }

    /// Number of registered peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Send a block request to a peer. Returns the reply channel; the
    /// caller waits on it for the [`Response`].
    pub fn request(
        &self,
        to: &PeerId,
        from: &PeerId,
        cid: &Cid,
    ) -> Result<Receiver<Response>, NodeError> {
        let (reply_tx, reply_rx) = channel();
        let tx = self
            .peers
            .get(to)
            .ok_or_else(|| NodeError::PeerNotFound(to.clone()))?;
        tx.send(Request {
            from: from.clone(),
            cid: *cid,
            reply: reply_tx,
        })
        .map_err(|_| NodeError::Network(format!("request channel to {to} is closed")))?;
        Ok(reply_rx)
    }
}

/// A node: an identity and the blocks it holds.
pub struct Node {
    identity: Identity,
    store: ContentStore,
}

impl Node {
    /// Create a node with a fresh identity under `display_name`.
    pub fn new(display_name: &str) -> Result<Self, NodeError> {
        Ok(Self {
            identity: Identity::generate(display_name)?,
            store: ContentStore::new(),
        })
    }

    /// This node's public peer id.
    pub fn peer_id(&self) -> &PeerId {
        self.identity.peer_id()
    }

    /// This node's shareable public identity.
    pub fn public_identity(&self) -> PublicIdentity {
        self.identity.public_identity()
    }

    /// Sign a message as this node.
    pub fn sign(&self, message: &[u8]) -> Ed25519Signature {
        self.identity.sign(message)
    }

    /// Store bytes locally, returning their CID.
    pub fn store_local(&mut self, data: &[u8]) -> Cid {
        self.store.put(data)
    }

    /// Store large content locally, split into verified chunks.
    /// Returns the root CID of the chunk manifest.
    pub fn store_local_chunked(&mut self, data: &[u8]) -> Cid {
        self.store.put_chunked(data)
    }

    /// Fetch bytes this node already holds.
    pub fn get_local(&self, cid: &Cid) -> Result<Vec<u8>, NodeError> {
        self.store.get(cid).map_err(NodeError::Storage)
    }

    /// Fetch chunked content this node already holds, verifying every chunk.
    pub fn get_local_chunked(&self, root: &Cid) -> Result<Vec<u8>, NodeError> {
        self.store.get_chunked(root).map_err(NodeError::Storage)
    }

    /// True if this node holds a block with this CID.
    pub fn has_local(&self, cid: &Cid) -> bool {
        self.store.has(cid)
    }

    /// Handle one pending request from `rx`, without blocking.
    /// Returns `true` if a request was answered, `false` if the queue was
    /// empty. A node answers only with blocks it actually holds; a CID it
    /// does not hold gets an empty answer, not an error to the requester —
    /// the requester turns that into [`NodeError::BlockNotFound`].
    pub fn handle_next(&mut self, rx: &Receiver<Request>) -> Result<bool, NodeError> {
        match rx.try_recv() {
            Ok(req) => {
                let data = self.store.get(&req.cid).ok();
                req.reply
                    .send(Response {
                        cid: req.cid,
                        data,
                    })
                    .map_err(|_| {
                        NodeError::Network("requester went away before the reply".to_string())
                    })?;
                Ok(true)
            }
            Err(TryRecvError::Empty) => Ok(false),
            Err(TryRecvError::Disconnected) => Err(NodeError::Network(
                "request channel closed".to_string(),
            )),
        }
    }

    /// Request a block from a peer on `net` and wait for the reply
    /// (up to 5 seconds). Fails with [`NodeError::PeerNotFound`] if the
    /// peer is not on the network, and [`NodeError::BlockNotFound`] if the
    /// peer does not hold the block.
    pub fn fetch_block(
        &self,
        net: &Network,
        peer: &PeerId,
        cid: &Cid,
    ) -> Result<Vec<u8>, NodeError> {
        let rx = net.request(peer, self.peer_id(), cid)?;
        let resp = rx
            .recv_timeout(Duration::from_secs(5))
            .map_err(|_| NodeError::Network("timed out waiting for block".to_string()))?;
        resp.data.ok_or(NodeError::BlockNotFound(*cid))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Two nodes on a fresh network, with Alice's request channel exposed
    /// so the test can drive her handler by hand.
    fn two_nodes() -> (Network, Node, Node, Receiver<Request>) {
        let mut net = Network::new();
        let alice = Node::new("alice").unwrap();
        let bob = Node::new("bob").unwrap();
        let (a_tx, a_rx) = channel();
        net.register(alice.peer_id().clone(), a_tx);
        assert_eq!(net.peer_count(), 1);
        (net, alice, bob, a_rx)
    }

    #[test]
    fn nodes_exchange_a_block() {
        let (net, mut alice, bob, a_rx) = two_nodes();
        let cid = alice.store_local(b"hello kernel");
        let alice_peer = alice.peer_id().clone();

        // Bob asks; Alice answers; Bob receives the bytes.
        let reply_rx = net.request(&alice_peer, bob.peer_id(), &cid).unwrap();
        assert!(alice.handle_next(&a_rx).unwrap());
        let resp = reply_rx.recv().unwrap();
        assert_eq!(resp.cid, cid);
        assert_eq!(resp.data.unwrap(), b"hello kernel");
    }

    #[test]
    fn fetch_block_convenience_with_threaded_peer() {
        let (net, mut alice, bob, a_rx) = two_nodes();
        let cid = alice.store_local(b"fetch me");
        let alice_peer = alice.peer_id().clone();

        // Alice serves requests on her own thread.
        let handle = std::thread::spawn(move || loop {
            match alice.handle_next(&a_rx) {
                Ok(true) => break,
                Ok(false) => std::thread::sleep(Duration::from_millis(1)),
                Err(_) => break,
            }
        });

        let data = bob.fetch_block(&net, &alice_peer, &cid).unwrap();
        assert_eq!(data, b"fetch me");
        handle.join().unwrap();
    }

    #[test]
    fn fetch_unknown_cid_returns_clean_error() {
        let (net, mut alice, bob, a_rx) = two_nodes();
        // A CID for content Alice was never given.
        let missing = Cid::from_bytes(kernel_crypto::sha256(b"alice never saw this"));
        let alice_peer = alice.peer_id().clone();

        let reply_rx = net.request(&alice_peer, bob.peer_id(), &missing).unwrap();
        assert!(alice.handle_next(&a_rx).unwrap());
        let resp = reply_rx.recv().unwrap();
        assert_eq!(resp.data, None);
        assert_eq!(resp.cid, missing);

        // The convenience call turns the empty answer into BlockNotFound.
        let handle = std::thread::spawn(move || loop {
            match alice.handle_next(&a_rx) {
                Ok(true) => break,
                Ok(false) => std::thread::sleep(Duration::from_millis(1)),
                Err(_) => break,
            }
        });
        assert_eq!(
            bob.fetch_block(&net, &alice_peer, &missing),
            Err(NodeError::BlockNotFound(missing))
        );
        handle.join().unwrap();
    }

    #[test]
    fn fetch_from_unknown_peer_fails() {
        let (net, _alice, bob, _a_rx) = two_nodes();
        let stranger = Node::new("stranger").unwrap();
        let cid = Cid::from_bytes(kernel_crypto::sha256(b"anything"));
        assert_eq!(
            bob.fetch_block(&net, stranger.peer_id(), &cid),
            Err(NodeError::PeerNotFound(stranger.peer_id().clone()))
        );
    }

    #[test]
    fn node_cannot_fetch_content_it_was_never_given() {
        // Bob knows the CID of some content, but Alice never received the
        // bytes. Asking Alice must fail — she cannot invent content.
        let (net, mut alice, bob, a_rx) = two_nodes();
        let secret_data = b"only bob knows these bytes";
        let cid = Cid::from_bytes(kernel_crypto::sha256(secret_data));
        assert!(!alice.has_local(&cid));

        let reply_rx = net
            .request(alice.peer_id(), bob.peer_id(), &cid)
            .unwrap();
        assert!(alice.handle_next(&a_rx).unwrap());
        let resp = reply_rx.recv().unwrap();
        assert_eq!(resp.data, None);
    }

    #[test]
    fn handle_next_empty_queue_returns_false() {
        let (_net, mut alice, _bob, a_rx) = two_nodes();
        assert_eq!(alice.handle_next(&a_rx).unwrap(), false);
    }

    #[test]
    fn local_store_and_chunked_round_trip() {
        let mut node = Node::new("alice").unwrap();
        let data: Vec<u8> = (0..700 * 1024).map(|i| (i % 251) as u8).collect();
        let root = node.store_local_chunked(&data);
        assert!(node.has_local(&root));
        assert_eq!(node.get_local_chunked(&root).unwrap(), data);
        assert_eq!(node.get_local(&root).unwrap().len(), 11 + 3 * 32); // manifest
    }

    #[test]
    fn nodes_authenticate_each_other() {
        let (_net, alice, bob, _a_rx) = two_nodes();
        let msg = b"prove it is you";
        let sig = alice.sign(msg);
        // Bob verifies against Alice's public identity.
        let alice_public = alice.public_identity();
        assert!(alice_public.peer_id().verify(msg, &sig).is_ok());
        // And Bob's own signature does not verify as Alice's.
        let bob_sig = bob.sign(msg);
        assert!(alice_public.peer_id().verify(msg, &bob_sig).is_err());
    }

    #[test]
    fn node_rejects_bad_display_name() {
        assert!(Node::new("").is_err());
        assert!(Node::new("has space").is_err());
    }
}
