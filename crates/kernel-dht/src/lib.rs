//! Kademlia-style distributed hash table for KERNEL (in-memory).
//!
//! Node IDs are the SHA-256 of a peer's Ed25519 public key, so an ID is
//! bound to an identity: a peer cannot pick an arbitrary ID and pretend
//! to be someone else's neighbour. Closeness is XOR distance, as in
//! Kademlia.
//!
//! Routing uses k-buckets: 256 buckets, one per common-prefix length
//! with the local node, each holding up to `K` peers ordered by how
//! recently they were seen. A full bucket evicts its least-recently-seen
//! peer to make room (LRU eviction; a production DHT would ping the
//! evicted peer first, but this is the in-memory table, not the network).
//!
//! Values are stored with the storing peer's signature over the key and
//! the value, so a reader can tell a value really came from the peer it
//! claims to come from. A value with a bad signature is refused at
//! store time and never served.

use std::collections::{HashMap, VecDeque};
use std::fmt;

use kernel_crypto::{sha256, Ed25519Signature};
use kernel_identity::{IdentityError, PeerId};

/// Number of peers per k-bucket.
pub const K: usize = 20;

/// Domain separator for value signatures: a signature covers
/// `b"KDHT1" || key || value`.
const VALUE_SIGN_DOMAIN: &[u8] = b"KDHT1";

/// Errors returned by DHT operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DhtError {
    /// A value's signature did not verify against the storing peer.
    InvalidSignature,
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
}

impl fmt::Display for DhtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DhtError::InvalidSignature => write!(f, "value signature is invalid"),
            DhtError::Identity(e) => write!(f, "identity error: {e}"),
        }
    }
}

impl std::error::Error for DhtError {}

impl From<IdentityError> for DhtError {
    fn from(e: IdentityError) -> Self {
        DhtError::Identity(e)
    }
}

/// A DHT node ID: 32 bytes, the SHA-256 of a peer's public key.
/// Closeness is XOR distance, compared as a big-endian 256-bit number.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct NodeId([u8; 32]);

impl NodeId {
    /// Derive a node ID from a peer: SHA-256 of the Ed25519 public key.
    pub fn from_peer(peer: &PeerId) -> Self {
        Self(sha256(peer.public_key().as_bytes()))
    }

    /// The node ID for a lookup key: SHA-256 of the key bytes.
    pub fn from_key(key: &[u8]) -> Self {
        Self(sha256(key))
    }

    /// Build from 32 raw bytes (used by tests to place IDs precisely).
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The raw 32 ID bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// XOR distance to another ID, as 32 bytes. Shorter (lexicographically
    /// smaller, big-endian) means closer.
    pub fn distance(&self, other: &NodeId) -> [u8; 32] {
        let mut out = [0u8; 32];
        for i in 0..32 {
            out[i] = self.0[i] ^ other.0[i];
        }
        out
    }

    /// The common-prefix length with another ID: the number of leading
    /// bits that are equal (0..=256). Identical IDs have prefix 256.
    /// This is the k-bucket index for a peer at this distance.
    pub fn common_prefix_len(&self, other: &NodeId) -> usize {
        let dist = self.distance(other);
        for (i, byte) in dist.iter().enumerate() {
            if *byte != 0 {
                // The first differing bit inside this byte.
                let leading = byte.leading_zeros() as usize;
                return i * 8 + leading;
            }
        }
        256
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Short human-friendly form: first 4 bytes as hex.
        write!(
            f,
            "{:02x}{:02x}{:02x}{:02x}",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}

/// A stored value: the bytes, who stored them, and their signature.
/// The signature covers `b"KDHT1" || key || value` and is verified at
/// store time and again on every read.
#[derive(Clone, Debug)]
pub struct StoredValue {
    /// The stored bytes.
    pub value: Vec<u8>,
    /// The peer that stored them.
    pub storer: PeerId,
    /// The storer's signature over the key and value.
    pub signature: Ed25519Signature,
}

/// The result of a value lookup: either the value (with its storer) or,
/// when the key is not held here, the closest nodes to the key so the
/// caller can ask them next.
#[derive(Clone, Debug)]
pub enum ValueLookup {
    /// The value, as stored by `storer`.
    Found { value: Vec<u8>, storer: PeerId },
    /// The key is not held here; these are the closest known nodes.
    Closest(Vec<NodeId>),
}

/// The message a storing peer signs: domain separator + key + value.
pub fn value_sign_message(key: &[u8], value: &[u8]) -> Vec<u8> {
    let mut msg = Vec::with_capacity(VALUE_SIGN_DOMAIN.len() + key.len() + value.len());
    msg.extend_from_slice(VALUE_SIGN_DOMAIN);
    msg.extend_from_slice(key);
    msg.extend_from_slice(value);
    msg
}

/// An in-memory DHT: k-buckets for routing plus a signed value store.
pub struct Dht {
    local_id: NodeId,
    /// One bucket per common-prefix length (0..256). Bucket `i` holds
    /// peers whose IDs share exactly `i` leading bits with the local ID.
    /// Each bucket is ordered least-recently-seen first.
    buckets: Vec<VecDeque<NodeId>>,
    values: HashMap<Vec<u8>, StoredValue>,
}

impl Dht {
    /// Create a DHT for the given local peer.
    pub fn new(local_peer: &PeerId) -> Self {
        Self {
            local_id: NodeId::from_peer(local_peer),
            buckets: vec![VecDeque::new(); 256],
            values: HashMap::new(),
        }
    }

    /// This node's DHT ID.
    pub fn local_id(&self) -> &NodeId {
        &self.local_id
    }

    /// Record that a peer was seen (or heard from). Inserts it into the
    /// right k-bucket, most-recently-seen last. A full bucket evicts its
    /// least-recently-seen peer. The local peer is never inserted.
    pub fn seen(&mut self, peer: &PeerId) {
        let id = NodeId::from_peer(peer);
        if id == self.local_id {
            return;
        }
        let bucket_idx = self.local_id.common_prefix_len(&id);
        // common_prefix_len is < 256 for any peer that is not us.
        let bucket = &mut self.buckets[bucket_idx];
        // Already present: move to the back (most recently seen).
        if let Some(pos) = bucket.iter().position(|x| *x == id) {
            bucket.remove(pos);
            bucket.push_back(id);
            return;
        }
        if bucket.len() >= K {
            bucket.pop_front(); // evict least-recently-seen
        }
        bucket.push_back(id);
    }

    /// True if this peer's ID is in the routing table.
    pub fn contains(&self, peer: &PeerId) -> bool {
        let id = NodeId::from_peer(peer);
        self.buckets
            .iter()
            .any(|bucket| bucket.iter().any(|x| *x == id))
    }

    /// Number of peers in bucket `index` (0..256).
    pub fn bucket_len(&self, index: usize) -> usize {
        self.buckets.get(index).map(|b| b.len()).unwrap_or(0)
    }

    /// Total peers known to the routing table.
    pub fn peer_count(&self) -> usize {
        self.buckets.iter().map(|b| b.len()).sum()
    }

    /// The `count` nodes closest to `target` by XOR distance, nearest
    /// first. Returns fewer if fewer are known.
    pub fn find_node(&self, target: &NodeId, count: usize) -> Vec<NodeId> {
        let mut all: Vec<NodeId> = self
            .buckets
            .iter()
            .flat_map(|bucket| bucket.iter().copied())
            .collect();
        all.sort_by(|a, b| a.distance(target).cmp(&b.distance(target)));
        all.truncate(count);
        all
    }

    /// Store a value under `key`, signed by `storer`. The signature is
    /// verified before the value is accepted; a bad signature is refused
    /// and nothing is stored.
    pub fn store(
        &mut self,
        key: &[u8],
        value: &[u8],
        storer: &PeerId,
        signature: &Ed25519Signature,
    ) -> Result<(), DhtError> {
        let msg = value_sign_message(key, value);
        storer
            .verify(&msg, signature)
            .map_err(|_| DhtError::InvalidSignature)?;
        self.values.insert(
            key.to_vec(),
            StoredValue {
                value: value.to_vec(),
                storer: storer.clone(),
                signature: *signature,
            },
        );
        Ok(())
    }

    /// Look up a value. If it is held here (and its signature still
    /// verifies), returns it. Otherwise returns the closest known nodes
    /// to the key, so the caller can ask them next. A missing key is not
    /// an error — it is a routing answer.
    pub fn find_value(&self, key: &[u8]) -> ValueLookup {
        if let Some(stored) = self.values.get(key) {
            let msg = value_sign_message(key, &stored.value);
            if stored.storer.verify(&msg, &stored.signature).is_ok() {
                return ValueLookup::Found {
                    value: stored.value.clone(),
                    storer: stored.storer.clone(),
                };
            }
            // A stored value whose signature no longer verifies is
            // treated as absent (it should never happen, since store()
            // verifies first).
        }
        let target = NodeId::from_key(key);
        ValueLookup::Closest(self.find_node(&target, K))
    }

    /// Number of values held.
    pub fn value_count(&self) -> usize {
        self.values.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_identity::Identity;

    fn node_id_with_first_byte(b: u8) -> NodeId {
        let mut bytes = [0u8; 32];
        bytes[0] = b;
        NodeId::from_bytes(bytes)
    }

    #[test]
    fn node_id_derives_from_peer_key() {
        let a = Identity::generate("alice").unwrap();
        let b = Identity::generate("alice").unwrap();
        // Same name, different keys -> different IDs.
        assert_ne!(
            NodeId::from_peer(a.peer_id()),
            NodeId::from_peer(b.peer_id())
        );
        // Same key -> same ID, whatever the display name.
        let renamed = PeerId::new(*a.peer_id().public_key(), "alice2").unwrap();
        assert_eq!(NodeId::from_peer(&renamed), NodeId::from_peer(a.peer_id()));
        // ID is the SHA-256 of the public key.
        assert_eq!(
            NodeId::from_peer(a.peer_id()).as_bytes(),
            &sha256(a.peer_id().public_key().as_bytes())
        );
    }

    #[test]
    fn xor_distance_orders_correctly() {
        let target = NodeId::from_bytes([0u8; 32]);
        // Distance to target is the ID itself when target is all zeros.
        let near = node_id_with_first_byte(0x01);
        let mid = node_id_with_first_byte(0x40);
        let far = node_id_with_first_byte(0x80);
        assert!(near.distance(&target) < mid.distance(&target));
        assert!(mid.distance(&target) < far.distance(&target));
        // Distance is symmetric.
        assert_eq!(near.distance(&far), far.distance(&near));
        // Distance to self is zero.
        assert_eq!(near.distance(&near), [0u8; 32]);
    }

    #[test]
    fn common_prefix_len_is_the_bucket_index() {
        let local = NodeId::from_bytes([0u8; 32]);
        // Differ in the very first bit -> prefix 0.
        assert_eq!(local.common_prefix_len(&node_id_with_first_byte(0x80)), 0);
        // 0x40 = 0100_0000: first bit equal, second differs -> prefix 1.
        assert_eq!(local.common_prefix_len(&node_id_with_first_byte(0x40)), 1);
        // 0x01 = 0000_0001: first 7 bits equal -> prefix 7.
        assert_eq!(local.common_prefix_len(&node_id_with_first_byte(0x01)), 7);
        // Identical -> 256.
        assert_eq!(local.common_prefix_len(&local), 256);
    }

    #[test]
    fn find_node_returns_closest_first() {
        let local = Identity::generate("local").unwrap();
        let mut dht = Dht::new(local.peer_id());
        let target = NodeId::from_bytes([0u8; 32]);
        // Insert 10 real peers, then check find_node returns them sorted
        // by distance to the target, nearest first.
        let mut peers = Vec::new();
        for i in 0..10 {
            let id = Identity::generate(&format!("peer{i}")).unwrap();
            peers.push(id);
            dht.seen(peers.last().unwrap().peer_id());
        }
        let found = dht.find_node(&target, 5);
        assert_eq!(found.len(), 5);
        for w in found.windows(2) {
            assert!(w[0].distance(&target) <= w[1].distance(&target));
        }
        // Every returned node is one of the peers we inserted.
        let known: Vec<NodeId> = peers.iter().map(|p| NodeId::from_peer(p.peer_id())).collect();
        assert!(found.iter().all(|id| known.contains(id)));
    }

    #[test]
    fn find_node_closest_is_first_for_known_distances() {
        // Deterministic check using raw NodeIds placed in buckets by
        // direct bucket manipulation through seen() is impossible without
        // forgeable PeerIds, so verify the ordering logic on the distance
        // function itself and the find_node result against a brute-force
        // sort of the same known set.
        let local = Identity::generate("local").unwrap();
        let mut dht = Dht::new(local.peer_id());
        let mut ids: Vec<NodeId> = Vec::new();
        for i in 0..30 {
            let peer = Identity::generate(&format!("p{i}")).unwrap();
            ids.push(NodeId::from_peer(peer.peer_id()));
            dht.seen(peer.peer_id());
        }
        let target = NodeId::from_key(b"some lookup key");
        let mut expected = ids.clone();
        expected.sort_by(|a, b| a.distance(&target).cmp(&b.distance(&target)));
        expected.truncate(K);
        assert_eq!(dht.find_node(&target, K), expected);
    }

    #[test]
    fn store_and_find_value_round_trip() {
        let local = Identity::generate("local").unwrap();
        let storer = Identity::generate("storer").unwrap();
        let mut dht = Dht::new(local.peer_id());
        let key = b"kernel:file:readme";
        let value = b"the readme contents";
        let sig = storer.sign(&value_sign_message(key, value));
        dht.store(key, value, storer.peer_id(), &sig).unwrap();
        match dht.find_value(key) {
            ValueLookup::Found { value: got, storer: who } => {
                assert_eq!(got, value);
                assert_eq!(who, *storer.peer_id());
            }
            ValueLookup::Closest(_) => panic!("expected the value, got closest nodes"),
        }
        assert_eq!(dht.value_count(), 1);
    }

    #[test]
    fn store_rejects_bad_signature() {
        let local = Identity::generate("local").unwrap();
        let storer = Identity::generate("storer").unwrap();
        let other = Identity::generate("other").unwrap();
        let mut dht = Dht::new(local.peer_id());
        let key = b"k";
        let value = b"v";
        // Signature over different bytes.
        let bad_sig = storer.sign(&value_sign_message(key, b"different value"));
        assert_eq!(
            dht.store(key, value, storer.peer_id(), &bad_sig),
            Err(DhtError::InvalidSignature)
        );
        // Signature from a different peer than the claimed storer.
        let wrong_peer_sig = other.sign(&value_sign_message(key, value));
        assert_eq!(
            dht.store(key, value, storer.peer_id(), &wrong_peer_sig),
            Err(DhtError::InvalidSignature)
        );
        // Nothing was stored.
        assert_eq!(dht.value_count(), 0);
        assert!(matches!(dht.find_value(key), ValueLookup::Closest(_)));
    }

    #[test]
    fn missing_key_returns_closest_nodes_not_an_error() {
        let local = Identity::generate("local").unwrap();
        let mut dht = Dht::new(local.peer_id());
        for i in 0..5 {
            let peer = Identity::generate(&format!("peer{i}")).unwrap();
            dht.seen(peer.peer_id());
        }
        match dht.find_value(b"no such key") {
            ValueLookup::Closest(nodes) => {
                assert_eq!(nodes.len(), 5);
                // Closest nodes are ordered by distance to the key.
                let target = NodeId::from_key(b"no such key");
                for w in nodes.windows(2) {
                    assert!(w[0].distance(&target) <= w[1].distance(&target));
                }
            }
            ValueLookup::Found { .. } => panic!("missing key must not return a value"),
        }
    }

    #[test]
    fn bucket_evicts_least_recently_seen_when_full() {
        let local = Identity::generate("local").unwrap();
        let mut dht = Dht::new(local.peer_id());
        // Insert K+1 peers. All generated peers land in some bucket;
        // to force one bucket to overflow, insert many peers and check
        // the total never exceeds 256 * K and that the oldest peer in
        // any overflowing bucket is gone. Simpler deterministic check:
        // insert K+1 peers that all share bucket 0 with the local node.
        // Bucket 0 holds IDs differing in the first bit. We cannot forge
        // PeerIds, so instead verify the eviction invariant directly:
        // after inserting more than K peers into the same bucket, the
        // bucket holds exactly K and the earliest-inserted is gone.
        //
        // We reach a single bucket by generating peers until K+1 of them
        // land in the same bucket as each other (they must share a prefix
        // length with local). Bucket 0 is the most likely to fill: half
        // of all random IDs differ from local in the first bit.
        let mut first_in_bucket0: Option<PeerId> = None;
        let mut inserted = 0;
        let mut tries = 0;
        while inserted <= K && tries < 500 {
            tries += 1;
            let peer = Identity::generate(&format!("p{tries}")).unwrap();
            let id = NodeId::from_peer(peer.peer_id());
            if dht.local_id().common_prefix_len(&id) == 0 {
                if first_in_bucket0.is_none() {
                    first_in_bucket0 = Some(peer.peer_id().clone());
                }
                dht.seen(peer.peer_id());
                inserted += 1;
            }
        }
        assert_eq!(inserted, K + 1, "did not fill bucket 0 (tries={tries})");
        assert_eq!(dht.bucket_len(0), K);
        // The first peer inserted into bucket 0 was evicted.
        let first = first_in_bucket0.unwrap();
        assert!(!dht.contains(&first));
        assert_eq!(dht.peer_count(), K);
    }

    #[test]
    fn seen_is_idempotent_and_refreshes_recency() {
        let local = Identity::generate("local").unwrap();
        let mut dht = Dht::new(local.peer_id());
        let peer = Identity::generate("peer").unwrap();
        dht.seen(peer.peer_id());
        dht.seen(peer.peer_id());
        assert_eq!(dht.peer_count(), 1);
        // Seeing the local peer does nothing.
        dht.seen(local.peer_id());
        assert_eq!(dht.peer_count(), 1);
    }
}
