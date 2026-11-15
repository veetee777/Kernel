//! Peer discovery for KERNEL: signed peer records gossiped over TCP.
//!
//! A peer record (`KPEER1`) says "I am this peer, reachable at this
//! address, as of this time", signed by the peer it names. Nodes exchange
//! full tables of records when they connect, so a network of nodes that
//! talk to each other converges on the same set of known peers.
//!
//! Rules, enforced here and nowhere else:
//! - A record with a bad signature is refused: never stored, never
//!   forwarded. Verification happens on insert, the single choke point.
//! - A stale record (an older timestamp for the same peer) never
//!   replaces a newer one. Equal timestamps keep the record already
//!   held, so convergence is deterministic.
//! - A node can only ever create a record for itself: records are
//!   signed with the creator's secret key, and verification checks the
//!   signature against the public key the record claims. Announcing
//!   another peer as yourself is a forgery, and it fails verification.
//!
//! Nothing here is JSON and nothing touches disk. The wire format is
//! binary, length-prefixed by [`kernel_transport`], and bounded by its
//! frame cap. This crate does not wire into the DHT; DHT integration
//! is separate work.

use std::collections::HashMap;
use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use kernel_crypto::{CryptoError, Ed25519PublicKey, Ed25519Signature};
use kernel_identity::{Identity, IdentityError, PeerId};
use kernel_transport::{Connection, Listener, TransportError};

/// Magic for one encoded peer record.
const RECORD_MAGIC: &[u8; 6] = b"KPEER1";
/// Domain separator for the signed payload.
const SIGN_DOMAIN: &[u8; 11] = b"KPEER1-SIG\0";
/// Magic for one encoded gossip batch.
const BATCH_MAGIC: &[u8; 6] = b"KPEERB";
/// Longest a discovery operation waits on a peer.
const DEFAULT_IO_TIMEOUT: Duration = Duration::from_secs(10);

/// Errors returned by discovery operations.
#[derive(Debug)]
pub enum DiscoveryError {
    /// A crypto operation failed.
    Crypto(CryptoError),
    /// An identity operation failed.
    Identity(IdentityError),
    /// A transport operation failed.
    Transport(TransportError),
    /// Bytes did not match the expected format.
    InvalidFormat(String),
    /// A record's signature did not verify against its claimed peer.
    InvalidSignature,
}

impl fmt::Display for DiscoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiscoveryError::Crypto(e) => write!(f, "crypto error: {e}"),
            DiscoveryError::Identity(e) => write!(f, "identity error: {e}"),
            DiscoveryError::Transport(e) => write!(f, "transport error: {e}"),
            DiscoveryError::InvalidFormat(msg) => write!(f, "invalid format: {msg}"),
            DiscoveryError::InvalidSignature => write!(f, "invalid signature"),
        }
    }
}

impl std::error::Error for DiscoveryError {}

impl From<CryptoError> for DiscoveryError {
    fn from(e: CryptoError) -> Self {
        DiscoveryError::Crypto(e)
    }
}

impl From<IdentityError> for DiscoveryError {
    fn from(e: IdentityError) -> Self {
        DiscoveryError::Identity(e)
    }
}

impl From<TransportError> for DiscoveryError {
    fn from(e: TransportError) -> Self {
        DiscoveryError::Transport(e)
    }
}

/// Current time as seconds since the Unix epoch.
fn now_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A signed peer record: who a peer is, where to reach them, and when
/// the record was made. The signature covers the peer id, the address,
/// and the timestamp, so none of them can be altered after signing.
///
/// Binary format (no JSON):
///
/// ```text
/// magic     "KPEER1" (6 bytes)
/// pubkey    32 bytes
/// name_len  u8, then that many name bytes (1..=32)
/// addr      family u8 (4 or 6), ip bytes (4 or 16), port u16 BE
/// timestamp u64 BE, seconds since the Unix epoch
/// signature 64 bytes
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeerRecord {
    peer: PeerId,
    addr: SocketAddr,
    timestamp: u64,
    signature: Ed25519Signature,
}

impl PeerRecord {
    /// Create a record for `identity` at `addr`, stamped at `timestamp`.
    /// Only the owner of the identity can create a valid record for it:
    /// the signature is made with the identity's secret key.
    pub fn new(identity: &Identity, addr: SocketAddr, timestamp: u64) -> PeerRecord {
        let peer = identity.peer_id().clone();
        let mut record = PeerRecord {
            peer,
            addr,
            timestamp,
            signature: Ed25519Signature::from_bytes(&[0u8; 64]),
        };
        let payload = record.signed_payload();
        record.signature = identity.sign(&payload);
        record
    }

    /// Create a record for `identity` at `addr`, stamped at the current time.
    pub fn announce(identity: &Identity, addr: SocketAddr) -> PeerRecord {
        Self::new(identity, addr, now_timestamp())
    }

    /// The peer this record names.
    pub fn peer(&self) -> &PeerId {
        &self.peer
    }

    /// The socket address the peer is reachable at.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// When the record was made, in seconds since the Unix epoch.
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// The Ed25519 signature over the record.
    pub fn signature(&self) -> &Ed25519Signature {
        &self.signature
    }

    /// The exact bytes the signature covers. Domain-separated, so a
    /// signature for a record cannot be replayed as something else.
    fn signed_payload(&self) -> Vec<u8> {
        let name = self.peer.display_name().as_bytes();
        let mut out = Vec::with_capacity(SIGN_DOMAIN.len() + 32 + 1 + name.len() + 19 + 8);
        out.extend_from_slice(SIGN_DOMAIN);
        out.extend_from_slice(self.peer.public_key().as_bytes());
        out.push(name.len() as u8);
        out.extend_from_slice(name);
        out.extend_from_slice(&addr_bytes(&self.addr));
        out.extend_from_slice(&self.timestamp.to_be_bytes());
        out
    }

    /// Verify the signature against the public key the record claims.
    /// Fails with [`DiscoveryError::InvalidSignature`] when the record
    /// was tampered with, or claims a peer its signer does not own.
    pub fn verify(&self) -> Result<(), DiscoveryError> {
        self.peer
            .verify(&self.signed_payload(), &self.signature)
            .map_err(|_| DiscoveryError::InvalidSignature)
    }

    /// Encode this record to its binary form.
    pub fn encode(&self) -> Vec<u8> {
        let name = self.peer.display_name().as_bytes();
        let addr = addr_bytes(&self.addr);
        let mut out = Vec::with_capacity(6 + 32 + 1 + name.len() + addr.len() + 8 + 64);
        out.extend_from_slice(RECORD_MAGIC);
        out.extend_from_slice(self.peer.public_key().as_bytes());
        out.push(name.len() as u8);
        out.extend_from_slice(name);
        out.extend_from_slice(&addr);
        out.extend_from_slice(&self.timestamp.to_be_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a record from its binary form. Structural only: it checks
    /// the magic and the lengths, but does not verify the signature —
    /// call [`PeerRecord::verify`] or insert through [`PeerTable`],
    /// which verifies.
    pub fn decode(bytes: &[u8]) -> Result<PeerRecord, DiscoveryError> {
        fn need(bytes: &[u8], at: usize, n: usize, what: &str) -> Result<(), DiscoveryError> {
            if bytes.len() < at + n {
                return Err(DiscoveryError::InvalidFormat(format!(
                    "truncated record: missing {what}"
                )));
            }
            Ok(())
        }
        let mut at = 0;
        need(bytes, at, 6, "magic")?;
        if &bytes[..6] != RECORD_MAGIC {
            return Err(DiscoveryError::InvalidFormat(
                "bad record magic".to_string(),
            ));
        }
        at += 6;
        need(bytes, at, 32, "public key")?;
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[at..at + 32]);
        at += 32;
        let public_key = Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| DiscoveryError::InvalidFormat(format!("bad public key: {e}")))?;
        need(bytes, at, 1, "name length")?;
        let name_len = bytes[at] as usize;
        at += 1;
        need(bytes, at, name_len, "name")?;
        let name = std::str::from_utf8(&bytes[at..at + name_len])
            .map_err(|_| DiscoveryError::InvalidFormat("name is not valid UTF-8".to_string()))?;
        let peer = PeerId::new(public_key, name)
            .map_err(|e| DiscoveryError::InvalidFormat(format!("bad peer id: {e}")))?;
        at += name_len;
        let (addr, used) = decode_addr(&bytes[at..])?;
        at += used;
        need(bytes, at, 8, "timestamp")?;
        let timestamp = u64::from_be_bytes(bytes[at..at + 8].try_into().unwrap());
        at += 8;
        need(bytes, at, 64, "signature")?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[at..at + 64]);
        at += 64;
        if at != bytes.len() {
            return Err(DiscoveryError::InvalidFormat(
                "trailing bytes after record".to_string(),
            ));
        }
        Ok(PeerRecord {
            peer,
            addr,
            timestamp,
            signature: Ed25519Signature::from_bytes(&sig_bytes),
        })
    }
}

/// Encode a socket address as: family byte (4 or 6), ip bytes, port u16 BE.
fn addr_bytes(addr: &SocketAddr) -> Vec<u8> {
    match addr.ip() {
        IpAddr::V4(ip) => {
            let mut out = Vec::with_capacity(7);
            out.push(4);
            out.extend_from_slice(&ip.octets());
            out.extend_from_slice(&addr.port().to_be_bytes());
            out
        }
        IpAddr::V6(ip) => {
            let mut out = Vec::with_capacity(19);
            out.push(6);
            out.extend_from_slice(&ip.octets());
            out.extend_from_slice(&addr.port().to_be_bytes());
            out
        }
    }
}

/// Decode an address encoded by [`addr_bytes`]. Returns the address and
/// the number of bytes consumed.
fn decode_addr(bytes: &[u8]) -> Result<(SocketAddr, usize), DiscoveryError> {
    if bytes.is_empty() {
        return Err(DiscoveryError::InvalidFormat(
            "truncated record: missing address".to_string(),
        ));
    }
    match bytes[0] {
        4 => {
            if bytes.len() < 7 {
                return Err(DiscoveryError::InvalidFormat(
                    "truncated record: missing IPv4 address".to_string(),
                ));
            }
            let ip = IpAddr::from([bytes[1], bytes[2], bytes[3], bytes[4]]);
            let port = u16::from_be_bytes([bytes[5], bytes[6]]);
            Ok((SocketAddr::new(ip, port), 7))
        }
        6 => {
            if bytes.len() < 19 {
                return Err(DiscoveryError::InvalidFormat(
                    "truncated record: missing IPv6 address".to_string(),
                ));
            }
            let mut octets = [0u8; 16];
            octets.copy_from_slice(&bytes[1..17]);
            let ip = IpAddr::from(octets);
            let port = u16::from_be_bytes([bytes[17], bytes[18]]);
            Ok((SocketAddr::new(ip, port), 19))
        }
        other => Err(DiscoveryError::InvalidFormat(format!(
            "unknown address family {other}"
        ))),
    }
}

/// The set of peers a node knows about, keyed by public key bytes.
/// A node knows at most one record per peer: the newest it has seen.
#[derive(Debug, Default)]
pub struct PeerTable {
    records: HashMap<[u8; 32], PeerRecord>,
}

impl PeerTable {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a record. The signature is verified first: a record with
    /// a bad signature is refused with [`DiscoveryError::InvalidSignature`]
    /// and never stored. A record older than (or as old as) the one
    /// already held for the same peer is kept out and reported as
    /// `Ok(false)`. A newer record replaces the old one and reports
    /// `Ok(true)`.
    pub fn insert(&mut self, record: PeerRecord) -> Result<bool, DiscoveryError> {
        record.verify()?;
        let key = *record.peer.public_key().as_bytes();
        match self.records.get(&key) {
            Some(old) if old.timestamp >= record.timestamp => Ok(false),
            _ => {
                self.records.insert(key, record);
                Ok(true)
            }
        }
    }

    /// The record held for a public key, if any.
    pub fn get(&self, key: &[u8; 32]) -> Option<&PeerRecord> {
        self.records.get(key)
    }

    /// The record held for a peer, if any.
    pub fn get_by_peer(&self, peer: &PeerId) -> Option<&PeerRecord> {
        self.records.get(peer.public_key().as_bytes())
    }

    /// Every record held, in no particular order.
    pub fn records(&self) -> Vec<PeerRecord> {
        self.records.values().cloned().collect()
    }

    /// Number of peers known.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True when no peers are known.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// Encode a gossip batch: every record a node knows, length-prefixed.
///
/// ```text
/// magic "KPEERB" (6 bytes)
/// count u32 BE
/// for each record: length u32 BE, then the encoded record
/// ```
pub fn encode_batch(records: &[PeerRecord]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(BATCH_MAGIC);
    out.extend_from_slice(&(records.len() as u32).to_be_bytes());
    for record in records {
        let bytes = record.encode();
        out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
        out.extend_from_slice(&bytes);
    }
    out
}

/// Decode a gossip batch. Structural only: records are not verified
/// here. Merge them through [`PeerTable::insert`], which verifies and
/// refuses bad ones.
pub fn decode_batch(bytes: &[u8]) -> Result<Vec<PeerRecord>, DiscoveryError> {
    if bytes.len() < 6 || &bytes[..6] != BATCH_MAGIC {
        return Err(DiscoveryError::InvalidFormat("bad batch magic".to_string()));
    }
    if bytes.len() < 10 {
        return Err(DiscoveryError::InvalidFormat(
            "truncated batch: missing count".to_string(),
        ));
    }
    let count = u32::from_be_bytes(bytes[6..10].try_into().unwrap()) as usize;
    let mut at = 10;
    let mut records = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        if bytes.len() < at + 4 {
            return Err(DiscoveryError::InvalidFormat(
                "truncated batch: missing record length".to_string(),
            ));
        }
        let len = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        at += 4;
        if bytes.len() < at + len {
            return Err(DiscoveryError::InvalidFormat(
                "truncated batch: missing record bytes".to_string(),
            ));
        }
        records.push(PeerRecord::decode(&bytes[at..at + len])?);
        at += len;
    }
    if at != bytes.len() {
        return Err(DiscoveryError::InvalidFormat(
            "trailing bytes after batch".to_string(),
        ));
    }
    Ok(records)
}

/// Merge a batch of received records into a table. Records with bad
/// signatures are refused and skipped; the rest are inserted under the
/// table's freshness rules. Returns `(stored, skipped)`: how many
/// records were newly stored, and how many were refused or stale.
/// One bad record never poisons the rest of the batch.
fn merge_batch(table: &mut PeerTable, records: Vec<PeerRecord>) -> (usize, usize) {
    let mut stored = 0;
    let mut skipped = 0;
    for record in records {
        match table.insert(record) {
            Ok(true) => stored += 1,
            Ok(false) => skipped += 1,
            Err(_) => skipped += 1,
        }
    }
    (stored, skipped)
}

/// A node that gossips peer records. It listens on one address, keeps
/// a table of known peers (including itself), and exchanges full tables
/// with any peer it connects to or accepts a connection from.
///
/// `DiscoveryNode` is safe to share between threads: the table is
/// behind a mutex and the listener accepts from `&self`.
pub struct DiscoveryNode {
    identity: Identity,
    listener: Listener,
    table: Mutex<PeerTable>,
}

impl DiscoveryNode {
    /// Bind `listen` (use port 0 for an ephemeral port) and announce
    /// this node at the address it actually bound to. The node's own
    /// record is the first entry in its table.
    pub fn new(identity: Identity, listen: SocketAddr) -> Result<Arc<Self>, DiscoveryError> {
        let listener = Listener::bind(listen)?;
        let addr = listener.local_addr()?;
        let own = PeerRecord::announce(&identity, addr);
        let mut table = PeerTable::new();
        table.insert(own)?;
        Ok(Arc::new(Self {
            identity,
            listener,
            table: Mutex::new(table),
        }))
    }

    /// This node's public peer id.
    pub fn peer_id(&self) -> &PeerId {
        self.identity.peer_id()
    }

    /// The address this node listens on.
    pub fn listen_addr(&self) -> Result<SocketAddr, DiscoveryError> {
        Ok(self.listener.local_addr()?)
    }

    /// A snapshot of every peer record this node knows.
    pub fn peers(&self) -> Vec<PeerRecord> {
        self.table.lock().unwrap().records()
    }

    /// Number of peers known, including this node itself.
    pub fn peer_count(&self) -> usize {
        self.table.lock().unwrap().len()
    }

    /// True when this node knows about `peer`.
    pub fn knows(&self, peer: &PeerId) -> bool {
        self.table.lock().unwrap().get_by_peer(peer).is_some()
    }

    /// Exchange full tables with the peer at `addr`: send everything
    /// this node knows, then merge everything the peer sends back.
    /// A refused record on either side is skipped, never stored.
    pub fn announce_to(&self, addr: SocketAddr) -> Result<(), DiscoveryError> {
        let mut conn = Connection::connect(addr)?;
        conn.set_read_timeout(Some(DEFAULT_IO_TIMEOUT))?;
        let batch = {
            let table = self.table.lock().unwrap();
            encode_batch(&table.records())
        };
        conn.send(&batch)?;
        let reply = conn.recv()?;
        conn.close();
        let records = decode_batch(&reply)?;
        let mut table = self.table.lock().unwrap();
        merge_batch(&mut table, records);
        Ok(())
    }

    /// Serve incoming gossip connections until `expected` connections
    /// have been handled, or `timeout` elapses. Each accepted
    /// connection reads the peer's batch, merges it, and answers with
    /// this node's own batch. Returns how many connections were
    /// handled. A peer that connects and sends nothing times out
    /// (per-connection read timeout) and still counts as handled, so
    /// this never waits forever on a broken peer.
    pub fn serve(&self, expected: usize, timeout: Duration) -> Result<usize, DiscoveryError> {
        self.listener.set_nonblocking(true)?;
        let deadline = Instant::now() + timeout;
        let mut handled = 0;
        while handled < expected && Instant::now() < deadline {
            match self.listener.accept() {
                Ok(mut conn) => {
                    handled += 1;
                    let _ = conn.set_read_timeout(Some(DEFAULT_IO_TIMEOUT));
                    if let Ok(batch) = conn.recv() {
                        if let Ok(records) = decode_batch(&batch) {
                            {
                                let mut table = self.table.lock().unwrap();
                                merge_batch(&mut table, records);
                            }
                            let reply = {
                                let table = self.table.lock().unwrap();
                                encode_batch(&table.records())
                            };
                            let _ = conn.send(&reply);
                        }
                    }
                    conn.close();
                }
                Err(TransportError::Io(e)) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(handled)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn addr(port: u16) -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], port))
    }

    fn loopback() -> SocketAddr {
        "127.0.0.1:0".parse().unwrap()
    }

    #[test]
    fn record_signs_and_verifies() {
        let id = Identity::generate("alice").unwrap();
        let record = PeerRecord::new(&id, addr(7001), 1_000);
        assert_eq!(record.peer(), id.peer_id());
        assert_eq!(record.addr(), addr(7001));
        assert_eq!(record.timestamp(), 1_000);
        assert!(record.verify().is_ok());
        // Encode/decode round trip keeps a valid record.
        let back = PeerRecord::decode(&record.encode()).unwrap();
        assert_eq!(back, record);
        assert!(back.verify().is_ok());
    }

    #[test]
    fn record_with_tampered_signature_is_refused() {
        let id = Identity::generate("alice").unwrap();
        let record = PeerRecord::new(&id, addr(7001), 1_000);
        let mut bytes = record.encode();
        // Flip a byte in the trailing signature.
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        let tampered = PeerRecord::decode(&bytes).unwrap();
        assert!(matches!(
            tampered.verify(),
            Err(DiscoveryError::InvalidSignature)
        ));
        // The table refuses it too: never stored.
        let mut table = PeerTable::new();
        assert!(matches!(
            table.insert(tampered),
            Err(DiscoveryError::InvalidSignature)
        ));
        assert!(table.is_empty());
    }

    #[test]
    fn record_claiming_another_peer_is_refused() {
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let record = PeerRecord::new(&alice, addr(7001), 1_000);
        let mut bytes = record.encode();
        // Swap the claimed public key to Bob's, keeping Alice's signature.
        // The signature no longer matches the claimed peer.
        bytes[6..38].copy_from_slice(bob.peer_id().public_key().as_bytes());
        let forged = PeerRecord::decode(&bytes).unwrap();
        assert_eq!(forged.peer().public_key(), bob.peer_id().public_key());
        assert!(matches!(
            forged.verify(),
            Err(DiscoveryError::InvalidSignature)
        ));
        let mut table = PeerTable::new();
        assert!(table.insert(forged).is_err());
        assert!(table.is_empty());
    }

    #[test]
    fn tampering_with_the_address_or_timestamp_breaks_the_signature() {
        let id = Identity::generate("alice").unwrap();
        let record = PeerRecord::new(&id, addr(7001), 1_000);
        let mut bytes = record.encode();
        // Change the port byte (the address is inside the signed payload).
        let port_at = 6 + 32 + 1 + "alice".len() + 5;
        bytes[port_at] ^= 0x01;
        let tampered = PeerRecord::decode(&bytes).unwrap();
        assert!(tampered.verify().is_err());
    }

    #[test]
    fn stale_record_does_not_overwrite_newer() {
        let id = Identity::generate("alice").unwrap();
        let mut table = PeerTable::new();
        let newer = PeerRecord::new(&id, addr(7001), 2_000);
        let older = PeerRecord::new(&id, addr(7002), 1_000);
        assert!(table.insert(newer.clone()).unwrap());
        assert!(!table.insert(older).unwrap());
        let kept = table.get_by_peer(id.peer_id()).unwrap();
        assert_eq!(kept.timestamp(), 2_000);
        assert_eq!(kept.addr(), addr(7001));
        // An equal timestamp does not replace the record already held.
        let same_time = PeerRecord::new(&id, addr(7003), 2_000);
        assert!(!table.insert(same_time).unwrap());
        assert_eq!(table.get_by_peer(id.peer_id()).unwrap().addr(), addr(7001));
        // A newer record does replace it.
        let newest = PeerRecord::new(&id, addr(7004), 3_000);
        assert!(table.insert(newest).unwrap());
        assert_eq!(table.get_by_peer(id.peer_id()).unwrap().addr(), addr(7004));
    }

    #[test]
    fn batch_encode_decode_round_trip() {
        let a = Identity::generate("alice").unwrap();
        let b = Identity::generate("bob").unwrap();
        let records = vec![
            PeerRecord::new(&a, addr(7001), 100),
            PeerRecord::new(&b, addr(7002), 200),
        ];
        let back = decode_batch(&encode_batch(&records)).unwrap();
        assert_eq!(back, records);
    }

    #[test]
    fn batch_with_one_bad_record_keeps_the_good_ones() {
        let a = Identity::generate("alice").unwrap();
        let b = Identity::generate("bob").unwrap();
        let good = PeerRecord::new(&a, addr(7001), 100);
        let mut bad_bytes = PeerRecord::new(&b, addr(7002), 200).encode();
        let last = bad_bytes.len() - 1;
        bad_bytes[last] ^= 0x01;
        let bad = PeerRecord::decode(&bad_bytes).unwrap();
        assert!(bad.verify().is_err());

        let mut table = PeerTable::new();
        let (stored, skipped) = merge_batch(&mut table, vec![good, bad]);
        assert_eq!(stored, 1);
        assert_eq!(skipped, 1);
        assert_eq!(table.len(), 1);
        assert!(table.get_by_peer(a.peer_id()).is_some());
        assert!(table.get_by_peer(b.peer_id()).is_none());
    }

    #[test]
    fn decode_rejects_malformed_records_and_batches() {
        let id = Identity::generate("alice").unwrap();
        let record = PeerRecord::new(&id, addr(7001), 1_000);
        let bytes = record.encode();
        // Bad magic.
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(PeerRecord::decode(&bad).is_err());
        // Truncated.
        assert!(PeerRecord::decode(&bytes[..bytes.len() - 10]).is_err());
        // Trailing bytes.
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(PeerRecord::decode(&longer).is_err());
        // Bad batch magic and truncation.
        let batch = encode_batch(&[record]);
        let mut bad_batch = batch.clone();
        bad_batch[0] = b'X';
        assert!(decode_batch(&bad_batch).is_err());
        assert!(decode_batch(&batch[..8]).is_err());
    }

    #[test]
    fn three_nodes_converge_over_loopback() {
        let a = DiscoveryNode::new(Identity::generate("alice").unwrap(), loopback()).unwrap();
        let b = DiscoveryNode::new(Identity::generate("bob").unwrap(), loopback()).unwrap();
        let c = DiscoveryNode::new(Identity::generate("carol").unwrap(), loopback()).unwrap();
        let a_addr = a.listen_addr().unwrap();
        let b_addr = b.listen_addr().unwrap();
        let c_addr = c.listen_addr().unwrap();

        // Each node starts knowing only itself.
        assert_eq!(a.peer_count(), 1);
        assert_eq!(b.peer_count(), 1);
        assert_eq!(c.peer_count(), 1);

        // Serve threads: each node accepts exactly 2 incoming connections
        // (one from each of the other two nodes).
        let serve_timeout = Duration::from_secs(10);
        let sa = Arc::clone(&a);
        let sb = Arc::clone(&b);
        let sc = Arc::clone(&c);
        let ta = thread::spawn(move || sa.serve(2, serve_timeout).unwrap());
        let tb = thread::spawn(move || sb.serve(2, serve_timeout).unwrap());
        let tc = thread::spawn(move || sc.serve(2, serve_timeout).unwrap());

        // Full mesh: every node announces to the other two. Each
        // announcement is a two-way table exchange.
        a.announce_to(b_addr).unwrap();
        a.announce_to(c_addr).unwrap();
        b.announce_to(a_addr).unwrap();
        b.announce_to(c_addr).unwrap();
        c.announce_to(a_addr).unwrap();
        c.announce_to(b_addr).unwrap();

        assert_eq!(ta.join().unwrap(), 2);
        assert_eq!(tb.join().unwrap(), 2);
        assert_eq!(tc.join().unwrap(), 2);

        // All three nodes now know all three peers.
        for node in [&a, &b, &c] {
            assert_eq!(node.peer_count(), 3);
            assert!(node.knows(a.peer_id()));
            assert!(node.knows(b.peer_id()));
            assert!(node.knows(c.peer_id()));
            // Every record a node holds verifies.
            for record in node.peers() {
                assert!(record.verify().is_ok());
            }
        }
        // The addresses learned are the real listen addresses.
        let a_record = b
            .peers()
            .into_iter()
            .find(|r| r.peer() == a.peer_id())
            .unwrap();
        assert_eq!(a_record.addr(), a_addr);
    }

    #[test]
    fn a_node_never_announces_another_peer_as_itself() {
        // The only way to make a record is with the owner's secret key.
        // A record naming Bob but signed by Alice fails verification,
        // so it can never enter a table or be gossiped on.
        let alice = Identity::generate("alice").unwrap();
        let bob = Identity::generate("bob").unwrap();
        let bob_record = PeerRecord::new(&bob, addr(7002), 1_000);
        // Alice tries to re-issue Bob's record under her own key by
        // re-signing Bob's payload bytes: she cannot, because the
        // payload names Bob's key and her signature is checked against
        // Bob's key, not hers.
        let mut forged = bob_record.clone();
        forged.signature = alice.sign(&forged.signed_payload());
        assert!(forged.verify().is_err());
        let mut table = PeerTable::new();
        assert!(table.insert(forged).is_err());
        assert!(table.is_empty());
    }
}
