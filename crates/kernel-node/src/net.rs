//! Real networking for a KERNEL node: block exchange over TCP.
//!
//! Two nodes on different machines talk through [`kernel_transport`]
//! framed connections. The wire protocol is tiny and explicit:
//!
//! * Request: `KREQ1` (5 bytes) + CID (32 bytes).
//! * Response: `KRES1` (5 bytes) + found flag (1 byte) + if found,
//!   payload length (u32 big-endian) + the block bytes.
//!
//! Trust rules, same as the in-process network:
//! * A node serves only blocks it actually holds. A CID it does not
//!   hold gets a not-found answer, not an error.
//! * A fetched block is accepted only if its bytes hash to the CID that
//!   was requested. Bytes that fail the hash check are rejected, never
//!   stored or returned.
//!
//! This module does not authenticate peers or encrypt traffic. Those
//! belong in the layers above (identity signatures, AEAD sessions).
//! It also does not discover peers — the caller supplies the address.
//! Peer discovery and NAT traversal are separate, future work.

use std::fmt;
use std::net::SocketAddr;

use kernel_crypto::sha256;
use kernel_storage::Cid;
use kernel_transport::{Connection, Listener, TransportError};

use crate::{Node, NodeError};

/// Magic bytes at the start of every block request.
const REQUEST_MAGIC: &[u8; 5] = b"KREQ1";
/// Magic bytes at the start of every block response.
const RESPONSE_MAGIC: &[u8; 5] = b"KRES1";

/// Errors returned by TCP block exchange.
#[derive(Debug)]
pub enum NetError {
    /// An underlying transport failure.
    Transport(TransportError),
    /// A node operation failed (e.g. storage).
    Node(NodeError),
    /// The peer sent a frame with the wrong magic.
    BadMagic,
    /// A frame was too short or malformed.
    Malformed(String),
    /// The peer does not hold the requested block.
    BlockNotFound(Cid),
    /// Bytes came back but do not hash to the requested CID.
    HashMismatch {
        /// The CID that was requested.
        wanted: Cid,
        /// The CID the received bytes actually hash to.
        got: Cid,
    },
}

impl fmt::Display for NetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetError::Transport(e) => write!(f, "transport error: {e}"),
            NetError::Node(e) => write!(f, "node error: {e}"),
            NetError::BadMagic => write!(f, "peer sent a frame with bad magic"),
            NetError::Malformed(msg) => write!(f, "malformed frame: {msg}"),
            NetError::BlockNotFound(cid) => write!(f, "peer does not hold block {cid}"),
            NetError::HashMismatch { wanted, got } => write!(
                f,
                "bytes hash to {got}, not the requested CID {wanted}"
            ),
        }
    }
}

impl std::error::Error for NetError {}

impl From<TransportError> for NetError {
    fn from(e: TransportError) -> Self {
        NetError::Transport(e)
    }
}

impl From<NodeError> for NetError {
    fn from(e: NodeError) -> Self {
        NetError::Node(e)
    }
}

fn encode_request(cid: &Cid) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + 32);
    out.extend_from_slice(REQUEST_MAGIC);
    out.extend_from_slice(cid.as_bytes());
    out
}

fn decode_request(frame: &[u8]) -> Result<Cid, NetError> {
    if frame.len() < 5 || &frame[..5] != REQUEST_MAGIC {
        return Err(NetError::BadMagic);
    }
    if frame.len() != 5 + 32 {
        return Err(NetError::Malformed(format!(
            "request is {} bytes, expected {}",
            frame.len(),
            5 + 32
        )));
    }
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&frame[5..]);
    Ok(Cid::from_bytes(bytes))
}

fn encode_response(data: Option<&[u8]>) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + 1 + 4 + data.map_or(0, |d| d.len()));
    out.extend_from_slice(RESPONSE_MAGIC);
    match data {
        Some(bytes) => {
            out.push(1);
            out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
            out.extend_from_slice(bytes);
        }
        None => out.push(0),
    }
    out
}

fn decode_response(frame: &[u8], wanted: &Cid) -> Result<Vec<u8>, NetError> {
    if frame.len() < 5 || &frame[..5] != RESPONSE_MAGIC {
        return Err(NetError::BadMagic);
    }
    if frame.len() < 6 {
        return Err(NetError::Malformed(
            "response is missing the found flag".to_string(),
        ));
    }
    match frame[5] {
        0 => Err(NetError::BlockNotFound(*wanted)),
        1 => {
            if frame.len() < 10 {
                return Err(NetError::Malformed(
                    "response is missing the payload length".to_string(),
                ));
            }
            let len = u32::from_be_bytes([frame[6], frame[7], frame[8], frame[9]]) as usize;
            if frame.len() != 10 + len {
                return Err(NetError::Malformed(format!(
                    "response payload is {} bytes, header says {}",
                    frame.len() - 10,
                    len
                )));
            }
            let data = &frame[10..];
            let got = Cid::from_bytes(sha256(data));
            if got != *wanted {
                return Err(NetError::HashMismatch {
                    wanted: *wanted,
                    got,
                });
            }
            Ok(data.to_vec())
        }
        flag => Err(NetError::Malformed(format!(
            "response found flag is {flag}, expected 0 or 1"
        ))),
    }
}

/// Handle one incoming connection: read a block request, answer from
/// the node's store, close. Serves only blocks the node holds.
pub fn serve_connection(mut conn: Connection, node: &mut Node) -> Result<(), NetError> {
    let frame = conn.recv()?;
    let cid = decode_request(&frame)?;
    let data = node.get_local(&cid).ok();
    conn.send(&encode_response(data.as_deref()))?;
    conn.close();
    Ok(())
}

/// Accept and serve up to `max_requests` connections on `listener`.
/// Each connection serves one request. Returns after the count is
/// reached or the listener errors.
pub fn serve_listener(
    listener: &Listener,
    node: &mut Node,
    max_requests: usize,
) -> Result<(), NetError> {
    for _ in 0..max_requests {
        let conn = listener.accept()?;
        serve_connection(conn, node)?;
    }
    Ok(())
}

/// Fetch a block from a peer at `addr` over TCP. Returns the bytes only
/// if they hash to `cid`; anything else is an error.
pub fn fetch_block_tcp<A: std::net::ToSocketAddrs>(
    addr: A,
    cid: &Cid,
) -> Result<Vec<u8>, NetError> {
    let mut conn = Connection::connect(addr)?;
    conn.send(&encode_request(cid))?;
    let frame = conn.recv()?;
    conn.close();
    decode_response(&frame, cid)
}

/// A peer's address on the network, for callers that keep a directory.
pub type PeerAddr = SocketAddr;

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_transport::Listener as TcpListener;

    fn loopback() -> SocketAddr {
        "127.0.0.1:0".parse().unwrap()
    }

    #[test]
    fn two_nodes_exchange_a_block_over_tcp() {
        let listener = TcpListener::bind(loopback()).unwrap();
        let addr = listener.local_addr().unwrap();

        let mut server_node = Node::new("alice").unwrap();
        let cid = server_node.store_local(b"hello over tcp");

        // Server: one request, then done.
        let server = std::thread::spawn(move || {
            serve_listener(&listener, &mut server_node, 1).unwrap();
        });

        let got = fetch_block_tcp(addr, &cid).unwrap();
        assert_eq!(got, b"hello over tcp");
        server.join().unwrap();
    }

    #[test]
    fn unknown_cid_over_tcp_is_a_clean_not_found() {
        let listener = TcpListener::bind(loopback()).unwrap();
        let addr = listener.local_addr().unwrap();

        let mut server_node = Node::new("alice").unwrap();
        let server = std::thread::spawn(move || {
            serve_listener(&listener, &mut server_node, 1).unwrap();
        });

        let missing = Cid::from_bytes(sha256(b"alice never saw this"));
        let err = fetch_block_tcp(addr, &missing).unwrap_err();
        assert!(matches!(err, NetError::BlockNotFound(_)));
        server.join().unwrap();
    }

    #[test]
    fn tampered_bytes_over_tcp_are_rejected_by_hash() {
        let listener = TcpListener::bind(loopback()).unwrap();
        let addr = listener.local_addr().unwrap();

        // A lying server: answers every request with the wrong bytes.
        let server = std::thread::spawn(move || {
            let mut conn = listener.accept().unwrap();
            let frame = conn.recv().unwrap();
            let cid = decode_request(&frame).unwrap();
            let _ = cid;
            // Send bytes that hash to a different CID than requested.
            conn.send(&encode_response(Some(b"these are not the bytes"))).unwrap();
            conn.close();
        });

        let wanted = Cid::from_bytes(sha256(b"the real bytes"));
        let err = fetch_block_tcp(addr, &wanted).unwrap_err();
        assert!(
            matches!(err, NetError::HashMismatch { .. }),
            "expected HashMismatch, got {err:?}"
        );
        server.join().unwrap();
    }

    #[test]
    fn bad_magic_is_rejected() {
        assert!(matches!(
            decode_request(b"NOPE1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
            Err(NetError::BadMagic)
        ));
        let cid = Cid::from_bytes(sha256(b"x"));
        assert!(matches!(
            decode_response(b"NOPE1\x01", &cid),
            Err(NetError::BadMagic)
        ));
    }

    #[test]
    fn chunked_block_round_trips_over_tcp() {
        let listener = TcpListener::bind(loopback()).unwrap();
        let addr = listener.local_addr().unwrap();

        let mut server_node = Node::new("alice").unwrap();
        let data: Vec<u8> = (0..700 * 1024).map(|i| (i % 251) as u8).collect();
        let root = server_node.store_local_chunked(&data);

        let server = std::thread::spawn(move || {
            serve_listener(&listener, &mut server_node, 1).unwrap();
        });

        // The chunked root is itself a block; fetching it returns the
        // manifest, which the client can then use to fetch chunks.
        let manifest = fetch_block_tcp(addr, &root).unwrap();
        assert!(!manifest.is_empty());
        server.join().unwrap();
    }
}
