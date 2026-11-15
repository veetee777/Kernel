//! Minimal STUN client: RFC 5389 binding request / response.
//!
//! Only what address discovery needs: build a binding request with a
//! random 96-bit transaction id, and parse a binding success response
//! to extract the XOR-MAPPED-ADDRESS. No MESSAGE-INTEGRITY, no
//! authentication, no TURN — a binding request is unauthenticated by
//! design, and this crate never sends anything else.
//!
//! IPv4 and IPv6 are both parsed. The transaction id is checked on
//! every response, so a stray or replayed packet from another exchange
//! cannot be mistaken for ours.

use std::fmt;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, UdpSocket};
use std::time::Duration;

use kernel_crypto::{random_bytes, CryptoError};

/// STUN binding request message type (RFC 5389 §6).
pub const BINDING_REQUEST: u16 = 0x0001;
/// STUN binding success response message type.
pub const BINDING_RESPONSE: u16 = 0x0101;
/// STUN error response message type (parsed only to reject it cleanly).
pub const ERROR_RESPONSE: u16 = 0x0111;
/// The fixed magic cookie every STUN message carries (RFC 5389 §6).
pub const MAGIC_COOKIE: u32 = 0x2112_A442;
/// XOR-MAPPED-ADDRESS attribute type (RFC 5389 §15.2).
pub const ATTR_XOR_MAPPED_ADDRESS: u16 = 0x0020;
/// Address family values in a mapped-address attribute.
pub const FAMILY_IPV4: u8 = 0x01;
pub const FAMILY_IPV6: u8 = 0x02;
/// Header size in bytes: type (2) + length (2) + cookie (4) + id (12).
pub const HEADER_LEN: usize = 20;
/// Transaction id length in bytes (96 bits).
pub const TRANSACTION_ID_LEN: usize = 12;

/// Errors from STUN operations.
#[derive(Debug)]
pub enum StunError {
    /// A socket operation failed.
    Io(io::Error),
    /// The bytes are not a well-formed binding response.
    Malformed(String),
    /// The response is for a different transaction.
    WrongTransactionId,
    /// The message is not a binding success response (carries the
    /// type that was actually received).
    NotABindingResponse(u16),
    /// The response has no XOR-MAPPED-ADDRESS attribute.
    NoMappedAddress,
    /// No response arrived within the timeout.
    Timeout,
    /// Randomness (for the transaction id) failed.
    Crypto(CryptoError),
}

impl fmt::Display for StunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StunError::Io(e) => write!(f, "io error: {e}"),
            StunError::Malformed(msg) => write!(f, "malformed STUN message: {msg}"),
            StunError::WrongTransactionId => {
                f.write_str("STUN response is for a different transaction")
            }
            StunError::NotABindingResponse(t) => {
                write!(f, "not a STUN binding response (type 0x{t:04x})")
            }
            StunError::NoMappedAddress => {
                f.write_str("STUN response has no XOR-MAPPED-ADDRESS")
            }
            StunError::Timeout => f.write_str("timed out waiting for a STUN response"),
            StunError::Crypto(e) => write!(f, "crypto error: {e}"),
        }
    }
}

impl std::error::Error for StunError {}

impl From<io::Error> for StunError {
    fn from(e: io::Error) -> Self {
        StunError::Io(e)
    }
}

impl From<CryptoError> for StunError {
    fn from(e: CryptoError) -> Self {
        StunError::Crypto(e)
    }
}

/// Build a binding request. Returns the 20 request bytes and the
/// transaction id the caller must expect in the response. Two calls
/// never produce the same id (barring a randomness failure).
pub fn build_binding_request() -> Result<(Vec<u8>, [u8; TRANSACTION_ID_LEN]), StunError> {
    let txid = random_bytes::<TRANSACTION_ID_LEN>()?;
    let mut req = Vec::with_capacity(HEADER_LEN);
    req.extend_from_slice(&BINDING_REQUEST.to_be_bytes());
    req.extend_from_slice(&0u16.to_be_bytes()); // no attributes
    req.extend_from_slice(&MAGIC_COOKIE.to_be_bytes());
    req.extend_from_slice(&txid);
    Ok((req, txid))
}

/// Parse a binding success response and return the XOR-MAPPED-ADDRESS
/// it reports. `expected_txid` is the id from
/// [`build_binding_request`]; a response for any other transaction is
/// refused.
pub fn parse_binding_response(
    bytes: &[u8],
    expected_txid: &[u8; TRANSACTION_ID_LEN],
) -> Result<SocketAddr, StunError> {
    if bytes.len() < HEADER_LEN {
        return Err(StunError::Malformed(format!(
            "response is {} bytes, shorter than the {}-byte header",
            bytes.len(),
            HEADER_LEN
        )));
    }
    let msg_type = u16::from_be_bytes([bytes[0], bytes[1]]);
    if msg_type != BINDING_RESPONSE {
        return Err(StunError::NotABindingResponse(msg_type));
    }
    let cookie = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if cookie != MAGIC_COOKIE {
        return Err(StunError::Malformed(format!(
            "bad magic cookie 0x{cookie:08x}"
        )));
    }
    if bytes[8..HEADER_LEN] != expected_txid[..] {
        return Err(StunError::WrongTransactionId);
    }
    let declared_len = u16::from_be_bytes([bytes[2], bytes[3]]) as usize;
    if bytes.len() < HEADER_LEN + declared_len {
        return Err(StunError::Malformed(format!(
            "declared attribute length {declared_len} exceeds the {} bytes received",
            bytes.len() - HEADER_LEN
        )));
    }

    // Walk the attributes (type 2 + length 2 + value, padded to 4).
    let mut off = HEADER_LEN;
    let end = HEADER_LEN + declared_len;
    while off + 4 <= end {
        let attr_type = u16::from_be_bytes([bytes[off], bytes[off + 1]]);
        let attr_len = u16::from_be_bytes([bytes[off + 2], bytes[off + 3]]) as usize;
        let value_start = off + 4;
        if value_start + attr_len > end {
            return Err(StunError::Malformed(format!(
                "attribute 0x{attr_type:04x} overruns the message"
            )));
        }
        if attr_type == ATTR_XOR_MAPPED_ADDRESS {
            return parse_xor_mapped_address(
                &bytes[value_start..value_start + attr_len],
                expected_txid,
            );
        }
        // Attributes are padded to a multiple of 4 bytes.
        off = value_start + attr_len.next_multiple_of(4);
    }
    Err(StunError::NoMappedAddress)
}

/// Decode one XOR-MAPPED-ADDRESS attribute value: reserved (1),
/// family (1), XOR'd port (2), XOR'd address (4 or 16). The port is
/// XOR'd with the top half of the magic cookie; the address is XOR'd
/// with the cookie (IPv4) or the cookie followed by the transaction
/// id (IPv6).
fn parse_xor_mapped_address(
    value: &[u8],
    txid: &[u8; TRANSACTION_ID_LEN],
) -> Result<SocketAddr, StunError> {
    if value.len() < 8 {
        return Err(StunError::Malformed(format!(
            "XOR-MAPPED-ADDRESS is {} bytes, need at least 8",
            value.len()
        )));
    }
    if value[0] != 0 {
        return Err(StunError::Malformed(
            "XOR-MAPPED-ADDRESS reserved byte is not zero".to_string(),
        ));
    }
    let port_xored = u16::from_be_bytes([value[2], value[3]]);
    let port = port_xored ^ (MAGIC_COOKIE >> 16) as u16;
    match value[1] {
        FAMILY_IPV4 => {
            if value.len() < 8 {
                return Err(StunError::Malformed(
                    "IPv4 XOR-MAPPED-ADDRESS is shorter than 8 bytes".to_string(),
                ));
            }
            let cookie = MAGIC_COOKIE.to_be_bytes();
            let addr = Ipv4Addr::new(
                value[4] ^ cookie[0],
                value[5] ^ cookie[1],
                value[6] ^ cookie[2],
                value[7] ^ cookie[3],
            );
            Ok(SocketAddr::new(IpAddr::V4(addr), port))
        }
        FAMILY_IPV6 => {
            if value.len() < 20 {
                return Err(StunError::Malformed(
                    "IPv6 XOR-MAPPED-ADDRESS is shorter than 20 bytes".to_string(),
                ));
            }
            let mut key = MAGIC_COOKIE.to_be_bytes().to_vec();
            key.extend_from_slice(txid);
            let mut octets = [0u8; 16];
            for (i, b) in octets.iter_mut().enumerate() {
                *b = value[4 + i] ^ key[i];
            }
            Ok(SocketAddr::new(
                IpAddr::V6(Ipv6Addr::from(octets)),
                port,
            ))
        }
        family => Err(StunError::Malformed(format!(
            "unknown address family 0x{family:02x} in XOR-MAPPED-ADDRESS"
        ))),
    }
}

/// A STUN client bound to one UDP socket. [`query`](Self::query)
/// connects the socket to the server (once — a second `connect()`
/// on the same socket is refused here), so one client socket serves
/// exactly one server; a caller that queries several servers needs
/// one client (and one socket) per server. The socket is left
/// connected to the server after the query.
pub struct StunClient<'a> {
    socket: &'a UdpSocket,
}

impl<'a> StunClient<'a> {
    /// Borrow a bound UDP socket. Nothing is sent until [`query`](Self::query).
    pub fn new(socket: &'a UdpSocket) -> Self {
        Self { socket }
    }

    /// Ask `server` for our public address. Connects the socket to
    /// the server, sends one binding request, and waits up to
    /// `timeout` for the response. The socket's read timeout is
    /// restored afterwards. A `Timeout` also covers the case where
    /// the server address has nothing listening on it (an ICMP
    /// port-unreachable means no answer, same as silence).
    pub fn query(&self, server: SocketAddr, timeout: Duration) -> Result<SocketAddr, StunError> {
        let (req, txid) = build_binding_request()?;
        crate::ensure_connected(self.socket, server)?;
        if let Err(e) = self.socket.send(&req) {
            if e.kind() == io::ErrorKind::ConnectionRefused {
                return Err(StunError::Timeout);
            }
            return Err(StunError::Io(e));
        }
        self.socket.set_read_timeout(Some(timeout))?;
        let mut buf = [0u8; 1024];
        let result = match self.socket.recv_from(&mut buf) {
            Ok((len, _from)) => parse_binding_response(&buf[..len], &txid),
            Err(e) if crate::is_no_answer(&e) => Err(StunError::Timeout),
            Err(e) => Err(StunError::Io(e)),
        };
        let _ = self.socket.set_read_timeout(None);
        result
    }
}

/// Build a canned binding success response reporting `mapped`.
/// Test helper, shared with the other modules' tests.
#[cfg(test)]
pub(crate) fn canned_response(
    txid: &[u8; TRANSACTION_ID_LEN],
    mapped: SocketAddr,
) -> Vec<u8> {
    let mut attr = Vec::new();
    attr.push(0u8); // reserved
    match mapped.ip() {
        IpAddr::V4(ip) => {
            attr.push(FAMILY_IPV4);
            attr.extend_from_slice(&(mapped.port() ^ (MAGIC_COOKIE >> 16) as u16).to_be_bytes());
            let cookie = MAGIC_COOKIE.to_be_bytes();
            for (i, b) in ip.octets().iter().enumerate() {
                attr.push(b ^ cookie[i]);
            }
        }
        IpAddr::V6(ip) => {
            attr.push(FAMILY_IPV6);
            attr.extend_from_slice(&(mapped.port() ^ (MAGIC_COOKIE >> 16) as u16).to_be_bytes());
            let mut key = MAGIC_COOKIE.to_be_bytes().to_vec();
            key.extend_from_slice(txid);
            for (i, b) in ip.octets().iter().enumerate() {
                attr.push(b ^ key[i]);
            }
        }
    }
    let mut resp = Vec::with_capacity(HEADER_LEN + 4 + attr.len());
    resp.extend_from_slice(&BINDING_RESPONSE.to_be_bytes());
    resp.extend_from_slice(&((4 + attr.len()) as u16).to_be_bytes());
    resp.extend_from_slice(&MAGIC_COOKIE.to_be_bytes());
    resp.extend_from_slice(txid);
    resp.extend_from_slice(&ATTR_XOR_MAPPED_ADDRESS.to_be_bytes());
    resp.extend_from_slice(&(attr.len() as u16).to_be_bytes());
    resp.extend_from_slice(&attr);
    // Attributes are padded to a multiple of 4; attr is 8 or 20
    // bytes here, already aligned.
    resp
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::UdpSocket;
    use std::thread;

    /// Build a canned binding success response reporting `mapped`.
    /// Used by the parse tests and by the fake STUN server below.
    fn fake_stun_server(report_as: SocketAddr) -> (SocketAddr, thread::JoinHandle<()>) {
        let server = UdpSocket::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let mut buf = [0u8; 1024];
            let (len, from) = server.recv_from(&mut buf).unwrap();
            assert_eq!(u16::from_be_bytes([buf[0], buf[1]]), BINDING_REQUEST);
            let mut txid = [0u8; TRANSACTION_ID_LEN];
            txid.copy_from_slice(&buf[8..HEADER_LEN]);
            let resp = canned_response(&txid, report_as);
            // Connect once, then send: send_to on an unconnected
            // socket is refused by this machine's sandbox.
            server.connect(from).unwrap();
            server.send(&resp).unwrap();
            // Keep the socket alive until the test ends so nothing else
            // reuses the port mid-test.
            let _ = len;
        });
        (addr, handle)
    }

    #[test]
    fn request_encodes_correctly() {
        let (req, txid) = build_binding_request().unwrap();
        assert_eq!(req.len(), HEADER_LEN);
        assert_eq!(u16::from_be_bytes([req[0], req[1]]), BINDING_REQUEST);
        assert_eq!(u16::from_be_bytes([req[2], req[3]]), 0);
        assert_eq!(
            u32::from_be_bytes([req[4], req[5], req[6], req[7]]),
            MAGIC_COOKIE
        );
        assert_eq!(&req[8..HEADER_LEN], &txid[..]);
        // Two requests never share a transaction id.
        let (_req2, txid2) = build_binding_request().unwrap();
        assert_ne!(txid, txid2);
    }

    #[test]
    fn response_parses_xor_mapped_address_ipv4() {
        let txid = [7u8; TRANSACTION_ID_LEN];
        let mapped: SocketAddr = "203.0.113.7:54321".parse().unwrap();
        let resp = canned_response(&txid, mapped);
        assert_eq!(parse_binding_response(&resp, &txid).unwrap(), mapped);
    }

    #[test]
    fn response_parses_xor_mapped_address_ipv6() {
        let txid = [9u8; TRANSACTION_ID_LEN];
        let mapped: SocketAddr = "[2001:db8::1]:12345".parse().unwrap();
        let resp = canned_response(&txid, mapped);
        assert_eq!(parse_binding_response(&resp, &txid).unwrap(), mapped);
    }

    #[test]
    fn response_with_wrong_transaction_id_refused() {
        let txid = [7u8; TRANSACTION_ID_LEN];
        let mapped: SocketAddr = "203.0.113.7:54321".parse().unwrap();
        let resp = canned_response(&txid, mapped);
        let other = [8u8; TRANSACTION_ID_LEN];
        assert!(matches!(
            parse_binding_response(&resp, &other),
            Err(StunError::WrongTransactionId)
        ));
    }

    #[test]
    fn response_with_wrong_type_refused() {
        let txid = [7u8; TRANSACTION_ID_LEN];
        let mapped: SocketAddr = "203.0.113.7:54321".parse().unwrap();
        let mut resp = canned_response(&txid, mapped);
        resp[0..2].copy_from_slice(&ERROR_RESPONSE.to_be_bytes());
        assert!(matches!(
            parse_binding_response(&resp, &txid),
            Err(StunError::NotABindingResponse(ERROR_RESPONSE))
        ));
    }

    #[test]
    fn response_with_bad_cookie_refused() {
        let txid = [7u8; TRANSACTION_ID_LEN];
        let mapped: SocketAddr = "203.0.113.7:54321".parse().unwrap();
        let mut resp = canned_response(&txid, mapped);
        resp[4..8].copy_from_slice(&0xDEAD_BEEFu32.to_be_bytes());
        assert!(matches!(
            parse_binding_response(&resp, &txid),
            Err(StunError::Malformed(_))
        ));
    }

    #[test]
    fn truncated_response_refused() {
        let txid = [7u8; TRANSACTION_ID_LEN];
        let mapped: SocketAddr = "203.0.113.7:54321".parse().unwrap();
        let resp = canned_response(&txid, mapped);
        assert!(matches!(
            parse_binding_response(&resp[..10], &txid),
            Err(StunError::Malformed(_))
        ));
        // Header intact but the attribute cut off.
        assert!(matches!(
            parse_binding_response(&resp[..HEADER_LEN + 4], &txid),
            Err(StunError::Malformed(_))
        ));
    }

    #[test]
    fn response_without_mapped_address_refused() {
        let txid = [7u8; TRANSACTION_ID_LEN];
        // A valid header with zero attributes.
        let mut resp = Vec::with_capacity(HEADER_LEN);
        resp.extend_from_slice(&BINDING_RESPONSE.to_be_bytes());
        resp.extend_from_slice(&0u16.to_be_bytes());
        resp.extend_from_slice(&MAGIC_COOKIE.to_be_bytes());
        resp.extend_from_slice(&txid);
        assert!(matches!(
            parse_binding_response(&resp, &txid),
            Err(StunError::NoMappedAddress)
        ));
    }

    #[test]
    fn fake_stun_server_round_trip() {
        let report_as: SocketAddr = "203.0.113.7:54321".parse().unwrap();
        let (server_addr, handle) = fake_stun_server(report_as);
        let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
        let client = StunClient::new(&socket);
        let got = client.query(server_addr, Duration::from_secs(2)).unwrap();
        assert_eq!(got, report_as);
        handle.join().unwrap();
    }

    #[test]
    fn query_times_out_when_no_server_answers() {
        let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
        // Nobody listens here; the request goes nowhere.
        let dead: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let client = StunClient::new(&socket);
        assert!(matches!(
            client.query(dead, Duration::from_millis(200)),
            Err(StunError::Timeout)
        ));
        // The socket is usable again after a timeout.
        assert!(socket.set_read_timeout(None).is_ok());
    }
}
