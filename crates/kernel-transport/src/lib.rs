//! TCP transport for KERNEL.
//!
//! Peers talk over plain TCP sockets with a tiny framing protocol: every
//! message is a 4-byte big-endian length prefix followed by that many
//! payload bytes. The frame length is capped (64 MiB) so a corrupt or
//! hostile peer cannot make a receiver allocate unbounded memory.
//!
//! This crate is deliberately small and dependency-free (std only). It
//! moves bytes; it does not authenticate peers, encrypt traffic, or
//! understand KERNEL messages. Authentication and encryption belong in
//! the layers above (identity signatures, AEAD sessions).

use std::fmt;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::time::Duration;

/// Maximum payload bytes in one frame (64 MiB). Anything larger is a
/// protocol error, not something to allocate for.
pub const MAX_FRAME_SIZE: usize = 64 * 1024 * 1024;

/// Errors returned by transport operations.
#[derive(Debug)]
pub enum TransportError {
    /// An underlying I/O failure (connect refused, reset, timeout, ...).
    Io(std::io::Error),
    /// The peer closed the connection (cleanly or mid-frame).
    ConnectionClosed,
    /// A frame claimed more than [`MAX_FRAME_SIZE`] bytes.
    FrameTooLarge(usize),
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransportError::Io(e) => write!(f, "I/O error: {e}"),
            TransportError::ConnectionClosed => write!(f, "connection closed by peer"),
            TransportError::FrameTooLarge(n) => {
                write!(f, "frame of {n} bytes exceeds maximum {MAX_FRAME_SIZE}")
            }
        }
    }
}

impl std::error::Error for TransportError {}

impl From<std::io::Error> for TransportError {
    fn from(e: std::io::Error) -> Self {
        // A closed peer surfaces as UnexpectedEof on read; report it as
        // a clean close rather than a raw I/O error so callers can tell
        // "peer went away" apart from a real failure.
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            TransportError::ConnectionClosed
        } else {
            TransportError::Io(e)
        }
    }
}

/// A framed TCP connection to one peer.
#[derive(Debug)]
pub struct Connection {
    stream: TcpStream,
}

impl Connection {
    /// Connect to a peer at `addr`. A refused connection (or any other
    /// failure) returns a [`TransportError::Io`]; callers can match on the
    /// inner [`std::io::ErrorKind`] (e.g. `ConnectionRefused`).
    pub fn connect<A: ToSocketAddrs>(addr: A) -> Result<Self, TransportError> {
        let stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;
        Ok(Self { stream })
    }

    /// Build a connection from an already-accepted [`TcpStream`].
    pub(crate) fn from_stream(stream: TcpStream) -> Result<Self, TransportError> {
        stream.set_nodelay(true)?;
        Ok(Self { stream })
    }

    /// The local address of this connection.
    pub fn local_addr(&self) -> Result<SocketAddr, TransportError> {
        Ok(self.stream.local_addr()?)
    }

    /// The peer's address.
    pub fn peer_addr(&self) -> Result<SocketAddr, TransportError> {
        Ok(self.stream.peer_addr()?)
    }

    /// Set a read timeout. `None` clears it (reads block indefinitely).
    /// A timed-out read surfaces as a [`TransportError::Io`] with kind
    /// `WouldBlock`.
    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> Result<(), TransportError> {
        Ok(self.stream.set_read_timeout(timeout)?)
    }

    /// Send one framed message. The payload is written as a 4-byte
    /// big-endian length followed by the payload bytes.
    pub fn send(&mut self, payload: &[u8]) -> Result<(), TransportError> {
        if payload.len() > MAX_FRAME_SIZE {
            return Err(TransportError::FrameTooLarge(payload.len()));
        }
        let len = (payload.len() as u32).to_be_bytes();
        self.stream.write_all(&len)?;
        self.stream.write_all(payload)?;
        self.stream.flush()?;
        Ok(())
    }

    /// Receive one framed message. Blocks until a full frame arrives.
    /// Returns [`TransportError::ConnectionClosed`] if the peer closes
    /// the connection (cleanly or mid-frame) instead of panicking.
    pub fn recv(&mut self) -> Result<Vec<u8>, TransportError> {
        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf)?;
        let len = u32::from_be_bytes(len_buf) as usize;
        if len > MAX_FRAME_SIZE {
            return Err(TransportError::FrameTooLarge(len));
        }
        let mut payload = vec![0u8; len];
        self.stream.read_exact(&mut payload)?;
        Ok(payload)
    }

    /// Half-close the write side, then close the socket. Best effort;
    /// errors are ignored because the peer may already be gone.
    pub fn close(self) {
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
    }
}

/// A listening TCP socket. Accepts incoming peer connections.
#[derive(Debug)]
pub struct Listener {
    listener: TcpListener,
}

impl Listener {
    /// Bind and listen on `addr` (use port 0 for an ephemeral port).
    pub fn bind<A: ToSocketAddrs>(addr: A) -> Result<Self, TransportError> {
        let listener = TcpListener::bind(addr)?;
        Ok(Self { listener })
    }

    /// The address this listener is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr, TransportError> {
        Ok(self.listener.local_addr()?)
    }

    /// Accept one incoming connection, blocking until a peer connects.
    pub fn accept(&self) -> Result<Connection, TransportError> {
        let (stream, _addr) = self.listener.accept()?;
        Connection::from_stream(stream)
    }

    /// Set the listener non-blocking. Useful for servers that poll for a
    /// shutdown flag instead of blocking forever in [`Listener::accept`].
    /// A non-blocking accept with no pending peer returns a
    /// [`TransportError::Io`] with kind `WouldBlock`.
    pub fn set_nonblocking(&self, nonblocking: bool) -> Result<(), TransportError> {
        Ok(self.listener.set_nonblocking(nonblocking)?)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpStream as StdTcpStream;

    fn loopback() -> SocketAddr {
        "127.0.0.1:0".parse().unwrap()
    }

    #[test]
    fn two_peers_exchange_messages_over_loopback() {
        let listener = Listener::bind(loopback()).unwrap();
        let addr = listener.local_addr().unwrap();

        // Server: accept, read one message, reply.
        let server = std::thread::spawn(move || {
            let mut conn = listener.accept().unwrap();
            let msg = conn.recv().unwrap();
            assert_eq!(msg, b"hello from client");
            conn.send(b"hello from server").unwrap();
            conn.close();
        });

        let mut client = Connection::connect(addr).unwrap();
        client.send(b"hello from client").unwrap();
        let reply = client.recv().unwrap();
        assert_eq!(reply, b"hello from server");
        client.close();
        server.join().unwrap();
    }

    #[test]
    fn large_message_round_trip() {
        let listener = Listener::bind(loopback()).unwrap();
        let addr = listener.local_addr().unwrap();
        // 1 MiB of deterministic data.
        let data: Vec<u8> = (0..1024 * 1024).map(|i| (i % 251) as u8).collect();
        let data_for_server = data.clone();

        let server = std::thread::spawn(move || {
            let mut conn = listener.accept().unwrap();
            let got = conn.recv().unwrap();
            assert_eq!(got, data_for_server);
            conn.send(&got).unwrap();
        });

        let mut client = Connection::connect(addr).unwrap();
        client.send(&data).unwrap();
        let echoed = client.recv().unwrap();
        assert_eq!(echoed, data);
        server.join().unwrap();
    }

    #[test]
    fn connection_refused_is_a_clean_error() {
        // Port 1 on loopback is (essentially) never listening.
        let refused: SocketAddr = "127.0.0.1:1".parse().unwrap();
        match Connection::connect(refused) {
            Err(TransportError::Io(e)) => {
                assert_eq!(e.kind(), std::io::ErrorKind::ConnectionRefused);
            }
            other => panic!("expected ConnectionRefused, got {other:?}"),
        }
    }

    #[test]
    fn peer_disconnect_is_handled_without_panic() {
        let listener = Listener::bind(loopback()).unwrap();
        let addr = listener.local_addr().unwrap();

        let server = std::thread::spawn(move || {
            let mut conn = listener.accept().unwrap();
            // The client connects and immediately drops. The read must
            // fail cleanly as ConnectionClosed, not panic or hang.
            conn.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let result = conn.recv();
            assert!(
                matches!(result, Err(TransportError::ConnectionClosed)),
                "expected ConnectionClosed, got {result:?}"
            );
        });

        // Connect, then drop the socket without sending anything.
        let client = Connection::connect(addr).unwrap();
        drop(client);
        server.join().unwrap();
    }

    #[test]
    fn oversized_frame_is_rejected() {
        let listener = Listener::bind(loopback()).unwrap();
        let addr = listener.local_addr().unwrap();

        let server = std::thread::spawn(move || {
            let mut conn = listener.accept().unwrap();
            conn.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            // A header claiming MAX_FRAME_SIZE + 1 bytes must be rejected
            // before any allocation for the payload.
            match conn.recv() {
                Err(TransportError::FrameTooLarge(n)) => {
                    assert_eq!(n, MAX_FRAME_SIZE + 1);
                }
                other => panic!("expected FrameTooLarge, got {other:?}"),
            }
        });

        // Write a raw oversized length header by hand (bypassing send(),
        // which refuses to write one).
        let mut raw = StdTcpStream::connect(addr).unwrap();
        raw.write_all(&((MAX_FRAME_SIZE + 1) as u32).to_be_bytes())
            .unwrap();
        raw.flush().unwrap();
        server.join().unwrap();
    }

    #[test]
    fn sending_an_oversized_payload_is_refused_locally() {
        let listener = Listener::bind(loopback()).unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let _conn = listener.accept().unwrap();
        });
        let mut client = Connection::connect(addr).unwrap();
        let huge = vec![0u8; MAX_FRAME_SIZE + 1];
        assert!(matches!(
            client.send(&huge),
            Err(TransportError::FrameTooLarge(_))
        ));
        server.join().unwrap();
    }

    #[test]
    fn multiple_sequential_messages_on_one_connection() {
        let listener = Listener::bind(loopback()).unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let mut conn = listener.accept().unwrap();
            for i in 0..5u8 {
                let msg = conn.recv().unwrap();
                assert_eq!(msg, vec![i; 10]);
                conn.send(&[i, i, i]).unwrap();
            }
        });
        let mut client = Connection::connect(addr).unwrap();
        for i in 0..5u8 {
            client.send(&vec![i; 10]).unwrap();
            let reply = client.recv().unwrap();
            assert_eq!(reply, vec![i, i, i]);
        }
        server.join().unwrap();
    }
}
