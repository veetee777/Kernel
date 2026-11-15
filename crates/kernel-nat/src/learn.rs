//! NAT classification from STUN observations.
//!
//! The classification is deliberately small and honest about what
//! each observation can and cannot prove:
//!
//! * Query two (or more) different STUN servers, one socket per
//!   server (a UDP socket can only be connected to one peer). If
//!   the mapped public address **changes** with the destination,
//!   the NAT assigns a new mapping per destination: **symmetric**.
//! * If the mapped address is the **same** for every server, the NAT
//!   preserves the mapping across destinations. That alone cannot
//!   separate full cone from restricted cone — the difference is
//!   whether a packet from an address the peer never contacted is
//!   let in. So an optional inbound probe settles it: the caller
//!   arranges for a third party (at `probe_from`, an address that is
//!   *not* one of the STUN servers) to send one packet to the mapped
//!   address while `learn_nat` listens. If it arrives: **full cone**.
//!   If nothing arrives before the timeout: **restricted**.
//! * If the mapping is stable but no probe was arranged, the answer
//!   is **unknown** — not a guess. A caller that wants the
//!   distinction must arrange the probe.
//!
//! On loopback there is no NAT at all, so the fake-server tests
//! report full cone when a probe arrives and restricted when it
//! does not — which is exactly what an open network looks like.

use std::fmt;
use std::io;
use std::net::{SocketAddr, UdpSocket};
use std::time::Duration;

use crate::stun::{StunClient, StunError};

/// Best-guess NAT type, from STUN observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NatType {
    /// Any outside address can send to the mapped address.
    FullCone,
    /// Only addresses the peer already contacted can send in.
    Restricted,
    /// A new mapping is created per destination; punching usually
    /// fails and the relay is the way through.
    Symmetric,
    /// Not enough evidence to classify (stable mapping, no probe).
    Unknown,
}

impl fmt::Display for NatType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NatType::FullCone => f.write_str("full cone"),
            NatType::Restricted => f.write_str("restricted cone"),
            NatType::Symmetric => f.write_str("symmetric"),
            NatType::Unknown => f.write_str("unknown"),
        }
    }
}

/// What address learning found: the public address the STUN servers
/// reported, the NAT type that best fits the observations, and every
/// mapped address observed (one per server queried).
#[derive(Debug, Clone)]
pub struct NatInfo {
    /// The public address, as reported by the first server queried.
    pub public_addr: SocketAddr,
    /// Best-guess NAT type.
    pub nat_type: NatType,
    /// Every mapped address observed, in query order.
    pub observations: Vec<SocketAddr>,
}

/// Learn this socket's public address and NAT type.
///
/// Each server in `servers` is queried from its own fresh socket
/// (one socket per server: a UDP socket can only be connected to
/// one peer, and this machine's sandbox only allows `send()` on a
/// connected socket). Each query uses a fresh transaction id. The
/// socket passed in is left unconnected — it is the socket the
/// probe must arrive on. If `probe_from` is `Some(addr)`, waits up
/// to `timeout` after the queries for one datagram from `addr` — the
/// caller arranges for that datagram to be sent to `public_addr`
/// (the rendezvous normally does this). The probe address must differ
/// from every server address, or a late STUN response could be
/// mistaken for the probe.
///
/// Honest limit: because the queries go out on per-server sockets,
/// the mapped addresses observed belong to those query sockets,
/// not to the socket passed in. On a machine where `send_to`
/// works, querying from the passed-in socket gives that socket's
/// own mapping; the code here cannot do that, because a connected
/// socket only receives from its peer and the probe has to arrive
/// on the passed-in socket afterwards.
///
/// Returns an error if no servers are given, or if any query fails.
pub fn learn_nat(
    socket: &UdpSocket,
    servers: &[SocketAddr],
    probe_from: Option<SocketAddr>,
    timeout: Duration,
) -> Result<NatInfo, StunError> {
    if servers.is_empty() {
        return Err(StunError::Malformed(
            "no STUN servers given".to_string(),
        ));
    }
    let mut observations = Vec::with_capacity(servers.len());
    for &server in servers {
        let qsock = UdpSocket::bind("0.0.0.0:0")?;
        let client = StunClient::new(&qsock);
        observations.push(client.query(server, timeout)?);
    }
    let public_addr = observations[0];
    let stable = observations.iter().all(|a| *a == public_addr);
    if !stable {
        return Ok(NatInfo {
            public_addr,
            nat_type: NatType::Symmetric,
            observations,
        });
    }
    let Some(probe) = probe_from else {
        return Ok(NatInfo {
            public_addr,
            nat_type: NatType::Unknown,
            observations,
        });
    };
    // Stable mapping: the probe decides full cone vs restricted.
    socket.set_read_timeout(Some(timeout))?;
    let mut buf = [0u8; 64];
    let nat_type = loop {
        match socket.recv_from(&mut buf) {
            Ok((_len, from)) if from == probe => break NatType::FullCone,
            Ok((_len, _from)) => continue, // not the probe; keep waiting
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => break NatType::Restricted,
            Err(e) => return Err(StunError::Io(e)),
        }
    };
    let _ = socket.set_read_timeout(None);
    Ok(NatInfo {
        public_addr,
        nat_type,
        observations,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stun::{canned_response, HEADER_LEN, TRANSACTION_ID_LEN};
    use std::net::UdpSocket;
    use std::thread;

    /// A fake STUN server that answers every request with a canned
    /// response reporting `report_as`, until `rounds` requests have
    /// been answered. Runs on its own thread.
    fn fake_stun_server(
        report_as: SocketAddr,
        rounds: usize,
    ) -> (SocketAddr, thread::JoinHandle<()>) {
        let server = UdpSocket::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let mut buf = [0u8; 1024];
            for _ in 0..rounds {
                let (len, from) = server.recv_from(&mut buf).unwrap();
                let _ = len;
                let mut txid = [0u8; TRANSACTION_ID_LEN];
                txid.copy_from_slice(&buf[8..HEADER_LEN]);
                let resp = canned_response(&txid, report_as);
                // Connect once; a second connect() is refused by
                // this machine's sandbox, so only connect when the
                // peer changed.
                if server.peer_addr().map(|a| a != from).unwrap_or(true) {
                    server.connect(from).unwrap();
                }
                server.send(&resp).unwrap();
            }
        });
        (addr, handle)
    }

    fn client_socket() -> UdpSocket {
        UdpSocket::bind("127.0.0.1:0").unwrap()
    }

    #[test]
    fn learn_returns_the_reported_address() {
        let report_as: SocketAddr = "203.0.113.7:54321".parse().unwrap();
        let (server, handle) = fake_stun_server(report_as, 1);
        let socket = client_socket();
        let info = learn_nat(&socket, &[server], None, Duration::from_secs(2)).unwrap();
        assert_eq!(info.public_addr, report_as);
        assert_eq!(info.observations, vec![report_as]);
        // Stable mapping, no probe arranged: honestly unknown.
        assert_eq!(info.nat_type, NatType::Unknown);
        handle.join().unwrap();
    }

    #[test]
    fn learn_detects_symmetric_nat() {
        let a: SocketAddr = "203.0.113.7:40001".parse().unwrap();
        let b: SocketAddr = "198.51.100.9:40002".parse().unwrap();
        let (server_a, handle_a) = fake_stun_server(a, 1);
        let (server_b, handle_b) = fake_stun_server(b, 1);
        let socket = client_socket();
        let info = learn_nat(&socket, &[server_a, server_b], None, Duration::from_secs(2))
            .unwrap();
        assert_eq!(info.public_addr, a);
        assert_eq!(info.observations, vec![a, b]);
        assert_eq!(info.nat_type, NatType::Symmetric);
        handle_a.join().unwrap();
        handle_b.join().unwrap();
    }

    #[test]
    fn learn_full_cone_when_probe_arrives() {
        let report_as: SocketAddr = "203.0.113.7:54321".parse().unwrap();
        let (server, server_handle) = fake_stun_server(report_as, 1);
        let socket = client_socket();
        let client_addr = socket.local_addr().unwrap();

        // A third party sends one packet to the mapped address — on
        // loopback the "mapped" address is the client's own address,
        // so the probe is sent there.
        let probe_sock = UdpSocket::bind("127.0.0.1:0").unwrap();
        let probe_addr = probe_sock.local_addr().unwrap();
        let probe_thread = thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            probe_sock.connect(client_addr).unwrap();
            probe_sock.send(b"KPROBE").unwrap();
        });
        let info = learn_nat(
            &socket,
            &[server],
            Some(probe_addr),
            Duration::from_secs(2),
        )
        .unwrap();
        assert_eq!(info.public_addr, report_as);
        assert_eq!(info.nat_type, NatType::FullCone);
        probe_thread.join().unwrap();
        server_handle.join().unwrap();
    }

    #[test]
    fn learn_restricted_when_probe_never_arrives() {
        let report_as: SocketAddr = "203.0.113.7:54321".parse().unwrap();
        let (server, server_handle) = fake_stun_server(report_as, 1);
        let socket = client_socket();
        // A probe address that never sends anything.
        let probe_sock = UdpSocket::bind("127.0.0.1:0").unwrap();
        let probe_addr = probe_sock.local_addr().unwrap();
        let info = learn_nat(
            &socket,
            &[server],
            Some(probe_addr),
            Duration::from_millis(300),
        )
        .unwrap();
        assert_eq!(info.public_addr, report_as);
        assert_eq!(info.nat_type, NatType::Restricted);
        server_handle.join().unwrap();
    }

    #[test]
    fn learn_refuses_empty_server_list() {
        let socket = client_socket();
        assert!(matches!(
            learn_nat(&socket, &[], None, Duration::from_secs(1)),
            Err(StunError::Malformed(_))
        ));
    }
}
