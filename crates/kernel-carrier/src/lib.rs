//! KERNEL carrier cell: Phases 0 to 2.
//!
//! Design: the carrier design doc, Sections 4, 5, 6, and 9. Option B: WiFi
//! sessions between phones and nodes, and between nodes. No radio code, no
//! cellular code, no JSON.
//!
//! STATUS
//! - Phase 0 (handshake, sessions, framing, pins, X25519): VERIFIED. 38 tests.
//! - Phase 1 (TCP transport, gateway table): VERIFIED. 60 tests in total with
//!   the PinSet persistence tests. Two nodes handshake and exchange sealed data
//!   over real TCP on 127.0.0.1.
//! - Phase 2 (voice, rekeying, Hello-flood protection): VERIFIED 2026-11-14 by
//!   Muse in this working copy: `cargo test` — 112 passed, 0 failed (106
//!   recovered/implemented tests + 6 new time-based rekey tests).
//!   The wire protocol changed: a responder now sends a cookie challenge
//!   before it reads a Hello, so Phase 1 nodes cannot handshake with Phase 2
//!   nodes.
//!
//! This working copy implements the pieces the other session's files were
//! missing: the Phase 2 frame kinds (Rekey, Cookie), the Phase 2 error
//! variants (BadCookie, RekeyPending, QueueFull), the `hmac` dependency, and
//! binary PinSet persistence. The files as sent, byte-for-byte, live in
//! `../kernel-carrier-phase0/` and were not touched.
//!
//! Modules
//! - `frame`: 12-byte binary header, bounded bodies. Kinds include Rekey and Cookie.
//! - `handshake`: SIGMA-style mutual authentication, pinned keys only.
//! - `session`: sealed Data frames, 64-entry replay window, rekeying
//!   (frame-count and time-based triggers, 10-minute default).
//! - `pins`: pinned public keys with sticky revocation, plus binary
//!   persistence (`KCPIN1` file format, atomic save).
//! - `flood`: cookie challenge, proof-of-work puzzle, per-address guard.
//! - `transport`: TCP, std only. Cookie, handshake, sealed frames, one deadline per handshake.
//! - `gateway`: peer key to socket address, pinned peers only.
//! - `voice`: codec trait with MockCodec, jitter buffer, call setup.
//! - `error`: refusal codes.
//!
//! Not in this crate yet: real audio codec (Opus), radio trait, multi-hop
//! routing, discovery, identity hiding.
//!
//! `unsafe` is forbidden crate-wide.

#![forbid(unsafe_code)]

pub mod error;
pub mod flood;
pub mod frame;
pub mod gateway;
pub mod handshake;
pub mod pins;
pub mod session;
pub mod transport;
pub mod voice;

mod kex;

pub use error::CarrierError;
pub use gateway::GatewayTable;
pub use handshake::{Initiator, Responder};
pub use pins::PinSet;
pub use session::{Event, Role, Session};
pub use transport::{
    accept_channel, connect_channel, connect_via, Listener, SecureChannel, TransportError,
};
pub use voice::{Codec, JitterBuffer, MockCodec, VoiceError};
