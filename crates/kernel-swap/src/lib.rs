//! Atomic swaps for KERNEL: hash-time-locked contracts (HTLC) over a
//! local ledger.
//!
//! A swap locks funds in escrow under a hashlock (SHA-256 of a secret
//! preimage) and a timeout. The claimer can take the funds by revealing
//! the preimage before the timeout; after the timeout the locker can
//! take them back. That is the whole atomicity trick: either the secret
//! is revealed and both sides of a two-party swap complete, or it is
//! not and both sides refund.
//!
//! This is an HTLC over a **local in-memory ledger**, not real
//! Bitcoin/Ethereum interop. Real chain interop needs chain clients,
//! on-chain scripts, and real chain finality — none of that exists
//! here. Timeouts use caller-supplied unix seconds; the state machine
//! never reads the system clock, so tests are deterministic.
//!
//! State machine: `Proposed -> Locked -> Claimed | Refunded | Expired`.
//! `Expired` is a timed-out lock whose funds are still in escrow
//! awaiting the locker's refund — it is not a loss of funds.

#![forbid(unsafe_code)]

pub mod error;
pub mod ledger;
pub mod swap;

pub use error::SwapError;
pub use ledger::{Ledger, MemLedger};
pub use swap::{Swap, SwapId, SwapState};
