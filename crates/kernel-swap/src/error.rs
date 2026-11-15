//! Errors returned by swap operations. Every failure mode is a clean
//! error value — a swap never panics, whatever the caller passes in.

use std::fmt;

/// Errors returned by swap and ledger operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwapError {
    /// `lock` was called on a swap that is not `Proposed`.
    NotProposed,
    /// An operation needed a `Locked` swap and the swap is not locked.
    NotLocked,
    /// The swap is already in a terminal state (`Claimed` or
    /// `Refunded`) and cannot move again.
    AlreadySettled,
    /// `lock` was called on a swap whose id already holds escrow.
    AlreadyLocked,
    /// The preimage does not hash to the swap's hashlock.
    WrongPreimage,
    /// The caller is not the swap's designated claimer.
    WrongClaimer,
    /// The caller is not the swap's locker.
    WrongLocker,
    /// A refund was attempted before the timeout.
    TooEarly,
    /// A claim was attempted at or after the timeout.
    TooLate,
    /// `expire` was called before the timeout.
    NotExpired,
    /// The lock amount is zero.
    ZeroAmount,
    /// The account does not hold enough to lock or debit.
    InsufficientFunds,
    /// A credit would overflow the account balance.
    Overflow,
    /// A release was attempted for a swap with no escrow.
    EscrowMissing,
}

impl fmt::Display for SwapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            SwapError::NotProposed => "swap is not in the Proposed state",
            SwapError::NotLocked => "swap is not Locked",
            SwapError::AlreadySettled => "swap is already settled (Claimed or Refunded)",
            SwapError::AlreadyLocked => "escrow already exists for this swap id",
            SwapError::WrongPreimage => "preimage does not match the hashlock",
            SwapError::WrongClaimer => "caller is not the swap's claimer",
            SwapError::WrongLocker => "caller is not the swap's locker",
            SwapError::TooEarly => "timeout has not passed yet",
            SwapError::TooLate => "timeout has passed",
            SwapError::NotExpired => "swap has not timed out yet",
            SwapError::ZeroAmount => "lock amount must be greater than zero",
            SwapError::InsufficientFunds => "insufficient funds",
            SwapError::Overflow => "balance would overflow",
            SwapError::EscrowMissing => "no escrow exists for this swap",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for SwapError {}
