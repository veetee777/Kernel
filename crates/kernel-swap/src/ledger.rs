//! The ledger abstraction a swap runs against, plus the in-memory
//! implementation used by tests (and by anything that wants a local
//! ledger without a chain behind it).
//!
//! Accounting rule, enforced by the implementation: a swap's funds are
//! accounted for exactly once in every path. Locking debits the locker
//! and records the same amount in escrow; claiming or refunding removes
//! the escrow and credits the same amount to exactly one account.
//! `MemLedger::total` (all balances plus all escrow) is constant across
//! every operation — tests assert it after every step.

use std::collections::HashMap;

use kernel_identity::PeerId;

use crate::error::SwapError;
use crate::swap::SwapId;

/// A minimal ledger a swap can lock funds into and release funds from.
/// Balances are plain integer amounts in the ledger's smallest unit;
/// the trait says nothing about what a unit is worth.
pub trait Ledger {
    /// Balance of `account`, or 0 for an account that never held funds.
    fn balance(&self, account: &PeerId) -> u64;

    /// Remove `amount` from `account`. Fails with
    /// [`SwapError::InsufficientFunds`] if the balance is too low.
    /// A zero debit always succeeds.
    fn debit(&mut self, account: &PeerId, amount: u64) -> Result<(), SwapError>;

    /// Add `amount` to `account`. Fails with [`SwapError::Overflow`]
    /// if the balance would overflow. A zero credit always succeeds.
    fn credit(&mut self, account: &PeerId, amount: u64) -> Result<(), SwapError>;

    /// Amount currently held in escrow for `swap_id`, or 0.
    fn escrowed(&self, swap_id: &SwapId) -> u64;

    /// Move `amount` from `account` into escrow for `swap_id`. Fails
    /// with [`SwapError::ZeroAmount`] for a zero amount,
    /// [`SwapError::AlreadyLocked`] if escrow already exists for the
    /// id, and [`SwapError::InsufficientFunds`] if the account is
    /// short. On any failure nothing moves.
    fn lock(&mut self, swap_id: &SwapId, from: &PeerId, amount: u64) -> Result<(), SwapError>;

    /// Move the escrowed amount for `swap_id` to `to` and return it.
    /// Fails with [`SwapError::EscrowMissing`] if nothing is escrowed
    /// for the id. On failure nothing moves.
    fn release(&mut self, swap_id: &SwapId, to: &PeerId) -> Result<u64, SwapError>;
}

/// An in-memory ledger: account balances plus per-swap escrow, both in
/// `HashMap`s. Nothing here touches a disk or a chain.
#[derive(Debug, Default, Clone)]
pub struct MemLedger {
    balances: HashMap<PeerId, u64>,
    escrow: HashMap<SwapId, u64>,
}

impl MemLedger {
    /// An empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create money for an account. This is setup-only (tests, demos):
    /// real ledgers do not mint on demand. Callers that assert funds
    /// conservation must mint before the measured window, never inside
    /// it.
    pub fn mint(&mut self, account: &PeerId, amount: u64) {
        let entry = self.balances.entry(account.clone()).or_insert(0);
        *entry = entry.saturating_add(amount);
    }

    /// Total money in the ledger: all balances plus all escrow.
    /// Constant across every swap operation; summed in `u128` so the
    /// total itself cannot overflow.
    pub fn total(&self) -> u128 {
        let balances: u128 = self.balances.values().map(|b| *b as u128).sum();
        let escrow: u128 = self.escrow.values().map(|e| *e as u128).sum();
        balances + escrow
    }

    /// Number of accounts that hold a balance (escrow excluded).
    pub fn account_count(&self) -> usize {
        self.balances.len()
    }
}

impl Ledger for MemLedger {
    fn balance(&self, account: &PeerId) -> u64 {
        self.balances.get(account).copied().unwrap_or(0)
    }

    fn debit(&mut self, account: &PeerId, amount: u64) -> Result<(), SwapError> {
        let balance = self.balance(account);
        if balance < amount {
            return Err(SwapError::InsufficientFunds);
        }
        self.balances.insert(account.clone(), balance - amount);
        Ok(())
    }

    fn credit(&mut self, account: &PeerId, amount: u64) -> Result<(), SwapError> {
        let balance = self.balance(account);
        let new = balance.checked_add(amount).ok_or(SwapError::Overflow)?;
        self.balances.insert(account.clone(), new);
        Ok(())
    }

    fn escrowed(&self, swap_id: &SwapId) -> u64 {
        self.escrow.get(swap_id).copied().unwrap_or(0)
    }

    fn lock(&mut self, swap_id: &SwapId, from: &PeerId, amount: u64) -> Result<(), SwapError> {
        if amount == 0 {
            return Err(SwapError::ZeroAmount);
        }
        if self.escrowed(swap_id) > 0 {
            return Err(SwapError::AlreadyLocked);
        }
        // Validate everything before moving anything.
        if self.balance(from) < amount {
            return Err(SwapError::InsufficientFunds);
        }
        self.debit(from, amount)?;
        self.escrow.insert(*swap_id, amount);
        Ok(())
    }

    fn release(&mut self, swap_id: &SwapId, to: &PeerId) -> Result<u64, SwapError> {
        let amount = self.escrowed(swap_id);
        if amount == 0 {
            return Err(SwapError::EscrowMissing);
        }
        // Validate the credit before moving anything, so a failure
        // cannot strand funds outside both escrow and the account.
        let new_balance = self
            .balance(to)
            .checked_add(amount)
            .ok_or(SwapError::Overflow)?;
        self.escrow.remove(swap_id);
        self.balances.insert(to.clone(), new_balance);
        Ok(amount)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swap::SwapId;

    fn alice() -> PeerId {
        kernel_identity::Identity::generate("alice")
            .unwrap()
            .peer_id()
            .clone()
    }

    fn bob() -> PeerId {
        kernel_identity::Identity::generate("bob")
            .unwrap()
            .peer_id()
            .clone()
    }

    #[test]
    fn unknown_account_has_zero_balance() {
        let ledger = MemLedger::new();
        assert_eq!(ledger.balance(&alice()), 0);
        assert_eq!(ledger.total(), 0);
    }

    #[test]
    fn debit_and_credit_move_funds() {
        let mut ledger = MemLedger::new();
        let a = alice();
        ledger.mint(&a, 100);
        assert_eq!(ledger.total(), 100);
        ledger.debit(&a, 40).unwrap();
        assert_eq!(ledger.balance(&a), 60);
        // A bare debit removes funds from the ledger (total drops);
        // conservation holds across lock/release, not across debit.
        assert_eq!(ledger.total(), 60);
        ledger.credit(&a, 40).unwrap();
        assert_eq!(ledger.balance(&a), 100);
        assert_eq!(ledger.total(), 100);
    }

    #[test]
    fn debit_more_than_balance_is_refused_and_moves_nothing() {
        let mut ledger = MemLedger::new();
        let a = alice();
        ledger.mint(&a, 10);
        assert_eq!(ledger.debit(&a, 11), Err(SwapError::InsufficientFunds));
        assert_eq!(ledger.balance(&a), 10);
        assert_eq!(ledger.total(), 10);
    }

    #[test]
    fn lock_moves_funds_into_escrow_and_release_moves_them_out() {
        let mut ledger = MemLedger::new();
        let a = alice();
        let b = bob();
        let id = SwapId::from_bytes([7u8; 32]);
        ledger.mint(&a, 100);
        assert_eq!(ledger.total(), 100);

        ledger.lock(&id, &a, 60).unwrap();
        assert_eq!(ledger.balance(&a), 40);
        assert_eq!(ledger.escrowed(&id), 60);
        assert_eq!(ledger.total(), 100);

        let released = ledger.release(&id, &b).unwrap();
        assert_eq!(released, 60);
        assert_eq!(ledger.balance(&b), 60);
        assert_eq!(ledger.escrowed(&id), 0);
        assert_eq!(ledger.total(), 100);
    }

    #[test]
    fn lock_twice_for_one_swap_id_is_refused() {
        let mut ledger = MemLedger::new();
        let a = alice();
        let id = SwapId::from_bytes([7u8; 32]);
        ledger.mint(&a, 100);
        ledger.lock(&id, &a, 60).unwrap();
        assert_eq!(ledger.lock(&id, &a, 10), Err(SwapError::AlreadyLocked));
        assert_eq!(ledger.balance(&a), 40);
        assert_eq!(ledger.escrowed(&id), 60);
        assert_eq!(ledger.total(), 100);
    }

    #[test]
    fn lock_of_zero_or_more_than_balance_is_refused() {
        let mut ledger = MemLedger::new();
        let a = alice();
        let id = SwapId::from_bytes([7u8; 32]);
        ledger.mint(&a, 10);
        assert_eq!(ledger.lock(&id, &a, 0), Err(SwapError::ZeroAmount));
        assert_eq!(ledger.lock(&id, &a, 11), Err(SwapError::InsufficientFunds));
        assert_eq!(ledger.balance(&a), 10);
        assert_eq!(ledger.escrowed(&id), 0);
        assert_eq!(ledger.total(), 10);
    }

    #[test]
    fn release_without_escrow_is_refused() {
        let mut ledger = MemLedger::new();
        let b = bob();
        let id = SwapId::from_bytes([7u8; 32]);
        assert_eq!(ledger.release(&id, &b), Err(SwapError::EscrowMissing));
        assert_eq!(ledger.balance(&b), 0);
        assert_eq!(ledger.total(), 0);
    }

    #[test]
    fn credit_overflow_is_refused_and_moves_nothing() {
        let mut ledger = MemLedger::new();
        let a = alice();
        ledger.mint(&a, u64::MAX);
        assert_eq!(ledger.credit(&a, 1), Err(SwapError::Overflow));
        assert_eq!(ledger.balance(&a), u64::MAX);
    }
}
