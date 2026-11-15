//! The HTLC swap state machine.
//!
//! ```text
//! Proposed -> Locked -> Claimed | Refunded | Expired
//! ```
//!
//! - `Proposed`: terms agreed (hashlock, amount, locker, claimer,
//!   timeout) but no funds moved yet.
//! - `Locked`: funds are in escrow on the ledger. The claimer can
//!   take them with the preimage before the timeout; the locker can
//!   take them back after the timeout.
//! - `Claimed`: the preimage was revealed in time and the funds went
//!   to the claimer. Terminal.
//! - `Refunded`: the timeout passed and the funds went back to the
//!   locker. Terminal.
//! - `Expired`: the timeout passed and the funds are still in escrow
//!   awaiting the locker's refund. A claim from `Expired` is refused
//!   even with the right preimage — only a refund can settle it.
//!
//! Time is caller-supplied unix seconds (`now`). The timeout is
//! exclusive for claims (`now < timeout` claims) and inclusive for
//! refunds and expiry (`now >= timeout` refunds). The machine never
//! reads the system clock.

use kernel_crypto::{random_bytes, sha256};
use kernel_identity::PeerId;

use crate::error::SwapError;
use crate::ledger::Ledger;

/// Identifier for one swap, unique per lock. Two swaps on two ledgers
/// (the two halves of a cross-ledger swap) have different ids; what
/// links them is the shared hashlock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SwapId(pub [u8; 32]);

impl SwapId {
    /// A fresh random id.
    pub fn random() -> Self {
        Self(random_bytes::<32>().unwrap_or([0u8; 32]))
    }

    /// An id from fixed bytes (tests, or ids derived by the caller).
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The raw bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// The states of a swap. See the module docs for the transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwapState {
    /// Terms set, no funds moved.
    Proposed,
    /// Funds in escrow.
    Locked,
    /// Claimed by the claimer with the preimage. Terminal.
    Claimed,
    /// Refunded to the locker after the timeout. Terminal.
    Refunded,
    /// Timed out with funds still in escrow. Only a refund settles it.
    Expired,
}

impl SwapState {
    /// True for `Claimed` and `Refunded` — the swap can never move again.
    pub fn is_terminal(&self) -> bool {
        matches!(self, SwapState::Claimed | SwapState::Refunded)
    }
}

/// One hash-time-locked contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Swap {
    id: SwapId,
    hashlock: [u8; 32],
    amount: u64,
    locker: PeerId,
    claimer: PeerId,
    timeout: u64,
    state: SwapState,
}

impl Swap {
    /// The hashlock for a secret preimage: SHA-256 of the preimage.
    pub fn hashlock_for(preimage: &[u8]) -> [u8; 32] {
        sha256(preimage)
    }

    /// Propose a swap. Nothing is funded yet; call [`Swap::lock`] to
    /// move the funds into escrow. `timeout` is absolute unix seconds.
    /// A zero amount is rejected here, not at lock time.
    pub fn new(
        id: SwapId,
        hashlock: [u8; 32],
        amount: u64,
        locker: PeerId,
        claimer: PeerId,
        timeout: u64,
    ) -> Result<Self, SwapError> {
        if amount == 0 {
            return Err(SwapError::ZeroAmount);
        }
        Ok(Self {
            id,
            hashlock,
            amount,
            locker,
            claimer,
            timeout,
            state: SwapState::Proposed,
        })
    }

    /// The swap's id.
    pub fn id(&self) -> SwapId {
        self.id
    }

    /// The hashlock (SHA-256 of the secret preimage).
    pub fn hashlock(&self) -> &[u8; 32] {
        &self.hashlock
    }

    /// Amount locked (or to be locked) in the ledger's smallest unit.
    pub fn amount(&self) -> u64 {
        self.amount
    }

    /// Who funded the lock and who gets the refund.
    pub fn locker(&self) -> &PeerId {
        &self.locker
    }

    /// Who may claim with the preimage.
    pub fn claimer(&self) -> &PeerId {
        &self.claimer
    }

    /// Absolute unix-seconds deadline. Claims need `now < timeout`;
    /// refunds and expiry need `now >= timeout`.
    pub fn timeout(&self) -> u64 {
        self.timeout
    }

    /// The stored state.
    pub fn state(&self) -> SwapState {
        self.state
    }

    /// The state as observed at `now`: a `Locked` swap at or past its
    /// timeout reads as `Expired`; every other state reads as stored.
    pub fn observed_state(&self, now: u64) -> SwapState {
        if self.state == SwapState::Locked && now >= self.timeout {
            SwapState::Expired
        } else {
            self.state
        }
    }

    /// Move the funds from the locker into escrow: `Proposed -> Locked`.
    /// Fails if the swap is not `Proposed`, if `now` is already at or
    /// past the timeout, or if the ledger refuses the lock
    /// (insufficient funds, duplicate id). On failure the state does
    /// not change and no funds move.
    pub fn lock<L: Ledger>(&mut self, ledger: &mut L, now: u64) -> Result<(), SwapError> {
        if self.state.is_terminal() {
            return Err(SwapError::AlreadySettled);
        }
        if self.state != SwapState::Proposed {
            return Err(SwapError::NotProposed);
        }
        if now >= self.timeout {
            return Err(SwapError::TooLate);
        }
        ledger.lock(&self.id, &self.locker, self.amount)?;
        self.state = SwapState::Locked;
        Ok(())
    }

    /// Claim the escrowed funds with the preimage: `Locked -> Claimed`.
    /// Only the designated claimer, only with the preimage whose
    /// SHA-256 matches the hashlock, only before the timeout, only
    /// once. Returns the amount claimed. On failure the state does not
    /// change and no funds move.
    pub fn claim<L: Ledger>(
        &mut self,
        ledger: &mut L,
        preimage: &[u8],
        caller: &PeerId,
        now: u64,
    ) -> Result<u64, SwapError> {
        if self.state.is_terminal() {
            return Err(SwapError::AlreadySettled);
        }
        if self.state == SwapState::Expired {
            return Err(SwapError::TooLate);
        }
        if self.state != SwapState::Locked {
            return Err(SwapError::NotLocked);
        }
        if caller != &self.claimer {
            return Err(SwapError::WrongClaimer);
        }
        if now >= self.timeout {
            return Err(SwapError::TooLate);
        }
        if sha256(preimage) != self.hashlock {
            return Err(SwapError::WrongPreimage);
        }
        let amount = ledger.release(&self.id, &self.claimer)?;
        self.state = SwapState::Claimed;
        Ok(amount)
    }

    /// Refund the escrowed funds to the locker: `Locked | Expired ->
    /// Refunded`. Only the locker, only at or after the timeout, only
    /// once. Returns the amount refunded. On failure the state does
    /// not change and no funds move.
    pub fn refund<L: Ledger>(
        &mut self,
        ledger: &mut L,
        caller: &PeerId,
        now: u64,
    ) -> Result<u64, SwapError> {
        if self.state.is_terminal() {
            return Err(SwapError::AlreadySettled);
        }
        if self.state != SwapState::Locked && self.state != SwapState::Expired {
            return Err(SwapError::NotLocked);
        }
        if caller != &self.locker {
            return Err(SwapError::WrongLocker);
        }
        if now < self.timeout {
            return Err(SwapError::TooEarly);
        }
        let amount = ledger.release(&self.id, &self.locker)?;
        self.state = SwapState::Refunded;
        Ok(amount)
    }

    /// Mark a timed-out lock as expired: `Locked -> Expired`. The funds
    /// stay in escrow — only a refund settles an expired swap. Fails if
    /// the swap is not `Locked` or the timeout has not passed.
    pub fn expire(&mut self, now: u64) -> Result<(), SwapError> {
        if self.state.is_terminal() {
            return Err(SwapError::AlreadySettled);
        }
        if self.state != SwapState::Locked {
            return Err(SwapError::NotLocked);
        }
        if now < self.timeout {
            return Err(SwapError::NotExpired);
        }
        self.state = SwapState::Expired;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::MemLedger;

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

    const SECRET: &[u8] = b"kernel-atomic-swap-secret";
    const TIMEOUT: u64 = 1_000_000;

    fn swap_funded(
        ledger: &mut MemLedger,
        id_byte: u8,
        amount: u64,
        locker: &PeerId,
        claimer: &PeerId,
        timeout: u64,
    ) -> Swap {
        let mut swap = Swap::new(
            SwapId::from_bytes([id_byte; 32]),
            Swap::hashlock_for(SECRET),
            amount,
            locker.clone(),
            claimer.clone(),
            timeout,
        )
        .unwrap();
        swap.lock(ledger, 0).unwrap();
        swap
    }

    #[test]
    fn happy_path_claim_moves_funds_and_conserves_them() {
        let alice = alice();
        let bob = bob();
        let mut ledger = MemLedger::new();
        ledger.mint(&alice, 100);
        let before = ledger.total();

        let mut swap = swap_funded(&mut ledger, 1, 60, &alice, &bob, TIMEOUT);
        assert_eq!(swap.state(), SwapState::Locked);
        assert_eq!(ledger.balance(&alice), 40);
        assert_eq!(ledger.escrowed(&swap.id()), 60);
        assert_eq!(ledger.total(), before);

        let claimed = swap.claim(&mut ledger, SECRET, &bob, 500).unwrap();
        assert_eq!(claimed, 60);
        assert_eq!(swap.state(), SwapState::Claimed);
        assert_eq!(ledger.balance(&bob), 60);
        assert_eq!(ledger.balance(&alice), 40);
        assert_eq!(ledger.escrowed(&swap.id()), 0);
        assert_eq!(ledger.total(), before);
    }

    #[test]
    fn wrong_preimage_is_refused_and_nothing_moves() {
        let alice = alice();
        let bob = bob();
        let mut ledger = MemLedger::new();
        ledger.mint(&alice, 100);
        let before = ledger.total();

        let mut swap = swap_funded(&mut ledger, 1, 60, &alice, &bob, TIMEOUT);
        assert_eq!(
            swap.claim(&mut ledger, b"not the secret", &bob, 500),
            Err(SwapError::WrongPreimage)
        );
        assert_eq!(swap.state(), SwapState::Locked);
        assert_eq!(ledger.balance(&alice), 40);
        assert_eq!(ledger.balance(&bob), 0);
        assert_eq!(ledger.escrowed(&swap.id()), 60);
        assert_eq!(ledger.total(), before);
    }

    #[test]
    fn double_claim_is_refused() {
        let alice = alice();
        let bob = bob();
        let mut ledger = MemLedger::new();
        ledger.mint(&alice, 100);
        let before = ledger.total();

        let mut swap = swap_funded(&mut ledger, 1, 60, &alice, &bob, TIMEOUT);
        swap.claim(&mut ledger, SECRET, &bob, 500).unwrap();
        assert_eq!(
            swap.claim(&mut ledger, SECRET, &bob, 600),
            Err(SwapError::AlreadySettled)
        );
        assert_eq!(ledger.balance(&bob), 60);
        assert_eq!(ledger.total(), before);
    }

    #[test]
    fn claim_by_the_wrong_party_is_refused() {
        let alice = alice();
        let bob = bob();
        let mut ledger = MemLedger::new();
        ledger.mint(&alice, 100);
        let before = ledger.total();

        let mut swap = swap_funded(&mut ledger, 1, 60, &alice, &bob, TIMEOUT);
        assert_eq!(
            swap.claim(&mut ledger, SECRET, &alice, 500),
            Err(SwapError::WrongClaimer)
        );
        assert_eq!(swap.state(), SwapState::Locked);
        assert_eq!(ledger.total(), before);
    }

    #[test]
    fn refund_before_timeout_is_refused() {
        let alice = alice();
        let bob = bob();
        let mut ledger = MemLedger::new();
        ledger.mint(&alice, 100);
        let before = ledger.total();

        let mut swap = swap_funded(&mut ledger, 1, 60, &alice, &bob, TIMEOUT);
        assert_eq!(
            swap.refund(&mut ledger, &alice, TIMEOUT - 1),
            Err(SwapError::TooEarly)
        );
        assert_eq!(swap.state(), SwapState::Locked);
        assert_eq!(ledger.balance(&alice), 40);
        assert_eq!(ledger.escrowed(&swap.id()), 60);
        assert_eq!(ledger.total(), before);
    }

    #[test]
    fn refund_after_timeout_returns_funds_exactly() {
        let alice = alice();
        let bob = bob();
        let mut ledger = MemLedger::new();
        ledger.mint(&alice, 100);
        let before = ledger.total();

        let mut swap = swap_funded(&mut ledger, 1, 60, &alice, &bob, TIMEOUT);
        let refunded = swap.refund(&mut ledger, &alice, TIMEOUT).unwrap();
        assert_eq!(refunded, 60);
        assert_eq!(swap.state(), SwapState::Refunded);
        assert_eq!(ledger.balance(&alice), 100);
        assert_eq!(ledger.balance(&bob), 0);
        assert_eq!(ledger.escrowed(&swap.id()), 0);
        assert_eq!(ledger.total(), before);
        // A settled swap cannot refund again.
        assert_eq!(
            swap.refund(&mut ledger, &alice, TIMEOUT + 1),
            Err(SwapError::AlreadySettled)
        );
        assert_eq!(ledger.total(), before);
    }

    #[test]
    fn refund_by_the_wrong_party_is_refused() {
        let alice = alice();
        let bob = bob();
        let mut ledger = MemLedger::new();
        ledger.mint(&alice, 100);
        let before = ledger.total();

        let mut swap = swap_funded(&mut ledger, 1, 60, &alice, &bob, TIMEOUT);
        assert_eq!(
            swap.refund(&mut ledger, &bob, TIMEOUT),
            Err(SwapError::WrongLocker)
        );
        assert_eq!(swap.state(), SwapState::Locked);
        assert_eq!(ledger.total(), before);
    }

    #[test]
    fn claim_after_timeout_is_refused() {
        let alice = alice();
        let bob = bob();
        let mut ledger = MemLedger::new();
        ledger.mint(&alice, 100);
        let before = ledger.total();

        let mut swap = swap_funded(&mut ledger, 1, 60, &alice, &bob, TIMEOUT);
        // Exactly at the deadline the claim is already too late.
        assert_eq!(
            swap.claim(&mut ledger, SECRET, &bob, TIMEOUT),
            Err(SwapError::TooLate)
        );
        assert_eq!(swap.state(), SwapState::Locked);
        assert_eq!(ledger.total(), before);
        // The locker can still take the funds back.
        assert_eq!(swap.refund(&mut ledger, &alice, TIMEOUT).unwrap(), 60);
        assert_eq!(ledger.total(), before);
    }

    #[test]
    fn expired_lock_cannot_be_claimed_even_with_the_right_preimage() {
        let alice = alice();
        let bob = bob();
        let mut ledger = MemLedger::new();
        ledger.mint(&alice, 100);
        let before = ledger.total();

        let mut swap = swap_funded(&mut ledger, 1, 60, &alice, &bob, TIMEOUT);
        // Past the timeout the lock reads as expired...
        assert_eq!(swap.observed_state(TIMEOUT), SwapState::Expired);
        // ...and the stored state can be marked expired explicitly.
        swap.expire(TIMEOUT).unwrap();
        assert_eq!(swap.state(), SwapState::Expired);
        assert_eq!(ledger.escrowed(&swap.id()), 60);
        // Even the right preimage cannot claim an expired lock.
        assert_eq!(
            swap.claim(&mut ledger, SECRET, &bob, TIMEOUT + 1),
            Err(SwapError::TooLate)
        );
        assert_eq!(ledger.total(), before);
        // The funds are not lost: the locker refunds them in full.
        assert_eq!(swap.refund(&mut ledger, &alice, TIMEOUT + 1).unwrap(), 60);
        assert_eq!(swap.state(), SwapState::Refunded);
        assert_eq!(ledger.balance(&alice), 100);
        assert_eq!(ledger.total(), before);
    }

    #[test]
    fn expire_before_timeout_is_refused() {
        let alice = alice();
        let bob = bob();
        let mut ledger = MemLedger::new();
        ledger.mint(&alice, 100);
        let before = ledger.total();

        let mut swap = swap_funded(&mut ledger, 1, 60, &alice, &bob, TIMEOUT);
        assert_eq!(swap.expire(TIMEOUT - 1), Err(SwapError::NotExpired));
        assert_eq!(swap.state(), SwapState::Locked);
        assert_eq!(ledger.total(), before);
    }

    #[test]
    fn lock_twice_is_refused_and_locking_an_expired_swap_is_refused() {
        let alice = alice();
        let bob = bob();
        let mut ledger = MemLedger::new();
        ledger.mint(&alice, 100);
        let before = ledger.total();

        let mut swap = swap_funded(&mut ledger, 1, 60, &alice, &bob, TIMEOUT);
        assert_eq!(swap.lock(&mut ledger, 10), Err(SwapError::NotProposed));
        assert_eq!(ledger.total(), before);

        // A swap proposed with a timeout already passed cannot lock.
        let mut late = Swap::new(
            SwapId::from_bytes([2u8; 32]),
            Swap::hashlock_for(SECRET),
            10,
            alice.clone(),
            bob.clone(),
            100,
        )
        .unwrap();
        assert_eq!(late.lock(&mut ledger, 100), Err(SwapError::TooLate));
        assert_eq!(late.state(), SwapState::Proposed);
        assert_eq!(ledger.total(), before);
    }

    #[test]
    fn zero_amount_swap_is_rejected_at_construction() {
        let alice = alice();
        let bob = bob();
        assert_eq!(
            Swap::new(
                SwapId::from_bytes([1u8; 32]),
                Swap::hashlock_for(SECRET),
                0,
                alice,
                bob,
                TIMEOUT,
            ),
            Err(SwapError::ZeroAmount)
        );
    }

    #[test]
    fn lock_with_insufficient_funds_is_refused() {
        let alice = alice();
        let bob = bob();
        let mut ledger = MemLedger::new();
        ledger.mint(&alice, 10);
        let before = ledger.total();

        let mut swap = Swap::new(
            SwapId::from_bytes([1u8; 32]),
            Swap::hashlock_for(SECRET),
            60,
            alice.clone(),
            bob.clone(),
            TIMEOUT,
        )
        .unwrap();
        assert_eq!(swap.lock(&mut ledger, 0), Err(SwapError::InsufficientFunds));
        assert_eq!(swap.state(), SwapState::Proposed);
        assert_eq!(ledger.balance(&alice), 10);
        assert_eq!(ledger.total(), before);
    }

    /// The full two-party atomic swap. Alice holds 100 on ledger A and
    /// wants Bob's 50 on ledger B; Bob holds 50 on B and wants Alice's
    /// 100 on A. Alice locks first (longer timeout), Bob locks second
    /// (shorter timeout), Alice claims on B — revealing the secret —
    /// and Bob claims on A with the revealed secret.
    #[test]
    fn two_party_cross_ledger_swap() {
        let alice = alice();
        let bob = bob();
        let mut ledger_a = MemLedger::new();
        let mut ledger_b = MemLedger::new();
        ledger_a.mint(&alice, 100);
        ledger_b.mint(&bob, 50);
        let total_a_before = ledger_a.total();
        let total_b_before = ledger_b.total();
        let timeout_a = 2_000_000u64;
        let timeout_b = 1_500_000u64;

        // Alice locks 100 on A for Bob; Bob locks 50 on B for Alice.
        // Same hashlock, different swap ids, different ledgers.
        let mut swap_a = Swap::new(
            SwapId::from_bytes([0xAAu8; 32]),
            Swap::hashlock_for(SECRET),
            100,
            alice.clone(),
            bob.clone(),
            timeout_a,
        )
        .unwrap();
        let mut swap_b = Swap::new(
            SwapId::from_bytes([0xBBu8; 32]),
            Swap::hashlock_for(SECRET),
            50,
            bob.clone(),
            alice.clone(),
            timeout_b,
        )
        .unwrap();
        swap_a.lock(&mut ledger_a, 0).unwrap();
        swap_b.lock(&mut ledger_b, 0).unwrap();
        assert_eq!(ledger_a.balance(&alice), 0);
        assert_eq!(ledger_b.balance(&bob), 0);
        assert_eq!(ledger_a.total(), total_a_before);
        assert_eq!(ledger_b.total(), total_b_before);

        // Alice claims Bob's 50 on B, revealing the secret. She must do
        // this before Bob's shorter timeout.
        let claimed_b = swap_b
            .claim(&mut ledger_b, SECRET, &alice, 1_000_000)
            .unwrap();
        assert_eq!(claimed_b, 50);
        assert_eq!(swap_b.state(), SwapState::Claimed);
        assert_eq!(ledger_b.balance(&alice), 50);
        assert_eq!(ledger_a.total(), total_a_before);
        assert_eq!(ledger_b.total(), total_b_before);

        // Bob uses the now-revealed secret to claim Alice's 100 on A,
        // before Alice's longer timeout.
        let claimed_a = swap_a
            .claim(&mut ledger_a, SECRET, &bob, 1_000_001)
            .unwrap();
        assert_eq!(claimed_a, 100);
        assert_eq!(swap_a.state(), SwapState::Claimed);
        assert_eq!(ledger_a.balance(&bob), 100);
        assert_eq!(ledger_a.total(), total_a_before);
        assert_eq!(ledger_b.total(), total_b_before);

        // End state: each holds the other's funds, nothing created or lost.
        assert_eq!(ledger_a.balance(&alice), 0);
        assert_eq!(ledger_a.balance(&bob), 100);
        assert_eq!(ledger_b.balance(&alice), 50);
        assert_eq!(ledger_b.balance(&bob), 0);
    }

    /// Abort path: Alice locks, Bob never locks, the lock times out and
    /// Alice refunds in full.
    #[test]
    fn abort_path_refunds_the_locker_in_full() {
        let alice = alice();
        let bob = bob();
        let mut ledger_a = MemLedger::new();
        let mut ledger_b = MemLedger::new();
        ledger_a.mint(&alice, 100);
        ledger_b.mint(&bob, 50);
        let total_a_before = ledger_a.total();
        let total_b_before = ledger_b.total();

        let mut swap_a = Swap::new(
            SwapId::from_bytes([0xAAu8; 32]),
            Swap::hashlock_for(SECRET),
            100,
            alice.clone(),
            bob.clone(),
            TIMEOUT,
        )
        .unwrap();
        swap_a.lock(&mut ledger_a, 0).unwrap();
        assert_eq!(ledger_a.balance(&alice), 0);
        assert_eq!(ledger_a.total(), total_a_before);

        // Bob never locks on B. Nothing happens on B at all.
        assert_eq!(ledger_b.balance(&bob), 50);
        assert_eq!(ledger_b.total(), total_b_before);

        // Past the timeout Alice takes her funds back, in full.
        let refunded = swap_a.refund(&mut ledger_a, &alice, TIMEOUT + 1).unwrap();
        assert_eq!(refunded, 100);
        assert_eq!(swap_a.state(), SwapState::Refunded);
        assert_eq!(ledger_a.balance(&alice), 100);
        assert_eq!(ledger_a.total(), total_a_before);
        assert_eq!(ledger_b.total(), total_b_before);
    }
}
