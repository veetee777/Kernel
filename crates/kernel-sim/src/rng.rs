//! A tiny deterministic PRNG for the Korn world simulation.
//!
//! xorshift64* (Vigna): one 64-bit state word, no wall clock, no system
//! randomness. The same seed always produces the same stream, which is
//! what makes the whole world a pure function of (seed, tick count).
//! The state is saved with the world so a loaded world continues
//! exactly where it left off.

/// xorshift64* generator: `state ^= state >> 12; state ^= state << 25;
/// state ^= state >> 27; return state * 0x2545F4914F6CDD1D`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XorShift64Star {
    state: u64,
}

impl XorShift64Star {
    /// Build from a seed. A zero seed is replaced with a fixed nonzero
    /// constant — an all-zero state would never move.
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
    }

    /// The raw state word, for saving and restoring.
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Restore a previously saved state word.
    pub fn from_state(state: u64) -> Self {
        Self::new(state)
    }

    /// Next random u64.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Next random u32 (high half of the next u64).
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform value in `[0, bound)`. `bound` must be nonzero.
    /// Uses rejection sampling so the result is unbiased.
    pub fn below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0, "XorShift64Star::below called with 0");
        // Largest multiple of `bound` that fits in u64; values at or
        // above it are rejected to avoid modulo bias.
        let limit = u64::MAX - (u64::MAX % bound);
        loop {
            let v = self.next_u64();
            if v < limit {
                return v % bound;
            }
        }
    }

    /// Uniform value in `[0.0, 1.0)`.
    pub fn next_f64(&mut self) -> f64 {
        // Top 53 bits over 2^53: full double precision, never 1.0.
        let v = self.next_u64() >> 11;
        (v as f64) / ((1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = XorShift64Star::new(42);
        let mut b = XorShift64Star::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = XorShift64Star::new(1);
        let mut b = XorShift64Star::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn zero_seed_still_moves() {
        let mut r = XorShift64Star::new(0);
        assert_ne!(r.next_u64(), 0);
        assert_ne!(r.next_u64(), r.next_u64());
    }

    #[test]
    fn below_stays_in_range() {
        let mut r = XorShift64Star::new(7);
        for _ in 0..1000 {
            assert!(r.below(10) < 10);
            assert!(r.below(1) == 0);
        }
    }

    #[test]
    fn f64_in_unit_range() {
        let mut r = XorShift64Star::new(7);
        for _ in 0..1000 {
            let v = r.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn state_round_trip() {
        let mut r = XorShift64Star::new(99);
        for _ in 0..10 {
            r.next_u64();
        }
        let saved = r.state();
        let mut restored = XorShift64Star::from_state(saved);
        assert_eq!(r.next_u64(), restored.next_u64());
    }
}
