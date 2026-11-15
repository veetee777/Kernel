//! Hello-flood protection (Phase 2).
//!
//! Each layer runs before the responder's next, more expensive step:
//!
//! 1. FloodGuard limits incoming connections per address. A refused
//!    connection is closed before any byte is written.
//! 2. The cookie challenge. The responder sends HMAC-SHA256 over the source
//!    address and a 30-second bucket, keyed by a random secret it holds. It
//!    stores nothing per attempt. The initiator must echo the cookie before
//!    the responder reads a Hello.
//! 3. Proof of work. The challenge carries a difficulty. The initiator must
//!    find a nonce such that SHA-256 over the cookie and nonce has that many
//!    leading zero bits: about 2^difficulty hashes for the initiator, one hash
//!    for the responder.
//!
//! No layer runs Diffie-Hellman, and no layer keeps state per attempt.
//!
//! Limits, stated plainly: over TCP a handshake cannot be spoofed from another
//! address, so the cookie is not anti-spoofing. It forces a round trip and binds
//! the challenge to an address. The guard and the puzzle bound the work that an
//! attacker with many real addresses can cause. A full table of tracked
//! addresses refuses new addresses until old entries expire.

use std::collections::HashMap;
use std::net::IpAddr;
use std::time::{Duration, Instant};

use getrandom::getrandom;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

use crate::error::CarrierError;

/// Cookie length: one HMAC-SHA256 output.
pub const COOKIE_LEN: usize = 32;
/// Puzzle nonce length.
pub const NONCE_LEN: usize = 8;
/// A cookie is valid for its own bucket and the one before it.
pub const COOKIE_BUCKET: Duration = Duration::from_secs(30);
/// Default puzzle difficulty, in leading zero bits.
pub const DEFAULT_DIFFICULTY: u8 = 12;
/// Highest difficulty the responder issues or the initiator will solve.
pub const MAX_DIFFICULTY: u8 = 20;
/// Addresses tracked by the guard at once.
pub const DEFAULT_MAX_TRACKED: usize = 4096;

type HmacSha256 = Hmac<Sha256>;

fn ip_bytes(ip: IpAddr) -> Vec<u8> {
    match ip {
        IpAddr::V4(a) => {
            let mut v = vec![4u8];
            v.extend_from_slice(&a.octets());
            v
        }
        IpAddr::V6(a) => {
            let mut v = vec![6u8];
            v.extend_from_slice(&a.octets());
            v
        }
    }
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut acc = 0u8;
    for (x, y) in a.iter().zip(b) {
        acc |= x ^ y;
    }
    acc == 0
}

/// Issues and verifies cookies. Holds one random secret for the life of the listener.
pub struct CookieState {
    secret: [u8; 32],
    difficulty: u8,
    started: Instant,
}

impl CookieState {
    pub fn new(difficulty: u8) -> Result<Self, CarrierError> {
        if difficulty > MAX_DIFFICULTY {
            return Err(CarrierError::BadCookie);
        }
        let mut secret = [0u8; 32];
        getrandom(&mut secret).map_err(|_| CarrierError::Random)?;
        Ok(Self {
            secret,
            difficulty,
            started: Instant::now(),
        })
    }

    pub fn difficulty(&self) -> u8 {
        self.difficulty
    }

    fn bucket(&self, now: Instant) -> u64 {
        now.saturating_duration_since(self.started).as_secs() / COOKIE_BUCKET.as_secs()
    }

    fn cookie_for(&self, ip: IpAddr, bucket: u64) -> Result<[u8; COOKIE_LEN], CarrierError> {
        let mut mac =
            HmacSha256::new_from_slice(&self.secret).map_err(|_| CarrierError::Internal)?;
        mac.update(b"KCAR-COOKIE-v1\0");
        mac.update(&ip_bytes(ip));
        mac.update(&bucket.to_be_bytes());
        Ok(mac.finalize().into_bytes().into())
    }

    /// The cookie to issue to `ip` at `now`.
    pub fn issue(&self, ip: IpAddr, now: Instant) -> Result<[u8; COOKIE_LEN], CarrierError> {
        self.cookie_for(ip, self.bucket(now))
    }

    /// True when `cookie` was issued to `ip` in the current or previous bucket.
    pub fn verify(&self, ip: IpAddr, now: Instant, cookie: &[u8; COOKIE_LEN]) -> bool {
        let b = self.bucket(now);
        let mut buckets = vec![b];
        if b > 0 {
            buckets.push(b - 1);
        }
        buckets
            .into_iter()
            .any(|bk| matches!(self.cookie_for(ip, bk), Ok(c) if ct_eq(&c, cookie)))
    }
}

/// SHA-256 over the puzzle domain, the cookie, and the nonce.
pub fn pow_hash(cookie: &[u8; COOKIE_LEN], nonce: u64) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"KCAR-POW-v1\0");
    h.update(cookie);
    h.update(nonce.to_be_bytes());
    h.finalize().into()
}

fn leading_zero_bits(bytes: &[u8]) -> u32 {
    let mut n = 0u32;
    for b in bytes {
        if *b == 0 {
            n += 8;
        } else {
            n += b.leading_zeros();
            break;
        }
    }
    n
}

/// True when `nonce` solves the puzzle for `cookie` at `difficulty`.
pub fn pow_ok(cookie: &[u8; COOKIE_LEN], nonce: u64, difficulty: u8) -> bool {
    leading_zero_bits(&pow_hash(cookie, nonce)) >= u32::from(difficulty)
}

/// Find a nonce for `cookie` at `difficulty`. Expected work: 2^difficulty hashes.
pub fn solve_pow(cookie: &[u8; COOKIE_LEN], difficulty: u8) -> Result<u64, CarrierError> {
    if difficulty > MAX_DIFFICULTY {
        return Err(CarrierError::BadCookie);
    }
    let mut nonce = 0u64;
    while !pow_ok(cookie, nonce, difficulty) {
        nonce = nonce.wrapping_add(1);
    }
    Ok(nonce)
}

#[derive(Clone, Copy)]
struct Window {
    started: Instant,
    count: u32,
}

/// Per-address connection limit over a fixed window, with a bounded table.
pub struct FloodGuard {
    max_per_window: u32,
    window: Duration,
    max_tracked: usize,
    table: HashMap<IpAddr, Window>,
}

impl FloodGuard {
    pub fn new(max_per_window: u32, window: Duration, max_tracked: usize) -> Self {
        Self {
            max_per_window,
            window,
            max_tracked,
            table: HashMap::new(),
        }
    }

    /// Number of addresses currently tracked.
    pub fn tracked(&self) -> usize {
        self.table.len()
    }

    /// Count one connection from `ip` at `now`. False means refuse it.
    pub fn admit(&mut self, ip: IpAddr, now: Instant) -> bool {
        let window = self.window;
        if !self.table.contains_key(&ip) && self.table.len() >= self.max_tracked {
            self.table
                .retain(|_, w| now.saturating_duration_since(w.started) < window);
            if self.table.len() >= self.max_tracked {
                return false;
            }
        }
        let max = self.max_per_window;
        let entry = self.table.entry(ip).or_insert(Window {
            started: now,
            count: 0,
        });
        if now.saturating_duration_since(entry.started) >= window {
            entry.started = now;
            entry.count = 0;
        }
        if entry.count >= max {
            return false;
        }
        entry.count += 1;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip4(a: u8, b: u8, c: u8, d: u8) -> IpAddr {
        IpAddr::from([a, b, c, d])
    }

    #[test]
    fn cookie_verifies_for_its_address_and_refuses_others() {
        let c = CookieState::new(0).expect("state");
        let now = Instant::now();
        let cookie = c.issue(ip4(10, 0, 0, 1), now).expect("issue");
        assert!(c.verify(ip4(10, 0, 0, 1), now, &cookie));
        assert!(!c.verify(ip4(10, 0, 0, 2), now, &cookie));
    }

    #[test]
    fn cookie_is_valid_for_two_buckets_then_expires() {
        let c = CookieState::new(0).expect("state");
        let now = Instant::now();
        let ip = ip4(10, 0, 0, 1);
        let cookie = c.issue(ip, now).expect("issue");
        // One bucket on: the previous bucket still counts.
        assert!(c.verify(ip, now + Duration::from_secs(31), &cookie));
        // Two buckets on: expired.
        assert!(!c.verify(ip, now + Duration::from_secs(61), &cookie));
    }

    #[test]
    fn a_solved_puzzle_verifies_and_zero_difficulty_is_free() {
        let cookie = [7u8; COOKIE_LEN];
        let n = solve_pow(&cookie, 8).expect("solve");
        assert!(pow_ok(&cookie, n, 8));
        assert!(pow_ok(&cookie, 0, 0));
        assert!(solve_pow(&cookie, MAX_DIFFICULTY + 1).is_err());
    }

    #[test]
    fn a_wrong_nonce_is_refused() {
        let cookie = [7u8; COOKIE_LEN];
        let n = solve_pow(&cookie, 8).expect("solve");
        let wrong = (n + 1..)
            .find(|m| !pow_ok(&cookie, *m, 8))
            .expect("a wrong nonce");
        assert!(!pow_ok(&cookie, wrong, 8));
    }

    #[test]
    fn guard_admits_up_to_the_limit_then_refuses() {
        let mut g = FloodGuard::new(3, Duration::from_secs(10), 16);
        let now = Instant::now();
        let ip = ip4(10, 0, 0, 1);
        assert!(g.admit(ip, now));
        assert!(g.admit(ip, now));
        assert!(g.admit(ip, now));
        assert!(!g.admit(ip, now));
    }

    #[test]
    fn guard_counts_each_address_separately() {
        let mut g = FloodGuard::new(1, Duration::from_secs(10), 16);
        let now = Instant::now();
        assert!(g.admit(ip4(10, 0, 0, 1), now));
        assert!(!g.admit(ip4(10, 0, 0, 1), now));
        assert!(g.admit(ip4(10, 0, 0, 2), now));
    }

    #[test]
    fn guard_window_resets() {
        let mut g = FloodGuard::new(1, Duration::from_secs(10), 16);
        let now = Instant::now();
        let ip = ip4(10, 0, 0, 1);
        assert!(g.admit(ip, now));
        assert!(!g.admit(ip, now));
        assert!(g.admit(ip, now + Duration::from_secs(11)));
    }

    #[test]
    fn guard_table_is_bounded() {
        let mut g = FloodGuard::new(1, Duration::from_secs(10), 2);
        let now = Instant::now();
        assert!(g.admit(ip4(10, 0, 0, 1), now));
        assert!(g.admit(ip4(10, 0, 0, 2), now));
        assert!(!g.admit(ip4(10, 0, 0, 3), now));
        assert_eq!(g.tracked(), 2);
        // After the window, the old entries are pruned and room opens.
        assert!(g.admit(ip4(10, 0, 0, 3), now + Duration::from_secs(11)));
    }
}
