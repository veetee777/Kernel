//! The play queue: next / previous over a playlist, seeded shuffle,
//! and repeat modes.
//!
//! A queue is built from an ordered list of track manifest CIDs (for
//! example from a [`crate::playlist::PlaylistDoc`]). It never touches
//! the network or the clock: shuffle order comes from a seeded
//! xorshift PRNG, so the same seed always gives the same order.
//!
//! Repeat modes:
//! - [`RepeatMode::Off`]: playback stops at the end of the list.
//! - [`RepeatMode::All`]: next / previous wrap around the list.
//! - [`RepeatMode::One`]: next / previous stay on the current track.

use kernel_storage::Cid;

/// Repeat behaviour at the ends of the queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RepeatMode {
    /// Stop at the end of the list.
    #[default]
    Off,
    /// Wrap around the list.
    All,
    /// Stay on the current track.
    One,
}

/// A small deterministic PRNG (xorshift64*). Same seed, same stream;
/// no system RNG, no wall clock. A seed of zero is replaced with a
/// fixed nonzero constant, since a zero xorshift state never moves.
struct XorShift64Star {
    state: u64,
}

impl XorShift64Star {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
    }

    /// Next `u64` in the stream.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A value in `[0, bound)`, unbiased for the small bounds a queue
    /// uses. `bound` must be nonzero.
    fn below(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        (self.next_u64() % bound as u64) as usize
    }
}

/// Fisher-Yates shuffle of `0..n`, driven by the seeded PRNG.
fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = XorShift64Star::new(seed);
    for i in (1..n).rev() {
        let j = rng.below(i + 1);
        order.swap(i, j);
    }
    order
}

/// An ordered play queue over track manifest CIDs.
#[derive(Debug, Clone)]
pub struct PlayQueue {
    /// Tracks in playlist order.
    tracks: Vec<Cid>,
    /// Play order: indices into `tracks`. Identity when shuffle is
    /// off, a seeded permutation when it is on.
    order: Vec<usize>,
    /// Position within `order`. `order.len()` means "past the end"
    /// (only reachable with [`RepeatMode::Off`]).
    pos: usize,
    repeat: RepeatMode,
    /// Seed of the active shuffle, if any.
    shuffle_seed: Option<u64>,
}

impl PlayQueue {
    /// A queue over `tracks`, in the given order, at the first track.
    pub fn new(tracks: Vec<Cid>) -> Self {
        let order: Vec<usize> = (0..tracks.len()).collect();
        Self {
            tracks,
            order,
            pos: 0,
            repeat: RepeatMode::Off,
            shuffle_seed: None,
        }
    }

    /// A queue from a playlist document, in the document's order.
    pub fn from_playlist(doc: &crate::playlist::PlaylistDoc) -> Self {
        Self::new(doc.tracks().to_vec())
    }

    /// Number of tracks in the queue.
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// True if the queue holds no tracks.
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// The track at the current position, if any.
    pub fn current(&self) -> Option<Cid> {
        self.order.get(self.pos).map(|&i| self.tracks[i])
    }

    /// The effective play order (the track CIDs in the order they
    /// will play), for inspection and testing.
    pub fn play_order(&self) -> Vec<Cid> {
        self.order.iter().map(|&i| self.tracks[i]).collect()
    }

    /// Current repeat mode.
    pub fn repeat(&self) -> RepeatMode {
        self.repeat
    }

    /// Set the repeat mode.
    pub fn set_repeat(&mut self, repeat: RepeatMode) {
        self.repeat = repeat;
    }

    /// Seed of the active shuffle, if shuffle is on.
    pub fn shuffle_seed(&self) -> Option<u64> {
        self.shuffle_seed
    }

    /// True if shuffle is on.
    pub fn is_shuffled(&self) -> bool {
        self.shuffle_seed.is_some()
    }

    /// Go back to the previous track. Repeat `One` stays on the
    /// current track; repeat `All` wraps to the last track; repeat
    /// `Off` returns `None` before the first track (and from the
    /// past-the-end position it returns the last track).
    pub fn prev(&mut self) -> Option<Cid> {
        if self.tracks.is_empty() {
            return None;
        }
        if self.repeat == RepeatMode::One {
            return self.current();
        }
        if self.pos == 0 {
            if self.repeat == RepeatMode::All {
                self.pos = self.order.len() - 1;
                self.current()
            } else {
                None
            }
        } else {
            self.pos -= 1;
            self.current()
        }
    }

    /// Turn shuffle on with `seed`: the play order becomes the seeded
    /// permutation, and the current track keeps playing (its position
    /// is preserved). The same seed always gives the same order.
    pub fn set_shuffle(&mut self, seed: u64) {
        let current = self.current();
        self.order = shuffled_indices(self.tracks.len(), seed);
        self.shuffle_seed = Some(seed);
        self.pos = match current {
            Some(cid) => self
                .order
                .iter()
                .position(|&i| self.tracks[i] == cid)
                .unwrap_or(0),
            None => 0,
        };
    }

    /// Turn shuffle off: the play order returns to playlist order,
    /// and the current track keeps playing.
    pub fn clear_shuffle(&mut self) {
        let current = self.current();
        self.order = (0..self.tracks.len()).collect();
        self.shuffle_seed = None;
        self.pos = match current {
            Some(cid) => self.tracks.iter().position(|&c| c == cid).unwrap_or(0),
            None => 0,
        };
    }
}

impl Iterator for PlayQueue {
    type Item = Cid;

    /// Advance to the next track. Repeat `One` stays on the current
    /// track; repeat `All` wraps to the first track; repeat `Off`
    /// returns `None` past the last track. Returns the new current
    /// track, or `None` when the queue is empty or finished.
    fn next(&mut self) -> Option<Self::Item> {
        if self.tracks.is_empty() {
            return None;
        }
        if self.repeat == RepeatMode::One {
            return self.current();
        }
        if self.pos + 1 < self.order.len() {
            self.pos += 1;
            self.current()
        } else if self.repeat == RepeatMode::All {
            self.pos = 0;
            self.current()
        } else {
            self.pos = self.order.len();
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cid(n: u8) -> Cid {
        let mut bytes = [0u8; 32];
        bytes[0] = n;
        Cid::from_bytes(bytes)
    }

    #[test]
    fn next_and_prev_with_repeat_off() {
        let mut q = PlayQueue::new(vec![cid(1), cid(2), cid(3)]);
        assert_eq!(q.current(), Some(cid(1)));
        assert_eq!(q.next(), Some(cid(2)));
        assert_eq!(q.next(), Some(cid(3)));
        assert_eq!(q.next(), None);
        assert_eq!(q.current(), None);
        // From past the end, prev returns the last track.
        assert_eq!(q.prev(), Some(cid(3)));
        assert_eq!(q.prev(), Some(cid(2)));
        assert_eq!(q.prev(), Some(cid(1)));
        assert_eq!(q.prev(), None);
        assert_eq!(q.current(), Some(cid(1)));
    }

    #[test]
    fn repeat_all_wraps_both_ways() {
        let mut q = PlayQueue::new(vec![cid(1), cid(2)]);
        q.set_repeat(RepeatMode::All);
        assert_eq!(q.next(), Some(cid(2)));
        assert_eq!(q.next(), Some(cid(1)));
        assert_eq!(q.next(), Some(cid(2)));
        assert_eq!(q.prev(), Some(cid(1)));
        assert_eq!(q.prev(), Some(cid(2)));
    }

    #[test]
    fn repeat_one_stays_put() {
        let mut q = PlayQueue::new(vec![cid(1), cid(2), cid(3)]);
        q.set_repeat(RepeatMode::One);
        assert_eq!(q.next(), Some(cid(1)));
        assert_eq!(q.next(), Some(cid(1)));
        assert_eq!(q.prev(), Some(cid(1)));
        q.set_repeat(RepeatMode::Off);
        assert_eq!(q.next(), Some(cid(2)));
    }

    #[test]
    fn shuffle_is_deterministic_and_seed_sensitive() {
        let tracks: Vec<Cid> = (1..=10).map(cid).collect();
        let mut a = PlayQueue::new(tracks.clone());
        a.set_shuffle(42);
        let mut b = PlayQueue::new(tracks.clone());
        b.set_shuffle(42);
        assert_eq!(a.play_order(), b.play_order());
        // The shuffled order is (overwhelmingly likely) not the
        // natural order, and a different seed gives a different order.
        assert_ne!(a.play_order(), tracks);
        let mut c = PlayQueue::new(tracks.clone());
        c.set_shuffle(7);
        assert_ne!(a.play_order(), c.play_order());
        // Same tracks, every one exactly once.
        let mut sorted = a.play_order();
        sorted.sort_by(|x, y| x.as_bytes().cmp(y.as_bytes()));
        let mut expected = tracks.clone();
        expected.sort_by(|x, y| x.as_bytes().cmp(y.as_bytes()));
        assert_eq!(sorted, expected);
        // Seed zero still shuffles deterministically.
        let mut d = PlayQueue::new(tracks.clone());
        d.set_shuffle(0);
        let mut e = PlayQueue::new(tracks.clone());
        e.set_shuffle(0);
        assert_eq!(d.play_order(), e.play_order());
    }

    #[test]
    fn shuffle_preserves_current_track_and_clear_restores_order() {
        let tracks: Vec<Cid> = (1..=5).map(cid).collect();
        let mut q = PlayQueue::new(tracks.clone());
        q.next(); // now on track 2
        assert_eq!(q.current(), Some(cid(2)));
        q.set_shuffle(99);
        assert_eq!(q.current(), Some(cid(2)));
        assert!(q.is_shuffled());
        assert_eq!(q.shuffle_seed(), Some(99));
        q.clear_shuffle();
        assert_eq!(q.current(), Some(cid(2)));
        assert_eq!(q.play_order(), tracks);
        assert!(!q.is_shuffled());
    }

    #[test]
    fn empty_queue_is_quiet() {
        let mut q = PlayQueue::new(vec![]);
        assert!(q.is_empty());
        assert_eq!(q.current(), None);
        assert_eq!(q.next(), None);
        assert_eq!(q.prev(), None);
        q.set_shuffle(1);
        assert_eq!(q.current(), None);
        q.set_repeat(RepeatMode::All);
        assert_eq!(q.next(), None);
    }
}
