//! Concordance: word -> verses containing it.
//!
//! Built from ingested text. A word maps to every verse whose text
//! contains it, in the order the verses were added (the library
//! re-sorts results into canonical order when it answers a query).
//! Words are normalized by [`crate::tokenize`] — case-insensitive,
//! punctuation stripped, split on whitespace. Exact words only:
//! "love" does not match "loved", and there is no stemming.

use std::collections::{HashMap, HashSet};

use crate::reference::VerseKey;
use crate::tokenize::tokenize;

/// Word -> verses containing it.
#[derive(Debug, Default, Clone)]
pub struct Concordance {
    index: HashMap<String, Vec<VerseKey>>,
}

impl Concordance {
    /// An empty concordance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Index one verse's text under its key. Each distinct word in
    /// the text is recorded once per verse, no matter how many times
    /// the word appears in that verse.
    pub fn add(&mut self, key: &VerseKey, text: &str) {
        let mut seen: HashSet<String> = HashSet::new();
        for word in tokenize(text) {
            if seen.insert(word.clone()) {
                self.index.entry(word).or_default().push(key.clone());
            }
        }
    }

    /// All verses containing `word`, in the order they were added.
    /// The word is normalized the same way as indexed text, so the
    /// lookup is case-insensitive and ignores surrounding punctuation.
    /// An unknown word returns an empty slice.
    pub fn lookup(&self, word: &str) -> &[VerseKey] {
        match crate::tokenize::normalize_word(word) {
            Some(normalized) => self
                .index
                .get(&normalized)
                .map(|v| v.as_slice())
                .unwrap_or(&[]),
            None => &[],
        }
    }

    /// Number of distinct words indexed.
    pub fn word_count(&self) -> usize {
        self.index.len()
    }

    /// True if the word is indexed at all.
    pub fn contains(&self, word: &str) -> bool {
        !self.lookup(word).is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(book: &str, chapter: u32, verse: u32) -> VerseKey {
        VerseKey::new(book, chapter, verse)
    }

    #[test]
    fn concordance_finds_every_occurrence() {
        let mut c = Concordance::new();
        c.add(&key("John", 3, 16), "For God so loved the world");
        c.add(&key("John", 3, 17), "God did not send his Son");
        c.add(&key("Psalm", 23, 1), "The Lord is my shepherd");
        assert_eq!(
            c.lookup("god"),
            vec![key("John", 3, 16), key("John", 3, 17)]
        );
        assert_eq!(
            c.lookup("GOD"),
            vec![key("John", 3, 16), key("John", 3, 17)]
        );
        assert_eq!(
            c.lookup("god,"),
            vec![key("John", 3, 16), key("John", 3, 17)]
        );
        assert_eq!(c.lookup("shepherd"), vec![key("Psalm", 23, 1)]);
        assert!(c.lookup("moses").is_empty());
        assert!(c.lookup("...").is_empty());
        assert!(c.contains("world"));
        assert!(!c.contains("worlds"));
    }

    #[test]
    fn one_verse_per_word_even_when_repeated() {
        let mut c = Concordance::new();
        c.add(&key("John", 1, 1), "love love love");
        assert_eq!(c.lookup("love"), vec![key("John", 1, 1)]);
        assert_eq!(c.word_count(), 1);
    }
}
