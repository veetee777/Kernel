//! Research queries over the library.
//!
//! Two queries, with exact semantics:
//!
//! - [`Library::verses_containing`] — all verses containing word X
//!   in book Y. The word matches by the concordance rules
//!   (case-insensitive, punctuation stripped, exact word — "love"
//!   does not match "loved"). The book is matched case-insensitively;
//!   an unknown book matches nothing. Results are in canonical
//!   order (books in first-ingested order, then chapter, then verse).
//!
//! - [`Library::related_verses`] — verses that share words with
//!   verse Z. Both verses are tokenized by the same rules; the
//!   score of a verse is the number of *distinct* words it shares
//!   with Z. Verses sharing at least one word are returned, sorted
//!   by score descending, then canonical order. Z itself is never
//!   returned. There is **no stopword list**: common words like
//!   "the", "and", and "of" count toward the score, so on real text
//!   the top results will often be verses that share the most common
//!   words, not the most meaningful ones. That is a known limit of
//!   this first version, not a semantic claim.

use std::collections::{HashMap, HashSet};

use crate::error::ScriptureError;
use crate::library::{Library, Scope, Verse};
use crate::reference::VerseKey;
use crate::tokenize::tokenize;

/// A verse related to a query verse, with its score.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelatedVerse {
    /// The related verse.
    pub verse: Verse,
    /// Number of distinct words shared with the query verse.
    pub shared_words: usize,
    /// The shared words, sorted alphabetically.
    pub shared: Vec<String>,
}

impl Library {
    /// All verses containing word X in book Y, in canonical order.
    /// Same word rules as the concordance. An unknown book matches
    /// nothing.
    pub fn verses_containing(&self, word: &str, book: &str) -> Vec<Verse> {
        self.search_word(word, &Scope::Book(book.to_string()))
    }

    /// Verses that share words with verse Z. Score is the count of
    /// distinct shared words; results sort by score descending,
    /// then canonical order. Z itself is excluded. The verse must
    /// exist in the library.
    pub fn related_verses(
        &self,
        book: &str,
        chapter: u32,
        verse: u32,
    ) -> Result<Vec<RelatedVerse>, ScriptureError> {
        let target = self.verse(book, chapter, verse).ok_or_else(|| {
            ScriptureError::VerseNotFound(format!("{} {}:{}", book, chapter, verse))
        })?;
        let target_words: HashSet<String> = tokenize(&target.text).into_iter().collect();
        let target_key = VerseKey::new(book, chapter, verse);

        let mut scored: Vec<RelatedVerse> = Vec::new();
        for key in self.ordered_keys() {
            if key == target_key {
                continue;
            }
            let Some(candidate) = self.verse_for_key(&key) else {
                continue;
            };
            let candidate_words: HashSet<String> = tokenize(&candidate.text).into_iter().collect();
            let mut shared: Vec<String> = target_words
                .intersection(&candidate_words)
                .cloned()
                .collect();
            if shared.is_empty() {
                continue;
            }
            shared.sort();
            scored.push(RelatedVerse {
                verse: candidate,
                shared_words: shared.len(),
                shared,
            });
        }
        scored.sort_by(|a, b| {
            b.shared_words
                .cmp(&a.shared_words)
                .then_with(|| self.cmp_canonical(&a.verse, &b.verse))
        });
        Ok(scored)
    }

    /// Compare two verses in canonical order.
    fn cmp_canonical(&self, a: &Verse, b: &Verse) -> std::cmp::Ordering {
        let key = |v: &Verse| {
            let k = VerseKey::from_reference(&v.reference).unwrap();
            (self.book_index(&k.book), k.chapter, k.verse)
        };
        key(a).cmp(&key(b))
    }
}

/// Word -> how many verses contain it. A tiny frequency helper for
/// research: useful to see which words dominate a library.
pub fn word_frequencies(library: &Library) -> HashMap<String, usize> {
    let mut freq = HashMap::new();
    for key in library.ordered_keys() {
        if let Some(verse) = library.verse_for_key(&key) {
            let words: HashSet<String> = tokenize(&verse.text).into_iter().collect();
            for word in words {
                *freq.entry(word).or_insert(0) += 1;
            }
        }
    }
    freq
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same small known text as the library tests: two books, shared
    /// words across verses.
    fn sample_text() -> &'static str {
        "John 3:16\tFor God so loved the world, that he gave his only Son.\n\
         John 3:17\tFor God did not send his Son into the world to condemn the world.\n\
         Psalm 23:1\tThe Lord is my shepherd; I shall not want.\n\
         Psalm 23:2\tHe makes me lie down in green pastures.\n"
    }

    fn sample_library() -> Library {
        let mut lib = Library::new();
        lib.ingest(sample_text()).unwrap();
        lib
    }

    #[test]
    fn verses_containing_in_one_book() {
        let lib = sample_library();
        let hits = lib.verses_containing("world", "John");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].reference.to_string(), "John 3:16");
        assert_eq!(hits[1].reference.to_string(), "John 3:17");
        assert!(lib.verses_containing("world", "Psalm").is_empty());
        assert!(lib.verses_containing("world", "Exodus").is_empty());
        // Case-insensitive book and word.
        assert_eq!(lib.verses_containing("WORLD", "JOHN").len(), 2);
        // Exact word only.
        assert!(lib.verses_containing("love", "John").is_empty());
        assert_eq!(lib.verses_containing("loved", "John").len(), 1);
    }

    #[test]
    fn related_verses_share_words() {
        let lib = sample_library();
        // John 3:16 shares "for", "god", "so"... with John 3:17, and
        // "the" with Psalm 23:1. John 3:17 shares the most.
        let related = lib.related_verses("John", 3, 16).unwrap();
        assert!(!related.is_empty());
        // The top result is John 3:17 (most shared words).
        assert_eq!(related[0].verse.reference.to_string(), "John 3:17");
        assert!(related[0].shared_words >= 3);
        assert!(related[0].shared.contains(&"god".to_string()));
        assert!(related[0].shared.contains(&"world".to_string()));
        // The query verse itself is never returned.
        assert!(related
            .iter()
            .all(|r| r.verse.reference.to_string() != "John 3:16"));
        // Scores sort descending.
        let scores: Vec<usize> = related.iter().map(|r| r.shared_words).collect();
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.cmp(a));
        assert_eq!(scores, sorted);
    }

    #[test]
    fn related_verses_requires_the_verse() {
        let lib = sample_library();
        assert_eq!(
            lib.related_verses("John", 3, 99),
            Err(ScriptureError::VerseNotFound("John 3:99".to_string()))
        );
    }

    #[test]
    fn related_verses_with_no_shared_words_is_empty() {
        let mut lib = Library::new();
        lib.ingest("John 1:1\tAlpha bravo charlie.\nPsalm 1:1\tDelta echo foxtrot.\n")
            .unwrap();
        assert!(lib.related_verses("John", 1, 1).unwrap().is_empty());
    }

    #[test]
    fn word_frequencies_counts_verses_not_occurrences() {
        let lib = sample_library();
        let freq = word_frequencies(&lib);
        assert_eq!(freq["world"], 2);
        assert_eq!(freq["god"], 2);
        assert_eq!(freq["shepherd"], 1);
        assert!(!freq.contains_key("moses"));
    }
}
