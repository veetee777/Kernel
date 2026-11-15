//! The scripture library: ingest, storage, search, and notes.
//!
//! Ingest format: one verse per line,
//! `Book Chapter:Verse<TAB>text`. Blank lines are skipped. A
//! malformed line is a clean [`ScriptureError::MalformedLine`] naming
//! its 1-based line number — never a panic. Each verse's text is
//! stored as one block in a [`ContentStore`], addressed by the
//! SHA-256 of the text, so the same text ingested twice stores once.
//!
//! Canonical order — the order results are returned in — is:
//! books in first-ingested order, then chapter, then verse.

use std::collections::HashMap;

use kernel_storage::{Cid, ContentStore};

use crate::concordance::Concordance;
use crate::error::ScriptureError;
use crate::notes::StudyNote;
use crate::reference::{normalize_book, Reference, VerseKey};
use crate::tokenize::{normalize_text, normalize_word};

/// One verse in the library: its reference, its text, and the CID
/// of the block holding the text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verse {
    /// The verse's reference (always a single verse).
    pub reference: Reference,
    /// The verse text, as ingested.
    pub text: String,
    /// The CID of the stored text block (SHA-256 of the text).
    pub cid: Cid,
}

/// Where a search looks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Scope {
    /// The whole library.
    All,
    /// One book, matched case-insensitively. An unknown book
    /// matches nothing.
    Book(String),
}

/// What [`Library::ingest`] did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IngestReport {
    /// Non-blank lines parsed.
    pub lines_seen: usize,
    /// Verses newly added to the library.
    pub verses_added: usize,
    /// Verses already present with the same text (stored once).
    pub verses_skipped: usize,
}

/// A library of verses, with a concordance and study notes.
#[derive(Default)]
pub struct Library {
    store: ContentStore,
    /// Verse key -> CID of the verse's text block.
    index: HashMap<VerseKey, Cid>,
    /// Canonical book display names, in first-ingested order.
    books: Vec<String>,
    /// Lowercased book name -> canonical display name.
    book_lookup: HashMap<String, String>,
    concordance: Concordance,
    /// Verse key -> notes attached to that verse.
    notes: HashMap<VerseKey, Vec<StudyNote>>,
}

impl Library {
    /// An empty library.
    pub fn new() -> Self {
        Self {
            store: ContentStore::new(),
            index: HashMap::new(),
            books: Vec::new(),
            book_lookup: HashMap::new(),
            concordance: Concordance::new(),
            notes: HashMap::new(),
        }
    }

    /// Number of verses in the library.
    pub fn verse_count(&self) -> usize {
        self.index.len()
    }

    /// Number of blocks in the content store (one per distinct verse
    /// text).
    pub fn block_count(&self) -> usize {
        self.store.len()
    }

    /// Canonical book names, in first-ingested order.
    pub fn book_names(&self) -> &[String] {
        &self.books
    }

    /// The concordance built from everything ingested so far.
    pub fn concordance(&self) -> &Concordance {
        &self.concordance
    }

    /// Ingest verses from plain text, one verse per line:
    /// `Book Chapter:Verse<TAB>text`. Blank lines are skipped. A
    /// line without a tab, with an unparsable reference, with a
    /// range or whole-chapter reference, or with empty text is a
    /// [`ScriptureError::MalformedLine`] naming its line number.
    /// Ingesting a verse already present with the same text is a
    /// no-op (stored once); the same verse with different text is
    /// [`ScriptureError::DuplicateVerse`]. Ingest stops at the
    /// first error; verses before it stay ingested.
    pub fn ingest(&mut self, text: &str) -> Result<IngestReport, ScriptureError> {
        let mut report = IngestReport {
            lines_seen: 0,
            verses_added: 0,
            verses_skipped: 0,
        };
        for (i, raw_line) in text.lines().enumerate() {
            let line_no = i + 1;
            if raw_line.trim().is_empty() {
                continue;
            }
            report.lines_seen += 1;
            let (ref_part, verse_text) =
                raw_line
                    .split_once('\t')
                    .ok_or_else(|| ScriptureError::MalformedLine {
                        line: line_no,
                        reason: "missing tab separator between reference and text".to_string(),
                    })?;
            let reference = Reference::parse(ref_part).map_err(|e| match e {
                ScriptureError::BadReference(msg) => ScriptureError::MalformedLine {
                    line: line_no,
                    reason: msg,
                },
                other => other,
            })?;
            let key = match VerseKey::from_reference(&reference) {
                Some(key) => key,
                None => {
                    return Err(ScriptureError::MalformedLine {
                        line: line_no,
                        reason: format!("ingest needs one verse per line, not '{reference}'"),
                    })
                }
            };
            let verse_text = verse_text.trim().to_string();
            if verse_text.is_empty() {
                return Err(ScriptureError::MalformedLine {
                    line: line_no,
                    reason: "verse text is empty".to_string(),
                });
            }
            if let Some(&existing_cid) = self.index.get(&key) {
                let existing = self.store.get(&existing_cid)?;
                if existing == verse_text.as_bytes() {
                    report.verses_skipped += 1;
                    continue;
                }
                return Err(ScriptureError::DuplicateVerse {
                    reference: reference.to_string(),
                    line: line_no,
                });
            }
            let cid = self.store.put(verse_text.as_bytes());
            self.index.insert(key.clone(), cid);
            self.register_book(&reference.book);
            self.concordance.add(&key, &verse_text);
            report.verses_added += 1;
        }
        Ok(report)
    }

    /// Record a book name if this is the first verse seen from it.
    fn register_book(&mut self, book: &str) {
        let lower = normalize_book(book);
        if let std::collections::hash_map::Entry::Vacant(entry) = self.book_lookup.entry(lower) {
            entry.insert(book.to_string());
            self.books.push(book.to_string());
        }
    }

    /// The canonical display name of a book, matched
    /// case-insensitively. `None` for a book with no verses.
    pub fn canonical_book(&self, book: &str) -> Option<&str> {
        self.book_lookup
            .get(&normalize_book(book))
            .map(|s| s.as_str())
    }

    /// One verse by book, chapter, and verse. `None` if the library
    /// does not hold it.
    pub fn verse(&self, book: &str, chapter: u32, verse: u32) -> Option<Verse> {
        let key = VerseKey::new(book, chapter, verse);
        self.verse_for_key(&key)
    }

    /// All verses a reference covers, in canonical order: a single
    /// verse, a range, or a whole chapter. Verses the library does
    /// not hold are simply absent from the result.
    pub fn verses(&self, reference: &Reference) -> Vec<Verse> {
        match reference.verse {
            None => {
                let chapter = reference.chapter;
                let prefix = normalize_book(&reference.book);
                let mut keys: Vec<&VerseKey> = self
                    .index
                    .keys()
                    .filter(|k| k.book == prefix && k.chapter == chapter)
                    .collect();
                keys.sort_by_key(|k| k.verse);
                keys.into_iter()
                    .filter_map(|k| self.verse_for_key(k))
                    .collect()
            }
            Some((start, end)) => (start..=end)
                .filter_map(|v| self.verse(&reference.book, reference.chapter, v))
                .collect(),
        }
    }

    /// Build a [`Verse`] for an indexed key. `None` if the key is
    /// not indexed or its block is missing (which cannot happen
    /// through the public API).
    pub(crate) fn verse_for_key(&self, key: &VerseKey) -> Option<Verse> {
        let cid = self.index.get(key)?;
        let bytes = self.store.get(cid).ok()?;
        let text = String::from_utf8(bytes).ok()?;
        let book = self
            .canonical_book(&key.book)
            .unwrap_or(&key.book)
            .to_string();
        Some(Verse {
            reference: Reference {
                book,
                chapter: key.chapter,
                verse: Some((key.verse, key.verse)),
            },
            text,
            cid: *cid,
        })
    }

    /// All verse keys, in canonical order (books in first-ingested
    /// order, then chapter, then verse).
    pub(crate) fn ordered_keys(&self) -> Vec<VerseKey> {
        let mut keys: Vec<VerseKey> = self.index.keys().cloned().collect();
        keys.sort_by_key(|k| (self.book_index(&k.book), k.chapter, k.verse));
        keys
    }

    /// Position of a book in first-ingested order; unknown books
    /// sort after all known ones.
    pub(crate) fn book_index(&self, lower_book: &str) -> usize {
        self.books
            .iter()
            .position(|b| normalize_book(b) == lower_book)
            .unwrap_or(usize::MAX)
    }

    /// Does a verse key fall inside a search scope?
    fn in_scope(&self, key: &VerseKey, scope: &Scope) -> bool {
        match scope {
            Scope::All => true,
            Scope::Book(book) => key.book == normalize_book(book),
        }
    }

    /// Verses containing one word, in canonical order. The word is
    /// normalized like indexed text (case-insensitive, punctuation
    /// stripped), so "God", "god", and "god," all match. Exact words
    /// only: "love" does not match "loved".
    pub fn search_word(&self, word: &str, scope: &Scope) -> Vec<Verse> {
        if normalize_word(word).is_none() {
            return Vec::new();
        }
        let mut keys: Vec<&VerseKey> = self
            .concordance
            .lookup(word)
            .iter()
            .filter(|k| self.in_scope(k, scope))
            .collect();
        keys.sort_by_key(|k| (self.book_index(&k.book), k.chapter, k.verse));
        keys.into_iter()
            .filter_map(|k| self.verse_for_key(k))
            .collect()
    }

    /// Verses containing an exact phrase, in canonical order. The
    /// phrase and the verse text are both normalized (lowercased,
    /// punctuation ignored, whitespace collapsed), so "God so loved"
    /// matches "For God, so loved the world" — words must appear
    /// consecutively, in order. An empty phrase matches nothing.
    pub fn search_phrase(&self, phrase: &str, scope: &Scope) -> Vec<Verse> {
        let needle = normalize_text(phrase);
        if needle.is_empty() {
            return Vec::new();
        }
        self.ordered_keys()
            .into_iter()
            .filter(|k| self.in_scope(k, scope))
            .filter_map(|k| self.verse_for_key(&k))
            .filter(|v| normalize_text(&v.text).contains(&needle))
            .collect()
    }

    /// Attach a signed study note to its verse. The note's signature
    /// is verified here; a bad signature is refused with
    /// [`ScriptureError::BadSignature`]. The verse must exist in the
    /// library, or the note is refused with
    /// [`ScriptureError::VerseNotFound`].
    pub fn add_note(&mut self, note: StudyNote) -> Result<(), ScriptureError> {
        if !note.verify() {
            return Err(ScriptureError::BadSignature);
        }
        let key = VerseKey::from_reference(&note.reference)
            .ok_or_else(|| ScriptureError::BadReference("note needs one verse".to_string()))?;
        if !self.index.contains_key(&key) {
            return Err(ScriptureError::VerseNotFound(note.reference.to_string()));
        }
        self.notes.entry(key).or_default().push(note);
        Ok(())
    }

    /// Notes attached to a verse, in the order they were added.
    /// Every note is re-verified against its author's key on read;
    /// a note whose signature no longer checks out is left out.
    pub fn notes_for(&self, book: &str, chapter: u32, verse: u32) -> Vec<&StudyNote> {
        let key = VerseKey::new(book, chapter, verse);
        match self.notes.get(&key) {
            Some(notes) => notes.iter().filter(|n| n.verify()).collect(),
            None => Vec::new(),
        }
    }

    /// Number of notes stored (including any that would fail
    /// re-verification on read).
    pub fn note_count(&self) -> usize {
        self.notes.values().map(|v| v.len()).sum()
    }

    /// Restore one verse from durable storage (used by
    /// [`crate::persistence`] replay). This is the ingest path
    /// without the text-format parsing: the reference is validated,
    /// the text is stored as one block, the book is registered, and
    /// the concordance is updated. A verse already present with the
    /// same text is a no-op (replay is idempotent, so a record
    /// written twice by a retried save is harmless); the same verse
    /// with different text is an error.
    pub(crate) fn restore_verse(
        &mut self,
        book: &str,
        chapter: u32,
        verse: u32,
        text: &str,
    ) -> Result<(), ScriptureError> {
        let reference = Reference::verse(book, chapter, verse)?;
        let key = VerseKey::from_reference(&reference).unwrap();
        if text.is_empty() {
            return Err(ScriptureError::Persistence(
                "verse text in log is empty".to_string(),
            ));
        }
        if let Some(&existing_cid) = self.index.get(&key) {
            let existing = self.store.get(&existing_cid)?;
            if existing == text.as_bytes() {
                return Ok(());
            }
            return Err(ScriptureError::Persistence(format!(
                "verse {reference} in log conflicts with stored text"
            )));
        }
        let cid = self.store.put(text.as_bytes());
        self.index.insert(key.clone(), cid);
        self.register_book(&reference.book);
        self.concordance.add(&key, text);
        Ok(())
    }

    /// The notes attached to a verse, exactly as stored (no
    /// re-verification filtering). Used by [`crate::persistence`]
    /// to find which notes are already durable.
    pub(crate) fn notes_for_key(&self, key: &VerseKey) -> &[StudyNote] {
        match self.notes.get(key) {
            Some(notes) => notes.as_slice(),
            None => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_identity::Identity;

    /// A small known text: two books, shared words across verses.
    pub fn sample_text() -> &'static str {
        "John 3:16\tFor God so loved the world, that he gave his only Son.\n\
         John 3:17\tFor God did not send his Son into the world to condemn the world.\n\
         Psalm 23:1\tThe Lord is my shepherd; I shall not want.\n\
         Psalm 23:2\tHe makes me lie down in green pastures.\n"
    }

    fn sample_library() -> Library {
        let mut lib = Library::new();
        let report = lib.ingest(sample_text()).unwrap();
        assert_eq!(report.verses_added, 4);
        lib
    }

    #[test]
    fn ingest_and_retrieve_round_trip() {
        let lib = sample_library();
        assert_eq!(lib.verse_count(), 4);
        assert_eq!(lib.book_names(), ["John", "Psalm"]);
        let v = lib.verse("John", 3, 16).unwrap();
        assert_eq!(
            v.text,
            "For God so loved the world, that he gave his only Son."
        );
        assert_eq!(v.reference.to_string(), "John 3:16");
        assert_eq!(v.cid, lib.verse("JOHN", 3, 16).unwrap().cid);
        assert!(lib.verse("John", 3, 99).is_none());
        assert!(lib.verse("Exodus", 1, 1).is_none());
    }

    #[test]
    fn ingest_whole_chapter_and_range() {
        let lib = sample_library();
        let chapter = lib.verses(&Reference::parse("Psalm 23").unwrap());
        assert_eq!(chapter.len(), 2);
        assert_eq!(chapter[0].reference.verse, Some((1, 1)));
        assert_eq!(chapter[1].reference.verse, Some((2, 2)));
        let range = lib.verses(&Reference::parse("John 3:16-17").unwrap());
        assert_eq!(range.len(), 2);
        // Missing verses are absent, not errors.
        let sparse = lib.verses(&Reference::parse("John 3:16-20").unwrap());
        assert_eq!(sparse.len(), 2);
    }

    #[test]
    fn malformed_lines_name_the_line() {
        let mut lib = Library::new();
        let bad = "John 3:16\tGood line.\nNo tab here\nJohn 3:17\tAlso good.\n";
        let err = lib.ingest(bad).unwrap_err();
        assert_eq!(
            err,
            ScriptureError::MalformedLine {
                line: 2,
                reason: "missing tab separator between reference and text".to_string(),
            }
        );
        // The good first line stayed ingested; ingest stops at the error.
        assert_eq!(lib.verse_count(), 1);

        let mut lib = Library::new();
        let err = lib
            .ingest("John 3:16-18\tA range is not one verse.\n")
            .unwrap_err();
        assert!(matches!(err, ScriptureError::MalformedLine { line: 1, .. }));

        let mut lib = Library::new();
        let err = lib.ingest("John 3:16\t\n").unwrap_err();
        assert!(matches!(err, ScriptureError::MalformedLine { line: 1, .. }));

        let mut lib = Library::new();
        let err = lib.ingest("John 0:16\tBad chapter.\n").unwrap_err();
        assert!(matches!(err, ScriptureError::MalformedLine { line: 1, .. }));

        // Blank lines are skipped and do not shift line numbers.
        let mut lib = Library::new();
        let err = lib
            .ingest("\n\nJohn 3:16\tGood.\n\nBad line\n")
            .unwrap_err();
        assert_eq!(
            err,
            ScriptureError::MalformedLine {
                line: 5,
                reason: "missing tab separator between reference and text".to_string(),
            }
        );
    }

    #[test]
    fn same_text_twice_stores_once() {
        let mut lib = Library::new();
        let first = lib.ingest(sample_text()).unwrap();
        assert_eq!(first.verses_added, 4);
        assert_eq!(lib.block_count(), 4);
        let second = lib.ingest(sample_text()).unwrap();
        assert_eq!(second.verses_added, 0);
        assert_eq!(second.verses_skipped, 4);
        assert_eq!(lib.block_count(), 4);
        // Same text under a different verse is still one block.
        let mut lib2 = Library::new();
        lib2.ingest("John 1:1\tSame words.\nPsalm 1:1\tSame words.\n")
            .unwrap();
        assert_eq!(lib2.block_count(), 1);
        assert_eq!(lib2.verse_count(), 2);
    }

    #[test]
    fn conflicting_duplicate_is_an_error() {
        let mut lib = Library::new();
        lib.ingest("John 3:16\tFirst text.\n").unwrap();
        let err = lib.ingest("John 3:16\tDifferent text.\n").unwrap_err();
        assert_eq!(
            err,
            ScriptureError::DuplicateVerse {
                reference: "John 3:16".to_string(),
                line: 1,
            }
        );
    }

    #[test]
    fn word_search_is_case_insensitive_and_exact() {
        let lib = sample_library();
        let world = lib.search_word("world", &Scope::All);
        assert_eq!(world.len(), 2);
        assert_eq!(world[0].reference.to_string(), "John 3:16");
        assert_eq!(world[1].reference.to_string(), "John 3:17");
        assert_eq!(lib.search_word("WORLD", &Scope::All).len(), 2);
        assert_eq!(lib.search_word("world,", &Scope::All).len(), 2);
        // Exact words only: "loved" does not match "love", and "son"
        // does not match "sons".
        assert_eq!(lib.search_word("love", &Scope::All).len(), 0);
        assert_eq!(lib.search_word("loved", &Scope::All).len(), 1);
        // Scoped to a book.
        assert_eq!(
            lib.search_word("world", &Scope::Book("Psalm".into())).len(),
            0
        );
        assert_eq!(
            lib.search_word("world", &Scope::Book("john".into())).len(),
            2
        );
        assert_eq!(
            lib.search_word("world", &Scope::Book("Exodus".into()))
                .len(),
            0
        );
        assert!(lib.search_word("...", &Scope::All).is_empty());
    }

    #[test]
    fn phrase_search_finds_consecutive_words() {
        let lib = sample_library();
        let hits = lib.search_phrase("God so loved", &Scope::All);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].reference.to_string(), "John 3:16");
        // Case-insensitive, punctuation ignored.
        assert_eq!(lib.search_phrase("GOD, SO LOVED", &Scope::All).len(), 1);
        // Words must be consecutive and in order.
        assert_eq!(lib.search_phrase("God world", &Scope::All).len(), 0);
        assert_eq!(lib.search_phrase("loved God", &Scope::All).len(), 0);
        // Scoped.
        assert_eq!(
            lib.search_phrase("the world", &Scope::Book("Psalm".into()))
                .len(),
            0
        );
        assert_eq!(lib.search_phrase("the world", &Scope::All).len(), 2);
        assert!(lib.search_phrase("", &Scope::All).is_empty());
        assert!(lib.search_phrase("   ", &Scope::All).is_empty());
    }

    #[test]
    fn search_results_are_in_canonical_order() {
        let mut lib = Library::new();
        // Ingest out of order: Psalm first, John second.
        lib.ingest("Psalm 23:1\tThe Lord is my shepherd.\nJohn 3:16\tFor God so loved the world.\nJohn 3:17\tGod did not send his Son into the world.\n")
            .unwrap();
        let hits = lib.search_word("world", &Scope::All);
        // Canonical order is first-ingested book first (Psalm has no
        // "world"), then John 3:16 before John 3:17.
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].reference.to_string(), "John 3:16");
        assert_eq!(hits[1].reference.to_string(), "John 3:17");
    }

    #[test]
    fn notes_attach_verify_and_tampered_are_refused() {
        let mut lib = sample_library();
        let id = Identity::generate("tristan").unwrap();
        let note = StudyNote::new(
            &id,
            &Reference::parse("John 3:16").unwrap(),
            "Love here is given, not earned.",
            1_700_000_000,
        )
        .unwrap();
        lib.add_note(note).unwrap();
        assert_eq!(lib.note_count(), 1);
        let notes = lib.notes_for("John", 3, 16);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].text, "Love here is given, not earned.");
        assert_eq!(notes[0].author.display_name(), "tristan");
        // A tampered note is refused on add.
        let mut bad = StudyNote::new(
            &id,
            &Reference::parse("John 3:16").unwrap(),
            "Untampered.",
            1_700_000_000,
        )
        .unwrap();
        bad.text = "Tampered.".to_string();
        assert_eq!(lib.add_note(bad), Err(ScriptureError::BadSignature));
        assert_eq!(lib.note_count(), 1);
        // A note on a verse the library does not hold is refused.
        let missing = StudyNote::new(
            &id,
            &Reference::parse("John 3:18").unwrap(),
            "No such verse.",
            1_700_000_000,
        )
        .unwrap();
        assert_eq!(
            lib.add_note(missing),
            Err(ScriptureError::VerseNotFound("John 3:18".to_string()))
        );
    }
}
