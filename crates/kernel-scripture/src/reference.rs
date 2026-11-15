//! Verse references: `John 3:16`, `John 3:16-18`, `Psalm 23`.
//!
//! A [`Reference`] is a book, a chapter, and either a single verse, a
//! verse range, or nothing (whole chapter). Book names keep the case
//! they were written with for display; matching is case-insensitive
//! through [`VerseKey`], which lowercases the book.

use std::fmt;

use crate::error::ScriptureError;

/// A single verse, addressed by book (lowercased), chapter, and verse.
/// This is the key the library indexes verses by.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VerseKey {
    /// Book name, lowercased and whitespace-collapsed.
    pub book: String,
    /// Chapter number, 1-based.
    pub chapter: u32,
    /// Verse number, 1-based.
    pub verse: u32,
}

impl VerseKey {
    /// Build a key, normalizing the book name (trimmed, internal
    /// whitespace collapsed to single spaces, lowercased).
    pub fn new(book: &str, chapter: u32, verse: u32) -> Self {
        Self {
            book: normalize_book(book),
            chapter,
            verse,
        }
    }

    /// The key for a single-verse [`Reference`]. Returns `None` for a
    /// whole-chapter reference or a verse range.
    pub fn from_reference(reference: &Reference) -> Option<Self> {
        match reference.verse {
            Some((start, end)) if start == end => {
                Some(Self::new(&reference.book, reference.chapter, start))
            }
            _ => None,
        }
    }
}

/// Normalize a book name for matching: trim, collapse internal
/// whitespace to single spaces, lowercase.
pub fn normalize_book(book: &str) -> String {
    book.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// A verse reference: a book, a chapter, and a verse or verse range.
///
/// - `John 3:16` — one verse.
/// - `John 3:16-18` — a range, both ends inclusive.
/// - `Psalm 23` — a whole chapter (`verse` is `None`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Reference {
    /// Book name as written (trimmed, whitespace collapsed, case kept).
    pub book: String,
    /// Chapter number, 1-based.
    pub chapter: u32,
    /// Verse range, both ends inclusive. `None` means the whole chapter.
    pub verse: Option<(u32, u32)>,
}

impl Reference {
    /// A single verse: `John 3:16`.
    pub fn verse(book: &str, chapter: u32, verse: u32) -> Result<Self, ScriptureError> {
        if chapter == 0 {
            return Err(ScriptureError::BadReference(
                "chapter must be at least 1".to_string(),
            ));
        }
        if verse == 0 {
            return Err(ScriptureError::BadReference(
                "verse must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            book: collapse_book(book)?,
            chapter,
            verse: Some((verse, verse)),
        })
    }

    /// A whole chapter: `Psalm 23`.
    pub fn chapter(book: &str, chapter: u32) -> Result<Self, ScriptureError> {
        if chapter == 0 {
            return Err(ScriptureError::BadReference(
                "chapter must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            book: collapse_book(book)?,
            chapter,
            verse: None,
        })
    }

    /// A verse range, both ends inclusive: `John 3:16-18`.
    pub fn range(book: &str, chapter: u32, start: u32, end: u32) -> Result<Self, ScriptureError> {
        if chapter == 0 {
            return Err(ScriptureError::BadReference(
                "chapter must be at least 1".to_string(),
            ));
        }
        if start == 0 || end == 0 {
            return Err(ScriptureError::BadReference(
                "verse must be at least 1".to_string(),
            ));
        }
        if start > end {
            return Err(ScriptureError::BadReference(format!(
                "verse range {start}-{end} is backwards"
            )));
        }
        Ok(Self {
            book: collapse_book(book)?,
            chapter,
            verse: Some((start, end)),
        })
    }

    /// Parse `John 3:16`, `John 3:16-18`, or `Psalm 23`. The book is
    /// everything before the final whitespace-separated token, so
    /// multi-word books (`1 John`, `Song of Solomon`) work. Bad input
    /// is a clean [`ScriptureError::BadReference`], never a panic.
    pub fn parse(input: &str) -> Result<Self, ScriptureError> {
        let input = input.trim();
        if input.is_empty() {
            return Err(ScriptureError::BadReference("empty reference".to_string()));
        }
        // The last whitespace-separated token is the chapter[:verse]
        // part; everything before it is the book name.
        let split = input
            .rfind(char::is_whitespace)
            .ok_or_else(|| ScriptureError::BadReference(format!("no chapter in '{input}'")))?;
        let (book_part, tail) = input.split_at(split);
        let tail = tail.trim_start();
        let book = collapse_book(book_part)?;
        if tail.is_empty() {
            return Err(ScriptureError::BadReference(format!(
                "no chapter in '{input}'"
            )));
        }
        match tail.split_once(':') {
            None => {
                // Whole chapter: `Psalm 23`.
                let chapter = parse_positive(tail, input)?;
                Ok(Self {
                    book,
                    chapter,
                    verse: None,
                })
            }
            Some((chapter_part, verse_part)) => {
                if verse_part.contains(':') {
                    return Err(ScriptureError::BadReference(format!(
                        "too many ':' in '{input}'"
                    )));
                }
                let chapter = parse_positive(chapter_part, input)?;
                match verse_part.split_once('-') {
                    None => {
                        let verse = parse_positive(verse_part, input)?;
                        Ok(Self {
                            book,
                            chapter,
                            verse: Some((verse, verse)),
                        })
                    }
                    Some((start_part, end_part)) => {
                        if end_part.contains('-') {
                            return Err(ScriptureError::BadReference(format!(
                                "too many '-' in '{input}'"
                            )));
                        }
                        let start = parse_positive(start_part, input)?;
                        let end = parse_positive(end_part, input)?;
                        if start > end {
                            return Err(ScriptureError::BadReference(format!(
                                "verse range {start}-{end} is backwards in '{input}'"
                            )));
                        }
                        Ok(Self {
                            book,
                            chapter,
                            verse: Some((start, end)),
                        })
                    }
                }
            }
        }
    }

    /// True for a single verse (`John 3:16`).
    pub fn is_single_verse(&self) -> bool {
        matches!(self.verse, Some((start, end)) if start == end)
    }

    /// True for a whole chapter (`Psalm 23`).
    pub fn is_whole_chapter(&self) -> bool {
        self.verse.is_none()
    }
}

impl fmt::Display for Reference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.verse {
            None => write!(f, "{} {}", self.book, self.chapter),
            Some((start, end)) if start == end => {
                write!(f, "{} {}:{}", self.book, self.chapter, start)
            }
            Some((start, end)) => {
                write!(f, "{} {}:{}-{}", self.book, self.chapter, start, end)
            }
        }
    }
}

/// Collapse a book name's whitespace (trim + single spaces), keeping
/// case. Rejects an empty name and names with characters outside
/// letters, digits, and spaces.
fn collapse_book(book: &str) -> Result<String, ScriptureError> {
    let collapsed = book.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return Err(ScriptureError::BadReference(
            "book name must not be empty".to_string(),
        ));
    }
    if !collapsed
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == ' ')
    {
        return Err(ScriptureError::BadReference(format!(
            "book name '{collapsed}' may only use letters, digits, and spaces"
        )));
    }
    Ok(collapsed)
}

/// Parse a strictly positive integer: ASCII digits only, at least 1.
/// Rejects signs, decimals, empty strings, and zero.
fn parse_positive(part: &str, whole: &str) -> Result<u32, ScriptureError> {
    if part.is_empty() || !part.chars().all(|c| c.is_ascii_digit()) {
        return Err(ScriptureError::BadReference(format!(
            "'{part}' is not a number in '{whole}'"
        )));
    }
    let n: u32 = part
        .parse()
        .map_err(|_| ScriptureError::BadReference(format!("'{part}' is too large in '{whole}'")))?;
    if n == 0 {
        return Err(ScriptureError::BadReference(format!(
            "chapter and verse numbers start at 1 in '{whole}'"
        )));
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_verse() {
        let r = Reference::parse("John 3:16").unwrap();
        assert_eq!(r.book, "John");
        assert_eq!(r.chapter, 3);
        assert_eq!(r.verse, Some((16, 16)));
        assert_eq!(r.to_string(), "John 3:16");
        assert!(r.is_single_verse());
    }

    #[test]
    fn parse_range() {
        let r = Reference::parse("John 3:16-18").unwrap();
        assert_eq!(r.verse, Some((16, 18)));
        assert_eq!(r.to_string(), "John 3:16-18");
        assert!(!r.is_single_verse());
    }

    #[test]
    fn parse_whole_chapter() {
        let r = Reference::parse("Psalm 23").unwrap();
        assert_eq!(r.book, "Psalm");
        assert_eq!(r.chapter, 23);
        assert_eq!(r.verse, None);
        assert_eq!(r.to_string(), "Psalm 23");
        assert!(r.is_whole_chapter());
    }

    #[test]
    fn parse_multiword_books() {
        assert_eq!(Reference::parse("1 John 4:8").unwrap().book, "1 John");
        assert_eq!(
            Reference::parse("Song of Solomon 2:4").unwrap().book,
            "Song of Solomon"
        );
        // Extra whitespace collapses; case is kept as written.
        assert_eq!(
            Reference::parse("  john   3:16 ").unwrap().to_string(),
            "john 3:16"
        );
    }

    #[test]
    fn parse_bad_references_are_clean_errors() {
        for bad in [
            "",
            "   ",
            "John",
            "John 3:",
            "John :16",
            "John 3:16-",
            "John 3:-18",
            "John 3:18-16",
            "John 0:16",
            "John 3:0",
            "Psalm 0",
            "John 3:16:17",
            "John 3:16-18-20",
            "John 3:abc",
            "John 3:-16",
            "John 3:+16",
            "John 3.16",
            "John@ 3:16",
        ] {
            assert!(
                Reference::parse(bad).is_err(),
                "expected an error for '{bad}'"
            );
        }
    }

    #[test]
    fn constructors_validate() {
        assert!(Reference::verse("John", 3, 16).is_ok());
        assert!(Reference::verse("John", 0, 16).is_err());
        assert!(Reference::verse("John", 3, 0).is_err());
        assert!(Reference::chapter("Psalm", 0).is_err());
        assert!(Reference::range("John", 3, 18, 16).is_err());
        assert!(Reference::range("John", 3, 16, 16).is_ok());
        assert!(Reference::verse("", 3, 16).is_err());
    }

    #[test]
    fn verse_key_normalizes_book_case() {
        assert_eq!(VerseKey::new("John", 3, 16), VerseKey::new("JOHN", 3, 16));
        assert_eq!(
            VerseKey::new("Song  of   Solomon", 2, 4),
            VerseKey::new("song of solomon", 2, 4)
        );
        let r = Reference::parse("John 3:16").unwrap();
        assert_eq!(
            VerseKey::from_reference(&r).unwrap(),
            VerseKey::new("john", 3, 16)
        );
        assert!(VerseKey::from_reference(&Reference::parse("Psalm 23").unwrap()).is_none());
        assert!(VerseKey::from_reference(&Reference::parse("John 3:16-18").unwrap()).is_none());
    }
}
