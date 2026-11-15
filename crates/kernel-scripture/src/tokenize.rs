//! Word normalization and tokenization.
//!
//! One definition, used by the concordance, the word search, the
//! phrase search, and the related-verses query, so all of them agree
//! on what a "word" is:
//!
//! - Split on whitespace.
//! - Lowercase (case-insensitive everywhere).
//! - Strip leading and trailing punctuation (anything that is not
//!   alphanumeric). Internal punctuation is kept: "don't" stays one
//!   word, "well-known" stays one word.
//! - Drop tokens that are empty after stripping.
//!
//! No stemming, no stopwords, no semantics: "love" and "loved" are
//! different words.

/// Normalize one raw token into a concordance word. Returns `None`
/// if nothing is left (the token was all punctuation).
pub fn normalize_word(raw: &str) -> Option<String> {
    let stripped = raw
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    if stripped.is_empty() {
        None
    } else {
        Some(stripped)
    }
}

/// Split text into normalized words, in order, with duplicates kept.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace().filter_map(normalize_word).collect()
}

/// Normalize whole text for phrase search: tokenize, then join with
/// single spaces. Two texts match as a phrase when the normalized
/// phrase appears as a substring of the normalized text — punctuation
/// is ignored and words must appear consecutively, in order.
pub fn normalize_text(text: &str) -> String {
    tokenize(text).join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_normalization() {
        assert_eq!(normalize_word("God,"), Some("god".to_string()));
        assert_eq!(normalize_word("\"Loved\""), Some("loved".to_string()));
        assert_eq!(normalize_word("don't"), Some("don't".to_string()));
        assert_eq!(normalize_word("well-known"), Some("well-known".to_string()));
        assert_eq!(normalize_word("..."), None);
        assert_eq!(normalize_word(""), None);
        assert_eq!(normalize_word("LOVED"), Some("loved".to_string()));
    }

    #[test]
    fn tokenize_keeps_order_and_duplicates() {
        assert_eq!(
            tokenize("God so loved, God so loved."),
            vec!["god", "so", "loved", "god", "so", "loved"]
        );
        assert_eq!(tokenize("   "), Vec::<String>::new());
    }

    #[test]
    fn normalize_text_collapses_punctuation_and_space() {
        assert_eq!(
            normalize_text("For  God, so   loved the world!"),
            "for god so loved the world"
        );
    }
}
