//! Errors returned by scripture operations.

use std::fmt;

use kernel_identity::IdentityError;
use kernel_storage::StorageError;

/// Errors returned by scripture operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScriptureError {
    /// A line of ingested text was malformed. `line` is 1-based and
    /// counts every line in the input, including blank lines.
    MalformedLine { line: usize, reason: String },
    /// A verse reference did not parse.
    BadReference(String),
    /// A verse that does not exist in the library was requested.
    VerseNotFound(String),
    /// Ingest found the same verse twice with different text.
    /// `line` is the 1-based line of the conflicting verse.
    DuplicateVerse { reference: String, line: usize },
    /// A study note's signature did not verify against its author.
    BadSignature,
    /// A study note's encoded bytes were malformed.
    BadNote(String),
    /// A persistence operation failed (I/O or a corrupt log file).
    /// The message names the problem; no raw I/O error is kept, so
    /// this stays `Clone` and `PartialEq` like the other variants.
    Persistence(String),
    /// Wraps a [`StorageError`].
    Storage(StorageError),
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
}

impl fmt::Display for ScriptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScriptureError::MalformedLine { line, reason } => {
                write!(f, "line {line}: {reason}")
            }
            ScriptureError::BadReference(msg) => write!(f, "bad reference: {msg}"),
            ScriptureError::VerseNotFound(reference) => {
                write!(f, "verse not found: {reference}")
            }
            ScriptureError::DuplicateVerse { reference, line } => write!(
                f,
                "line {line}: verse {reference} already ingested with different text"
            ),
            ScriptureError::BadSignature => {
                write!(f, "study note signature is invalid")
            }
            ScriptureError::BadNote(msg) => write!(f, "bad study note: {msg}"),
            ScriptureError::Persistence(msg) => write!(f, "persistence error: {msg}"),
            ScriptureError::Storage(e) => write!(f, "storage error: {e}"),
            ScriptureError::Identity(e) => write!(f, "identity error: {e}"),
        }
    }
}

impl std::error::Error for ScriptureError {}

impl From<StorageError> for ScriptureError {
    fn from(e: StorageError) -> Self {
        ScriptureError::Storage(e)
    }
}

impl From<std::io::Error> for ScriptureError {
    fn from(e: std::io::Error) -> Self {
        ScriptureError::Persistence(e.to_string())
    }
}

impl From<IdentityError> for ScriptureError {
    fn from(e: IdentityError) -> Self {
        ScriptureError::Identity(e)
    }
}
