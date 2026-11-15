//! Bible study and research pipeline for KERNEL.
//!
//! A **library** is a set of verses ingested from plain text, one verse
//! per line, in an explicit format:
//!
//! ```text
//! John 3:16<TAB>For God so loved the world...
//! Psalm 23:1<TAB>The Lord is my shepherd...
//! ```
//!
//! Verses are stored content-addressed in a [`kernel_storage::ContentStore`]
//! (text blocks by CID), so the same text always has the same address and
//! ingesting the same text twice stores it once. References like
//! `John 3:16`, `John 3:16-18`, and `Psalm 23` (whole chapter) parse into
//! a structured [`Reference`]. A concordance maps each word to the verses
//! containing it; search covers exact phrases and single words, scoped to
//! a book or the whole library. Study notes are signed records attached
//! to a verse, verified against the author's key on read. Research
//! queries answer "all verses containing X in book Y" and "verses that
//! share words with verse Z".
//!
//! # Honest limits
//!
//! - **No real Bible text is bundled here.** The caller supplies the
//!   text — finding a public-domain text is the caller's job.
//! - No translation, no audio, no commentary, no cross-references
//!   from any published study Bible.
//! - The concordance is exact-word based, not semantic: "love" and
//!   "loved" are different words, and no stemming is done.
//! - Related-verses is shared-words only, not meaning — and there is
//!   no stopword list, so common words like "the" count.
//! - Storage is in-memory (`ContentStore`); nothing here persists to
//!   disk. Persistence is the caller's job (see `kernel-store-disk`).
//! - Verse text is stored as one block per verse; a whole book is
//!   many blocks, not one.

pub mod concordance;
pub mod error;
pub mod library;
pub mod notes;
pub mod query;
pub mod reference;
pub mod tokenize;

pub use concordance::Concordance;
pub use error::ScriptureError;
pub use library::{IngestReport, Library, Scope, Verse};
pub use notes::StudyNote;
pub use query::RelatedVerse;
pub use reference::{Reference, VerseKey};
