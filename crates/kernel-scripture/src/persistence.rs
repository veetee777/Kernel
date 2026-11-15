//! Durable binary persistence for the scripture library.
//!
//! One append-only binary log file, `scripture.klog`, in the store
//! directory. The format follows the `kernel-store-disk` `KBLK1`
//! pattern: every record is framed as
//!
//! ```text
//! KSCR1 (5 bytes) + kind (1 byte) + payload length (u32 LE)
//!       + payload bytes + SHA-256(kind || payload) (32 bytes)
//! ```
//!
//! Record kinds: `1` is one ingested verse (canonical display book
//! name, chapter, verse, text); `2` is one signed study note (its
//! `KNOTE1` encoding). The verse text is stored inline in the record
//! — the log alone rebuilds the library exactly, with no second file.
//! The concordance is not stored: it is rebuilt from the verse texts
//! during replay, so it can never drift from the verses.
//!
//! A [`ScriptureStore`] replays the log on open, rebuilding the
//! library, the book order, and the concordance. [`ScriptureStore::save`]
//! appends only the verses and notes that are not durable yet, then
//! fsyncs once for the batch — a save that returns has its records on
//! disk. Replay is idempotent: a verse record already present with
//! the same text, or a note record already attached, is skipped, so
//! a save that is interrupted and retried cannot corrupt the log with
//! duplicates.
//!
//! Corruption behavior, by design:
//!
//! - A log whose **first** record is corrupt (bad magic, unknown
//!   kind, oversize length, failed hash, undecodable payload, or a
//!   note whose signature does not verify) is **refused** with a
//!   clean [`ScriptureError::Persistence`] error — the file is not a
//!   scripture log at all, and silently treating it as empty would
//!   hide the damage.
//! - Corruption **later** in the log cuts the tail: replay stops at
//!   the first bad record and the file is truncated back to the last
//!   good record, exactly like `kernel-store-disk`. A crash
//!   mid-write leaves a torn final record, which is cut the same
//!   way; everything before it survives.
//! - A note whose signature fails verification on load is a corrupt
//!   record: it is refused and cuts the tail there. Notes can only
//!   enter the library through the verified `add_note` path, so a
//!   bad signature in the log means the bytes were tampered with.
//!
//! No JSON on disk, no SQLite, no Postgres — binary records only.
//! Time is never read here: notes carry caller-supplied timestamps,
//! and replay uses none of its own.

use std::collections::{HashMap, HashSet};
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use kernel_crypto::sha256;

use crate::error::ScriptureError;
use crate::library::Library;
use crate::notes::StudyNote;
use crate::reference::VerseKey;

/// Magic bytes at the start of every log record.
const RECORD_MAGIC: &[u8; 5] = b"KSCR1";
/// Name of the log file inside the store directory.
const LOG_FILE_NAME: &str = "scripture.klog";
/// Largest payload a record may carry (64 MiB). A length above this
/// is corruption, not a verse — it also bounds the replay allocation.
const MAX_PAYLOAD: u32 = 64 * 1024 * 1024;
/// Record kind: one ingested verse.
const KIND_VERSE: u8 = 1;
/// Record kind: one signed study note (KNOTE1 encoding).
const KIND_NOTE: u8 = 2;

/// A scripture library with a durable append-only log behind it.
///
/// Open replays the log into a [`Library`]; mutate the library
/// through [`ScriptureStore::library_mut`], then [`ScriptureStore::save`]
/// to make the new verses and notes durable. Only what is not already
/// in the log is appended.
pub struct ScriptureStore {
    library: Library,
    file: File,
    /// Full path of the log file.
    path: PathBuf,
    /// Verse keys already written to the log.
    persisted_verses: HashSet<VerseKey>,
    /// Notes already written to the log, per verse (a prefix count:
    /// notes are append-only, so the durable notes for a verse are
    /// always the first `n` of its list).
    persisted_notes: HashMap<VerseKey, usize>,
}

impl ScriptureStore {
    /// Open the store in `dir`, creating the directory and the log
    /// file if they do not exist. Replays the log to rebuild the
    /// library; a torn or corrupt tail is cut, a corrupt first
    /// record is refused with an error.
    pub fn open(dir: &Path) -> Result<Self, ScriptureError> {
        create_dir_all(dir)?;
        let path = dir.join(LOG_FILE_NAME);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        let (library, persisted_verses, persisted_notes) = replay(&mut file)?;
        Ok(Self {
            library,
            file,
            path,
            persisted_verses,
            persisted_notes,
        })
    }

    /// Load the library stored in `dir`, without keeping the store
    /// open. This is [`ScriptureStore::open`] followed by
    /// [`ScriptureStore::into_library`].
    pub fn load(dir: &Path) -> Result<Library, ScriptureError> {
        Ok(Self::open(dir)?.into_library())
    }

    /// The library, rebuilt from the log.
    pub fn library(&self) -> &Library {
        &self.library
    }

    /// The library, mutably. Ingest verses and attach notes here,
    /// then call [`ScriptureStore::save`] to make them durable.
    pub fn library_mut(&mut self) -> &mut Library {
        &mut self.library
    }

    /// Path of the log file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Take the library out of the store, dropping the log handle.
    pub fn into_library(self) -> Library {
        self.library
    }

    /// Append every verse and note that is not durable yet, then
    /// fsync. A save that returns has its records on disk. A save
    /// with nothing new to write touches nothing and returns
    /// immediately.
    pub fn save(&mut self) -> Result<(), ScriptureError> {
        let mut buf: Vec<u8> = Vec::new();
        let mut new_verses: Vec<VerseKey> = Vec::new();
        let mut new_notes: Vec<(VerseKey, usize)> = Vec::new();

        // Verses first, in canonical order: a note's verse is always
        // already in the log before the note that cites it.
        for key in self.library.ordered_keys() {
            if self.persisted_verses.contains(&key) {
                continue;
            }
            let verse = self
                .library
                .verse_for_key(&key)
                .expect("indexed key has a verse");
            let verse_no = verse.reference.verse.unwrap_or((0, 0)).0;
            buf.extend_from_slice(&frame_record(
                KIND_VERSE,
                &encode_verse_payload(
                    &verse.reference.book,
                    verse.reference.chapter,
                    verse_no,
                    &verse.text,
                ),
            ));
            new_verses.push(key);
        }
        // Notes next, in canonical verse order, insertion order
        // within a verse.
        for key in self.library.ordered_keys() {
            let persisted = self.persisted_notes.get(&key).copied().unwrap_or(0);
            let notes = self.library.notes_for_key(&key);
            if notes.len() > persisted {
                let mut count = 0;
                for note in &notes[persisted..] {
                    buf.extend_from_slice(&frame_record(KIND_NOTE, &note.encode()));
                    count += 1;
                }
                new_notes.push((key, count));
            }
        }

        if buf.is_empty() {
            return Ok(());
        }
        self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(&buf)?;
        self.file.sync_all()?;
        // Only now, with the batch durable, is the state "persisted".
        // A failed write leaves the tracking untouched, so the next
        // save retries the same records — and replay skips duplicates.
        for key in new_verses {
            self.persisted_verses.insert(key);
        }
        for (key, count) in new_notes {
            *self.persisted_notes.entry(key).or_insert(0) += count;
        }
        Ok(())
    }
}

/// Frame one record: magic + kind + length + payload + hash.
fn frame_record(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(10 + payload.len() + 32);
    out.extend_from_slice(RECORD_MAGIC);
    out.push(kind);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    let mut hashed = vec![kind];
    hashed.extend_from_slice(payload);
    out.extend_from_slice(&sha256(&hashed));
    out
}

/// Encode a verse record payload: book + chapter + verse + text,
/// all length-prefixed where variable-length.
fn encode_verse_payload(book: &str, chapter: u32, verse: u32, text: &str) -> Vec<u8> {
    let mut out = Vec::new();
    push_len_bytes(&mut out, book.as_bytes());
    out.extend_from_slice(&chapter.to_le_bytes());
    out.extend_from_slice(&verse.to_le_bytes());
    push_len_bytes(&mut out, text.as_bytes());
    out
}

/// Append a u32-LE length prefix plus the bytes.
fn push_len_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// Replay the log, rebuilding the library and the persisted-state
/// tracking. Stops at the first truncated or corrupt record and
/// truncates the file to the last good record end, so the log is
/// always appendable. A corrupt *first* record is an error, not a
/// cut: the file is not a scripture log at all.
fn replay(
    file: &mut File,
) -> Result<
    (
        Library,
        HashSet<VerseKey>,
        HashMap<VerseKey, usize>,
    ),
    ScriptureError,
> {
    let mut lib = Library::new();
    let mut persisted_verses = HashSet::new();
    let mut persisted_notes: HashMap<VerseKey, usize> = HashMap::new();
    let mut reader = BufReader::new(file.try_clone()?);
    let mut offset: u64 = 0;
    loop {
        let record_start = offset;
        let first = record_start == 0;
        let mut magic = [0u8; 5];
        match reader.read_exact(&mut magic) {
            Ok(()) => {}
            // A short read at a record boundary is a torn write:
            // stop, keep everything before it.
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        if &magic != RECORD_MAGIC {
            if first {
                return Err(ScriptureError::Persistence(
                    "not a scripture log (bad magic)".to_string(),
                ));
            }
            break;
        }
        let mut kind_buf = [0u8; 1];
        let mut len_buf = [0u8; 4];
        if reader.read_exact(&mut kind_buf).is_err()
            || reader.read_exact(&mut len_buf).is_err()
        {
            break;
        }
        let kind = kind_buf[0];
        if kind != KIND_VERSE && kind != KIND_NOTE {
            if first {
                return Err(ScriptureError::Persistence(format!(
                    "unknown record kind {kind} in scripture log"
                )));
            }
            break;
        }
        let len = u32::from_le_bytes(len_buf);
        if len > MAX_PAYLOAD {
            if first {
                return Err(ScriptureError::Persistence(
                    "record length exceeds the 64 MiB limit".to_string(),
                ));
            }
            break;
        }
        let mut payload = vec![0u8; len as usize];
        if reader.read_exact(&mut payload).is_err() {
            break;
        }
        let mut hash = [0u8; 32];
        if reader.read_exact(&mut hash).is_err() {
            break;
        }
        let mut hashed = vec![kind];
        hashed.extend_from_slice(&payload);
        if sha256(&hashed) != hash {
            if first {
                return Err(ScriptureError::Persistence(
                    "first record failed hash verification".to_string(),
                ));
            }
            break;
        }
        // The framing is sound; now decode and apply. A failure
        // here is a corrupt record, handled like a bad hash.
        let key = match kind {
            KIND_VERSE => apply_verse(&mut lib, &payload),
            KIND_NOTE => apply_note(&mut lib, &payload),
            _ => unreachable!("kind checked above"),
        };
        let key = match key {
            Ok(key) => key,
            Err(e) => {
                if first {
                    return Err(e);
                }
                break;
            }
        };
        match kind {
            KIND_VERSE => {
                persisted_verses.insert(key);
            }
            KIND_NOTE => {
                *persisted_notes.entry(key).or_insert(0) += 1;
            }
            _ => unreachable!("kind checked above"),
        }
        offset = record_start + 10 + len as u64 + 32;
    }
    // Cut any torn or corrupt tail so later appends have no hole.
    let file_len = reader.seek(SeekFrom::End(0))?;
    if file_len != offset {
        file.set_len(offset)?;
    }
    file.seek(SeekFrom::End(0))?;
    Ok((lib, persisted_verses, persisted_notes))
}

/// Decode a verse payload and restore the verse. Returns the verse
/// key. A verse already present with the same text is a no-op
/// (idempotent replay); anything else malformed is an error.
fn apply_verse(lib: &mut Library, payload: &[u8]) -> Result<VerseKey, ScriptureError> {
    let mut cursor = Cursor::new(payload);
    let book = cursor.str_bytes()?;
    let chapter = cursor.u32()?;
    let verse = cursor.u32()?;
    let text = cursor.str_bytes()?;
    cursor.done()?;
    lib.restore_verse(&book, chapter, verse, &text)?;
    Ok(VerseKey::new(&book, chapter, verse))
}

/// Decode a note payload and attach the note, verifying its
/// signature. Returns the verse key the note attaches to. A note
/// already attached (identical encoding) is a no-op.
fn apply_note(lib: &mut Library, payload: &[u8]) -> Result<VerseKey, ScriptureError> {
    let note = StudyNote::decode(payload).map_err(|e| match e {
        ScriptureError::BadNote(msg) => {
            ScriptureError::Persistence(format!("bad note record: {msg}"))
        }
        other => other,
    })?;
    let key = VerseKey::from_reference(&note.reference).ok_or_else(|| {
        ScriptureError::Persistence("note record is not for a single verse".to_string())
    })?;
    if lib
        .notes_for_key(&key)
        .iter()
        .any(|n| n.encode() == payload)
    {
        return Ok(key);
    }
    lib.add_note(note)?;
    Ok(key)
}

/// A bounds-checked cursor over a record payload.
struct Cursor<'a> {
    bytes: &'a [u8],
    off: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, off: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], ScriptureError> {
        if self.bytes.len() - self.off < n {
            return Err(ScriptureError::Persistence(
                "record payload is truncated".to_string(),
            ));
        }
        let slice = &self.bytes[self.off..self.off + n];
        self.off += n;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, ScriptureError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn len_bytes(&mut self) -> Result<&'a [u8], ScriptureError> {
        let len = self.u32()? as usize;
        self.take(len)
    }

    fn str_bytes(&mut self) -> Result<String, ScriptureError> {
        let bytes = self.len_bytes()?;
        String::from_utf8(bytes.to_vec()).map_err(|_| {
            ScriptureError::Persistence("record text is not UTF-8".to_string())
        })
    }

    fn done(&self) -> Result<(), ScriptureError> {
        if self.off != self.bytes.len() {
            return Err(ScriptureError::Persistence(
                "trailing bytes in record payload".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::Reference;
    use kernel_identity::Identity;
    use std::fs;

    /// The same small text as `library::tests::sample_text` (that
    /// helper is private to its module).
    fn sample_text() -> &'static str {
        "John 3:16\tFor God so loved the world, that he gave his only Son.\n\
         John 3:17\tFor God did not send his Son into the world to condemn the world.\n\
         Psalm 23:1\tThe Lord is my shepherd; I shall not want.\n\
         Psalm 23:2\tHe makes me lie down in green pastures.\n"
    }

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kernel-scripture-persist-{}-{}",
            std::process::id(),
            name
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn sample_library_with_note() -> Library {
        let mut lib = Library::new();
        lib.ingest(sample_text()).unwrap();
        let id = Identity::generate("tristan").unwrap();
        let note = StudyNote::new(
            &id,
            &Reference::parse("John 3:16").unwrap(),
            "Love here is given, not earned.",
            1_700_000_000,
        )
        .unwrap();
        lib.add_note(note).unwrap();
        lib
    }

    fn log_bytes(dir: &Path) -> Vec<u8> {
        fs::read(dir.join(LOG_FILE_NAME)).unwrap()
    }

    /// Length of the first framed record in `bytes`, by its own
    /// header. Used to cut or flip inside a later record.
    fn first_record_len(bytes: &[u8]) -> usize {
        let len = u32::from_le_bytes(bytes[6..10].try_into().unwrap()) as usize;
        10 + len + 32
    }

    #[test]
    fn round_trip_preserves_library() {
        let dir = test_dir("round-trip");
        {
            let mut store = ScriptureStore::open(&dir).unwrap();
            *store.library_mut() = sample_library_with_note();
            store.save().unwrap();
        }
        let lib = ScriptureStore::load(&dir).unwrap();
        assert_eq!(lib.verse_count(), 4);
        assert_eq!(lib.block_count(), 4);
        assert_eq!(lib.book_names(), ["John", "Psalm"]);
        let v = lib.verse("John", 3, 16).unwrap();
        assert_eq!(
            v.text,
            "For God so loved the world, that he gave his only Son."
        );
        assert_eq!(lib.search_word("world", &crate::library::Scope::All).len(), 2);
        // The concordance rebuilt from the replayed verses matches
        // the original exactly.
        let expected = sample_library_with_note();
        assert_eq!(
            lib.concordance().word_count(),
            expected.concordance().word_count()
        );
        assert_eq!(lib.note_count(), 1);
        let notes = lib.notes_for("John", 3, 16);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].text, "Love here is given, not earned.");
        assert_eq!(notes[0].author.display_name(), "tristan");
        assert!(notes[0].verify());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_store_loads_cleanly() {
        let dir = test_dir("empty");
        let lib = ScriptureStore::load(&dir).unwrap();
        assert_eq!(lib.verse_count(), 0);
        assert_eq!(lib.note_count(), 0);
        // Saving an empty library writes nothing and still loads.
        {
            let mut store = ScriptureStore::open(&dir).unwrap();
            store.save().unwrap();
        }
        assert_eq!(log_bytes(&dir).len(), 0);
        assert_eq!(ScriptureStore::load(&dir).unwrap().verse_count(), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_is_idempotent() {
        let dir = test_dir("idempotent");
        {
            let mut store = ScriptureStore::open(&dir).unwrap();
            *store.library_mut() = sample_library_with_note();
            store.save().unwrap();
        }
        let size_after_first = log_bytes(&dir).len();
        {
            let mut store = ScriptureStore::open(&dir).unwrap();
            store.save().unwrap();
        }
        assert_eq!(log_bytes(&dir).len(), size_after_first);
        assert_eq!(ScriptureStore::load(&dir).unwrap().verse_count(), 4);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn incremental_save_appends() {
        let dir = test_dir("incremental");
        {
            let mut store = ScriptureStore::open(&dir).unwrap();
            store.library_mut().ingest(sample_text()).unwrap();
            store.save().unwrap();
        }
        let size_after_first = log_bytes(&dir).len();
        {
            let mut store = ScriptureStore::open(&dir).unwrap();
            assert_eq!(store.library().verse_count(), 4);
            store
                .library_mut()
                .ingest("Exodus 20:1\tAnd God spoke all these words.\n")
                .unwrap();
            let id = Identity::generate("tristan").unwrap();
            let note = StudyNote::new(
                &id,
                &Reference::parse("Exodus 20:1").unwrap(),
                "The words begin.",
                1_700_000_001,
            )
            .unwrap();
            store.library_mut().add_note(note).unwrap();
            store.save().unwrap();
        }
        // The second save appended, it did not rewrite from scratch.
        assert!(log_bytes(&dir).len() > size_after_first);
        let lib = ScriptureStore::load(&dir).unwrap();
        assert_eq!(lib.verse_count(), 5);
        assert_eq!(lib.book_names(), ["John", "Psalm", "Exodus"]);
        assert_eq!(lib.note_count(), 1);
        assert_eq!(
            lib.verse("Exodus", 20, 1).unwrap().text,
            "And God spoke all these words."
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn flipped_byte_in_first_record_is_refused() {
        let dir = test_dir("flip-first");
        {
            let mut store = ScriptureStore::open(&dir).unwrap();
            *store.library_mut() = sample_library_with_note();
            store.save().unwrap();
        }
        let mut bytes = log_bytes(&dir);
        bytes[12] ^= 0xff; // inside the first record's payload
        fs::write(dir.join(LOG_FILE_NAME), &bytes).unwrap();
        assert!(ScriptureStore::load(&dir).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn flipped_byte_in_later_record_cuts_tail() {
        let dir = test_dir("flip-later");
        {
            let mut store = ScriptureStore::open(&dir).unwrap();
            *store.library_mut() = sample_library_with_note();
            store.save().unwrap();
        }
        let mut bytes = log_bytes(&dir);
        // Flip a byte inside the second record's payload.
        let second = first_record_len(&bytes);
        bytes[second + 12] ^= 0xff;
        fs::write(dir.join(LOG_FILE_NAME), &bytes).unwrap();
        let lib = ScriptureStore::load(&dir).unwrap();
        // The first verse survived; everything from the bad record
        // on was cut.
        assert_eq!(lib.verse_count(), 1);
        assert!(lib.verse("John", 3, 16).is_some());
        assert_eq!(lib.note_count(), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncated_file_cuts_torn_tail() {
        let dir = test_dir("truncated");
        {
            let mut store = ScriptureStore::open(&dir).unwrap();
            *store.library_mut() = sample_library_with_note();
            store.save().unwrap();
        }
        let bytes = log_bytes(&dir);
        let first_len = first_record_len(&bytes);
        // Cut the file in the middle of the second record.
        let cut = first_len + 5;
        fs::write(dir.join(LOG_FILE_NAME), &bytes[..cut]).unwrap();
        let lib = ScriptureStore::load(&dir).unwrap();
        assert_eq!(lib.verse_count(), 1);
        assert!(lib.verse("John", 3, 16).is_some());
        // The store is usable after repair: new saves append cleanly.
        let mut store = ScriptureStore::open(&dir).unwrap();
        store
            .library_mut()
            .ingest("Exodus 20:1\tAnd God spoke all these words.\n")
            .unwrap();
        store.save().unwrap();
        let lib = ScriptureStore::load(&dir).unwrap();
        assert_eq!(lib.verse_count(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn garbage_file_is_refused() {
        let dir = test_dir("garbage");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(LOG_FILE_NAME), b"this is not a scripture log").unwrap();
        assert!(ScriptureStore::load(&dir).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn tampered_note_signature_is_refused_on_load() {
        let dir = test_dir("bad-note-sig");
        {
            let mut store = ScriptureStore::open(&dir).unwrap();
            *store.library_mut() = sample_library_with_note();
            store.save().unwrap();
        }
        // Append a hand-framed note record whose framing hash is
        // correct but whose signature does not verify: the note text
        // was edited after signing.
        let id = Identity::generate("tristan").unwrap();
        let mut note = StudyNote::new(
            &id,
            &Reference::parse("John 3:16").unwrap(),
            "A genuine note.",
            1_700_000_002,
        )
        .unwrap();
        note.text = "Forged after signing.".to_string();
        let framed = frame_record(KIND_NOTE, &note.encode());
        {
            let mut f = OpenOptions::new()
                .append(true)
                .open(dir.join(LOG_FILE_NAME))
                .unwrap();
            f.write_all(&framed).unwrap();
            f.sync_all().unwrap();
        }
        let lib = ScriptureStore::load(&dir).unwrap();
        // The library loaded; the forged note did not.
        assert_eq!(lib.verse_count(), 4);
        assert_eq!(lib.note_count(), 1);
        assert_eq!(
            lib.notes_for("John", 3, 16)[0].text,
            "Love here is given, not earned."
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn duplicate_record_is_skipped() {
        let dir = test_dir("duplicate");
        {
            let mut store = ScriptureStore::open(&dir).unwrap();
            store.library_mut().ingest(sample_text()).unwrap();
            store.save().unwrap();
        }
        // Hand-append a byte-identical copy of the first verse
        // record, as a retried save would after an interrupted write.
        let bytes = log_bytes(&dir);
        let first_len = first_record_len(&bytes);
        let duplicate = bytes[..first_len].to_vec();
        {
            let mut f = OpenOptions::new()
                .append(true)
                .open(dir.join(LOG_FILE_NAME))
                .unwrap();
            f.write_all(&duplicate).unwrap();
            f.sync_all().unwrap();
        }
        let lib = ScriptureStore::load(&dir).unwrap();
        assert_eq!(lib.verse_count(), 4);
        assert_eq!(lib.block_count(), 4);
        let _ = fs::remove_dir_all(&dir);
    }
}
