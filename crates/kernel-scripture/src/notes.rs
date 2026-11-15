//! Signed study notes attached to a verse.
//!
//! A note is an author, a single verse, text, and a caller-supplied
//! timestamp, signed with the author's Ed25519 key. The signature
//! covers the author key, the author's display name, the verse, the
//! timestamp, and the text, under the domain separator `KNOTE1`.
//! A note whose signature does not verify is refused on add and
//! filtered out on read — a tampered note never reaches a reader.
//!
//! Binary encoding (explicit, no JSON):
//! `KNOTE1` + author public key (32) + name length (u32 LE) + name
//! bytes + book length (u32 LE) + book bytes + chapter (u32 LE) +
//! verse (u32 LE) + timestamp (u64 LE) + text length (u32 LE) + text
//! bytes + signature (64). Decoding is strict: bad magic, truncation,
//! or trailing bytes are all errors.

use kernel_crypto::{Ed25519PublicKey, Ed25519Signature};
use kernel_identity::{Identity, PeerId};

use crate::error::ScriptureError;
use crate::reference::Reference;

/// Domain separator for note signatures and the encoding magic.
const NOTE_DOMAIN: &[u8; 6] = b"KNOTE1";

/// A signed study note attached to one verse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StudyNote {
    /// The author. Only the public half is stored.
    pub author: PeerId,
    /// The verse this note is about (always a single verse).
    pub reference: Reference,
    /// The note text.
    pub text: String,
    /// Caller-supplied timestamp (seconds since the Unix epoch).
    /// Notes carry no clock of their own.
    pub timestamp: u64,
    /// The author's signature over author, verse, timestamp, and text.
    pub signature: Ed25519Signature,
}

impl StudyNote {
    /// Create and sign a note. The reference must be a single verse;
    /// a range or whole-chapter reference is refused.
    pub fn new(
        identity: &Identity,
        reference: &Reference,
        text: &str,
        timestamp: u64,
    ) -> Result<Self, ScriptureError> {
        if !reference.is_single_verse() {
            return Err(ScriptureError::BadReference(format!(
                "a study note attaches to one verse, not '{reference}'"
            )));
        }
        let author = identity.peer_id().clone();
        let mut note = Self {
            author,
            reference: reference.clone(),
            text: text.to_string(),
            timestamp,
            // Placeholder; replaced by the real signature below.
            signature: Ed25519Signature::from_bytes(&[0u8; 64]),
        };
        let signature = identity.sign(&note.signing_bytes());
        note.signature = signature;
        Ok(note)
    }

    /// True if the signature verifies against the author's key for
    /// this note's contents.
    pub fn verify(&self) -> bool {
        self.author
            .verify(&self.signing_bytes(), &self.signature)
            .is_ok()
    }

    /// The bytes the signature covers: domain + author key + author
    /// name + book + chapter + verse + timestamp + text, all
    /// length-prefixed where variable-length.
    fn signing_bytes(&self) -> Vec<u8> {
        let verse = self.reference.verse.unwrap_or((0, 0)).0;
        let mut out = Vec::new();
        out.extend_from_slice(NOTE_DOMAIN);
        out.extend_from_slice(self.author.public_key().as_bytes());
        push_len_bytes(&mut out, self.author.display_name().as_bytes());
        push_len_bytes(&mut out, self.reference.book.as_bytes());
        out.extend_from_slice(&self.reference.chapter.to_le_bytes());
        out.extend_from_slice(&verse.to_le_bytes());
        out.extend_from_slice(&self.timestamp.to_le_bytes());
        push_len_bytes(&mut out, self.text.as_bytes());
        out
    }

    /// Encode the note to bytes. `decode(encode(note)) == note`.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = self.signing_bytes();
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode bytes produced by [`StudyNote::encode`]. Strict: bad
    /// magic, truncation, trailing bytes, an invalid public key, or
    /// an invalid display name are all errors.
    pub fn decode<'a>(bytes: &'a [u8]) -> Result<Self, ScriptureError> {
        let mut off = 0usize;
        let take =
            |bytes: &'a [u8], off: &mut usize, n: usize| -> Result<&'a [u8], ScriptureError> {
                if bytes.len() - *off < n {
                    return Err(ScriptureError::BadNote("note is truncated".to_string()));
                }
                let slice = &bytes[*off..*off + n];
                *off += n;
                Ok(slice)
            };
        let magic = take(bytes, &mut off, 6)?;
        if magic != NOTE_DOMAIN {
            return Err(ScriptureError::BadNote("bad KNOTE1 magic".to_string()));
        }
        let key_bytes: [u8; 32] = take(bytes, &mut off, 32)?.try_into().unwrap();
        let public_key = Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| ScriptureError::BadNote(format!("invalid author public key: {e}")))?;
        let name = read_len_bytes(bytes, &mut off)?;
        let name = String::from_utf8(name.to_vec())
            .map_err(|_| ScriptureError::BadNote("author name is not UTF-8".to_string()))?;
        let book = read_len_bytes(bytes, &mut off)?;
        let book = String::from_utf8(book.to_vec())
            .map_err(|_| ScriptureError::BadNote("book name is not UTF-8".to_string()))?;
        let chapter = u32::from_le_bytes(take(bytes, &mut off, 4)?.try_into().unwrap());
        let verse = u32::from_le_bytes(take(bytes, &mut off, 4)?.try_into().unwrap());
        let timestamp = u64::from_le_bytes(take(bytes, &mut off, 8)?.try_into().unwrap());
        let text = read_len_bytes(bytes, &mut off)?;
        let text = String::from_utf8(text.to_vec())
            .map_err(|_| ScriptureError::BadNote("note text is not UTF-8".to_string()))?;
        let sig_bytes: [u8; 64] = take(bytes, &mut off, 64)?.try_into().unwrap();
        if off != bytes.len() {
            return Err(ScriptureError::BadNote(
                "trailing bytes after note".to_string(),
            ));
        }
        let author = PeerId::new(public_key, &name)
            .map_err(|e| ScriptureError::BadNote(format!("invalid author identity: {e}")))?;
        let reference = Reference::verse(&book, chapter, verse)?;
        Ok(Self {
            author,
            reference,
            text,
            timestamp,
            signature: Ed25519Signature::from_bytes(&sig_bytes),
        })
    }
}

/// Append a u32-LE length prefix plus the bytes.
fn push_len_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// Read a u32-LE length-prefixed byte string. The length must fit in
/// the remaining input.
fn read_len_bytes<'a>(bytes: &'a [u8], off: &mut usize) -> Result<&'a [u8], ScriptureError> {
    if bytes.len() - *off < 4 {
        return Err(ScriptureError::BadNote("note is truncated".to_string()));
    }
    let len = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap()) as usize;
    *off += 4;
    if bytes.len() - *off < len {
        return Err(ScriptureError::BadNote("note is truncated".to_string()));
    }
    let slice = &bytes[*off..*off + len];
    *off += len;
    Ok(slice)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn author() -> Identity {
        Identity::generate("tristan").unwrap()
    }

    fn note() -> StudyNote {
        StudyNote::new(
            &author(),
            &Reference::parse("John 3:16").unwrap(),
            "God's love is for the world, not a few.",
            1_700_000_000,
        )
        .unwrap()
    }

    #[test]
    fn note_signs_and_verifies() {
        let n = note();
        assert!(n.verify());
        assert_eq!(n.author.display_name(), "tristan");
        assert_eq!(n.reference.to_string(), "John 3:16");
    }

    #[test]
    fn note_round_trip() {
        let n = note();
        let decoded = StudyNote::decode(&n.encode()).unwrap();
        assert_eq!(decoded, n);
        assert!(decoded.verify());
    }

    #[test]
    fn tampered_note_fails_verification() {
        let mut n = note();
        n.text.push_str(" (edited)");
        assert!(!n.verify());
        // Tampering with the bytes and re-decoding keeps a valid
        // structure but the signature no longer matches.
        let mut bytes = n.encode();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff; // flip a bit inside the signature
        let decoded = StudyNote::decode(&bytes).unwrap();
        assert!(!decoded.verify());
    }

    #[test]
    fn note_requires_a_single_verse() {
        let id = author();
        assert!(StudyNote::new(&id, &Reference::parse("John 3:16-18").unwrap(), "x", 0).is_err());
        assert!(StudyNote::new(&id, &Reference::parse("Psalm 23").unwrap(), "x", 0).is_err());
    }

    #[test]
    fn decode_rejects_garbage() {
        assert!(StudyNote::decode(b"").is_err());
        assert!(StudyNote::decode(b"KNOTE1garbage").is_err());
        let mut bytes = note().encode();
        bytes.push(0); // trailing byte
        assert!(StudyNote::decode(&bytes).is_err());
        assert!(StudyNote::decode(&bytes[..bytes.len() - 10]).is_err());
    }
}
