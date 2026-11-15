//! Editable playlists.
//!
//! A [`PlaylistDoc`] is the working form of a playlist: a
//! human-readable id, a title, and an ordered list of track manifest
//! CIDs, with add / remove / reorder. It is distinct from the signed
//! [`crate::Playlist`] record, which is the published, immutable form
//! — a doc is what you edit before you publish.
//!
//! A doc saves to an explicit binary format (no JSON) and loads back
//! exactly: `KPLST1` + id length (u16 LE) + id bytes + title length
//! (u16 LE) + title bytes + track count (u32 LE) + track CIDs (32 each)
//! + SHA-256 checksum of everything before it. Loading refuses bad
//! magic, truncated or trailing bytes, bad names, an over-large track
//! count, and any checksum mismatch — a flipped byte anywhere is a
//! clean error, never a half-loaded playlist.

use kernel_crypto::sha256;
use kernel_storage::Cid;

use crate::{validate_human_name, TrackError, TrackStore, MAX_PLAYLIST_TRACKS};

/// Magic bytes at the start of every saved playlist document.
const DOC_MAGIC: &[u8; 6] = b"KPLST1";

/// An editable playlist: id, title, and track CIDs in play order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaylistDoc {
    /// Human-readable playlist id (unique within a library).
    pub id: String,
    /// Human-readable playlist title.
    pub title: String,
    /// Track manifest CIDs, in play order. Duplicates are allowed —
    /// the same track may appear twice in a playlist.
    pub tracks: Vec<Cid>,
}

impl PlaylistDoc {
    /// A new, empty playlist document. Id and title follow the human
    /// naming rules (1-128 printable ASCII, no path separators).
    pub fn new(id: &str, title: &str) -> Result<Self, TrackError> {
        validate_human_name(id, "playlist")?;
        validate_human_name(title, "playlist")?;
        Ok(Self {
            id: id.to_string(),
            title: title.to_string(),
            tracks: Vec::new(),
        })
    }

    /// Human-readable playlist id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Playlist title.
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Track manifest CIDs, in play order.
    pub fn tracks(&self) -> &[Cid] {
        &self.tracks
    }

    /// Number of tracks.
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// True if the playlist holds no tracks.
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Append a track. The track must be one the `store` holds, or
    /// the add is refused with [`TrackError::NotFound`]. The playlist
    /// cap ([`MAX_PLAYLIST_TRACKS`]) is enforced.
    pub fn add(&mut self, cid: Cid, store: &TrackStore) -> Result<(), TrackError> {
        if store.get_manifest(&cid).is_none() {
            return Err(TrackError::NotFound(cid));
        }
        if self.tracks.len() >= MAX_PLAYLIST_TRACKS {
            return Err(TrackError::InvalidManifest(format!(
                "playlist must have at most {MAX_PLAYLIST_TRACKS} tracks"
            )));
        }
        self.tracks.push(cid);
        Ok(())
    }

    /// Remove the track at `index`, returning its CID. An out-of-range
    /// index is a clean error.
    pub fn remove(&mut self, index: usize) -> Result<Cid, TrackError> {
        if index >= self.tracks.len() {
            return Err(TrackError::InvalidManifest(format!(
                "track index {index} out of range for a playlist of {} tracks",
                self.tracks.len()
            )));
        }
        Ok(self.tracks.remove(index))
    }

    /// Move the track at `from` to position `to`, shifting the tracks
    /// between them. Both indices must be in range.
    pub fn move_track(&mut self, from: usize, to: usize) -> Result<(), TrackError> {
        let n = self.tracks.len();
        if from >= n || to >= n {
            return Err(TrackError::InvalidManifest(format!(
                "move {from} -> {to} out of range for a playlist of {n} tracks"
            )));
        }
        if from == to {
            return Ok(());
        }
        let cid = self.tracks.remove(from);
        self.tracks.insert(to, cid);
        Ok(())
    }

    /// Encode to the explicit binary document format.
    pub fn save(&self) -> Vec<u8> {
        let id_bytes = self.id.as_bytes();
        let title_bytes = self.title.as_bytes();
        let mut out = Vec::with_capacity(
            6 + 2 + id_bytes.len() + 2 + title_bytes.len() + 4 + 32 * self.tracks.len() + 32,
        );
        out.extend_from_slice(DOC_MAGIC);
        out.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(id_bytes);
        out.extend_from_slice(&(title_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(title_bytes);
        out.extend_from_slice(&(self.tracks.len() as u32).to_le_bytes());
        for cid in &self.tracks {
            out.extend_from_slice(cid.as_bytes());
        }
        let checksum = sha256(&out);
        out.extend_from_slice(&checksum);
        out
    }

    /// Decode a document produced by [`PlaylistDoc::save`]. Refuses
    /// anything that is not exactly that format, including a single
    /// flipped byte (the checksum catches it).
    pub fn load(bytes: &[u8]) -> Result<Self, TrackError> {
        let bad = |msg: String| TrackError::InvalidManifest(msg);
        if bytes.len() < 6 + 2 + 2 + 4 + 32 || &bytes[..6] != DOC_MAGIC {
            return Err(bad("missing KPLST1 magic".to_string()));
        }
        // The last 32 bytes are the checksum over everything before.
        let (body, checksum) = bytes.split_at(bytes.len() - 32);
        if sha256(body) != checksum {
            return Err(bad("playlist checksum mismatch".to_string()));
        }
        let id_len = u16::from_le_bytes(body[6..8].try_into().expect("2-byte slice")) as usize;
        let title_off = 8 + id_len;
        if body.len() < title_off + 2 {
            return Err(bad("playlist too short for title length".to_string()));
        }
        let title_len = u16::from_le_bytes(
            body[title_off..title_off + 2]
                .try_into()
                .expect("2-byte slice"),
        ) as usize;
        let count_off = title_off + 2 + title_len;
        if body.len() < count_off + 4 {
            return Err(bad("playlist too short for track count".to_string()));
        }
        let count = u32::from_le_bytes(
            body[count_off..count_off + 4]
                .try_into()
                .expect("4-byte slice"),
        ) as usize;
        if count > MAX_PLAYLIST_TRACKS {
            return Err(bad(format!(
                "playlist claims {count} tracks, max is {MAX_PLAYLIST_TRACKS}"
            )));
        }
        let expected = count_off + 4 + 32 * count;
        if body.len() != expected {
            return Err(bad(format!(
                "playlist body is {} bytes, expected {expected}",
                body.len()
            )));
        }
        let id = String::from_utf8(body[8..8 + id_len].to_vec())
            .map_err(|_| bad("playlist id is not valid UTF-8".to_string()))?;
        validate_human_name(&id, "playlist")?;
        let title_start = title_off + 2;
        let title = String::from_utf8(body[title_start..title_start + title_len].to_vec())
            .map_err(|_| bad("playlist title is not valid UTF-8".to_string()))?;
        validate_human_name(&title, "playlist")?;
        let mut off = count_off + 4;
        let mut tracks = Vec::with_capacity(count);
        for _ in 0..count {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&body[off..off + 32]);
            tracks.push(Cid::from_bytes(arr));
            off += 32;
        }
        Ok(Self { id, title, tracks })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_identity::Identity;

    fn store_with_tracks(n: usize) -> (TrackStore, Identity, Vec<Cid>) {
        let alice = Identity::generate("alice").unwrap();
        let mut store = TrackStore::new();
        let mut cids = Vec::new();
        for i in 0..n {
            let cid = store
                .put_track(
                    &format!("Track {i}"),
                    "Nova",
                    60,
                    format!("audio {i}").as_bytes(),
                    &alice,
                )
                .unwrap();
            cids.push(cid);
        }
        (store, alice, cids)
    }

    #[test]
    fn crud_add_remove_reorder() {
        let (store, _alice, cids) = store_with_tracks(3);
        let mut doc = PlaylistDoc::new("mix-001", "Morning Mix").unwrap();
        assert!(doc.is_empty());
        doc.add(cids[0], &store).unwrap();
        doc.add(cids[1], &store).unwrap();
        doc.add(cids[2], &store).unwrap();
        assert_eq!(doc.tracks(), &[cids[0], cids[1], cids[2]]);
        // Duplicates are allowed.
        doc.add(cids[0], &store).unwrap();
        assert_eq!(doc.len(), 4);
        // Reorder: move the last track to the front.
        doc.move_track(3, 0).unwrap();
        assert_eq!(doc.tracks(), &[cids[0], cids[0], cids[1], cids[2]]);
        // Remove the middle duplicate.
        assert_eq!(doc.remove(1).unwrap(), cids[0]);
        assert_eq!(doc.tracks(), &[cids[0], cids[1], cids[2]]);
        // Out-of-range operations are clean errors.
        assert!(doc.remove(3).is_err());
        assert!(doc.move_track(0, 3).is_err());
        assert!(doc.move_track(3, 0).is_err());
        // Moving a track onto itself is a no-op.
        doc.move_track(1, 1).unwrap();
        assert_eq!(doc.tracks(), &[cids[0], cids[1], cids[2]]);
    }

    #[test]
    fn add_refuses_unknown_track() {
        let (store, _alice, _cids) = store_with_tracks(1);
        let mut doc = PlaylistDoc::new("mix-002", "Empty Mix").unwrap();
        let ghost = Cid::from_bytes(kernel_crypto::sha256(b"no such track"));
        assert_eq!(doc.add(ghost, &store), Err(TrackError::NotFound(ghost)));
        assert!(doc.is_empty());
    }

    #[test]
    fn save_load_round_trip_is_exact() {
        let (store, _alice, cids) = store_with_tracks(3);
        let mut doc = PlaylistDoc::new("mix-003", "Road Trip").unwrap();
        doc.add(cids[2], &store).unwrap();
        doc.add(cids[0], &store).unwrap();
        doc.add(cids[0], &store).unwrap();
        let bytes = doc.save();
        let loaded = PlaylistDoc::load(&bytes).unwrap();
        assert_eq!(loaded, doc);
        // Saving the loaded doc reproduces the exact same bytes.
        assert_eq!(loaded.save(), bytes);
        // An empty playlist round-trips too.
        let empty = PlaylistDoc::new("empty-001", "Nothing Yet").unwrap();
        assert_eq!(PlaylistDoc::load(&empty.save()).unwrap(), empty);
    }

    #[test]
    fn corrupted_saves_are_refused() {
        let (store, _alice, cids) = store_with_tracks(2);
        let mut doc = PlaylistDoc::new("mix-004", "Fragile").unwrap();
        doc.add(cids[0], &store).unwrap();
        doc.add(cids[1], &store).unwrap();
        let bytes = doc.save();
        // A flipped byte anywhere breaks the checksum.
        for flip_at in [0, 6, 10, bytes.len() / 2, bytes.len() - 1] {
            let mut bad = bytes.clone();
            bad[flip_at] ^= 0xff;
            assert!(
                PlaylistDoc::load(&bad).is_err(),
                "flipped byte at {flip_at} was accepted"
            );
        }
        // Truncation and trailing bytes are refused.
        assert!(PlaylistDoc::load(&bytes[..bytes.len() - 1]).is_err());
        assert!(PlaylistDoc::load(b"").is_err());
        assert!(PlaylistDoc::load(b"garbage").is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(PlaylistDoc::load(&trailing).is_err());
        // A count that overruns the body is refused (checksum fixed
        // up so the failure is the length check, not the checksum).
        let mut doctored = bytes.clone();
        let count_off = 6 + 2 + doc.id.len() + 2 + doc.title.len();
        doctored[count_off] = 255;
        let sum = sha256(&doctored[..doctored.len() - 32]);
        let tail = doctored.len() - 32;
        doctored[tail..].copy_from_slice(&sum);
        assert!(PlaylistDoc::load(&doctored).is_err());
    }

    #[test]
    fn names_are_validated() {
        assert!(PlaylistDoc::new("", "Title").is_err());
        assert!(PlaylistDoc::new("id/with/slash", "Title").is_err());
        assert!(PlaylistDoc::new("ok-id", "").is_err());
        assert!(PlaylistDoc::new("ok-id", "Fine Title").is_ok());
    }
}
