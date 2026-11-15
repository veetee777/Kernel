//! Kernelodys — music on top of KERNEL content-addressed storage.
//!
//! A track is a human title and artist plus audio bytes. The bytes are
//! stored chunked in the content store, and the track is described by a
//! signed manifest: title, artist, duration in seconds, the chunked
//! content root CID, the owner, and the owner's signature.
//!
//! A playlist is a signed record: a name plus an ordered list of track
//! manifest CIDs, signed by the owner. Resolving a playlist means
//! resolving each track CID through [`TrackStore::get_track`], which
//! verifies the manifest signature and every chunk hash.
//!
//! The manifest is itself a block in the content store, addressed by its
//! own CID. Anyone holding the manifest CID can fetch the track — and the
//! signature plus per-chunk hash verification mean a tampered manifest
//! or a tampered chunk fails loudly instead of silently returning wrong
//! bytes.
//!
//! Nothing here touches an audio device. This crate stores, verifies,
//! and lists tracks and playlists, and decodes 16-bit PCM WAV bytes
//! held in memory (see [`wav`]). Playback is the [`tuning`] module
//! plus the [`playback`] engine: Kernelodys plays at A = 432 Hz, and
//! [`tuning::play_432`] retunes decoded PCM to 432 before the engine
//! streams it to a sink in fixed-size chunks. [`playback::WavFileSink`]
//! renders a track to a real `.wav` file on disk; a device sink
//! (ALSA, AAudio) is future work, not this build.
//!
//! Manifest formats (explicit binary, no JSON):
//! `KTRACK1` + title length (u16 LE) + title bytes + artist length
//! (u16 LE) + artist bytes + duration (u64 LE) + chunk root CID (32) +
//! owner public key (32) + signature (64).
//! `KPLAY1` + name length (u16 LE) + name bytes + track count (u32 LE)
//! + track CIDs (32 each) + owner public key (32) + signature (64).

use std::collections::HashMap;
use std::fmt;

use kernel_crypto::Ed25519Signature;
use kernel_identity::{Identity, IdentityError, PeerId};
use kernel_storage::{Cid, ContentStore, StorageError};

/// Playback tuning: Kernelodys plays at A = 432 Hz.
pub mod tuning;

/// WAV decoding: 16-bit PCM WAV bytes in memory, decoded and retuned
/// to 432 Hz by [`wav::play_wav_432`].
pub mod wav;

/// Real playback: a sample-accurate engine that streams retuned audio
/// to a sink in fixed-size chunks, with pause/resume/stop and underrun
/// counting. See [`playback::PlaybackEngine`].
pub mod playback;

/// Magic bytes at the start of every track manifest.
const TRACK_MAGIC: &[u8; 7] = b"KTRACK1";
/// Domain separator for track manifest signatures.
const TRACK_SIGN_DOMAIN: &[u8] = b"KTRACK1";
/// Magic bytes at the start of every playlist record.
const PLAYLIST_MAGIC: &[u8; 6] = b"KPLAY1";
/// Domain separator for playlist signatures.
const PLAYLIST_SIGN_DOMAIN: &[u8] = b"KPLAY1";

/// Maximum length of a track title, artist, or playlist name.
pub const MAX_NAME_LEN: usize = 128;
/// Maximum track duration: 24 hours, in seconds.
pub const MAX_DURATION_SECS: u64 = 86_400;
/// Maximum tracks in one playlist.
pub const MAX_PLAYLIST_TRACKS: usize = 10_000;

/// Errors returned by track and playlist operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackError {
    /// Wraps a [`StorageError`].
    Storage(StorageError),
    /// Wraps an [`IdentityError`].
    Identity(IdentityError),
    /// A title broke the naming rules.
    InvalidTitle(String),
    /// An artist name broke the naming rules.
    InvalidArtist(String),
    /// A duration was out of range.
    InvalidDuration(String),
    /// A playlist name broke the naming rules.
    InvalidName(String),
    /// A manifest or playlist record was malformed.
    InvalidManifest(String),
    /// A manifest's or playlist's signature did not verify.
    BadSignature,
    /// No track or playlist with this manifest CID is known here.
    NotFound(Cid),
}

impl fmt::Display for TrackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrackError::Storage(e) => write!(f, "storage error: {e}"),
            TrackError::Identity(e) => write!(f, "identity error: {e}"),
            TrackError::InvalidTitle(msg) => write!(f, "invalid track title: {msg}"),
            TrackError::InvalidArtist(msg) => write!(f, "invalid artist: {msg}"),
            TrackError::InvalidDuration(msg) => write!(f, "invalid duration: {msg}"),
            TrackError::InvalidName(msg) => write!(f, "invalid playlist name: {msg}"),
            TrackError::InvalidManifest(msg) => write!(f, "invalid manifest: {msg}"),
            TrackError::BadSignature => write!(f, "manifest signature is invalid"),
            TrackError::NotFound(cid) => write!(f, "track not found: {cid}"),
        }
    }
}

impl std::error::Error for TrackError {}

impl From<StorageError> for TrackError {
    fn from(e: StorageError) -> Self {
        TrackError::Storage(e)
    }
}

impl From<IdentityError> for TrackError {
    fn from(e: IdentityError) -> Self {
        TrackError::Identity(e)
    }
}

/// Rules for human names (titles, artists, playlist names): 1-128
/// printable ASCII characters, no path separators. Names are human
/// labels, not hashes, and must never be able to escape into a path.
fn validate_human_name(name: &str, kind: &str) -> Result<(), TrackError> {
    let err = |msg: String| match kind {
        "title" => TrackError::InvalidTitle(msg),
        "artist" => TrackError::InvalidArtist(msg),
        _ => TrackError::InvalidName(msg),
    };
    if name.is_empty() {
        return Err(err(format!("{kind} must not be empty")));
    }
    if name.len() > MAX_NAME_LEN {
        return Err(err(format!("{kind} must be at most {MAX_NAME_LEN} characters")));
    }
    if name.contains('/') || name.contains('\\') {
        return Err(err(format!("{kind} must not contain a path separator")));
    }
    if !name.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
        return Err(err(format!("{kind} must be printable ASCII")));
    }
    Ok(())
}

/// Rules for durations: at least 1 second, at most [`MAX_DURATION_SECS`].
fn validate_duration(duration_secs: u64) -> Result<(), TrackError> {
    if duration_secs == 0 {
        return Err(TrackError::InvalidDuration(
            "duration must be at least 1 second".to_string(),
        ));
    }
    if duration_secs > MAX_DURATION_SECS {
        return Err(TrackError::InvalidDuration(format!(
            "duration must be at most {MAX_DURATION_SECS} seconds"
        )));
    }
    Ok(())
}

/// The message an owner signs for a track manifest.
fn track_sign_message(
    title: &str,
    artist: &str,
    duration_secs: u64,
    root: &Cid,
    owner: &PeerId,
) -> Vec<u8> {
    let mut msg = Vec::with_capacity(
        TRACK_SIGN_DOMAIN.len() + title.len() + artist.len() + 8 + 32 + 32,
    );
    msg.extend_from_slice(TRACK_SIGN_DOMAIN);
    msg.extend_from_slice(title.as_bytes());
    msg.extend_from_slice(artist.as_bytes());
    msg.extend_from_slice(&duration_secs.to_le_bytes());
    msg.extend_from_slice(root.as_bytes());
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg
}

/// The message an owner signs for a playlist record.
fn playlist_sign_message(name: &str, tracks: &[Cid], owner: &PeerId) -> Vec<u8> {
    let mut msg =
        Vec::with_capacity(PLAYLIST_SIGN_DOMAIN.len() + name.len() + 4 + 32 * tracks.len() + 32);
    msg.extend_from_slice(PLAYLIST_SIGN_DOMAIN);
    msg.extend_from_slice(name.as_bytes());
    msg.extend_from_slice(&(tracks.len() as u32).to_le_bytes());
    for cid in tracks {
        msg.extend_from_slice(cid.as_bytes());
    }
    msg.extend_from_slice(owner.public_key().as_bytes());
    msg
}

/// A display name derived from a public key: the first 8 hex chars of
/// the key. Used when a manifest is decoded and only the key survives
/// (verification needs the key, not the human name).
fn fingerprint_name(key: &kernel_crypto::Ed25519PublicKey) -> String {
    let hex = kernel_crypto::to_hex(key.as_bytes());
    format!("peer-{}", &hex[..8])
}

/// A signed track manifest: title, artist, duration, chunked-content
/// root, owner, signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrackManifest {
    /// Human-readable track title.
    pub title: String,
    /// Human-readable artist name.
    pub artist: String,
    /// Track duration in seconds.
    pub duration_secs: u64,
    /// The owner who signed this manifest.
    pub owner: PeerId,
    /// Root CID of the chunked audio content (a `KCHUNK1` manifest block).
    pub root: Cid,
    /// The owner's signature over title, artist, duration, root, owner key.
    pub signature: Ed25519Signature,
}

impl TrackManifest {
    /// Build and sign a manifest as `identity`.
    pub fn new(
        title: &str,
        artist: &str,
        duration_secs: u64,
        root: Cid,
        identity: &Identity,
    ) -> Result<Self, TrackError> {
        validate_human_name(title, "title")?;
        validate_human_name(artist, "artist")?;
        validate_duration(duration_secs)?;
        let owner = identity.peer_id().clone();
        let msg = track_sign_message(title, artist, duration_secs, &root, &owner);
        let signature = identity.sign(&msg);
        Ok(Self {
            title: title.to_string(),
            artist: artist.to_string(),
            duration_secs,
            owner,
            root,
            signature,
        })
    }

    /// True if the signature verifies against the owner for this exact
    /// title, artist, duration, and root. Any tampering fails this check.
    pub fn verify(&self) -> bool {
        let msg = track_sign_message(
            &self.title,
            &self.artist,
            self.duration_secs,
            &self.root,
            &self.owner,
        );
        self.owner.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary manifest format.
    pub fn encode(&self) -> Vec<u8> {
        let title_bytes = self.title.as_bytes();
        let artist_bytes = self.artist.as_bytes();
        let mut out = Vec::with_capacity(
            7 + 2 + title_bytes.len() + 2 + artist_bytes.len() + 8 + 32 + 32 + 64,
        );
        out.extend_from_slice(TRACK_MAGIC);
        out.extend_from_slice(&(title_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(title_bytes);
        out.extend_from_slice(&(artist_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(artist_bytes);
        out.extend_from_slice(&self.duration_secs.to_le_bytes());
        out.extend_from_slice(self.root.as_bytes());
        out.extend_from_slice(self.owner.public_key().as_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a manifest produced by [`TrackManifest::encode`].
    /// Rejects anything that is not exactly that format.
    pub fn decode(bytes: &[u8]) -> Result<Self, TrackError> {
        if bytes.len() < 7 + 2 + 2 + 8 + 32 + 32 + 64 || &bytes[..7] != TRACK_MAGIC {
            return Err(TrackError::InvalidManifest(
                "missing KTRACK1 magic".to_string(),
            ));
        }
        let title_len = u16::from_le_bytes(bytes[7..9].try_into().expect("2-byte slice")) as usize;
        let artist_len_off = 9 + title_len;
        if bytes.len() < artist_len_off + 2 {
            return Err(TrackError::InvalidManifest(
                "manifest too short for artist length".to_string(),
            ));
        }
        let artist_len =
            u16::from_le_bytes(bytes[artist_len_off..artist_len_off + 2].try_into().expect("2-byte slice"))
                as usize;
        let expected = artist_len_off + 2 + artist_len + 8 + 32 + 32 + 64;
        if bytes.len() != expected {
            return Err(TrackError::InvalidManifest(format!(
                "manifest is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let mut off = 9;
        let title = String::from_utf8(bytes[off..off + title_len].to_vec()).map_err(|_| {
            TrackError::InvalidManifest("manifest title is not valid UTF-8".to_string())
        })?;
        validate_human_name(&title, "title")?;
        off += title_len;
        off += 2; // skip the artist length field
        let artist = String::from_utf8(bytes[off..off + artist_len].to_vec()).map_err(|_| {
            TrackError::InvalidManifest("manifest artist is not valid UTF-8".to_string())
        })?;
        validate_human_name(&artist, "artist")?;
        off += artist_len;
        let duration_secs =
            u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8-byte slice"));
        off += 8;
        validate_duration(duration_secs)?;
        let mut root_bytes = [0u8; 32];
        root_bytes.copy_from_slice(&bytes[off..off + 32]);
        let root = Cid::from_bytes(root_bytes);
        off += 32;
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let owner_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| TrackError::InvalidManifest(e.to_string()))?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let owner = PeerId::new(owner_key, &fingerprint_name(&owner_key))?;
        Ok(Self {
            title,
            artist,
            duration_secs,
            owner,
            root,
            signature,
        })
    }
}

/// A signed playlist record: a name plus an ordered list of track
/// manifest CIDs, signed by the owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Playlist {
    /// Human-readable playlist name.
    pub name: String,
    /// Track manifest CIDs, in play order.
    pub tracks: Vec<Cid>,
    /// The owner who signed this playlist.
    pub owner: PeerId,
    /// The owner's signature over name, track list, and owner key.
    pub signature: Ed25519Signature,
}

impl Playlist {
    /// Build and sign a playlist as `identity`.
    pub fn new(name: &str, tracks: &[Cid], identity: &Identity) -> Result<Self, TrackError> {
        validate_human_name(name, "playlist")?;
        if tracks.len() > MAX_PLAYLIST_TRACKS {
            return Err(TrackError::InvalidManifest(format!(
                "playlist must have at most {MAX_PLAYLIST_TRACKS} tracks"
            )));
        }
        let owner = identity.peer_id().clone();
        let msg = playlist_sign_message(name, tracks, &owner);
        let signature = identity.sign(&msg);
        Ok(Self {
            name: name.to_string(),
            tracks: tracks.to_vec(),
            owner,
            signature,
        })
    }

    /// True if the signature verifies against the owner for this exact
    /// name and track list.
    pub fn verify(&self) -> bool {
        let msg = playlist_sign_message(&self.name, &self.tracks, &self.owner);
        self.owner.verify(&msg, &self.signature).is_ok()
    }

    /// Encode to the explicit binary playlist format.
    pub fn encode(&self) -> Vec<u8> {
        let name_bytes = self.name.as_bytes();
        let mut out =
            Vec::with_capacity(6 + 2 + name_bytes.len() + 4 + 32 * self.tracks.len() + 32 + 64);
        out.extend_from_slice(PLAYLIST_MAGIC);
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(&(self.tracks.len() as u32).to_le_bytes());
        for cid in &self.tracks {
            out.extend_from_slice(cid.as_bytes());
        }
        out.extend_from_slice(self.owner.public_key().as_bytes());
        out.extend_from_slice(&self.signature.to_bytes());
        out
    }

    /// Decode a playlist produced by [`Playlist::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, TrackError> {
        if bytes.len() < 6 + 2 + 4 + 32 + 64 || &bytes[..6] != PLAYLIST_MAGIC {
            return Err(TrackError::InvalidManifest(
                "missing KPLAY1 magic".to_string(),
            ));
        }
        let name_len = u16::from_le_bytes(bytes[6..8].try_into().expect("2-byte slice")) as usize;
        let count_off = 8 + name_len;
        if bytes.len() < count_off + 4 {
            return Err(TrackError::InvalidManifest(
                "playlist too short for track count".to_string(),
            ));
        }
        let count = u32::from_le_bytes(
            bytes[count_off..count_off + 4]
                .try_into()
                .expect("4-byte slice"),
        ) as usize;
        if count > MAX_PLAYLIST_TRACKS {
            return Err(TrackError::InvalidManifest(format!(
                "playlist claims {count} tracks, max is {MAX_PLAYLIST_TRACKS}"
            )));
        }
        let expected = count_off + 4 + 32 * count + 32 + 64;
        if bytes.len() != expected {
            return Err(TrackError::InvalidManifest(format!(
                "playlist is {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let name = String::from_utf8(bytes[8..8 + name_len].to_vec()).map_err(|_| {
            TrackError::InvalidManifest("playlist name is not valid UTF-8".to_string())
        })?;
        validate_human_name(&name, "playlist")?;
        let mut off = count_off + 4;
        let mut tracks = Vec::with_capacity(count);
        for _ in 0..count {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes[off..off + 32]);
            tracks.push(Cid::from_bytes(arr));
            off += 32;
        }
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        let owner_key = kernel_crypto::Ed25519PublicKey::from_bytes(&key_bytes)
            .map_err(|e| TrackError::InvalidManifest(e.to_string()))?;
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&bytes[off..off + 64]);
        let signature = Ed25519Signature::from_bytes(&sig_bytes);
        let owner = PeerId::new(owner_key, &fingerprint_name(&owner_key))?;
        Ok(Self {
            name,
            tracks,
            owner,
            signature,
        })
    }
}

/// The metadata of a track, as a human sees it — no CIDs, no hashes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrackMetadata {
    /// Human-readable track title.
    pub title: String,
    /// Human-readable artist name.
    pub artist: String,
    /// Track duration in seconds.
    pub duration_secs: u64,
    /// Who owns (published) this track.
    pub owner: PeerId,
}

/// A track and playlist store: a content store plus indexes of
/// manifests and playlists. Tracks are addressed by manifest CID;
/// humans see titles and artists.
pub struct TrackStore {
    store: ContentStore,
    /// Track manifests this node holds, by manifest CID.
    tracks: HashMap<Cid, TrackManifest>,
    /// Playlists this node holds, by record CID.
    playlists: HashMap<Cid, Playlist>,
}

impl Default for TrackStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackStore {
    /// An empty track store.
    pub fn new() -> Self {
        Self {
            store: ContentStore::new(),
            tracks: HashMap::new(),
            playlists: HashMap::new(),
        }
    }

    /// Number of tracks held.
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// True if no tracks are held.
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// The manifest for a track, if this node holds it.
    pub fn get_manifest(&self, manifest_cid: &Cid) -> Option<&TrackManifest> {
        self.tracks.get(manifest_cid)
    }

    /// The playlist record for a playlist CID, if held.
    pub fn get_playlist_record(&self, playlist_cid: &Cid) -> Option<&Playlist> {
        self.playlists.get(playlist_cid)
    }

    /// Store a track: chunk the audio bytes, build and sign the
    /// manifest, and store the manifest as a block. Returns the
    /// manifest CID — the track's address.
    pub fn put_track(
        &mut self,
        title: &str,
        artist: &str,
        duration_secs: u64,
        data: &[u8],
        identity: &Identity,
    ) -> Result<Cid, TrackError> {
        validate_human_name(title, "title")?;
        validate_human_name(artist, "artist")?;
        validate_duration(duration_secs)?;
        let root = self.store.put_chunked(data);
        let manifest = TrackManifest::new(title, artist, duration_secs, root, identity)?;
        let manifest_cid = self.store.put(&manifest.encode());
        self.tracks.insert(manifest_cid, manifest);
        Ok(manifest_cid)
    }

    /// Fetch a track by manifest CID: verify the manifest signature,
    /// then reassemble the chunked audio (every chunk hash-verified).
    /// Returns the audio bytes and human-readable metadata.
    pub fn get_track(
        &self,
        manifest_cid: &Cid,
    ) -> Result<(Vec<u8>, TrackMetadata), TrackError> {
        let manifest = match self.tracks.get(manifest_cid) {
            Some(m) => m.clone(),
            None => {
                let bytes = self.store.get(manifest_cid).map_err(|e| match e {
                    StorageError::NotFound(cid) => TrackError::NotFound(cid),
                    other => TrackError::Storage(other),
                })?;
                TrackManifest::decode(&bytes)?
            }
        };
        if !manifest.verify() {
            return Err(TrackError::BadSignature);
        }
        let data = self
            .store
            .get_chunked(&manifest.root)
            .map_err(TrackError::Storage)?;
        let metadata = TrackMetadata {
            title: manifest.title,
            artist: manifest.artist,
            duration_secs: manifest.duration_secs,
            owner: manifest.owner,
        };
        Ok((data, metadata))
    }

    /// All tracks held, as (manifest CID, title), sorted by title
    /// for a stable, human-readable listing.
    pub fn list_tracks(&self) -> Vec<(Cid, String)> {
        let mut out: Vec<(Cid, String)> = self
            .tracks
            .iter()
            .map(|(cid, m)| (*cid, m.title.clone()))
            .collect();
        out.sort_by(|a, b| a.1.cmp(&b.1));
        out
    }

    /// Create a playlist over tracks this node already holds. Every
    /// track CID must be a track in this store, or the playlist is
    /// refused. Returns the playlist record CID.
    pub fn create_playlist(
        &mut self,
        name: &str,
        track_cids: &[Cid],
        identity: &Identity,
    ) -> Result<Cid, TrackError> {
        validate_human_name(name, "playlist")?;
        for cid in track_cids {
            if !self.tracks.contains_key(cid) {
                return Err(TrackError::NotFound(*cid));
            }
        }
        let playlist = Playlist::new(name, track_cids, identity)?;
        let playlist_cid = self.store.put(&playlist.encode());
        self.playlists.insert(playlist_cid, playlist);
        Ok(playlist_cid)
    }

    /// Fetch a playlist by record CID, verifying its signature.
    pub fn get_playlist(&self, playlist_cid: &Cid) -> Result<Playlist, TrackError> {
        let playlist = match self.playlists.get(playlist_cid) {
            Some(p) => p.clone(),
            None => {
                let bytes = self.store.get(playlist_cid).map_err(|e| match e {
                    StorageError::NotFound(cid) => TrackError::NotFound(cid),
                    other => TrackError::Storage(other),
                })?;
                Playlist::decode(&bytes)?
            }
        };
        if !playlist.verify() {
            return Err(TrackError::BadSignature);
        }
        Ok(playlist)
    }

    /// All playlists held, as (record CID, name), sorted by name.
    pub fn list_playlists(&self) -> Vec<(Cid, String)> {
        let mut out: Vec<(Cid, String)> = self
            .playlists
            .iter()
            .map(|(cid, p)| (*cid, p.name.clone()))
            .collect();
        out.sort_by(|a, b| a.1.cmp(&b.1));
        out
    }

    /// Mutable access to the underlying content store. Test-only:
    /// lets tests simulate corruption (see `put_unchecked`).
    #[cfg(test)]
    pub fn store_mut(&mut self) -> &mut ContentStore {
        &mut self.store
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_storage::decode_manifest;

    fn test_data(n: usize) -> Vec<u8> {
        (0..n).map(|i| ((i * 37 + 11) % 251) as u8).collect()
    }

    #[test]
    fn manifest_encode_decode_round_trip_and_verify() {
        let alice = Identity::generate("alice").unwrap();
        let root = Cid::from_bytes(kernel_crypto::sha256(b"audio chunks"));
        let manifest = TrackManifest::new("Midnight", "Nova", 217, root, &alice).unwrap();
        assert!(manifest.verify());
        let encoded = manifest.encode();
        let decoded = TrackManifest::decode(&encoded).unwrap();
        assert_eq!(decoded.title, "Midnight");
        assert_eq!(decoded.artist, "Nova");
        assert_eq!(decoded.duration_secs, 217);
        assert_eq!(decoded.root, root);
        assert_eq!(decoded.signature, manifest.signature);
        assert_eq!(decoded.owner.public_key(), manifest.owner.public_key());
        assert!(decoded.verify());
        assert!(TrackManifest::decode(b"garbage").is_err());
        assert!(TrackManifest::decode(&encoded[..encoded.len() - 1]).is_err());
    }

    #[test]
    fn track_round_trip_small_and_large() {
        let alice = Identity::generate("alice").unwrap();
        let mut tracks = TrackStore::new();
        let small = b"fake mp3 bytes";
        let cid = tracks
            .put_track("Intro", "Nova", 45, small, &alice)
            .unwrap();
        let (data, meta) = tracks.get_track(&cid).unwrap();
        assert_eq!(data, small);
        assert_eq!(meta.title, "Intro");
        assert_eq!(meta.artist, "Nova");
        assert_eq!(meta.duration_secs, 45);
        assert_eq!(meta.owner, *alice.peer_id());
        assert_eq!(tracks.len(), 1);

        let large = test_data(600 * 1024);
        let big_cid = tracks
            .put_track("Epic", "Nova", 360, &large, &alice)
            .unwrap();
        let (data, meta) = tracks.get_track(&big_cid).unwrap();
        assert_eq!(data, large);
        assert_eq!(meta.title, "Epic");
        assert_eq!(tracks.len(), 2);
        // Same title + artist + duration + content -> same manifest CID.
        assert_eq!(
            tracks.put_track("Intro", "Nova", 45, small, &alice).unwrap(),
            cid
        );
    }

    #[test]
    fn tampered_chunk_fails() {
        let alice = Identity::generate("alice").unwrap();
        let mut tracks = TrackStore::new();
        let data = test_data(600 * 1024);
        let cid = tracks
            .put_track("Epic", "Nova", 360, &data, &alice)
            .unwrap();
        let manifest = tracks.get_manifest(&cid).unwrap().clone();
        let manifest_bytes = tracks.store_mut().get(&manifest.root).unwrap();
        let chunk_cids = decode_manifest(&manifest_bytes).unwrap();
        let mut bad = tracks.store_mut().get(&chunk_cids[1]).unwrap();
        bad[0] ^= 0xff;
        tracks.store_mut().put_unchecked(&chunk_cids[1], bad);
        match tracks.get_track(&cid) {
            Err(TrackError::Storage(StorageError::CorruptChunk { cid: got, .. })) => {
                assert_eq!(got, chunk_cids[1]);
            }
            other => panic!("expected CorruptChunk, got {other:?}"),
        }
    }

    #[test]
    fn tampered_manifest_signature_fails() {
        let alice = Identity::generate("alice").unwrap();
        let mut tracks = TrackStore::new();
        let cid = tracks
            .put_track("Secret", "Nova", 120, b"audio", &alice)
            .unwrap();
        let mut bytes = tracks.store_mut().get(&cid).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let tampered_cid = tracks.store_mut().put(&bytes);
        assert_ne!(tampered_cid, cid);
        assert_eq!(
            tracks.get_track(&tampered_cid),
            Err(TrackError::BadSignature)
        );
        assert_eq!(tracks.get_track(&cid).unwrap().0, b"audio");
    }

    #[test]
    fn playlist_round_trip_two_tracks() {
        let alice = Identity::generate("alice").unwrap();
        let mut tracks = TrackStore::new();
        let a = tracks.put_track("One", "Nova", 60, b"aaa", &alice).unwrap();
        let b = tracks.put_track("Two", "Nova", 90, b"bbb", &alice).unwrap();
        let playlist_cid = tracks
            .create_playlist("Favorites", &[a, b], &alice)
            .unwrap();
        let playlist = tracks.get_playlist(&playlist_cid).unwrap();
        assert_eq!(playlist.name, "Favorites");
        assert_eq!(playlist.tracks, vec![a, b]);
        assert_eq!(playlist.owner, *alice.peer_id());
        assert!(playlist.verify());
        // Play order resolves through get_track.
        let (first, _) = tracks.get_track(&playlist.tracks[0]).unwrap();
        assert_eq!(first, b"aaa");
        let (second, _) = tracks.get_track(&playlist.tracks[1]).unwrap();
        assert_eq!(second, b"bbb");
        // Tampered playlist signature fails.
        let mut bytes = tracks.store_mut().get(&playlist_cid).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let bad_cid = tracks.store_mut().put(&bytes);
        assert_eq!(
            tracks.get_playlist(&bad_cid),
            Err(TrackError::BadSignature)
        );
    }

    #[test]
    fn playlist_requires_existing_tracks() {
        let alice = Identity::generate("alice").unwrap();
        let mut tracks = TrackStore::new();
        let ghost = Cid::from_bytes(kernel_crypto::sha256(b"no such track"));
        assert_eq!(
            tracks.create_playlist("Empty", &[ghost], &alice),
            Err(TrackError::NotFound(ghost))
        );
        // An empty playlist is allowed.
        let empty = tracks.create_playlist("Empty", &[], &alice).unwrap();
        assert!(tracks.get_playlist(&empty).unwrap().tracks.is_empty());
    }

    #[test]
    fn list_tracks_and_playlists_sorted() {
        let alice = Identity::generate("alice").unwrap();
        let mut tracks = TrackStore::new();
        tracks.put_track("Zulu", "Nova", 60, b"1", &alice).unwrap();
        tracks.put_track("Alpha", "Nova", 60, b"2", &alice).unwrap();
        tracks.put_track("Mike", "Nova", 60, b"3", &alice).unwrap();
        let listed = tracks.list_tracks();
        let titles: Vec<&str> = listed.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(titles, vec!["Alpha", "Mike", "Zulu"]);
        let a = tracks.list_tracks()[0].0;
        tracks.create_playlist("Zoo", &[a], &alice).unwrap();
        tracks.create_playlist("Apple", &[a], &alice).unwrap();
        let playlists = tracks.list_playlists();
        let names: Vec<&str> = playlists.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, vec!["Apple", "Zoo"]);
    }

    #[test]
    fn name_and_duration_rules() {
        let alice = Identity::generate("alice").unwrap();
        let mut tracks = TrackStore::new();
        assert!(tracks.put_track("", "Nova", 60, b"x", &alice).is_err());
        assert!(tracks.put_track("a/b", "Nova", 60, b"x", &alice).is_err());
        assert!(tracks.put_track("Ok", "", 60, b"x", &alice).is_err());
        assert!(tracks.put_track("Ok", "Nova", 0, b"x", &alice).is_err());
        assert!(tracks
            .put_track("Ok", "Nova", MAX_DURATION_SECS + 1, b"x", &alice)
            .is_err());
        assert!(tracks
            .put_track("Ok", "Nova", MAX_DURATION_SECS, b"x", &alice)
            .is_ok());
    }

    #[test]
    fn get_unknown_track_is_not_found() {
        let tracks = TrackStore::new();
        let cid = Cid::from_bytes(kernel_crypto::sha256(b"never stored"));
        assert_eq!(tracks.get_track(&cid), Err(TrackError::NotFound(cid)));
    }
}
