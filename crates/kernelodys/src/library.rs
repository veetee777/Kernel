//! The library index: find WAV tracks held in a content store and
//! index their metadata.
//!
//! [`LibraryIndex::scan`] walks every block in a [`ContentStore`] and
//! keeps the ones that are complete 16-bit PCM WAV files. For each
//! one it records the title and artist from the WAV's `LIST`/`INFO`
//! chunk (`INAM` / `IART`) when present, falling back to a file name
//! the caller supplies (or `"Untitled"`), and the duration from the
//! sample count. [`LibraryIndex::search_library`] finds tracks by
//! title or artist substring, case-insensitive.
//!
//! Only complete WAV files are indexed: a block that is a chunk of a
//! larger chunked track, a truncated file, or a non-PCM / non-16-bit
//! WAV is skipped, not guessed at. The index reads WAV headers only —
//! it cannot identify a track by ear, and it never plays audio.

use std::collections::HashMap;

use kernel_storage::{Cid, ContentStore};

/// Fallback title when a WAV carries no `INAM` and the caller
/// supplied no file name.
pub const UNTITLED: &str = "Untitled";
/// Fallback artist when a WAV carries no `IART`.
pub const UNKNOWN_ARTIST: &str = "Unknown artist";

/// One indexed track: what the library knows about a WAV block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibraryEntry {
    /// CID of the block holding the whole WAV file.
    pub cid: Cid,
    /// Title from `INAM`, or the caller-supplied file name, or
    /// [`UNTITLED`].
    pub title: String,
    /// Artist from `IART`, or [`UNKNOWN_ARTIST`].
    pub artist: String,
    /// Duration in whole seconds, from the sample count.
    pub duration_secs: u64,
    /// Sample rate in samples per second per channel.
    pub sample_rate: u32,
    /// Channel count (1 mono, 2 stereo).
    pub channels: u16,
}

fn u32le(bytes: &[u8], off: usize) -> Option<u32> {
    bytes
        .get(off..off + 4)
        .map(|s| u32::from_le_bytes(s.try_into().expect("4-byte slice")))
}

fn u16le(bytes: &[u8], off: usize) -> Option<u16> {
    bytes
        .get(off..off + 2)
        .map(|s| u16::from_le_bytes(s.try_into().expect("2-byte slice")))
}

/// Header facts a WAV must carry to be indexed.
struct WavHeader {
    sample_rate: u32,
    channels: u16,
    frames: u64,
    title: Option<String>,
    artist: Option<String>,
}

/// Clean a text value from an INFO subchunk: strip trailing NULs and
/// surrounding whitespace. Returns `None` for an empty result.
fn clean_info_text(raw: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(raw);
    let trimmed = text.trim_end_matches('\0').trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Parse the header of a complete 16-bit PCM WAV file, including the
/// `LIST`/`INFO` chunk when present. Returns `None` for anything that
/// is not a complete, supported WAV: non-RIFF bytes, a declared size
/// that runs past the buffer, a missing or truncated chunk, a
/// non-PCM format, non-16-bit samples, or a channel count other than
/// 1 or 2.
fn parse_wav_header(bytes: &[u8]) -> Option<WavHeader> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    if let Some(riff_size) = u32le(bytes, 4) {
        if riff_size != 0 && 8usize.saturating_add(riff_size as usize) > bytes.len() {
            return None;
        }
    }
    let mut fmt: Option<(usize, usize)> = None;
    let mut data_len: Option<usize> = None;
    let mut title: Option<String> = None;
    let mut artist: Option<String> = None;
    let mut off = 12usize;
    while off + 8 <= bytes.len() {
        let id = &bytes[off..off + 4];
        let size = u32le(bytes, off + 4)? as usize;
        let payload = off + 8;
        let end = payload.checked_add(size)?;
        if end > bytes.len() {
            return None;
        }
        match id {
            b"fmt " => fmt = Some((payload, size)),
            b"data" => data_len = Some(size),
            b"LIST" => {
                // A LIST chunk holds typed subchunks; INFO is the
                // metadata type. Subchunks are word-aligned.
                if size >= 4 && &bytes[payload..payload + 4] == b"INFO" {
                    let mut sub = payload + 4;
                    let list_end = end;
                    while sub + 8 <= list_end {
                        let sub_id = &bytes[sub..sub + 4];
                        let sub_size = u32le(bytes, sub + 4)? as usize;
                        let sub_payload = sub + 8;
                        let sub_end = sub_payload.checked_add(sub_size)?;
                        if sub_end > list_end {
                            return None;
                        }
                        match sub_id {
                            b"INAM" => {
                                title =
                                    title.or_else(|| clean_info_text(&bytes[sub_payload..sub_end]));
                            }
                            b"IART" => {
                                artist = artist
                                    .or_else(|| clean_info_text(&bytes[sub_payload..sub_end]));
                            }
                            _ => {}
                        }
                        sub = sub_end + (sub_size & 1);
                    }
                }
            }
            _ => {}
        }
        off = end + (size & 1);
        if off > bytes.len() {
            off = bytes.len();
        }
    }
    let (fmt_off, fmt_len) = fmt?;
    if fmt_len < 16 {
        return None;
    }
    if u16le(bytes, fmt_off)? != 1 {
        return None;
    }
    let channels = u16le(bytes, fmt_off + 2)?;
    if channels != 1 && channels != 2 {
        return None;
    }
    let sample_rate = u32le(bytes, fmt_off + 4)?;
    if sample_rate == 0 {
        return None;
    }
    if u16le(bytes, fmt_off + 14)? != 16 {
        return None;
    }
    let block_align = u16le(bytes, fmt_off + 12)? as usize;
    if block_align != channels as usize * 2 {
        return None;
    }
    let data_len = data_len?;
    if data_len % block_align != 0 {
        return None;
    }
    let frames = (data_len / block_align) as u64;
    Some(WavHeader {
        sample_rate,
        channels,
        frames,
        title,
        artist,
    })
}

/// An index of the WAV tracks in a content store, built by scanning.
#[derive(Debug, Default, Clone)]
pub struct LibraryIndex {
    entries: Vec<LibraryEntry>,
}

impl LibraryIndex {
    /// Scan every block in `store`, indexing the complete 16-bit PCM
    /// WAV files. `names` maps a block CID to a file name the caller
    /// knows it by; the name is the title fallback when the WAV has
    /// no `INAM`. Entries come out sorted by title, then artist, for
    /// a stable listing.
    pub fn scan(store: &ContentStore, names: &HashMap<Cid, String>) -> Self {
        let mut entries = Vec::new();
        for (cid, bytes) in store.blocks() {
            let Some(header) = parse_wav_header(&bytes) else {
                continue;
            };
            let title = header
                .title
                .or_else(|| names.get(&cid).cloned())
                .unwrap_or_else(|| UNTITLED.to_string());
            let artist = header.artist.unwrap_or_else(|| UNKNOWN_ARTIST.to_string());
            entries.push(LibraryEntry {
                cid,
                title,
                artist,
                duration_secs: header.frames / header.sample_rate as u64,
                sample_rate: header.sample_rate,
                channels: header.channels,
            });
        }
        entries.sort_by(|a, b| a.title.cmp(&b.title).then_with(|| a.artist.cmp(&b.artist)));
        Self { entries }
    }

    /// Number of indexed tracks.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if no tracks are indexed.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All indexed entries, sorted by title then artist.
    pub fn entries(&self) -> &[LibraryEntry] {
        &self.entries
    }

    /// Find tracks whose title or artist contains `query`
    /// (case-insensitive substring). An empty query returns every
    /// entry.
    pub fn search_library(&self, query: &str) -> Vec<&LibraryEntry> {
        let q = query.to_lowercase();
        self.entries
            .iter()
            .filter(|e| {
                q.is_empty()
                    || e.title.to_lowercase().contains(&q)
                    || e.artist.to_lowercase().contains(&q)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 16-bit PCM WAV in memory, optionally with a LIST/INFO
    /// chunk carrying INAM/IART.
    fn build_wav(
        channels: u16,
        sample_rate: u32,
        samples: &[i16],
        info: Option<(&str, &str)>,
    ) -> Vec<u8> {
        let block_align = channels * 2;
        let byte_rate = sample_rate * block_align as u32;
        let mut fmt = Vec::with_capacity(16);
        fmt.extend_from_slice(&1u16.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&sample_rate.to_le_bytes());
        fmt.extend_from_slice(&byte_rate.to_le_bytes());
        fmt.extend_from_slice(&block_align.to_le_bytes());
        fmt.extend_from_slice(&16u16.to_le_bytes());

        let mut data = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            data.extend_from_slice(&s.to_le_bytes());
        }

        let mut chunks = Vec::new();
        let push_chunk = |id: &[u8; 4], payload: &[u8], chunks: &mut Vec<u8>| {
            chunks.extend_from_slice(id);
            chunks.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            chunks.extend_from_slice(payload);
            if payload.len() % 2 == 1 {
                chunks.push(0);
            }
        };
        push_chunk(b"fmt ", &fmt, &mut chunks);
        if let Some((title, artist)) = info {
            let mut list = Vec::new();
            list.extend_from_slice(b"INFO");
            let sub = |id: &[u8; 4], value: &str, list: &mut Vec<u8>| {
                let mut payload = value.as_bytes().to_vec();
                payload.push(0);
                list.extend_from_slice(id);
                list.extend_from_slice(&(payload.len() as u32).to_le_bytes());
                list.extend_from_slice(&payload);
                if payload.len() % 2 == 1 {
                    list.push(0);
                }
            };
            sub(b"INAM", title, &mut list);
            sub(b"IART", artist, &mut list);
            push_chunk(b"LIST", &list, &mut chunks);
        }
        push_chunk(b"data", &data, &mut chunks);

        let mut out = Vec::with_capacity(12 + chunks.len());
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&((4 + chunks.len()) as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(&chunks);
        out
    }

    fn samples_for_secs(sample_rate: u32, secs: u64) -> Vec<i16> {
        vec![1000i16; (sample_rate as u64 * secs) as usize]
    }

    #[test]
    fn scan_indexes_wav_files_with_info_metadata() {
        let mut store = ContentStore::new();
        let a = build_wav(
            1,
            8000,
            &samples_for_secs(8000, 90),
            Some(("Midnight", "Nova")),
        );
        // Stereo: twice as many interleaved samples for the same
        // number of frames (45 seconds).
        let stereo_samples = vec![1000i16; (8000 * 45 * 2) as usize];
        let b = build_wav(2, 8000, &stereo_samples, Some(("Sunrise", "Echo")));
        let cid_a = store.put(&a);
        let cid_b = store.put(&b);
        // Non-WAV blocks are ignored.
        store.put(b"not audio at all");
        store.put(b"KTRACK1fake manifest bytes");

        let index = LibraryIndex::scan(&store, &HashMap::new());
        assert_eq!(index.len(), 2);
        let titles: Vec<&str> = index.entries().iter().map(|e| e.title.as_str()).collect();
        assert_eq!(titles, vec!["Midnight", "Sunrise"]);
        let midnight = &index.entries()[0];
        assert_eq!(midnight.cid, cid_a);
        assert_eq!(midnight.artist, "Nova");
        assert_eq!(midnight.duration_secs, 90);
        assert_eq!(midnight.sample_rate, 8000);
        assert_eq!(midnight.channels, 1);
        let sunrise = &index.entries()[1];
        assert_eq!(sunrise.cid, cid_b);
        assert_eq!(sunrise.artist, "Echo");
        assert_eq!(sunrise.duration_secs, 45);
        assert_eq!(sunrise.channels, 2);
    }

    #[test]
    fn scan_falls_back_to_file_name_and_unknown_artist() {
        let mut store = ContentStore::new();
        let wav = build_wav(1, 8000, &samples_for_secs(8000, 30), None);
        let cid = store.put(&wav);
        let mut names = HashMap::new();
        names.insert(cid, "road-song.wav".to_string());
        let index = LibraryIndex::scan(&store, &names);
        assert_eq!(index.len(), 1);
        assert_eq!(index.entries()[0].title, "road-song.wav");
        assert_eq!(index.entries()[0].artist, UNKNOWN_ARTIST);
        // No name at all: Untitled.
        let index = LibraryIndex::scan(&store, &HashMap::new());
        assert_eq!(index.entries()[0].title, UNTITLED);
    }

    #[test]
    fn scan_skips_truncated_and_unsupported_wavs() {
        let mut store = ContentStore::new();
        let full = build_wav(1, 8000, &samples_for_secs(8000, 10), None);
        // Truncated mid-data.
        let cut = full[..full.len() - 100].to_vec();
        store.put(&cut);
        // A WAV that declares more bytes than it holds.
        let mut lying = full.clone();
        lying[4..8].copy_from_slice(&1_000_000u32.to_le_bytes());
        store.put(&lying);
        // Non-PCM (format tag 3) and 8-bit variants.
        let mut float_fmt = full.clone();
        float_fmt[20..22].copy_from_slice(&3u16.to_le_bytes());
        store.put(&float_fmt);
        let mut eight_bit = full.clone();
        eight_bit[34..36].copy_from_slice(&8u16.to_le_bytes());
        store.put(&eight_bit);
        // Only the intact file is indexed.
        let good = store.put(&full);
        let index = LibraryIndex::scan(&store, &HashMap::new());
        assert_eq!(index.len(), 1);
        assert_eq!(index.entries()[0].cid, good);
    }

    #[test]
    fn search_is_case_insensitive_substring_on_title_and_artist() {
        let mut store = ContentStore::new();
        let mk = |title: &str, artist: &str| {
            build_wav(1, 8000, &samples_for_secs(8000, 5), Some((title, artist)))
        };
        store.put(&mk("Midnight Drive", "Nova"));
        store.put(&mk("Sunrise", "Echo Park"));
        store.put(&mk("Night Owl", "nova rays"));
        let index = LibraryIndex::scan(&store, &HashMap::new());
        assert_eq!(index.len(), 3);
        fn titles(hits: Vec<&LibraryEntry>) -> Vec<&str> {
            hits.iter().map(|e| e.title.as_str()).collect()
        }
        assert_eq!(
            titles(index.search_library("night")),
            vec!["Midnight Drive", "Night Owl"]
        );
        assert_eq!(
            titles(index.search_library("NOVA")),
            vec!["Midnight Drive", "Night Owl"]
        );
        assert_eq!(titles(index.search_library("echo")), vec!["Sunrise"]);
        assert_eq!(titles(index.search_library("zzz")), Vec::<&str>::new());
        assert_eq!(index.search_library("").len(), 3);
    }
}
