//! Pinned public keys with sticky revocation.
//!
//! Pins are local decisions. Nothing here learns a key from the network.
//! A key must be pinned out of band before a handshake with it succeeds.
//! There is no trust-on-first-use.
//!
//! Persistence (Phase 1): a PinSet saves to one binary file. The format is
//! fixed and has no JSON:
//!
//! ```text
//! magic   "KCPIN1" (6 bytes)
//! pinned  count u32 LE, then that many 32-byte keys
//! revoked count u32 LE, then that many 32-byte keys
//! ```
//!
//! A save writes a sibling `.tmp` file and renames it over the target, so a
//! crash mid-write never leaves a half-written pin file. A load refuses a
//! bad magic, a truncated file, or trailing bytes. Disk errors surface as
//! `CarrierError::Internal`; the caller decides what that means.

use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::error::CarrierError;

pub type PublicKey = [u8; 32];

const MAGIC: &[u8; 6] = b"KCPIN1";

#[derive(Debug, Default)]
pub struct PinSet {
    pinned: HashSet<PublicKey>,
    revoked: HashSet<PublicKey>,
}

impl PinSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pin a key. Refused if the key was ever revoked in this set.
    pub fn pin(&mut self, key: PublicKey) -> Result<(), CarrierError> {
        if self.revoked.contains(&key) {
            return Err(CarrierError::RevokedPeer);
        }
        self.pinned.insert(key);
        Ok(())
    }

    /// Revoke a key. Sticky: it cannot be pinned again in this set.
    pub fn revoke(&mut self, key: PublicKey) {
        self.pinned.remove(&key);
        self.revoked.insert(key);
    }

    /// Ok only for a pinned key that has not been revoked.
    pub fn check(&self, key: &PublicKey) -> Result<(), CarrierError> {
        if self.revoked.contains(key) {
            Err(CarrierError::RevokedPeer)
        } else if self.pinned.contains(key) {
            Ok(())
        } else {
            Err(CarrierError::UnpinnedPeer)
        }
    }

    /// Encode this set to its binary form. See the module docs.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            MAGIC.len() + 8 + (self.pinned.len() + self.revoked.len()) * 32,
        );
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(self.pinned.len() as u32).to_le_bytes());
        for k in &self.pinned {
            out.extend_from_slice(k);
        }
        out.extend_from_slice(&(self.revoked.len() as u32).to_le_bytes());
        for k in &self.revoked {
            out.extend_from_slice(k);
        }
        out
    }

    /// Decode a set from its binary form. Refuses bad magic, truncation,
    /// trailing bytes, or a count that cannot fit the remaining bytes.
    pub fn decode(bytes: &[u8]) -> Result<PinSet, CarrierError> {
        fn take_u32(b: &[u8], at: &mut usize) -> Result<u32, CarrierError> {
            let end = *at + 4;
            if end > b.len() {
                return Err(CarrierError::MalformedFrame);
            }
            let v = u32::from_le_bytes(b[*at..end].try_into().unwrap());
            *at = end;
            Ok(v)
        }
        fn take_keys(
            b: &[u8],
            at: &mut usize,
            n: u32,
        ) -> Result<HashSet<PublicKey>, CarrierError> {
            let n = n as usize;
            if n > (b.len() - *at) / 32 {
                return Err(CarrierError::MalformedFrame);
            }
            let mut set = HashSet::with_capacity(n);
            for _ in 0..n {
                let k: PublicKey = b[*at..*at + 32].try_into().unwrap();
                *at += 32;
                set.insert(k);
            }
            Ok(set)
        }

        if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
            return Err(CarrierError::MalformedFrame);
        }
        let mut at = MAGIC.len();
        let pinned_n = take_u32(bytes, &mut at)?;
        let pinned = take_keys(bytes, &mut at, pinned_n)?;
        let revoked_n = take_u32(bytes, &mut at)?;
        let revoked = take_keys(bytes, &mut at, revoked_n)?;
        if at != bytes.len() {
            return Err(CarrierError::MalformedFrame);
        }
        Ok(PinSet { pinned, revoked })
    }

    /// Save this set to `path`, atomically: a sibling `.tmp` file is
    /// written and renamed over the target.
    pub fn save(&self, path: &Path) -> Result<(), CarrierError> {
        let tmp = path.with_file_name(format!(
            "{}.tmp",
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("pins")
        ));
        let bytes = self.encode();
        File::create(&tmp)
            .and_then(|mut f| f.write_all(&bytes).and_then(|_| f.sync_all()))
            .map_err(|_| CarrierError::Internal)?;
        std::fs::rename(&tmp, path).map_err(|_| CarrierError::Internal)?;
        Ok(())
    }

    /// Load a set from `path`. A missing or unreadable file is an error.
    pub fn load(path: &Path) -> Result<PinSet, CarrierError> {
        let mut bytes = Vec::new();
        File::open(path)
            .and_then(|mut f| f.read_to_end(&mut bytes))
            .map_err(|_| CarrierError::Internal)?;
        Self::decode(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const K: PublicKey = [4u8; 32];
    const OTHER: PublicKey = [5u8; 32];

    #[test]
    fn pinned_key_passes_and_unknown_key_is_refused() {
        let mut p = PinSet::new();
        p.pin(K).expect("pin");
        assert!(p.check(&K).is_ok());
        assert_eq!(p.check(&OTHER).err().expect("refuse"), CarrierError::UnpinnedPeer);
    }

    #[test]
    fn revocation_is_sticky() {
        let mut p = PinSet::new();
        p.pin(K).expect("pin");
        p.revoke(K);
        assert_eq!(p.check(&K).err().expect("refuse"), CarrierError::RevokedPeer);
        assert_eq!(p.pin(K).err().expect("refuse"), CarrierError::RevokedPeer);
    }

    fn set_with(keys: &[PublicKey], revoked: &[PublicKey]) -> PinSet {
        let mut p = PinSet::new();
        for k in keys {
            p.pin(*k).expect("pin");
        }
        for k in revoked {
            p.revoke(*k);
        }
        p
    }

    #[test]
    fn round_trip_keeps_pins_and_revocations() {
        let p = set_with(&[K, OTHER], &[K]);
        let back = PinSet::decode(&p.encode()).expect("decode");
        assert!(back.check(&OTHER).is_ok());
        assert_eq!(back.check(&K).err().expect("refuse"), CarrierError::RevokedPeer);
    }

    #[test]
    fn revocation_stays_sticky_after_reload() {
        let p = set_with(&[K], &[K]);
        let mut back = PinSet::decode(&p.encode()).expect("decode");
        assert_eq!(back.pin(K).err().expect("refuse"), CarrierError::RevokedPeer);
    }

    #[test]
    fn empty_set_round_trips() {
        let back = PinSet::decode(&PinSet::new().encode()).expect("decode");
        assert_eq!(back.check(&K).err().expect("refuse"), CarrierError::UnpinnedPeer);
    }

    #[test]
    fn bad_magic_is_refused() {
        let mut bytes = set_with(&[K], &[]).encode();
        bytes[0] = b'X';
        assert_eq!(
            PinSet::decode(&bytes).err().expect("refuse"),
            CarrierError::MalformedFrame
        );
    }

    #[test]
    fn truncated_and_trailing_bytes_are_refused() {
        let bytes = set_with(&[K, OTHER], &[]).encode();
        for cut in [1, MAGIC.len() + 1, bytes.len() - 1] {
            assert_eq!(
                PinSet::decode(&bytes[..cut]).err().expect("refuse"),
                CarrierError::MalformedFrame,
                "cut at {cut}"
            );
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert_eq!(
            PinSet::decode(&longer).err().expect("refuse"),
            CarrierError::MalformedFrame
        );
    }

    #[test]
    fn save_and_load_round_trip_on_disk() {
        let dir = std::env::temp_dir().join(format!("kcarrier-pins-{}", std::process::id()));
        let path = dir.join("pins.bin");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let p = set_with(&[K, OTHER], &[K]);
        p.save(&path).expect("save");
        let back = PinSet::load(&path).expect("load");
        assert!(back.check(&OTHER).is_ok());
        assert_eq!(back.check(&K).err().expect("refuse"), CarrierError::RevokedPeer);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_of_a_missing_file_is_an_error() {
        let path = std::env::temp_dir().join(format!(
            "kcarrier-pins-missing-{}-pins.bin",
            std::process::id()
        ));
        assert!(PinSet::load(&path).is_err());
    }
}
