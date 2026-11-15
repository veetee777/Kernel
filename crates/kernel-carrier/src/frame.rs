//! Binary frame layer.
//!
//! Header, 12 bytes, length big-endian:
//!
//! ```text
//! offset  size  field
//! 0       4     magic   = b"KCAR"
//! 4       1     version = 1
//! 5       1     kind    (Hello=1, Response=2, Confirm=3, Data=4, Rekey=5, Cookie=6)
//! 6       2     reserved, must be zero
//! 8       4     body length, u32 big-endian, at most MAX_BODY
//! ```
//!
//! The body is binary. Nothing here is JSON, and nothing here touches disk.

use crate::error::CarrierError;

pub const MAGIC: [u8; 4] = *b"KCAR";
pub const VERSION: u8 = 1;
pub const HEADER_LEN: usize = 12;
pub const MAX_BODY: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Hello = 1,
    Response = 2,
    Confirm = 3,
    Data = 4,
    /// Phase 2: a rekey request or response, sealed under the keys in force.
    Rekey = 5,
    /// Phase 2: the cookie challenge (responder) and the cookie echo
    /// (initiator), before any Hello is read.
    Cookie = 6,
}

impl Kind {
    fn from_byte(b: u8) -> Result<Kind, CarrierError> {
        match b {
            1 => Ok(Kind::Hello),
            2 => Ok(Kind::Response),
            3 => Ok(Kind::Confirm),
            4 => Ok(Kind::Data),
            5 => Ok(Kind::Rekey),
            6 => Ok(Kind::Cookie),
            _ => Err(CarrierError::UnexpectedKind),
        }
    }
}

/// A parsed header. Keeps the exact 12 bytes it came from, so callers can
/// use them as associated data without re-encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub kind: Kind,
    pub body_len: usize,
    raw: [u8; HEADER_LEN],
}

impl Header {
    /// The exact header bytes as they appeared on the wire.
    pub fn raw(&self) -> &[u8; HEADER_LEN] {
        &self.raw
    }
}

/// Build a header for a body of `body_len` bytes.
pub fn header_bytes(kind: Kind, body_len: usize) -> Result<[u8; HEADER_LEN], CarrierError> {
    if body_len > MAX_BODY {
        return Err(CarrierError::Oversize);
    }
    let len = u32::try_from(body_len).map_err(|_| CarrierError::Oversize)?;
    let mut h = [0u8; HEADER_LEN];
    h[0..4].copy_from_slice(&MAGIC);
    h[4] = VERSION;
    h[5] = kind as u8;
    // h[6..8] stay zero (reserved).
    h[8..12].copy_from_slice(&len.to_be_bytes());
    Ok(h)
}

/// Encode a whole frame: header followed by body.
pub fn encode(kind: Kind, body: &[u8]) -> Result<Vec<u8>, CarrierError> {
    let mut out = Vec::with_capacity(HEADER_LEN + body.len());
    out.extend_from_slice(&header_bytes(kind, body.len())?);
    out.extend_from_slice(body);
    Ok(out)
}

/// Parse the first 12 bytes of `bytes` as a header. Checks magic, version,
/// kind, reserved bytes, and the body bound. Does not check the body length.
pub fn decode_header(bytes: &[u8]) -> Result<Header, CarrierError> {
    if bytes.len() < HEADER_LEN {
        return Err(CarrierError::MalformedFrame);
    }
    let mut raw = [0u8; HEADER_LEN];
    raw.copy_from_slice(&bytes[..HEADER_LEN]);
    if &raw[0..4] != &MAGIC[..] {
        return Err(CarrierError::MalformedFrame);
    }
    if raw[4] != VERSION {
        return Err(CarrierError::UnsupportedVersion);
    }
    let kind = Kind::from_byte(raw[5])?;
    if raw[6] != 0 || raw[7] != 0 {
        return Err(CarrierError::MalformedFrame);
    }
    let body_len = u32::from_be_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;
    if body_len > MAX_BODY {
        return Err(CarrierError::Oversize);
    }
    Ok(Header { kind, body_len, raw })
}

/// Parse one whole frame. The buffer must hold exactly one header and
/// exactly `body_len` body bytes.
pub fn decode(bytes: &[u8]) -> Result<(Header, &[u8]), CarrierError> {
    let header = decode_header(bytes)?;
    let body = &bytes[HEADER_LEN..];
    if body.len() != header.body_len {
        return Err(CarrierError::MalformedFrame);
    }
    Ok((header, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flip(frame: &[u8], idx: usize, value: u8) -> Vec<u8> {
        let mut v = frame.to_vec();
        v[idx] = value;
        v
    }

    #[test]
    fn round_trip() {
        let body = b"hello carrier";
        let f = encode(Kind::Data, body).expect("encode");
        assert_eq!(f.len(), HEADER_LEN + body.len());
        let (h, b) = decode(&f).expect("decode");
        assert_eq!(h.kind, Kind::Data);
        assert_eq!(h.body_len, body.len());
        assert_eq!(b, &body[..]);
    }

    #[test]
    fn header_layout_is_fixed() {
        let f = encode(Kind::Hello, &[7u8; 3]).expect("encode");
        assert_eq!(&f[0..4], &b"KCAR"[..]);
        assert_eq!(f[4], 1);
        assert_eq!(f[5], 1);
        assert_eq!(&f[6..8], &[0u8, 0u8][..]);
        assert_eq!(&f[8..12], &[0u8, 0u8, 0u8, 3u8][..]);
    }

    #[test]
    fn refuses_bad_magic() {
        let f = encode(Kind::Data, b"x").expect("encode");
        let bad = flip(&f, 0, b'X');
        assert_eq!(decode(&bad).err().expect("refuse"), CarrierError::MalformedFrame);
    }

    #[test]
    fn refuses_unsupported_version() {
        let f = encode(Kind::Data, b"x").expect("encode");
        let bad = flip(&f, 4, 2);
        assert_eq!(decode(&bad).err().expect("refuse"), CarrierError::UnsupportedVersion);
    }

    #[test]
    fn refuses_unknown_kind() {
        let f = encode(Kind::Data, b"x").expect("encode");
        let bad = flip(&f, 5, 9);
        assert_eq!(decode(&bad).err().expect("refuse"), CarrierError::UnexpectedKind);
    }

    #[test]
    fn refuses_nonzero_reserved_bytes() {
        let f = encode(Kind::Data, b"x").expect("encode");
        let bad = flip(&f, 6, 1);
        assert_eq!(decode(&bad).err().expect("refuse"), CarrierError::MalformedFrame);
    }

    #[test]
    fn refuses_length_mismatch_both_ways() {
        let f = encode(Kind::Data, b"abc").expect("encode");
        let mut longer = f.clone();
        longer.push(0);
        assert_eq!(decode(&longer).err().expect("refuse"), CarrierError::MalformedFrame);
        let shorter = &f[..f.len() - 1];
        assert_eq!(decode(shorter).err().expect("refuse"), CarrierError::MalformedFrame);
    }

    #[test]
    fn refuses_truncated_header() {
        assert_eq!(decode(&[0u8; 5]).err().expect("refuse"), CarrierError::MalformedFrame);
    }

    #[test]
    fn refuses_oversize_claim_in_header() {
        // Claims 65_537 body bytes: 0x0001_0001.
        let h = [b'K', b'C', b'A', b'R', 1, 4, 0, 0, 0x00, 0x01, 0x00, 0x01];
        assert_eq!(decode_header(&h).err().expect("refuse"), CarrierError::Oversize);
    }

    #[test]
    fn refuses_oversize_on_encode() {
        let big = vec![0u8; MAX_BODY + 1];
        assert_eq!(encode(Kind::Data, &big).err().expect("refuse"), CarrierError::Oversize);
    }

    #[test]
    fn phase2_kinds_round_trip() {
        for (kind, byte) in [(Kind::Rekey, 5u8), (Kind::Cookie, 6u8)] {
            let f = encode(kind, b"p2").expect("encode");
            assert_eq!(f[5], byte);
            let (h, b) = decode(&f).expect("decode");
            assert_eq!(h.kind, kind);
            assert_eq!(b, b"p2");
        }
    }

    #[test]
    fn all_kind_bytes_decode() {
        let want = [
            (1u8, Kind::Hello),
            (2, Kind::Response),
            (3, Kind::Confirm),
            (4, Kind::Data),
            (5, Kind::Rekey),
            (6, Kind::Cookie),
        ];
        for (b, kind) in want {
            assert_eq!(Kind::from_byte(b).expect("known kind"), kind);
        }
        assert_eq!(
            Kind::from_byte(7).err().expect("refuse"),
            CarrierError::UnexpectedKind
        );
    }
}
