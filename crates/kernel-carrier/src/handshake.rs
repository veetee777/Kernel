//! Mutual authentication: a SIGMA-style handshake with three messages.
//!
//! ```text
//! I -> R  Hello     eph_i | nonce_i | pub_i | pub_r_expected   (128 bytes)
//! R -> I  Response  eph_r | nonce_r | pub_r | sig_r            (160 bytes)
//! I -> R  Confirm   sig_i                                       (64 bytes)
//! ```
//!
//! sig_r = Ed25519_R("KCAR-HS2-SIG-v1\0" || frame1 || eph_r | nonce_r | pub_r)
//! sig_i = Ed25519_I("KCAR-HS3-SIG-v1\0" || frame1 || frame2)
//! th    = SHA-256(frame1 || frame2)
//! shared = X25519(eph_i_secret, eph_r) = X25519(eph_r_secret, eph_i)
//! i2r, r2i = HKDF-SHA256(salt = th, ikm = shared, info = label), 32 bytes each
//!
//! Both sides must have the other's key pinned. The ephemeral keys give the
//! session forward secrecy. The signatures bind both ephemeral keys, both
//! identities, and the full transcript, so no message can be replayed into a
//! different session or altered in transit.
//!
//! Known limit: pub_i travels in the clear in Hello, so a passive observer
//! learns which identity is connecting. Identity hiding is not provided.
//! Known limit: no cookie or puzzle before the responder's DH, so Hello
//! floods cost the responder one DH and one signature each.

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use getrandom::getrandom;
use hkdf::Hkdf;
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::error::CarrierError;
use crate::frame::{self, Header, Kind};
use crate::kex::{self, Secret};
use crate::pins::PinSet;
use crate::session::{Role, Session};

pub(crate) const HELLO_BODY: usize = 128;
pub(crate) const RESPONSE_BODY: usize = 160;
const RESPONSE_SIGNED_BODY: usize = 96;
pub(crate) const CONFIRM_BODY: usize = 64;

const LABEL_RESPONSE_SIG: &[u8] = b"KCAR-HS2-SIG-v1\0";
const LABEL_CONFIRM_SIG: &[u8] = b"KCAR-HS3-SIG-v1\0";
const LABEL_I2R: &[u8] = b"KCAR-v1 key i2r";
const LABEL_R2I: &[u8] = b"KCAR-v1 key r2i";

fn arr32(b: &[u8]) -> Result<[u8; 32], CarrierError> {
    b.try_into().map_err(|_| CarrierError::MalformedFrame)
}

fn sig_from(b: &[u8]) -> Result<Signature, CarrierError> {
    let arr: [u8; 64] = b.try_into().map_err(|_| CarrierError::MalformedFrame)?;
    Ok(Signature::from_bytes(&arr))
}

fn concat(parts: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for p in parts {
        v.extend_from_slice(p);
    }
    v
}

fn expect_kind(h: &Header, want: Kind) -> Result<(), CarrierError> {
    if h.kind == want {
        Ok(())
    } else {
        Err(CarrierError::UnexpectedKind)
    }
}

fn transcript(frame1: &[u8], frame2: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(frame1);
    h.update(frame2);
    h.finalize().into()
}

fn derive(shared: &[u8; 32], th: &[u8; 32]) -> Result<(Secret, Secret), CarrierError> {
    let hk = Hkdf::<Sha256>::new(Some(&th[..]), &shared[..]);
    let mut i2r = Zeroizing::new([0u8; 32]);
    let mut r2i = Zeroizing::new([0u8; 32]);
    hk.expand(LABEL_I2R, &mut i2r[..])
        .map_err(|_| CarrierError::Internal)?;
    hk.expand(LABEL_R2I, &mut r2i[..])
        .map_err(|_| CarrierError::Internal)?;
    Ok((i2r, r2i))
}

/// Initiator state between messages. Holds the secret ephemeral key.
pub struct Initiator {
    my_key: SigningKey,
    peer_pub: [u8; 32],
    eph: Secret,
    frame1: Vec<u8>,
}

impl Initiator {
    /// Start a handshake with a pinned responder. Returns the Hello bytes.
    pub fn start(
        my_key: &SigningKey,
        peer_pub: [u8; 32],
        pins: &PinSet,
    ) -> Result<(Initiator, Vec<u8>), CarrierError> {
        pins.check(&peer_pub)?;
        let (eph, eph_pub) = kex::ephemeral()?;
        let mut nonce = [0u8; 32];
        getrandom(&mut nonce).map_err(|_| CarrierError::Random)?;
        let init_pub = my_key.verifying_key().to_bytes();

        let body = concat(&[&eph_pub[..], &nonce[..], &init_pub[..], &peer_pub[..]]);
        let frame1 = frame::encode(Kind::Hello, &body)?;
        let state = Initiator {
            my_key: my_key.clone(),
            peer_pub,
            eph,
            frame1: frame1.clone(),
        };
        Ok((state, frame1))
    }

    /// Consume the Response, verify the responder, and produce the Confirm
    /// bytes together with the initiator's session.
    pub fn finish(
        self,
        frame2: &[u8],
        pins: &PinSet,
    ) -> Result<(Session, Vec<u8>), CarrierError> {
        let (hdr, body) = frame::decode(frame2)?;
        expect_kind(&hdr, Kind::Response)?;
        if body.len() != RESPONSE_BODY {
            return Err(CarrierError::MalformedFrame);
        }
        let resp_eph = arr32(&body[0..32])?;
        let resp_pub = arr32(&body[64..96])?;
        if resp_pub != self.peer_pub {
            return Err(CarrierError::WrongResponder);
        }
        pins.check(&resp_pub)?;

        let vk = VerifyingKey::from_bytes(&resp_pub).map_err(|_| CarrierError::InvalidKey)?;
        let sig = sig_from(&body[96..160])?;
        let signed = concat(&[
            LABEL_RESPONSE_SIG,
            &self.frame1[..],
            &body[..RESPONSE_SIGNED_BODY],
        ]);
        vk.verify_strict(&signed, &sig)
            .map_err(|_| CarrierError::BadSignature)?;

        let shared = kex::shared(&self.eph, &resp_eph)?;
        let th = transcript(&self.frame1, frame2);
        let (i2r, r2i) = derive(&shared, &th)?;

        let confirm_signed = concat(&[LABEL_CONFIRM_SIG, &self.frame1[..], frame2]);
        let confirm_sig = self.my_key.sign(&confirm_signed).to_bytes();
        let frame3 = frame::encode(Kind::Confirm, &confirm_sig)?;
        Ok((Session::new(Role::Initiator, i2r, r2i, resp_pub), frame3))
    }
}

/// Responder state between messages. Holds the shared secret and both
/// transcript frames until the Confirm arrives.
pub struct Responder {
    init_pub: [u8; 32],
    shared: Secret,
    frame1: Vec<u8>,
    frame2: Vec<u8>,
}

impl Responder {
    /// Answer a Hello addressed to us from a pinned initiator. Returns the
    /// Response bytes.
    pub fn accept(
        my_key: &SigningKey,
        pins: &PinSet,
        frame1: &[u8],
    ) -> Result<(Responder, Vec<u8>), CarrierError> {
        let (hdr, body) = frame::decode(frame1)?;
        expect_kind(&hdr, Kind::Hello)?;
        if body.len() != HELLO_BODY {
            return Err(CarrierError::MalformedFrame);
        }
        let init_eph = arr32(&body[0..32])?;
        let init_pub = arr32(&body[64..96])?;
        let target = arr32(&body[96..128])?;

        let my_pub = my_key.verifying_key().to_bytes();
        if target != my_pub {
            return Err(CarrierError::WrongResponder);
        }
        pins.check(&init_pub)?;

        let (eph, eph_pub) = kex::ephemeral()?;
        let shared = kex::shared(&eph, &init_eph)?;
        let mut nonce = [0u8; 32];
        getrandom(&mut nonce).map_err(|_| CarrierError::Random)?;

        let unsigned = concat(&[&eph_pub[..], &nonce[..], &my_pub[..]]);
        let sig = my_key
            .sign(&concat(&[LABEL_RESPONSE_SIG, frame1, &unsigned[..]]))
            .to_bytes();
        let frame2 = frame::encode(Kind::Response, &concat(&[&unsigned[..], &sig[..]]))?;

        let state = Responder {
            init_pub,
            shared,
            frame1: frame1.to_vec(),
            frame2: frame2.clone(),
        };
        Ok((state, frame2))
    }

    /// Verify the Confirm and produce the responder's session.
    pub fn finish(self, frame3: &[u8], pins: &PinSet) -> Result<Session, CarrierError> {
        let (hdr, body) = frame::decode(frame3)?;
        expect_kind(&hdr, Kind::Confirm)?;
        if body.len() != CONFIRM_BODY {
            return Err(CarrierError::MalformedFrame);
        }
        pins.check(&self.init_pub)?;

        let vk = VerifyingKey::from_bytes(&self.init_pub).map_err(|_| CarrierError::InvalidKey)?;
        let sig = sig_from(body)?;
        let signed = concat(&[LABEL_CONFIRM_SIG, &self.frame1[..], &self.frame2[..]]);
        vk.verify_strict(&signed, &sig)
            .map_err(|_| CarrierError::BadSignature)?;

        let th = transcript(&self.frame1, &self.frame2);
        let (i2r, r2i) = derive(&self.shared, &th)?;
        Ok(Session::new(Role::Responder, i2r, r2i, self.init_pub))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn public(k: &SigningKey) -> [u8; 32] {
        k.verifying_key().to_bytes()
    }

    fn pins_with(keys: &[&SigningKey]) -> PinSet {
        let mut p = PinSet::new();
        for k in keys {
            p.pin(public(k)).expect("pin");
        }
        p
    }

    #[test]
    fn handshake_completes_and_sessions_carry_data_both_ways() {
        let a = key(1);
        let b = key(2);
        let pins_a = pins_with(&[&b]);
        let pins_b = pins_with(&[&a]);

        let (ini, f1) = Initiator::start(&a, public(&b), &pins_a).expect("start");
        let (resp, f2) = Responder::accept(&b, &pins_b, &f1).expect("accept");
        let (mut ini_s, f3) = ini.finish(&f2, &pins_a).expect("finish");
        let mut resp_s = resp.finish(&f3, &pins_b).expect("confirm");

        assert_eq!(ini_s.peer(), public(&b));
        assert_eq!(resp_s.peer(), public(&a));

        let up = ini_s.seal(b"from initiator").expect("seal");
        assert_eq!(resp_s.open(&up).expect("open"), b"from initiator".to_vec());
        let down = resp_s.seal(b"from responder").expect("seal");
        assert_eq!(ini_s.open(&down).expect("open"), b"from responder".to_vec());
    }

    #[test]
    fn two_handshakes_produce_independent_sessions() {
        let a = key(1);
        let b = key(2);
        let pins_a = pins_with(&[&b]);
        let pins_b = pins_with(&[&a]);

        let run = || {
            let (ini, f1) = Initiator::start(&a, public(&b), &pins_a).expect("start");
            let (resp, f2) = Responder::accept(&b, &pins_b, &f1).expect("accept");
            let (ini_s, f3) = ini.finish(&f2, &pins_a).expect("finish");
            let resp_s = resp.finish(&f3, &pins_b).expect("confirm");
            (ini_s, resp_s)
        };
        let (mut first_i, mut first_r) = run();
        let (_second_i, mut second_r) = run();

        // A frame from the first session must open in the first responder
        // and must not open in the second, which has different keys.
        let f = first_i.seal(b"x").expect("seal");
        assert!(first_r.open(&f).is_ok());
        assert_eq!(second_r.open(&f).err().expect("refuse"), CarrierError::AuthFailed);
    }

    #[test]
    fn unpinned_responder_is_refused_before_anything_is_sent() {
        let a = key(1);
        let b = key(2);
        let empty = PinSet::new();
        let err = Initiator::start(&a, public(&b), &empty)
            .err()
            .expect("refuse");
        assert_eq!(err, CarrierError::UnpinnedPeer);
    }

    #[test]
    fn unpinned_initiator_is_refused_at_accept() {
        let a = key(1);
        let b = key(2);
        let pins_a = pins_with(&[&b]);
        let (_ini, f1) = Initiator::start(&a, public(&b), &pins_a).expect("start");
        let empty = PinSet::new();
        let err = Responder::accept(&b, &empty, &f1).err().expect("refuse");
        assert_eq!(err, CarrierError::UnpinnedPeer);
    }

    #[test]
    fn responder_with_a_different_key_is_refused() {
        let a = key(1);
        let b = key(2);
        let c = key(3);
        let pins_a = pins_with(&[&b]);
        let (_ini, f1) = Initiator::start(&a, public(&b), &pins_a).expect("start");
        let pins_c = pins_with(&[&a]);
        let err = Responder::accept(&c, &pins_c, &f1).err().expect("refuse");
        assert_eq!(err, CarrierError::WrongResponder);
    }

    #[test]
    fn tampered_response_signature_is_refused() {
        let a = key(1);
        let b = key(2);
        let pins_a = pins_with(&[&b]);
        let pins_b = pins_with(&[&a]);
        let (ini, f1) = Initiator::start(&a, public(&b), &pins_a).expect("start");
        let (_resp, mut f2) = Responder::accept(&b, &pins_b, &f1).expect("accept");
        let last = f2.len() - 1;
        f2[last] ^= 0x01;
        let err = ini.finish(&f2, &pins_a).err().expect("refuse");
        assert_eq!(err, CarrierError::BadSignature);
    }

    #[test]
    fn man_in_the_middle_altering_the_hello_is_caught() {
        // The attacker swaps the initiator's ephemeral key. The responder
        // answers the altered Hello. The initiator's transcript differs, so
        // the responder's signature does not verify.
        let a = key(1);
        let b = key(2);
        let pins_a = pins_with(&[&b]);
        let pins_b = pins_with(&[&a]);
        let (ini, f1) = Initiator::start(&a, public(&b), &pins_a).expect("start");
        let mut altered = f1.clone();
        altered[frame::HEADER_LEN] ^= 0x01;
        let (_resp, f2) = Responder::accept(&b, &pins_b, &altered).expect("responder answers");
        let err = ini.finish(&f2, &pins_a).err().expect("refuse");
        assert_eq!(err, CarrierError::BadSignature);
    }

    #[test]
    fn tampered_confirm_is_refused() {
        let a = key(1);
        let b = key(2);
        let pins_a = pins_with(&[&b]);
        let pins_b = pins_with(&[&a]);
        let (ini, f1) = Initiator::start(&a, public(&b), &pins_a).expect("start");
        let (resp, f2) = Responder::accept(&b, &pins_b, &f1).expect("accept");
        let (_s, mut f3) = ini.finish(&f2, &pins_a).expect("finish");
        let last = f3.len() - 1;
        f3[last] ^= 0x01;
        let err = resp.finish(&f3, &pins_b).err().expect("refuse");
        assert_eq!(err, CarrierError::BadSignature);
    }

    #[test]
    fn key_revoked_after_accept_is_refused_at_confirm() {
        let a = key(1);
        let b = key(2);
        let pins_a = pins_with(&[&b]);
        let mut pins_b = pins_with(&[&a]);
        let (ini, f1) = Initiator::start(&a, public(&b), &pins_a).expect("start");
        let (resp, f2) = Responder::accept(&b, &pins_b, &f1).expect("accept");
        let (_s, f3) = ini.finish(&f2, &pins_a).expect("finish");
        pins_b.revoke(public(&a));
        let err = resp.finish(&f3, &pins_b).err().expect("refuse");
        assert_eq!(err, CarrierError::RevokedPeer);
    }

    #[test]
    fn zero_ephemeral_key_is_refused() {
        let a = key(1);
        let b = key(2);
        let pins_b = pins_with(&[&a]);
        let body = concat(&[
            &[0u8; 32][..],
            &[5u8; 32][..],
            &public(&a)[..],
            &public(&b)[..],
        ]);
        let f1 = frame::encode(Kind::Hello, &body).expect("encode");
        let err = Responder::accept(&b, &pins_b, &f1).err().expect("refuse");
        assert_eq!(err, CarrierError::WeakSecret);
    }

    #[test]
    fn wrong_frame_kind_at_a_step_is_refused() {
        let a = key(1);
        let b = key(2);
        let pins_b = pins_with(&[&a]);
        let body = vec![0u8; HELLO_BODY];
        let not_hello = frame::encode(Kind::Data, &body).expect("encode");
        let err = Responder::accept(&b, &pins_b, &not_hello)
            .err()
            .expect("refuse");
        assert_eq!(err, CarrierError::UnexpectedKind);
    }
}
