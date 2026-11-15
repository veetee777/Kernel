//! X25519 ephemeral key agreement, using curve25519-dalek (already in the
//! lockfile). x25519-dalek is not in the lockfile, so it is not used.

use curve25519_dalek::constants::X25519_BASEPOINT;
use curve25519_dalek::montgomery::MontgomeryPoint;
use getrandom::getrandom;
use zeroize::Zeroizing;

use crate::error::CarrierError;

/// A 32-byte X25519 secret. Zeroized on drop.
pub type Secret = Zeroizing<[u8; 32]>;

/// Fresh ephemeral keypair: (secret, public).
pub fn ephemeral() -> Result<(Secret, [u8; 32]), CarrierError> {
    let mut secret = Zeroizing::new([0u8; 32]);
    getrandom(&mut secret[..]).map_err(|_| CarrierError::Random)?;
    let public = X25519_BASEPOINT.mul_clamped(*secret).to_bytes();
    Ok((secret, public))
}

/// Public key for a given secret. No randomness. Used by the test vector.
#[cfg(test)]
pub fn public_from_secret(secret: &[u8; 32]) -> [u8; 32] {
    X25519_BASEPOINT.mul_clamped(*secret).to_bytes()
}

/// X25519 shared secret. Refuses the all-zero output that a small-order
/// peer point produces (RFC 7748, section 6.1).
pub fn shared(secret: &Secret, peer_public: &[u8; 32]) -> Result<Secret, CarrierError> {
    let out = Zeroizing::new(MontgomeryPoint(*peer_public).mul_clamped(**secret).to_bytes());
    let mut acc = 0u8;
    for b in out.iter() {
        acc |= *b;
    }
    if acc == 0 {
        return Err(CarrierError::WeakSecret);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex32(s: &str) -> [u8; 32] {
        let mut out = [0u8; 32];
        for i in 0..32 {
            out[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex");
        }
        out
    }

    /// RFC 7748 section 6.1. If this fails, check the transcription of the
    /// hex strings against the RFC before suspecting the code.
    #[test]
    fn rfc7748_section_6_1_vector() {
        let alice_sk = hex32("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a");
        let alice_pk = hex32("8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a");
        let bob_pk = hex32("de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f");
        let expected = hex32("4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742");

        assert_eq!(public_from_secret(&alice_sk), alice_pk);
        let secret: Secret = Zeroizing::new(alice_sk);
        let got = shared(&secret, &bob_pk).expect("shared");
        assert_eq!(*got, expected);
    }

    #[test]
    fn shared_secret_is_symmetric_for_random_keys() {
        let (a_sk, a_pk) = ephemeral().expect("eph");
        let (b_sk, b_pk) = ephemeral().expect("eph");
        let ab = shared(&a_sk, &b_pk).expect("ab");
        let ba = shared(&b_sk, &a_pk).expect("ba");
        assert_eq!(*ab, *ba);
    }

    #[test]
    fn all_zero_peer_point_is_refused() {
        let (sk, _) = ephemeral().expect("eph");
        assert_eq!(shared(&sk, &[0u8; 32]).err().expect("refuse"), CarrierError::WeakSecret);
    }
}
