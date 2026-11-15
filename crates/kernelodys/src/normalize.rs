//! Peak normalization for Kernelodys.
//!
//! Tracks are recorded at different loudnesses. Normalization computes
//! one gain per track from its peak amplitude, so a loud track and a
//! quiet track play at the same peak level. The gain is applied by the
//! playback engine ([`crate::playback::PlaybackEngine::set_gain`]).
//!
//! The target peak is full scale (`32767`). A track whose peak is
//! already full scale gets gain `1.0`; a quiet track gets a gain above
//! `1.0` that brings its peak up to full scale. The gain is capped at
//! [`MAX_GAIN`] so a near-silent track does not get amplified into
//! noise, and the engine clamps every scaled sample to the `i16`
//! range, so the output can never clip past full scale. Linear
//! interpolation in the 432 Hz retune cannot raise a peak (each
//! interpolated value lies between two original samples), so a gain
//! computed on the decoded samples stays safe after retuning.

/// The peak level normalized tracks aim for: full scale.
pub const TARGET_PEAK: i16 = 32767;
/// Maximum gain normalization ever applies. A track quieter than
/// `TARGET_PEAK / MAX_GAIN` stays below full scale rather than having
/// its noise floor amplified without limit.
pub const MAX_GAIN: f32 = 20.0;

/// Peak amplitude of interleaved `i16` samples: the largest absolute
/// value, as a `u32` so `-32768` counts as `32768`. Silence peaks at 0.
pub fn peak_amplitude(samples: &[i16]) -> u32 {
    samples
        .iter()
        .map(|&s| (s as i32).unsigned_abs())
        .max()
        .unwrap_or(0)
}

/// Gain that brings a track's peak up to [`TARGET_PEAK`].
///
/// - Silence (peak 0) gets `1.0`: there is nothing to normalize and
///   amplifying silence would only amplify nothing.
/// - A track already at full scale gets `1.0`.
/// - Anything else gets `TARGET_PEAK / peak`, capped at [`MAX_GAIN`].
///
/// The result is always finite and in `[0.0, MAX_GAIN]`. Apply it with
/// [`apply_gain`], which clamps every sample, so the output peak never
/// exceeds full scale.
pub fn normalize_gain(samples: &[i16]) -> f32 {
    let peak = peak_amplitude(samples);
    if peak == 0 {
        return 1.0;
    }
    (TARGET_PEAK as f32 / peak as f32).min(MAX_GAIN)
}

/// Scale samples by `gain`, rounding to the nearest integer and
/// clamping to the `i16` range, so output can never clip past full
/// scale. A non-finite or negative gain is treated as `1.0`; a gain
/// above [`MAX_GAIN`] is clamped to it. Gain `1.0` returns the samples
/// unchanged.
pub fn apply_gain(samples: &[i16], gain: f32) -> Vec<i16> {
    let gain = if gain.is_finite() && gain >= 0.0 {
        gain.min(MAX_GAIN)
    } else {
        1.0
    };
    if gain == 1.0 {
        return samples.to_vec();
    }
    samples
        .iter()
        .map(|&s| (s as f32 * gain).round().clamp(-32768.0, 32767.0) as i16)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_gets_gain_one() {
        assert_eq!(normalize_gain(&[]), 1.0);
        assert_eq!(normalize_gain(&[0, 0, 0]), 1.0);
        assert_eq!(peak_amplitude(&[]), 0);
        assert_eq!(peak_amplitude(&[0, -5, 3]), 5);
        // -32768 counts as 32768.
        assert_eq!(peak_amplitude(&[-32768]), 32768);
    }

    #[test]
    fn full_scale_gets_gain_one() {
        assert_eq!(normalize_gain(&[32767]), 1.0);
        assert_eq!(normalize_gain(&[-32768, 100]), 32767.0 / 32768.0);
    }

    #[test]
    fn quiet_track_gain_bring_peak_to_full_scale() {
        let quiet = vec![10000i16, -10000, 5000, -8000];
        let gain = normalize_gain(&quiet);
        assert!((gain - 32767.0 / 10000.0).abs() < 1e-6);
        let scaled = apply_gain(&quiet, gain);
        assert_eq!(peak_amplitude(&scaled), 32767);
    }

    #[test]
    fn loud_and_quiet_normalize_to_the_same_peak() {
        let loud = vec![30000i16, -30000, 20000];
        let quiet = vec![5000i16, -5000, 3000];
        let loud_out = apply_gain(&loud, normalize_gain(&loud));
        let quiet_out = apply_gain(&quiet, normalize_gain(&quiet));
        assert_eq!(peak_amplitude(&loud_out), 32767);
        assert_eq!(peak_amplitude(&quiet_out), 32767);
    }

    #[test]
    fn gain_is_capped_and_output_never_clips() {
        // Peak 100 would want gain 327.67; the cap is 20.
        let faint = vec![100i16, -100];
        assert_eq!(normalize_gain(&faint), MAX_GAIN);
        // Even an absurd gain clamps every sample to the i16 range.
        let hot = vec![30000i16, -30000, 32767, -32768];
        let scaled = apply_gain(&hot, 100.0);
        // -32768 stays -32768 (its magnitude is 32768, the largest
        // representable); nothing exceeds the i16 range.
        assert_eq!(scaled.iter().max(), Some(&32767));
        assert_eq!(scaled.iter().min(), Some(&-32768));
        assert_eq!(peak_amplitude(&scaled), 32768);
        // Non-finite and negative gains are treated as 1.0.
        assert_eq!(apply_gain(&hot, f32::NAN), hot);
        assert_eq!(apply_gain(&hot, -2.0), hot);
        assert_eq!(apply_gain(&hot, f32::INFINITY), hot);
    }
}
