//! Kernelodys plays at **A = 432 Hz**, not concert pitch A = 440 Hz.
//!
//! That is the law for this app: every track is heard tuned to 432.
//! Audio that was recorded at A = 440 is retuned by playing it slower,
//! at a rate of `432 / 440` (~0.9818x). Pitch drops to 432 and the track
//! runs ~1.85% longer. That is the honest trade: this retune does **not**
//! preserve tempo. A tempo-preserving pitch shift (phase vocoder or
//! similar) is future work, not something this module pretends to do.
//!
//! The entry point for playback is [`play_432`]: feed it decoded PCM
//! samples (mono `f32`, any sample rate) and it returns the same audio
//! retuned to 432. The caller sends the result to the audio device.
//! Nothing here decodes files or touches an audio device; this module
//! is pure sample math, so it can be tested exactly.

/// Concert pitch: A4 = 440 Hz. This is what most recordings are tuned to.
pub const A4_STANDARD_HZ: f64 = 440.0;
/// Kernelodys pitch: A4 = 432 Hz. This is what the listener hears.
pub const A4_KERNEL_HZ: f64 = 432.0;
/// Playback rate that retunes A440 audio down to A432.
pub const RETUNE_RATIO: f64 = A4_KERNEL_HZ / A4_STANDARD_HZ; // 0.981818...
/// How much longer a retuned track runs, as a factor.
pub const DURATION_FACTOR: f64 = A4_STANDARD_HZ / A4_KERNEL_HZ; // 1.018518...

/// Retune one frequency from concert pitch to Kernelodys pitch.
/// `retune_frequency(440.0)` is `432.0`.
pub fn retune_frequency(freq_440: f64) -> f64 {
    freq_440 * RETUNE_RATIO
}

/// Equal-tempered frequency of a MIDI note number at a given A4.
/// MIDI 69 is A4, 81 is A5, 57 is A3.
pub fn note_frequency(midi_note: i32, a4_hz: f64) -> f64 {
    a4_hz * 2f64.powf((midi_note - 69) as f64 / 12.0)
}

/// Frequency of a MIDI note as Kernelodys plays it (A4 = 432 Hz).
pub fn note_frequency_432(midi_note: i32) -> f64 {
    note_frequency(midi_note, A4_KERNEL_HZ)
}

/// How long a track runs when retuned to 432 Hz, in seconds.
pub fn retuned_duration_secs(duration_secs: f64) -> f64 {
    duration_secs * DURATION_FACTOR
}

/// Retune decoded PCM audio from A440 to A432 by linear-interpolation
/// resampling. The output is longer than the input by the duration
/// factor; a pure tone at 440 Hz comes out at 432 Hz.
///
/// An empty input returns an empty output.
pub fn resample_to_432(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let out_len = (samples.len() as f64 * DURATION_FACTOR).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    let last = *samples.last().unwrap();
    for i in 0..out_len {
        // Position in the input. Advances by 432/440 per output sample,
        // so the wave stretches and its pitch drops to 432.
        let pos = i as f64 * RETUNE_RATIO;
        let idx = pos.floor() as usize;
        let frac = pos - idx as f64;
        let a = samples[idx];
        let b = samples.get(idx + 1).copied().unwrap_or(last);
        out.push((a as f64 * (1.0 - frac) + b as f64 * frac) as f32);
    }
    out
}

/// Playback entry point: retune decoded PCM samples to 432 Hz.
/// The caller feeds decoded audio in and sends the returned samples
/// to the audio device. See the module docs for the tempo trade.
pub fn play_432(samples: &[f32]) -> Vec<f32> {
    resample_to_432(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq_hz: f64, sample_rate: f64, secs: f64) -> Vec<f32> {
        let n = (sample_rate * secs) as usize;
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq_hz * i as f64 / sample_rate).sin() as f32)
            .collect()
    }

    /// Dominant frequency of a mono sine, from zero crossings.
    fn measured_freq(samples: &[f32], sample_rate: f64) -> f64 {
        let mut crossings = 0usize;
        for w in samples.windows(2) {
            if (w[0] < 0.0 && w[1] >= 0.0) || (w[0] >= 0.0 && w[1] < 0.0) {
                crossings += 1;
            }
        }
        let seconds = samples.len() as f64 / sample_rate;
        crossings as f64 / 2.0 / seconds
    }

    #[test]
    fn retune_ratio_is_432_over_440() {
        assert!((RETUNE_RATIO - 432.0 / 440.0).abs() < 1e-12);
        assert!((retune_frequency(440.0) - 432.0).abs() < 1e-9);
        assert!((retune_frequency(880.0) - 864.0).abs() < 1e-9);
    }

    #[test]
    fn note_frequencies_at_432() {
        assert!((note_frequency_432(69) - 432.0).abs() < 1e-9);
        assert!((note_frequency_432(81) - 864.0).abs() < 1e-9);
        assert!((note_frequency_432(57) - 216.0).abs() < 1e-9);
        // Standard pitch is untouched by the 432 table.
        assert!((note_frequency(69, A4_STANDARD_HZ) - 440.0).abs() < 1e-9);
    }

    #[test]
    fn retuned_track_runs_longer() {
        let d = retuned_duration_secs(100.0);
        assert!((d - 100.0 * 440.0 / 432.0).abs() < 1e-9);
        assert!(d > 100.0 && d < 102.0);
    }

    #[test]
    fn resample_keeps_silence_and_level() {
        assert!(resample_to_432(&[]).is_empty());
        let silence = vec![0.0f32; 4410];
        assert!(resample_to_432(&silence).iter().all(|&s| s == 0.0));
        let level = vec![0.5f32; 1000];
        let out = resample_to_432(&level);
        assert!(out.iter().all(|&s| (s - 0.5).abs() < 1e-6));
    }

    #[test]
    fn resample_stretches_output_by_duration_factor() {
        let input = vec![0.0f32; 4320];
        let out = resample_to_432(&input);
        assert_eq!(out.len(), (4320.0f64 * 440.0 / 432.0).ceil() as usize);
    }

    #[test]
    fn a_440_sine_comes_out_at_432() {
        let sample_rate = 44_100.0;
        let input = sine(440.0, sample_rate, 1.0);
        assert!((measured_freq(&input, sample_rate) - 440.0).abs() < 1.0);
        let out = play_432(&input);
        let got = measured_freq(&out, sample_rate);
        assert!(
            (got - 432.0).abs() < 2.0,
            "retuned sine measured at {got} Hz, expected ~432 Hz"
        );
    }
}
