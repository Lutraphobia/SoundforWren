"""harmonic_motion.py — three new perceptual lenses for Wren's pipeline.

Built 2026-05-10 PT under Walt's "explore lane" charter (ledger #1168).
Zero new dependencies — all three features use librosa 0.11.0 primitives
that were already installed but unused.

Three lenses:

1. TONNETZ TRAJECTORY
   Maps the song through 6-D tonal centroid space. The total path-length
   (summed Euclidean step) is a measure of harmonic *restlessness*: how
   much the music moves through the harmonic neighborhood, even when key
   detection reports a single stable key. Catches modulations that
   chroma-averaging flattens.

2. PYIN VIBRATO
   Estimates fundamental frequency on the harmonic-enhanced (HPSS)
   signal, then characterizes the dominant vocal-range frequency band's
   periodic pitch wobble: rate (Hz), depth (cents), regularity (0-1).
   Until we have Demucs stems, this is the closest we can get to
   "delivery" without lyrics.

3. TEMPOGRAM STABILITY
   Single-tempo summary hides whether a song breathes or locks to grid.
   Tempogram gives local-tempo per frame; std-dev around the median
   tempo is the stability metric. Low std = locked (e.g. EDM); high std
   = elastic (e.g. live ballad). Also reports the tempo-shift count
   (how many times local tempo crosses ±5% of median).

Returns a dict suitable to merge into sensory_report.extract_features
output.
"""
from __future__ import annotations

import numpy as np
import librosa


def tonnetz_trajectory(y: np.ndarray, sr: int, hop_length: int = 512) -> dict:
    """6-D tonnetz trajectory metrics.

    Returns:
        path_length: total Euclidean distance traveled (per-second normalized).
        mean_step: average per-frame step size (harmonic activity rate).
        max_step: largest single step (modulation marker).
        restlessness: path_length / duration_sec (0 = static drone, >2 = very mobile).
        harmonic_band_changes: count of large steps (>2x mean) — modulation events.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=sr)  # (6, T)

    # Per-frame step in 6-D space
    diffs = np.diff(tonnetz, axis=1)
    steps = np.linalg.norm(diffs, axis=0)
    path_length = float(steps.sum())
    mean_step = float(steps.mean()) if len(steps) else 0.0
    max_step = float(steps.max()) if len(steps) else 0.0

    duration_sec = (tonnetz.shape[1] * hop_length) / float(sr)
    restlessness = path_length / max(duration_sec, 1e-6)

    threshold = mean_step * 2.0 if mean_step > 0 else 0.0
    harmonic_band_changes = int((steps > threshold).sum())

    return {
        "path_length": path_length,
        "mean_step": mean_step,
        "max_step": max_step,
        "restlessness": restlessness,
        "harmonic_band_changes": harmonic_band_changes,
        "tonnetz_shape": list(tonnetz.shape),
    }


def vibrato_summary(y_harmonic: np.ndarray, sr: int,
                    fmin: float = 80.0, fmax: float = 1000.0,
                    hop_length: int = 512) -> dict:
    """Vibrato characterization on the harmonic signal (HPSS-enhanced).

    Uses pyin to extract f0; then measures the periodic wobble in
    semitones-from-running-mean (a vibrato is typically 4-7 Hz, ±50 cents).

    Returns:
        f0_mean_hz: mean voiced f0 (Hz)
        voiced_ratio: fraction of frames where pitch was detected
        vibrato_rate_hz: dominant wobble frequency (Hz)
        vibrato_depth_cents: std of pitch around running mean, in cents
        vibrato_regularity: spectral-peak prominence of the dominant rate (0-1)
        delivery_register: "flat"|"steady"|"swung"|"intense" coarse label
    """
    # pyin returns (f0, voiced_flag, voiced_prob)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y_harmonic, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )
    voiced_ratio = float(np.mean(voiced_flag)) if len(voiced_flag) else 0.0

    f0_voiced = f0[voiced_flag]
    if len(f0_voiced) < 32:
        return {
            "f0_mean_hz": 0.0,
            "voiced_ratio": voiced_ratio,
            "vibrato_rate_hz": 0.0,
            "vibrato_depth_cents": 0.0,
            "vibrato_regularity": 0.0,
            "delivery_register": "insufficient-pitch",
        }

    f0_mean = float(np.nanmean(f0_voiced))

    # Convert to cents-from-running-mean (1-second running window)
    frame_rate = sr / hop_length  # frames per second
    win = max(int(round(frame_rate * 1.0)), 8)  # 1 second
    # Use a simple moving-average via convolution; nan-safe
    f0_filled = np.where(voiced_flag, f0, np.nan)
    cents = 1200.0 * np.log2(f0_filled / f0_mean)
    # interpolate over NaN gaps for spectral analysis
    isnan = np.isnan(cents)
    if isnan.any():
        idx = np.arange(len(cents))
        cents[isnan] = np.interp(idx[isnan], idx[~isnan], cents[~isnan]) if (~isnan).any() else 0.0
    # Detrend with running mean
    kernel = np.ones(win) / win
    running = np.convolve(cents, kernel, mode="same")
    detrended = cents - running

    # Cap to keep outliers from blowing up depth measurement
    detrended = np.clip(detrended, -300.0, 300.0)
    depth_cents = float(np.std(detrended))

    # Spectral analysis: dominant rate in 3-9 Hz vibrato band
    n = len(detrended)
    if n < 64:
        return {
            "f0_mean_hz": f0_mean,
            "voiced_ratio": voiced_ratio,
            "vibrato_rate_hz": 0.0,
            "vibrato_depth_cents": depth_cents,
            "vibrato_regularity": 0.0,
            "delivery_register": "too-short",
        }
    spec = np.abs(np.fft.rfft(detrended * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / frame_rate)
    band = (freqs >= 3.0) & (freqs <= 9.0)
    if not band.any() or spec[band].max() <= 0:
        rate = 0.0
        regularity = 0.0
    else:
        in_band_spec = spec[band]
        in_band_freqs = freqs[band]
        peak_idx = int(np.argmax(in_band_spec))
        rate = float(in_band_freqs[peak_idx])
        # regularity = peak / total in-band energy (0..1)
        total = float(in_band_spec.sum())
        regularity = float(in_band_spec[peak_idx] / total) if total > 0 else 0.0

    # Coarse delivery register
    if depth_cents < 15:
        register = "flat"
    elif depth_cents < 35 and regularity < 0.15:
        register = "steady"
    elif depth_cents < 60 and regularity >= 0.15:
        register = "swung"
    else:
        register = "intense"

    return {
        "f0_mean_hz": f0_mean,
        "voiced_ratio": voiced_ratio,
        "vibrato_rate_hz": rate,
        "vibrato_depth_cents": depth_cents,
        "vibrato_regularity": regularity,
        "delivery_register": register,
    }


def tempogram_stability(y: np.ndarray, sr: int, hop_length: int = 512) -> dict:
    """Tempo stability via tempogram local-tempo curve.

    Returns:
        median_local_tempo: median across the song
        tempo_std_bpm: std-dev of local tempo (low = locked, high = elastic)
        tempo_stability: 1 / (1 + std/median) — 1.0 = perfectly locked, ~0.5 = loose
        tempo_shift_count: # of times local tempo crosses +/-5% of median
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # Tempogram: rows = lag (BPM), cols = time
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )
    # Per-frame dominant tempo
    tempo_axis = librosa.tempo_frequencies(tempogram.shape[0], sr=sr, hop_length=hop_length)
    # Restrict to plausible range
    mask = (tempo_axis >= 40.0) & (tempo_axis <= 240.0)
    if not mask.any():
        return {
            "median_local_tempo": 0.0,
            "tempo_std_bpm": 0.0,
            "tempo_stability": 0.0,
            "tempo_shift_count": 0,
        }
    tg = tempogram[mask, :]
    ta = tempo_axis[mask]
    local_idx = np.argmax(tg, axis=0)
    local_tempo = ta[local_idx]
    median_t = float(np.median(local_tempo))
    std_t = float(np.std(local_tempo))
    stability = 1.0 / (1.0 + (std_t / max(median_t, 1e-6)))
    threshold = 0.05 * median_t
    deviation = local_tempo - median_t
    sign = np.sign(deviation)
    crossings = int(np.sum(np.abs(np.diff(sign[np.abs(deviation) > threshold])) > 0))

    return {
        "median_local_tempo": median_t,
        "tempo_std_bpm": std_t,
        "tempo_stability": stability,
        "tempo_shift_count": crossings,
    }


def harmonic_motion_report(y: np.ndarray, sr: int,
                            y_harmonic: np.ndarray | None = None,
                            hop_length: int = 512) -> dict:
    """Run all three lenses and return a single dict."""
    if y_harmonic is None:
        y_harmonic, _ = librosa.effects.hpss(y)
    return {
        "tonnetz": tonnetz_trajectory(y, sr, hop_length=hop_length),
        "vibrato": vibrato_summary(y_harmonic, sr, hop_length=hop_length),
        "tempo_stability": tempogram_stability(y, sr, hop_length=hop_length),
    }


if __name__ == "__main__":
    import sys
    import json
    import sensory_report  # local

    if len(sys.argv) < 2:
        print("usage: harmonic_motion.py <audio_path>")
        sys.exit(1)
    path = sys.argv[1]
    print(f"Loading {path}...")
    y, sr = sensory_report.load_audio(path)
    print(f"sr={sr} duration={len(y)/sr:.1f}s")
    print("Running harmonic_motion lenses...")
    out = harmonic_motion_report(y, sr)
    print(json.dumps(out, indent=2))
