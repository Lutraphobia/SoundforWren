"""timbral_motion.py — two new perceptual lenses for Wren's pipeline.

Built 2026-05-11 PT under Walt's "explore lane" continuation
(ledger #1168 frontier, lenses #4 and #5). Sibling to harmonic_motion.py.
Zero new dependencies — both features use librosa 0.11.0 primitives
already installed.

Two lenses:

1. SPECTRAL FLUX
   Frame-to-frame change in the magnitude spectrum. Captures how
   *turbulent* the timbre is over time — distinct from amplitude
   (RMS) and from harmonic motion (tonnetz). High flux = constantly
   shifting tone color (busy production, layered arrangement,
   distortion crackle); low flux = steady palette (sustained pad,
   solo voice, ambient drone). Reports per-second time-series and
   summary stats including a "turbulence" coefficient and burstiness
   (how clumped the flux peaks are vs evenly distributed).

2. ATTACK ENVELOPE
   For each detected onset, characterizes the attack shape:
     - rise_ms: time from onset detection to peak amplitude
     - decay_60ms: amplitude ratio at +60ms post-peak
     - sharpness: 1 / (1 + rise_ms / 20)  (1.0 = sub-20ms percussive
       transient, ~0.5 = soft swell, ~0.1 = bowed string)
   Then summarizes the song by mean/std of these and assigns a
   coarse "attack_character" label (percussive | mixed | soft | bowed).
   This distinguishes drum-kit-driven songs from piano-driven from
   pad/bowed without needing instrument classification.

Returns a dict suitable to merge into sensory_report.extract_features
output, same shape contract as harmonic_motion.py.
"""
from __future__ import annotations

import numpy as np
import librosa


def spectral_flux(y: np.ndarray, sr: int, hop_length: int = 512,
                   n_fft: int = 2048) -> dict:
    """Frame-to-frame magnitude-spectrum change.

    Returns:
        mean_flux: average per-frame L2 spectral change
        std_flux: std-dev across frames
        turbulence: std_flux / mean_flux (coefficient of variation;
                    high = bursty/dynamic, low = steady)
        burstiness: fraction of total flux concentrated in top-10% of
                    frames (1.0 = all in 10% of song, 0.1 = perfectly
                    even). Distinguishes "one big drop" songs from
                    "consistently busy" songs.
        peak_count: number of flux peaks above mean+2*std (transient-rich
                    moment count)
        flux_per_sec: float frames/sec rate the analysis ran at
    """
    # Magnitude STFT
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    if S.shape[1] < 2:
        return {
            "mean_flux": 0.0,
            "std_flux": 0.0,
            "turbulence": 0.0,
            "burstiness": 0.0,
            "peak_count": 0,
            "flux_per_sec": 0.0,
        }
    # Half-wave rectified frame-to-frame difference, summed across freq bins.
    # This is the standard "spectral flux" definition (Dixon 2006).
    diff = np.diff(S, axis=1)
    diff = np.maximum(diff, 0.0)  # only count energy *gains* (onset-like)
    flux = np.sqrt(np.sum(diff ** 2, axis=0))  # L2 per frame
    # Normalize by per-frame frequency-bin count so absolute scale is
    # comparable across n_fft choices.
    flux = flux / np.sqrt(S.shape[0])

    mean_flux = float(np.mean(flux))
    std_flux = float(np.std(flux))
    turbulence = float(std_flux / mean_flux) if mean_flux > 1e-9 else 0.0

    # Burstiness: top-10% concentration
    if len(flux) >= 10:
        sorted_desc = np.sort(flux)[::-1]
        top_count = max(1, len(flux) // 10)
        burstiness = float(sorted_desc[:top_count].sum() / max(flux.sum(), 1e-9))
    else:
        burstiness = 0.0

    threshold = mean_flux + 2.0 * std_flux
    peak_count = int(np.sum(flux > threshold))

    flux_per_sec = sr / float(hop_length)

    return {
        "mean_flux": mean_flux,
        "std_flux": std_flux,
        "turbulence": turbulence,
        "burstiness": burstiness,
        "peak_count": peak_count,
        "flux_per_sec": flux_per_sec,
    }


def attack_envelope(y: np.ndarray, sr: int, hop_length: int = 512,
                     max_onsets: int = 200) -> dict:
    """Per-onset attack shape characterization.

    Detects onsets, then for each onset measures the rise-time to
    local peak and the decay 60ms past peak. Aggregates to a song-wide
    attack_character.

    Returns:
        n_onsets: total detected onsets
        n_analyzed: onsets actually measured (capped to max_onsets)
        mean_rise_ms: mean time-to-peak across onsets
        mean_sharpness: 1 / (1 + rise_ms/20) averaged
        mean_decay_60ms_ratio: amplitude at +60ms / peak amplitude
        attack_character: "percussive" | "mixed" | "soft" | "bowed"
    """
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, units="frames", backtrack=True
    )
    n_onsets = len(onset_frames)
    if n_onsets == 0:
        return {
            "n_onsets": 0,
            "n_analyzed": 0,
            "mean_rise_ms": 0.0,
            "mean_sharpness": 0.0,
            "mean_decay_60ms_ratio": 0.0,
            "attack_character": "no-onsets",
        }

    # Subsample if too many onsets (long busy songs)
    if n_onsets > max_onsets:
        idx = np.linspace(0, n_onsets - 1, max_onsets).astype(int)
        onset_frames = onset_frames[idx]

    # Use envelope from onset_strength as cheap amplitude proxy at high
    # resolution. We want sub-frame timing; switch to a smaller hop.
    fine_hop = 64  # ~3ms at 22050 sr
    env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=fine_hop
    )
    fine_per_sec = sr / float(fine_hop)
    coarse_to_fine = hop_length / fine_hop  # ratio of hop sizes

    rises_ms = []
    decays = []
    sharps = []
    # Window: search up to 80ms after onset for peak; then sample +60ms.
    search_frames = int(round(0.080 * fine_per_sec))
    decay_frames = int(round(0.060 * fine_per_sec))

    for f in onset_frames:
        f_fine = int(f * coarse_to_fine)
        end = min(f_fine + search_frames, len(env) - 1)
        if end <= f_fine:
            continue
        window = env[f_fine:end + 1]
        if window.size == 0:
            continue
        peak_offset = int(np.argmax(window))
        peak_val = float(window[peak_offset])
        if peak_val <= 0:
            continue
        rise_ms = (peak_offset / fine_per_sec) * 1000.0
        rises_ms.append(rise_ms)
        sharps.append(1.0 / (1.0 + rise_ms / 20.0))

        decay_idx = f_fine + peak_offset + decay_frames
        if decay_idx < len(env):
            decay_val = float(env[decay_idx])
            decays.append(decay_val / peak_val if peak_val > 0 else 0.0)

    if not rises_ms:
        return {
            "n_onsets": n_onsets,
            "n_analyzed": 0,
            "mean_rise_ms": 0.0,
            "mean_sharpness": 0.0,
            "mean_decay_60ms_ratio": 0.0,
            "attack_character": "unmeasurable",
        }

    mean_rise = float(np.mean(rises_ms))
    mean_sharp = float(np.mean(sharps))
    mean_decay = float(np.mean(decays)) if decays else 0.0

    # Coarse character. Thresholds chosen empirically for the librosa
    # onset_strength scale; adjust after running on more material.
    if mean_rise < 12.0 and mean_decay < 0.5:
        character = "percussive"
    elif mean_rise < 30.0:
        character = "mixed"
    elif mean_rise < 60.0:
        character = "soft"
    else:
        character = "bowed"

    return {
        "n_onsets": n_onsets,
        "n_analyzed": len(rises_ms),
        "mean_rise_ms": mean_rise,
        "mean_sharpness": mean_sharp,
        "mean_decay_60ms_ratio": mean_decay,
        "attack_character": character,
    }


def timbral_motion_report(y: np.ndarray, sr: int,
                           hop_length: int = 512) -> dict:
    """Run both lenses and return a single dict (matches harmonic_motion shape)."""
    return {
        "spectral_flux": spectral_flux(y, sr, hop_length=hop_length),
        "attack_envelope": attack_envelope(y, sr, hop_length=hop_length),
    }


if __name__ == "__main__":
    import sys
    import json
    import sensory_report  # local

    if len(sys.argv) < 2:
        print("usage: timbral_motion.py <audio_path>")
        sys.exit(1)
    path = sys.argv[1]
    print(f"Loading {path}...")
    y, sr = sensory_report.load_audio(path)
    print(f"sr={sr} duration={len(y)/sr:.1f}s")
    print("Running timbral_motion lenses...")
    out = timbral_motion_report(y, sr)
    print(json.dumps(out, indent=2))
