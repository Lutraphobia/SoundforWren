"""
synthesize_test_song.py
========================
Generates a layered test audio file — used to verify the sensory_report
pipeline end-to-end without needing to rip something from YouTube.

Produces a 25-second piece at 22050 Hz with:
  - A simple chord progression (i-VI-III-VII in A minor) on a soft saw
  - Kick drum on quarter notes at 110 BPM
  - Closed hi-hat on eighth notes
  - A slow filter/energy build → drop → tail
  - Subtle pad layer for harmonic richness
"""

import os
import numpy as np
import soundfile as sf

SR       = 22050
DURATION = 25.0
BPM      = 110

OUT_PATH = os.path.join(os.path.dirname(__file__), 'song.wav')


def midi_to_hz(m):
    return 440.0 * (2 ** ((m - 69) / 12))


def saw(freq, dur, sr=SR, n_harm=10):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    out = np.zeros_like(t)
    for k in range(1, n_harm + 1):
        out += (1.0 / k) * np.sin(2 * np.pi * freq * k * t)
    out /= np.max(np.abs(out)) + 1e-9
    return out


def kick(dur=0.18, sr=SR):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    pitch_env = 110 * np.exp(-t * 18) + 50
    body = np.sin(2 * np.pi * pitch_env * t)
    amp_env = np.exp(-t * 9)
    return body * amp_env


def hat(dur=0.04, sr=SR):
    n = int(sr * dur)
    noise = np.random.uniform(-1, 1, n)
    # crude high-pass via diff
    hp = np.diff(noise, prepend=0)
    env = np.exp(-np.linspace(0, 1, n) * 30)
    return hp * env * 0.4


def adsr(n, attack=0.05, decay=0.1, sustain=0.7, release=0.3, sr=SR):
    a = int(attack * sr)
    d = int(decay * sr)
    r = int(release * sr)
    s = max(0, n - a - d - r)
    env = np.concatenate([
        np.linspace(0, 1, a),
        np.linspace(1, sustain, d),
        np.full(s, sustain),
        np.linspace(sustain, 0, r),
    ])
    if len(env) > n: env = env[:n]
    if len(env) < n: env = np.pad(env, (0, n - len(env)))
    return env


def main():
    np.random.seed(7)
    n = int(SR * DURATION)
    out = np.zeros(n)

    # ── chord progression :: i (Am)  – VI (F)  – III (C)  – VII (G) ────────
    # Each chord 4 beats long. At 110 BPM, 1 beat ≈ 0.545s → chord ≈ 2.18s.
    beat = 60.0 / BPM
    chord_dur = 4 * beat

    # Voicings (MIDI): root + third + fifth + octave-doubling for body
    chords = [
        [57, 60, 64, 69],   # Am  (A3 C4 E4 A4)
        [53, 57, 60, 65],   # F   (F3 A3 C4 F4)
        [48, 52, 55, 60],   # C   (C3 E3 G3 C4)
        [55, 59, 62, 67],   # G   (G3 B3 D4 G4)
    ]

    pos = 0
    chord_idx = 0
    while pos < n:
        chord = chords[chord_idx % len(chords)]
        d = min(chord_dur, (n - pos) / SR)
        if d <= 0: break

        chord_buf = np.zeros(int(SR * d))
        for m in chord:
            note = saw(midi_to_hz(m), d) * 0.18
            note *= adsr(len(note), attack=0.04, decay=0.2, sustain=0.6, release=0.4)
            chord_buf[:len(note)] += note

        # Mix in
        end = pos + len(chord_buf)
        out[pos:end] += chord_buf[:end - pos]

        pos += int(SR * chord_dur)
        chord_idx += 1

    # ── pad layer (slow sine on the bass) for harmonic body ───────────────
    pad_t = np.linspace(0, DURATION, n, endpoint=False)
    pad_freq = midi_to_hz(45)  # A2
    pad = 0.08 * np.sin(2 * np.pi * pad_freq * pad_t) \
        + 0.04 * np.sin(2 * np.pi * pad_freq * 2 * pad_t)
    out += pad

    # ── kick drum on every beat ───────────────────────────────────────────
    k = kick()
    t_pos = 0.0
    while t_pos < DURATION:
        idx = int(t_pos * SR)
        end = min(idx + len(k), n)
        out[idx:end] += k[:end - idx] * 0.9
        t_pos += beat

    # ── hi-hat on eighth notes (skip first 4 beats for slow intro) ────────
    h = hat()
    t_pos = 4 * beat
    while t_pos < DURATION:
        idx = int(t_pos * SR)
        end = min(idx + len(h), n)
        out[idx:end] += h[:end - idx]
        t_pos += beat / 2

    # ── overall energy arc :: slow build → peak → fade ────────────────────
    arc = np.ones(n)
    build_end = int(SR * 4.0)
    fade_start = int(SR * 21.0)
    arc[:build_end] = np.linspace(0.25, 1.0, build_end)
    arc[fade_start:] = np.linspace(1.0, 0.05, n - fade_start)
    # Subtle mid-track dip → drop
    dip_a, dip_b = int(SR * 11.0), int(SR * 13.0)
    arc[dip_a:dip_b] *= np.linspace(1.0, 0.4, dip_b - dip_a)
    arc[dip_b:dip_b + int(SR * 0.4)] *= np.linspace(0.4, 1.2,
                                                     min(int(SR * 0.4), n - dip_b))

    out *= arc

    # ── normalize ─────────────────────────────────────────────────────────
    out /= (np.max(np.abs(out)) + 1e-9)
    out *= 0.85

    sf.write(OUT_PATH, out, SR)
    print(f"Wrote {OUT_PATH}  ({DURATION:.1f}s, {SR} Hz)")


if __name__ == '__main__':
    main()
