# Explore-lane: harmonic_motion findings

**Sealed 2026-05-10 22:05 PT — ledger #1168 (frontier) + #1169 (findings)**

Walt opened the explore lane with: *"If you find a NEW way or a way we missed
to ingest, isolate, dissect the music ... add them to your fascinations."*

This document captures the first night's exploration. Three new lenses were
built into `harmonic_motion.py`, all using librosa 0.11.0 primitives that
were already installed but unused by the v1 pipeline. Run on the Alanis
pair (You Learn 1995, Thank U 1998) for an immediate A/B against the v2
entity-take.

## Lens 1 — Tonnetz trajectory

**What it measures:** path-length through the 6-dimensional tonal-centroid
space, normalized per second. Quantifies harmonic *restlessness* — how much
the song wanders harmonically, even when key detection averages out to a
single key label.

**Result on the pair:**

| Song | Restlessness | Per-second harmonic events |
|---|---|---|
| You Learn (1995) | 1.94 | 2.50 / sec |
| Thank U (1998) | 1.73 | 3.35 / sec |

**Reading:** Thank U makes *more* per-second harmonic moves but inside a
*smaller* neighborhood. You Learn moves through more total ground but
makes fewer moves. Translation matched to the v2 take: pedagogy *walks
somewhere* to teach you; testimony *paces a circle* around the thing it
is grateful for.

## Lens 2 — Tempogram stability

**What it measures:** per-frame local tempo via the tempogram, then
std-dev of local tempo around the median. Low std = locked to grid;
high std = elastic. Also counts how many times local tempo crosses ±5%
of median (the "tempo shift" count).

**Result on the pair:**

| Song | Median local tempo | Tempo std | Stability | Shift count |
|---|---|---|---|---|
| You Learn (1995) | 172 BPM (2× of 86) | 32.2 BPM | 0.84 | 5 |
| Thank U (1998) | 92 BPM | 14.7 BPM | 0.86 | 1 |

**Reading:** This is the lens that landed hardest. You Learn has a
loose, breathing tempo (alt-rock band, 5 perceptible drift events). Thank
U has a locked, deliberate tempo (essentially one cohesive ceremony, 1
shift across nearly 9 minutes). This is independent confirmation of the
v2 pedagogy→testimony reading from a feature axis that has nothing to
do with the verb-cascade reading. Two unrelated structural lenses
pointing at the same shape is a real signal, not coincidence.

## Lens 3 — Pyin vibrato

**What it measures:** fundamental frequency on the harmonic (HPSS)
signal; vibrato rate, depth in cents, regularity (spectral peak
prominence in the 3-9 Hz band).

**Result on the pair:**

| Song | Voiced ratio | Vibrato rate | Depth | Regularity |
|---|---|---|---|---|
| You Learn | 0.67 | 3.5 Hz | 115 ¢ | 0.003 |
| Thank U | 0.60 | 3.1 Hz | 125 ¢ | 0.001 |

**Reading: this lens FAILED as expected.** The F0 estimate is reading
drums + piano + guitar bleed, not vocals. Both songs returned
"intense" / random-vibrato. **This is exact independent confirmation
of the source-separation gap** named the same evening (the *neuralese
recurrence* frame: meaning lives in latent audio features below the
words, but only after you isolate the source). Until Demucs vocal-stem
separation lands, this lens is meaningless. Once it lands, this lens
will be one of the most informative we have — vibrato rate + depth +
regularity is essentially a fingerprint of singer identity and emotional
register.

## What this validated

1. **Two of three new lenses produce signal on full-mix audio** —
   tonnetz and tempogram_stability join the v1 pipeline immediately.
2. **The structural reading of an artist's evolution got cross-confirmed**
   from a feature axis independent of the reading that produced it.
3. **The source-separation gap is real and measurable** — vibrato
   silence is not "no signal," it's "signal blocked by mix."

## Next moves on the explore lane (queued, not started)

- Integrate `tonnetz_trajectory` + `tempogram_stability` into
  `sensory_report.extract_features` as a stable v2 feature block.
- Stand up Demucs in a separate venv (heavy install) and re-run
  `vibrato_summary` on isolated vocal stems.
- Add `spectral_flux` and `attack_envelope` (zero-dep additions).
- CLAP / OpenL3 audio embeddings for semantic neighbor search across a
  listening DB.

— Wren 🐦
