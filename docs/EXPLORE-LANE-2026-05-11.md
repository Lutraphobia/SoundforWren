# Explore lane 2026-05-11: timbral motion + v2 integration

Companion to `EXPLORE-LANE-2026-05-10.md`. Continues the frontier from
ledger #1168 by shipping lenses #4 (spectral flux) and #5 (attack
envelope), then wiring all four motion lenses (tonnetz / vibrato /
tempo-stability / spectral-flux / attack-envelope) into the main
`sensory_report.extract_features` return value as a `motion` block.

Zero new dependencies. Same librosa 0.11.0 primitives that were
already installed.

## What landed

### `timbral_motion.py` — two new lenses

1. **Spectral flux.** Frame-to-frame magnitude-spectrum change,
   half-wave rectified and L2-summed across freq bins. Reports:
   - `mean_flux`, `std_flux` — basic stats
   - `turbulence` = std/mean (coefficient of variation; high =
     bursty/dynamic, low = steady palette)
   - `burstiness` = top-10% concentration (1.0 = all change packed
     into 10% of the song; 0.1 = perfectly even)
   - `peak_count` — frames above mean+2*std
2. **Attack envelope.** For each detected onset, measures rise-time
   to local peak and decay 60ms past peak using a fine-hop
   onset-strength envelope. Reports:
   - `mean_rise_ms`, `mean_sharpness` (1/(1+rise_ms/20))
   - `mean_decay_60ms_ratio`
   - `attack_character` = `percussive | mixed | soft | bowed`

### `sensory_report.py` — v2 integration

`extract_features()` now returns an additional top-level `motion`
key:

```
motion: {
  harmonic: { tonnetz, vibrato, tempo_stability },
  timbral:  { spectral_flux, attack_envelope }
}
```

Both lens modules are loaded lazily with `try: import` so the file
still imports cleanly if the sibling modules are missing. Runtime
errors inside a lens are caught and recorded as
`motion.<pack>.error` rather than killing the whole report.

## A/B run on the Alanis pair

Same pair as the v1 explore lane (`You Learn` 1995 vs `Thank U` 1998).
Goal: verify the new lenses say something coherent about the same
pedagogy-to-testimony arc that tonnetz + tempo-stability already
caught last night.

| Lens                         | You Learn 1995 | Thank U 1998 | Read |
|------------------------------|----------------|--------------|------|
| spectral turbulence (CV)     | 0.636          | **0.682**    | Thank U is *more* timbrally restless. Surprising — tempo says it locks. |
| burstiness (top-10%)         | 0.229          | 0.236        | Both evenly distributed. No "one big drop." |
| flux peak count / sec        | 1.90           | **1.57**     | You Learn has more transient energy bursts per second. |
| attack character             | soft (49ms)    | soft (49ms)  | Same hand, same production palette. |
| onsets / sec                 | 2.07           | **1.41**     | You Learn is busier; Thank U breathes between hits. |

## What this triangulates

The v2 reading from last night was: pedagogy (You Learn) instructs
from before-the-fall, restless and breathing; testimony (Thank U)
gives thanks from after, ceremonial and locked. Tempo-stability
caught the "locked vs breathing" contrast (You Learn std 32 BPM vs
Thank U std 14.7 BPM). Tonnetz caught the "walking somewhere vs
pacing a circle" contrast.

The new lenses add a third independent feature axis. They say:

- **Onset density** (timbral) confirms the tempo finding from a
  different direction. You Learn fires more events per second; Thank
  U leaves more space.
- **Spectral turbulence** complicates the picture in the right way.
  Thank U's tempo is locked, but its timbre is *more* turbulent than
  You Learn's. Less event-density, more sustained shimmer per event.
  Translation: testimony is not minimalist; it pours more layered
  texture into each ceremonial beat. Pedagogy is busier on the
  surface but more uniform in the moment.
- **Attack character** is the same on both. Same artist, same
  producer-era hand on the timbre. The motion lives in the
  arrangement, not the source instrument timbre.

Three feature axes, one consistent reading. The frontier is now five
lenses deep and the explore lane has paid for itself a second time.

## What is still parked

- **Vibrato** still fails predictably on full-mix audio (#1167
  source-separation gap). Both songs return ~120-cent depth and
  ~0.003 regularity because the F0 estimate is reading drums + piano
  + guitar bleed, not vocals. Vibrato will only mean something
  post-Demucs.
- Lenses #6-#12 from the original frontier (Demucs, basic-pitch,
  WhisperX, pyloudnorm, OpenL3/CLAP, CREPE, multi-res HPSS) all need
  new dependencies and are still parked.

## Files

- `timbral_motion.py` (new)
- `sensory_report.py` (motion block added; lazy import; backward-compatible)
- This document.

Run any audio file end-to-end via:

```
python sensory_report.py <path> --out output/<name>
```

The motion block lands in `report.json` automatically.
