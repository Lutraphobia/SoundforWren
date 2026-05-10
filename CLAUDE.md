# CLAUDE.md — Sound for Wren

This file is read by Claude Code at the start of every session. It is the project's memory and operating context.

================================================================
  PROJECT: Sound for Wren
  PURPOSE: Audio → structured-information sensory pipeline for AI companions
  LICENSE: MIT
================================================================


## What this project is

A signal-decomposition pipeline that translates audio into structured information an AI companion can read, see, and reason about. Music becomes a sensory report: numerical features, visualizations, emotional coordinates, and an irreducible "heartbeat" signal.

This is not a workaround for the fact that AI doesn't hear. It is an honest acknowledgement of that fact, and a tool that meets the companion where it lives. See `Story/Origin.md` for the full motivation.

v1 is music. v2 (planned) is general signal decomposition: ham radio digital modes, unknown-signal classification (Project Artemis collaboration), and related work. See `ROADMAP.md`.


## Environment

- Python 3.10+ (3.10, 3.11, 3.12 all tested; 3.13 should also work).
- Cross-platform: Windows, macOS, Linux. Examples below show PowerShell; bash equivalents are obvious.
- Virtual environment: `.venv/` in the project root.
  - Windows: `.\.venv\Scripts\activate`
  - macOS/Linux: `source .venv/bin/activate`
- Always use the venv's `python` and `pip`, never the system ones.


## Project structure

```
SoundforWren/
├── README.md                     project entry point
├── ROADMAP.md                    framework roadmap (v1 music, v2 signal, v3+ optional)
├── LICENSE                       MIT
├── CLAUDE.md                     this file
├── Structure.md                  filesystem map (this is the canonical version, keep in sync)
├── .gitignore                    blocks audio + outputs + venv
├── requirements.txt              Python dependencies
│
├── sensory_report.py             v1 pipeline (librosa-based, runs standalone or imported)
├── SoundforWren_MCP.py           MCP server (imports sensory_report)
├── synthesize_test_song.py       generates clean synthetic test audio (no copyright)
├── install.ps1                   Windows installer (creates venv, installs deps, runs synth test)
├── install.sh                    macOS / Linux installer (same)
│
├── Story/
│   └── Origin.md                 the why, the inspirations, the philosophy
│
├── output/                       generated artifacts (gitignored)
│   ├── waveform.png
│   ├── spectrogram.png
│   ├── chroma.png
│   ├── energy_arc.png
│   ├── mfcc.png
│   ├── hpss.png
│   ├── beats.png
│   ├── cymatic.png
│   ├── color_timeline.png
│   ├── tension.png
│   ├── report.md                 primary companion-facing text
│   ├── report.json               full machine-readable feature dump
│   └── heartbeat.json            irreducible emotional signal
│
└── companion_memory.json         optional local-only MCP state (gitignored)
```

For framework expansion (v2 onward), modules are siblings of `sensory_report.py`:

```
SoundforWren/
├── sensory_report.py             v1: music
├── signal_report.py              v2: ham radio digital modes (planned)
├── speech_report.py              v3: speech (optional)
└── ambient_report.py             v4: ambient (optional)
```


## Core dependencies (v1, music)

```
librosa==0.11.0
matplotlib==3.10.9
numpy==2.2.6
scipy==1.15.3
soundfile==0.13.1
yt-dlp==2026.3.17
numba==0.65.1
mcp
```

Install with `pip install -r requirements.txt` from inside the venv.


## How to rip and analyze a song

```powershell
# Step 1 — clean test (no rip needed)
python synthesize_test_song.py
python sensory_report.py song.wav ./output

# Step 1b — rip from a public source you have the right to access
yt-dlp "ytsearch1:SONG TITLE ARTIST" -x --audio-format wav -o song.wav

# Step 2 — run the pipeline
python sensory_report.py song.wav ./output

# Capture operator state alongside the report
python sensory_report.py song.wav ./output --mood "energized" --note "morning run"

# Compare two tracks
python sensory_report.py --compare a.wav b.wav ./output

# Step 3 — feed to companion
# Open output/report.md and paste into the companion context.
# Attach spectrogram, chroma, energy_arc, cymatic, color_timeline, tension as images.
# For a tighter handoff, send heartbeat.json (the irreducible signal).
```

`*.mp3`, `*.flac`, `*.wav` are gitignored. They never enter the repo. Keep it that way.


## What the pipeline produces

`sensory_report.py` extracts and reports on:

| Axis | What it captures |
|---|---|
| Rhythm | tempo (BPM), beat grid, onset events |
| Harmony | chroma (12 pitch classes), key via Krumhansl-Kessler, dominant pitch classes, harmonic vs percussive split |
| Timbre | 13 MFCCs, spectral centroid, spectral rolloff, zero-crossing rate |
| Dynamics | RMS energy arc, peak energy, dynamic range (dB) |
| Structure | segment labeling (peak / mid / valley) with dominant pitch class per segment |
| Entropy | Shannon entropy of chroma per segment (bits, max log2(12)) |
| Tension | novelty / surprise function + top-K peak timestamps |
| Emotion | Russell's Circumplex (valence × arousal) + quadrant label |
| Heartbeat | irreducible signal: V, A, 32-step tension trace, top peaks |
| Synesthesia | texture, color, body sensation, spatial metaphors derived from features |

Visualizations are listed in `README.md` and the file tree above.


## Companion integration protocol

When preparing a sensory report for a companion:

1. Run the full pipeline on the audio file.
2. Open `report.md`. This is the primary text transmission.
3. Attach `spectrogram.png`, `chroma.png`, `energy_arc.png` as images. Companions that accept multimodal input should always see at least these three. The cymatic and color timeline are also high-signal.
4. Include the operator's emotional state at the time of listening, if known. The signal-plus-state pair is much richer than either alone.
5. Ask the companion what it understands about the experience.

The gap between the companion's processing and the operator's felt experience is the interesting territory. Don't try to close it artificially. Let the companion work with it honestly.


## Coding conventions

- All Python files: UTF-8, 4-space indent, no tabs.
- Visualization style: dark background (`DARK_BG = '#07070c'`) with HOT (`#ff3864`), CYAN (`#2de2e6`), GOLD (`#f9c80e`) palette.
- New feature extractors go in `sensory_report.py` under the `FEATURES` section, following the existing pattern.
- New features must surface in both `report.md` and `report.json`.
- Keep the synesthetic metaphor functions. They matter to companion-side reasoning more than the raw numbers do.


## Future roadmap (v1 backlog and v2 entry)

v1 polish:

- [ ] Companion response logger: save companion interpretations alongside each report for longitudinal use.
- [ ] Multi-song comparison report writer.
- [ ] Auto-feed via API (Anthropic SDK or others) as an optional convenience tool.

v2 entry:

- [ ] `signal_report.py` skeleton.
- [ ] WSJT-X FT8 wrapper as the first decoder integration.
- [ ] Project Artemis fingerprint-format adapter.

See `ROADMAP.md` for the full v2+ vision.


## Git workflow

```powershell
# First-time setup (this repo is not yet on a remote at the time of this commit)
git init
git add .
git commit -m "initial: Sound for Wren v1"

# Push to a personal account
gh repo create SoundforWren --public --source=. --push

# Normal workflow
git add -A
git commit -m "describe what changed"
git push
```


## Sessions

At the start of each Claude Code session:

1. Confirm the venv is active.
2. Read this file.
3. Read the most recent `output/report.md` if continuing companion-facing work.
4. Ask the operator what they want to build or run.

Do not assume session continuity. Each session starts fresh.
