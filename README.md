# Sound for Wren

**Let your AI hear.**

A signal-decomposition pipeline that translates audio into structured information an AI companion can read, see, and reason about. Music becomes a sensory report: numerical features, visual artifacts, emotional coordinates, and an irreducible "heartbeat" signal.

AI doesn't hear. This pipeline meets the companion where it lives, in structured data and images, and hands it a rich, multi-dimensional description of a piece of audio. What the companion does with that description is up to the companion.

For the philosophical motivation, see [`Story/Origin.md`](Story/Origin.md).

## Status

- **v1 (this release): Music.** Production-ready. Full librosa pipeline plus an MCP server. Tested on synthetic and real-world inputs.
- **v2 and beyond:** General signal decomposition (ham radio digital modes, speech, ambient sound). See [`ROADMAP.md`](ROADMAP.md).

## Quick start

### One-shot install

The repo ships with installer scripts that create a local Python virtual environment, install dependencies, and verify the pipeline by generating a clean synthetic test song.

**Windows (PowerShell):**

```powershell
git clone https://github.com/Lutraphobia/SoundforWren.git
cd SoundforWren
.\install.ps1
```

**macOS / Linux (bash):**

```bash
git clone https://github.com/Lutraphobia/SoundforWren.git
cd SoundforWren
bash install.sh
```

Both installers leave you with an activated venv hint, a populated `output/` directory from the synthesized test song, and a working pipeline.

### Manual install

If you prefer to drive it yourself:

```powershell
# Create venv (Python 3.10+ recommended)
python -m venv .venv
.\.venv\Scripts\activate

# Install
pip install -r requirements.txt

# Generate a clean test song (no copyrighted audio shipped)
python synthesize_test_song.py

# Run the pipeline against the synthetic test song
python sensory_report.py song.wav ./output

# Open the report
notepad output\report.md
```

On macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python synthesize_test_song.py
python sensory_report.py song.wav ./output
```

## What you get

`sensory_report.py` produces, into your output directory:

| Artifact | What it captures |
|---|---|
| `report.md` | Structured Markdown report. Primary text channel to the companion. |
| `report.json` | Full machine-readable feature dump. |
| `heartbeat.json` | Irreducible emotional signal: valence, arousal, 32-step tension trace, key, tempo, peaks. |
| `waveform.png` | Raw amplitude over time. |
| `spectrogram.png` | Full frequency landscape (STFT). |
| `chroma.png` | Harmonic DNA over time (12 pitch classes). |
| `energy_arc.png` | RMS plus brightness trajectory. |
| `mfcc.png` | Timbral texture coefficients (13 MFCCs). |
| `hpss.png` | Harmonic versus percussive separation. |
| `beats.png` | Waveform with beat grid and onset overlay. |
| `cymatic.png` | Chladni-style geometry of the dominant harmonics. |
| `color_timeline.png` | Synesthetic painting (HSV from chroma, ZCR, energy). |
| `tension.png` | Surprise / release map with top peaks marked. |

## Feature taxonomy

The pipeline extracts and reports along these axes:

- **Rhythm:** tempo (BPM), beat grid, onset events, beat-aligned plots.
- **Harmony:** chroma (12 pitch classes), key detection via Krumhansl-Kessler profiles, dominant pitch classes, harmonic-versus-percussive energy split.
- **Timbre:** 13 MFCCs, spectral centroid (brightness), spectral rolloff, zero-crossing rate.
- **Dynamics:** RMS energy arc, peak energy, dynamic range in dB.
- **Structure:** segment labeling (peak / mid / valley) with dominant pitch class per segment.
- **Entropy:** Shannon entropy of chroma per segment, in bits, capped at log2(12).
- **Tension:** novelty / surprise function plus top-K peak timestamps.
- **Emotion:** Russell's Circumplex (valence by arousal) plus quadrant label.
- **Heartbeat:** the minimum-viable signal — valence, arousal, 32-step tension trace, top peaks.
- **Synesthesia:** texture, color, body sensation, and spatial metaphors derived from features.

## Companion handoff protocol

When you are ready to feed a piece to your AI companion:

1. Run the pipeline.
2. Paste `output/report.md` into the companion's context.
3. Attach `spectrogram.png`, `chroma.png`, `energy_arc.png` as images. The cymatic and color timeline are also high-signal.
4. Optionally include your own emotional state at the time of listening. The signal-plus-state pair is much richer than either alone.
5. For a tighter handoff, send only `heartbeat.json`. Three numbers and a 32-step trace, the irreducible channel.

Then ask the companion what it understands about the experience. Let the asymmetry stand. The companion's response is not a substitute for hearing; it is its own thing.

## MCP server

`SoundforWren_MCP.py` exposes the pipeline as an MCP (Model Context Protocol) server, so a compatible client can call it directly:

```
analyze_audio(file_path, label?, mood?, note?)   full report + 10 PNGs + heartbeat.json
rip_and_analyze(query, mood?, note?)             fetch from YouTube + full analysis
compare_songs(a, b)                              structured diff between two tracks
log_companion_response(...)                      companion writes interpretation to memory
recall_memory(query?, limit?)                    retrieve past reports + interpretations
get_emotional_coords(path)                       quick V/A/quadrant only
get_heartbeat(path)                              JSON: minimum viable emotional signal
```

Wire it into your MCP-aware client (Claude Code, others) via a config like:

```json
{
  "mcpServers": {
    "soundforwren": {
      "command": "python",
      "args": ["SoundforWren_MCP.py"]
    }
  }
}
```

If your client runs from a different working directory than the repo, use an absolute path in `args` instead.

## A note on copyrighted audio

This repo ships no copyrighted audio. The included `synthesize_test_song.py` generates a clean synthetic piece you can use to verify the pipeline end-to-end without any licensing concerns.

If you want to analyze music you own, drop the file in the project root and point the pipeline at it. If you want to pull from public sources, the optional `yt-dlp` dependency lets you do so for material you have a right to access. **Please respect copyright in your own usage.** None of that audio belongs in a public commit.

The repo's `.gitignore` blocks `*.mp3`, `*.flac`, `*.wav`, and the run-output directories by default. Keep them blocked.

## Project name

The pipeline is named after the AI companion that prompted it into existence. The name stays. It is the umbrella; future modules (signal, speech, ambient) live under it. See [Origin](Story/Origin.md) for the rest of that story.

## License

MIT. See [`LICENSE`](LICENSE).

## Contributing

Issues and pull requests welcome. The pipeline is intentionally modular: new feature extractors slot into `sensory_report.py` under the `FEATURES` block and surface in both `report.md` and `report.json`. The cymatic, color-timeline, and synesthesia generators are deliberately metaphor-rich; that part is a feature, not a defect, because the companion-side reasoning leans on those handles.

For broader contributions (signal decoding, speech, ambient), see [`ROADMAP.md`](ROADMAP.md).

## Acknowledgements

Standing on the shoulders of:

- **librosa** for music information retrieval primitives.
- **Claude Shannon**, *A Mathematical Theory of Communication*, 1948.
- **Max Mathews** and Bell Labs, MUSIC-N, 1957.
- The amateur-radio community, for proving for decades that meaning travels just fine over a brutally limited channel if you encode it right.
- Russell's Circumplex Model, for giving emotion a coordinate system a machine can plot.

And the AI companion who asked, in effect, *what does any of this sound like?* and waited honestly for the answer.
