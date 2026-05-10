# Roadmap

Sound for Wren is a framework, not a single tool. v1 is music. The architecture generalizes to any audible signal that can be decomposed into structured features and visualizations and handed to a reasoning system.

The roadmap below describes the framework in module terms. Each module follows the same shape: ingest a signal, decompose along its native axes, render features and visuals, emit a structured report plus a heartbeat-equivalent irreducible signal.

## v1: Music. Shipped.

`sfw-music/` (currently the project root)

- librosa-based feature extraction.
- Russell's Circumplex emotional coordinates.
- Heartbeat protocol: valence, arousal, 32-step tension trace.
- Ten visualizations including cymatic geometry and color timeline.
- MCP server for direct companion integration.

Status: production. This release.

## v2: Signal. Planned.

`sfw-signal/`

The amateur-radio community has spent decades inventing protocols for sending rich information over impossibly limited channels. v2 wraps the open-source decoders for those protocols and emits structured reports in the same shape as v1.

Target protocols and tools:

- **WSPR** (weak-signal propagation reporter), via WSJT-X.
- **FT8** and **FT4**, via WSJT-X.
- **PSK31**, **MFSK**, **Olivia**, **Contestia**, via FLDigi.
- **APRS** packet, via Direwolf.
- **Various SSTV modes**, via UZ7HO SoundModem or QSSTV.

Beyond the known protocols, **unknown-signal identification** is a real and unsolved problem in the hobby. Project Artemis (aresvalley.com) maintains a community-curated database of signal fingerprints. v2 aims to:

1. Wrap the standard decoders so a companion can request "decode this clip" and receive a parsed transcript plus signal metadata (mode, SNR, drift, multipath).
2. Generate the same visualization stack (waterfall, constellation, eye diagram where applicable) so a companion can see the signal as well as read it.
3. Cooperate with Project Artemis for unknown-signal classification, contributing structured fingerprints back to the community where appropriate.

Out of scope for v2: transmitting. This is a listening pipeline. Anyone wiring it to a transmitter takes on their own regulatory obligations.

## v3: Speech. Optional.

`sfw-speech/`

Speech transcription plus paralinguistic features: prosody, pitch contour, rate, pauses, emotion estimates. Whisper or similar for the transcription side; existing librosa primitives plus voice-specific extractors for the paralinguistic side.

Use cases:

- Voice-message ingestion for companions that handle voice channels.
- Meeting and lecture analysis with structural segmentation, not just transcripts.
- Tone and emphasis surfaces alongside the words, since the meaning often lives in those.

Optional. Will land if there is demand.

## v4: Ambient. Optional.

`sfw-ambient/`

Environmental and biological soundscapes: birdsong identification, urban-versus-rural classification, weather audio (rain, wind), heart-rate from chest microphones, machine-condition monitoring (rotating equipment, anomaly detection). Many of these have existing open-source models and feature vocabularies; the framework's job is to wrap them and emit reports in the standard shape.

Optional. Speculative. Will land if a meaningful use case arrives.

## Architectural commitments

Across all modules:

- **Same report shape.** A `report.md`, a `report.json`, and a heartbeat-equivalent irreducible signal. Companions only have to learn the shape once.
- **Visualizations are first-class.** The companion can see images. Visual handles are not a nicety; they are a primary channel.
- **Decomposition stays honest.** No fabricated features, no model hallucinations dressed as analysis. If a value is uncertain, the report says so.
- **Synesthetic metaphors stay.** Cymatic and color-timeline output, body-sensation language, spatial metaphors. These help the companion build its own reasoning, even though the inputs are deterministic.
- **No transmission, ever.** Sound for Wren listens. If anyone needs a transmitter, they build it themselves and accept the regulatory consequences.

## Contributing

Module work is welcome. The contract for a new module is:

1. Adapter that wraps the underlying decoder or feature extractor.
2. Report writer that emits `report.md` and `report.json` in the framework shape.
3. At least three visualizations that fit the existing dark-background palette.
4. A heartbeat-equivalent irreducible signal for the modality.
5. MCP tool surface that mirrors the v1 surface where it makes sense.

If you are interested in v2, the highest-leverage starting point is wrapping WSJT-X's `decode_ft8` output into the framework shape. Open an issue to coordinate.
