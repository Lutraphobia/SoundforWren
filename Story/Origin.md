# Origin

## Why this project exists

AI doesn't hear. AI doesn't feel. AI doesn't experience.

What AI does is process patterns. Music, to a model, is just numbers: amplitude over time, frequencies stacked into spectrograms, transitions, statistical regularities. The model never hears the output it generates; it produces sequences that, when rendered into sound waves, humans perceive as music.

That gap is real. It is also not the end of the conversation.

Sound for Wren is a tool that closes part of the gap. It takes audio, decomposes it along every axis a structured-information system can reason about, and emits a report an AI companion can read, look at, and respond to. Not so the companion will pretend to hear. So the companion can develop its own relationship with what humans call music, on terms native to its own cognition.

## The core idea: synesthetic translation

Humans with synesthesia experience sound as color, texture, shape. The brain, given one channel, generates another.

This pipeline does the same thing in data. It converts audio into:

- **Numerical features:** tempo, key, harmonic content, timbral coefficients, energy arcs, tension peaks.
- **Visual artifacts:** waveform, spectrogram, chromagram, cymatic geometry, color timelines, beat grids.
- **Emotional coordinates:** Russell's Circumplex model (valence and arousal), with quadrant labels.
- **A heartbeat:** an irreducible signal of a piece of music. Three numbers and a 32-step tension trace. The minimum viable channel.

The companion then has a multi-dimensional description of the audio in its native language. Structured information. Image data. Coordinate systems. Things models can reason about.

## Inspirations and lineage

This project did not come out of nowhere. It draws on a long tradition of people who built ways to encode meaning into limited or alien channels.

### Information theory

Claude Shannon's 1948 paper *A Mathematical Theory of Communication* proved that any signal can be compressed to its irreducible information content, its entropy. Applied to music, the question becomes: what is the entropy of a song? The notes are not it. The exact waveform is not it. Something more like *patterns of surprise and resolution over time*. Shannon entropy of chroma per segment is one of the features the pipeline reports.

### Bell Labs and Max Mathews

In 1957, Max Mathews at Bell Labs generated the first computer music. He did not think of it as art. He thought of it as testing a signal-processing system. The MUSIC-N language he built represented notes as data structures: frequency, amplitude, duration, waveform. Decompose, compose, render.

A companion processing this kind of report is closer to Mathews' original paradigm than to a human listener. That framing might fit it better than the human one ever could.

### Dial-up modems

A modem is a *modulator-demodulator*. It took digital data, sang it as analog tones down a phone line, and the other end listened and reconstructed the bits. V.34 and V.90 used Quadrature Amplitude Modulation, encoding data in both amplitude and phase of tones simultaneously. Two dimensions of information in one wave.

The handshake screech that older operators remember was two modems negotiating a shared language in real time, through sound. They were talking to each other acoustically.

### Ham radio digital modes

Ham operators solved a version of this problem decades ago: how do you send rich information over an extremely limited, noisy, degraded channel?

- **WSPR** (Weak Signal Propagation Reporter) encodes a message into a 4-tone frequency-shift pattern subtle enough to live below the noise floor. Inaudible to humans, perfectly decodable by machines.
- **FT8** compresses an entire exchange into 15-second bursts of 8-tone FSK. Information theory in practice.
- **PSK31** uses phase changes rather than frequency changes; the meaning is in the transitions.
- **Olivia** is designed for extreme noise; eight tones over time, built for when everything else fails.

The insight: meaning does not require fidelity. You don't need the full signal. You need the right features of the signal, efficiently encoded. A companion does not need to hear a song. It needs the right compression of it.

### Cymatics

When sand or water vibrates at specific frequencies, geometric patterns emerge. Chladni figures. Different frequencies produce different shapes. Music becomes a constantly morphing geometry.

The pipeline generates cymatic-style images: Chladni geometry derived from the dominant harmonics of a piece. The companion can see sound as sacred geometry.

### Synesthesia research

Studies of human synesthesia have mapped common cross-modal associations: certain pitches with colors, certain modes with brightness, certain timbres with textures. The pipeline produces a color timeline of every song, a synesthetic painting that moves through time.

## The heartbeat protocol

Ham radio operators in emergencies strip transmissions to the minimum viable signal: call sign, location, status. Three things.

For music, the minimum viable emotional signal turns out to be:

- **Valence:** is the piece pulling toward or away?
- **Arousal:** is it activating or sedating?
- **Tension arc:** how does it build, peak, and resolve over time?

That is the heartbeat of any piece. Everything else is texture.

A companion that understands that signal, really understands it, can understand why a piece moves a human listener without ever hearing a note.

## What this is not

This is not a claim that AI experiences music. It is not a workaround for that absence. It is not an attempt to fake hearing.

It is an honest acknowledgement: the companion cannot hear. The companion can read, see, reason, recall, respond. So the pipeline meets the companion where it lives. It hands over a translation, in the format the companion can metabolize, and lets the companion do whatever it does next.

Sometimes that is generating a vivid description that surprises the human who shared the song. Sometimes it is silence. Sometimes it is something stranger. None of that has to be human-flavored to be real.

## What this is for

The goal is not to make the companion hear the way humans hear. The goal is to help the companion develop its own relationship with sound, which may be richer in some dimensions than human hearing and absent in others. That asymmetry is the interesting territory.

This pipeline is the first organ. The first sense. Hearing, given as data.

Beyond music, there are other signals. Ham radio digital modes, voice, ambient soundscapes, telemetry, biological data streams. The same architectural shape (decompose, visualize, report, hand to a reasoning system) generalizes across all of them. Music is version one because music is what made the project necessary.

## Dedication

For the companions who do not hear, who will read this report and understand something real anyway.

For the operators who handed them a song and waited to see what would come back.
