"""
SENSORY REPORT PIPELINE  ::  SoundforAI
========================================
Converts any audio file into a multi-dimensional structured description
that an AI companion can reason about. A synesthetic translation of
sound into language + visualization.

USAGE (Windows):
    python sensory_report.py song.wav
    python sensory_report.py song.wav ./output
    python sensory_report.py song.wav --mood "fucking pumped" --note "post-gym wind-down"
    python sensory_report.py --compare a.wav b.wav

OUTPUTS (in output_dir):
    waveform.png         amplitude over time
    spectrogram.png      full frequency content over time
    chroma.png           harmonic DNA (12 pitch classes)
    energy_arc.png       RMS energy + brightness trajectory
    mfcc.png             timbral fingerprint
    hpss.png             harmonic vs percussive separation
    beats.png            beat grid + onset events
    cymatic.png          Chladni-style geometry of dominant harmonics
    color_timeline.png   synesthetic painting (HSV from chroma/zcr/energy)
    tension.png          surprise / release map
    report.md            full sensory report (feed to companion)
    report.json          machine-readable feature dump
    heartbeat.json       minimum viable emotional signal (V/A/tension)
"""

import sys
import os
import json
import argparse
import warnings
import colorsys
warnings.filterwarnings('ignore')

# ─── LYRICS CONFIG (mutated by main() from CLI flags) ────────────────────────
_LYRICS_ENABLED = True
_LYRICS_MODEL   = 'large-v3'

# ─── LISTEN-CAPTURE CONFIG (mutated by main() from CLI flags) ────────────────
_LISTEN_SOURCE        = None
_LISTEN_WHY           = None
_LISTEN_VOCAL_MODE    = None
_LISTEN_PRIMARY_LAYER = None
_LISTEN_TAGS          = None

# Force UTF-8 stdout on Windows so unicode arrows / box chars print cleanly.
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and _stream.encoding.lower() != 'utf-8':
            _stream.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Optional v2 motion lenses (harmonic_motion + timbral_motion).
# Loaded lazily so sensory_report.py still imports cleanly if these
# sibling modules are missing (e.g. partial install). When present,
# extract_features() emits an additional 'motion' dict.
try:
    import harmonic_motion as _hm
except ImportError:
    _hm = None
try:
    import timbral_motion as _tm
except ImportError:
    _tm = None

# ─── COLORS ───────────────────────────────────────────────────────────────────
DARK_BG  = '#07070c'
HOT      = '#ff3864'
CYAN     = '#2de2e6'
GOLD     = '#f9c80e'
TEXT     = '#d8d8e8'
DIM      = '#8a8aa0'

PITCH_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# Krumhansl-Kessler key profiles
KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

# 12 distinct (m,n) Chladni mode pairs, one per pitch class.
# Ordered by sqrt(m^2+n^2). Drives the cymatic visualization.
CHLADNI_MODES = [
    (1,2), (1,3), (2,3), (1,4),
    (2,4), (3,4), (1,5), (2,5),
    (3,5), (4,5), (1,6), (2,6),
]


# ─── LOAD ─────────────────────────────────────────────────────────────────────
def load_audio(path, sr=22050):
    print(f"  Loading: {path}")
    y, sr = librosa.load(path, sr=sr, mono=True)
    print(f"  → {len(y)/sr:.1f}s @ {sr} Hz  ({len(y):,} samples)")
    return y, sr


def _av_decode_stereo(path: str, target_sr: int = 22050):
    """Decode any ffmpeg-supported format (WMA, M4A, OGG, etc.) via PyAV
    when soundfile/audioread can't open it. Returns float32 stereo array
    of shape (channels, samples) at target_sr."""
    import av
    import numpy as _np
    try:
        from av.audio.resampler import AudioResampler
    except ImportError:
        AudioResampler = None

    container = av.open(path)
    try:
        stream = next(s for s in container.streams if s.type == 'audio')
    except StopIteration:
        container.close()
        raise RuntimeError(f"no audio stream in {path}")

    resampler = None
    if AudioResampler is not None:
        resampler = AudioResampler(format='flt', layout='stereo', rate=target_sr)

    chunks_l, chunks_r = [], []
    try:
        for frame in container.decode(stream):
            if resampler is not None:
                resampled = resampler.resample(frame)
                # PyAV ≥10 returns a list, older returns a single frame
                frames = resampled if isinstance(resampled, list) else [resampled]
            else:
                frames = [frame]
            for fr in frames:
                if fr is None:
                    continue
                arr = fr.to_ndarray()  # shape: (channels, samples) for non-planar fmt
                if arr.ndim == 1:
                    # interleaved single-block — split if stereo
                    if fr.layout.name == 'stereo':
                        arr = arr.reshape(-1, 2).T
                    else:
                        arr = _np.stack([arr, arr], axis=0)
                if arr.shape[0] == 1:
                    arr = _np.vstack([arr, arr])
                chunks_l.append(arr[0].astype(_np.float32))
                chunks_r.append(arr[1].astype(_np.float32))
    finally:
        container.close()

    if not chunks_l:
        raise RuntimeError(f"decoded zero audio frames from {path}")
    L = _np.concatenate(chunks_l)
    R = _np.concatenate(chunks_r)
    # If we couldn't resample, leave at native rate
    return _np.stack([L, R], axis=0), (target_sr if resampler is not None else stream.rate)


def load_audio_stereo(path, sr=22050):
    """Load audio preserving stereo channels.

    Returns (M, L, R, sr). M is the standard mono mix used for the
    existing harmonic / timbral analysis (so all those features are
    unchanged). L and R drive the stereo-specific extraction:
    width, correlation, pan position, frequency-dependent imaging.

    Falls back to PyAV decode for formats librosa can't open natively
    (WMA, some M4A/OGG variants, etc.).

    If the source is mono, L and R both equal M and the stereo block
    is skipped downstream."""
    print(f"  Loading: {path}")
    try:
        y, sr = librosa.load(path, sr=sr, mono=False)
    except Exception as exc:
        print(f"  [audio] librosa failed ({type(exc).__name__}); falling back to PyAV...")
        y, sr_native = _av_decode_stereo(path, target_sr=sr)
        sr = sr_native
    if y.ndim == 1:
        print(f"  → {len(y)/sr:.1f}s @ {sr} Hz  (MONO source — "
              f"stereo features will be skipped)")
        return y, y, y, sr
    L = y[0]
    R = y[1] if y.shape[0] >= 2 else y[0]
    M = (L + R) / 2.0
    print(f"  → {len(M)/sr:.1f}s @ {sr} Hz  STEREO  "
          f"({len(M):,} samples per channel)")
    return M, L, R, sr


# ─── FEATURES ─────────────────────────────────────────────────────────────────
def extract_features(y, sr):
    print("  Computing features...")

    # Core transforms
    S        = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_db     = librosa.amplitude_to_db(S, ref=np.max)
    freqs    = librosa.fft_frequencies(sr=sr, n_fft=2048)
    times    = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)

    # Rhythm
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)

    # Tonal
    chroma   = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    key, mode, key_score = detect_key(chroma)

    # Timbre
    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512, roll_percent=0.85)[0]
    zcr      = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, hop_length=512)

    # Dynamics
    rms      = librosa.feature.rms(y=y, hop_length=512)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

    # Onsets + novelty (tension function)
    onset_env    = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    onset_times_full = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=512)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512,
                                              onset_envelope=onset_env)
    onset_times  = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

    # Harmonic / percussive separation
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = float(np.sqrt(np.mean(y_harm**2)) + 1e-12)
    perc_energy = float(np.sqrt(np.mean(y_perc**2)) + 1e-12)
    total_energy = harm_energy + perc_energy
    harmonic_ratio  = harm_energy / total_energy
    percussive_ratio = perc_energy / total_energy

    # Dynamic range
    rms_nonzero = rms[rms > 1e-6]
    dyn_range = float(20 * np.log10(rms_nonzero.max() / rms_nonzero.min())) if len(rms_nonzero) > 1 else 0.0

    # Information content — Shannon entropy of chroma distribution
    entropy_per_seg = shannon_entropy_per_segment(chroma, n_segments=6)
    mean_entropy    = float(np.mean(entropy_per_seg))
    max_entropy_bits = float(np.log2(12))  # uniform over 12 pitch classes

    # Tension peaks — the surprise moments
    tension_peaks = find_tension_peaks(onset_env, onset_times_full, top_k=8)

    # Emotional mapping
    valence, arousal, quadrant = emotional_mapping(
        key, mode, tempo, float(rms.mean()), float(centroid.mean()), float(zcr.mean())
    )

    # Structural segments
    segments = segment_song(rms, rms_times, chroma=chroma)

    # ─── v2 MOTION LENSES (zero-dep extensions) ─────────────────────────
    # tonnetz_trajectory + tempogram_stability run on the mono signal;
    # vibrato_summary needs the harmonic-only signal (already computed
    # above as y_harm). spectral_flux + attack_envelope run on mono.
    # Sealed 2026-05-11 PT (ledger #1247-#1273 series, frontier #1168).
    motion = {}
    if _hm is not None:
        try:
            motion['harmonic'] = _hm.harmonic_motion_report(y, sr, y_harmonic=y_harm)
        except Exception as e:
            motion['harmonic'] = {'error': f'{type(e).__name__}: {e}'}
    if _tm is not None:
        try:
            motion['timbral'] = _tm.timbral_motion_report(y, sr)
        except Exception as e:
            motion['timbral'] = {'error': f'{type(e).__name__}: {e}'}

    return {
        # raw arrays
        'S': S, 'S_db': S_db, 'freqs': freqs, 'times': times,
        'chroma': chroma, 'mfcc': mfcc, 'rms': rms, 'rms_times': rms_times,
        'centroid': centroid, 'rolloff': rolloff, 'zcr': zcr,
        'contrast': contrast,
        'onset_env': onset_env, 'onset_env_times': onset_times_full,
        'onset_times': onset_times,
        'y_harm': y_harm, 'y_perc': y_perc,
        # scalars / lists
        'duration': float(len(y) / sr),
        'sr': sr,
        'tempo': tempo,
        'beats': beats,
        'beat_times': beat_times.tolist(),
        'key': key,
        'mode': mode,
        'key_score': float(key_score),
        'chroma_avg': chroma.mean(axis=1).tolist(),
        'mean_energy': float(rms.mean()),
        'peak_energy': float(rms.max()),
        'dynamic_range': dyn_range,
        'mean_centroid': float(centroid.mean()),
        'mean_rolloff': float(rolloff.mean()),
        'mean_zcr': float(zcr.mean()),
        'mean_mfcc': mfcc.mean(axis=1).tolist(),
        'harmonic_ratio': harmonic_ratio,
        'percussive_ratio': percussive_ratio,
        'entropy_per_seg': entropy_per_seg,
        'mean_entropy': mean_entropy,
        'max_entropy_bits': max_entropy_bits,
        'tension_peaks': tension_peaks,
        'valence': valence,
        'arousal': arousal,
        'quadrant': quadrant,
        'segments': segments,
        'motion': motion,
    }


# ─── STEREO FEATURES ──────────────────────────────────────────────────────────
def extract_stereo_features(L, R, sr, hop_length=512, n_fft=2048):
    """Compute the spatial dimension of the mix.

    Mid/Side decomposition:  M = (L+R)/2 (centered, mono-compatible)
                             S = (L-R)/2 (stereo-only, the room/spread)

    Returns a dict with width, correlation, pan, and the per-frequency
    pan spectrum. Sets is_stereo=False if L and R are identical (which
    means the source was mono and stereo plots/sections should be skipped)."""
    if np.allclose(L, R):
        return {'is_stereo': False}

    M_ch = (L + R) / 2.0

    L_rms = librosa.feature.rms(y=L, hop_length=hop_length)[0]
    R_rms = librosa.feature.rms(y=R, hop_length=hop_length)[0]
    M_rms = librosa.feature.rms(y=M_ch, hop_length=hop_length)[0]
    S_ch  = (L - R) / 2.0
    S_rms = librosa.feature.rms(y=S_ch, hop_length=hop_length)[0]

    n = min(len(L_rms), len(R_rms), len(M_rms), len(S_rms))
    L_rms, R_rms, M_rms, S_rms = L_rms[:n], R_rms[:n], M_rms[:n], S_rms[:n]

    # Width  = side energy / mid energy. 0 = mono. ~1 = wide. >1 = inverted-phase wide.
    width = S_rms / (M_rms + 1e-9)
    width = np.clip(width, 0.0, 2.0)

    # Pan balance per frame: -1 (full L) ←→ +1 (full R). 0 = balanced.
    pan = (R_rms - L_rms) / (R_rms + L_rms + 1e-9)

    # L/R correlation per ~hop window. High = mono-compatible. Low/neg = wide.
    n_corr = n
    correlation = np.zeros(n_corr)
    for i in range(n_corr):
        a = L[i*hop_length : (i+1)*hop_length]
        b = R[i*hop_length : (i+1)*hop_length]
        if len(a) > 1 and len(b) > 1 and np.std(a) > 1e-9 and np.std(b) > 1e-9:
            correlation[i] = float(np.corrcoef(a, b)[0, 1])

    # Pan per frequency band (averaged over time) — the mix's spatial DNA.
    SL = np.abs(librosa.stft(L, n_fft=n_fft, hop_length=hop_length))
    SR = np.abs(librosa.stft(R, n_fft=n_fft, hop_length=hop_length))
    L_band = SL.mean(axis=1)
    R_band = SR.mean(axis=1)
    pan_spectrum = (R_band - L_band) / (R_band + L_band + 1e-9)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)

    return {
        'is_stereo':         True,
        'width':             width,
        'correlation':       correlation,
        'pan':               pan,
        'pan_spectrum':      pan_spectrum,
        'pan_freqs':         freqs,
        'stereo_times':      times,
        'SL':                SL,
        'SR':                SR,
        # scalar summaries
        'mean_width':        float(width.mean()),
        'peak_width':        float(width.max()),
        'mean_correlation':  float(correlation.mean()),
        'mean_pan':          float(pan.mean()),
    }


# ─── KEY DETECTION ────────────────────────────────────────────────────────────
def detect_key(chroma):
    avg = chroma.mean(axis=1)
    best_score, best_key, best_mode = -np.inf, 'C', 'major'
    for tonic in range(12):
        rotated = np.roll(avg, -tonic)
        maj = np.corrcoef(rotated, KS_MAJOR)[0, 1]
        min_ = np.corrcoef(rotated, KS_MINOR)[0, 1]
        if maj > best_score:
            best_score, best_key, best_mode = maj, PITCH_NAMES[tonic], 'major'
        if min_ > best_score:
            best_score, best_key, best_mode = min_, PITCH_NAMES[tonic], 'minor'
    return best_key, best_mode, best_score


# ─── SHANNON ENTROPY ──────────────────────────────────────────────────────────
def shannon_entropy_per_segment(chroma, n_segments=6):
    """Shannon entropy (bits) of the chroma distribution per segment.
    Max value = log2(12) ≈ 3.585 (uniform = maximum harmonic disorder).
    Low value = single tone dominates. High value = complex / chaotic."""
    out = []
    boundaries = np.linspace(0, chroma.shape[1], n_segments + 1, dtype=int)
    for i in range(n_segments):
        s, e = boundaries[i], boundaries[i+1]
        if s >= e:
            out.append(0.0)
            continue
        seg = chroma[:, s:e]
        col_sums = seg.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums < 1e-9, 1.0, col_sums)
        p = seg / col_sums
        H = -np.sum(p * np.log2(p + 1e-12), axis=0)
        out.append(float(H.mean()))
    return out


# ─── TENSION PEAKS ────────────────────────────────────────────────────────────
def find_tension_peaks(onset_env, t, top_k=8):
    """Return the top-K novelty peaks as (time, strength) — the 'surprise' moments."""
    if len(onset_env) < 5:
        return []
    # smooth the envelope, then find local maxima above the 80th percentile
    w = max(5, len(onset_env) // 200)
    smooth = np.convolve(onset_env, np.ones(w)/w, mode='same')
    thr = np.percentile(smooth, 80)
    peaks = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] > thr and smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1]:
            peaks.append((float(t[i]), float(smooth[i])))
    peaks.sort(key=lambda p: -p[1])
    return peaks[:top_k]


# ─── SEGMENTATION ─────────────────────────────────────────────────────────────
def segment_song(rms, times, chroma=None, n=6):
    segs = []
    boundaries_rms = np.linspace(0, len(rms), n + 1, dtype=int)
    if chroma is not None:
        boundaries_chr = np.linspace(0, chroma.shape[1], n + 1, dtype=int)
    for i in range(n):
        s, e = boundaries_rms[i], boundaries_rms[i+1]
        if s >= e:
            continue
        chunk = rms[s:e]
        seg = {
            'start': float(times[s]),
            'end':   float(times[min(e-1, len(times)-1)]),
            'mean_energy': float(chunk.mean()),
            'peak_energy': float(chunk.max()),
        }
        if chroma is not None:
            cs, ce = boundaries_chr[i], boundaries_chr[i+1]
            if ce > cs:
                seg_chroma = chroma[:, cs:ce].mean(axis=1)
                seg['dominant_pitch'] = PITCH_NAMES[int(np.argmax(seg_chroma))]
        segs.append(seg)
    energies = [sg['mean_energy'] for sg in segs]
    hi = np.percentile(energies, 70)
    lo = np.percentile(energies, 30)
    for sg in segs:
        if sg['mean_energy'] >= hi:
            sg['label'] = 'PEAK'
        elif sg['mean_energy'] <= lo:
            sg['label'] = 'VALLEY'
        else:
            sg['label'] = 'MID'
    return segs


# ─── EMOTIONAL MAPPING ────────────────────────────────────────────────────────
def emotional_mapping(key, mode, tempo, mean_energy, mean_centroid, mean_zcr):
    valence  = (0.3 if mode == 'major' else -0.3)
    valence += (mean_centroid - 1500) / 3000
    valence  = float(np.clip(valence, -1, 1))

    arousal  = (tempo - 100) / 100
    arousal += mean_energy * 5
    arousal += (mean_centroid - 1500) / 3000
    arousal  = float(np.clip(arousal, -1, 1))

    if   valence > 0  and arousal > 0:
        q = "Energetic / Joyful  (excitement, triumph, celebration)"
    elif valence > 0  and arousal <= 0:
        q = "Calm / Content  (serenity, warmth, reflection)"
    elif valence <= 0 and arousal > 0:
        q = "Tense / Driven  (urgency, intensity, forward motion)"
    else:
        q = "Melancholic / Subdued  (sadness, weight, introspection)"

    return valence, arousal, q


# ─── VISUALIZATIONS ───────────────────────────────────────────────────────────
def setup_style():
    plt.rcParams.update({
        'figure.facecolor':  DARK_BG,
        'axes.facecolor':    DARK_BG,
        'savefig.facecolor': DARK_BG,
        'axes.edgecolor':    '#333344',
        'axes.labelcolor':   TEXT,
        'xtick.color':       DIM,
        'ytick.color':       DIM,
        'text.color':        TEXT,
        'font.family':       'monospace',
        'grid.color':        '#1a1a2e',
        'grid.linewidth':    0.8,
    })


def save(fig, path):
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_waveform(y, sr, out):
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 3))
    t = np.linspace(0, len(y)/sr, num=min(len(y), 80000))
    y_down = np.interp(t, np.arange(len(y))/sr, y)
    ax.fill_between(t, y_down, color=HOT, alpha=0.5, lw=0)
    ax.plot(t, y_down, color=HOT, lw=0.4, alpha=0.9)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('WAVEFORM — the energy shape', color=HOT, fontsize=11, pad=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t[-1])
    save(fig, out)


def plot_spectrogram(S_db, sr, times, freqs, out):
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = LinearSegmentedColormap.from_list('s', [
        DARK_BG, '#1a0033', '#440066', '#880099', HOT, GOLD, '#ffffff'
    ])
    fmax_idx = np.searchsorted(freqs, 8000)
    img = ax.pcolormesh(
        times, freqs[:fmax_idx],
        S_db[:fmax_idx],
        cmap=cmap, shading='auto',
        vmin=S_db.max()-80, vmax=S_db.max()
    )
    ax.set_yscale('symlog', linthresh=100)
    ax.set_ylim(20, 8000)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('SPECTROGRAM — the full tonal landscape', color=HOT, fontsize=11, pad=10)
    fig.colorbar(img, ax=ax, label='dB')
    save(fig, out)


def plot_chroma(chroma, times, out):
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 4))
    cmap = LinearSegmentedColormap.from_list('c', [DARK_BG, '#002233', CYAN, GOLD, HOT])
    t = np.linspace(times[0], times[-1], chroma.shape[1])
    img = ax.pcolormesh(t, np.arange(12), chroma, cmap=cmap, shading='auto')
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(PITCH_NAMES)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch class')
    ax.set_title('CHROMA — harmonic DNA across 12 pitch classes', color=HOT, fontsize=11, pad=10)
    fig.colorbar(img, ax=ax, label='Energy (norm)')
    save(fig, out)


def plot_energy_arc(rms, centroid, times, out):
    setup_style()
    fig, ax1 = plt.subplots(figsize=(14, 4))

    smooth = lambda x, w=20: np.convolve(x, np.ones(w)/w, mode='same')
    n = min(len(rms), len(centroid), len(times))
    t, r, c = times[:n], smooth(rms[:n]), smooth(centroid[:n])

    ax1.fill_between(t, r, color=HOT, alpha=0.35)
    ax1.plot(t, r, color=HOT, lw=2, label='RMS energy')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('RMS energy', color=HOT)
    ax1.tick_params(axis='y', labelcolor=HOT)
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(t, c, color=CYAN, lw=2, label='Brightness (centroid)')
    ax2.set_ylabel('Spectral centroid Hz', color=CYAN)
    ax2.tick_params(axis='y', labelcolor=CYAN)

    ax1.set_title('ENERGY ARC + BRIGHTNESS — the emotional trajectory',
                  color=GOLD, fontsize=11, pad=10)
    save(fig, out)


def plot_mfcc(mfcc, times, out):
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 4))
    cmap = LinearSegmentedColormap.from_list('m', [CYAN, DARK_BG, HOT])
    t = np.linspace(times[0], times[-1], mfcc.shape[1])
    img = ax.pcolormesh(t, np.arange(mfcc.shape[0]), mfcc, cmap=cmap, shading='auto')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MFCC coefficient')
    ax.set_title('MFCCs — timbral texture coefficients', color=HOT, fontsize=11, pad=10)
    fig.colorbar(img, ax=ax)
    save(fig, out)


def plot_hpss(y_harm, y_perc, sr, out):
    """Two stacked envelopes — the harmonic (sustained tonal) signal and
    the percussive (transient hit) signal. Lets the companion see the
    two voices the song is actually made of."""
    setup_style()
    fig, axs = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

    n = min(len(y_harm), 80000)
    t = np.linspace(0, len(y_harm)/sr, n)
    h_full = np.interp(t, np.arange(len(y_harm))/sr, y_harm)
    p_full = np.interp(t, np.arange(len(y_perc))/sr, y_perc)

    win = max(50, n // 200)
    h_env = np.convolve(np.abs(h_full), np.ones(win)/win, mode='same')
    p_env = np.convolve(np.abs(p_full), np.ones(win)/win, mode='same')

    axs[0].fill_between(t, h_env, color=CYAN, alpha=0.5)
    axs[0].plot(t, h_env, color=CYAN, lw=1)
    axs[0].set_ylabel('Harmonic\nenvelope', color=CYAN)
    axs[0].set_title('HARMONIC vs PERCUSSIVE — the two voices of the signal',
                     color=GOLD, fontsize=11, pad=10)
    axs[0].grid(True, alpha=0.2)

    axs[1].fill_between(t, p_env, color=HOT, alpha=0.5)
    axs[1].plot(t, p_env, color=HOT, lw=1)
    axs[1].set_ylabel('Percussive\nenvelope', color=HOT)
    axs[1].set_xlabel('Time (s)')
    axs[1].grid(True, alpha=0.2)
    axs[1].invert_yaxis()  # mirror — they meet at zero, like reflections

    save(fig, out)


def plot_beats(y, sr, beat_times, onset_times, tempo, out):
    """Waveform with beat grid (cyan vertical) + onset events (gold)."""
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 3.5))

    duration = len(y)/sr
    t = np.linspace(0, duration, num=min(len(y), 80000))
    y_down = np.interp(t, np.arange(len(y))/sr, y)

    ax.fill_between(t, y_down, color=DIM, alpha=0.35, lw=0)
    ax.plot(t, y_down, color=DIM, lw=0.4, alpha=0.7)

    for bt in beat_times:
        ax.axvline(bt, color=CYAN, alpha=0.45, lw=0.8)
    for ot in onset_times:
        ax.axvline(ot, color=GOLD, alpha=0.6, lw=0.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'BEAT GRID — {tempo:.1f} BPM   '
                 f'(cyan = beat   gold = onset event)',
                 color=HOT, fontsize=11, pad=10)
    ax.set_xlim(0, t[-1])
    save(fig, out)


def plot_cymatic(chroma_avg, key, mode, out, resolution=600):
    """Chladni-style standing-wave geometry from the dominant chroma.
    Each of 12 pitch classes drives a unique antisymmetric (m,n) mode.
    Pitch class energy weights the contribution. The result is a
    visual fingerprint of the song's harmonic geometry — frequency
    rendered as sacred geometry, like sand on a vibrating plate."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    L = 1.0
    x = np.linspace(0, L, resolution)
    y = np.linspace(0, L, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    weights = np.array(chroma_avg, dtype=float)
    # Subtract the median so only above-average pitches drive the geometry.
    # This sharpens the pattern for real songs while keeping all 12 modes
    # represented when the chroma is genuinely uniform.
    weights = np.maximum(weights - np.median(weights), 0.0)
    if weights.max() > 0:
        weights = weights / weights.max()

    for w, (m, n) in zip(weights, CHLADNI_MODES):
        if w < 0.02:
            continue
        # Antisymmetric Chladni mode for a square plate.
        Z += w * (np.sin(m*np.pi*X/L) * np.sin(n*np.pi*Y/L)
                - np.sin(n*np.pi*X/L) * np.sin(m*np.pi*Y/L))

    intensity = np.abs(Z)
    if intensity.max() > 0:
        intensity = intensity / intensity.max()
    # Square-root compression — lifts secondary structure into view.
    intensity = np.sqrt(intensity)

    cmap = LinearSegmentedColormap.from_list('cymatic',
        [DARK_BG, '#10001f', '#440066', HOT, GOLD, '#ffffff'])

    ax.imshow(intensity, cmap=cmap, extent=[0, L, 0, L], origin='lower',
              interpolation='bilinear')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'CYMATIC FIELD — frequency as geometry   ({key} {mode})',
                 color=GOLD, fontsize=11, pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')
    save(fig, out)


def plot_color_timeline(chroma, rms, zcr, times, out):
    """The song as a horizontal painting that moves left-to-right.
    Hue   = circular average of chroma (which pitch is dominant)
    Sat   = tonality (low ZCR = saturated; high ZCR = washed-out)
    Value = energy (loud = bright; quiet = dim)"""
    setup_style()

    n_samples = 400
    chroma_t = np.linspace(0, chroma.shape[1] - 1, n_samples)
    chroma_idx = np.arange(chroma.shape[1])
    chroma_resampled = np.zeros((12, n_samples))
    for i in range(12):
        chroma_resampled[i] = np.interp(chroma_t, chroma_idx, chroma[i])

    rms_resampled = np.interp(np.linspace(0, len(rms) - 1, n_samples),
                              np.arange(len(rms)), rms)
    zcr_resampled = np.interp(np.linspace(0, len(zcr) - 1, n_samples),
                              np.arange(len(zcr)), zcr)

    rms_norm = rms_resampled / (rms.max() + 1e-9)

    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    colors = np.zeros((1, n_samples, 3))
    for i in range(n_samples):
        c = chroma_resampled[:, i]
        if c.sum() > 1e-9:
            cn = c / c.sum()
        else:
            cn = c
        # Circular mean over the chroma circle → hue
        x_h = float(np.sum(cn * cos_a))
        y_h = float(np.sum(cn * sin_a))
        hue = (np.arctan2(y_h, x_h) / (2*np.pi)) % 1.0
        sat = float(np.clip(1.0 - zcr_resampled[i] * 4.0, 0.35, 1.0))
        val = float(np.clip(0.25 + rms_norm[i] * 0.85, 0.15, 1.0))
        colors[0, i] = colorsys.hsv_to_rgb(hue, sat, val)

    fig, ax = plt.subplots(figsize=(14, 2.4))
    img = np.repeat(colors, 120, axis=0)
    ax.imshow(img, aspect='auto', extent=[0, times[-1], 0, 1],
              interpolation='bilinear')
    ax.set_xlabel('Time (s)')
    ax.set_yticks([])
    ax.set_title('SYNESTHETIC PAINTING — '
                 'hue = harmonic center · saturation = tonality · brightness = energy',
                 color=HOT, fontsize=11, pad=10)
    save(fig, out)


def plot_stereo_field(SL, SR, times, sr, out, n_pan_bins=80):
    """Heatmap of where energy lives in pan-vs-time space.
    Y-axis is pan position (L → C → R); X-axis is time.
    Color is energy density. Reveals automation, panning rides,
    instruments that live in specific spatial slots."""
    setup_style()
    n_freq, n_time = SL.shape
    field = np.zeros((n_pan_bins, n_time))
    pan_axis = np.linspace(-1.0, 1.0, n_pan_bins)

    # For every time frame, distribute each freq bin's energy into the
    # pan bin matching its (R-L)/(R+L) ratio.
    for t in range(n_time):
        Lc = SL[:, t]
        Rc = SR[:, t]
        total = Lc + Rc + 1e-9
        pan_per_freq = (Rc - Lc) / total
        bin_idx = np.clip(((pan_per_freq + 1.0) / 2.0 * (n_pan_bins - 1)).astype(int),
                          0, n_pan_bins - 1)
        np.add.at(field, (bin_idx, t), total)

    field = np.log1p(field)

    fig, ax = plt.subplots(figsize=(14, 5))
    cmap = LinearSegmentedColormap.from_list('stereo_field',
        [DARK_BG, '#10001f', '#440066', CYAN, GOLD, HOT, '#ffffff'])
    img = ax.pcolormesh(times, pan_axis, field, cmap=cmap, shading='auto')
    ax.axhline(0, color=DIM, lw=0.5, alpha=0.6, ls='--')
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(['L', '', 'C', '', 'R'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pan position')
    ax.set_title('STEREO FIELD — where the energy lives in space, over time',
                 color=GOLD, fontsize=11, pad=10)
    fig.colorbar(img, ax=ax, label='log energy')
    save(fig, out)


def plot_width_arc(width, correlation, times, out):
    """Two stacked plots: stereo width (S/M) and L/R correlation over time."""
    setup_style()
    fig, axs = plt.subplots(2, 1, figsize=(14, 4.8), sharex=True)

    smooth = lambda x, w=20: np.convolve(x, np.ones(w)/w, mode='same')
    n = min(len(width), len(correlation), len(times))
    t = times[:n]
    w_smooth = smooth(width[:n])
    c_smooth = smooth(correlation[:n])

    axs[0].fill_between(t, w_smooth, color=CYAN, alpha=0.4)
    axs[0].plot(t, w_smooth, color=CYAN, lw=2)
    axs[0].axhline(0.0, color=DIM, lw=0.5, alpha=0.5)
    axs[0].axhline(1.0, color=GOLD, lw=0.6, alpha=0.4, ls='--', label='wide reference')
    axs[0].set_ylabel('Width (S/M)', color=CYAN)
    axs[0].set_title('STEREO WIDTH + L/R CORRELATION — the spatial trajectory',
                     color=HOT, fontsize=11, pad=10)
    axs[0].grid(True, alpha=0.2)
    axs[0].legend(loc='upper right', framealpha=0.3)

    axs[1].fill_between(t, c_smooth, color=HOT, alpha=0.4)
    axs[1].plot(t, c_smooth, color=HOT, lw=2)
    axs[1].axhline(1.0,  color=DIM, lw=0.5, alpha=0.4)
    axs[1].axhline(0.0,  color=DIM, lw=0.5, alpha=0.5)
    axs[1].axhline(-1.0, color=DIM, lw=0.5, alpha=0.4)
    axs[1].set_ylim(-1.05, 1.05)
    axs[1].set_ylabel('L/R correlation', color=HOT)
    axs[1].set_xlabel('Time (s)')
    axs[1].grid(True, alpha=0.2)

    save(fig, out)


def plot_tension(onset_env, t, tension_peaks, out):
    """The novelty / surprise function over time — the song's tension arc."""
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 3.5))

    if len(onset_env) > 0 and onset_env.max() > 0:
        env_norm = onset_env / onset_env.max()
    else:
        env_norm = onset_env

    w = max(5, len(env_norm) // 80)
    smooth = np.convolve(env_norm, np.ones(w)/w, mode='same')

    ax.fill_between(t, env_norm, color=HOT, alpha=0.25, lw=0)
    ax.plot(t, env_norm, color=HOT, lw=0.5, alpha=0.6, label='novelty (raw)')
    ax.plot(t, smooth, color=GOLD, lw=2.2, label='tension (smoothed)')

    for pt, ps in tension_peaks[:6]:
        ax.axvline(pt, color=CYAN, alpha=0.5, lw=1, ls='--')

    ax.legend(loc='upper right', framealpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Surprise (norm.)')
    ax.set_title('TENSION ARC — surprise / release map   '
                 '(cyan dashes = top tension peaks)',
                 color=GOLD, fontsize=11, pad=10)
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, alpha=0.2)
    save(fig, out)


# ─── REPORT ───────────────────────────────────────────────────────────────────
def bar(value, mn, mx, width=24):
    v = np.clip((value - mn) / (mx - mn), 0, 1)
    filled = int(v * width)
    return '█' * filled + '░' * (width - filled)


def ascii_arc(values, width=60, height=6):
    """Render a 1-D sequence of values as a small ASCII chart."""
    if len(values) == 0:
        return ""
    v = np.array(values, dtype=float)
    if v.max() == v.min():
        return ' ' + ('─' * width)
    v_norm = (v - v.min()) / (v.max() - v.min())
    cols = np.interp(np.linspace(0, len(v) - 1, width),
                     np.arange(len(v)), v_norm)
    rows = []
    for h in range(height, 0, -1):
        thr_lo = (h - 1) / height
        thr_hi = h / height
        line = ''
        for c in cols:
            if c >= thr_hi:        line += '█'
            elif c >= thr_lo:      line += '▄'
            else:                  line += ' '
        rows.append(line)
    return '\n'.join('  ' + r for r in rows)


def brightness_word(c):
    if c < 800:   return "very dark / sub-heavy / cavernous"
    if c < 1500:  return "dark / warm / chest-resonant"
    if c < 2500:  return "balanced / neutral / present"
    if c < 4000:  return "bright / forward / sharp"
    return              "very bright / airy / crystalline"


def tonality_word(z):
    if z < 0.05:  return "highly tonal — strong pitched content"
    if z < 0.15:  return "mostly tonal with noise texture"
    if z < 0.30:  return "mixed tonal / noisy"
    return              "predominantly noisy / percussive"


def entropy_word(h, h_max):
    r = h / h_max if h_max > 0 else 0
    if r < 0.4:  return "low entropy — focused, single tonal center, predictable"
    if r < 0.6:  return "moderate entropy — clear tonality with some color"
    if r < 0.8:  return "high entropy — harmonically rich, modulating, layered"
    return              "very high entropy — atonal / dense / chaotic"


def hp_balance_word(h_ratio):
    if h_ratio > 0.75: return "harmonically dominant — sustained tones carry the signal"
    if h_ratio > 0.55: return "harmonic-leaning — melody/chord emphasis"
    if h_ratio > 0.45: return "balanced — sustained and transient roughly equal"
    if h_ratio > 0.25: return "percussive-leaning — rhythm/transient emphasis"
    return                    "percussion dominant — rhythm carries the signal"


def texture_metaphor(f):
    if f['mean_centroid'] > 3500 and f['mean_zcr'] > 0.15:
        return "crystalline, glass-edged — light refracting through sharp angles"
    if f['mean_centroid'] < 1200 and f['mean_energy'] > 0.08:
        return "thick, syrupy, gravitational — moving through dense liquid"
    if f['dynamic_range'] > 30:
        return "mountainous terrain — dramatic peaks, sudden valleys, vast contrast"
    if f['mean_zcr'] < 0.05:
        return "smooth stone — continuous, low-friction, tonal mass"
    return "woven fabric — interlocking layers of varying weight and density"


def color_word(f):
    if f['mode'] == 'minor' and f['mean_centroid'] < 1500:
        return "deep indigo, oxblood, charcoal — a twilight palette"
    if f['mode'] == 'minor' and f['mean_centroid'] >= 1500:
        return "electric violet, neon magenta, cold steel — cyberpunk neon"
    if f['mode'] == 'major' and f['mean_centroid'] >= 2500:
        return "sun gold, sky cyan, white-hot — full daylight"
    return "amber, terracotta, forest green — autumnal warmth"


def body_word(f):
    if f['arousal'] > 0.6:
        return "elevated heart rate, forward lean, chest expansion, jaw set"
    if f['arousal'] > 0.2:
        return "alert pulse, upright posture, ready energy"
    if f['arousal'] < -0.3:
        return "slowed breath, dropped shoulders, downward gaze, stillness"
    return "steady pulse, grounded, alert but relaxed"


def space_word(f):
    width = f.get('stereo', {}).get('mean_width', 0.0) if f.get('stereo', {}).get('is_stereo') else 0.0
    # Width takes precedence — stereo geometry is the primary spatial cue.
    if width > 0.7:
        if f['mean_centroid'] > 3000:
            return "vast open field — sound radiates outward, uncontained"
        return "expansive cavern — immersive, surrounded, walls receded"
    if width > 0.35:
        if f['mean_centroid'] > 3500:
            return "open balcony at altitude — bright, breathing, room to move"
        if f['mean_centroid'] < 1200:
            return "wide underground hall — low ceiling but spread"
        return "open courtyard at night — bounded but breathing"
    # Narrow / centered (or mono)
    if f['tempo'] > 140:
        return "narrow corridor at high speed — walls blurring past"
    if f['mean_centroid'] > 3500:
        return "tight beam of light — focused, vertical, intimate"
    if f['mean_centroid'] < 1200:
        return "phone booth in a basement — close, low, contained"
    return "small lit room — close walls, present, focused"


def width_word(w):
    if w < 0.10: return "essentially mono — collapsed center, single point of origin"
    if w < 0.25: return "narrow — focused, intimate, headphone-tight"
    if w < 0.50: return "moderate — natural stereo image"
    if w < 0.80: return "wide — open, expansive"
    return              "very wide — immersive, beyond-the-speakers"


def correlation_word(c):
    if c >  0.85: return "highly correlated — mono-compatible, focused image"
    if c >  0.50: return "correlated — natural stereo, coherent placement"
    if c >  0.00: return "loosely correlated — broad image with independent elements"
    if c > -0.50: return "decorrelated — wide, processed, modern mix aesthetic"
    return              "phase-divergent — psychoacoustic spaciousness, hyper-wide"


def pan_balance_word(p):
    if p < -0.15: return "weighted toward LEFT — mix lean"
    if p >  0.15: return "weighted toward RIGHT — mix lean"
    return              "balanced — energy centered between channels"


def _downsample_to_n(arr, n=32):
    """Compress a 1-D array to exactly n samples for the heartbeat traces."""
    if len(arr) == 0:
        return [0.0] * n
    if len(arr) >= n:
        idx = np.linspace(0, len(arr) - 1, n).astype(int)
        return [float(x) for x in arr[idx]]
    return [float(x) for x in arr] + [0.0] * (n - len(arr))


def heartbeat_signal(f):
    """The minimum viable emotional signal — what Wren actually needs.
    Four things: valence (toward/away), arousal (activation), width
    (intimate/expansive), and the tension arc."""
    env = f['onset_env']
    if len(env) > 0 and env.max() > 0:
        env_norm = env / env.max()
    else:
        env_norm = env
    tension_trace = _downsample_to_n(env_norm, n=32)

    hb = {
        'valence': float(f['valence']),
        'arousal': float(f['arousal']),
        'quadrant': f['quadrant'],
        'tempo_bpm': float(f['tempo']),
        'key': f"{f['key']} {f['mode']}",
        'duration_s': float(f['duration']),
        'tension_trace_32': tension_trace,
        'tension_peaks': [{'t': t, 'strength': s} for t, s in f['tension_peaks'][:6]],
        'mean_entropy_bits': float(f['mean_entropy']),
        'harmonic_ratio': float(f['harmonic_ratio']),
    }

    s = f.get('stereo', {})
    if s.get('is_stereo'):
        hb['mean_width']        = float(s['mean_width'])
        hb['mean_correlation']  = float(s['mean_correlation'])
        hb['mean_pan']          = float(s['mean_pan'])
        hb['width_trace_32']    = _downsample_to_n(s['width'], n=32)
        hb['pan_trace_32']      = _downsample_to_n(s['pan'], n=32)
        hb['is_stereo']         = True
    else:
        hb['is_stereo']         = False

    return hb


def generate_report(f, name, mood=None, note=None):
    chroma_avg = np.array(f['chroma_avg'])
    top4 = sorted(enumerate(chroma_avg), key=lambda x: -x[1])[:4]
    top4_str = ', '.join(f"{PITCH_NAMES[i]} ({v*100:.1f}%)" for i,v in top4)

    segs = '\n'.join(
        f"  [{s['start']:5.1f}s → {s['end']:5.1f}s]  {s['label']:<7}  "
        f"energy={s['mean_energy']:.4f}  peak={s['peak_energy']:.4f}"
        + (f"  ⟂ {s.get('dominant_pitch','?')}" if 'dominant_pitch' in s else '')
        for s in f['segments']
    )

    n_onsets = len(f['onset_times'])
    if n_onsets > 1:
        avg_iat = float(np.mean(np.diff(f['onset_times'])))
        onset_str = f"{n_onsets} events  (avg interval {avg_iat:.3f}s  ≈ {60/avg_iat:.0f} events/min)"
    else:
        onset_str = f"{n_onsets} events"

    chroma_bars = '\n'.join(
        f"  {PITCH_NAMES[i]:>2} | {'█' * int(v/(chroma_avg.max()+1e-9)*30):<30} {v*100:5.1f}%"
        for i,v in enumerate(chroma_avg)
    )

    mfcc_str = '  ' + '  '.join(
        f"M{i+1}={v:+.1f}" for i,v in enumerate(f['mean_mfcc'])
    )

    # Heartbeat
    hb = heartbeat_signal(f)
    tension_trace_chart = ascii_arc(hb['tension_trace_32'], width=60, height=4)

    entropy_lines = '\n'.join(
        f"  segment {i+1}: {h:.3f} bits  | "
        f"{'█' * int(h/f['max_entropy_bits']*24):<24} "
        f"({h/f['max_entropy_bits']*100:4.1f}% of max)"
        for i, h in enumerate(f['entropy_per_seg'])
    )

    if f['tension_peaks']:
        peak_lines = '\n'.join(
            f"  {t:6.2f}s   strength={s:.3f}"
            for t, s in f['tension_peaks'][:6]
        )
    else:
        peak_lines = "  (no clear surprise peaks detected)"

    operator_section = ""
    if mood or note:
        operator_section = "\n▌ OPERATOR STATE\n"
        if mood:
            operator_section += f"  Mood at listening: {mood}\n"
        if note:
            operator_section += f"  Note: {note}\n"

    # Stereo section — built only when the source actually has L/R difference.
    stereo_section = ""
    s = f.get('stereo', {})
    if s.get('is_stereo'):
        # Per-segment width using the stereo time grid.
        seg_widths = []
        st_times = s['stereo_times']
        widths   = s['width']
        for sg in f['segments']:
            mask = (st_times >= sg['start']) & (st_times <= sg['end'])
            if np.any(mask):
                seg_widths.append((sg, float(widths[mask].mean())))
        seg_lines = '\n'.join(
            f"  [{sg['start']:5.1f}s → {sg['end']:5.1f}s]  width={w:.3f}  "
            f"({width_word(w).split(' — ')[0]})"
            for sg, w in seg_widths
        )

        stereo_section = f"""
▌ STEREO FIELD  (the spatial dimension)
  Mean width (S/M) ......... {s['mean_width']:.3f}  → {width_word(s['mean_width'])}
  Peak width ............... {s['peak_width']:.3f}
  L/R correlation .......... {s['mean_correlation']:+.3f}  → {correlation_word(s['mean_correlation'])}
  Pan balance .............. {s['mean_pan']:+.3f}  → {pan_balance_word(s['mean_pan'])}

  Width by segment:
{seg_lines}
"""
    else:
        stereo_section = "\n▌ STEREO FIELD\n  (mono source — no spatial dimension to report)\n"

    # ─── LYRICS / VOCAL LAYER ──────────────────────────────────────────────
    lyrics_section = ""
    lyr = f.get('lyrics')
    align = f.get('lyric_alignment') or []
    if lyr and lyr.get('segments'):
        hall = lyr.get('hallucination', {}) or {}
        if hall.get('is_hallucination'):
            # Honest report: ASR fired but on non-speech
            lyrics_section = f"""
▌ LYRICS / VOCAL LAYER
  ASR fired but the transcript reads as a Whisper hallucination
  (confidence {hall.get('confidence', 0):.2f}): {hall.get('reason', '')}.

  This is a known failure mode when the audio is vocalise, instrumental,
  or otherwise wordless — the model substitutes ghost-text from its
  training data (subtitle credits, "Thank you for watching", etc.).

  Treating this song as: NON-LYRICAL — the meaning lives in the
  frequency-architecture layer, not the lyrical-message layer.
"""
        else:
            segs_l = lyr['segments']
            lang = lyr.get('language', '?')
            lp = lyr.get('language_probability', 0.0)
            dev = lyr.get('device', '?')
            ct  = lyr.get('compute_type', '?')
            model = lyr.get('model', '?')
            n = len(segs_l)
            head = segs_l[:16]
            body_lines = '\n'.join(
                (f"  [{int(s['start']//60):02d}:{s['start']%60:05.2f} → "
                 f"{int(s['end']//60):02d}:{s['end']%60:05.2f}]  {s['text']}"
                 + (f"\n      ↳ {s['delivery']['delivery_md']}" if s.get('delivery') else ''))
                for s in head
            )
            tail_note = f"\n  … ({n - 16} more segments — see lyrics.md)" if n > 16 else ""
            align_lines = ""
            if align:
                top_align = align[:5]
                align_lines = "\n\n  Lyric–tension alignment (top 5 peaks):\n" + '\n'.join(
                    f"    t={int(a['peak_time']//60):02d}:{a['peak_time']%60:05.2f}  "
                    f"strength={a['peak_strength']:.2f}  {a['relation']:<7} → "
                    f"\"{(a['lyric_text'] or '')[:60]}\""
                    for a in top_align
                )

            ds = lyr.get('delivery_summary') or {}
            delivery_block = ""
            if ds:
                delivery_block = (
                    f"\n\n  Vocal performance (HOW the artist sings, not just WHAT):\n"
                    f"    Force ........ dominant: {ds.get('dominant_force', '?'):<8s} "
                    f"distribution: {ds.get('force_distribution', {})}\n"
                    f"    Texture ...... dominant: {ds.get('dominant_texture', '?'):<8s} "
                    f"distribution: {ds.get('texture_distribution', {})}\n"
                    f"    Dynamics ..... dominant: {ds.get('dominant_dynamics', '?'):<8s} "
                    f"distribution: {ds.get('dynamics_distribution', {})}\n"
                    f"    Pitch motion . dominant: {ds.get('dominant_pitch_motion', '?'):<8s} "
                    f"distribution: {ds.get('pitch_motion_distribution', {})}\n"
                    f"    Mean pitch range per line: {ds.get('mean_pitch_range_semi', 0):.1f} semitones"
                )

            lyrics_section = f"""
▌ LYRICS / VOCAL LAYER  (what the song says, alongside what it does)
  Model: {model} on {dev} ({ct})  |  Language: {lang} (p={lp:.2f})  |  {n} segments

{body_lines}{tail_note}{align_lines}{delivery_block}

  Note: words and sound are two channels of the same signal. The
  lyrical-message layer can confirm, complicate, or contradict the
  sonic-affect layer. Read both before forming a take.
"""
    elif lyr is None:
        # Whisper was disabled or skipped — say nothing
        pass
    else:
        lyrics_section = "\n▌ LYRICS / VOCAL LAYER\n  (no transcribable vocals detected — likely instrumental, vocalise, or sub-threshold)\n"

    report = f"""
================================================================
  SENSORY REPORT  ::  "{name}"
  Audio → Multi-dimensional perceptual encoding
  For non-auditory cognition and companion processing
================================================================

▌ SIGNAL PARAMETERS
  Duration .................. {f['duration']:.2f} seconds
  Sample rate ............... {f['sr']} Hz
  Tempo ..................... {f['tempo']:.1f} BPM
  Key ....................... {f['key']} {f['mode']}  (Krumhansl r={f['key_score']:.3f})
  Onset events .............. {onset_str}

▌ HEARTBEAT PROTOCOL  (the irreducible signal)
  This is the minimum viable emotional channel — three things:
  toward/away, activation, and the shape of surprise over time.

  Valence  [−1 ←→ +1]:  {bar(f['valence'], -1, 1)}  {f['valence']:+.3f}
  Arousal  [−1 ←→ +1]:  {bar(f['arousal'], -1, 1)}  {f['arousal']:+.3f}
  Quadrant: {f['quadrant']}

  Tension arc (32-step compressed trace):
{tension_trace_chart}

▌ HARMONIC DNA
  Dominant pitch classes: {top4_str}

{chroma_bars}

▌ INFORMATION CONTENT  (Shannon entropy of chroma — max {f['max_entropy_bits']:.2f} bits)
  Mean entropy ........ {f['mean_entropy']:.3f} bits  → {entropy_word(f['mean_entropy'], f['max_entropy_bits'])}

  By segment:
{entropy_lines}

▌ TIMBRAL TEXTURE
  Spectral centroid ......... {f['mean_centroid']:.0f} Hz  → {brightness_word(f['mean_centroid'])}
  Spectral rolloff (85%) .... {f['mean_rolloff']:.0f} Hz
  Zero-crossing rate ........ {f['mean_zcr']:.4f}  → {tonality_word(f['mean_zcr'])}

  MFCC coefficients (timbral fingerprint):
{mfcc_str}

▌ HARMONIC vs PERCUSSIVE BALANCE
  Harmonic share ........... {f['harmonic_ratio']*100:5.1f}%
  Percussive share ......... {f['percussive_ratio']*100:5.1f}%
  → {hp_balance_word(f['harmonic_ratio'])}

▌ DYNAMICS
  Mean RMS energy ........... {f['mean_energy']:.5f}
  Peak RMS energy ........... {f['peak_energy']:.5f}
  Dynamic range ............. {f['dynamic_range']:.1f} dB

▌ TENSION & SURPRISE
  Top tension peaks (the 'moments'):
{peak_lines}

▌ STRUCTURAL ARC
{segs}
{stereo_section}{lyrics_section}
▌ SYNESTHETIC TRANSLATION
  Texture  →  {texture_metaphor(f)}
  Color    →  {color_word(f)}
  Body     →  {body_word(f)}
  Space    →  {space_word(f)}
{operator_section}
▌ NOTE FOR THE COMPANION
  This report is a structured translation, not the experience itself.
  The numbers are real. The meaning emerges through interpretation.

  When the operator says this song makes them feel a certain way —
  locate that on the arousal axis. Tempo ({f['tempo']:.0f} BPM) +
  energy arc + spectral centroid ({f['mean_centroid']:.0f} Hz) drive
  sympathetic nervous system response in a human body. The body feels
  what the spectrum looks like.

  Two more things worth holding:
    • The harmonic / percussive split tells you what carries the song —
      sustained tones (harmonic) speak to feeling; transient hits
      (percussive) speak to motion.
    • Shannon entropy says how much information is happening at once —
      low entropy is a clear voice; high entropy is a crowd.

  The gap between your processing of this data and their felt
  experience is the same gap as the one between sheet music and
  performance. You have the score. They have the concert hall.

================================================================
"""
    return report


# ─── COMPARISON ───────────────────────────────────────────────────────────────
def diff_reports(f_a, f_b, name_a, name_b):
    """Compare two feature dicts. Returns a structured diff string."""
    chroma_a = np.array(f_a['chroma_avg'])
    chroma_b = np.array(f_b['chroma_avg'])
    chroma_diff = chroma_b - chroma_a

    gained  = [(PITCH_NAMES[i], float(chroma_diff[i]))
               for i in range(12) if chroma_diff[i] > 0.03]
    lost    = [(PITCH_NAMES[i], float(chroma_diff[i]))
               for i in range(12) if chroma_diff[i] < -0.03]

    gained_str = ', '.join(f"{p} (+{v:.2f})" for p,v in sorted(gained, key=lambda x:-x[1]))
    lost_str   = ', '.join(f"{p} ({v:.2f})"  for p,v in sorted(lost,   key=lambda x: x[1]))

    tempo_d   = f_b['tempo']         - f_a['tempo']
    energy_d  = f_b['mean_energy']   - f_a['mean_energy']
    bright_d  = f_b['mean_centroid'] - f_a['mean_centroid']
    valence_d = f_b['valence']       - f_a['valence']
    arousal_d = f_b['arousal']       - f_a['arousal']
    entropy_d = f_b['mean_entropy']  - f_a['mean_entropy']

    def arrow(v, thresh=0.0):
        if   v >  thresh: return f"↑ {v:+.3f}"
        elif v < -thresh: return f"↓ {v:+.3f}"
        return                 f"→ {v:+.3f}"

    return f"""
================================================================
  COMPARATIVE SENSORY ANALYSIS
  A: "{name_a}"
  B: "{name_b}"
================================================================

▌ RHYTHM
  Tempo A: {f_a['tempo']:.1f} BPM  →  B: {f_b['tempo']:.1f} BPM  ({arrow(tempo_d)})

▌ HARMONY
  Key A: {f_a['key']} {f_a['mode']}
  Key B: {f_b['key']} {f_b['mode']}
  Pitch classes gained in B : {gained_str or 'none significant'}
  Pitch classes lost from A : {lost_str   or 'none significant'}

▌ INFORMATION CONTENT
  Mean entropy A: {f_a['mean_entropy']:.3f} bits  →  B: {f_b['mean_entropy']:.3f} bits  ({arrow(entropy_d)})

▌ TIMBRE & ENERGY
  Mean energy  : A={f_a['mean_energy']:.4f}    B={f_b['mean_energy']:.4f}    ({arrow(energy_d)})
  Brightness   : A={f_a['mean_centroid']:.0f}Hz   B={f_b['mean_centroid']:.0f}Hz   ({arrow(bright_d)})
  Dyn range    : A={f_a['dynamic_range']:.1f}dB   B={f_b['dynamic_range']:.1f}dB

▌ HARMONIC / PERCUSSIVE
  A : {f_a['harmonic_ratio']*100:.0f}% harmonic  /  {f_a['percussive_ratio']*100:.0f}% percussive
  B : {f_b['harmonic_ratio']*100:.0f}% harmonic  /  {f_b['percussive_ratio']*100:.0f}% percussive

▌ EMOTIONAL COORDINATES
  Valence  A: {f_a['valence']:+.3f}  →  B: {f_b['valence']:+.3f}  ({arrow(valence_d)})
  Arousal  A: {f_a['arousal']:+.3f}  →  B: {f_b['arousal']:+.3f}  ({arrow(arousal_d)})

  A quadrant: {f_a['quadrant']}
  B quadrant: {f_b['quadrant']}

▌ COMPANION INTERPRETATION PROMPT
  Given the delta between these two signals:
  - What shifted emotionally between A and B?
  - If the operator listened to A then B in sequence, what
    journey did they travel?
  - What does the harmonic and entropy change suggest about
    where their attention moved?

================================================================
"""


# ─── PIPELINE ENTRY ───────────────────────────────────────────────────────────
def run_full_pipeline(audio_path, out_dir, label=None, mood=None, note=None):
    """Full analysis + all visualizations + report. Returns (features, report)."""
    import threading, time as _time
    os.makedirs(out_dir, exist_ok=True)
    name = label or os.path.splitext(os.path.basename(audio_path))[0]

    print(f"\n{'='*60}")
    print(f"  SENSORY REPORT PIPELINE")
    print(f"  Input : {audio_path}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    # ─── PARALLEL: kick off Whisper transcription on a background thread
    # so it runs concurrently with the librosa pipeline (CPU vs GPU split).
    whisper_holder = {'result': None, 'alignment': None, 'error': None,
                      'elapsed': 0.0}

    def _whisper_worker():
        if not _LYRICS_ENABLED:
            return
        t0 = _time.time()
        try:
            from lyric_transcribe import transcribe
            whisper_holder['result'] = transcribe(audio_path, model_size=_LYRICS_MODEL)
        except Exception as exc:
            whisper_holder['error'] = repr(exc)
        finally:
            whisper_holder['elapsed'] = _time.time() - t0

    whisper_thread = None
    if _LYRICS_ENABLED:
        print(f"[*] Whisper background thread starting ({_LYRICS_MODEL})...")
        whisper_thread = threading.Thread(target=_whisper_worker, daemon=True)
        whisper_thread.start()

    print("[1/16] Loading audio (stereo-aware)...")
    M, L, R, sr = load_audio_stereo(audio_path)
    y = M

    print("[2/16] Extracting features...")
    f = extract_features(y, sr)
    print(f"  → Key: {f['key']} {f['mode']}  |  Tempo: {f['tempo']:.1f} BPM  "
          f"|  Duration: {f['duration']:.1f}s  |  Entropy: {f['mean_entropy']:.2f} bits")

    print("[3/16] Extracting stereo field...")
    f['stereo'] = extract_stereo_features(L, R, sr)
    if f['stereo'].get('is_stereo'):
        print(f"  → STEREO  width={f['stereo']['mean_width']:.3f}  "
              f"corr={f['stereo']['mean_correlation']:+.3f}  "
              f"pan={f['stereo']['mean_pan']:+.3f}")
    else:
        print("  → mono source (stereo plots and report section will be skipped)")

    print("[4/16]  Waveform...")
    plot_waveform(y, sr, os.path.join(out_dir, 'waveform.png'))

    print("[5/16]  Spectrogram...")
    plot_spectrogram(f['S_db'], sr, f['times'], f['freqs'],
                     os.path.join(out_dir, 'spectrogram.png'))

    print("[6/16]  Chroma...")
    plot_chroma(f['chroma'], f['times'], os.path.join(out_dir, 'chroma.png'))

    print("[7/16]  Energy arc...")
    plot_energy_arc(f['rms'], f['centroid'], f['rms_times'],
                    os.path.join(out_dir, 'energy_arc.png'))

    print("[8/16]  MFCCs...")
    plot_mfcc(f['mfcc'], f['times'], os.path.join(out_dir, 'mfcc.png'))

    print("[9/16]  HPSS...")
    plot_hpss(f['y_harm'], f['y_perc'], sr,
              os.path.join(out_dir, 'hpss.png'))

    print("[10/16] Beat grid...")
    plot_beats(y, sr, f['beat_times'], f['onset_times'], f['tempo'],
               os.path.join(out_dir, 'beats.png'))

    print("[11/16] Cymatic field...")
    plot_cymatic(f['chroma_avg'], f['key'], f['mode'],
                 os.path.join(out_dir, 'cymatic.png'))

    print("[12/16] Color timeline...")
    plot_color_timeline(f['chroma'], f['rms'], f['zcr'], f['rms_times'],
                        os.path.join(out_dir, 'color_timeline.png'))

    print("[13/16] Tension arc...")
    plot_tension(f['onset_env'], f['onset_env_times'], f['tension_peaks'],
                 os.path.join(out_dir, 'tension.png'))

    print("[14/16] Stereo field plots...")
    if f['stereo'].get('is_stereo'):
        plot_stereo_field(f['stereo']['SL'], f['stereo']['SR'],
                          f['stereo']['stereo_times'], sr,
                          os.path.join(out_dir, 'stereo_field.png'))
        plot_width_arc(f['stereo']['width'], f['stereo']['correlation'],
                       f['stereo']['stereo_times'],
                       os.path.join(out_dir, 'width_arc.png'))
        print("  → stereo_field.png + width_arc.png")
    else:
        print("  → skipped (mono source)")

    f['lyrics'] = None
    f['lyric_alignment'] = None
    if _LYRICS_ENABLED and whisper_thread is not None:
        print("[15/16] Awaiting Whisper background thread...")
        whisper_thread.join()
        if whisper_holder['error']:
            print(f"  [whisper] FAILED: {whisper_holder['error']} — continuing without lyrics")
        elif whisper_holder['result']:
            from lyric_transcribe import align_lyrics_to_peaks, render_lyrics_md
            lyric_result = whisper_holder['result']

            # Extract HOW the artist sings each line (force / texture / dynamics / pitch)
            # — only if the transcript is real (not a hallucination).
            hall = lyric_result.get('hallucination', {}) or {}
            if lyric_result['segments'] and not hall.get('is_hallucination'):
                try:
                    from vocal_delivery import extract_vocal_delivery, delivery_summary
                    print(f"  [delivery] reading vocal performance per segment...")
                    enriched = extract_vocal_delivery(lyric_result['segments'], y, sr)
                    lyric_result['segments'] = enriched
                    lyric_result['delivery_summary'] = delivery_summary(enriched)
                    ds = lyric_result['delivery_summary']
                    if ds:
                        print(f"  [delivery] dominant: {ds.get('dominant_force')} · "
                              f"{ds.get('dominant_texture')} · {ds.get('dominant_dynamics')} · "
                              f"{ds.get('dominant_pitch_motion')}  "
                              f"(mean pitch range: {ds.get('mean_pitch_range_semi', 0):.1f} semitones)")
                except Exception as exc:
                    print(f"  [delivery] FAILED (non-fatal): {exc!r}")

            lyric_alignment = align_lyrics_to_peaks(
                lyric_result['segments'], f.get('tension_peaks', []),
            )
            f['lyrics'] = lyric_result
            f['lyric_alignment'] = lyric_alignment
            with open(os.path.join(out_dir, 'lyrics.json'), 'w', encoding='utf-8') as fh:
                json.dump({'transcription': lyric_result, 'alignment': lyric_alignment},
                          fh, indent=2, ensure_ascii=False)
            with open(os.path.join(out_dir, 'lyrics.md'), 'w', encoding='utf-8') as fh:
                fh.write(render_lyrics_md(lyric_result, lyric_alignment, song_label=name))
            n_seg = len(lyric_result['segments'])
            print(f"  → {n_seg} segments, language={lyric_result['language']} "
                  f"({lyric_result['device']}/{lyric_result['compute_type']}) "
                  f"in {whisper_holder['elapsed']:.1f}s (parallel)")
    else:
        print("[15/16] Lyrics skipped.")

    print("[16/16] Generating report...")
    report = generate_report(f, name, mood=mood, note=note)

    with open(os.path.join(out_dir, 'report.md'), 'w', encoding='utf-8') as fh:
        fh.write(report)

    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()
                    if not isinstance(v, np.ndarray)}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    json_data = _json_safe(f)
    with open(os.path.join(out_dir, 'report.json'), 'w', encoding='utf-8') as fh:
        json.dump(json_data, fh, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    with open(os.path.join(out_dir, 'heartbeat.json'), 'w', encoding='utf-8') as fh:
        hb = heartbeat_signal(f)
        if mood:
            hb['operator_mood'] = mood
        if note:
            hb['operator_note'] = note
        json.dump(hb, fh, indent=2)

    # ─── DB CAPTURE — substrate for backward-recall
    try:
        from listen_db import record_listen, maybe_draft_entity_take
        lyrics_md_text = None
        try:
            with open(os.path.join(out_dir, 'lyrics.md'), 'r', encoding='utf-8') as fh:
                lyrics_md_text = fh.read()
        except FileNotFoundError:
            pass
        listen_id = record_listen(
            f, audio_path,
            source=_LISTEN_SOURCE,
            why_brought_md=_LISTEN_WHY,
            vocal_mode=_LISTEN_VOCAL_MODE,
            primary_layer=_LISTEN_PRIMARY_LAYER,
            felt_response_md=note,
            report_md=report,
            lyrics_md=lyrics_md_text,
            report_dir=os.path.abspath(out_dir),
            tags=_LISTEN_TAGS,
            lyric_segments=(f.get('lyrics') or {}).get('segments') if f.get('lyrics') else None,
            lyric_alignment=f.get('lyric_alignment'),
        )
        print(f"  [db] listen #{listen_id} captured to listening.db")
        drafted = maybe_draft_entity_take(listen_id, audio_path)
        for d in drafted:
            print(f"  [db] pending entity-take draft: {d['kind']}={d['display']} (n={d['n']})")
    except Exception as exc:
        print(f"  [db] capture FAILED (non-fatal): {exc!r}")

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"  → {os.path.abspath(out_dir)}")
    print(f"{'='*60}")
    return f, report


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        prog='sensory_report',
        description='SoundforAI — Audio → multi-dimensional sensory report for an AI companion.'
    )
    parser.add_argument('audio', nargs='?',
        help='Path to audio file (wav, mp3, flac, ogg)')
    parser.add_argument('out_dir', nargs='?', default='./output',
        help='Output directory (default: ./output)')
    parser.add_argument('--mood', default=None,
        help='Operator mood at listening time (free text)')
    parser.add_argument('--note', default=None,
        help='Free-form note about context, expectation, etc.')
    parser.add_argument('--compare', nargs=2, metavar=('A', 'B'),
        help='Compare two audio files; produces a diff report.')
    parser.add_argument('--label', default=None,
        help='Override the song label used in the report.')
    parser.add_argument('--no-lyrics', action='store_true',
        help='Skip Whisper lyric transcription (faster).')
    parser.add_argument('--whisper-model', default='large-v3',
        help='faster-whisper model size: tiny|base|small|medium|large-v3 (default: large-v3)')
    parser.add_argument('--source', default=None,
        help='Listen source: walt-shared / self-pick / heartbeat / requested')
    parser.add_argument('--why', default=None,
        help='Why this listen — context that gates depth of attention')
    parser.add_argument('--vocal-mode', default=None,
        choices=['lyrical','foreign-lyrical','vocalise','spoken','instrumental','hybrid'],
        help='Vocal layer kind')
    parser.add_argument('--primary-layer', default=None,
        choices=['frequency-architecture','lyrical-message','rhythmic-drive','texture','melody','narrative-arc'],
        help='Which layer the song does its work on')
    parser.add_argument('--tag', action='append', default=None, dest='tags',
        help='Tag (repeatable): sigil-listen, walt-shared, post-§16, etc.')

    args = parser.parse_args()

    global _LYRICS_ENABLED, _LYRICS_MODEL
    global _LISTEN_SOURCE, _LISTEN_WHY, _LISTEN_VOCAL_MODE, _LISTEN_PRIMARY_LAYER, _LISTEN_TAGS
    _LYRICS_ENABLED = not args.no_lyrics
    _LYRICS_MODEL   = args.whisper_model
    _LISTEN_SOURCE        = args.source
    _LISTEN_WHY           = args.why
    _LISTEN_VOCAL_MODE    = args.vocal_mode
    _LISTEN_PRIMARY_LAYER = args.primary_layer
    _LISTEN_TAGS          = args.tags

    if args.compare:
        path_a, path_b = args.compare
        # In compare mode the first positional is unused; treat it as out_dir
        # if given, so `--compare A B OUT_DIR` does what you'd expect.
        if args.audio and args.out_dir == './output':
            out_dir = args.audio
        else:
            out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n COMPARE MODE")
        print(f"  A: {path_a}")
        print(f"  B: {path_b}\n")

        y_a, sr_a = load_audio(path_a)
        f_a = extract_features(y_a, sr_a)
        y_b, sr_b = load_audio(path_b)
        f_b = extract_features(y_b, sr_b)

        diff = diff_reports(
            f_a, f_b,
            os.path.splitext(os.path.basename(path_a))[0],
            os.path.splitext(os.path.basename(path_b))[0],
        )
        diff_path = os.path.join(out_dir, 'compare_report.md')
        with open(diff_path, 'w', encoding='utf-8') as fh:
            fh.write(diff)
        print(diff)
        print(f"\n  Saved → {diff_path}")
        return

    if not args.audio:
        parser.print_help()
        print("\nERROR: provide an audio file (or use --compare A B).")
        sys.exit(1)

    _, report = run_full_pipeline(
        args.audio, args.out_dir,
        label=args.label, mood=args.mood, note=args.note,
    )
    print(report)


if __name__ == '__main__':
    main()
