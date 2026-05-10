"""
SoundforWren_MCP.py
===================
MCP server exposing audio analysis as callable tools for an AI companion.

USAGE:
    python SoundforWren_MCP.py

WIRE INTO AN MCP-AWARE CLIENT (e.g. Claude Code via .claude/mcp.json):
    {
      "mcpServers": {
        "soundforwren": {
          "command": "python",
          "args": ["SoundforWren_MCP.py"]
        }
      }
    }

The "args" path is resolved relative to the MCP client's working directory.
If you run the client from a different directory, use an absolute path instead.
"""

import asyncio
import json
import os
import subprocess
import warnings
import datetime
warnings.filterwarnings('ignore')

import numpy as np

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from sensory_report import (
    load_audio,
    extract_features,
    generate_report,
    heartbeat_signal,
    diff_reports,
    run_full_pipeline,
    PITCH_NAMES,
)

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, 'output')
MEMORY_FILE   = os.path.join(PROJECT_ROOT, 'companion_memory.json')
TEMP_AUDIO    = os.path.join(PROJECT_ROOT, 'song_temp.wav')

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── memory helpers ────────────────────────────────────────────────────────────
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"entries": []}


def save_memory(mem):
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(mem, f, indent=2, default=str)


def next_report_id(mem):
    if not mem["entries"]:
        return "RPT-001"
    last = mem["entries"][-1].get("report_id", "RPT-000")
    n = int(last.split("-")[1]) + 1
    return f"RPT-{n:03d}"


def memory_entry(features, label, source_path, mood=None, note=None, query=None):
    """Build a structured memory record from a feature dict."""
    entry = {
        "report_id":    None,  # filled by caller
        "timestamp":    datetime.datetime.now().isoformat(),
        "song_name":    label,
        "file_path":    source_path,
        "key":          f"{features['key']} {features['mode']}",
        "tempo":        features['tempo'],
        "valence":      features['valence'],
        "arousal":      features['arousal'],
        "quadrant":     features['quadrant'],
        "mean_entropy": features['mean_entropy'],
        "harmonic_ratio": features['harmonic_ratio'],
        "report_path":  os.path.join(OUTPUT_DIR, 'report.md'),
        "companion_response": None,
    }
    if mood:  entry["operator_mood"] = mood
    if note:  entry["operator_note"] = note
    if query: entry["query"] = query
    return entry


# ── MCP SERVER ────────────────────────────────────────────────────────────────
app = Server("soundforwren")


@app.list_tools()
async def list_tools():
    return [

        types.Tool(
            name="analyze_audio",
            description=(
                "Full sensory analysis of a local audio file. "
                "Returns tempo, key, chroma, MFCCs, spectral features, "
                "Shannon entropy, harmonic/percussive split, tension peaks, "
                "emotional coordinates (valence/arousal), synesthetic metaphors, "
                "and structural segmentation. Saves 10 visualization PNGs and "
                "a heartbeat.json (the irreducible emotional signal) "
                "to the output folder."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute or relative path to audio file (wav, mp3, flac, ogg)"
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional human-readable name for this track"
                    },
                    "mood": {
                        "type": "string",
                        "description": "Optional: operator's mood at listening time (free text)"
                    },
                    "note": {
                        "type": "string",
                        "description": "Optional: any context or expectation about this track"
                    }
                },
                "required": ["file_path"]
            }
        ),

        types.Tool(
            name="rip_and_analyze",
            description=(
                "Search YouTube for a song by name, rip it as WAV, "
                "and run full sensory analysis. One call does everything. "
                "Use when the operator names a track they want analyzed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Song search query e.g. 'Burial Archangel' or 'Aphex Twin Windowlicker'"
                    },
                    "mood": {
                        "type": "string",
                        "description": "Optional: operator's mood at listening time"
                    },
                    "note": {
                        "type": "string",
                        "description": "Optional: any context about this track"
                    }
                },
                "required": ["query"]
            }
        ),

        types.Tool(
            name="compare_songs",
            description=(
                "Compare two audio files and produce a structured diff: "
                "what changed harmonically, timbrally, energetically, "
                "in entropy, and emotionally between them. Useful for "
                "understanding the operator's taste or the journey "
                "between two listening states."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_a": {"type": "string", "description": "Path to first audio file"},
                    "file_b": {"type": "string", "description": "Path to second audio file"},
                    "label_a": {"type": "string", "description": "Name for song A"},
                    "label_b": {"type": "string", "description": "Name for song B"},
                },
                "required": ["file_a", "file_b"]
            }
        ),

        types.Tool(
            name="log_companion_response",
            description=(
                "Store the companion's interpretation of a sensory report "
                "into persistent memory. Call this after processing a report "
                "to record your understanding, emotional read, and any "
                "observations about the operator's state. This builds "
                "longitudinal musical memory over time."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "report_id":      {"type": "string", "description": "Report ID e.g. RPT-001"},
                    "song_name":      {"type": "string", "description": "Human-readable song name"},
                    "interpretation": {"type": "string", "description": "Full interpretation of the sensory data"},
                    "emotional_read": {"type": "string", "description": "Read on operator's emotional state"},
                    "key_observations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific observations worth remembering long-term"
                    },
                    "operator_state": {"type": "string", "description": "What the operator said about how the song felt"}
                },
                "required": ["report_id", "song_name", "interpretation"]
            }
        ),

        types.Tool(
            name="recall_memory",
            description=(
                "Retrieve past sensory reports and companion interpretations "
                "from memory. Use to find patterns, recall how a specific song "
                "was experienced, or review the operator's listening history."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Optional search term"},
                    "limit": {"type": "integer", "description": "Max entries (default 5)"}
                },
                "required": []
            }
        ),

        types.Tool(
            name="get_emotional_coords",
            description=(
                "Quick emotional coordinate extraction only — valence, arousal, "
                "quadrant, key, tempo. Faster than full analysis. Use when you "
                "just need the emotional fingerprint without full feature extraction."
            ),
            inputSchema={
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"]
            }
        ),

        types.Tool(
            name="get_heartbeat",
            description=(
                "Return only the Heartbeat Protocol — the irreducible emotional "
                "signal of the song: valence, arousal, tension trace (32 steps), "
                "tension peaks, key, tempo, entropy, harmonic ratio. This is "
                "the minimum viable signal for processing music as feeling. "
                "Returns structured JSON, not prose."
            ),
            inputSchema={
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"]
            }
        ),

    ]


# ── TOOL HANDLERS ─────────────────────────────────────────────────────────────
@app.call_tool()
async def call_tool(name: str, arguments: dict):

    # ── analyze_audio ─────────────────────────────────────────────────────────
    if name == "analyze_audio":
        path  = arguments["file_path"]
        label = arguments.get("label")
        mood  = arguments.get("mood")
        note  = arguments.get("note")

        if not os.path.exists(path):
            return [types.TextContent(type="text",
                text=f"ERROR: File not found: {path}")]

        features, report = run_full_pipeline(
            path, OUTPUT_DIR,
            label=label, mood=mood, note=note,
        )

        mem = load_memory()
        rid = next_report_id(mem)
        entry = memory_entry(
            features,
            label or os.path.splitext(os.path.basename(path))[0],
            path, mood=mood, note=note,
        )
        entry["report_id"] = rid
        mem["entries"].append(entry)
        save_memory(mem)

        return [types.TextContent(type="text",
            text=f"[REPORT ID: {rid}]\n{report}\n\n"
                 f"Visualizations saved to: {OUTPUT_DIR}\n"
                 f"  10 PNGs (waveform, spectrogram, chroma, energy_arc, mfcc,\n"
                 f"  hpss, beats, cymatic, color_timeline, tension)\n"
                 f"  + report.md, report.json, heartbeat.json\n\n"
                 f"Use log_companion_response('{rid}', ...) to record your interpretation."
        )]


    # ── rip_and_analyze ───────────────────────────────────────────────────────
    if name == "rip_and_analyze":
        query = arguments["query"]
        mood  = arguments.get("mood")
        note  = arguments.get("note")

        try:
            result = subprocess.run([
                "yt-dlp",
                f"ytsearch1:{query}",
                "-x", "--audio-format", "wav",
                "-o", TEMP_AUDIO,
                "--print", "title",
                "--no-playlist",
            ], check=True, capture_output=True, text=True)
            actual_title = result.stdout.strip().split('\n')[0] or query
        except subprocess.CalledProcessError as e:
            return [types.TextContent(type="text",
                text=f"ERROR: yt-dlp failed: {e.stderr}")]

        if not os.path.exists(TEMP_AUDIO):
            return [types.TextContent(type="text",
                text="ERROR: Audio file not created. yt-dlp may have failed silently.")]

        features, report = run_full_pipeline(
            TEMP_AUDIO, OUTPUT_DIR,
            label=actual_title, mood=mood, note=note,
        )

        mem = load_memory()
        rid = next_report_id(mem)
        entry = memory_entry(
            features, actual_title, TEMP_AUDIO,
            mood=mood, note=note, query=query,
        )
        entry["report_id"] = rid
        mem["entries"].append(entry)
        save_memory(mem)

        return [types.TextContent(type="text",
            text=f"[REPORT ID: {rid}] Ripped: {actual_title}\n\n{report}\n\n"
                 f"Use log_companion_response('{rid}', ...) to record your interpretation."
        )]


    # ── compare_songs ─────────────────────────────────────────────────────────
    if name == "compare_songs":
        path_a  = arguments["file_a"]
        path_b  = arguments["file_b"]
        label_a = arguments.get("label_a", os.path.basename(path_a))
        label_b = arguments.get("label_b", os.path.basename(path_b))

        for p in [path_a, path_b]:
            if not os.path.exists(p):
                return [types.TextContent(type="text",
                    text=f"ERROR: File not found: {p}")]

        y_a, sr_a = load_audio(path_a); f_a = extract_features(y_a, sr_a)
        y_b, sr_b = load_audio(path_b); f_b = extract_features(y_b, sr_b)

        return [types.TextContent(type="text",
            text=diff_reports(f_a, f_b, label_a, label_b))]


    # ── log_companion_response ────────────────────────────────────────────────
    if name == "log_companion_response":
        rid             = arguments["report_id"]
        song_name       = arguments["song_name"]
        interpretation  = arguments["interpretation"]
        emotional_read  = arguments.get("emotional_read", "")
        observations    = arguments.get("key_observations", [])
        operator_state  = arguments.get("operator_state", "")

        mem = load_memory()
        matched = False
        for entry in mem["entries"]:
            if entry.get("report_id") == rid:
                entry["companion_response"] = {
                    "timestamp":        datetime.datetime.now().isoformat(),
                    "interpretation":   interpretation,
                    "emotional_read":   emotional_read,
                    "key_observations": observations,
                    "operator_state":   operator_state,
                }
                matched = True
                break

        if not matched:
            mem["entries"].append({
                "report_id":   rid,
                "timestamp":   datetime.datetime.now().isoformat(),
                "song_name":   song_name,
                "companion_response": {
                    "timestamp":        datetime.datetime.now().isoformat(),
                    "interpretation":   interpretation,
                    "emotional_read":   emotional_read,
                    "key_observations": observations,
                    "operator_state":   operator_state,
                }
            })

        save_memory(mem)

        return [types.TextContent(type="text",
            text=f"Memory written. Report {rid} — '{song_name}' logged.\n"
                 f"Observations stored: {len(observations)}\n"
                 f"Total memory entries: {len(mem['entries'])}"
        )]


    # ── recall_memory ─────────────────────────────────────────────────────────
    if name == "recall_memory":
        query = arguments.get("query", "").lower()
        limit = arguments.get("limit", 5)

        mem     = load_memory()
        entries = mem["entries"]

        if query:
            entries = [
                e for e in entries
                if query in e.get("song_name", "").lower()
                or query in e.get("key", "").lower()
                or query in e.get("quadrant", "").lower()
                or query in json.dumps(e.get("companion_response", {})).lower()
                or query in e.get("operator_mood", "").lower()
            ]

        entries = entries[-limit:]

        if not entries:
            return [types.TextContent(type="text",
                text="No memory entries found matching that query.")]

        lines = []
        for e in entries:
            lines.append(f"\n{'─'*60}")
            lines.append(f"  {e.get('report_id','???')}  ::  {e.get('song_name','unknown')}")
            lines.append(f"  {e.get('timestamp','')[:19]}")
            lines.append(f"  Key: {e.get('key','')}  |  Tempo: {e.get('tempo',0):.1f} BPM")
            lines.append(f"  Valence: {e.get('valence',0):+.3f}  Arousal: {e.get('arousal',0):+.3f}")
            if 'mean_entropy' in e:
                lines.append(f"  Entropy: {e.get('mean_entropy',0):.3f} bits  "
                             f"H/P: {e.get('harmonic_ratio',0)*100:.0f}%")
            lines.append(f"  {e.get('quadrant','')}")
            if e.get("operator_mood"):
                lines.append(f"  Operator mood: {e['operator_mood']}")
            cr = e.get("companion_response")
            if cr:
                lines.append("\n  COMPANION INTERPRETATION:")
                lines.append(f"  {cr.get('interpretation','')[:400]}...")
                if cr.get('key_observations'):
                    lines.append("\n  KEY OBSERVATIONS:")
                    for obs in cr['key_observations']:
                        lines.append(f"    • {obs}")
            else:
                lines.append("  [no companion response logged yet]")

        return [types.TextContent(type="text", text='\n'.join(lines))]


    # ── get_emotional_coords ──────────────────────────────────────────────────
    if name == "get_emotional_coords":
        path = arguments["file_path"]
        if not os.path.exists(path):
            return [types.TextContent(type="text",
                text=f"ERROR: File not found: {path}")]

        y, sr    = load_audio(path)
        features = extract_features(y, sr)

        result = f"""
EMOTIONAL COORDINATES :: {os.path.basename(path)}
{'─'*50}
Key      : {features['key']} {features['mode']}
Tempo    : {features['tempo']:.1f} BPM
Valence  : {features['valence']:+.4f}
Arousal  : {features['arousal']:+.4f}
Quadrant : {features['quadrant']}
Centroid : {features['mean_centroid']:.0f} Hz
Energy   : {features['mean_energy']:.5f} RMS
Entropy  : {features['mean_entropy']:.3f} bits
H / P    : {features['harmonic_ratio']*100:.0f}% / {features['percussive_ratio']*100:.0f}%
"""
        return [types.TextContent(type="text", text=result)]


    # ── get_heartbeat ─────────────────────────────────────────────────────────
    if name == "get_heartbeat":
        path = arguments["file_path"]
        if not os.path.exists(path):
            return [types.TextContent(type="text",
                text=f"ERROR: File not found: {path}")]

        y, sr    = load_audio(path)
        features = extract_features(y, sr)
        hb       = heartbeat_signal(features)

        return [types.TextContent(type="text",
            text=json.dumps(hb, indent=2))]


    # ── unknown tool ──────────────────────────────────────────────────────────
    return [types.TextContent(type="text", text=f"ERROR: Unknown tool '{name}'")]


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
