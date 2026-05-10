#!/usr/bin/env bash
# install.sh — Sound for Wren installer (macOS / Linux)
#
# Creates a local Python virtual environment in .venv, installs all
# dependencies from requirements.txt, then runs synthesize_test_song.py
# to verify the pipeline end-to-end with a clean synthetic input.

set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
SKIP_SYNTH="${SKIP_SYNTH:-0}"

echo
echo "Sound for Wren installer"
echo "------------------------"

# 1. Locate Python
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python interpreter '$PYTHON_BIN' not found on PATH." >&2
    echo "Install Python 3.10+ and try again. Override with PYTHON=/path/to/python ./install.sh" >&2
    exit 1
fi
echo "Using Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"

# 2. Confirm version >= 3.10
PY_VER=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=${PY_VER%%.*}
PY_MINOR=${PY_VER##*.}
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "Python $PY_VER detected; 3.10 or newer is required." >&2
    exit 1
fi
echo "Python version: $PY_VER (ok)"

# 3. Create venv
if [ -d ".venv" ]; then
    echo ".venv already exists; reusing it."
else
    echo "Creating virtual environment in .venv ..."
    "$PYTHON_BIN" -m venv .venv
fi

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "venv creation failed; expected $VENV_PY to exist." >&2
    exit 1
fi

# 4. Install dependencies
echo "Upgrading pip ..."
"$VENV_PY" -m pip install --quiet --upgrade pip

echo "Installing requirements.txt ..."
"$VENV_PY" -m pip install -r requirements.txt

# 5. Synthesize test song to verify the pipeline
if [ "$SKIP_SYNTH" != "1" ]; then
    echo "Generating clean synthetic test song ..."
    "$VENV_PY" synthesize_test_song.py
fi

echo
echo "Install complete."
echo
echo "To activate the venv in your shell:"
echo "  source .venv/bin/activate"
echo
echo "To analyze audio:"
echo "  python sensory_report.py song.wav ./output"
echo
echo "To start the MCP server:"
echo "  python SoundforWren_MCP.py"
