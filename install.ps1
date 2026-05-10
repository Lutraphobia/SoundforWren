# install.ps1 — Sound for Wren installer (Windows)
#
# Creates a local Python virtual environment in .venv, installs all
# dependencies from requirements.txt, then runs synthesize_test_song.py
# to verify the pipeline end-to-end with a clean synthetic input.

[CmdletBinding()]
param(
    [string]$Python = 'python',
    [switch]$SkipSynth
)

$ErrorActionPreference = 'Stop'

Write-Host ""
Write-Host "Sound for Wren installer" -ForegroundColor Cyan
Write-Host "------------------------" -ForegroundColor Cyan

# 1. Locate Python
$pythonCmd = Get-Command $Python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "Python interpreter '$Python' not found on PATH. Install Python 3.10+ from https://www.python.org/downloads/ and try again."
    exit 1
}
$pythonPath = $pythonCmd.Source
Write-Host "Using Python: $pythonPath"

# 2. Confirm version >= 3.10
$verRaw = & $pythonPath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
if (-not $verRaw) { Write-Error "Could not query Python version."; exit 1 }
$ver = [Version]$verRaw
$min = [Version]'3.10'
if ($ver -lt $min) {
    Write-Error "Python $verRaw detected; 3.10 or newer is required."
    exit 1
}
Write-Host "Python version: $verRaw (ok)"

# 3. Create venv
if (Test-Path .\.venv) {
    Write-Host ".venv already exists; reusing it."
} else {
    Write-Host "Creating virtual environment in .venv ..."
    & $pythonPath -m venv .venv
}

$venvPython = Join-Path -Path (Resolve-Path .\.venv).Path -ChildPath 'Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Error "venv creation failed; expected $venvPython to exist."
    exit 1
}

# 4. Install dependencies
Write-Host "Upgrading pip ..."
& $venvPython -m pip install --quiet --upgrade pip

Write-Host "Installing requirements.txt ..."
& $venvPython -m pip install -r requirements.txt

# 5. Synthesize test song to verify the pipeline
if (-not $SkipSynth) {
    Write-Host "Generating clean synthetic test song ..."
    & $venvPython synthesize_test_song.py
}

Write-Host ""
Write-Host "Install complete." -ForegroundColor Green
Write-Host ""
Write-Host "To activate the venv in your shell:"
Write-Host "  .\.venv\Scripts\activate"
Write-Host ""
Write-Host "To analyze audio:"
Write-Host "  python sensory_report.py song.wav ./output"
Write-Host ""
Write-Host "To start the MCP server:"
Write-Host "  python SoundforWren_MCP.py"
