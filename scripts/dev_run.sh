#!/usr/bin/env bash
# Development loop: set up the engine's virtualenv and run the app against it
# (no bundle needed). Requires macOS, Xcode 16+, uv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv sync --project "$ROOT/engine" --extra dev --frozen
export VERBATIM_ENGINE_PYTHON="$ROOT/engine/.venv/bin/python"
exec swift run --package-path "$ROOT/app" Verbatim
