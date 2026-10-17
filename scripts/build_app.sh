#!/usr/bin/env bash
# Build Verbatim.app: the SwiftUI app plus an embedded, relocatable Python
# runtime with the transcription engine installed. The result launches from
# Finder without any Terminal setup. Models are NOT bundled; the app downloads
# them only when the user confirms.
#
# Requirements (build machine): macOS on Apple Silicon, Xcode 16+ (Swift 6), uv.
#
# Usage: scripts/build_app.sh            # -> build/Verbatim.app
# Env:   OUT=dir  CONFIG=release|debug  BUNDLE_ID=com.example.verbatim
#        PYTHON_VERSION=3.12  SIGN_IDENTITY=- (ad-hoc; Developer ID for distribution)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/build}"
CONFIG="${CONFIG:-release}"
BUNDLE_ID="${BUNDLE_ID:-com.sematrica.verbatim}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
SIGN_IDENTITY="${SIGN_IDENTITY:--}"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/engine/pyproject.toml" | head -1)"
BUILD_NUMBER="${BUILD_NUMBER:-$(git -C "$ROOT" rev-list --count HEAD 2>/dev/null || echo 1)}"
APP="$OUT/Verbatim.app"
RES="$APP/Contents/Resources"
PY="$RES/python"

[[ "$(uname -s)" == "Darwin" ]] || { echo "error: build_app.sh must run on macOS" >&2; exit 1; }
[[ "$(uname -m)" == "arm64" ]] || echo "warning: building on $(uname -m); the bundle targets Apple Silicon" >&2
command -v uv >/dev/null || { echo "error: uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }

echo "==> Building Swift app ($CONFIG)"
swift build --package-path "$ROOT/app" -c "$CONFIG" --arch arm64
BIN_DIR="$(swift build --package-path "$ROOT/app" -c "$CONFIG" --arch arm64 --show-bin-path)"

echo "==> Assembling bundle at $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$RES"
cp "$BIN_DIR/Verbatim" "$APP/Contents/MacOS/Verbatim"
sed -e "s/__BUNDLE_ID__/$BUNDLE_ID/" -e "s/__VERSION__/$VERSION/" -e "s/__BUILD__/$BUILD_NUMBER/" \
    "$ROOT/app/Bundle/Info.plist" > "$APP/Contents/Info.plist"
plutil -lint "$APP/Contents/Info.plist" >/dev/null

echo "==> Embedding CPython $PYTHON_VERSION (python-build-standalone, relocatable)"
# Only uv-managed builds are relocatable (a Homebrew or python.org Python is not).
uv python install --python-preference only-managed "$PYTHON_VERSION"
PY_EXE="$(uv python find --python-preference only-managed "$PYTHON_VERSION")"
echo "    using $PY_EXE"
PY_HOME="$(cd "$(dirname "$PY_EXE")/.." && pwd -P)"
ditto "$PY_HOME" "$PY"
# The copy is ours to manage; uv marks its managed interpreters as externally managed.
find "$PY/lib" -maxdepth 2 -name EXTERNALLY-MANAGED -delete
PY_BIN="$PY/bin/python3"
[[ -x "$PY_BIN" ]] || { echo "error: embedded python not found at $PY_BIN" >&2; exit 1; }

echo "==> Installing the engine and its locked dependencies"
REQS="$(mktemp -t verbatim-reqs)"
uv export --project "$ROOT/engine" --frozen --no-dev --no-hashes --no-emit-project \
    --format requirements-txt > "$REQS"
uv pip install --python "$PY_BIN" --no-deps -r "$REQS"
uv pip install --python "$PY_BIN" --no-deps "$ROOT/engine"
rm -f "$REQS"
if "$PY_BIN" -c "import torch" 2>/dev/null; then
    echo "error: torch was installed into the bundle (it is not needed at runtime)" >&2; exit 1
fi

echo "==> Trimming and precompiling"
rm -rf "$PY/lib/python$PYTHON_VERSION/test" "$PY/lib/python$PYTHON_VERSION/idlelib" \
       "$PY/lib/python$PYTHON_VERSION/tkinter" "$PY/lib/python$PYTHON_VERSION/turtledemo" "$PY/share"
# Bytecode is generated now because the signed bundle must stay read-only at runtime.
"$PY_BIN" -m compileall -q -j 0 "$PY/lib" >/dev/null

echo "==> Recording build provenance"
{
    echo "Verbatim $VERSION ($BUILD_NUMBER) built $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "python: $("$PY_BIN" -c 'import sys; print(sys.version)')"
    echo
    echo "== Python packages =="
    uv pip list --python "$PY_BIN" 2>/dev/null
    echo
    echo "== FFmpeg build bundled by PyAV (review for LGPL/GPL obligations) =="
    "$PY_BIN" -c 'import av; print("FFmpeg", av.ffmpeg_version_info); print(av.library_versions)'
    LIBAVCODEC="$(find "$PY/lib" -path '*av*' -name 'libavcodec*.dylib' | head -1)"
    if [[ -n "$LIBAVCODEC" ]]; then
        strings "$LIBAVCODEC" 2>/dev/null | grep -m1 -E '^--(prefix|enable|disable)' || true
    fi
} > "$RES/BUILD_INFO.txt"

echo "==> Verifying the embedded engine"
"$PY_BIN" -I -m verbatim_engine --version
"$PY_BIN" -I -c "import verbatim_engine, faster_whisper, ctranslate2, av, onnxruntime; print('engine imports ok')"
# YouTube import: yt-dlp, its pinned JavaScript components and the deno runtime run from the bundle
"$PY_BIN" -I -c "import subprocess, deno, yt_dlp, yt_dlp_ejs; print('yt-dlp', yt_dlp.version.__version__, '|',
subprocess.run([deno.find_deno_bin(), '--version'], capture_output=True, text=True, check=True).stdout.splitlines()[0])"

echo "==> Signing ($([[ "$SIGN_IDENTITY" == "-" ]] && echo ad-hoc || echo "$SIGN_IDENTITY"))"
# Sign nested code first (inside-out), then the app.
find "$PY" -type f \( -name '*.so' -o -name '*.dylib' \) -print0 \
    | xargs -0 -n 64 codesign --force --sign "$SIGN_IDENTITY" --timestamp=none
find "$PY/bin" -type f -perm -u+x -print0 | while IFS= read -r -d '' f; do
    if file "$f" | grep -q 'Mach-O'; then codesign --force --sign "$SIGN_IDENTITY" --timestamp=none "$f"; fi
done
codesign --force --sign "$SIGN_IDENTITY" --timestamp=none "$APP"
codesign --verify --strict "$APP"

du -sh "$APP" | awk '{print "==> Done: '"$APP"' (" $1 ")"}'
