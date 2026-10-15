#!/usr/bin/env bash
# Run the packaged app's built-in self-test: import MEDIA, transcribe it with
# the real model in MODEL_DIR through the app's own model layer, open the
# transcript workspace, snapshot the window, and quit.
#   scripts/app_self_test.sh build/Verbatim.app MEDIA MODEL_DIR BACKEND EXPECTED_PHRASE
set -euo pipefail
APP="$1"; MEDIA="$2"; MODEL_DIR="$3"; BACKEND="$4"; EXPECTED="$5"
OUT="$(mktemp -d)"
echo "==> Self-test ($BACKEND) on $(basename "$MEDIA")"
VERBATIM_SELF_TEST_MEDIA="$MEDIA" VERBATIM_SELF_TEST_MODEL_DIR="$MODEL_DIR" \
VERBATIM_SELF_TEST_BACKEND="$BACKEND" VERBATIM_SELF_TEST_OUTPUT="$OUT" \
    "$APP/Contents/MacOS/Verbatim" > "$OUT/app.log" 2>&1 &
app_pid=$!
for _ in $(seq 1 1800); do
    kill -0 "$app_pid" 2>/dev/null || break
    sleep 0.5
done
if kill -0 "$app_pid" 2>/dev/null; then
    echo "error: app did not finish the self-test" >&2; kill "$app_pid"; tail -40 "$OUT/app.log"; exit 1
fi
wait "$app_pid" || true
cat "$OUT/result.json"
python3 - "$OUT/result.json" "$EXPECTED" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
assert result["status"] == "completed", result
assert sys.argv[2].lower() in result["text"].lower(), result["text"]
assert all(s.endswith(":completed") for s in result["stages"]), result["stages"]
print("self-test passed:", result["segments"], "segment(s), language", result["language"])
PY
if pgrep -f "$APP/Contents/Resources/python/bin/python3 -I -m verbatim_engine" >/dev/null; then
    echo "error: engine still running after the app quit" >&2; exit 1
fi
if [[ -f "$OUT/window.png" ]]; then
    echo "----BEGIN WINDOW PNG ($BACKEND)----"
    base64 -b 120 -i "$OUT/window.png"
    echo "----END WINDOW PNG----"
fi
