#!/usr/bin/env bash
# Launch the built app like Finder does and check that it starts its embedded
# engine, then quits cleanly without leaving the engine running.
#   scripts/smoke_test_app.sh build/Verbatim.app
set -euo pipefail
APP="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
PY="$APP/Contents/Resources/python/bin/python3"
BUNDLE_ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$APP/Contents/Info.plist")"
LOG_DIR="$HOME/Library/Application Support/Verbatim/logs"

echo "==> Embedded engine runs in isolated mode"
TMP_DATA="$(mktemp -d)"
"$PY" -I -m verbatim_engine info --data-dir "$TMP_DATA" | "$PY" -I -c '
import json, sys
info = json.load(sys.stdin)
print("engine", info["version"], "protocol", info["protocol_version"], "python", info["python"]["version"])
for b in info["backends"]:
    detail = b["unavailable_reason"] or b["capabilities"]["acceleration"]
    print("  %-15s available=%s %s" % (b["name"], b["available"], detail))
assert info["protocol_version"] == 1
assert any(b["name"] == "faster-whisper" and b["available"] for b in info["backends"])
'

echo "==> Launching $APP"
open -n "$APP"
engine_pid=""
for _ in $(seq 1 120); do
    engine_pid="$(pgrep -f "$APP/Contents/Resources/python/bin/python3 -I -m verbatim_engine serve" | head -1 || true)"
    [[ -n "$engine_pid" ]] && break
    sleep 0.5
done
app_pid="$(pgrep -f "$APP/Contents/MacOS/Verbatim" | head -1 || true)"
if [[ -z "$engine_pid" ]]; then
    echo "error: the app did not start its engine" >&2
    tail -50 "$LOG_DIR/engine-stderr.log" 2>/dev/null || true
    exit 1
fi
parent="$(ps -o ppid= -p "$engine_pid" | tr -d ' ')"
echo "app pid $app_pid, engine pid $engine_pid (parent $parent)"
[[ "$parent" == "$app_pid" ]] || { echo "error: engine is not a child of the app" >&2; exit 1; }

# Give the app a moment to finish its first requests, then check the log.
sleep 3
grep -q "engine serving" "$LOG_DIR/engine.log" || { echo "error: engine log has no start-up line" >&2; exit 1; }

echo "==> Quitting the app"
osascript -e "tell application id \"$BUNDLE_ID\" to quit" 2>/dev/null || kill -TERM "$app_pid"
for _ in $(seq 1 60); do
    if ! kill -0 "$engine_pid" 2>/dev/null && ! kill -0 "$app_pid" 2>/dev/null; then
        echo "app and engine exited cleanly"
        exit 0
    fi
    sleep 0.5
done
echo "error: app or engine still running after quit" >&2
ps -p "$app_pid,$engine_pid" || true
exit 1
