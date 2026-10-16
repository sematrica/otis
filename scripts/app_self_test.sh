#!/usr/bin/env bash
# Run the packaged app's built-in self-test: import MEDIA, transcribe it with
# the real model in MODEL_DIR through the app's own model layer, open the
# transcript workspace, snapshot the window, and quit.
#   scripts/app_self_test.sh build/Verbatim.app MEDIA MODEL_DIR BACKEND EXPECTED_PHRASE [SPEAKERS]
# With SPEAKERS=N the speaker identification and alignment models must be
# installed in the app's models folder; N speakers are expected, a speaker is
# renamed and the rename undone.
set -euo pipefail
APP="$1"; MEDIA="$2"; MODEL_DIR="$3"; BACKEND="$4"; EXPECTED="$5"; SPEAKERS="${6:-0}"
OUT="$(mktemp -d)"
echo "==> Self-test ($BACKEND) on $(basename "$MEDIA")"
VERBATIM_SELF_TEST_MEDIA="$MEDIA" VERBATIM_SELF_TEST_MODEL_DIR="$MODEL_DIR" \
VERBATIM_SELF_TEST_BACKEND="$BACKEND" VERBATIM_SELF_TEST_OUTPUT="$OUT" \
VERBATIM_SELF_TEST_SPEAKERS="$([[ "$SPEAKERS" != 0 ]] && echo 1 || echo 0)" \
    "$APP/Contents/MacOS/Verbatim" > "$OUT/app.log" 2>&1 &
app_pid=$!
for _ in $(seq 1 360); do
    kill -0 "$app_pid" 2>/dev/null || break
    sleep 0.5
done
if kill -0 "$app_pid" 2>/dev/null; then
    echo "error: app did not finish the self-test" >&2
    echo "---- progress ----"; cat "$OUT/progress.log" 2>/dev/null || true
    echo "---- app log ----"; tail -40 "$OUT/app.log" || true
    echo "---- main thread sample ----"
    sample "$app_pid" 2 -file "$OUT/sample.txt" >/dev/null 2>&1 || true
    awk '/^Call graph:/{p=1} p{print} /^Total number in stack/{exit}' "$OUT/sample.txt" | head -120 || true
    kill "$app_pid"
    exit 1
fi
status=0
wait "$app_pid" || status=$?
echo "app exit status: $status"
if [[ ! -f "$OUT/result.json" || $status -ne 0 ]]; then
    echo "---- progress ----"; cat "$OUT/progress.log" 2>/dev/null || true
    echo "---- app log ----"; tail -80 "$OUT/app.log" || true
    echo "---- engine log ----"; tail -40 "$HOME/Library/Application Support/Verbatim/logs/engine.log" 2>/dev/null || true
    report="$(ls -t "$HOME"/Library/Logs/DiagnosticReports/Verbatim*.ips 2>/dev/null | head -1 || true)"
    if [[ -n "$report" ]]; then
        echo "---- crash report: $report ----"
        python3 - "$report" <<'PY'
import json, sys
text = open(sys.argv[1]).read()
header, _, body = text.partition("\n")
crash = json.loads(body)
print("exception:", json.dumps(crash.get("exception")))
print("termination:", json.dumps(crash.get("termination")))
for key in ("asi", "lastExceptionBacktrace", "ktriageinfo"):
    if key in crash:
        print(key + ":", json.dumps(crash[key])[:3000])
images = crash.get("usedImages", [])
thread = crash["threads"][crash.get("faultingThread", 0)]
print("faulting thread", crash.get("faultingThread"), thread.get("queue"))
for frame in thread.get("frames", [])[:40]:
    image = images[frame["imageIndex"]].get("name", "?") if frame.get("imageIndex", -1) < len(images) else "?"
    print(f"  {image:28} {frame.get('symbol', hex(frame.get('imageOffset', 0)))}")
PY
    fi
    exit 1
fi
cat "$OUT/result.json"
python3 - "$OUT/result.json" "$EXPECTED" "$SPEAKERS" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
speakers = int(sys.argv[3])
assert result["status"] == "completed", result
assert sys.argv[2].lower() in result["text"].lower(), result["text"]
optional = set() if speakers else {"diarize", "align"}  # skipped when their models are not installed
for entry in result["stages"]:
    stage, status = entry.split(":")
    assert status == "completed" or (stage in optional and status == "skipped"), result["stages"]
if speakers:
    assert len(result["speakers"]) == speakers, result["speakers"]
    assert result["aligned_words"] >= 0.7 * result["words"], result
    assert result["renamed"] == "Diane" and result["after_undo"] == "Speaker 1" and result["can_redo"], result
print("self-test passed:", result["segments"], "segment(s), language", result["language"],
      "speakers", result.get("speakers"), "aligned words", result.get("aligned_words"), "/", result.get("words"))
PY
if pgrep -f "$APP/Contents/Resources/python/bin/python3 -I -m verbatim_engine" >/dev/null; then
    echo "error: engine still running after the app quit" >&2; exit 1
fi
if [[ -f "$OUT/window.png" ]]; then
    echo "----BEGIN WINDOW PNG ($BACKEND)----"
    base64 -b 120 -i "$OUT/window.png"
    echo "----END WINDOW PNG----"
fi
