#!/usr/bin/env python3
"""Measure word-timestamp accuracy on synthetic speech with known word timings.

Speech comes from libespeak-ng, which reports where each word starts (see
``tests/support/espeak.py``). The audio goes through the engine's own
decoder, a real Whisper model and, optionally, a forced-alignment model; the
recognised words are matched to the reference words by text and their start
and end times compared. Only words recognised correctly are scored.

Synthetic speech is cleaner and more regular than real recordings: these
numbers describe timing precision, not real-world accuracy.

    python scripts/benchmark_word_timing.py --whisper /path/to/faster-whisper-small \
        --aligner nemo:/path/to/nemo-ctc-model [--aligner mms:/path/to/mms] [--json out.json]
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))

from support import espeak  # noqa: E402

from verbatim_engine.alignment.aligner import WordAligner  # noqa: E402
from verbatim_engine.alignment.models import MMSModel, NemoCTCModel  # noqa: E402
from verbatim_engine.asr.merge import normalize_token  # noqa: E402
from verbatim_engine.audio.extract import extract_audio  # noqa: E402
from verbatim_engine.audio.wav import WavReader  # noqa: E402

LEAD_SILENCE = 0.5


def build_audio(text: str, voice: str, out_dir: Path) -> tuple[Path, list[espeak.TimedWord]]:
    samples, rate, words = espeak.synthesize_timed(text, voice=voice)
    pad = np.zeros(int(LEAD_SILENCE * rate), np.float32)
    samples = np.concatenate([pad, samples, pad])
    raw = out_dir / f"{voice.replace('/', '_')}.raw.wav"
    with wave.open(str(raw), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    wav = out_dir / f"{voice.replace('/', '_')}.wav"
    extract_audio(raw, wav)
    for word in words:
        word.start += LEAD_SILENCE
        word.end += LEAD_SILENCE
    return wav, words


class _W:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _S:
    def __init__(self, start, end, words):
        self.start, self.end, self.words = start, end, words


def transcribe(model_dir: str, wav: Path) -> list[_S]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_dir, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(wav), language="en", word_timestamps=True, beam_size=5)
    return [_S(s.start, s.end, [_W(w.start, w.end, w.word) for w in s.words]) for s in segments]


def match(reference: list[espeak.TimedWord], hypothesis: list[_W]) -> list[tuple[espeak.TimedWord, _W]]:
    ref_tokens = [normalize_token(w.text) for w in reference]
    hyp_tokens = [normalize_token(w.text) for w in hypothesis]
    pairs = []
    for block in difflib.SequenceMatcher(a=ref_tokens, b=hyp_tokens, autojunk=False).get_matching_blocks():
        for k in range(block.size):
            pairs.append((reference[block.a + k], hypothesis[block.b + k]))
    return pairs


def metrics(pairs) -> dict:
    if not pairs:
        return {"words": 0}
    start = np.array([abs(h.start - r.start) for r, h in pairs])
    end = np.array([abs(h.end - r.end) for r, h in pairs])
    bias_start = float(np.mean([h.start - r.start for r, h in pairs]))
    bias_end = float(np.mean([h.end - r.end for r, h in pairs]))
    out = {"words": len(pairs)}
    for name, err in (("start", start), ("end", end)):
        out[f"{name}_mean_abs_ms"] = round(float(err.mean()) * 1000, 1)
        out[f"{name}_median_abs_ms"] = round(float(np.median(err)) * 1000, 1)
        for limit in (0.05, 0.1, 0.2):
            out[f"{name}_within_{int(limit * 1000)}ms"] = round(float((err <= limit).mean()), 3)
    out["start_bias_ms"] = round(bias_start * 1000, 1)
    out["end_bias_ms"] = round(bias_end * 1000, 1)
    return out


def load_aligner(spec: str):
    kind, _, path = spec.partition(":")
    if kind == "nemo":
        return NemoCTCModel(Path(path), model_id="nemo", display_name="NeMo CTC", threads=4)
    if kind == "mms":
        return MMSModel(Path(path), model_id="mms", display_name="MMS", threads=4)
    raise SystemExit(f"unknown aligner kind {kind!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--whisper", required=True, help="faster-whisper model folder")
    parser.add_argument("--aligner", action="append", default=[], help="kind:path (nemo or mms)")
    parser.add_argument("--voice", action="append", default=[], help="espeak-ng voice(s)")
    parser.add_argument("--json", help="write results here")
    args = parser.parse_args()
    if not espeak.available():
        raise SystemExit("libespeak-ng is required")
    texts = json.loads((HERE.parent / "tests" / "corpus" / "tts_texts.json").read_text())
    text = " ".join(texts["english_paragraphs"])
    voices = args.voice or ["en-us"]
    aligners = [(spec, load_aligner(spec)) for spec in args.aligner]
    report: dict = {"text_words": len(text.split()), "voices": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for voice in voices:
            wav, reference = build_audio(text, voice, Path(tmp))
            segments = transcribe(args.whisper, wav)
            hyp_words = [w for s in segments for w in s.words]
            result = {"whisper": metrics(match(reference, hyp_words))}
            for spec, model in aligners:
                with WavReader(wav) as reader:
                    aligned = WordAligner(model).align(reader, segments, "en")
                words = []
                statuses = {}
                for seg, res in zip(segments, aligned):
                    statuses[res.status] = statuses.get(res.status, 0) + 1
                    if res.status == "aligned":
                        words.extend(_W(t.start, t.end, w.text) for t, w in zip(res.words, seg.words))
                    else:
                        words.extend(seg.words)
                kind, _, path = spec.partition(":")
                result[f"{kind}:{Path(path).name}"] = {**metrics(match(reference, words)), "segments": statuses}
            report["voices"][voice] = result
            print(json.dumps({voice: result}, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
