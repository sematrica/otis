"""Accuracy benchmark corpus (Stage 3).

Every item has audio and whatever references exist for it: a verified
transcript (WER, CER), a speaker annotation (DER), exact word times
(timestamp accuracy). Real recordings come from pinned URLs checked against
SHA-256; derived items (real speech with noise or silence added) and
synthetic items (espeak-ng) are generated deterministically with fixed seeds.

Categories follow the specification's test list: single speaker, two-person,
multi-speaker, overlapping speech, background noise, long recordings,
multilingual, silence (interrupted processing is a procedure, run by the
benchmark script). Synthetic speech is cleaner than real recordings; results
on it describe behaviour, not real-world accuracy.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import conversation, espeak

ENGINE = Path(__file__).resolve().parents[2]
CORPUS = ENGINE / "tests" / "corpus"
SR = 16000

#: LibriSpeech utterances (CC BY 4.0) with their transcripts, as bundled in the
#: sherpa-onnx export of the English alignment model (same pinned file).
LIBRISPEECH_TARBALL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-conformer-small.tar.bz2",
    "83dcb462aece5bef4e8072c267419389f0b8d1f91152d8851765f284ff664caa",
)

#: An original French paragraph written for this corpus (read by espeak-ng).
FRENCH = (
    "Le matin, le marché ouvre avant le lever du soleil. Les marchands installent leurs tables et "
    "comptent les caisses de légumes. Une cliente demande le prix des pommes, puis elle achète du pain "
    "et rentre chez elle à pied."
)


@dataclass
class Item:
    id: str
    categories: list[str]
    wav: Path
    language: str | None
    real: bool
    #: verified transcript (WER, CER)
    reference: str | None = None
    #: speaker annotation [(start, end, label)] (DER)
    turns: list[tuple[float, float, str]] | None = None
    #: exact word times [(start, end, text)] (timestamp accuracy)
    words: list[tuple[float, float, str]] | None = None
    #: the audio contains no speech: every word is an insertion
    no_speech: bool = False
    notes: str = ""
    speakers: int | None = None
    extra: dict = field(default_factory=dict)


def _write(path: Path, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _load(path: Path, cache: Path) -> np.ndarray:
    from verbatim_engine.audio.extract import extract_audio
    from verbatim_engine.audio.wav import WavReader

    out = cache / "decoded" / f"{path.stem}.16k.wav"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        extract_audio(path, out)
    with WavReader(out) as reader:
        return reader.read(0.0, reader.duration)


def fetch_librispeech(cache: Path) -> dict[str, tuple[Path, str]]:
    """The three LibriSpeech utterances: {name: (wav, transcript)}."""
    folder = cache / "librispeech"
    marker = folder / "trans.txt"
    if not marker.is_file():
        import httpx2

        url, digest = LIBRISPEECH_TARBALL
        response = httpx2.get(url, timeout=httpx2.Timeout(30.0, read=300.0), follow_redirects=True)
        response.raise_for_status()
        data = response.content
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("checksum mismatch for the LibriSpeech test files")
        folder.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as tar:
            for member in tar.getmembers():
                if member.isfile() and "/test_wavs/" in member.name:
                    name = Path(member.name).name
                    if name.endswith((".wav", ".txt")):
                        (folder / name).write_bytes(tar.extractfile(member).read())
    out = {}
    for line in marker.read_text().splitlines():
        if line.strip():
            name, text = line.split(None, 1)
            out[Path(name).stem] = (folder / name, text.strip())
    return out


def _noise_like(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    gain = np.sqrt(np.mean(signal ** 2) / (np.mean(noise ** 2) + 1e-12) / 10 ** (snr_db / 10))
    return signal + noise * gain


def _babble(sources: list[np.ndarray], length: int) -> np.ndarray:
    out = np.zeros(length)
    for s in sources:
        tiled = np.tile(s, int(np.ceil(length / s.size)))[:length]
        out += tiled / np.sqrt(np.mean(tiled ** 2) + 1e-12)
    return out


def build(cache: Path, *, speaker_corpus: Path | None = None, long: bool = False, synthetic: bool = True) -> list[Item]:
    cache.mkdir(parents=True, exist_ok=True)
    items: list[Item] = []
    rng = np.random.default_rng(1234)

    # real single-speaker speech ----------------------------------------------
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    jfk_ref = next(s for s in manifest["samples"] if s["id"] == "jfk")["reference"]
    jfk_path = CORPUS / "jfk.wav"
    items.append(Item("jfk", ["single_speaker"], jfk_path, "en", True, reference=jfk_ref, speakers=1,
                      notes="1961 inaugural address (public domain)"))
    libri = fetch_librispeech(cache)
    for name, (path, text) in sorted(libri.items()):
        items.append(Item(f"librispeech-{name}", ["single_speaker"], path, "en", True, reference=text, speakers=1,
                          notes="LibriSpeech (CC BY 4.0)" + (", 8 kHz telephone band" if name == "8k" else "")))

    # real speech made harder ---------------------------------------------------
    jfk = _load(jfk_path, cache)
    others = [_load(p, cache) for p, _ in libri.values()]
    for snr in (5, 0):
        items.append(Item(f"jfk-white-{snr}db", ["noise"], _write(cache / f"jfk_white{snr}.wav",
                          _noise_like(jfk, rng.normal(size=jfk.size), snr)), "en", True, reference=jfk_ref,
                          speakers=1, notes=f"JFK with white noise at {snr} dB SNR"))
        items.append(Item(f"jfk-babble-{snr}db", ["noise", "overlap"], _write(cache / f"jfk_babble{snr}.wav",
                          _noise_like(jfk, _babble(others, jfk.size), snr)), "en", True, reference=jfk_ref,
                          speakers=1, notes=f"JFK over three other real speakers (babble) at {snr} dB SNR"))
    faint = lambda seconds: rng.normal(size=int(seconds * SR)) * 0.002  # noqa: E731
    items.append(Item("jfk-long-silences", ["silence"], _write(cache / "jfk_silences.wav",
                      np.concatenate([faint(12), jfk, faint(25)])), "en", True, reference=jfk_ref, speakers=1,
                      notes="JFK between 12 s and 25 s of near-silence (hallucination trigger)"))
    items.append(Item("silence-30s", ["silence"], _write(cache / "silence30.wav", faint(30)), None, False,
                      reference="", no_speech=True, notes="30 s of faint noise: no speech at all"))

    # real multi-speaker recordings (speaker annotation only) -------------------
    if speaker_corpus is not None:
        from verbatim_engine.metrics.der import parse_rttm

        refs: dict[str, list] = {}
        for rttm in sorted(speaker_corpus.glob("*.rttm")):
            refs.update(parse_rttm(rttm.read_text()))
        for uri, turns in sorted(refs.items()):
            wav = speaker_corpus / f"{uri}.wav"
            if not wav.is_file():
                continue
            labels = {label for *_t, label in turns}
            if uri == "sample":
                items.append(Item("call", ["two_person"], wav, "en", True, turns=turns, speakers=len(labels),
                                  notes="Two-person telephone call (pyannote sample); no verified transcript"))
            else:
                items.append(Item(f"ami-{uri}", ["multi_speaker", "overlap"], wav, "en", True, turns=turns,
                                  speakers=len(labels), notes="AMI meeting excerpt (CC BY 4.0), far-field"))

    if not synthetic or not espeak.available():
        return items

    # synthetic conversations with exact references ------------------------------
    texts = json.loads((CORPUS / "tts_texts.json").read_text())
    paragraphs = texts["english_paragraphs"]
    for voices, name, categories in ((conversation.VOICES[:2], "conversation-2", ["two_person", "overlap"]),
                                     (conversation.VOICES, "conversation-3", ["multi_speaker", "overlap"])):
        conv = conversation.build(paragraphs[:6], voices, cache / "synthetic", seed=1, overlap_every=4, name=name)
        reference = " ".join(sentence for _speaker, sentence in conv.texts)
        items.append(Item(name, categories, conv.wav, "en", False, reference=reference, turns=conv.turns,
                          words=[(a, b, t) for a, b, t, _s in conv.words], speakers=len(voices),
                          notes="espeak-ng voices; every 4th sentence overlaps the previous one"))
    for lang, text in (("de", texts["german"]), ("es", texts["spanish"]), ("fr", FRENCH)):
        path = cache / "synthetic" / f"tts_{lang}.wav"
        samples, rate, timed = espeak.synthesize_timed(text, voice=lang)
        raw = cache / "synthetic" / f"tts_{lang}.raw.wav"
        with wave.open(str(raw), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            pad = np.zeros(int(0.5 * rate), np.float32)
            w.writeframes((np.clip(np.concatenate([pad, samples, pad]), -1, 1) * 32767).astype("<i2").tobytes())
        from verbatim_engine.audio.extract import extract_audio

        extract_audio(raw, path)
        items.append(Item(f"tts-{lang}", ["multilingual"], path, lang, False, reference=text,
                          words=[(w.start + 0.5, w.end + 0.5, w.text) for w in timed], speakers=1,
                          notes=f"espeak-ng, {lang}"))
    if long:
        sentences = (paragraphs * 3)[:30]
        conv = conversation.build(sentences, conversation.VOICES, cache / "synthetic", seed=7, overlap_every=6,
                                  name="long")
        reference = " ".join(sentence for _speaker, sentence in conv.texts)
        items.append(Item("long-conversation", ["long", "multi_speaker"], conv.wav, "en", False, reference=reference,
                          turns=conv.turns, words=[(a, b, t) for a, b, t, _s in conv.words], speakers=3,
                          notes="espeak-ng, three voices"))
    return items
