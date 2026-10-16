"""CTC acoustic models usable for forced alignment.

Each model turns 16 kHz audio into per-frame log-probabilities over its
token vocabulary, and turns a transcript word into tokens it can emit (or
``None`` if it cannot spell the word, e.g. digits or an unsupported script).

* :class:`NemoCTCModel` - NVIDIA NeMo Conformer-CTC (English, SentencePiece
  tokens, 40 ms frames), in the ONNX export published by the sherpa-onnx
  project.
* :class:`MMSModel` - Meta's MMS-300M forced-alignment model (wav2vec 2.0,
  characters, 20 ms frames, more than 1,100 languages through romanisation).
"""

from __future__ import annotations

import abc
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np

from ..audio import SAMPLE_RATE
from ..errors import EngineError

_WORD_SPLIT = re.compile(r"[\s\-‐‑‒–—/_]+")


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _session(path: Path, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, threads)
    options.inter_op_num_threads = 1
    try:
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    except Exception as exc:  # onnxruntime raises its own types
        raise EngineError(f"The alignment model could not be loaded: {exc}", code="model_invalid") from exc


class CTCModel(abc.ABC):
    id: str
    display_name: str
    frame_seconds: float
    blank: int
    vocab_size: int
    #: Where, between two words' token spikes, connected speech is split
    #: (see :mod:`.refine`); measured per model on speech with known word times.
    connected_split: float = 0.62

    @abc.abstractmethod
    def supports(self, language: str | None) -> bool: ...

    @abc.abstractmethod
    def emissions(self, audio: np.ndarray) -> np.ndarray:
        """(T, vocab_size) log-probabilities; frame t covers
        [t * frame_seconds, (t + 1) * frame_seconds) of ``audio``."""

    @abc.abstractmethod
    def tokenize(self, word: str, language: str | None) -> list[int] | None: ...

    def describe(self) -> dict:
        return {"id": self.id, "display_name": self.display_name, "frame_seconds": self.frame_seconds}


# --------------------------------------------------------------------------- NeMo


class NemoCTCModel(CTCModel):
    """NeMo ``EncDecCTCModelBPE`` exported to ONNX (log-probs output).

    Features follow sherpa-onnx's preprocessing for these exports: 80-bin
    Kaldi filterbanks on int16-scaled audio (no frame snipping, no dither),
    normalised per feature over the input.
    """

    def __init__(self, model_dir: Path, *, model_id: str, display_name: str, threads: int):
        self.id = model_id
        self.display_name = display_name
        self.session = _session(model_dir / "model.onnx", threads)
        meta = self.session.get_modelmeta().custom_metadata_map
        if meta.get("model_type") != "EncDecCTCModelBPE" or meta.get("normalize_type") != "per_feature":
            raise EngineError("Unexpected NeMo CTC model.", code="model_invalid")
        self.subsampling = int(meta.get("subsampling_factor", "4"))
        self.frame_seconds = 0.01 * self.subsampling
        pieces: dict[str, int] = {}
        for line in (model_dir / "tokens.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                pieces[parts[0]] = int(parts[1])
        self.vocab_size = len(pieces)
        if int(meta.get("vocab_size", self.vocab_size)) != self.vocab_size:
            raise EngineError("Alignment model tokens do not match the model.", code="model_invalid")
        self.blank = pieces.get("<blk>", self.vocab_size - 1)
        self._pieces = {p: i for p, i in pieces.items() if not p.startswith("<")}
        self._max_piece = max(len(p) for p in self._pieces)
        # Zipf's law: p(rank r) = 1 / (r * H), H = harmonic number of the vocabulary
        harmonic = math.log(len(self._pieces)) + 0.5772
        self._log_prob = {i: -math.log(i + 1) - math.log(harmonic) for i in self._pieces.values()}
        self._inputs = [i.name for i in self.session.get_inputs()]
        self._output = self.session.get_outputs()[0].name

    def supports(self, language: str | None) -> bool:
        return language == "en"

    def features(self, audio: np.ndarray) -> np.ndarray:
        import kaldi_native_fbank as knf

        options = knf.FbankOptions()
        options.frame_opts.dither = 0.0
        options.frame_opts.snip_edges = False
        options.frame_opts.samp_freq = SAMPLE_RATE
        options.mel_opts.num_bins = 80
        computer = knf.OnlineFbank(options)
        computer.accept_waveform(SAMPLE_RATE, (audio * 32768.0).astype(np.float32))
        computer.input_finished()
        if computer.num_frames_ready == 0:
            return np.zeros((0, 80), dtype=np.float32)
        feats = np.stack([computer.get_frame(i) for i in range(computer.num_frames_ready)])
        return (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-5)

    def emissions(self, audio: np.ndarray) -> np.ndarray:
        feats = self.features(audio).astype(np.float32)
        if feats.shape[0] < 4 * self.subsampling:  # too short for the network
            return np.zeros((0, self.vocab_size), dtype=np.float32)
        out = self.session.run(
            [self._output],
            {self._inputs[0]: feats.T[None], self._inputs[1]: np.array([feats.shape[0]], dtype=np.int64)},
        )[0][0]
        return out.astype(np.float32)

    def tokenize(self, word: str, language: str | None) -> list[int] | None:
        """SentencePiece pieces spelling the word, or None if impossible.

        The model was trained on its tokenizer's segmentation, and a different
        split of the same word (``▁h el lo`` instead of ``▁he ll o``) scores
        badly. Only the piece list survives in the export, so the unigram
        tokenizer is approximated: pieces are ranked by frequency, so a piece's
        log-probability is taken as Zipf's law predicts from its rank, and the
        most probable segmentation wins. This reproduces the model's own
        segmentation for 98.5% of the words it decoded in our test audio
        (the "fewest pieces" rule: 88.5%).
        """
        parts = [p for p in _WORD_SPLIT.split(_normalise_latin(word)) if p]
        if not parts:
            return None
        tokens: list[int] = []
        for part in parts:
            if not re.fullmatch(r"[a-z']+", part):
                return None
            text = "▁" + part
            n = len(text)
            best: list[tuple[float, list[int]] | None] = [None] * (n + 1)
            best[0] = (0.0, [])
            for i in range(n):
                if best[i] is None:
                    continue
                for j in range(i + 1, min(n, i + self._max_piece) + 1):
                    piece = self._pieces.get(text[i:j])
                    if piece is None:
                        continue
                    score = best[i][0] + self._log_prob[piece]
                    if best[j] is None or score > best[j][0]:
                        best[j] = (score, best[i][1] + [piece])
            if best[n] is None:
                return None
            tokens.extend(best[n][1])
        return tokens


def _normalise_latin(word: str) -> str:
    """Lower-case, typographic apostrophes to ', drop punctuation and accents."""
    word = unicodedata.normalize("NFKC", word).casefold().replace("’", "'").replace("‘", "'")
    word = "".join(ch for ch in unicodedata.normalize("NFKD", word) if not unicodedata.combining(ch))
    kept = []
    for ch in word:
        cat = unicodedata.category(ch)
        if ch == "'" or ch.isalnum() or ch.isspace() or ch in "-‐‑‒–—/_":
            kept.append(ch)
        elif cat.startswith("P") or cat.startswith("S"):
            continue
        else:
            kept.append(ch)
    return "".join(kept).strip("'")


# --------------------------------------------------------------------------- MMS

#: Vocabulary of the MMS-300M forced-alignment model (torchaudio ``MMS_FA``).
MMS_VOCAB = {
    "<blank>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "a": 4, "i": 5, "e": 6, "n": 7, "o": 8, "u": 9,
    "t": 10, "s": 11, "r": 12, "m": 13, "k": 14, "l": 15, "d": 16, "g": 17, "h": 18, "y": 19, "b": 20,
    "p": 21, "w": 22, "c": 23, "v": 24, "j": 25, "z": 26, "f": 27, "'": 28, "q": 29, "x": 30,
}


class MMSModel(CTCModel):
    """wav2vec 2.0 MMS forced aligner (ONNX export with a ``logits`` output).

    Non-Latin scripts are romanised with ``uroman`` (the procedure the model
    was trained with); words that do not romanise to its alphabet are left to
    the aligner's wildcard.
    """

    #: Context added on each side of the audio given to the network (its
    #: receptive field is large; edges are otherwise poorly modelled).
    CONTEXT_SECONDS = 1.0

    def __init__(self, model_dir: Path, *, model_id: str, display_name: str, threads: int):
        self.id = model_id
        self.display_name = display_name
        files = sorted(model_dir.glob("*.onnx"))
        if not files:
            raise EngineError("The alignment model file is missing.", code="model_invalid")
        self.session = _session(files[0], threads)
        self._input = self.session.get_inputs()[0].name
        self._output = self.session.get_outputs()[0].name
        self.frame_seconds = 0.02
        self.blank = MMS_VOCAB["<blank>"]
        self.vocab_size = len(MMS_VOCAB)
        self._romanizer = None
        meta_path = model_dir / "verbatim-model.json"
        self._meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}

    def supports(self, language: str | None) -> bool:
        return True

    def emissions(self, audio: np.ndarray) -> np.ndarray:
        context = int(self.CONTEXT_SECONDS * SAMPLE_RATE)
        audio = audio.astype(np.float32)
        audio = (audio - audio.mean()) / (audio.std() + 1e-7)  # the model expects normalised input
        padded = np.concatenate([np.zeros(context, np.float32), audio, np.zeros(context, np.float32)])
        logits = self.session.run([self._output], {self._input: padded[None]})[0][0]
        skip = int(round(self.CONTEXT_SECONDS / self.frame_seconds))
        frames = int(np.ceil(audio.size / SAMPLE_RATE / self.frame_seconds))
        logits = logits[skip : skip + frames]
        return _log_softmax(logits.astype(np.float32))

    def _romanize(self, text: str, language: str | None) -> str:
        if all(ord(ch) < 0x250 for ch in text):  # Latin script: no romanisation needed
            return text
        if self._romanizer is None:
            try:
                import uroman

                self._romanizer = uroman.Uroman()
            except Exception:  # noqa: BLE001 - optional dependency missing or broken
                self._romanizer = False
        if not self._romanizer:
            return text
        try:
            return str(self._romanizer.romanize_string(text))
        except Exception:  # noqa: BLE001
            return text

    def tokenize(self, word: str, language: str | None) -> list[int] | None:
        text = _normalise_latin(self._romanize(word, language))
        tokens = [MMS_VOCAB[ch] for ch in text if ch in MMS_VOCAB and MMS_VOCAB[ch] > 3]
        letters = [ch for ch in text if not ch.isspace() and ch not in "-‐‑‒–—/_"]
        if not tokens or len(tokens) < len(letters):  # something it cannot spell (digits, symbols)
            return None
        return tokens


def load_model(spec, path: Path, threads: int) -> CTCModel:
    """Instantiate the acoustic model for a catalog alignment entry."""
    if spec.family.startswith("en-conformer-"):
        model = NemoCTCModel(path, model_id=spec.id, display_name=spec.display_name, threads=threads)
        # measured with scripts/benchmark_word_timing.py (synthetic speech)
        model.connected_split = 0.45 if spec.family == "en-conformer-medium" else 0.62
        return model
    if spec.family == "mms-300m":
        return MMSModel(path, model_id=spec.id, display_name=spec.display_name, threads=threads)
    raise EngineError(f"No loader for alignment model {spec.id}.", code="model_invalid")
