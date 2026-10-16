"""Catalog of downloadable models.

Three kinds of models:

* ``asr`` - Whisper models for the speech-recognition backends, downloaded
  from HuggingFace repositories;
* ``diarization`` - the speaker segmentation and speaker embedding networks;
* ``alignment`` - CTC acoustic models for forced word alignment.

Diarization and English alignment models are fixed files from GitHub release
assets, pinned by SHA-256 checksum. Download sizes are checked by
``scripts/verify_model_catalog.py`` (run in CI). Memory figures are estimates
of peak use for one loaded model; they have not been measured on the target
machine and are presented to the user as estimates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..errors import NotFound

# Decimal units, matching how macOS Finder and HuggingFace report sizes.
GB = 1000**3


@dataclass(frozen=True)
class RemoteFile:
    """A file downloaded from a fixed URL and checked against its SHA-256."""

    url: str
    sha256: str
    size: int
    #: Saved under this name (plain files) ...
    name: str | None = None
    #: ... or a ``.tar.bz2`` archive from which these (member, saved name)
    #: pairs are extracted; the rest of the archive is discarded.
    members: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ModelSpec:
    id: str
    backend: str
    family: str
    display_name: str
    repo_id: str | None
    parameters: str
    approx_download_bytes: int
    #: compute type / dtype -> estimated peak memory in bytes
    memory_estimates: dict[str, int] = field(default_factory=dict)
    allow_patterns: tuple[str, ...] = ()
    multilingual: bool = True
    revision: str | None = None
    notes: str = ""
    kind: str = "asr"
    #: Files from fixed URLs (instead of a HuggingFace repository).
    files: tuple[RemoteFile, ...] = ()
    #: Files that must exist for the model to count as installed (non-ASR).
    required_files: tuple[str, ...] = ()
    #: Language codes the model supports; empty means "any".
    languages: tuple[str, ...] = ()
    license: str = ""
    homepage: str = ""

    @property
    def install_group(self) -> str:
        return self.backend if self.kind == "asr" else self.kind

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allow_patterns"] = list(self.allow_patterns)
        data["files"] = [{"url": f.url, "size": f.size, "sha256": f.sha256} for f in self.files]
        data["required_files"] = list(self.required_files)
        data["languages"] = list(self.languages)
        return data


_FW_PATTERNS = ("config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*")
_MLX_PATTERNS = ("config.json", "weights.safetensors", "weights.npz")


def _fw(family: str, name: str, repo: str, params: str, size: int, int8: int, fp32: int, notes: str = "") -> ModelSpec:
    return ModelSpec(
        id=f"faster-whisper/{family}",
        backend="faster-whisper",
        family=family,
        display_name=name,
        repo_id=repo,
        parameters=params,
        approx_download_bytes=size,
        memory_estimates={"int8": int8, "float32": fp32},
        allow_patterns=_FW_PATTERNS,
        notes=notes,
    )


def _mlx(family: str, name: str, repo: str, params: str, size: int, fp16: int, notes: str = "") -> ModelSpec:
    return ModelSpec(
        id=f"mlx/{family}",
        backend="mlx",
        family=family,
        display_name=name,
        repo_id=repo,
        parameters=params,
        approx_download_bytes=size,
        memory_estimates={"float16": fp16},
        allow_patterns=_MLX_PATTERNS,
        notes=notes,
    )


CATALOG: tuple[ModelSpec, ...] = (
    _fw("small", "Whisper Small", "Systran/faster-whisper-small", "244M", int(0.48 * GB), int(0.8 * GB), int(1.4 * GB)),
    _fw("medium", "Whisper Medium", "Systran/faster-whisper-medium", "769M", int(1.53 * GB), int(1.6 * GB), int(3.5 * GB)),
    _fw("large-v3", "Whisper Large-v3", "Systran/faster-whisper-large-v3", "1550M", int(3.09 * GB), int(2.6 * GB), int(6.8 * GB),
        "Highest accuracy. Slow on CPU; float32 (Maximum Accuracy) needs about 7 GB."),
    _fw("large-v3-turbo", "Whisper Large-v3 Turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "809M",
        int(1.62 * GB), int(1.7 * GB), int(3.8 * GB), "Pruned decoder: much faster than Large-v3, slightly less accurate."),
    _mlx("small", "Whisper Small", "mlx-community/whisper-small-mlx", "244M", int(0.48 * GB), int(1.0 * GB)),
    _mlx("medium", "Whisper Medium", "mlx-community/whisper-medium-mlx", "769M", int(1.53 * GB), int(2.3 * GB)),
    _mlx("large-v3", "Whisper Large-v3", "mlx-community/whisper-large-v3-mlx", "1550M", int(3.09 * GB), int(4.2 * GB),
         "Highest accuracy. Greedy decoding only (no beam search in mlx-whisper)."),
    _mlx("large-v3-turbo", "Whisper Large-v3 Turbo", "mlx-community/whisper-large-v3-turbo", "809M",
         int(1.62 * GB), int(2.4 * GB), "Pruned decoder: much faster than Large-v3, slightly less accurate."),
)

_SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
MB = 1000**2

DIARIZATION_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="diarization/pyannote-wespeaker",
        backend="onnx",
        family="pyannote-wespeaker",
        kind="diarization",
        display_name="Speaker diarization (pyannote segmentation-3.0 + WeSpeaker ResNet34)",
        repo_id=None,
        parameters="1.5M + 6.6M",
        approx_download_bytes=6_958_444 + 26_530_550,
        memory_estimates={"float32": int(0.4 * GB)},
        files=(
            RemoteFile(
                url=f"{_SHERPA}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
                sha256="24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
                size=6_958_444,
                members=(
                    ("sherpa-onnx-pyannote-segmentation-3-0/model.onnx", "segmentation.onnx"),
                    ("sherpa-onnx-pyannote-segmentation-3-0/LICENSE", "LICENSE-segmentation.txt"),
                ),
            ),
            RemoteFile(
                url=f"{_SHERPA}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",
                sha256="e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012",
                size=26_530_550,
                name="embedding.onnx",
            ),
        ),
        required_files=("segmentation.onnx", "embedding.onnx"),
        license="segmentation-3.0: MIT (CNRS); WeSpeaker ResNet34 (VoxCeleb): CC BY 4.0",
        homepage="https://huggingface.co/pyannote/segmentation-3.0",
        notes="The models of pyannote's speaker-diarization-3.1 pipeline, converted to ONNX by the "
        "sherpa-onnx project. Works for any language.",
    ),
)


def _nemo(size: str, display: str, params: str, archive_bytes: int, archive_sha: str, memory: int, notes: str) -> ModelSpec:
    folder = f"sherpa-onnx-nemo-ctc-en-conformer-{size}"
    return ModelSpec(
        id=f"alignment/en-conformer-{size}",
        backend="onnx",
        family=f"en-conformer-{size}",
        kind="alignment",
        display_name=display,
        repo_id=None,
        parameters=params,
        approx_download_bytes=archive_bytes,
        memory_estimates={"float32": memory},
        multilingual=False,
        languages=("en",),
        files=(
            RemoteFile(
                url=f"{_SHERPA}/asr-models/{folder}.tar.bz2",
                sha256=archive_sha,
                size=archive_bytes,
                members=((f"{folder}/model.onnx", "model.onnx"), (f"{folder}/tokens.txt", "tokens.txt")),
            ),
        ),
        required_files=("model.onnx", "tokens.txt"),
        license="CC BY 4.0 (NVIDIA NeMo stt_en_conformer_ctc_" + size + ")",
        homepage=f"https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_conformer_ctc_{size}",
        notes=notes,
    )


ALIGNMENT_MODELS: tuple[ModelSpec, ...] = (
    _nemo("small", "English word alignment (Conformer-CTC Small)", "13M", 76_482_338,
          "83dcb462aece5bef4e8072c267419389f0b8d1f91152d8851765f284ff664caa", int(0.5 * GB),
          "English only. Measured word-start error on synthetic speech: about 35 ms (Whisper alone: about 90 ms)."),
    _nemo("medium", "English word alignment (Conformer-CTC Medium)", "30M", 165_685_608,
          "08cb7b6ebc516a2577c5b152230730ebf5f937507260305ea592c7accd7f899b", int(0.8 * GB),
          "English only. Larger, but less precise than Small in our measurements (about 45 ms word-start error)."),
    ModelSpec(
        id="alignment/mms-300m",
        backend="onnx",
        family="mms-300m",
        kind="alignment",
        display_name="Multilingual word alignment (MMS-300M)",
        repo_id="deskpai/ctc_forced_aligner",
        parameters="300M",
        approx_download_bytes=int(1.26 * GB),
        memory_estimates={"float32": int(2.5 * GB)},
        # the repository's only model file (an ONNX export of torchaudio's MMS_FA)
        allow_patterns=("04ac86b67129634da93aea76e0147ef3.onnx",),
        required_files=("*.onnx",),
        license="CC BY-NC 4.0 (Meta MMS) - non-commercial use only",
        homepage="https://huggingface.co/deskpai/ctc_forced_aligner",
        notes="Over 1,100 languages (non-Latin scripts are romanised); the most precise aligner we measured "
        "(about 20 ms word-start error in English, German and Spanish). Its licence does not allow commercial use.",
    ),
)

ALL_MODELS: tuple[ModelSpec, ...] = CATALOG + DIARIZATION_MODELS + ALIGNMENT_MODELS
_BY_ID = {spec.id: spec for spec in ALL_MODELS}


def get_spec(model_id: str) -> ModelSpec:
    try:
        return _BY_ID[model_id]
    except KeyError:
        raise NotFound(f"Unknown model '{model_id}'.", data={"model_id": model_id, "known": sorted(_BY_ID)}) from None


def specs_for_family(family: str) -> list[ModelSpec]:
    return [spec for spec in CATALOG if spec.family == family]


def specs_of_kind(kind: str) -> list[ModelSpec]:
    return [spec for spec in ALL_MODELS if spec.kind == kind]
