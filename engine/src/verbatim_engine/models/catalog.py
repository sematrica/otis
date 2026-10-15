"""Catalog of downloadable speech-recognition models.

Download sizes are approximate and are checked against the HuggingFace API by
``scripts/verify_model_catalog.py`` (run in CI). Memory figures are estimates
of peak use for one loaded model while decoding; they have not been measured
on the target machine yet and are presented to the user as estimates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..errors import NotFound

# Decimal units, matching how macOS Finder and HuggingFace report sizes.
GB = 1000**3


@dataclass(frozen=True)
class ModelSpec:
    id: str
    backend: str
    family: str
    display_name: str
    repo_id: str
    parameters: str
    approx_download_bytes: int
    #: compute type / dtype -> estimated peak memory in bytes
    memory_estimates: dict[str, int] = field(default_factory=dict)
    allow_patterns: tuple[str, ...] = ()
    multilingual: bool = True
    revision: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allow_patterns"] = list(self.allow_patterns)
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

_BY_ID = {spec.id: spec for spec in CATALOG}


def get_spec(model_id: str) -> ModelSpec:
    try:
        return _BY_ID[model_id]
    except KeyError:
        raise NotFound(f"Unknown model '{model_id}'.", data={"model_id": model_id, "known": sorted(_BY_ID)}) from None


def specs_for_family(family: str) -> list[ModelSpec]:
    return [spec for spec in CATALOG if spec.family == family]
