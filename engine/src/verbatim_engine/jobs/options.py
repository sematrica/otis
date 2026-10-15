"""Processing modes and job option validation/resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..asr.base import DecodeOptions, LoadOptions
from ..asr.registry import AUTO_ORDER, BACKENDS, backend_class
from ..errors import BackendUnavailable, InvalidParams, ModelNotInstalled
from ..models.catalog import ModelSpec, get_spec
from ..models.manager import ModelManager

# Whisper's language codes (large-v3 vocabulary, includes Cantonese).
WHISPER_LANGUAGES = (
    "en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg "
    "lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so "
    "af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su yue"
).split()


@dataclass(frozen=True)
class ModePreset:
    id: str
    label: str
    description: str
    family: str
    beam_size: int
    best_of: int
    cpu_compute_type: str
    #: What this mode does *not* do yet, shown verbatim in the UI.
    limitations: str = ""


MODES: dict[str, ModePreset] = {
    "fast": ModePreset(
        id="fast",
        label="Fast",
        description="Whisper Small with greedy decoding. Quickest; less accurate on difficult audio.",
        family="small",
        beam_size=1,
        best_of=1,
        cpu_compute_type="int8",
    ),
    "balanced": ModePreset(
        id="balanced",
        label="Balanced",
        description="Whisper Large-v3 Turbo with beam search where the backend supports it.",
        family="large-v3-turbo",
        beam_size=5,
        best_of=5,
        cpu_compute_type="int8",
    ),
    "maximum": ModePreset(
        id="maximum",
        label="Maximum Accuracy",
        description="Whisper Large-v3 with beam search where supported (float32 on the CPU backend).",
        family="large-v3",
        beam_size=5,
        best_of=5,
        cpu_compute_type="float32",
        limitations="Multi-pass verification and targeted re-transcription are not implemented yet; "
        "this mode currently runs a single high-accuracy pass.",
    ),
}

DEFAULT_CHUNK_SECONDS = 600.0
CHUNK_PAD_SECONDS = 1.0


def modes_payload() -> list[dict[str, Any]]:
    return [asdict(m) for m in MODES.values()]


def normalize_options(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Validate client-supplied job options and fill defaults. The result is
    stored with the job and is what a resumed job runs with."""
    raw = dict(raw or {})
    mode = raw.pop("mode", "balanced")
    if mode not in MODES:
        raise InvalidParams(f"Unknown mode '{mode}'.", data={"modes": list(MODES)})
    backend = raw.pop("backend", "auto") or "auto"
    if backend != "auto" and backend not in BACKENDS:
        raise InvalidParams(f"Unknown backend '{backend}'.", data={"backends": ["auto", *BACKENDS]})
    model_id = raw.pop("model_id", None) or None
    if model_id is not None:
        spec = get_spec(model_id)
        if backend not in ("auto", spec.backend):
            raise InvalidParams(f"Model '{model_id}' runs on the {spec.backend} backend, not {backend}.")
        backend = spec.backend
    model_path = raw.pop("model_path", None) or None
    if model_path is not None:
        if backend == "auto":
            raise InvalidParams("A custom model folder requires an explicit backend.")
        if model_id is not None:
            raise InvalidParams("Specify either model_id or model_path, not both.")
        if not Path(model_path).is_absolute():
            raise InvalidParams("model_path must be absolute.")
    language = raw.pop("language", None)
    if language in ("", "auto"):
        language = None
    if language is not None and language not in WHISPER_LANGUAGES:
        raise InvalidParams(f"Unsupported language code '{language}'.")
    chunk_seconds = float(raw.pop("chunk_seconds", DEFAULT_CHUNK_SECONDS))
    if not 20.0 <= chunk_seconds <= 3600.0:
        raise InvalidParams("chunk_seconds must be between 20 and 3600.")
    vad = bool(raw.pop("vad", True))
    if raw:
        raise InvalidParams(f"Unknown option(s): {', '.join(sorted(raw))}.")
    return {
        "mode": mode,
        "backend": backend,
        "model_id": model_id,
        "model_path": model_path,
        "language": language,
        "chunk_seconds": chunk_seconds,
        "vad": vad,
    }


@dataclass
class ResolvedJob:
    mode: ModePreset
    backend: str
    model_dir: Path
    model: dict[str, Any]
    load: LoadOptions
    decode: DecodeOptions
    chunk_seconds: float
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode.id,
            "backend": self.backend,
            "model": self.model,
            "chunk_seconds": self.chunk_seconds,
            "warnings": self.warnings,
        }


def _candidate_specs(family: str, backend: str) -> list[ModelSpec]:
    names = AUTO_ORDER if backend == "auto" else (backend,)
    specs = []
    for name in names:
        spec_id = f"{name}/{family}"
        try:
            specs.append(get_spec(spec_id))
        except Exception:  # noqa: BLE001 - family not offered for this backend
            continue
    return specs


def resolve(options: dict[str, Any], models: ModelManager) -> ResolvedJob:
    mode = MODES[options["mode"]]
    warnings: list[str] = []

    if options["model_path"]:
        backend = options["backend"]
        available, reason = backend_class(backend).availability()
        if not available:
            raise BackendUnavailable(reason or f"{backend} is unavailable", data={"backend": backend})
        model_dir = models.validate_custom_dir(backend, options["model_path"])
        model = {"id": None, "display_name": f"Custom model ({model_dir.name})", "path": str(model_dir), "revision": None}
    else:
        if options["model_id"]:
            specs = [get_spec(options["model_id"])]
        else:
            specs = _candidate_specs(mode.family, options["backend"])
        usable = [s for s in specs if backend_class(s.backend).availability()[0]]
        if not usable:
            reasons = {s.backend: backend_class(s.backend).availability()[1] for s in specs}
            raise BackendUnavailable("No speech-recognition backend is available for this model.", data={"reasons": reasons})
        installed = [s for s in usable if models.status(s)["installed"]]
        if not installed:
            suggestion = usable[0]
            raise ModelNotInstalled(
                f"{suggestion.display_name} is not installed. Download it in Settings → Models.",
                data={
                    "model_id": suggestion.id,
                    "display_name": suggestion.display_name,
                    "approx_download_bytes": suggestion.approx_download_bytes,
                    "alternatives": [s.id for s in usable[1:]],
                },
            )
        spec = installed[0]
        if spec is not usable[0]:
            warnings.append(
                f"Using the {spec.backend} backend because {usable[0].display_name} for {usable[0].backend} is not installed."
            )
        backend = spec.backend
        status = models.status(spec)
        model_dir = Path(status["path"])
        model = {"id": spec.id, "display_name": spec.display_name, "path": str(model_dir), "revision": status["revision"]}

    caps = backend_class(backend).capabilities()
    if mode.beam_size > 1 and not caps.beam_search:
        warnings.append(f"Beam search is not supported by the {backend} backend; greedy decoding is used.")
    if mode.limitations:
        warnings.append(mode.limitations)

    decode = DecodeOptions(
        language=options["language"],
        beam_size=mode.beam_size if caps.beam_search else 1,
        best_of=mode.best_of,
        word_timestamps=caps.word_timestamps,
        vad=options["vad"],
    )
    load = LoadOptions(compute_type=mode.cpu_compute_type)
    return ResolvedJob(
        mode=mode,
        backend=backend,
        model_dir=model_dir,
        model=model,
        load=load,
        decode=decode,
        chunk_seconds=options["chunk_seconds"],
        warnings=warnings,
    )
