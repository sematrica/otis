"""Processing modes and job option validation/resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..asr.base import DecodeOptions, LoadOptions
from ..asr.registry import AUTO_ORDER, BACKENDS, backend_class
from ..errors import BackendUnavailable, InvalidParams, ModelNotInstalled
from ..models.catalog import ModelSpec, get_spec, specs_of_kind
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
    #: Forced word alignment by default (Whisper's own word timings otherwise).
    align: bool = True
    #: Re-transcribe doubtful passages and choose by acoustic evidence (passes 6-7).
    recheck: bool = False


MODES: dict[str, ModePreset] = {
    "fast": ModePreset(
        id="fast",
        label="Fast",
        description="Whisper Small with greedy decoding and speaker identification; word times from "
        "Whisper; quality checks flag doubtful passages. Quickest; less accurate on difficult audio.",
        family="small",
        beam_size=1,
        best_of=1,
        cpu_compute_type="int8",
        align=False,
    ),
    "balanced": ModePreset(
        id="balanced",
        label="Balanced",
        description="Whisper Large-v3 Turbo with beam search where the backend supports it, speaker "
        "identification, forced word alignment and quality checks.",
        family="large-v3-turbo",
        beam_size=5,
        best_of=5,
        cpu_compute_type="int8",
    ),
    "maximum": ModePreset(
        id="maximum",
        label="Maximum Accuracy",
        description="Whisper Large-v3 with beam search where supported (float32 on the CPU backend), "
        "speaker identification, forced word alignment, quality checks, and re-transcription of doubtful "
        "passages with the wording chosen by acoustic evidence. Slowest.",
        family="large-v3",
        beam_size=5,
        best_of=5,
        cpu_compute_type="float32",
        limitations="Wordings are compared acoustically only in languages an installed word-alignment model "
        "supports; elsewhere, differing wordings are listed for review instead.",
        recheck=True,
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
    diarize = _tristate(raw.pop("diarize", "auto"), "diarize")
    align = _tristate(raw.pop("align", "auto" if MODES[mode].align else False), "align")
    alignment_model = raw.pop("alignment_model", None) or None
    if alignment_model is not None:
        spec = get_spec(alignment_model)
        if spec.kind != "alignment":
            raise InvalidParams(f"'{alignment_model}' is not an alignment model.")
    speakers = _speaker_bounds(raw.pop("speakers", None))
    recheck = _tristate(raw.pop("recheck", "auto"), "recheck")
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
        "diarize": diarize,
        "speakers": speakers,
        "align": align,
        "alignment_model": alignment_model,
        "recheck": recheck,
    }


def _tristate(value: Any, name: str) -> bool | str:
    """True (required), False (off) or "auto" (when the model is installed)."""
    if value in (True, False, "auto"):
        return value
    raise InvalidParams(f"Option '{name}' must be true, false or \"auto\".")


def _speaker_bounds(value: Any) -> dict[str, int | None]:
    value = dict(value or {})
    out: dict[str, int | None] = {}
    for key in ("num", "min", "max"):
        v = value.pop(key, None)
        if v is not None and (not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 50):
            raise InvalidParams(f"speakers.{key} must be a whole number from 1 to 50.")
        out[key] = v
    if value:
        raise InvalidParams(f"Unknown speakers option(s): {', '.join(sorted(value))}.")
    if out["num"] is not None and (out["min"] is not None or out["max"] is not None):
        raise InvalidParams("Give either an exact number of speakers or a range, not both.")
    if out["min"] is not None and out["max"] is not None and out["min"] > out["max"]:
        raise InvalidParams("The minimum number of speakers is larger than the maximum.")
    return out


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
    #: Diarization model folder, or None when speakers are not identified.
    diarization_dir: Path | None = None
    diarization_skip_reason: str | None = None
    speaker_bounds: dict[str, int | None] = field(default_factory=dict)
    #: Installed alignment models that may be used (chosen per language once
    #: the language is known), most preferred first.
    alignment_candidates: list[tuple[ModelSpec, Path]] = field(default_factory=list)
    alignment_skip_reason: str | None = None
    #: Targeted re-transcription (passes 6-7).
    recheck: bool = False
    recheck_skip_reason: str | None = None
    #: Installed acoustic models for comparing wordings (independent of the
    #: word-alignment setting), most preferred first.
    scoring_candidates: list[tuple[ModelSpec, Path]] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode.id,
            "backend": self.backend,
            "model": self.model,
            "chunk_seconds": self.chunk_seconds,
            "warnings": self.warnings,
            "diarization": {
                "enabled": self.diarization_dir is not None,
                "skip_reason": self.diarization_skip_reason,
                "speakers": {k: v for k, v in self.speaker_bounds.items() if v is not None},
            },
            "alignment": {
                "enabled": bool(self.alignment_candidates),
                "candidates": [spec.id for spec, _ in self.alignment_candidates],
                "skip_reason": self.alignment_skip_reason,
            },
            "recheck": {
                "enabled": self.recheck,
                "skip_reason": self.recheck_skip_reason,
                "acoustic_models": [spec.id for spec, _ in self.scoring_candidates],
            },
        }

    def alignment_model_for(self, language: str | None) -> tuple[ModelSpec, Path] | None:
        return _first_for(self.alignment_candidates, language)

    def scoring_model_for(self, language: str | None) -> tuple[ModelSpec, Path] | None:
        return _first_for(self.scoring_candidates, language)


def _first_for(candidates: list[tuple[ModelSpec, Path]], language: str | None) -> tuple[ModelSpec, Path] | None:
    if language is None:
        return None
    for spec, path in candidates:
        if not spec.languages or language in spec.languages:
            return spec, path
    return None


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

    diarization_dir, diarization_skip = _resolve_diarization(options, models)
    alignment, alignment_skip = _resolve_alignment(options, models)
    recheck_setting = options.get("recheck", "auto")
    recheck = mode.recheck if recheck_setting == "auto" else bool(recheck_setting)
    recheck_skip = None
    if not recheck:
        recheck_skip = ("Re-checking doubtful passages runs in Maximum Accuracy mode." if recheck_setting == "auto"
                        else "Re-checking doubtful passages was turned off for this job.")
    scoring = _installed_alignment(models, options.get("language")) if recheck else []

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
        diarization_dir=diarization_dir,
        diarization_skip_reason=diarization_skip,
        speaker_bounds=dict(options.get("speakers") or {}),
        alignment_candidates=alignment,
        alignment_skip_reason=alignment_skip,
        recheck=recheck,
        recheck_skip_reason=recheck_skip,
        scoring_candidates=scoring,
    )


#: Preference among installed alignment models (first match for the language
#: wins), by measured word-timing precision (docs/STAGE2_REPORT.md). MMS is
#: only used if the user installed it after seeing its non-commercial licence.
ALIGNMENT_PREFERENCE = ("alignment/mms-300m", "alignment/en-conformer-small", "alignment/en-conformer-medium")


def _resolve_diarization(options: dict[str, Any], models: ModelManager) -> tuple[Path | None, str | None]:
    setting = options.get("diarize", "auto")
    if setting is False:
        return None, "Speaker identification was turned off for this job."
    installed = models.installed("diarization")
    if installed:
        return Path(models.status(installed[0])["path"]), None
    spec = specs_of_kind("diarization")[0]
    if setting is True:
        raise ModelNotInstalled(
            "The speaker identification model is not installed. Download it in Settings → Models.",
            data={"model_id": spec.id, "display_name": spec.display_name,
                  "approx_download_bytes": spec.approx_download_bytes},
        )
    return None, "The speaker identification model is not installed."


def _resolve_alignment(options: dict[str, Any], models: ModelManager) -> tuple[list[tuple[ModelSpec, Path]], str | None]:
    setting = options.get("align", "auto")
    if setting is False:
        return [], "Word alignment is off for this job; word times come from Whisper."
    if options.get("alignment_model"):
        spec = get_spec(options["alignment_model"])
        path = models.require_installed(spec)
        return [(spec, path)], None
    language = options.get("language")
    order = _installed_alignment(models, language)
    if order:
        return order, None
    if setting is True:
        suggestion = get_spec("alignment/en-conformer-small" if language in (None, "en") else "alignment/mms-300m")
        raise ModelNotInstalled(
            "No word-alignment model for this language is installed. Download one in Settings → Models.",
            data={"model_id": suggestion.id, "display_name": suggestion.display_name,
                  "approx_download_bytes": suggestion.approx_download_bytes},
        )
    return [], "No word-alignment model is installed; word times come from Whisper."


def _installed_alignment(models: ModelManager, language: str | None) -> list[tuple[ModelSpec, Path]]:
    installed = {spec.id: spec for spec in models.installed("alignment")}
    order = [installed[i] for i in ALIGNMENT_PREFERENCE if i in installed]
    order += [s for i, s in installed.items() if i not in ALIGNMENT_PREFERENCE]
    if language is not None:
        order = [s for s in order if not s.languages or language in s.languages]
    return [(s, Path(models.status(s)["path"])) for s in order]
