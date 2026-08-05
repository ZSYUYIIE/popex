#!/usr/bin/env python3
"""Safe local checks for PopEx's optional separation runtime.

Only standard-library modules are imported until trusted local paths are checked.
"""
from __future__ import annotations

import argparse, json, math, os, platform, re, shutil, stat, struct, sys, tempfile, time, wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

PREFIX = "popex-separation-doctor-"
PROFILES = {("Linux", "x86_64"): "linux-x86_64-cpu-cpython313", ("Windows", "x86_64"): "windows-x86_64-cpu-cpython313"}
LOCK_KEYS = {"demucs", "torch", "huggingface_hub", "safetensors", "PyYAML"}
TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
CONTROL = re.compile(r"[\x00-\x1f\x7f]+")
SECRET = re.compile(r"(?i)\b(token|password|secret|authorization|api[_-]?key|access[_-]?key|credential)\b\s*[:=]\s*[^\s,;]+")
BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+\-/=]+")
URL = re.compile(r"(?i)https?://[^\s\]\[<>()\"']+")
WINPATH = re.compile(r"(?i)(?:\b[A-Z]:[\\/]|\\\\)[^\s,;\"']+")
POSIXPATH = re.compile(r"(?<![:\w])/(?:[^\s,;\"']+)")


class DoctorError(RuntimeError):
    def __init__(self, state: str, message: str, code: int = 2):
        self.state, self.message, self.code = state, safe(message), code
        super().__init__(self.message)


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise DoctorError("unavailable", f"Invalid doctor arguments: {message}")


@dataclass(frozen=True, slots=True)
class Config:
    worker: Path
    runtime_lock: Path
    cache_root: Path
    expected_profile: str
    device: str = "cpu"
    timeout: float = 3600.0
    temp_parent: Path | None = None


@dataclass(frozen=True, slots=True)
class Identity:
    os_name: str
    architecture: str
    python: str
    profile: str


def parser() -> Parser:
    p = Parser(prog="popex-separation-doctor")
    subs = p.add_subparsers(dest="mode", required=True)
    for name in ("check", "validate"):
        q = subs.add_parser(name)
        q.add_argument("--worker", required=True)
        q.add_argument("--runtime-lock", required=True)
        q.add_argument("--cache-root", required=True)
        q.add_argument("--expected-profile", required=True)
        q.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
        q.add_argument("--timeout-seconds", type=positive, default=3600.0)
        if name == "validate":
            q.add_argument("--allow-model-download", action="store_true")
            q.add_argument("--temporary-root")
    return p


def positive(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if number <= 0 or not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def privacy(passive: bool) -> None:
    for key in list(os.environ):
        if any(x in key.upper() for x in ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY")):
            os.environ.pop(key, None)
    os.environ.update({
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_UPDATE_CHECK": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "PYTHONNOUSERSITE": "1", "DO_NOT_TRACK": "1", "TQDM_DISABLE": "1", "NO_COLOR": "1",
    })
    if passive:
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)


def identity() -> Identity:
    machine = platform.machine().lower()
    arch = "x86_64" if machine in {"x86_64", "amd64"} else machine
    profile = PROFILES.get((platform.system(), arch))
    if profile is None:
        raise DoctorError("unavailable", "Only Windows or Linux x86-64 CPU profiles are supported.")
    if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 13):
        raise DoctorError("unavailable", "CPython 3.13 is required for the validated CPU profiles.")
    return Identity(platform.system().lower(), arch, "3.13", profile)


def raw(path: Path) -> str:
    value = os.fspath(path)
    if not value or "\0" in value:
        raise DoctorError("unavailable", "A trusted path is invalid.")
    return value


def normalized(path: Path) -> Path:
    value = raw(path)
    if not path.is_absolute():
        raise DoctorError("unavailable", "Trusted paths must be absolute.")
    result = Path(os.path.normpath(value))
    if os.path.normcase(str(result)) != os.path.normcase(value):
        raise DoctorError("unavailable", "Trusted paths must be normalized.")
    return result


def no_links(path: Path) -> None:
    current = path
    while True:
        if current.exists() or current.is_symlink():
            try:
                mode = current.lstat().st_mode
            except OSError as exc:
                raise DoctorError("unavailable", "A trusted path is unavailable.") from exc
            if stat.S_ISLNK(mode):
                raise DoctorError("unavailable", "Trusted paths may not use symbolic links.")
        if current.parent == current:
            return
        current = current.parent


def trusted_file(path: Path, missing: str, executable: bool = False) -> Path:
    path = normalized(path)
    if not path.exists():
        raise DoctorError(missing, "A required runtime file is missing.")
    no_links(path)
    try:
        mode, resolved = path.lstat().st_mode, path.resolve(strict=True)
    except OSError as exc:
        raise DoctorError(missing, "A required runtime file is unavailable.") from exc
    if not stat.S_ISREG(mode) or os.path.normcase(str(resolved)) != os.path.normcase(str(path)):
        raise DoctorError("unavailable", "A trusted runtime file is unsafe.")
    if executable and os.name != "nt" and not os.access(resolved, os.X_OK):
        raise DoctorError("runtime_missing", "The worker is not executable.")
    return resolved


def trusted_dir(path: Path, create: bool) -> Path:
    path = normalized(path)
    no_links(path)
    if not path.exists():
        if not create:
            raise DoctorError("unavailable", "A trusted directory is missing.")
        try:
            path.mkdir(parents=True)
        except OSError as exc:
            raise DoctorError("unavailable", "The private cache could not be created safely.") from exc
    no_links(path)
    try:
        mode, resolved = path.lstat().st_mode, path.resolve(strict=True)
    except OSError as exc:
        raise DoctorError("unavailable", "A trusted directory is unavailable.") from exc
    if not stat.S_ISDIR(mode) or os.path.normcase(str(resolved)) != os.path.normcase(str(path)):
        raise DoctorError("unavailable", "A trusted directory is unsafe.")
    return resolved


def validate_lock(path: Path, profile: str) -> None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DoctorError("unavailable", "The runtime lock is invalid.") from exc
    packages = value.get("packages") if isinstance(value, dict) else None
    if value.get("schemaVersion") != 1 or value.get("runtimeProfile") != profile:
        raise DoctorError("unavailable", "The runtime lock does not match the expected profile.")
    if not isinstance(packages, dict) or set(packages) != LOCK_KEYS or packages.get("demucs") != "4.1.0":
        raise DoctorError("unavailable", "The runtime lock package set is invalid.")
    if any(not isinstance(v, str) or not v for v in packages.values()):
        raise DoctorError("unavailable", "The runtime lock contains an invalid version.")


def validate_config(c: Config, probe: Callable[[], Identity] = identity) -> Identity:
    ident = probe()
    if not TOKEN.fullmatch(c.expected_profile) or c.expected_profile != ident.profile:
        raise DoctorError("unavailable", "The expected profile does not match this platform.")
    if c.device != "cpu":
        raise DoctorError("unavailable", "The doctor validates the CPU profile only.")
    if c.timeout <= 0 or not math.isfinite(c.timeout):
        raise DoctorError("unavailable", "The timeout is invalid.")
    trusted_file(c.worker, "runtime_missing", True)
    validate_lock(trusted_file(c.runtime_lock, "unavailable"), c.expected_profile)
    trusted_dir(c.cache_root, True)
    if c.temp_parent is not None:
        trusted_dir(c.temp_parent, False)
    return ident


def make_client(c: Config) -> Any:
    try:
        from app.separation_runtime import SeparationRuntimeClient
        return SeparationRuntimeClient(
            c.worker,
            c.cache_root,
            runtime_lock_path=c.runtime_lock,
            expected_runtime_profile=c.expected_profile,
            command_timeouts={"separate": c.timeout},
        )
    except Exception as exc:
        raise DoctorError("unavailable", "The trusted runtime configuration is unavailable.") from exc


def error_code(exc: BaseException) -> str | None:
    value = getattr(exc, "code", None)
    if isinstance(value, str):
        return value
    return getattr(getattr(exc, "detail", None), "code", None)


def check(
    c: Config,
    client_factory: Callable[[Config], Any] = make_client,
    probe: Callable[[], Identity] = identity,
) -> dict[str, Any]:
    ident = validate_config(c, probe)
    client = client_factory(c)
    try:
        runtime = client.runtime_probe()
    except Exception as exc:
        missing = error_code(exc) == "RUNTIME_MISSING" or exc.__class__.__name__ == "RuntimeMissingError"
        raise DoctorError(
            "runtime_missing" if missing else "unavailable",
            "The optional runtime could not be started." if missing else "The optional runtime is unavailable.",
        ) from None
    profile = getattr(runtime, "runtime_profile", None)
    if profile != c.expected_profile:
        raise DoctorError("unavailable", "The worker reported an unexpected profile.")
    state, model = "ready", None
    try:
        model = client.model_probe()
    except Exception as exc:
        if error_code(exc) == "MODEL_DOWNLOAD_REQUIRED":
            state = "download_required"
        elif error_code(exc) == "RUNTIME_MISSING":
            raise DoctorError("runtime_missing", "The optional runtime could not be started.") from None
        else:
            raise DoctorError("unavailable", "The local model state could not be verified.") from None
    if model is not None and getattr(model, "offline_ready", None) is not True:
        raise DoctorError("unavailable", "The local model is not verified for offline use.")
    source = model or runtime
    return {
        "schemaVersion": 1,
        "mode": "check",
        "state": state,
        "platform": ident.os_name,
        "architecture": ident.architecture,
        "python": ident.python,
        "runtimeProfile": profile,
        "workerVersion": token(getattr(source, "worker_version", None)),
        "demucsVersion": token(getattr(source, "demucs_version", None)),
        "modelPrepared": state == "ready",
        "modelDownloadPerformed": False,
        "audioRemainsLocal": True,
    }


def validate(
    c: Config,
    allowed: bool,
    client_factory: Callable[[Config], Any] = make_client,
    runner: Callable[[Config, Any, Path], Mapping[str, Any]] | None = None,
    probe: Callable[[], Identity] = identity,
) -> dict[str, Any]:
    if allowed is not True:
        raise DoctorError("unavailable", "Explicit --allow-model-download consent is required.")
    ident = validate_config(c, probe)
    client = client_factory(c)
    parent = trusted_dir(c.temp_parent or Path(tempfile.gettempdir()), False)
    root = Path(tempfile.mkdtemp(prefix=PREFIX, dir=parent))
    result, failure = None, None
    try:
        details = dict((runner or app_validation)(c, client, root))
        result = {
            "schemaVersion": 1,
            "mode": "validate",
            "state": "validated",
            "platform": ident.os_name,
            "architecture": ident.architecture,
            "python": ident.python,
            "runtimeProfile": c.expected_profile,
            "modelDownloadAuthorized": True,
            "syntheticAudio": True,
            "audioRemainsLocal": True,
            **details,
        }
    except BaseException as exc:
        failure = exc
    removed = cleanup(root, parent)
    if not removed:
        raise DoctorError(
            "unavailable",
            "Temporary validation data requires manual cleanup from the chosen temporary parent.",
        )
    if failure is not None:
        raise failure
    assert result is not None
    result["temporaryDataRemoved"] = True
    return result


def cleanup(root: Path, parent: Path) -> bool:
    try:
        if not root.name.startswith(PREFIX) or root.is_symlink() or root.resolve().parent != parent.resolve():
            return False
        shutil.rmtree(root)
        return not root.exists()
    except OSError:
        return False


def app_validation(c: Config, runtime_client: Any, root: Path) -> Mapping[str, Any]:
    try:
        from fastapi.testclient import TestClient
        from app import db
        from app.config import Settings
        from app.main import create_app
        from app.media import secure_job_dir
        from app.separation import load_stem_manifest
    except Exception as exc:
        raise DoctorError("unavailable", "The installed PopEx base application is unavailable.") from exc

    settings = Settings(
        data_dir=root,
        allowed_hosts=("example.invalid",),
        max_duration_seconds=60,
        max_filesize_mb=10,
        max_upload_mb=10,
        audio_quality="192",
        ffmpeg_binary=sys.executable,
        ffprobe_binary=sys.executable,
        audio_analysis_enabled=False,
        stem_separation_enabled=True,
        stem_separation_worker_executable=c.worker,
        stem_separation_runtime_lock=c.runtime_lock,
        stem_separation_cache_dir=c.cache_root,
        stem_separation_runtime_profile=c.expected_profile,
        stem_separation_device="cpu",
        stem_separation_timeout_seconds=math.ceil(c.timeout),
    )
    app = create_app(settings=settings, separation_runtime_client=runtime_client)
    job_id = uuid4().hex
    with TestClient(app) as web:
        db.create_job(
            settings.database_path,
            job_id,
            source_type="upload",
            original_filename="synthetic-validation-audio",
        )
        job_dir = secure_job_dir(settings, job_id, create=True)
        synthetic(job_dir / "analysis.wav")
        synthetic(job_dir / "synthetic-source.wav")
        (job_dir / "metadata.json").write_text('{"synthetic":true}', encoding="utf-8")
        db.update_job(
            settings.database_path,
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            message="Synthetic analysis audio is ready.",
            source_file_name="synthetic-source.wav",
            normalized_file_name="analysis.wav",
            metadata_file_name="metadata.json",
            preparation_status="completed",
            analysis_status="completed",
            analysis_version="doctor-synthetic-v1",
            analyzed_at="2026-01-01T00:00:00+00:00",
            error=None,
            analysis_error=None,
        )
        first = web.get(f"/api/jobs/{job_id}")
        payload = first.json() if first.status_code == 200 else {}
        initial = payload.get("separation", {}).get("runtime", {}).get("state")
        if initial not in {"ready", "download_required"}:
            raise DoctorError("unavailable", "The runtime is not actionable.")
        start = web.post(
            f"/api/jobs/{job_id}/separate",
            json={"allowModelDownload": initial == "download_required"},
        )
        if start.status_code != 202:
            raise DoctorError("unavailable", "The synthetic separation request was rejected.")
        deadline = time.monotonic() + c.timeout + 30
        while time.monotonic() < deadline:
            current = web.get(f"/api/jobs/{job_id}").json()
            state = current.get("separation", {}).get("status")
            if state == "completed":
                break
            if state == "failed":
                raise DoctorError("unavailable", "Synthetic separation failed.")
            time.sleep(0.2)
        else:
            raise DoctorError("unavailable", "Synthetic separation timed out.")
        details = web.get(f"/api/jobs/{job_id}/stems")
        if details.status_code != 200:
            raise DoctorError("unavailable", "Synthetic stem details are unavailable.")
        detail_json = details.json()
        kinds = [x.get("kind") for x in detail_json.get("stems", [])]
        if kinds != ["vocals", "bass", "drums", "other"]:
            raise DoctorError("unavailable", "Four stems were not published.")
        for kind in kinds:
            preview = web.get(f"/api/jobs/{job_id}/stems/{kind}/preview")
            download = web.get(f"/api/jobs/{job_id}/stems/{kind}/download")
            if (
                preview.status_code != 200
                or not preview.content
                or not preview.headers.get("content-type", "").startswith("audio/wav")
            ):
                raise DoctorError("unavailable", "A stem preview failed validation.")
            if (
                download.status_code != 200
                or not download.content
                or f"{kind}.wav" not in download.headers.get("content-disposition", "")
            ):
                raise DoctorError("unavailable", "A stem download failed validation.")
        manifest = load_stem_manifest(job_id, settings)
        if manifest is None or manifest.payload.get("schemaVersion") != 3:
            raise DoctorError("unavailable", "The schema-3 manifest is unavailable.")
        public = json.dumps({"start": start.json(), "details": detail_json})
        if any(
            str(p) in public or str(p).replace("\\", "/") in public
            for p in (c.worker, c.runtime_lock, c.cache_root, root)
        ):
            raise DoctorError("unavailable", "A trusted path entered the public API.")
    return {
        "initialState": initial,
        "modelDownloadPerformed": initial == "download_required",
        "manifestSchemaVersion": 3,
        "stems": kinds,
        "previewsVerified": 4,
        "downloadsVerified": 4,
        "sampleRate": 44100,
        "channels": 2,
    }


def synthetic(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setparams((2, 2, 44100, 44100, "NONE", "not compressed"))
        for i in range(44100):
            out.writeframesraw(
                struct.pack(
                    "<hh",
                    int(7000 * ((i % 97) / 48 - 1)),
                    int(5000 * ((i % 131) / 65 - 1)),
                )
            )
        out.writeframes(b"")


def token(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and TOKEN.fullmatch(value.strip()) else None


def safe(value: object, fallback: str = "The validation could not be completed.") -> str:
    if not isinstance(value, str):
        return fallback
    text = CONTROL.sub(" ", value)
    text = BEARER.sub("[redacted]", text)
    text = SECRET.sub("[redacted]", text)
    text = URL.sub("[redacted]", text)
    text = WINPATH.sub("[redacted]", text)
    text = POSIXPATH.sub("[redacted]", text)
    text = " ".join(text.split())
    return (text[:239] + "…") if len(text) > 240 else (text or fallback)


def failure(mode: str, exc: DoctorError) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "mode": mode,
        "state": exc.state if exc.state in {"runtime_missing", "unavailable"} else "unavailable",
        "message": exc.message,
        "modelDownloadPerformed": False,
        "audioRemainsLocal": True,
    }


def summary(result: Mapping[str, Any]) -> str:
    state = result.get("state")
    if state == "ready":
        return "PopEx separation check: runtime and verified model are ready."
    if state == "download_required":
        return "PopEx separation check: explicit validation consent is required before the first model download."
    if state == "validated":
        return "PopEx synthetic four-stem validation passed; temporary data was removed."
    if state == "runtime_missing":
        return "PopEx separation doctor: the optional runtime is missing."
    return "PopEx separation doctor: " + safe(result.get("message"), "validation unavailable")


def from_args(a: argparse.Namespace) -> Config:
    return Config(
        Path(a.worker),
        Path(a.runtime_lock),
        Path(a.cache_root),
        a.expected_profile,
        a.device,
        float(a.timeout_seconds),
        Path(a.temporary_root) if getattr(a, "temporary_root", None) else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    mode = "check"
    try:
        a = parser().parse_args(list(argv) if argv is not None else None)
        mode = a.mode
        if mode == "validate" and a.allow_model_download is not True:
            raise DoctorError("unavailable", "Explicit --allow-model-download consent is required.")
        privacy(mode == "check")
        c = from_args(a)
        result = check(c) if mode == "check" else validate(c, a.allow_model_download)
        code = 0
    except DoctorError as exc:
        result, code = failure(mode, exc), exc.code
    except KeyboardInterrupt:
        result, code = failure(mode, DoctorError("unavailable", "Validation was cancelled.")), 130
    except Exception:
        result, code = failure(mode, DoctorError("unavailable", "Validation failed unexpectedly.")), 2
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    print(summary(result), file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
