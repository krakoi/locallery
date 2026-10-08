"""Global defaults and working-directory library configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .videos import VideoSettings

PREPROCESS = "pillow-jpegli-q90-srgb-white-oriented-fit1280-v1"
DEFAULTS = {
    "server": {"host": "127.0.0.1", "port": 3000},
    "embedding": {
        "model": "google/embeddinggemma-2",
        "device": "auto",
        "dtype": "auto",
    },
    "video": {
        "fps": 1,
        "max_frames": 32,
        "overflow_strategy": "uniform",
        "add_timestamps": True,
    },
}


@dataclass(frozen=True)
class Config:
    library: Path
    storage: Path
    host: str = "127.0.0.1"
    port: int = 3000
    model: str = "google/embeddinggemma-2"
    revision: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    video: VideoSettings = field(default_factory=VideoSettings)


def inside(root: Path, path: Path) -> bool:
    return path.is_relative_to(root)


def mapping(value, name):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def directory(path: Path):
    if path.is_symlink():
        raise ValueError(f"Application directory must not be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)


def load(path: Path):
    return mapping(yaml.safe_load(path.read_text()), str(path)) if path.exists() else {}


def read_config(cwd: Path | None = None, global_directory: Path | None = None):
    cwd = (cwd or Path.cwd()).resolve()
    global_directory = (
        global_directory
        or Path(os.environ.get("LOCALLERY_HOME", Path.home() / ".locallery"))
    ).absolute()
    directory(global_directory)
    global_file = global_directory / "config.yml"
    if not global_file.exists():
        try:
            with global_file.open("x") as file:
                yaml.safe_dump(DEFAULTS, file, sort_keys=False)
        except FileExistsError:
            pass
    local_dir = cwd / ".locallery"
    directory(local_dir)
    directory(local_dir / "data")
    global_raw, local_raw = load(global_file), load(local_dir / "config.yml")
    for raw in (global_raw, local_raw):
        if "storage" in raw:
            raise ValueError("storage is automatic; remove storage from config.yml")
        for section in ("library", "server", "embedding", "video"):
            mapping(raw.get(section), section)
    server = (
        DEFAULTS["server"]
        | mapping(global_raw.get("server"), "server")
        | mapping(local_raw.get("server"), "server")
    )
    embedding = (
        DEFAULTS["embedding"]
        | mapping(global_raw.get("embedding"), "embedding")
        | mapping(local_raw.get("embedding"), "embedding")
    )
    # Old llama-server connection settings are harmless during migration.
    for name in ("base_url", "timeout_seconds", "concurrency"):
        if name in embedding:
            print(
                f"Ignoring legacy embedding.{name}; inference now runs in Python",
                flush=True,
            )
    allowed = {
        "model",
        "revision",
        "device",
        "dtype",
        "base_url",
        "timeout_seconds",
        "concurrency",
    }
    if unknown := set(embedding) - allowed:
        raise ValueError(f"Unknown embedding settings: {', '.join(sorted(unknown))}")

    def text(value, name):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
        return value

    host = text(server["host"], "server.host")
    port = server["port"]
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("server.port must be an integer between 1 and 65535")
    library = (
        cwd
        / text(
            mapping(local_raw.get("library"), "library").get("path", "."),
            "library.path",
        )
    ).resolve()
    storage = (local_dir / "data").resolve()
    if inside(storage, library) or library == local_dir.resolve():
        raise ValueError("library.path must be outside application storage")
    model = text(embedding["model"], "embedding.model")
    # Explicit filesystem model paths are relative to the working directory.
    if model.startswith((".", "/", "~")):
        model = str((cwd / Path(model).expanduser()).resolve())
    revision = embedding.get("revision")
    if revision is not None:
        revision = text(revision, "embedding.revision")
    device = text(embedding["device"], "embedding.device")
    if device not in ("auto", "cpu", "cuda", "mps") and not (
        device.startswith("cuda:") and device[5:].isdigit()
    ):
        raise ValueError("embedding.device must be auto, cpu, cuda, cuda:N, or mps")
    dtype = embedding["dtype"]
    if dtype not in ("auto", "float32", "bfloat16"):
        raise ValueError(
            "embedding.dtype must be auto, float32, or bfloat16; float16 is unsafe for this model"
        )
    video = (
        DEFAULTS["video"]
        | mapping(global_raw.get("video"), "video")
        | mapping(local_raw.get("video"), "video")
    )
    if set(video) - set(DEFAULTS["video"]):
        raise ValueError("Unknown video settings")
    fps = video["fps"]
    if isinstance(fps, bool) or not isinstance(fps, (float, int)) or not 0 < fps <= 60:
        raise ValueError("video.fps must be greater than 0 and at most 60")
    frames = video["max_frames"]
    if isinstance(frames, bool) or not isinstance(frames, int) or not 1 <= frames <= 48:
        raise ValueError("video.max_frames must be an integer between 1 and 48")
    if video["overflow_strategy"] not in ("uniform", "truncate"):
        raise ValueError("video.overflow_strategy must be uniform or truncate")
    if not isinstance(video["add_timestamps"], bool):
        raise ValueError("video.add_timestamps must be true or false")
    return Config(
        library,
        storage,
        host,
        port,
        model,
        revision,
        device,
        dtype,
        VideoSettings(**video),
    )
