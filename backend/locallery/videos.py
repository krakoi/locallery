"""Bounded frame sampling with official Gemma selection and FFmpeg seeking."""

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from .images import make_preview

SUPPORTED_VIDEOS = {
    ".mp4",
    ".m4v",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".mpeg",
    ".mpg",
    ".mts",
    ".m2ts",
    ".3gp",
}
VIDEO_PREPROCESS = "ffmpeg-seek-official-sampling-jpegli1280-v1"


@dataclass(frozen=True)
class VideoSettings:
    fps: float = 1
    max_frames: int = 32
    overflow_strategy: str = "uniform"
    add_timestamps: bool = True


def video_fingerprint(image_fingerprint, settings):
    return hashlib.sha256(
        json.dumps(
            [image_fingerprint, VIDEO_PREPROCESS, asdict(settings)], sort_keys=True
        ).encode()
    ).hexdigest()


def probe_video(source):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
            "-of",
            "json",
            str(source),
        ],
        capture_output=True,
        timeout=60,
    )
    if result.returncode:
        raise ValueError(
            f"ffprobe failed: {result.stderr.decode(errors='replace').strip()}"
        )
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError("Video contains no video stream")
    stream = data["streams"][0]
    rate = stream.get("avg_frame_rate", "0/0")
    try:
        fps = float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        fps = float(Fraction(stream.get("r_frame_rate", "0/1")))
    duration = 0.0
    for value in (stream.get("duration"), data.get("format", {}).get("duration")):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0:
            duration = parsed
            break
    if (
        not math.isfinite(fps)
        or fps <= 0
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError(
            "Video needs a valid frame rate and duration for sampling and timestamps"
        )
    total = (
        int(stream["nb_frames"])
        if stream.get("nb_frames", "N/A") != "N/A"
        else max(1, round(duration * fps))
    )
    if total <= 0:
        raise ValueError("Video contains no frames")
    return dict(
        total_num_frames=total,
        fps=fps,
        duration=duration,
        height=stream["height"],
        width=stream["width"],
        video_backend="ffmpeg",
    )


def prepare_video(
    source: Path,
    asset_id: str,
    storage: Path,
    settings: VideoSettings,
    report=lambda message: None,
):
    from transformers.models.embedding_gemma2.video_processing_embedding_gemma2 import (
        EmbeddingGemma2VideoProcessor,
    )
    from transformers.video_utils import VideoMetadata

    metadata = probe_video(source)
    # Use the reference processor's index selection rather than reproducing its policy.
    indices = EmbeddingGemma2VideoProcessor().sample_frames(
        VideoMetadata(**metadata),
        fps=settings.fps,
        max_frames=settings.max_frames,
        overflow_strategy=settings.overflow_strategy,
    )
    directory = storage / "videos" / asset_id
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for number, index in enumerate(indices):
        report(f"Sampling video frame {number + 1}/{len(indices)} · {source.name}")
        timestamp = float(index) / metadata["fps"]
        # Input-side seek decodes a short GOP near each target instead of the whole clip.
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-ss",
                str(timestamp),
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-vf",
                "scale=w='min(1280,iw*sar)':h='min(1280,ih)':force_original_aspect_ratio=decrease,setsar=1",
                "-f",
                "image2pipe",
                "-c:v",
                "png",
                "pipe:1",
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode or not result.stdout:
            raise ValueError(
                f"Could not extract video frame at {timestamp:.3f}s: {result.stderr.decode(errors='replace').strip()}"
            )
        temporary = directory / "frame.png"
        try:
            temporary.write_bytes(result.stdout)
            cache, width, height = make_preview(
                temporary, f"{asset_id}-{number:03}", storage
            )
            target = directory / f"{number:03}.jpg"
            Path(cache).replace(target)
            frames.append(target.name)
        finally:
            temporary.unlink(missing_ok=True)
    metadata.update(
        frames_indices=[int(index) for index in indices], width=width, height=height
    )
    manifest = directory / "frames.json"
    manifest.write_text(json.dumps(dict(metadata=metadata, frames=frames)))
    poster = directory / frames[len(frames) // 2]
    return str(poster), width, height, metadata["duration"], str(manifest)


def cache_exists(manifest):
    try:
        file = Path(manifest)
        data = json.loads(file.read_text())
        frames = data["frames"]
        return (
            bool(frames)
            and len(frames) <= 48
            and all((file.parent / name).is_file() for name in frames)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def load_video_cache(manifest):
    from transformers.video_utils import VideoMetadata

    file = Path(manifest)
    data = json.loads(file.read_text())
    frames = []
    for name in data["frames"]:
        path = (file.parent / name).resolve()
        if not path.is_relative_to(file.parent.resolve()):
            raise ValueError("Frame is outside the video cache")
        with Image.open(path) as image:
            frames.append(np.array(image.convert("RGB")))
    return np.stack(frames), VideoMetadata(**data["metadata"])
