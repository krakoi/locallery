import hashlib
import json
import subprocess
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from locallery.config import Config, read_config
from locallery.db import open_database
from locallery.embedding import normalize
from locallery.indexer import opaque, scan
from locallery.server import create_app
from locallery.service import Service
from locallery.vectors import Vectors
from locallery.videos import (
    VideoSettings,
    cache_exists,
    load_video_cache,
    prepare_video,
    video_fingerprint,
)
from PIL import Image
from transformers.models.embedding_gemma2.video_processing_embedding_gemma2 import (
    EmbeddingGemma2VideoProcessor,
)
from transformers.video_utils import VideoMetadata


class VideoEmbedder:
    def __init__(self, config=None):
        self.fingerprint = "video-test"
        self.calls = []

    def load(self):
        pass

    def embed(self, query=None, image=None, video=None):
        self.calls.append((query, image, video))
        vector = np.zeros(768, dtype=np.float32)
        if video:
            frames, metadata = load_video_cache(video)
            assert frames.shape[0] == len(metadata.frames_indices)
            vector[:3] = frames.mean(axis=(0, 1, 2)) / 255
        elif image:
            with Image.open(image) as source:
                vector[:3] = np.array(source.convert("RGB").resize((1, 1)))[0, 0] / 255
        if query:
            vector[0] += 1
        return normalize(vector)


@pytest.fixture
def video_library(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    file = source / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x96:r=10:d=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(file),
        ],
        check=True,
    )
    config = Config(source, tmp_path / "data")
    db = open_database(config)
    yield config, db, file
    db.close()


def test_reference_uniform_sampling_for_hour_long_video():
    metadata = VideoMetadata(total_num_frames=108000, fps=30, duration=3600)
    processor = EmbeddingGemma2VideoProcessor()
    indices = processor.sample_frames(
        metadata, fps=1, max_frames=32, overflow_strategy="uniform"
    )
    assert len(indices) == 32
    assert indices[0] == 0 and indices[-1] / 30 == 3599
    assert np.all(np.diff(indices) > 0)
    truncated = processor.sample_frames(
        metadata, fps=1, max_frames=32, overflow_strategy="truncate"
    )
    assert truncated[-1] / 30 == 31


def test_cached_frames_duration_poster_and_read_only_source(video_library):
    config, db, source = video_library
    before = source.read_bytes(), source.stat().st_mtime_ns
    poster, width, height, duration, manifest = prepare_video(
        source, "video", config.storage, VideoSettings()
    )
    assert (width, height) == (160, 96)
    assert duration == 2
    assert cache_exists(manifest)
    frames, metadata = load_video_cache(manifest)
    assert frames.shape == (2, 96, 160, 3)
    assert metadata.timestamps == [0, 1]
    assert metadata.total_num_frames == 20 and metadata.fps == 10
    with Image.open(poster) as preview:
        assert preview.format == "JPEG"
    assert before == (source.read_bytes(), source.stat().st_mtime_ns)


def test_video_incremental_cache_reuse_and_video_only_invalidation(video_library):
    config, db, source = video_library
    Image.new("RGB", (20, 10), "blue").save(config.library / "photo.png")
    embedder = VideoEmbedder()
    progress = []
    scan(config, db, embedder, progress.append)
    assert progress[-1]["indexed"] == 2 and progress[-1]["failed"] == 0
    assert len(embedder.calls) == 2
    scan(config, db, embedder, progress.append)
    assert progress[-1]["indexed"] == 0 and len(embedder.calls) == 2
    duplicate = config.library / "copy.mp4"
    duplicate.write_bytes(source.read_bytes())
    scan(config, db, embedder, progress.append)
    assert len(embedder.calls) == 2
    changed = replace(config, video=VideoSettings(max_frames=1))
    scan(changed, db, embedder, progress.append)
    assert progress[-1]["indexed"] == 1
    assert len(embedder.calls) == 3 and embedder.calls[-1][2]
    # Missing sampled data rebuilds once for both content-identical videos.
    manifest = db.execute(
        "SELECT video_manifest FROM assets WHERE media_type='video' AND fingerprint=?",
        (video_fingerprint(embedder.fingerprint, changed.video),),
    ).fetchone()[0]
    from pathlib import Path

    data = json.loads(Path(manifest).read_text())
    (Path(manifest).parent / data["frames"][0]).unlink()
    scan(changed, db, embedder, progress.append)
    assert progress[-1]["indexed"] == 1 and len(embedder.calls) == 4
    duplicate.unlink()
    source.unlink()
    scan(changed, db, embedder, progress.append)
    assert db.execute("SELECT count(*) FROM images").fetchone()[0] == 1


def test_video_refinement_uses_full_video_not_poster(video_library):
    config, db, source = video_library
    embedder = VideoEmbedder()
    scan(config, db, embedder, lambda progress: None)
    service = Service(config, lambda progress: None)
    service.db, service.embedder, service.vectors = db, embedder, Vectors(db)
    service.vectors.rebuild()
    service.progress["stage"] = "ready"
    reference = opaque("image:clip.mp4")
    assert service.get_image(reference)["mediaType"] == "video"
    assert service.get_image(reference)["duration"] == 2
    assert service.search({"referenceImageId": reference})["total"] == 0
    service.search({"referenceImageId": reference, "query": "outside"})
    assert embedder.calls[-1][0] == "outside"
    assert embedder.calls[-1][1] is None and embedder.calls[-1][2].endswith(
        "frames.json"
    )


def test_corrupt_video_is_reported_and_retried(video_library):
    config, db, source = video_library
    invalid = config.library / "broken.mov"
    invalid.write_bytes(b"broken")
    embedder = VideoEmbedder()
    progress = []
    scan(config, db, embedder, progress.append)
    assert progress[-1]["failed"] == 1
    invalid.write_bytes(source.read_bytes())
    scan(config, db, embedder, progress.append)
    assert progress[-1]["failed"] == 0
    assert len(embedder.calls) == 1


def test_video_yaml_validation(tmp_path):
    local = tmp_path / ".locallery"
    local.mkdir()
    file = local / "config.yml"
    for value in (
        "max_frames: 0",
        "max_frames: 49",
        "fps: 0",
        "overflow_strategy: wrong",
        "add_timestamps: 1",
    ):
        file.write_text("video:\n  " + value + "\n")
        with pytest.raises(ValueError, match="video"):
            read_config(tmp_path, tmp_path / "global")
    file.write_text("video:\n  fps: 0.5\n  max_frames: 16\n  add_timestamps: false\n")
    settings = read_config(tmp_path, tmp_path / "global").video
    assert settings == VideoSettings(fps=0.5, max_frames=16, add_timestamps=False)


def test_video_original_supports_byte_range_requests(video_library):
    import time

    config, _, source = video_library
    app = create_app(config, VideoEmbedder)
    with TestClient(app) as client:
        deadline = time.monotonic() + 15
        while client.get("/api/status").json()["busy"] and time.monotonic() < deadline:
            time.sleep(0.02)
        assert client.get("/api/status").json()["stage"] == "ready"
        image = client.get("/api/images").json()["items"][0]
        response = client.get(
            f"/api/images/{image['id']}/original", headers={"Range": "bytes=0-99"}
        )
        assert response.status_code == 206
        assert response.headers["content-type"] == "video/mp4"
        assert response.content == source.read_bytes()[:100]
        assert response.headers["accept-ranges"] == "bytes"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == db_hash(config)


def db_hash(config):
    db = open_database(config)
    try:
        return db.execute("SELECT hash FROM images").fetchone()[0]
    finally:
        db.close()


def test_video_processor_receives_original_metadata_without_resampling(video_library):
    from types import SimpleNamespace

    import torch
    from locallery.embedding import Embedder

    config, db, source = video_library
    _, _, _, _, manifest = prepare_video(
        source, "prompt-test", config.storage, VideoSettings()
    )
    recorded = {}

    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            recorded["messages"] = messages
            assert kwargs["tokenize"] is False
            return "<|video|>task: search result | query: red"

        def __call__(self, **kwargs):
            recorded.update(kwargs)
            return Inputs(
                input_ids=torch.ones((1, 2), dtype=torch.int64),
                attention_mask=torch.ones((1, 2)),
            )

    class Model:
        def __call__(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.ones((1, 2, 768)))

    embedder = Embedder(config)
    embedder.model, embedder.processor, embedder.device = Model(), Processor(), "cpu"
    vector = embedder.embed(query="red", video=manifest)
    assert np.isclose(np.linalg.norm(vector), 1)
    assert recorded["do_sample_frames"] is False
    assert recorded["add_timestamps"] is True
    assert recorded["videos"][0].shape[0] == 2
    assert recorded["video_metadata"][0].timestamps == [0, 1]
    assert recorded["messages"][0]["content"][0] == {"type": "video"}
    assert recorded["text"].startswith("<|video|>")
