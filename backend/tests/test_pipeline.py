"""Batching, bounded preparation, failure isolation and unchanged cache checks."""

import threading
from dataclasses import replace

import numpy as np
import pytest
from locallery.cancellation import ScanCancelled
from locallery.config import Config, read_config
from locallery.db import open_database, vector_from
from locallery.embedding import Embedder, normalize
from locallery.images import make_preview
from locallery.pipeline import ImagePipeline, batch_size_for
from PIL import Image


class BatchEmbedder:
    fingerprint = "batch-test"
    device = "cpu"

    def __init__(self):
        self.batches = []
        self.threads = []
        self.cleared = 0

    def embed_images(self, paths):
        self.batches.append(list(paths))
        self.threads.append(threading.get_ident())
        vectors = []
        for path in paths:
            with Image.open(path) as image:
                vector = np.zeros(768, dtype=np.float32)
                vector[:3] = np.array(image.convert("RGB").resize((1, 1)))[0, 0] / 255
            vectors.append(normalize(vector))
        return np.stack(vectors)

    def release_device_cache(self):
        self.cleared += 1


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = Config(source, tmp_path / "data", batch_size=3)
    db = open_database(config)
    yield config, db, BatchEmbedder()
    db.close()


def files(config, count):
    paths = []
    for number in range(count):
        path = f"{number}.png"
        Image.new("RGB", (20 + number, 10 + number), (20 + number * 12, 80, 180)).save(
            config.library / path
        )
        paths.append(path)
    return paths


def pipeline(setup, config=None, check=lambda: None):
    base, db, embedder = setup
    results = []
    runner = ImagePipeline(
        config or base, db, embedder, lambda *args: results.append(args), check
    )
    return runner, results


def test_full_partial_batches_order_and_bounded_window(setup):
    config, db, embedder = setup
    paths = files(config, 7)
    runner, results = pipeline(setup)
    runner.run(paths)
    assert [len(batch) for batch in embedder.batches] == [3, 3, 1]
    assert runner.metrics["max_pending_files"] <= runner.window == 6
    assert runner.metrics["max_preparation_tasks"] <= runner.window
    assert set(embedder.threads) == {threading.get_ident()}
    assert sum(kind == "indexed" for kind, _, _ in results) == 7
    for row in db.execute(
        "SELECT i.path,a.vector,a.cache FROM images i JOIN assets a ON a.id=i.asset_id"
    ):
        vector = vector_from(row["vector"])
        with Image.open(row["cache"]) as cached:
            color = np.array(cached.resize((1, 1)))[0, 0] / 255
        assert np.allclose(vector[:3], color / np.linalg.norm(color))
    restarted, results = pipeline(
        setup, replace(config, batch_size=1, preparation_workers=4)
    )
    restarted.run(paths)
    assert all(kind == "unchanged" for kind, _, _ in results)
    assert len(embedder.batches) == 3
    assert restarted.metrics["max_preparation_tasks"] == 0


def test_duplicate_assets_share_preview_across_windows(setup, monkeypatch):
    import locallery.pipeline as module

    config, db, embedder = setup
    paths = files(config, 8)
    (config.library / "duplicate.png").write_bytes(
        (config.library / paths[0]).read_bytes()
    )
    paths.insert(1, "duplicate.png")
    (config.library / "late.png").write_bytes((config.library / paths[0]).read_bytes())
    paths.append("late.png")
    previews = []

    def prepare(*args):
        previews.append(args[1])
        return make_preview(*args)

    monkeypatch.setattr(module, "make_preview", prepare)
    runner, results = pipeline(setup)
    runner.run(paths)
    assert len(previews) == len(set(previews)) == 8
    assert sum(map(len, embedder.batches)) == 8
    assert sum(kind == "unchanged" for kind, _, _ in results) == 2
    assert db.execute("SELECT count(*) FROM images").fetchone()[0] == 10


def test_preparation_overlaps_inference_and_uses_other_threads(setup, monkeypatch):
    import locallery.pipeline as module

    config, _, embedder = setup
    config = replace(config, batch_size=2)
    paths = files(config, 8)
    started, prepared = threading.Event(), threading.Event()
    lock = threading.Lock()
    calls, threads = [], []

    def prepare(*args):
        with lock:
            calls.append(args[0])
            number = len(calls)
            threads.append(threading.get_ident())
        if number > 2:
            assert started.wait(2)
        result = make_preview(*args)
        if number > 2:
            prepared.set()
        return result

    original = embedder.embed_images

    def infer(paths):
        started.set()
        assert prepared.wait(2), "Preparation did not run during model inference"
        return original(paths)

    monkeypatch.setattr(module, "make_preview", prepare)
    monkeypatch.setattr(embedder, "embed_images", infer)
    runner, _ = pipeline(setup, config)
    runner.run(paths)
    assert threading.get_ident() not in threads
    assert runner.metrics["max_pending_files"] <= 4


@pytest.mark.parametrize("failure", ["exception", "invalid", "shape"])
def test_batch_failures_isolate_one_image(setup, monkeypatch, failure):
    config, db, embedder = setup
    paths = files(config, 5)
    original = embedder.embed_images
    bad = [None]

    def infer(paths):
        if bad[0] is None:
            bad[0] = paths[0]
        vectors = original(paths)
        if bad[0] in paths:
            if failure == "exception":
                raise ValueError("one bad image")
            if failure == "shape":
                return vectors[:, :10]
            vectors[paths.index(bad[0]), 0] = np.nan
        return vectors

    monkeypatch.setattr(embedder, "embed_images", infer)
    runner, results = pipeline(setup)
    runner.run(paths)
    assert (
        db.execute("SELECT count(*) FROM images WHERE status='ready'").fetchone()[0]
        == 4
    )
    assert sum(kind == "failed" for kind, _, _ in results) == 1


@pytest.mark.parametrize("backend", ["torch", "mps"])
def test_oom_halves_batch_size_for_remainder(setup, monkeypatch, backend):
    import torch

    config, db, embedder = setup
    config = replace(config, batch_size=4)
    paths = files(config, 12)
    calls = []
    original = embedder.embed_images

    def infer(paths):
        calls.append(len(paths))
        if len(paths) > 2:
            if backend == "mps":
                raise RuntimeError("MPS backend out of memory (simulated)")
            raise torch.OutOfMemoryError("simulated allocation failure")
        return original(paths)

    monkeypatch.setattr(embedder, "embed_images", infer)
    runner, _ = pipeline(setup, config)
    runner.run(paths)
    assert calls[0] == 4 and max(calls[1:]) == 2
    assert runner.batch_size == 2 and embedder.cleared == 1
    assert (
        db.execute("SELECT count(*) FROM images WHERE status='ready'").fetchone()[0]
        == 12
    )


def test_singleton_oom_records_error_and_continues(setup, monkeypatch):
    import torch

    config, db, embedder = setup
    original = embedder.embed_images

    def infer(paths):
        if len(embedder.batches) == 0:
            embedder.batches.append(list(paths))
            raise torch.OutOfMemoryError("simulated singleton failure")
        return original(paths)

    monkeypatch.setattr(embedder, "embed_images", infer)
    runner, results = pipeline(setup, replace(config, batch_size=1))
    runner.run(files(config, 3))
    assert sum(kind == "failed" for kind, _, _ in results) == 1
    assert (
        db.execute("SELECT count(*) FROM images WHERE status='ready'").fetchone()[0]
        == 2
    )


def test_cancelled_inference_is_not_retried_or_recorded_as_failure(setup, monkeypatch):
    _, db, embedder = setup
    calls = []

    def cancelled(paths):
        calls.append(paths)
        raise ScanCancelled()

    monkeypatch.setattr(embedder, "embed_images", cancelled)
    runner, results = pipeline(setup)
    with pytest.raises(ScanCancelled):
        runner.run(files(setup[0], 5))
    assert len(calls) == 1
    assert not any(kind == "failed" for kind, _, _ in results)
    assert db.execute("SELECT count(*) FROM images").fetchone()[0] == 0


def test_corrupt_preview_shared_failure_and_retry(setup):
    config, db, embedder = setup
    paths = files(config, 1)
    (config.library / "bad.png").write_bytes(b"bad image")
    (config.library / "copy.png").write_bytes(b"bad image")
    runner, results = pipeline(setup)
    runner.run(paths + ["bad.png", "copy.png"])
    assert sum(kind == "failed" for kind, _, _ in results) == 2
    assert sum(map(len, embedder.batches)) == 1
    (config.library / "bad.png").write_bytes((config.library / paths[0]).read_bytes())
    (config.library / "copy.png").write_bytes((config.library / paths[0]).read_bytes())
    runner, _ = pipeline(setup)
    runner.run(["bad.png", "copy.png"])
    assert (
        db.execute("SELECT count(*) FROM images WHERE status='ready'").fetchone()[0]
        == 3
    )
    assert sum(map(len, embedder.batches)) == 1


def test_failed_encoding_preserves_cache_and_cleans_temporary_files(setup, monkeypatch):
    config, _, _ = setup
    source = config.library / files(config, 1)[0]
    cache = config.storage / "previews" / "asset.jpg"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"previous cache")

    def fail(*args, **kwargs):
        raise ValueError("encoder failed")

    monkeypatch.setattr("locallery.images.run_command", fail)
    with pytest.raises(ValueError, match="encoder failed"):
        make_preview(source, "asset", config.storage)
    assert cache.read_bytes() == b"previous cache"
    assert list(cache.parent.iterdir()) == [cache]


def test_batched_processor_padding_float32_pooling_and_order():
    from types import SimpleNamespace

    import torch

    recorded = []

    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            recorded.append((messages, kwargs))
            return Inputs(
                input_ids=torch.ones((2, 3), dtype=torch.int64),
                attention_mask=torch.tensor([[1, 1, 0], [1, 0, 0]]),
            )

    class Model:
        def __call__(self, **kwargs):
            tokens = torch.zeros((2, 3, 768), dtype=torch.bfloat16)
            tokens[0, 0, 0], tokens[0, 1, 1], tokens[0, 2, 2] = 2, 2, 100
            tokens[1, 0, 3], tokens[1, 1:, 4] = 3, 100
            return SimpleNamespace(last_hidden_state=tokens)

    embedder = Embedder(None)
    embedder.processor, embedder.model, embedder.device = Processor(), Model(), "cpu"
    result = embedder.embed_images(["wide.jpg", "tall.jpg"])
    assert result.shape == (2, 768) and result.dtype == np.float32
    assert result[0, 0] == result[0, 1] and result[0, 2] == 0
    assert result[1, 3] == 1 and result[1, 4] == 0
    messages, kwargs = recorded[0]
    assert kwargs["processor_kwargs"] == {"padding": True}
    assert "padding" not in kwargs
    assert len(messages) == 2 and all(len(chat) == 1 for chat in messages)
    assert messages[1][0]["content"] == [{"type": "image", "url": "tall.jpg"}]
    assert embedder.embed_images([]).shape == (0, 768)


@pytest.mark.parametrize(
    "device,expected", [("cpu", 1), ("mps", 1), ("cuda", 4), ("cuda:1", 4)]
)
def test_device_aware_batch_default(tmp_path, device, expected):
    embedder = BatchEmbedder()
    embedder.device = device
    config = Config(tmp_path, tmp_path / "data")
    assert batch_size_for(config, embedder) == expected
    assert batch_size_for(replace(config, batch_size=8), embedder) == 8


@pytest.mark.parametrize(
    "setting,value",
    [
        ("batch_size", True),
        ("batch_size", False),
        ("batch_size", 0),
        ("batch_size", 65),
        ("batch_size", 1.5),
        ("batch_size", "4"),
        ("batch_size", None),
        ("preparation_workers", True),
        ("preparation_workers", 0),
        ("preparation_workers", 17),
        ("preparation_workers", 1.5),
        ("preparation_workers", "2"),
    ],
)
def test_invalid_pipeline_settings(tmp_path, setting, value):
    import yaml

    global_dir = tmp_path / "global"
    global_dir.mkdir()
    section = "embedding" if setting == "batch_size" else "indexing"
    (global_dir / "config.yml").write_text(yaml.safe_dump({section: {setting: value}}))
    with pytest.raises(ValueError, match=setting):
        read_config(tmp_path, global_dir)


def test_pipeline_settings_defaults_local_overrides(tmp_path):
    config = read_config(tmp_path, tmp_path / "global")
    assert config.batch_size == "auto" and config.preparation_workers == 2
    (tmp_path / ".locallery" / "config.yaml").write_text(
        "embedding:\n  batch_size: 8\nindexing:\n  preparation_workers: 4\n"
    )
    config = read_config(tmp_path, tmp_path / "global")
    assert config.batch_size == 8 and config.preparation_workers == 4


def test_interrupted_scan_preserves_records_and_reconciles_on_restart(setup):
    from locallery.indexer import scan

    config, db, embedder = setup
    old = files(config, 1)[0]
    runner, _ = pipeline(setup)
    runner.run([old])
    (config.library / old).unlink()
    new_paths = files(config, 6)[1:]
    (config.library / "0.png").unlink()
    stopped = threading.Event()

    def check():
        if stopped.is_set():
            raise ScanCancelled()

    def report(progress):
        if progress["processed"] >= 1:
            stopped.set()

    with pytest.raises(ScanCancelled):
        scan(config, db, embedder, report, check)
    assert (
        db.execute("SELECT count(*) FROM images WHERE path=?", (old,)).fetchone()[0]
        == 1
    )
    assert (
        db.execute("SELECT count(*) FROM images WHERE status='ready'").fetchone()[0]
        >= 2
    )
    assert scan(config, db, embedder, lambda progress: None)
    assert db.execute("SELECT count(*) FROM images").fetchone()[0] == len(new_paths)
    assert (
        db.execute("SELECT count(*) FROM images WHERE path=?", (old,)).fetchone()[0]
        == 0
    )
