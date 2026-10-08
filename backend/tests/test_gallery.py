import time
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from locallery.config import Config, read_config
from locallery.db import open_database
from locallery.embedding import Embedder, normalize
from locallery.groups import generate_groups
from locallery.images import make_preview
from locallery.indexer import opaque, scan
from locallery.server import create_app
from locallery.service import Service
from locallery.vectors import Vectors
from PIL import Image


class FakeEmbedder:
    def __init__(self, config=None):
        self.fingerprint = "python-fixture-v1"
        self.calls = []
        self.fail = False

    def load(self):
        pass

    def embed(self, query=None, image=None):
        self.calls.append((query, image))
        if self.fail:
            raise ValueError("Invalid embedding")
        vector = np.zeros(768, dtype=np.float32)
        if image:
            with Image.open(image) as source:
                vector[:3] = (
                    np.asarray(source.convert("RGB").resize((1, 1)))[0, 0] / 255
                )
        if query:
            vector[{"red": 0, "green": 1, "blue": 2}.get(query, 0)] += 2
        return normalize(vector)


@pytest.fixture
def library(tmp_path):
    source, storage = tmp_path / "source", tmp_path / "data"
    source.mkdir()
    config = Config(source, storage)
    db = open_database(config)
    embedder = FakeEmbedder()
    yield config, db, embedder
    db.close()


def image(config, path, color="red", size=(32, 24)):
    file = config.library / path
    file.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(file)
    return file


def rescan(library):
    config, db, embedder = library
    progress = []
    complete = scan(config, db, embedder, progress.append)
    return complete, progress[-1]


def test_configuration_bootstrap_and_legacy_settings(tmp_path):
    global_dir = tmp_path / "global"
    config = read_config(tmp_path, global_dir)
    assert config.library == tmp_path
    assert config.storage == tmp_path / ".locallery/data"
    assert (global_dir / "config.yml").exists()
    assert not (tmp_path / ".locallery/config.yml").exists()
    local = tmp_path / ".locallery/config.yml"
    local.write_text(
        "library:\n  path: photos\nembedding:\n  base_url: http://127.0.0.1:4096/v1\n  device: cpu\n"
    )
    assert read_config(tmp_path, global_dir).library == tmp_path / "photos"
    local.write_text("embedding:\n  dtype: float16\n")
    with pytest.raises(ValueError, match="float16"):
        read_config(tmp_path, global_dir)
    local.write_text("library:\n  path: .locallery/data\n")
    with pytest.raises(ValueError, match="outside"):
        read_config(tmp_path, global_dir)


def test_symlinked_app_directory_rejected(tmp_path):
    (tmp_path / "source").mkdir()
    (tmp_path / ".locallery").symlink_to(tmp_path / "source", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        read_config(tmp_path, tmp_path / "global")


def test_incremental_source_preservation_and_restart(library):
    config, db, embedder = library
    source = image(config, "trip/red.png", size=(2000, 1000))
    before = source.read_bytes(), source.stat().st_mtime_ns
    complete, progress = rescan(library)
    assert complete and progress["indexed"] == 1
    row = db.execute("SELECT * FROM assets").fetchone()
    assert (row["width"], row["height"]) == (1280, 640)
    with Image.open(row["cache"]) as preview:
        assert preview.format == "JPEG"
    assert before == (source.read_bytes(), source.stat().st_mtime_ns)
    assert rescan(library)[1]["unchanged"] == 1
    # A fresh connection and fresh worker retain the same cache identity.
    restarted = open_database(config)
    try:
        fresh = FakeEmbedder()
        scan(config, restarted, fresh, lambda progress: None)
        assert fresh.calls == []
    finally:
        restarted.close()


def test_changes_duplicates_removals_and_interrupted_scan(library):
    config, db, embedder = library
    original = image(config, "red.png")
    rescan(library)
    image(config, "red.png", "blue")
    duplicate = config.library / "copy.png"
    duplicate.write_bytes(original.read_bytes())
    assert rescan(library)[1]["indexed"] == 1
    assert len(embedder.calls) == 2
    duplicate.unlink()
    rescan(library)
    assert db.execute("SELECT count(*) FROM images").fetchone()[0] == 1
    db.execute("UPDATE images SET status='pending'")
    assert rescan(library)[1]["unchanged"] == 1
    assert len(embedder.calls) == 2


def test_unreadable_failure_retry_and_unavailable_root(library):
    config, db, embedder = library
    image(config, "red.png")
    embedder.fail = True
    assert rescan(library)[1]["failed"] == 1
    embedder.fail = False
    assert rescan(library)[1]["failed"] == 0
    (config.library / "bad.png").write_text("not an image")
    assert rescan(library)[1]["failed"] == 1
    missing = replace(config, library=config.library / "unmounted")
    assert not scan(missing, db, embedder, lambda progress: None)
    assert db.execute("SELECT count(*) FROM images").fetchone()[0] == 2


def test_fingerprint_change_and_exclusions(library):
    config, db, embedder = library
    source = image(config, "red.png")
    hidden = config.library / ".locallery"
    hidden.mkdir()
    (hidden / "ignored.png").write_bytes(source.read_bytes())
    (config.library / "link.png").symlink_to(source)
    assert rescan(library)[1]["indexed"] == 1
    embedder.fingerprint = "python-fixture-v2"
    assert rescan(library)[1]["indexed"] == 1
    assert db.execute("SELECT count(*) FROM images").fetchone()[0] == 1


def test_orientation_and_no_upscale(library):
    config, _, _ = library
    source = config.library / "oriented.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (60, 30), "red").save(source, exif=exif)
    cache, width, height = make_preview(source, "oriented", config.storage)
    assert (width, height) == (30, 60)
    with Image.open(cache) as preview:
        assert 274 not in preview.getexif()


def test_descendant_scope_reference_refinement_and_groups(library):
    config, db, embedder = library
    for path, color in (
        ("trip/red.png", "red"),
        ("trip/sub/blue.png", "blue"),
        ("trip-other/red.png", "red"),
        ("📷/sub/red.png", "red"),
    ):
        image(config, path, color)
    rescan(library)
    vectors = Vectors(db)
    vectors.rebuild()
    matches = vectors.search(embedder.embed(query="red"), opaque("folder:trip"))
    paths = {
        db.execute("SELECT path FROM images WHERE key=?", (key,)).fetchone()[0]
        for key, _ in matches
    }
    assert paths == {"trip/red.png", "trip/sub/blue.png"}
    assert len(vectors.search(embedder.embed(query="red"), opaque("folder:📷"))) == 1
    service = Service(config, lambda progress: None)
    service.db, service.vectors, service.embedder = db, vectors, embedder
    service.progress["stage"] = "ready"
    reference = opaque("image:trip/red.png")
    results = service.search({"referenceImageId": reference, "query": "blue"})
    assert all(item["id"] != reference for item in results["items"])
    assert embedder.calls[-1][0] == "blue" and embedder.calls[-1][1].endswith(".jpg")
    generate_groups(db, lambda done, total: None)
    assert (
        db.execute("SELECT count(*) FROM images WHERE group_id IS NOT NULL").fetchone()[
            0
        ]
        == 4
    )
    before = [tuple(row) for row in db.execute("SELECT * FROM groups ORDER BY id")]
    generate_groups(db, lambda done, total: None)
    assert before == [
        tuple(row) for row in db.execute("SELECT * FROM groups ORDER BY id")
    ]
    assert service.get_images(1, 2)["total"] == 4
    assert len(service.get_images(2, 2)["items"]) == 2
    assert len(service.get_images(1, 96, opaque("folder:trip"))["items"]) == 1


def test_normalize():
    for vector in ([1, 2], np.zeros(768), np.full(768, np.nan), np.full(768, np.inf)):
        with pytest.raises(ValueError):
            normalize(vector)
    assert np.isclose(np.linalg.norm(normalize(np.ones(768))), 1)


def test_http_blocking_contract_rescan_errors_and_pagination(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = Config(source, tmp_path / "data")
    image(config, "red.png")

    class SlowEmbedder(FakeEmbedder):
        def load(self):
            time.sleep(0.15)

    app = create_app(config, SlowEmbedder)
    with TestClient(app) as client:
        assert client.get("/api/status").json()["busy"]
        assert client.get("/api/images").status_code == 503
        assert client.post("/api/rescan").status_code == 409
        deadline = time.monotonic() + 10
        while client.get("/api/status").json()["busy"] and time.monotonic() < deadline:
            time.sleep(0.02)
        assert client.get("/api/status").json()["stage"] == "ready"
        response = client.get("/api/images?pageSize=1").json()
        assert response["total"] == 1 and response["pageSize"] == 1
        reference = response["items"][0]["id"]
        assert (
            client.get(f"/api/images/{reference}/preview").headers["content-type"]
            == "image/jpeg"
        )
        assert client.get(f"/api/images/{reference}/original").status_code == 200
        assert client.post("/api/search", json={"query": "red"}).json()["total"] == 1
        assert (
            client.post("/api/search", json={"referenceImageId": reference}).json()[
                "total"
            ]
            == 0
        )
        assert client.post("/api/search", content="[]").status_code == 400
        assert client.post("/api/search", content="x" * 10001).status_code == 413
        assert client.get("/api/images?page=0").status_code == 400
        assert client.get("/api/images?folderId=missing").status_code == 404
        assert client.post("/api/rescan").status_code == 202


def test_transformers_request_and_float32_pooling(monkeypatch):
    import torch

    recorded = []

    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            recorded.append(messages)
            return Inputs(
                input_ids=torch.ones((1, 3), dtype=torch.int64),
                attention_mask=torch.tensor([[1, 1, 0]]),
            )

    class Model:
        def __call__(self, **kwargs):
            from types import SimpleNamespace

            tokens = torch.zeros((1, 3, 768))
            tokens[0, 0, 0], tokens[0, 1, 1], tokens[0, 2, 2] = 2, 2, 100
            return SimpleNamespace(last_hidden_state=tokens)

    embedder = Embedder(None)
    embedder.processor, embedder.model, embedder.device = Processor(), Model(), "cpu"
    result = embedder.embed("blue", "/cache/preview.jpg")
    assert np.isclose(result[0], result[1]) and result[2] == 0
    content = recorded[0][0]["content"]
    assert content[0] == {"type": "image", "url": "/cache/preview.jpg"}
    assert content[1]["text"] == "task: search result | query: blue"


def test_scoped_approximate_ranking(library):
    config, db, embedder = library
    # Force the large-folder path while retaining exact reference scores.
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((10020, 768), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    db.execute("INSERT INTO folders VALUES ('scope','scope','root','scope')")
    db.execute("BEGIN")
    for index, vector in enumerate(vectors):
        db.execute(
            "INSERT INTO assets(id,hash,fingerprint,cache,width,height,vector) VALUES (?,?,?,?,?,?,?)",
            (str(index), str(index), "test", "unused", 1, 1, vector.tobytes()),
        )
        path = f"scope/{index}.jpg" if index < 10010 else f"scope-other/{index}.jpg"
        db.execute(
            "INSERT INTO images(id,path,folder_id,size,mtime,status,asset_id) VALUES (?,?,?,1,'1','ready',?)",
            (str(index), path, "scope" if index < 10010 else "root", str(index)),
        )
    db.execute("COMMIT")
    ranking = Vectors(db)
    ranking.rebuild()
    actual = ranking.search(vectors[0], "scope", limit=10)
    exact = set((np.argsort(-(vectors[:10010] @ vectors[0]))[:10] + 1).tolist())
    assert len(exact & {key for key, _ in actual}) >= 8
    assert all(key <= 10010 for key, _ in actual)


def test_model_load_pins_processor_and_weights_to_same_commit(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import torch
    import transformers
    from transformers.utils import hub

    commit = "a" * 40
    calls = []
    model_config = SimpleNamespace(model_type="embedding_gemma2", audio_config={})

    class Model:
        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

    def load_config(model, **kwargs):
        calls.append(("config", kwargs["revision"]))
        return model_config

    def load_processor(model, **kwargs):
        calls.append(("processor", kwargs["revision"]))
        return object()

    def load_model(model, **kwargs):
        calls.append(("model", kwargs["revision"]))
        assert kwargs["dtype"] == torch.float32
        assert kwargs["config"].audio_config is None
        return Model()

    monkeypatch.setattr(
        hub,
        "cached_file",
        lambda *args, **kwargs: f"/cache/snapshots/{commit}/config.json",
    )
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", load_config)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load_model)
    embedder = Embedder(Config(tmp_path, tmp_path / "data", device="cpu"))
    embedder.load()
    assert calls == [("config", commit), ("processor", commit), ("model", commit)]
    assert len(embedder.fingerprint) == 64
    embedder.load()
    assert len(calls) == 3
