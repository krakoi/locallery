"""Bounded image preparation; only the coordinator owns SQLite and inference."""

import hashlib
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import numpy as np

from .cancellation import ScanCancelled
from .embedding import normalize
from .images import make_preview


def opaque(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def batch_size_for(config, embedder):
    if config.batch_size != "auto":
        return config.batch_size
    device = getattr(embedder, "device", config.device)
    return 4 if device.startswith("cuda") else 1


def is_device_oom(error):
    import torch

    return isinstance(error, torch.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and str(error).startswith(
            ("CUDA out of memory", "MPS backend out of memory", "HIP out of memory")
        )
    )


def hash_file(source, check_running):
    check_running()
    if source.is_symlink():
        raise ValueError("Source changed to a symlink; rescan")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            check_running()
            digest.update(chunk)
    check_running()
    return digest.hexdigest()


def timed(function, *args):
    started = time.monotonic()
    result = function(*args)
    return result, time.monotonic() - started


@dataclass
class File:
    path: str
    size: int = 0
    mtime: str = ""
    digest: str | None = None

    @property
    def folder_id(self):
        parent = str(PurePosixPath(self.path).parent)
        return "root" if parent == "." else opaque("folder:" + parent)


@dataclass
class Asset:
    id: str
    files: list[File] = field(default_factory=list)
    preview: tuple | None = None


class ImagePipeline:
    def __init__(self, config, db, embedder, report, check_running=lambda: None):
        self.config, self.db, self.embedder = config, db, embedder
        self.report, self.check_running = report, check_running
        self.batch_size = batch_size_for(config, embedder)
        self.window = 2 * max(config.preparation_workers, self.batch_size)
        self.pending_files = 0
        self.futures, self.assets, self.failures = {}, {}, {}
        self.ready = []
        self.metrics = dict(
            hash_seconds=0.0,
            preparation_seconds=0.0,
            inference_seconds=0.0,
            max_pending_files=0,
            max_preparation_tasks=0,
        )

    def submit(self, pool, stage, item, function, *args):
        future = pool.submit(timed, function, *args)
        self.futures[future] = (stage, item)
        self.metrics["max_preparation_tasks"] = max(
            self.metrics["max_preparation_tasks"], len(self.futures)
        )

    def finish(self, file, kind, asset_id=None, error=None, pending=True):
        self.check_running()
        if error is None:
            self.db.execute(
                """INSERT INTO images(id,path,folder_id,size,mtime,hash,status,error,asset_id)
                VALUES (?,?,?,?,?,?,'ready',NULL,?) ON CONFLICT(path) DO UPDATE SET
                folder_id=excluded.folder_id,size=excluded.size,mtime=excluded.mtime,
                hash=excluded.hash,status='ready',error=NULL,asset_id=excluded.asset_id""",
                (
                    opaque("image:" + file.path),
                    file.path,
                    file.folder_id,
                    file.size,
                    file.mtime,
                    file.digest,
                    asset_id,
                ),
            )
        else:
            self.db.execute(
                """INSERT INTO images(id,path,folder_id,size,mtime,hash,status,error,asset_id)
                VALUES (?,?,?,?,?,?,'error',?,NULL) ON CONFLICT(path) DO UPDATE SET
                folder_id=excluded.folder_id,size=excluded.size,mtime=excluded.mtime,
                hash=excluded.hash,status='error',error=excluded.error,asset_id=NULL""",
                (
                    opaque("image:" + file.path),
                    file.path,
                    file.folder_id,
                    file.size,
                    file.mtime,
                    file.digest,
                    str(error),
                ),
            )
        if pending:
            self.pending_files -= 1
        self.report(kind, file.path, error)

    def admit(self, pool, path):
        self.check_running()
        file = File(path)
        pending = False
        try:
            source = self.config.library / path
            stat = source.lstat()
            if source.is_symlink():
                raise ValueError("Source changed to a symlink; rescan")
            file.size, file.mtime = stat.st_size, str(stat.st_mtime_ns)
            old = self.db.execute(
                "SELECT i.*,a.fingerprint,a.cache FROM images i LEFT JOIN assets a ON a.id=i.asset_id WHERE i.path=?",
                (path,),
            ).fetchone()
            if (
                old
                and old["status"] == "ready"
                and old["size"] == file.size
                and old["mtime"] == file.mtime
                and old["fingerprint"] == self.embedder.fingerprint
                and os.path.isfile(old["cache"])
            ):
                # Avoid rewriting metadata and preserve existing IDs on unchanged scans.
                self.report("unchanged", path, None)
                return
            self.pending_files += 1
            pending = True
            self.metrics["max_pending_files"] = max(
                self.metrics["max_pending_files"], self.pending_files
            )
            self.submit(pool, "hash", file, hash_file, source, self.check_running)
        except ScanCancelled:
            raise
        except Exception as error:
            self.finish(file, "failed", error=error, pending=pending)

    def failed_asset(self, asset, error):
        self.failures[asset.id] = str(error)
        for file in asset.files:
            self.finish(file, "failed", error=error)
        self.assets.pop(asset.id)

    def collect(self, pool, future):
        stage, item = self.futures.pop(future)
        try:
            result, elapsed = future.result()
        except ScanCancelled:
            raise
        except Exception as error:
            if stage == "hash":
                self.finish(item, "failed", error=error)
            else:
                self.failed_asset(item, error)
            return
        self.check_running()
        if stage == "preview":
            self.metrics["preparation_seconds"] += elapsed
            item.preview = result
            self.ready.append(item)
            return
        self.metrics["hash_seconds"] += elapsed
        item.digest = result
        asset_id = result + "-" + self.embedder.fingerprint
        cached = self.db.execute(
            "SELECT cache FROM assets WHERE id=?", (asset_id,)
        ).fetchone()
        if cached and os.path.isfile(cached["cache"]):
            self.finish(item, "unchanged", asset_id)
        elif asset_id in self.failures:
            self.finish(item, "failed", error=self.failures[asset_id])
        elif asset_id in self.assets:
            self.assets[asset_id].files.append(item)
        else:
            asset = Asset(asset_id, [item])
            self.assets[asset_id] = asset
            self.submit(
                pool,
                "preview",
                asset,
                make_preview,
                self.config.library / item.path,
                asset_id,
                self.config.storage,
                self.check_running,
            )

    def infer(self, assets):
        self.check_running()
        error_message, oom = None, False
        started = time.monotonic()
        try:
            vectors = np.asarray(
                self.embedder.embed_images([asset.preview[0] for asset in assets]),
                dtype=np.float32,
            )
            if vectors.shape != (len(assets), 768):
                raise ValueError("Image batch returned an unexpected embedding shape")
            vectors = np.stack([normalize(vector) for vector in vectors])
        except ScanCancelled:
            raise
        except Exception as error:
            error_message = str(error)
            oom = is_device_oom(error)
        finally:
            self.metrics["inference_seconds"] += time.monotonic() - started
        # Leave the except block before clearing cache: its traceback owns failed tensors.
        self.check_running()
        if error_message is not None:
            if oom:
                self.batch_size = min(self.batch_size, max(1, len(assets) // 2))
                release = getattr(self.embedder, "release_device_cache", None)
                if release:
                    release()
            if len(assets) == 1:
                self.failed_asset(assets[0], error_message)
            else:
                split = min(self.batch_size, max(1, len(assets) // 2))
                for offset in range(0, len(assets), split):
                    self.infer(assets[offset : offset + split])
            return
        for asset, vector in zip(assets, vectors):
            self.check_running()
            cache, width, height = asset.preview
            self.db.execute(
                "INSERT OR REPLACE INTO assets(id,hash,fingerprint,cache,width,height,vector,media_type,duration,video_manifest) VALUES (?,?,?,?,?,?,?,'image',NULL,NULL)",
                (
                    asset.id,
                    asset.files[0].digest,
                    self.embedder.fingerprint,
                    cache,
                    width,
                    height,
                    vector.astype("<f4").tobytes(),
                ),
            )
            for index, file in enumerate(asset.files):
                self.finish(file, "indexed" if index == 0 else "unchanged", asset.id)
            self.assets.pop(asset.id)

    def run(self, paths):
        paths, exhausted = iter(paths), False
        pool = ThreadPoolExecutor(
            max_workers=self.config.preparation_workers,
            thread_name_prefix="locallery-prepare",
        )
        try:
            while not exhausted or self.futures or self.ready:
                self.check_running()
                while not exhausted and self.pending_files < self.window:
                    self.check_running()
                    path = next(paths, None)
                    if path is None:
                        exhausted = True
                        break
                    self.admit(pool, path)
                # Harvest completed preparation before deciding whether a batch is full.
                for future in list(self.futures):
                    if future.done():
                        self.collect(pool, future)
                if len(self.ready) >= self.batch_size or (
                    self.ready and not self.futures
                ):
                    assets = self.ready[: self.batch_size]
                    del self.ready[: len(assets)]
                    self.report(
                        "batch",
                        f"Embedding {len(assets)} images (batch limit {self.batch_size})",
                        None,
                    )
                    self.infer(assets)
                elif self.futures:
                    self.report(
                        "preparing",
                        f"Preparing images · {self.pending_files} pending",
                        None,
                    )
                    done, _ = wait(
                        self.futures, timeout=0.1, return_when=FIRST_COMPLETED
                    )
                    for future in done:
                        self.collect(pool, future)
            assert self.pending_files == 0
        finally:
            for future in self.futures:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
