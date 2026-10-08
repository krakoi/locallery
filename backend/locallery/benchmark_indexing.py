"""Opt-in real-model image benchmark; sources are read-only, trials isolated."""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from .config import SOURCE_ROOT, Config
from .db import open_database
from .embedding import Embedder
from .images import SUPPORTED, make_preview
from .pipeline import ImagePipeline, is_device_oom


def sample_images(source, limit):
    paths = []
    for root, directories, names in os.walk(source, followlinks=False):
        directories[:] = sorted(
            name
            for name in directories
            if name != ".locallery" and not (Path(root) / name).is_symlink()
        )
        for name in sorted(names):
            file = Path(root) / name
            if file.suffix.lower() in SUPPORTED and not file.is_symlink():
                paths.append(file.relative_to(source).as_posix())
                if len(paths) == limit:
                    return paths
    return paths


def verify_batches(embedder, caches, dtype):
    singles = np.stack([embedder.embed(image=cache) for cache in caches])
    minimum = 1.0
    for batch_size in (1, 2, 4, 8):
        batched = np.concatenate(
            [
                embedder.embed_images(caches[offset : offset + batch_size])
                for offset in range(0, len(caches), batch_size)
            ]
        )
        minimum = min(minimum, float(np.min(np.sum(singles * batched, axis=1))))
    threshold = 0.999 if dtype == "bfloat16" else 0.9999
    if minimum < threshold:
        raise ValueError(f"Batched/single cosine {minimum:.8f} is below {threshold}")
    return {"minimum_cosine": minimum, "threshold": threshold}


def warm_up(embedder, cache, batch_size):
    while True:
        try:
            embedder.embed_images([cache] * batch_size)
            return
        except Exception as error:
            if not is_device_oom(error) or batch_size == 1:
                raise
        # Clear failed tensors' traceback before releasing cached device allocations.
        embedder.release_device_cache()
        batch_size = max(1, batch_size // 2)


def trial(args):
    import torch

    source, storage = args.source.resolve(), args.storage.resolve()
    if storage.is_relative_to(source):
        raise ValueError("Benchmark storage must be outside the source folder")
    config = Config(
        source,
        storage,
        model=args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
        preparation_workers=args.workers,
        batch_size=args.batch,
    )
    paths = sample_images(source, args.limit)
    if not paths:
        raise ValueError("No supported images in benchmark sample")
    embedder = Embedder(config)
    embedder.load()
    cache, _, _ = make_preview(source / paths[0], "warmup", storage / "warmup")
    warm_up(embedder, cache, args.batch)
    if embedder.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    db = open_database(config)
    counts = {"indexed": 0, "unchanged": 0, "failed": 0}

    def report(kind, path, error):
        if kind in counts:
            counts[kind] += 1
        if error:
            print(f"Benchmark item failed: {path}: {error}", flush=True)

    try:
        pipeline = ImagePipeline(config, db, embedder, report)
        started = time.monotonic()
        pipeline.run(paths)
        if embedder.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = dict(
            workers=args.workers,
            batch_size=args.batch,
            effective_batch_size=pipeline.batch_size,
            files=len(paths),
            **counts,
            elapsed_seconds=elapsed,
            files_per_second=len(paths) / elapsed,
            **pipeline.metrics,
            peak_rss_mib=peak_rss / (1024 * 1024 if sys.platform == "darwin" else 1024),
            cuda_peak_mib=torch.cuda.max_memory_allocated() / 1024**2
            if embedder.device.startswith("cuda")
            else None,
            fingerprint=embedder.fingerprint,
        )
        if counts["failed"]:
            raise ValueError("Benchmark scan reported failed files")
        if args.verify:
            caches = [
                row[0] for row in db.execute("SELECT cache FROM assets ORDER BY id")
            ]
            result["equivalence"] = verify_batches(embedder, caches, args.dtype)
        return result
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--model", default="google/embeddinggemma-2")
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare actual batched/single vectors in the first trial",
    )
    parser.add_argument("--trial-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--storage", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--batch", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.limit < 1 or not args.source.is_dir():
        parser.error("Provide an existing source directory and a positive limit")
    if args.cache_dir is not None:
        args.cache_dir = (SOURCE_ROOT / args.cache_dir.expanduser()).resolve()
    if args.trial_worker:
        print(json.dumps(trial(args)), flush=True)
        return
    root = Path(".benchmark-results").resolve()
    run = root / "indexing" / uuid.uuid4().hex[:12]
    if run.is_relative_to(args.source.resolve()):
        parser.error("Run from a directory outside the source collection")
    run.mkdir(parents=True)
    results = []
    for workers in (1, 2, 4):
        for batch in (1, 2, 4, 8):
            print(
                f"Benchmark: {workers} preparation workers, batch {batch}", flush=True
            )
            command = [
                sys.executable,
                "-m",
                "locallery.benchmark_indexing",
                str(args.source.resolve()),
                "--trial-worker",
                "--storage",
                str(run / f"w{workers}-b{batch}"),
                "--workers",
                str(workers),
                "--batch",
                str(batch),
                "--limit",
                str(args.limit),
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--model",
                args.model,
            ]
            if args.revision:
                command += ["--revision", args.revision]
            if args.cache_dir:
                command += ["--cache-dir", str(args.cache_dir)]
            if args.verify and not results:
                command += ["--verify"]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(completed.stdout + completed.stderr)
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            results.append(result)
            print(
                f"  {result['files_per_second']:.2f} files/s · {result['elapsed_seconds']:.2f}s · {result['peak_rss_mib']:.0f} MiB",
                flush=True,
            )
            document = dict(
                device=args.device,
                dtype=args.dtype,
                files=sample_images(args.source.resolve(), args.limit),
                trial_storage=str(run),
                results=results,
            )
            (root / "indexing-latest.json").write_text(
                json.dumps(document, indent=2) + "\n"
            )
    print(f"Saved {root / 'indexing-latest.json'}")


if __name__ == "__main__":
    main()
