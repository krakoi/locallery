"""Synthetic native-index benchmark; invoke explicitly, never on startup."""

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from .vectors import new_index


def main():
    maximum = int(sys.argv[1]) if len(sys.argv) > 1 else 500000
    if maximum < 1:
        raise ValueError("Vector count must be positive")
    rng = np.random.default_rng(42)
    index = new_index()
    queries = rng.standard_normal((10, 768), dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    exact = [[] for _ in queries]
    results = []
    count, started = 0, time.perf_counter()
    for target in sorted({min(maximum, n) for n in (10000, 50000, maximum)}):
        while count < target:
            size = min(1024, target - count)
            vectors = rng.standard_normal((size, 768), dtype=np.float32)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            keys = np.arange(count, count + size, dtype=np.uint64)
            index.add(keys, vectors, threads=4)
            scores = vectors @ queries.T
            for q in range(len(queries)):
                candidates = np.argsort(-scores[:, q])[:10]
                exact[q].extend((float(scores[n, q]), int(keys[n])) for n in candidates)
                exact[q] = sorted(exact[q], reverse=True)[:10]
            count += size
        times, recalls = [], []
        for q, vector in enumerate(queries):
            before = time.perf_counter()
            matches = index.search(vector, 10, threads=1)
            times.append((time.perf_counter() - before) * 1000)
            recalls.append(
                len(set(map(int, matches.keys)) & {key for _, key in exact[q]}) / 10
            )
        result = dict(
            vectors=count,
            constructionSeconds=time.perf_counter() - started,
            searchMedianMs=float(np.median(times)),
            rssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            recallAt10=float(np.mean(recalls)),
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    directory = Path(".benchmark-results")
    directory.mkdir(exist_ok=True)
    (directory / "python-latest.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
