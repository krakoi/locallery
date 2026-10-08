"""Deterministic spherical clustering, independent of folder navigation."""

import json
import math

import numpy as np

from .db import vector_batches, vector_from
from .embedding import normalize
from .vectors import new_index


def generate_groups(db, report, check_running=lambda: None):
    check_running()
    count = db.execute("SELECT count(*) FROM images WHERE status='ready'").fetchone()[0]
    db.executescript("DELETE FROM groups; UPDATE images SET group_id=NULL;")
    if not count:
        return
    rows = db.execute(
        "SELECT a.vector FROM images i JOIN assets a ON a.id=i.asset_id WHERE i.status='ready' ORDER BY i.id LIMIT 10000"
    ).fetchall()
    sample = np.stack(
        [normalize(vector_from(row["vector"])[:256], 256) for row in rows]
    )
    k = min(32, max(1, math.isqrt(count)))
    centers = sample[np.arange(k) * len(sample) // k].copy()
    updates = np.zeros(k, dtype=np.int64)
    for vectors in (sample, sample[::-1]):
        for vector in vectors:
            check_running()
            group = int(np.argmax(centers @ vector))
            updates[group] += 1
            rate = 1 / math.sqrt(updates[group] + 1)
            centers[group] = normalize((1 - rate) * centers[group] + rate * vector, 256)
    index = new_index(256)
    index.add(np.arange(k, dtype=np.uint64), centers, threads=1)
    counts = [0] * k
    representatives = [[] for _ in range(k)]
    done = 0
    for rows in vector_batches(db):
        check_running()
        values = np.stack(
            [normalize(vector_from(row["vector"])[:256], 256) for row in rows]
        )
        matches = index.search(values, 1, threads=1)
        assignments = []
        for row, group, distance in zip(
            rows,
            np.asarray(matches.keys).reshape(-1),
            np.asarray(matches.distances).reshape(-1),
        ):
            group = int(group)
            assignments.append((group, row["key"]))
            counts[group] += 1
            representatives[group].append((float(distance), row["key"]))
            representatives[group].sort()
            del representatives[group][4:]
        db.execute("BEGIN")
        try:
            db.executemany("UPDATE images SET group_id=? WHERE key=?", assignments)
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
        done += len(rows)
        report(done, count)
    db.executemany(
        "INSERT INTO groups VALUES (?,?,?)",
        [
            (
                group,
                counts[group],
                json.dumps([key for _, key in representatives[group]]),
            )
            for group in range(k)
            if counts[group]
        ],
    )
