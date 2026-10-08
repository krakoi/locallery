"""Bounded-memory exact ranking and scoped native approximate indexes."""

import numpy as np
from usearch.index import Index

from .db import scope_clause, scope_path, vector_batches, vector_from


def new_index(dimensions=768):
    return Index(
        ndim=dimensions,
        metric="cos",
        dtype="f16",
        connectivity=16,
        expansion_add=128,
        expansion_search=128,
    )


def add_rows(index, rows):
    index.add(
        np.array([r["key"] for r in rows], dtype=np.uint64),
        np.stack([vector_from(r["vector"]) for r in rows]),
        threads=4,
    )


class Vectors:
    def __init__(self, db):
        self.db = db
        self.global_index = new_index()
        self.scoped = None

    def rebuild(self, progress=lambda count: None):
        self.global_index = new_index()
        self.scoped = None
        count = 0
        for rows in vector_batches(self.db):
            add_rows(self.global_index, rows)
            count += len(rows)
            progress(count)

    def search(self, vector, folder_id=None, excluded_key=None, limit=500):
        path = scope_path(self.db, folder_id)
        clause, args = scope_clause(path)
        count = self.db.execute(
            f"SELECT count(*) FROM images i WHERE i.status='ready'{clause}", args
        ).fetchone()[0]
        best = []
        if count <= 10000:
            for rows in vector_batches(self.db, path):
                scores = np.stack([vector_from(row["vector"]) for row in rows]) @ vector
                best.extend(
                    (row["key"], float(score))
                    for row, score in zip(rows, scores)
                    if row["key"] != excluded_key
                )
        else:
            index = self.global_index
            if path:
                if self.scoped is None or self.scoped[0] != path:
                    scoped = new_index()
                    for rows in vector_batches(self.db, path):
                        add_rows(scoped, rows)
                    self.scoped = (path, scoped)
                index = self.scoped[1]
            matches = index.search(vector, min(count, limit * 2 + 1), threads=1)
            for key in matches.keys:
                key = int(key)
                if key != excluded_key:
                    row = self.db.execute(
                        "SELECT a.vector FROM images i JOIN assets a ON a.id=i.asset_id WHERE i.key=? AND i.status='ready'",
                        (key,),
                    ).fetchone()
                    if row:
                        best.append((key, float(vector_from(row["vector"]) @ vector)))
        return sorted(best, key=lambda pair: (-pair[1], pair[0]))[:limit]
