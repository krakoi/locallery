import { Index, MetricKind, ScalarKind } from 'usearch';
import type { Database } from 'bun:sqlite';
import { descendantClause, scopePath, vectorFrom } from './db';

export type Match = { key: number; score: number };

export function dot(a: Float32Array, b: Float32Array) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    s += a[i] * b[i];
  }
  return s;
}

export function newIndex(dimensions = 768) {
  return new Index({
    dimensions,
    metric: MetricKind.Cos,
    quantization: ScalarKind.F16,
    connectivity: 16,
    expansion_add: 128,
    expansion_search: 128,
    multi: false,
  });
}

export function* vectorBatches(db: Database, path = '', size = 1024) {
  const scope = descendantClause(path);
  let last = 0;
  while (true) {
    const rows = db
      .query(
        `SELECT i.key,a.vector FROM images i JOIN assets a ON a.id=i.asset_id WHERE i.status='ready' AND i.key>?${scope.sql} ORDER BY i.key LIMIT ?`,
      )
      .all(last, ...scope.args, size) as { key: number; vector: Uint8Array }[];
    if (!rows.length) {
      return;
    }
    yield rows;
    last = rows.at(-1)!.key;
  }
}

export class Vectors {
  private global = newIndex();
  private scoped: { path: string; index: Index } | null = null;
  constructor(private db: Database) {}

  async rebuild(progress?: (count: number) => void) {
    this.global = newIndex();
    this.scoped = null;
    let count = 0;
    for (const rows of vectorBatches(this.db)) {
      this.addRows(this.global, rows);
      count += rows.length;
      progress?.(count);
      await Bun.sleep(0);
    }
  }

  private addRows(index: Index, rows: { key: number; vector: Uint8Array }[]) {
    const keys = new BigUint64Array(rows.length);
    const values = new Float32Array(rows.length * 768);
    for (let i = 0; i < rows.length; i++) {
      keys[i] = BigInt(rows[i].key);
      values.set(vectorFrom(rows[i].vector), i * 768);
    }
    index.add(keys, values, 4);
  }

  search(
    vector: Float32Array,
    folderId?: string,
    excludedKey?: number,
    limit = 500,
  ): Match[] {
    const path = scopePath(this.db, folderId);
    const scope = descendantClause(path);
    const { count } = this.db
      .query(
        `SELECT COUNT(*) AS count FROM images i WHERE i.status='ready'${scope.sql}`,
      )
      .get(...scope.args) as { count: number };

    if (count <= 10000) {
      const best: Match[] = [];

      for (const rows of vectorBatches(this.db, path)) {
        for (const row of rows) {
          if (row.key !== excludedKey) {
            best.push({
              key: row.key,
              score: dot(vector, vectorFrom(row.vector)),
            });
          }
        }
      }

      return best
        .sort((a, b) => b.score - a.score || a.key - b.key)
        .slice(0, limit);
    }

    let index = this.global;

    if (path && count !== this.global.size()) {
      if (this.scoped?.path !== path) {
        const scoped = newIndex();

        for (const rows of vectorBatches(this.db, path)) {
          this.addRows(scoped, rows);
        }

        this.scoped = { path, index: scoped };
      }

      index = this.scoped.index;
    }

    const matches = index.search(vector, Math.min(count, limit * 2 + 1), 1);
    const best: Match[] = [];
    const lookup = this.db.query(
      `SELECT a.vector FROM images i JOIN assets a ON a.id=i.asset_id WHERE i.key=? AND i.status='ready'`,
    );

    for (const key of matches.keys) {
      const number = Number(key);
      if (number === excludedKey) {
        continue;
      }

      const row = lookup.get(number) as { vector: Uint8Array } | null;
      if (row) {
        best.push({ key: number, score: dot(vector, vectorFrom(row.vector)) });
      }
    }

    return best
      .sort((a, b) => b.score - a.score || a.key - b.key)
      .slice(0, limit);
  }
}
