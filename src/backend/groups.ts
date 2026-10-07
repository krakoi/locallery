import type { Database } from 'bun:sqlite';
import { normalize } from './embedding';
import { vectorFrom } from './db';
import { dot, newIndex, vectorBatches } from './vectors';

export async function generateGroups(
  db: Database,
  progress: (processed: number, total: number) => void,
) {
  const { count } = db
    .query(`SELECT COUNT(*) AS count FROM images WHERE status='ready'`)
    .get() as { count: number };
  db.exec('DELETE FROM groups; UPDATE images SET group_id=NULL;');
  if (!count) {
    return;
  }

  const sampleRows = db
    .query(
      `SELECT a.vector FROM images i JOIN assets a ON a.id=i.asset_id WHERE i.status='ready' ORDER BY i.id LIMIT 10000`,
    )
    .all() as { vector: Uint8Array }[];
  const sample = sampleRows.map((r) =>
    normalize(vectorFrom(r.vector).subarray(0, 256), 256),
  );
  const k = Math.min(32, Math.max(1, Math.floor(Math.sqrt(count))));
  const centers = Array.from({ length: k }, (_, i) =>
    sample[Math.floor((i * sample.length) / k)].slice(),
  );
  const updates = new Uint32Array(k);

  // Two deterministic streaming passes train bounded-memory spherical centroids.
  for (let pass = 0; pass < 2; pass++) {
    for (let n = 0; n < sample.length; n++) {
      const v = sample[pass === 0 ? n : sample.length - 1 - n];
      let best = 0;
      let bestScore = -Infinity;

      for (let j = 0; j < k; j++) {
        const score = dot(v, centers[j]);

        if (score > bestScore) {
          bestScore = score;
          best = j;
        }
      }

      const rate = 1 / Math.sqrt(++updates[best] + 1);
      const center = centers[best];

      for (let d = 0; d < 256; d++) {
        center[d] = (1 - rate) * center[d] + rate * v[d];
      }

      centers[best] = normalize(center, 256);

      if (n % 1000 === 0) {
        await Bun.sleep(0);
      }
    }
  }

  const index = newIndex(256);
  centers.forEach((c, i) => index.add(BigInt(i), c));
  const counts = new Uint32Array(k);
  const representatives: { key: number; distance: number }[][] = Array.from(
    { length: k },
    () => [],
  );
  const update = db.query('UPDATE images SET group_id=? WHERE key=?');
  let done = 0;

  for (const rows of vectorBatches(db)) {
    const values = new Float32Array(rows.length * 256);

    rows.forEach((r, i) =>
      values.set(
        normalize(vectorFrom(r.vector).subarray(0, 256), 256),
        i * 256,
      ),
    );

    const matches = index.search(values, 1, 1);

    db.transaction(() => {
      for (let i = 0; i < rows.length; i++) {
        const g = Number(matches.keys[i]);
        update.run(g, rows[i].key);
        counts[g]++;

        const reps = representatives[g];
        reps.push({ key: rows[i].key, distance: matches.distances[i] });
        reps.sort((a, b) => a.distance - b.distance || a.key - b.key);

        if (reps.length > 4) {
          reps.pop();
        }
      }
    })();

    done += rows.length;
    progress(done, count);
    await Bun.sleep(0);
  }

  const insert = db.query('INSERT INTO groups VALUES (?,?,?)');
  db.transaction(() => {
    for (let i = 0; i < k; i++) {
      if (counts[i]) {
        insert.run(
          i,
          counts[i],
          JSON.stringify(representatives[i].map((r) => r.key)),
        );
      }
    }
  })();
}
