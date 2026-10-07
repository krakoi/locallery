import { mkdir } from 'node:fs/promises';
import { newIndex, dot } from '../src/backend/vectors';
import { normalize } from '../src/backend/embedding';

const target = Number(process.argv[2] || 500000);
if (!Number.isInteger(target) || target < 1000) {
  throw new Error('Pass an image count >= 1000');
}

const index = newIndex();
const dimensions = 768;

function vector(key: number) {
  const v = new Float32Array(dimensions);
  let seed = key + 1;

  for (let d = 0; d < dimensions; d++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    v[d] = (seed / 4294967296 - 0.5) * 0.02;
  }

  v[key % 64] += 1;
  return normalize(v);
}

const measurements: unknown[] = [];
const started = performance.now();

for (let offset = 0; offset < target; offset += 1000) {
  const n = Math.min(1000, target - offset);
  const keys = new BigUint64Array(n);
  const values = new Float32Array(n * dimensions);

  for (let i = 0; i < n; i++) {
    keys[i] = BigInt(offset + i);
    values.set(vector(offset + i), i * dimensions);
  }

  index.add(keys, values, 4);
  const count = offset + n;

  if (count % 10000 === 0) {
    console.log(
      `${count.toLocaleString()} / ${target.toLocaleString()} vectors · ${((performance.now() - started) / 1000).toFixed(1)}s · RSS ${(process.memoryUsage().rss / 1048576).toFixed(0)} MiB`,
    );
  }

  if ([10000, 50000, target].includes(count)) {
    const q = vector(Math.floor(count * 0.37));
    const times: number[] = [];
    let searchKeys: bigint[] = [];

    for (let j = 0; j < 10; j++) {
      const start = performance.now();
      searchKeys = [...index.search(q, 10, 1).keys];
      times.push(performance.now() - start);
    }

    const exactStart = performance.now();
    let best: { key: number; score: number }[] = [];

    for (let i = 0; i < count; i++) {
      const score = dot(q, vector(i));

      if (best.length < 10 || score > best[best.length - 1].score) {
        best.push({ key: i, score });
        best.sort((a, b) => b.score - a.score);
        best = best.slice(0, 10);
      }
    }

    const expected = new Set(best.map((b) => BigInt(b.key)));
    const recall = searchKeys.filter((k) => expected.has(k)).length / 10;

    const measurement = {
      count,
      buildSeconds: (performance.now() - started) / 1000,
      searchP50ms: times.sort((a, b) => a - b)[5],
      exactMs: performance.now() - exactStart,
      recallAt10: recall,
      rssMiB: process.memoryUsage().rss / 1048576,
    };

    measurements.push(measurement);
    console.log(JSON.stringify(measurement));
  }
}

await mkdir('.benchmark-results', { recursive: true });
await Bun.write(
  '.benchmark-results/latest.json',
  JSON.stringify(
    {
      date: new Date().toISOString(),
      dimensions,
      distribution: '64 clusters with deterministic low-amplitude noise',
      measurements,
    },
    null,
    2,
  ),
);
