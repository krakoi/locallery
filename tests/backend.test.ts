import { beforeAll, afterAll, test, expect } from 'bun:test';
import { mkdir, writeFile, unlink, rm, stat } from 'node:fs/promises';
import { join } from 'node:path';
import sharp from 'sharp';
import type { Config } from '../src/backend/config';
import { readConfig, fingerprint } from '../src/backend/config';
import { openDatabase, scopePath, vectorFrom } from '../src/backend/db';
import { scan, opaque } from '../src/backend/indexer';
import { embed, normalize } from '../src/backend/embedding';
import { Vectors } from '../src/backend/vectors';
import { generateGroups } from '../src/backend/groups';
import type { Progress } from '../src/shared/types';

const directory = join(
  process.cwd(),
  '.test-artifacts',
  'backend-' + crypto.randomUUID(),
);

let config: Config,
  db: ReturnType<typeof openDatabase>,
  server: ReturnType<typeof Bun.serve>;
const requests: any[] = [];
let bad = false,
  offline = false;
const progress: Progress[] = [];

async function image(path: string, color: string, width = 32, height = 24) {
  await mkdir(join(config.library.path, path, '..'), { recursive: true });
  await sharp({ create: { width, height, channels: 3, background: color } })
    .png()
    .toFile(join(config.library.path, path));
}

async function rescan() {
  progress.length = 0;
  return scan(config, db, (p) => progress.push(p));
}

beforeAll(async () => {
  await mkdir(directory, { recursive: true });
  server = Bun.serve({
    port: 0,
    hostname: '127.0.0.1',
    async fetch(req) {
      if (offline) {
        return new Response('offline', { status: 503 });
      }
      const body = (await req.json()) as any;
      requests.push(body);
      if (bad) {
        return Response.json({ data: [{ embedding: [0] }] });
      }
      const content = body.input[0].content;
      const v = new Float32Array(768);
      const media = content.find((p: any) => p.type === 'image_url');
      if (media) {
        const b = Buffer.from(media.image_url.url.split(',')[1], 'base64');
        const { data } = await sharp(b)
          .resize(1, 1)
          .raw()
          .toBuffer({ resolveWithObject: true });
        v[0] = data[0] / 255;
        v[1] = data[1] / 255;
        v[2] = data[2] / 255;
      }
      const query = content.find((p: any) => p.type === 'text')?.text || '';
      if (query.includes('blue')) {
        v[2] += 2;
      }
      if (query.includes('red')) {
        v[0] += 2;
      }
      if (!v.some((x) => x)) {
        v[0] = 1;
      }
      return Response.json({ data: [{ embedding: Array.from(v) }] });
    },
  });
  config = {
    library: { path: join(directory, 'source') },
    storage: { path: join(directory, 'data') },
    server: { host: '127.0.0.1', port: 3000 },
    embedding: {
      base_url: `http://127.0.0.1:${server.port}/v1`,
      model: 'embeddinggemma-2',
      revision: 'test',
      concurrency: 2,
      timeout_seconds: 2,
    },
  };
  await mkdir(config.library.path, { recursive: true });
  db = openDatabase(config);
});

afterAll(async () => {
  db.close();
  server.stop(true);
  await rm(directory, { recursive: true, force: true });
});

test('validates configuration and protects input folder', async () => {
  const yaml = join(directory, 'config.yaml');
  await writeFile(
    yaml,
    'library:\n  path: ./source\nstorage:\n  path: ./source/cache\n',
  );
  expect(() => readConfig(yaml)).toThrow('outside');
  await writeFile(
    yaml,
    'library:\n  path: ./source\nstorage:\n  path: ./data\nserver:\n  port: nope\n',
  );
  expect(() => readConfig(yaml)).toThrow('server.port');
});
test('incremental scan encodes cjpegli previews and preserves source bytes', async () => {
  await image('trip/red.png', 'red', 2000, 1000);
  await image('trip/sub/blue.png', 'blue');
  await image('trip-other/green.png', 'green');
  const before = await Bun.file(
    join(config.library.path, 'trip/red.png'),
  ).arrayBuffer();
  const beforeStat = await stat(join(config.library.path, 'trip/red.png'));
  expect(await rescan()).toBe(true);
  expect(requests.length).toBe(3);
  expect(
    db.query("SELECT COUNT(*) AS n FROM images WHERE status='ready'").get(),
  ).toEqual({ n: 3 });
  const preview = db
    .query(
      "SELECT a.cache,a.width,a.height FROM assets a JOIN images i ON i.asset_id=a.id WHERE i.path='trip/red.png'",
    )
    .get() as any;
  expect(preview.width).toBe(1280);
  expect(preview.height).toBe(640);
  expect((await sharp(preview.cache).metadata()).format).toBe('jpeg');
  expect(
    Buffer.from(
      await Bun.file(join(config.library.path, 'trip/red.png')).arrayBuffer(),
    ),
  ).toEqual(Buffer.from(before));
  expect((await stat(join(config.library.path, 'trip/red.png'))).mtimeMs).toBe(
    beforeStat.mtimeMs,
  );
  const calls = requests.length;
  await rescan();
  expect(requests.length).toBe(calls);
  expect(progress.at(-1)?.unchanged).toBe(3);
  expect(progress.at(-1)?.processed).toBe(3);
});
test('changed, new, deleted and duplicate files reconcile without redundant inference', async () => {
  const calls = requests.length;
  await image('trip/red.png', 'blue', 2000, 1000);
  await image('new/purple.png', 'purple');
  await mkdir(join(config.library.path, 'copy'), { recursive: true });
  await writeFile(
    join(config.library.path, 'copy/purple.png'),
    new Uint8Array(
      await Bun.file(join(config.library.path, 'new/purple.png')).arrayBuffer(),
    ),
  );
  await unlink(join(config.library.path, 'trip-other/green.png'));
  await rescan();
  expect(requests.length - calls).toBe(2);
  expect(
    db.query("SELECT COUNT(*) AS n FROM images WHERE status='ready'").get(),
  ).toEqual({ n: 4 });
  expect(
    db.query("SELECT * FROM images WHERE path='trip-other/green.png'").get(),
  ).toBeNull();
});
test('exact scoped ranking includes descendants and excludes sibling prefixes', async () => {
  const vectors = new Vectors(db);
  await vectors.rebuild();
  const q = await embed(config, 'blue');
  const result = vectors.search(q, opaque('folder:trip'));
  const keys = result.map((r) => r.key);
  const paths = keys.map(
    (k) => (db.query('SELECT path FROM images WHERE key=?').get(k) as any).path,
  );
  expect(paths.sort()).toEqual(['trip/red.png', 'trip/sub/blue.png']);
  expect(() => scopePath(db, 'missing')).toThrow();
  const reference = db
    .query(
      "SELECT i.key,a.vector,a.cache FROM images i JOIN assets a ON a.id=i.asset_id WHERE i.path='trip/red.png'",
    )
    .get() as any;
  expect(
    vectors
      .search(vectorFrom(reference.vector), 'root', reference.key)
      .some((m) => m.key === reference.key),
  ).toBe(false);
  await embed(config, 'red', reference.cache);
  const parts = requests.at(-1).input[0].content;
  expect(parts[0].text).toBe('task: search result | query: red');
  expect(parts[1].image_url.url.startsWith('data:image/jpeg;base64,')).toBe(
    true,
  );
});
test('invalid embeddings, outage and unreadable images retry on the next scan', async () => {
  await image('broken/retry.png', 'orange');
  bad = true;
  await rescan();
  expect(progress.at(-1)?.failed).toBe(1);
  bad = false;
  await rescan();
  expect(progress.at(-1)?.failed).toBe(0);
  await image('broken/offline.png', 'yellow');
  offline = true;
  await rescan();
  expect(progress.at(-1)?.failed).toBe(1);
  offline = false;
  await rescan();
  expect(progress.at(-1)?.failed).toBe(0);
  await writeFile(join(config.library.path, 'broken/bad.png'), 'not an image');
  await rescan();
  expect(progress.at(-1)?.failed).toBe(1);
  await unlink(join(config.library.path, 'broken/bad.png'));
});
test('missing source root preserves indexed records', async () => {
  const original = config.library.path;
  config.library.path = join(directory, 'not-mounted');
  const before = (db.query('SELECT COUNT(*) AS n FROM images').get() as any).n;
  expect(await rescan()).toBe(false);
  expect((db.query('SELECT COUNT(*) AS n FROM images').get() as any).n).toBe(
    before,
  );
  config.library.path = original;
});
test('revision changes invalidate embeddings, global grouping persists assignments', async () => {
  config.embedding.revision = 'test-2';
  const before = requests.length;
  await rescan();
  expect(requests.length).toBeGreaterThan(before);
  const old = db
    .query(
      'SELECT COUNT(*) AS n FROM images i JOIN assets a ON a.id=i.asset_id WHERE a.fingerprint<>? AND i.status=?',
    )
    .get(fingerprint(config), 'ready') as any;
  expect(old.n).toBe(0);
  await generateGroups(db, () => {});
  const assignments = db
    .query(
      "SELECT COUNT(*) AS n FROM images WHERE status='ready' AND group_id IS NOT NULL",
    )
    .get() as any;
  const count = db
    .query("SELECT COUNT(*) AS n FROM images WHERE status='ready'")
    .get() as any;
  expect(assignments.n).toBe(count.n);
  const first = db.query('SELECT * FROM groups ORDER BY id').all();
  await generateGroups(db, () => {});
  expect(db.query('SELECT * FROM groups ORDER BY id').all()).toEqual(first);
});
test('normalization rejects invalid vector shapes, values and zero vectors', () => {
  expect(() => normalize([1, 2])).toThrow();
  expect(() => normalize(new Float32Array(768))).toThrow();
  const invalid = new Float32Array(768);
  invalid[0] = NaN;
  expect(() => normalize(invalid)).toThrow();
});

test('descendant scope handles emoji folder names using SQLite character lengths', async () => {
  await image('📷 holiday/sub/red.png', 'red');
  await rescan();
  const vectors = new Vectors(db);
  await vectors.rebuild();
  const results = vectors.search(
    await embed(config, 'red'),
    opaque('folder:📷 holiday'),
  );
  expect(results.length).toBe(1);
  const row = db
    .query('SELECT path FROM images WHERE key=?')
    .get(results[0].key) as any;
  expect(row.path).toBe('📷 holiday/sub/red.png');
});
