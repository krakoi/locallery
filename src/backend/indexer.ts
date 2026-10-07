import type { Database } from 'bun:sqlite';
import { readdir, stat, access } from 'node:fs/promises';
import { join, extname, basename, posix } from 'node:path';
import type { Config } from './config';
import { fingerprint } from './config';
import { supported, makePreview } from './images';
import { embed } from './embedding';
import { vectorBlob } from './db';
import type { Progress } from '../shared/types';

export const initialProgress = (): Progress => ({
  busy: true,
  stage: 'starting',
  discovered: 0,
  processed: 0,
  total: 0,
  unchanged: 0,
  indexed: 0,
  skipped: 0,
  failed: 0,
  rate: 0,
  etaSeconds: null,
  message: 'Opening library',
  errors: [],
  startedAt: Date.now(),
});

export function opaque(path: string) {
  return new Bun.CryptoHasher('sha256').update(path).digest('hex').slice(0, 24);
}

export async function scan(
  config: Config,
  db: Database,
  report: (progress: Progress) => void,
): Promise<boolean> {
  const p = initialProgress();
  p.stage = 'discovering';

  let complete = true;
  const files: string[] = [];
  const seen = new Set<string>();
  const seenFolders = new Set(['root']);

  const fp = fingerprint(config);

  db.query(
    "UPDATE images SET status='pending' WHERE asset_id IN (SELECT id FROM assets WHERE fingerprint<>?)",
  ).run(fp);

  const emit = () => report({ ...p, errors: [...p.errors] });

  const error = (path: string, e: unknown) => {
    p.failed++;

    if (p.errors.length < 30) {
      p.errors.push({
        path,
        message: e instanceof Error ? e.message : String(e),
      });
    }
  };

  const folder = db.query('INSERT OR REPLACE INTO folders VALUES (?,?,?,?)');

  async function walk(relative: string, parent: string | null) {
    const id = relative ? opaque('folder:' + relative) : 'root';
    seenFolders.add(id);

    folder.run(id, relative, parent, relative ? basename(relative) : 'Library');

    try {
      const entries = await readdir(join(config.library.path, relative), {
        withFileTypes: true,
      });

      for (const entry of entries) {
        if (entry.name === '.locallery' || entry.isSymbolicLink()) {
          p.skipped++;
          continue;
        }
        const path = relative ? relative + '/' + entry.name : entry.name;

        if (entry.isDirectory()) {
          await walk(path, id);
        } else if (entry.isFile()) {
          if (supported.has(extname(entry.name).toLowerCase())) {
            files.push(path);
            seen.add(path);
            p.discovered++;
          } else {
            p.skipped++;
          }
        }

        if ((p.discovered + p.skipped) % 100 === 0) {
          emit();
        }
      }
    } catch (e) {
      complete = false;
      error(relative || config.library.path, e);
    }
  }

  emit();
  await walk('', null);

  p.total = files.length;
  p.stage = 'processing';
  p.message = 'Checking files and generating image embeddings';
  emit();

  const processingStart = Date.now();

  const find = db.query<
    { asset_id: string | null; status: string; size: number; mtime: string },
    [string]
  >('SELECT * FROM images WHERE path=?');
  const asset = db.query<{ fingerprint: string; cache: string }, [string]>(
    'SELECT * FROM assets WHERE id=?',
  );
  const save = db.query(
    `INSERT INTO images(id,path,folder_id,size,mtime,hash,status,error,asset_id) VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET size=excluded.size,mtime=excluded.mtime,hash=excluded.hash,status=excluded.status,error=excluded.error,asset_id=excluded.asset_id`,
  );

  let next = 0;
  const assetJobs = new Map<string, Promise<boolean>>();

  async function processFile(path: string) {
    let size = 0;
    let mtime = '';
    let hash: string | null = null;

    try {
      const source = join(config.library.path, path);
      const s = await stat(source, { bigint: true });
      size = Number(s.size);
      mtime = s.mtimeNs.toString();

      const old = find.get(path);
      const previous = old?.asset_id ? asset.get(old.asset_id) : null;

      if (
        old?.status === 'ready' &&
        old.size === size &&
        old.mtime === mtime &&
        previous?.fingerprint === fp &&
        (await Bun.file(previous.cache).exists())
      ) {
        p.unchanged++;
        return;
      }

      const hasher = new Bun.CryptoHasher('sha256');
      const reader = Bun.file(source).stream().getReader();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            break;
          }
          hasher.update(value);
        }
      } finally {
        reader.releaseLock();
      }

      hash = hasher.digest('hex');
      const assetId = hash + '-' + fp;

      let job = assetJobs.get(assetId);
      const owner = !job;

      if (!job) {
        job = (async () => {
          let cached = asset.get(assetId);
          if (cached) {
            try {
              await access(cached.cache);
            } catch {
              cached = null;
            }
          }
          if (cached) {
            return false;
          }

          const preview = await makePreview(source, assetId, config);
          const vector = await embed(config, undefined, preview.cache);

          db.query('INSERT OR REPLACE INTO assets VALUES (?,?,?,?,?,?,?)').run(
            assetId,
            hash,
            fp,
            preview.cache,
            preview.width,
            preview.height,
            vectorBlob(vector),
          );
          return true;
        })();
        assetJobs.set(assetId, job);
      }

      try {
        const created = await job;
        if (owner && created) {
          p.indexed++;
        } else {
          p.unchanged++;
        }
      } finally {
        if (owner) {
          assetJobs.delete(assetId);
        }
      }

      const parent = posix.dirname(path);
      save.run(
        opaque('image:' + path),
        path,
        parent === '.' ? 'root' : opaque('folder:' + parent),
        size,
        mtime,
        hash,
        'ready',
        null,
        assetId,
      );
    } catch (e) {
      error(path, e);

      const parent = posix.dirname(path);
      save.run(
        opaque('image:' + path),
        path,
        parent === '.' ? 'root' : opaque('folder:' + parent),
        size,
        mtime,
        hash,
        'error',
        e instanceof Error ? e.message : String(e),
        null,
      );
    } finally {
      p.processed++;
      const elapsed = (Date.now() - processingStart) / 1000;
      p.rate = elapsed ? p.processed / elapsed : 0;
      p.etaSeconds = p.rate ? (p.total - p.processed) / p.rate : null;
      emit();
    }
  }

  await Promise.all(
    Array.from({ length: config.embedding.concurrency }, async () => {
      while (next < files.length) {
        await processFile(files[next++]);
      }
    }),
  );

  if (complete) {
    const remove = db.query('DELETE FROM images WHERE path=?');
    const removeFolder = db.query('DELETE FROM folders WHERE id=?');

    db.transaction(() => {
      for (const row of db.query('SELECT path FROM images').all() as {
        path: string;
      }[]) {
        if (!seen.has(row.path)) {
          remove.run(row.path);
        }
      }

      for (const row of db.query('SELECT id FROM folders').all() as {
        id: string;
      }[]) {
        if (!seenFolders.has(row.id)) {
          removeFolder.run(row.id);
        }
      }
    })();
  }

  p.message = complete
    ? 'Scan finished'
    : 'Scan incomplete; previous records preserved';
  emit();
  return complete;
}
