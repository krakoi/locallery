import { join, extname, resolve } from 'node:path';
import { realpath } from 'node:fs/promises';
import { readConfig, inside } from './config';
import { openDatabase, folders, getImage, imageSelect, toImage } from './db';
import { initialProgress } from './indexer';
import type { SearchRequest, SearchResult } from '../shared/types';

const config = readConfig(process.env.LOCALLERY_CONFIG || 'config.yaml');
const db = openDatabase(config);
let progress = initialProgress(),
  lastLog = 0;
const worker = new Worker(new URL('./worker.ts', import.meta.url).href);
const subscribers = new Set<ReadableStreamDefaultController<Uint8Array>>();
const encoder = new TextEncoder();
const pending = new Map<
  string,
  {
    resolve: (result: SearchResult) => void;
    reject: (error: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  }
>();

function publish() {
  const data = encoder.encode(`data: ${JSON.stringify(progress)}\n\n`);
  for (const controller of subscribers) {
    try {
      controller.enqueue(data);
    } catch {
      subscribers.delete(controller);
    }
  }
}

worker.onmessage = (event) => {
  const m = event.data;
  if (m.type === 'progress') {
    progress = m.progress;
    publish();
    if (Date.now() - lastLog > 1000 || !progress.busy) {
      console.log(
        `[${progress.stage}] ${progress.message} | ${progress.processed}/${progress.total} · indexed ${progress.indexed} · unchanged ${progress.unchanged} · skipped ${progress.skipped} · errors ${progress.failed} · ${progress.rate.toFixed(1)}/s${progress.etaSeconds === null ? '' : ` · ETA ${Math.ceil(progress.etaSeconds)}s`}`,
      );
      lastLog = Date.now();
    }
  } else if (m.type === 'result') {
    const p = pending.get(m.id);
    if (p) {
      clearTimeout(p.timer);
      pending.delete(m.id);
      if (m.error) {
        p.reject(new Error(m.error));
      } else {
        p.resolve(m.result);
      }
    }
  }
};
worker.onerror = (event) => {
  progress = {
    ...progress,
    busy: false,
    stage: 'error',
    message: event.message,
  };
  publish();
  for (const p of pending.values()) {
    clearTimeout(p.timer);
    p.reject(new Error('Background worker failed'));
  }
  pending.clear();
};

function rpc(request: SearchRequest) {
  return new Promise<SearchResult>((resolve, reject) => {
    const id = crypto.randomUUID(),
      timer = setTimeout(
        () => {
          pending.delete(id);
          reject(new Error('Search timed out'));
        },
        Math.max(600000, config.embedding.timeout_seconds * 1000 + 120000),
      );
    pending.set(id, { resolve, reject, timer });
    worker.postMessage({ type: 'search', id, request });
  });
}

const json = (value: unknown, status = 200) => Response.json(value, { status });

function pagination(url: URL) {
  const page = Number(url.searchParams.get('page') || 1),
    pageSize = Number(url.searchParams.get('pageSize') || 96);
  if (
    !Number.isInteger(page) ||
    page < 1 ||
    !Number.isInteger(pageSize) ||
    pageSize < 1 ||
    pageSize > 200
  ) {
    throw new Error('Invalid pagination');
  }
  return { page, pageSize };
}

function stream() {
  let controller: ReadableStreamDefaultController<Uint8Array>;
  return new Response(
    new ReadableStream<Uint8Array>({
      start(c) {
        controller = c;
        subscribers.add(c);
        c.enqueue(encoder.encode(`data: ${JSON.stringify(progress)}\n\n`));
      },
      cancel() {
        subscribers.delete(controller);
      },
    }),
    {
      headers: {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
        'x-accel-buffering': 'no',
      },
    },
  );
}

const keepalive = setInterval(() => {
  for (const c of subscribers) {
    try {
      c.enqueue(encoder.encode(': keepalive\n\n'));
    } catch {
      subscribers.delete(c);
    }
  }
}, 15000);

const server = Bun.serve({
  hostname: config.server.host,
  port: config.server.port,
  idleTimeout: 255,
  async fetch(req) {
    const url = new URL(req.url);
    const path = url.pathname;

    try {
      if (path === '/api/status' && req.method === 'GET') {
        return json(progress);
      }
      if (path === '/api/events' && req.method === 'GET') {
        return stream();
      }

      if (path === '/api/rescan' && req.method === 'POST') {
        if (progress.busy || pending.size) {
          return json({ error: 'A job is already running' }, 409);
        }
        progress = { ...initialProgress(), message: 'Starting rescan' };
        publish();
        worker.postMessage({ type: 'rescan' });
        return json({ accepted: true }, 202);
      }

      if (path.startsWith('/api/') && progress.busy) {
        return json({ error: 'Indexing is underway' }, 503);
      }

      if (path === '/api/folders' && req.method === 'GET') {
        return json(folders(db));
      }

      if (path === '/api/images' && req.method === 'GET') {
        const { page, pageSize } = pagination(url);
        let condition = '';
        const args: (string | number)[] = [];

        const folderId = url.searchParams.get('folderId');
        if (folderId) {
          if (!db.query('SELECT id FROM folders WHERE id=?').get(folderId)) {
            return json({ error: 'Folder not found' }, 404);
          }
          condition += ' AND i.folder_id=?';
          args.push(folderId);
        }

        const groupId = url.searchParams.get('groupId');
        if (groupId !== null) {
          const group = Number(groupId);
          if (!Number.isInteger(group) || group < 0) {
            throw new Error('Invalid group');
          }
          condition += ' AND i.group_id=?';
          args.push(group);
        }

        const { total } = db
          .query(
            `SELECT COUNT(*) AS total FROM images i WHERE i.status='ready'${condition}`,
          )
          .get(...args) as { total: number };

        const rows = db
          .query(
            `${imageSelect}${condition} ORDER BY i.path COLLATE NOCASE LIMIT ? OFFSET ?`,
          )
          .all(...args, pageSize, (page - 1) * pageSize);
        return json({ items: rows.map(toImage), total, page, pageSize });
      }

      if (path === '/api/search' && req.method === 'POST') {
        if (Number(req.headers.get('content-length')) > 10000) {
          return json({ error: 'Request too large' }, 413);
        }
        const body = await req.text();
        if (body.length > 10000) {
          return json({ error: 'Request too large' }, 413);
        }

        const raw = JSON.parse(body);
        for (const key of ['query', 'folderId', 'referenceImageId']) {
          if (
            raw[key] !== undefined &&
            (typeof raw[key] !== 'string' || raw[key].length > 2000)
          ) {
            throw new Error(`Invalid ${key}`);
          }
        }

        return json(
          await rpc({
            query: raw.query,
            folderId: raw.folderId,
            referenceImageId: raw.referenceImageId,
          }),
        );
      }

      if (path === '/api/groups' && req.method === 'GET') {
        return json(
          db
            .query<{ id: number; count: number; representatives: string }, []>(
              'SELECT * FROM groups ORDER BY count DESC,id',
            )
            .all()
            .map((group) => ({
              id: group.id,
              count: group.count,
              representatives: JSON.parse(group.representatives)
                .map((key: number) => {
                  const row = db.query(`${imageSelect} AND i.key=?`).get(key);
                  return row ? toImage(row) : null;
                })
                .filter(Boolean),
            })),
        );
      }

      const match = path.match(
        /^\/api\/images\/([a-f0-9]{24})(?:\/(preview|original))?$/,
      );
      if (match && req.method === 'GET') {
        const image = getImage(db, match[1]);
        if (!image) {
          return json({ error: 'Image not found' }, 404);
        }
        if (!match[2]) {
          return json(image);
        }

        let file: string;

        if (match[2] === 'preview') {
          file = (
            db
              .query(
                'SELECT a.cache FROM images i JOIN assets a ON a.id=i.asset_id WHERE i.id=?',
              )
              .get(match[1]) as { cache: string }
          ).cache;
        } else {
          file = await realpath(join(config.library.path, image.path));
          if (!inside(config.library.path, file)) {
            return json({ error: 'File is outside the image library' }, 403);
          }
        }

        const source = Bun.file(file);
        if (!(await source.exists())) {
          return json(
            { error: 'File no longer exists; rescan the library' },
            404,
          );
        }

        return new Response(source, {
          headers: {
            'content-type': match[2] === 'preview' ? 'image/jpeg' : source.type,
            'cache-control': 'no-cache',
            'x-content-type-options': 'nosniff',
          },
        });
      }

      if (path.startsWith('/api/')) {
        return json({ error: 'Not found' }, 404);
      }

      if (req.method !== 'GET' && req.method !== 'HEAD') {
        return new Response('Method not allowed', { status: 405 });
      }

      const root = resolve('dist');
      const candidate = resolve(root, '.' + decodeURIComponent(path));
      if (!inside(root, candidate)) {
        return new Response('Forbidden', { status: 403 });
      }

      const file = Bun.file(candidate);
      if (extname(candidate) && (await file.exists())) {
        return new Response(file);
      }

      const index = Bun.file(join(root, 'index.html'));
      return (await index.exists())
        ? new Response(index, { headers: { 'content-type': 'text/html' } })
        : new Response(
            'Frontend not built. Run bun run build, or use bun run dev.',
            { status: 503 },
          );
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      console.error(message);
      return json(
        { error: message },
        message.includes('not found') ? 404 : 400,
      );
    }
  },
});

console.log(
  `Locallery → http://${config.server.host}:${server.port} · llama-server → ${config.embedding.base_url}`,
);
worker.postMessage({ type: 'init', config });

function shutdown() {
  clearInterval(keepalive);
  worker.terminate();
  server.stop(true);
  db.close();
  process.exit(0);
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
