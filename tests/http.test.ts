import { test, expect } from 'bun:test';
import { mkdir, rm } from 'node:fs/promises';
import { join } from 'node:path';
import sharp from 'sharp';

test('HTTP lifecycle blocks indexing, emits SSE, serves images and supports manual rescan', async () => {
  const root = join(
    process.cwd(),
    '.test-artifacts',
    'http-' + crypto.randomUUID(),
  );
  await mkdir(join(root, 'source'), { recursive: true });
  await sharp({
    create: { width: 32, height: 24, channels: 3, background: 'red' },
  })
    .png()
    .toFile(join(root, 'source', 'red.png'));
  const model = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    async fetch(req) {
      const path = new URL(req.url).pathname;
      if (path === '/v1/models') {
        return Response.json({ data: [{ id: 'mock' }] });
      }
      if (path === '/props') {
        return Response.json({ model_path: 'mock.gguf', total_slots: 1 });
      }
      await Bun.sleep(300);
      return Response.json({
        data: [{ embedding: [1, ...new Array(767).fill(0)] }],
      });
    },
  });
  const reservation = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    fetch: () => new Response(),
  });
  const port = reservation.port;
  reservation.stop(true);
  await mkdir(join(root, '.locallery'), { recursive: true });
  await Bun.write(
    join(root, '.locallery', 'config.yml'),
    `library:\n  path: ./source\nserver:\n  port: ${port}\nembedding:\n  base_url: http://127.0.0.1:${model.port}/v1\n`,
  );
  const child = Bun.spawn(
    ['bun', join(process.cwd(), 'src/backend/server.ts')],
    {
      cwd: root,
      env: { ...process.env, LOCALLERY_HOME: join(root, 'global') },
      stdout: 'ignore',
      stderr: 'pipe',
    },
  );
  const base = `http://127.0.0.1:${port}/api`;
  async function status() {
    return (await (await fetch(base + '/status')).json()) as any;
  }
  async function ready() {
    for (let i = 0; i < 200; i++) {
      const p = await status();
      if (!p.busy) {
        expect(p.stage).toBe('ready');
        return p;
      }
      await Bun.sleep(10);
    }
    throw new Error('Backend did not finish');
  }
  try {
    let connected = false;
    for (let i = 0; i < 200; i++) {
      try {
        await status();
        connected = true;
        break;
      } catch {
        await Bun.sleep(10);
      }
    }
    expect(connected).toBe(true);
    expect((await fetch(base + '/images')).status).toBe(503);
    const events = await fetch(base + '/events');
    expect(events.headers.get('content-type')).toBe('text/event-stream');
    const reader = events.body!.getReader();
    const first = new TextDecoder().decode((await reader.read()).value);
    expect(first).toContain('"busy":true');
    await reader.cancel();
    const firstScan = await ready();
    expect(firstScan.indexed).toBe(1);
    const list = (await (await fetch(base + '/images')).json()) as any;
    expect(list.total).toBe(1);
    const id = list.items[0].id;
    expect(
      (await fetch(base + `/images/${id}/preview`)).headers.get('content-type'),
    ).toBe('image/jpeg');
    expect((await fetch(base + `/images/${id}/original`)).status).toBe(200);
    expect((await fetch(base + '/images?folderId=missing')).status).toBe(404);
    expect((await fetch(base + '/images?page=-1')).status).toBe(400);
    const result = (await (
      await fetch(base + '/search', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ referenceImageId: id }),
      })
    ).json()) as any;
    expect(result.total).toBe(0);
    expect(
      ((await (await fetch(base + '/groups')).json()) as any[]).length,
    ).toBe(1);
    expect((await fetch(base + '/rescan', { method: 'POST' })).status).toBe(
      202,
    );
    const second = await ready();
    expect(second.indexed).toBe(0);
    expect(second.unchanged).toBe(1);
  } finally {
    child.kill();
    await child.exited;
    model.stop(true);
    await rm(root, { recursive: true, force: true });
  }
}, 10000);
