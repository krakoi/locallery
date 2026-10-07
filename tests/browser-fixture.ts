// Run explicitly to inspect pagination and nested folders without changing real photos.
import { mkdir, rm } from 'node:fs/promises';
import { join } from 'node:path';
import sharp from 'sharp';

const root = join(process.cwd(), '.test-artifacts', 'browser-fixture');
await mkdir(join(root, 'source', 'album', 'nested'), { recursive: true });

const png = await sharp({
  create: { width: 240, height: 180, channels: 3, background: '#819c67' },
})
  .png()
  .toBuffer();
for (let i = 0; i < 201; i++) {
  await Bun.write(
    join(
      root,
      'source',
      'album',
      i === 200 ? 'nested/last.png' : `image-${String(i).padStart(3, '0')}.png`,
    ),
    png,
  );
}

const mock = Bun.serve({
  hostname: '127.0.0.1',
  port: 0,
  fetch(req) {
    const path = new URL(req.url).pathname;
    if (path === '/v1/models') {
      return Response.json({ data: [{ id: 'mock' }] });
    }
    if (path === '/props') {
      return Response.json({ model_path: 'mock.gguf', total_slots: 1 });
    }
    return Response.json({
      data: [{ embedding: [1, ...new Array(767).fill(0)] }],
    });
  },
});

await mkdir(join(root, '.locallery'), { recursive: true });
await Bun.write(
  join(root, '.locallery', 'config.yml'),
  `library:\n  path: ./source\nserver:\n  port: 3002\nembedding:\n  base_url: http://127.0.0.1:${mock.port}/v1\n`,
);

const child = Bun.spawn(['bun', join(process.cwd(), 'src/backend/server.ts')], {
  cwd: root,
  env: { ...process.env, LOCALLERY_HOME: join(root, 'global') },
  stdout: 'inherit',
  stderr: 'inherit',
});

async function stop() {
  child.kill();
  await child.exited;
  mock.stop(true);
  await rm(root, { recursive: true, force: true });
  process.exit();
}
process.on('SIGINT', () => void stop());
process.on('SIGTERM', () => void stop());
await child.exited;
await stop();
