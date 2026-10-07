import { test, expect } from 'bun:test';
import { mkdir, rm, symlink } from 'node:fs/promises';
import { join } from 'node:path';
import sharp from 'sharp';
import { makePreview } from '../src/backend/images';
import { readConfig } from '../src/backend/config';

test('rotates EXIF orientation, avoids upscaling, and rejects symlinked storage inside source', async () => {
  const root = join(
    process.cwd(),
    '.test-artifacts',
    'cache-' + crypto.randomUUID(),
  );
  await mkdir(join(root, 'source'), { recursive: true });
  try {
    const source = join(root, 'source', 'oriented.jpg');
    await sharp({
      create: { width: 60, height: 30, channels: 3, background: 'red' },
    })
      .withMetadata({ orientation: 6 })
      .jpeg()
      .toFile(source);
    const config = { storage: { path: join(root, 'data') } } as any;
    const p = await makePreview(source, 'orientation', config);
    expect(p.width).toBe(30);
    expect(p.height).toBe(60);
    const info = await sharp(p.cache).metadata();
    expect(info.orientation).toBeUndefined();
    await symlink(join(root, 'source'), join(root, '.locallery'));
    expect(() =>
      readConfig({ cwd: root, globalDirectory: join(root, 'global') }),
    ).toThrow('symlink');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
