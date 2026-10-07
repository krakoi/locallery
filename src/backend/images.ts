import sharp from 'sharp';
import { join } from 'node:path';
import { mkdir, unlink } from 'node:fs/promises';
import type { Config } from './config';

export const supported = new Set([
  '.jpg',
  '.jpeg',
  '.png',
  '.webp',
  '.avif',
  '.gif',
  '.tif',
  '.tiff',
]);

export async function makePreview(
  source: string,
  assetId: string,
  config: Config,
) {
  const directory = join(config.storage.path, 'previews');
  await mkdir(directory, { recursive: true });
  const cache = join(directory, assetId + '.jpg'),
    input = join(directory, assetId + '.png');
  try {
    const { data, info } = await sharp(source, {
      page: 0,
      pages: 1,
      limitInputPixels: 268402689,
    })
      .rotate()
      .resize({
        width: 1280,
        height: 1280,
        fit: 'inside',
        withoutEnlargement: true,
      })
      .flatten({ background: '#fff' })
      .toColourspace('srgb')
      .png()
      .toBuffer({ resolveWithObject: true });
    await Bun.write(input, data);
    const process = Bun.spawn(['cjpegli', input, cache, '--quality=90'], {
      stdout: 'ignore',
      stderr: 'pipe',
    });
    const stderr = await new Response(process.stderr).text();
    if ((await process.exited) !== 0) {
      throw new Error(`cjpegli failed: ${stderr.trim()}`);
    }
    return { cache, width: info.width, height: info.height };
  } finally {
    await unlink(input).catch(() => {});
  }
}
