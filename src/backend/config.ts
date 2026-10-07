import { parse } from 'yaml';
import { resolve, relative, isAbsolute, dirname } from 'node:path';
import { existsSync, realpathSync, readFileSync } from 'node:fs';

export interface Config {
  library: { path: string };
  storage: { path: string };
  server: { host: string; port: number };
  embedding: {
    base_url: string;
    model: string;
    revision: string;
    concurrency: number;
    timeout_seconds: number;
  };
}

export const PREPROCESS = 'jpegli-q90-srgb-white-oriented-fit1280-v1';

export function inside(root: string, path: string) {
  const r = relative(root, path);
  return (
    r === '' ||
    (!r.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`) &&
      r !== '..' &&
      !isAbsolute(r))
  );
}

function canonical(path: string): string {
  if (existsSync(path)) {
    return realpathSync(path);
  }
  const parent = dirname(path);
  return parent === path
    ? path
    : resolve(canonical(parent), relative(parent, path));
}

export function readConfig(file: string): Config {
  const raw = parse(readFileSync(file, 'utf8'));

  if (
    !raw ||
    typeof raw !== 'object' ||
    typeof raw.library?.path !== 'string' ||
    !raw.library.path.trim()
  ) {
    throw new Error('library.path must be a nonempty string');
  }

  for (const section of ['library', 'storage', 'server', 'embedding']) {
    const value = raw[section];
    if (
      value !== undefined &&
      (!value || typeof value !== 'object' || Array.isArray(value))
    ) {
      throw new Error(`${section} must be a YAML mapping`);
    }
  }

  const base = dirname(resolve(file));

  const text = (v: unknown, d: string, n: string) => {
    if (v === undefined) {
      return d;
    }
    if (typeof v !== 'string' || !v.trim()) {
      throw new Error(`${n} must be a nonempty string`);
    }
    return v;
  };

  const number = (
    v: unknown,
    d: number,
    min: number,
    max: number,
    n: string,
  ) => {
    if (v === undefined) {
      return d;
    }
    if (typeof v !== 'number' || !Number.isInteger(v) || v < min || v > max) {
      throw new Error(`${n} must be an integer between ${min} and ${max}`);
    }
    return v;
  };

  const config: Config = {
    library: { path: canonical(resolve(base, raw.library.path)) },
    storage: {
      path: canonical(
        resolve(base, text(raw.storage?.path, './data', 'storage.path')),
      ),
    },
    server: {
      host: text(raw.server?.host, '127.0.0.1', 'server.host'),
      port: number(raw.server?.port, 3000, 1, 65535, 'server.port'),
    },
    embedding: {
      base_url: text(
        raw.embedding?.base_url,
        'http://127.0.0.1:4096/v1',
        'embedding.base_url',
      ).replace(/\/$/, ''),
      model: text(raw.embedding?.model, 'embeddinggemma-2', 'embedding.model'),
      revision: text(
        raw.embedding?.revision,
        'embeddinggemma-2-Q8_0',
        'embedding.revision',
      ),
      concurrency: number(
        raw.embedding?.concurrency,
        1,
        1,
        16,
        'embedding.concurrency',
      ),
      timeout_seconds: number(
        raw.embedding?.timeout_seconds,
        60,
        1,
        3600,
        'embedding.timeout_seconds',
      ),
    },
  };

  if (inside(config.library.path, config.storage.path)) {
    throw new Error('storage.path must be outside library.path');
  }

  const url = new URL(config.embedding.base_url);
  if (!['http:', 'https:'].includes(url.protocol)) {
    throw new Error('embedding.base_url must use HTTP or HTTPS');
  }
  return config;
}

export function fingerprint(config: Config) {
  return new Bun.CryptoHasher('sha256')
    .update(
      JSON.stringify([
        config.embedding.model,
        config.embedding.revision,
        768,
        PREPROCESS,
      ]),
    )
    .digest('hex');
}
