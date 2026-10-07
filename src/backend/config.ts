import { parse, stringify } from 'yaml';
import { homedir } from 'node:os';
import { resolve, relative, isAbsolute, dirname } from 'node:path';
import {
  existsSync,
  realpathSync,
  readFileSync,
  mkdirSync,
  writeFileSync,
  lstatSync,
} from 'node:fs';

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

const defaults = {
  server: { host: '127.0.0.1', port: 3000 },
  embedding: { base_url: 'http://127.0.0.1:4096/v1', timeout_seconds: 60 },
};

type Mapping = Record<string, unknown>;

function mapping(value: unknown, name: string): Mapping {
  if (value === undefined || value === null) {
    return {};
  }
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${name} must be a YAML mapping`);
  }
  return value as Mapping;
}

function load(file: string) {
  return existsSync(file)
    ? mapping(parse(readFileSync(file, 'utf8')), file)
    : {};
}

function directory(path: string) {
  if (existsSync(path) && lstatSync(path).isSymbolicLink()) {
    throw new Error(`Application directory must not be a symlink: ${path}`);
  }
  mkdirSync(path, { recursive: true });
}

export function readConfig(
  options: { cwd?: string; globalDirectory?: string } = {},
): Config {
  const cwd = canonical(resolve(options.cwd || process.cwd()));
  const globalDirectory = resolve(
    options.globalDirectory ||
      process.env.LOCALLERY_HOME ||
      resolve(homedir(), '.locallery'),
  );
  directory(globalDirectory);
  const globalFile = resolve(globalDirectory, 'config.yml');
  if (!existsSync(globalFile)) {
    try {
      writeFileSync(globalFile, stringify(defaults), { flag: 'wx' });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') {
        throw error;
      }
    }
  }
  const localDirectory = resolve(cwd, '.locallery');
  directory(localDirectory);
  directory(resolve(localDirectory, 'data'));
  const global = load(globalFile);
  const local = load(resolve(localDirectory, 'config.yml'));
  for (const raw of [global, local]) {
    if ('storage' in raw) {
      throw new Error('storage is automatic; remove storage from config.yml');
    }
    for (const section of ['library', 'server', 'embedding']) {
      if (raw[section] !== undefined) {
        mapping(raw[section], section);
      }
    }
    const embedding = mapping(raw.embedding, 'embedding');
    for (const name of ['model', 'revision', 'concurrency']) {
      if (name in embedding) {
        throw new Error(
          `embedding.${name} is discovered from llama-server; remove it from config.yml`,
        );
      }
    }
  }
  const server = {
    ...mapping(global.server, 'server'),
    ...mapping(local.server, 'server'),
  };
  const embedding = {
    ...mapping(global.embedding, 'embedding'),
    ...mapping(local.embedding, 'embedding'),
  };
  const library = mapping(local.library, 'library');
  const text = (value: unknown, fallback: string, name: string) => {
    if (value === undefined) {
      return fallback;
    }
    if (typeof value !== 'string' || !value.trim()) {
      throw new Error(`${name} must be a nonempty string`);
    }
    return value;
  };
  const number = (
    value: unknown,
    fallback: number,
    min: number,
    max: number,
    name: string,
  ) => {
    if (value === undefined) {
      return fallback;
    }
    if (
      typeof value !== 'number' ||
      !Number.isInteger(value) ||
      value < min ||
      value > max
    ) {
      throw new Error(`${name} must be an integer between ${min} and ${max}`);
    }
    return value;
  };
  const config: Config = {
    library: {
      path: canonical(resolve(cwd, text(library.path, '.', 'library.path'))),
    },
    storage: { path: canonical(resolve(localDirectory, 'data')) },
    server: {
      host: text(server.host, defaults.server.host, 'server.host'),
      port: number(server.port, defaults.server.port, 1, 65535, 'server.port'),
    },
    embedding: {
      base_url: text(
        embedding.base_url,
        defaults.embedding.base_url,
        'embedding.base_url',
      ).replace(/\/$/, ''),
      timeout_seconds: number(
        embedding.timeout_seconds,
        defaults.embedding.timeout_seconds,
        1,
        3600,
        'embedding.timeout_seconds',
      ),
      // Runtime values populated from llama-server before scanning.
      model: '',
      revision: '',
      concurrency: 1,
    },
  };
  if (
    inside(config.storage.path, config.library.path) ||
    config.library.path === canonical(localDirectory)
  ) {
    throw new Error('library.path must be outside application storage');
  }
  const url = new URL(config.embedding.base_url);
  if (
    !['http:', 'https:'].includes(url.protocol) ||
    !url.pathname.endsWith('/v1')
  ) {
    throw new Error(
      'embedding.base_url must use HTTP or HTTPS and end with /v1',
    );
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
