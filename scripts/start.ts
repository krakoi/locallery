import { resolve } from 'node:path';

const project = resolve(import.meta.dir, '..');
const extra = process.env.LOCALLERY_TORCH_EXTRA || 'cpu';
const args = [
  'uv',
  'run',
  '--project',
  project,
  '--extra',
  extra,
  'python',
  '-m',
  'locallery',
  ...Bun.argv.slice(2),
];
const child = Bun.spawn(args, {
  stdout: 'inherit',
  stderr: 'inherit',
  stdin: 'inherit',
});
for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.on(signal, () => child.kill(signal));
}
process.exit(await child.exited);
