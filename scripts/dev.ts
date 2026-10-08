import { resolve } from 'node:path';

const project = resolve(import.meta.dir, '..');
const extra = process.env.LOCALLERY_TORCH_EXTRA || 'cpu';
const configProcess = Bun.spawn(
  [
    'uv',
    'run',
    '--project',
    project,
    '--extra',
    extra,
    'python',
    '-m',
    'locallery',
    '--print-config',
    '--select-library',
  ],
  { stdin: 'inherit', stdout: 'pipe', stderr: 'inherit' },
);
const output = await new Response(configProcess.stdout).text();
if (await configProcess.exited) {
  throw new Error('Could not load Python backend configuration');
}
const config = JSON.parse(output.trim().split('\n').at(-1)!);
const backend = Bun.spawn(
  ['bun', 'scripts/start.ts', '--reload', '--library', config.library],
  {
    stdin: 'inherit',
    stdout: 'inherit',
    stderr: 'inherit',
  },
);
const frontend = Bun.spawn(['bun', 'x', 'vite', '--host', '127.0.0.1'], {
  env: {
    ...process.env,
    LOCALLERY_BACKEND: `http://${config.host}:${config.port}`,
  },
  stdout: 'inherit',
  stderr: 'inherit',
});
function stop() {
  backend.kill();
  frontend.kill();
  process.exit();
}
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
await Promise.race([backend.exited, frontend.exited]);
stop();
