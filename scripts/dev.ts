import { readConfig } from '../src/backend/config';

const config = readConfig();
const backend = Bun.spawn(['bun', '--watch', 'src/backend/server.ts'], {
  stdout: 'inherit',
  stderr: 'inherit',
});
const frontend = Bun.spawn(['bun', 'x', 'vite', '--host', '127.0.0.1'], {
  env: {
    ...process.env,
    LOCALLERY_BACKEND: `http://${config.server.host}:${config.server.port}`,
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
