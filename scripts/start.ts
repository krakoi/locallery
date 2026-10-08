import { backendCommand } from './python';

const args = backendCommand(Bun.argv.slice(2));
const child = Bun.spawn(args, {
  stdout: 'inherit',
  stderr: 'inherit',
  stdin: 'inherit',
});
for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.on(signal, () => child.kill(signal));
}
process.exit(await child.exited);
