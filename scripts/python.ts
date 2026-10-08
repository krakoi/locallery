import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

const project = resolve(import.meta.dir, '..');
const python = resolve(project, '.venv', 'bin', 'python');

export function backendCommand(args: string[] = []) {
  if (!existsSync(python)) {
    throw new Error(
      `Python environment missing. Run ${resolve(project, 'install.sh')} first.`,
    );
  }
  return [python, '-m', 'locallery', ...args];
}
