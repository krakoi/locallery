#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ ! -x "$repo_dir/.venv/bin/python" ]]; then
  printf 'Python environment missing. Run %s/install.sh first.\n' "$repo_dir" >&2
  exit 1
fi

# Build in the repository; retain the caller's library directory for the backend.
(
  cd -- "$repo_dir"
  bun run build
)

exec "$repo_dir/.venv/bin/python" -m locallery "$@"
