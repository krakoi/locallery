#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# Build in the repository; retain the caller's library directory for the backend.
(
  cd -- "$repo_dir"
  bun run build
)

exec uv run --project "$repo_dir" --extra "${LOCALLERY_TORCH_EXTRA:-cpu}" python -m locallery
