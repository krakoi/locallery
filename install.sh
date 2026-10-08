#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
selection=""

usage() {
  printf 'Usage: %s [--cpu | --cuda]\n' "$0"
  printf 'Install Python/JavaScript dependencies and build the frontend.\n'
  printf 'Without a flag, detect NVIDIA hardware and prompt for the PyTorch build.\n'
}

for argument in "$@"; do
  case "$argument" in
    --cpu | --cuda)
      if [[ -n "$selection" ]]; then
        printf 'Choose only one of --cpu or --cuda.\n' >&2
        exit 1
      fi
      selection="${argument#--}"
      ;;
    --help | -h) usage; exit 0 ;;
    *) usage >&2; exit 1 ;;
  esac
done

missing=0
for dependency in bun uv cjpegli; do
  if ! command -v "$dependency" >/dev/null 2>&1; then
    printf 'Missing dependency: %s (see README.md for installation links).\n' "$dependency" >&2
    missing=1
  fi
done
if [[ "$missing" == 1 ]]; then
  exit 1
fi
for dependency in ffmpeg ffprobe; do
  if ! command -v "$dependency" >/dev/null 2>&1; then
    printf '%s is missing; install FFmpeg for video indexing, or set video.enabled: false.\n' "$dependency" >&2
  fi
done

nvidia_gpu_found() {
  local devices device vendor device_class
  if command -v nvidia-smi >/dev/null 2>&1; then
    if devices="$(nvidia-smi --list-gpus 2>/dev/null)" && [[ "$devices" == *GPU* ]]; then
      return 0
    fi
  fi
  if command -v lspci >/dev/null 2>&1; then
    if devices="$(lspci 2>/dev/null)"; then
      while IFS= read -r device; do
        if [[ "$device" == *NVIDIA* && ( "$device" == *'VGA compatible controller'* || "$device" == *'3D controller'* || "$device" == *'Display controller'* ) ]]; then
          return 0
        fi
      done <<< "$devices"
      return 1
    fi
  fi
  # Linux fallback when PCI utilities or the NVIDIA driver are unavailable.
  for device in /sys/bus/pci/devices/*; do
    if [[ -r "$device/vendor" && -r "$device/class" ]]; then
      read -r vendor < "$device/vendor" || continue
      read -r device_class < "$device/class" || continue
      if [[ "$vendor" == 0x10de && "$device_class" == 0x03* ]]; then
        return 0
      fi
    fi
  done
  return 1
}

if [[ -z "$selection" ]]; then
  if nvidia_gpu_found; then
    default="cuda"
    printf 'NVIDIA GPU detected. Default PyTorch build: CUDA.\n'
  else
    default="cpu"
    printf 'No NVIDIA GPU detected. Default PyTorch build: CPU.\n'
  fi
  if [[ -t 0 ]]; then
    while true; do
      printf 'PyTorch build [cuda/cpu] (%s): ' "$default"
      if ! IFS= read -r answer; then
        printf '\nInstallation cancelled.\n' >&2
        exit 1
      fi
      case "${answer:-$default}" in
        [Cc][Pp][Uu]) selection="cpu"; break ;;
        [Cc][Uu][Dd][Aa]) selection="cuda"; break ;;
        *) printf 'Enter cuda or cpu, or press Enter for %s.\n' "$default" ;;
      esac
    done
  else
    selection="$default"
    printf 'Using %s without a prompt (override with --cpu or --cuda).\n' "$selection"
  fi
fi

cd -- "$repo_dir"
printf 'Installing dependencies with the %s PyTorch build...\n' "$selection"
bun install --frozen-lockfile
uv sync --locked --extra "$selection"
bun run build

"$repo_dir/.venv/bin/python" - "$selection" <<'PY'
import sys
import torch

print(f"PyTorch {torch.__version__} installed.")
if sys.argv[1] == "cuda" and not torch.cuda.is_available():
    print("CUDA is not available to PyTorch; device: auto will use CPU until the GPU driver is working.")
PY

printf '\nInstallation complete. Start with bun run start, or run.sh from your album folder.\n'
