"""Cooperative cancellation of work owned by the service thread."""

import subprocess
import time


class ScanCancelled(Exception):
    """Shutdown requested; stop processing without recording a media error."""


def run_command(args, timeout, check_running):
    """Reap decoder children promptly when a scan is stopped."""
    check_running()
    with subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        deadline = time.monotonic() + timeout
        try:
            while True:
                check_running()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(args, timeout)
                try:
                    stdout, stderr = process.communicate(timeout=min(0.1, remaining))
                    return subprocess.CompletedProcess(
                        args, process.returncode, stdout, stderr
                    )
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            process.kill()
            process.communicate()
            raise
