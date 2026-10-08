"""Real Uvicorn/SIGINT regression checks with an open progress stream."""

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

HARNESS = """
import sys, time
from pathlib import Path
import numpy as np
import uvicorn
import locallery.server as server
from locallery.config import Config
mode = sys.argv[4].removeprefix("double-")
server.SHUTDOWN_TIMEOUT = 1.5

class Embedder:
    fingerprint = 'shutdown-test'
    def __init__(self, config): pass
    def load(self):
        if mode == 'load': time.sleep(60)
    def embed(self, **kwargs):
        print('EMBED', flush=True)
        time.sleep(60 if mode == 'inference' else 0.15)
        vector = np.zeros(768, dtype=np.float32)
        vector[0] = 1
        return vector

app = server.create_app(Config(Path(sys.argv[1]), Path(sys.argv[2])), Embedder)
runner = getattr(server, 'GalleryServer', uvicorn.Server)
try:
    runner(uvicorn.Config(app, host='127.0.0.1', port=int(sys.argv[3]),
                          timeout_graceful_shutdown=1)).run()
except KeyboardInterrupt:
    pass
"""


@pytest.mark.parametrize(
    "mode",
    ["idle", "scan", "load", "inference", "reload", "double-load", "double-inference"],
)
def test_ctrl_c_closes_sse_and_stops_scan(tmp_path, mode):
    scenario = mode.removeprefix("double-")
    import socket

    source = tmp_path / "source"
    source.mkdir()
    if scenario in ("scan", "inference"):
        for number in range(60):
            Image.new("RGB", (8, 8), (number, 0, 0)).save(source / f"{number}.png")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = os.environ | {"PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    command = [sys.executable, "-c", HARNESS]
    if mode == "reload":
        harness_file = tmp_path / "shutdown_app.py"
        harness_file.write_text(
            HARNESS.split("app = server.create_app", 1)[0]
            + """
def create_app():
    return server.create_app(Config(Path(sys.argv[1]), Path(sys.argv[2])), Embedder)

if __name__ == '__main__':
    from uvicorn.supervisors import ChangeReload
    config = uvicorn.Config('shutdown_app:create_app', factory=True, reload=True,
                           reload_dirs=[str(Path(__file__).parent)],
                           host='127.0.0.1', port=int(sys.argv[3]),
                           timeout_graceful_shutdown=1)
    runner = server.GalleryServer(config)
    ChangeReload(config, target=runner.run, sockets=[config.bind_socket()]).run()
"""
        )
        command = [sys.executable, str(harness_file)]
        env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env["PYTHONPATH"]
    process = subprocess.Popen(
        command + [str(source), str(tmp_path / "data"), str(port), mode],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    stream = None
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                stream = urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/events", timeout=1
                )
                assert stream.readline().startswith(b"data:")
                break
            except (urllib.error.URLError, TimeoutError):
                if time.monotonic() >= deadline:
                    pytest.fail("Server did not start")
                time.sleep(0.05)
        if scenario in ("scan", "inference"):
            time.sleep(0.4)
        started = time.monotonic()
        process.send_signal(signal.SIGINT)
        if mode.startswith("double-"):
            time.sleep(0.15)
            process.send_signal(signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=4)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            pytest.fail("Ctrl+C left the scan running:\n" + output[-2500:])
        assert time.monotonic() - started < 4
        assert "timeout graceful shutdown exceeded" not in output, output
        assert "Exception in ASGI application" not in output, output
        assert "Traceback" not in output, output
        if scenario in ("scan", "inference"):
            assert output.count("EMBED\n") < 10, output
        if scenario in ("load", "inference"):
            assert process.returncode == 130, output
            if mode.startswith("double-"):
                assert time.monotonic() - started < 1
            else:
                assert "Active operation did not stop" in output
        else:
            assert process.returncode in (0, -signal.SIGINT), output
        if scenario == "scan":
            import sqlite3

            with sqlite3.connect(tmp_path / "data" / "library.sqlite") as db:
                committed = db.execute(
                    "SELECT count(*) FROM images WHERE status='ready'"
                ).fetchone()[0]
            assert 0 < committed < 10
            assert len(list(source.glob("*.png"))) == 60
    finally:
        if stream:
            stream.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def test_cancelled_decoder_process_is_reaped(tmp_path):
    from threading import Event, Timer

    from locallery.cancellation import ScanCancelled, run_command

    stopping = Event()
    pidfile = tmp_path / "decoder.pid"

    def check_running():
        if stopping.is_set():
            raise ScanCancelled()

    timer = Timer(0.4, stopping.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(ScanCancelled):
            run_command(
                [
                    sys.executable,
                    "-c",
                    "import os,sys,time; open(sys.argv[1],'w').write(str(os.getpid())); time.sleep(30)",
                    str(pidfile),
                ],
                timeout=60,
                check_running=check_running,
            )
    finally:
        timer.cancel()
    assert time.monotonic() - started < 2
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
