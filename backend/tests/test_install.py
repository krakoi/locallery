"""Installer and launch behavior with fake dependencies; no downloads."""

import os
import pty
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def installation(tmp_path):
    project = tmp_path / "project with spaces"
    binaries = tmp_path / "commands"
    album = tmp_path / "album"
    project.mkdir()
    binaries.mkdir()
    album.mkdir()
    for script in ("install.sh", "run.sh"):
        shutil.copy2(ROOT / script, project / script)
    calls = tmp_path / "calls"
    env = os.environ | {"PATH": str(binaries), "INSTALL_CALLS": str(calls)}
    (binaries / "dirname").symlink_to(shutil.which("dirname"))

    def executable(path, body):
        path.write_text("#!/bin/bash\nset -e\n" + body)
        path.chmod(0o755)

    for command in ("bun", "uv", "cjpegli", "ffmpeg", "ffprobe"):
        executable(
            binaries / command,
            f'printf "%s|%s|%s\\n" "{command}" "$PWD" "$*" >> "$INSTALL_CALLS"\n'
            + (
                'if [[ "${FAIL_SYNC:-}" == 1 ]]; then exit 7; fi\n'
                if command == "uv"
                else ""
            ),
        )
    executable(
        binaries / "nvidia-smi",
        'if [[ "${GPU_DRIVER:-}" == working ]]; then printf "GPU 0: NVIDIA test GPU\\n"; else exit 1; fi\n',
    )
    executable(
        binaries / "lspci",
        'if [[ "${GPU_HARDWARE:-}" == nvidia ]]; then printf "01:00.0 VGA compatible controller: NVIDIA test GPU\\n"; fi\n'
        'printf "01:00.1 Audio device: NVIDIA audio\\n"\n',
    )
    python = project / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    executable(
        python,
        'printf "python|%s|%s\\n" "$PWD" "$*" >> "$INSTALL_CALLS"\n',
    )
    return project, binaries, album, calls, env


def invoke(installation, args=(), env=None, answer=None, script="install.sh"):
    project, _, album, _, base_env = installation
    command = ["/bin/bash", str(project / script), *args]
    options = dict(
        cwd=album, env=base_env | (env or {}), capture_output=True, text=True
    )
    if answer is None:
        return subprocess.run(command, stdin=subprocess.DEVNULL, timeout=5, **options)
    master, slave = pty.openpty()
    try:
        with subprocess.Popen(
            command,
            stdin=slave,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=album,
            env=options["env"],
            text=True,
        ) as process:
            os.write(master, answer.encode())
            stdout, stderr = process.communicate(timeout=5)
            return subprocess.CompletedProcess(
                command, process.returncode, stdout, stderr
            )
    finally:
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize(
    "driver,hardware,expected",
    [("working", "", "cuda"), ("broken", "nvidia", "cuda"), ("", "", "cpu")],
)
def test_hardware_detection_and_noninteractive_default(
    installation, driver, hardware, expected
):
    result = invoke(installation, env={"GPU_DRIVER": driver, "GPU_HARDWARE": hardware})
    assert result.returncode == 0, result.stderr
    project, _, _, calls, _ = installation
    assert calls.read_text().splitlines() == [
        f"bun|{project}|install --frozen-lockfile",
        f"uv|{project}|sync --locked --extra {expected}",
        f"bun|{project}|run build",
        f"python|{project}|- {expected}",
    ]


@pytest.mark.parametrize(
    "answer,expected", [("\n", "cuda"), ("cpu\n", "cpu"), ("wrong\nCPU\n", "cpu")]
)
def test_interactive_default_and_selection(installation, answer, expected):
    result = invoke(installation, env={"GPU_DRIVER": "working"}, answer=answer)
    assert result.returncode == 0, result.stderr
    assert "PyTorch build [cuda/cpu] (cuda):" in result.stdout
    assert f"--extra {expected}" in installation[3].read_text()


@pytest.mark.parametrize("selection", ["cpu", "cuda"])
def test_explicit_selection_skips_prompt_and_detection(installation, selection):
    result = invoke(
        installation, args=[f"--{selection}"], env={"GPU_HARDWARE": "nvidia"}
    )
    assert result.returncode == 0, result.stderr
    assert "detected" not in result.stdout and "PyTorch build [" not in result.stdout
    assert f"--extra {selection}" in installation[3].read_text()


def test_missing_dependency_stops_before_installation(installation):
    _, binaries, _, calls, _ = installation
    (binaries / "cjpegli").unlink()
    result = invoke(installation, args=["--cpu"])
    assert result.returncode == 1
    assert "Missing dependency: cjpegli" in result.stderr
    assert not calls.exists()


def test_missing_video_tools_report_requirement(installation):
    _, binaries, _, _, _ = installation
    (binaries / "ffmpeg").unlink()
    (binaries / "ffprobe").unlink()
    result = invoke(installation, args=["--cpu"])
    assert result.returncode == 0, result.stderr
    assert "video.enabled: false" in result.stderr


def test_failed_sync_stops_before_build(installation):
    result = invoke(installation, args=["--cpu"], env={"FAIL_SYNC": "1"})
    assert result.returncode == 7
    assert "run build" not in installation[3].read_text()


@pytest.mark.parametrize("args", [["--wrong"], ["--cpu", "--cuda"]])
def test_invalid_arguments_do_not_install(installation, args):
    result = invoke(installation, args=args)
    assert result.returncode == 1
    assert not installation[3].exists()


def test_run_preserves_working_directory_and_environment(installation):
    project, _, album, calls, _ = installation
    result = invoke(
        installation, args=["--library", "photos with spaces"], script="run.sh"
    )
    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines() == [
        f"bun|{project}|run build",
        f"python|{album}|-m locallery --library photos with spaces",
    ]


def test_run_without_installation_reports_installer(installation):
    project, _, _, calls, _ = installation
    (project / ".venv" / "bin" / "python").unlink()
    result = invoke(installation, script="run.sh")
    assert result.returncode == 1
    assert "install.sh first" in result.stderr
    assert not calls.exists()
