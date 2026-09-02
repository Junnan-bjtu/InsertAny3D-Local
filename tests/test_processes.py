from __future__ import annotations

import base64
import json
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from insertany3d.executors import CommandExecutor
from insertany3d.processes import ProcessIdentity, ProcessSupervisor, WindowsJobRunner, WindowsJobRunnerError


class FakeWindowsBackend:
    def __init__(self, remaining=()):
        self.remaining = tuple(remaining)
        self.bound = []
        self.terminated = []
        self.released = []

    def bind(self, pid):
        self.bound.append(pid)

    def terminate(self, pid, *, grace_seconds, poll_seconds):
        self.terminated.append((pid, grace_seconds, poll_seconds))
        return self.remaining

    def release(self, pid):
        self.released.append(pid)


class ProcessSupervisorTests(unittest.TestCase):
    def test_wsl_unity_command_is_wrapped_in_windows_kill_on_close_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "windows_job_runner.ps1"
            script.write_text("# fixture", encoding="utf-8")
            runner = WindowsJobRunner(
                platform_name="posix",
                powershell_executable="powershell.exe",
                script_path=script,
            )
            wrapped = runner.wrap(
                ["/mnt/q/programs/Unity/Editor/Unity.exe", "-batchmode", "-projectPath", r"Q:\Project"],
                cwd=Path("/mnt/q/work/attempt"),
            )
        self.assertEqual(wrapped[0], "powershell.exe")
        payload = wrapped[wrapped.index("-CommandJsonBase64") + 1]
        decoded = base64.b64decode(payload, validate=True).decode("utf-8")
        self.assertEqual(json.loads(decoded)[0], r"Q:\programs\Unity\Editor\Unity.exe")
        self.assertEqual(wrapped[-1], r"Q:\work\attempt")

    def test_wsl_unity_fails_closed_when_job_wrapper_is_missing(self) -> None:
        runner = WindowsJobRunner(
            platform_name="posix",
            powershell_executable="powershell.exe",
            script_path="/definitely/missing/windows_job_runner.ps1",
        )
        with self.assertRaises(WindowsJobRunnerError):
            runner.wrap(["Unity.exe", "-batchmode"], cwd=Path("/tmp"))

    def test_command_executor_launches_the_job_wrapper_not_unmanaged_unity(self) -> None:
        class RecordingJobRunner:
            def __init__(self):
                self.calls = []

            def wrap(self, command, *, cwd):
                self.calls.append((list(command), cwd))
                return ["powershell-job-wrapper", "fixture"]

        class RecordingSupervisor:
            def __init__(self):
                self.bound = []
                self.released = []

            def bind(self, pid): self.bound.append(pid)
            def release(self, pid): self.released.append(pid)

        process = Mock(pid=321, returncode=0)
        process.poll.return_value = 0
        job = RecordingJobRunner()
        supervisor = RecordingSupervisor()
        controller = Mock()
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "attempt" / "output.staging"
            staging.mkdir(parents=True)
            item = SimpleNamespace(staging_dir=staging, stage_id=1, lease_token="lease")
            with patch("insertany3d.executors.subprocess.Popen", return_value=process) as popen, patch(
                "insertany3d.executors.os.getpgid", return_value=321
            ):
                CommandExecutor(
                    process_supervisor=supervisor,
                    windows_job_runner=job,
                ).execute(controller, item, ["/mnt/q/tools/Unity.exe", "-batchmode"])
        self.assertEqual(popen.call_args.args[0], ["powershell-job-wrapper", "fixture"])
        self.assertEqual(job.calls[0][0][0], "/mnt/q/tools/Unity.exe")
        self.assertEqual(supervisor.bound, [321])
        self.assertEqual(supervisor.released, [321])

    def test_command_executor_cancellation_terminates_the_job_wrapper(self) -> None:
        class RecordingJobRunner:
            def wrap(self, command, *, cwd):
                del command, cwd
                return ["powershell-job-wrapper", "fixture"]

        class RecordingSupervisor:
            def __init__(self):
                self.terminated = []
                self.released = []

            def bind(self, pid):
                del pid

            def terminate(self, identity, *, diagnostics_dir):
                self.terminated.append((identity, diagnostics_dir))
                return SimpleNamespace(completed=True)

            def release(self, pid):
                self.released.append(pid)

        process = Mock(pid=654, returncode=None)
        process.poll.return_value = None
        supervisor = RecordingSupervisor()
        controller = Mock()
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "attempt" / "output.staging"
            staging.mkdir(parents=True)
            item = SimpleNamespace(staging_dir=staging, stage_id=1, lease_token="lease")
            with patch("insertany3d.executors.subprocess.Popen", return_value=process) as popen, patch(
                "insertany3d.executors.os.getpgid", return_value=654
            ):
                result = CommandExecutor(
                    process_supervisor=supervisor,
                    windows_job_runner=RecordingJobRunner(),
                ).execute(controller, item, ["Unity.exe"], canceled=lambda: True)

        self.assertEqual(popen.call_args.args[0], ["powershell-job-wrapper", "fixture"])
        self.assertEqual(len(supervisor.terminated), 1)
        self.assertEqual(supervisor.released, [654])
        self.assertEqual(result.error_code, "canceled")
        self.assertTrue(result.cleanup_completed)

    def test_powershell_wrapper_assigns_suspended_process_before_resume(self) -> None:
        script = Path(__file__).resolve().parents[1] / "tools" / "windows_job_runner.ps1"
        text = script.read_text(encoding="utf-8")
        self.assertIn("JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE", text)
        self.assertIn("Check(CreateProcess(null, commandLine", text)
        self.assertIn("$command = [string[]]($CommandJson | ConvertFrom-Json)", text)
        self.assertLess(text.index("Check(CreateProcess("), text.index("Check(AssignProcessToJobObject("))
        self.assertLess(text.index("Check(AssignProcessToJobObject("), text.index("ResumeThread(process.hThread)"))

    def test_command_executor_binds_and_releases_process_tracking(self) -> None:
        class RecordingSupervisor:
            def __init__(self):
                self.bound = []
                self.released = []

            def bind(self, pid):
                self.bound.append(pid)

            def release(self, pid):
                self.released.append(pid)

        process = Mock(pid=123, returncode=0)
        process.poll.return_value = 0
        supervisor = RecordingSupervisor()
        controller = Mock()
        with tempfile.TemporaryDirectory(prefix="insertany3d_process_test_") as directory:
            staging = Path(directory) / "attempt" / "output.staging"
            staging.mkdir(parents=True)
            item = SimpleNamespace(
                staging_dir=staging,
                stage_id=1,
                lease_token="lease",
            )
            with patch("insertany3d.executors.subprocess.Popen", return_value=process), patch(
                "insertany3d.executors.os.getpgid", return_value=123
            ):
                result = CommandExecutor(process_supervisor=supervisor).execute(
                    controller, item, ["fake-command"]
                )

        self.assertFalse(result.succeeded)
        self.assertEqual(result.error_code, "compile_or_contract")
        self.assertEqual(supervisor.bound, [123])
        self.assertEqual(supervisor.released, [123])

    def test_windows_backend_binds_and_terminates_the_owned_tree(self) -> None:
        backend = FakeWindowsBackend()
        supervisor = ProcessSupervisor(
            platform_name="nt",
            windows_backend=backend,
            identity_checker=lambda _identity: True,
            grace_seconds=3.0,
            poll_seconds=0.2,
            py_spy_path="",
        )
        identity = ProcessIdentity(123, 123, "windows-process-time-v1", 456)

        supervisor.bind(identity.pid)
        result = supervisor.terminate(identity)
        supervisor.release(identity.pid)

        self.assertEqual(backend.bound, [123])
        self.assertEqual(backend.terminated, [(123, 3.0, 0.2)])
        self.assertEqual(backend.released, [123])
        self.assertTrue(result.completed)
        self.assertTrue(result.identity_matched)
        self.assertEqual(result.remaining_pids, ())

    def test_windows_backend_reports_remaining_descendants(self) -> None:
        backend = FakeWindowsBackend(remaining=(124, 125))
        supervisor = ProcessSupervisor(
            platform_name="nt",
            windows_backend=backend,
            identity_checker=lambda _identity: True,
            py_spy_path="",
        )
        result = supervisor.terminate(ProcessIdentity(123, 123, "boot", 456))

        self.assertFalse(result.completed)
        self.assertEqual(result.remaining_pids, (124, 125))

    def test_identity_mismatch_never_calls_windows_termination(self) -> None:
        backend = FakeWindowsBackend()
        supervisor = ProcessSupervisor(
            platform_name="nt",
            windows_backend=backend,
            identity_checker=lambda _identity: False,
            py_spy_path="",
        )
        result = supervisor.terminate(ProcessIdentity(123, 123, "old-boot", 456))

        self.assertTrue(result.completed)
        self.assertFalse(result.identity_matched)
        self.assertEqual(backend.terminated, [])

    def test_posix_termination_keeps_term_then_kill_behavior(self) -> None:
        supervisor = ProcessSupervisor(
            platform_name="posix",
            identity_checker=lambda _identity: True,
            grace_seconds=0.0,
            py_spy_path="",
        )
        identity = ProcessIdentity(123, 321, "boot", 456)
        with patch.object(supervisor, "_signal_group") as signal_group, patch.object(
            supervisor, "_wait_group", side_effect=[(123,), ()]
        ):
            result = supervisor.terminate(identity)

        self.assertEqual(
            signal_group.call_args_list,
            [call(321, signal.SIGTERM), call(321, signal.SIGKILL)],
        )
        self.assertTrue(result.completed)


if __name__ == "__main__":
    unittest.main()
