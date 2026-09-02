from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from insertany3d.apiyi import (
    AdaptiveImageQueue,
    ApiOutcome,
    ConcurrentControllerError,
    ImageApiClient,
    bucket_key,
    token_fingerprint,
)
from insertany3d.processes import current_boot_id, process_start_ticks
from insertany3d.store import SchedulerStore


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.path == "/timeout":
            time.sleep(0.2)
            self._send(200)
        elif self.path == "/drop":
            self.connection.close()
        elif self.path == "/429":
            self._send(429, {"Retry-After": "7", "X-Request-Id": "fake-429", "Authorization": "must-not-persist"})
        elif self.path == "/503":
            self._send(503)
        elif self.path == "/400":
            self._send(400)
        elif self.path == "/403":
            self._send(403)
        else:
            self._send(200)

    def _send(self, status: int, headers=None):
        body = json.dumps({"status": status}).encode()
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def log_message(self, *_args):
        return


class ApiYiTests(unittest.TestCase):
    def test_local_fake_api_classifies_status_and_uncertain_delivery(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            client = ImageApiClient(timeout_seconds=0.05)
            self.assertEqual(client.post_json(base + "/200", "secret", {}).status, "succeeded")
            limited = client.post_json(base + "/429", "secret", {})
            self.assertEqual(limited.error_code, "http_429")
            self.assertEqual(limited.retry_after_seconds, 7)
            self.assertEqual(limited.response_headers, {"retry-after": "7", "x-request-id": "fake-429"})
            self.assertEqual(client.post_json(base + "/503", "secret", {}).error_code, "http_503")
            self.assertFalse(client.post_json(base + "/400", "secret", {}).retryable)
            self.assertFalse(client.post_json(base + "/403", "secret", {}).retryable)
            self.assertTrue(client.post_json(base + "/timeout", "secret", {}).delivery_unknown)
            self.assertTrue(client.post_json(base + "/drop", "secret", {}).delivery_unknown)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_adaptive_queue_isolated_by_token_and_model_and_halves_on_429(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_apiyi_") as directory:
            with SchedulerStore(Path(directory) / "state.sqlite3") as store:
                queue = AdaptiveImageQueue(store, "secret-a", "image-model", initial_limit=4, maximum_limit=24, jitter=lambda _a, _b: 0)
                other = AdaptiveImageQueue(store, "secret-b", "image-model", initial_limit=4, maximum_limit=24)
                self.assertNotIn(token_fingerprint("secret-a"), {"secret-a", "secret-b"})
                self.assertNotEqual(bucket_key("secret-a", "image-model"), other.key)
                self.assertTrue(queue.acquire(timeout=0))
                queue.release(ApiOutcome("failed_retryable", "http_429", True, False, 429, 7, {}))
                state = queue.state
                self.assertEqual(state["current_limit"], 2)
                self.assertGreater(state["cooldown_until"], 0)
                self.assertEqual(other.state["current_limit"], 4)

    def test_adaptive_queue_enforces_limit_across_threads(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_apiyi_threads_") as directory:
            with SchedulerStore(Path(directory) / "state.sqlite3") as store:
                queue = AdaptiveImageQueue(store, "secret", "image-model", initial_limit=2, maximum_limit=2)
                lock = threading.Lock()
                active = 0
                peak = 0
                errors = []

                def run() -> None:
                    nonlocal active, peak
                    try:
                        if not queue.acquire(timeout=2):
                            raise RuntimeError("timed out waiting for fake API slot")
                        with lock:
                            active += 1
                            peak = max(peak, active)
                        time.sleep(0.01)
                        with lock:
                            active -= 1
                        queue.release(ApiOutcome("succeeded", None, False, False, 200, None, {}))
                    except BaseException as exc:
                        errors.append(exc)

                threads = [threading.Thread(target=run) for _ in range(8)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=3)
                self.assertFalse(any(thread.is_alive() for thread in threads))
                self.assertEqual(errors, [])
                self.assertEqual(peak, 2)

    def test_same_token_model_instances_share_gate_and_foreign_owner_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_apiyi_owner_") as directory:
            with SchedulerStore(Path(directory) / "state.sqlite3") as store:
                first = AdaptiveImageQueue(store, "secret", "model", initial_limit=1, maximum_limit=1)
                second = AdaptiveImageQueue(store, "secret", "model", initial_limit=1, maximum_limit=1)
                self.assertTrue(first.acquire(timeout=0))
                self.assertFalse(second.acquire(timeout=0))
                first.release(ApiOutcome("succeeded", None, False, False, 200, None, {}))
                self.assertTrue(second.acquire(timeout=0))
                second.release(ApiOutcome("succeeded", None, False, False, 200, None, {}))
                with store.transaction() as connection:
                    connection.execute(
                        """UPDATE api_bucket_owners SET owner_token='foreign', pid=?, host_boot_id=?,
                           process_start_ticks=? WHERE bucket_key=?""",
                        (os.getpid(), current_boot_id(), process_start_ticks(os.getpid()), first.key),
                    )
                with self.assertRaises(ConcurrentControllerError):
                    AdaptiveImageQueue(store, "secret", "model", initial_limit=1, maximum_limit=1)

    def test_fixed_mode_does_not_ramp_after_clean_successes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_apiyi_fixed_") as directory:
            with SchedulerStore(Path(directory) / "state.sqlite3") as store:
                queue = AdaptiveImageQueue(store, "fixed-secret", "model", mode="fixed", fixed_limit=3)
                for _ in range(25):
                    self.assertTrue(queue.acquire(timeout=0))
                    queue.release(ApiOutcome("succeeded", None, False, False, 200, None, {}))
                self.assertEqual(queue.state["current_limit"], 3)

    def test_adaptive_mode_defaults_to_three_and_caps_at_five(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_apiyi_defaults_") as directory:
            with SchedulerStore(Path(directory) / "state.sqlite3") as store:
                queue = AdaptiveImageQueue(store, "default-secret", "model", initial_limit=3, maximum_limit=5, clean_window=1)
                for _ in range(3):
                    self.assertTrue(queue.acquire(timeout=0))
                    queue.release(ApiOutcome("succeeded", None, False, False, 200, None, {}))
                self.assertEqual(queue.state["current_limit"], 5)


if __name__ == "__main__":
    unittest.main()
