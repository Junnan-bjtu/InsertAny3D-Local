"""APIYi image request classification and a token/model-scoped local gate."""

from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .store import SchedulerStore
from .processes import current_boot_id, process_start_ticks


SAFE_RESPONSE_HEADERS = frozenset(
    {
        "retry-after",
        "x-request-id",
        "request-id",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
    }
)


class ConcurrentControllerError(RuntimeError):
    pass


@dataclass
class _LocalGate:
    condition: threading.Condition
    in_flight: int = 0


_GATES_LOCK = threading.Lock()
_GATES: dict[tuple[str, str], _LocalGate] = {}


@dataclass(frozen=True)
class ApiOutcome:
    status: str
    error_code: str | None
    retryable: bool
    delivery_unknown: bool
    http_status: int | None
    retry_after_seconds: float | None
    response_headers: dict[str, str]
    body: bytes | None = None


def token_fingerprint(token: str) -> str:
    """Return a non-secret stable queue key; the token itself is never stored."""
    if not token:
        raise ValueError("API token 不能为空")
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def bucket_key(token: str, model: str) -> str:
    if not model.strip():
        raise ValueError("model 不能为空")
    return f"{token_fingerprint(token)}:{model}"


def classify_http(status: int, headers: Mapping[str, str] | None = None, body: bytes | None = None) -> ApiOutcome:
    safe = _safe_headers(headers or {})
    retry_after = _retry_after(safe.get("retry-after"))
    if 200 <= status < 300:
        return ApiOutcome("succeeded", None, False, False, status, None, safe, body)
    if status == 429:
        return ApiOutcome("failed_retryable", "http_429", True, False, status, retry_after, safe, body)
    if status == 503 or status >= 500:
        return ApiOutcome("failed_retryable", "http_503", True, False, status, retry_after, safe, body)
    if status in {400, 403}:
        return ApiOutcome("failed_terminal", f"http_{status}", False, False, status, None, safe, body)
    return ApiOutcome("failed_terminal", f"http_{status}", False, False, status, None, safe, body)


class ImageApiClient:
    """A synchronous JSON client with no automatic retry after uncertain delivery."""

    def __init__(self, *, timeout_seconds: float = 360.0):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")
        self.timeout_seconds = timeout_seconds

    def post_json(self, endpoint: str, token: str, payload: Mapping[str, Any]) -> ApiOutcome:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "x-goog-api-key": token,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
                return classify_http(int(response.status), response.headers, body)
        except urllib.error.HTTPError as exc:
            return classify_http(exc.code, exc.headers, exc.read())
        except (TimeoutError, socket.timeout):
            return ApiOutcome("waiting_manual", "delivery_unknown", False, True, None, None, {}, None)
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                return ApiOutcome("waiting_manual", "delivery_unknown", False, True, None, None, {}, None)
            return ApiOutcome("failed_retryable", "transient_network", True, False, None, None, {}, None)
        except (ConnectionError, OSError):
            # The server may have accepted bytes before disconnecting.  Treat a
            # broken established request as potentially billed, not as 429.
            return ApiOutcome("waiting_manual", "delivery_unknown", False, True, None, None, {}, None)


class AdaptiveImageQueue:
    """Local concurrency control isolated by API token fingerprint and model."""

    def __init__(
        self,
        store: SchedulerStore,
        token: str,
        model: str,
        *,
        initial_limit: int = 4,
        maximum_limit: int = 24,
        mode: str = "adaptive",
        fixed_limit: int | None = None,
        clean_window: int = 20,
        clock: Callable[[], float] = time.time,
        jitter: Callable[[float, float], float] = random.uniform,
    ):
        if mode not in {"fixed", "adaptive"}:
            raise ValueError("API 并发模式必须是 fixed 或 adaptive")
        if mode == "fixed":
            fixed_limit = initial_limit if fixed_limit is None else fixed_limit
            initial_limit = maximum_limit = fixed_limit
        if not 0 < initial_limit <= maximum_limit:
            raise ValueError("并发必须满足 0 < initial_limit <= maximum_limit")
        self.store = store
        self.fingerprint = token_fingerprint(token)
        self.model = model
        self.key = f"{self.fingerprint}:{model}"
        self.clean_window = clean_window
        self.mode = mode
        self.clock = clock
        self.jitter = jitter
        registry_key = (str(store.path.resolve()), self.key)
        with _GATES_LOCK:
            self._gate = _GATES.setdefault(registry_key, _LocalGate(threading.Condition()))
        now = clock()
        pid = os.getpid()
        boot_id = current_boot_id()
        start_ticks = process_start_ticks(pid) or 0
        owner_token = f"{boot_id}:{pid}:{start_ticks}"
        with store.transaction() as connection:
            connection.execute(
                """INSERT INTO api_buckets(bucket_key, token_fingerprint, model, current_limit, maximum_limit, updated_at)
                   VALUES(?, ?, ?, ?, ?, ?)
                   ON CONFLICT(bucket_key) DO NOTHING""",
                (self.key, self.fingerprint, model, initial_limit, maximum_limit, now),
            )
            owner = connection.execute("SELECT * FROM api_bucket_owners WHERE bucket_key=?", (self.key,)).fetchone()
            if owner is not None and owner["owner_token"] != owner_token:
                owner_alive = (
                    owner["host_boot_id"] == current_boot_id()
                    and process_start_ticks(int(owner["pid"])) == int(owner["process_start_ticks"])
                )
                if owner_alive:
                    raise ConcurrentControllerError(
                        f"API 队列 {self.key} 已由进程 {owner['pid']} 控制；多进程并发会突破同一令牌额度"
                    )
            connection.execute(
                """INSERT INTO api_bucket_owners(
                       bucket_key, owner_token, pid, host_boot_id, process_start_ticks, updated_at
                   ) VALUES(?, ?, ?, ?, ?, ?)
                   ON CONFLICT(bucket_key) DO UPDATE SET owner_token=excluded.owner_token, pid=excluded.pid,
                     host_boot_id=excluded.host_boot_id, process_start_ticks=excluded.process_start_ticks,
                     updated_at=excluded.updated_at""",
                (self.key, owner_token, pid, boot_id, start_ticks, now),
            )

    @property
    def state(self) -> dict[str, Any]:
        row = self.store.row("SELECT * FROM api_buckets WHERE bucket_key=?", (self.key,))
        assert row is not None
        value = dict(row)
        value["in_flight"] = self._gate.in_flight
        return value

    def acquire(self, *, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._gate.condition:
            while True:
                state = self.state
                now = self.clock()
                if now >= float(state["cooldown_until"]) and self._gate.in_flight < int(state["current_limit"]):
                    self._gate.in_flight += 1
                    return True
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._gate.condition.wait(min(remaining, 0.1) if remaining is not None else 0.1)

    def release(self, outcome: ApiOutcome, *, attempt_index: int = 0) -> None:
        now = self.clock()
        with self.store.transaction() as connection:
            row = connection.execute("SELECT * FROM api_buckets WHERE bucket_key=?", (self.key,)).fetchone()
            assert row is not None
            limit = int(row["current_limit"])
            clean = int(row["clean_successes"])
            cooldown = float(row["cooldown_until"])
            if outcome.error_code == "http_429":
                limit = max(1, limit // 2)
                clean = 0
                base = outcome.retry_after_seconds
                if base is None:
                    base = (5.0, 15.0, 45.0)[min(max(attempt_index, 0), 2)]
                cooldown = max(cooldown, now + base + self.jitter(0.0, min(1.0, base * 0.1)))
            elif outcome.error_code == "http_503":
                clean = 0
                base = outcome.retry_after_seconds or (5.0, 15.0, 45.0)[min(max(attempt_index, 0), 2)]
                cooldown = max(cooldown, now + base + self.jitter(0.0, min(1.0, base * 0.1)))
            elif outcome.status == "succeeded" and self.mode == "adaptive":
                clean += 1
                if clean >= self.clean_window and limit < int(row["maximum_limit"]):
                    limit += 1
                    clean = 0
            else:
                clean = 0
            connection.execute(
                "UPDATE api_buckets SET current_limit=?, clean_successes=?, cooldown_until=?, updated_at=? WHERE bucket_key=?",
                (limit, clean, cooldown, now, self.key),
            )
        with self._gate.condition:
            self._gate.in_flight = max(0, self._gate.in_flight - 1)
            self._gate.condition.notify_all()


def _safe_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items() if str(key).lower() in SAFE_RESPONSE_HEADERS}


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
