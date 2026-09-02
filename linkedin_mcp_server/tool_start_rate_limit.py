"""Cross-process minimum spacing between MCP tool-call starts."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from linkedin_mcp_server.profile_lease import acquire_locked_fd
from linkedin_mcp_server.session_state import auth_root_dir

logger = logging.getLogger(__name__)

_STATE_FILE = "tool-start-rate-limit.lock"
_LOCK_POLL_SECONDS = 0.05


class ToolStartPermit:
    """A locked start slot that is recorded immediately before tool execution."""

    def __init__(self, fd: int | None, clock: Callable[[], float]) -> None:
        self._fd = fd
        self._clock = clock

    def commit(self) -> None:
        """Record the actual start and release the cross-process lock."""
        if self._fd is None:
            return
        ToolStartRateLimiter._write_timestamp(self._fd, self._clock())
        self.close()

    def close(self) -> None:
        """Release an uncommitted slot, for cancellation and failed calls."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class ToolStartRateLimiter:
    """Reserve tool-call starts at least ``interval_seconds`` apart.

    The timestamp and its kernel lock share one owner-only file under the auth
    root. Consequently independent MCP server processes using the same profile
    coordinate without holding the browser profile lease while they wait.
    """

    def __init__(
        self,
        interval_seconds: float,
        *,
        state_path: Path | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._interval = interval_seconds
        self._state_path = state_path
        self._clock = clock
        self._sleep = sleep

    @property
    def enabled(self) -> bool:
        return self._interval > 0

    def _path(self) -> Path:
        return self._state_path or auth_root_dir() / _STATE_FILE

    @staticmethod
    def _read_timestamp(fd: int) -> float | None:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 128)
        try:
            value = float(raw.decode("ascii").strip())
        except (UnicodeDecodeError, ValueError):
            return None
        return value if math.isfinite(value) and value >= 0 else None

    @staticmethod
    def _write_timestamp(fd: int, timestamp: float) -> None:
        encoded = f"{timestamp:.9f}\n".encode("ascii")
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, encoded)
        os.ftruncate(fd, len(encoded))
        # open_lock_file creates new files as 0600. Harden an old file too in
        # case it predates this code or was copied in with broader permissions.
        if os.name != "nt":
            os.fchmod(fd, 0o600)

    async def acquire(
        self,
        report_wait: Callable[[float], Awaitable[None]] | None = None,
    ) -> ToolStartPermit:
        """Wait cancellably and return a locked, uncommitted start slot."""
        if not self.enabled:
            return ToolStartPermit(None, self._clock)

        path = self._path()
        while True:
            fd = acquire_locked_fd(path, exclusive=True)
            if fd is None:
                await self._sleep(_LOCK_POLL_SECONDS)
                continue

            now = self._clock()
            previous = self._read_timestamp(fd)
            delay = 0.0
            if previous is not None:
                # A clock correction or a state file copied from a machine
                # whose clock was ahead must not create an unbounded wait.
                delay = min(
                    self._interval,
                    max(0.0, previous + self._interval - now),
                )
            if delay <= 0:
                return ToolStartPermit(fd, self._clock)
            os.close(fd)  # closing releases the kernel lock on every backend

            if report_wait is not None:
                await report_wait(delay)
            logger.debug("Waiting %.3fs for the minimum tool-start interval", delay)
            await self._sleep(delay)

    async def wait(
        self,
        report_wait: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        """Wait and reserve now; useful when the caller is itself the start."""
        permit = await self.acquire(report_wait)
        permit.commit()
