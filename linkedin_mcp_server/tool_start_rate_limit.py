"""Cross-process minimum spacing between MCP tool-call starts."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import stat
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from linkedin_mcp_server.common_utils import is_still_at
from linkedin_mcp_server.profile_lease import acquire_locked_fd
from linkedin_mcp_server.session_state import auth_root_dir

logger = logging.getLogger(__name__)

_STATE_FILE = "tool-start-rate-limit.lock"
_LOCK_POLL_SECONDS = 0.05


class ToolStartRateLimitStateError(RuntimeError):
    """The pacing state path is not a safe regular file."""


def _validate_state_file(fd: int, path: Path) -> None:
    details = os.fstat(fd)
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise ToolStartRateLimitStateError(
            f"Tool-start pacing state {path} must be a single-link regular file"
        )
    if os.name != "nt" and details.st_uid != os.geteuid():
        raise ToolStartRateLimitStateError(
            f"Tool-start pacing state {path} is not owned by the current user"
        )
    if not is_still_at(fd, path):
        raise ToolStartRateLimitStateError(
            f"Tool-start pacing state {path} was replaced while it was opened"
        )


def _acquire_state_file(path: Path) -> int | None:
    """Reject a link at the owned state path, then use the project lock helper."""
    try:
        entry = path.lstat()
    except FileNotFoundError:
        entry = None
    if entry is not None:
        attributes = getattr(entry, "st_file_attributes", 0)
        reparse = attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if stat.S_ISLNK(entry.st_mode) or reparse:
            raise ToolStartRateLimitStateError(
                f"Tool-start pacing state {path} is a link or reparse point"
            )

    fd = acquire_locked_fd(path, exclusive=True)
    if fd is None:
        return None
    try:
        _validate_state_file(fd, path)
        return fd
    except BaseException:
        os.close(fd)
        raise


class ToolStartPermit:
    """A locked start slot that is recorded immediately before tool execution."""

    def __init__(
        self,
        fd: int | None,
        clock: Callable[[], float],
        path: Path | None = None,
    ) -> None:
        self._fd = fd
        self._clock = clock
        self._path = path

    def commit(self) -> None:
        """Record the actual start and release the cross-process lock."""
        if self._fd is None:
            return
        assert self._path is not None
        _validate_state_file(self._fd, self._path)
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
            fd = _acquire_state_file(path)
            if fd is None:
                await self._sleep(_LOCK_POLL_SECONDS)
                continue

            now = self._clock()
            previous = self._read_timestamp(fd)
            delay = 0.0
            if previous is not None and previous <= now:
                delay = max(0.0, previous + self._interval - now)
            # A future timestamp means the wall clock moved backwards. Reset it
            # on this start instead of rereading it after an unbounded series of
            # interval-sized sleeps.
            if delay <= 0:
                return ToolStartPermit(fd, self._clock, path)
            os.close(fd)

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
