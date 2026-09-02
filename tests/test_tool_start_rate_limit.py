from __future__ import annotations

import asyncio
import multiprocessing
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import mcp.types as mt
import pytest
from fastmcp.server.middleware import MiddlewareContext

from linkedin_mcp_server.sequential_tool_middleware import (
    SequentialToolExecutionMiddleware,
)
from linkedin_mcp_server.tool_start_rate_limit import (
    ToolStartRateLimiter,
    ToolStartRateLimitStateError,
)


def _reserve_in_process(state_path: str, interval: float, ready, output) -> None:
    async def run() -> None:
        ready.wait()
        limiter = ToolStartRateLimiter(interval, state_path=Path(state_path))
        await limiter.wait()
        output.put(time.time())

    asyncio.run(run())


class TestToolStartRateLimiter:
    async def test_disabled_does_not_create_state(self, tmp_path):
        path = tmp_path / "rate.lock"

        await ToolStartRateLimiter(0, state_path=path).wait()

        assert not path.exists()

    async def test_sequential_calls_wait_for_the_remaining_interval(self, tmp_path):
        now = [10.0]
        sleeps: list[float] = []

        async def sleep(delay: float) -> None:
            sleeps.append(delay)
            now[0] += delay

        limiter = ToolStartRateLimiter(
            2.0,
            state_path=tmp_path / "rate.lock",
            clock=lambda: now[0],
            sleep=sleep,
        )

        await limiter.wait()
        await limiter.wait()

        assert sleeps == [2.0]
        assert now[0] == 12.0

    async def test_symlink_state_fails_closed_without_touching_target(self, tmp_path):
        victim = tmp_path / "victim.txt"
        victim.write_text("do not change", encoding="utf-8")
        state = tmp_path / "rate.lock"
        try:
            state.symlink_to(victim)
        except OSError as exc:
            pytest.skip(f"symlinks are unavailable: {exc}")

        limiter = ToolStartRateLimiter(1, state_path=state)

        with pytest.raises(ToolStartRateLimitStateError, match="link|reparse"):
            await limiter.wait()
        assert victim.read_text(encoding="utf-8") == "do not change"

    async def test_future_timestamp_is_reset_without_repeated_sleep(self, tmp_path):
        state = tmp_path / "rate.lock"
        state.write_text("100.0\n", encoding="ascii")

        async def unexpected_sleep(_delay: float) -> None:
            pytest.fail("a future timestamp must not cause a retry loop")

        limiter = ToolStartRateLimiter(
            5,
            state_path=state,
            clock=lambda: 10.0,
            sleep=unexpected_sleep,
        )

        await asyncio.wait_for(limiter.wait(), timeout=1)

        assert float(state.read_text(encoding="ascii")) == 10.0

    async def test_concurrent_calls_are_spaced(self, tmp_path):
        limiter = ToolStartRateLimiter(0.05, state_path=tmp_path / "rate.lock")

        async def reserve() -> float:
            await limiter.wait()
            return time.monotonic()

        starts = sorted(await asyncio.gather(reserve(), reserve()))

        assert starts[1] - starts[0] >= 0.04

    async def test_wait_is_cancellable(self, tmp_path):
        limiter = ToolStartRateLimiter(30, state_path=tmp_path / "rate.lock")
        await limiter.wait()
        waiting = asyncio.Event()

        async def report_wait(_delay: float) -> None:
            waiting.set()

        task = asyncio.create_task(limiter.wait(report_wait))
        await asyncio.wait_for(waiting.wait(), timeout=1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.skipif(os.name == "nt", reason="spawn timing is too noisy on CI")
    def test_processes_sharing_an_auth_root_coordinate(self, tmp_path):
        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        output = context.Queue()
        path = tmp_path / "rate.lock"
        processes = [
            context.Process(
                target=_reserve_in_process,
                args=(str(path), 0.25, ready, output),
            )
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        ready.set()
        starts = sorted(output.get(timeout=10) for _ in processes)
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0

        assert starts[1] - starts[0] >= 0.20
        assert path.stat().st_mode & 0o777 == 0o600


class TestRateLimitedSequentialMiddleware:
    @staticmethod
    def _context():
        fastmcp_context = MagicMock()
        fastmcp_context.request_context = object()
        fastmcp_context.report_progress = AsyncMock()
        return MiddlewareContext(
            message=mt.CallToolRequestParams(name="test_tool", arguments={}),
            method="tools/call",
            fastmcp_context=fastmcp_context,
        )

    async def test_rate_wait_precedes_profile_ownership(self, monkeypatch):
        middleware = SequentialToolExecutionMiddleware(min_tool_interval_seconds=1)
        order: list[str] = []

        permit = MagicMock()

        async def acquire(_report) -> MagicMock:
            order.append("rate_wait")
            return permit

        async def run_profile(*_args, on_start) -> MagicMock:
            order.append("profile")
            on_start()
            return MagicMock()

        monkeypatch.setattr(middleware._start_limiter, "acquire", acquire)
        monkeypatch.setattr(middleware, "_run_owning_the_profile", run_profile)

        await middleware.on_call_tool(self._context(), AsyncMock())

        assert order == ["rate_wait", "profile"]
        permit.commit.assert_called_once_with()
        permit.close.assert_called_once_with()

    async def test_reports_rate_limit_wait(self, monkeypatch):
        middleware = SequentialToolExecutionMiddleware(min_tool_interval_seconds=1)
        context = self._context()

        permit = MagicMock()

        async def acquire(report) -> MagicMock:
            await report(0.75)
            return permit

        monkeypatch.setattr(middleware._start_limiter, "acquire", acquire)
        monkeypatch.setattr(
            middleware,
            "_run_owning_the_profile",
            AsyncMock(return_value=MagicMock()),
        )

        await middleware.on_call_tool(context, AsyncMock())

        context.fastmcp_context.report_progress.assert_any_await(
            progress=0,
            total=100,
            message="Minimum tool interval active; waiting 0.8s before starting",
        )
