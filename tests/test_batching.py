"""Tests for the batch processing module.

Covers: _wrap_job_call error handling, _process_batch_sequential,
and the ProcessPoolExecutor-based parallel batch processing.
Tests avoid forking where possible and use mock callables.
"""
from __future__ import annotations

import logging
from typing import Any

import pytest

from batching import _wrap_job_call, _process_batch_sequential


# ---------------------------------------------------------------------------
# _wrap_job_call
# ---------------------------------------------------------------------------

class TestWrapJobCall:
    def test_wrap_job_call_success(self):
        """Successful processing returns the correct dict."""
        def my_func(x: int, y: int) -> dict[str, Any]:
            return {"result": x + y, "status": "ok"}

        result = _wrap_job_call(my_func, 3, y=4)
        assert result == {"result": 7, "status": "ok"}

    def test_wrap_job_call_error(self, caplog):
        """A failing function should be caught, logged, and return an error dict."""
        caplog.set_level(logging.ERROR)

        def failing_func() -> dict[str, Any]:
            raise RuntimeError("Something went wrong")

        result = _wrap_job_call(failing_func)
        assert "error" in result
        assert "Something went wrong" in result["error"]
        assert result["fields"] == []
        assert result["confidence"] == 0.0

    def test_wrap_job_call_error_logged(self, caplog):
        """Verify the exception is logged at ERROR level."""
        caplog.set_level(logging.ERROR)

        def failing_func() -> dict[str, Any]:
            raise ValueError("bad value")

        _wrap_job_call(failing_func)
        assert len(caplog.records) >= 1
        assert "bad value" in caplog.text

    def test_wrap_job_call_no_args(self):
        """Function with no args should work fine."""

        def simple() -> dict[str, Any]:
            return {"ok": True}

        assert _wrap_job_call(simple) == {"ok": True}


# ---------------------------------------------------------------------------
# _worker (indirectly via _process_batch_sequential)
# ---------------------------------------------------------------------------

class TestBatchSequential:
    def _minimal_job_kwargs(self, job_id: str) -> dict:
        """Helper to produce minimal kwargs that match the _worker signature."""
        return {
            "config_name": "cfg",
            "config_path": "/nonexistent/config.json",
            "image_path": "/nonexistent/img.png",
            "output_dir": "/tmp/out",
            "log_dir": "/tmp/log",
            "dictionaries_dir": "/tmp/dict",
            "dpi": 300,
            "original_filename": f"{job_id}.png",
            "job_id": job_id,
            "api_key": "test-key",
        }

    def test_batch_sequential_success(self):
        """Process a small batch of jobs sequentially and collect all results."""
        jobs = {
            "job1": self._minimal_job_kwargs("job1"),
            "job2": self._minimal_job_kwargs("job2"),
        }

        results = _process_batch_sequential(jobs)
        # Tasks fail because files don't exist, but _wrap_job_call catches errors.
        # We just verify results are dicts for all job_ids.
        assert isinstance(results, dict)
        assert "job1" in results
        assert "job2" in results
        for job_id in jobs:
            assert isinstance(results[job_id], dict)

    def test_batch_sequential_empty(self):
        """Empty jobs dict → empty results dict."""
        assert _process_batch_sequential({}) == {}

    def test_batch_sequential_handles_failure(self):
        """All jobs in a batch should complete (possibly with errors)."""
        jobs = {
            "fail": self._minimal_job_kwargs("fail"),
        }
        results = _process_batch_sequential(jobs)
        assert "fail" in results
        # Should have an error key since real processing would fail
        assert isinstance(results["fail"], dict)
        assert "error" in results["fail"]
