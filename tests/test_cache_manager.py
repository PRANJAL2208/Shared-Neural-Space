"""Test suite for the cache manager."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from src.preprocessing.cache_manager import CacheManager


@pytest.fixture
def tmp_cache(tmp_path):
    """Fresh temporary cache directory for each test."""
    cache = tmp_path / "raw"
    cache.mkdir()
    return cache


def _write_fake_recording(path: Path, size_bytes: int = 1024) -> Path:
    """Write a fake binary file to simulate a downloaded recording."""
    path.mkdir(parents=True, exist_ok=True)
    fake_file = path / "recording.bin"
    fake_file.write_bytes(b"\x00" * size_bytes)
    return path


class TestCacheManagerContextManager:
    """Verify that raw recordings are deleted after the context exits."""

    def test_cleanup_on_success(self, tmp_cache):
        """Recording directory is deleted after successful processing."""
        recording_dir = tmp_cache / "sub-01"
        _write_fake_recording(recording_dir)
        assert recording_dir.exists()

        manager = CacheManager(cache_dir=tmp_cache)
        with manager.recording("sub-01", recording_path=recording_dir):
            assert recording_dir.exists(), "Directory should exist inside context"

        assert not recording_dir.exists(), "Directory should be deleted after context"

    def test_cleanup_on_exception(self, tmp_cache):
        """Recording directory is still deleted when the processing block raises."""
        recording_dir = tmp_cache / "sub-02"
        _write_fake_recording(recording_dir)

        manager = CacheManager(cache_dir=tmp_cache)
        with pytest.raises(ValueError, match="simulated processing failure"):
            with manager.recording("sub-02", recording_path=recording_dir):
                raise ValueError("simulated processing failure")

        assert not recording_dir.exists(), (
            "Directory must be cleaned up even after an exception"
        )

    def test_cleanup_on_keyboard_interrupt(self, tmp_cache):
        """Recording directory is still deleted on KeyboardInterrupt."""
        recording_dir = tmp_cache / "sub-03"
        _write_fake_recording(recording_dir)

        manager = CacheManager(cache_dir=tmp_cache)
        with pytest.raises(KeyboardInterrupt):
            with manager.recording("sub-03", recording_path=recording_dir):
                raise KeyboardInterrupt

        assert not recording_dir.exists()

    def test_nonexistent_path_does_not_raise(self, tmp_cache):
        """Context manager tolerates a path that was never created."""
        ghost_path = tmp_cache / "sub-99"
        manager = CacheManager(cache_dir=tmp_cache)
        # Should not raise even though the path doesn't exist
        with manager.recording("sub-99", recording_path=ghost_path):
            pass

    def test_yields_none_when_no_path(self, tmp_cache):
        """When recording_path is None, context yields None."""
        manager = CacheManager(cache_dir=tmp_cache)
        with manager.recording("sub-01") as path:
            assert path is None


class TestCacheManagerDiskUsage:
    def test_disk_usage_reports_bytes(self, tmp_cache):
        manager = CacheManager(cache_dir=tmp_cache)
        # Empty cache
        assert manager.disk_usage_bytes() == 0

        # Write 4096 bytes
        fake = tmp_cache / "fake.bin"
        fake.write_bytes(b"\x00" * 4096)
        assert manager.disk_usage_bytes() == 4096

    def test_clear_all(self, tmp_cache):
        manager = CacheManager(cache_dir=tmp_cache)
        (tmp_cache / "a.bin").write_bytes(b"\x00" * 512)
        (tmp_cache / "b.bin").write_bytes(b"\x00" * 512)

        manager.clear_all()
        assert manager.disk_usage_bytes() == 0
        assert tmp_cache.exists(), "Cache dir itself should still exist after clear_all"
