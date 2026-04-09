"""
tests/unit_static/test_event_log_extended.py — extended EventLog and event tests.

Covers behaviour not already exercised by test_eval_contract.py:

  * CacheUpgradeEvent ratio computation in __post_init__
  * UpgradeFailureEvent fields and to_dict
  * EventLog file-based flush (writes events.jsonl to tmp_path)
  * Incremental / append flush semantics (only new events are written)
  * EventLog.upgrade_count, failure_count, summary
  * EventLog.events returns a copy (not the internal list)

No MLX required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from turboquant.runtime.events import (
    CacheUpgradeEvent,
    EventLog,
    UpgradeFailureEvent,
)


# ── CacheUpgradeEvent ─────────────────────────────────────────────────────────


def test_cache_upgrade_event_type_is_upgrade() -> None:
    ev = CacheUpgradeEvent(layer_index=0, token_index=0, old_bytes=0, new_bytes=0)
    assert ev.event_type == "upgrade"


def test_cache_upgrade_event_ratio_computed_when_new_bytes_positive() -> None:
    ev = CacheUpgradeEvent(
        layer_index=0,
        token_index=64,
        old_bytes=131072,
        new_bytes=32768,
    )
    assert abs(ev.ratio - 4.0) < 1e-6


def test_cache_upgrade_event_ratio_zero_when_new_bytes_zero() -> None:
    ev = CacheUpgradeEvent(layer_index=0, token_index=0, old_bytes=131072, new_bytes=0)
    assert ev.ratio == 0.0


def test_cache_upgrade_event_ratio_rounded_to_4_places() -> None:
    ev = CacheUpgradeEvent(layer_index=0, token_index=0, old_bytes=100000, new_bytes=3)
    # 100000/3 = 33333.333… → rounded to 4 decimal places
    assert ev.ratio == round(100000 / 3, 4)


def test_cache_upgrade_event_to_dict_contains_expected_keys() -> None:
    ev = CacheUpgradeEvent(
        layer_index=2,
        token_index=128,
        old_type="KVCache",
        new_type="TurboQuantKCache",
        old_bytes=262144,
        new_bytes=65536,
    )
    d = ev.to_dict()
    assert d["event_type"] == "upgrade"
    assert d["layer_index"] == 2
    assert d["token_index"] == 128
    assert d["old_type"] == "KVCache"
    assert d["new_type"] == "TurboQuantKCache"
    assert d["old_bytes"] == 262144
    assert d["new_bytes"] == 65536
    assert "ratio" in d
    assert "timestamp_utc" in d


# ── UpgradeFailureEvent ───────────────────────────────────────────────────────


def test_upgrade_failure_event_type_is_upgrade_failure() -> None:
    ev = UpgradeFailureEvent(layer_index=1, token_index=32, reason="NaN in K scales")
    assert ev.event_type == "upgrade_failure"


def test_upgrade_failure_event_reverted_is_always_true() -> None:
    ev = UpgradeFailureEvent(layer_index=0, token_index=0, reason="scale == 0")
    assert ev.reverted is True


def test_upgrade_failure_event_to_dict_contains_expected_keys() -> None:
    ev = UpgradeFailureEvent(
        layer_index=3,
        token_index=256,
        reason="Inf in packed codes",
        exception_type="CompressionFailureError",
    )
    d = ev.to_dict()
    assert d["event_type"] == "upgrade_failure"
    assert d["layer_index"] == 3
    assert d["token_index"] == 256
    assert d["reason"] == "Inf in packed codes"
    assert d["reverted"] is True
    assert d["exception_type"] == "CompressionFailureError"
    assert "timestamp_utc" in d


def test_upgrade_failure_event_default_exception_type_is_empty_string() -> None:
    ev = UpgradeFailureEvent(layer_index=0, token_index=0, reason="unknown")
    assert ev.exception_type == ""


# ── EventLog — in-memory mode ─────────────────────────────────────────────────


def test_event_log_starts_empty() -> None:
    log = EventLog(artifact_dir=None)
    assert log.events == []
    assert log.upgrade_count() == 0
    assert log.failure_count() == 0


def test_event_log_records_upgrade_event() -> None:
    log = EventLog(artifact_dir=None)
    log.record(CacheUpgradeEvent(layer_index=0, token_index=64, old_bytes=0, new_bytes=0))
    assert len(log.events) == 1
    assert log.upgrade_count() == 1
    assert log.failure_count() == 0


def test_event_log_records_failure_event() -> None:
    log = EventLog(artifact_dir=None)
    log.record(UpgradeFailureEvent(layer_index=1, token_index=128, reason="bad"))
    assert len(log.events) == 1
    assert log.failure_count() == 1
    assert log.upgrade_count() == 0


def test_event_log_records_mixed_events() -> None:
    log = EventLog(artifact_dir=None)
    log.record(CacheUpgradeEvent(layer_index=0, token_index=64, old_bytes=0, new_bytes=0))
    log.record(UpgradeFailureEvent(layer_index=1, token_index=128, reason="err"))
    log.record(CacheUpgradeEvent(layer_index=2, token_index=200, old_bytes=0, new_bytes=0))
    assert log.upgrade_count() == 2
    assert log.failure_count() == 1


def test_event_log_summary_keys() -> None:
    log = EventLog(artifact_dir=None)
    log.record(CacheUpgradeEvent(layer_index=0, token_index=0, old_bytes=0, new_bytes=0))
    summary = log.summary()
    assert "upgrades" in summary
    assert "failures" in summary
    assert "total_events" in summary
    assert summary["upgrades"] == 1
    assert summary["failures"] == 0
    assert summary["total_events"] == 1


def test_event_log_flush_returns_none_when_no_artifact_dir() -> None:
    log = EventLog(artifact_dir=None)
    log.record(CacheUpgradeEvent(layer_index=0, token_index=0, old_bytes=0, new_bytes=0))
    result = log.flush()
    assert result is None


def test_event_log_flush_no_artifact_dir_events_still_accessible() -> None:
    log = EventLog(artifact_dir=None)
    ev = CacheUpgradeEvent(layer_index=0, token_index=0, old_bytes=0, new_bytes=0)
    log.record(ev)
    log.flush()
    assert len(log.events) == 1


def test_event_log_events_returns_a_copy() -> None:
    log = EventLog(artifact_dir=None)
    log.record(CacheUpgradeEvent(layer_index=0, token_index=0, old_bytes=0, new_bytes=0))
    copy1 = log.events
    copy2 = log.events
    assert copy1 is not copy2


# ── EventLog — file-based flush ───────────────────────────────────────────────


def test_event_log_flush_creates_events_jsonl(tmp_path: Path) -> None:
    log = EventLog(artifact_dir=tmp_path)
    log.record(
        CacheUpgradeEvent(
            layer_index=0,
            token_index=64,
            old_bytes=131072,
            new_bytes=32768,
        )
    )
    out = log.flush()
    assert out is not None
    assert out.name == "events.jsonl"
    assert out.exists()


def test_event_log_flush_writes_valid_json_lines(tmp_path: Path) -> None:
    log = EventLog(artifact_dir=tmp_path)
    log.record(
        CacheUpgradeEvent(
            layer_index=0,
            token_index=64,
            old_bytes=131072,
            new_bytes=32768,
        )
    )
    log.flush()
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["event_type"] == "upgrade"
    assert parsed["layer_index"] == 0


def test_event_log_flush_appends_incrementally(tmp_path: Path) -> None:
    """Second flush must append only the new events, not re-write old ones."""
    log = EventLog(artifact_dir=tmp_path)
    log.record(
        CacheUpgradeEvent(layer_index=0, token_index=64, old_bytes=100, new_bytes=10)
    )
    log.flush()  # first flush — 1 event written

    log.record(
        UpgradeFailureEvent(layer_index=1, token_index=128, reason="NaN")
    )
    log.flush()  # second flush — only the new failure event is appended

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2, "Both events should be present after two flushes"
    types = [json.loads(ln)["event_type"] for ln in lines]
    assert types == ["upgrade", "upgrade_failure"]


def test_event_log_flush_returns_none_when_no_new_events(tmp_path: Path) -> None:
    log = EventLog(artifact_dir=tmp_path)
    log.record(
        CacheUpgradeEvent(layer_index=0, token_index=0, old_bytes=0, new_bytes=0)
    )
    log.flush()
    result = log.flush()  # second call — nothing new
    assert result is None


def test_event_log_flush_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    log = EventLog(artifact_dir=nested)
    log.record(
        CacheUpgradeEvent(layer_index=0, token_index=0, old_bytes=0, new_bytes=0)
    )
    log.flush()
    assert (nested / "events.jsonl").exists()


def test_event_log_failure_event_written_to_jsonl(tmp_path: Path) -> None:
    log = EventLog(artifact_dir=tmp_path)
    log.record(
        UpgradeFailureEvent(
            layer_index=2,
            token_index=256,
            reason="scale == 0",
            exception_type="CompressionFailureError",
        )
    )
    log.flush()
    parsed = json.loads(
        (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip()
    )
    assert parsed["event_type"] == "upgrade_failure"
    assert parsed["reason"] == "scale == 0"
    assert parsed["reverted"] is True
