"""
tests/unit_static/test_metrics_tracker.py — MetricsTracker unit tests.

Covers:
  * Initial state (all counters zero)
  * set_dense_bytes / set_compressed_bytes
  * record_step accumulation (tokens, latency, peak latency)
  * Derived properties: ratio, tok_per_sec, avg_latency_ms
  * to_dict structure and key presence
  * write() produces metrics.json in the expected location
  * write() with an EventLog flushes events.jsonl and includes summary

No MLX required.
"""

from __future__ import annotations

import json
from pathlib import Path

from turboquant.metrics.tracker import MetricsTracker
from turboquant.runtime.events import CacheUpgradeEvent, EventLog


# ── Initial state ─────────────────────────────────────────────────────────────


def test_metrics_tracker_initial_ratio_is_zero() -> None:
    tracker = MetricsTracker()
    assert tracker.ratio == 0.0


def test_metrics_tracker_initial_tok_per_sec_is_zero() -> None:
    tracker = MetricsTracker()
    assert tracker.tok_per_sec == 0.0


def test_metrics_tracker_initial_avg_latency_is_zero() -> None:
    tracker = MetricsTracker()
    assert tracker.avg_latency_ms == 0.0


# ── Byte measurements ─────────────────────────────────────────────────────────


def test_set_dense_bytes() -> None:
    tracker = MetricsTracker()
    tracker.set_dense_bytes(1_000_000)
    d = tracker.to_dict()
    assert d["dense_bytes"] == 1_000_000


def test_set_compressed_bytes() -> None:
    tracker = MetricsTracker()
    tracker.set_compressed_bytes(250_000)
    d = tracker.to_dict()
    assert d["compressed_bytes"] == 250_000


def test_ratio_is_dense_over_compressed() -> None:
    tracker = MetricsTracker()
    tracker.set_dense_bytes(1_000_000)
    tracker.set_compressed_bytes(250_000)
    assert abs(tracker.ratio - 4.0) < 1e-6


def test_ratio_is_zero_when_compressed_bytes_zero() -> None:
    tracker = MetricsTracker()
    tracker.set_dense_bytes(500_000)
    tracker.set_compressed_bytes(0)
    assert tracker.ratio == 0.0


def test_ratio_is_rounded_to_4_places() -> None:
    tracker = MetricsTracker()
    tracker.set_dense_bytes(100_000)
    tracker.set_compressed_bytes(3)
    assert tracker.ratio == round(100_000 / 3, 4)


# ── record_step ───────────────────────────────────────────────────────────────


def test_record_step_accumulates_tokens() -> None:
    tracker = MetricsTracker()
    tracker.record_step(tokens_generated=5, latency_ms=10.0)
    tracker.record_step(tokens_generated=3, latency_ms=8.0)
    d = tracker.to_dict()
    assert d["total_tokens"] == 8
    assert d["total_steps"] == 2


def test_record_step_accumulates_latency() -> None:
    tracker = MetricsTracker()
    tracker.record_step(tokens_generated=1, latency_ms=10.0)
    tracker.record_step(tokens_generated=1, latency_ms=20.0)
    assert abs(tracker.avg_latency_ms - 15.0) < 1e-6


def test_record_step_tracks_peak_latency() -> None:
    tracker = MetricsTracker()
    tracker.record_step(tokens_generated=1, latency_ms=5.0)
    tracker.record_step(tokens_generated=1, latency_ms=50.0)
    tracker.record_step(tokens_generated=1, latency_ms=3.0)
    d = tracker.to_dict()
    assert abs(d["peak_latency_ms"] - 50.0) < 1e-6


def test_tok_per_sec_computed_correctly() -> None:
    tracker = MetricsTracker()
    tracker.record_step(tokens_generated=10, latency_ms=500.0)
    # 10 tokens / 0.5 s = 20 tok/s
    assert abs(tracker.tok_per_sec - 20.0) < 0.01


def test_tok_per_sec_zero_when_no_steps() -> None:
    tracker = MetricsTracker()
    assert tracker.tok_per_sec == 0.0


# ── to_dict ───────────────────────────────────────────────────────────────────


def test_to_dict_contains_all_required_keys() -> None:
    tracker = MetricsTracker(run_id="test-run", model="llama")
    d = tracker.to_dict()
    required = {
        "run_id", "model", "config_fingerprint",
        "dense_bytes", "compressed_bytes", "ratio",
        "tok_per_sec", "avg_latency_ms", "peak_latency_ms",
        "total_tokens", "total_steps",
        "mlx_version", "elapsed_s",
    }
    assert required.issubset(d.keys())


def test_to_dict_run_id_matches_constructor() -> None:
    tracker = MetricsTracker(run_id="my-run")
    assert tracker.to_dict()["run_id"] == "my-run"


def test_to_dict_model_matches_constructor() -> None:
    tracker = MetricsTracker(model="gemma")
    assert tracker.to_dict()["model"] == "gemma"


def test_to_dict_auto_run_id_is_8_chars_if_uuid() -> None:
    tracker = MetricsTracker()
    # UUID4 short ID is first 8 chars
    assert len(tracker.run_id) == 8


def test_to_dict_config_fingerprint_none_by_default() -> None:
    tracker = MetricsTracker()
    assert tracker.to_dict()["config_fingerprint"] is None


def test_to_dict_config_fingerprint_set_via_constructor() -> None:
    tracker = MetricsTracker(config_fingerprint="abc123")
    assert tracker.to_dict()["config_fingerprint"] == "abc123"


# ── write() ───────────────────────────────────────────────────────────────────


def test_write_creates_metrics_json(tmp_path: Path) -> None:
    tracker = MetricsTracker(run_id="test-write", artifact_root=tmp_path)
    tracker.set_dense_bytes(100_000)
    tracker.set_compressed_bytes(25_000)
    result = tracker.write()
    assert (tmp_path / "test-write" / "metrics.json").exists()
    assert result["ratio"] == 4.0


def test_write_returns_dict_matching_written_json(tmp_path: Path) -> None:
    tracker = MetricsTracker(run_id="test-match", artifact_root=tmp_path)
    tracker.record_step(tokens_generated=2, latency_ms=100.0)
    result = tracker.write()
    on_disk = json.loads(
        (tmp_path / "test-match" / "metrics.json").read_text(encoding="utf-8")
    )
    assert result["total_tokens"] == on_disk["total_tokens"]
    assert result["run_id"] == on_disk["run_id"]


def test_write_with_event_log_flushes_events_jsonl(tmp_path: Path) -> None:
    tracker = MetricsTracker(run_id="with-events", artifact_root=tmp_path)
    log = EventLog()
    log.record(
        CacheUpgradeEvent(
            layer_index=0,
            token_index=64,
            old_bytes=100_000,
            new_bytes=25_000,
        )
    )
    tracker.write(event_log=log)
    assert (tmp_path / "with-events" / "events.jsonl").exists()


def test_write_with_event_log_includes_summary_in_metrics(tmp_path: Path) -> None:
    tracker = MetricsTracker(run_id="ev-summary", artifact_root=tmp_path)
    log = EventLog()
    log.record(
        CacheUpgradeEvent(layer_index=0, token_index=0, old_bytes=0, new_bytes=0)
    )
    result = tracker.write(event_log=log)
    assert "events" in result
    assert result["events"]["upgrades"] == 1
    assert result["events"]["total_events"] == 1
