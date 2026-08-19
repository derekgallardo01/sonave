"""tests/test_training_scheduler.py — Unit & Integration Tests for Automated Training Scheduler."""
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from src.pipeline.training_scheduler import TrainingScheduler, DEFAULT_CONFIG


def test_scheduler_config_loading_and_saving(tmp_path):
    config_file = tmp_path / "scheduler_config.json"
    sched = TrainingScheduler(config_path=config_file)

    cfg = sched.load_config()
    assert cfg["cadence"] == "weekly"
    assert cfg["day_of_week"] == 6

    # Update and save
    cfg["cadence"] = "daily"
    cfg["hour_utc"] = 3
    sched.save_config(cfg)

    # Reload
    sched2 = TrainingScheduler(config_path=config_file)
    cfg2 = sched2.load_config()
    assert cfg2["cadence"] == "daily"
    assert cfg2["hour_utc"] == 3


def test_scheduler_next_run_calculation(tmp_path):
    config_file = tmp_path / "scheduler_config.json"
    sched = TrainingScheduler(config_path=config_file)

    # Weekly
    sched.config["cadence"] = "weekly"
    ts_weekly = sched.get_next_run_timestamp()
    assert "Sunday" in ts_weekly or "UTC" in ts_weekly

    # Daily
    sched.config["cadence"] = "daily"
    ts_daily = sched.get_next_run_timestamp()
    assert "UTC" in ts_daily

    # Threshold
    sched.config["cadence"] = "threshold"
    sched.config["capture_threshold_hours"] = 5.0
    ts_thresh = sched.get_next_run_timestamp()
    assert "+5.0h" in ts_thresh or "+5h" in ts_thresh

    # Manual
    sched.config["cadence"] = "manual"
    ts_manual = sched.get_next_run_timestamp()
    assert "Manual" in ts_manual


def test_scheduler_threshold_trigger(tmp_path):
    config_file = tmp_path / "scheduler_config.json"
    sched = TrainingScheduler(config_path=config_file)
    sched.config["cadence"] = "threshold"
    sched.config["capture_threshold_hours"] = 5.0
    sched.config["last_retrain_hours"] = 10.0

    # 12 hours accumulated (diff = 2.0 < 5.0 -> False)
    assert not sched.should_trigger_threshold_retrain(12.0)

    # 16 hours accumulated (diff = 6.0 >= 5.0 -> True)
    assert sched.should_trigger_threshold_retrain(16.0)


def test_scheduler_regression_gate(tmp_path):
    config_file = tmp_path / "scheduler_config.json"
    sched = TrainingScheduler(config_path=config_file)

    # Passing candidate (EER = 4.5% <= 10.0%, Catch = 96.0% >= 90.0%)
    passing_metrics = {"equal_error_rate_pct": 4.5, "catch_rate_at_1pct_far": 96.0}
    gate_pass = sched.evaluate_regression_gate(passing_metrics)
    assert gate_pass["gate_passed"] is True
    assert gate_pass["verdict"] == "PROMOTED_TO_PRODUCTION"

    # Failing candidate due to high EER (EER = 14.5% > 10.0%)
    failing_metrics = {"equal_error_rate_pct": 14.5, "catch_rate_at_1pct_far": 95.0}
    gate_fail = sched.evaluate_regression_gate(failing_metrics)
    assert gate_fail["gate_passed"] is False
    assert gate_fail["verdict"] == "REJECTED_REGRESSION_DETECTED"


def test_api_training_schedule_endpoints():
    import sys
    _RAILWAY = Path(__file__).resolve().parent.parent / "railway"
    if str(_RAILWAY) not in sys.path:
        sys.path.insert(0, str(_RAILWAY))

    from app import app
    client = TestClient(app)

    # GET schedule
    r = client.get("/api/training/schedule")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert "cadence" in d
    assert "next_run_display" in d
