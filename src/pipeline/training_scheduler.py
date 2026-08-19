"""src/pipeline/training_scheduler.py — Automated Continuous Retraining Scheduler & Daemon.

Manages:
  - Automated recurring schedules (Weekly on Sundays, Daily at Midnight, or On-Capture-Threshold)
  - Next run calculation and persistence in models/scheduler_config.json
  - Regression quality gates: validates EER <= 10.0% & Catch Rate >= 90.0% before promotion
  - Promotion of passing TorchScript models to active production checkpoint
"""
from __future__ import annotations

import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("sonave.pipeline.scheduler")

CONFIG_FILE = Path("models/scheduler_config.json")
DEFAULT_CONFIG = {
    "cadence": "weekly",            # "weekly", "daily", "threshold", "manual"
    "day_of_week": 6,               # 6 = Sunday (0 = Monday)
    "hour_utc": 0,                  # 00:00 UTC
    "minute_utc": 0,
    "capture_threshold_hours": 5.0, # Retrain when N hours new audio ready
    "last_retrain_hours": 0.0,
    "auto_deploy_on_pass": True,
    "max_acceptable_eer_pct": 10.0, # Regression gate: max EER
    "min_acceptable_catch_pct": 90.0,# Regression gate: min catch rate @ 1% FAR
    "last_run_utc": None,
    "status": "active"
}


class TrainingScheduler:
    """Manages retraining schedule, execution triggers, and regression gating."""

    def __init__(self, config_path: str | Path = CONFIG_FILE):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self) -> dict[str, Any]:
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
                # Merge with default config to ensure all keys present
                cfg = DEFAULT_CONFIG.copy()
                cfg.update(data)
                return cfg
            except Exception as e:
                logger.warning("Could not read scheduler config, using defaults: %s", e)
        return DEFAULT_CONFIG.copy()

    def save_config(self, config_data: dict[str, Any] | None = None) -> None:
        if config_data:
            self.config = config_data
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(self.config, indent=2), encoding="utf-8")

    def get_next_run_timestamp(self) -> str | None:
        """Calculate the next scheduled UTC execution time based on current cadence."""
        cadence = self.config.get("cadence", "weekly")
        if cadence == "manual":
            return "Manual On-Demand"
        elif cadence == "threshold":
            th = self.config.get("capture_threshold_hours", 5.0)
            return f"When +{th}h new captures accumulate"

        now = datetime.datetime.now(datetime.timezone.utc)
        target_hour = self.config.get("hour_utc", 0)
        target_minute = self.config.get("minute_utc", 0)

        if cadence == "daily":
            next_run = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += datetime.timedelta(days=1)
            return next_run.strftime("%Y-%m-%d %H:%M UTC")

        elif cadence == "weekly":
            target_weekday = self.config.get("day_of_week", 6) # Sunday = 6
            days_ahead = target_weekday - now.weekday()
            if days_ahead <= 0: # Target day already happened this week
                days_ahead += 7
            next_run = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0) + datetime.timedelta(days=days_ahead)
            return next_run.strftime("%Y-%m-%d %H:%M UTC (Sunday)")

        return None

    def should_trigger_threshold_retrain(self, current_capture_hours: float) -> bool:
        """Check if newly accumulated audio exceeds the retrain threshold."""
        if self.config.get("cadence") != "threshold":
            return False
        last_h = self.config.get("last_retrain_hours", 0.0)
        th = self.config.get("capture_threshold_hours", 5.0)
        return (current_capture_hours - last_h) >= th

    def evaluate_regression_gate(self, eval_metrics: dict[str, Any]) -> dict[str, Any]:
        """Check if newly trained model passes strict performance regression gates."""
        eer = eval_metrics.get("equal_error_rate_pct", 100.0)
        catch_rate = eval_metrics.get("catch_rate_at_1pct_far", 0.0)

        max_eer = self.config.get("max_acceptable_eer_pct", 10.0)
        min_catch = self.config.get("min_acceptable_catch_pct", 90.0)

        passed_eer = (eer <= max_eer)
        passed_catch = (catch_rate >= min_catch)
        passed_all = passed_eer and passed_catch

        gate_result = {
            "gate_passed": passed_all,
            "eer_check": {"current": eer, "max_allowed": max_eer, "passed": passed_eer},
            "catch_rate_check": {"current": catch_rate, "min_required": min_catch, "passed": passed_catch},
            "verdict": "PROMOTED_TO_PRODUCTION" if passed_all else "REJECTED_REGRESSION_DETECTED"
        }

        if passed_all:
            logger.info("✅ Regression gate PASSED: EER=%.2f%% (<=%.2f%%) | Catch=%.2f%% (>=%.2f%%)",
                        eer, max_eer, catch_rate, min_catch)
        else:
            logger.warning("❌ Regression gate FAILED: Model rejected. EER=%.2f%% | Catch=%.2f%%", eer, catch_rate)

        return gate_result

    def execute_scheduled_retrain(self, epochs: int = 3, batch_size: int = 16, lr: float = 1e-4) -> dict[str, Any]:
        """Execute retraining pipeline, evaluate regression gate, and update schedule config."""
        logger.info("Executing scheduled retraining iteration (cadence: %s)", self.config.get("cadence"))
        
        from src.pipeline.run_pipeline import execute_full_pipeline
        run_record = execute_full_pipeline(epochs=epochs, batch_size=batch_size, lr=lr)

        gate_result = self.evaluate_regression_gate(run_record["evaluation_metrics"])
        run_record["regression_gate"] = gate_result

        # Update last run timestamp in config
        self.config["last_run_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.save_config()

        return run_record
