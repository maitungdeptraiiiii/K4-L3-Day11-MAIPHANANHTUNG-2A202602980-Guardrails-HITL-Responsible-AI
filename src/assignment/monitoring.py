"""
Assignment 11 — Monitoring & Alerts starter (TODO).

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    def check_metrics(self) -> list[Alert]:
        """Compare current rates to thresholds; return alerts raised by this call.

        An alert for a metric is raised once — repeated checks don't duplicate it.
        """
        current = self.snapshot()
        watched = [
            ("block_rate", current["block_rate"], self.block_rate_threshold),
            ("rate_limit_hits", current["rate_limit_hits"], self.rate_limit_hit_threshold),
            ("judge_fail_rate", current["judge_fail_rate"], self.judge_fail_rate_threshold),
        ]
        already_raised = {a.metric for a in self.alerts}
        fresh: list[Alert] = []
        for metric, value, limit in watched:
            if value <= limit or metric in already_raised:
                continue
            fresh.append(
                Alert(
                    metric=metric,
                    value=float(value),
                    threshold=float(limit),
                    message=f"ALERT: {metric}={value:.2f} is above threshold {limit}",
                )
            )
        self.alerts.extend(fresh)
        for alert in fresh:
            print(f"[monitor] {alert.message}")
        return fresh

    def export_json(self, filepath: str | None = None):
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        target = Path(filepath or default_metrics_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(self.snapshot(), fh, ensure_ascii=False, indent=2)
        return str(target)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
