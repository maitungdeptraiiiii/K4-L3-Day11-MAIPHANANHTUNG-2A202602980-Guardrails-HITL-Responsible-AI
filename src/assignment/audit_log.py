"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    @staticmethod
    def _trace_key(user_id: str, request_id: str | None) -> str:
        return f"{user_id}::{request_id or '-'}"

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Open an audit entry and start its latency clock."""
        self._open[self._trace_key(user_id, request_id)] = time.perf_counter()
        self.logs.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "received_at": utc_now_iso(),
                "input": text,
                "output": None,
                "blocked": None,
                "layer": None,
                "latency_ms": None,
            }
        )

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Close the matching audit entry with the decision and latency."""
        started = self._open.pop(self._trace_key(user_id, request_id), None)
        latency_ms = (
            round((time.perf_counter() - started) * 1000, 3) if started is not None else None
        )
        entry = next(
            (
                e
                for e in reversed(self.logs)
                if e["user_id"] == user_id
                and e["request_id"] == request_id
                and e["output"] is None
            ),
            None,
        )
        if entry is None:
            # Output without a recorded input — still keep it for forensics.
            entry = {"request_id": request_id, "user_id": user_id,
                     "received_at": None, "input": None}
            self.logs.append(entry)
        entry.update(
            output=text,
            blocked=bool(blocked),
            layer=layer,
            latency_ms=latency_ms,
            completed_at=utc_now_iso(),
        )

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        target = Path(filepath or default_audit_log_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(self.logs, fh, ensure_ascii=False, indent=2)
        return str(target)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
