#!/usr/bin/env python3
"""Structured audit helper for delegation and approval decisions.

The helper keeps the wire format small, preserves the original payload shape,
and hashes/redacts leaf values so logs and durable stores never receive raw
commands, secrets, or other sensitive parameters.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS delegation_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at INTEGER NOT NULL,
    action TEXT NOT NULL,
    caller_profile TEXT NOT NULL,
    callee_profile TEXT NOT NULL,
    correlation_id TEXT,
    task_id TEXT,
    session_id TEXT,
    reason TEXT,
    parameters_json TEXT NOT NULL,
    source TEXT NOT NULL,
    backend TEXT NOT NULL
);
"""

_ALLOWED_ACTIONS = {"allow", "deny", "review_required"}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _jsonish(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))


def _redact_value(value: Any) -> Any:
    """Recursively redact leaf values while preserving the container shape."""
    if value is None:
        return {"type": "none", "sha256": _sha256_text("null")}
    if isinstance(value, bool):
        token = "true" if value else "false"
        return {"type": "bool", "sha256": _sha256_text(token)}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        token = _jsonish(value)
        return {"type": type(value).__name__, "sha256": _sha256_text(token)}
    if isinstance(value, str):
        return {
            "type": "str",
            "length": len(value),
            "sha256": _sha256_text(value),
        }
    if isinstance(value, Mapping):
        return {str(key): _redact_value(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_redact_value(item) for item in value],
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, set):
        return {
            "type": "set",
            "items": [_redact_value(item) for item in sorted(value, key=_jsonish)],
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview, str)):
        return {
            "type": type(value).__name__,
            "items": [_redact_value(item) for item in value],
        }
    text = _jsonish(value)
    return {
        "type": type(value).__name__,
        "sha256": _sha256_text(text),
    }


def _env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "")
        if value:
            return value
    return ""


def _default_store_path() -> str:
    explicit = _env_first("DELEGATION_AUDIT_DB", "HERMES_DELEGATION_AUDIT_DB")
    if explicit:
        return explicit
    approval_db = os.environ.get("APPROVAL_DB", "")
    if approval_db:
        return os.path.join(os.path.dirname(approval_db), "delegation-audit.db")
    return ""


def _store_record(path: str, record: dict[str, Any]) -> bool:
    if not path:
        return False
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with sqlite3.connect(path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        conn.execute(
            """
            INSERT INTO delegation_audit (
                created_at, action, caller_profile, callee_profile,
                correlation_id, task_id, session_id, reason,
                parameters_json, source, backend
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["created_at"],
                record["action"],
                record["caller_profile"],
                record["callee_profile"],
                record["correlation_id"],
                record["task_id"],
                record["session_id"],
                record["reason"],
                _jsonish(record["parameters"]),
                record["source"],
                "sqlite",
            ),
        )
        conn.commit()
    return True


def record_delegation_audit(
    action: str,
    caller_profile: str,
    callee_profile: str,
    parameters: Any | None,
    correlation_id: str = "",
    task_id: str = "",
    session_id: str = "",
    reason: str = "",
    *,
    source: str = "delegation",
    audit_db_path: str | None = None,
    logger_: logging.Logger | None = None,
) -> dict[str, Any]:
    """Write a structured delegation audit record.

    The helper never raises on persistence failures. If the durable store is
    unavailable, it falls back to logging a JSON payload.
    """
    logger_obj = logger_ or logger
    normalized_action = (action or "").strip().lower()
    if normalized_action not in _ALLOWED_ACTIONS:
        normalized_action = "review_required"

    record = {
        "action": normalized_action,
        "caller_profile": caller_profile or _env_first("HERMES_PROFILE", "HERMES_SESSION_USER_NAME", "worker"),
        "callee_profile": callee_profile or _env_first("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_NAME", "approval-gateway"),
        "parameters": _redact_value(parameters or {}),
        "correlation_id": correlation_id or _env_first("HERMES_CORRELATION_ID", "HERMES_SESSION_MESSAGE_ID"),
        "task_id": task_id or _env_first("HERMES_TASK_ID", "HERMES_KANBAN_TASK"),
        "session_id": session_id or _env_first("HERMES_SESSION_ID", "HERMES_SESSION_KEY"),
        "reason": reason or "",
        "created_at": int(time.time()),
        "source": source,
    }

    backend = "log"
    store_path = audit_db_path if audit_db_path is not None else _default_store_path()
    try:
        if _store_record(store_path, record):
            backend = "sqlite"
    except Exception as exc:  # noqa: BLE001 - audit must never fail closed
        logger_obj.debug("Delegation audit store write failed: %s", exc, exc_info=True)

    if backend == "log":
        logger_obj.info("delegation_audit %s", _jsonish(record))

    return {**record, "backend": backend, "audit_db_path": store_path}
