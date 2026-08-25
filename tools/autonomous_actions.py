"""Fleet-wide append-only audit log for autonomous agent actions.

Modeled on the decommissioned openclaw's ``autonomous-actions.jsonl``. One
JSON object per line (JSONL), append-only, durable, and fleet-visible so every
bot (code, mgmt, pt, life) writes to the same file.

Every record carries at minimum::

    {"action": "...", "time": "2026-08-25T19:00:00Z", "bot": "mgmt",
     "outcome": "success|failure|blocked", "detail": "..."}

Design contract (mirrors :mod:`tools.delegation_audit`):
- **Never raises and never blocks execution.** Audit is observability, not
  policy — a write failure is surfaced (stderr fallback + warning log +
  failure counter) but never propagates.
- **Append-only.** Records are opened with ``O_APPEND``; the module never
  truncates or rewrites the file.
- **Self-contained.** No framework imports required for the core path; optional
  imports (``hermes_constants``) are guarded so the module also runs standalone
  as a script or from a cron job outside the venv.

Log path resolution order:
1. ``AUTONOMOUS_ACTIONS_LOG`` env var (explicit override).
2. Fleet-visible path when ``HERMES_HOME`` is set (profile/custom deployment):
   ``<hermes_root>/shared/autonomous-actions.jsonl`` (e.g.
   ``/opt/hermes/shared/autonomous-actions.jsonl``) so all profiles share one log.
3. ``~/.hermes/autonomous-actions.jsonl`` otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_logger = None
try:
    import logging

    _logger = logging.getLogger(__name__)
except Exception:  # pragma: no cover - logging import is not the point
    _logger = None

_append_lock = threading.Lock()

# Canonical outcome values.
OUTCOME_SUCCESS = "success"
OUTCOME_FAILURE = "failure"
OUTCOME_BLOCKED = "blocked"
_CANONICAL_OUTCOMES = {OUTCOME_SUCCESS, OUTCOME_FAILURE, OUTCOME_BLOCKED}

# A single write is one JSON line; cap detail length so a pathological command
# string (or a redaction miss) can't bloat a line unboundedly.
_MAX_DETAIL_CHARS = 2000


def _log_debug(msg: str, *args: Any, exc_info: bool = False) -> None:
    if _logger is not None:
        _logger.debug(msg, *args, exc_info=exc_info)


def _log_warning(msg: str, *args: Any, exc_info: bool = False) -> None:
    if _logger is not None:
        _logger.warning(msg, *args, exc_info=exc_info)


def _hermes_home() -> Path | None:
    env = os.environ.get("HERMES_HOME", "").strip()
    if env:
        return Path(env)
    return None


def _hermes_root() -> Path:
    """Best-effort root dir (parent of a ``profiles/`` layout), else home."""
    home = _hermes_home()
    if home is not None:
        if home.parent.name == "profiles":
            return home.parent.parent
        return home
    return Path.home() / ".hermes"


def _default_shared_dir() -> Path:
    """Fleet-visible shared directory, derived without framework imports."""
    try:
        from hermes_constants import get_default_hermes_root  # local guarded import

        root = get_default_hermes_root()
        if root is not None:
            return Path(root) / "shared"
    except Exception:
        pass
    root = _hermes_root()
    if root.name == "shared":
        return root
    return root / "shared"


def resolve_log_path() -> str:
    """Resolve the audit-log file path (see module docstring)."""
    override = os.environ.get("AUTONOMOUS_ACTIONS_LOG", "").strip()
    if override:
        return override

    home = _hermes_home()
    if home is not None:
        # HERMES_HOME is set: use the fleet-visible shared location so every
        # profile appends to the same log rather than its own profile dir.
        return str(_default_shared_dir() / "autonomous-actions.jsonl")

    return str(Path.home() / ".hermes" / "autonomous-actions.jsonl")


def resolve_bot() -> str:
    """Resolve the acting bot/profile name."""
    env = os.environ.get("HERMES_PROFILE", "").strip()
    if env:
        return env
    home = _hermes_home()
    if home is not None:
        name = home.name
        if home.parent.name == "profiles" and name:
            return name
        return name
    # Legacy / non-profile deployments: fall back to any session user hint.
    for key in ("HERMES_SESSION_USER_NAME", "USER", "LOGNAME"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return "unknown"


def _iso_utc_now(ts: float | None = None) -> str:
    moment = ts if ts is not None else time.time()
    return (
        datetime.fromtimestamp(moment, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _truncate(value: str) -> str:
    if len(value) <= _MAX_DETAIL_CHARS:
        return value
    return value[:_MAX_DETAIL_CHARS] + "…[truncated]"


def build_record(
    action: str,
    outcome: str,
    *,
    bot: str | None = None,
    detail: str = "",
    ts: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a single audit record (pure; does not write)."""
    record: dict[str, Any] = {
        "action": (action or "").strip(),
        "time": _iso_utc_now(ts),
        "bot": (bot or "").strip() or resolve_bot(),
        "outcome": (outcome or "").strip(),
        "detail": _truncate(detail or ""),
    }
    # Best-effort correlation context, mirroring delegation_audit.
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if task_id:
        record["task_id"] = task_id
    session_id = os.environ.get("HERMES_SESSION_ID", "").strip()
    if session_id:
        record["session_id"] = session_id
    if extra:
        for key, value in extra.items():
            if key not in record:
                record[key] = value
    return record


# --- Write-failure observability -------------------------------------------------
#
# A failed append must not be silent: even though append_action() never raises
# (so a write error can never block or alter an approval decision), the failure
# is surfaced three ways so an operator can detect a broken audit trail:
#   1. a WARNING-level log line (visible in the gateway logs),
#   2. a best-effort stderr write (durable — captured by systemd/journald even
#      when the JSONL file path is unwritable),
#   3. an in-process failure counter (a metric a health check can scrape).
_write_failure_count = 0


def get_write_failure_count() -> int:
    """Number of append failures observed this process (metric)."""
    with _append_lock:
        return _write_failure_count


def _note_write_failure(error: str, line: str) -> None:
    global _write_failure_count
    with _append_lock:
        _write_failure_count += 1
    _log_warning("autonomous-actions append failed: %s", error)
    try:
        sys.stderr.write(f"[autonomous-actions] append failed: {error}\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 - stderr fallback is itself best-effort
        pass


def append_action(
    action: str,
    outcome: str,
    *,
    bot: str | None = None,
    detail: str = "",
    ts: float | None = None,
    path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one autonomous-action record as a JSON line.

    Returns the record written. Never raises — on any failure the record is
    still returned (with ``"written": False``) and the failure is surfaced via
    :func:`_note_write_failure`, so callers and operators can observe it.
    """
    record = build_record(action, outcome, bot=bot, detail=detail, ts=ts, extra=extra)
    log_path = path or resolve_log_path()
    line = json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n"
    written = False
    error: str | None = None
    try:
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with _append_lock:
            fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line.encode("utf-8"))
                written = True
            finally:
                os.close(fd)
    except Exception as exc:  # noqa: BLE001 - audit must never fail closed
        error = f"{type(exc).__name__}: {exc}"
        _note_write_failure(error, line)
    record["written"] = written
    if error:
        record["error"] = error
    return record


# ---------------------------------------------------------------------------
# Reading / review
# ---------------------------------------------------------------------------


def _parse_time(value: str) -> float:
    """Parse an ISO-8601 (optionally 'Z'-suffixed) timestamp to epoch seconds."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).timestamp()


def read_actions(
    path: str | None = None,
    *,
    bot: str | None = None,
    since: str | None = None,
    until: str | None = None,
    outcome: str | None = None,
    action_contains: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read and filter audit records from the JSONL log (newest first).

    ``since``/``until`` accept ISO-8601 timestamps; ``action_contains`` is a
    case-insensitive substring match on the action field.
    """
    log_path = path or resolve_log_path()
    if not os.path.exists(log_path):
        return []

    since_ts = _parse_time(since) if since else None
    until_ts = _parse_time(until) if until else None
    records: list[dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if bot and rec.get("bot") != bot:
                continue
            if outcome and rec.get("outcome") != outcome:
                continue
            if action_contains and action_contains.lower() not in str(rec.get("action", "")).lower():
                continue
            try:
                ts = _parse_time(rec["time"]) if rec.get("time") else 0.0
            except (ValueError, TypeError):
                ts = 0.0
            if since_ts is not None and ts < since_ts:
                continue
            if until_ts is not None and ts > until_ts:
                continue
            records.append(rec)

    records.sort(key=lambda r: r.get("time", ""), reverse=True)
    if limit is not None and limit >= 0:
        records = records[:limit]
    return records


def _render_record(rec: dict[str, Any]) -> str:
    time_str = rec.get("time", "-")
    bot = rec.get("bot", "-")
    outcome = rec.get("outcome", "-")
    action = rec.get("action", "-")
    detail = rec.get("detail", "")
    line = f"{time_str}  [{bot:6}]  {outcome:9}  {action}"
    if detail:
        line += f"  ::  {detail}"
    return line


def _cli_append(args: argparse.Namespace) -> int:
    rec = append_action(
        args.action,
        args.outcome,
        bot=args.bot,
        detail=args.detail or "",
        extra={"reason": args.reason} if args.reason else None,
    )
    if rec.get("written"):
        print(json.dumps(rec, ensure_ascii=False))
        return 0
    print(f"ERROR: failed to append: {rec.get('error', 'unknown')}", file=sys.stderr)
    return 1


def _cli_list(args: argparse.Namespace) -> int:
    recs = read_actions(
        bot=args.bot,
        since=args.since,
        until=args.until,
        outcome=args.outcome,
        action_contains=args.action,
        limit=args.limit,
    )
    if args.json:
        for rec in recs:
            print(json.dumps(rec, ensure_ascii=False))
    else:
        for rec in recs:
            print(_render_record(rec))
    if not recs:
        print(f"(no matching records in {resolve_log_path()})", file=sys.stderr)
    return 0


def _cli_stats(args: argparse.Namespace) -> int:
    recs = read_actions(bot=args.bot, since=args.since, until=args.until)
    by_bot: dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    for rec in recs:
        by_bot[rec.get("bot", "?")] = by_bot.get(rec.get("bot", "?"), 0) + 1
        by_outcome[rec.get("outcome", "?")] = by_outcome.get(rec.get("outcome", "?"), 0) + 1
    print(f"total: {len(recs)}")
    print("by bot:", json.dumps(by_bot, sort_keys=True) if by_bot else "{}")
    print("by outcome:", json.dumps(by_outcome, sort_keys=True) if by_outcome else "{}")
    print(f"write failures: {get_write_failure_count()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autonomous-actions",
        description="Append-only fleet audit log for autonomous agent actions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_append = sub.add_parser("append", help="Append one action record")
    p_append.add_argument("--action", required=True, help="What was done")
    p_append.add_argument("--outcome", required=True, help="success | failure | blocked")
    p_append.add_argument("--bot", default=None, help="Profile name (default: auto)")
    p_append.add_argument("--detail", default="", help="Human-readable detail / command")
    p_append.add_argument("--reason", default="", help="Optional reason field")
    p_append.set_defaults(func=_cli_append)

    p_list = sub.add_parser("list", help="List / filter actions (newest first)")
    _add_filter_args(p_list)
    p_list.set_defaults(func=_cli_list)

    p_review = sub.add_parser("review", help="Alias for 'list' (default: last 50)")
    _add_filter_args(p_review)
    p_review.set_defaults(func=_cli_list)

    p_stats = sub.add_parser("stats", help="Counts by bot and outcome")
    p_stats.add_argument("--bot", default=None)
    p_stats.add_argument("--since", default=None)
    p_stats.add_argument("--until", default=None)
    p_stats.set_defaults(func=_cli_stats)

    return parser


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bot", default=None, help="Filter by profile name")
    p.add_argument("--since", default=None, help="ISO-8601 lower bound (inclusive)")
    p.add_argument("--until", default=None, help="ISO-8601 upper bound (inclusive)")
    p.add_argument("--outcome", default=None, help="Filter by outcome")
    p.add_argument("--action", default=None, help="Substring match on action")
    p.add_argument("--limit", type=int, default=50, help="Max records (default 50)")
    p.add_argument("--json", action="store_true", help="Emit JSON lines")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
