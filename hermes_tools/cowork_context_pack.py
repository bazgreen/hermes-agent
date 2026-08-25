"""Cowork context packs.

This module builds compact, structured collaboration bundles for bot-to-bot
handoffs and summons.

Schema shape (version 1):

{
  "schema_version": 1,
  "mode": "task" | "summon",
  "request": {
    "query": str | None,
    "recipient_profile": str | None,
    "recipient_capability": str | None,
    "goal_summary": str | None,
    "next_action": str | None,
  },
  "task": { ... } | None,
  "related_tasks": {
    "parents": [...],
    "children": [...],
  },
  "comments": [...],
  "review_verdicts": [...],
  "artifacts": [...],
  "decisions": [...],
  "constraints": [...],
  "memory": {
    "shared": [...],
    "profile": [...],
  },
}

The pack is intentionally bounded and sanitized:
- secrets are redacted deterministically
- long values are truncated with a visible marker
- task comments and memory snippets are selected by relevance and capped
- the markdown rendering is concise enough for a kanban comment or handoff note
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from hermes_cli import kanban_db as kb
from hermes_constants import get_hermes_home

SCHEMA_VERSION = 1
DEFAULT_COMMENT_LIMIT = 6
DEFAULT_RELATED_LIMIT = 6
DEFAULT_MEMORY_LIMIT = 4
DEFAULT_ARTIFACT_LIMIT = 8
DEFAULT_FIELD_LIMIT = 1200
DEFAULT_BODY_LIMIT = 2200
DEFAULT_COMMENT_BODY_LIMIT = 900

_SECRET_KEY_RE = re.compile(
    r"(?i)(secret|token|password|passwd|api[_-]?key|auth|cookie|bearer|private[_-]?key)"
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)\b("
    r"bearer\s+[a-z0-9._-]{12,}|"
    r"gh[pousr]_[A-Za-z0-9_]{8,}|"
    r"xox[baprs]-[A-Za-z0-9-]{8,}|"
    r"sk-[A-Za-z0-9]{8,}|"
    r"AKIA[0-9A-Z]{16}|"
    r"ASIA[0-9A-Z]{16}|"
    r"[A-Fa-f0-9]{32,}"
    r")\b"
)
_URL_RE = re.compile(r"https?://[^\s<>)\]]+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]{3,}")


def _truncate_text(text: Any, limit: int) -> Any:
    if text is None:
        return None
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    omitted = len(stripped) - limit
    return stripped[:limit] + f"… [truncated, {omitted} chars omitted]"


def _looks_sensitive_key(path: Sequence[str]) -> bool:
    if not path:
        return False
    joined = ".".join(path)
    return bool(_SECRET_KEY_RE.search(joined))


def _looks_like_secret_value(value: str) -> bool:
    return bool(_SECRET_VALUE_RE.search(value))


def _redact_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        if _looks_like_secret_value(stripped):
            return "[REDACTED]"
        return stripped
    return value


def _sanitize(obj: Any, path: Sequence[str] = (), limit: int = DEFAULT_FIELD_LIMIT) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for raw_key, raw_value in obj.items():
            key = str(raw_key)
            child_path = tuple(path) + (key,)
            if _looks_sensitive_key(child_path):
                out[key] = "[REDACTED]"
                continue
            out[key] = _sanitize(raw_value, child_path, limit=limit)
        return out
    if isinstance(obj, (list, tuple)):
        return [_sanitize(item, path, limit=limit) for item in obj]
    if isinstance(obj, set):
        return [_sanitize(item, path, limit=limit) for item in sorted(obj, key=lambda x: repr(x))]
    if isinstance(obj, str):
        value = _redact_value(obj)
        if isinstance(value, str):
            return _truncate_text(value, limit)
        return value
    if isinstance(obj, (int, float, bool)):
        return obj
    return _truncate_text(str(obj), limit)


def _tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(text):
        token = raw.casefold().strip("-_./:")
        if len(token) < 3:
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def _extract_urls(text: str | None) -> list[str]:
    if not text:
        return []
    urls: list[str] = []
    seen: set[str] = set()
    for raw in _URL_RE.findall(text):
        url = raw.rstrip(".,;)]>")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _split_memory_entries(text: str) -> list[str]:
    if not text.strip():
        return []
    if "§" in text:
        raw_entries = [chunk.strip() for chunk in text.split("§")]
        return [entry for entry in raw_entries if entry]

    entries: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                entries.append(" ".join(current).strip())
                current = []
            continue
        if stripped.startswith("---"):
            continue
        if stripped.startswith("#"):
            continue
        current.append(stripped)
    if current:
        entries.append(" ".join(current).strip())
    return [entry for entry in entries if entry]


def _score_entry(entry: str, terms: Sequence[str]) -> int:
    if not terms:
        return 0
    haystack = entry.casefold()
    score = 0
    for term in terms:
        if term and term in haystack:
            score += 1
    return score


def _select_relevant_entries(
    entries: Sequence[str],
    terms: Sequence[str],
    *,
    limit: int,
    source: str,
) -> list[dict[str, Any]]:
    scored = [
        (idx, _score_entry(entry, terms), entry)
        for idx, entry in enumerate(entries)
        if entry.strip()
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    out: list[dict[str, Any]] = []
    for idx, score, entry in scored:
        if score <= 0:
            continue
        out.append(
            {
                "source": source,
                "index": idx,
                "content": _truncate_text(entry, DEFAULT_FIELD_LIMIT),
            }
        )
        if len(out) >= limit:
            break
    return out


def _memory_paths(profile_memory_path: str | Path | None, shared_memory_path: str | Path | None) -> tuple[list[Path], list[Path]]:
    home = get_hermes_home()
    profile_candidates = [
        Path(profile_memory_path).expanduser() if profile_memory_path else home / "memories" / "MEMORY.md",
    ]
    user_candidate = home / "memories" / "USER.md"
    if user_candidate not in profile_candidates:
        profile_candidates.append(user_candidate)

    shared_candidates = [
        Path(shared_memory_path).expanduser() if shared_memory_path else Path("/opt/hermes/profiles/shared/memory/FLEET.md"),
        Path("/opt/hermes/repo/shared/memory/FLEET.md"),
    ]
    return profile_candidates, shared_candidates


def _load_memory_snippets(
    *,
    terms: Sequence[str],
    profile_memory_path: str | Path | None,
    shared_memory_path: str | Path | None,
    memory_limit: int,
) -> dict[str, list[dict[str, Any]]]:
    profile_candidates, shared_candidates = _memory_paths(profile_memory_path, shared_memory_path)
    profile_entries: list[dict[str, Any]] = []
    shared_entries: list[dict[str, Any]] = []

    for path in profile_candidates:
        p = Path(path)
        if not p.exists():
            continue
        entries = _split_memory_entries(_read_text_file(p))
        profile_entries.extend(_select_relevant_entries(entries, terms, limit=memory_limit, source=str(p)))

    for path in shared_candidates:
        p = Path(path)
        if not p.exists():
            continue
        entries = _split_memory_entries(_read_text_file(p))
        shared_entries.extend(_select_relevant_entries(entries, terms, limit=memory_limit, source=str(p)))

    return {"profile": profile_entries[:memory_limit], "shared": shared_entries[:memory_limit]}


def _compact_task(task: kb.Task, *, body_limit: int = DEFAULT_BODY_LIMIT) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": _truncate_text(task.title, DEFAULT_FIELD_LIMIT),
        "body": _truncate_text(task.body or "", body_limit) if task.body else None,
        "assignee": task.assignee,
        "status": task.status,
        "priority": task.priority,
        "tenant": task.tenant,
        "workspace_kind": task.workspace_kind,
        "workspace_path": task.workspace_path,
        "branch_name": task.branch_name,
        "created_by": task.created_by,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "result": _truncate_text(task.result or "", DEFAULT_FIELD_LIMIT) if task.result else None,
        "parents": [],
        "children": [],
    }


def _compact_related_task(conn, task_id: str, relation: str) -> dict[str, Any]:
    task = kb.get_task(conn, task_id)
    if not task:
        return {"id": task_id, "missing": True, "relation": relation}
    out = _compact_task(task)
    out["relation"] = relation
    out["parents"] = kb.parent_ids(conn, task_id)
    out["children"] = kb.child_ids(conn, task_id)
    return out


def _review_verdicts_from_comments(comments: Sequence[kb.Comment], task_status: str) -> list[dict[str, Any]]:
    verdicts: list[dict[str, Any]] = []
    if task_status == "review":
        verdicts.append({"source": "task", "verdict": "in_review"})

    for comment in comments:
        text = (comment.body or "").casefold()
        if "review-required" in text or "needs review" in text:
            verdict = "review_required"
        elif "changes requested" in text or "rejected" in text or "blocked" in text:
            verdict = "rejected"
        elif "approved" in text or "looks good" in text:
            verdict = "approved"
        else:
            continue
        verdicts.append(
            {
                "source": f"comment:{comment.id}",
                "author": comment.author,
                "created_at": comment.created_at,
                "verdict": verdict,
                "excerpt": _truncate_text(comment.body, DEFAULT_COMMENT_BODY_LIMIT),
            }
        )
    return verdicts[-DEFAULT_COMMENT_LIMIT:]


def _comment_records(comments: Sequence[kb.Comment], *, limit: int) -> list[dict[str, Any]]:
    selected = list(comments[-limit:]) if len(comments) > limit else list(comments)
    out: list[dict[str, Any]] = []
    for comment in selected:
        out.append(
            {
                "id": comment.id,
                "author": comment.author,
                "created_at": comment.created_at,
                "body": _truncate_text(comment.body, DEFAULT_COMMENT_BODY_LIMIT),
                "urls": _extract_urls(comment.body),
            }
        )
    return out


def _artifact_records(task: kb.Task | None, attachments: Sequence[kb.Attachment], *, extra_artifacts: Sequence[Any] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for att in attachments:
        out.append(
            {
                "type": "attachment",
                "filename": att.filename,
                "stored_path": att.stored_path,
                "content_type": att.content_type,
                "size": att.size,
            }
        )
    if task is not None:
        for url in _extract_urls(task.body):
            out.append({"type": "url", "url": url, "source": "task.body"})
    for artifact in extra_artifacts or []:
        if isinstance(artifact, Mapping):
            out.append(_sanitize(dict(artifact), limit=DEFAULT_FIELD_LIMIT))
        else:
            out.append({"type": "note", "content": _truncate_text(str(artifact), DEFAULT_FIELD_LIMIT)})
    # Deduplicate by JSON fingerprint while keeping order.
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in out:
        fingerprint = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(item)
    return unique


def _base_request(
    *,
    query: str | None,
    recipient_profile: str | None,
    recipient_capability: str | None,
    goal_summary: str | None,
    next_action: str | None,
) -> dict[str, Any]:
    return {
        "query": _truncate_text(query, DEFAULT_FIELD_LIMIT) if query else None,
        "recipient_profile": _truncate_text(recipient_profile, DEFAULT_FIELD_LIMIT) if recipient_profile else None,
        "recipient_capability": _truncate_text(recipient_capability, DEFAULT_FIELD_LIMIT) if recipient_capability else None,
        "goal_summary": _truncate_text(goal_summary, DEFAULT_FIELD_LIMIT) if goal_summary else None,
        "next_action": _truncate_text(next_action, DEFAULT_FIELD_LIMIT) if next_action else None,
    }


def build_context_pack(
    *,
    task_id: str | None = None,
    query: str | None = None,
    board: str | None = None,
    recipient_profile: str | None = None,
    recipient_capability: str | None = None,
    goal_summary: str | None = None,
    next_action: str | None = None,
    decisions: Sequence[str] | None = None,
    constraints: Sequence[str] | None = None,
    extra_artifacts: Sequence[Any] | None = None,
    comment_limit: int = DEFAULT_COMMENT_LIMIT,
    related_limit: int = DEFAULT_RELATED_LIMIT,
    memory_limit: int = DEFAULT_MEMORY_LIMIT,
    artifact_limit: int = DEFAULT_ARTIFACT_LIMIT,
    profile_memory_path: str | Path | None = None,
    shared_memory_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a bounded JSON-serialisable cowork context pack.

    Either ``task_id`` or ``query`` must be provided. When ``task_id`` is
    supplied the pack includes the task graph, the latest comments, review
    verdicts, attachments, and relevant memory snippets.
    When ``query`` is supplied without a task, the pack becomes a free-form
    summon bundle that carries the request metadata plus relevant memory.
    """

    if task_id is None and not (query and query.strip()):
        raise ValueError("task_id or query is required")

    task: kb.Task | None = None
    comments: list[kb.Comment] = []
    related_tasks = {"parents": [], "children": []}
    artifacts: list[dict[str, Any]] = []
    review_verdicts: list[dict[str, Any]] = []
    task_payload: dict[str, Any] | None = None
    terms: list[str] = []
    request_query: str | None = query
    effective_goal_summary: str | None = goal_summary

    if task_id is not None:
        conn = kb.connect(board=board)
        try:
            task = kb.get_task(conn, task_id)
            if task is None:
                raise ValueError(f"unknown task {task_id}")
            request_query = request_query or task.title
            effective_goal_summary = effective_goal_summary or task.body or task.title
            comments = kb.list_comments(conn, task_id)
            attachments = kb.list_attachments(conn, task_id)
            parent_ids = kb.parent_ids(conn, task_id)
            child_ids = kb.child_ids(conn, task_id)
            related_tasks = {
                "parents": [_compact_related_task(conn, pid, "parent") for pid in parent_ids[:related_limit]],
                "children": [_compact_related_task(conn, cid, "child") for cid in child_ids[:related_limit]],
            }
            task_payload = _compact_task(task)
            task_payload["parents"] = parent_ids
            task_payload["children"] = child_ids
            review_verdicts = _review_verdicts_from_comments(comments, task.status)
            artifacts = _artifact_records(task, attachments, extra_artifacts=extra_artifacts)
            terms.extend(_tokenize(task.title))
            terms.extend(_tokenize(task.body))
            for comment in comments:
                terms.extend(_tokenize(comment.body))
        finally:
            conn.close()
    else:
        task_payload = None
        artifacts = _artifact_records(None, [], extra_artifacts=extra_artifacts)
        effective_goal_summary = effective_goal_summary or request_query

    request_terms = [
        request_query or "",
        recipient_profile or "",
        recipient_capability or "",
        effective_goal_summary or "",
        next_action or "",
    ]
    for text in request_terms:
        terms.extend(_tokenize(text))

    if decisions is None:
        decisions = []
    if constraints is None:
        constraints = []

    memory = _load_memory_snippets(
        terms=terms,
        profile_memory_path=profile_memory_path,
        shared_memory_path=shared_memory_path,
        memory_limit=memory_limit,
    )

    pack = {
        "schema_version": SCHEMA_VERSION,
        "mode": "task" if task_id is not None else "summon",
        "request": _base_request(
            query=request_query,
            recipient_profile=recipient_profile,
            recipient_capability=recipient_capability,
            goal_summary=effective_goal_summary,
            next_action=next_action,
        ),
        "task": task_payload,
        "related_tasks": related_tasks,
        "comments": _comment_records(comments, limit=comment_limit),
        "review_verdicts": review_verdicts,
        "artifacts": artifacts[:artifact_limit],
        "decisions": [_truncate_text(item, DEFAULT_FIELD_LIMIT) for item in decisions][:artifact_limit],
        "constraints": [_truncate_text(item, DEFAULT_FIELD_LIMIT) for item in constraints][:artifact_limit],
        "memory": memory,
    }
    return _sanitize(pack, limit=DEFAULT_FIELD_LIMIT)


def render_context_pack_markdown(pack: Mapping[str, Any]) -> str:
    """Render a compact markdown summary of a context pack."""
    data = _sanitize(dict(pack), limit=DEFAULT_FIELD_LIMIT)
    lines: list[str] = []
    lines.append(f"# Cowork context pack v{data.get('schema_version', SCHEMA_VERSION)}")
    lines.append("")

    request = data.get("request") or {}
    if any(request.get(key) for key in ("query", "recipient_profile", "recipient_capability", "goal_summary", "next_action")):
        lines.append("## Request")
        for label, key in (
            ("Query", "query"),
            ("Recipient", "recipient_profile"),
            ("Capability", "recipient_capability"),
            ("Goal", "goal_summary"),
            ("Next action", "next_action"),
        ):
            value = request.get(key)
            if value:
                lines.append(f"- {label}: {value}")
        lines.append("")

    task = data.get("task")
    if isinstance(task, Mapping):
        lines.append("## Task")
        lines.append(f"- {task.get('id')}: {task.get('title')}")
        lines.append(f"- Status: {task.get('status')} | Assignee: {task.get('assignee') or '(unassigned)'}")
        if task.get("workspace_path"):
            lines.append(f"- Workspace: {task.get('workspace_kind')} @ {task.get('workspace_path')}")
        parents = task.get("parents") or []
        children = task.get("children") or []
        if parents:
            lines.append(f"- Parents: {', '.join(map(str, parents))}")
        if children:
            lines.append(f"- Children: {', '.join(map(str, children))}")
        body = task.get("body")
        if body:
            lines.append("- Body: "+_truncate_text(body, 300))
        lines.append("")

    related = data.get("related_tasks") or {}
    for section_name, items in (("Parents", related.get("parents") or []), ("Children", related.get("children") or [])):
        if not items:
            continue
        lines.append(f"## {section_name}")
        for item in items[:DEFAULT_RELATED_LIMIT]:
            if item.get("missing"):
                lines.append(f"- {item.get('id')} (missing)")
                continue
            lines.append(
                f"- {item.get('id')} — {item.get('title')} "
                f"[{item.get('status')}, {item.get('assignee') or 'unassigned'}]"
            )
        lines.append("")

    comments = data.get("comments") or []
    if comments:
        lines.append("## Recent comments")
        for comment in comments:
            lines.append(
                f"- {comment.get('author')} at {comment.get('created_at')}: "
                f"{_truncate_text(comment.get('body'), 220)}"
            )
        lines.append("")

    verdicts = data.get("review_verdicts") or []
    if verdicts:
        lines.append("## Review verdicts")
        for verdict in verdicts:
            lines.append(
                f"- {verdict.get('verdict')} from {verdict.get('source')}: "
                f"{_truncate_text(verdict.get('excerpt') or '', 220)}"
            )
        lines.append("")

    artifacts = data.get("artifacts") or []
    if artifacts:
        lines.append("## Artifacts")
        for artifact in artifacts[:DEFAULT_ARTIFACT_LIMIT]:
            if artifact.get("type") == "attachment":
                lines.append(
                    f"- attachment {artifact.get('filename')} → {artifact.get('stored_path')}"
                )
            elif artifact.get("type") == "url":
                lines.append(f"- url {artifact.get('url')}")
            else:
                lines.append(f"- {artifact}")
        lines.append("")

    memory = data.get("memory") or {}
    shared = memory.get("shared") or []
    profile = memory.get("profile") or []
    if shared or profile:
        lines.append("## Relevant memory")
        for label, items in (("Shared", shared), ("Profile", profile)):
            if not items:
                continue
            lines.append(f"### {label}")
            for item in items[:DEFAULT_MEMORY_LIMIT]:
                content = item.get("content") if isinstance(item, Mapping) else item
                source = item.get("source") if isinstance(item, Mapping) else ""
                lines.append(f"- {content} ({source})")
        lines.append("")

    decisions = data.get("decisions") or []
    if decisions:
        lines.append("## Decisions")
        for item in decisions:
            lines.append(f"- {item}")
        lines.append("")

    constraints = data.get("constraints") or []
    if constraints:
        lines.append("## Constraints")
        for item in constraints:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def context_pack_to_json(pack: Mapping[str, Any]) -> str:
    """Return canonical JSON for logging, comments, or transport."""
    return json.dumps(_sanitize(dict(pack), limit=DEFAULT_FIELD_LIMIT), ensure_ascii=False, sort_keys=True, indent=2)
