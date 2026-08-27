from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_tools import (
    build_context_pack,
    context_pack_to_json,
    handoff_request,
    render_context_pack_markdown,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _write_memory_file(path: Path, entries: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n§\n".join(entries), encoding="utf-8")
    return path


def test_context_pack_task_graph_comments_redaction_and_memory(kanban_home, tmp_path):
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="Parent task", body="Parent body")
        kb.complete_task(conn, parent, summary="parent done")
        child = kb.create_task(
            conn,
            title="Build cowork context pack",
            body="Deliver a JSON pack + markdown rendering for handoffs. See https://example.com/spec",
            assignee="code",
        )
        kb.link_tasks(conn, parent, child)
        kb.add_comment(
            conn,
            child,
            author="reviewer",
            body="review-required: please check the auth flow and sk-abc123def456 secret handling",
        )
        kb.add_comment(
            conn,
            child,
            author="code",
            body="I am on it. Reference: https://example.com/context-pack",
        )
        attachment_path = tmp_path / "artifact.txt"
        attachment_path.write_text("artifact payload", encoding="utf-8")
        kb.add_attachment(
            conn,
            child,
            filename="artifact.txt",
            stored_path=str(attachment_path),
            content_type="text/plain",
            size=attachment_path.stat().st_size,
            uploaded_by="code",
        )
    finally:
        conn.close()

    profile_memory = _write_memory_file(
        tmp_path / "profile-memory.md",
        [
            "irrelevant note about recipes",
            "context pack builders should be deterministic and concise",
        ],
    )
    shared_memory = _write_memory_file(
        tmp_path / "fleet-memory.md",
        [
            "shared concern about context packs for bot handoffs",
            "another unrelated fleet note",
        ],
    )

    pack = build_context_pack(
        task_id=child,
        recipient_profile="reviewer",
        recipient_capability="code review",
        next_action="Review the pack and suggest the next edit.",
        profile_memory_path=profile_memory,
        shared_memory_path=shared_memory,
    )

    assert pack["schema_version"] == 1
    assert pack["mode"] == "task"
    assert pack["task"]["id"] == child
    assert pack["task"]["parents"] == [parent]
    assert pack["related_tasks"]["parents"][0]["id"] == parent
    assert pack["related_tasks"]["children"] == []
    assert any("[REDACTED]" in c["body"] for c in pack["comments"])
    assert any(v["verdict"] == "review_required" for v in pack["review_verdicts"])
    assert any(a["type"] == "attachment" for a in pack["artifacts"])
    assert any(a["type"] == "url" for a in pack["artifacts"])
    assert pack["memory"]["profile"]
    assert "deterministic" in pack["memory"]["profile"][0]["content"]
    assert pack["memory"]["shared"]
    assert "bot handoffs" in pack["memory"]["shared"][0]["content"]

    rendered = render_context_pack_markdown(pack)
    assert rendered.startswith("# Cowork context pack v1")
    assert "## Task" in rendered
    assert "## Review verdicts" in rendered
    assert "artifact.txt" in rendered
    assert "review_required" in rendered
    assert "***" not in rendered

    json_payload = context_pack_to_json(pack)
    assert '"schema_version": 1' in json_payload
    assert "sk-abc123def456" not in json_payload


def test_context_pack_free_form_request_and_markdown(tmp_path):
    profile_memory = _write_memory_file(
        tmp_path / "profile-memory.md",
        ["reviewers prefer concise context packs"],
    )
    shared_memory = _write_memory_file(
        tmp_path / "fleet-memory.md",
        ["shared memory about summon requests and recipient metadata"],
    )

    pack = build_context_pack(
        query="Need a reviewer for the new cowork context pack",
        recipient_profile="reviewer",
        recipient_capability="code review",
        goal_summary="Ship the handoff pack infrastructure",
        next_action="Read the pack and reply with gaps.",
        profile_memory_path=profile_memory,
        shared_memory_path=shared_memory,
    )

    assert pack["mode"] == "summon"
    assert pack["task"] is None
    assert pack["request"]["query"] == "Need a reviewer for the new cowork context pack"
    assert pack["request"]["recipient_profile"] == "reviewer"
    assert pack["request"]["recipient_capability"] == "code review"
    assert pack["request"]["goal_summary"] == "Ship the handoff pack infrastructure"
    assert pack["memory"]["profile"]
    assert pack["memory"]["shared"]

    rendered = render_context_pack_markdown(pack)
    assert "## Request" in rendered
    assert "Recipient: reviewer" in rendered
    assert "Capability: code review" in rendered
    assert "Next action: Read the pack and reply with gaps." in rendered


def test_context_pack_missing_or_unknown_task():
    with pytest.raises(ValueError):
        build_context_pack()

    with pytest.raises(ValueError, match="unknown task"):
        build_context_pack(task_id="t_missing")


@pytest.mark.asyncio
async def test_handoff_request_carries_context_pack(monkeypatch):
    class _MockResponse:
        def __init__(self, status_code, json_data):
            self.status_code = status_code
            self._json = json_data

        def json(self):
            return self._json

    class _MockClient:
        def __init__(self):
            self.last_request = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, *, json, headers):
            self.last_request = {"url": url, "json": json, "headers": headers}
            return _MockResponse(200, {"result": "ok"})

    client = _MockClient()
    pack = {"schema_version": 1, "mode": "summon", "request": {"query": "hello"}}

    monkeypatch.setattr("hermes_tools.handoff.httpx.AsyncClient", lambda **_: client)

    result = await handoff_request(
        from_bot="code",
        to_url="http://127.0.0.1:8642/api/handoff",
        action="summon",
        query="hello",
        secret="secret",
        context_pack=pack,
    )

    assert result == {"result": "ok"}
    assert client.last_request["json"]["context_pack"]["schema_version"] == 1
    assert client.last_request["json"]["query"] == "hello"


@pytest.mark.asyncio
async def test_summon_prompt_includes_context_pack_rendering():
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    seen: dict[str, str] = {}

    async def fake_run_agent(*, user_message, conversation_history, session_id):
        seen["user_message"] = user_message
        seen["session_id"] = session_id
        return {"final_response": "ok", "completed": True, "api_calls": 1}, {"total_tokens": 1}

    adapter._run_agent = fake_run_agent
    pack = build_context_pack(query="Need help", recipient_profile="reviewer")

    result = await adapter._run_handoff_summon(
        query="Need help",
        timeout=1.0,
        context_pack=pack,
    )

    assert result["success"] is True
    assert "Structured cowork context pack" in seen["user_message"]
    assert "Need help" in seen["user_message"]
    assert seen["session_id"].startswith("handoff_summon_")
