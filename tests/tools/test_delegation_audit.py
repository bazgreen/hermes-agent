import json
import sqlite3

from tools.delegation_audit import record_delegation_audit


def test_record_delegation_audit_persists_redacted_payload_and_ids(tmp_path, monkeypatch):
    db_path = tmp_path / "delegation-audit.db"
    monkeypatch.setenv("DELEGATION_AUDIT_DB", str(db_path))

    record = record_delegation_audit(
        action="allow",
        caller_profile="code",
        callee_profile="mgmt",
        parameters={
            "command": "rm -rf /tmp/secret",
            "nested": {"token": "abc123", "count": 7},
            "items": ["one", "two"],
        },
        correlation_id="corr-123",
        task_id="task-456",
        session_id="sess-789",
        reason="approved by reviewer",
        source="handoff_request",
    )

    assert record["backend"] == "sqlite"
    assert record["caller_profile"] == "code"
    assert record["callee_profile"] == "mgmt"
    assert record["correlation_id"] == "corr-123"
    assert record["task_id"] == "task-456"
    assert record["session_id"] == "sess-789"
    assert record["reason"] == "approved by reviewer"

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT action, caller_profile, callee_profile, correlation_id, task_id, session_id, reason, parameters_json, source, backend FROM delegation_audit ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    assert row == (
        "allow",
        "code",
        "mgmt",
        "corr-123",
        "task-456",
        "sess-789",
        "approved by reviewer",
        row[7],
        "handoff_request",
        "sqlite",
    )

    params = json.loads(row[7])
    assert params["command"]["sha256"] != "rm -rf /tmp/secret"
    assert params["nested"]["token"]["sha256"] != "abc123"
    assert params["nested"]["count"]["sha256"]
    assert [item["sha256"] for item in params["items"]]


def test_record_delegation_audit_logs_when_no_store(monkeypatch, caplog):
    monkeypatch.delenv("DELEGATION_AUDIT_DB", raising=False)
    monkeypatch.delenv("HERMES_DELEGATION_AUDIT_DB", raising=False)
    monkeypatch.delenv("APPROVAL_DB", raising=False)

    with caplog.at_level("INFO"):
        record = record_delegation_audit(
            action="deny",
            caller_profile="code",
            callee_profile="approval-gateway",
            parameters={"secret": "super-secret"},
            correlation_id="corr",
            task_id="task",
            session_id="sess",
            reason="blocked by policy",
            source="approval_gateway",
            audit_db_path="",
        )

    assert record["backend"] == "log"
    assert any("delegation_audit" in msg for msg in caplog.messages)
    assert "super-secret" not in caplog.text
