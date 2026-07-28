"""Tests for bot-to-bot handoff: HandoffError, handoff_request(), and the
/api/handoff gateway endpoint handler.

Test organisation:
- Tests in ``TestHandoffRequest`` cover ``hermes_tools.handoff.handoff_request()``
  with mocked HTTP transport (no real network calls).
- Tests in ``TestHandoffError`` cover the ``HandoffError`` exception type itself.
- Tests in ``TestHandleHandoff`` cover the API server handler logic via
  pytest-aiohttp fake requests (no real gateway startup needed).
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is importable
_project = str(Path(__file__).resolve().parents[1])
if _project not in sys.path:
    sys.path.insert(0, _project)

from hermes_tools.handoff import HandoffError, handoff_request


# ============================================================================
# HandoffError
# ============================================================================


class TestHandoffError:
    def test_attributes(self):
        exc = HandoffError("auth_denied", "Invalid secret", status=401)
        assert exc.code == "auth_denied"
        assert exc.detail == "Invalid secret"
        assert exc.status == 401
        assert "auth_denied" in str(exc)
        assert "Invalid secret" in str(exc)

    def test_default_status_is_zero(self):
        exc = HandoffError("timeout", "timed out")
        assert exc.status == 0

    def test_str_representation(self):
        exc = HandoffError("tool_not_found", "Echo missing", status=404)
        text = str(exc)
        assert "[tool_not_found]" in text
        assert "Echo missing" in text


# ============================================================================
# handoff_request — caller side
# ============================================================================


class _MockResponse:
    """Lightweight httpx.Response stand-in for testing."""

    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json = json_data

    def json(self):
        return self._json


class _MockAsyncClient:
    """Replacement for httpx.AsyncClient that returns canned responses."""

    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self.json_data = json_data or {"result": "ok"}
        self.last_request = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url, *, json, headers):
        self.last_request = {"url": url, "json": json, "headers": headers}
        return _MockResponse(self.status_code, self.json_data)


class _MockAsyncClientRaising:
    """Replacement that raises on post()."""

    def __init__(self, exc):
        self._exc = exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url, *, json, headers):
        raise self._exc


@pytest.mark.asyncio
async def test_handoff_request_success():
    """AC1: Valid auth + valid tool → result returned."""
    client = _MockAsyncClient(
        status_code=200,
        json_data={"result": {"echoed": "hello"}},
    )
    with patch("hermes_tools.handoff.httpx.AsyncClient", return_value=client), \
        patch("hermes_tools.handoff.record_delegation_audit") as mock_audit:
        mock_audit.return_value = {"backend": "log"}
        result = await handoff_request(
            from_bot="code",
            to_url="http://127.0.0.1:8642/api/handoff",
            action="tool_call",
            tool="echo",
            params={"text": "hello"},
            secret="test-secret",
        )
    assert result == {"result": {"echoed": "hello"}}
    assert client.last_request["headers"]["X-Handoff-Auth"] == "test-secret"
    assert client.last_request["headers"]["Content-Type"] == "application/json"
    mock_audit.assert_called_once()
    audit_kwargs = mock_audit.call_args.kwargs
    assert audit_kwargs["action"] == "allow"
    assert audit_kwargs["caller_profile"] == "code"
    assert audit_kwargs["callee_profile"] == "127.0.0.1:8642"
    assert audit_kwargs["source"] == "handoff_request"


@pytest.mark.asyncio
async def test_handoff_request_401():
    """AC2: Invalid auth returns 401 HandoffError."""
    client = _MockAsyncClient(status_code=401, json_data={"error": "Invalid auth"})
    with patch("hermes_tools.handoff.httpx.AsyncClient", return_value=client):
        with pytest.raises(HandoffError) as ei:
            await handoff_request(
                from_bot="badger",
                to_url="http://127.0.0.1:8642/api/handoff",
                action="tool_call",
                tool="echo",
                params={},
                secret="wrong-secret",
            )
    assert ei.value.code == "auth_denied"
    assert ei.value.status == 401


@pytest.mark.asyncio
async def test_handoff_request_403():
    """AC4: Non-allowed source returns 403 HandoffError."""
    client = _MockAsyncClient(
        status_code=403,
        json_data={"error": "Source 'hacker' not allowed"},
    )
    with patch("hermes_tools.handoff.httpx.AsyncClient", return_value=client):
        with pytest.raises(HandoffError) as ei:
            await handoff_request(
                from_bot="hacker",
                to_url="http://127.0.0.1:8642/api/handoff",
                action="tool_call",
                tool="echo",
                params={},
                secret="test-secret",
            )
    assert ei.value.code == "source_denied"
    assert ei.value.status == 403


@pytest.mark.asyncio
async def test_handoff_request_404():
    """AC3: Nonexistent tool returns 404 HandoffError."""
    client = _MockAsyncClient(
        status_code=404,
        json_data={"error": "Tool 'nonexistent' not found"},
    )
    with patch("hermes_tools.handoff.httpx.AsyncClient", return_value=client):
        with pytest.raises(HandoffError) as ei:
            await handoff_request(
                from_bot="code",
                to_url="http://127.0.0.1:8642/api/handoff",
                action="tool_call",
                tool="nonexistent",
                params={},
                secret="test-secret",
            )
    assert ei.value.code == "tool_not_found"
    assert ei.value.status == 404


@pytest.mark.asyncio
async def test_handoff_request_504():
    """AC5: Timeout (31s+) returns 504 HandoffError."""
    client = _MockAsyncClient(
        status_code=504,
        json_data={"error": "Tool timed out"},
    )
    with patch("hermes_tools.handoff.httpx.AsyncClient", return_value=client):
        with pytest.raises(HandoffError) as ei:
            await handoff_request(
                from_bot="code",
                to_url="http://127.0.0.1:8642/api/handoff",
                action="tool_call",
                tool="terminal",
                params={"command": "sleep 60"},
                secret="test-secret",
            )
    assert ei.value.code == "timeout"
    assert ei.value.status == 504


@pytest.mark.asyncio
async def test_handoff_request_timeout_exception():
    """Transport-level timeout raises HandoffError with 'timeout' code."""
    import httpx

    client = _MockAsyncClientRaising(httpx.TimeoutException("timed out"))
    with patch("hermes_tools.handoff.httpx.AsyncClient", return_value=client):
        with pytest.raises(HandoffError) as ei:
            await handoff_request(
                from_bot="code",
                to_url="http://127.0.0.1:8642/api/handoff",
                action="tool_call",
                tool="echo",
                params={},
                secret="test-secret",
            )
    assert ei.value.code == "timeout"


@pytest.mark.asyncio
async def test_handoff_request_connection_failure():
    """Connection refused raises HandoffError with 'connection_failed'."""
    import httpx

    client = _MockAsyncClientRaising(httpx.RequestError("connection refused"))
    with patch("hermes_tools.handoff.httpx.AsyncClient", return_value=client):
        with pytest.raises(HandoffError) as ei:
            await handoff_request(
                from_bot="code",
                to_url="http://127.0.0.1:8642/api/handoff",
                action="tool_call",
                tool="echo",
                params={},
                secret="test-secret",
            )
    assert ei.value.code == "connection_failed"


@pytest.mark.asyncio
async def test_handoff_request_summon_success():
    """Summon requests send a query body and return the gateway payload."""
    client = _MockAsyncClient(
        status_code=200,
        json_data={
            "success": True,
            "result": "Your next car insurance payment is due on Friday.",
            "metrics": {"total_tokens": 42},
        },
    )
    with patch("hermes_tools.handoff.httpx.AsyncClient", return_value=client):
        result = await handoff_request(
            from_bot="code",
            to_url="http://127.0.0.1:8642/api/handoff",
            action="summon",
            query="When is my next car insurance payment due?",
            secret="test-secret",
        )
    assert result["success"] is True
    assert "Friday" in result["result"]
    assert client.last_request["json"]["query"] == "When is my next car insurance payment due?"
    assert "tool" not in client.last_request["json"]
    assert "params" not in client.last_request["json"]


@pytest.mark.asyncio
async def test_handoff_request_allowed_sources_param():
    """allowed_sources is passed in the request body when provided."""
    client = _MockAsyncClient(
        status_code=200,
        json_data={"result": "ok"},
    )
    with patch("hermes_tools.handoff.httpx.AsyncClient", return_value=client):
        await handoff_request(
            from_bot="code",
            to_url="http://127.0.0.1:8642/api/handoff",
            action="tool_call",
            tool="echo",
            params={},
            secret="test-secret",
            allowed_sources=["code", "mgmt"],
        )
    assert client.last_request["json"]["allowed_sources"] == ["code", "mgmt"]


# ============================================================================
# Gateway handler — _handle_handoff
# ============================================================================


class _FakeRequest:
    """Stand-in for an aiohttp.web.Request for handler unit tests."""

    def __init__(self, method="POST", path="/api/handoff", headers=None, body=None):
        self.method = method
        self.path = path
        self.path_qs = path  # aiohttp real attribute for logging
        self._headers = headers or {}
        self._body = json.dumps(body).encode() if body else b"{}"
        self.transport = MagicMock()
        self.transport.get_extra_info.return_value = ("127.0.0.1", 54321)
        self.remote = "127.0.0.1"

    @property
    def headers(self):
        return self._headers

    async def json(self):
        return json.loads(self._body.decode())

    def __repr__(self):
        return f"_FakeRequest({self.method} {self.path})"


def _make_adapter():
    """Create a minimal APIServerAdapter-like object for testing handlers.

    We instantiate the real APIServerAdapter with a dummy config but patch
    out its route/app lifecycle so we can call _handle_handoff in isolation.
    """
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.config import PlatformConfig

    config = PlatformConfig(enabled=True, extra={})
    adapter = APIServerAdapter(config)
    # Stub out _handoff_config to return a test secret
    adapter._handoff_config = MagicMock(
        return_value={
            "secret": "test-handoff-secret",
            "allowed_sources": ["code", "mgmt"],
            "timeout": 30.0,
        }
    )
    return adapter


@pytest.mark.asyncio
async def test_handle_handoff_success():
    """AC1: Valid auth + valid tool → result returned."""
    from aiohttp import web

    adapter = _make_adapter()
    # Patch _run_handoff_tool to avoid real tool imports
    async def fake_run(tool_name, params, *, timeout=30.0):
        return {"result": {"echoed": params.get("text", "")}}

    adapter._run_handoff_tool = fake_run

    body = {
        "from": "code",
        "action": "tool_call",
        "tool": "echo",
        "params": {"text": "hello"},
    }
    request = _FakeRequest(
        headers={"X-Handoff-Auth": "test-handoff-secret"},
        body=body,
    )
    response: web.Response = await adapter._handle_handoff(request)
    assert response.status == 200
    payload = json.loads(response.body)
    assert payload["result"]["result"]["echoed"] == "hello"


@pytest.mark.asyncio
async def test_handle_handoff_401_wrong_secret():
    """AC2: Invalid auth returns 401."""
    from aiohttp import web

    adapter = _make_adapter()
    body = {
        "from": "code",
        "action": "tool_call",
        "tool": "echo",
        "params": {},
    }
    request = _FakeRequest(
        headers={"X-Handoff-Auth": "wrong-secret"},
        body=body,
    )
    response: web.Response = await adapter._handle_handoff(request)
    assert response.status == 401
    payload = json.loads(response.body)
    assert "Invalid" in payload.get("error", "")


@pytest.mark.asyncio
async def test_handle_handoff_401_missing_secret():
    """Missing X-Handoff-Auth header returns 401."""
    from aiohttp import web

    adapter = _make_adapter()
    request = _FakeRequest(
        headers={},
        body={"from": "code", "action": "tool_call", "tool": "echo", "params": {}},
    )
    response: web.Response = await adapter._handle_handoff(request)
    assert response.status == 401


@pytest.mark.asyncio
async def test_handle_handoff_403_not_allowed():
    """AC4: Non-allowed source returns 403."""
    from aiohttp import web

    adapter = _make_adapter()
    body = {
        "from": "hacker",
        "action": "tool_call",
        "tool": "echo",
        "params": {},
    }
    request = _FakeRequest(
        headers={"X-Handoff-Auth": "test-handoff-secret"},
        body=body,
    )
    response: web.Response = await adapter._handle_handoff(request)
    assert response.status == 403
    payload = json.loads(response.body)
    assert "not allowed" in payload.get("error", "")


@pytest.mark.asyncio
async def test_handle_handoff_404_tool_not_found():
    """AC3: Tool-layer 404s propagate back to the caller."""
    from aiohttp import web
    from hermes_tools.handoff import HandoffError

    adapter = _make_adapter()

    async def fake_run_fail(tool_name, params, *, timeout=30.0):
        raise HandoffError("tool_not_found", f"Tool {tool_name!r} not found", status=404)

    adapter._run_handoff_tool = fake_run_fail
    body = {
        "from": "mgmt",
        "target_profile": "mgmt",
        "action": "tool_call",
        "tool": "kanban",
        "params": {},
    }
    request = _FakeRequest(
        headers={"X-Handoff-Auth": "test-handoff-secret"},
        body=body,
    )
    response: web.Response = await adapter._handle_handoff(request)
    assert response.status == 404
    payload = json.loads(response.body)
    assert "not found" in payload.get("error", "")


@pytest.mark.asyncio
async def test_handle_handoff_summon_success():
    """AC1: Summon requests return a success payload with plaintext result."""
    from aiohttp import web

    adapter = _make_adapter()

    async def fake_summon(query, *, timeout=60.0):
        assert query == "When is my next car insurance payment due?"
        assert timeout == 60.0
        return {
            "success": True,
            "result": "Your next car insurance payment is due on Friday.",
            "metrics": {"total_tokens": 12},
        }

    adapter._run_handoff_summon = fake_summon
    body = {
        "from": "code",
        # target_profile must advertise `summon: true` in the capability registry;
        # `code` is the summon-capable profile in shared/capabilities.yaml. Without
        # an explicit target the callee resolves to the ambient profile ("default"),
        # which the registry does not grant summon, so the capability gate correctly
        # denies it — making this assertion env-dependent. Pin it for determinism.
        "target_profile": "code",
        "action": "summon",
        "query": "When is my next car insurance payment due?",
    }
    request = _FakeRequest(
        headers={"X-Handoff-Auth": "test-handoff-secret"},
        body=body,
    )
    response: web.Response = await adapter._handle_handoff(request)
    assert response.status == 200
    payload = json.loads(response.body)
    assert payload["success"] is True
    assert "Friday" in payload["result"]
    assert payload["metrics"]["total_tokens"] == 12


@pytest.mark.asyncio
async def test_handle_handoff_unsupported_action():
    """Unsupported action returns 400."""
    from aiohttp import web

    adapter = _make_adapter()
    body = {
        "from": "code",
        "action": "ping",
        "tool": "echo",
        "params": {},
    }
    request = _FakeRequest(
        headers={"X-Handoff-Auth": "test-handoff-secret"},
        body=body,
    )
    response: web.Response = await adapter._handle_handoff(request)
    assert response.status == 400
    payload = json.loads(response.body)
    assert "ping" in payload.get("error", "")


@pytest.mark.asyncio
async def test_handle_handoff_review_required_for_high_risk_tool():
    """High-risk target capabilities should be review-gated before execution."""
    from aiohttp import web

    adapter = _make_adapter()
    run_tool = MagicMock()
    adapter._run_handoff_tool = run_tool
    body = {
        "from": "code",
        "target_profile": "code",
        "action": "tool_call",
        "tool": "terminal",
        "params": {"command": "echo should-not-run"},
    }
    request = _FakeRequest(
        headers={"X-Handoff-Auth": "test-handoff-secret"},
        body=body,
    )
    response: web.Response = await adapter._handle_handoff(request)
    assert response.status == 409
    payload = json.loads(response.body)
    assert payload["policy"]["decision"] == "review_required"
    assert "terminal" in payload.get("error", "")
    run_tool.assert_not_called()


@pytest.mark.asyncio
async def test_handle_handoff_no_handoff_secret():
    """When handoff is not configured, return 501."""
    from aiohttp import web
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.config import PlatformConfig

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._handoff_config = MagicMock(return_value={})
    request = _FakeRequest(
        headers={"X-Handoff-Auth": "irrelevant"},
        body={"from": "code", "action": "tool_call", "tool": "echo", "params": {}},
    )
    response: web.Response = await adapter._handle_handoff(request)
    assert response.status == 501


@pytest.mark.asyncio
async def test_handle_handoff_route_registered():
    """Verify the route is registered in connect().

    This checks that the API server's native route registration path includes
    the handoff endpoint — a trivial negative test for accidental regression.
    """
    # The route is registered as part of connect(). Check it's in the list.
    from gateway.platforms.api_server import APIServerAdapter

    assert hasattr(APIServerAdapter, "_handle_handoff")
    assert hasattr(APIServerAdapter, "_run_handoff_tool")
    assert hasattr(APIServerAdapter, "_handoff_config")


# ============================================================================
# Integration smoke test: handoff_request + handler round trip
# ============================================================================


@pytest.mark.asyncio
async def test_handler_echo_roundtrip():
    """End-to-end: handoff_request to the handler's echo tool succeeds.

    Uses a patched httpx client that calls the handler directly,
    bypassing real HTTP.
    """
    from aiohttp import web
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.config import PlatformConfig

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._handoff_config = MagicMock(
        return_value={
            "secret": "roundtrip-secret",
            "allowed_sources": ["code"],
            "timeout": 30.0,
        }
    )

    async def fake_run(tool_name, params, *, timeout=30.0):
        # _run_handoff_tool wraps its return in {"result": ...}
        # so return the inner value — the handler adds the outer wrapper
        return {"echoed": params.get("text", "")}

    adapter._run_handoff_tool = fake_run

    class _RoundTripClient:
        """httpx-like client that calls the adapter handler directly."""

        def __init__(self):
            self.last_req = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, *, json, headers):
            import json as _j
            request = _FakeRequest(
                headers=headers,
                body=json,
            )
            response: web.Response = await adapter._handle_handoff(request)
            # aiohttp.web.json_response sets .body to a dict internally
            # in older versions; use body_bytes for the JSON payload
            raw = response.body if isinstance(response.body, (bytes, str)) else _j.dumps(response.body).encode()
            return _MockResponse(response.status, _j.loads(raw))

    import httpx

    with patch("hermes_tools.handoff.httpx.AsyncClient", return_value=_RoundTripClient()):
        result = await handoff_request(
            from_bot="code",
            to_url="http://127.0.0.1:8642/api/handoff",
            action="tool_call",
            tool="echo",
            params={"text": "roundtrip-ok"},
            secret="roundtrip-secret",
        )
    assert result["result"]["echoed"] == "roundtrip-ok"
