"""Bot-to-bot handoff: direct structured requests between Hermes gateways.

Usage by a bot agent::

    from hermes_tools import handoff_request

    result = await handoff_request(
        from_bot="code",
        to_url="http://127.0.0.1:8645/api/handoff",
        action="tool_call",
        tool="terminal",
        params={"command": "date"},
        secret="<shared-handoff-secret>",
    )

This module also provides the error type (`HandoffError`) used by both
callers and gateway endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

from tools.delegation_audit import record_delegation_audit

logger = logging.getLogger(__name__)

DEFAULT_HANDOFF_TIMEOUT = 30.0


class HandoffError(Exception):
    """Raised on transport, auth, or server-side handoff failures.

    Attributes:
        code: Machine-readable error code (e.g. ``timeout``, ``auth_denied``,
            ``tool_not_found``, ``server_error``, ``unexpected_status``).
        status: HTTP status code, or 0 for transport-level errors.
        detail: Human-readable explanation.
    """

    def __init__(
        self,
        code: str,
        detail: str,
        status: int = 0,
    ):
        self.code = code
        self.status = status
        self.detail = detail
        super().__init__(f"[{code}] {detail}")


async def handoff_request(
    from_bot: str,
    to_url: str,
    action: str,
    tool: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    secret: str = "",
    timeout: float = DEFAULT_HANDOFF_TIMEOUT,
    allowed_sources: Optional[list[str]] = None,
    query: Optional[str] = None,
    context_pack: Optional[Dict[str, Any]] = None,
    target_profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a structured handoff request to another Hermes gateway.

    This is the **caller** side — used by bots that want to invoke a tool
    on another bot's gateway.

    Args:
        from_bot: Name of the requesting bot (used for source allowlisting).
        to_url: Callee's endpoint URL (e.g. ``http://127.0.0.1:8642/api/handoff``).
        action: Action to perform. Currently only ``tool_call`` is supported.
        tool: Tool name to invoke on the callee (e.g. ``terminal``, ``read_file``).
        params: Tool parameters as a JSON-serializable dict.
        secret: Shared ``X-Handoff-Auth`` secret (must match the callee's config).
        timeout: Total timeout in seconds (default 30).  The callee also applies
            a 30s internal timeout, so this should be >= 30.
        allowed_sources: If set, the callee will verify that ``from_bot`` is in
            this list.  Passed through to the request for callee-side enforcement.

    Returns:
        The tool's result dict (``{"result": ...}`` on success).

    Raises:
        HandoffError: On transport failure, auth rejection, or callee-side error.
    """
    body: Dict[str, Any] = {
        "from": from_bot,
        "action": action,
    }
    if allowed_sources is not None:
        body["allowed_sources"] = allowed_sources

    if action == "tool_call":
        if not tool:
            raise ValueError("tool is required for action='tool_call'")
        body["tool"] = tool
        body["params"] = params or {}
    elif action == "summon":
        if not query:
            raise ValueError("query is required for action='summon'")
        body["query"] = query
    else:
        if tool is not None:
            body["tool"] = tool
        if params is not None:
            body["params"] = params
        if query is not None:
            body["query"] = query

    if context_pack is not None:
        body["context_pack"] = context_pack
    if target_profile is not None:
        body["target_profile"] = target_profile

    headers = {
        "Content-Type": "application/json",
        "X-Handoff-Auth": secret,
    }

    correlation_id = ""
    task_id = ""
    session_id = ""
    if isinstance(context_pack, dict):
        correlation_id = str(context_pack.get("correlation_id") or "")
        task_id = str(context_pack.get("task_id") or "")
        session_id = str(context_pack.get("session_id") or "")

    callee_profile = urlparse(to_url).netloc or to_url
    audit_params = {
        "action": action,
        "tool": tool,
        "params": params or {},
        "query": query,
        "allowed_sources": allowed_sources or [],
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(to_url, json=body, headers=headers)
    except httpx.TimeoutException as exc:
        record_delegation_audit(
            action="review_required",
            caller_profile=from_bot,
            callee_profile=callee_profile,
            parameters={**audit_params, "timeout_seconds": timeout},
            correlation_id=correlation_id,
            task_id=task_id,
            session_id=session_id,
            reason=str(exc),
            source="handoff_request",
        )
        raise HandoffError("timeout", f"Handoff request timed out after {timeout}s", status=0)
    except httpx.RequestError as exc:
        record_delegation_audit(
            action="deny",
            caller_profile=from_bot,
            callee_profile=callee_profile,
            parameters=audit_params,
            correlation_id=correlation_id,
            task_id=task_id,
            session_id=session_id,
            reason=str(exc),
            source="handoff_request",
        )
        raise HandoffError(
            "connection_failed",
            f"Could not connect to {to_url}: {exc}",
            status=0,
        )

    try:
        result = _process_response(response, tool or action)
    except HandoffError as exc:
        action_name = "review_required" if exc.code == "timeout" else "deny"
        record_delegation_audit(
            action=action_name,
            caller_profile=from_bot,
            callee_profile=callee_profile,
            parameters={**audit_params, "status_code": response.status_code},
            correlation_id=correlation_id,
            task_id=task_id,
            session_id=session_id,
            reason=exc.detail,
            source="handoff_request",
        )
        raise

    record_delegation_audit(
        action="allow",
        caller_profile=from_bot,
        callee_profile=callee_profile,
        parameters={**audit_params, "status_code": response.status_code},
        correlation_id=correlation_id,
        task_id=task_id,
        session_id=session_id,
        reason="handoff succeeded",
        source="handoff_request",
    )
    return result


def resolve_handoff_target(
    target: str,
    *,
    registry: "CapabilityRegistry | None" = None,
):
    """Resolve a direct profile name or a capability alias for routing.

    This is the optional capability-aware helper for summon/handoff routers.
    Existing direct-profile callers can keep passing profile names unchanged;
    capability-aware callers can pass a capability alias and use the returned
    resolution to choose the concrete profile.
    """
    from hermes_cli.capability_registry import resolve_target

    return resolve_target(target, registry=registry)


def _process_response(response: httpx.Response, tool: str) -> Dict[str, Any]:
    """Parse the handoff gateway response and raise on errors."""
    if response.status_code == 401:
        raise HandoffError("auth_denied", "Invalid X-Handoff-Auth secret", status=401)
    if response.status_code == 403:
        raise HandoffError("source_denied", "Caller not in allowed_sources", status=403)
    if response.status_code == 404:
        raise HandoffError("tool_not_found", f"Tool {tool!r} not found on callee", status=404)
    if response.status_code == 504:
        raise HandoffError("timeout", "Handoff tool execution timed out (31s+)", status=504)

    try:
        payload = response.json()
    except json.JSONDecodeError:
        raise HandoffError(
            "bad_response",
            f"Non-JSON response (HTTP {response.status_code})",
            status=response.status_code,
        )

    if response.status_code >= 500:
        detail = payload.get("error", str(payload))
        raise HandoffError("server_error", str(detail), status=response.status_code)

    if response.status_code != 200:
        detail = payload.get("error", str(payload))
        raise HandoffError("unexpected_status", str(detail), status=response.status_code)

    if "error" in payload:
        raise HandoffError(
            "tool_error",
            str(payload["error"]),
            status=response.status_code,
        )

    return payload  # {"result": ...}
