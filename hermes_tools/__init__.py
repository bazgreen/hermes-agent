"""Hermes tools for bot-to-bot handoff and internal communication.

This package provides utilities that let Hermes bots communicate directly
with each other without going through Mgmt as a routing bottleneck.

Public API:
- handoff_request(from_bot, to_url, action, tool, params, secret, timeout=30)
- HandoffError — raised on transport or auth failures
"""

from hermes_tools.handoff import handoff_request, HandoffError, resolve_handoff_target

__all__ = [
    "handoff_request",
    "HandoffError",
    "resolve_handoff_target",
]
