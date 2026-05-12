"""Transport layer types and registry for provider response normalization.

Usage:
    from agent.transports import get_transport
    transport = get_transport("chat_completions")
    result = transport.normalize_response(raw_response)
"""

from typing import Optional, Type, Dict

from agent.transports.types import (
    NormalizedResponse,
    ToolCall,
    Usage,
    build_tool_call,
    map_finish_reason,
)  # noqa: F401

_REGISTRY: Dict[str, Type] = {}
_discovered = True


def register_transport(api_mode: str, transport_cls: type) -> None:
    """Register a transport class for an api_mode string."""
    _REGISTRY[api_mode] = transport_cls


def get_transport(api_mode: str = "chat_completions"):
    """Get a transport instance for the given api_mode.

    Only supports 'chat_completions' (OpenAI-compatible API).
    Returns None if no transport is registered for this api_mode.
    """
    cls = _REGISTRY.get(api_mode)
    if cls is None:
        return None
    return cls()


def _discover_transports() -> None:
    """Discover and register all available transports."""
    global _discovered
    if _discovered:
        return
    # Import ChatCompletionsTransport which auto-registers itself
    # Must import the module, not just the class, to trigger registration
    import agent.transports.chat_completions  # noqa: F401
    _discovered = True


# Auto-discover transports on module load
_discover_transports()
