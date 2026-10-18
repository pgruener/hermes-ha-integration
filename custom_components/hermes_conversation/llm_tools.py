"""Bridge Home Assistant LLM API tools to Hermes split-runtime client tools."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .api import ClientToolExecutor, HermesClientToolCall, HermesClientToolResult

_LOGGER = logging.getLogger(__name__)

# Keys strict OpenAI-style function schemas reject at the top level.
_UNSUPPORTED_SCHEMA_KEYS = frozenset({"oneOf", "anyOf", "allOf", "enum", "not"})


@dataclass(slots=True, frozen=True)
class SplitRuntimeTurn:
    """Client tools offered to Hermes for one conversation turn."""

    tools: list[dict[str, Any]]
    api_prompt: str
    execute_tool: ClientToolExecutor


def _schema_to_openapi(
    schema: Any, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    try:
        import probatio  # noqa: PLC0415

        return probatio.to_openapi(schema, custom_serializer=custom_serializer)
    except ImportError:
        from voluptuous_openapi import convert  # noqa: PLC0415

        return convert(schema, custom_serializer=custom_serializer)


def format_tool(
    tool: Any, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format a Home Assistant ``llm.Tool`` as an OpenAI function tool."""
    schema = _schema_to_openapi(tool.parameters, custom_serializer)
    if _UNSUPPORTED_SCHEMA_KEYS.intersection(schema):
        schema = {k: v for k, v in schema.items() if k not in _UNSUPPORTED_SCHEMA_KEYS}
    function: dict[str, Any] = {"name": tool.name, "parameters": schema}
    if tool.description:
        function["description"] = tool.description
    return {"type": "function", "function": function}


def build_split_runtime_turn(llm_api: Any) -> SplitRuntimeTurn | None:
    """Return the client tools for ``llm_api``, or None when it offers none."""
    tools = list(getattr(llm_api, "tools", None) or ())
    if not tools:
        return None
    custom_serializer = getattr(llm_api, "custom_serializer", None)

    async def _execute(call: HermesClientToolCall) -> HermesClientToolResult:
        return await async_execute_tool(llm_api, call)

    return SplitRuntimeTurn(
        tools=[format_tool(tool, custom_serializer) for tool in tools],
        api_prompt=getattr(llm_api, "api_prompt", "") or "",
        execute_tool=_execute,
    )


async def async_execute_tool(
    llm_api: Any, call: HermesClientToolCall
) -> HermesClientToolResult:
    """Execute one relayed tool call through Home Assistant's LLM API."""
    from homeassistant.helpers import llm  # noqa: PLC0415

    try:
        result = await llm_api.async_call_tool(
            llm.ToolInput(tool_name=call.name, tool_args=call.arguments, id=call.call_id)
        )
    except Exception as err:  # noqa: BLE001 - every failure goes back to the model
        _LOGGER.warning("Home Assistant tool %s failed: %s", call.name, err)
        return HermesClientToolResult(
            output={"error": type(err).__name__, "error_text": str(err)},
            error=True,
        )
    # Newer HA releases wrap the JSON object in ``ToolResult``.
    if hasattr(result, "data"):
        return HermesClientToolResult(output=result.data, error=bool(result.error))
    return HermesClientToolResult(output=result)
