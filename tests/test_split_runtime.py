from __future__ import annotations

import json
import sys
import types
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest import mock

from tests.test_support import FakeChatLog, FakeConfigEntry, FakeConversationInput, FakeHass
from tests.test_api import FakeResponse, FakeSession
from custom_components.hermes_conversation.api import (
    HermesApiClient,
    HermesClientToolCall,
    HermesClientToolResult,
    HermesConnectionError,
    HermesStreamSetupError,
)
from custom_components.hermes_conversation.const import CONF_API_KEY, CONF_LLM_HASS_API
from custom_components.hermes_conversation.conversation import HermesConversationAgent
from custom_components.hermes_conversation.llm_tools import (
    async_execute_tool,
    build_split_runtime_turn,
)


def _sse(*events: dict[str, Any]) -> list[str]:
    return [": open\n\n"] + [f"id: {i}\ndata: {json.dumps(e)}\n\n" for i, e in enumerate(events)]


def _client(session) -> HermesApiClient:
    return HermesApiClient(session, "hermes.local", 8642, "secret", use_ssl=False)


TOOLS = [{"type": "function", "function": {"name": "play_radio", "parameters": {}}}]


@dataclass
class FakeToolInput:
    tool_name: str
    tool_args: dict[str, Any]
    id: str = "generated"
    external: bool = False


@dataclass
class FakeToolResult:
    data: Any
    error: bool = False


class FakeLlmTool:
    def __init__(self, name, description="", parameters=None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}


class FakeLlmApi:
    def __init__(self, tools, result=None, error=None):
        self.tools = tools
        self.api_prompt = "Call Home Assistant tools to control devices."
        self.custom_serializer = None
        self.calls: list[FakeToolInput] = []
        self._result = result if result is not None else {"success": True}
        self._error = error

    async def async_call_tool(self, tool_input):
        self.calls.append(tool_input)
        if self._error is not None:
            raise self._error
        return self._result


def _llm_modules():
    llm = types.ModuleType("homeassistant.helpers.llm")
    llm.ToolInput = FakeToolInput
    probatio = types.ModuleType("probatio")
    probatio.to_openapi = lambda schema, custom_serializer=None: {
        "type": "object",
        "properties": dict(schema),
    }
    return {"homeassistant.helpers.llm": llm, "probatio": probatio}


class SplitRuntimeApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_relays_tool_call_and_streams_reply(self):
        session = FakeSession(
            [
                FakeResponse(status=202, json_data={"run_id": "run_1", "status": "queued"}),
                FakeResponse(
                    chunks=_sse(
                        {"event": "message.delta", "delta": "Moment. "},
                        {
                            "event": "tool_call.request",
                            "call_id": "call_1",
                            "name": "play_radio",
                            "arguments": {"station_uuid": "abc"},
                        },
                        {"event": "tool_call.responded", "call_id": "call_1"},
                        {"event": "message.delta", "delta": "Radio Eins läuft."},
                        {"event": "run.completed", "output": "Moment. Radio Eins läuft."},
                    )
                ),
                FakeResponse(json_data={"resolved": True}),
            ]
        )
        executed: list[HermesClientToolCall] = []

        async def execute(call):
            executed.append(call)
            return HermesClientToolResult(output={"playing": True})

        client = _client(session)
        chunks = [
            chunk
            async for chunk in client.async_stream_run(
                [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "earlier"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "Spiel Radio Eins"},
                ],
                tools=TOOLS,
                execute_tool=execute,
            )
        ]

        self.assertEqual(chunks, ["Moment. ", "Radio Eins läuft."])
        self.assertEqual(
            executed,
            [HermesClientToolCall("call_1", "play_radio", {"station_uuid": "abc"})],
        )
        run_request = session.calls[0]
        self.assertEqual(run_request["url"], "http://hermes.local:8642/v1/runs")
        self.assertEqual(run_request["json"]["input"], "Spiel Radio Eins")
        self.assertEqual(run_request["json"]["instructions"], "Be brief.")
        self.assertEqual(
            run_request["json"]["conversation_history"],
            [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}],
        )
        self.assertEqual(run_request["json"]["tools"], TOOLS)
        self.assertNotIn("session_id", run_request["json"])
        self.assertEqual(session.calls[1]["url"], "http://hermes.local:8642/v1/runs/run_1/events")
        self.assertEqual(
            session.calls[2]["json"],
            {"call_id": "call_1", "output": {"playing": True}, "error": False},
        )
        self.assertEqual(client.last_session_id, "run_1")

    async def test_run_continues_given_session_and_falls_back_to_output(self):
        session = FakeSession(
            [
                FakeResponse(status=202, json_data={"run_id": "run_2"}),
                FakeResponse(chunks=_sse({"event": "run.completed", "output": "Fertig."})),
            ]
        )
        client = _client(session)

        chunks = [
            chunk
            async for chunk in client.async_stream_run(
                [{"role": "user", "content": "hi"}],
                tools=TOOLS,
                execute_tool=None,
                session_id="sess-9",
            )
        ]

        self.assertEqual(chunks, ["Fertig."])
        self.assertEqual(session.calls[0]["json"]["session_id"], "sess-9")
        self.assertEqual(client.last_session_id, "sess-9")

    async def test_rejected_admission_is_a_stream_setup_error(self):
        session = FakeSession([FakeResponse(status=400)])

        with self.assertRaises(HermesStreamSetupError):
            async for _ in _client(session).async_stream_run(
                [{"role": "user", "content": "hi"}], tools=TOOLS, execute_tool=None
            ):
                pass

    async def test_failed_run_raises_without_stop(self):
        session = FakeSession(
            [
                FakeResponse(status=202, json_data={"run_id": "run_3"}),
                FakeResponse(chunks=_sse({"event": "run.failed", "error": "boom"})),
            ]
        )

        with self.assertRaises(HermesConnectionError):
            async for _ in _client(session).async_stream_run(
                [{"role": "user", "content": "hi"}], tools=TOOLS, execute_tool=None
            ):
                pass
        self.assertEqual(len(session.calls), 2)

    async def test_truncated_stream_stops_the_run(self):
        session = FakeSession(
            [
                FakeResponse(status=202, json_data={"run_id": "run_4"}),
                FakeResponse(chunks=_sse({"event": "message.delta", "delta": "Hal"})),
                FakeResponse(json_data={}),
            ]
        )

        with self.assertRaises(HermesConnectionError):
            async for _ in _client(session).async_stream_run(
                [{"role": "user", "content": "hi"}], tools=TOOLS, execute_tool=None
            ):
                pass
        self.assertEqual(session.calls[-1]["url"], "http://hermes.local:8642/v1/runs/run_4/stop")

    async def test_capabilities_probe_is_cached(self):
        session = FakeSession(
            [FakeResponse(json_data={"runtime": {"split_runtime": True}})]
        )
        client = _client(session)

        self.assertTrue(await client.async_split_runtime_available())
        self.assertTrue(await client.async_split_runtime_available())
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(session.calls[0]["url"], "http://hermes.local:8642/v1/capabilities")

    async def test_capabilities_probe_defaults_to_unavailable(self):
        session = FakeSession([FakeResponse(json_data={"runtime": {"split_runtime": False}})])

        self.assertFalse(await _client(session).async_split_runtime_available())


class LlmToolsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, _llm_modules())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_formats_ha_tools_as_openai_functions(self):
        api = FakeLlmApi([FakeLlmTool("play_radio", "Play a station", {"station_uuid": "str"})])

        turn = build_split_runtime_turn(api)

        self.assertEqual(
            turn.tools,
            [
                {
                    "type": "function",
                    "function": {
                        "name": "play_radio",
                        "parameters": {"type": "object", "properties": {"station_uuid": "str"}},
                        "description": "Play a station",
                    },
                }
            ],
        )
        self.assertEqual(turn.api_prompt, api.api_prompt)

    def test_api_without_tools_offers_no_turn(self):
        self.assertIsNone(build_split_runtime_turn(FakeLlmApi([])))

    async def test_execute_unwraps_tool_result(self):
        api = FakeLlmApi([], result=FakeToolResult(data={"speech": "ok"}, error=True))

        result = await async_execute_tool(api, HermesClientToolCall("c1", "play_radio", {"a": 1}))

        self.assertEqual(result, HermesClientToolResult(output={"speech": "ok"}, error=True))
        self.assertEqual(api.calls, [FakeToolInput("play_radio", {"a": 1}, id="c1")])

    async def test_execute_reports_exceptions_as_errors(self):
        api = FakeLlmApi([], error=ValueError("bad entity"))

        result = await async_execute_tool(api, HermesClientToolCall("c1", "play_radio", {}))

        self.assertTrue(result.error)
        self.assertEqual(result.output, {"error": "ValueError", "error_text": "bad entity"})


class SplitClient:
    def __init__(self, *, available=True):
        self.available = available
        self.last_session_id = None
        self.calls: list[dict[str, Any]] = []

    async def async_split_runtime_available(self):
        return self.available

    async def async_stream_run(self, messages, *, tools, execute_tool, session_id=None):
        self.calls.append({"method": "run", "messages": messages, "tools": tools})
        await execute_tool(HermesClientToolCall("call_1", "play_radio", {"station_uuid": "abc"}))
        self.last_session_id = session_id or "run_1"
        yield "Radio Eins läuft."

    async def async_stream_message(self, messages, session_id=None):
        self.calls.append({"method": "stream", "messages": messages})
        yield "plain"


class LlmInput(FakeConversationInput):
    def as_llm_context(self, domain):
        return SimpleNamespace(platform=domain)


class SplitConversationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, _llm_modules())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.llm_api = FakeLlmApi([FakeLlmTool("play_radio", "Play a station")])
        llm_api = self.llm_api
        self.provided: list[tuple] = []
        provided = self.provided

        async def provide(chat_log, llm_context, api_ids, prompt, extra):
            provided.append((llm_context.platform, api_ids))
            chat_log.llm_api = llm_api

        patcher = mock.patch.object(
            FakeChatLog, "async_provide_llm_data", provide, create=True
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _entry(self, api_ids):
        return FakeConfigEntry(
            data={CONF_API_KEY: "secret"},
            options={CONF_LLM_HASS_API: api_ids},
        )

    async def test_selected_llm_api_tools_run_in_home_assistant(self):
        client = SplitClient()
        agent = HermesConversationAgent(FakeHass(), self._entry(["assist"]), client, session_map={})

        result = await agent.async_process(LlmInput("Spiel Radio Eins", conversation_id="c1"))

        self.assertEqual(result.response.speech["plain"]["speech"], "Radio Eins läuft.")
        self.assertEqual(self.provided, [("hermes_conversation", ["assist"])])
        self.assertEqual(client.calls[0]["method"], "run")
        self.assertEqual(client.calls[0]["tools"][0]["function"]["name"], "play_radio")
        self.assertIn(self.llm_api.api_prompt, client.calls[0]["messages"][0]["content"])
        self.assertEqual(self.llm_api.calls[0].tool_name, "play_radio")
        self.assertEqual(self.llm_api.calls[0].id, "call_1")

    async def test_route_without_split_runtime_keeps_plain_chat(self):
        client = SplitClient(available=False)
        agent = HermesConversationAgent(FakeHass(), self._entry(["assist"]), client, session_map={})

        await agent.async_process(LlmInput("hi", conversation_id="c2"))

        self.assertEqual(client.calls[0]["method"], "stream")
        self.assertEqual(self.provided, [])

    async def test_no_selected_llm_api_keeps_plain_chat(self):
        client = SplitClient()
        agent = HermesConversationAgent(FakeHass(), self._entry([]), client, session_map={})

        await agent.async_process(LlmInput("hi", conversation_id="c3"))

        self.assertEqual(client.calls[0]["method"], "stream")
