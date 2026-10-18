"""HTTP client for the Hermes Agent API."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, AsyncGenerator

import aiohttp

from .compat import normalize_host, normalize_profile, normalize_profile_route
from .const import (
    API_CAPABILITIES,
    API_CHAT_COMPLETIONS,
    API_HEALTH,
    API_HEALTH_DETAILED,
    API_MODELS,
    API_RUNS,
    DEFAULT_MODEL,
    DEFAULT_STREAM_TIMEOUT,
    DEFAULT_TIMEOUT,
    ERROR_CATEGORY_AUTH,
    ERROR_CATEGORY_MALFORMED,
    ERROR_CATEGORY_REDIRECT,
    ERROR_CATEGORY_SERVER,
    ERROR_CATEGORY_TIMEOUT,
    ERROR_CATEGORY_UNREACHABLE,
    ERROR_CATEGORY_UNSUPPORTED,
    HEALTH_STATUSES,
    REDIRECT_STATUS_CODES,
    SPLIT_RUNTIME_CAPABILITY_TTL_SECONDS,
    UNSUPPORTED_DETAILED_STATUS_CODES,
    ProfileRouteFamily,
)

_LOGGER = logging.getLogger(__name__)
_NATIVE_ROUTE_CANARY_PROFILE = "hermes"


class HermesApiError(Exception):
    """Base exception for Hermes API errors."""


class HermesConnectionError(HermesApiError):
    """Cannot reach the Hermes Agent API."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_category: str = ERROR_CATEGORY_UNREACHABLE,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_category = error_category


class HermesAuthError(HermesApiError):
    """Authentication failed."""


class HermesStreamSetupError(HermesApiError):
    """Explicitly known pre-dispatch streaming rejection from an adapter.

    An HTTP failure from a submitted chat POST is never proof of pre-dispatch
    rejection: the server may already have executed model or tool work.
    """


@dataclass(slots=True)
class HermesApiResult:
    """Result wrapper for Hermes API chat-completions calls."""

    text: str
    session_id: str | None


@dataclass(slots=True, frozen=True)
class HermesClientToolCall:
    """A split-runtime tool call Hermes relayed for local execution."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True, frozen=True)
class HermesClientToolResult:
    """Outcome of a locally executed split-runtime tool call."""

    output: Any
    error: bool = False


ClientToolExecutor = Callable[[HermesClientToolCall], Awaitable[HermesClientToolResult]]

_RUN_TERMINAL_FAILURES = frozenset(
    {"run.failed", "run.cancelled", "run.interrupted"}
)


@dataclass(slots=True, frozen=True)
class HermesDetailedHealthResult:
    """Whitelisted optional /health/detailed outcome; never stores raw payload."""

    available: bool
    status: str | None
    error_category: str | None
    auth_failed: bool = False


class HermesApiClient:
    """Client for the Hermes Agent OpenAI-compatible API."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        host: str,
        port: int,
        api_key: str | None = None,
        use_ssl: bool = True,
        verify_ssl: bool = False,
        profile: str | None = None,
        model: str | None = None,
        request_timeout: int = DEFAULT_TIMEOUT,
        stream_timeout: int = DEFAULT_STREAM_TIMEOUT,
        profile_route: ProfileRouteFamily | str | None = None,
    ) -> None:
        self._session = session
        scheme = "https" if use_ssl else "http"
        normalized_host = normalize_host(host)
        url_host = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
        root_url = f"{scheme}://{url_host}:{port}"
        normalized_route = normalize_profile_route(profile_route)
        normalized_profile = normalize_profile(profile, normalized_route)
        self._root_url = root_url
        self._profile = normalized_profile
        self._profile_route = normalized_route
        route_prefix = (
            "profile"
            if normalized_route is ProfileRouteFamily.ADDON
            else "p"
        )
        self._base_url = (
            f"{root_url}/{route_prefix}/{normalized_profile}"
            if normalized_profile
            else root_url
        )
        self._api_key = api_key
        self._model = model or DEFAULT_MODEL
        self._request_timeout = max(1, int(request_timeout))
        self._stream_timeout = max(self._request_timeout, int(stream_timeout))
        # ssl=False disables certificate verification (for self-signed certs)
        self._ssl: bool | None = None if not use_ssl else (None if verify_ssl else False)
        self._last_session_id: str | None = None
        self._split_runtime_cache: tuple[bool, float] | None = None

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def last_session_id(self) -> str | None:
        """Most recent X-Hermes-Session-Id observed from the API."""
        return self._last_session_id

    def _headers(self, session_id: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if session_id:
            headers["X-Hermes-Session-Id"] = session_id
        return headers

    def _classify_status_error(self, status: int) -> HermesConnectionError:
        if status in REDIRECT_STATUS_CODES:
            category = ERROR_CATEGORY_REDIRECT
        elif status >= 500:
            category = ERROR_CATEGORY_SERVER
        else:
            category = ERROR_CATEGORY_UNREACHABLE
        return HermesConnectionError(
            "Hermes API request failed",
            status_code=status,
            error_category=category,
        )

    async def _async_verify_native_profile_route(self) -> bool:
        """Verify that a named native route selects and identifies its profile."""
        if not (
            self._profile_route is ProfileRouteFamily.NATIVE
            and self._profile
        ):
            return False

        timeout = aiohttp.ClientTimeout(total=10)
        canary_url = (
            f"{self._root_url}/p/{_NATIVE_ROUTE_CANARY_PROFILE}{API_HEALTH}"
        )
        try:
            async with self._session.get(
                canary_url,
                headers={},
                timeout=timeout,
                ssl=self._ssl,
                allow_redirects=False,
            ) as resp:
                if resp.status != 404:
                    raise self._classify_status_error(resp.status)

            async with self._session.get(
                f"{self._base_url}{API_HEALTH}",
                headers={},
                timeout=timeout,
                ssl=self._ssl,
                allow_redirects=False,
            ) as resp:
                if resp.status != 200:
                    raise self._classify_status_error(resp.status)
                health = await resp.json()
                if not isinstance(health, dict) or (
                    health.get("status") != "ok"
                    or health.get("platform") != "hermes-agent"
                ):
                    raise HermesConnectionError(
                        "Native profile route did not identify a Hermes Agent API",
                        error_category=ERROR_CATEGORY_MALFORMED,
                    )
            return True
        except HermesConnectionError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as err:
            raise HermesConnectionError(
                "Cannot verify native Hermes profile routing",
                error_category=_transport_error_category(err),
            ) from err

    async def async_check_connection(self) -> bool:
        """Check Hermes identity, reachability, and Bearer authentication."""
        timeout = aiohttp.ClientTimeout(total=10)
        legacy_health_missing = False
        try:
            native_route_verified = (
                await self._async_verify_native_profile_route()
            )

            if not native_route_verified:
                async with self._session.get(
                    f"{self._base_url}{API_HEALTH}",
                    headers={},
                    timeout=timeout,
                    ssl=self._ssl,
                    allow_redirects=False,
                ) as resp:
                    if resp.status == 404:
                        # Hermes releases before 2026-03-28 do not expose
                        # /v1/health. The authenticated models probe below remains
                        # authoritative for both API identity and reachability.
                        legacy_health_missing = True
                    elif resp.status != 200:
                        raise self._classify_status_error(resp.status)
                    else:
                        health = await resp.json()
                        if not isinstance(health, dict) or (
                            health.get("status") != "ok"
                            or health.get("platform") != "hermes-agent"
                        ):
                            raise HermesConnectionError(
                                "Health endpoint did not identify a Hermes Agent API",
                                error_category=ERROR_CATEGORY_MALFORMED,
                            )

            async with self._session.get(
                f"{self._base_url}{API_MODELS}",
                headers=self._headers(),
                timeout=timeout,
                ssl=self._ssl,
                allow_redirects=False,
            ) as resp:
                if resp.status in (401, 403):
                    raise HermesAuthError("Invalid API key")
                if resp.status != 200:
                    raise self._classify_status_error(resp.status)
                models = await resp.json()
                if not isinstance(models, dict) or not isinstance(
                    models.get("data"), list
                ):
                    raise HermesConnectionError(
                        "Models endpoint returned an invalid Hermes API response",
                        error_category=ERROR_CATEGORY_MALFORMED,
                    )
                if legacy_health_missing and not any(
                    isinstance(model, dict) and model.get("owned_by") == "hermes"
                    for model in models["data"]
                ):
                    raise HermesConnectionError(
                        "Legacy models endpoint did not identify a Hermes Agent API"
                    )
            return True
        except (HermesAuthError, HermesConnectionError):
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as err:
            raise HermesConnectionError(
                "Cannot connect to Hermes Agent",
                error_category=_transport_error_category(err),
            ) from err

    async def async_get_detailed_health(self) -> HermesDetailedHealthResult:
        """Fetch optional authenticated /health/detailed on the selected route."""
        await self._async_verify_native_profile_route()
        timeout = aiohttp.ClientTimeout(total=10)
        try:
            async with self._session.get(
                f"{self._base_url}{API_HEALTH_DETAILED}",
                headers=self._headers(),
                timeout=timeout,
                ssl=self._ssl,
                allow_redirects=False,
            ) as resp:
                if resp.status in (401, 403):
                    return HermesDetailedHealthResult(
                        available=False,
                        status=None,
                        error_category=ERROR_CATEGORY_AUTH,
                        auth_failed=True,
                    )
                if (
                    resp.status in REDIRECT_STATUS_CODES
                    or resp.status in UNSUPPORTED_DETAILED_STATUS_CODES
                ):
                    return HermesDetailedHealthResult(
                        available=False,
                        status=None,
                        error_category=ERROR_CATEGORY_UNSUPPORTED,
                    )
                if resp.status != 200:
                    return HermesDetailedHealthResult(
                        available=False,
                        status=None,
                        error_category=(
                            ERROR_CATEGORY_SERVER
                            if resp.status >= 500
                            else ERROR_CATEGORY_UNREACHABLE
                        ),
                    )
                payload = await resp.json()
                return _parse_detailed_health(payload)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as err:
            return HermesDetailedHealthResult(
                available=False,
                status=None,
                error_category=_transport_error_category(err),
            )

    async def async_get_models(self) -> list[dict[str, Any]]:
        """Fetch available models from /v1/models."""
        await self._async_verify_native_profile_route()
        try:
            async with self._session.get(
                f"{self._base_url}{API_MODELS}",
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=10),
                ssl=self._ssl,
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data.get("data", [])
        except Exception:
            return []

    async def async_send_message(
        self,
        messages: list[dict[str, str]],
        session_id: str | None = None,
    ) -> HermesApiResult:
        """Send a non-streaming chat completion request."""
        await self._async_verify_native_profile_route()
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }

        try:
            async with self._session.post(
                f"{self._base_url}{API_CHAT_COMPLETIONS}",
                headers=self._headers(session_id=session_id),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._request_timeout),
                ssl=self._ssl,
            ) as resp:
                if resp.status in (401, 403):
                    raise HermesAuthError("Invalid API key")
                if resp.status >= 400:
                    raise self._classify_status_error(resp.status)
                data = await resp.json()
                if _chat_completion_failed(data) or resp.headers.get("X-Hermes-Completed", "").lower() == "false":
                    raise HermesConnectionError(
                        "Hermes API reported an incomplete chat response",
                        error_category=ERROR_CATEGORY_SERVER,
                    )
                resolved_session_id = resp.headers.get("X-Hermes-Session-Id") or session_id
                self._last_session_id = resolved_session_id
                return HermesApiResult(
                    text=self._extract_content(data),
                    session_id=resolved_session_id,
                )
        except HermesApiError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            raise HermesConnectionError(
                "Connection error",
                error_category=_transport_error_category(err),
            ) from err

    async def async_stream_message(
        self,
        messages: list[dict[str, str]],
        session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Send a streaming chat completion request. Yields content deltas."""
        await self._async_verify_native_profile_route()
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }

        try:
            async with self._session.post(
                f"{self._base_url}{API_CHAT_COMPLETIONS}",
                headers=self._headers(session_id=session_id),
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=self._stream_timeout,
                    sock_read=self._request_timeout,
                ),
                ssl=self._ssl,
            ) as resp:
                if resp.status in (401, 403):
                    raise HermesAuthError("Invalid API key")
                if resp.status >= 400:
                    raise self._classify_status_error(resp.status)

                self._last_session_id = resp.headers.get("X-Hermes-Session-Id") or session_id

                # Parse SSE stream
                buffer = ""
                event_name = "message"
                terminal_failed = False
                try:
                    async for chunk in resp.content.iter_any():
                        buffer += chunk.decode("utf-8", errors="replace")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.rstrip("\r")

                            if not line:
                                event_name = "message"
                                continue
                            line = line.strip()
                            if line.startswith(":"):
                                continue
                            if line.startswith("event:"):
                                event_name = line[6:].strip() or "message"
                                continue
                            if line == "data: [DONE]":
                                if terminal_failed:
                                    raise HermesConnectionError(
                                        "Hermes API reported an incomplete chat response",
                                        error_category=ERROR_CATEGORY_SERVER,
                                    )
                                return
                            if not line.startswith("data: "):
                                continue
                            if event_name == "error":
                                terminal_failed = True
                                continue
                            if event_name != "message" or terminal_failed:
                                continue

                            try:
                                data = json.loads(line[6:])
                                if _chat_completion_failed(data):
                                    terminal_failed = True
                                    continue
                                delta = (
                                    data.get("choices", [{}])[0]
                                    .get("delta", {})
                                    .get("content")
                                )
                                if isinstance(delta, str) and delta:
                                    yield delta
                            except (json.JSONDecodeError, IndexError):
                                continue
                except (aiohttp.ClientError, asyncio.TimeoutError) as err:
                    if terminal_failed:
                        raise HermesConnectionError(
                            "Hermes API reported an incomplete chat response",
                            error_category=ERROR_CATEGORY_SERVER,
                        ) from err
                    raise
                if terminal_failed:
                    raise HermesConnectionError(
                        "Hermes API reported an incomplete chat response",
                        error_category=ERROR_CATEGORY_SERVER,
                    )
                raise HermesConnectionError("Stream ended without a completion marker")

        except HermesApiError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            raise HermesConnectionError(
                "Stream connection error",
                error_category=_transport_error_category(err),
            ) from err

    async def async_split_runtime_available(self) -> bool:
        """Return whether the selected route advertises split-runtime client tools."""
        now = time.monotonic()
        if self._split_runtime_cache is not None:
            available, checked_at = self._split_runtime_cache
            if now - checked_at < SPLIT_RUNTIME_CAPABILITY_TTL_SECONDS:
                return available

        available = False
        try:
            await self._async_verify_native_profile_route()
            async with self._session.get(
                f"{self._base_url}{API_CAPABILITIES}",
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=10),
                ssl=self._ssl,
                allow_redirects=False,
            ) as resp:
                if resp.status == 200:
                    payload = await resp.json()
                    runtime = payload.get("runtime") if isinstance(payload, dict) else None
                    available = (
                        isinstance(runtime, dict)
                        and runtime.get("split_runtime") is True
                    )
        except (HermesApiError, aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            _LOGGER.debug("Hermes capabilities probe failed", exc_info=True)

        self._split_runtime_cache = (available, now)
        return available

    async def async_stream_run(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]],
        execute_tool: ClientToolExecutor,
        session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Run one turn via /v1/runs with client tools. Yields content deltas.

        Tool calls Hermes relays as ``tool_call.request`` are executed through
        ``execute_tool`` and answered on ``/v1/runs/{run_id}/tool_result``.
        """
        await self._async_verify_native_profile_route()
        body: dict[str, Any] = {"model": self._model, "tools": tools}
        body.update(_split_run_messages(messages))
        if session_id:
            body["session_id"] = session_id

        try:
            async with self._session.post(
                f"{self._base_url}{API_RUNS}",
                headers=self._headers(),
                json=body,
                timeout=aiohttp.ClientTimeout(total=self._request_timeout),
                ssl=self._ssl,
            ) as resp:
                if resp.status in (401, 403):
                    raise HermesAuthError("Invalid API key")
                if resp.status == 400:
                    # Admission validation (e.g. a tool name reserved by Hermes):
                    # the run was never started, so a plain chat retry is safe.
                    raise HermesStreamSetupError("Hermes rejected the split-runtime run")
                if resp.status >= 400:
                    raise self._classify_status_error(resp.status)
                admission = await resp.json()
        except HermesApiError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as err:
            raise HermesConnectionError(
                "Connection error",
                error_category=_transport_error_category(err),
            ) from err

        run_id = admission.get("run_id") if isinstance(admission, dict) else None
        if not isinstance(run_id, str) or not run_id:
            raise HermesConnectionError(
                "Hermes run admission returned no run_id",
                error_category=ERROR_CATEGORY_MALFORMED,
            )
        # Without an explicit session_id Hermes keys the new session by run_id.
        self._last_session_id = session_id or run_id

        finished = False
        yielded_content = False
        try:
            async for event in self._async_iter_run_events(run_id):
                event_name = event.get("event")
                if event_name == "message.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str) and delta:
                        yielded_content = True
                        yield delta
                elif event_name == "tool_call.request":
                    await self._async_answer_tool_call(run_id, event, execute_tool)
                elif event_name == "run.completed":
                    finished = True
                    output = event.get("output")
                    if not yielded_content and isinstance(output, str) and output:
                        yield output
                    return
                elif event_name in _RUN_TERMINAL_FAILURES:
                    finished = True
                    raise HermesConnectionError(
                        "Hermes API reported an incomplete run",
                        error_category=ERROR_CATEGORY_SERVER,
                    )
            raise HermesConnectionError("Run stream ended without a terminal event")
        except HermesApiError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            raise HermesConnectionError(
                "Run stream connection error",
                error_category=_transport_error_category(err),
            ) from err
        finally:
            if not finished:
                await self._async_stop_run(run_id)

    async def _async_iter_run_events(
        self, run_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Yield JSON payloads from the run's SSE stream until it closes."""
        async with self._session.get(
            f"{self._base_url}{API_RUNS}/{run_id}/events",
            headers={**self._headers(), "Accept": "text/event-stream"},
            timeout=aiohttp.ClientTimeout(
                total=self._stream_timeout,
                sock_read=self._request_timeout,
            ),
            ssl=self._ssl,
        ) as resp:
            if resp.status in (401, 403):
                raise HermesAuthError("Invalid API key")
            if resp.status >= 400:
                raise self._classify_status_error(resp.status)

            buffer = ""
            async for chunk in resp.content.iter_any():
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        payload = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        yield payload

    async def _async_answer_tool_call(
        self,
        run_id: str,
        event: dict[str, Any],
        execute_tool: ClientToolExecutor,
    ) -> None:
        call_id = event.get("call_id")
        name = event.get("name")
        if not isinstance(call_id, str) or not isinstance(name, str):
            raise HermesConnectionError(
                "Hermes relayed a malformed client tool call",
                error_category=ERROR_CATEGORY_MALFORMED,
            )
        arguments = event.get("arguments")
        call = HermesClientToolCall(
            call_id=call_id,
            name=name,
            arguments=arguments if isinstance(arguments, dict) else {},
        )
        result = await execute_tool(call)
        async with self._session.post(
            f"{self._base_url}{API_RUNS}/{run_id}/tool_result",
            headers=self._headers(),
            json={"call_id": call_id, "output": result.output, "error": result.error},
            timeout=aiohttp.ClientTimeout(total=self._request_timeout),
            ssl=self._ssl,
        ) as resp:
            if resp.status in (401, 403):
                raise HermesAuthError("Invalid API key")
            if resp.status == 409:
                # The relay already expired or was cancelled; the run stream
                # reports how the turn ends.
                _LOGGER.warning("Hermes no longer awaits client tool call %s", name)
                return
            if resp.status >= 400:
                raise self._classify_status_error(resp.status)

    async def _async_stop_run(self, run_id: str) -> None:
        """Best-effort stop so Hermes does not wait for an abandoned tool result."""
        try:
            async with self._session.post(
                f"{self._base_url}{API_RUNS}/{run_id}/stop",
                headers=self._headers(),
                json={},
                timeout=aiohttp.ClientTimeout(total=5),
                ssl=self._ssl,
            ):
                pass
        except Exception:
            _LOGGER.debug("Stopping Hermes run %s failed", run_id, exc_info=True)

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        """Extract the assistant message content from a chat completion response."""
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return data.get("error", {}).get("message", "(No response)")


def _split_run_messages(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Map chat messages onto /v1/runs ``instructions``/``conversation_history``/``input``."""
    system_parts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    turns = [m for m in messages if m.get("role") != "system"]
    fields: dict[str, Any] = {"input": turns[-1]["content"] if turns else ""}
    if system_parts:
        fields["instructions"] = "\n\n".join(system_parts)
    if len(turns) > 1:
        fields["conversation_history"] = [
            {"role": m["role"], "content": m["content"]} for m in turns[:-1]
        ]
    return fields


def _chat_completion_failed(data: Any) -> bool:
    """Interpret Hermes terminal metadata without exposing its raw error details."""
    if not isinstance(data, dict):
        return False
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        if choices[0].get("finish_reason") in ("error", "length"):
            return True
    hermes = data.get("hermes")
    if isinstance(hermes, dict) and (
        hermes.get("completed") is False
        or hermes.get("partial") is True
        or hermes.get("failed") is True
    ):
        return True
    return "error" in data and data["error"] is not None


def _transport_error_category(err: BaseException) -> str:
    if isinstance(err, asyncio.TimeoutError):
        return ERROR_CATEGORY_TIMEOUT
    if isinstance(err, ValueError):
        return ERROR_CATEGORY_MALFORMED
    return ERROR_CATEGORY_UNREACHABLE


def _parse_detailed_health(payload: Any) -> HermesDetailedHealthResult:
    if not isinstance(payload, dict):
        return HermesDetailedHealthResult(
            available=False,
            status=None,
            error_category=ERROR_CATEGORY_MALFORMED,
        )
    status = payload.get("status")
    if not isinstance(status, str) or status not in HEALTH_STATUSES:
        return HermesDetailedHealthResult(
            available=False,
            status=None,
            error_category=ERROR_CATEGORY_MALFORMED,
        )
    platform = payload.get("platform")
    if platform is not None and platform != "hermes-agent":
        return HermesDetailedHealthResult(
            available=False,
            status=None,
            error_category=ERROR_CATEGORY_MALFORMED,
        )
    return HermesDetailedHealthResult(
        available=True,
        status=status,
        error_category=None,
    )
