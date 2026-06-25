"""Runtime patch for OpenHarness Codex subscription transport retries.

The upstream Codex client retries httpx.NetworkError, but httpx raises
RemoteProtocolError for interrupted chunked/SSE responses.  That class is a
TransportError, not a NetworkError, so transient "incomplete chunked read"
disconnects otherwise terminate a whole benchmark trial.
"""

from __future__ import annotations

import os


_OH_NETWORK_ERROR_MARKERS = (
    "network error",
    "incomplete chunked read",
    "peer closed connection",
    "server disconnected",
    "remote protocol",
    "connection reset",
    "connection aborted",
    "transport error",
)


def _patch_codex_retry() -> None:
    try:
        import httpx
        from openharness.api import codex_client
    except Exception:
        return

    original = codex_client.CodexApiClient._is_retryable
    try:
        codex_client.MAX_RETRIES = max(
            int(getattr(codex_client, "MAX_RETRIES", 3)),
            int(os.environ.get("CLAW_OPENHARNESS_CODEX_MAX_RETRIES", "8")),
        )
    except Exception:
        pass

    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, httpx.TransportError):
            return True
        message = str(exc).lower()
        if any(
            term in message
            for term in (
                "incomplete chunked read",
                "peer closed connection",
                "server disconnected",
                "remote protocol",
                "connection reset",
            )
        ):
            return True
        try:
            return bool(original(exc))
        except Exception:
            return False

    codex_client.CodexApiClient._is_retryable = staticmethod(_is_retryable)


def _is_oh_network_error(message: str) -> bool:
    msg = message.lower()
    return any(marker in msg for marker in _OH_NETWORK_ERROR_MARKERS)


def _first_oh_network_error(events: list[dict]) -> str | None:
    for ev in events:
        if ev.get("type") != "error":
            continue
        message = str(ev.get("message", ""))
        if _is_oh_network_error(message):
            return message
    return None


def _patch_claw_openharness_network_failures() -> None:
    try:
        from claw_anything.agents.openharness_agent import OpenHarnessAgent
    except Exception:
        return

    if getattr(OpenHarnessAgent, "_claw_oh_network_patch", False):
        return

    original_run_oh = OpenHarnessAgent._run_oh_subprocess
    original_execute = OpenHarnessAgent._execute

    def _run_oh_subprocess(self, *args, **kwargs):
        events, return_code = original_run_oh(self, *args, **kwargs)
        self._claw_last_oh_network_error = _first_oh_network_error(events)
        return events, return_code

    def _execute(self, *args, **kwargs):
        original_execute(self, *args, **kwargs)
        message = getattr(self, "_claw_last_oh_network_error", None)
        if message:
            raise RuntimeError(f"[oh-network] {message[:500]}")

    OpenHarnessAgent._run_oh_subprocess = _run_oh_subprocess
    OpenHarnessAgent._execute = _execute
    OpenHarnessAgent._claw_oh_network_patch = True


_patch_codex_retry()
_patch_claw_openharness_network_failures()
