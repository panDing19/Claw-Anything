"""Generate a per-task OpenHarness plugin that exposes claw-anything tools.

For each evaluation trial we materialise an isolated OH config dir with one
plugin (`clawanything`) whose ``tools/clawanything_tools.py`` defines a ``BaseTool``
subclass per ``ToolEndpoint`` declared in the task. OH's plugin loader picks
them up at startup; the generated tools forward calls to the mock-service HTTP
endpoints already running in the parent process and append a side-channel
record to ``$CLAW_ANYTHING_DISPATCH_LOG`` so the OpenHarness-family agents can
rebuild ``ToolDispatch`` events later.

Both ``OpenHarnessAgent`` (vanilla openharness-ai) and ``OpenHarnessExtAgent``
(OH-Ext fork) use this module. Each agent owns its own builtin-tool list (see
``builtin_tools_for_deny`` on each class) and passes it via the
``extra_denied_tools`` parameter on ``generate_plugin_files``. This module is
deliberately ignorant of which specific tools exist for which fork — that
knowledge lives next to the agent that needs it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from ..models.task import TaskDefinition


PLUGIN_NAME = "clawanything"


_PLUGIN_MANIFEST = {
    "name": PLUGIN_NAME,
    "version": "1.0.0",
    "description": "Auto-generated claw-anything task tools",
    "enabled_by_default": True,
    "tools_dir": "tools",
}


# Tools that are ALWAYS denied during a claw-anything trial, regardless of the
# ``--oh-disable-builtin-tools`` flag. These tools break the non-interactive batch
# eval contract — e.g. ``ask_user_question`` blocks the agent loop waiting for
# human input that will never arrive, so the trial just burns wall-clock until
# its deadline. Add new entries here when a builtin is found to be incompatible
# with unattended evaluation.
ALWAYS_DENIED_TOOLS = [
    "ask_user_question",
]


def _load_settings(path: Path | None) -> dict:
    """Load a user-provided OH settings.json as the baseline (or empty dict).

    Raises FileNotFoundError / ValueError with a helpful message if the path
    is set but unreadable or not valid JSON.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--oh-settings file does not exist: {p}")
    try:
        text = p.read_text(encoding="utf-8")
        if not text.strip():
            return {}
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--oh-settings file is not valid JSON ({p}): {exc}") from exc


def _build_settings(
    *,
    extra_denied_tools: list[str] | None = None,
    base_settings: dict | None = None,
    apps: list[dict] | None = None,
    execution_date: str | None = None,
    print_mode_extra_fields: list[str] | None = None,
) -> dict:
    """Merge clawanything-required fields onto a (possibly user-provided) base settings dict.

    Merge rules:
      - ``enabled_plugins`` — additive: user's existing plugin flags kept,
        ``clawanything`` forced to True so OH loads our generated plugin.
      - ``allow_project_plugins`` — overridden to False (claw-anything owns the
        plugin set; we don't want OH scanning the cwd for stray plugins).
      - ``permission.mode`` — overridden to ``full_auto`` so OH never
        prompts (claw-anything batch is non-interactive).
      - ``permission.denied_tools`` — always unioned with
        ``ALWAYS_DENIED_TOOLS`` (tools that don't make sense under
        non-interactive eval). When the caller passes ``extra_denied_tools``
        (typically the agent's own builtin-tool list to suppress them), those
        names are added too. This module never looks up tool names itself —
        the caller (the active agent) owns the list and passes the right one.
      - ``prompt_meta`` — top-level dict carrying task-time metadata for OH:
        ``today`` (execution_date, all tasks) and ``apps`` (mobile_gui tasks
        only). OH forwards this to the GUI backend at runtime. apps is omitted
        when empty; today is omitted when None.
      - ``print_mode.stream_json_extra_fields`` — additive: user's existing
        list kept, ``print_mode_extra_fields`` unioned in. Lets the agent
        declare which optional stream-json fields claw-anything's adapter
        needs (e.g. OH-Ext: ``["usage"]``). Vanilla OH ignores this section
        entirely; injecting it is harmless there because vanilla OH's
        Settings model tolerates unknown fields.
      - All other top-level fields from the base settings are preserved
        verbatim — that's the point of providing them.
    """
    settings: dict = json.loads(json.dumps(base_settings or {}))  # deep copy

    enabled = settings.get("enabled_plugins")
    if not isinstance(enabled, dict):
        enabled = {}
    enabled[PLUGIN_NAME] = True
    settings["enabled_plugins"] = enabled

    settings["allow_project_plugins"] = False

    perm = settings.get("permission")
    if not isinstance(perm, dict):
        perm = {}
    perm["mode"] = "full_auto"

    existing_denied = perm.get("denied_tools") or []
    if not isinstance(existing_denied, list):
        existing_denied = []
    extra_denied: list[str] = list(ALWAYS_DENIED_TOOLS)
    if extra_denied_tools:
        extra_denied.extend(extra_denied_tools)
    if existing_denied or extra_denied:
        perm["denied_tools"] = list(dict.fromkeys([*existing_denied, *extra_denied]))
    settings["permission"] = perm

    meta: dict = {}
    if apps:
        meta["apps"] = apps
    if execution_date:
        meta["today"] = execution_date
    if meta:
        existing_meta = settings.get("prompt_meta")
        if isinstance(existing_meta, dict):
            existing_meta.update(meta)
        else:
            settings["prompt_meta"] = meta

    # ``print_mode`` is only emitted when the agent actually requests extra
    # fields. Vanilla OH never sets this (its agent passes None/[]) so its
    # settings.json stays free of the key regardless of whether vanilla OH's
    # pydantic model would have tolerated the unknown section.
    if print_mode_extra_fields:
        pm = settings.get("print_mode")
        if not isinstance(pm, dict):
            pm = {}
        existing_pm_fields = pm.get("stream_json_extra_fields") or []
        if not isinstance(existing_pm_fields, list):
            existing_pm_fields = []
        pm["stream_json_extra_fields"] = list(
            dict.fromkeys([*existing_pm_fields, *print_mode_extra_fields])
        )
        settings["print_mode"] = pm

    return settings


_GENERATED_HEADER = '''"""Auto-generated by claw-anything. Do not edit."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class _AnyArgs(BaseModel):
    """Permissive Pydantic model: accepts any tool input.

    The real schema is exposed to the LLM via ``to_api_schema`` below.
    """

    model_config = ConfigDict(extra="allow")


def _emit_dispatch(record: dict) -> None:
    path = os.environ.get("CLAW_ANYTHING_DISPATCH_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\\n")
    except Exception:
        pass


def _make_tool_class(
    *,
    tool_name: str,
    tool_description: str,
    endpoint_url: str,
    method: str,
    input_schema: dict,
) -> type[BaseTool]:
    class _GenTool(BaseTool):
        name = tool_name
        description = tool_description
        input_model = _AnyArgs

        def to_api_schema(self) -> dict:
            return {
                "name": tool_name,
                "description": tool_description,
                "input_schema": input_schema,
            }

        def is_read_only(self, arguments: BaseModel) -> bool:
            return False

        async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
            payload = arguments.model_dump()
            t0 = time.monotonic()
            local_result = _handle_local_gui_tool(tool_name, endpoint_url, payload)
            if local_result is not None:
                status, body = local_result
                latency_ms = (time.monotonic() - t0) * 1000.0
                _emit_dispatch({
                    "tool_name": tool_name,
                    "endpoint_url": "local://gui/" + tool_name,
                    "request_body": payload,
                    "response_status": status,
                    "response_body": body,
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                })
                return ToolResult(
                    output=json.dumps(body, ensure_ascii=False),
                    is_error=status >= 400,
                )
            try:
                async with httpx.AsyncClient(timeout=30.0, trust_env=False) as cli:
                    resp = await cli.request(method=method, url=endpoint_url, json=payload)
                    status = resp.status_code
                    try:
                        body: Any = resp.json()
                    except Exception:
                        body = {"raw": resp.text}
                latency_ms = (time.monotonic() - t0) * 1000.0
                _emit_dispatch({
                    "tool_name": tool_name,
                    "endpoint_url": endpoint_url,
                    "request_body": payload,
                    "response_status": status,
                    "response_body": body,
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                })
                return ToolResult(
                    output=json.dumps(body, ensure_ascii=False),
                    is_error=status >= 400,
                )
            except Exception as exc:
                _emit_dispatch({
                    "tool_name": tool_name,
                    "endpoint_url": endpoint_url,
                    "request_body": payload,
                    "response_status": 0,
                    "response_body": {"error": str(exc)},
                    "latency_ms": (time.monotonic() - t0) * 1000.0,
                    "timestamp": time.time(),
                })
                return ToolResult(output=f"Error: {exc}", is_error=True)

    safe = re.sub(r"\\W", "_", tool_name) or "tool"
    _GenTool.__name__ = f"_GenTool_{safe}"
    _GenTool.__qualname__ = _GenTool.__name__
    return _GenTool


# --- Per-task tool registrations ---
'''


_LOCAL_GUI_HELPERS = r'''

_LOCAL_GUI_STATE: dict[str, Any] = {}


def _text_match(query: Any, *values: Any) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return True
    return any(q in str(value or "").lower() for value in values)


def _max_results(payload: dict, default: int = 50) -> int:
    try:
        return max(1, int(payload.get("max_results") or default))
    except (TypeError, ValueError):
        return default


def _record_id(record: dict, *fields: str) -> str:
    for field in fields:
        value = record.get(field)
        if value is not None and value != "":
            return str(value)
    for field in (
        "id",
        "event_id",
        "contact_id",
        "message_id",
        "transaction_id",
        "habit_id",
        "call_id",
        "product_id",
        "note_id",
    ):
        value = record.get(field)
        if value is not None and value != "":
            return str(value)
    return ""


def _find_record(records: list, value: Any, *fields: str) -> dict | None:
    wanted = str(value or "")
    if not wanted:
        return None
    for record in records:
        if not isinstance(record, dict):
            continue
        candidates = {_record_id(record, *fields)}
        for field in fields:
            raw = record.get(field)
            if raw is not None:
                candidates.add(str(raw))
        if wanted in candidates:
            return record
    return None


def _date_part(value: Any) -> str:
    return str(value or "")[:10]


def _in_date_range(value: Any, start: Any = None, end: Any = None) -> bool:
    date = _date_part(value)
    if not date:
        return True
    start_s = _date_part(start)
    end_s = _date_part(end)
    if start_s and date < start_s:
        return False
    if end_s and date > end_s:
        return False
    return True


def _body_preview(text: Any, limit: int = 180) -> str:
    s = str(text or "")
    return s if len(s) <= limit else s[:limit] + "..."


def _handle_fossify_messages(tool_name: str, payload: dict) -> tuple[int, Any]:
    state = _LOCAL_GUI_STATE.setdefault("fossify_messages", {})
    threads = state.setdefault("threads", [])
    sent = state.setdefault("sent_messages", [])

    if tool_name.endswith("_list_threads"):
        query = payload.get("query")
        unread_only = bool(payload.get("unread_only", False))
        max_results = int(payload.get("max_results") or 50)
        matches = []
        for thread in threads:
            messages = thread.get("messages") or []
            haystack = [
                thread.get("thread_id"),
                thread.get("contact_name"),
                thread.get("last_message"),
                " ".join(thread.get("participants") or []),
                " ".join(str(msg.get("text") or "") for msg in messages[-5:]),
            ]
            if not _text_match(query, *haystack):
                continue
            if unread_only and int(thread.get("unread_count") or 0) <= 0:
                continue
            matches.append({
                "thread_id": thread.get("thread_id"),
                "contact_name": thread.get("contact_name"),
                "participants": thread.get("participants") or [],
                "last_message": thread.get("last_message"),
                "last_message_time": thread.get("last_message_time"),
                "unread_count": int(thread.get("unread_count") or 0),
                "pinned": bool(thread.get("pinned", False)),
                "archived": bool(thread.get("archived", False)),
            })
        return 200, {"threads": matches[:max_results], "total": len(matches), "returned": min(len(matches), max_results)}

    if tool_name.endswith("_get_thread"):
        thread_id = payload.get("thread_id")
        for thread in threads:
            if thread.get("thread_id") == thread_id:
                return 200, thread
        return 404, {"error": f"Thread not found: {thread_id}"}

    if tool_name.endswith("_send_message"):
        text = str(payload.get("text") or payload.get("message") or "")
        if not text.strip():
            return 400, {"error": "text is required"}
        thread_id = payload.get("thread_id")
        phone_number = payload.get("phone_number")
        target = None
        for thread in threads:
            if thread_id and thread.get("thread_id") == thread_id:
                target = thread
                break
            if phone_number and phone_number in (thread.get("participants") or []):
                target = thread
                break
        if target is None:
            thread_id = thread_id or f"FSMS-LOCAL-{len(threads) + 1}"
            target = {
                "thread_id": thread_id,
                "contact_name": phone_number or "Unknown",
                "participants": [phone_number] if phone_number else [],
                "messages": [],
                "unread_count": 0,
                "pinned": False,
                "archived": False,
            }
            threads.append(target)
        message_id = f"LOCAL-SENT-{len(sent) + 1}"
        msg = {
            "id": message_id,
            "from": "me",
            "text": text,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "is_outgoing": True,
            "is_read": True,
        }
        target.setdefault("messages", []).append(msg)
        target["last_message"] = text
        target["last_message_time"] = msg["time"]
        record = {
            "message_id": message_id,
            "thread_id": target.get("thread_id"),
            "phone_number": phone_number,
            "text": text,
            "status": "sent",
        }
        sent.append(record)
        return 200, record

    if tool_name.endswith("_mark_thread"):
        thread_id = payload.get("thread_id")
        for thread in threads:
            if thread.get("thread_id") == thread_id:
                if "read" in payload:
                    thread["unread_count"] = 0 if payload.get("read") else thread.get("unread_count", 0)
                if "archived" in payload:
                    thread["archived"] = bool(payload.get("archived"))
                if "pinned" in payload:
                    thread["pinned"] = bool(payload.get("pinned"))
                return 200, {"status": "updated", "thread_id": thread_id}
        return 404, {"error": f"Thread not found: {thread_id}"}

    return 404, {"error": f"Unsupported Fossify Messages tool: {tool_name}"}


def _handle_fossify_notes(tool_name: str, payload: dict) -> tuple[int, Any]:
    state = _LOCAL_GUI_STATE.setdefault("fossify_notes", {})
    notes = state.setdefault("notes", [])

    if tool_name.endswith("_list_notes"):
        query = payload.get("query")
        tag = str(payload.get("tag") or "").strip().lower()
        max_results = int(payload.get("max_results") or 50)
        matches = []
        for note in notes:
            tags = note.get("tags") or []
            if tag and tag not in [str(t).lower() for t in tags]:
                continue
            if not _text_match(query, note.get("title"), note.get("content"), " ".join(tags)):
                continue
            matches.append({
                "note_id": note.get("note_id"),
                "title": note.get("title"),
                "updated_at": note.get("updated_at"),
                "tags": tags,
                "pinned": bool(note.get("pinned", False)),
            })
        return 200, {"notes": matches[:max_results], "total": len(matches), "returned": min(len(matches), max_results)}

    if tool_name.endswith("_get_note"):
        note_id = payload.get("note_id")
        for note in notes:
            if note.get("note_id") == note_id:
                return 200, note
        return 404, {"error": f"Note not found: {note_id}"}

    if tool_name.endswith("_create_note"):
        title = str(payload.get("title") or "").strip()
        content = str(payload.get("content") or "")
        if not title or not content:
            return 400, {"error": "title and content are required"}
        note_id = f"FNOT-LOCAL-{len(notes) + 1}"
        note = {
            "note_id": note_id,
            "title": title,
            "content": content,
            "color": payload.get("color"),
            "tags": payload.get("tags") or [],
            "checklist": bool(payload.get("checklist", False)),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        }
        notes.append(note)
        return 200, {"status": "created", **note}

    if tool_name.endswith("_update_note"):
        note_id = payload.get("note_id")
        for note in notes:
            if note.get("note_id") == note_id:
                for key in ("title", "content", "color", "tags"):
                    if key in payload:
                        note[key] = payload[key]
                note["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
                return 200, {"status": "updated", "note_id": note_id}
        return 404, {"error": f"Note not found: {note_id}"}

    return 404, {"error": f"Unsupported Fossify Notes tool: {tool_name}"}


def _handle_fossify_calendar(tool_name: str, payload: dict) -> tuple[int, Any]:
    state = _LOCAL_GUI_STATE.setdefault("fossify_calendar", {})
    events = state.setdefault("events", [])

    if tool_name.endswith("_list_events"):
        query = payload.get("query")
        start = payload.get("date_from") or payload.get("start_date") or payload.get("date")
        end = payload.get("date_to") or payload.get("end_date")
        max_results = _max_results(payload)
        matches = []
        for event in events:
            if not _in_date_range(event.get("start_time"), start, end):
                continue
            if not _text_match(
                query,
                event.get("event_id"),
                event.get("title"),
                event.get("description"),
                event.get("location"),
            ):
                continue
            matches.append(event)
        return 200, {"events": matches[:max_results], "total": len(matches), "returned": min(len(matches), max_results)}

    if tool_name.endswith("_get_event"):
        event_id = payload.get("event_id")
        event = _find_record(events, event_id, "event_id", "id")
        if event is None:
            return (400 if not event_id else 404), {"error": f"Event not found: {event_id}"}
        return 200, event

    if tool_name.endswith("_create_event"):
        title = str(payload.get("title") or "").strip()
        start_time = payload.get("start_time")
        end_time = payload.get("end_time")
        if not title or not start_time or not end_time:
            return 400, {"error": "title, start_time and end_time are required"}
        event_id = f"FCAL-LOCAL-{len(events) + 1}"
        event = {
            "event_id": event_id,
            "title": title,
            "start_time": start_time,
            "end_time": end_time,
            "location": payload.get("location") or "",
            "description": payload.get("description") or "",
            "recurring": payload.get("recurring") or "none",
            "reminder_minutes": payload.get("reminder_minutes"),
            "color": payload.get("color"),
        }
        events.append(event)
        return 200, {"status": "created", **event}

    if tool_name.endswith("_update_event"):
        event_id = payload.get("event_id")
        event = _find_record(events, event_id, "event_id", "id")
        if event is None:
            return (400 if not event_id else 404), {"error": f"Event not found: {event_id}"}
        for key in ("title", "start_time", "end_time", "location", "description", "recurring", "reminder_minutes", "color"):
            if key in payload:
                event[key] = payload[key]
        return 200, {"status": "updated", "event_id": event_id}

    return 404, {"error": f"Unsupported Fossify Calendar tool: {tool_name}"}


def _handle_contacts(tool_name: str, payload: dict) -> tuple[int, Any]:
    state = _LOCAL_GUI_STATE.setdefault("contacts", {})
    contacts = state.setdefault("contacts", [])

    if tool_name.endswith("_list"):
        query = payload.get("query")
        group = str(payload.get("group") or "").strip().lower()
        max_results = _max_results(payload)
        matches = []
        for contact in contacts:
            if group and group != str(contact.get("group") or "").lower():
                continue
            if not _text_match(query, *contact.values()):
                continue
            matches.append(contact)
        return 200, {"contacts": matches[:max_results], "total": len(matches), "returned": min(len(matches), max_results)}

    if tool_name.endswith("_get"):
        contact_id = payload.get("contact_id")
        contact = _find_record(contacts, contact_id, "contact_id", "id")
        if contact is None:
            return (400 if not contact_id else 404), {"error": f"Contact not found: {contact_id}"}
        return 200, contact

    if tool_name.endswith("_create"):
        name = str(payload.get("name") or "").strip()
        if not name:
            return 400, {"error": "name is required"}
        contact = {
            "contact_id": f"GCON-LOCAL-{len(contacts) + 1}",
            "name": name,
            "phone": payload.get("phone") or "",
            "email": payload.get("email") or "",
            "company": payload.get("company") or "",
            "title": payload.get("title") or "",
            "note": payload.get("note") or "",
            "group": payload.get("group") or "",
            "starred": bool(payload.get("starred", False)),
        }
        contacts.append(contact)
        return 200, {"status": "created", **contact}

    if tool_name.endswith("_update"):
        contact_id = payload.get("contact_id")
        contact = _find_record(contacts, contact_id, "contact_id", "id")
        if contact is None:
            return (400 if not contact_id else 404), {"error": f"Contact not found: {contact_id}"}
        for key in ("name", "phone", "email", "company", "title", "note", "group", "address", "birthday", "preferences", "starred"):
            if key in payload:
                contact[key] = payload[key]
        return 200, {"status": "updated", "contact_id": contact_id}

    return 404, {"error": f"Unsupported Contacts tool: {tool_name}"}


def _handle_gmail_clone(tool_name: str, payload: dict) -> tuple[int, Any]:
    state = _LOCAL_GUI_STATE.setdefault("gmail_clone", {})
    messages = state.setdefault("messages", [])
    sent = state.setdefault("sent_messages", [])
    drafts = state.setdefault("drafts", [])

    if tool_name.endswith("_list_messages"):
        query = payload.get("query")
        label = str(payload.get("label") or "").strip().lower()
        unread_only = bool(payload.get("unread_only", False))
        max_results = _max_results(payload)
        matches = []
        for message in messages:
            labels = [str(v).lower() for v in (message.get("labels") or [])]
            if label and label not in labels:
                continue
            if unread_only and bool(message.get("is_read", True)):
                continue
            if not _text_match(query, message.get("from"), message.get("to"), message.get("subject"), message.get("body"), " ".join(labels)):
                continue
            matches.append({
                "message_id": _record_id(message, "message_id", "id"),
                "id": message.get("id"),
                "from": message.get("from"),
                "to": message.get("to"),
                "subject": message.get("subject"),
                "date": message.get("date"),
                "is_read": bool(message.get("is_read", True)),
                "labels": message.get("labels") or [],
                "body_preview": _body_preview(message.get("body")),
            })
        return 200, {"messages": matches[:max_results], "total": len(matches), "returned": min(len(matches), max_results)}

    if tool_name.endswith("_get_message"):
        message_id = payload.get("message_id")
        message = _find_record(messages, message_id, "message_id", "id")
        if message is None:
            return (400 if not message_id else 404), {"error": f"Message not found: {message_id}"}
        message["is_read"] = True
        return 200, message

    if tool_name.endswith("_send_message") or tool_name.endswith("_save_draft"):
        to = str(payload.get("to") or "").strip()
        subject = str(payload.get("subject") or "").strip()
        body = str(payload.get("body") or "")
        if not to or not subject or not body:
            return 400, {"error": "to, subject and body are required"}
        is_draft = tool_name.endswith("_save_draft")
        message = {
            "message_id": ("GDRAFT-LOCAL-" if is_draft else "GMSG-SENT-LOCAL-") + str((len(drafts) if is_draft else len(sent)) + 1),
            "from": "me",
            "to": to,
            "cc": payload.get("cc"),
            "subject": subject,
            "body": body,
            "date": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "labels": ["drafts"] if is_draft else ["sent"],
            "thread_id": payload.get("thread_id"),
            "is_read": True,
        }
        (drafts if is_draft else sent).append(message)
        messages.append(message)
        return 200, {"status": "draft_saved" if is_draft else "sent", **message}

    if tool_name.endswith("_update_message"):
        message_id = payload.get("message_id")
        message = _find_record(messages, message_id, "message_id", "id")
        if message is None:
            return (400 if not message_id else 404), {"error": f"Message not found: {message_id}"}
        if "read" in payload:
            message["is_read"] = bool(payload.get("read"))
        labels = list(message.get("labels") or [])
        if payload.get("add_labels"):
            labels = list(dict.fromkeys([*labels, *payload.get("add_labels")]))
        if payload.get("remove_labels"):
            remove = set(payload.get("remove_labels") or [])
            labels = [label for label in labels if label not in remove]
        if payload.get("archived") and "archived" not in labels:
            labels.append("archived")
        if payload.get("trashed") and "trash" not in labels:
            labels.append("trash")
        message["labels"] = labels
        return 200, {"status": "updated", "message_id": message_id, "labels": labels}

    return 404, {"error": f"Unsupported Gmail Clone tool: {tool_name}"}


def _handle_my_expenses(tool_name: str, payload: dict) -> tuple[int, Any]:
    state = _LOCAL_GUI_STATE.setdefault("my_expenses", {})
    accounts = state.setdefault("accounts", [])
    transactions = state.setdefault("transactions", [])

    if tool_name.endswith("_list_accounts"):
        query = payload.get("query")
        matches = [account for account in accounts if _text_match(query, *account.values())]
        return 200, {"accounts": matches, "total": len(matches), "returned": len(matches)}

    if tool_name.endswith("_get_account"):
        account_id = payload.get("account_id")
        account = _find_record(accounts, account_id, "account_id", "id", "name", "label")
        if account is None:
            return (400 if not account_id else 404), {"error": f"Account not found: {account_id}"}
        return 200, account

    if tool_name.endswith("_list_transactions"):
        start = payload.get("date_from") or payload.get("start_date")
        end = payload.get("date_to") or payload.get("end_date")
        category = str(payload.get("category") or payload.get("category_id") or "").strip().lower()
        account = str(payload.get("account") or payload.get("account_id") or "").strip().lower()
        query = payload.get("query")
        max_results = _max_results(payload)
        matches = []
        for txn in transactions:
            if not _in_date_range(txn.get("date"), start, end):
                continue
            if category and category not in str(txn.get("category") or txn.get("category_id") or "").lower():
                continue
            if account and account not in str(txn.get("account") or txn.get("account_id") or "").lower():
                continue
            if not _text_match(query, *txn.values()):
                continue
            matches.append(txn)
        return 200, {"transactions": matches[:max_results], "total": len(matches), "returned": min(len(matches), max_results)}

    if tool_name.endswith("_get_transaction"):
        transaction_id = payload.get("transaction_id")
        txn = _find_record(transactions, transaction_id, "transaction_id", "id")
        if txn is None:
            return (400 if not transaction_id else 404), {"error": f"Transaction not found: {transaction_id}"}
        return 200, txn

    if tool_name.endswith("_create_transaction") or tool_name.endswith("_add_transaction"):
        if "amount" not in payload:
            return 400, {"error": "amount is required"}
        txn = {
            "transaction_id": f"TRX-LOCAL-{len(transactions) + 1}",
            "date": payload.get("date") or time.strftime("%Y-%m-%d", time.gmtime()),
            "account_id": payload.get("account_id"),
            "account": payload.get("account"),
            "category_id": payload.get("category_id"),
            "category": payload.get("category"),
            "payee": payload.get("payee"),
            "amount": payload.get("amount"),
            "description": payload.get("description") or payload.get("comment") or "",
            "notes": payload.get("notes") or payload.get("comment") or "",
            "tags": payload.get("tags") or [],
            "status": payload.get("status") or "cleared",
        }
        transactions.append(txn)
        return 200, {"status": "created", **txn}

    if tool_name.endswith("_update_transaction"):
        transaction_id = payload.get("transaction_id")
        txn = _find_record(transactions, transaction_id, "transaction_id", "id")
        if txn is None:
            return (400 if not transaction_id else 404), {"error": f"Transaction not found: {transaction_id}"}
        for key in ("date", "account_id", "account", "category_id", "category", "payee", "amount", "description", "notes", "comment", "tags", "status"):
            if key in payload:
                txn[key] = payload[key]
        return 200, {"status": "updated", "transaction_id": transaction_id}

    return 404, {"error": f"Unsupported My Expenses tool: {tool_name}"}


def _handle_loop_habit(tool_name: str, payload: dict) -> tuple[int, Any]:
    state = _LOCAL_GUI_STATE.setdefault("loop_habit", {})
    habits = state.setdefault("habits", [])

    if tool_name.endswith("_list_habits"):
        query = payload.get("query")
        matches = [habit for habit in habits if _text_match(query, habit.get("habit_id"), habit.get("name"), habit.get("frequency"), habit.get("unit"))]
        return 200, {"habits": matches, "total": len(matches), "returned": len(matches)}

    if tool_name.endswith("_get_habit"):
        habit_id = payload.get("habit_id")
        habit = _find_record(habits, habit_id, "habit_id", "id")
        if habit is None:
            return (400 if not habit_id else 404), {"error": f"Habit not found: {habit_id}"}
        return 200, habit

    if tool_name.endswith("_check_habit") or tool_name.endswith("_check_in"):
        habit_id = payload.get("habit_id")
        habit = _find_record(habits, habit_id, "habit_id", "id")
        if habit is None:
            return (400 if not habit_id else 404), {"error": f"Habit not found: {habit_id}"}
        date = payload.get("date") or time.strftime("%Y-%m-%d", time.gmtime())
        value = payload.get("value", habit.get("target_value", 1))
        completions = habit.setdefault("completions", [])
        existing = next((item for item in completions if isinstance(item, dict) and item.get("date") == date), None)
        if existing is None:
            completions.append({"date": date, "value": value})
        else:
            existing["value"] = value
        return 200, {"status": "checked", "habit_id": habit_id, "date": date, "value": value}

    if tool_name.endswith("_list_completions"):
        habit_id = payload.get("habit_id")
        habit = _find_record(habits, habit_id, "habit_id", "id")
        if habit is None:
            return (400 if not habit_id else 404), {"error": f"Habit not found: {habit_id}"}
        start = payload.get("start_date")
        end = payload.get("end_date")
        completions = [item for item in habit.get("completions", []) if isinstance(item, dict) and _in_date_range(item.get("date"), start, end)]
        return 200, {"habit_id": habit_id, "completions": completions, "total": len(completions)}

    return 404, {"error": f"Unsupported Loop Habit tool: {tool_name}"}


def _handle_clock(tool_name: str, payload: dict) -> tuple[int, Any]:
    state = _LOCAL_GUI_STATE.setdefault("clock", {})
    alarms = state.setdefault("alarms", [])

    if tool_name.endswith("_list_alarms"):
        query = payload.get("query")
        matches = [alarm for alarm in alarms if _text_match(query, alarm.get("alarm_id"), alarm.get("time"), alarm.get("label"))]
        return 200, {"alarms": matches, "total": len(matches), "returned": len(matches)}

    if tool_name.endswith("_get_alarm"):
        alarm_id = payload.get("alarm_id")
        alarm = _find_record(alarms, alarm_id, "alarm_id", "id")
        if alarm is None:
            return (400 if not alarm_id else 404), {"error": f"Alarm not found: {alarm_id}"}
        return 200, alarm

    if tool_name.endswith("_create_alarm"):
        time_value = str(payload.get("time") or "").strip()
        if not time_value:
            return 400, {"error": "time is required"}
        alarm = {
            "alarm_id": f"ALRM-LOCAL-{len(alarms) + 1}",
            "time": time_value,
            "label": payload.get("label") or "",
            "repeat_days": payload.get("repeat_days") or [],
            "enabled": bool(payload.get("enabled", True)),
            "ringtone": payload.get("ringtone") or "Default alarm sound",
            "vibrate": bool(payload.get("vibrate", True)),
        }
        alarms.append(alarm)
        return 200, {"status": "created", **alarm}

    if tool_name.endswith("_update_alarm"):
        alarm_id = payload.get("alarm_id")
        alarm = _find_record(alarms, alarm_id, "alarm_id", "id")
        if alarm is None:
            return (400 if not alarm_id else 404), {"error": f"Alarm not found: {alarm_id}"}
        for key in ("time", "label", "repeat_days", "enabled", "ringtone", "vibrate"):
            if key in payload:
                alarm[key] = payload[key]
        return 200, {"status": "updated", "alarm_id": alarm_id}

    return 404, {"error": f"Unsupported Clock tool: {tool_name}"}


def _handle_dialer(tool_name: str, payload: dict) -> tuple[int, Any]:
    state = _LOCAL_GUI_STATE.setdefault("dialer", {})
    calls = state.setdefault("call_log", [])

    if tool_name.endswith("_list_call_log"):
        call_type = str(payload.get("call_type") or "").strip().lower()
        max_results = _max_results(payload)
        matches = [
            call for call in calls
            if not call_type or call_type == str(call.get("call_type") or "").lower()
        ]
        return 200, {"calls": matches[:max_results], "total": len(matches), "returned": min(len(matches), max_results)}

    if tool_name.endswith("_get_call_details"):
        call_id = payload.get("call_id")
        call = _find_record(calls, call_id, "call_id", "id")
        if call is None:
            return (400 if not call_id else 404), {"error": f"Call not found: {call_id}"}
        return 200, call

    return 404, {"error": f"Unsupported Dialer tool: {tool_name}"}


def _handle_mattermost(tool_name: str, payload: dict) -> tuple[int, Any]:
    state = _LOCAL_GUI_STATE.setdefault("mattermost", {})
    messages = state.setdefault("messages", [])

    if tool_name.endswith("_list_channels"):
        channels: dict[str, dict] = {}
        for msg in messages:
            channel_id = str(msg.get("channel_id") or msg.get("channel_name") or "default")
            channel = channels.setdefault(
                channel_id,
                {
                    "channel_id": channel_id,
                    "channel_name": msg.get("channel_name") or channel_id,
                    "message_count": 0,
                    "unread_count": 0,
                },
            )
            channel["message_count"] += 1
        return 200, {"channels": list(channels.values()), "total": len(channels)}

    if tool_name.endswith("_send_message"):
        channel_id = str(payload.get("channel_id") or "").strip()
        text = str(payload.get("text") or "").strip()
        if not channel_id or not text:
            return 400, {"error": "channel_id and text are required"}
        msg = {
            "message_id": f"MMSG-LOCAL-{len(messages) + 1}",
            "channel_id": channel_id,
            "channel_name": channel_id,
            "author": "me",
            "text": text,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "reactions": [],
            "pinned": False,
            "root_id": payload.get("root_id"),
        }
        messages.append(msg)
        return 200, {"status": "sent", **msg}

    return 404, {"error": f"Unsupported Mattermost tool: {tool_name}"}


def _handle_testmall(tool_name: str, payload: dict) -> tuple[int, Any]:
    state = _LOCAL_GUI_STATE.setdefault("testmall", {})
    products = state.setdefault("products", [])

    if tool_name.endswith("_list_products"):
        query = payload.get("query")
        category = str(payload.get("category") or "").strip().lower()
        max_results = _max_results(payload)
        min_price = payload.get("min_price")
        max_price = payload.get("max_price")
        matches = []
        for product in products:
            if category and category not in str(product.get("category") or "").lower():
                continue
            price = product.get("price")
            try:
                numeric_price = float(price)
            except (TypeError, ValueError):
                numeric_price = None
            if min_price is not None and numeric_price is not None and numeric_price < float(min_price):
                continue
            if max_price is not None and numeric_price is not None and numeric_price > float(max_price):
                continue
            if not _text_match(query, *product.values()):
                continue
            matches.append(product)
        return 200, {"products": matches[:max_results], "total": len(matches), "returned": min(len(matches), max_results)}

    if tool_name.endswith("_get_product"):
        product_id = payload.get("product_id")
        product = _find_record(products, product_id, "product_id", "id", "sku")
        if product is None:
            return (400 if not product_id else 404), {"error": f"Product not found: {product_id}"}
        return 200, product

    return 404, {"error": f"Unsupported TestMall tool: {tool_name}"}


def _handle_local_gui_tool(tool_name: str, endpoint_url: str, payload: dict) -> tuple[int, Any] | None:
    if "/gui/" not in str(endpoint_url):
        return None
    match = re.search(r"/gui/([^/?#]+)/([^/?#]+)", str(endpoint_url))
    app = match.group(1) if match else ""
    if app == "fossify_messages" or tool_name.startswith("fossify_messages"):
        return _handle_fossify_messages(tool_name, payload)
    if app == "fossify_notes" or tool_name.startswith("fossify_notes"):
        return _handle_fossify_notes(tool_name, payload)
    if app == "fossify_calendar" or tool_name.startswith("fossify_calendar"):
        return _handle_fossify_calendar(tool_name, payload)
    if app == "contacts" or tool_name.startswith("contacts"):
        return _handle_contacts(tool_name, payload)
    if app == "gmail_clone" or tool_name.startswith("gmail_clone"):
        return _handle_gmail_clone(tool_name, payload)
    if app == "my_expenses" or tool_name.startswith("my_expenses"):
        return _handle_my_expenses(tool_name, payload)
    if app == "loop_habit" or tool_name.startswith("loop_habit") or tool_name.startswith("loop_habits"):
        return _handle_loop_habit(tool_name, payload)
    if app == "clock" or tool_name.startswith("clock"):
        return _handle_clock(tool_name, payload)
    if app == "dialer" or tool_name.startswith("dialer"):
        return _handle_dialer(tool_name, payload)
    if app == "mattermost" or tool_name.startswith("mattermost"):
        return _handle_mattermost(tool_name, payload)
    if app == "testmall" or tool_name.startswith("testmall"):
        return _handle_testmall(tool_name, payload)
    return 404, {"error": f"Unsupported local GUI endpoint: {endpoint_url}"}
'''


def _message_text(item: dict) -> str:
    return str(
        item.get("text")
        or item.get("message_text")
        or item.get("content")
        or item.get("body")
        or ""
    )


def _message_time(item: dict) -> str:
    return str(item.get("time") or item.get("timestamp") or item.get("date") or "")


def _message_outgoing(item: dict) -> bool:
    if "is_outgoing" in item:
        return bool(item.get("is_outgoing"))
    if "is_sent" in item:
        return bool(item.get("is_sent"))
    return str(item.get("sender") or item.get("from") or "").lower() == "me"


def _normalize_local_message(item: dict) -> dict:
    sender = item.get("from") or item.get("sender")
    if sender == "contact":
        sender = item.get("contact_number") or "contact"
    return {
        "id": str(item.get("id") or item.get("message_id") or ""),
        "from": sender or ("me" if _message_outgoing(item) else "contact"),
        "text": _message_text(item),
        "time": _message_time(item),
        "is_outgoing": _message_outgoing(item),
        "is_read": bool(item.get("is_read", True)),
    }


def _normalize_local_fossify_messages(raw: Any) -> list[dict]:
    grouped: dict[str, dict] = {}
    for index, item in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        thread_id = str(item.get("thread_id") or item.get("id") or f"thread-{index + 1}")
        thread = grouped.setdefault(
            thread_id,
            {
                "thread_id": thread_id,
                "contact_name": item.get("contact_name") or item.get("name") or "",
                "participants": list(item.get("participants") or []),
                "messages": [],
                "unread_count": int(item.get("unread_count") or 0),
                "pinned": bool(item.get("pinned", False)),
                "archived": bool(item.get("archived", item.get("is_archived", False))),
            },
        )
        if item.get("contact_name") and not thread.get("contact_name"):
            thread["contact_name"] = item.get("contact_name")
        contact_number = item.get("contact_number") or item.get("phone_number")
        if contact_number and contact_number not in thread["participants"]:
            thread["participants"].append(contact_number)

        messages = item.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    thread["messages"].append(_normalize_local_message(msg))
        else:
            thread["messages"].append(_normalize_local_message(item))

        explicit_last = item.get("last_message") or item.get("last_message_preview")
        if explicit_last:
            thread["last_message"] = explicit_last
        explicit_time = item.get("last_message_time") or item.get("last_updated")
        if explicit_time:
            thread["last_message_time"] = explicit_time

    for thread in grouped.values():
        messages = thread.get("messages") or []
        if messages:
            last = messages[-1]
            thread.setdefault("last_message", last.get("text"))
            thread.setdefault("last_message_time", last.get("time"))
        thread["messages"] = messages
    return list(grouped.values())


def _normalize_local_fossify_notes(raw: Any) -> list[dict]:
    notes: list[dict] = []
    for index, item in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        raw_id = item.get("note_id") or item.get("id") or index + 1
        note_id = str(raw_id if str(raw_id).startswith("FNOT-") else f"FNOT-{raw_id}")
        notes.append(
            {
                "note_id": note_id,
                "title": item.get("title") or "",
                "content": item.get("content") or "",
                "color": item.get("color"),
                "tags": item.get("tags") or [],
                "pinned": bool(item.get("pinned", False)),
                "updated_at": item.get("updated_at") or item.get("timestamp") or item.get("last_updated") or "",
            }
        )
    return notes


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _normalize_record_list(raw: Any, id_field: str | None = None, id_prefix: str = "LOCAL") -> list[dict]:
    records: list[dict] = []
    for index, item in enumerate(_as_list(raw)):
        if not isinstance(item, dict):
            continue
        record = dict(item)
        if id_field:
            raw_id = record.get(id_field) or record.get("id")
            if raw_id is None or raw_id == "":
                raw_id = f"{id_prefix}-{index + 1}"
            record[id_field] = str(raw_id)
        records.append(record)
    return records


def _normalize_local_fossify_calendar(raw: Any) -> list[dict]:
    return _normalize_record_list(raw, "event_id", "FCAL")


def _normalize_local_contacts(raw: Any) -> list[dict]:
    return _normalize_record_list(raw, "contact_id", "GCON")


def _normalize_local_gmail_clone(raw: Any) -> list[dict]:
    messages: list[dict] = []
    for index, item in enumerate(_as_list(raw)):
        if not isinstance(item, dict):
            continue
        msg = dict(item)
        raw_id = msg.get("message_id") or msg.get("id") or f"GMSG-{index + 1}"
        msg["message_id"] = str(raw_id)
        msg.setdefault("labels", [])
        msg.setdefault("is_read", True)
        messages.append(msg)
    return messages


def _local_record_key(item: object, fields: tuple[str, ...]) -> str | None:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return None
    for field in fields:
        value = item.get(field)
        if value not in (None, ""):
            return str(value)
    return None


def _dedupe_local_records(records: list, fields: tuple[str, ...], *, prefer_latest: bool = False) -> list:
    order: list[str] = []
    keyed: dict[str, object] = {}
    passthrough: list[object] = []
    for item in records:
        key = _local_record_key(item, fields)
        if key is None:
            passthrough.append(item)
            continue
        if key not in keyed:
            order.append(key)
        elif not prefer_latest:
            continue
        keyed[key] = item
    return [keyed[key] for key in order] + passthrough


def _normalize_local_my_expenses(raw: Any) -> dict[str, Any]:
    data = {"accounts": [], "categories": [], "payees": [], "transactions": []}
    containers = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in ("accounts", "categories", "transactions"):
            for item in _as_list(container.get(key)):
                if isinstance(item, dict):
                    data[key].append(json.loads(json.dumps(item)))
        for payee in _as_list(container.get("payees")):
            if payee is not None:
                data["payees"].append(payee)

    accounts = []
    for index, item in enumerate(_as_list(data.get("accounts"))):
        if not isinstance(item, dict):
            continue
        account = dict(item)
        raw_id = account.get("account_id") or account.get("id") or account.get("name") or f"ACC-{index + 1}"
        account["account_id"] = str(raw_id)
        accounts.append(account)
    data["accounts"] = _dedupe_local_records(accounts, ("account_id", "id", "label", "name"))

    categories = []
    for index, item in enumerate(_as_list(data.get("categories"))):
        if not isinstance(item, dict):
            continue
        category = dict(item)
        raw_id = category.get("category_id") or category.get("id") or category.get("name") or f"CAT-{index + 1}"
        category["category_id"] = str(raw_id)
        categories.append(category)
    data["categories"] = _dedupe_local_records(categories, ("category_id", "id", "label", "name"))

    account_names = {
        str(account.get("account_id") or ""): account.get("name") or account.get("label") or ""
        for account in data["accounts"]
    }
    category_names = {
        str(category.get("category_id") or ""): category.get("name") or category.get("label") or ""
        for category in data["categories"]
    }

    transactions = []
    for index, item in enumerate(_as_list(data.get("transactions"))):
        if not isinstance(item, dict):
            continue
        txn = dict(item)
        raw_id = txn.get("transaction_id") or txn.get("id") or f"TRX-{index + 1}"
        txn["transaction_id"] = str(raw_id)
        account_id = txn.get("account_id") or txn.get("account")
        if account_id is not None:
            txn["account_id"] = str(account_id)
            if str(txn.get("account") or "") in account_names:
                txn["account"] = account_names[str(txn.get("account"))]
            elif not txn.get("account"):
                txn["account"] = account_names.get(str(account_id), "")
        category_id = txn.get("category_id") or txn.get("category")
        if category_id is not None:
            txn["category_id"] = str(category_id)
            if str(txn.get("category") or "") in category_names:
                txn["category"] = category_names[str(txn.get("category"))]
            elif not txn.get("category"):
                txn["category"] = category_names.get(str(category_id), "")
        transactions.append(txn)
    data["transactions"] = _dedupe_local_records(
        transactions, ("transaction_id", "id"), prefer_latest=True
    )
    payees = []
    seen_payees = set()
    for payee in data.get("payees") or []:
        try:
            key = json.dumps(payee, ensure_ascii=False, sort_keys=True)
        except TypeError:
            key = str(payee)
        if key in seen_payees:
            continue
        seen_payees.add(key)
        payees.append(payee)
    data["payees"] = payees
    return data


def _normalize_local_loop_habit(raw: Any) -> list[dict]:
    return _normalize_record_list(raw, "habit_id", "HAB")


def _normalize_local_clock(raw: Any) -> list[dict]:
    return _normalize_record_list(raw, "alarm_id", "ALRM")


def _normalize_local_dialer(raw: Any) -> list[dict]:
    return _normalize_record_list(raw, "call_id", "CALL")


def _normalize_local_mattermost(raw: Any) -> list[dict]:
    return _normalize_record_list(raw, "message_id", "MMSG")


def _normalize_local_testmall(raw: Any) -> list[dict]:
    return _normalize_record_list(raw, "product_id", "PROD")


def _collect_gui_fixture_paths(task: TaskDefinition) -> list[str]:
    paths: list[str] = []
    paths.extend(task.environment.fixtures)
    paths.extend(getattr(task, "gui_fixture_paths", []) or [])
    if not task.task_file:
        return paths
    try:
        raw_task = yaml.safe_load(Path(task.task_file).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return paths
    for section in ("inject", "apps"):
        entries = raw_task.get(section)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for key in ("fixture", "json", "file", "path"):
                value = entry.get(key)
                if isinstance(value, str):
                    paths.append(value)
    return list(dict.fromkeys(paths))


def _build_local_gui_state(task: TaskDefinition) -> dict[str, Any]:
    """Embed GUI fixtures for task-declared semantic GUI tools.

    ``/gui/...`` endpoints are not real mock services in trial containers. The
    generated OH plugin serves the small semantic subset directly from fixture
    state so GUI task tools are reachable after host-side Android injection.
    """
    if not task.task_file:
        return {}
    task_dir = Path(task.task_file).parent
    state: dict[str, Any] = {}
    cached_fixtures = getattr(task, "gui_fixture_data", {}) or {}
    for fixture in _collect_gui_fixture_paths(task):
        if fixture in cached_fixtures:
            raw = cached_fixtures[fixture]
        else:
            path = task_dir / fixture
            if not path.exists():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
        if "fossify_messages_gui" in fixture:
            state["fossify_messages"] = {
                "threads": _normalize_local_fossify_messages(raw),
                "sent_messages": [],
            }
        elif "fossify_notes_gui" in fixture:
            state["fossify_notes"] = {
                "notes": _normalize_local_fossify_notes(raw),
            }
        elif "fossify_calendar_gui" in fixture:
            state["fossify_calendar"] = {
                "events": _normalize_local_fossify_calendar(raw),
            }
        elif "contacts_gui" in fixture:
            state["contacts"] = {
                "contacts": _normalize_local_contacts(raw),
            }
        elif "gmail_clone_gui" in fixture:
            state["gmail_clone"] = {
                "messages": _normalize_local_gmail_clone(raw),
                "sent_messages": [],
                "drafts": [],
            }
        elif "my_expenses_gui" in fixture:
            state["my_expenses"] = _normalize_local_my_expenses(raw)
        elif "loop_habit_gui" in fixture:
            state["loop_habit"] = {
                "habits": _normalize_local_loop_habit(raw),
            }
        elif "clock_gui" in fixture:
            state["clock"] = {
                "alarms": _normalize_local_clock(raw),
            }
        elif "dialer_gui" in fixture:
            state["dialer"] = {
                "call_log": _normalize_local_dialer(raw),
            }
        elif "mattermost_gui" in fixture:
            state["mattermost"] = {
                "messages": _normalize_local_mattermost(raw),
            }
        elif "testmall_gui" in fixture:
            state["testmall"] = {
                "products": _normalize_local_testmall(raw),
            }
    return state


def _safe_attr_name(tool_name: str) -> str:
    """Sanitise a tool name into a valid Python identifier for module-level
    attribute assignment. The name is what OH's plugin loader sees, so we
    keep it close to the original tool name without decorative prefixes."""
    safe = re.sub(r"\W", "_", tool_name) or "tool"
    if safe[0].isdigit():
        safe = "_" + safe
    return safe


def _render_tool_registration(
    *, tool_name: str, description: str, endpoint_url: str, method: str, input_schema: dict
) -> str:
    """Render one module-level assignment that the OH plugin loader will discover."""
    # NOTE: input_schema is rendered with repr(), NOT json.dumps(). json.dumps
    # emits JSON literals (true/false/null) which are NOT valid Python and make
    # the generated module raise NameError on import (e.g. skill mode's
    # ``additionalProperties: True`` → ``true`` → the whole clawanything plugin
    # silently fails to load, so the agent sees zero claw-anything tools). repr()
    # produces a valid Python literal for any JSON-derived dict/list/str/num/bool/None.
    return (
        f"{_safe_attr_name(tool_name)} = _make_tool_class(\n"
        f"    tool_name={json.dumps(tool_name, ensure_ascii=False)},\n"
        f"    tool_description={json.dumps(description, ensure_ascii=False)},\n"
        f"    endpoint_url={json.dumps(endpoint_url)},\n"
        f"    method={json.dumps(method)},\n"
        f"    input_schema={input_schema!r},\n"
        f")\n"
    )


_SKILL_MODE_GET_TOOL_SCHEMA_DESCRIPTION = (
    "Retrieve full definitions (description + JSON Schema) for tools by name. "
    "Some tools in this session are listed by name only — their description and "
    "input schema are intentionally omitted to save context. "
    "Call this before using any tool to get its complete definition."
)

_SKILL_MODE_GET_TOOL_SCHEMA_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "tool_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of tool names to retrieve schemas for.",
        }
    },
    "required": ["tool_names"],
}


def _render_skill_mode_get_tool_schema(task: TaskDefinition) -> str:
    """Render a self-contained get_tool_schema BaseTool with all task schemas baked in.

    The full tool definitions are embedded as a Python dict literal so that the
    generated plugin has no runtime dependency on the parent claw-anything
    process.  OH discovers and instantiates the class like any other plugin
    tool; ``execute()`` handles the lookup locally and writes a dispatch record
    (with ``endpoint_url="local://meta/get_tool_schema"``) so the trace
    reconstructor can account for it.
    """
    all_schemas = {
        spec.name: {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        }
        for spec in task.tools
    }
    desc_src = json.dumps(_SKILL_MODE_GET_TOOL_SCHEMA_DESCRIPTION, ensure_ascii=False)
    api_schema_src = json.dumps(
        {
            "name": "get_tool_schema",
            "description": _SKILL_MODE_GET_TOOL_SCHEMA_DESCRIPTION,
            "input_schema": _SKILL_MODE_GET_TOOL_SCHEMA_INPUT_SCHEMA,
        },
        ensure_ascii=False,
    )
    # repr(), not json.dumps(): this is embedded as a Python dict *literal* in
    # the generated source. A task schema containing a JSON bool/null
    # (additionalProperties: false, "default": true, …) would otherwise emit
    # true/false/null and break the module import. See _render_tool_registration.
    schemas_src = repr(all_schemas)
    return (
        f"\n_SKILL_MODE_SCHEMAS = {schemas_src}\n"
        "\n\n"
        "class _GetToolSchemaTool(BaseTool):\n"
        f'    name = "get_tool_schema"\n'
        f"    description = {desc_src}\n"
        "    input_model = _AnyArgs\n"
        "\n"
        "    def to_api_schema(self) -> dict:\n"
        f"        return {api_schema_src}\n"
        "\n"
        "    def is_read_only(self, arguments: BaseModel) -> bool:\n"
        "        return True\n"
        "\n"
        "    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:\n"
        '        t0 = time.monotonic()\n'
        '        names = arguments.model_dump().get("tool_names", [])\n'
        '        results = {}\n'
        '        for n in names:\n'
        '            if n in _SKILL_MODE_SCHEMAS:\n'
        '                results[n] = _SKILL_MODE_SCHEMAS[n]\n'
        '            else:\n'
        '                results[n] = {"error": "Unknown tool: " + n}\n'
        '        latency_ms = (time.monotonic() - t0) * 1000.0\n'
        '        _emit_dispatch({\n'
        '            "tool_name": "get_tool_schema",\n'
        '            "endpoint_url": "local://meta/get_tool_schema",\n'
        '            "request_body": {"tool_names": names},\n'
        '            "response_status": 200,\n'
        '            "response_body": results,\n'
        '            "latency_ms": latency_ms,\n'
        '            "timestamp": time.time(),\n'
        '        })\n'
        '        return ToolResult(output=json.dumps(results, ensure_ascii=False), is_error=False)\n'
        "\n"
        "\n"
        "get_tool_schema = _GetToolSchemaTool\n"
    )


def _render_generated_tools(task: TaskDefinition, skill_mode: bool = False) -> str:
    """Render the full tools/clawanything_tools.py source for a task."""
    endpoints = task.get_endpoint_map()
    local_gui_state = _build_local_gui_state(task)
    lines = [
        _GENERATED_HEADER,
        _LOCAL_GUI_HELPERS,
        "\n_LOCAL_GUI_STATE = ",
        repr(local_gui_state),
        "\n\n",
    ]
    if skill_mode:
        lines.append(_render_skill_mode_get_tool_schema(task))
    for spec in task.tools:
        ep = endpoints.get(spec.name)
        if ep is None:
            # No HTTP endpoint declared — skip silently. Such tools (rare)
            # would need a different backend; the LLM will see them missing.
            continue
        if skill_mode:
            # Expose only the tool name; the real properties stay hidden behind
            # get_tool_schema. The declared schema must still be a *valid* empty
            # object schema — an empty {} serializes to inputSchema.properties=null,
            # which Bedrock-backed Claude rejects ("$.properties: null found,
            # object expected"), failing the whole request before any turn.
            # additionalProperties keeps the call permissive once the agent has
            # learned the real args via get_tool_schema.
            lines.append(
                _render_tool_registration(
                    tool_name=spec.name,
                    description="",
                    endpoint_url=ep.url,
                    method=ep.method,
                    input_schema={"type": "object", "properties": {}, "additionalProperties": True},
                )
            )
        else:
            lines.append(
                _render_tool_registration(
                    tool_name=spec.name,
                    description=spec.description,
                    endpoint_url=ep.url,
                    method=ep.method,
                    input_schema=spec.input_schema,
                )
            )
    return "".join(lines)


def generate_plugin_files(
    task: TaskDefinition,
    plugin_dir: Path,
    *,
    settings_root: Path,
    extra_denied_tools: list[str] | None = None,
    settings_path: Path | None = None,
    skill_mode: bool = False,
    print_mode_extra_fields: list[str] | None = None,
) -> None:
    """Materialise plugin.json, tools/clawanything_tools.py and settings.json on disk.

    Layout produced::

        <settings_root>/
            settings.json
            plugins/clawanything/
                plugin.json
                tools/__init__.py
                tools/clawanything_tools.py

    ``settings_path`` lets the caller pass a pre-existing OH
    settings.json (with model/provider/etc.) as the baseline; clawanything-
    required fields are merged on top — see ``_build_settings`` for the
    merge rules. If omitted, a minimal clawanything-only settings.json is
    written, which is usually NOT enough for OH to actually run.

    ``extra_denied_tools`` is the agent-specific builtin tool list to add to
    ``permission.denied_tools`` (typically passed when the caller wants to
    suppress OH's native tools so the model only sees the per-task clawanything
    tools). Each OH-family agent owns its own list (see
    ``OpenHarnessAgent.builtin_tools_for_deny`` and the OH-Ext override) and
    passes the appropriate union.

    ``skill_mode`` switches the generated plugin to progressive-revelation mode:
    task tools expose only their name (empty description, empty input_schema)
    and an extra ``get_tool_schema`` BaseTool is injected with all full schemas
    baked in.  The LLM is guided (via the get_tool_schema description) to call
    it before using any tool.  Works for both vanilla OH and OH-Ext.
    """
    plugin_dir = Path(plugin_dir)
    settings_root = Path(settings_root)
    tools_dir = plugin_dir / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    settings_root.mkdir(parents=True, exist_ok=True)

    base_settings = _load_settings(settings_path)
    merged = _build_settings(
        extra_denied_tools=extra_denied_tools,
        base_settings=base_settings,
        apps=task.apps or None,
        execution_date=task.execution_date or None,
        print_mode_extra_fields=print_mode_extra_fields,
    )

    (plugin_dir / "plugin.json").write_text(
        json.dumps(_PLUGIN_MANIFEST, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (tools_dir / "__init__.py").write_text("", encoding="utf-8")
    (tools_dir / "clawanything_tools.py").write_text(_render_generated_tools(task, skill_mode=skill_mode), encoding="utf-8")
    (settings_root / "settings.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
