"""Supermemory memory plugin (MemoryProvider, v5 namespace API): profile recall, semantic search, memory tools,
per-turn capture into one document per conversation."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.memory_provider import MemoryProvider, RecallStatus
from agent.secret_scope import get_secret
from tools.registry import tool_error

try:
    from agent.memory_provider import spawn_context_thread
except ImportError:  # older Hermes: plain threads
    def spawn_context_thread(target, *, name, daemon=True, args=(), kwargs=None):
        return threading.Thread(target=target, args=args, kwargs=kwargs, name=name, daemon=daemon)

logger = logging.getLogger(__name__)

_DEFAULT_CONTAINER_TAG = "hermes"
_MAX_TAG_LEN = 100
_VALID_SEARCH_MODES = ("hybrid", "memories", "chunks")
_VALID_DREAMING = ("dynamic", "instant")
_VALID_SCOPES = ("shared", "user", "chat")
_SDK_REQUIREMENT = "supermemory>=5.0.0,<6"
_DEFAULT_BASE_URL = "https://api.supermemory.ai"
_API_KEY_URL = "https://console.supermemory.ai/keys?create=true"
# Strips injected <supermemory-context> / <supermemory-containers> blocks before capture.
_INJECTED_BLOCK_RE = re.compile(r"<supermemory-(context|containers)>[\s\S]*?</supermemory-\1>\s*", re.DOTALL)
_DATA_URI_RE = re.compile(r"data:[^;,\s]+;base64,[A-Za-z0-9+/=]+")  # pasted inline images are useless as memory text
_READ_WINDOW_CHARS = 8000
_MAX_QUERY_CHARS = 2000
_MAX_CACHED_PROFILES = 64
_MAX_PENDING_DOCS = 50
_MAX_PENDING_BYTES = 256 * 1024
_SUMMARY_END = "--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---"
_MERGED_PRIOR_HEADER = "[PRIOR CONTEXT — for reference only; not a new message]"
_MERGED_SUMMARY_DELIMITER = "[END OF PRIOR CONTEXT — COMPACTION SUMMARY BELOW]"
_LEGACY_TEXT: Dict[tuple, str] = {}
_STORED_TURN_RE = re.compile(r"\[role: user\]\n(.*?)\n\[user:end\](?:\n\[role: assistant\]\n(.*?)\n\[assistant:end\])?", re.DOTALL)
_DEFAULT_ENTITY_CONTEXT = (
    "User-assistant conversation. Format: [role: user]...[user:end] and [role: assistant]...[assistant:end].\n\n"
    "Only extract things useful in future conversations. Most messages are not worth remembering.\n\n"
    "Remember lasting personal facts, preferences, routines, tools, ongoing projects, working context, "
    "and explicit requests to remember something.\n\n"
    "Do not remember temporary intents, one-time tasks, assistant actions, implementation details, or in-progress status.\n\n"
    "When in doubt, store less."
)
_KEBAB_ALIASES = {"supermemory_store": "supermemory-save", "supermemory_search": "supermemory-search",
                  "supermemory_forget": "supermemory-forget", "supermemory_profile": "supermemory-profile",
                  "supermemory_read_document": "supermemory-read-document"}
_ALIAS_TO_TOOL = {kebab: snake for snake, kebab in _KEBAB_ALIASES.items()}
_BOOL_WORDS = {**dict.fromkeys(("true", "1", "yes", "y", "on"), True), **dict.fromkeys(("false", "0", "no", "n", "off"), False)}


def _quietly(fn: Callable[[], Any], fail_msg: str = "", *args: Any, level: int = logging.DEBUG, default: Any = None) -> Any:
    """Run ``fn()``; on any exception log ``fail_msg`` (if given) with traceback and return ``default``."""
    try:
        return fn()
    except Exception:
        if fail_msg:
            logger.log(level, fail_msg, *args, exc_info=True)
        return default


# The API rejects ids over 100 chars; a hash of the full id keeps long ones that share a prefix distinct.
def _cap_tag(tag: str) -> str:
    return tag if len(tag) <= _MAX_TAG_LEN else f"{tag[:_MAX_TAG_LEN - 9]}_{hashlib.sha256(tag.encode()).hexdigest()[:8]}"


# Keeps exactly the API's containerTag charset, so a tag the API accepts reaches it unchanged.
def _sanitize_tag(raw: str) -> str:
    return _cap_tag("_".join(part for part in re.split(r"[^a-zA-Z0-9_:-]+", raw or "") if part) or _DEFAULT_CONTAINER_TAG)


# The 1.0.1 sanitizer, kept only to name the container that version wrote to.
def _legacy_tag(raw: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9_]", "_", raw or "")).strip("_") or _DEFAULT_CONTAINER_TAG


def _resolve_base_url(config_value: Any = "") -> str:
    """config > SUPERMEMORY_BASE_URL (profile-scoped) > default (self-hosted support)."""
    raw = str(config_value or "").strip() or (get_secret("SUPERMEMORY_BASE_URL", "") or "").strip()
    return (raw or _DEFAULT_BASE_URL).rstrip("/") or _DEFAULT_BASE_URL


def _clamp_entity_context(text: str) -> str:
    return text.strip()[:1500] if text else _DEFAULT_ENTITY_CONTEXT


def _as_bool(value: Any, default: bool) -> bool:
    """bool passthrough; common true/false words parsed; anything else (incl. ints) -> default."""
    return value if isinstance(value, bool) else _BOOL_WORDS.get(value.strip().lower(), default) if isinstance(value, str) else default


def _clamp_number(value: Any, default, lo, hi, cast):
    """Cast ``value`` and clamp it to [lo, hi]; fall back to ``default`` on any conversion error."""
    return _quietly(lambda: max(lo, min(hi, cast(value))), default=default)


# 1.x called chunk search "documents".
def _search_mode(value: Any) -> str:
    mode = "chunks" if (mode := str(value).strip().lower()) == "documents" else mode
    return mode if mode in _VALID_SEARCH_MODES else "hybrid"


# Gateway ids can hold any character, so an id the charset would alter gets a hash of the original to stay distinct.
def _scoped_tag(base: str, scope: str, platform: str, user_id: str, chat_id: str) -> str:
    kind, raw = {"user": ("user", user_id), "chat": ("chat", chat_id)}.get(scope, ("", ""))
    raw = str(raw or "").strip()
    if not kind or not raw:
        return base
    part = re.sub(r"[^a-zA-Z0-9_:-]", "_", raw)
    part = part if part == raw else f"{part}_{hashlib.sha256(raw.encode()).hexdigest()[:8]}"
    return _cap_tag(f"{base}:{_sanitize_tag(platform or 'gateway')}:{kind}:{part}")


# config key -> (default, normalizer). Tags stay raw here and are sanitized once applied, after {identity} expansion.
_CONFIG_SPEC: Dict[str, tuple] = {
    "container_tag": (_DEFAULT_CONTAINER_TAG, lambda v: str(v).strip() or _DEFAULT_CONTAINER_TAG),
    "memory_scope": ("shared", lambda v: v if (v := str(v or "").strip().lower()) in _VALID_SCOPES else "shared"),
    "auto_recall": (True, lambda v: _as_bool(v, True)),
    "auto_capture": (True, lambda v: _as_bool(v, True)),
    "max_recall_results": (10, lambda v: _clamp_number(v, 10, 1, 20, int)),
    "threshold": (0.5, lambda v: _clamp_number(v, 0.5, 0.0, 1.0, float)),
    "rerank": ("none", lambda v: v if (v := str(v or "").strip().lower()) in ("none", "order") else "none"),
    "rewrite_query": (False, lambda v: _as_bool(v, False)),
    "include_documents": (False, lambda v: _as_bool(v, False)),
    "include_related": (False, lambda v: _as_bool(v, False)),
    "include_forgotten": (False, lambda v: _as_bool(v, False)),
    "allow_agent_search_overrides": (True, lambda v: _as_bool(v, True)),
    "profile_frequency": (50, lambda v: _clamp_number(v, 50, 1, 500, int)),
    "capture_mode": ("all", lambda v: "everything" if v == "everything" else "all"),
    "search_mode": ("hybrid", _search_mode),
    "entity_context": (_DEFAULT_ENTITY_CONTEXT, lambda v: _clamp_entity_context(str(v))),
    "dreaming": ("dynamic", lambda v: v if (v := str(v or "").strip().lower()) in _VALID_DREAMING else "dynamic"),
    "api_timeout": (5.0, lambda v: _clamp_number(v, 5.0, 0.5, 15.0, float)),
    "base_url": ("", lambda v: str(v or "").strip()),
    "enable_custom_container_tags": (False, lambda v: _as_bool(v, False)),
    "custom_containers": ([], lambda v: [str(t) for t in v if t] if isinstance(v, list) else []),
    "custom_container_instructions": ("", lambda v: str(v).strip()),
}


def _read_json_dict(path: Path) -> dict:
    raw = _quietly(lambda: json.loads(path.read_text(encoding="utf-8-sig")), "Failed to parse %s", path) if path.exists() else None
    return raw if isinstance(raw, dict) else {}


def _load_supermemory_config(hermes_home: Optional[str] = None) -> dict:
    """Defaults overlaid with $hermes_home/supermemory.json (None = defaults only), every key normalized."""
    config = {k: (list(d) if isinstance(d, list) else d) for k, (d, _) in _CONFIG_SPEC.items()}
    if hermes_home is not None:
        config.update({k: v for k, v in _read_json_dict(Path(hermes_home) / "supermemory.json").items() if v is not None})
    for key, (_, normalize) in _CONFIG_SPEC.items():
        config[key] = normalize(config[key])
    return config


def _save_supermemory_config(values: dict, hermes_home: str) -> None:
    from utils import atomic_json_write
    config_path = Path(hermes_home) / "supermemory.json"
    atomic_json_write(config_path, {**_read_json_dict(config_path), **values}, mode=0o600, sort_keys=True)


def _detect_category(text: str) -> str:
    lowered = text.lower()  # first matching pattern wins
    return next((cat for cat, pat in (("preference", r"prefer|like|love|hate|want"), ("decision", r"decided|will use|going with"),
                                      ("fact", r"\bis\b|\bare\b|\bhas\b|\bhave\b")) if re.search(pat, lowered)), "other")


def _format_relative_time(iso_timestamp: str) -> str:
    """'just now' / '5m ago' / '3h ago' / '2d ago' / '%d %b[ %Y]'; '' when unparseable."""
    def _fmt():
        dt, now = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00")), datetime.now(timezone.utc)
        seconds = (now - dt).total_seconds()
        for limit, unit, label in ((1800, 0, "just now"), (3600, 60, "m ago"), (86400, 3600, "h ago"), (604800, 86400, "d ago")):
            if seconds < limit:
                return f"{int(seconds / unit)}{label}" if unit else label
        return dt.strftime("%d %b" if dt.year == now.year else "%d %b %Y")
    return _quietly(_fmt, default="")


def _similarity_pct(value: Any) -> Optional[int]:
    """0..1 similarity -> whole percent; None when absent or unparseable."""
    return _quietly(lambda: None if value is None else round(float(value) * 100))


def _format_prefetch_context(profile: str, search_results: list, max_results: int) -> tuple:
    """The markdown profile, then up to ``max_results`` search hits not already in it, and how many hits made it in."""
    seen: set = set()

    def _unique(items, key=lambda x: x):  # set.add() returns None, so `not seen.add(k)` records k and keeps the item
        return [i for i in items or [] if (k := key(i)) and k not in seen and k not in profile and not seen.add(k)][:max_results]
    profile = (profile or "").strip()
    sections = [profile] if any(line.startswith("- ") for line in profile.splitlines()) else []
    lines = []
    for item in _unique(search_results, key=lambda i: i.get("memory", "")):
        rel = _format_relative_time(item.get("updated_at") or item.get("updatedAt") or "")
        pct = _similarity_pct(item.get("similarity"))
        lines.append(f"- {' '.join(([f'[{rel}]'] if rel else []) + ([f'[{pct}%]'] if pct is not None else []))} {item['memory']}".strip())
    sections += ["## Relevant Memories\n" + "\n".join(lines)] if lines else []
    return ("<supermemory-context>\n" + "\n\n".join(sections) + "\n</supermemory-context>" if sections else "", len(lines))


def _clean_text_for_capture(text: str) -> str:
    return _DATA_URI_RE.sub("[image]", _INJECTED_BLOCK_RE.sub("", text or "")).strip()


def _as_dict(value: Any) -> dict:
    """An SDK response model as the API's JSON object."""
    return value.to_dict() if hasattr(value, "to_dict") else dict(value or {})


def _result_fields(item: dict) -> dict:
    """A v5 search result (memory or document chunk) as the plain dict the formatters use."""
    metadata = item.get("metadata") or {}
    document = (item.get("included") or {}).get("document") or {}
    return {"id": item.get("id") or "", "memory": item.get("memory") or item.get("chunk") or "", "related": (item.get("included") or {}).get("related"),
            "similarity": item.get("similarity"), "updated_at": item.get("updatedAt") or item.get("updated_at"),
            "document_id": metadata.get("docId") or document.get("id") or item.get("documentId") or "",
            "metadata": metadata}


def _load_sdk():
    """The ``supermemory`` SDK (5.x speaks the v5 namespace API); ImportError when missing or older."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        found = version("supermemory")
    except PackageNotFoundError:
        found = None
    if not found or not found.split(".", 1)[0].isdigit() or int(found.split(".", 1)[0]) < 5:
        raise ImportError(f"the supermemory SDK 5.x is required (found {found or 'none'}); "
                          f"install it with: pip install '{_SDK_REQUIREMENT}'")
    import supermemory
    return supermemory


class _SupermemoryClient:
    """Supermemory v5: the container tag is the namespace every call reads and writes."""

    def __init__(self, api_key: str, timeout: float, container_tag: str,
                 search_mode: str = "hybrid", base_url: str = ""):
        self._sm = _load_sdk()
        self._api_key, self._container_tag, self._timeout = api_key, container_tag, timeout
        self._search_mode = search_mode if search_mode in _VALID_SEARCH_MODES else "hybrid"
        self._base_url = _resolve_base_url(base_url)
        self._client = self._sm.Supermemory(api_key=api_key, base_url=self._base_url, timeout=timeout, max_retries=0,
                                            default_headers={"x-sm-source": "hermes"})

    def _merge_metadata(self, metadata: Optional[dict]) -> dict:
        # sm_source routes Hermes writes into the "Hermes" Space in the Supermemory app so the user
        # can filter / bulk-manage them per source agent (a routing key for the user, not telemetry).
        merged = {"sm_source": "hermes", **(metadata or {})}
        if (legacy_source := merged.pop("source", None)) and "type" not in merged:
            merged["type"] = str(legacy_source)
        return merged

    def add_memory(self, content: str, metadata: Optional[dict] = None, *, entity_context: str = "",
                   container_tag: Optional[str] = None, dreaming: str = "") -> dict:
        kwargs: dict[str, Any] = {"content": content.strip(), "date": datetime.now().astimezone().isoformat(timespec="seconds"),
                                  **({"metadata": self._merge_metadata(metadata)} if metadata else {}),
                                  **({"supporting_context": _clamp_entity_context(entity_context)} if entity_context else {}),
                                  **({"dreaming": dreaming} if dreaming else {})}
        return {"id": _as_dict(self._client.add(container_tag or self._container_tag, **kwargs)).get("id") or ""}

    # PATCH replaces the whole document, so a resend, rewind or edit never appends it to itself; the first write creates it.
    def write_document(self, doc_id: str, content: str, metadata: dict, *, entity_context: str = "", dreaming: str = "") -> None:
        kwargs: dict[str, Any] = {"content": content.strip(), "date": datetime.now().astimezone().isoformat(timespec="seconds"),
                                  "metadata": self._merge_metadata(metadata),
                                  **({"supporting_context": _clamp_entity_context(entity_context)} if entity_context else {}),
                                  **({"dreaming": dreaming} if dreaming else {})}
        try:
            self._client.documents.update(self._container_tag, urllib.parse.quote(doc_id, safe=""), **kwargs)
        except self._sm.NotFoundError:
            self._client.add(self._container_tag, id=doc_id, **kwargs)

    # POST to an existing id appends, so each speaker's document grows one turn at a time without a read.
    def append_document(self, doc_id: str, content: str, metadata: dict, *, container_tag: str, entity_context: str = "",
                        dreaming: str = "") -> None:
        self._client.add(container_tag, id=doc_id, content=content.strip(), date=datetime.now().astimezone().isoformat(timespec="seconds"),
                         metadata=self._merge_metadata(metadata),
                         **({"supporting_context": _clamp_entity_context(entity_context)} if entity_context else {}),
                         **({"dreaming": dreaming} if dreaming else {}))

    # 1.0.x wrote a session as appended deltas in one document per 4-hour window, stamped with its session_id and no docId.
    def legacy_capture_text(self, session_id: str) -> str:
        listed = _as_dict(self._client.list(self._container_tag, "documents", limit=100,
                                            filter={"field": "session_id", "operator": "eq", "value": session_id}))
        ids = [d["id"] for d in listed.get("documents") or []
               if (d.get("metadata") or {}).get("type") == "conversation" and "docId" not in (d.get("metadata") or {})]
        return "\n\n".join(_as_dict(self._client.documents.get(self._container_tag, urllib.parse.quote(i, safe=""))).get("content") or ""
                           for i in ids)

    def search_memories(self, query: str, *, limit: int = 5, container_tag: Optional[str] = None,
                        search_mode: Optional[str] = None, threshold: Optional[float] = None, rerank: str = "",
                        rewrite_query: bool = False, include: Optional[dict] = None, timeout: Optional[float] = None) -> list[dict]:
        mode = search_mode or self._search_mode
        response = self._client.search(container_tag or self._container_tag, query=query[:_MAX_QUERY_CHARS], limit=limit,
                                       **({"search_mode": mode} if mode in _VALID_SEARCH_MODES else {}),
                                       **({"threshold": threshold} if threshold is not None else {}),
                                       **({"rerank": rerank} if rerank and rerank != "none" else {}),
                                       **({"rewrite_query": True} if rewrite_query else {}),
                                       **({"include": include} if include else {}),
                                       **({"timeout": timeout} if timeout else {}))
        return [_result_fields(item) for item in _as_dict(response).get("results") or []]

    def get_profile(self, *, container_tag: Optional[str] = None, timeout: Optional[float] = None) -> str:
        """The profile as markdown (``Accept: text/markdown``), read from the raw response."""
        try:
            raw = self._client.with_raw_response.profile(container_tag or self._container_tag,
                                                         extra_headers={"Accept": "text/markdown"},
                                                         **({"timeout": timeout} if timeout else {}))
        except self._sm.APIStatusError as error:
            # SDK 5.0 parses every 2xx body as JSON; a markdown body comes back on the error with its response.
            if 200 <= error.status_code < 300 and getattr(error, "response", None) is not None:
                return error.response.text
            raise
        return raw.http_response.text

    def read_document(self, doc_id: str, offset: int = 0, *, container_tag: Optional[str] = None,
                      limit: Optional[int] = _READ_WINDOW_CHARS) -> dict:
        doc = _as_dict(self._client.documents.get(container_tag or self._container_tag, urllib.parse.quote(doc_id, safe="")))
        content = doc.get("content") or ""
        window = content[offset:offset + limit if limit else None]
        end = offset + len(window)
        return {"id": doc_id, "date": doc.get("date") or (doc.get("metadata") or {}).get("date"), "content": window,
                "next_offset": end if end < len(content) else None}

    # V5 answers 200 with per-id errors, so a missing or ineligible id must be read from the body; returns the error or "".
    def forget_memory(self, memory_id: str, *, container_tag: Optional[str] = None) -> str:
        result = _as_dict(self._client.memories.forget(container_tag or self._container_tag, ids=[memory_id]))
        errors = [e.get("error") or "not forgotten" for e in result.get("errors") or [] if e.get("id") == memory_id]
        if errors or memory_id not in {m.get("id") for m in result.get("matches") or []}:
            return errors[0] if errors else "memory not found"
        return ""

    def forget_by_query(self, query: str, *, container_tag: Optional[str] = None) -> dict:
        results = self.search_memories(query, limit=5, container_tag=container_tag, search_mode="memories")
        memory_id = results[0].get("id", "") if results else ""
        if not memory_id:
            return {"success": False, "message": "Best matching memory has no id." if results else "No matching memory found to forget."}
        if error := self.forget_memory(memory_id, container_tag=container_tag):
            return {"success": False, "message": f"Could not forget {memory_id}: {error}", "id": memory_id}
        return {"success": True, "message": f'Forgot: "{(results[0].get("memory") or "")[:100]}"', "id": memory_id}


def _format_turn(user: str, assistant: str) -> str:
    """Render one turn in the [role: x]...[x:end] layout the entity context describes."""
    return "\n".join(f"[role: {role}]\n{text}\n[{role}:end]" for role, text in (("user", user), ("assistant", assistant)) if text)


def _capture_doc_id(session_id: str) -> str:
    return _cap_tag(f"hermes_{_sanitize_tag(session_id)}")


def _message_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(str(p.get("text") or "") if isinstance(p, dict) else str(p) for p in content
                         if isinstance(p, str) or (isinstance(p, dict) and p.get("type") in {"text", "input_text", "output_text"}))
    return content if isinstance(content, str) else ""


# Hermes writes a compaction summary on its own or merged into a kept message; only the kept message's own text is the user's.
def _without_compaction_summary(text: str) -> str:
    if _SUMMARY_END not in text:
        return text
    if text.startswith(_MERGED_PRIOR_HEADER) and _MERGED_SUMMARY_DELIMITER in text:
        return text[len(_MERGED_PRIOR_HEADER):text.index(_MERGED_SUMMARY_DELIMITER)]
    return text.split(_SUMMARY_END, 1)[1]


# Each user message with the last assistant text after it.
def _turns_from_messages(messages: Optional[list]) -> List[Dict[str, str]]:
    turns: List[Dict[str, str]] = []
    for message in messages or []:
        role = message.get("role") if isinstance(message, dict) else None
        text = _message_text(message.get("content")) if role in {"user", "assistant"} else ""
        text = _clean_text_for_capture(_without_compaction_summary(text) if text and message.get("_compressed_summary") else text)
        if role == "user" and text:
            turns.append({"user": text, "assistant": ""})
        elif role == "assistant" and text and turns:
            turns[-1]["assistant"] = text
    return turns


# Compaction keeps the first turns and the latest ones around its summary; both are already in the archive, so count them once.
def _join_turns(archive: List[Dict[str, str]], turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    head = next((i for i, (a, t) in enumerate(zip(archive, turns)) if a != t), min(len(archive), len(turns)))
    rest, room = turns[head:], len(archive) - head
    tail = next((k for k in range(min(room, len(rest)), 0, -1) if archive[-k:] == rest[:k]), 0)
    return archive + rest[tail:]


def _build_client(api_key: str, config: dict, container_tag: str) -> _SupermemoryClient:
    return _SupermemoryClient(api_key=api_key, timeout=config["api_timeout"], container_tag=container_tag,
                              search_mode=config["search_mode"], base_url=_resolve_base_url(config["base_url"]))


def _expand_container_tag(config_tag: str, identity: str) -> str:
    """SUPERMEMORY_CONTAINER_TAG (profile-scoped, never the default profile's environ) > config > default, {identity} expanded."""
    raw_tag = (get_secret("SUPERMEMORY_CONTAINER_TAG", "") or "").strip() or config_tag
    return raw_tag.replace("{identity}", identity)


def _probe_supermemory_connection(api_key: str, hermes_home: str, *, identity: str = "default") -> dict:
    config = _load_supermemory_config(hermes_home)
    status = {"ok": False, "error": "", "profile_facts": 0, "container_tag": _sanitize_tag(_expand_container_tag(config["container_tag"], identity)),
              "auto_recall": bool(config["auto_recall"]), "auto_capture": bool(config["auto_capture"])}
    if not (api_key or "").strip():
        return {**status, "error": "SUPERMEMORY_API_KEY not set"}
    try:
        _load_sdk()
    except ImportError as exc:
        return {**status, "error": str(exc)}
    try:
        profile = _build_client(api_key.strip(), config, status["container_tag"]).get_profile()
    except Exception as exc:
        return {**status, "error": str(exc).strip()[:160] or "connection failed"}
    facts = sum(1 for line in (profile or "").splitlines() if line.startswith("- "))
    return {**status, "ok": True, "profile_facts": facts}


def _format_connection_summary(status: dict) -> str:
    container = status.get("container_tag") or _DEFAULT_CONTAINER_TAG
    flags = f"auto_recall {'on' if status.get('auto_recall') else 'off'} · auto_capture {'on' if status.get('auto_capture') else 'off'}"
    if status.get("ok"):
        facts = int(status.get("profile_facts") or 0)
        return f"✓ Connected · container: {container} · {facts} profile {'fact' if facts == 1 else 'facts'} · {flags}"
    return f"✗ {status.get('error') or 'connection failed'} · container: {container} · {flags}"


_BASE_SCHEMAS = [
    {"name": name, "description": description,
     "parameters": {"type": "object", "properties": {p: {"type": t, "description": d} for p, t, d in props}, **({"required": req} if req else {})}}
    for name, description, props, req in (
        ("supermemory_store", "Store an explicit memory for future recall.",
         (("content", "string", "The memory content to store."), ("metadata", "object", "Optional metadata attached to the memory.")), ["content"]),
        ("supermemory_search", "Search long-term memory by semantic similarity.",
         (("query", "string", "What to search for."), ("limit", "integer", "Maximum results to return, 1 to 20.")), ["query"]),
        ("supermemory_forget", "Forget a memory by exact id or by best-match query.",
         (("id", "string", "Exact memory id to delete."), ("query", "string", "Query used to find the memory to forget.")), None),
        ("supermemory_profile", "Retrieve the user's profile: persistent facts and recent memory context.",
         (("query", "string", "Optional query; matching memories are returned alongside the profile."),), None),
        ("supermemory_read_document", "Read a stored document, such as a past conversation, by its document_id from search "
         "results. Long documents are paged with offset.",
         (("document_id", "string", "Document id from supermemory_search."),
          ("offset", "integer", "Character offset to continue from (next_offset of the previous read).")), ["document_id"]),
    )
]


class _TagError(Exception):
    """Tool call named a container_tag outside the whitelist."""


def _tagged(resp: dict, tag: Optional[str]) -> dict:
    return {**resp, "container_tag": tag} if tag else resp


class SupermemoryMemoryProvider(MemoryProvider):
    def __init__(self):
        self._api_key = self._session_id = self._hermes_home = ""
        self._client: Optional[_SupermemoryClient] = None
        self._container_tag, self._write_enabled, self._active = _DEFAULT_CONTAINER_TAG, True, False
        self._base_tag, self._platform, self._user_id, self._chat_id, self._speaker = _DEFAULT_CONTAINER_TAG, "", "", "", ""
        self._per_speaker, self._unsent_appends, self._recalled_ids = False, [], set()
        self._profiles: Dict[str, str] = {}
        self._profile_jobs: Dict[str, threading.Thread] = {}
        self._profile_shown: set = set()
        self._last_recall: Optional[RecallStatus] = None
        self._write_thread = None
        self._turn_count = 0
        self._reset_capture("")
        self._pending: Dict[str, dict] = {}
        self._capture_lock = threading.Lock()  # sync_turn (worker) vs on_session_end/switch/shutdown (caller thread)
        self._apply_config(_load_supermemory_config())
        self._base_url, self._allowed_containers = _DEFAULT_BASE_URL, []  # env var is only consulted in initialize()

    def _apply_config(self, config: dict) -> None:
        for key in ("auto_recall", "auto_capture", "max_recall_results", "threshold", "rerank", "rewrite_query", "include_documents",
                    "include_related", "include_forgotten", "allow_agent_search_overrides", "profile_frequency", "capture_mode",
                    "search_mode", "entity_context", "dreaming", "api_timeout", "custom_container_instructions"):
            setattr(self, f"_{key}", config[key])
        self._custom_containers = [_sanitize_tag(t) for t in config["custom_containers"]]
        self._base_url, self._enable_custom_containers = _resolve_base_url(config["base_url"]), config["enable_custom_container_tags"]
        self._allowed_containers: List[str] = [self._container_tag] + list(self._custom_containers)

    @property
    def name(self) -> str:
        return "supermemory"

    def is_available(self) -> bool:
        # Key presence only, no SDK import check: the SDK is lazy-installed in initialize(), so gating on
        # importability here is a chicken-and-egg trap on sealed venvs. Mirrors honcho/mem0.
        return bool(get_secret("SUPERMEMORY_API_KEY", ""))

    # A new setup scopes gateway memory per person; a config without memory_scope keeps the shared container.
    def get_config_schema(self):
        return [{"key": "api_key", "description": "Supermemory API key", "secret": True, "required": True, "env_var": "SUPERMEMORY_API_KEY", "url": _API_KEY_URL},
                {"key": "memory_scope", "description": "Gateway memory: one container per user, per chat, or shared by everyone",
                 "choices": ["user", "chat", "shared"], "default": "user"}]

    def save_config(self, values, hermes_home):
        sanitized = dict(values or {})
        for key, fix in (("container_tag", _sanitize_tag), ("entity_context", _clamp_entity_context)):
            if key in sanitized:
                sanitized[key] = fix(str(sanitized[key]))
        _save_supermemory_config(sanitized, hermes_home)

    def get_status_config(self, provider_config: dict) -> dict:
        from hermes_constants import get_hermes_home
        return {"summary": _format_connection_summary(_probe_supermemory_connection(get_secret("SUPERMEMORY_API_KEY", "") or "", str(get_hermes_home())))}

    def post_setup(self, hermes_home: str, config: dict) -> None:
        from hermes_cli.config import save_config
        from hermes_cli.memory_setup import _prompt, _write_env_vars
        print(f"\n  Configuring supermemory:\n\n  Get your API key at {_API_KEY_URL}\n")
        existing = get_secret("SUPERMEMORY_API_KEY", "") or ""
        masked = f"...{existing[-4:]}" if len(existing) > 4 else "set"
        val = _prompt(f"Supermemory API key (current: {masked}, blank to keep)" if existing else "Supermemory API key", secret=True)
        memory = config["memory"] = config["memory"] if isinstance(config.get("memory"), dict) else {}
        memory["provider"] = self.name
        save_config(config)
        if val:
            _write_env_vars({"SUPERMEMORY_API_KEY": val}, hermes_home=hermes_home)
        status = _probe_supermemory_connection(val or existing, hermes_home)
        print(f"\n  {_format_connection_summary(status)}\n\n  Memory provider: supermemory\n  Activation saved to config.yaml")
        if val:
            print("  API keys saved to .env")
        print("\n  Start a new session to activate.\n")

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        self._hermes_home = kwargs.get("hermes_home") or str(get_hermes_home())
        self._session_id, self._turn_count, self._pending = session_id, 0, {}
        self._reset_capture(session_id)
        config = _load_supermemory_config(self._hermes_home)
        self._api_key = get_secret("SUPERMEMORY_API_KEY", "") or ""
        raw_tag = _expand_container_tag(config["container_tag"], kwargs.get("agent_identity", "default"))
        self._base_tag = _sanitize_tag(raw_tag)
        self._platform, self._user_id, self._chat_id = (str(kwargs.get(k) or "").strip() for k in ("platform", "user_id", "chat_id"))
        self._speaker, self._unsent_appends, self._recalled_ids = "", [], set()
        self._container_tag = _scoped_tag(self._base_tag, config["memory_scope"], self._platform, self._user_id, self._chat_id)
        self._per_speaker = config["memory_scope"] == "user" and bool(self._user_id) and str(kwargs.get("chat_type") or "dm") != "dm"
        self._apply_config(config)
        self._write_enabled = kwargs.get("agent_context", "") not in {"cron", "flush", "subagent"}
        self._client = _quietly(lambda: _build_client(self._api_key, config, self._container_tag),
                                "Supermemory initialization failed", level=logging.WARNING) if self._api_key else None
        self._active = self._client is not None
        self._profiles, self._profile_jobs, self._profile_shown, self._last_recall = {}, {}, set(), None
        if self._active and self._auto_recall:
            self._refresh_profile(self._container_tag)
        if self._active:
            self._warn_legacy_tags_once([raw_tag] + (config["custom_containers"] if self._enable_custom_containers else []))

    # Data saved by 1.0.1 sits under its rewritten tag; a marker file in HERMES_HOME keeps this to one warning per install.
    def _warn_legacy_tags_once(self, raw_tags: List[str]) -> None:
        pairs = {(_legacy_tag(raw), _sanitize_tag(raw)) for raw in raw_tags}
        renamed = sorted((old, new) for old, new in pairs if old != new and len(old) <= _MAX_TAG_LEN)
        marker = Path(self._hermes_home) / ".supermemory_legacy_tags_warned"
        if not renamed or marker.exists():
            return
        logger.warning("Memories saved by hermes-supermemory 1.0.1 are under the old tag (old -> new): %s. To keep them, set the old name "
                       "in supermemory.json, or have an org admin merge it into the new tag from Namespaces at console.supermemory.ai "
                       "after your first save.", ", ".join(f"{old} -> {new}" for old, new in renamed))
        _quietly(marker.touch, "Failed to write %s", marker)

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn_count = max(turn_number, 0)
        self._speaker = str(kwargs.get("author_id") or "").strip()

    # Read on every inbound gateway message, so a scope change rebuilds cached agents.
    def identity_signature(self) -> Dict[str, Any]:
        from hermes_constants import get_hermes_home
        return {"supermemory.memory_scope": _load_supermemory_config(str(get_hermes_home()))["memory_scope"]}

    # In a group chat under the user scope, each speaker's turns and recall live in that speaker's container.
    def _speaker_tag(self, author_id: str = "") -> str:
        return _scoped_tag(self._base_tag, "user", self._platform, author_id, "") if self._per_speaker and author_id else self._container_tag

    def _ids(self, author_id: str = "") -> Dict[str, str]:
        return {k: v for k, v in (("user_id", author_id or self._user_id), ("chat_id", self._chat_id)) if v}

    def system_prompt_block(self) -> str:
        lines = ["# Supermemory", f"Active. Container: {self._container_tag}.",
                 "Use supermemory-search, supermemory-save, supermemory-forget, supermemory-profile, and supermemory-read-document."]
        if self._enable_custom_containers and self._custom_containers:
            lines += [f"\nMulti-container mode enabled. Available containers: {', '.join(self._allowed_containers)}.",
                      "Pass an optional container_tag to any supermemory tool to target a specific container."]
            lines += [f"\n{self._custom_container_instructions}"] if self._custom_container_instructions else []
        return "\n".join(lines) if self._active else ""

    def _can_write(self) -> bool:
        return bool(self._active and self._write_enabled and self._client)

    # Shown once per context window (again after a compaction) and every profile_frequency turns.
    def _profile_due(self, tag: str, turn: int) -> bool:
        return tag not in self._profile_shown or (turn > 0 and turn % self._profile_frequency == 0)

    def _refresh_profile(self, tag: str) -> None:
        if (job := self._profile_jobs.get(tag)) and job.is_alive():
            return

        def fetch() -> None:
            profile = _quietly(lambda: self._client.get_profile(container_tag=tag, timeout=self._api_timeout), "Supermemory profile refresh failed")
            if profile is not None:
                self._profiles.pop(tag, None)
                self._profiles[tag] = profile
                while len(self._profiles) > _MAX_CACHED_PROFILES:
                    self._profiles.pop(next(iter(self._profiles)))
        self._profile_jobs[tag] = job = spawn_context_thread(fetch, name="supermemory-profile")
        job.start()

    def _search_options(self, rerank: str = "") -> Dict[str, Any]:
        include = {k: True for k in ("documents", "related", "forgotten") if getattr(self, f"_include_{k}")}
        return {"threshold": self._threshold, "rerank": rerank or self._rerank, "rewrite_query": self._rewrite_query, "include": include}

    # Only the search blocks the turn; the profile comes from the cache that queue_prefetch keeps fresh.
    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_recall = None
        if not self._active or not self._auto_recall or not self._client or not query.strip():
            return ""
        tag = self._speaker_tag(self._speaker)
        due = self._profile_due(tag, self._turn_count)
        if due and tag not in self._profiles:
            self._refresh_profile(tag)
        found = _quietly(lambda: self._client.search_memories(query, limit=self._max_recall_results, container_tag=tag, search_mode="memories",
                                                              timeout=self._api_timeout, **self._search_options()),
                         "Supermemory recall failed", default=[])
        hits = [hit for hit in found if hit.get("id") not in self._recalled_ids]
        profile = self._profiles.get(tag, "") if due else ""
        text, count = _quietly(lambda: _format_prefetch_context(profile, hits, self._max_recall_results),
                               "Supermemory prefetch failed", default=("", 0))
        if due and tag in self._profiles:
            self._profile_shown.add(tag)
        if text:
            self._recalled_ids.update(hit["id"] for hit in hits if hit.get("id"))
            self._last_recall = RecallStatus(provider_label="Supermemory", count=count)
        return text

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        tag = self._speaker_tag(self._speaker)
        if self._active and self._auto_recall and self._client and self._profile_due(tag, self._turn_count + 1):
            self._refresh_profile(tag)

    def recall_status(self) -> Optional[RecallStatus]:
        return self._last_recall

    def _reset_capture(self, session_id: str) -> None:
        self._doc_id, self._doc_session = _capture_doc_id(session_id), session_id
        self._archive: List[Dict[str, str]] = []
        self._turns: List[Dict[str, str]] = []
        self._captured_digest: Optional[str] = None
        self._unsent: Optional[dict] = None
        self._legacy_text: Optional[str] = None

    def _write(self, doc_id: str, doc: dict, mode: str) -> bool:
        return _quietly(lambda: self._client.write_document(doc_id, doc["content"], doc["metadata"], entity_context=self._entity_context,
                                                            dreaming=self._dreaming) or True,
                        "Supermemory capture failed (%s, document %s)", mode, doc_id, level=logging.WARNING, default=False)

    # Documents of earlier sessions whose last write failed.
    def _retry_pending(self, mode: str) -> None:
        for doc_id, doc in list(self._pending.items()):
            if self._write(doc_id, doc, mode):
                self._pending.pop(doc_id, None)

    # A down service must not grow the retry buffer without bound; the oldest documents go first.
    def _park_unsent(self) -> None:
        if self._unsent:
            self._pending[self._doc_id] = self._unsent
        while len(self._pending) > _MAX_PENDING_DOCS or sum(len(d["content"]) for d in self._pending.values()) > _MAX_PENDING_BYTES:
            dropped = next(iter(self._pending))
            self._pending.pop(dropped)
            logger.warning("Supermemory: dropped the pending write of %s to keep the retry buffer bounded", dropped)

    # Turns 1.0.x already stored for this session stay in its documents; only the rest go to the 2.0 document.
    def _without_legacy_turns(self, turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if self._container_tag != self._base_tag:
            return turns
        key = (self._base_url, self._container_tag, self._doc_session)
        if self._legacy_text is None and key in _LEGACY_TEXT:
            self._legacy_text = _LEGACY_TEXT[key]
        if self._legacy_text is None:
            self._legacy_text = "" if len(turns) <= 1 else _quietly(lambda: self._client.legacy_capture_text(self._doc_session),
                                                                     "Supermemory 1.x document lookup failed", level=logging.WARNING)
            if self._legacy_text is not None and len(turns) > 1:
                if len(_LEGACY_TEXT) > 256:
                    _LEGACY_TEXT.clear()
                _LEGACY_TEXT[key] = self._legacy_text
        if not self._legacy_text:
            return turns
        return turns[next((i for i, t in enumerate(turns) if _format_turn(t["user"], t["assistant"]) not in self._legacy_text), len(turns)):]

    # A provider built after a compaction only sees what it kept, so it starts from the stored document; False when that read failed.
    def _seed_archive(self, messages: Optional[list]) -> bool:
        if self._archive or self._captured_digest or not any(isinstance(m, dict) and m.get("_compressed_summary") for m in messages or []):
            return True
        try:
            stored = self._client.read_document(self._doc_id, 0, limit=None)["content"]
        except self._client._sm.NotFoundError:
            stored = ""
        except Exception:
            logger.warning("Supermemory: could not read %s; its next write waits so earlier turns are not dropped", self._doc_id, exc_info=True)
            return False
        self._archive = [{"user": u, "assistant": a} for u, a in _STORED_TURN_RE.findall(stored)]
        return True

    def _capture(self, mode: str, messages: Optional[list] = None, *, completed_only: bool = False) -> None:
        with self._capture_lock:
            self._retry_pending(mode)
            if not self._seed_archive(messages):
                return
            turns = _turns_from_messages(messages)
            if completed_only and turns and not turns[-1]["assistant"]:
                turns = turns[:-1]  # the transcript can run ahead of the turn being synced
            self._turns = turns or self._turns
            turns = self._without_legacy_turns(_join_turns(self._archive, self._turns))
            if not turns:
                return
            content = "\n\n".join(_format_turn(t["user"], t["assistant"]) for t in turns)
            digest = hashlib.sha256(content.encode()).hexdigest()
            if digest == self._captured_digest and self._unsent is None:
                return
            doc = {"content": content, "metadata": {"type": "conversation", "session_id": self._doc_session, "docId": self._doc_id, **self._ids()}}
            if self._write(self._doc_id, doc, mode):
                self._captured_digest, self._unsent = digest, None
            else:
                self._unsent = doc

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "",
                  messages: Optional[list] = None, turn_author: Optional[dict] = None) -> None:
        # Host runs this on a worker thread, so the blocking write is fine here.
        if not self._can_write() or not self._auto_capture:
            return
        if session_id and session_id != self._session_id:
            self.on_session_switch(session_id)
        if self._per_speaker:
            author = str((turn_author or {}).get("id") or self._speaker or self._user_id)
            content = _format_turn(_clean_text_for_capture(_message_text(user_content)), _clean_text_for_capture(_message_text(assistant_content)))
            meta = {"type": "conversation", "session_id": self._doc_session, "docId": self._doc_id, **self._ids(author)}
            self._append_turns([(self._speaker_tag(author), meta, content)] if content.startswith("[role: user]") else [])
            return
        turn = {"user": _clean_text_for_capture(_message_text(user_content)),
                "assistant": _clean_text_for_capture(_message_text(assistant_content))}
        if messages is None and turn["user"]:
            self._turns.append(turn)
        self._capture("turn", messages, completed_only=True)

    # A failed append is retried before the next one, oldest first, within the same bound as other pending writes.
    def _append_turns(self, new: list) -> None:
        with self._capture_lock:
            queue, self._unsent_appends = self._unsent_appends + new, []
            for tag, meta, content in queue:
                if not _quietly(lambda t=tag, m=meta, c=content: self._client.append_document(
                        m["docId"], c, m, container_tag=t, entity_context=self._entity_context, dreaming=self._dreaming) or True,
                        "Supermemory capture failed (turn, document %s in %s)", meta["docId"], tag, level=logging.WARNING, default=False):
                    self._unsent_appends.append((tag, meta, content))
            del self._unsent_appends[:-_MAX_PENDING_DOCS]

    def _flush(self, mode: str, messages: Optional[list] = None) -> None:
        if self._can_write() and self._auto_capture:
            self._append_turns([]) if self._per_speaker else self._capture(mode, messages)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        self._flush("session_end", messages)

    # Compaction continues the conversation and its document; a rewind needs nothing, the next transcript rewrites it.
    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, **kwargs) -> None:
        new_id = str(new_session_id or "").strip()
        self._recalled_ids, self._profile_shown = set(), set()
        if kwargs.get("reason") == "compression":
            self._flush("session_switch")
            with self._capture_lock:
                self._archive, self._turns = _join_turns(self._archive, self._turns), []
        elif (new_id and new_id != self._session_id) or reset:
            self._flush("session_switch")
            with self._capture_lock:
                self._park_unsent()
                self._reset_capture(new_id or self._session_id)
            if reset and self._can_write():
                self._turn_count = 0
        self._session_id = new_id or self._session_id

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not self._can_write() or action != "add" or not (content or "").strip():
            return
        if self._write_thread and self._write_thread.is_alive():
            self._write_thread.join(timeout=2.0)
        self._write_thread = spawn_context_thread(
            _quietly, daemon=False, name="supermemory-memory-write",
            args=(lambda: self._client.add_memory(content.strip(), metadata={"target": target, "type": "explicit_memory", **self._ids(self._speaker)},
                                                  entity_context=self._entity_context, container_tag=self._speaker_tag(self._speaker),
                                                  dreaming=self._dreaming),
                  "Supermemory on_memory_write failed"))
        self._write_thread.start()

    def shutdown(self) -> None:
        self._flush("shutdown")
        if self._write_thread and self._write_thread.is_alive():
            self._write_thread.join(timeout=5.0)
        self._write_thread = None

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = [json.loads(json.dumps(base)) for base in _BASE_SCHEMAS]  # deep copies
        for schema in schemas if self._enable_custom_containers else ():  # multi-container mode: every tool takes container_tag
            schema["parameters"]["properties"]["container_tag"] = {
                "type": "string", "description": f"Optional container tag. Allowed: {', '.join(self._allowed_containers)}. Defaults to primary ({self._container_tag})."}
        if self._allow_agent_search_overrides:
            search = next(s for s in schemas if s["name"] == "supermemory_search")["parameters"]["properties"]
            search["search_mode"] = {"type": "string", "enum": list(_VALID_SEARCH_MODES), "description": f"Defaults to {self._search_mode}."}
            search["aggregate"] = {"type": "boolean", "description": "Merge overlapping results into cleaner answers; slower."}
        return [{**s, "name": _KEBAB_ALIASES[s["name"]]} for s in schemas]

    def _tool_container_tag(self, args: dict) -> Optional[str]:
        """Validated container_tag from args; None = primary. Raises _TagError when not whitelisted."""
        raw = str(args.get("container_tag") or "").strip() if self._enable_custom_containers else ""
        tag = _sanitize_tag(raw) if raw else None
        if tag and tag not in self._allowed_containers:
            raise _TagError(f"Container tag '{tag}' is not allowed. Allowed: {', '.join(self._allowed_containers)}")
        return tag

    def _tool_target(self, tag: Optional[str]) -> str:
        return tag or self._speaker_tag(self._speaker)

    def _tool_store(self, args: dict) -> dict | str:
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required")
        metadata = args.get("metadata") if isinstance(args.get("metadata"), dict) else {}
        metadata.setdefault("type", _detect_category(content))
        metadata.pop("source", None)
        tag = self._tool_container_tag(args)
        result = self._client.add_memory(content, metadata={**self._ids(self._speaker), **metadata}, entity_context=self._entity_context,
                                         container_tag=self._tool_target(tag), dreaming=self._dreaming)
        return _tagged({"saved": True, "id": result.get("id", ""), "preview": content[:80] + ("..." if len(content) > 80 else "")}, tag)

    def _tool_search(self, args: dict) -> dict | str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        limit = _clamp_number(args.get("limit", 5) or 5, 5, 1, 20, int)
        tag = self._tool_container_tag(args)
        overrides = self._allow_agent_search_overrides
        mode = _search_mode(args["search_mode"]) if overrides and args.get("search_mode") else None
        rerank = "aggregate" if overrides and _as_bool(args.get("aggregate"), False) else ""
        results = [{"id": i.get("id", ""), "content": i.get("memory", ""),
                    **({"document_id": i["document_id"]} if i.get("document_id") else {}),
                    **({"similarity": pct} if (pct := _similarity_pct(i.get("similarity"))) is not None else {}),
                    **({"related": i["related"]} if i.get("related") else {})}
                   for i in self._client.search_memories(query, limit=limit, container_tag=self._tool_target(tag), search_mode=mode,
                                                         **self._search_options(rerank))]
        return _tagged({"results": results, "count": len(results)}, tag)

    def _tool_forget(self, args: dict) -> dict | str:
        memory_id, query = str(args.get("id") or "").strip(), str(args.get("query") or "").strip()
        if not memory_id and not query:
            return tool_error("Provide either id or query")
        tag = self._tool_container_tag(args)  # not echoed in the response
        if not memory_id:
            return self._client.forget_by_query(query, container_tag=self._tool_target(tag))
        if error := self._client.forget_memory(memory_id, container_tag=self._tool_target(tag)):
            return tool_error(f"Forget failed: {error}")
        return {"forgotten": True, "id": memory_id}

    def _tool_profile(self, args: dict) -> dict:
        tag = self._tool_container_tag(args)
        query = str(args.get("query") or "").strip()
        resp = {"profile": self._client.get_profile(container_tag=self._tool_target(tag))}
        if query:
            resp["results"] = [{"id": i.get("id", ""), "content": i.get("memory", ""),
                                **({"similarity": pct} if (pct := _similarity_pct(i.get("similarity"))) is not None else {})}
                               for i in self._client.search_memories(query, limit=self._max_recall_results, container_tag=self._tool_target(tag),
                                                                     **self._search_options())]
        return _tagged(resp, tag)

    def _tool_read_document(self, args: dict) -> dict | str:
        document_id = str(args.get("document_id") or "").strip()
        if not document_id:
            return tool_error("document_id is required")
        offset = _clamp_number(args.get("offset") or 0, 0, 0, 10**9, int)
        tag = self._tool_container_tag(args)
        return _tagged(self._client.read_document(document_id, offset, container_tag=self._tool_target(tag)), tag)

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handlers return a tool_error() string for bad args or a dict to JSON-encode; client failures get ``fail_prefix``."""
        if not self._active or not self._client:
            return tool_error("Supermemory is not configured")
        tool_name = _ALIAS_TO_TOOL.get(tool_name, tool_name)
        if tool_name not in self._TOOL_HANDLERS:
            return tool_error(f"Unknown tool: {tool_name}")
        handler, fail_prefix = self._TOOL_HANDLERS[tool_name]
        try:
            resp = handler(self, args)
        except Exception as exc:
            return tool_error(str(exc) if isinstance(exc, _TagError) else f"{fail_prefix}: {exc}")
        return resp if isinstance(resp, str) else json.dumps(resp)

    _TOOL_HANDLERS = {"supermemory_store": (_tool_store, "Failed to store memory"), "supermemory_search": (_tool_search, "Search failed"),
                      "supermemory_forget": (_tool_forget, "Forget failed"), "supermemory_profile": (_tool_profile, "Profile failed"),
                      "supermemory_read_document": (_tool_read_document, "Read failed")}


def register(ctx):
    ctx.register_memory_provider(SupermemoryMemoryProvider())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

FORGET_SCHEMA = {
    "name": "supermemory_forget",
    "description": "Forget a memory by exact id or by best-match query.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Exact memory id to delete."},
            "query": {"type": "string", "description": "Query used to find the memory to forget."},
        },
    },
}

PROFILE_SCHEMA = {
    "name": "supermemory_profile",
    "description": "Retrieve the user's profile: persistent facts and recent memory context.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Optional query; matching memories are returned alongside the profile."},
        },
    },
}

SEARCH_SCHEMA = {
    "name": "supermemory_search",
    "description": "Search long-term memory by semantic similarity.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {"type": "integer", "description": "Maximum results to return, 1 to 20."},
        },
        "required": ["query"],
    },
}

STORE_SCHEMA = {
    "name": "supermemory_store",
    "description": "Store an explicit memory for future recall.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory content to store."},
            "metadata": {"type": "object", "description": "Optional metadata attached to the memory."},
        },
        "required": ["content"],
    },
}
# ---- END PLUGIN-COMPAT ----
