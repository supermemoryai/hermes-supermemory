"""Supermemory provider contracts. The real ``supermemory`` SDK runs against a scripted httpx transport (no sockets)."""

import inspect
import json
import os
import stat
from datetime import datetime
from urllib.parse import unquote

import httpx
import pytest

import hermes_supermemory as sm
from hermes_supermemory import SupermemoryMemoryProvider

supermemory = pytest.importorskip("supermemory")

PROFILE = "# Profile\n\n## Static\n- Morgan runs a bakery\n\n## Recent Memories\n- Switched to oat milk"


class FakeAPI:
    """Stands in for the v5 API behind the SDK's transport: records requests, replays scripted failures.

    Document writes follow mono's ingest rules: POST to an existing id keeps the stored text when the new text
    does not extend it (``old + delimiter + new``), and PATCH replaces the content. ``lose`` applies a write
    and then fails the response, like a response lost on the way back."""

    def __init__(self):
        self.requests, self.failures, self.lose, self.docs = [], [], [], {}
        self.results = [{"id": "m1", "memory": "Morgan prefers email", "similarity": 0.91,
                         "metadata": {"docId": "d1"}}]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.raw_path.decode().split("?", 1)[0]
        self.requests.append({"method": request.method, "path": path, "body": body,
                              "headers": {k.lower(): v for k, v in request.headers.items()}})
        for failure in list(self.failures):
            if path.endswith(failure[0]) or (failure[0] == "/document" and "/document/" in path):
                self.failures.remove(failure)
                return httpx.Response(failure[1], json={"error": "refused"})
        response = self.route(request.method, path, body)
        if self.lose and request.method in ("POST", "PATCH") and "/document" in path:
            self.lose.pop()
            return httpx.Response(503, json={"error": "lost"})
        return response

    def route(self, method, path, body):
        if path.endswith("/profile"):
            return httpx.Response(200, text=PROFILE, headers={"content-type": "text/markdown"})
        if path.endswith("/search"):
            return httpx.Response(200, json={"results": self.results})
        if path.endswith("/document") and method == "POST":
            doc_id = body.get("id") or f"auto-{len(self.docs)}"
            old = self.docs.get(doc_id, {}).get("content", "")
            content = body["content"] if not old or body["content"].startswith(old) else old + "\n\n" + body["content"]
            self.docs[doc_id] = {**self.docs.get(doc_id, {}), **body, "id": doc_id, "content": content}
            return httpx.Response(200, json={"id": doc_id, "status": "queued"})
        if "/document/" in path and method == "PATCH":
            doc_id = unquote(path.rsplit("/", 1)[1])
            if doc_id not in self.docs:
                return httpx.Response(404, json={"error": "Document not found"})
            self.docs[doc_id] = {**self.docs[doc_id], **body, "id": doc_id}
            return httpx.Response(200, json={"id": doc_id, "status": "queued"})
        if path.endswith("/list/documents"):
            wanted = next((p["value"] for p in [body.get("filter") or {}] if p.get("field") == "session_id"), None)
            docs = [{"id": k, "type": "text", "metadata": v.get("metadata") or {}, "system": {"status": "done"}}
                    for k, v in self.docs.items() if (v.get("metadata") or {}).get("session_id") == wanted]
            return httpx.Response(200, json={"documents": docs, "chunks": [], "memories": [],
                                             "pagination": {"currentPage": 1, "totalItems": len(docs), "totalPages": 1, "limit": 100}})
        if "/document/" in path:
            doc_id = unquote(path.rsplit("/", 1)[1])
            if doc_id not in self.docs:
                return httpx.Response(404, json={"error": "Document not found"})
            return httpx.Response(200, json={"type": "text", "metadata": {}, "system": {"status": "done"}, **self.docs[doc_id]})
        if "forget" in path:
            return httpx.Response(200, json={"count": len(body.get("ids") or []), "errors": [], "matches": []})
        return httpx.Response(200, json={})

    def writes(self):
        return [r for r in self.requests if (r["method"] == "POST" and r["path"].endswith("/document"))
                or (r["method"] == "PATCH" and "/document/" in r["path"])]

    def captures(self):
        return {k: v["content"] for k, v in self.docs.items() if (v.get("metadata") or {}).get("type") == "conversation"}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    sm._LEGACY_TEXT.clear()
    for name in list(os.environ):
        if name.startswith("SUPERMEMORY_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def api(monkeypatch, tmp_path):
    fake = FakeAPI()
    real = supermemory.Supermemory
    monkeypatch.setattr(supermemory, "Supermemory",
                        lambda **options: real(http_client=httpx.Client(transport=httpx.MockTransport(fake)), **options))
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    return fake


def make_provider(home, session_id="session-1", **kwargs):
    provider = SupermemoryMemoryProvider()
    provider.initialize(session_id, hermes_home=str(home), platform="cli", **kwargs)
    return provider


def transcript(*turns):
    messages = []
    for user, reply in turns:
        messages += [{"role": "user", "content": user},
                     {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function",
                                                                             "function": {"name": "x", "arguments": "{}"}}]},
                     {"role": "tool", "tool_call_id": "c", "content": "ok"},
                     {"role": "assistant", "content": reply}]
    return messages


TURN1 = "[role: user]\nBook the usual table\n[user:end]\n[role: assistant]\nBooked.\n[assistant:end]"
TURN2 = "[role: user]\nAnd a cab at 7.\n[user:end]\n[role: assistant]\nCab booked.\n[assistant:end]"


def test_needs_only_an_api_key(monkeypatch):
    assert SupermemoryMemoryProvider().is_available() is False
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "k")
    assert SupermemoryMemoryProvider().is_available() is True


def test_config_file_and_normalization(tmp_path):
    (tmp_path / "supermemory.json").write_text(json.dumps(
        {"container_tag": "hermes-{identity}", "max_recall_results": 50, "search_mode": "documents", "dreaming": "Instant"}))
    config = sm._load_supermemory_config(str(tmp_path))
    assert (config["max_recall_results"], config["search_mode"], config["dreaming"]) == (20, "chunks", "instant")
    assert sm._sanitize_tag(sm._expand_container_tag(config["container_tag"], "coder")) == "hermes-coder"
    assert sm._load_supermemory_config(None)["dreaming"] == "dynamic"
    assert sm._load_supermemory_config(None)["profile_frequency"] == 50


def test_each_turn_rewrites_one_growing_document(api, tmp_path):
    provider = make_provider(tmp_path)
    first = transcript(("Book the usual table", "Booked."))
    provider.sync_turn("Book the usual table", "Booked.", messages=first)
    assert api.captures() == {"hermes_session-1": TURN1}  # stored right after the turn
    second = first + transcript(("And a cab at 7.", "Cab booked."))
    provider.sync_turn("And a cab at 7.", "Cab booked.", messages=second)
    provider.on_session_end(second)  # nothing new
    provider.shutdown()
    writes = api.writes()
    assert [(w["method"], w["path"]) for w in writes] == [("PATCH", "/ns/hermes/document/hermes_session-1"),
                                                          ("POST", "/ns/hermes/document"),
                                                          ("PATCH", "/ns/hermes/document/hermes_session-1")]
    body = writes[1]["body"]
    assert body["id"] == "hermes_session-1" and body["content"] == TURN1
    assert body["metadata"] == {"sm_source": "hermes", "type": "conversation", "session_id": "session-1",
                                "docId": "hermes_session-1"}
    assert body["supportingContext"] == sm._DEFAULT_ENTITY_CONTEXT and body["dreaming"] == "dynamic"
    assert datetime.fromisoformat(body["date"]).tzinfo is not None  # the write time
    assert writes[2]["body"]["content"] == TURN1 + "\n\n" + TURN2 and "id" not in writes[2]["body"]
    assert api.captures() == {"hermes_session-1": TURN1 + "\n\n" + TURN2}


def test_content_is_stored_as_given(api, tmp_path):
    provider = make_provider(tmp_path)
    user = "[Tue 2026-04-28 13:40:53 CEST] Book the usual table"
    provider.sync_turn(user, "Booked.", messages=transcript((user, "Booked.")))
    assert api.writes()[0]["body"]["content"].startswith(f"[role: user]\n{user}\n")


def test_turn_still_in_progress_is_left_out(api, tmp_path):
    provider = make_provider(tmp_path)
    running_ahead = transcript(("Book the usual table", "Booked.")) + [{"role": "user", "content": "Next"}]
    provider.sync_turn("Book the usual table", "Booked.", messages=running_ahead)
    assert api.writes()[-1]["body"]["content"] == TURN1


def test_refused_write_is_logged_not_raised_and_retried(api, tmp_path, caplog):
    provider = make_provider(tmp_path)
    api.failures = [("/document", 405)]
    messages = transcript(("Book the usual table", "Booked."))
    provider.sync_turn("Book the usual table", "Booked.", messages=messages)  # does not raise
    assert "Supermemory capture failed" in caplog.text and api.captures() == {}
    provider.on_session_end(messages)
    assert api.captures() == {"hermes_session-1": TURN1}


def test_new_provider_per_turn_keeps_one_document(api, tmp_path):
    # Hermes' gateway builds a new provider for every turn of a session and hands it the whole transcript.
    first = transcript(("Book the usual table", "Booked."))
    second = first + transcript(("And a cab at 7.", "Cab booked."))
    for user, reply, messages in (("Book the usual table", "Booked.", first), ("And a cab at 7.", "Cab booked.", second)):
        provider = make_provider(tmp_path, session_id="local-ab12")
        provider.sync_turn(user, reply, session_id="local-ab12", messages=messages)
        provider.on_session_end(messages)
        provider.shutdown()
    assert api.captures() == {"hermes_local-ab12": TURN1 + "\n\n" + TURN2}


def test_session_reset_starts_a_new_document(api, tmp_path):
    provider = make_provider(tmp_path)
    provider.sync_turn("Book the usual table", "Booked.", messages=transcript(("Book the usual table", "Booked.")))
    provider.on_session_switch("session-2", reset=True)
    provider.sync_turn("And a cab at 7.", "Cab booked.", messages=transcript(("And a cab at 7.", "Cab booked.")))
    assert api.captures() == {"hermes_session-1": TURN1, "hermes_session-2": TURN2}


def test_non_primary_contexts_never_write(api, tmp_path):
    provider = make_provider(tmp_path, agent_context="cron")
    provider.sync_turn("Book the usual table", "Booked.", messages=transcript(("Book the usual table", "Booked.")))
    provider.on_memory_write("add", "user", "likes tea")
    provider.shutdown()
    assert api.writes() == []


def test_recall_is_full_profile_plus_search(api, tmp_path):
    text = make_provider(tmp_path).prefetch("Book the usual table")
    assert text.startswith("<supermemory-context>\n") and text.endswith("\n</supermemory-context>")
    assert PROFILE in text  # every section, including Recent Memories
    assert "## Relevant Memories\n- [91%] Morgan prefers email" in text
    search = next(r for r in api.requests if r["path"].endswith("/search"))
    assert search["path"] == "/ns/hermes/search"
    assert search["body"] == {"query": "Book the usual table", "limit": 10, "searchMode": "hybrid"}
    assert search["headers"]["x-sm-source"] == "hermes" and search["headers"]["authorization"] == "Bearer test-key"
    profile = next(r for r in api.requests if r["path"].endswith("/profile"))
    assert profile["path"] == "/ns/hermes/profile" and profile["headers"]["accept"] == "text/markdown"


def test_recall_survives_one_failing_call(api, tmp_path):
    api.failures = [("/profile", 503)]
    text = make_provider(tmp_path).prefetch("Book the usual table")
    assert "Morgan prefers email" in text and "Morgan runs a bakery" not in text


def test_recalled_context_is_not_captured(api, tmp_path):
    provider = make_provider(tmp_path)
    recalled = provider.prefetch("Book the usual table")
    provider.sync_turn(f"Book the usual table\n\n{recalled}", "Booked.")
    assert api.writes()[0]["body"]["content"] == TURN1


def test_tools_and_kebab_aliases(api, tmp_path):
    provider = make_provider(tmp_path)
    schemas = provider.get_tool_schemas()
    assert "query" in next(s for s in schemas if s["name"] == "supermemory_profile")["parameters"]["properties"]
    names = [s["name"] for s in schemas]
    assert names == ["supermemory_store", "supermemory_search", "supermemory_forget", "supermemory_profile",
                     "supermemory_read_document", "supermemory-save", "supermemory-search", "supermemory-forget",
                     "supermemory-profile", "supermemory-read-document"]
    saved = json.loads(provider.handle_tool_call("supermemory-save", {"content": "The shop is closed on Sundays."}))
    assert saved["saved"] is True
    store = api.writes()[-1]["body"]
    assert "id" not in store and store["content"] == "The shop is closed on Sundays."
    assert store["metadata"] == {"sm_source": "hermes", "type": "fact"}

    found = json.loads(provider.handle_tool_call("supermemory_search", {"query": "email", "limit": 3}))
    assert found == {"results": [{"id": "m1", "content": "Morgan prefers email", "document_id": "d1", "similarity": 91}],
                     "count": 1}
    assert json.loads(provider.handle_tool_call("supermemory_profile", {})) == {"profile": PROFILE}
    focused = json.loads(provider.handle_tool_call("supermemory_profile", {"query": "email"}))
    assert focused == {"profile": PROFILE, "results": [{"id": "m1", "content": "Morgan prefers email", "similarity": 91}]}

    assert json.loads(provider.handle_tool_call("supermemory_forget", {"id": "m1"})) == {"forgotten": True, "id": "m1"}
    assert api.requests[-1]["body"] == {"ids": ["m1"]} and "/ns/hermes/" in api.requests[-1]["path"]
    by_query = json.loads(provider.handle_tool_call("supermemory-forget", {"query": "email"}))
    assert by_query["success"] is True and by_query["id"] == "m1"


def test_read_document_pages_and_encodes_ids(api, tmp_path):
    provider = make_provider(tmp_path)
    api.docs["d1"] = {"id": "d1", "content": "x" * 9000, "date": "2026-02-01"}
    page = json.loads(provider.handle_tool_call("supermemory_read_document", {"document_id": "d1"}))
    assert page == {"id": "d1", "date": "2026-02-01", "content": "x" * 8000, "next_offset": 8000}
    rest = json.loads(provider.handle_tool_call("supermemory-read-document", {"document_id": "d1", "offset": 8000}))
    assert rest["content"] == "x" * 1000 and rest["next_offset"] is None
    api.docs["a b/c?#d"] = {"id": "a b/c?#d", "content": "y"}
    provider.handle_tool_call("supermemory_read_document", {"document_id": "a b/c?#d"})
    assert api.requests[-1]["path"] == "/ns/hermes/document/a%20b%2Fc%3F%23d"
    missing = json.loads(provider.handle_tool_call("supermemory_read_document", {"document_id": "nope"}))
    assert missing["error"].startswith("Read failed:")


def test_builtin_memory_add_is_mirrored(api, tmp_path):
    provider = make_provider(tmp_path)
    provider.on_memory_write("add", "user", "likes tea")
    provider.on_memory_write("replace", "user", "likes coffee")
    provider.shutdown()
    bodies = [w["body"] for w in api.writes()]
    assert bodies == [{"content": "likes tea", "date": bodies[0]["date"], "supportingContext": sm._DEFAULT_ENTITY_CONTEXT,
                       "dreaming": "dynamic",
                       "metadata": {"sm_source": "hermes", "target": "user", "type": "explicit_memory"}}]


def test_custom_containers_are_namespaces(api, tmp_path):
    (tmp_path / "supermemory.json").write_text(json.dumps(
        {"enable_custom_container_tags": True, "custom_containers": ["project-alpha"]}))
    provider = make_provider(tmp_path)
    provider.handle_tool_call("supermemory_search", {"query": "x", "container_tag": "project-alpha"})
    assert api.requests[-1]["path"] == "/ns/project-alpha/search"
    refused = json.loads(provider.handle_tool_call("supermemory_search", {"query": "x", "container_tag": "other"}))
    assert "not allowed" in refused["error"]


def test_dreaming_can_be_instant(api, tmp_path):
    (tmp_path / "supermemory.json").write_text(json.dumps({"dreaming": "instant"}))
    provider = make_provider(tmp_path)
    provider.sync_turn("Book the usual table", "Booked.")
    assert api.writes()[0]["body"]["dreaming"] == "instant"


def test_pre_v5_sdk_disables_the_provider(api, tmp_path, monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda name: "3.50.0")
    provider = make_provider(tmp_path)
    assert provider._client is None
    assert json.loads(provider.handle_tool_call("supermemory_search", {"query": "x"}))["error"] == "Supermemory is not configured"


def test_memory_manager_dispatch(api, tmp_path):
    # Hermes passes optional sync_turn kwargs (messages, turn_author) only to signatures that accept them.
    from agent.memory_manager import MemoryManager

    manager = MemoryManager()
    manager.add_provider(make_provider(tmp_path))
    extra = ({"turn_author": {"id": "u1", "name": "Morgan", "is_bot": False}}
             if "turn_author" in inspect.signature(manager.sync_all).parameters else {})
    manager.sync_all("Book the usual table", "Booked.", session_id="session-1",
                     messages=transcript(("Book the usual table", "Booked.")), **extra)
    manager.flush_pending(timeout=10)
    manager.shutdown_all()
    assert api.captures() == {"hermes_session-1": TURN1}


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits not enforced on Windows")
def test_save_config_sets_owner_only_permissions(tmp_path):
    SupermemoryMemoryProvider().save_config({"container_tag": "my tag"}, str(tmp_path))
    path = tmp_path / "supermemory.json"
    assert json.loads(path.read_text())["container_tag"] == "my_tag"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


TURN3 = "[role: user]\nMake it 8 instead.\n[user:end]\n[role: assistant]\nMoved to 8.\n[assistant:end]"
SUMMARY = ("[CONTEXT COMPACTION — REFERENCE ONLY] The user booked a table.\n\n"
           "--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---")


def test_rewind_and_edit_rewrite_the_document(api, tmp_path):
    provider = make_provider(tmp_path)
    both = transcript(("Book the usual table", "Booked."), ("And a cab at 7.", "Cab booked."))
    provider.sync_turn("And a cab at 7.", "Cab booked.", messages=both)
    provider.on_session_switch("session-1", rewound=True)
    rewound = transcript(("Book the usual table", "Booked."), ("Make it 8 instead.", "Moved to 8."))
    provider.sync_turn("Make it 8 instead.", "Moved to 8.", messages=rewound)
    assert api.captures() == {"hermes_session-1": TURN1 + "\n\n" + TURN3}
    edited = transcript(("Book the usual table", "Booked."), ("Make it 8:30 instead.", "Moved to 8:30."))
    make_provider(tmp_path).sync_turn("Make it 8:30 instead.", "Moved to 8:30.", messages=edited)
    assert api.captures()["hermes_session-1"].count("Book the usual table") == 1
    assert "Moved to 8:30." in api.captures()["hermes_session-1"] and "Moved to 8." not in api.captures()["hermes_session-1"]


@pytest.mark.parametrize("new_provider", [False, True], ids=["same-provider", "provider-per-turn"])
def test_compaction_keeps_one_copy(api, tmp_path, new_provider):
    provider = make_provider(tmp_path)
    before = transcript(("Book the usual table", "Booked."), ("And a cab at 7.", "Cab booked."))
    provider.sync_turn("And a cab at 7.", "Cab booked.", messages=before)
    provider.on_session_switch("session-1", reason="compression")
    compacted = [{"role": "user", "content": SUMMARY, "_compressed_summary": True}] + transcript(
        ("And a cab at 7.", "Cab booked."), ("Make it 8 instead.", "Moved to 8."))
    if new_provider:
        provider = make_provider(tmp_path)
    provider.sync_turn("Make it 8 instead.", "Moved to 8.", messages=compacted)
    assert api.captures() == {"hermes_session-1": "\n\n".join((TURN1, TURN2, TURN3))}


def test_compaction_into_a_child_session_keeps_writing_the_same_document(api, tmp_path):
    provider = make_provider(tmp_path)
    before = transcript(("Book the usual table", "Booked."), ("And a cab at 7.", "Cab booked."))
    provider.sync_turn("And a cab at 7.", "Cab booked.", messages=before)
    provider.on_session_switch("session-1-c1", parent_session_id="session-1", reason="compression")
    compacted = [{"role": "user", "content": SUMMARY, "_compressed_summary": True}] + transcript(
        ("And a cab at 7.", "Cab booked."), ("Make it 8 instead.", "Moved to 8."))
    provider.sync_turn("Make it 8 instead.", "Moved to 8.", session_id="session-1-c1", messages=compacted)
    assert api.captures() == {"hermes_session-1": "\n\n".join((TURN1, TURN2, TURN3))}


def test_summary_merged_into_a_kept_message_is_left_out():
    merged = ("[PRIOR CONTEXT — for reference only; not a new message]\nAnd a cab at 7.\n\n"
              "[END OF PRIOR CONTEXT — COMPACTION SUMMARY BELOW]\n\n" + SUMMARY)
    turns = sm._turns_from_messages([{"role": "user", "content": merged, "_compressed_summary": True},
                                     {"role": "assistant", "content": "Cab booked."}])
    assert turns == [{"user": "And a cab at 7.", "assistant": "Cab booked."}]


def test_resend_after_a_lost_response_keeps_one_copy(api, tmp_path):
    provider = make_provider(tmp_path)
    first = transcript(("Book the usual table", "Booked."))
    api.lose = [True]
    provider.sync_turn("Book the usual table", "Booked.", messages=first)
    second = first + transcript(("And a cab at 7.", "Cab booked."))
    api.lose = [True]
    provider.sync_turn("And a cab at 7.", "Cab booked.", messages=second)
    provider.on_session_end(second)
    assert api.captures() == {"hermes_session-1": TURN1 + "\n\n" + TURN2}


# 1.0.x appended turn 1 to its 4-hour window document; the session goes on under 2.0.
def test_upgraded_session_continues_without_storing_turns_twice(api, tmp_path):
    api.docs["session-1_2026-10-20_b2"] = {"id": "session-1_2026-10-20_b2", "content": TURN1,
                                           "metadata": {"sm_source": "hermes", "type": "conversation", "session_id": "session-1"}}
    messages = transcript(("Book the usual table", "Booked."), ("And a cab at 7.", "Cab booked."))
    make_provider(tmp_path).sync_turn("And a cab at 7.", "Cab booked.", messages=messages)
    assert api.captures() == {"session-1_2026-10-20_b2": TURN1, "hermes_session-1": TURN2}
    listed = next(r for r in api.requests if r["path"].endswith("/list/documents"))
    assert listed["body"]["filter"] == {"field": "session_id", "operator": "eq", "value": "session-1"}


def test_upgrade_lookup_failure_heals_on_the_next_write(api, tmp_path):
    api.docs["session-1_2026-10-20_b2"] = {"id": "session-1_2026-10-20_b2", "content": TURN1,
                                           "metadata": {"type": "conversation", "session_id": "session-1"}}
    api.failures = [("/list/documents", 503)]
    provider = make_provider(tmp_path)
    two = transcript(("Book the usual table", "Booked."), ("And a cab at 7.", "Cab booked."))
    provider.sync_turn("And a cab at 7.", "Cab booked.", messages=two)
    three = two + transcript(("Make it 8 instead.", "Moved to 8."))
    provider.sync_turn("Make it 8 instead.", "Moved to 8.", messages=three)
    assert api.captures()["hermes_session-1"] == TURN2 + "\n\n" + TURN3


def test_failed_documents_of_past_sessions_are_retried_and_bounded(api, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(sm, "_MAX_PENDING_DOCS", 2)
    provider = make_provider(tmp_path, session_id="s0")
    api.failures = [("/document", 503)] * 100
    for n in range(4):
        provider.sync_turn("Book the usual table", "Booked.", messages=transcript(("Book the usual table", "Booked.")))
        provider.on_session_switch(f"s{n + 1}", reset=True)
    assert list(provider._pending) == ["hermes_s2", "hermes_s3"] and "keep the retry buffer bounded" in caplog.text
    api.failures = []
    provider.shutdown()
    assert sorted(api.captures()) == ["hermes_s2", "hermes_s3"]


def test_profile_is_recalled_on_the_first_turn_and_every_profile_frequency_turns(api, tmp_path):
    (tmp_path / "supermemory.json").write_text(json.dumps({"profile_frequency": 3}))
    provider = make_provider(tmp_path)
    recalled = []
    for turn in range(1, 7):
        provider.on_turn_start(turn, "hi")
        recalled.append("Morgan runs a bakery" in provider.prefetch("Book the usual table"))
    assert recalled == [True, False, True, False, False, True]


@pytest.mark.parametrize("raw, namespace", [
    ("team-coder", "team-coder"),
    ("team:shared", "team:shared"),
    ("my.project", "my_project"),
    ("x" * 120, "x" * 91 + "_" + sm.hashlib.sha256(("x" * 120).encode()).hexdigest()[:8]),
], ids=["hyphen", "colon", "dot", "over-100"])
def test_container_tags_reach_v5_as_namespaces(api, tmp_path, raw, namespace):
    (tmp_path / "supermemory.json").write_text(json.dumps({"container_tag": raw}))
    provider = make_provider(tmp_path)
    provider.handle_tool_call("supermemory_search", {"query": "q"})
    provider.prefetch("q")
    provider.sync_turn("Book the usual table", "Booked.", messages=transcript(("Book the usual table", "Booked.")))
    assert {r["path"].split("/")[2] for r in api.requests} == {namespace}
    assert len(namespace) <= 100 and sm.re.fullmatch(r"[a-zA-Z0-9_:-]+", namespace)


def test_legacy_tag_warning_still_fires_once(api, tmp_path, caplog):
    (tmp_path / "supermemory.json").write_text(json.dumps({"container_tag": "hermes-{identity}"}))
    make_provider(tmp_path, agent_identity="coder")
    make_provider(tmp_path, agent_identity="coder")
    assert caplog.text.count("hermes_coder -> hermes-coder") == 1


def test_long_session_ids_keep_document_ids_within_100_characters(api, tmp_path):
    provider = make_provider(tmp_path, session_id="s" * 120)
    provider.sync_turn("Book the usual table", "Booked.", messages=transcript(("Book the usual table", "Booked.")))
    (doc_id,) = api.captures()
    assert len(doc_id) == 100 and doc_id.startswith("hermes_sss")


def test_setup_reads_the_key_through_hermes_and_leaves_the_environment_alone(api, tmp_path, monkeypatch):
    import hermes_cli.config
    import hermes_cli.memory_setup

    monkeypatch.setattr(hermes_cli.config, "save_config", lambda config: None)
    monkeypatch.setattr(hermes_cli.memory_setup, "_prompt", lambda *a, **k: "")
    seen = []
    monkeypatch.setattr(sm, "get_secret", lambda name, default=None: seen.append(name) or "scoped-key")
    before = dict(os.environ)
    SupermemoryMemoryProvider().post_setup(str(tmp_path), {})
    assert "SUPERMEMORY_API_KEY" in seen and dict(os.environ) == before
    assert api.requests[-1]["headers"]["authorization"] == "Bearer scoped-key"
    assert "os.environ" not in inspect.getsource(sm)
