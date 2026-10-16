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
    """Stands in for the v5 API behind the SDK's transport: records requests, replays scripted failures."""

    def __init__(self):
        self.requests, self.failures, self.docs = [], [], {}
        self.results = [{"id": "m1", "memory": "Morgan prefers email", "similarity": 0.91,
                         "metadata": {"docId": "d1"}}]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.raw_path.decode().split("?", 1)[0]
        self.requests.append({"method": request.method, "path": path, "body": body,
                              "headers": {k.lower(): v for k, v in request.headers.items()}})
        for failure in list(self.failures):
            if path.endswith(failure[0]):
                self.failures.remove(failure)
                return httpx.Response(failure[1], json={"error": "refused"})
        if path.endswith("/profile"):
            return httpx.Response(200, text=PROFILE, headers={"content-type": "text/markdown"})
        if path.endswith("/search"):
            return httpx.Response(200, json={"results": self.results})
        if path.endswith("/document") and request.method == "POST":
            doc_id = body.get("id") or f"auto-{len(self.docs)}"
            self.docs[doc_id] = {**body, "id": doc_id}
            return httpx.Response(200, json={"id": doc_id, "status": "queued"})
        if "/document/" in path:
            doc_id = unquote(path.rsplit("/", 1)[1])
            if doc_id not in self.docs:
                return httpx.Response(404, json={"error": "Document not found"})
            return httpx.Response(200, json=self.docs[doc_id])
        if "forget" in path:
            return httpx.Response(200, json={"count": len(body.get("ids") or []), "errors": [], "matches": []})
        return httpx.Response(200, json={})

    def writes(self):
        return [r for r in self.requests if r["method"] == "POST" and r["path"].endswith("/document")]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
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
    assert sm._resolve_container_tag(config["container_tag"], "coder") == "hermes_coder"
    assert sm._load_supermemory_config(None)["dreaming"] == "dynamic"
    assert "profile_frequency" not in sm._load_supermemory_config(None)


def test_each_turn_upserts_one_growing_document(api, tmp_path):
    provider = make_provider(tmp_path)
    first = transcript(("Book the usual table", "Booked."))
    provider.sync_turn("Book the usual table", "Booked.", messages=first)
    assert len(api.writes()) == 1  # stored right after the turn
    second = first + transcript(("And a cab at 7.", "Cab booked."))
    provider.sync_turn("And a cab at 7.", "Cab booked.", messages=second)
    provider.on_session_end(second)  # nothing new
    provider.shutdown()
    writes = api.writes()
    assert [w["path"] for w in writes] == ["/ns/hermes/document"] * 2
    body = writes[0]["body"]
    assert body["id"] == "hermes_session_1" and body["content"] == TURN1
    assert body["metadata"] == {"sm_source": "hermes", "type": "conversation", "session_id": "session-1",
                                "docId": "hermes_session_1"}
    assert body["supportingContext"] == sm._DEFAULT_ENTITY_CONTEXT and body["dreaming"] == "dynamic"
    assert datetime.fromisoformat(body["date"]).tzinfo is not None  # the write time
    assert writes[1]["body"]["id"] == "hermes_session_1" and writes[1]["body"]["content"] == TURN1 + "\n\n" + TURN2
    assert list(api.docs) == ["hermes_session_1"]


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
    assert "Supermemory capture failed" in caplog.text
    provider.on_session_end(messages)
    assert len(api.writes()) == 2 and api.docs["hermes_session_1"]["content"] == TURN1


def test_new_provider_per_turn_keeps_one_document(api, tmp_path):
    # Hermes' gateway builds a new provider for every turn of a session and hands it the whole transcript.
    first = transcript(("Book the usual table", "Booked."))
    second = first + transcript(("And a cab at 7.", "Cab booked."))
    for user, reply, messages in (("Book the usual table", "Booked.", first), ("And a cab at 7.", "Cab booked.", second)):
        provider = make_provider(tmp_path, session_id="local-ab12")
        provider.sync_turn(user, reply, session_id="local-ab12", messages=messages)
        provider.on_session_end(messages)
        provider.shutdown()
    assert [w["body"]["id"] for w in api.writes()] == ["hermes_local_ab12"] * 2
    assert api.docs["hermes_local_ab12"]["content"] == TURN1 + "\n\n" + TURN2


def test_session_reset_starts_a_new_document(api, tmp_path):
    provider = make_provider(tmp_path)
    provider.sync_turn("Book the usual table", "Booked.", messages=transcript(("Book the usual table", "Booked.")))
    provider.on_session_switch("session-2", reset=True)
    provider.sync_turn("And a cab at 7.", "Cab booked.", messages=transcript(("And a cab at 7.", "Cab booked.")))
    assert {k: v["content"] for k, v in api.docs.items()} == {"hermes_session_1": TURN1, "hermes_session_2": TURN2}


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
    names = [s["name"] for s in provider.get_tool_schemas()]
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
    assert api.requests[-1]["path"] == "/ns/project_alpha/search"
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
    assert [w["body"]["id"] for w in api.writes()] == ["hermes_session_1"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits not enforced on Windows")
def test_save_config_sets_owner_only_permissions(tmp_path):
    SupermemoryMemoryProvider().save_config({"container_tag": "my tag"}, str(tmp_path))
    path = tmp_path / "supermemory.json"
    assert json.loads(path.read_text())["container_tag"] == "my_tag"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
