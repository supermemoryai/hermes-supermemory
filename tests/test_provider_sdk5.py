"""SDK5 / v5 API contract tests for the provider (synthetic data, mock HTTP transport only)."""

from __future__ import annotations

import importlib.metadata
import time
import tomllib
from datetime import datetime, timezone

import httpx
import pytest

from conftest import API_KEY, BASE_URL, ROOT, STAMP, search_hit

CAPTURE_PATH = "/ns/hermes/document"


# ---- dependency declaration / SDK gate -------------------------------------------------------

def test_installed_sdk_is_inside_the_declared_range():
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    assert declared == ["supermemory>=5.0.0,<6"]
    assert "  - supermemory>=5.0.0,<6\n" in (ROOT / "plugin.yaml").read_text()
    assert importlib.metadata.version("supermemory").split(".")[0] == "5"


@pytest.mark.parametrize("version,accepted", [("3.62.0", False), ("4.0.0", False), ("5.0.0", True), ("5.3.1", True), ("6.0.0", False)])
def test_sdk_gate_accepts_only_the_v5_major(harness, monkeypatch, version, accepted):
    monkeypatch.setattr(harness.mod.importlib.metadata, "version", lambda _name: version)
    provider = harness.provider()
    assert provider._active is accepted
    status = harness.mod._probe_supermemory_connection(API_KEY, str(harness.home))
    assert status["ok"] is accepted
    if not accepted:
        assert "needs supermemory>=5,<6" in status["error"]
        assert harness.wire.requests == []


# ---- capture: append by stable id, isolation, retries ------------------------------------------

def test_turn_capture_appends_by_stable_id_with_source_metadata(harness):
    provider = harness.provider()
    provider.sync_turn("Hi <supermemory-context>recalled</supermemory-context>there", "Hello")
    provider.sync_turn("see data:image/png;base64,AAAA", "Nice")
    provider.on_session_end([{"role": "user", "content": "already captured"}])
    provider.shutdown()

    assert [(r["method"], r["path"]) for r in harness.wire.requests] == [("POST", CAPTURE_PATH)] * 2
    first, second = (r["body"] for r in harness.wire.requests)
    assert first["id"] == second["id"] and first["id"].startswith("session_a_")
    assert first["content"] == "[role: user]\nHi there\n[user:end]\n[role: assistant]\nHello\n[assistant:end]"
    assert "[image]" in second["content"] and "base64" not in second["content"]
    assert first["taskType"] == "memory"
    assert first["supportingContext"] == harness.mod._DEFAULT_ENTITY_CONTEXT
    assert first["metadata"]["sm_source"] == "hermes"
    assert first["metadata"]["type"] == "conversation" and first["metadata"]["session_id"] == "session-a"
    assert not {"containerTags", "customId", "dreaming", "entityContext"} & set(first)


@pytest.mark.parametrize("failure", [503, 409, 429, httpx.ReadTimeout("synthetic lost acknowledgement"),
                                     httpx.ConnectError("synthetic refused")])
def test_failed_capture_is_sent_once_and_retried_only_by_the_provider(harness, failure):
    provider = harness.provider()
    harness.wire.overrides["document"] = failure
    provider.sync_turn("remember me", "ok")
    assert len(harness.wire.requests) == 1  # SDK retries are off: no hidden duplicate appends
    assert len(provider._pending_turns) == 1
    sent = harness.wire.requests[0]["body"]

    del harness.wire.overrides["document"]
    provider.on_session_end([])
    assert len(harness.wire.requests) == 2 and harness.wire.requests[1]["body"]["content"] == sent["content"]
    assert harness.wire.requests[1]["body"]["id"] == sent["id"]
    assert provider._pending_turns == []
    provider.shutdown()
    assert len(harness.wire.requests) == 2  # nothing left to resend after a successful retry


def test_pending_retry_carries_its_original_session(harness):
    provider = harness.provider()
    harness.wire.overrides["document"] = 503
    provider.sync_turn("old session turn", "ok")
    del harness.wire.overrides["document"]
    provider.on_session_switch("session-b", reset=True)
    provider.sync_turn("new session turn", "ok")
    sessions = [r["body"]["metadata"]["session_id"] for r in harness.wire.requests]
    assert sessions == ["session-a", "session-a", "session-b"]
    assert provider._pending_turns == []


def test_pending_buffer_stays_bounded(harness):
    provider = harness.provider()
    harness.wire.overrides["document"] = 503
    for i in range(harness.mod._MAX_PENDING_TURNS + 3):
        provider.sync_turn(f"turn {i}", "ok")
    assert len(provider._pending_turns) == harness.mod._MAX_PENDING_TURNS
    assert provider._pending_turns[0]["user"] == "turn 3"  # oldest dropped first


@pytest.mark.parametrize("context", ["cron", "flush", "subagent"])
def test_non_primary_contexts_never_write(harness, context):
    provider = harness.provider(agent_context=context)
    provider.sync_turn("user", "assistant")
    provider.on_memory_write("add", "memory", "mirrored fact")
    provider.on_session_switch("session-b")
    provider.on_session_end([])
    provider.shutdown()
    assert harness.wire.requests == []


def test_auto_capture_and_auto_recall_can_be_disabled(harness):
    harness.write_config(auto_capture=False, auto_recall=False)
    provider = harness.provider()
    provider.sync_turn("not stored", "not stored")
    assert provider.prefetch("anything") == ""
    provider.shutdown()
    assert harness.wire.requests == []


def test_explicit_store_and_memory_mirror_use_the_v5_add(harness):
    provider = harness.provider()
    result = harness.tool(provider, "supermemory-save", {"content": "I prefer tea", "metadata": {"source": "x"}})
    assert result == {"saved": True, "id": "doc-generated", "preview": "I prefer tea"}
    body = harness.wire.requests[-1]["body"]
    assert harness.wire.requests[-1]["path"] == CAPTURE_PATH
    assert body["metadata"] == {"sm_source": "hermes", "type": "preference"} and "id" not in body

    provider.on_memory_write("add", "user", "mirrored fact")
    provider.shutdown()
    assert harness.wire.requests[-1]["body"]["metadata"] == {"sm_source": "hermes", "target": "user", "type": "explicit_memory"}


# ---- namespaces ----------------------------------------------------------------------------------

def test_container_tag_template_and_env_become_the_namespace(harness):
    harness.write_config(container_tag="team-{identity}")
    provider = harness.provider(agent_identity="coder")
    harness.tool(provider, "supermemory_search", {"query": "q"})
    assert harness.wire.requests[-1]["path"] == "/ns/team_coder/search"  # same sanitized tag as before v5

    harness.secrets["SUPERMEMORY_CONTAINER_TAG"] = "from-env"
    provider = harness.provider()
    harness.tool(provider, "supermemory_search", {"query": "q"})
    assert harness.wire.requests[-1]["path"] == "/ns/from_env/search"


def test_custom_containers_are_whitelisted_and_ignored_when_disabled(harness):
    provider = harness.provider()
    for name, args in [("supermemory_store", {"content": "x"}), ("supermemory_search", {"query": "q"}),
                       ("supermemory_profile", {}), ("supermemory_forget", {"id": "m1"})]:
        assert "error" not in harness.tool(provider, name, {**args, "container_tag": "project_alpha"})
    assert {r["path"].split("/")[2] for r in harness.wire.requests} == {"hermes"}

    harness.write_config(enable_custom_container_tags=True, custom_containers=["project_alpha"])
    provider = harness.provider()
    count = len(harness.wire.requests)
    result = harness.tool(provider, "supermemory_search", {"query": "q", "container_tag": "project_alpha"})
    assert result["container_tag"] == "project_alpha"
    assert harness.wire.requests[-1]["path"] == "/ns/project_alpha/search"
    assert "error" in harness.tool(provider, "supermemory_search", {"query": "q", "container_tag": "project_beta"})
    assert len(harness.wire.requests) == count + 1
    provider.sync_turn("auto capture", "stays primary")
    assert harness.wire.requests[-1]["path"] == CAPTURE_PATH


# ---- search ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode,wire_mode", [("hybrid", "hybrid"), ("memories", "memories"), ("chunks", "chunks"), ("documents", "chunks")])
def test_search_modes_and_pre_v5_defaults_on_the_wire(harness, mode, wire_mode):
    harness.write_config(search_mode=mode)
    provider = harness.provider()
    result = harness.tool(provider, "supermemory-search", {"query": "tea", "limit": 7})
    request = harness.wire.requests[-1]
    assert (request["method"], request["path"]) == ("POST", "/ns/hermes/search")
    assert request["body"] == {"query": "tea", "limit": 7, "searchMode": wire_mode,
                               "threshold": 0.6, "rerank": "none", "rewriteQuery": False}
    assert result == {"count": 2, "results": [{"id": "m1", "content": "Synthetic memory", "similarity": 90},
                                              {"id": "c1", "content": "Synthetic chunk", "similarity": 90}]}


@pytest.mark.parametrize("fields,text", [
    ({"memory": "memory text", "chunk": "chunk text"}, "memory text"),
    ({"memory": "  ", "chunk": "chunk text"}, "chunk text"),
    ({"memory": None, "chunk": None, "chunks": [{"content": "first"}, {"content": " "}, {"content": "second"}]}, "first\nsecond"),
    ({}, ""),
])
def test_hit_text_falls_back_from_memory_to_chunk_text(harness, fields, text):
    harness.wire.overrides["search"] = {"results": [search_hit("h1", **fields)], "searchTime": 1.0}
    provider = harness.provider()
    assert harness.tool(provider, "supermemory_search", {"query": "q"})["results"][0]["content"] == text
    hit = provider._client.search_memories("q")[0]
    assert hit["updated_at"] == STAMP and hit["metadata"] == {"fixture": True}


# ---- profile / recall ----------------------------------------------------------------------------

def test_profile_without_query_is_one_request(harness):
    provider = harness.provider()
    result = harness.tool(provider, "supermemory_profile", {})
    assert harness.wire.ops() == ["profile"] and harness.wire.requests[0]["body"] == {}
    assert result == {"profile": "## User Profile (Persistent)\n- Persistent fact\n\n## Recent Context\n- Recent fact",
                      "static_count": 1, "dynamic_count": 1}


def test_profile_with_query_adds_a_separate_search(harness):
    provider = harness.provider()
    result = harness.tool(provider, "supermemory_profile", {"query": "tea"})
    assert sorted(harness.wire.ops()) == ["profile", "search"]
    assert next(r["body"] for r in harness.wire.requests if r["op"] == "profile") == {}
    assert next(r["body"]["query"] for r in harness.wire.requests if r["op"] == "search") == "tea"
    assert "ProfileFact" not in result["profile"] and "- Persistent fact" in result["profile"]
    assert [r["content"] for r in result["results"]] == ["Synthetic memory", "Synthetic chunk"]
    assert "errors" not in result


def test_prefetch_renders_profile_and_chunk_hits_deduped(harness):
    harness.wire.overrides["search"] = {"results": [search_hit("m1", memory="Persistent fact"), search_hit("c1", chunk="Synthetic chunk")],
                                        "searchTime": 1.0}
    harness.write_config(max_recall_results=4)
    provider = harness.provider()
    text = provider.prefetch("x" * 300)
    assert text.startswith("<supermemory-context>") and text.endswith("</supermemory-context>")
    assert text.count("Persistent fact") == 1 and "Recent fact" in text and "Synthetic chunk" in text
    search = next(r["body"] for r in harness.wire.requests if r["op"] == "search")
    assert len(search["query"]) == 200 and search["limit"] == 4

    provider.on_turn_start(2, "next")
    text = provider.prefetch("next")
    assert "User Profile (Persistent)" not in text and "Relevant Memories" in text


@pytest.mark.parametrize("failed,kept", [("profile", "Synthetic memory"), ("search", "Persistent fact")])
@pytest.mark.parametrize("failure", [401, 503, httpx.ReadTimeout("synthetic timeout")])
def test_one_failed_recall_leg_keeps_the_other(harness, failed, kept, failure):
    harness.wire.overrides[failed] = failure
    provider = harness.provider()
    assert kept in provider.prefetch("query")
    assert len(harness.wire.requests) == 2  # one request per leg, no SDK retries
    result = harness.tool(provider, "supermemory_profile", {"query": "query"})
    assert set(result["errors"]) == {failed}


def test_both_recall_legs_failing_is_an_error(harness):
    harness.wire.overrides.update(profile=503, search=503)
    provider = harness.provider()
    assert provider.prefetch("query") == ""
    assert "Profile failed" in harness.tool(provider, "supermemory_profile", {"query": "q"})["error"]
    assert not harness.mod._probe_supermemory_connection(API_KEY, str(harness.home))["ok"]


def test_slow_search_leg_is_bounded_by_the_timeout(harness):
    harness.write_config(api_timeout=0.5)
    harness.wire.delays["search"] = 1.5  # MockTransport ignores httpx timeouts, so only the provider's bound applies
    provider = harness.provider()
    started = time.monotonic()
    text = provider.prefetch("query")
    assert time.monotonic() - started < 1.2
    assert "Persistent fact" in text and "Synthetic memory" not in text


# ---- forget ----------------------------------------------------------------------------------------

def test_forget_by_exact_id(harness):
    provider = harness.provider()
    assert harness.tool(provider, "supermemory-forget", {"id": "m1"}) == {"forgotten": True, "id": "m1"}
    request = harness.wire.requests[-1]
    assert (request["method"], request["path"], request["body"]) == ("DELETE", "/ns/hermes/memories", {"ids": ["m1"]})


@pytest.mark.parametrize("response", [
    {"count": 0, "errors": [], "matches": []},
    {"count": 0, "errors": [{"id": "m1", "error": "not found"}], "matches": []},
    {"count": 1, "errors": [], "matches": [{"id": "other", "memory": "x"}]},
    {"count": 2, "errors": [], "matches": [{"id": "m1", "memory": "x"}, {"id": "m2", "memory": "y"}]},
])
def test_forget_never_claims_success_without_an_exact_match(harness, response):
    harness.wire.overrides["memories"] = response
    result = harness.tool(harness.provider(), "supermemory_forget", {"id": "m1"})
    assert result["error"].startswith("Forget failed: Supermemory did not confirm forgetting m1")


def test_forget_by_query_searches_memories_only(harness):
    harness.write_config(search_mode="hybrid")
    provider = harness.provider()
    result = harness.tool(provider, "supermemory_forget", {"query": "synthetic"})
    assert result == {"success": True, "message": 'Forgot: "Synthetic memory"', "id": "m1"}
    search, forget = harness.wire.requests[-2:]
    assert search["body"]["searchMode"] == "memories"
    assert forget["body"] == {"ids": ["m1"]}

    harness.wire.overrides["search"] = {"results": [], "searchTime": 1.0}
    assert harness.tool(provider, "supermemory_forget", {"query": "nothing"})["success"] is False


# ---- endpoint, auth, status ------------------------------------------------------------------------

def test_every_request_uses_the_configured_endpoint_and_key(harness):
    provider = harness.provider()
    provider.sync_turn("u", "a")
    provider.prefetch("q")
    harness.tool(provider, "supermemory_forget", {"id": "m1"})
    assert {r["host"] for r in harness.wire.requests} == {"supermemory.test"}
    assert all(r["url"].startswith(BASE_URL + "/ns/") for r in harness.wire.requests)
    assert {r["headers"]["authorization"] for r in harness.wire.requests} == {f"Bearer {API_KEY}"}
    assert {r["headers"]["x-sm-source"] for r in harness.wire.requests} == {"hermes"}


def test_base_url_falls_back_to_the_scoped_env_then_hosted_default(harness):
    harness.write_config(base_url="")
    harness.secrets["SUPERMEMORY_BASE_URL"] = "http://env-local.test:6767/"
    harness.tool(harness.provider(), "supermemory_search", {"query": "q"})
    del harness.secrets["SUPERMEMORY_BASE_URL"]
    harness.tool(harness.provider(), "supermemory_search", {"query": "q"})
    assert [r["host"] for r in harness.wire.requests] == ["env-local.test", "api.supermemory.ai"]


def test_unreachable_local_endpoint_fails_closed(harness):
    harness.wire.overrides.update({op: httpx.ConnectError("synthetic refused") for op in ("document", "search", "profile")})
    provider = harness.provider()
    provider.sync_turn("u", "a")
    assert provider.prefetch("q") == ""
    assert "error" in harness.tool(provider, "supermemory_search", {"query": "q"})
    assert not harness.mod._probe_supermemory_connection(API_KEY, str(harness.home))["ok"]
    assert {r["host"] for r in harness.wire.requests} == {"supermemory.test"}  # never retried against the hosted API
    assert len(provider._pending_turns) == 1


def test_status_probe_counts_profile_facts(harness):
    status = harness.mod._probe_supermemory_connection(API_KEY, str(harness.home))
    assert status["ok"] and status["profile_facts"] == 2 and status["container_tag"] == "hermes"
    assert harness.wire.ops() == ["profile"]


def test_capture_bucket_id_is_unchanged(harness):
    custom_id = harness.mod._capture_custom_id
    assert custom_id("session-a", datetime(2026, 1, 1, 3, 59, tzinfo=timezone.utc)) == "session_a_2026-01-01_b0"
    assert custom_id("session-a", datetime(2026, 1, 1, 4, 0, tzinfo=timezone.utc)) == "session_a_2026-01-01_b1"
