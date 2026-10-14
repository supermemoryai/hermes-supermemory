"""Offline harness: the real provider module and the real installed supermemory SDK (its request
serializers and response models), with HTTP answered by an ``httpx.MockTransport``.

Only the Hermes host modules the provider imports are stubbed (mirroring their public signatures),
so the tests run without a hermes-agent checkout. Sockets are blocked: nothing reaches a network.
"""

from __future__ import annotations

import contextvars
import importlib.util
import json
import socket
import sys
import threading
import time
import types
from pathlib import Path

import httpx
import pytest
import supermemory

ROOT = Path(__file__).resolve().parents[1]
API_KEY = "sk-synthetic-test-key"
BASE_URL = "http://supermemory.test:6767"
STAMP = "2026-01-01T03:00:00Z"


def search_hit(hit_id: str = "m1", **fields) -> dict:
    """One v5 SearchResult as the API serializes it (camelCase)."""
    return {"id": hit_id, "metadata": {"fixture": True}, "similarity": 0.9, "isLatest": True, "isInference": False,
            "system": {"createdAt": STAMP, "updatedAt": STAMP}, **fields}


def default_response(op: str, body: dict) -> dict:
    if op == "document":
        return {"id": body.get("id", "doc-generated"), "status": "queued"}
    if op == "search":
        return {"results": [search_hit("m1", memory="Synthetic memory"), search_hit("c1", chunk="Synthetic chunk")],
                "searchTime": 1.0}
    if op == "profile":
        return {"profile": {"static": [{"id": "p1", "memory": "Persistent fact"}],
                            "dynamic": [{"id": "p2", "memory": "Recent fact"}], "buckets": {}}}
    if op == "memories":
        return {"count": len(body["ids"]), "errors": [], "matches": [{"id": i, "memory": "Synthetic memory"} for i in body["ids"]]}
    raise AssertionError(f"unexpected operation {op!r}")


class Wire:
    """Records every request; ``overrides[op]`` is a JSON body, an HTTP status, an exception, or a callable."""

    def __init__(self):
        self.requests: list[dict] = []
        self.overrides: dict = {}
        self.delays: dict[str, float] = {}
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        op = request.url.path.rsplit("/", 1)[-1]
        with self._lock:
            self.requests.append({"method": request.method, "url": str(request.url), "host": request.url.host,
                                  "path": request.url.path, "op": op, "body": body, "headers": dict(request.headers)})
        if op in self.delays:
            time.sleep(self.delays[op])
        value = self.overrides.get(op)
        if callable(value) and not isinstance(value, type):
            value = value(body)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, int):
            return httpx.Response(value, json={"error": "synthetic failure"})
        return httpx.Response(200, json=value if value is not None else default_response(op, body))

    def ops(self) -> list[str]:
        return [r["op"] for r in self.requests]


def _install_host_stubs(monkeypatch, secrets: dict, home: Path) -> None:
    provider_mod = types.ModuleType("agent.memory_provider")

    class MemoryProvider:  # the real base is an ABC with default no-op hooks
        pass

    def spawn_context_thread(target, *, name, daemon=True, args=(), kwargs=None):
        ctx = contextvars.copy_context()
        return threading.Thread(target=lambda *a, **k: ctx.run(target, *a, **k), args=args, kwargs=kwargs,
                                name=name, daemon=daemon)

    provider_mod.MemoryProvider, provider_mod.spawn_context_thread = MemoryProvider, spawn_context_thread
    secret_mod = types.ModuleType("agent.secret_scope")
    secret_mod.get_secret = lambda key, default="": secrets.get(key, default)
    secret_mod.is_multiplex_active = lambda: False
    registry_mod = types.ModuleType("tools.registry")
    registry_mod.tool_error = lambda message, **extra: json.dumps({"error": str(message), **extra})
    constants_mod = types.ModuleType("hermes_constants")
    constants_mod.get_hermes_home = lambda: home
    utils_mod = types.ModuleType("utils")
    utils_mod.atomic_json_write = lambda path, data, **kw: Path(path).write_text(json.dumps(data))
    for name, mod in {"agent": types.ModuleType("agent"), "agent.memory_provider": provider_mod,
                      "agent.secret_scope": secret_mod, "tools": types.ModuleType("tools"),
                      "tools.registry": registry_mod, "hermes_constants": constants_mod, "utils": utils_mod}.items():
        monkeypatch.setitem(sys.modules, name, mod)


class Harness:
    def __init__(self, module, wire: Wire, secrets: dict, home: Path):
        self.mod, self.wire, self.secrets, self.home = module, wire, secrets, home

    def write_config(self, **config) -> None:
        (self.home / "supermemory.json").write_text(json.dumps({"base_url": BASE_URL, "api_timeout": 1.0, **config}))

    def provider(self, session_id: str = "session-a", **init_kwargs):
        provider = self.mod.SupermemoryMemoryProvider()
        provider.initialize(session_id, hermes_home=str(self.home), **init_kwargs)
        return provider

    @staticmethod
    def tool(provider, name: str, args: dict) -> dict:
        return json.loads(provider.handle_tool_call(name, args))


@pytest.fixture
def harness(monkeypatch, tmp_path):
    def no_network(*_args, **_kwargs):
        raise AssertionError("network access is forbidden in these tests")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)

    secrets = {"SUPERMEMORY_API_KEY": API_KEY}
    _install_host_stubs(monkeypatch, secrets, tmp_path)
    spec = importlib.util.spec_from_file_location("hermes_supermemory_under_test", ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    wire, real_client, clients = Wire(), supermemory.Supermemory, []

    def client_factory(**kwargs):  # the real SDK client, with only its HTTP transport replaced
        client = real_client(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(wire)))
        clients.append(client)
        return client
    monkeypatch.setattr(supermemory, "Supermemory", client_factory)

    h = Harness(module, wire, secrets, tmp_path)
    h.write_config()
    yield h
    for client in clients:
        client._http_client.close()
