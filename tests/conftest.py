"""Load the plugin (the repository root is the package) as ``hermes_supermemory``.

Hermes Agent must be importable (``agent``, ``hermes_cli``, ``utils``), e.g. installed in the same venv.
"""

import importlib.util
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import pytest

_ROOT = Path(__file__).resolve().parents[1]

if "hermes_supermemory" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "hermes_supermemory", _ROOT / "__init__.py", submodule_search_locations=[str(_ROOT)])
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["hermes_supermemory"] = _module
    _spec.loader.exec_module(_module)

auth = sys.modules["hermes_supermemory"]._auth
ISSUER = "https://api.supermemory.ai/api/auth"


class FakeIssuer:
    """The authorization server behind ``auth._request``: records calls and issues numbered tokens."""

    def __init__(self):
        self.calls, self.issued, self.refresh_error = [], 0, ""

    def __call__(self, url, *, form=None, body=None, token=""):
        self.calls.append({"url": url, "form": form, "body": body})
        path = urlsplit(url).path
        if path == "/.well-known/oauth-authorization-server/api/auth":
            return {"issuer": ISSUER, **{f"{name}_endpoint": f"{ISSUER}/oauth2/{name}"
                                         for name in ("authorization", "token", "registration", "revocation")}}
        if path.endswith("/oauth2/registration"):
            return {"client_id": "client-1", "token_endpoint_auth_method": "none"}
        if path.endswith("/oauth2/revocation"):
            return {}
        if path.endswith("/oauth2/token"):
            if form["grant_type"] == "refresh_token" and self.refresh_error:
                raise auth.AuthError("refused", self.refresh_error)
            self.issued += 1
            return {"access_token": f"access-{self.issued}", "refresh_token": f"refresh-{self.issued}",
                    "token_type": "Bearer", "expires_in": 3600}
        raise AssertionError(f"unexpected request to {url}")

    def token_requests(self):
        return [c["form"] for c in self.calls if c["url"].endswith("/oauth2/token")]


@pytest.fixture(autouse=True)
def credentials_dir(tmp_path, monkeypatch):
    """Each test signs in to its own ~/.supermemory-hermes."""
    monkeypatch.setattr(auth, "CREDENTIALS_DIR", tmp_path / ".supermemory-hermes")
    return tmp_path / ".supermemory-hermes"


@pytest.fixture
def issuer(monkeypatch):
    fake = FakeIssuer()
    monkeypatch.setattr(auth, "_request", fake)
    return fake


def save_sign_in(expires_in_ms=3_600_000, **overrides):
    """Credentials as a completed sign-in leaves them, valid for ``expires_in_ms``."""
    auth._write_json(auth._credentials_file(), {
        "type": "oauth", **auth.oauth_config(""), "clientId": "client-1", "redirectUri": auth.REDIRECT_URI,
        "authorizationEndpoint": f"{ISSUER}/oauth2/authorization", "tokenEndpoint": f"{ISSUER}/oauth2/token",
        "revocationEndpoint": f"{ISSUER}/oauth2/revocation", "accessToken": "access-0", "refreshToken": "refresh-0",
        "expiresAt": int(time.time() * 1000) + expires_in_ms, "savedAt": "2026-11-01T00:00:00.000Z", **overrides})
