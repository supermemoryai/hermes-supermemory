"""OAuth sign-in contracts: the login the Supermemory CLI spawns, refresh, and logout."""

import base64
import hashlib
import json
import socket
import stat
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlsplit

import pytest
from conftest import ISSUER, auth, save_sign_in


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def callback_status(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def test_print_url_login_prints_one_link_and_finishes_when_the_code_arrives(issuer, monkeypatch, capsys, credentials_dir):
    port = free_port()
    monkeypatch.setattr(auth, "AUTH_PORT", port)
    monkeypatch.setattr(auth, "REDIRECT_URI", f"http://127.0.0.1:{port}/callback")
    result = {}
    thread = threading.Thread(target=lambda: result.update(saved=auth.login(auth.oauth_config(""), print_url=True, timeout=10)))
    thread.start()
    stdout = ""
    for _ in range(100):
        stdout += capsys.readouterr().out
        if stdout.endswith("\n"):
            break
        time.sleep(0.05)
    lines = stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith("SUPERMEMORY_AUTHORIZE_URL=")
    link = urlsplit(lines[0].removeprefix("SUPERMEMORY_AUTHORIZE_URL="))
    params = {k: v[0] for k, v in parse_qs(link.query).items()}
    assert f"{link.scheme}://{link.netloc}{link.path}" == f"{ISSUER}/oauth2/authorization"
    assert (params["client_id"], params["redirect_uri"], params["code_challenge_method"]) == (
        "client-1", f"http://127.0.0.1:{port}/callback", "S256")
    assert (params["scope"], params["resource"]) == ("openid profile email offline_access", "https://mcp.supermemory.ai/mcp")

    assert callback_status(f"{params['redirect_uri']}?code=stolen&state=wrong") == 400
    assert callback_status(f"{params['redirect_uri']}?code=c1&state={params['state']}&iss={ISSUER}") == 200
    thread.join(10)

    exchange = issuer.token_requests()[0]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(exchange["code_verifier"].encode()).digest()).rstrip(b"=").decode()
    assert (exchange["code"], exchange["grant_type"], challenge) == ("c1", "authorization_code", params["code_challenge"])
    saved = json.loads((credentials_dir / "credentials.json").read_text())
    assert result["saved"] == saved
    assert (saved["type"], saved["accessToken"], saved["refreshToken"]) == ("oauth", "access-1", "refresh-1")
    assert saved["expiresAt"] > time.time() * 1000 and saved["savedAt"].endswith("Z")
    assert stat.S_IMODE((credentials_dir / "credentials.json").stat().st_mode) == 0o600
    assert issuer.calls[1]["body"]["redirect_uris"] == [f"http://127.0.0.1:{port}/callback"]


def test_an_expiring_token_is_refreshed_and_the_file_rewritten(issuer):
    config = auth.oauth_config("")
    save_sign_in(expires_in_ms=30_000)
    assert auth.access_token(config) == "access-1"
    assert issuer.token_requests() == [{"client_id": "client-1", "resource": "https://mcp.supermemory.ai/mcp",
                                        "grant_type": "refresh_token", "refresh_token": "refresh-0"}]
    assert auth.load_credentials()["refreshToken"] == "refresh-1"
    assert auth.access_token(config) == "access-1"
    assert auth.access_token(config, rejected="access-1") == "access-2"


def test_a_rejected_refresh_token_asks_to_sign_in_again(issuer):
    config = auth.oauth_config("")
    save_sign_in(expires_in_ms=0)
    issuer.refresh_error = "invalid_grant"
    with pytest.raises(auth.AuthError, match="sign in again") as raised:
        auth.access_token(config)
    assert raised.value.code == "AUTH_REQUIRED"
    assert "refreshToken" not in auth.load_credentials() and not auth.has_oauth_credentials(config)


def test_logout_revokes_the_refresh_token_and_deletes_the_file(issuer, credentials_dir):
    save_sign_in()
    assert auth.logout()["refreshToken"] == "refresh-0"
    assert issuer.calls[-1]["form"] == {"client_id": "client-1", "token": "refresh-0", "token_type_hint": "refresh_token"}
    assert not (credentials_dir / "credentials.json").exists()
    assert auth.logout() is None


def test_oauth_is_for_supermemory_cloud_only(monkeypatch):
    with pytest.raises(auth.AuthError, match="SUPERMEMORY_API_KEY"):
        auth.oauth_config("http://localhost:6767")
    monkeypatch.setenv("SUPERMEMORY_MCP_URL", "https://mcp.example.com/mcp")
    assert auth.oauth_config("https://memory.example.com")["resource"] == "https://mcp.example.com/mcp"
