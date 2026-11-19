"""Supermemory OAuth sign-in for Hermes, stdlib only so any python3 runs it: auth.py login [--print-url] | logout | status."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets
import shlex
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

CREDENTIALS_DIR = Path.home() / ".supermemory-hermes"
AUTH_PORT = 19884
REDIRECT_URI = f"http://127.0.0.1:{AUTH_PORT}/callback"
OAUTH_SCOPE = "openid profile email offline_access"
DEFAULT_BASE_URL = "https://api.supermemory.ai"
LINK_PREFIX = "SUPERMEMORY_AUTHORIZE_URL="
REQUEST_TIMEOUT = 10.0
REFRESH_SKEW_MS = 60_000
_CLIENT_FIELDS = ("apiBaseUrl", "issuer", "resource", "clientId", "redirectUri",
                  "authorizationEndpoint", "tokenEndpoint", "revocationEndpoint")
_KNOWN_ERRORS = ("invalid_grant", "invalid_client", "access_denied")
_TOKEN_GONE_STATUSES = (400, 401)


def _page(heading: str, color: str, message: str) -> str:
    return (f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>{heading} - Supermemory</title></head>'
            '<body style="font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh;'
            f'margin:0;background:#0a0a0a;color:#fafafa"><div style="text-align:center"><h1 style="color:{color}">'
            f"{heading}</h1><p>{message}</p></div></body></html>")


_SUCCESS_HTML = _page("Connected!", "#22c55e", "You can close this window and return to Hermes.")
_DENIED_HTML = _page("Authorization denied", "#ef4444", "Supermemory was not connected. Run login again to retry.")
_FAILED_HTML = _page("Connection failed", "#ef4444", "Return to Hermes and run login again.")


class AuthError(Exception):
    def __init__(self, message: str, code: str = "oauth_error", status: Optional[int] = None):
        super().__init__(message)
        self.code, self.status = code, status


def _plugin_version() -> str:
    try:
        import tomllib
        return tomllib.loads(Path(__file__).with_name("pyproject.toml").read_text("utf-8"))["project"]["version"]
    except Exception:
        return "unknown"


PLUGIN_VERSION = _plugin_version()


def login_command() -> str:
    return f"python3 {shlex.quote(str(Path(__file__).resolve()))} login"


def auth_required() -> AuthError:
    return AuthError(f"Supermemory sign-in required. Run `{login_command()}` to sign in again, or set SUPERMEMORY_API_KEY.",
                     "AUTH_REQUIRED")


def _credentials_file() -> Path:
    return CREDENTIALS_DIR / "credentials.json"


def _client_file() -> Path:
    return CREDENTIALS_DIR / "oauth-client.json"


def _read_json(path: Path) -> Optional[dict]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, data: dict) -> None:
    CREDENTIALS_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _try_lock(handle) -> bool:
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


# The Hermes session and this CLI share the credentials; the OS drops the lock if its holder dies.
@contextmanager
def _auth_lock(timeout: float) -> Iterator[None]:
    CREDENTIALS_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    with open(CREDENTIALS_DIR / "auth.lock", "a+b") as handle:
        while not _try_lock(handle):
            if time.monotonic() >= deadline:
                raise AuthError("Supermemory authentication is busy. Try again.", "AUTH_BUSY")
            time.sleep(0.05)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _secure_url(value: Any) -> str:
    url = urllib.parse.urlsplit(str(value or ""))
    loopback = url.hostname in ("localhost", "127.0.0.1", "::1")
    if (not url.hostname or url.username or url.password or url.query or url.fragment
            or not (url.scheme == "https" or (url.scheme == "http" and loopback))):
        raise AuthError("OAuth requires HTTPS (HTTP is allowed only for loopback development).", "insecure_url")
    return str(value).rstrip("/")


def _issuer_endpoint(value: Any, issuer: str) -> str:
    endpoint = _secure_url(value)
    if urllib.parse.urlsplit(endpoint)[:2] != urllib.parse.urlsplit(issuer)[:2]:
        raise AuthError("OAuth endpoint does not belong to the configured issuer.")
    return endpoint


def oauth_config(base_url: str = "") -> Dict[str, str]:
    """Where to sign in for ``base_url``; AuthError for a custom backend without SUPERMEMORY_MCP_URL."""
    api = _secure_url(base_url or DEFAULT_BASE_URL)
    resource = os.environ.get("SUPERMEMORY_MCP_URL", "")
    if not resource:
        parts = urllib.parse.urlsplit(api)
        if not parts.netloc.startswith("api."):
            raise AuthError("OAuth sign-in is available for Supermemory cloud only. Set SUPERMEMORY_API_KEY for a custom "
                            "or self-hosted backend (or SUPERMEMORY_MCP_URL if it supports OAuth).", "unsupported_backend")
        resource = urllib.parse.urlunsplit((parts.scheme, "mcp." + parts.netloc[4:], "/mcp", "", ""))
    return {"apiBaseUrl": api, "issuer": _secure_url(os.environ.get("SUPERMEMORY_OAUTH_ISSUER") or f"{api}/api/auth"),
            "resource": _secure_url(resource)}


def _same_config(saved: Optional[dict], config: Dict[str, str]) -> bool:
    return bool(saved) and all(saved.get(key) == config[key] for key in ("apiBaseUrl", "issuer", "resource"))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _request(url: str, *, form: Optional[dict] = None, body: Optional[dict] = None, token: str = "") -> dict:
    headers = {"Accept": "application/json", "User-Agent": f"hermes-supermemory/{PLUGIN_VERSION}"}
    data = None
    if form is not None:
        data, headers["Content-Type"] = urllib.parse.urlencode(form).encode(), "application/x-www-form-urlencoded"
    elif body is not None:
        data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
    if token:
        headers.update({"Authorization": f"Bearer {token}", "x-sm-source": "hermes"})
    try:
        with _OPENER.open(urllib.request.Request(url, data=data, headers=headers), timeout=REQUEST_TIMEOUT) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    except (urllib.error.URLError, OSError) as error:
        raise AuthError(f"Could not reach Supermemory ({getattr(error, 'reason', error)}).", "network") from error
    try:
        payload = json.loads(raw or b"null")
    except ValueError:
        payload = None
    result = payload if isinstance(payload, dict) else {}
    if not 200 <= status < 300:
        code = result.get("error") if result.get("error") in _KNOWN_ERRORS else "oauth_request_failed"
        raise AuthError(f"Supermemory authorization failed ({code}, HTTP {status}).", code, status)
    return result


def _register_client(config: Dict[str, str]) -> dict:
    saved = _read_json(_client_file())
    if _same_config(saved, config) and saved.get("clientId") and saved.get("redirectUri") == REDIRECT_URI:
        return saved
    issuer = urllib.parse.urlsplit(config["issuer"])
    metadata = _request(f"{issuer.scheme}://{issuer.netloc}/.well-known/oauth-authorization-server{issuer.path.rstrip('/')}")
    if metadata.get("issuer") != config["issuer"]:
        raise AuthError("OAuth discovery returned an unexpected issuer.")
    endpoints = {key: _issuer_endpoint(metadata.get(field), config["issuer"]) for key, field in (
        ("authorizationEndpoint", "authorization_endpoint"), ("tokenEndpoint", "token_endpoint"),
        ("revocationEndpoint", "revocation_endpoint"), ("registrationEndpoint", "registration_endpoint"))}
    registration = _request(endpoints.pop("registrationEndpoint"), body={
        "client_name": "Supermemory for Hermes",
        "client_uri": "https://github.com/supermemoryai/hermes-supermemory",
        "software_id": "hermes-supermemory",
        "software_version": PLUGIN_VERSION,
        "redirect_uris": [REDIRECT_URI],
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "scope": OAUTH_SCOPE,
    })
    client_id = registration.get("client_id")
    if not isinstance(client_id, str) or not client_id or registration.get("token_endpoint_auth_method") != "none":
        raise AuthError("OAuth registration did not return a public client.")
    client = {**config, "clientId": client_id, "redirectUri": REDIRECT_URI, **endpoints}
    _write_json(_client_file(), client)
    return client


def _exchange(client: dict, parameters: Dict[str, str], previous: Optional[dict] = None) -> dict:
    data = _request(_issuer_endpoint(client["tokenEndpoint"], client["issuer"]),
                    form={"client_id": client["clientId"], "resource": client["resource"], **parameters})
    refresh_token = data.get("refresh_token") or (previous or {}).get("refreshToken")
    expires_in = data.get("expires_in")
    if (not isinstance(data.get("access_token"), str) or not data["access_token"]
            or str(data.get("token_type") or "").lower() != "bearer"
            or isinstance(expires_in, bool) or not isinstance(expires_in, (int, float)) or not 0 < expires_in < float("inf")
            or not isinstance(refresh_token, str) or not refresh_token):
        raise AuthError("OAuth returned an invalid token response.")
    credentials = {"type": "oauth", **{key: client[key] for key in _CLIENT_FIELDS},
                   "accessToken": data["access_token"], "refreshToken": refresh_token,
                   "expiresAt": int(time.time() * 1000 + expires_in * 1000),
                   "savedAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")}
    _write_json(_credentials_file(), credentials)
    return credentials


def load_credentials() -> Optional[dict]:
    saved = _read_json(_credentials_file())
    return saved if saved and saved.get("type") == "oauth" else None


def _fresh(saved: dict, rejected: str) -> bool:
    token = saved.get("accessToken")
    return bool(token) and token != rejected and (saved.get("expiresAt") or 0) > time.time() * 1000 + REFRESH_SKEW_MS


def has_oauth_credentials(config: Dict[str, str]) -> bool:
    saved = load_credentials()
    return _same_config(saved, config) and (bool(saved.get("refreshToken")) or _fresh(saved, ""))


def access_token(config: Dict[str, str], rejected: str = "") -> str:
    """A usable access token, refreshed (and the file rewritten) when it is near expiry or equals ``rejected``."""
    saved = load_credentials()
    if not _same_config(saved, config):
        raise auth_required()
    if _fresh(saved, rejected):
        return saved["accessToken"]
    with _auth_lock(REQUEST_TIMEOUT):
        saved = load_credentials()
        if not _same_config(saved, config):
            raise auth_required()
        if _fresh(saved, rejected):
            return saved["accessToken"]
        if not saved.get("refreshToken"):
            raise auth_required()
        try:
            return _exchange(saved, {"grant_type": "refresh_token", "refresh_token": saved["refreshToken"]}, saved)["accessToken"]
        except AuthError as error:
            if error.code not in ("invalid_grant", "invalid_client"):
                raise
            _write_json(_credentials_file(), {k: v for k, v in saved.items() if k not in ("accessToken", "refreshToken")})
            if error.code == "invalid_client":
                _client_file().unlink(missing_ok=True)
            raise auth_required() from error


def logout() -> Optional[dict]:
    """Revoke the refresh token, then delete the credentials; returns what was saved, or None."""
    with _auth_lock(REQUEST_TIMEOUT):
        saved = load_credentials()
        if saved and saved.get("refreshToken"):
            try:
                _request(_issuer_endpoint(saved["revocationEndpoint"], saved["issuer"]),
                         form={"client_id": saved["clientId"], "token": saved["refreshToken"], "token_type_hint": "refresh_token"})
            except AuthError as error:
                if error.status not in _TOKEN_GONE_STATUSES:
                    raise
        _credentials_file().unlink(missing_ok=True)
        return saved


# Each connection gets its own daemon thread, so an idle browser socket blocks neither the callback nor shutdown.
class _CallbackServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False


def _callback_handler(client: dict, state: str, verifier: str, outcome: dict, done: threading.Event):
    claim = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        timeout = REQUEST_TIMEOUT

        def log_message(self, *args: Any) -> None:
            pass

        def _reply(self, status: int, body: str = "", html: bool = False) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            for name, value in (("Cache-Control", "no-store"), ("Referrer-Policy", "no-referrer"),
                                ("Content-Type", f"text/{'html' if html else 'plain'}; charset=utf-8"),
                                ("Content-Length", str(len(payload)))):
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            url = urllib.parse.urlsplit(self.path)
            params = {key: values[0] for key, values in urllib.parse.parse_qs(url.query, keep_blank_values=True).items()}
            if url.path != "/callback":
                return self._reply(404)
            if (not hmac.compare_digest(params.get("state", "").encode("utf-8"), state.encode("utf-8"))
                    or ("iss" in params and params["iss"] != client["issuer"])):
                return self._reply(400, "Invalid authorization state.")
            if not claim.acquire(blocking=False):
                return self._reply(409)
            try:
                if "error" in params:
                    raise AuthError("Authorization was denied.", "access_denied")
                if not params.get("code"):
                    raise AuthError("Authorization code missing.")
                outcome["credentials"] = _exchange(client, {"grant_type": "authorization_code", "code": params["code"],
                                                            "code_verifier": verifier, "redirect_uri": REDIRECT_URI})
                self._reply(200, _SUCCESS_HTML, html=True)
            except Exception as error:
                code = getattr(error, "code", "")
                self._reply(400, _DENIED_HTML if code == "access_denied" else _FAILED_HTML, html=True)
                if code == "invalid_client":
                    _client_file().unlink(missing_ok=True)
                outcome["error"] = error
            finally:
                done.set()

    return Handler


def _auth_timeout() -> float:
    try:
        configured = float(os.environ.get("SUPERMEMORY_AUTH_TIMEOUT", ""))
    except ValueError:
        configured = 0
    return configured / 1000 if configured > 0 else 300.0


def _open_browser(url: str) -> bool:
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False


def login(config: Dict[str, str], *, print_url: bool = False, timeout: Optional[float] = None) -> dict:
    """Run the browser sign-in and save the credentials; ``print_url`` prints the link instead of opening a browser."""
    timeout = timeout or _auth_timeout()
    deadline = time.monotonic() + timeout
    with _auth_lock(timeout):
        client = _register_client(config)
        state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
        authorize_url = _issuer_endpoint(client["authorizationEndpoint"], client["issuer"]) + "?" + urllib.parse.urlencode({
            "client_id": client["clientId"], "redirect_uri": REDIRECT_URI, "response_type": "code", "scope": OAUTH_SCOPE,
            "resource": client["resource"], "code_challenge": challenge, "code_challenge_method": "S256", "state": state,
            "prompt": "consent"})
        outcome: dict = {}
        done = threading.Event()
        try:
            server = _CallbackServer(("127.0.0.1", AUTH_PORT), _callback_handler(client, state, verifier, outcome, done))
        except OSError as error:
            raise AuthError(f"Cannot open the sign-in callback port {AUTH_PORT}. Retry after other logins finish.",
                            "port_in_use") from error
        threading.Thread(target=server.serve_forever, name="supermemory-oauth-callback", daemon=True).start()
        try:
            if print_url:
                print(f"{LINK_PREFIX}{authorize_url}", flush=True)
                print(f"Open this link to sign in to Supermemory:\n{authorize_url}", file=sys.stderr, flush=True)
            else:
                opened = _open_browser(authorize_url)
                print(("Opening your browser to sign in to Supermemory. If it does not open, visit:" if opened
                       else "Open this link to sign in to Supermemory:") + f"\n{authorize_url}", flush=True)
            done.wait(max(0.0, deadline - time.monotonic()))
        finally:
            server.shutdown()
            server.server_close()
    if "credentials" in outcome:
        return outcome["credentials"]
    raise outcome.get("error") or AuthError("Sign-in timed out. Run login again.", "AUTH_TIMEOUT")


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()


def _dotenv(path: Path) -> Dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return {}
    values: Dict[str, str] = {}
    for line in lines:
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and key.strip() and not key.startswith("#"):
            values[key.strip()] = value.strip().strip("'\"")
    return values


def _setting(name: str) -> Tuple[str, str]:
    """(value, where it came from): the environment first, then $HERMES_HOME/.env."""
    if os.environ.get(name, "").strip():
        return os.environ[name].strip(), f"{name} environment variable"
    env_file = _hermes_home() / ".env"
    value = _dotenv(env_file).get(name, "").strip()
    return value, str(env_file) if value else ""


def _base_url() -> str:
    raw = str((_read_json(_hermes_home() / "supermemory.json") or {}).get("base_url") or "").strip()
    return (raw or _setting("SUPERMEMORY_BASE_URL")[0] or DEFAULT_BASE_URL).rstrip("/") or DEFAULT_BASE_URL


def mask_key(key: str) -> str:
    return f"{key[:3]}...{key[-4:]}" if len(key) > 12 else "set"


def _cmd_login(print_url: bool) -> int:
    out = sys.stderr if print_url else sys.stdout
    try:
        login(oauth_config(_base_url()), print_url=print_url)
    except AuthError as error:
        print(f"✗ Sign-in failed: {error}", file=sys.stderr)
        return 1
    print("✓ Signed in to Supermemory. New Hermes sessions use this sign-in.", file=out)
    api_key, source = _setting("SUPERMEMORY_API_KEY")
    if api_key:
        print(f"SUPERMEMORY_API_KEY ({source}) still takes precedence. Remove it to use this sign-in.", file=out)
    return 0


def _cmd_logout() -> int:
    try:
        saved = logout()
    except AuthError as error:
        print(f"✗ Logout failed: {error}\nSaved credentials were kept. Try again when the network is available.", file=sys.stderr)
        return 1
    print("✓ Signed out. The refresh token was revoked and the saved sign-in deleted." if saved else "No saved sign-in found.")
    api_key, source = _setting("SUPERMEMORY_API_KEY")
    if api_key:
        print(f"SUPERMEMORY_API_KEY ({source}) is still set, so Hermes keeps using that API key.")
    return 0


def _cmd_status() -> int:
    base_url = _base_url()
    api_key, source = _setting("SUPERMEMORY_API_KEY")
    print(f"Supermemory for Hermes\nAPI URL: {base_url}")
    token = api_key
    if api_key:
        print(f"Auth: API key {mask_key(api_key)} ({source})")
    else:
        saved, problem = load_credentials(), ""
        try:
            config = oauth_config(base_url)
            token = access_token(config) if _same_config(saved, config) else ""
        except AuthError as error:
            problem = str(error)
        if not token:
            print("Auth: not configured")
            print(problem or (("Saved sign-in is for a different API URL. " if saved else "")
                              + f"Run `{login_command()}` to sign in, or set SUPERMEMORY_API_KEY."))
            return 0
        print(f"Auth: OAuth (signed in {saved.get('savedAt') or 'earlier'}, {_credentials_file()})")
    try:
        session = _request(f"{base_url}/session", token=token)
    except AuthError as error:
        print(f"Connected: no ({error})")
        return 0
    user, org = session.get("user") or {}, session.get("org") or {}
    print("Connected: yes")
    for label, value in (("Email", user.get("email")), ("Name", user.get("name")), ("Organization", org.get("name"))):
        if isinstance(value, str) and value:
            print(f"{label}: {value}")
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="auth.py", description="Sign the Hermes Supermemory plugin in or out.")
    commands = parser.add_subparsers(dest="command", required=True)
    login_parser = commands.add_parser("login", help="Sign in with your Supermemory account (opens a browser)")
    login_parser.add_argument("--print-url", action="store_true",
                              help=f"Print one {LINK_PREFIX}<url> line instead of opening a browser")
    commands.add_parser("logout", help="Revoke and delete the saved sign-in")
    commands.add_parser("status", help="Show how Hermes authenticates to Supermemory")
    args = parser.parse_args(argv)
    try:
        if args.command == "login":
            return _cmd_login(args.print_url)
        return _cmd_logout() if args.command == "logout" else _cmd_status()
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
