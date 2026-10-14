> The [Supermemory](https://supermemory.ai) memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent), maintained by Supermemory. It was previously bundled inside Hermes core. See [MAINTAINERS.md](MAINTAINERS.md).

# Supermemory Memory Provider

Semantic long-term memory with profile recall, semantic search, explicit memory tools, and per-turn conversation capture into one document per session per 4-hour window for richer profiles.

## Requirements

- `supermemory>=5,<6` (the Python SDK for the v5 API; `hermes plugins install` installs it from `pyproject.toml`)
- Hosted: API key from [console.supermemory.ai/keys](https://console.supermemory.ai/keys?create=true)
- Self-hosted: a running [Supermemory local](https://supermemory.ai/docs/self-hosting/overview) server that serves the v5 API (`/ns/{namespace}/...`) and the API key it prints on first boot

## Setup

```bash
hermes plugins install supermemoryai/hermes-supermemory
hermes plugins enable supermemory
hermes memory setup    # select "supermemory"
```

Or manually:

```bash
hermes config set memory.provider supermemory
echo 'SUPERMEMORY_API_KEY=***' >> ~/.hermes/.env
```

For a fully self-hosted setup, start Supermemory local and note the API key it
prints on first boot:

```bash
npx supermemory local
```

Before running `hermes memory setup`, add the local endpoint to
`$HERMES_HOME/supermemory.json`:

```json
{
  "base_url": "http://localhost:6767"
}
```

Then run `hermes memory setup` and enter the local server's API key. Configuring
the endpoint first ensures the setup connection probe also stays local.

## Config

Config file: `$HERMES_HOME/supermemory.json`

| Key | Default | Description |
|-----|---------|-------------|
| `base_url` | `https://api.supermemory.ai` | API endpoint for hosted or self-hosted Supermemory. Takes priority over `SUPERMEMORY_BASE_URL`. |
| `container_tag` | `hermes` | Container tag used for search and writes. Supports `{identity}` template for profile-scoped tags (e.g. `hermes-{identity}` → `hermes-coder`). |
| `auto_recall` | `true` | Inject relevant memory context before turns |
| `auto_capture` | `true` | Store cleaned user-assistant turns after each response |
| `max_recall_results` | `10` | Max recalled items to format into context |
| `profile_frequency` | `50` | Include profile facts on first turn and every N turns |
| `capture_mode` | `all` | Skip tiny or trivial turns by default |
| `search_mode` | `hybrid` | Search mode: `hybrid` (memories + source chunks), `memories` (memories only), `chunks` (source chunks only; `documents` is accepted as the pre-v5 name) |
| `entity_context` | built-in default | Extraction guidance passed to Supermemory |
| `api_timeout` | `5.0` | Timeout for SDK requests |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SUPERMEMORY_API_KEY` | API key (required) |
| `SUPERMEMORY_BASE_URL` | Compatibility fallback for the API endpoint when `base_url` is not configured |
| `SUPERMEMORY_CONTAINER_TAG` | Override container tag (takes priority over config file) |

Base URL precedence is `supermemory.json` → `SUPERMEMORY_BASE_URL` →
`https://api.supermemory.ai`. Hermes resolves it once and uses the same endpoint
for SDK operations and setup/status probes.

## Tools

Kebab-case names are registered for the agent; snake_case aliases remain supported.

| Tool | Alias | Description |
|------|-------|-------------|
| `supermemory-save` | `supermemory_store` | Store an explicit memory |
| `supermemory-search` | `supermemory_search` | Search memories by semantic similarity |
| `supermemory-forget` | `supermemory_forget` | Forget a memory by ID or best-match query |
| `supermemory-profile` | `supermemory_profile` | Retrieve persistent profile and recent context |

## Source attribution

All Supermemory API calls send `x-sm-source: hermes`, and document writes stamp
`metadata.sm_source: hermes`. This is a **functional routing key, not telemetry**:
it groups Hermes-written memories into a dedicated "Hermes" Space in the
Supermemory app, so you can filter, browse, and bulk-manage them per source agent
(alongside Codex, Claude Code, etc.) from the Supermemory UI.

## Behavior

When enabled, Hermes can:

- prefetch relevant memory context before each turn
- write each completed user/assistant turn to **one document per session per 4-hour window** (document `id` = `<session>_<date>_b<0-5>`; a v5 `POST` with an existing `id` appends the delta rather than replacing the document), matching the capture shape of the other Supermemory agent integrations
- retry failed turn writes on the next turn, session end, `/reset`, or shutdown. Delivery is **at-least-once**, not exactly-once: SDK-level retries are disabled so one hook sends each batch once, but if the API accepted a write and the response was lost (timeout, dropped connection), the provider's retry appends the same turns again. The retry buffer is in memory only (lost when the process exits) and capped at 50 turns / 256 KiB, dropping the oldest turns first
- route every SDK and probe request through the configured hosted or self-hosted endpoint
- expose explicit tools for search, store, forget, and profile access


## Upgrading from 1.x (SDK 3)

2.0 talks to the namespace-first v5 API through `supermemory>=5,<6`. Tools, config keys and environment
variables are unchanged. What to know:

- **No data migration.** The configured container tag (after the same sanitization as before) is used as the
  v5 namespace, so existing memories stay addressable.
- **Self-hosted servers** must serve the v5 API. An older local server will reject the requests; the
  configured `base_url` is still used for everything, never silently replaced by the hosted API.
- **Search keeps its pre-v5 behaviour.** v5 lowered the default similarity threshold to 0.3; the provider
  sends `threshold: 0.6`, `rerank: "none"` and `rewriteQuery: false` explicitly. Hybrid hits that carry
  their text in `chunk` (source passages) are now rendered instead of showing up empty.
- **Profile + query.** The v5 profile endpoint has no query parameter, so a query (automatic recall or
  `supermemory-profile` with `query`) fetches the profile and runs a separate search concurrently, bounded by
  `api_timeout`. If one of the two fails, the other is still used and the tool reports the failure under `errors`.
- **Forget is confirmed.** Forgetting by id succeeds only when the API reports that exact id as forgotten;
  forgetting by query searches memories only (chunk ids are not memory ids).
- If another component pins an older SDK into the Hermes venv, the provider stays inactive and
  `hermes memory status` reports the installed version instead of failing on every request.

## Profile-Scoped Containers

Use `{identity}` in the `container_tag` to scope memories per Hermes profile:

```json
{
  "container_tag": "hermes-{identity}"
}
```

For a profile named `coder`, this resolves to `hermes-coder`. The default profile resolves to `hermes-default`. Without `{identity}`, all profiles share the same container.

## Multi-Container Mode

For advanced setups (e.g. OpenClaw-style multi-workspace), you can enable custom container tags so the agent can read/write across multiple named containers:

```json
{
  "container_tag": "hermes",
  "enable_custom_container_tags": true,
  "custom_containers": ["project-alpha", "project-beta", "shared-knowledge"],
  "custom_container_instructions": "Use project-alpha for coding tasks, project-beta for research, and shared-knowledge for team-wide facts."
}
```

When enabled:
- `supermemory-search`, `supermemory-save`, `supermemory-forget`, and `supermemory-profile` accept an optional `container_tag` parameter
- The tag must be in the whitelist: primary container + `custom_containers`
- Automatic operations (turn capture, prefetch, memory write mirroring) always use the **primary** container only
- Custom container instructions are injected into the system prompt

## Development

The tests run the provider against the real installed SDK with an in-process mock HTTP transport
(no network, no Hermes checkout needed):

```bash
uv run --isolated --no-project --python 3.11 --with 'supermemory>=5.0.0,<6' --with pytest python -m pytest tests -q
```

## Support

- [Supermemory Discord](https://supermemory.link/discord)
- [support@supermemory.com](mailto:support@supermemory.com)
- [supermemory.ai](https://supermemory.ai)
