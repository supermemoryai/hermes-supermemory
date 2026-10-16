> The [Supermemory](https://supermemory.ai) memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent), maintained by Supermemory. It was previously bundled inside Hermes core. See [MAINTAINERS.md](MAINTAINERS.md).

# Supermemory Memory Provider

Semantic long-term memory on Supermemory's v5 namespace API: profile recall, semantic search, explicit memory tools, and per-turn conversation capture into one document per conversation.

## Requirements

- The official SDK, `supermemory>=5.0.0,<6` (declared in `pyproject.toml`; Hermes installs it with the plugin)
- Hosted: API key from [console.supermemory.ai/keys](https://console.supermemory.ai/keys?create=true)
- Self-hosted: a running [Supermemory local](https://supermemory.ai/docs/self-hosting/overview) server and the API key it prints on first boot

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
| `container_tag` | `hermes` | The Supermemory namespace used for search and writes. Supports `{identity}` template for profile-scoped tags (e.g. `hermes-{identity}` → `hermes-coder`). |
| `auto_recall` | `true` | Inject relevant memory context before turns |
| `auto_capture` | `true` | Store cleaned user-assistant turns after each response |
| `max_recall_results` | `10` | Max search results recalled with the profile (1-20) |
| `capture_mode` | `all` | Skip tiny or trivial turns by default |
| `search_mode` | `hybrid` | Search mode: `hybrid`, `memories` (memories only), `chunks` (document excerpts only; `documents` is accepted as before) |
| `entity_context` | built-in default | Extraction guidance, sent as the namespace's `supportingContext` with every write |
| `dreaming` | `dynamic` | Supermemory's processing mode, sent with every write: `dynamic` groups related documents so memories form from coherent context; `instant` processes each document right away (billed as one extra operation per document) |
| `api_timeout` | `5.0` | Timeout for SDK requests |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SUPERMEMORY_API_KEY` | API key (required) |
| `SUPERMEMORY_BASE_URL` | Compatibility fallback for the API endpoint when `base_url` is not configured |
| `SUPERMEMORY_CONTAINER_TAG` | Override the container tag / namespace (takes priority over config file) |

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
| `supermemory-profile` | `supermemory_profile` | Retrieve the user's profile (markdown) |
| `supermemory-read-document` | `supermemory_read_document` | Read a stored document (e.g. a past conversation) by the `document_id` from search results, paged by `offset` |

## Source attribution

All Supermemory API calls send `x-sm-source: hermes`, and document writes stamp
`metadata.sm_source: hermes`. This is a **functional routing key, not telemetry**:
it groups Hermes-written memories into a dedicated "Hermes" Space in the
Supermemory app, so you can filter, browse, and bulk-manage them per source agent
(alongside Codex, Claude Code, etc.) from the Supermemory UI.

## Behavior

When enabled, Hermes can:

- before each turn, recall the full profile and the top `max_recall_results` search results for the message (fetched in parallel, each bounded by `api_timeout`; if one fails the other is still used)
- after each completed turn, write the conversation so far as **one document per conversation** (id `hermes_<session id>`); the whole document is re-sent under the same id, so it is replaced, never duplicated, and stays current while a long session runs
- skip a write whose content has not changed, and retry a failed one on the next turn, session end, `/reset`, or shutdown; a refused write is logged and never breaks the turn
- store messages as given (only the plugin's own recalled `<supermemory-context>` block and inline base64 images are left out)
- copy built-in `memory` tool additions into Supermemory
- route every SDK and probe request through the configured hosted or self-hosted endpoint
- expose explicit tools for search, store, forget, profile, and reading documents


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
- every tool accepts an optional `container_tag` parameter (each container is its own namespace)
- The tag must be in the whitelist: primary container + `custom_containers`
- Automatic operations (turn capture, prefetch, memory write mirroring) always use the **primary** container only
- Custom container instructions are injected into the system prompt

## Upgrading from 1.x

2.0 moves the plugin from the 3.x SDK (container tags) to the 5.x SDK and Supermemory's v5 namespace API.

- **Data:** reads and writes go to the namespace named by `container_tag` (default `hermes`). Memories written by 1.x
  under container tags are not migrated by the plugin.
- **Capture:** one document per conversation, replaced after every turn, instead of one appended document per session per
  4-hour window.
- **Recall:** the full markdown profile plus the top search results on every turn. `profile_frequency` is gone.
- **Tools:** new `supermemory-read-document` / `supermemory_read_document`. `supermemory_profile` no longer takes a
  `query`, and search results carry a `document_id`.
- **Settings:** `search_mode: documents` is now `chunks` (the old value still works); new `dreaming` (default `dynamic`, or `instant`). Everything else in
  `supermemory.json` and the environment is read as before.

## Support

- [Supermemory Discord](https://supermemory.link/discord)
- [support@supermemory.com](mailto:support@supermemory.com)
- [supermemory.ai](https://supermemory.ai)
