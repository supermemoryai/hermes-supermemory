> The [Supermemory](https://supermemory.ai) memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent), maintained by Supermemory. It was previously bundled inside Hermes core. See [MAINTAINERS.md](MAINTAINERS.md).

# Supermemory Memory Provider

Semantic long-term memory with profile recall, semantic search, explicit memory tools, and per-turn conversation capture into one document per conversation.

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
| `container_tag` | `hermes` | Container tag (namespace) used for search and writes. Supports `{identity}` template for profile-scoped tags (e.g. `hermes-{identity}` → `hermes-coder`). Letters, digits, `_`, `-` and `:` are kept; other characters become `_`. Tags over 100 characters are shortened to 100, ending in `_` and an 8-character hash of the full tag so they stay distinct. |
| `auto_recall` | `true` | Inject relevant memory context before turns |
| `auto_capture` | `true` | Store cleaned user-assistant turns after each response |
| `max_recall_results` | `10` | Max search results recalled with the profile (1-20) |
| `profile_frequency` | `50` | Include profile facts on the first turn and every N turns |
| `capture_mode` | `all` | Skip tiny or trivial turns by default |
| `search_mode` | `hybrid` | Search mode: `hybrid`, `memories` (memories only), `chunks` (document excerpts only; `documents` is accepted as before) |
| `entity_context` | built-in default | Extraction guidance passed to Supermemory |
| `dreaming` | `dynamic` | `dynamic` groups related conversations before forming memories; `instant` processes each one right away (billed as one extra operation) |
| `api_timeout` | `5.0` | Timeout for SDK requests |

### Upgrading from 1.0.1

1.0.1 changed `-` and `:` in tags to `_`, so `hermes-coder` was saved as `hermes_coder`. From 1.0.2 tags are kept as written, so memories saved before the upgrade stay under the old name. The plugin logs the old and new name once.

To keep those memories, either set the old name in `supermemory.json` (`"container_tag": "hermes_coder"`), or have an org admin merge `hermes_coder` into `hermes-coder` from the Namespaces page at console.supermemory.ai after your first save on 1.0.2.

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
| `supermemory-profile` | `supermemory_profile` | Retrieve the user's profile, optionally with memories matching a `query` |
| `supermemory-read-document` | `supermemory_read_document` | Read a stored document, such as a past conversation, by the `document_id` from search results |

## Source attribution

All Supermemory API calls send `x-sm-source: hermes`, and document writes stamp
`metadata.sm_source: hermes`. This is a **functional routing key, not telemetry**:
it groups Hermes-written memories into a dedicated "Hermes" Space in the
Supermemory app, so you can filter, browse, and bulk-manage them per source agent
(alongside Codex, Claude Code, etc.) from the Supermemory UI.

## Behavior

When enabled, Hermes can:

- before each turn, recall relevant memories, plus profile facts on the first turn and every `profile_frequency` turns
- after each completed turn, save the conversation so far as **one document per conversation**, rewritten in full, so undo, edits and context compaction never store it twice
- skip a save when nothing changed, and retry a failed one later (up to 50 conversations); a failed save is logged and never breaks the turn
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

- Your memories stay searchable under the same `container_tag`. Nothing needs migrating.
- New conversations are saved as one document each instead of one per 4-hour window. A conversation that was open during the upgrade keeps its earlier turns where 1.x saved them, and only new turns go into the new document.
- `search_mode: documents` is now called `chunks`; the old value still works.

## Support

- [Supermemory Discord](https://supermemory.link/discord)
- [support@supermemory.com](mailto:support@supermemory.com)
- [supermemory.ai](https://supermemory.ai)
