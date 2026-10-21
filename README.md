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
| `capture_writes` | `replace` | `replace` rewrites each conversation document in full. `append_delta` sends only new turns and is **only for a backend built to append them**; see [Append-only backends](#append-only-backends-capture_writes-append_delta) |

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
- after each completed turn, save the conversation so far as **one document per conversation**, rewritten in full (`capture_writes: replace`, the default), so undo, edits and context compaction never store it twice
- skip a save when nothing changed, and retry a failed one later (up to 50 conversations); a failed save is logged and never breaks the turn
- store messages as given (only the plugin's own recalled `<supermemory-context>` block and inline base64 images are left out)
- copy built-in `memory` tool additions into Supermemory
- route every SDK and probe request through the configured hosted or self-hosted endpoint
- expose explicit tools for search, store, forget, profile, and reading documents


## Append-only backends (`capture_writes: append_delta`)

> **Leave this off for Supermemory itself (hosted or self-hosted).** The Supermemory API does not promise that a
> second `POST` with the same document id appends to it. Turning this on against a backend that replaces or merges
> on same-id `POST` can lose or garble conversation history.

By default, each capture `PATCH`es the conversation document with the whole conversation, and `POST`s it when the `PATCH`
returns 404. Some deployments put an authenticated gateway in front of the API that refuses replacing `PATCH`es, to
protect history it has already accumulated. With the default, that gateway refuses the capture: it is logged and kept for
retry, and it is never turned into a `POST`. For such a backend only:

```json
{
  "capture_writes": "append_delta"
}
```

The plugin then `POST`s only the turns the document does not hold yet to `/ns/<container_tag>/document`, with the same `id` each
time. The metadata is the default mode's (`sm_source`, `type: conversation`, `session_id`, `docId`) plus `turn_offset`, the
number of the conversation's turns the plugin counts as stored before this one. It never `PATCH`es. The backend **must**:

- append the `content` of a `POST` to an existing `id` to the stored text, unchanged and in order;
- ignore an exact retry: a `POST` to the same namespace whose JSON body is identical to one it already appended. A failed
  send is resent as that identical request (same id, content, date and metadata) before anything newer. A response lost
  after the backend stored the turn looks the same as a failure, so without this the turn is stored twice. Delivery is
  at least once, not exactly once. Compare the whole body: the same turn sent again within a second has the same content
  and `date` (whole seconds) and differs only in `turn_offset`, and it is a new turn;
- answer `GET /ns/<container_tag>/document/<id>` with the accumulated text (404 before the first write). A new provider
  (Hermes' gateway builds one per turn) reads it once to learn which turns are already stored. If that read fails, the
  plugin sends nothing until a later read succeeds, rather than resending the whole conversation.

Document ids are `hermes_<session id>`. A conversation keeps the id of the session it started in, through context
compaction into child sessions; a reset or new session starts a new id.

Captures without a session id go to `hermes_background_<YYYY-MM-DD>_b<0-5>`, one document per four-hour UTC window, shared by
all such captures. A failed send keeps its window, and a provider that runs across a window boundary sends only its new
turns to the next window. Without a session id there is no conversation to resume, though: a provider that starts fresh
learns only what the current window holds, and that may include other captures. It sends again the turns of its
transcript that an earlier window holds, or that it cannot line up with the current window (storing them twice), and it
may skip a turn identical to one another capture already stored there.

Undo and edits cannot remove turns that were already sent, including a failed send that is being retried: the edited or
new turns are appended after them.

As in the default mode, failed sends are retried from memory only (up to 50 pending sends, 256 KiB). If Hermes exits
first, they are sent only when that conversation is captured again.

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
