> The [Supermemory](https://supermemory.ai) memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent), maintained by Supermemory. It was previously bundled inside Hermes core. See [MAINTAINERS.md](MAINTAINERS.md).

# Supermemory Memory Provider

Semantic long-term memory with profile recall, semantic search, explicit memory tools, and per-turn conversation capture into one document per conversation.

## Requirements

- The official SDK, `supermemory>=5.0.0,<6` (declared in `pyproject.toml`; Hermes installs it with the plugin)
- Hosted: a Supermemory account (sign in with your browser), or an API key from [console.supermemory.ai/keys](https://console.supermemory.ai/keys?create=true)
- Self-hosted: a running [Supermemory local](https://supermemory.ai/docs/self-hosting/overview) server and the API key it prints on first boot

## Setup

```bash
hermes plugins install supermemoryai/hermes-supermemory
hermes plugins enable supermemory
hermes memory setup    # select "supermemory", then press Enter to sign in with your browser
```

Or manually:

```bash
hermes config set memory.provider supermemory
python3 ~/.hermes/plugins/supermemory/auth.py login
```

### Sign in

The plugin signs in with your Supermemory account over OAuth. `login` opens your browser; on a
machine without one, add `--print-url` and open the printed link anywhere (the callback listens on
`127.0.0.1:19884`, so a remote box needs that port forwarded). Credentials are saved to
`~/.supermemory-hermes/credentials.json` (mode 0600) and refreshed automatically. If the sign-in is
revoked or expires, run `login` again.

```bash
python3 ~/.hermes/plugins/supermemory/auth.py login     # sign in (opens a browser)
python3 ~/.hermes/plugins/supermemory/auth.py status    # OAuth or API key, and the account
python3 ~/.hermes/plugins/supermemory/auth.py logout    # revoke the refresh token and delete the file
```

Use `$HERMES_HOME/plugins/supermemory/auth.py` if `HERMES_HOME` is not `~/.hermes`.

### API key (fallback)

An API key still works exactly as before and takes priority over the OAuth sign-in. Paste it during
`hermes memory setup`, or:

```bash
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
the endpoint first ensures the setup connection probe also stays local. OAuth sign-in is only
available for Supermemory cloud, so self-hosted and custom endpoints always use an API key.

## Config

Config files: `$HERMES_HOME/supermemory.json`, and `$HERMES_HOME/supermemory/config.json`, which the config panel in the Hermes dashboard and Desktop writes. A value set in the panel wins, and once that file exists `hermes memory setup` writes there too. On the first run an existing `supermemory.json` is copied there, so after that, edit settings in the panel or in `supermemory/config.json`.

| Key | Default | Description |
|-----|---------|-------------|
| `base_url` | `https://api.supermemory.ai` | API endpoint for hosted or self-hosted Supermemory. Takes priority over `SUPERMEMORY_BASE_URL`. |
| `namespace` | `hermes` | Namespace used for search and writes. Supports `{identity}` for profile-scoped namespaces (e.g. `hermes-{identity}` → `hermes-coder`). ASCII letters, digits, `_`, `-` and `:` are kept. Each run of other characters becomes one `_`, and runs at the start or end are dropped. Namespaces over 100 characters are shortened to 100, ending in `_` and an 8-character hash of the full name so they stay distinct. |
| `memory_scope` | `shared` | Gateway bots only: `user` gives each person their own namespace, `chat` each conversation, `shared` one for everyone. CLI and Desktop always use `namespace`. New setups choose `user`. |
| `auto_recall` | `true` | Inject relevant memory context before turns |
| `auto_capture` | `true` | Store cleaned user-assistant turns after each response |
| `max_recall_results` | `10` | Max search results recalled with the profile (1-20) |
| `threshold` | unset | Minimum similarity (0-1). Automatic recall uses 0.5 when unset and searches memories only; the search tool uses the server default unless this is set |
| `profile_frequency` | `50` | Include profile facts on the first turn, after a compaction, and every N turns |
| `search_mode` | `hybrid` | Default mode of the search tool: `hybrid`, `memories` (memories only), `chunks` (document excerpts only; `documents` is accepted as before) |
| `rerank` | `none` | `order` re-ranks results. The search tool can also ask for `aggregate`, which merges overlapping results |
| `rewrite_query` | `false` | Rewrite the query before searching (about 400 ms slower) |
| `include_documents`, `include_related`, `include_forgotten` | `false` | Add each result's source document, related memories, or forgotten memories |
| `allow_agent_search_overrides` | `true` | Let the search tool pick `search_mode` and `aggregate`; `false` always uses the config |
| `entity_context` | built-in default | Extraction guidance passed to Supermemory |
| `dreaming` | `dynamic` | `dynamic` groups related conversations before forming memories; `instant` processes each one right away (billed as one extra operation) |
| `api_timeout` | `5.0` | Timeout for SDK requests |

### Gateway memory scope

Existing bots keep `shared`, so every user shares one namespace. To switch, choose "Per user" in the panel or set `"memory_scope": "user"` in `supermemory/config.json`; memories saved while shared stay in the old namespace.

### Upgrading from 1.0.1

1.0.1 changed `-` and `:` in container tags to `_`, so `hermes-coder` was saved as `hermes_coder`. Names set with the 1.x keys (`container_tag`, `custom_containers`, `SUPERMEMORY_CONTAINER_TAG`) keep that spelling, so memories stay where they are, and the panel shows the saved name (for example `hermes_coder`). Names set with `namespace`, `custom_namespaces` or `SUPERMEMORY_NAMESPACE` keep `-` and `:`, and other characters are handled as described for `namespace` above.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SUPERMEMORY_API_KEY` | API key. Optional when signed in with OAuth; takes priority over the sign-in when set |
| `SUPERMEMORY_BASE_URL` | Compatibility fallback for the API endpoint when `base_url` is not configured |
| `SUPERMEMORY_NAMESPACE` | Override the namespace (takes priority over config file). 1.x's `SUPERMEMORY_CONTAINER_TAG` still works |

Base URL precedence is `supermemory.json` → `SUPERMEMORY_BASE_URL` →
`https://api.supermemory.ai`. Hermes resolves it once and uses the same endpoint
for SDK operations and setup/status probes.

## Tools

Only the kebab-case names are registered. 1.x also registered snake_case names such as `supermemory_search`; Hermes now refuses those, and the model retries with the names it is given.

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

- before each turn, recall relevant memories not already shown in the conversation, plus profile facts on the first turn, after a compaction, and every `profile_frequency` turns; the profile is fetched in the background, so only the search delays a turn
- show "recalled N memories" when it adds context
- after each completed turn, save the conversation so far as **one document per conversation**, rewritten in full, so undo, edits and context compaction never store it twice
- skip a save when nothing changed, and retry a failed one later (up to 50 conversations); a failed save is logged and never breaks the turn
- store messages as given (only the plugin's own recalled `<supermemory-context>` block and inline base64 images are left out)
- copy built-in `memory` tool additions, edits and removals into Supermemory
- route every SDK and probe request through the configured hosted or self-hosted endpoint
- expose explicit tools for search, store, forget, profile, and reading documents


## Profile-Scoped Namespaces

Use `{identity}` in the `namespace` to scope memories per Hermes profile:

```json
{
  "namespace": "hermes-{identity}"
}
```

For a profile named `coder`, this resolves to `hermes-coder`. The default profile resolves to `hermes-default`. Without `{identity}`, all profiles share the same namespace.

## Multi-Namespace Mode

For advanced setups (e.g. OpenClaw-style multi-workspace), you can enable custom namespaces so the agent can read/write across multiple named namespaces:

```json
{
  "namespace": "hermes",
  "enable_custom_namespaces": true,
  "custom_namespaces": ["project-alpha", "project-beta", "shared-knowledge"],
  "custom_namespace_instructions": "Use project-alpha for coding tasks, project-beta for research, and shared-knowledge for team-wide facts."
}
```

When enabled:
- every tool accepts an optional `namespace` parameter
- The namespace must be in the whitelist: primary namespace + `custom_namespaces`
- Automatic operations (turn capture, prefetch, memory write mirroring) always use the **primary** namespace only
- Custom namespace instructions are injected into the system prompt

## Upgrading from 1.x

- 1.x called namespaces container tags. `container_tag`, `enable_custom_container_tags`, `custom_containers`, `custom_container_instructions`, `SUPERMEMORY_CONTAINER_TAG` and the tools' `container_tag` argument still work, and your memories stay searchable under the same name. Nothing needs migrating.
- New conversations are saved as one document each instead of one per 4-hour window. A conversation that was open during the upgrade keeps its earlier turns where 1.x saved them, and only new turns go into the new document.
- `search_mode: documents` is now called `chunks`; the old value still works.

## Support

- [Supermemory Discord](https://supermemory.link/discord)
- [support@supermemory.com](mailto:support@supermemory.com)
- [supermemory.ai](https://supermemory.ai)
