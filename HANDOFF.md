# Handoff notes — hermes-supermemory

This is the standalone `supermemory` memory provider for [Hermes](https://github.com/NousResearch/hermes-agent).
It previously shipped inside `NousResearch/hermes-agent` under `plugins/memory/supermemory/`. Nous Research is
moving every memory provider out of the core tree, so it now lives here and is maintained by
[Supermemory](https://supermemory.ai). It is listed on the
[Hermes plugin catalog](https://hermes-agent.nousresearch.com/plugins).

What it is: Supermemory cloud memory.

## Install (as a user)

```
hermes plugins install supermemoryai/hermes-supermemory
hermes plugins enable supermemory
hermes memory setup            # or: set memory.provider: supermemory in config.yaml
```

Dependencies in `pyproject.toml` are installed into the Hermes venv automatically and survive
`hermes update`.

## What changed versus the in-tree copy

Mechanical only; behaviour is identical.

- Added `pyproject.toml` declaring: `supermemory>=3.50.0,<4`.
- Dropped the `tools.lazy_deps.ensure("memory.supermemory")` call (see below). This is the only code change.
- Single-file package, so no self-imports needed rewriting and no shared helpers were vendored.
- Fixed the API key link: it was `http://app.supermemory.ai/integrations?connect=hermes`, which is no longer
  a Supermemory host. It is now `https://console.supermemory.ai/keys?create=true`.

## Maintenance notes

- The in-tree `tools.lazy_deps.ensure("memory.supermemory")` calls were removed: they pinned the exact
  (old) version in Hermes' lazy-deps registry and downgraded newer installs (hermes-agent#86992).
  `pyproject.toml` is now the only dependency authority; bump it when you need a newer client.
- The setup wizard imports private helpers from `hermes_cli.memory_setup` (`_curses_select`,
  `_prompt`, ...). Those are Hermes internals, not API; expect to own a copy or drop the wizard
  hook if they move.
- Tests were not copied: the in-tree tests import `plugins.memory.supermemory` and depend on the
  hermes-agent test harness. See `tests/plugins/memory/` in hermes-agent for the originals.
- While the in-tree copy still exists, a same-named user plugin is shadowed by it
  (bundled providers win on name). It takes effect the moment core drops `plugins/memory/supermemory`.

Original authors are preserved in hermes-agent's history: `git log -- plugins/memory/supermemory`.
