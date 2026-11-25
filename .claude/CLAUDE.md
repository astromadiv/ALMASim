# ALMASim handbook

## Documentation

- Every change updates the docs in the same PR. Before committing, grep `docs/` and `README.md` for every passage the change touches (functions, options, quantities, outputs) and update each one. Regenerate `docs/cli.md` when CLI help changes. List the doc files in the PR's "What changes".
- The docs always describe what ALMASim does now: its current capabilities, behaviour and outputs.
- Never reference the past in the docs. No "before #NNN", "used to", "until", "since", "previously", "now", "no longer" or "changed in". No notes about old outputs or migration, and no troubleshooting rows for behaviour that no longer exists. Give a reason as a fact about the present, for example "the row's Int.Time is the whole observing unit's, so ...", not as a story of what was fixed.
- History belongs in commit messages, PR descriptions and the changelog, never in `docs/`.

## Running almasim: the MCP server

The `almasim` MCP server is installed in the user's Claude Code config (shared install on zwicky, `almasim mcp`). Its tools are named `mcp__almasim__<tool>`; if they are deferred, load them with ToolSearch first. If the server is missing or fails to connect, say so and fall back to `almasim submit` (never run a heavy command on the login node). Full reference: `docs/mcp.md`.

- Use the tools, not a shell, to run released `almasim` commands:
  1. `list_commands` to find the command, `describe_command` to read its options. Do not compose options from memory.
  2. `submit_command` with `argv` as a list of words without "almasim" (e.g. `["noise", "compare", "--model", "alma@7", ...]`), plus `cpus`, `memory` ("16G"), `time` ("02:00:00"), `title`, and `after` (run ids that must succeed first) when needed. The CLI's parser checks the line first; a refusal means nothing was submitted, so fix the line and submit again.
  3. `show_run`, `run_logs` (`stderr: true` for errors) and `wait_for_run` (at most 30 min per call; call again while the state is not succeeded, failed or cancelled) to follow it. `list_runs` for an overview, `cancel_run` for the user's own runs.
- `run_here` takes only `noise show`, `noise components` and `noise schema`.
- Ask the user before submitting `noise promote` (it releases a model for the team) and before anything that overwrites results.
- The server runs the shared install's version. To run code from a branch or worktree that is not released there, the tools do not apply: use `sbatch` with that checkout, as before.
- Runs count against the Slurm limits in memory (keep ours at 8 or fewer, touch only our own).
- The user's own commands: `/mcp__almasim__runs`, `/mcp__almasim__run <task>`, `/mcp__almasim__validate-noise-model`.
