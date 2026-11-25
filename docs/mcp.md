# MCP server

`almasim mcp` serves the `almasim` commands to an AI assistant over the
[Model Context Protocol](https://modelcontextprotocol.io) (MCP). The
assistant gets tools that list the commands, describe their options, submit
them as runs and follow the runs; you get the server's prompts as commands
in the assistant. It is a third way into the same runs as
[`almasim submit`](cli.md) and the [web console](console.md): a run the
assistant submits shows up in `almasim jobs list` and in the console, and
the other way round.

The server speaks MCP on its standard input and output, so nothing listens
on a port: the assistant starts `almasim mcp` as a process of yours and
talks to it directly. It needs the `mcp` extra:

```bash
pip install "almasim[mcp]"
```

The team's shared install includes it.

## Install it in an assistant

**Claude Code**, for every project of yours (`--scope user`) or for one
project (`--scope project`, which writes a `.mcp.json` the project can
commit):

```bash
claude mcp add --scope user almasim -- almasim mcp
```

On zwicky, with the shared install, name the program and the site profile
in full, so that the server starts whatever the shell's PATH is:

```bash
claude mcp add --scope user almasim \
  -e ALMASIM_SITE_PROFILE=/SKAO_data/cycle11/almasim/site.toml \
  -- /SKAO_data/cycle11/almasim/env/bin/almasim mcp
```

`claude mcp list` shows whether the server connects, and `/mcp` inside
Claude Code lists its tools. A session started before the server was added
does not have it: start a new one.

**Any other MCP client** (Claude Desktop and others) takes the same three
values in its configuration file:

```json
{
  "mcpServers": {
    "almasim": {
      "command": "/SKAO_data/cycle11/almasim/env/bin/almasim",
      "args": ["mcp"],
      "env": {"ALMASIM_SITE_PROFILE": "/SKAO_data/cycle11/almasim/site.toml"}
    }
  }
}
```

The server runs where the assistant runs. To submit to a cluster's Slurm,
run the assistant on the cluster's login node, or give a client on your
laptop `ssh` as the command (`"command": "ssh", "args": ["you@cluster",
"/path/to/almasim", "mcp"]`).

`almasim mcp --profile <file>` reads that user profile instead of
`$ALMASIM_PROFILE` or `~/.config/almasim/almasim.toml`. The profile decides
where each command runs and what it asks for, as it does for `almasim
submit` ([the job profile](console.md)).

## The tools

| Tool | What it does |
| --- | --- |
| `list_commands` | Every command: its name, what it does, where it runs (`slurm`, `server` or `here`) and whether it needs the internet. |
| `describe_command` | One command's help and every argument and option: flags, type, default, whether it is required, help. |
| `submit_command` | Runs `almasim <argv>` as a run: a Slurm job, or a process on the server's node where the profile says so. Takes `cpus`, `memory`, `time`, `partition`, `account`, a `title` and `after` (run ids that must succeed first). |
| `list_runs` | The runs, newest first: yours, or the team's with `everyone`. |
| `show_run` | One run: command, resources, state, exit code, folder. |
| `run_logs` | The end of a run's `stdout.log` or `stderr.log`. |
| `wait_for_run` | Waits until a run ends, for at most 30 minutes per call, and returns it. |
| `cancel_run` | Stops a run of yours. |
| `run_here` | Runs a light command on the server's node and returns what it prints: `noise show`, `noise components`, `noise schema`. |

The commands and their options come from the CLI's own definitions, the
ones the [CLI reference](cli.md) and the console's forms are made from, so
the assistant sees every command and option the installed version has and
none it lacks.

`submit_command` checks the command line with the CLI's own parser before
anything is submitted: an unknown command, a misspelt option or a missing
required one comes back as the tool's error, with the parser's message, and
no run is made. The command then goes through the same policy as `almasim
submit`: the profile's executor and resources for the command, the Slurm
backend for the commands that fan out, and the profile's WSClean and CASA
data.

Only `run_here` runs anything on the node the server runs on, and it takes
three commands that read small files and answer in seconds. Everything else
is a run, so an assistant on a login node does no heavy work there.

Long logs and outputs are cut to their last 20 000 characters, with a line
saying how much was left out.

The tools that only read (`list_commands`, `describe_command`, `list_runs`,
`show_run`, `run_logs`, `wait_for_run`) are marked read-only, so a client
can run them without asking. `submit_command`, `cancel_run` and `run_here`
are not: a client asks you first, unless you allowed them. `noise promote`
releases a noise model for the whole team; the server's instructions tell
the assistant to ask you before submitting it.

## The prompts, as commands

An MCP client offers a server's prompts as commands of its own. In Claude
Code they are slash commands:

| Command | What the assistant does |
| --- | --- |
| `/mcp__almasim__runs` | Reports on your runs: what is running or queued, what failed, and for each failed run why, from its logs. |
| `/mcp__almasim__run <task>` | Finds the command for the task, reads its options, shows you the command line and resources, submits it and gives you the run's id and folder. |
| `/mcp__almasim__validate-noise-model <model> <windows> <metadata> <measured>` | Submits `noise validate` for the model on the real windows, waits, and reports the windows validated and failed, each metric's median ratio and verdict, and whether the model passes. |

You can also just ask: "image this MS as a cube with almasim", "why did my
last run fail?", "validate alma@7 on these windows". The assistant picks
the tools itself.

## What the assistant can and cannot do

- It acts as you: the server is your process, the runs are yours, and Slurm
  sees your user.
- It runs only `almasim` commands, the ones `almasim submit` takes. It has
  no tool to run a shell command or to read or write a file; a command's
  own inputs and outputs are the paths you or the assistant name in its
  command line.
- It cancels only your runs and deletes none.
