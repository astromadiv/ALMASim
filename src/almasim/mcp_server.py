"""``almasim mcp``: the commands as tools for an AI assistant (Model Context Protocol).

The server speaks MCP over stdin and stdout, so an assistant (Claude Code,
Claude Desktop, any MCP client) starts it as ``almasim mcp`` and gets:

- the commands and their options, read from the CLI itself
  (:func:`almasim.jobs.catalog.catalog`), so a tool never offers an option the
  CLI lacks;
- ``submit_command``: any of them as a run, exactly as ``almasim submit`` and
  the web console submit it (same profile, same run folders, same policy);
- the runs: list, show, logs, wait, cancel;
- ``run_here`` for the few commands that only read small files;
- prompts, which a client offers as commands of its own (in Claude Code:
  ``/mcp__almasim__runs``).

Each tool is a plain function here; :func:`build_server` registers them.
Install with the ``mcp`` extra (``pip install almasim[mcp]``).
"""

from __future__ import annotations

import contextlib
import functools
import getpass
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from almasim.jobs.model import command_name

# The commands that only read small files and answer in seconds: these run on
# the node the server runs on. Everything else goes through submit_command.
LIGHT_COMMANDS = {
    "noise show": "A released noise model, or the list of them.",
    "noise components": "The components a noise model can hold.",
    "noise schema": "The JSON schema of a noise model file.",
}
LIGHT_TIMEOUT_S = 120
# What a tool returns of a log or of a command's output, at most: an assistant
# reads all of it, so the end of a long log is worth more than its whole.
MAX_OUTPUT_CHARS = 20000
MAX_WAIT_S = 1800

INSTRUCTIONS = """\
ALMASim simulates ALMA observations and images real and simulated data.
Find a command with list_commands and read its options with describe_command
before building a command line. submit_command runs it as a run (a Slurm job
where the profile says so); follow it with show_run, run_logs and
wait_for_run. Never run heavy almasim commands in a shell on a login node:
submit them. `noise promote` releases a noise model for the whole team: ask
the user before submitting it.
"""

# The user profile `almasim mcp --profile` named; None: the environment's.
_user_profile: Path | None = None


class ToolError(ValueError):
    """What a tool could not do, as the assistant should read it."""


def _profile():
    from almasim.jobs.profile import ProfileError, load_profile

    try:
        return load_profile(user=_user_profile)
    except ProfileError as exc:
        raise ToolError(str(exc)) from exc


def _clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"[... {len(text) - limit} earlier characters left out ...]\n{text[-limit:]}"


def _run_json(run) -> dict[str, Any]:
    from almasim.jobs.service import short_id

    job, status = run.job, run.status
    return {
        "id": job.id,
        "short_id": short_id(job.id),
        "state": run.state.value,
        "reason": run.reason,
        "command": f"almasim {shlex.join(job.argv)}",
        "title": job.title,
        "owner": job.owner,
        "executor": job.executor,
        "slurm_job_id": job.slurm_job_id,
        "resources": job.resources.to_dict(),
        "created": datetime.fromtimestamp(job.created_at).isoformat(timespec="seconds"),
        "host": status.host,
        "exit_code": status.exit_code,
        "folder": str(run.run_dir),
    }


def _find(profile, run_id: str) -> Path:
    from almasim.jobs.service import RunError, find_run

    try:
        return find_run(profile, run_id)
    except RunError as exc:
        raise ToolError(str(exc)) from exc


def _current(run_dir: Path):
    """The run with Slurm's word on it, as ``almasim jobs show`` reads it."""
    from almasim.jobs.executors.slurm import poll
    from almasim.jobs.service import view

    job_id = view(run_dir).job.slurm_job_id
    return view(run_dir, poll([job_id]).get(job_id) if job_id else None)


def list_commands() -> list[dict[str, Any]]:
    """Every almasim command: its name, what it does and where it runs.

    ``runs_on`` is "slurm" or "server" for the commands submit_command takes,
    and "here" for the light ones run_here takes.
    """
    from almasim.jobs.catalog import catalog

    commands = [
        {
            "name": info.name,
            "summary": info.summary,
            "runs_on": info.executor,
            "needs_internet": info.needs_internet,
        }
        for info in catalog(_profile())
    ]
    commands += [
        {"name": name, "summary": summary, "runs_on": "here", "needs_internet": False}
        for name, summary in LIGHT_COMMANDS.items()
    ]
    return commands


def describe_command(command: str) -> dict[str, Any]:
    """One command's help and every argument and option it takes.

    ``command`` is a name from list_commands, e.g. "image image-from-ms". Each
    parameter has its flags, type, default, whether it is required, and help.
    """
    from almasim.jobs.catalog import catalog

    name = " ".join(command.split())
    if name.startswith("almasim "):
        name = name[len("almasim ") :]
    for info in catalog(_profile()):
        if info.name == name:
            return asdict(info)
    if name in LIGHT_COMMANDS:
        return {"name": name, "runs_on": "here", "help": _almasim([*name.split(), "--help"])}
    raise ToolError(f"unknown command {name!r}: list_commands names them all")


def submit_command(
    argv: list[str],
    cpus: int | None = None,
    memory: str | None = None,
    time: str | None = None,
    partition: str | None = None,
    account: str | None = None,
    title: str = "",
    after: list[str] | None = None,
) -> dict[str, Any]:
    """Run ``almasim <argv>`` as a run (a Slurm job where the profile says so).

    ``argv`` is the command line without "almasim", one word per item, e.g.
    ["noise", "compare", "--model", "alma@7", ...]. The CLI's own parser checks
    it first, so a wrong option fails here and nothing is submitted. ``cpus``,
    ``memory`` ("16G"), ``time`` ("02:00:00"), ``partition`` and ``account``
    override the profile's defaults for the command. ``after`` lists run ids
    that must succeed first. Returns the run: its id, state and folder.
    """
    from almasim.jobs import ResourcesError
    from almasim.jobs.catalog import validate
    from almasim.jobs.executors.slurm import SlurmError
    from almasim.jobs.policy import PolicyError
    from almasim.jobs.service import RunError, find_run
    from almasim.jobs.service import submit_command as submit

    profile = _profile()
    if argv and argv[0] == "almasim":
        argv = argv[1:]
    if command_name(argv) in LIGHT_COMMANDS:
        raise ToolError(f"almasim {command_name(argv)} is a light command: use run_here")
    problems = validate(argv) if argv else ["give the command line as a list of words"]
    if problems:
        raise ToolError("; ".join(problems))
    resources = {
        "partition": partition,
        "account": account,
        "cpus": cpus,
        "memory": memory,
        "time": time,
    }
    try:
        parents = [find_run(profile, ref) for ref in after or []]
        run = submit(argv, profile, resources=resources, title=title, after=parents)
    except (PolicyError, ResourcesError, SlurmError, RunError) as exc:
        raise ToolError(str(exc)) from exc
    return _run_json(run)


def list_runs(limit: int = 20, everyone: bool = False) -> list[dict[str, Any]]:
    """The runs, newest first: the user's own, or the team's with ``everyone``."""
    from almasim.jobs import service

    profile = _profile()
    service.advance_pipelines(profile)
    owner = None if everyone else getpass.getuser()
    runs = service.list_runs(profile, owner=owner, limit=max(1, limit))
    return [_run_json(run) for run in runs]


def show_run(run_id: str) -> dict[str, Any]:
    """One run: its command, resources, state, exit code and folder.

    ``run_id`` is the id, or its first or last characters (at least 4).
    """
    return _run_json(_current(_find(_profile(), run_id)))


def run_logs(run_id: str, stderr: bool = False, tail_lines: int = 200) -> str:
    """The end of a run's output: stdout.log, or stderr.log with ``stderr``."""
    from almasim.jobs.wrapper import STDERR_NAME, STDOUT_NAME

    path = _find(_profile(), run_id) / (STDERR_NAME if stderr else STDOUT_NAME)
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return _clip("\n".join(lines[-max(1, tail_lines) :]))


def wait_for_run(run_id: str, timeout_s: float = 300.0) -> dict[str, Any]:
    """Wait until the run ends or ``timeout_s`` (at most 1800) passes; the run then.

    A run whose state is not succeeded, failed or cancelled is still going:
    call again to keep waiting.
    """
    from almasim.jobs.service import wait

    profile = _profile()
    timeout = min(max(0.0, timeout_s), MAX_WAIT_S)
    return _run_json(wait(_find(profile, run_id), profile=profile, timeout=timeout))


def cancel_run(run_id: str) -> dict[str, Any]:
    """Stop a run (the user's own only)."""
    from almasim.jobs import service
    from almasim.jobs.executors.slurm import SlurmError

    try:
        return _run_json(service.cancel_run(_find(_profile(), run_id)))
    except (service.RunError, SlurmError) as exc:
        raise ToolError(str(exc)) from exc


def _almasim(argv: list[str]) -> str:
    """``almasim <argv>`` in a process of its own; its output, or ToolError."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "almasim.cli", *argv],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,  # the server's stdin carries the protocol
            timeout=LIGHT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"almasim {shlex.join(argv)} took over {LIGHT_TIMEOUT_S} s") from exc
    if result.returncode != 0:
        raise ToolError(
            _clip(f"almasim {shlex.join(argv)} exited {result.returncode}:\n{result.stderr}")
        )
    return _clip(result.stdout)


def run_here(argv: list[str]) -> str:
    """Run a light command on this node and return what it prints.

    Only the commands list_commands marks ``runs_on: here`` (noise show, noise
    components, noise schema), e.g. ["noise", "show"] for the released noise
    models. Every other command goes through submit_command.
    """
    if argv and argv[0] == "almasim":
        argv = argv[1:]
    if command_name(argv) not in LIGHT_COMMANDS:
        raise ToolError(
            f"run_here takes only {', '.join(sorted(LIGHT_COMMANDS))}: "
            "submit the other commands with submit_command"
        )
    return _almasim(argv)


def prompt_runs() -> str:
    """Report on my ALMASim runs: what is running, what failed and why."""
    return (
        "List my ALMASim runs with list_runs. For every run that failed, read the end of "
        "its stderr and stdout with run_logs and say in one line why it failed. Then list "
        "what is still running or queued, with each run's short id and command."
    )


def prompt_run(task: str) -> str:
    """Run an ALMASim task: find the command, check its options, submit it."""
    return (
        f"Do this with ALMASim: {task}\n\n"
        "Find the command with list_commands and read its options with describe_command. "
        "Show me the command line and the resources you will ask for, then submit it with "
        "submit_command and tell me the run's short id and folder. Do not wait for it to "
        "end unless I ask."
    )


def prompt_validate_noise_model(model: str, windows: str, metadata: str, measured: str) -> str:
    """Validate a noise model on real windows and report whether it passes."""
    return (
        f"Validate the ALMASim noise model {model} against real data.\n\n"
        "Read describe_command for `noise validate`, then submit it with submit_command: "
        f"--model {model} --windows {windows} --metadata {metadata} --measured {measured}, "
        "an --out folder you propose to me first, and --allow-partial. When the run ends "
        "(wait_for_run), read its log with run_logs and report: how many windows were "
        "validated and how many failed, each metric's median ratio with its verdict, and "
        "whether the model passes. Name the windows the comparison left out and why."
    )


TOOLS: tuple[Callable[..., Any], ...] = (
    list_commands,
    describe_command,
    submit_command,
    list_runs,
    show_run,
    run_logs,
    wait_for_run,
    cancel_run,
    run_here,
)
# Tools that only read: a client may run these without asking the user.
READ_ONLY = frozenset(
    {"list_commands", "describe_command", "list_runs", "show_run", "run_logs", "wait_for_run"}
)
PROMPTS: dict[str, Callable[..., str]] = {
    "runs": prompt_runs,
    "run": prompt_run,
    "validate-noise-model": prompt_validate_noise_model,
}


def _in_thread(tool: Callable[..., Any]) -> Callable[..., Any]:
    """``tool`` as the server calls it: in a worker thread, so a long wait does
    not hold up the protocol, and with anything it prints sent to stderr
    (stdout carries the protocol)."""
    import anyio

    def call(*args: Any, **kwargs: Any) -> Any:
        with contextlib.redirect_stdout(sys.stderr):
            return tool(*args, **kwargs)

    @functools.wraps(tool)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await anyio.to_thread.run_sync(functools.partial(call, *args, **kwargs))

    return wrapper


def build_server(user_profile: Path | None = None):
    """The MCP server, with every tool and prompt registered."""
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    global _user_profile
    _user_profile = user_profile
    # Warnings only: the SDK logs every request at INFO on stderr.
    server = FastMCP("almasim", instructions=INSTRUCTIONS, log_level="WARNING")
    for tool in TOOLS:
        server.add_tool(
            _in_thread(tool),
            name=tool.__name__,
            description=tool.__doc__,
            annotations=ToolAnnotations(readOnlyHint=tool.__name__ in READ_ONLY),
        )
    for name, prompt in PROMPTS.items():
        server.prompt(name=name, description=prompt.__doc__)(prompt)
    return server


def run(user_profile: Path | None = None) -> None:
    """Serve over stdin and stdout until the client closes them."""
    build_server(user_profile).run()
