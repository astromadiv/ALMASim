"""almasim mcp: the commands, almasim submit and the runs as an assistant's tools."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from almasim import cli, mcp_server
from almasim.jobs import JobState, RunStatus, write_manifest, write_status
from almasim.jobs.executors import slurm as slurm_exec
from almasim.jobs.executors.slurm import SlurmState
from almasim.jobs.policy import KNOWN_COMMANDS
from almasim.mcp_server import ToolError

pytest.importorskip("mcp")

pytestmark = pytest.mark.unit


class FakeSlurm:
    def __init__(self):
        self.states: dict[str, SlurmState] = {}
        self.cancelled: list[str] = []

    def submit(self, job, run_dir, profile, **kwargs):
        job.slurm_job_id = str(4242 + len(self.states))
        job.state = JobState.QUEUED
        write_manifest(run_dir, job)
        self.states[job.slurm_job_id] = SlurmState("PENDING", "Priority")
        return job.slurm_job_id

    def poll(self, ids, **kwargs):
        return {i: self.states[i] for i in ids if i in self.states}

    def cancel(self, job, **kwargs):
        self.cancelled.append(job.slurm_job_id)


@pytest.fixture
def slurm(monkeypatch, tmp_path):
    """A stand-in Slurm and a profile whose runs go under tmp_path."""
    fake = FakeSlurm()
    for name in ("submit", "poll", "cancel"):
        monkeypatch.setattr(slurm_exec, name, getattr(fake, name))
    user = tmp_path / "almasim.toml"
    user.write_text(f'runs_root = "{tmp_path / "runs"}"\n')
    monkeypatch.setenv("ALMASIM_PROFILE", str(user))
    monkeypatch.delenv("ALMASIM_SITE_PROFILE", raising=False)
    monkeypatch.setattr(mcp_server, "_user_profile", None)
    return fake


def test_the_commands_are_the_clis_own(slurm):
    commands = {c["name"]: c for c in mcp_server.list_commands()}
    assert set(commands) == KNOWN_COMMANDS | set(mcp_server.LIGHT_COMMANDS)
    assert commands["noise validate"]["runs_on"] == "slurm"
    assert commands["noise show"]["runs_on"] == "here"
    assert commands["metadata"]["needs_internet"] is True

    described = mcp_server.describe_command("almasim  noise   compare")
    flags = {flag for p in described["params"] for flag in p["flags"]}
    assert {"--model", "--windows", "--measured", "--out", "--criteria"} <= flags
    with pytest.raises(ToolError, match="unknown command"):
        mcp_server.describe_command("noise nothing")


def test_a_command_is_submitted_shown_listed_and_cancelled(slurm):
    run = mcp_server.submit_command(["almasim", "doctor", "--slurm"], cpus=2, title="check")
    assert run["state"] == "queued" and run["slurm_job_id"] == "4242"
    assert run["command"] == "almasim doctor --slurm"
    assert run["resources"]["cpus"] == 2 and run["title"] == "check"
    assert Path(run["folder"]).is_dir()

    assert mcp_server.show_run(run["short_id"])["id"] == run["id"]
    assert [r["id"] for r in mcp_server.list_runs()] == [run["id"]]

    second = mcp_server.submit_command(["doctor"], after=[run["short_id"]])
    assert [r["id"] for r in mcp_server.list_runs(limit=1)] == [second["id"]]

    mcp_server.cancel_run(run["id"])
    assert slurm.cancelled == ["4242"]


def test_a_command_line_the_cli_rejects_is_never_submitted(slurm, tmp_path):
    for argv, message in (
        (["rm", "-rf", "/"], "unknown command"),
        (["noise", "compare", "--no-such-option"], "No such option"),
        (["noise", "show"], "use run_here"),
        ([], "list of words"),
    ):
        with pytest.raises(ToolError, match=message):
            mcp_server.submit_command(argv)
    with pytest.raises(ToolError, match="memory"):
        mcp_server.submit_command(["doctor"], memory="a lot")
    with pytest.raises(ToolError, match="no run"):
        mcp_server.show_run("ZZZZZZ")
    assert not list((tmp_path / "runs").glob("*/*"))


def test_logs_give_the_end_of_the_output_and_waiting_returns_the_ended_run(slurm):
    run = mcp_server.submit_command(["doctor"])
    folder = Path(run["folder"])
    assert mcp_server.run_logs(run["id"]) == ""
    (folder / "stdout.log").write_text("".join(f"line {i}\n" for i in range(50)))
    (folder / "stderr.log").write_text("it broke\n")
    assert mcp_server.run_logs(run["id"], tail_lines=2) == "line 48\nline 49"
    assert mcp_server.run_logs(run["id"], stderr=True) == "it broke"
    clipped = mcp_server._clip("x" * 50, limit=10)
    assert clipped.endswith("x" * 10) and "40 earlier characters left out" in clipped

    assert mcp_server.wait_for_run(run["id"], timeout_s=0)["state"] == "queued"
    write_status(folder, RunStatus(started_at=1.0, finished_at=2.0, exit_code=3))
    ended = mcp_server.wait_for_run(run["id"])
    assert ended["state"] == "failed" and ended["exit_code"] == 3


def test_only_light_commands_run_here(slurm, monkeypatch):
    calls: list[list[str]] = []

    def standin(argv, **kwargs):
        calls.append(argv)
        assert kwargs["stdin"] == subprocess.DEVNULL, "the protocol's stdin is not the command's"
        if "schema" in argv:
            return subprocess.CompletedProcess(argv, 1, "", "no schema")
        if "components" in argv:
            raise subprocess.TimeoutExpired(argv, 1)
        return subprocess.CompletedProcess(argv, 0, "alma@7\n", "")

    monkeypatch.setattr(mcp_server.subprocess, "run", standin)
    assert mcp_server.run_here(["almasim", "noise", "show"]) == "alma@7\n"
    assert calls[0][1:] == ["-m", "almasim.cli", "noise", "show"]
    assert mcp_server.describe_command("noise show")["help"] == "alma@7\n"
    with pytest.raises(ToolError, match="exited 1:\nno schema"):
        mcp_server.run_here(["noise", "schema"])
    with pytest.raises(ToolError, match="took over"):
        mcp_server.run_here(["noise", "components"])
    with pytest.raises(ToolError, match="submit the other commands"):
        mcp_server.run_here(["noise", "validate", "--model", "alma@7"])


def test_a_profile_that_cannot_be_read_is_the_tools_error(slurm, monkeypatch, tmp_path):
    monkeypatch.setenv("ALMASIM_PROFILE", str(tmp_path / "missing.toml"))
    with pytest.raises(ToolError):
        mcp_server.list_runs()


def test_the_server_offers_every_tool_and_prompt(slurm, capsys):
    server = mcp_server.build_server()

    async def session():
        tools = {tool.name: tool for tool in await server.list_tools()}
        prompts = {prompt.name for prompt in await server.list_prompts()}
        listed = await server.call_tool("list_runs", {"limit": 5})
        prompt = await server.get_prompt("run", {"task": "image my MS"})
        return tools, prompts, listed, prompt

    tools, prompts, listed, prompt = asyncio.run(session())
    assert set(tools) == {tool.__name__ for tool in mcp_server.TOOLS}
    assert tools["submit_command"].inputSchema["required"] == ["argv"]
    assert "CLI's own parser" in tools["submit_command"].description
    assert tools["show_run"].annotations.readOnlyHint is True
    assert tools["cancel_run"].annotations.readOnlyHint is False
    assert prompts == set(mcp_server.PROMPTS)
    assert "image my MS" in prompt.messages[0].content.text
    assert listed is not None
    assert capsys.readouterr().out == "", "stdout carries the protocol"

    for name, prompt_function in mcp_server.PROMPTS.items():
        arguments = prompt_function.__code__.co_varnames[: prompt_function.__code__.co_argcount]
        assert prompt_function(*arguments), name


def test_the_command_names_the_extra_when_the_sdk_is_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def without_mcp(name, *args, **kwargs):
        if name == "mcp":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_mcp)
    result = CliRunner().invoke(cli.app, ["mcp"])
    assert result.exit_code == 1 and "almasim[mcp]" in result.output


def test_the_command_serves_with_the_named_profile(monkeypatch, tmp_path):
    served: list[Path | None] = []
    monkeypatch.setattr(mcp_server, "run", served.append)
    result = CliRunner().invoke(cli.app, ["mcp", "--profile", str(tmp_path / "me.toml")])
    assert result.exit_code == 0, result.output
    assert served == [tmp_path / "me.toml"]
