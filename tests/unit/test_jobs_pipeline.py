"""Pipelines: runs that wait for runs (web console plan, J1.8)."""

from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner

from almasim import cli
from almasim.jobs import JobState, read_manifest, write_manifest
from almasim.jobs.executors import slurm as slurm_exec
from almasim.jobs.executors.slurm import SlurmState
from almasim.jobs.pipeline import (
    Step,
    archive_to_images,
    noise_model_from_windows,
    simulate,
    submit_pipeline,
)
from almasim.jobs.policy import plan_job
from almasim.jobs.profile import CommandPolicy, Profile
from almasim.jobs.rundir import RunStatus, write_status
from almasim.jobs.service import NOT_STARTED, advance_pipelines, list_runs, submit_command, wait

pytestmark = pytest.mark.unit


class FakeSlurm:
    def __init__(self):
        self.next_id = 7000
        self.states: dict[str, SlurmState] = {}
        self.after: dict[str, list[str]] = {}
        self.cancelled: list[str] = []

    def submit(self, job, run_dir, profile, *, after=(), **kwargs):
        job.slurm_job_id = str(self.next_id)
        self.next_id += 1
        job.state = JobState.QUEUED
        write_manifest(run_dir, job)
        self.after[job.slurm_job_id] = list(after)
        self.states[job.slurm_job_id] = SlurmState("PENDING", "Dependency" if after else "Priority")
        return job.slurm_job_id

    def poll(self, ids, **kwargs):
        return {i: self.states[i] for i in ids if i in self.states}

    def cancel(self, job, **kwargs):
        self.cancelled.append(job.slurm_job_id)


@pytest.fixture
def slurm(monkeypatch):
    fake = FakeSlurm()
    monkeypatch.setattr(slurm_exec, "submit", fake.submit)
    monkeypatch.setattr(slurm_exec, "poll", fake.poll)
    monkeypatch.setattr(slurm_exec, "cancel", fake.cancel)
    return fake


@pytest.fixture
def profile(tmp_path) -> Profile:
    return Profile(
        runs_root=tmp_path / "runs",
        python=sys.executable,
        commands={"doctor": CommandPolicy(executor="server")},  # a quick server-side step
    )


def _finish(run_dir, exit_code=0):
    write_status(run_dir, RunStatus(started_at=1.0, finished_at=2.0, exit_code=exit_code))


SLURM_STEP = Step("check", ["image", "clean-quality", "/out"])


def test_slurm_steps_chain_with_afterok(profile, slurm):
    first, second, third = submit_pipeline([SLURM_STEP] * 3, profile, title="night")
    assert slurm.after[second.job.slurm_job_id] == [first.job.slurm_job_id]
    assert slurm.after[third.job.slurm_job_id] == [second.job.slurm_job_id]
    assert second.job.parents == [first.job.id]
    assert third.job.title == "night: check"
    assert third.job.params == {"pipeline": "night", "step": 3, "steps": 3}


def test_a_step_after_a_server_step_waits_then_starts(profile, slurm):
    server_step, slurm_step = submit_pipeline(
        [Step("doctor", ["doctor", "--help"]), SLURM_STEP], profile
    )
    assert server_step.job.executor == "server"
    assert slurm_step.state is JobState.PENDING and slurm_step.job.slurm_job_id is None

    assert wait(server_step.run_dir, poll_s=0.2, timeout=60).state is JobState.SUCCEEDED
    advance_pipelines(profile)
    moved = read_manifest(slurm_step.run_dir)
    assert moved.state is JobState.QUEUED and moved.slurm_job_id is not None
    assert slurm.after[moved.slurm_job_id] == []  # its parent had already succeeded


def test_when_a_step_fails_the_rest_do_not_start(profile, slurm):
    a, b, c = submit_pipeline(
        [Step("doctor", ["doctor", "--help"]), SLURM_STEP, SLURM_STEP], profile
    )
    wait(a.run_dir, poll_s=0.2, timeout=60)
    _finish(a.run_dir, exit_code=1)  # make the server step a failure
    advance_pipelines(profile)
    advance_pipelines(profile)  # c follows b
    by_id = {r.job.id: r for r in list_runs(profile)}
    for run in (b, c):
        assert by_id[run.job.id].state is JobState.CANCELLED
        assert by_id[run.job.id].reason == NOT_STARTED


def test_a_step_after_a_failed_run_is_not_started(profile, slurm):
    done = submit_command(["image", "clean-quality", "/x"], profile)
    _finish(done.run_dir, exit_code=2)
    slurm.states.pop(done.job.slurm_job_id)  # aged out of Slurm
    later = submit_command(["image", "clean-quality", "/y"], profile, after=[done.run_dir])
    assert later.state is JobState.CANCELLED and later.reason == NOT_STARTED


def test_slurm_dependencies_that_cannot_be_met_are_cancelled(profile, slurm):
    first, second = submit_pipeline([SLURM_STEP, SLURM_STEP], profile)
    slurm.states[second.job.slurm_job_id] = SlurmState("PENDING", "DependencyNeverSatisfied")
    assert advance_pipelines(profile) == [second.run_dir]
    assert slurm.cancelled == [second.job.slurm_job_id]
    assert read_manifest(second.run_dir).reason == NOT_STARTED


def test_the_archive_template_hands_files_from_step_to_step(tmp_path, profile):
    steps = archive_to_images(tmp_path / "night", ["--band", "3"], imaging=["--cube"])
    assert [s.title for s in steps] == [
        "Query the archive",
        "Download and calibrate",
        "Imaging parameters",
        "Imaging",
        "Clean quality",
    ]
    night = tmp_path / "night"
    assert steps[0].argv[:3] == ["metadata", "--save-csv", str(night / "metadata.csv")]
    assert str(night / "metadata.csv") in steps[1].argv
    assert steps[2].argv[2:] == [str(night / "ms" / "calibrated_ms"), str(night / "params.csv")]
    assert steps[3].argv[-1] == "--cube"
    assert steps[4].argv == ["image", "clean-quality", str(night / "images")]
    plans = [plan_job(s.argv, profile) for s in steps]
    assert [p.executor for p in plans] == ["slurm"] * 5
    assert "--yes" in plans[0].argv and "--yes" in plans[1].argv  # both ask otherwise
    assert plans[1].argv[plans[1].argv.index("--backend") + 1] == "slurm"


def test_the_simulation_template(tmp_path):
    [step] = simulate(tmp_path / "sims", tmp_path / "m.csv", ["--source-type", "point"])
    csv, sims = str(tmp_path / "m.csv"), str(tmp_path / "sims")
    assert step.argv[:5] == ["simulation", "--metadata-path", csv, "--output-dir", sims]


def test_submit_after_from_the_cli(tmp_path, monkeypatch, slurm):
    user = tmp_path / "almasim.toml"
    user.write_text(f'runs_root = "{tmp_path / "runs"}"\n')
    monkeypatch.setenv("ALMASIM_PROFILE", str(user))
    monkeypatch.delenv("ALMASIM_SITE_PROFILE", raising=False)
    runner = CliRunner()
    runner.invoke(cli.app, ["submit", "image", "clean-quality", "/a"])
    [first] = list((tmp_path / "runs").glob("*/*"))
    result = runner.invoke(
        cli.app, ["submit", "--after", first.name[-8:], "image", "pb-correct", "p.csv", "/a"]
    )
    assert result.exit_code == 0, result.output
    assert "after:" in result.output
    assert list(slurm.after.values())[-1] == [read_manifest(first).slurm_job_id]


def test_the_noise_model_template_measures_reduces_and_validates(tmp_path, profile):
    folder = tmp_path / "noise"
    steps = noise_model_from_windows(
        folder, tmp_path / "windows.csv", tmp_path / "rows.csv", fit=["--realizations", "2"]
    )
    assert [s.argv[:2] for s in steps] == [
        ["noise", "extract"],
        ["noise", "fit"],
        ["noise", "aggregate"],
        ["noise", "validate"],
    ]
    candidate = str(folder / "candidate.json")
    assert steps[1].argv[-2:] == ["--realizations", "2"]
    assert candidate in steps[2].argv and steps[3].argv[3] == candidate
    plans = [plan_job(s.argv, profile) for s in steps]
    assert [p.executor for p in plans] == ["slurm"] * 4
    assert "--backend" in plans[0].argv and "--backend" in plans[3].argv
