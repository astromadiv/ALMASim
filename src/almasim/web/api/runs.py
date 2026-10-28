"""Runs as the API returns them (web console plan, S2.3).

The models only describe the JSON (for the OpenAPI schema the web app's
types are generated from, U3.1); the routes return ``run_json`` dicts.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from almasim.jobs.model import JobState
from almasim.jobs.service import RunView, short_id


class RunResources(BaseModel):
    partition: str | None = None
    account: str | None = None
    cpus: int
    memory: str
    time: str


class Run(BaseModel):
    id: str
    short_id: str
    owner: str
    command: str
    argv: list[str]
    title: str
    executor: str
    resources: RunResources
    params: dict[str, Any]
    parents: list[str]
    slurm_job_id: str | None
    state: JobState
    finished: bool
    reason: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    host: str | None
    exit_code: int | None
    signal: str | None
    run_dir: str


class RunDetail(Run):
    files: list[str]
    dashboard: bool = Field(
        False,
        description="Whether the run serves a Dask dashboard now, relayed at "
        "/api/v2/jobs/{run_id}/dashboard/status.",
    )


class RunList(BaseModel):
    runs: list[Run]
    user: str


class Accounting(BaseModel):
    """``sacct``'s rows for a run's Slurm job (the job, then its steps); none for a server job."""

    slurm_job_id: str | None
    rows: list[dict[str, str]]


class LogSlice(BaseModel):
    text: str
    offset: int
    next_offset: int
    size: int


def run_json(run: RunView) -> dict[str, Any]:
    job, status = run.job, run.status
    return {
        "id": job.id,
        "short_id": short_id(job.id),
        "owner": job.owner,
        "command": job.command,
        "argv": job.argv,
        "title": job.title,
        "executor": job.executor,
        "resources": job.resources.to_dict(),
        "params": job.params,
        "parents": job.parents,
        "slurm_job_id": job.slurm_job_id,
        "state": run.state.value,
        "finished": run.state.finished,
        "reason": run.reason,
        "created_at": job.created_at,
        "started_at": status.started_at,
        "finished_at": status.finished_at,
        "host": status.host,
        "exit_code": status.exit_code,
        "signal": status.signal,
        "run_dir": str(run.run_dir),
    }
