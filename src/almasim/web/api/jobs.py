"""Submit, list and control runs over HTTP (web console plan, S2.3).

The routes are thin: everything goes through almasim.jobs.service, the same
code ``almasim submit`` and ``almasim jobs`` use, so the browser and the
terminal see the same runs.
"""

from __future__ import annotations

import codecs
import getpass
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from almasim.jobs.catalog import validate
from almasim.jobs.executors.slurm import SlurmError
from almasim.jobs.model import JobState, ResourcesError
from almasim.jobs.policy import PolicyError
from almasim.jobs.rundir import ManifestError, read_dashboard
from almasim.jobs.service import (
    RunError,
    cancel_run,
    delete_run,
    find_run,
    list_runs,
    resubmit,
    submit_command,
    view,
)
from almasim.jobs.wrapper import STDERR_NAME, STDOUT_NAME

from .runs import Accounting, LogSlice, Run, RunDetail, RunList, run_json

router = APIRouter(prefix="/api/v2/jobs", tags=["jobs"])

LOG_CHUNK = 256 * 1024


class SubmitRequest(BaseModel):
    argv: list[str] = Field(..., min_length=1, description="The command line after `almasim`.")
    resources: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    title: str = ""
    after: list[str] = Field(default_factory=list, description="Run ids to wait for.")


def _run_dir(request: Request, run_id: str):
    try:
        return find_run(request.app.state.profile, run_id)
    except RunError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("", status_code=201, response_model=Run)
def submit(body: SubmitRequest, request: Request) -> dict[str, Any]:
    problems = validate(body.argv)
    if problems:
        raise HTTPException(status_code=422, detail={"problems": problems})
    profile = request.app.state.profile
    parents = [_run_dir(request, ref) for ref in body.after]
    try:
        run = submit_command(
            body.argv,
            profile,
            resources=body.resources,
            params=body.params,
            title=body.title,
            after=parents,
        )
    except (PolicyError, ResourcesError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SlurmError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    request.app.state.events.poke()
    return run_json(run)


@router.get("", response_model=RunList)
def list_jobs(
    request: Request,
    owner: str = Query("me", description="`me`, `all`, or a user name."),
    state: list[str] = Query(default_factory=list, description="Only runs in these states."),
    limit: int = Query(50, ge=1, le=1000),
) -> dict[str, Any]:
    who = getpass.getuser() if owner == "me" else None if owner == "all" else owner
    wanted = {JobState(s) for s in state} if state else None
    runs = list_runs(request.app.state.profile, owner=who, limit=limit)
    if wanted is not None:
        runs = [r for r in runs if r.state in wanted]
    return {"runs": [run_json(r) for r in runs], "user": getpass.getuser()}


@router.get("/{run_id}", response_model=RunDetail)
def get_job(run_id: str, request: Request) -> dict[str, Any]:
    run_dir = _run_dir(request, run_id)
    try:
        current = _fresh(run_dir)
    except ManifestError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    files = sorted(p.name for p in run_dir.iterdir() if not p.name.startswith("."))
    dashboard = not current.state.finished and read_dashboard(run_dir) is not None
    return {**run_json(current), "files": files, "dashboard": dashboard}


def _fresh(run_dir):
    from almasim.jobs.executors.slurm import poll

    job_id = view(run_dir).job.slurm_job_id
    return view(run_dir, poll([job_id]).get(job_id) if job_id else None)


@router.delete("/{run_id}", status_code=204)
def delete(run_id: str, request: Request) -> None:
    """Remove a finished run's folder; only its owner may (409 otherwise)."""
    try:
        delete_run(_run_dir(request, run_id))
    except RunError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    request.app.state.events.poke()


@router.post("/{run_id}/cancel", response_model=Run)
def cancel(run_id: str, request: Request) -> dict[str, Any]:
    try:
        run = cancel_run(_run_dir(request, run_id))
    except RunError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SlurmError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    request.app.state.events.poke()
    return run_json(run)


@router.post("/{run_id}/resubmit", status_code=201, response_model=Run)
def resubmit_job(run_id: str, request: Request) -> dict[str, Any]:
    try:
        run = resubmit(_run_dir(request, run_id), request.app.state.profile)
    except (RunError, PolicyError, ResourcesError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SlurmError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    request.app.state.events.poke()
    return run_json(run)


@router.get("/{run_id}/accounting", response_model=Accounting)
def job_accounting(run_id: str, request: Request) -> dict[str, Any]:
    """Slurm's accounting for the run's job: state, elapsed, node, CPUs, peak memory."""
    from almasim.jobs.executors.slurm import accounting

    slurm_id = view(_run_dir(request, run_id)).job.slurm_job_id
    if not slurm_id:
        return {"slurm_job_id": None, "rows": []}
    try:
        rows = accounting(slurm_id)
    except SlurmError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"slurm_job_id": slurm_id, "rows": rows}


@router.get("/{run_id}/log", response_model=LogSlice)
def read_log(
    run_id: str,
    request: Request,
    stream: str = Query("stdout", pattern="^(stdout|stderr)$"),
    offset: int = Query(0, ge=0, description="Byte offset to read from."),
    limit: int = Query(LOG_CHUNK, ge=1, le=4 * LOG_CHUNK),
) -> dict[str, Any]:
    """A slice of a run's log; poll with ``next_offset`` to follow it."""
    path = _run_dir(request, run_id) / (STDOUT_NAME if stream == "stdout" else STDERR_NAME)
    if not path.exists():
        return {"text": "", "offset": offset, "next_offset": offset, "size": 0}
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(min(offset, size))
        chunk = f.read(limit)
    # Never split a UTF-8 character: a partial one at the end waits for the next read.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text = decoder.decode(chunk, final=False)
    used = len(chunk) - len(decoder.getstate()[0])
    return {"text": text, "offset": offset, "next_offset": offset + used, "size": size}
