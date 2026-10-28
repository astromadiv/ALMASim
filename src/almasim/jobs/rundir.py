"""Run folders: where a job's manifest, status and logs live (web console plan, J1.1).

    <runs root>/<yyyy-mm-dd>/<job id>/
        job.json      the manifest (Job.to_dict()), written by whoever submits
        status.json   started / host / finished / exit code, written by the wrapper
        stdout.log, stderr.log
        dashboard.json  where the run's Dask dashboard is, while it serves one

The runs root may be shared by a group (the plan puts it under
/SKAO_data/cycle11): folders are made group-writable with the setgid bit, so
everything below keeps the group, and each file is written atomically (a
temporary file renamed over the old one), so a reader never sees half a file.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import Job, JobState, is_job_id

MANIFEST_NAME = "job.json"
STATUS_NAME = "status.json"
DASHBOARD_NAME = "dashboard.json"
# The run folder of the job a process belongs to, set by the wrapper for the
# command it runs, so code deep in a job (a Dask cluster) can leave a note there.
RUN_DIR_ENV = "ALMASIM_RUN_DIR"

# rwxrwsr-x: the group can add runs under a day folder; setgid keeps the group.
_DIR_MODE = 0o2775
_FILE_MODE = 0o664


class ManifestError(ValueError):
    """A run folder whose job.json is missing, unreadable or not a job."""


def _make_dir(path: Path) -> None:
    if path.is_dir():
        return
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(_DIR_MODE)
    except PermissionError:  # created by someone else in a race; theirs is fine
        pass


def create_run_dir(root: str | os.PathLike[str], job: Job) -> Path:
    """Make ``<root>/<date>/<job id>/`` for ``job`` (date from its creation, UTC)."""
    day = time.strftime("%Y-%m-%d", time.gmtime(job.created_at))
    root = Path(root)
    _make_dir(root)
    _make_dir(root / day)
    run_dir = root / day / job.id
    run_dir.mkdir()  # a second job with the same id is a bug: let it raise
    run_dir.chmod(_DIR_MODE)
    return run_dir


def _write_json(path: Path, data: dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.chmod(tmp, _FILE_MODE)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_manifest(run_dir: str | os.PathLike[str], job: Job) -> Path:
    path = Path(run_dir) / MANIFEST_NAME
    _write_json(path, job.to_dict())
    return path


def read_manifest(run_dir: str | os.PathLike[str]) -> Job:
    path = Path(run_dir) / MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Job.from_dict(data)
    except FileNotFoundError as exc:
        raise ManifestError(f"{path}: no manifest") from exc
    except (OSError, ValueError, TypeError) as exc:
        raise ManifestError(f"{path}: {exc}") from exc


@dataclass(frozen=True)
class RunStatus:
    """What the wrapper reported about one run; absent fields were not reached."""

    started_at: float | None = None
    host: str | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    signal: str | None = None  # e.g. "SIGTERM" when scancel stopped it


def read_status(run_dir: str | os.PathLike[str]) -> RunStatus:
    """The wrapper's status.json, or an empty status before it wrote one."""
    try:
        data = json.loads((Path(run_dir) / STATUS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return RunStatus()
    known = {k: data[k] for k in RunStatus.__dataclass_fields__ if k in data}
    return RunStatus(**known)


def write_status(run_dir: str | os.PathLike[str], status: RunStatus) -> None:
    fields = {k: v for k, v in status.__dict__.items() if v is not None}
    _write_json(Path(run_dir) / STATUS_NAME, fields)


@dataclass(frozen=True)
class DashboardAddress:
    """Where a running job's Dask dashboard listens (the console relays it)."""

    host: str
    port: int


def write_dashboard(run_dir: str | os.PathLike[str], address: DashboardAddress) -> None:
    _write_json(Path(run_dir) / DASHBOARD_NAME, {"host": address.host, "port": address.port})


def clear_dashboard(run_dir: str | os.PathLike[str]) -> None:
    (Path(run_dir) / DASHBOARD_NAME).unlink(missing_ok=True)


def read_dashboard(run_dir: str | os.PathLike[str]) -> DashboardAddress | None:
    """The run's dashboard address, or None when it has none.

    Only a note this user wrote counts: the console relays to whatever it
    names, and in a group-writable runs root anyone could write one.
    """
    path = Path(run_dir) / DASHBOARD_NAME
    try:
        if path.stat().st_uid != os.getuid():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return DashboardAddress(host=str(data["host"]), port=int(data["port"]))
    except (OSError, ValueError, TypeError, KeyError):
        return None


def current_state(job: Job, status: RunStatus) -> JobState:
    """The state the run folder shows, before asking Slurm.

    The wrapper's report wins over the manifest's last recorded state: it
    is written on the node that ran the command. Slurm's view (queued,
    lost) is merged in by the executor (J1.4).
    """
    if status.finished_at is not None:
        if status.signal in ("SIGTERM", "SIGINT"):
            return JobState.CANCELLED
        return JobState.SUCCEEDED if status.exit_code == 0 else JobState.FAILED
    if status.started_at is not None:
        return JobState.RUNNING
    return job.state


def iter_runs(root: str | os.PathLike[str], *, newest_first: bool = True) -> Iterator[Path]:
    """Every run folder under ``root``, by day then id (ids sort by time).

    Folders whose names are not dates or job ids are skipped, so a runs root
    can hold other files (a README, a team's notes).
    """
    root = Path(root)
    if not root.is_dir():
        return
    days = sorted(
        (p for p in root.iterdir() if p.is_dir() and _is_day(p.name)), reverse=newest_first
    )
    for day in days:
        try:
            runs = sorted(
                (p for p in day.iterdir() if p.is_dir() and is_job_id(p.name)),
                reverse=newest_first,
            )
        except PermissionError:  # a day folder another user made unreadable
            continue
        yield from runs


def _is_day(name: str) -> bool:
    try:
        time.strptime(name, "%Y-%m-%d")
    except ValueError:
        return False
    return len(name) == 10


def is_group_shared(path: str | os.PathLike[str]) -> bool:
    """Whether ``path`` has the setgid bit, i.e. runs under it keep the group."""
    return bool(Path(path).stat().st_mode & stat.S_ISGID)
