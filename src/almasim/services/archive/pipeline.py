"""Archive post-processing: extract, unpack and calibrate, one process per UID (MADIV-28 T3.5).

Moved out of ``cli/products.py``. Each UID's unpack or calibration runs in
its own subprocess (``python -m almasim.cli products unpack|calibrate`` for
that UID), in turn or on Slurm workers, so a CASA crash on one execution
block never takes the batch down. Progress goes to ``echo`` (called like
``typer.echo``; the log by default); bad inputs raise :class:`StageError`
or :class:`StageUsageError`, which the commands map to their exit codes.
"""

from __future__ import annotations

import logging
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep, time
from typing import Any, Protocol

from tqdm.auto import tqdm

from almasim.services.compute.progress import (
    FAILED_FUTURE_STATES,
    TaskFailure,
    error_headline,
    future_done,
    future_status,
    heartbeat_lines,
)
from almasim.services.runtime.env import casa_bundled_lib_dir as _casa_bundled_lib_dir
from almasim.services.runtime.env import casa_library_path
from almasim.services.utils import dedupe_keep_order

logger = logging.getLogger(__name__)


class Echo(Protocol):
    def __call__(self, message: str, *, err: bool = False) -> Any: ...


def _log_echo(message: str, *, err: bool = False) -> None:
    (logger.error if err else logger.info)(message)


class StageError(RuntimeError):
    """A stage cannot run: nothing to process (the command exits 1)."""


class StageUsageError(ValueError):
    """A stage was asked for an unsupported combination of options (exit 2)."""


@dataclass
class ArchiveStageResult:
    """What :func:`run_archive_stages` produced, and the UIDs that failed."""

    raw_ms: list[str] = field(default_factory=list)
    calibrated_ms: list[str] = field(default_factory=list)
    failures: list[TaskFailure] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def create_backend(*args, **kwargs):
    from almasim.services.compute import create_backend as _create_backend

    return _create_backend(*args, **kwargs)


_NO_PROGRESS_TIMEOUT_S = 3600


_PROGRESS_HEARTBEAT_S = 60.0


def _safe_uid(uid: str) -> str:
    return uid.replace("/", "_").replace(":", "_")


def _stage_log_path(output_root: Any, uid: str, stage: str) -> Path:
    """Per-UID log file written by the worker subprocess wrappers.

    The file lives under ``<output_root>/logs`` so that, on Slurm, the submit
    node can follow it while the job runs on a compute node.
    """
    return Path(str(output_root)).expanduser().resolve() / "logs" / f"{_safe_uid(uid)}.{stage}.log"


def _compute_jobs_with_progress(  # noqa: C901
    *,
    backend: Any,
    jobs: list[Any],
    job_uids: list[str],
    stage_label: str,
    no_progress_timeout: float = _NO_PROGRESS_TIMEOUT_S,
    continue_on_error: bool = False,
    log_paths: list[Path] | None = None,
    failures: list[TaskFailure] | None = None,
    heartbeat_interval: float = _PROGRESS_HEARTBEAT_S,
) -> list[Any]:
    """Run a stage and show per-UID progress for asynchronous backends.

    With ``continue_on_error`` the result list contains ``None`` for every job
    that failed (details are appended to ``failures``) instead of raising on the
    first failure. When ``log_paths`` is given, the last line of every running
    UID's log is printed every ``heartbeat_interval`` seconds so long stages are
    not silent.
    """
    if not jobs:
        return []

    futures = backend.compute(jobs, sync=False)
    if not isinstance(futures, list):
        futures = [futures]

    def _uid(index: int) -> str:
        return job_uids[index] if index < len(job_uids) else f"job-{index + 1}"

    def _log_path(index: int) -> Path | None:
        if log_paths is not None and index < len(log_paths):
            return log_paths[index]
        return None

    def _record_failure(index: int, error: str) -> None:
        if failures is not None:
            log_path = _log_path(index)
            failures.append(
                TaskFailure(_uid(index), error, str(log_path) if log_path is not None else None)
            )

    completed: set[int] = set()
    failed: set[int] = set()
    stage_started_at = time()
    last_progress_time = stage_started_at
    last_heartbeat_time = stage_started_at
    with tqdm(total=len(futures), desc=stage_label, unit="uid", leave=True) as progress_bar:
        progress_bar.set_postfix_str(f"completed 0/{len(futures)}")
        while len(completed) < len(futures):
            prev_completed = len(completed)
            for index, future in enumerate(futures):
                if index in completed:
                    continue

                state = future_status(future)
                if state in FAILED_FUTURE_STATES:
                    failed.add(index)

                if not future_done(future):
                    continue

                completed.add(index)
                progress_bar.update(1)
                uid = _uid(index)
                status_text = state if state else "finished"
                progress_bar.write(f"{stage_label} completed for {uid} [{status_text}]")
                if index in failed:
                    error_text = f"job ended in state {status_text!r}"
                    try:
                        exc = future.exception()
                        if exc is not None:
                            error_text = str(exc)
                    except Exception:  # noqa: BLE001 - the job's own error is best effort
                        pass
                    _record_failure(index, error_text)
                    progress_bar.write(f"  {uid} error: {error_headline(error_text)}")
                    log_path = _log_path(index)
                    if log_path is not None:
                        progress_bar.write(f"  {uid} log: {log_path}")

            if len(completed) > prev_completed:
                last_progress_time = time()

            progress_bar.set_postfix_str(
                f"completed {len(completed)}/{len(futures)} failed {len(failed)}"
            )
            if len(completed) < len(futures):
                now = time()
                if now - last_progress_time > no_progress_timeout:
                    pending = len(futures) - len(completed)
                    raise RuntimeError(
                        f"{stage_label}: {pending} future(s) made no progress for "
                        f"{no_progress_timeout:.0f}s — workers may have died. "
                        "Re-run with --skip-existing to resume."
                    )
                if log_paths is not None and now - last_heartbeat_time >= heartbeat_interval:
                    last_heartbeat_time = now
                    lines, running = heartbeat_lines(
                        (
                            (_uid(i), _log_path(i))
                            for i in range(len(futures))
                            if i not in completed
                        ),
                        started_at=stage_started_at,
                    )
                    for line in lines:
                        progress_bar.write(line)
                    queued = len(futures) - len(completed) - running
                    progress_bar.write(
                        f"{stage_label}: {running} running, {queued} waiting for a worker, "
                        f"{len(completed)} done"
                    )
                sleep(0.5)

    if not continue_on_error:
        return backend.gather(futures)

    results: list[Any] = []
    for index, future in enumerate(futures):
        if index in failed:
            results.append(None)
            continue
        try:
            results.append(backend.gather([future])[0])
        except Exception as exc:  # noqa: BLE001 - a failed job is recorded, not raised
            failed.add(index)
            _record_failure(index, str(exc))
            results.append(None)
    return results


def _run_uid_stage(
    *,
    backend: Any,
    postprocess_backend: str,
    task_fn: Any,
    job_kwargs: list[dict[str, Any]],
    job_uids: list[str],
    stage_label: str,
    log_paths: list[Path],
    continue_on_error: bool,
    failures: list[TaskFailure],
    echo: Echo = _log_echo,
) -> list[list[str]]:
    """Run one per-UID task for every UID, returning one output list per UID.

    Failed UIDs yield an empty list and are recorded in ``failures`` when
    ``continue_on_error`` is set; otherwise the first failure propagates.
    """
    if not job_uids:
        return []

    if postprocess_backend == "slurm":
        task = backend.delayed(task_fn)
        jobs = [task(**kwargs) for kwargs in job_kwargs]
        results = _compute_jobs_with_progress(
            backend=backend,
            jobs=jobs,
            job_uids=job_uids,
            stage_label=stage_label,
            continue_on_error=continue_on_error,
            log_paths=log_paths,
            failures=failures,
        )
        return [list(result) if result else [] for result in results]

    outputs: list[list[str]] = []
    total = len(job_uids)
    for index, (uid, kwargs) in enumerate(zip(job_uids, job_kwargs, strict=True), start=1):
        log_path = log_paths[index - 1] if index - 1 < len(log_paths) else None
        echo(
            f"{stage_label} [{index}/{total}] starting {uid}"
            + (f" (log: {log_path})" if log_path else "")
        )
        try:
            result = task_fn(**kwargs)
        except Exception as exc:
            if not continue_on_error:
                raise
            failures.append(TaskFailure(uid, str(exc), str(log_path) if log_path else None))
            echo(
                f"{stage_label} [{index}/{total}] FAILED {uid}: {error_headline(str(exc))}",
                err=True,
            )
            outputs.append([])
            continue
        echo(f"{stage_label} [{index}/{total}] finished {uid}")
        outputs.append(list(result))
    return outputs


def _extract_asdm_uids_from_download_root(download_root: Path) -> list[str]:
    from almasim.services.archive import find_asdm_directories

    asdm_dirs = find_asdm_directories(download_root)
    return [path.name.removesuffix(".asdm.sdm") for path in asdm_dirs]


def _extract_uids_from_raw_ms_root(raw_ms_root: Path) -> list[str]:
    from almasim.services.archive.calibration import find_raw_ms_directories

    return [path.name.removesuffix(".ms") for path in find_raw_ms_directories(raw_ms_root)]


def _stage_child_env() -> tuple[Path, dict[str, str]]:
    """The project root and environment for an unpack or calibrate worker child.

    ``PYTHONPATH`` finds this checkout's sources, ``PYTHONUNBUFFERED`` keeps
    the streamed log live, and ``LD_LIBRARY_PATH`` puts the system libraries
    first (GLIBC conflicts with spack builds) but casatools' bundled libraries
    ahead of them: the system libpmix otherwise shadows the bundled one and
    breaks the CASA extension modules.
    """
    import os

    project_root = Path(__file__).resolve().parents[4]
    src_root = project_root / "src"
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_root}:{existing_pythonpath}" if existing_pythonpath else str(src_root)
    )
    env["LD_LIBRARY_PATH"] = casa_library_path(
        env.get("LD_LIBRARY_PATH", ""), _casa_bundled_lib_dir()
    )
    env["PYTHONUNBUFFERED"] = "1"
    return project_root, env


def _worker_failure(return_code: int, tail_lines: Sequence[str]) -> str:
    """``Worker exited with return code N``, plus the last crash line if there is one."""
    error = f"Worker exited with return code {return_code}"
    crash = next(
        (line for line in reversed(tail_lines) if "what():" in line or "Error:" in line),
        "",
    ).strip()
    return f"{error}: {crash}" if crash else error


def _unpack_single_uid(
    *,
    input_root: str,
    raw_output_root: str,
    asdm_uid: str,
    casa_data_root: str | None,
    skip_casa_data_update: bool,
    overwrite: bool,
) -> list[str]:
    """Run one UID's ASDM import in a fresh subprocess.

    CASA's importasdm maintains process-level global state that is not reset
    between calls in the same Python process. Running as a subprocess guarantees
    a clean CASA environment for every UID, matching the approach used by
    _calibrate_single_uid.
    """
    import sys
    from pathlib import Path as _Path

    from almasim.services.runtime.child import run_stage_subprocess

    # Point workers at the pre-downloaded CASA runtime data so they never try
    # to download it themselves (compute nodes typically have no internet).
    effective_casa_data = casa_data_root or str(_Path(raw_output_root) / ".casa-data")

    cmd = [
        sys.executable,
        "-m",
        "almasim.cli",
        "products",
        "unpack",
        "--input-root",
        input_root,
        "--output-dir",
        raw_output_root,
        "--asdm-uid",
        asdm_uid,
        "--backend",
        "sync",
        "--casa-data-root",
        effective_casa_data,
        "--skip-casa-data-update",
    ]
    if overwrite:
        cmd.append("--overwrite-outputs")

    project_root, env = _stage_child_env()

    # Mirror everything the child prints into a per-UID log on the (shared) output
    # filesystem so the run can be followed with ``tail -f`` even on Slurm.
    log_path = _stage_log_path(raw_output_root, asdm_uid, "unpack")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    return_code, tail_lines = run_stage_subprocess(cmd, log_path, cwd=project_root, env=env)
    if return_code != 0:
        tail_text = "\n".join(tail_lines)
        # importasdm can be killed outright (OOM, walltime), so the in-process
        # handler never runs. This is the only place a marker can be written for
        # that case -- and without one, the half-written MS directory left behind
        # looks finished to the next run.
        from almasim.services.archive.unpack import write_raw_ms_failure_marker

        error = _worker_failure(return_code, tail_lines)
        try:
            write_raw_ms_failure_marker(raw_output_root, asdm_uid, error, log_path=log_path)
        except OSError as marker_error:  # never mask the unpack failure
            print(f"Could not write failure marker for {asdm_uid}: {marker_error}", flush=True)
        raise RuntimeError(
            f"Unpack failed for {asdm_uid}.\n"
            f"Return code: {return_code}\n"
            f"Full log: {log_path}\n"
            f"Last {len(tail_lines)} log lines:\n{tail_text}"
        )

    # Discover output paths produced by the subprocess.
    working_dir = _Path(raw_output_root) / "working"
    expected = working_dir / f"{asdm_uid}.ms"
    if expected.is_dir():
        return [str(expected)]
    # Fallback: scan for any MS produced for this UID.
    return [str(p) for p in working_dir.glob(f"{asdm_uid}*.ms") if p.is_dir()]


def _calibrate_single_uid(
    *,
    input_root: str,
    raw_ms_root: str,
    calibrated_output_root: str,
    asdm_uid: str,
    casa_data_root: str | None,
    skip_casa_data_update: bool,
    overwrite: bool,
    clean_intermediate: bool,
    keep_working_copies: bool = False,
) -> list[str]:
    """Run one UID's calibration in a fresh subprocess.

    CASA's calibrater tool maintains process-level global state that is not
    reset between calls in the same Python process. Running as a subprocess
    guarantees a clean CASA environment for every UID.
    """
    import sys
    from pathlib import Path as _Path

    from almasim.services.runtime.child import run_stage_subprocess

    # Point workers at the pre-downloaded CASA runtime data so they never try
    # to download it themselves (compute nodes typically have no internet).
    effective_casa_data = casa_data_root or str(_Path(calibrated_output_root) / ".casa-data")

    cmd = [
        sys.executable,
        "-m",
        "almasim.cli",
        "products",
        "calibrate",
        "--input-root",
        input_root,
        "--raw-ms-root",
        raw_ms_root,
        "--output-dir",
        calibrated_output_root,
        "--asdm-uid",
        asdm_uid,
        "--backend",
        "sync",
        "--casa-data-root",
        effective_casa_data,
        "--skip-casa-data-update",
    ]
    if overwrite:
        cmd.append("--overwrite-outputs")
    if keep_working_copies:
        cmd.append("--keep-working-copies")

    project_root, env = _stage_child_env()

    # Mirror everything the child prints into a per-UID log on the (shared) output
    # filesystem so the run can be followed with ``tail -f`` even on Slurm.
    log_path = _stage_log_path(calibrated_output_root, asdm_uid, "calibrate")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    return_code, tail_lines = run_stage_subprocess(cmd, log_path, cwd=project_root, env=env)
    if return_code != 0:
        tail_text = "\n".join(tail_lines)
        # An uncaught casacore C++ exception kills the child outright, so the
        # in-process handler never runs: this is the only place a marker can be
        # written for that class of failure, and the only place the abandoned
        # working copy can be reclaimed.
        from almasim.services.archive.calibration import write_calibration_failure_marker

        error = _worker_failure(return_code, tail_lines)
        try:
            write_calibration_failure_marker(
                calibrated_output_root, asdm_uid, error, log_path=log_path
            )
        except OSError as marker_error:  # never mask the calibration failure
            print(f"Could not write failure marker for {asdm_uid}: {marker_error}", flush=True)
        if not keep_working_copies:
            working_dir = _Path(calibrated_output_root) / "working" / f"{asdm_uid}.calibration"
            if working_dir.is_dir():
                import shutil

                shutil.rmtree(working_dir, ignore_errors=True)
        raise RuntimeError(
            f"Calibration failed for {asdm_uid}.\n"
            f"Return code: {return_code}\n"
            f"Full log: {log_path}\n"
            f"Last {len(tail_lines)} log lines:\n{tail_text}"
        )

    # Discover output paths produced by the subprocess.
    output_path = _Path(calibrated_output_root)
    expected = output_path / f"{asdm_uid}.ms.split.cal"
    if expected.is_dir():
        return [str(expected)]
    # Fallback: scan for any split.cal produced for this UID.
    return [str(p) for p in output_path.glob(f"{asdm_uid}*.split.cal") if p.is_dir()]


def _preflight_casa_data(
    output_root: Path,
    casa_data_root: Path | None,
    skip_casa_data_update: bool,
    echo: Echo = _log_echo,
) -> Path:
    """Ensure CASA runtime data is populated on the master node before workers start.

    Slurm compute nodes typically have no internet access, so the data must be
    downloaded once from the submit node and written to a shared filesystem path
    that all workers can read. Returns the directory the workers should use.
    """
    from almasim.services.casa.runtime_data import ensure_casa_runtime_data, find_existing_casa_data

    casa_data = find_existing_casa_data(output_root, output_root, casa_data_root)
    echo(f"Preflight: ensuring CASA runtime data at {casa_data} …")
    ensure_casa_runtime_data(casa_data, skip_update=skip_casa_data_update)
    return Path(casa_data)


def _one_uid_per_worker(postprocess_backend_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Make each Slurm worker run one UID at a time.

    dask-jobqueue gives a worker as many task slots as it has cores, so a
    10-worker/20-core cluster would run up to 200 UIDs at once: each one copies
    the whole raw MeasurementSet and starts its own CASA process, which floods
    the shared filesystem and the nodes' memory. The real work happens in a
    subprocess, so a single task slot per worker is enough and makes
    ``--slurm-workers`` mean what it says: the number of UIDs in flight.

    An explicit ``worker_extra_args`` from the caller is left untouched.
    """
    if "worker_extra_args" in postprocess_backend_kwargs:
        return postprocess_backend_kwargs
    return {**postprocess_backend_kwargs, "worker_extra_args": ["--nthreads", "1"]}


def _partition_completed_calibrations(
    output_root: Path, uids: list[str], overwrite: bool, echo: Echo = _log_echo
) -> tuple[list[str], list[str]]:
    """Split ``uids`` into those still to calibrate and the outputs of finished ones.

    A UID counts as finished when ``<uid>.ms.split.cal`` exists together with
    its ``.done`` marker; with ``overwrite`` everything is redone.
    """
    from almasim.services.archive import calibrated_output_path, is_calibration_complete

    if overwrite:
        return list(uids), []
    todo: list[str] = []
    done_outputs: list[str] = []
    for uid in uids:
        if is_calibration_complete(output_root, uid):
            done_outputs.append(str(calibrated_output_path(output_root, uid)))
        else:
            todo.append(uid)
    if done_outputs:
        echo(
            f"Skipping {len(done_outputs)} already calibrated UID(s) with completion markers "
            "(pass --overwrite-outputs to redo them)."
        )
    return todo, done_outputs


def _drop_uncalibrated_deliveries(
    input_root: Path, output_root: Path, uids: list[str], echo: Echo = _log_echo
) -> list[str]:
    """Drop the UIDs ALMA delivered without calibration; the ones left to calibrate.

    Such an execution block (QA0 SemiPass) has nothing to apply, so it is
    reported as skipped, not as a failure that fails the whole run.
    """
    from almasim.services.archive.calibration import (
        calibration_failure_marker_path,
        calibration_not_delivered,
    )

    todo: list[str] = []
    skipped: list[str] = []
    for uid in uids:
        (skipped if calibration_not_delivered(input_root, uid) else todo).append(uid)
    if skipped:
        echo(
            f"Skipping {len(skipped)} UID(s) that ALMA delivered without calibration "
            "(left out of the delivered pipeline run, e.g. QA0 SemiPass): " + ", ".join(skipped)
        )
        for uid in skipped:
            # A failure recorded by a run before this check no longer applies.
            calibration_failure_marker_path(output_root, uid).unlink(missing_ok=True)
    return todo


def _run_unpack_jobs(
    *,
    input_root: Path,
    output_root: Path,
    asdm_uids: list[str],
    postprocess_backend: str,
    postprocess_backend_kwargs: dict[str, Any],
    casa_data_root: Path | None,
    skip_casa_data_update: bool,
    overwrite_outputs: bool,
    continue_on_error: bool = True,
    failures: list[TaskFailure] | None = None,
    echo: Echo = _log_echo,
) -> list[str]:
    """Import ASDMs into raw MeasurementSets, one isolated subprocess per UID.

    A single UID (which is also how each worker subprocess re-enters this
    function) runs in-process. Several UIDs always run one subprocess each so a
    CASA crash on one execution block cannot take the rest of the batch down;
    failed UIDs are appended to ``failures`` when ``continue_on_error`` is set.
    """
    from almasim.services.archive import create_measurement_sets

    if failures is None:
        failures = []

    effective_uids = asdm_uids or dedupe_keep_order(
        _extract_asdm_uids_from_download_root(input_root)
    )
    if not effective_uids:
        raise StageError("No ASDM directories found to unpack.")

    if postprocess_backend == "sync" and len(effective_uids) == 1:
        return [
            str(path)
            for path in create_measurement_sets(
                input_root=input_root,
                output_root=output_root,
                asdm_uid=effective_uids[0],
                casa_data_root=casa_data_root,
                skip_casa_data_update=skip_casa_data_update,
                overwrite=overwrite_outputs,
                logger_fn=echo,
            )
        ]

    if postprocess_backend_kwargs.get("n_workers") == 0:
        postprocess_backend_kwargs = {
            **postprocess_backend_kwargs,
            "n_workers": len(effective_uids),
        }
    if postprocess_backend == "slurm":
        postprocess_backend_kwargs = _one_uid_per_worker(postprocess_backend_kwargs)

    worker_casa_data = _preflight_casa_data(
        output_root, casa_data_root, skip_casa_data_update, echo
    )
    skip_casa_data_update = True  # workers reuse what master just populated

    log_paths = [_stage_log_path(output_root, uid, "unpack") for uid in effective_uids]
    echo(f"Per-UID unpack logs: {log_paths[0].parent}/<uid>.unpack.log")
    job_kwargs = [
        {
            "input_root": str(input_root),
            "raw_output_root": str(output_root),
            "asdm_uid": uid,
            "casa_data_root": str(worker_casa_data),
            "skip_casa_data_update": skip_casa_data_update,
            "overwrite": overwrite_outputs,
        }
        for uid in effective_uids
    ]

    stage_label = "Slurm unpack" if postprocess_backend == "slurm" else "Unpack"
    if postprocess_backend == "slurm":
        with create_backend(postprocess_backend, **postprocess_backend_kwargs) as backend:
            unpack_results = _run_uid_stage(
                backend=backend,
                postprocess_backend=postprocess_backend,
                task_fn=_unpack_single_uid,
                job_kwargs=job_kwargs,
                job_uids=effective_uids,
                stage_label=stage_label,
                log_paths=log_paths,
                continue_on_error=continue_on_error,
                failures=failures,
                echo=echo,
            )
    else:
        unpack_results = _run_uid_stage(
            backend=None,
            postprocess_backend=postprocess_backend,
            task_fn=_unpack_single_uid,
            job_kwargs=job_kwargs,
            job_uids=effective_uids,
            stage_label=stage_label,
            log_paths=log_paths,
            continue_on_error=continue_on_error,
            failures=failures,
            echo=echo,
        )
    outputs: list[str] = []
    for result in unpack_results:
        outputs.extend(result)
    return outputs


def _run_calibrate_jobs(  # noqa: C901
    *,
    input_root: Path,
    raw_ms_root: Path,
    output_root: Path,
    asdm_uids: list[str],
    postprocess_backend: str,
    postprocess_backend_kwargs: dict[str, Any],
    casa_data_root: Path | None,
    skip_casa_data_update: bool,
    overwrite_outputs: bool,
    clean_intermediate: bool,
    continue_on_error: bool = True,
    failures: list[TaskFailure] | None = None,
    keep_working_copies: bool = False,
    echo: Echo = _log_echo,
) -> list[str]:
    """Calibrate raw MeasurementSets, one isolated subprocess per UID.

    UIDs whose calibrated output already carries a completion marker are skipped
    and reported as outputs, so an interrupted batch can simply be re-run.

    A single UID (which is also how each worker subprocess re-enters this
    function) runs in-process. Several UIDs always run one subprocess each so a
    CASA crash (for example an uncaught ``casacore::ArrayError``) on one
    execution block cannot take the rest of the batch down; failed UIDs are
    appended to ``failures`` when ``continue_on_error`` is set.
    """
    from almasim.services.archive import (
        cleanup_intermediate_calibration_data,
        create_calibrated_measurement_sets,
    )

    if failures is None:
        failures = []

    if clean_intermediate and postprocess_backend == "slurm":
        raise StageUsageError(
            "--clean-intermediate-files is not supported with --postprocess-backend=slurm."
        )

    effective_uids = asdm_uids or dedupe_keep_order(_extract_uids_from_raw_ms_root(raw_ms_root))
    if not effective_uids:
        raise StageError("No raw MeasurementSets found to calibrate.")
    effective_uids = _drop_uncalibrated_deliveries(input_root, output_root, effective_uids, echo)
    if not effective_uids:
        echo("No UID left to calibrate.")
        return []

    if postprocess_backend == "sync" and len(effective_uids) == 1:
        return [
            str(path)
            for path in create_calibrated_measurement_sets(
                input_root=input_root,
                raw_ms_root=raw_ms_root,
                output_root=output_root,
                asdm_uid=effective_uids[0],
                casa_data_root=casa_data_root,
                skip_casa_data_update=skip_casa_data_update,
                overwrite=overwrite_outputs,
                clean_intermediate=clean_intermediate,
                logger_fn=echo,
                remove_working_copy=not keep_working_copies,
            )
        ]

    effective_uids, done_outputs = _partition_completed_calibrations(
        output_root, effective_uids, overwrite_outputs, echo
    )
    if not effective_uids:
        echo("All requested UIDs are already calibrated; nothing to do.")
        return done_outputs

    if postprocess_backend_kwargs.get("n_workers") == 0:
        postprocess_backend_kwargs = {
            **postprocess_backend_kwargs,
            "n_workers": len(effective_uids),
        }
    if postprocess_backend == "slurm":
        postprocess_backend_kwargs = _one_uid_per_worker(postprocess_backend_kwargs)

    worker_casa_data = _preflight_casa_data(
        output_root, casa_data_root, skip_casa_data_update, echo
    )
    skip_casa_data_update = True  # workers reuse what master just populated

    log_paths = [_stage_log_path(output_root, uid, "calibrate") for uid in effective_uids]
    echo(f"Per-UID calibration logs: {log_paths[0].parent}/<uid>.calibrate.log")
    echo(f"Calibrating {len(effective_uids)} UID(s)")
    job_kwargs = [
        {
            "input_root": str(input_root),
            "raw_ms_root": str(raw_ms_root),
            "calibrated_output_root": str(output_root),
            "asdm_uid": uid,
            "casa_data_root": str(worker_casa_data),
            "skip_casa_data_update": skip_casa_data_update,
            "overwrite": overwrite_outputs,
            "clean_intermediate": False,
            "keep_working_copies": keep_working_copies,
        }
        for uid in effective_uids
    ]

    stage_label = "Slurm calibrate" if postprocess_backend == "slurm" else "Calibrate"
    if postprocess_backend == "slurm":
        with create_backend(postprocess_backend, **postprocess_backend_kwargs) as backend:
            calibrate_results = _run_uid_stage(
                backend=backend,
                postprocess_backend=postprocess_backend,
                task_fn=_calibrate_single_uid,
                job_kwargs=job_kwargs,
                job_uids=effective_uids,
                stage_label=stage_label,
                log_paths=log_paths,
                continue_on_error=continue_on_error,
                failures=failures,
                echo=echo,
            )
    else:
        calibrate_results = _run_uid_stage(
            backend=None,
            postprocess_backend=postprocess_backend,
            task_fn=_calibrate_single_uid,
            job_kwargs=job_kwargs,
            job_uids=effective_uids,
            stage_label=stage_label,
            log_paths=log_paths,
            continue_on_error=continue_on_error,
            failures=failures,
            echo=echo,
        )
    outputs: list[str] = list(done_outputs)
    for result in calibrate_results:
        outputs.extend(result)

    if clean_intermediate:
        if failures:
            echo(
                "Skipping --clean-intermediate-files because "
                f"{len(failures)} UID(s) failed; raw data is kept for a re-run.",
                err=True,
            )
        else:
            cleanup_intermediate_calibration_data(
                raw_ms_root,
                output_root,
                [Path(path) for path in outputs],
                logger_fn=echo,
            )
    return outputs


def _safe_extract_tar_archive(
    archive_path: Path, destination: Path, echo: Echo = _log_echo
) -> list[Path]:
    """Extract a tarball while refusing absolute or escaping paths.

    If the archive is flat (no common top-level directory), all contents are
    placed inside a subdirectory named after the archive stem so they never
    spill into the destination root. Extraction is atomic per directory (see
    :mod:`almasim.services.extraction`), so a concurrent unpack never sees a
    half-extracted ASDM.
    """
    from almasim.services.extraction import extract_tar_atomically

    return extract_tar_atomically(
        archive_path,
        destination,
        flat_archive_subdir=True,
        on_skipped_member=lambda name: echo(f"Skipping unsafe archive member: {name}", err=True),
    )


def _find_archives(root: Path, recursive: bool) -> list[Path]:
    candidates = root.rglob("*") if recursive else root.iterdir()
    archives = []
    for path in candidates:
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(".tar") or name.endswith(".tgz") or name.endswith(".tar.gz"):
            archives.append(path)
    return sorted(archives)


def _archive_done_marker(archive_path: Path) -> Path:
    return archive_path.parent / (archive_path.name + ".done")


def _extract_single_archive(
    *,
    archive_path: str,
    destination: str,
    delete_archive: bool,
) -> list[str]:
    """Extract one tarball on a Slurm worker; optionally delete it afterwards.

    Imports lazily so cloudpickle ships only a reference to the installed
    ``almasim`` package rather than this module's globals.
    """
    from pathlib import Path as _Path

    from almasim.services.extraction import extract_tar_atomically

    archive = _Path(archive_path)
    extracted = [
        str(path)
        for path in extract_tar_atomically(archive, _Path(destination), flat_archive_subdir=True)
    ]

    # Write marker so re-runs with --skip-existing can detect completion.
    (_Path(archive_path + ".done")).write_text("")

    if delete_archive:
        archive.unlink(missing_ok=True)
    return extracted


def _run_extract_jobs(
    *,
    source: Path,
    target: Path,
    archives: list[Path],
    postprocess_backend: str,
    postprocess_backend_kwargs: dict[str, Any],
    delete_archives: bool,
    skip_existing: bool = False,
    echo: Echo = _log_echo,
) -> tuple[list[str], list[str]]:
    if skip_existing:
        pending = [a for a in archives if not _archive_done_marker(a).exists()]
        skipped = len(archives) - len(pending)
        if skipped:
            echo(f"Skipped {skipped} already-extracted archive(s).")
        archives = pending

    if not archives:
        echo("Nothing to extract.")
        return [], []

    if postprocess_backend == "sync":
        extracted_files: list[str] = []
        failed_archives: list[str] = []
        for archive_path in archives:
            try:
                files = _safe_extract_tar_archive(archive_path, target, echo)
                extracted_files.extend(str(p) for p in files)
                echo(f"Extracted {archive_path}")
                _archive_done_marker(archive_path).write_text("")
                if delete_archives:
                    archive_path.unlink(missing_ok=True)
            except (tarfile.TarError, OSError, ValueError) as exc:
                failed_archives.append(str(archive_path))
                echo(f"Failed to extract {archive_path}: {exc}", err=True)
        return extracted_files, failed_archives

    archive_labels = [a.name for a in archives]
    if postprocess_backend_kwargs.get("n_workers") == 0:
        postprocess_backend_kwargs = {
            **postprocess_backend_kwargs,
            "n_workers": len(archives),
        }

    with create_backend(postprocess_backend, **postprocess_backend_kwargs) as backend:
        extract_task = backend.delayed(_extract_single_archive)
        extract_jobs = [
            extract_task(
                archive_path=str(archive_path),
                destination=str(target),
                delete_archive=delete_archives,
            )
            for archive_path in archives
        ]
        extract_results = _compute_jobs_with_progress(
            backend=backend,
            jobs=extract_jobs,
            job_uids=archive_labels,
            stage_label="Slurm extract",
        )

    extracted_files = []
    for result in extract_results:
        extracted_files.extend(result)
    return extracted_files, []


def _run_parallel_archive_jobs(  # noqa: C901
    *,
    download_root: Path,
    archive_output_root: Path,
    unpack_ms: bool,
    generate_calibrated_visibilities: bool,
    postprocess_backend: str,
    postprocess_backend_kwargs: dict[str, Any],
    casa_data_root: Path | None,
    skip_casa_data_update: bool,
    overwrite_archive_outputs: bool,
    continue_on_error: bool = True,
    failures: list[TaskFailure] | None = None,
    keep_working_copies: bool = False,
    echo: Echo = _log_echo,
) -> tuple[list[str], list[str]]:
    if failures is None:
        failures = []
    raw_ms_root = archive_output_root / "raw_ms"
    calibrated_ms_root = archive_output_root / "calibrated_ms"

    if unpack_ms:
        asdm_uids = _extract_asdm_uids_from_download_root(download_root)
    elif generate_calibrated_visibilities:
        asdm_uids = _extract_uids_from_raw_ms_root(raw_ms_root)
    else:
        return [], []

    asdm_uids = dedupe_keep_order(asdm_uids)
    if not asdm_uids:
        raise StageError("No ASDM/raw-MS inputs found for archive post-processing.")

    if postprocess_backend_kwargs.get("n_workers") == 0:
        postprocess_backend_kwargs = {
            **postprocess_backend_kwargs,
            "n_workers": len(asdm_uids),
        }
    if postprocess_backend == "slurm":
        postprocess_backend_kwargs = _one_uid_per_worker(postprocess_backend_kwargs)

    echo(
        "Running archive post-processing with "
        f"backend={postprocess_backend} for {len(asdm_uids)} UID(s)..."
    )
    raw_outputs: list[str] = []
    calibrated_outputs: list[str] = []

    # One CASA runtime data directory for both stages, populated on the submit node.
    worker_casa_data = _preflight_casa_data(
        archive_output_root, casa_data_root, skip_casa_data_update, echo
    )
    skip_casa_data_update = True

    stage_prefix = "Slurm " if postprocess_backend == "slurm" else ""
    with create_backend(postprocess_backend, **postprocess_backend_kwargs) as backend:
        calibrate_uids = list(asdm_uids)
        if unpack_ms:
            unpack_log_paths = [_stage_log_path(raw_ms_root, uid, "unpack") for uid in asdm_uids]
            echo(f"Per-UID unpack logs: {unpack_log_paths[0].parent}/<uid>.unpack.log")
            unpack_results = _run_uid_stage(
                backend=backend,
                postprocess_backend=postprocess_backend,
                task_fn=_unpack_single_uid,
                job_kwargs=[
                    {
                        "input_root": str(download_root),
                        "raw_output_root": str(raw_ms_root),
                        "asdm_uid": uid,
                        "casa_data_root": str(worker_casa_data),
                        "skip_casa_data_update": skip_casa_data_update,
                        "overwrite": overwrite_archive_outputs,
                    }
                    for uid in asdm_uids
                ],
                job_uids=asdm_uids,
                stage_label=f"{stage_prefix}unpack",
                log_paths=unpack_log_paths,
                continue_on_error=continue_on_error,
                failures=failures,
                echo=echo,
            )
            calibrate_uids = []
            for uid, result in zip(asdm_uids, unpack_results, strict=True):
                raw_outputs.extend(result)
                if result:
                    calibrate_uids.append(uid)
            skipped = [uid for uid in asdm_uids if uid not in calibrate_uids]
            if skipped and generate_calibrated_visibilities:
                echo(
                    f"Skipping calibration for {len(skipped)} UID(s) whose unpack failed: "
                    + ", ".join(skipped),
                    err=True,
                )

        if generate_calibrated_visibilities and calibrate_uids:
            calibrate_uids, done_outputs = _partition_completed_calibrations(
                calibrated_ms_root, calibrate_uids, overwrite_archive_outputs, echo
            )
            calibrated_outputs.extend(done_outputs)
            calibrate_uids = _drop_uncalibrated_deliveries(
                download_root, calibrated_ms_root, calibrate_uids, echo
            )

        if generate_calibrated_visibilities and calibrate_uids:
            calibrate_log_paths = [
                _stage_log_path(calibrated_ms_root, uid, "calibrate") for uid in calibrate_uids
            ]
            echo(f"Per-UID calibration logs: {calibrate_log_paths[0].parent}/<uid>.calibrate.log")
            calibrate_results = _run_uid_stage(
                backend=backend,
                postprocess_backend=postprocess_backend,
                task_fn=_calibrate_single_uid,
                job_kwargs=[
                    {
                        "input_root": str(download_root),
                        "raw_ms_root": str(raw_ms_root),
                        "calibrated_output_root": str(calibrated_ms_root),
                        "asdm_uid": uid,
                        "casa_data_root": str(worker_casa_data),
                        "skip_casa_data_update": skip_casa_data_update,
                        "overwrite": overwrite_archive_outputs,
                        "clean_intermediate": False,
                        "keep_working_copies": keep_working_copies,
                    }
                    for uid in calibrate_uids
                ],
                job_uids=calibrate_uids,
                stage_label=f"{stage_prefix}calibrate",
                log_paths=calibrate_log_paths,
                continue_on_error=continue_on_error,
                failures=failures,
                echo=echo,
            )
            for result in calibrate_results:
                calibrated_outputs.extend(result)

    return raw_outputs, calibrated_outputs


def run_archive_stages(
    download_root: Path,
    archive_output_root: Path,
    *,
    unpack_ms: bool = True,
    calibrate: bool = True,
    postprocess_backend: str = "sync",
    postprocess_backend_kwargs: dict[str, Any] | None = None,
    casa_data_root: Path | None = None,
    skip_casa_data_update: bool = False,
    overwrite: bool = False,
    keep_working_copies: bool = False,
    echo: Echo = _log_echo,
) -> ArchiveStageResult:
    """Unpack and calibrate every UID under ``download_root``, one subprocess per UID.

    Raw MSs go to ``<archive_output_root>/raw_ms`` and calibrated ones to
    ``calibrated_ms``. A failed UID is recorded in the result's ``failures``
    (with a ``.failed`` marker next to its output) and the others go on.
    """
    result = ArchiveStageResult()
    result.raw_ms, result.calibrated_ms = _run_parallel_archive_jobs(
        download_root=Path(download_root),
        archive_output_root=Path(archive_output_root),
        unpack_ms=unpack_ms,
        generate_calibrated_visibilities=calibrate,
        postprocess_backend=postprocess_backend,
        postprocess_backend_kwargs=dict(postprocess_backend_kwargs or {}),
        casa_data_root=casa_data_root,
        skip_casa_data_update=skip_casa_data_update,
        overwrite_archive_outputs=overwrite,
        continue_on_error=True,
        failures=result.failures,
        keep_working_copies=keep_working_copies,
        echo=echo,
    )
    return result
