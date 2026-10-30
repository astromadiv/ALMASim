"""Run imaging tasks in process or on Slurm workers (MADIV-28 T3.6).

One failed task never ends the batch unless asked to.

Moved out of ``cli/image.py``; the command maps :class:`ImagingRunAborted`
and :class:`WorkersUnavailableError` to its exit codes.
"""

from __future__ import annotations

import logging
from time import sleep, time
from typing import Any, Protocol

from tqdm import tqdm

from almasim.services.compute.progress import TaskFailure, error_headline, heartbeat_lines
from almasim.services.imaging.options import ExecOptions, ImagingOptions
from almasim.services.imaging.wsclean_plan import ImagingTask

logger = logging.getLogger(__name__)


class Echo(Protocol):
    def __call__(self, message: str, err: bool = False) -> Any: ...


def _log_echo(message: str, err: bool = False) -> None:
    (logger.error if err else logger.info)(message)


class ImagingRunAborted(RuntimeError):
    """A task failed and the run was not to continue past failures."""


class WorkersUnavailableError(RuntimeError):
    """The Slurm workers did not come up."""


# How long the Slurm driver waits for its workers before it starts anyway.
_WORKER_WAIT_S = 1800.0


_IMAGING_HEARTBEAT_S = 60.0


def _record_imaging_failure(task: ImagingTask, error: str, failures: list[TaskFailure]) -> None:
    """Record a failed task and make sure a ``.failed`` marker exists for it.

    The worker-side wrapper writes the marker itself; this covers the cases
    where it never ran or was lost with its worker (walltime, node failure).
    """
    from almasim.services.imaging.markers import (
        imaging_failure_marker_path,
        imaging_log_path,
        imaging_marker_path,
        write_imaging_failure_marker,
    )

    log_path = imaging_log_path(task.output_dir)
    if (
        not imaging_failure_marker_path(task.output_dir).is_file()
        and not imaging_marker_path(task.output_dir).is_file()
    ):
        try:
            write_imaging_failure_marker(
                task.output_dir,
                ms_path=str(task.ms_path),
                spw=task.spw,
                error=error,
                log_path=log_path if log_path.is_file() else None,
            )
        except OSError:
            pass
    failures.append(TaskFailure(task.label, error, str(log_path) if log_path.is_file() else None))


def run_imaging_tasks(  # noqa: C901
    tasks: list[ImagingTask],
    imaging: ImagingOptions,
    execution: ExecOptions,
    *,
    casa_data: str | None = None,
    echo: Echo = _log_echo,
    failures: list[TaskFailure] | None = None,
    heartbeat_interval: float = _IMAGING_HEARTBEAT_S,
) -> list[dict[str, Any]]:
    """Run WSClean tasks, one subprocess each, and never let one failure end the run.

    Every task ends with a ``.done`` or ``.failed`` marker next to its output
    directory. With ``continue_on_error`` (the default) failed tasks are
    collected in ``failures`` and the rest of the batch keeps going; otherwise
    the first failure stops the run with :class:`ImagingRunAborted`. Progress
    goes to ``echo`` (called like ``typer.echo``; the log by default).
    """
    from almasim.services.imaging.markers import imaging_log_path
    from almasim.services.imaging.task import run_wsclean_task

    backend_kind, cores_per_task = execution.backend, execution.cores_per_task
    n_jobs, node_cores = execution.n_jobs, execution.node_cores
    continue_on_error = execution.continue_on_error
    # The keyword arguments every task gets (MADIV-28 T4.1).
    shared_kwargs: dict[str, Any] = {
        "threads": cores_per_task,
        "timeout": execution.task_timeout,
        "scratch_root": execution.scratch_root,
        "retries": execution.task_retries,
        **imaging.task_kwargs(casa_data),
    }

    if failures is None:
        failures = []
    if not tasks:
        return []

    results: list[dict[str, Any]] = []
    if backend_kind == "sync":
        for index, task in enumerate(tasks, start=1):
            log_path = imaging_log_path(task.output_dir)
            echo(f"Image [{index}/{len(tasks)}] {task.label} (log: {log_path})")
            try:
                results.append(
                    run_wsclean_task(
                        command=task.command,
                        output_dir=str(task.output_dir),
                        ms_path=str(task.ms_path),
                        spw=task.spw,
                        field_ids=list(task.field_ids),
                        cube_command=task.cube_command,
                        pb_dish_diameter=task.dish_diameter if imaging.pbcor else None,
                        provenance=task.provenance,
                        **shared_kwargs,
                    )
                )
            except Exception as exc:
                _record_imaging_failure(task, str(exc), failures)
                echo(f"  FAILED {task.label}: {error_headline(str(exc))}", err=True)
                if not continue_on_error:
                    raise ImagingRunAborted(task.label) from exc
        return results

    from almasim.services.compute import create_backend

    stage_label = "Slurm image"
    started_at = time()
    with create_backend(
        "slurm",
        queue=execution.queue,
        node_cores=node_cores,
        memory=execution.memory,
        walltime=execution.walltime,
        n_workers=n_jobs,
        project=execution.project,
        scheduler_host=execution.scheduler_host,
        scheduler_interface=execution.scheduler_interface,
    ) as backend:
        # Dask assigns a resource-restricted task to a worker the moment it is
        # submitted and does not move it later (tasks last hours, so work
        # stealing never sees a reason to). Submitting everything as soon as
        # the first worker registered put all 46 tasks of job 3879 on one
        # worker while four sat idle. So: wait for the workers, then keep only
        # as many tasks in flight as they have slots, topping up as each ends;
        # the scheduler then places every new task on a worker with room.
        wait_for_workers = getattr(backend, "wait_for_workers", None)
        if callable(wait_for_workers):
            echo(f"Waiting up to {_WORKER_WAIT_S / 60:.0f} min for {n_jobs} worker(s)...")
            try:
                ready = wait_for_workers(n_jobs, timeout=_WORKER_WAIT_S)
            except RuntimeError as exc:
                raise WorkersUnavailableError(str(exc)) from exc
            echo(f"{ready} worker(s) registered; starting.")
        slots = max(int(n_jobs), 1) * max(int(node_cores) // max(int(cores_per_task), 1), 1)

        def _submit(task: ImagingTask) -> Any:
            return backend.submit_callable(
                run_wsclean_task,
                cores=cores_per_task,
                command=task.command,
                output_dir=str(task.output_dir),
                ms_path=str(task.ms_path),
                spw=task.spw,
                field_ids=list(task.field_ids),
                cube_command=task.cube_command,
                pb_dish_diameter=task.dish_diameter if imaging.pbcor else None,
                provenance=task.provenance,
                **shared_kwargs,
            )

        queued = list(range(len(tasks)))
        pending: dict[int, Any] = {}
        while queued and len(pending) < slots:
            index = queued.pop(0)
            pending[index] = _submit(tasks[index])
        total = len(tasks)
        last_heartbeat = started_at
        with tqdm(total=total, desc=stage_label, unit="task", leave=True) as bar:
            while pending:
                progressed = False
                for index, future in list(pending.items()):
                    if not future.done():
                        continue
                    del pending[index]
                    progressed = True
                    bar.update(1)
                    if queued:
                        next_index = queued.pop(0)
                        pending[next_index] = _submit(tasks[next_index])
                    task = tasks[index]
                    error: str | None = None
                    try:
                        task_error = future.exception()
                        error = str(task_error) if task_error is not None else None
                    except Exception as future_error:  # noqa: BLE001 - lost/cancelled future
                        error = str(future_error) or "task lost"
                    if error is None:
                        try:
                            results.append(future.result())
                        except Exception as result_error:  # noqa: BLE001 - recorded as the task's error
                            error = str(result_error)
                    if error is None:
                        bar.write(f"{stage_label} completed {task.label}")
                        continue
                    _record_imaging_failure(task, error, failures)
                    bar.write(f"  {task.label} FAILED: {error_headline(error)}")
                    if not continue_on_error:
                        for other in pending.values():
                            cancel = getattr(other, "cancel", None)
                            if callable(cancel):
                                cancel()
                        raise ImagingRunAborted(task.label)
                done = total - len(pending) - len(queued)
                bar.set_postfix_str(f"completed {done}/{total} failed {len(failures)}")
                if not pending:
                    break
                now = time()
                if now - last_heartbeat >= heartbeat_interval:
                    last_heartbeat = now
                    lines, running = heartbeat_lines(
                        (
                            (tasks[index].label, imaging_log_path(tasks[index].output_dir))
                            for index in pending
                        ),
                        started_at=started_at,
                    )
                    for line in lines:
                        bar.write(line)
                    bar.write(
                        f"{stage_label}: {running} running, "
                        f"{len(pending) - running + len(queued)} waiting, "
                        f"{total - len(pending) - len(queued)} done, {len(failures)} failed"
                    )
                if not progressed:
                    sleep(0.5)
    return results
