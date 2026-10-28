"""Calibrate every raw MS of a download (MADIV-28 T4.8).

Each raw MS gets the delivered pipeline flags and calibration
(:mod:`.apply`), then its science windows are split out (:mod:`.split`).
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from ...casa import tasks as casa_tasks
from ...casa.environment import configure_casa_environment
from ...casa.runtime_data import (
    ensure_casa_runtime_data,
    find_existing_casa_data,
    has_casa_runtime_data,
)
from ...ms.verify import verify_measurement_set
from ...utils import LogFn, report
from ..unpack import casa_log_path
from .apply import apply_delivered_calibration
from .cleanup import cleanup_intermediate_calibration_data
from .discovery import (
    calibration_not_delivered,
    find_calibration_directory,
    find_raw_ms_directories,
)
from .split import (
    calibrated_output_path,
    calibration_failure_marker_path,
    is_calibration_complete,
    split_calibrated_science_ms,
    write_calibration_failure_marker,
)

logger = logging.getLogger(__name__)


def _prepare_casa_environment(
    input_path: Path,
    output_path: Path,
    casa_data_root: str | os.PathLike[str] | None,
) -> tuple[Path, Path | None]:
    """Choose CASA runtime data and an isolated writable workspace for calibration jobs."""
    workspace_root = Path(tempfile.mkdtemp(prefix="almasim-casa-"))

    if casa_data_root is not None:
        resolved = Path(casa_data_root).expanduser().resolve()
        return resolved, workspace_root

    # Searches the output root, then the input tree (one walk; it is slow on NFS).
    existing = find_existing_casa_data(input_path, output_path)
    if has_casa_runtime_data(existing):
        return existing.resolve(), workspace_root

    return workspace_root / ".casa-data", workspace_root


@dataclass(frozen=True)
class CalibrationOptions:
    """How :func:`calibrate_measurement_sets` treats each raw MS (MADIV-28 T4.6)."""

    asdm_uid: str | None = None
    casa_data_root: str | os.PathLike[str] | None = None
    skip_casa_data_update: bool = False
    overwrite: bool = False
    clean_intermediate: bool = False
    original_data_root: str | os.PathLike[str] | None = None
    continue_on_error: bool = False
    remove_working_copy: bool = True
    require_pipeline_flags: bool = True


class _CalibrationTasks(NamedTuple):
    applycal: Any
    clearcal: Any
    flagmanager: Any
    mstransform: Any


def create_calibrated_measurement_sets(
    input_root: str | os.PathLike[str],
    raw_ms_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    asdm_uid: str | None = None,
    casa_data_root: str | os.PathLike[str] | None = None,
    skip_casa_data_update: bool = False,
    overwrite: bool = False,
    clean_intermediate: bool = False,
    original_data_root: str | os.PathLike[str] | None = None,
    logger_fn: LogFn = None,
    continue_on_error: bool = False,
    remove_working_copy: bool = True,
    require_pipeline_flags: bool = True,
) -> list[Path]:
    """Create calibrated science MS products from raw MS and ALMA calibration products.

    The delivered ``Pipeline_Final`` flags are restored onto every raw MS before
    its calibration tables are applied (:func:`restore_pipeline_flags`); a
    delivery without them is a recorded failure unless
    ``require_pipeline_flags`` is False.

    UIDs whose output already carries a completion marker are skipped (and
    returned) unless ``overwrite`` is set. UIDs ALMA delivered without
    calibration (:func:`calibration_not_delivered`) are skipped and not
    returned: they are neither calibrated nor a failure. Partial outputs or working copies
    without a marker are leftovers of a failed run and are replaced. With
    ``remove_working_copy`` the per-UID working directory (a full copy of the
    raw MS plus calibration tables) is deleted as soon as the UID finishes.

    Every failure writes a ``<uid>.ms.split.cal.failed`` marker recording the
    error, so a rerun can tell "never attempted" from "attempted and failed".
    With ``continue_on_error`` a Python-level failure on one raw MS is logged and
    the remaining raw MSs are still processed; a ``RuntimeError`` summarising the
    failed UIDs is raised at the end. Note that a hard crash inside CASA (for
    example an uncaught ``casacore`` C++ exception) terminates the whole process
    and cannot be skipped here; the CLI isolates each UID in its own subprocess
    for that reason, and that wrapper writes the failure marker instead. The
    working copy is reclaimed in either case, so a failed UID leaves no
    multi-GB raw copy behind in ``working/``.
    """
    options = CalibrationOptions(
        asdm_uid=asdm_uid,
        casa_data_root=casa_data_root,
        skip_casa_data_update=skip_casa_data_update,
        overwrite=overwrite,
        clean_intermediate=clean_intermediate,
        original_data_root=original_data_root,
        continue_on_error=continue_on_error,
        remove_working_copy=remove_working_copy,
        require_pipeline_flags=require_pipeline_flags,
    )
    return calibrate_measurement_sets(
        input_root, raw_ms_root, output_root, options, logger_fn=logger_fn
    )


def calibrate_measurement_sets(
    input_root: str | os.PathLike[str],
    raw_ms_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    options: CalibrationOptions,
    *,
    logger_fn: LogFn = None,
) -> list[Path]:
    """:func:`create_calibrated_measurement_sets` with its options in one object."""
    input_path = Path(input_root).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    raw_mss = find_raw_ms_directories(raw_ms_root, options.asdm_uid)

    casa_data, workspace_root = _prepare_casa_environment(
        input_path, output_path, options.casa_data_root
    )
    try:
        casa_log = casa_log_path(output_path, "calibrate", options.asdm_uid)
        configure_casa_environment(
            output_path,
            casa_data,
            workspace_root=workspace_root,
            log_file=casa_log,
            log_to_terminal=True,
        )
        report(logger, logger_fn, f"CASA log file: {casa_log}")
        ensure_casa_runtime_data(
            casa_data, skip_update=options.skip_casa_data_update, logger_fn=logger_fn
        )

        _casa = casa_tasks()
        tasks = _CalibrationTasks(
            _casa.applycal, _casa.clearcal, _casa.flagmanager, _casa.mstransform
        )

        report(logger, logger_fn, f"Found {len(raw_mss)} raw MeasurementSet(s) to calibrate")
        calibrated_mss: list[Path] = []
        failures: list[tuple[str, str]] = []
        for index, raw_ms in enumerate(raw_mss, start=1):
            uid = raw_ms.name.removesuffix(".ms")
            progress = f"[{index}/{len(raw_mss)}]"
            if is_calibration_complete(output_path, uid) and not options.overwrite:
                existing = calibrated_output_path(output_path, uid)
                report(
                    logger, logger_fn, f"{progress} {uid} already calibrated, skipping: {existing}"
                )
                calibrated_mss.append(existing)
                continue
            if calibration_not_delivered(input_path, uid):
                report(
                    logger,
                    logger_fn,
                    f"{progress} {uid} was delivered without calibration (e.g. QA0 SemiPass), "
                    "skipping",
                )
                calibration_failure_marker_path(output_path, uid).unlink(missing_ok=True)
                continue

            report(logger, logger_fn, f"{progress} Calibrating {uid}")
            working_dir = output_path / "working" / f"{uid}.calibration"
            # A fresh attempt invalidates whatever the previous one concluded.
            calibration_failure_marker_path(output_path, uid).unlink(missing_ok=True)
            try:
                calibrated_mss.append(
                    _calibrate_one(
                        raw_ms, uid, input_path, output_path, working_dir, tasks, options, logger_fn
                    )
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                write_calibration_failure_marker(output_path, uid, error)
                if not options.continue_on_error:
                    raise
                logger.debug("Calibration of %s failed", uid, exc_info=True)
                failures.append((uid, error))
                report(
                    logger,
                    logger_fn,
                    f"{progress} Calibration FAILED for {uid}: {exc}",
                    logging.ERROR,
                )
                continue
            finally:
                if options.remove_working_copy and working_dir.is_dir():
                    shutil.rmtree(working_dir, ignore_errors=True)
                    report(logger, logger_fn, f"Removed working copy: {working_dir}")
            report(logger, logger_fn, f"{progress} Calibration finished for {uid}")

        if failures:
            summary = "; ".join(f"{uid} ({error})" for uid, error in failures)
            raise RuntimeError(
                f"Calibration failed for {len(failures)} of {len(raw_mss)} UID(s): {summary}"
            )

        if options.clean_intermediate:
            cleanup_intermediate_calibration_data(
                raw_ms_root,
                output_path,
                calibrated_mss,
                original_data_root=options.original_data_root,
                logger_fn=logger_fn,
            )

        return calibrated_mss
    finally:
        if workspace_root is not None:
            shutil.rmtree(workspace_root, ignore_errors=True)


def _calibrate_one(
    raw_ms: Path,
    uid: str,
    input_path: Path,
    output_path: Path,
    working_dir: Path,
    tasks: _CalibrationTasks,
    options: CalibrationOptions,
    logger_fn: LogFn,
) -> Path:
    """Flag, calibrate and split one raw MS; the calibrated science MS."""
    partial_output = calibrated_output_path(output_path, uid)
    if partial_output.exists() and not options.overwrite:
        report(
            logger, logger_fn, f"Replacing incomplete output from a previous run: {partial_output}"
        )
    if (working_dir / raw_ms.name).exists() and not options.overwrite:
        report(
            logger, logger_fn, f"Replacing stale working copy from a previous run: {working_dir}"
        )
    # A raw MS whose import was killed opens fine but has no rows;
    # clearcal would then abort the whole process inside casacore
    # (minMax on an empty array), which no handler here can catch.
    # Checking first turns that into an ordinary recorded failure.
    verify_measurement_set(raw_ms)
    calibration_dir = find_calibration_directory(input_path, uid)
    working_ms = apply_delivered_calibration(
        tasks.applycal,
        tasks.clearcal,
        raw_ms,
        calibration_dir,
        working_dir,
        overwrite=True,
        logger_fn=logger_fn,
        flagmanager=tasks.flagmanager,
        require_pipeline_flags=options.require_pipeline_flags,
    )
    return split_calibrated_science_ms(
        tasks.mstransform,
        working_ms,
        output_path,
        overwrite=True,
        logger_fn=logger_fn,
    )
