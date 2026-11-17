"""Import ALMA ASDM directories into raw MeasurementSets (MADIV-28 T4.8).

Only the raw ASDM-to-MS import via ``casatasks.importasdm``: no calibration,
splitting, imaging or restoring of pipeline products.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

from ..casa import tasks as casa_tasks
from ..casa.environment import configure_casa_environment
from ..casa.runtime_data import ensure_casa_runtime_data, find_existing_casa_data
from ..markers import StageMarkers
from ..ms.remove import remove_measurement_set
from ..ms.verify import verify_measurement_set
from ..utils import LogFn, report
from .asdm import asdm_name, find_asdm_directories, verify_asdm_directory
from .delivery import RAW_FOLDER, eb_member_folder

logger = logging.getLogger(__name__)


def casa_log_path(output_root: str | os.PathLike[str], stage: str, uid: str | None) -> Path:
    """Return the CASA log file used for one ``stage`` (unpack/calibrate) of one UID."""
    safe_uid = (uid or "all").replace("/", "_").replace(":", "_")
    return Path(output_root).expanduser().resolve() / "logs" / f"casa-{stage}-{safe_uid}.log"


def raw_ms_path(output_root: str | os.PathLike[str], uid: str) -> Path:
    """Where ``uid``'s raw MS goes: next to its ASDM, ``<member>/raw/<uid>.ms``, when
    ``output_root`` is a delivery tree that places ``uid``; ``<output_root>/working/<uid>.ms``
    otherwise."""
    root = Path(output_root).expanduser().resolve()
    member = eb_member_folder(root, uid)
    if member is not None:
        return member / RAW_FOLDER / f"{uid}.ms"
    return root / "working" / f"{uid}.ms"


def raw_ms_marker_path(output_root: str | os.PathLike[str], uid: str) -> Path:
    """Return the completion marker written next to a finished raw MS."""
    return raw_ms_path(output_root, uid).with_name(f"{uid}.ms.done")


def raw_ms_failure_marker_path(output_root: str | os.PathLike[str], uid: str) -> Path:
    """Return the failure marker written when ``uid`` could not be imported."""
    return raw_ms_path(output_root, uid).with_name(f"{uid}.ms.failed")


def is_unpack_complete(output_root: str | os.PathLike[str], uid: str) -> bool:
    """Return True when ``uid`` has both a raw MS directory and its marker."""
    return raw_ms_path(output_root, uid).is_dir() and raw_ms_marker_path(output_root, uid).is_file()


def _raw_ms_markers(output_root: str | os.PathLike[str], uid: str) -> StageMarkers:
    return StageMarkers(
        raw_ms_marker_path(output_root, uid), raw_ms_failure_marker_path(output_root, uid)
    )


def write_raw_ms_marker(output_root: str | os.PathLike[str], uid: str, rows: int) -> Path:
    """Record that ``uid`` imported successfully, with the row count as proof."""
    return _raw_ms_markers(output_root, uid).write_done(
        uid=uid, output=str(raw_ms_path(output_root, uid)), rows=rows
    )


def write_raw_ms_failure_marker(
    output_root: str | os.PathLike[str],
    uid: str,
    error: str,
    *,
    log_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Record why ``uid`` could not be imported."""
    return _raw_ms_markers(output_root, uid).write_failed(
        error, stage="unpack", log_path=log_path, uid=uid
    )


def create_measurement_set(
    importasdm: Callable[..., object],
    raw_asdm: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    overwrite: bool = False,
    logger_fn: LogFn = None,
) -> Path:
    """Create one raw MeasurementSet from one ASDM directory.

    An import is only considered finished once the resulting MS has at least one
    row and a ``<uid>.ms.done`` marker. Directory existence is deliberately not
    enough: a killed ``importasdm`` leaves a complete-looking directory holding
    tens of GB whose row count was never committed, and treating that as done is
    what let truncated MeasurementSets flow into calibration, where they abort
    CASA with ``minMax - Array has no elements``.

    An existing MS without a marker is verified rather than trusted: if it has
    rows it is grandfathered in and marked, otherwise it is re-imported.
    """
    raw_asdm_path = Path(raw_asdm).expanduser().resolve()
    asdm_uid = asdm_name(raw_asdm_path)
    output_ms = raw_ms_path(output_root, asdm_uid)
    working_dir = output_ms.parent

    if not raw_asdm_path.is_dir():
        raise RuntimeError(f"Cannot find raw ASDM directory: {raw_asdm_path}")

    working_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("Working directory: %s", working_dir)

    output_root_path = Path(output_root).expanduser().resolve()

    if not overwrite and output_ms.exists():
        if raw_ms_marker_path(output_root_path, asdm_uid).is_file():
            report(
                logger,
                logger_fn,
                f"Raw MeasurementSet already exists, skipping import: {output_ms}",
            )
            return output_ms
        # No marker: either an import from before markers existed, or one that
        # was killed mid-write. Only the row count tells the two apart, and
        # trusting the directory is what let truncated MSs reach calibration.
        try:
            rows = verify_measurement_set(output_ms)
        except RuntimeError as exc:
            report(
                logger,
                logger_fn,
                f"Existing MeasurementSet is unusable, re-importing: {output_ms} ({exc})",
                logging.WARNING,
            )
        else:
            write_raw_ms_marker(output_root_path, asdm_uid, rows)
            report(
                logger,
                logger_fn,
                f"Verified existing raw MeasurementSet ({rows:,} rows), skipping import: "
                f"{output_ms}",
            )
            return output_ms

    report(logger, logger_fn, f"Creating raw MeasurementSet from {raw_asdm_path}")
    # Reaching here means we have decided to import: either the caller asked for
    # it, or whatever is on disk is unusable. importasdm must be allowed to
    # replace it, so ``overwrite`` is unconditionally True below.
    raw_ms_marker_path(output_root_path, asdm_uid).unlink(missing_ok=True)
    raw_ms_failure_marker_path(output_root_path, asdm_uid).unlink(missing_ok=True)
    # A link at the MS's path (a tree whose MSs were moved in by hand) is
    # replaced together with the MS it points to.
    for old in (output_ms, output_ms.with_name(output_ms.name + ".flagversions")):
        if old.is_symlink():
            remove_measurement_set(old)
    current_dir = Path.cwd()
    try:
        # Refuse a half-extracted ASDM up front, as a recorded failure. Left to
        # importasdm it becomes an opaque "File not found" after minutes of work.
        verify_asdm_directory(raw_asdm_path)
        try:
            os.chdir(working_dir)
            importasdm(
                asdm=str(raw_asdm_path),
                vis=str(output_ms),
                overwrite=True,
            )
        finally:
            os.chdir(current_dir)
        rows = verify_measurement_set(output_ms)
    except Exception as exc:
        write_raw_ms_failure_marker(output_root_path, asdm_uid, f"{type(exc).__name__}: {exc}")
        raise

    write_raw_ms_marker(output_root_path, asdm_uid, rows)
    report(logger, logger_fn, f"Created raw MeasurementSet ({rows:,} rows): {output_ms}")
    return output_ms


def create_measurement_sets(
    input_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    asdm_uid: str | None = None,
    casa_data_root: str | os.PathLike[str] | None = None,
    skip_casa_data_update: bool = False,
    overwrite: bool = False,
    logger_fn: LogFn = None,
) -> list[Path]:
    """Create raw MeasurementSets for all matching ASDMs below ``input_root``.

    Each UID that finishes carries a ``<uid>.ms.done`` marker recording its row
    count; each that fails carries a ``<uid>.ms.failed`` marker recording the
    error. See :func:`create_measurement_set` for why the row count matters.
    """
    asdm_dirs = find_asdm_directories(input_root, asdm_uid)

    # Decide what is already finished *before* importing CASA. A UID whose MS
    # carries a ``.ms.done`` marker needs no casatools at all, and importing
    # them anyway is what exposed finished UIDs to CASA start-up failures
    # (which the subprocess wrapper then recorded as ``.ms.failed``).
    finished: dict[Path, Path] = {}
    if not overwrite:
        for raw_asdm in asdm_dirs:
            uid = asdm_name(raw_asdm)
            if is_unpack_complete(output_root, uid):
                ms = raw_ms_path(output_root, uid)
                finished[raw_asdm] = ms
                report(
                    logger, logger_fn, f"Raw MeasurementSet already exists, skipping import: {ms}"
                )
    if asdm_dirs and len(finished) == len(asdm_dirs):
        return [finished[raw_asdm] for raw_asdm in asdm_dirs]

    casa_data = find_existing_casa_data(input_root, output_root, casa_data_root)
    configure_casa_environment(
        output_root,
        casa_data,
        log_file=casa_log_path(output_root, "unpack", asdm_uid),
        log_to_terminal=True,
    )
    ensure_casa_runtime_data(casa_data, skip_update=skip_casa_data_update, logger_fn=logger_fn)

    importasdm = casa_tasks().importasdm

    report(
        logger,
        logger_fn,
        f"Found {len(asdm_dirs)} ASDM director{'y' if len(asdm_dirs) == 1 else 'ies'}",
    )
    return [
        finished[raw_asdm]
        if raw_asdm in finished
        else create_measurement_set(
            importasdm,
            raw_asdm,
            output_root,
            overwrite=overwrite,
            logger_fn=logger_fn,
        )
        for raw_asdm in asdm_dirs
    ]
