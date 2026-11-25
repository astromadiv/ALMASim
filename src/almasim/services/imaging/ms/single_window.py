"""Copy one spectral window (and some fields) of a MeasurementSet into its own MS.

Moved out of archive_imaging.py (MADIV-28 module split).
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Sequence
from pathlib import Path

from almasim.services.instrument.constants import DEFAULT_MAX_WEIGHT
from almasim.services.ms.tables import (
    open_table,
    polarization_of_data_description,
    spw_of_data_description,
)
from almasim.services.runtime.child import ChildFailed, append_to_log, call_in_child
from almasim.services.runtime.env import wsclean_environment


def extract_single_window_ms(
    ms_path: str | os.PathLike[str],
    spw: int,
    field_ids: Sequence[int] | None,
    out_path: str | os.PathLike[str],
    *,
    max_weight: float | None = DEFAULT_MAX_WEIGHT,
) -> dict[str, object]:
    """Copy one spectral window (and optionally some fields) into a new MS.

    One attempt; ``extract_single_window_ms_subprocess`` retries lock-file
    failures (see ``_LOCK_FILE_ERROR``) in a new process each time.

    Rows whose WEIGHT exceeds ``max_weight`` are left out (see
    ``DEFAULT_MAX_WEIGHT``; ``None`` or 0 keeps every row); the result
    reports how many were dropped.

    WSClean 3.7 reads out of bounds in its reordering step when an MS holds
    several spectral windows and ``-spws`` selects one (an ALMA split MS
    lists 60 windows, 4 of them populated); the crash is heap-layout
    dependent, so about half the tasks die. Handing WSClean an MS with a
    single spectral window lets it use its contiguous reader (``-no-reorder``)
    and skip that code entirely. The copy is a deep copy of the selected rows
    with all subtables; DATA_DESCRIPTION and SPECTRAL_WINDOW are collapsed to
    the one window (renumbered 0) so WSClean sees exactly one band.
    """
    return _extract_single_window_ms_once(ms_path, spw, field_ids, out_path, max_weight=max_weight)


def _extract_single_window_ms_once(
    ms_path: str | os.PathLike[str],
    spw: int,
    field_ids: Sequence[int] | None,
    out_path: str | os.PathLike[str],
    *,
    max_weight: float | None,
) -> dict[str, object]:
    from casacore.tables import taql

    src = str(ms_path)
    out = Path(out_path)
    shutil.rmtree(out, ignore_errors=True)
    # The source MS is only read, by several tasks at once (one per spectral
    # window, on different nodes) over NFS: shared reads (see ms.tables).
    with open_table(f"{src}/DATA_DESCRIPTION", shared=True) as dd_table:
        spw_of_dd = spw_of_data_description(dd_table)
        pol_of_dd = polarization_of_data_description(dd_table)
    dd_ids = [i for i, s in enumerate(spw_of_dd) if s == int(spw)]
    if not dd_ids:
        raise RuntimeError(f"Spectral window {spw} has no DATA_DESCRIPTION row in {src}")
    where = f"DATA_DESC_ID in [{','.join(str(i) for i in dd_ids)}]"
    if field_ids:
        where += f" and FIELD_ID in [{','.join(str(int(f)) for f in field_ids)}]"
    with open_table(src, shared=True) as main_in:
        counted = taql(f"select from $1 where {where}", tables=[main_in])
        n_all = int(counted.nrows())
        counted.close()
        if n_all == 0:
            raise RuntimeError(
                f"No visibilities for spectral window {spw} fields {field_ids} in {src}"
            )
        if max_weight:
            where += f" and max(WEIGHT) <= {float(max_weight)!r}"
        selection = taql(f"select from $1 where {where}", tables=[main_in])
        try:
            n_rows = int(selection.nrows())
            if n_rows == 0:
                raise RuntimeError(
                    f"All {n_all} visibilities for spectral window {spw} fields {field_ids} "
                    f"in {src} have WEIGHT > {max_weight:g}"
                )
            selection.copy(str(out), deep=True)
        finally:
            selection.close()

    with open_table(f"{out}/DATA_DESCRIPTION", readonly=False) as dd_out:
        dd_out.removerows([i for i in range(dd_out.nrows()) if i not in dd_ids])
        dd_out.putcol("SPECTRAL_WINDOW_ID", [0] * len(dd_ids))
        dd_out.putcol("POLARIZATION_ID", [pol_of_dd[i] for i in dd_ids])
    with open_table(f"{out}/SPECTRAL_WINDOW", readonly=False) as spw_out:
        spw_out.removerows([i for i in range(spw_out.nrows()) if i != int(spw)])
    renumber = {old: new for new, old in enumerate(dd_ids)}
    with open_table(out, readonly=False) as main:
        main.putcol("DATA_DESC_ID", [renumber[int(x)] for x in main.getcol("DATA_DESC_ID")])
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    return {
        "rows": n_rows,
        "bytes": size,
        "data_desc_ids": dd_ids,
        "dropped_rows": n_all - n_rows,
        "max_weight": float(max_weight) if max_weight else None,
    }


# casacore keeps a list of the processes using a table in its table.lock file
# and writes it with pwrite even when no read lock is taken. With several
# tasks of one MS reading it from different nodes over NFS that write fails
# now and then ("LockFile.cc : 300 Failed AlwaysAssert tracePWRITE", 4 of
# ~370 tasks on 2026-09-25). It is transient, so the extraction is retried,
# each attempt in a fresh child process: casacore keeps the half-written
# output of a failed copy in its process-wide table cache, and a retry in the
# same process then fails with "Cannot copy/rename; target table ... is still
# open (is in the table cache)" (job 3841, 2 tasks on 2026-09-26).
_LOCK_FILE_ERROR = "LockFile.cc"


_LOCK_FILE_ATTEMPTS = 5


_LOCK_FILE_RETRY_SECONDS = 20.0


def extract_single_window_ms_subprocess(
    ms_path: str | os.PathLike[str],
    spw: int,
    field_ids: Sequence[int] | None,
    out_path: str | os.PathLike[str],
    *,
    timeout: float | None = None,
    log_path: str | os.PathLike[str] | None = None,
    max_weight: float | None = DEFAULT_MAX_WEIGHT,
    attempts: int = _LOCK_FILE_ATTEMPTS,
    retry_seconds: float = _LOCK_FILE_RETRY_SECONDS,
) -> dict[str, object]:
    """Run ``extract_single_window_ms`` in a child Python process.

    casacore does the deep copy in C++ without releasing the GIL, for tens of
    seconds to minutes per task. Called in a Dask worker thread that freezes
    the worker's event loop, the scheduler stops receiving heartbeats and
    drops the worker, and dask-jobqueue then cancels its Slurm job with every
    task on it (production job 3828, 2026-09-25: all five workers lost within
    two minutes of starting). A child process keeps the worker responsive,
    exactly like the WSClean subprocess. The child's stderr is appended to
    ``log_path`` when given.

    A child that dies of a casacore lock-file error (see ``_LOCK_FILE_ERROR``)
    is rerun up to ``attempts`` times, ``retry_seconds`` apart, each time in a
    new process; any other failure raises at once. ``timeout`` applies to
    each attempt.
    """
    kwargs = {
        "ms_path": str(ms_path),
        "spw": int(spw),
        "field_ids": [int(f) for f in field_ids] if field_ids else None,
        "out_path": str(out_path),
        "max_weight": float(max_weight) if max_weight else None,
    }
    started = time.time()
    for attempt in range(1, max(int(attempts), 1) + 1):
        try:
            result = call_in_child(
                "almasim.services.imaging.ms.single_window:extract_single_window_ms",
                kwargs,
                what="extraction",
                env=wsclean_environment(1),
                timeout=timeout,
                timeout_noun="task timeout",
                log_path=log_path,
            )
            break
        except ChildFailed as exc:
            if _LOCK_FILE_ERROR not in exc.stderr or attempt >= attempts:
                raise
            append_to_log(
                log_path,
                f"extraction attempt {attempt}/{attempts} hit a casacore lock-file error; "
                f"retrying in a new process in {retry_seconds:g} s\n",
            )
            time.sleep(retry_seconds)
    result["seconds"] = round(time.time() - started, 1)
    result["attempts"] = attempt
    return result
