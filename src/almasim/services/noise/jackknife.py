"""The continuum noise of an imaged window with its sky cancelled (MADIV-28).

``cont.sigma`` (:mod:`.image_stats`) is measured on the cleaned residual, so
it carries whatever cleaning leaves of the sky: calibration residue around a
bright source, extended emission, sidelobes of emission outside the clean
mask. A simulated window's sky is a point source, so on a bright real field
the two residuals differ by more than their noise. The jackknife measures
the noise alone. The window is extracted as its imaging extracted it (the
same window and fields, rows over the maximum WEIGHT left out), and every
other integration of each baseline has its sign flipped: whatever stays the
same from one integration to the next, the sky, cancels, while the noise and
the correlation of its channels stay. The result is imaged dirty with the
imaging task's own continuum command (same grid, weighting, field and
channels; no cleaning) over the whole band, and ``jk.sigma`` is the robust
noise of the inner part of that image, where ``cont.sigma`` is measured.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..imaging.markers import imaging_log_path
from ..imaging.ms.single_window import extract_single_window_ms_subprocess
from ..imaging.products import IMAGE_FILENAME
from ..imaging.stats import robust_sigma
from ..imaging.wsclean.command import WSCleanCommand, _dirty_command
from ..instrument.constants import DEFAULT_MAX_WEIGHT
from ..runtime.child import call_in_child
from ..runtime.env import wsclean_environment
from .image_stats import INNER, _planes, _radius

NAME = "jackknife"
CONTINUUM_NAME = IMAGE_FILENAME.removesuffix("-image.fits")  # the continuum's -name: wsclean
DATA_COLUMNS = ("CORRECTED_DATA", "DATA")
CHUNK_ROWS = 20_000


def continuum_command(task_dir: str | Path) -> list[str] | None:
    """The continuum WSClean command the imaging task's log records (its first
    pass: the one without ``-continue``), or None when it records none."""
    log = imaging_log_path(task_dir)
    if not log.is_file():
        return None
    for line in log.read_text(errors="replace").splitlines():
        if not line.startswith("# "):
            continue
        parts = line[2:].split()
        if len(parts) < 2 or Path(parts[0]).name != "wsclean" or "-continue" in parts:
            continue
        name = WSCleanCommand.parse(parts).value("-name")
        if name is not None and Path(name).name == CONTINUUM_NAME:
            return parts
    return None


def alternating_signs(antenna1: np.ndarray, antenna2: np.ndarray, time: np.ndarray) -> np.ndarray:
    """+1 and -1 in turn along each baseline's integrations, in time order."""
    order = np.lexsort((time, antenna2, antenna1))
    baseline = np.stack([antenna1[order], antenna2[order]])
    new = np.r_[True, np.any(baseline[:, 1:] != baseline[:, :-1], axis=0)]
    starts = np.flatnonzero(new)
    lengths = np.diff(np.r_[starts, order.size])
    within = np.arange(order.size) - np.repeat(starts, lengths)
    signs = np.empty(order.size)
    signs[order] = np.where(within % 2 == 0, 1.0, -1.0)
    return signs


def flip_alternate_integrations(ms: str, opener=None) -> dict[str, object]:
    """Flip the sign of every other integration of each baseline in the MS's
    data columns, in place; the column the image reads (CORRECTED_DATA when
    present) and the rows. Run in a child process (``flip_in_child``)."""
    from ..ms.tables import open_table

    with open_table(ms, readonly=False, opener=opener) as table:
        signs = alternating_signs(
            np.asarray(table.getcol("ANTENNA1")),
            np.asarray(table.getcol("ANTENNA2")),
            np.asarray(table.getcol("TIME")),
        )
        columns = [c for c in DATA_COLUMNS if c in table.colnames()]
        if not columns:
            raise ValueError(f"{ms}: no data column ({', '.join(DATA_COLUMNS)})")
        for column in columns:
            for start in range(0, signs.size, CHUNK_ROWS):
                chunk = signs[start : start + CHUNK_ROWS]
                data = table.getcol(column, start, chunk.size)
                table.putcol(column, data * chunk[:, None, None].astype(data.real.dtype), start)
    return {"column": columns[0], "rows": int(signs.size)}


def flip_in_child(ms: Path, *, timeout: float | None = None) -> dict[str, object]:
    """``flip_alternate_integrations`` in a new Python process: the simulated
    MS was written with casatools, and python-casacore in the same process
    can abort (free(): invalid pointer)."""
    return call_in_child(
        "almasim.services.noise.jackknife:flip_alternate_integrations",
        {"ms": str(ms)},
        what="jackknife sign flip",
        env=wsclean_environment(1),
        timeout=timeout,
        timeout_noun="task timeout",
    )


def jackknife_command(
    command: Sequence[str], work: Path, ms: Path, column: str, *, threads: int,
    wsclean_bin: str | None = None,
) -> list[str]:  # fmt: skip
    """The continuum ``command`` as a dirty image of the jackknifed ``ms``
    (``column``) into ``work/jackknife``."""
    dirty = WSCleanCommand.parse(_dirty_command(command, work, NAME, column))
    dirty = dirty.with_option("-temp-dir", work).with_option("-j", threads)
    return replace(dirty, program=wsclean_bin or dirty.program, ms=str(ms)).argv()


def jackknife_noise(
    task_dir: str | Path,
    ms: str | Path,
    spw: int,
    field_ids: Sequence[int] | None,
    work: Path,
    *,
    threads: int = 4,
    timeout: float | None = None,
    wsclean_bin: str | None = None,
) -> dict[str, float]:
    """``{"jk.sigma": ...}``: the jackknifed continuum noise of the window that
    the imaging task ``task_dir`` imaged from spectral window ``spw`` of
    ``ms``. Empty when the task's log records no continuum command."""
    command = continuum_command(task_dir)
    if command is None:
        return {}
    work.mkdir(parents=True, exist_ok=True)
    copy = work / f"{NAME}.ms"
    shutil.rmtree(copy, ignore_errors=True)
    try:
        extract_single_window_ms_subprocess(
            ms, spw, field_ids, copy, timeout=timeout, max_weight=DEFAULT_MAX_WEIGHT
        )
        column = str(flip_in_child(copy, timeout=timeout)["column"])
        argv = jackknife_command(
            command, work, copy, column, threads=threads, wsclean_bin=wsclean_bin
        )
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=wsclean_environment(threads),
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout).strip()[-1500:]
            raise RuntimeError(f"jackknife image of {task_dir} failed: {tail}")
        dirty = work / f"{NAME}-dirty.fits"
        plane = _planes(dirty)[0][0]
        sigma = robust_sigma(plane[_radius(plane.shape) < INNER])
    finally:
        shutil.rmtree(copy, ignore_errors=True)
        for leftover in work.glob(f"{NAME}-*.fits"):
            leftover.unlink(missing_ok=True)
    return {"jk.sigma": float(sigma)}
