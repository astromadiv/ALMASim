"""One window and field of every execution block of a member, combined into one MS.

The execution blocks of a member observe the same windows on different days.
Their channels are topocentric (``MEAS_FREQ_REF`` TOPO), so one window starts
at a frequency that moves by up to a few MHz from one EB to the next, as the
Earth turns and orbits: channel n of one EB is not channel n of another.
:func:`build_member_window_ms` copies the window and field out of each EB
(``extract_single_window_ms``), regrids every copy onto one LSRK channel grid
with CASA ``mstransform``, and concatenates them with CASA ``concat`` into an
MS of one window and one field, the input of a member's imaging task.

The grid is the earliest EB's window in LSRK: its channel count and width,
starting at its first channel's LSRK frequency. The other EBs cover it up to
their shift, so the edge channels hold fewer EBs than the rest.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from almasim.services.imaging.ms.single_window import (
    DEFAULT_MAX_WEIGHT,
    extract_single_window_ms_subprocess,
)
from almasim.services.imaging.provenance import eb_uid_from_ms
from almasim.services.runtime.child import call_in_child
from almasim.services.runtime.env import casa_environment


def build_member_window_ms(
    windows: Sequence[tuple[str | os.PathLike[str], int, Sequence[int]]],
    out_path: str | os.PathLike[str],
    *,
    casa_data: str,
    timeout: float | None = None,
    log_path: str | os.PathLike[str] | None = None,
    max_weight: float | None = DEFAULT_MAX_WEIGHT,
) -> dict[str, object]:
    """Combine ``windows`` ((MS, window, field ids) per EB, earliest first) into ``out_path``.

    Each EB's copy and LSRK regridding are made next to ``out_path`` and
    removed once concatenated. Returns the rows kept and dropped (``max_weight``,
    as in ``extract_single_window_ms``), the size, the grid (``channels``,
    ``start_hz``, ``width_hz``) and each EB's own LSRK start (``shifts_hz``,
    relative to the grid's).
    """
    if not windows:
        raise ValueError("build_member_window_ms needs at least one execution block")
    out = Path(out_path)
    started = time.time()
    copies: list[Path] = []
    kept = dropped = 0
    try:
        for ms, spw, fields in windows:
            eb = eb_uid_from_ms(ms) or Path(ms).name.split(".")[0]
            copy = out.with_name(f"{eb}.{out.name}")
            extracted: dict[str, Any] = extract_single_window_ms_subprocess(
                ms,
                int(spw),
                list(fields) or None,
                copy,
                timeout=timeout,
                log_path=log_path,
                max_weight=max_weight,
            )
            copies.append(copy)
            kept += int(extracted.get("rows") or 0)
            dropped += int(extracted.get("dropped_rows") or 0)
        combined = call_in_child(
            "almasim.services.imaging.ms.member_window:regrid_and_concat",
            {
                "window_mss": [str(c) for c in copies],
                "out_path": str(out),
                "casa_data": str(casa_data),
                "log_file": str(log_path) if log_path else str(out.with_suffix(".casa.log")),
            },
            what="combining the execution blocks",
            env=casa_environment(),
            timeout=timeout,
            log_path=log_path,
        )
    finally:
        for copy in copies:
            shutil.rmtree(copy, ignore_errors=True)
            shutil.rmtree(_lsrk_path(copy), ignore_errors=True)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    return {
        **combined,
        "execution_blocks": len(windows),
        "rows": kept,
        "dropped_rows": dropped,
        "max_weight": float(max_weight) if max_weight else None,
        "bytes": size,
        "seconds": round(time.time() - started, 1),
    }


def _lsrk_path(copy: Path) -> Path:
    return copy.with_name(f"{copy.name.removesuffix('.ms')}.lsrk.ms")


def regrid_and_concat(
    window_mss: Sequence[str], out_path: str, casa_data: str, log_file: str
) -> dict[str, object]:
    """CASA: regrid single-window MSs onto the first one's LSRK grid and concatenate them.

    Runs in a CASA child process (``build_member_window_ms``). Each input holds
    one window and the target field's rows (its other fields stay in FIELD);
    the output holds one window and one field, both numbered 0.
    """
    import numpy as np

    from almasim.services.casa import tasks as casa_tasks
    from almasim.services.casa import tools as casa_tools
    from almasim.services.imaging.selfcal.casa_ops import _configure_casa

    _configure_casa(casa_data, log_file)
    tasks, tools = casa_tasks(), casa_tools()

    def lsrk_frequencies(ms_path: str) -> tuple[np.ndarray, list[int]]:
        table = tools.table()
        table.open(ms_path)
        try:
            fields = sorted({int(f) for f in table.getcol("FIELD_ID")})
        finally:
            table.close()
        ms = tools.ms()
        ms.open(ms_path)
        try:
            frequencies = ms.cvelfreqs(
                spwids=[0], fieldids=fields[:1], mode="channel", outframe="LSRK"
            )
        finally:
            ms.close()
        return np.sort(np.asarray(frequencies, dtype=float)), fields

    reference, _ = lsrk_frequencies(window_mss[0])
    width = float(np.median(np.diff(reference))) if reference.size > 1 else 0.0
    start = float(reference[0])
    shifts: list[float] = []
    regridded: list[str] = []
    for ms_path in window_mss:
        own, fields = lsrk_frequencies(ms_path)
        shifts.append(round(float(own[0]) - start, 1))
        target = str(_lsrk_path(Path(ms_path)))
        shutil.rmtree(target, ignore_errors=True)
        tasks.mstransform(
            vis=ms_path,
            outputvis=target,
            field=",".join(str(f) for f in fields),
            datacolumn="data",
            regridms=True,
            mode="frequency",
            nchan=int(reference.size),
            start=f"{start!r}Hz",
            width=f"{width!r}Hz",
            outframe="LSRK",
            interpolation="linear",
        )
        regridded.append(target)
    shutil.rmtree(out_path, ignore_errors=True)
    tasks.concat(vis=regridded, concatvis=out_path, dirtol="1arcsec", copypointing=False)
    table = tools.table()
    counts = {}
    for name in ("SPECTRAL_WINDOW", "FIELD", "DATA_DESCRIPTION"):
        table.open(f"{out_path}/{name}")
        try:
            counts[name] = int(table.nrows())
        finally:
            table.close()
    if counts["SPECTRAL_WINDOW"] != 1 or counts["FIELD"] != 1:
        raise RuntimeError(
            f"The combined MS has {counts['SPECTRAL_WINDOW']} spectral windows and "
            f"{counts['FIELD']} fields; one of each was expected (fields more than 1 arcsec "
            "apart, or windows on different grids, are not merged)"
        )
    return {
        "channels": int(reference.size),
        "start_hz": start,
        "width_hz": width,
        "shifts_hz": shifts,
    }
