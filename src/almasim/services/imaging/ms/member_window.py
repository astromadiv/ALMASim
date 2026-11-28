"""One window and field of every execution block of a member, combined into one MS.

The execution blocks of a member observe the same windows on different days.
Their channels are topocentric (``MEAS_FREQ_REF`` TOPO), so one window starts
at a frequency that moves by up to a few MHz from one EB to the next, as the
Earth turns and orbits: channel n of one EB is not channel n of another.
:func:`build_member_window_ms` copies the window and field out of each EB
(``extract_single_window_ms``), regrids every copy onto one LSRK channel grid
with CASA ``mstransform``, and concatenates them with CASA ``concat`` into an
MS of one window and one field, the input of a member's imaging task.

``mstransform`` regrids only onto channels an input covers over its whole
time range, so the grid is where all the EBs' windows overlap in LSRK, a few
channels fewer than one EB's window, and every channel holds every EB.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Mapping, Sequence
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
    channels_out: int | None = None,
    grid: Sequence[float] | None = None,
    selfcal: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, object]:
    """Combine ``windows`` ((MS, window, field ids) per EB, earliest first) into ``out_path``.

    Each EB's copy and LSRK regridding are made next to ``out_path`` and
    removed once concatenated. Returns the rows kept and dropped (``max_weight``,
    as in ``extract_single_window_ms``), the size, the grid (``channels``,
    ``start_hz``, ``width_hz``), each EB's own LSRK start (``shifts_hz``,
    relative to the first's) and, with ``channels_out``, the cube's channel
    count on the grid (``regrid_and_concat``). With ``grid`` (first channel's
    centre in Hz, channel width in Hz, channel count) the grid is that one, as
    far as the EBs cover it, and ``grid_note`` in the result says which of its
    channels were made. With ``selfcal`` (``{eb: applycal entries}``, from a replay
    of the ALMA pipeline's self-calibration: ``alma_replay.entries_for``) each
    EB's window is self-calibrated with them before it is regridded, and
    ``selfcal_applied`` in the result lists those EBs.
    """
    if not windows:
        raise ValueError("build_member_window_ms needs at least one execution block")
    out = Path(out_path)
    started = time.time()
    copies: list[Path] = []
    calibrations: list[dict[str, Any] | None] = []
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
            entries = (selfcal or {}).get(eb)
            calibrations.append(
                {"eb": eb, "spw": int(spw), "entries": [dict(e) for e in entries]}
                if entries
                else None
            )
            kept += int(extracted.get("rows") or 0)
            dropped += int(extracted.get("dropped_rows") or 0)
        combined = call_in_child(
            "almasim.services.imaging.ms.member_window:regrid_and_concat",
            {
                "window_mss": [str(c) for c in copies],
                "out_path": str(out),
                "casa_data": str(casa_data),
                "log_file": str(log_path) if log_path else str(out.with_suffix(".casa.log")),
                "channels_out": channels_out,
                "grid": [float(v) for v in grid] if grid else None,
                "selfcal": calibrations if any(calibrations) else None,
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
    window_mss: Sequence[str],
    out_path: str,
    casa_data: str,
    log_file: str,
    channels_out: int | None = None,
    grid: Sequence[float] | None = None,
    selfcal: Sequence[Mapping[str, Any] | None] | None = None,
) -> dict[str, object]:
    """CASA: regrid single-window MSs onto one LSRK grid and concatenate them.

    Runs in a CASA child process (``build_member_window_ms``). Each input holds
    one window and the target field's rows (its other fields stay in FIELD);
    the output holds one window and one field, both numbered 0.

    ``mstransform`` regrids only onto channels the input covers over its whole
    time range, and no narrower than its own. Each EB's LSRK coverage is read
    from a regridding of its first and last scans; the grid is where they all
    overlap, less a channel at each end, with the widest EB's channel width.
    With ``channels_out`` (the cube's WSClean ``-channels-out``, planned for
    the first EB's channel count) the grid's channel count is a multiple of
    the native channels per cube channel, and ``channels_out`` in the result
    is the cube's channel count on it.

    With ``grid`` (first channel's centre in Hz, channel width in Hz, channel
    count: the channels of a delivered cube) the grid is those channels that
    lie inside the EBs' overlap, each as wide as the delivered cube's and
    centred where its is; ``channels_out`` is their count and ``grid_note``
    names them. A grid narrower than the EBs' own channels, or outside their
    overlap, is not used, and ``grid_note`` says so.

    ``selfcal``: for each input, None or its EB (``eb``), its window's id in
    the EB (``spw``) and the replayed self-calibration's ``entries``
    (``alma_replay.apply_selfcal``), applied before the regridding, which then
    reads CORRECTED_DATA.
    """
    import math

    import numpy as np

    from almasim.services.casa import tasks as casa_tasks
    from almasim.services.casa import tools as casa_tools
    from almasim.services.imaging.selfcal.casa_ops import _configure_casa

    _configure_casa(casa_data, log_file)
    tasks, tools = casa_tasks(), casa_tools()

    # Each input's column: CORRECTED_DATA once self-calibrated.
    data_column: dict[str, str] = {}
    applied: list[str] = []
    for ms_path, calibration in zip(window_mss, selfcal or [None] * len(window_mss), strict=True):
        data_column[ms_path] = "data"
        if calibration:
            from almasim.services.imaging.selfcal.alma_replay import apply_selfcal

            apply_selfcal(tasks, ms_path, calibration["entries"], int(calibration["spw"]))
            data_column[ms_path] = "corrected"
            applied.append(str(calibration["eb"]))

    def column(table_path: str, name: str, row: int | None = None):
        table = tools.table()
        table.open(table_path)
        try:
            return table.getcol(name) if row is None else table.getcell(name, row)
        finally:
            table.close()

    def lsrk_coverage(ms_path: str) -> dict[str, object]:
        fields = sorted({int(f) for f in column(ms_path, "FIELD_ID")})
        scans = sorted({int(s) for s in column(ms_path, "SCAN_NUMBER")})
        probe = f"{ms_path.removesuffix('.ms')}.probe.ms"
        shutil.rmtree(probe, ignore_errors=True)
        try:
            tasks.mstransform(
                vis=ms_path,
                outputvis=probe,
                field=",".join(str(f) for f in fields),
                scan=f"{scans[0]},{scans[-1]}",
                datacolumn=data_column[ms_path],
                regridms=True,
                mode="channel",
                outframe="LSRK",
            )
            frequencies = np.asarray(column(f"{probe}/SPECTRAL_WINDOW", "CHAN_FREQ", 0), float)
            widths = np.abs(np.asarray(column(f"{probe}/SPECTRAL_WINDOW", "CHAN_WIDTH", 0)))
        finally:
            shutil.rmtree(probe, ignore_errors=True)
        return {
            "fields": fields,
            "low": float(frequencies.min()),
            "high": float(frequencies.max()),
            "width": float(widths.max()),
            "channels": int(frequencies.size),
        }

    coverage = [lsrk_coverage(ms_path) for ms_path in window_mss]
    # A hair wider than the widest: mstransform refuses a narrower channel.
    width = max(float(c["width"]) for c in coverage) * (1 + 1e-9)  # type: ignore[arg-type]
    low = max(float(c["low"]) for c in coverage) + width  # type: ignore[arg-type]
    high = min(float(c["high"]) for c in coverage) - width  # type: ignore[arg-type]
    channels = int(math.floor((high - low) / width + 1e-6)) + 1 if high >= low else 0
    native = int(coverage[0]["channels"])  # type: ignore[call-overload]
    per_cube_channel = native // channels_out if channels_out and native % channels_out == 0 else 1
    grid_note = ""
    if grid:
        chosen = delivered_channels(grid, low - width, high + width, width)
        if chosen is None:
            grid_note = (
                "the delivered cube's channels are not used (narrower than the execution "
                "blocks' own, or outside where they overlap)"
            )
        else:
            first, count = chosen
            low, width, channels, per_cube_channel = (
                float(grid[0]) + first * float(grid[1]),
                float(grid[1]),
                count,
                1,
            )
            grid_note = (
                f"on the delivered cube's channels {first}~{first + count - 1} "
                f"of its {int(grid[2])}"
            )
    channels -= channels % per_cube_channel
    if channels < per_cube_channel or channels < 1:
        raise RuntimeError(
            "The execution blocks' windows do not overlap in LSRK: "
            + ", ".join(f"{c['low']:.0f}-{c['high']:.0f} Hz" for c in coverage)
        )
    regridded: list[str] = []
    for ms_path, info in zip(window_mss, coverage, strict=True):
        target = str(_lsrk_path(Path(ms_path)))
        shutil.rmtree(target, ignore_errors=True)
        tasks.mstransform(
            vis=ms_path,
            outputvis=target,
            field=",".join(str(f) for f in info["fields"]),  # type: ignore[attr-defined]
            datacolumn=data_column[ms_path],
            regridms=True,
            mode="frequency",
            nchan=channels,
            start=f"{low!r}Hz",
            width=f"{width!r}Hz",
            outframe="LSRK",
            interpolation="linear",
        )
        regridded.append(target)
    shutil.rmtree(out_path, ignore_errors=True)
    tasks.concat(vis=regridded, concatvis=out_path, dirtol="1arcsec", copypointing=False)
    counts = {
        name: len(column(f"{out_path}/{name}", "FLAG_ROW")) for name in ("SPECTRAL_WINDOW", "FIELD")
    }
    if counts["SPECTRAL_WINDOW"] != 1 or counts["FIELD"] != 1:
        raise RuntimeError(
            f"The combined MS has {counts['SPECTRAL_WINDOW']} spectral windows and "
            f"{counts['FIELD']} fields; one of each was expected (fields more than 1 arcsec "
            "apart, or windows on different grids, are not merged)"
        )
    reference = float(coverage[0]["low"])  # type: ignore[arg-type]
    return {
        "channels": channels,
        "native_channels": native,
        "start_hz": low,
        "width_hz": width,
        "shifts_hz": [round(float(c["low"]) - reference, 1) for c in coverage],  # type: ignore[arg-type]
        "channels_out": channels // per_cube_channel if channels_out else None,
        "grid_note": grid_note,
        "selfcal_applied": applied,
    }


def delivered_channels(
    grid: Sequence[float], low: float, high: float, native_width: float
) -> tuple[int, int] | None:
    """Which channels of ``grid`` (first centre Hz, width Hz, count) can be
    regridded from data whose channel centres run from ``low`` to ``high``,
    ``native_width`` wide: ``(index of the first, how many)``, or None.

    A channel is kept when it lies within the data's band, edge to edge. A
    grid narrower than the data's channels is refused: ``mstransform`` does
    not regrid onto narrower channels.
    """
    import math

    start, width, count = float(grid[0]), float(grid[1]), int(grid[2])
    if width < native_width * (1 - 1e-6) or count < 1:
        return None
    band_low, band_high = low - native_width / 2, high + native_width / 2
    slack = 1e-6 * width
    first = max(math.ceil((band_low - (start - width / 2) - slack) / width), 0)
    last = min(math.floor((band_high - (start + width / 2) + slack) / width), count - 1)
    if last < first:
        return None
    return first, last - first + 1
