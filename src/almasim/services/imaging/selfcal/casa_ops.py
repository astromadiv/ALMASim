"""CASA operations for self-calibration: gaincal, applycal and gain tables.

Moved out of archive_imaging.py (MADIV-28 module split).
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Sequence
from pathlib import Path

from almasim.services.imaging import cont_dat
from almasim.services.imaging.continuum import _channel_ranges, line_free_channels
from almasim.services.imaging.ms.chunks import (
    _default_rows_per_chunk,
    _has_spectrum_weight,
    _parallel_hands,
    _read_rows,
    _source_spectrum,
)

from ...casa import tasks as casa_tasks
from ...casa import tools as casa_tools
from ...ms.tables import antenna_names, open_table


def _configure_casa(casa_data: str, log_file: str) -> Path:
    """Point CASA at ``casa_data`` and ``log_file`` with a private workspace (removed at exit)."""
    import atexit
    import tempfile

    from almasim.services.casa.environment import configure_casa_environment

    workspace = Path(tempfile.mkdtemp(prefix="almasim-selfcal-casa-"))
    atexit.register(shutil.rmtree, workspace, True)
    configure_casa_environment(workspace, casa_data, workspace_root=workspace, log_file=log_file)
    return workspace


def selfcal_prepare(
    ms_path: str,
    rows_per_chunk: int | None = None,
    pipeline_ms: str | None = None,
    ous_index: str | None = None,
) -> dict[str, object]:
    """CASA selections for self-calibrating ``ms_path``: the line-free channels of
    DATA's source spectrum (``spw``) and the antenna with the most
    cross-correlation rows holding an unflagged line-free parallel-hand sample
    (``refant``); a row unflagged as a whole can still have every such sample
    flagged.

    With ``pipeline_ms`` the line channels of the ALMA pipeline's ``cont.dat``
    are left out too, as ``subtract_uv_continuum`` leaves them out, and the
    result carries its ``cont_dat`` and ``cont_dat_line_channels``."""
    import numpy as np
    from casacore.tables import table

    pipeline_line, pipeline_note = (
        cont_dat.pipeline_line_channels(ms_path, pipeline_ms, ous_index)
        if pipeline_ms is not None
        else (None, "")
    )
    pipeline: dict[str, object] = {}
    ms = table(str(ms_path), ack=False)
    try:
        if rows_per_chunk is None:
            rows_per_chunk = _default_rows_per_chunk(ms, "DATA")
        measured = _source_spectrum(ms, "DATA", rows_per_chunk)
        if measured is None:
            raise RuntimeError("the MS has no rows")
        spectrum, usable = measured
        line_free = line_free_channels(spectrum, usable)
        if pipeline_ms is not None:
            line_free, pipeline = cont_dat.without_pipeline_lines(
                line_free, pipeline_line, pipeline_note
            )
        if not line_free.any():
            line_free = usable
        ant1, ant2 = ms.getcol("ANTENNA1"), ms.getcol("ANTENNA2")
        has_data = np.zeros(ms.nrows(), dtype=bool)
        spectrum_weight = _has_spectrum_weight(ms)
        for start in range(0, ms.nrows(), rows_per_chunk):
            n = min(rows_per_chunk, ms.nrows() - start)
            _, _, flag = _read_rows(ms, "DATA", start, n, spectrum_weight)
            corrs = _parallel_hands(flag.shape[2])
            has_data[start : start + n] = (~flag[:, line_free][..., corrs]).any(axis=(1, 2))
        good = has_data & (ant1 != ant2)
    finally:
        ms.close()
    with open_table(f"{ms_path}/ANTENNA") as antenna:
        names = antenna_names(antenna)
    counts = np.bincount(ant1[good], minlength=len(names)) + np.bincount(
        ant2[good], minlength=len(names)
    )
    return {
        "spw": "0:" + _channel_ranges(line_free),
        "refant": str(names[int(np.argmax(counts))]),
        "line_free_channels": int(line_free.sum()),
        **pipeline,
    }


def adopt_corrected_data(ms_path: str, rows_per_chunk: int | None = None) -> dict[str, object]:
    """Make an incoming CORRECTED_DATA the task copy's DATA, and drop it.

    Self-calibration solves from DATA, writes CORRECTED_DATA and, when a
    round is rejected, drops it: an MS that arrived with its calibration in
    CORRECTED_DATA would then be solved from, and fall back to, the
    uncalibrated DATA. The single-window MS is the task's own copy.
    """
    from casacore.tables import table

    ms = table(str(ms_path), readonly=False, ack=False)
    try:
        if "CORRECTED_DATA" not in ms.colnames():
            return {"adopted": False}
        if rows_per_chunk is None:
            rows_per_chunk = _default_rows_per_chunk(ms, "CORRECTED_DATA")
        for start in range(0, ms.nrows(), rows_per_chunk):
            n = min(rows_per_chunk, ms.nrows() - start)
            ms.putcol(
                "DATA", ms.getcol("CORRECTED_DATA", startrow=start, nrow=n), startrow=start, nrow=n
            )
        ms.removecols("CORRECTED_DATA")
        return {"adopted": True}
    finally:
        ms.close()


def casa_selfcal_solve(
    ms_path: str,
    caltable: str,
    *,
    solint: str,
    spw: str,
    refant: str,
    casa_data: str,
    log_file: str,
    min_snr: float = 3.0,
    calmode: str = "p",
    pre_apply: Sequence[str] = (),
) -> dict[str, object]:
    """CASA ``gaincal`` of DATA against MODEL_DATA: one solution per antenna for
    both polarizations (gaintype T), over ``spw`` (line-free).

    ``calmode`` is ``p`` (phase) or ``ap`` (amplitude and phase); an ``ap``
    solve is made on top of the ``pre_apply`` tables (the kept phase
    solution) and normalised (``solnorm``) so the flux scale stays put.
    """
    import numpy as np

    _configure_casa(casa_data, log_file)
    gaincal = casa_tasks().gaincal
    table = casa_tools().table

    started = time.time()
    gaincal(
        vis=ms_path,
        caltable=caltable,
        solint=solint,
        spw=spw,
        refant=refant,
        calmode=calmode,
        gaintype="T",
        minsnr=min_snr,
        minblperant=4,
        solnorm=calmode == "ap",
        gaintable=list(pre_apply),
        interp=["linear"] * len(pre_apply),
    )
    tb = table()
    tb.open(caltable)
    try:
        flags = tb.getcol("FLAG")
        gains = tb.getcol("CPARAM")
        times = tb.getcol("TIME")
    finally:
        tb.close()
    good = ~flags
    phases = np.degrees(np.abs(np.angle(gains[good]))) if good.any() else np.zeros(1)
    amplitudes = np.abs(gains[good]) if good.any() else np.ones(1)
    return {
        "calmode": calmode,
        "median_amp_dev": round(float(np.median(np.abs(amplitudes - 1.0))), 3),
        "solint": solint,
        "intervals": int(np.unique(times).size),
        "solutions": int(good.sum()),
        "possible": int(flags.size),
        "median_phase_deg": round(float(np.median(phases)), 1),
        "seconds": round(time.time() - started, 1),
    }


def casa_selfcal_apply(
    ms_path: str, caltables: Sequence[str], *, casa_data: str, log_file: str
) -> dict:
    """CASA ``applycal`` of ``caltables`` into CORRECTED_DATA. ``calonly``: data
    whose solution failed stay as they were, unflagged; weights unchanged."""
    _configure_casa(casa_data, log_file)
    applycal = casa_tasks().applycal

    applycal(
        vis=ms_path,
        gaintable=list(caltables),
        interp=["linear"] * len(caltables),
        calwt=[False] * len(caltables),
        applymode="calonly",
        flagbackup=False,
    )
    return {"applied": True}


def export_gain_table(caltable: str, csv_path: str) -> dict[str, object]:
    """A CASA gain table as CSV: one row per solution (time, antenna, polarization),
    with amplitude, phase in degrees and the flag. Read with casacore, no CASA."""
    import csv

    import numpy as np
    from casacore.tables import table

    tb = table(str(caltable), ack=False)
    try:
        times = tb.getcol("TIME")
        antennas = tb.getcol("ANTENNA1")
        gains = tb.getcol("CPARAM")  # (rows, channels, polarizations)
        flags = tb.getcol("FLAG")
    finally:
        tb.close()
    with open_table(f"{caltable}/ANTENNA") as antenna:
        names = antenna_names(antenna)
    rows = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["time_mjd_s", "antenna", "polarization", "amplitude", "phase_deg", "flagged"]
        )
        for row in range(len(times)):
            antenna = int(antennas[row])
            for pol in range(gains.shape[2]):
                gain = complex(gains[row, 0, pol])
                writer.writerow(
                    [
                        f"{times[row]:.3f}",
                        str(names[antenna]) if antenna < len(names) else antenna,
                        pol,
                        f"{abs(gain):.6g}",
                        f"{np.degrees(np.angle(gain)):.4g}",
                        bool(flags[row, 0, pol]),
                    ]
                )
                rows += 1
    return {"rows": rows}
