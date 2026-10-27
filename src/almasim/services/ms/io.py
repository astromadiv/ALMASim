"""Write and read ALMASim's MSv2 with casatools or python-casacore (MADIV-28 T4.9)."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .backends import _get_backend
from .subtables import (
    MS_CORR_TYPES,
    antenna_table,
    data_description_table,
    empty_table,
    feed_table,
    field_table,
    history_table,
    main_table,
    observation_table,
    polarization_table,
    processor_table,
    source_table,
    spectral_window_table,
    state_table,
    write_table,
)


def export_native_ms(
    *,
    ms_path: str | Path,
    visibility_table: dict[str, Any],
    project_name: str,
    source_name: str,
    telescope_name: str = "ALMA",
    rng: np.random.Generator | None = None,
) -> str:
    """Write a CASA MSv2 to *ms_path* using casatools or python-casacore.

    ``rng`` draws the two hands' independent noise (``subtables._hands``);
    without it they are copies of Stokes I.
    """
    backend = _get_backend()
    ms_path = str(Path(ms_path).expanduser().resolve())
    vt = visibility_table

    channel_freq_hz = np.asarray(vt["channel_freq_hz"], dtype=np.float64)
    field_ra_rad = float(vt["field_ra_rad"])
    field_dec_rad = float(vt["field_dec_rad"])
    time_mjd_s = np.asarray(vt["time_mjd_s"], dtype=np.float64)
    nchan = channel_freq_hz.shape[0]
    time_range = np.array([time_mjd_s.min(), time_mjd_s.max()])

    # The main table first: casatools creates the MS root directory with it. Its
    # links to the subtables ("Table: <path>" keywords) are put last, once the
    # subtables exist: casatools stores a link to a missing table as plain
    # text, which casacore and the imagers cannot follow.
    main = main_table(ms_path, vt, rng)
    links = {k: v for k, v in main.keywords.items() if str(v).startswith("Table: ")}
    main = replace(main, keywords={k: v for k, v in main.keywords.items() if k not in links})
    for spec in (
        main,
        antenna_table(
            list(vt["antenna_names"]),
            np.asarray(vt["antenna_positions_m"], dtype=np.float64),
            vt.get("antenna_diameters_m"),
        ),
        spectral_window_table(channel_freq_hz, nchan),
        polarization_table(),
        data_description_table(),
        field_table(source_name, field_ra_rad, field_dec_rad, time_range),
        observation_table(telescope_name, project_name, _execution_ranges(vt, time_mjd_s)),
        source_table(source_name, field_ra_rad, field_dec_rad, time_range),
        state_table(),
        history_table(project_name, time_range),
        feed_table(len(vt["antenna_names"]), time_range),
        processor_table(),
        empty_table("FLAG_CMD"),
        empty_table("POINTING"),
    ):
        write_table(backend, ms_path, spec)
    tb = backend.open(ms_path, readonly=False)
    tb.putkeywords(links)
    # The names of FLAG_CATEGORY's categories, which WSClean requires: CASA's.
    tb.putcolkeyword("FLAG_CATEGORY", "CATEGORY", ["FLAG_CMD", "ORIGINAL", "USER"])
    tb.flush()
    tb.close()
    return ms_path


def _execution_ranges(vt: dict[str, Any], time_mjd_s: np.ndarray) -> np.ndarray:
    """Each execution block's (start, end) time, in OBSERVATION_ID order."""
    ids = vt.get("observation_id")
    if ids is None:
        return np.array([[time_mjd_s.min(), time_mjd_s.max()]])
    ids = np.asarray(ids)
    ranges = []
    for i in range(int(ids.max()) + 1):
        times = time_mjd_s[ids == i] if np.any(ids == i) else time_mjd_s
        ranges.append([times.min(), times.max()])
    return np.array(ranges)


def read_native_ms(ms_path: str | Path) -> dict[str, Any]:
    """Read a CASA MSv2 back into a visibility-table dict."""
    backend = _get_backend()
    ms_path = str(Path(ms_path).expanduser().resolve())  # lgtm[py/path-injection]

    tb = backend.open(ms_path, readonly=True)
    # Visibility tables use (nrows, ncorr, nchan); the MS stores (nrows, nchan, ncorr).
    data = backend.get(tb, "DATA").transpose(0, 2, 1).astype(np.complex64)
    model_data = backend.get(tb, "MODEL_DATA").transpose(0, 2, 1).astype(np.complex64)
    flag = backend.get(tb, "FLAG").transpose(0, 2, 1)
    uvw_m = backend.get(tb, "UVW").astype(np.float64)
    antenna1 = tb.getcol("ANTENNA1").astype(np.int32)
    antenna2 = tb.getcol("ANTENNA2").astype(np.int32)
    time_mjd_s = tb.getcol("TIME").astype(np.float64)
    interval_s = tb.getcol("INTERVAL").astype(np.float64)
    exposure_s = tb.getcol("EXPOSURE").astype(np.float64)
    weight = backend.get(tb, "WEIGHT").astype(np.float32)
    sigma = backend.get(tb, "SIGMA").astype(np.float32)
    tb.close()

    pol = backend.open(os.path.join(ms_path, "POLARIZATION"), readonly=True)
    corr_types = tuple(int(c) for c in np.asarray(pol.getcell("CORR_TYPE", 0)).ravel())
    pol.close()
    if corr_types == MS_CORR_TYPES:
        # I = (XX + YY) / 2: back to the visibility table's single Stokes I
        # correlation, weighing what the two hands weigh together.
        data = data.mean(axis=1, keepdims=True).astype(np.complex64)
        model_data = model_data.mean(axis=1, keepdims=True).astype(np.complex64)
        flag = flag.any(axis=1, keepdims=True)
        weight = weight.sum(axis=1, keepdims=True).astype(np.float32)
        sigma = (
            np.sqrt(np.sum(np.square(sigma.astype(np.float64)), axis=1, keepdims=True)) / 2
        ).astype(np.float32)

    spw = backend.open(os.path.join(ms_path, "SPECTRAL_WINDOW"), readonly=True)
    channel_freq_hz = np.asarray(spw.getcell("CHAN_FREQ", 0), dtype=np.float64)
    spw.close()

    ant = backend.open(os.path.join(ms_path, "ANTENNA"), readonly=True)
    antenna_names = list(ant.getcol("NAME"))
    antenna_positions_m = backend.get(ant, "POSITION").astype(np.float64)
    antenna_diameters_m = np.asarray(ant.getcol("DISH_DIAMETER"), dtype=np.float64)
    ant.close()

    fld = backend.open(os.path.join(ms_path, "FIELD"), readonly=True)
    phase_dir = fld.getcell("PHASE_DIR", 0)
    field_ra_rad = float(np.asarray(phase_dir).flat[0])
    field_dec_rad = float(np.asarray(phase_dir).flat[1])
    source_name = str(fld.getcell("NAME", 0))
    fld.close()

    obs = backend.open(os.path.join(ms_path, "OBSERVATION"), readonly=True)
    telescope_name = str(obs.getcell("TELESCOPE_NAME", 0))
    project_name = str(obs.getcell("PROJECT", 0))
    obs.close()

    return {
        "uvw_m": uvw_m,
        "antenna1": antenna1,
        "antenna2": antenna2,
        "time_mjd_s": time_mjd_s,
        "interval_s": interval_s,
        "exposure_s": exposure_s,
        "data": data,
        "model_data": model_data,
        "flag": flag,
        "weight": weight,
        "sigma": sigma,
        "channel_freq_hz": channel_freq_hz,
        "antenna_names": antenna_names,
        "antenna_positions_m": antenna_positions_m,
        "antenna_diameters_m": antenna_diameters_m,
        "field_ra_rad": field_ra_rad,
        "field_dec_rad": field_dec_rad,
        "source_name": source_name,
        "telescope_name": telescope_name,
        "project_name": project_name,
    }
