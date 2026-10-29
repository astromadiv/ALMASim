"""What ALMASim writes into each MeasurementSet table, and one writer for all (MADIV-28 T4.9).

Each ``*_table`` function returns a :class:`TableSpec`: the table's columns
and values. :func:`write_table` creates any of them, so the main table and the
nine subtables share one code path.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import schema
from .schema import Column


@dataclass(frozen=True)
class TableSpec:
    """One table to write.

    ``values`` holds whole columns when ``by_column`` (the main and ANTENNA
    tables), otherwise the cells of the table's single row.
    """

    name: str  # the subtable's name; "" for the main table
    columns: Sequence[Column]
    values: Mapping[str, Any]
    nrow: int = 1
    by_column: bool = False
    keywords: Mapping[str, Any] = field(default_factory=dict)
    info: Mapping[str, Any] | None = None


def write_table(backend, ms_path: str, spec: TableSpec) -> None:
    """Create ``spec``'s table under ``ms_path`` and fill it."""
    path = os.path.join(ms_path, spec.name) if spec.name else ms_path
    tb = backend.create(path, [column.desc() for column in spec.columns], nrow=spec.nrow)
    arrays = {column.name for column in spec.columns if column.ndim}
    for name, value in spec.values.items():
        if spec.by_column:
            if name in arrays:
                backend.put(tb, name, value)
            else:
                tb.putcol(name, value)
        elif name in arrays:
            backend.put_cell(tb, name, 0, value)
        else:
            tb.putcell(name, 0, value)
    for key, value in spec.keywords.items():
        tb.putkeyword(key, value)
    if spec.info is not None:
        tb.putinfo(dict(spec.info))
    tb.flush()
    tb.close()


def _as_xx_yy(values: Any, dtype: Any, axis: int) -> np.ndarray:
    """A single (Stokes I) correlation repeated as XX and YY along ``axis``."""
    array = np.asarray(values, dtype=dtype)
    if array.shape[axis] != 1:
        raise ValueError(f"expected one (Stokes I) correlation, got {array.shape[axis]}")
    return np.repeat(array, len(MS_CORR_TYPES), axis=axis)


def _channel_noise(
    rng: np.random.Generator, shape: tuple[int, ...], kernel: Any | None
) -> np.ndarray:
    """Unit complex noise of ``shape`` (rows, correlations, channels), its
    channels correlated through the 3-tap ``kernel`` (unit norm, so the level
    stays 1); white when None."""
    if kernel is None:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    taps = np.asarray(kernel, dtype=np.float64)
    nchan = shape[-1]
    wide = (*shape[:-1], nchan + len(taps) - 1)
    white = rng.standard_normal(wide) + 1j * rng.standard_normal(wide)
    out = np.zeros(shape, dtype=np.complex128)
    for k, w in enumerate(taps):
        out += w * white[..., k : k + nchan]
    return out


def _hands(
    vt: Mapping[str, Any], rng: np.random.Generator | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DATA, WEIGHT and SIGMA of the two hands from Stokes I.

    XX = I + d and YY = I - d, with d independent noise at each visibility's
    own level (``sigma_spectrum``, else its row's SIGMA), its channels
    correlated as I's (``channel_kernel``, the correlator's smoothing, which
    each hand has on its own): each hand then carries independent noise of
    sqrt(2) times Stokes I's, as two real polarisations do, and (XX + YY) / 2
    is still I. Each hand's WEIGHT is half
    Stokes I's, so that the two together weigh what I does; copying I into
    both hands at I's weight made an imager expect sqrt(2) less noise than the
    data carry. Without ``rng`` the hands are copies of I, at half its weight.
    """
    data = np.asarray(vt["data"], dtype=np.complex64)
    if data.shape[1] != 1:
        raise ValueError(f"expected one (Stokes I) correlation, got {data.shape[1]}")
    hands = np.repeat(data, len(MS_CORR_TYPES), axis=1)
    if rng is not None:
        noise = vt.get("sigma_spectrum")
        noise = np.asarray(
            noise if noise is not None else np.asarray(vt["sigma"])[:, :, None], dtype=np.float32
        )
        delta = _channel_noise(rng, data.shape, vt.get("channel_kernel")) * noise
        hands[:, :1] += delta.astype(np.complex64)
        hands[:, 1:] -= delta.astype(np.complex64)
    weight = _as_xx_yy(np.asarray(vt["weight"], dtype=np.float32) / 2, np.float32, axis=1)
    sigma = _as_xx_yy(np.asarray(vt["sigma"], dtype=np.float32) * np.sqrt(2), np.float32, axis=1)
    return hands, weight, sigma


def main_table(
    ms_path: str, vt: Mapping[str, Any], rng: np.random.Generator | None = None
) -> TableSpec:
    """The visibilities, in the MS's (nrows, nchan, ncorr) layout, Stokes I as XX and
    YY with independent noise (``_hands``)."""
    data, weight, sigma = _hands(vt, rng)
    flag = _as_xx_yy(vt["flag"], bool, axis=1)
    time_mjd_s = np.asarray(vt["time_mjd_s"], dtype=np.float64)
    nrows = data.shape[0]
    values = {
        "TIME": time_mjd_s,
        "TIME_CENTROID": time_mjd_s,
        "INTERVAL": np.asarray(vt["interval_s"], dtype=np.float64),
        "EXPOSURE": np.asarray(vt["exposure_s"], dtype=np.float64),
        "ANTENNA1": np.asarray(vt["antenna1"], dtype=np.int32),
        "ANTENNA2": np.asarray(vt["antenna2"], dtype=np.int32),
        "FEED1": np.zeros(nrows, dtype=np.int32),
        "FEED2": np.zeros(nrows, dtype=np.int32),
        "DATA_DESC_ID": np.zeros(nrows, dtype=np.int32),
        "FIELD_ID": np.zeros(nrows, dtype=np.int32),
        "ARRAY_ID": np.zeros(nrows, dtype=np.int32),
        "OBSERVATION_ID": (
            np.asarray(vt["observation_id"], dtype=np.int32)
            if vt.get("observation_id") is not None
            else np.zeros(nrows, dtype=np.int32)
        ),
        "PROCESSOR_ID": np.full(nrows, -1, dtype=np.int32),
        "SCAN_NUMBER": (
            np.asarray(vt["scan_number"], dtype=np.int32)
            if vt.get("scan_number") is not None
            else np.ones(nrows, dtype=np.int32)
        ),
        "STATE_ID": np.zeros(nrows, dtype=np.int32),
        "FLAG_ROW": np.all(flag, axis=(1, 2)),
        # The visibility table's (nrows, ncorr, nchan) becomes (nrows, nchan, ncorr).
        "UVW": np.asarray(vt["uvw_m"], dtype=np.float64),
        "DATA": data.transpose(0, 2, 1),
        "MODEL_DATA": _as_xx_yy(vt["model_data"], np.complex64, axis=1).transpose(0, 2, 1),
        "FLAG": flag.transpose(0, 2, 1),
        "WEIGHT": weight,  # (nrows, ncorr)
        "SIGMA": sigma,
    }
    keywords: dict[str, Any] = {"MS_VERSION": 2.0}
    keywords.update(
        {name: "Table: " + os.path.join(ms_path, name) for name in schema.SUBTABLE_NAMES}
    )
    return TableSpec(
        "",
        schema.MAIN_COLUMNS,
        values,
        nrow=nrows,
        by_column=True,
        keywords=keywords,
        info={"type": "Measurement Set", "subType": "ALMASim"},
    )


def antenna_table(
    antenna_names: Sequence[str],
    antenna_positions_m: np.ndarray,
    dish_diameters_m: np.ndarray | None = None,
) -> TableSpec:
    """One row per antenna, with its own dish diameter (12 m each when not given)."""
    nant = len(antenna_names)
    diameters = (
        np.full(nant, 12.0)
        if dish_diameters_m is None
        else np.asarray(dish_diameters_m, dtype=np.float64)
    )
    if diameters.shape != (nant,):
        raise ValueError(f"{diameters.shape[0]} dish diameters for {nant} antennas")
    values = {
        "NAME": list(antenna_names),
        "STATION": list(antenna_names),
        "TYPE": ["GROUND-BASED"] * nant,
        "MOUNT": ["alt-az"] * nant,
        "DISH_DIAMETER": diameters,
        "FLAG_ROW": np.zeros(nant, dtype=bool),
        "POSITION": antenna_positions_m,  # (nant, 3)
        "OFFSET": np.zeros((nant, 3)),
    }
    return TableSpec("ANTENNA", schema.ANTENNA_COLUMNS, values, nrow=nant, by_column=True)


def spectral_window_table(channel_freq_hz: np.ndarray, nchan: int) -> TableSpec:
    chan_width = np.gradient(channel_freq_hz).astype(np.float64) if nchan > 1 else np.array([1.0])
    values = {
        "NUM_CHAN": int(nchan),
        "CHAN_FREQ": channel_freq_hz,
        "CHAN_WIDTH": chan_width,
        "EFFECTIVE_BW": np.abs(chan_width),
        "RESOLUTION": np.abs(chan_width),
        "REF_FREQUENCY": float(np.median(channel_freq_hz)),
        "TOTAL_BANDWIDTH": float(np.sum(np.abs(chan_width))),
        "NAME": "ALMASim SPW 0",
        "MEAS_FREQ_REF": 5,
        "NET_SIDEBAND": 1,
        "FLAG_ROW": False,
        "FREQ_GROUP": 0,
        "FREQ_GROUP_NAME": "",
        "IF_CONV_CHAIN": 0,
    }
    return TableSpec("SPECTRAL_WINDOW", schema.SPECTRAL_WINDOW_COLUMNS, values)


# XX and YY (casacore Stokes types 9 and 12): the simulated Stokes I in both,
# so imagers that form I = (XX + YY) / 2 recover it (MADIV-28 T5.12).
MS_CORR_TYPES = (9, 12)


def polarization_table() -> TableSpec:
    values = {
        "NUM_CORR": len(MS_CORR_TYPES),
        "CORR_TYPE": np.array(MS_CORR_TYPES, dtype=np.int32),
        "CORR_PRODUCT": np.array([[0, 0], [1, 1]], dtype=np.int32),  # (ncorr, 2)
        "FLAG_ROW": False,
    }
    return TableSpec("POLARIZATION", schema.POLARIZATION_COLUMNS, values)


def data_description_table() -> TableSpec:
    values = {"SPECTRAL_WINDOW_ID": 0, "POLARIZATION_ID": 0, "FLAG_ROW": False}
    return TableSpec("DATA_DESCRIPTION", schema.DATA_DESCRIPTION_COLUMNS, values)


def field_table(
    source_name: str, ra_rad: float, dec_rad: float, time_range: np.ndarray
) -> TableSpec:
    direction = np.array([[ra_rad, dec_rad]])  # shape (1, 2): one polynomial term
    values = {
        "NAME": source_name,
        "CODE": "",
        "NUM_POLY": 0,
        "SOURCE_ID": 0,
        "TIME": float(np.mean(time_range)),
        "FLAG_ROW": False,
        "DELAY_DIR": direction,
        "PHASE_DIR": direction,
        "REFERENCE_DIR": direction,
    }
    return TableSpec("FIELD", schema.FIELD_COLUMNS, values)


def observation_table(telescope_name: str, project_name: str, time_ranges: np.ndarray) -> TableSpec:
    """One row per execution block, ``time_ranges`` its (start, end) each
    (one (2,) range: one execution)."""
    ranges = np.asarray(time_ranges, dtype=np.float64).reshape(-1, 2)
    n = len(ranges)
    values = {
        "TELESCOPE_NAME": [telescope_name] * n,
        "OBSERVER": ["ALMASim"] * n,
        "PROJECT": [project_name] * n,
        "SCHEDULE_TYPE": ["ALMASim"] * n,
        "RELEASE_DATE": ranges[:, 1].copy(),
        "FLAG_ROW": np.zeros(n, dtype=bool),
        "TIME_RANGE": ranges,
        "LOG": np.array([["ALMASim simulation"]] * n),
        "SCHEDULE": np.array([["ALMASim"]] * n),
    }
    return TableSpec("OBSERVATION", schema.OBSERVATION_COLUMNS, values, nrow=n, by_column=True)


def feed_table(n_antennas: int, time_range: np.ndarray) -> TableSpec:
    """One feed per antenna, two linear receptors (X and Y) at right angles."""
    values = {
        "ANTENNA_ID": np.arange(n_antennas, dtype=np.int32),
        "FEED_ID": np.zeros(n_antennas, dtype=np.int32),
        "SPECTRAL_WINDOW_ID": np.full(n_antennas, -1, dtype=np.int32),
        "TIME": np.full(n_antennas, float(np.mean(time_range))),
        "INTERVAL": np.full(n_antennas, float(np.ptp(time_range)) + 1.0),
        "NUM_RECEPTORS": np.full(n_antennas, 2, dtype=np.int32),
        "BEAM_ID": np.full(n_antennas, -1, dtype=np.int32),
        "BEAM_OFFSET": np.zeros((n_antennas, 2, 2)),
        "POLARIZATION_TYPE": np.tile(np.array(["X", "Y"]), (n_antennas, 1)),
        "POL_RESPONSE": np.tile(np.eye(2, dtype=np.complex64), (n_antennas, 1, 1)),
        "POSITION": np.zeros((n_antennas, 3)),
        "RECEPTOR_ANGLE": np.tile(np.array([0.0, np.pi / 2]), (n_antennas, 1)),
    }
    return TableSpec("FEED", schema.FEED_COLUMNS, values, nrow=n_antennas, by_column=True)


def processor_table() -> TableSpec:
    values = {
        "TYPE": "CORRELATOR",
        "SUB_TYPE": "ALMASim",
        "TYPE_ID": -1,
        "MODE_ID": -1,
        "FLAG_ROW": False,
    }
    return TableSpec("PROCESSOR", schema.PROCESSOR_COLUMNS, values)


def empty_table(name: str) -> TableSpec:
    """A required subtable with its columns and no rows (FLAG_CMD, POINTING)."""
    columns = {"FLAG_CMD": schema.FLAG_CMD_COLUMNS, "POINTING": schema.POINTING_COLUMNS}[name]
    return TableSpec(name, columns, {}, nrow=0)


def source_table(
    source_name: str, ra_rad: float, dec_rad: float, time_range: np.ndarray
) -> TableSpec:
    values = {
        "SOURCE_ID": 0,
        "TIME": float(np.mean(time_range)),
        "INTERVAL": float(time_range[1] - time_range[0]),
        "SPECTRAL_WINDOW_ID": -1,
        "NUM_LINES": 0,
        "NAME": source_name,
        "CODE": "",
        "CALIBRATION_GROUP": 0,
        "DIRECTION": np.array([ra_rad, dec_rad]),
        "PROPER_MOTION": np.zeros(2),
    }
    return TableSpec("SOURCE", schema.SOURCE_COLUMNS, values)


def state_table() -> TableSpec:
    values = {
        "SIG": True,
        "REF": False,
        "CAL": 0.0,
        "LOAD": 0.0,
        "SUB_SCAN": 0,
        # The intent the imaging pipeline selects target fields by.
        "OBS_MODE": "OBSERVE_TARGET#ON_SOURCE",
        "FLAG_ROW": False,
    }
    return TableSpec("STATE", schema.STATE_COLUMNS, values)


def history_table(
    project_name: str, time_range: np.ndarray, app_params: Sequence[str] = ()
) -> TableSpec:
    values = {
        "TIME": float(np.mean(time_range)),
        "OBSERVATION_ID": 0,
        "MESSAGE": "ALMASim MSv2 export",
        "PRIORITY": "NORMAL",
        "ORIGIN": "ALMASim",
        "OBJECT_ID": 0,
        "APPLICATION": "ALMASim",
        "CLI_COMMAND": np.array(["export_native_ms"]),
        "APP_PARAMS": np.array([project_name, *app_params]),
    }
    return TableSpec("HISTORY", schema.HISTORY_COLUMNS, values)
