"""WSClean geometry of a MeasurementSet: windows, fields, beam, field of view (MADIV-28 T3.6).

Moved out of ``cli/image.py``: one row per spectral window and field, with
the synthesized beam, the primary beam's field of view and the spectral
columns the cube planning needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.time import Time
from tqdm import tqdm

from almasim.services.instrument.geometry import (
    MIN_IMAGE_PIXELS,  # noqa: F401  (re-exported)
    cell_arcsec,
    image_size,
    max_baseline_m,
    primary_beam_fov_arcsec,
    synthesized_beam_arcsec,
)
from almasim.services.ms.tables import (
    TARGET_INTENT,
    intent_matches,
    open_table,
    spw_of_data_description,
)

DAY_IN_SECONDS = 3600 * 24


class MissingDependencyError(ImportError):
    """An optional dependency a command needs is not installed."""


def import_casacore_tables() -> Any:
    """``casacore.tables.table``; :class:`MissingDependencyError` without python-casacore."""
    try:
        from casacore.tables import table

        return table
    except ImportError as exc:
        raise MissingDependencyError(
            "Missing optional dependency `python-casacore` required to read metadata from "
            "the measurement set. Install it with the `ms-casacore` extra."
        ) from exc


def science_selection(input_ms: Path) -> tuple[dict[int, int], list[int]]:
    """Which spectral windows carry visibilities, and which fields are the science target.

    An ALMA MeasurementSet lists every spectral window the correlator was set
    up with (WVR, channel-average, pointing …), but after ``split`` only the
    science windows have rows; imaging the others is wasted work. The main
    table also holds the calibrator scans, so the target fields are those
    observed with the ``OBSERVE_TARGET`` intent.

    Returns ``({spw_id: n_rows}, [target field ids])``.
    """
    rows_per_spw, target_fields, _ = _field_scan(input_ms)
    return rows_per_spw, target_fields


def _field_scan(
    input_ms: Path,
    intents: Sequence[str] = (TARGET_INTENT,),
) -> tuple[dict[int, int], list[int], dict[tuple[int, int], int]]:
    """``science_selection`` plus ``{(spw_id, field_id): n_rows}`` for every field.

    The selected fields are those with rows under one of ``intents``
    (``OBSERVE_TARGET`` by default).

    The per-field counts cover all of a field's rows in a window, whatever the
    intent, which is what the single-window extraction copies for it.
    """
    opener = import_casacore_tables()
    with open_table(f"{input_ms}::DATA_DESCRIPTION", opener=opener) as data_description:
        spw_of_dd = np.asarray(spw_of_data_description(data_description))
    with open_table(f"{input_ms}::STATE", opener=opener) as states:
        obs_modes = list(states.getcol("OBS_MODE")) if states.nrows() else []
    with open_table(str(input_ms), opener=opener) as main:
        dd_ids = np.asarray(main.getcol("DATA_DESC_ID"))
        field_ids = np.asarray(main.getcol("FIELD_ID"))
        state_ids = np.asarray(main.getcol("STATE_ID"))
    unique_dd, counts = np.unique(dd_ids, return_counts=True)
    rows_per_spw: dict[int, int] = {}
    for dd_id, count in zip(unique_dd, counts, strict=True):
        spw = int(spw_of_dd[int(dd_id)])
        rows_per_spw[spw] = rows_per_spw.get(spw, 0) + int(count)

    rows_per_spw_field: dict[tuple[int, int], int] = {}
    pairs, pair_counts = np.unique(np.stack([dd_ids, field_ids]), axis=1, return_counts=True)
    for (dd_id, field_id), count in zip(pairs.T, pair_counts, strict=True):
        key = (int(spw_of_dd[int(dd_id)]), int(field_id))
        rows_per_spw_field[key] = rows_per_spw_field.get(key, 0) + int(count)

    target_states = {i for i, mode in enumerate(obs_modes) if intent_matches(mode, intents)}
    target_fields: list[int] = []
    if target_states:
        mask = np.isin(state_ids, list(target_states))
        target_fields = sorted(int(f) for f in np.unique(field_ids[mask]))
    return rows_per_spw, target_fields, rows_per_spw_field


SPECTRAL_COLUMNS = (
    "n_channels",
    "channel_width_hz",
    "spectral_resolution_hz",
    "total_bandwidth_hz",
)


def spectral_window_properties(spectral_windows: Any) -> dict[str, np.ndarray]:
    """Channelisation of every row of an open SPECTRAL_WINDOW table.

    ``channel_width_hz`` and ``spectral_resolution_hz`` are the median of the
    per-channel CHAN_WIDTH and RESOLUTION (absolute: ALMA lower-sideband
    windows have negative widths). ALMA's online Hanning smoothing makes the
    resolution of an unaveraged FDM window twice its channel width (3840 x
    488 kHz channels, 977 kHz resolution); an online-averaged window has a
    resolution about equal to its width.
    """
    n_rows = spectral_windows.nrows()
    widths = np.empty(n_rows)
    resolutions = np.empty(n_rows)
    for row in range(n_rows):
        widths[row] = np.median(np.abs(spectral_windows.getcell("CHAN_WIDTH", row)))
        resolutions[row] = np.median(np.abs(spectral_windows.getcell("RESOLUTION", row)))
    return {
        "n_channels": np.asarray(spectral_windows.getcol("NUM_CHAN"), dtype=int),
        "channel_width_hz": widths,
        "spectral_resolution_hz": resolutions,
        "total_bandwidth_hz": np.abs(np.asarray(spectral_windows.getcol("TOTAL_BANDWIDTH"))),
    }


def add_spectral_columns(parameters: pd.DataFrame) -> pd.DataFrame:
    """Fill ``SPECTRAL_COLUMNS`` from each MS when a CSV predates them.

    Reads only the SPECTRAL_WINDOW subtables, so an existing parameter CSV
    can be imaged as cubes without rerunning ``compute-parameters``.
    """
    if all(column in parameters.columns for column in SPECTRAL_COLUMNS):
        return parameters
    casacore_table = import_casacore_tables()
    filled = parameters.copy()
    for column in SPECTRAL_COLUMNS:
        if column not in filled.columns:
            filled[column] = np.nan
    for filename, rows in tqdm(filled.groupby("filename").groups.items(), desc="Spectral windows"):
        with open_table(f"{filename}::SPECTRAL_WINDOW", opener=casacore_table) as spectral_windows:
            spectral = spectral_window_properties(spectral_windows)
        spws = filled.loc[rows, "spectral_window_id"].astype(int).to_numpy()
        for column, values in spectral.items():
            filled.loc[rows, column] = values[spws]
    filled["n_channels"] = filled["n_channels"].astype(int)
    return filled


def image_pixels(
    imaging_parameters: pd.Series,
    fov_fraction: float,
    beam_sampling: float,
    min_beams_across: float = 0.0,
) -> int:
    """Image side in pixels: the FOV over the cell (beam / ``beam_sampling``).

    ``min_beams_across`` widens images narrower than that many synthesized
    beams: a compact configuration's primary beam spans few of them.
    """
    return image_size(
        imaging_parameters["fov_per_frequency"] * fov_fraction,
        cell_arcsec(imaging_parameters["synthetized_beam_size"], beam_sampling),
        min_pixels=min_beams_across * beam_sampling,
    )


def compute_imaging_parameters(
    input_ms: Path,
    science_only: bool = True,
    intents: Sequence[str] | None = None,
) -> pd.DataFrame:
    """WSClean geometry for ``input_ms``: one row per spectral window and field.

    Every row images one field: ``target_field_ids`` holds its single id and
    ``field_name`` its name. WSClean's ``-field`` needs all listed fields to
    share a phase centre, and multi-target observations do not: ``-field 2,4``
    on HOPS-358 and HOPS-373, 12.4' apart, gridded both onto one image
    (uid___A002_X10d9399_X32fd, Cycle 11).

    Only (window, field) pairs that hold visibilities get a row. With
    ``science_only`` (the default) the fields are those observed with one of
    ``intents`` (``OBSERVE_TARGET`` unless given, e.g.
    ``CALIBRATE_PHASE#ON_SOURCE`` for the phase calibrators); an MS without
    such a field gives no rows. Otherwise every field is listed, calibrators
    included, and ``intents`` must not be given.
    """
    if intents and not science_only:
        raise ValueError("intents select fields; they cannot be combined with all fields")
    opener = import_casacore_tables()
    with open_table(f"{input_ms}::SPECTRAL_WINDOW", opener=opener) as spectral_windows:
        reference_frequencies = spectral_windows.getcol("REF_FREQUENCY")
        window_names = np.asarray([str(n) for n in spectral_windows.getcol("NAME")], dtype=object)
        spectral = spectral_window_properties(spectral_windows)
    with open_table(f"{input_ms}::OBSERVATION", opener=opener) as observation:
        start_mjd, end_mjd = observation[0]["TIME_RANGE"]
    start_datetime = Time(start_mjd / DAY_IN_SECONDS, format="mjd").to_datetime()
    end_datetime = Time(end_mjd / DAY_IN_SECONDS, format="mjd").to_datetime()
    with open_table(f"{input_ms}::ANTENNA", opener=opener) as antennas:
        min_dish_diameter = np.min(antennas.getcol("DISH_DIAMETER"))
        antenna_pos = antennas.getcol("POSITION")
    max_baseline_size = max_baseline_m(antenna_pos)
    # The primary beam's half-power width (the Airy model of primary_beam).
    fov_per_frequency = primary_beam_fov_arcsec(float(min_dish_diameter), reference_frequencies)
    synthetized_beam_size = synthesized_beam_arcsec(reference_frequencies, max_baseline_size)
    spectral_window_id = np.arange(reference_frequencies.size, dtype=int)

    _, selected_fields, rows_per_spw_field = _field_scan(
        input_ms, tuple(intents) if intents else (TARGET_INTENT,)
    )
    # (index into the spectral-window arrays, field id, n rows): only pairs
    # with visibilities, so every row is something WSClean can image.
    entries: list[tuple[int, int, int]] = []
    for index, spw in enumerate(int(s) for s in spectral_window_id):
        fields = sorted(
            f
            for (s, f), rows in rows_per_spw_field.items()
            if s == spw and rows > 0 and (not science_only or f in selected_fields)
        )
        entries.extend((index, f, rows_per_spw_field[(spw, f)]) for f in fields)
    field_names: list[str] = []
    if entries:
        with open_table(f"{input_ms}::FIELD", opener=opener) as field_table:
            field_names = [str(name) for name in field_table.getcol("NAME")]
    keep = np.array([index for index, _, _ in entries], dtype=int)
    n_rows = len(entries)

    derived_parameters = pd.DataFrame(
        {
            "filename": [str(input_ms.resolve())] * n_rows,
            "spectral_window_id": spectral_window_id[keep],
            "spectral_window_name": window_names[keep],
            "reference_frequency": reference_frequencies[keep],
            "fov_per_frequency": fov_per_frequency[keep],
            "max_baseline_size": [max_baseline_size] * n_rows,
            "synthetized_beam_size": synthetized_beam_size[keep],
            **{column: values[keep] for column, values in spectral.items()},
            "start_mjd_s": start_mjd,
            "end_mjd_s": end_mjd,
            "duration_s": end_mjd - start_mjd,
            "start_datetime": start_datetime.isoformat(),
            "end_datetime": end_datetime.isoformat(),
            "n_visibility_rows": [rows for _, _, rows in entries],
            "target_field_ids": [str(f) for _, f, _ in entries],
            "field_name": [field_names[f] for _, f, _ in entries],
        }
    )
    return derived_parameters
