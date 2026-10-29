"""Row-wise visibility helpers for MS-style export."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.time import Time


@dataclass
class VisibilityTable:
    """Internal row-wise visibility representation."""

    uvw_m: np.ndarray
    antenna1: np.ndarray
    antenna2: np.ndarray
    time_mjd_s: np.ndarray
    interval_s: np.ndarray
    exposure_s: np.ndarray
    data: np.ndarray
    model_data: np.ndarray
    flag: np.ndarray
    weight: np.ndarray
    sigma: np.ndarray
    channel_freq_hz: np.ndarray
    antenna_names: list[str]
    antenna_positions_m: np.ndarray
    source_name: str
    field_ra_rad: float
    field_dec_rad: float
    observation_date: str
    config_name: str | None = None
    array_type: str | None = None
    # One dish diameter per antenna (m); 12 m each when unknown.
    antenna_diameters_m: np.ndarray | None = None
    # Each row's scan (1, 2, ...) when the track is split into scans
    # (corruptions.CorruptionModel.scan_length_s); one scan when None.
    scan_number: np.ndarray | None = None
    # Each row's execution block (the MS's OBSERVATION_ID); 0 for all when None.
    observation_id: np.ndarray | None = None
    # Each visibility's own noise (nrows, ncorr, nchan), per real and imaginary
    # part; the MS writer gives the two hands independent noise from it.
    sigma_spectrum: np.ndarray | None = None
    # The 3-tap kernel correlating neighbouring channels' noise
    # (corruptions.spectral_kernel); the hands' own noise goes through it too.
    # None: channels independent.
    channel_kernel: np.ndarray | None = None

    def as_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for serialization."""
        return {
            "uvw_m": self.uvw_m,
            "antenna1": self.antenna1,
            "antenna2": self.antenna2,
            "time_mjd_s": self.time_mjd_s,
            "interval_s": self.interval_s,
            "exposure_s": self.exposure_s,
            "data": self.data,
            "model_data": self.model_data,
            "flag": self.flag,
            "weight": self.weight,
            "sigma": self.sigma,
            "channel_freq_hz": self.channel_freq_hz,
            "antenna_names": self.antenna_names,
            "antenna_positions_m": self.antenna_positions_m,
            "antenna_diameters_m": self.antenna_diameters_m,
            "source_name": self.source_name,
            "field_ra_rad": self.field_ra_rad,
            "field_dec_rad": self.field_dec_rad,
            "observation_date": self.observation_date,
            "config_name": self.config_name,
            "array_type": self.array_type,
            **({"scan_number": self.scan_number} if self.scan_number is not None else {}),
            **({"observation_id": self.observation_id} if self.observation_id is not None else {}),
            **({"sigma_spectrum": self.sigma_spectrum} if self.sigma_spectrum is not None else {}),
            **({"channel_kernel": self.channel_kernel} if self.channel_kernel is not None else {}),
        }


def build_channel_visibility_rows(
    *,
    modelfft: np.ndarray,
    gains: np.ndarray,
    noise_samples: np.ndarray,
    u_waves: np.ndarray,
    v_waves: np.ndarray,
    w_waves: np.ndarray,
    antnum: np.ndarray,
    nH: int,
    Nphf: int,
    uv_pixsize: float,
    noise_std: float,
    mean_wavelength_m: float,
    noise_scale: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build per-channel row-wise visibility samples before UV gridding.

    ``noise_scale`` (baselines): each baseline's noise over the channel's, as
    in ``noise_samples`` (its antennas' own noise shape); the rows' sigma
    carries it too. None: 1.
    """
    nbas = int(antnum.shape[0])
    nrows = int(nbas * nH)
    antenna1 = np.repeat(antnum[:, 0].astype(np.int32), nH)
    antenna2 = np.repeat(antnum[:, 1].astype(np.int32), nH)
    time_index = np.tile(np.arange(nH, dtype=np.int32), nbas)
    u_flat = np.asarray(u_waves, dtype=np.float32).reshape(nrows)
    v_flat = np.asarray(v_waves, dtype=np.float32).reshape(nrows)
    w_flat = np.asarray(w_waves, dtype=np.float32).reshape(nrows)
    gains_flat = np.asarray(gains, dtype=np.complex64).reshape(nrows)
    noise_flat = np.asarray(noise_samples, dtype=np.complex64).reshape(nrows)

    pix_u = np.rint(u_flat / uv_pixsize).astype(np.int32)
    pix_v = np.rint(v_flat / uv_pixsize).astype(np.int32)
    valid = (np.abs(pix_u) < Nphf) & (np.abs(pix_v) < Nphf)

    model_data = np.zeros(nrows, dtype=np.complex64)
    data = np.zeros(nrows, dtype=np.complex64)
    if np.any(valid):
        p_v = pix_v[valid] + Nphf
        m_u = -pix_u[valid] + Nphf
        # The gridding path samples model visibilities at (v, -u).
        model_vis_grid = np.fft.fftshift(modelfft)
        sampled = model_vis_grid[p_v, m_u].astype(np.complex64)
        model_data[valid] = sampled
        data[valid] = sampled * gains_flat[valid] + noise_flat[valid] * np.abs(gains_flat[valid])

    # Each row's noise as calibrated data's weights carry it: the channel's noise
    # times the gain amplitude the data were scaled by (1 without corruptions)
    # and the baseline's own noise scale.
    sigma = (max(float(noise_std), 1e-12) * np.abs(gains_flat)).astype(np.float32)
    if noise_scale is not None:
        sigma *= np.repeat(np.asarray(noise_scale, dtype=np.float32), nH)
    weight = (1.0 / np.square(np.maximum(sigma, 1e-12))).astype(np.float32)
    uvw_m = np.stack(
        [
            u_flat * mean_wavelength_m,
            v_flat * mean_wavelength_m,
            w_flat * mean_wavelength_m,
        ],
        axis=1,
    ).astype(np.float64)

    return {
        "uvw_m": uvw_m,
        "antenna1": antenna1,
        "antenna2": antenna2,
        "time_index": time_index,
        "valid": valid.astype(np.bool_),
        "model_data": model_data,
        "data": data,
        "weight": weight,
        "sigma": sigma,
    }


def assemble_visibility_table(
    *,
    channel_rows: list[dict[str, np.ndarray]],
    channel_freq_hz: np.ndarray,
    scan_time_s: float,
    observation_date: str,
    antenna_names: list[str],
    antenna_positions_m: np.ndarray,
    source_name: str,
    field_ra_rad: float,
    field_dec_rad: float,
    config_name: str | None = None,
    array_type: str | None = None,
    antenna_diameters_m: np.ndarray | None = None,
    scan_length_s: float = 0.0,
    observation_id: int | None = None,
    channel_kernel: np.ndarray | None = None,
) -> VisibilityTable:
    """Assemble channel-wise row samples into an MS-like visibility table.

    ``scan_length_s`` > 0 numbers the rows' scans, blocks of that length;
    ``observation_id`` marks every row as that execution block's;
    ``channel_kernel`` is the kernel the channels' noise was correlated with.
    """
    if not channel_rows:
        raise ValueError("channel_rows must not be empty")

    reference = channel_rows[0]
    nrows = int(reference["uvw_m"].shape[0])
    nchan = int(len(channel_rows))
    ncorr = 1
    data = np.zeros((nrows, ncorr, nchan), dtype=np.complex64)
    model_data = np.zeros_like(data)
    flag = np.ones((nrows, ncorr, nchan), dtype=np.bool_)
    weight_spectrum = np.zeros((nrows, ncorr, nchan), dtype=np.float32)
    sigma_spectrum = np.zeros((nrows, ncorr, nchan), dtype=np.float32)

    for chan_idx, rows in enumerate(channel_rows):
        data[:, 0, chan_idx] = rows["data"]
        model_data[:, 0, chan_idx] = rows["model_data"]
        flag[:, 0, chan_idx] = ~rows["valid"]
        weight_spectrum[:, 0, chan_idx] = rows["weight"]
        sigma_spectrum[:, 0, chan_idx] = rows["sigma"]

    start = Time(f"{observation_date}T00:00:00", format="isot", scale="utc")
    time_offsets_s = reference["time_index"].astype(np.float64) * float(scan_time_s)
    times_mjd_s = start.mjd * 86400.0 + time_offsets_s
    interval_s = np.full(nrows, float(scan_time_s), dtype=np.float64)
    exposure_s = np.full(nrows, float(scan_time_s), dtype=np.float64)

    # Collapse per-channel noise metadata to row-level MS WEIGHT/SIGMA as the
    # ALMA pipeline's statwt does: the inverse of the row's mean variance over its
    # channels (a mean of the channels' weights would overweight the quiet ones).
    variance = np.mean(np.square(sigma_spectrum.astype(np.float64)), axis=2)
    sigma = np.sqrt(variance).astype(np.float32)
    weight = (1.0 / np.maximum(variance, 1e-24)).astype(np.float32)

    scan_number = None
    if scan_length_s > 0:
        scan_number = (time_offsets_s // float(scan_length_s)).astype(np.int32) + 1
    return VisibilityTable(
        sigma_spectrum=sigma_spectrum,
        channel_kernel=None if channel_kernel is None else np.asarray(channel_kernel, np.float64),
        scan_number=scan_number,
        observation_id=(
            None if observation_id is None else np.full(nrows, int(observation_id), np.int32)
        ),
        uvw_m=np.asarray(reference["uvw_m"], dtype=np.float64),
        antenna1=np.asarray(reference["antenna1"], dtype=np.int32),
        antenna2=np.asarray(reference["antenna2"], dtype=np.int32),
        time_mjd_s=times_mjd_s,
        interval_s=interval_s,
        exposure_s=exposure_s,
        data=data,
        model_data=model_data,
        flag=flag,
        weight=weight,
        sigma=sigma,
        channel_freq_hz=np.asarray(channel_freq_hz, dtype=np.float64),
        antenna_names=list(antenna_names),
        antenna_positions_m=np.asarray(antenna_positions_m, dtype=np.float64),
        source_name=str(source_name),
        field_ra_rad=float(field_ra_rad),
        field_dec_rad=float(field_dec_rad),
        observation_date=str(observation_date),
        config_name=config_name,
        array_type=array_type,
        antenna_diameters_m=(
            None
            if antenna_diameters_m is None
            else np.asarray(antenna_diameters_m, dtype=np.float64)
        ),
    )


# Per-row columns of a visibility table; the rest describe the whole table.
_ROW_COLUMNS = (
    "uvw_m",
    "antenna1",
    "antenna2",
    "time_mjd_s",
    "interval_s",
    "exposure_s",
    "data",
    "model_data",
    "flag",
    "weight",
    "sigma",
)


def _diameters(table: Mapping[str, Any]) -> np.ndarray:
    """A table's per-antenna dish diameters (m), 12 m each when it has none."""
    diameters = table.get("antenna_diameters_m")
    if diameters is None:
        return np.full(len(table["antenna_names"]), 12.0)
    return np.asarray(diameters, dtype=np.float64)


def concatenate_visibility_tables(tables: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Concatenate multiple visibility tables across rows."""
    if not tables:
        raise ValueError("tables must not be empty")
    if len(tables) == 1:
        return dict(tables[0])

    first = tables[0]
    # One ANTENNA list, each antenna once in the order the tables name them,
    # every table's ANTENNA1/ANTENNA2 mapped onto it: a 7 m configuration's rows
    # used to point at the 12 m antennas of the first table, and the execution
    # blocks of one array share its antennas (each block a subset of them).
    antennas = _joint_antennas(tables)
    return {
        **{
            key: np.concatenate(
                antennas["rows"][key]
                if key in antennas["rows"]
                else [np.asarray(t[key]) for t in tables],
                axis=0,
            )
            for key in _ROW_COLUMNS
        },
        "channel_freq_hz": np.asarray(first["channel_freq_hz"]),
        "antenna_names": antennas["names"],
        "antenna_positions_m": np.asarray(antennas["positions"], dtype=np.float64).reshape(-1, 3),
        "antenna_diameters_m": np.asarray(antennas["diameters"], dtype=np.float64),
        "source_name": first["source_name"],
        "field_ra_rad": float(first["field_ra_rad"]),
        "field_dec_rad": float(first["field_dec_rad"]),
        "observation_date": first["observation_date"],
        "config_name": first.get("config_name"),
        "array_type": first.get("array_type"),
        **_scan_numbers(tables),
        **_observation_ids(tables),
        **_sigma_spectra(tables),
        **_channel_kernel(tables),
    }


def _joint_antennas(tables: list[Mapping[str, Any]]) -> dict[str, Any]:
    """The tables' antennas, each name once (first position and diameter), and each
    table's ANTENNA1/ANTENNA2 as indices into them."""
    index: dict[str, int] = {}
    names: list[str] = []
    positions: list[np.ndarray] = []
    diameters: list[float] = []
    rows: dict[str, list[np.ndarray]] = {"antenna1": [], "antenna2": []}
    for table in tables:
        table_positions = np.asarray(table["antenna_positions_m"]).reshape(-1, 3)
        local = []
        for name, position, diameter in zip(
            table["antenna_names"], table_positions, _diameters(table), strict=True
        ):
            if name not in index:
                index[name] = len(names)
                names.append(name)
                positions.append(position)
                diameters.append(float(diameter))
            local.append(index[name])
        mapping = np.asarray(local, dtype=np.int32)
        for key in rows:
            rows[key].append(mapping[np.asarray(table[key], dtype=np.int64)])
    return {"names": names, "positions": positions, "diameters": diameters, "rows": rows}


def _scan_numbers(tables: list[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """The concatenated scan numbers, if any table has them (one scan for those
    without), each table's after the previous tables': scans do not repeat."""
    if all(t.get("scan_number") is None for t in tables):
        return {}
    columns, last = [], 0
    for t in tables:
        scans = (
            np.asarray(t["scan_number"])
            if t.get("scan_number") is not None
            else np.ones(len(t["antenna1"]), np.int32)
        )
        columns.append(scans + last)
        last += int(scans.max(initial=0))
    return {"scan_number": np.concatenate(columns).astype(np.int32)}


def _sigma_spectra(tables: list[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """The concatenated per-visibility noise, when every table has it."""
    if any(t.get("sigma_spectrum") is None for t in tables):
        return {}
    return {"sigma_spectrum": np.concatenate([np.asarray(t["sigma_spectrum"]) for t in tables])}


def _channel_kernel(tables: list[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """The tables' channel kernel, when they all share one (one window's channels)."""
    kernels = [t.get("channel_kernel") for t in tables]
    if any(k is None for k in kernels):
        return {}
    first = np.asarray(kernels[0], dtype=np.float64)
    if not all(np.allclose(np.asarray(k, dtype=np.float64), first) for k in kernels[1:]):
        return {}
    return {"channel_kernel": first}


def _observation_ids(tables: list[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """The concatenated execution blocks, if any table has them (0 for those without)."""
    if all(t.get("observation_id") is None for t in tables):
        return {}
    return {
        "observation_id": np.concatenate(
            [
                np.asarray(t["observation_id"])
                if t.get("observation_id") is not None
                else np.zeros(len(t["antenna1"]), np.int32)
                for t in tables
            ]
        ).astype(np.int32)
    }
