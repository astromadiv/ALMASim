"""Simulation parameters and their coercion (MADIV-28 T5.7)."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

LogFn = Callable[[str], None] | None
StopFn = bool | Callable[[], bool]


_FORBIDDEN_ROOTS = (Path("/proc"), Path("/sys"), Path("/dev"))


def _expand_path(path_like: Path | str) -> str:
    """An absolute path with ``~`` expanded, refused under /proc, /sys or /dev."""
    raw = str(path_like)
    if "\x00" in raw:
        raise ValueError(f"Path contains null bytes: {path_like!r}")
    resolved = Path(raw).expanduser().resolve()
    for forbidden in _FORBIDDEN_ROOTS:
        if resolved.is_relative_to(forbidden):
            raise ValueError(
                f"Path '{path_like}' resolves to a restricted system location: {resolved}"
            )
    return str(resolved)


def _optional(convert: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return lambda value: None if value is None else convert(value)


def _cpu_count(value: Any) -> int:
    return int(value if value is not None else (os.cpu_count() or 1))


def _name_or_none(value: Any) -> str | None:
    return str(value) if value else None


def _coerced(convert: Callable[[Any], Any], **kwargs: Any) -> Any:
    """A field that ``SimulationParams.__post_init__`` passes through ``convert``."""
    return field(metadata={"coerce": convert}, **kwargs)


def _corruption_mapping(value: Any) -> dict[str, Any] | None:
    """A corruption model's fields from a preset, JSON, JSON file or mapping,
    checked; None for none (also NaN, a table's empty cell)."""
    from ..interferometry.corruptions import resolve_corruptions

    if isinstance(value, float) and value != value:
        return None
    return resolve_corruptions(value)


@dataclass
class SimulationParams:
    """Everything one simulation run needs.

    Values are coerced on construction (``__post_init__``, driven by each
    field's ``coerce`` metadata): numbers become ``float``/``int``, paths
    become absolute (and are refused under /proc, /sys and /dev), ``ncpu``
    defaults to the machine's CPU count.
    """

    idx: int = _coerced(int)
    source_name: str = _coerced(str)
    member_ouid: str = _coerced(str)
    main_dir: str = _coerced(_expand_path)
    output_dir: str = _coerced(_expand_path)
    tng_dir: str = _coerced(_expand_path)
    galaxy_zoo_dir: str = _coerced(_expand_path)
    hubble_dir: str = _coerced(_expand_path)
    project_name: str = _coerced(str)
    ra: float = _coerced(float)
    dec: float = _coerced(float)
    band: float = _coerced(float)
    ang_res: float = _coerced(float)
    vel_res: float = _coerced(float)
    fov: float = _coerced(float)
    obs_date: str = _coerced(str)
    pwv: float = _coerced(float)
    int_time: float = _coerced(float)
    bandwidth: float = _coerced(float)
    freq: float = _coerced(float)
    freq_support: str = _coerced(str)
    cont_sens: float = _coerced(float)
    antenna_array: str = _coerced(str)
    n_pix: float | None = None
    n_channels: int | None = None
    source_type: str = _coerced(str, default="point")
    tng_api_key: str | None = None
    ncpu: int | None = _coerced(_cpu_count, default=None)
    rest_frequency: Any = None
    redshift: Any = None
    lum_infrared: Any = None
    snr: float | None = None
    n_lines: Any = None
    line_names: Any = None
    save_mode: str = _coerced(str, default="fits")
    persist: bool = _coerced(bool, default=True)
    ml_dataset_path: str | None = _coerced(_optional(_expand_path), default=None)
    inject_serendipitous: bool = _coerced(bool, default=False)
    output_subdir_name: str | None = _coerced(_name_or_none, default=None)
    remote: bool = _coerced(bool, default=False)
    observation_configs: Any | None = None
    ground_temperature_k: float = _coerced(float, default=270.0)
    correlator: str | None = None
    elevation_deg: float | None = None
    line_sens_10kms: float | None = _coerced(_optional(float), default=None)
    source_offset_x_arcsec: float = _coerced(float, default=0.0)
    source_offset_y_arcsec: float = _coerced(float, default=0.0)
    background_mode: str = _coerced(str, default="none")
    background_level: float = _coerced(float, default=1.0)
    background_seed: int | None = _coerced(_optional(int), default=None)
    # Seeds every random draw of the run (sky, noise, background unless
    # background_seed is set; MADIV-28 T5.1); None draws fresh entropy.
    seed: int | None = _coerced(_optional(int), default=None)
    external_skymodel_path: str | None = _coerced(_optional(_expand_path), default=None)
    external_component_table_path: str | None = _coerced(_optional(_expand_path), default=None)
    external_alignment_mode: str = _coerced(str, default="observation")
    external_header_mode: str = _coerced(str, default="observation")
    external_header_overrides: dict[str, Any] | None = None
    ms_export: bool = _coerced(bool, default=False)
    ms_export_dir: str | None = _coerced(_optional(_expand_path), default=None)
    ms_save_mode: str = _coerced(str, default="msv2")
    imaging_algorithm: str = _coerced(str, default="legacy")
    # "legacy": the metadata FOV, a fifth of the beam per cell. "planner": the
    # imaging planner's geometry (services.instrument.geometry: the primary
    # beam's FOV, 8 cells per beam); MADIV-28 T2.11, until the golden
    # simulation outputs are re-baselined on it.
    instrument_geometry: str = _coerced(str, default="legacy")
    kaggle_token: str | None = _coerced(_optional(_expand_path), default=None)
    # What the visibilities carry besides white thermal noise: the fields of
    # interferometry.corruptions.CorruptionModel (a mapping, its JSON, a JSON
    # file or a preset name such as "alma"); None adds nothing (MADIV-28).
    corruptions: dict[str, Any] | None = _coerced(_corruption_mapping, default=None)
    # The MOUS's execution blocks: how many observe it (the archive's ASDM
    # count; None: Int.Time split into blocks of at most
    # planning.MAX_EXECUTION_TIME_S on source) and how many of them to simulate
    # (None: all of them).
    n_executions: int | None = _coerced(_optional(int), default=None)
    executions: int | None = _coerced(_optional(int), default=None)

    def __post_init__(self) -> None:
        for spec in fields(self):
            convert = spec.metadata.get("coerce")
            if convert is not None:
                setattr(self, spec.name, convert(getattr(self, spec.name)))

    @classmethod
    def from_values(cls, **values: Any) -> SimulationParams:
        """Build :class:`SimulationParams` from explicit values (same as the constructor).

        ``n_pix`` and ``n_channels`` may be ``None`` to allow downstream simulation
        stages to derive suitable defaults from the observing configuration.
        """
        return cls(**values)

    # Metadata columns (archive name first, then the snake_case alias) of the
    # values a metadata row provides.
    _ROW_COLUMNS: ClassVar[dict[str, tuple[str, ...]]] = {
        "source_name": ("ALMA_source_name", "source_name", "target_name"),
        "member_ouid": ("member_ous_uid", "member_ouid"),
        "band": ("Band", "band"),
        "ra": ("RA",),
        "dec": ("Dec",),
        "ang_res": ("Ang.res.", "ang_res"),
        "vel_res": ("Vel.res.", "vel_res"),
        "fov": ("FOV", "fov"),
        "obs_date": ("Obs.date", "obs_date"),
        "pwv": ("PWV", "pwv"),
        "int_time": ("Int.Time", "int_time"),
        "bandwidth": ("Bandwidth", "bandwidth"),
        "freq": ("Freq", "frequency"),
        "freq_support": ("Freq.sup.", "frequency_support"),
        "cont_sens": ("Cont_sens_mJybeam", "cont_sens"),
        "antenna_array": ("antenna_arrays", "antenna_array"),
    }
    # Values a row may provide; an explicit keyword argument wins.
    _OPTIONAL_ROW_COLUMNS: ClassVar[dict[str, tuple[str, ...]]] = {
        "line_sens_10kms": ("Line_sens_10kms_mJybeam", "line_sens_10kms"),
        "rest_frequency": ("rest_frequency",),
        "redshift": ("redshift",),
        "lum_infrared": ("lum_infrared",),
        "n_lines": ("n_lines",),
        "line_names": ("line_names",),
        "n_executions": ("n_executions",),
    }

    @classmethod
    def from_metadata_row(
        cls, row: pd.Series | Mapping[str, Any], **values: Any
    ) -> SimulationParams:
        """Build :class:`SimulationParams` from a metadata row plus explicit values.

        The row supplies the observation (``_ROW_COLUMNS``, all required) and
        optionally the line and source values (``_OPTIONAL_ROW_COLUMNS``); the
        keyword arguments supply the rest and override the optional ones.
        """
        payload = dict(row.to_dict()) if hasattr(row, "to_dict") else dict(row)

        def _get(keys: tuple[str, ...]) -> Any:
            for key in keys:
                value = payload.get(key)
                if value is None:
                    continue
                try:
                    if pd.isna(value):
                        continue
                except (TypeError, ValueError):
                    pass
                return value
            return None

        row_values: dict[str, Any] = {}
        for name, keys in cls._ROW_COLUMNS.items():
            value = _get(keys)
            if value is None:
                raise KeyError(f"Missing required metadata column(s): {list(keys)}")
            row_values[name] = value
        for name, keys in cls._OPTIONAL_ROW_COLUMNS.items():
            if values.get(name) is None:
                row_values[name] = _get(keys)
        return cls(**{**values, **row_values})


def _clean(value: Any) -> Any:
    """``None`` for a missing value (None or NaN), the value otherwise."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
