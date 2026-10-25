"""The model cube a run simulates: frequencies, beam, cell and size (MADIV-28 T5.7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import astropy.units as U
import numpy as np

from ..external_skymodel import infer_external_cube_geometry, is_external_source_type
from ..instrument.channels import Channels, channels_from_resolution
from ..instrument.geometry import (
    cell_arcsec,
    image_size,
    primary_beam_fov_arcsec,
    synthesized_beam_arcsec,
)
from ..instrument.spectral_window import SpectralWindow
from ..interferometry import antenna as ual_antenna
from ..interferometry.frequency import freq_supp_extractor
from ..interferometry.utils import closest_power_of_2
from ..observation_plan import build_single_pointing_observation_plan
from .params import SimulationParams, _clean

# Cells per synthesised beam in the planner geometry (the imaging planner's default).
PLANNER_BEAM_SAMPLING = 8


@dataclass(frozen=True)
class CubeGeometry:
    """The model cube a run simulates, and the observation values that set it.

    One resolver (:func:`resolve_cube_geometry`) serves the footprint estimate
    and the run, so ``/estimate`` reports the cube a run would make.
    """

    central_freq: U.Quantity
    band_range: U.Quantity
    delta_freq: U.Quantity
    observation_plan: Any
    max_baseline: U.Quantity
    fov: U.Quantity
    beam_size: U.Quantity
    cell_size: U.Quantity
    n_pix: int
    n_channels: int
    # The window's channels as ALMA makes them (instrument.channels): the native
    # channels each averages and the radiometer's bandwidth. None: as given.
    channels: Channels | None = None

    @property
    def channel_width_hz(self) -> float:
        return self.delta_freq.to(U.Hz).value

    @property
    def channel_averaging(self) -> int:
        return self.channels.averaged if self.channels is not None else 1

    @property
    def noise_bandwidth_hz(self) -> float:
        """The bandwidth of each channel's radiometer noise."""
        if self.channels is not None and self.channels.noise_bandwidth_hz is not None:
            return self.channels.noise_bandwidth_hz
        return self.channel_width_hz

    def channel_frequencies_hz(self) -> np.ndarray:
        """The centre frequency of every channel, symmetric about ``central_freq``."""
        return self.spectral_window().frequencies_hz()

    def spectral_window(self) -> SpectralWindow:
        """The cube's channels: the one grid the uv sampling and the MS use too (T5.4)."""
        return SpectralWindow.centred(
            self.central_freq.to(U.Hz).value, self.channel_width_hz, self.n_channels
        )


def resolve_cube_geometry(params: SimulationParams, *, strict: bool = True) -> CubeGeometry:
    """Frequencies, beam, cell and cube size for a run.

    The cell is a fifth of the synthesised beam unless ``n_pix`` is given, in
    which case the field of view sets it; the channel count is the spectral
    window's unless ``n_channels`` is given. An external model in "preserve"
    header mode keeps its own size. With ``strict=False`` (the estimate) an
    unreadable external model falls back to the computed size.
    """
    source_freq = params.freq * U.GHz
    band_range, central_freq, t_channels, delta_freq = freq_supp_extractor(
        params.freq_support, source_freq
    )
    # The archive gives the window's resolution; its channels are narrower.
    channels = channels_from_resolution(delta_freq.to(U.Hz).value, band_range.to(U.Hz).value)
    delta_freq = (channels.width_hz * U.Hz).to(U.GHz)
    t_channels = int((band_range / delta_freq).decompose().value + 1e-6)
    observation_plan = build_single_pointing_observation_plan(params)
    max_baseline_km = max(
        ual_antenna.get_max_baseline_from_antenna_array(config.antenna_array, params.main_dir)
        for config in observation_plan.configs
    )
    max_baseline = max_baseline_km * U.km
    planner = str(params.instrument_geometry).lower() == "planner"
    if planner:
        # The imaging planner's geometry (MADIV-28 T2.11): the primary beam of
        # the smallest dish as the FOV, PLANNER_BEAM_SAMPLING cells per beam.
        frequency_hz = central_freq.to(U.Hz).value
        dish = min(config.antenna_diameter_m for config in observation_plan.configs)
        fov = primary_beam_fov_arcsec(dish, frequency_hz) * U.arcsec
        beam_size = synthesized_beam_arcsec(frequency_hz, max_baseline.to(U.m).value) * U.arcsec
        cell_size = cell_arcsec(beam_size, PLANNER_BEAM_SAMPLING)
    else:
        fov = params.fov * 3600 * U.arcsec
        beam_size = ual_antenna.estimate_alma_beam_size(
            central_freq, max_baseline, return_value=False
        )
        cell_size = beam_size / 5

    external_geometry = None
    if is_external_source_type(params.source_type) and params.external_header_mode == "preserve":
        try:
            external_geometry = infer_external_cube_geometry(
                source_type=params.source_type,
                skymodel_path=params.external_skymodel_path,
                component_table_path=params.external_component_table_path,
            )
        except Exception:
            if strict:
                raise

    n_pix = _clean(params.n_pix)
    if n_pix is None and external_geometry is not None:
        n_pix = int(external_geometry["n_pix"])
    elif n_pix is None:
        n_pix = image_size(1.5 * fov.value, cell_size.value, rounding="pow2")
    else:
        n_pix = closest_power_of_2(int(n_pix))
        cell_size = fov / n_pix

    n_channels = _clean(params.n_channels)
    if n_channels is None and external_geometry is not None:
        n_channels = int(external_geometry["n_channels"])
    elif n_channels is None:
        n_channels = t_channels
    else:
        band_range = n_channels * delta_freq

    return CubeGeometry(
        central_freq=central_freq,
        band_range=band_range,
        delta_freq=delta_freq,
        observation_plan=observation_plan,
        max_baseline=max_baseline,
        fov=fov,
        beam_size=beam_size,
        cell_size=cell_size,
        n_pix=n_pix,
        n_channels=n_channels,
        channels=channels,
    )


def estimate_simulation_footprint(params: SimulationParams) -> dict[str, Any]:
    """Estimate cube dimensions and raw storage footprint before running a simulation."""
    geometry = resolve_cube_geometry(params, strict=False)
    n_pix, n_channels = int(geometry.n_pix), int(geometry.n_channels)
    cell_size, beam_size = geometry.cell_size, geometry.beam_size
    voxels = int(n_channels) * int(n_pix) * int(n_pix)
    float32_gb = (voxels * np.dtype(np.float32).itemsize) / float(1024**3)
    complex64_gb = (voxels * np.dtype(np.complex64).itemsize) / float(1024**3)
    # Approximate raw footprint for the standard persisted image/cube products.
    estimated_standard_output_gb = 6.0 * float32_gb + complex64_gb

    return {
        "n_pix": int(n_pix),
        "n_channels": int(n_channels),
        "cube_shape": [int(n_channels), int(n_pix), int(n_pix)],
        "cube_voxels": voxels,
        "cell_size_arcsec": float(cell_size.to(U.arcsec).value),
        "beam_size_arcsec": float(beam_size.to(U.arcsec).value),
        "raw_single_cube_gb": float(float32_gb),
        "raw_complex_cube_gb": float(complex64_gb),
        "estimated_standard_output_gb": float(estimated_standard_output_gb),
        "note": (
            "Raw uncompressed estimate. Actual NPZ/HDF5 output can be smaller due to compression."
        ),
    }


def resolve_source_pixel_position(
    *,
    wcs: Any,
    ra: U.Quantity,
    dec: U.Quantity,
    central_freq: U.Quantity,
    n_pix: int,
    cell_size_arcsec: float,
    offset_x_arcsec: float = 0.0,
    offset_y_arcsec: float = 0.0,
) -> tuple[float, float]:
    """Resolve the source pixel position from phase center plus explicit offsets."""
    pos_x, pos_y, _ = wcs.sub(3).wcs_world2pix(ra, dec, central_freq, 0)
    pos_x += float(offset_x_arcsec) / max(float(cell_size_arcsec), 1e-12)
    pos_y += float(offset_y_arcsec) / max(float(cell_size_arcsec), 1e-12)
    pos_x = float(np.clip(pos_x, 0, n_pix - 1))
    pos_y = float(np.clip(pos_y, 0, n_pix - 1))
    return pos_x, pos_y
