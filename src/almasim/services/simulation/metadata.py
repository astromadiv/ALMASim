"""The run's parameter record, metadata and effective SNR (MADIV-28 T5.7)."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import astropy.units as U
import numpy as np

from ..astro.parameters import write_sim_parameters
from ..json_safe import json_safe as _json_safe
from .geometry import CubeGeometry
from .params import SimulationParams
from .skymodel_stage import SkyCube
from .spectral_setup import SpectralSetup


def resolve_snr(params: SimulationParams, log: Callable[[str], None]) -> float:
    """``params.snr``, or one derived from the metadata sensitivities."""
    effective_snr = params.snr
    if effective_snr is None:
        if (
            params.line_sens_10kms is not None
            and params.cont_sens is not None
            and params.cont_sens > 0.0
        ):
            effective_snr = float(np.clip(params.line_sens_10kms / params.cont_sens, 1.0, 100.0))
            log(
                "Auto-derived SNR from metadata sensitivities "
                f"(line/continuum): {effective_snr:.3f}"
            )
        elif params.cont_sens is not None and params.cont_sens > 0.0:
            effective_snr = 5.0
            log("Auto-derived SNR fallback from metadata continuum sensitivity: 5.000")
        else:
            effective_snr = 1.3
            log("Falling back to default SNR=1.3 because auto-derivation was not possible")
    else:
        effective_snr = float(effective_snr)
    return effective_snr


@dataclass
class SimParamsRecord:
    """The values ``write_sim_parameters`` writes to ``sim_params_<idx>.txt`` (MADIV-28 T5.8)."""

    path: str
    source_name: str
    member_ouid: str
    ra: Any
    dec: Any
    ang_res: Any
    vel_res: Any
    int_time: Any
    band: Any
    band_range: Any
    central_freq: Any
    redshift: Any
    line_fluxes: Any
    line_names: Any
    line_frequencies: Any
    continuum: Any
    fov: Any
    beam_size: Any
    cell_size: Any
    n_pix: int
    n_channels: int
    snapshot: int | None
    subhalo: Any
    lum_infrared: Any
    fwhm_z: Any
    source_type: str
    fwhm_x: Any = None
    fwhm_y: Any = None
    angle: Any = None
    noise_model: str | None = None

    def write(self) -> str:
        """Write the parameter file; its path."""
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values["continum"] = values.pop("continuum")
        write_sim_parameters(**values)
        return self.path


def sim_params_record(
    params: SimulationParams,
    geometry: CubeGeometry,
    spectral: SpectralSetup,
    sky: SkyCube,
    output_dir_abs: str,
) -> SimParamsRecord:
    """The run's parameter record."""
    return SimParamsRecord(
        path=os.path.join(output_dir_abs, f"sim_params_{params.idx}.txt"),
        source_name=params.source_name,
        member_ouid=params.member_ouid,
        ra=params.ra * U.deg,
        dec=params.dec * U.deg,
        ang_res=params.ang_res * U.arcsec,
        vel_res=params.vel_res * U.km / U.s,
        int_time=params.int_time * U.s,
        band=params.band,
        band_range=geometry.band_range,
        central_freq=geometry.central_freq,
        redshift=spectral.redshift,
        line_fluxes=spectral.line_fluxes,
        line_names=spectral.line_names,
        line_frequencies=spectral.line_frequency,
        continuum=spectral.continuum,
        fov=geometry.fov,
        beam_size=geometry.beam_size,
        cell_size=geometry.cell_size,
        n_pix=geometry.n_pix,
        n_channels=geometry.n_channels,
        snapshot=sky.snapshot,
        subhalo=sky.tng_subhaloid,
        lum_infrared=spectral.lum_infrared,
        fwhm_z=spectral.fwhm_z,
        source_type=params.source_type,
        fwhm_x=sky.fwhm_x,
        fwhm_y=sky.fwhm_y,
        angle=sky.angle,
        noise_model=_noise_model_line(params),
    )


def simulation_metadata(
    params: SimulationParams,
    geometry: CubeGeometry,
    spectral: SpectralSetup,
    sky: SkyCube,
    *,
    persist_outputs: bool,
) -> dict[str, Any]:
    """The JSON-safe description of the run stored with its products."""
    return {
        "idx": params.idx,
        "project_name": params.project_name,
        "source_name": params.source_name,
        "member_ouid": params.member_ouid,
        "source_type": params.source_type,
        "save_mode": params.save_mode,
        "persist": persist_outputs,
        "ra_deg": params.ra,
        "dec_deg": params.dec,
        "band": params.band,
        "ang_res_arcsec": params.ang_res,
        "vel_res_kms": params.vel_res,
        "fov_arcsec": geometry.fov.value,
        "obs_date": params.obs_date,
        "pwv": params.pwv,
        "int_time_s": params.int_time,
        "bandwidth_ghz": params.bandwidth,
        "source_freq_ghz": params.freq,
        "freq_support": params.freq_support,
        "cont_sens": params.cont_sens,
        "antenna_array": params.antenna_array,
        "n_pix": geometry.n_pix,
        "n_channels": geometry.n_channels,
        "central_freq_ghz": geometry.central_freq.to(U.GHz).value,
        "band_range_ghz": geometry.band_range.to(U.GHz).value,
        "channel_width_hz": geometry.delta_freq.to(U.Hz).value,
        "channel_averaging": geometry.channel_averaging,
        "beam_size_arcsec": geometry.beam_size.to(U.arcsec).value,
        "cell_size_arcsec": geometry.cell_size.to(U.arcsec).value,
        "redshift": spectral.redshift,
        "rest_frequency_ghz": spectral.rest_frequency.to(U.GHz).value,
        "line_names": _json_safe(spectral.line_names),
        "line_fluxes": _json_safe(spectral.line_fluxes),
        "line_frequency_hz": _json_safe(spectral.line_frequency),
        "continuum": _json_safe(spectral.continuum),
        "observation_plan": geometry.observation_plan.as_dict(),
        "source_offset_x_arcsec": params.source_offset_x_arcsec,
        "source_offset_y_arcsec": params.source_offset_y_arcsec,
        "source_pixel_x": float(sky.pos_x),
        "source_pixel_y": float(sky.pos_y),
        "background_mode": params.background_mode,
        "background_level": params.background_level,
        "background_seed": params.background_seed,
        "background_total_flux_jy": float(np.sum(sky.background_cube)),
        "external_input": _json_safe(sky.external_input_metadata),
        "imaging_algorithm": params.imaging_algorithm,
        # Recorded only when the run used one, so runs without keep their metadata.
        **({"noise_model": params.noise_model.identity()} if params.noise_model else {}),
    }


def _noise_model_line(params: SimulationParams) -> str | None:
    """The noise model's label and SHA-256 for the parameter file; None without one."""
    model = params.noise_model
    return None if model is None else f"{model.label()} (sha256 {model.sha256()})"
