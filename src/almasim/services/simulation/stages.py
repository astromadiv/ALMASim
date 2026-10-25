"""The simulation stages: clean cube, observation and image products (MADIV-28 T5.7)."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import astropy.units as U
import numpy as np

from ... import skymodels as usm
from .. import interferometry as uin
from ..external_skymodel import is_external_source_type
from ..imaging import build_image_products
from ..instrument.constants import GAUSSIAN_BEAM_AREA
from ..interferometry.corruptions import CorruptionModel
from ..utils import as_progress_emitter
from .context import RunContext
from .geometry import CubeGeometry, resolve_cube_geometry
from .metadata import SimParamsRecord, resolve_snr, sim_params_record, simulation_metadata
from .params import LogFn, SimulationParams, StopFn
from .planning import (
    InterferometerRun,
    TotalPowerRun,
    noise_reference,
    plan_observation_runs,
)
from .rng import run_rng, stream_entropy
from .skymodel_stage import build_sky_cube
from .spectral_setup import resolve_spectral_setup


@dataclass
class CleanCubeStage:
    """Intermediate clean-cube stage output."""

    datacube: Any
    model_cube: np.ndarray
    header: Any
    output_dir_abs: str
    sim_output_dir: str | None
    sim_params_payload: SimParamsRecord
    interferometer_kwargs: dict[str, Any]
    interferometer_runs: list[InterferometerRun]
    total_power_runs: list[TotalPowerRun]
    metadata: dict[str, Any]
    observation_plan: dict[str, Any]
    channel_frequencies_hz: np.ndarray
    channel_width_hz: float
    cell_size_arcsec: float
    background_cube: np.ndarray | None
    external_input_metadata: dict[str, Any] | None
    seed: int | None = None
    corruptions: CorruptionModel | None = None
    # Each channel's continuum share of the source (SpectralSetup.continuum_fraction):
    # the per-baseline errors act on it only. None: on all the emission.
    continuum_fraction: np.ndarray | None = None


def generate_clean_cube(
    params: SimulationParams,
    *,
    logger: LogFn = None,
    status_callback: Callable[[str], None] | None = None,
    progress_emitter=None,
    compute_backend=None,
    terminal_logger=None,
    stop_requested: StopFn = False,
) -> CleanCubeStage:
    """Generate the clean sky cube and derived observation metadata."""
    ctx = RunContext(
        logger,
        status_callback,
        stop_requested,
        remote=bool(params.remote),
        rng=run_rng(params.seed, "sky"),
    )
    log = ctx.log
    progress_adapter = as_progress_emitter(progress_emitter)

    ctx.check_cancelled()
    ctx.status("Computing Max baseline")
    geometry = resolve_cube_geometry(params)
    central_freq, band_range, delta_freq = (
        geometry.central_freq,
        geometry.band_range,
        geometry.delta_freq,
    )
    fov, beam_size = geometry.fov, geometry.beam_size
    n_pix, n_channels = geometry.n_pix, geometry.n_channels
    output_dir_abs = os.path.abspath(params.output_dir)
    persist_outputs = bool(params.persist and params.save_mode != "memory")
    sim_output_dir = (
        (
            os.path.join(output_dir_abs, params.output_subdir_name)
            if params.output_subdir_name
            else output_dir_abs
        )
        if persist_outputs
        else None
    )
    if sim_output_dir is not None:
        os.makedirs(sim_output_dir, exist_ok=True)

    log(f"RA: {params.ra * U.deg}")
    log(f"DEC: {params.dec * U.deg}")
    log(f"Integration Time: {params.int_time * U.s}")

    if progress_adapter is not None:
        progress_adapter.emit(100)
    log(f"Field of view: {round(fov.value, 3)} arcsec")

    beam_area = GAUSSIAN_BEAM_AREA * beam_size**2
    beam_solid_angle = np.pi * (beam_size / 2) ** 2
    cont_sens = params.cont_sens * U.mJy / (U.arcsec**2)
    cont_sens_jy = (cont_sens * beam_solid_angle).to(U.Jy)
    log(f"Minimum detectable continum: {cont_sens_jy}")

    spectral = resolve_spectral_setup(params, geometry, cont_sens_jy, ctx)

    log(f"Central Frequency: {central_freq}")
    log(f"Beam size: {round(beam_size.value, 4)} arcsec")
    log(f"Spectral Window: {band_range}")
    log(f"Freq Support: {delta_freq}")
    log(f"Cube Dimensions: {n_pix} x {n_pix} x {n_channels}")
    log(f"Redshift: {round(spectral.redshift, 3)}")
    log(f"Source frequency: {round(params.freq, 2)} GHz")
    log(f"Band: {params.band}")
    log(f"Velocity resolution: {round(params.vel_res, 2)} Km/s")
    log(f"Angular resolution: {round(params.ang_res, 3)} arcsec")
    log(f"Infrared Luminosity: {spectral.lum_infrared:.2e}")

    sky = build_sky_cube(
        params,
        geometry,
        spectral,
        cont_sens=cont_sens,
        cont_sens_jy=cont_sens_jy,
        ctx=ctx,
        progress_adapter=progress_adapter,
        compute_backend=compute_backend,
        terminal_logger=terminal_logger,
    )
    model = usm.channel_first(sky.datacube) / beam_area.value
    totflux = np.sum(model)
    log(f"Total Flux injected in model cube: {round(totflux, 3)} Jy")
    log("Observing with ALMA")

    sim_params_payload = sim_params_record(params, geometry, spectral, sky, output_dir_abs)
    metadata = simulation_metadata(params, geometry, spectral, sky, persist_outputs=persist_outputs)
    effective_snr = resolve_snr(params, log)
    metadata["effective_snr"] = effective_snr
    metadata["snr_mode"] = "auto" if params.snr is None else "manual"
    channel_frequencies_hz = geometry.channel_frequencies_hz()
    interferometer_runs, total_power_runs = plan_observation_runs(
        params,
        geometry,
        model_cube=model,
        header=sky.header,
        channel_frequencies_hz=channel_frequencies_hz,
        effective_snr=effective_snr,
        base_noise_reference=noise_reference(spectral.line_fluxes, beam_area, effective_snr),
        output_dir_abs=output_dir_abs,
        persist_outputs=persist_outputs,
    )
    interferometer_kwargs = (
        interferometer_runs[0].interferometer_kwargs() if interferometer_runs else {}
    )

    return CleanCubeStage(
        seed=params.seed,
        corruptions=_window_corruptions(params, geometry),
        datacube=sky.datacube,
        model_cube=model,
        header=sky.header,
        output_dir_abs=output_dir_abs,
        sim_output_dir=sim_output_dir,
        sim_params_payload=sim_params_payload,
        interferometer_kwargs=interferometer_kwargs,
        interferometer_runs=interferometer_runs,
        total_power_runs=total_power_runs,
        metadata=metadata,
        observation_plan=geometry.observation_plan.as_dict(),
        channel_frequencies_hz=channel_frequencies_hz,
        channel_width_hz=geometry.channel_width_hz,
        cell_size_arcsec=geometry.cell_size.to(U.arcsec).value,
        background_cube=sky.background_cube.astype(np.float32),
        external_input_metadata=sky.external_input_metadata,
        continuum_fraction=(
            None
            if is_external_source_type(params.source_type)
            else spectral.continuum_fraction(geometry.n_channels)
        ),
    )


def _window_corruptions(params: SimulationParams, geometry: CubeGeometry) -> CorruptionModel | None:
    """The run's corruption model on its window's channels."""
    model = CorruptionModel.from_mapping(params.corruptions)
    return None if model is None else model.for_window(geometry.channel_averaging)


def simulate_observation(  # noqa: C901
    clean_cube_stage: CleanCubeStage,
    *,
    compute_backend=None,
    robust: float = 0.0,
    terminal_logger=None,
    interferometer_progress_callback: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Run the interferometric observation stage from a prepared clean cube.

    Without a ``compute_backend`` the channels are imaged in-process
    (:class:`SyncBackend`); the Interferometer needs one to scatter the cube.
    """
    if compute_backend is None:
        from ..compute.sync import SyncBackend

        compute_backend = SyncBackend()
    per_config_results = []
    total_configs = max(
        len(clean_cube_stage.interferometer_runs) + len(clean_cube_stage.total_power_runs),
        1,
    )
    progress_index = 0

    for config_index, run in enumerate(clean_cube_stage.interferometer_runs):
        interferometer = uin.Interferometer(
            backend=compute_backend,
            robust=robust,
            logger=terminal_logger,
            noise_entropy=stream_entropy(clean_cube_stage.seed, "interferometer", config_index),
            corruptions=clean_cube_stage.corruptions,
            continuum_fraction=clean_cube_stage.continuum_fraction,
            **run.interferometer_kwargs(),
        )
        if interferometer_progress_callback is not None:

            def scaled_progress(progress: int, *, idx=progress_index):
                base = idx / total_configs
                scaled = int(round((base + (progress / 100.0) / total_configs) * 100))
                interferometer_progress_callback(scaled)

            interferometer.progress_signal.connect(scaled_progress)
        config_result = interferometer.run_interferometric_sim()
        config_result["config_name"] = run.config_name
        config_result["array_type"] = run.array_type
        per_config_results.append(config_result)
        progress_index += 1

    int_results = None
    if per_config_results:
        int_results = uin.combine_interferometric_results(
            per_config_results,
            config_weights=[
                max(float(run.integration_time), 0.0)
                for run in clean_cube_stage.interferometer_runs
            ],
        )

    tp_per_config_results = []
    # One generator for every TP configuration, from the run's seed.
    tp_rng = run_rng(clean_cube_stage.seed, "total_power")
    for tp_run in clean_cube_stage.total_power_runs:
        tp_result = uin.simulate_total_power_observation(
            clean_cube_stage.model_cube,
            freqs_hz=clean_cube_stage.channel_frequencies_hz,
            cell_size_arcsec=clean_cube_stage.cell_size_arcsec,
            noise_profile=tp_run.noise,
            antenna_diameter_m=tp_run.antenna_diameter_m,
            config_name=tp_run.config_name,
            array_type=tp_run.array_type,
            rng=tp_rng,
        )
        tp_per_config_results.append(tp_result)
        if interferometer_progress_callback is not None:
            scaled = int(round(((progress_index + 1) / total_configs) * 100))
            interferometer_progress_callback(scaled)
        progress_index += 1

    tp_results = None
    if tp_per_config_results:
        tp_results = uin.combine_total_power_results(
            tp_per_config_results,
            config_weights=[
                max(float(run.integration_time_s), 0.0) for run in clean_cube_stage.total_power_runs
            ],
        )

    measurement_results: dict[str, Any] = {
        "observation_plan": clean_cube_stage.observation_plan,
        "int_results": int_results,
        "tp_results": tp_results,
        "model_cube": clean_cube_stage.model_cube,
    }
    if int_results is not None:
        measurement_results.update(int_results)
    if tp_results is not None:
        measurement_results.update(
            {key: value for key, value in tp_results.items() if key.startswith("tp_")}
        )
    return measurement_results


def image_products(
    simulation_results: dict[str, Any],
    *,
    reconstruction_epsilon: float = 1e-3,
) -> dict[str, Any]:
    """Construct image-domain products from measurement-stage outputs."""
    simulation_results = dict(simulation_results)
    products = build_image_products(
        int_results=simulation_results.get("int_results"),
        tp_results=simulation_results.get("tp_results"),
        reconstruction_epsilon=reconstruction_epsilon,
    )
    simulation_results.update(products)
    return simulation_results
