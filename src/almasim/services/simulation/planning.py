"""One interferometer or total-power run per observing configuration (MADIV-28 T5.7)."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from datetime import date, timedelta
from typing import Any

import astropy.units as U
import numpy as np

from .. import interferometry as uin
from ..interferometry.atmosphere import receiver_ripple
from ..interferometry.corruptions import CorruptionModel
from ..interferometry.observing import hour_angle_samples
from .geometry import CubeGeometry
from .params import SimulationParams
from .rng import run_rng


@dataclass(frozen=True)
class InterferometerRun:
    """One interferometric configuration of a run: what its Interferometer is built from
    (MADIV-28 T5.8)."""

    idx: int
    skymodel: np.ndarray
    main_dir: str
    output_dir: str
    ra: U.Quantity
    dec: U.Quantity
    central_freq: U.Quantity
    bandwidth: U.Quantity
    fov: float
    antenna_array: str
    noise: Any
    snr: float
    integration_time: float  # hours
    observation_date: str
    header: Any
    save_mode: str
    persist: bool
    antenna_diameter_m: float
    config_name: str | None
    array_type: str | None
    imaging_algorithm: str
    spectral_window: Any
    # Seconds per hour-angle sample, the same for each of a MOUS's execution
    # blocks; None: from the integration time.
    sample_time_s: float | None = None
    # Which execution block of the MOUS this run is (0, 1, ...).
    execution: int | None = None

    def interferometer_kwargs(self) -> dict[str, Any]:
        """The run as ``Interferometer`` keyword arguments."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class TotalPowerRun:
    """One total-power configuration of a run (MADIV-28 T5.8)."""

    config_name: str | None
    array_type: str
    antenna_array: str
    antenna_diameter_m: float
    integration_time_s: float
    noise: Any


# The longest an execution block observes its source: ALMA runs a scheduling
# block for about an hour, and the longest time on source among the 100
# MADIV-16 windows was 2,964 s.
MAX_EXECUTION_TIME_S = 3000.0


def execution_count(total_time_s: float, n_executions: int | None = None) -> int:
    """How many execution blocks observe a MOUS of ``total_time_s`` on source: the
    archive's count when known, else enough blocks of at most MAX_EXECUTION_TIME_S."""
    if n_executions is not None and n_executions >= 1:
        return int(n_executions)
    return max(1, math.ceil(float(total_time_s) / MAX_EXECUTION_TIME_S))


def execution_date(obs_date: str, execution: int) -> str:
    """The date execution block ``execution`` observes on: one a day from ``obs_date``."""
    try:
        start = date.fromisoformat(str(obs_date)[:10])
    except ValueError:
        return str(obs_date)
    return (start + timedelta(days=int(execution))).isoformat()


def _count_antennas(antenna_array: str) -> int:
    """Count antennas encoded in an ALMA antenna-array string."""
    tokens = [token for token in str(antenna_array).split() if token.strip()]
    return max(len(tokens), 2)


def noise_reference(line_fluxes: Any, beam_area: U.Quantity, effective_snr: float) -> float:
    """The noise the per-channel noise profiles are calibrated to: a tenth of the
    faintest line's peak per beam, over the SNR."""
    line_flux_magnitudes = np.abs(np.asarray(line_fluxes, dtype=float))
    nonzero_line_fluxes = line_flux_magnitudes[line_flux_magnitudes > 0]
    min_line_flux = float(np.min(nonzero_line_fluxes)) if nonzero_line_fluxes.size > 0 else 0.0
    return 0.1 * (min_line_flux / beam_area.value) / effective_snr


def plan_observation_runs(
    params: SimulationParams,
    geometry: CubeGeometry,
    *,
    model_cube: np.ndarray,
    header: Any,
    channel_frequencies_hz: np.ndarray,
    effective_snr: float,
    base_noise_reference: float,
    output_dir_abs: str,
    persist_outputs: bool,
) -> tuple[list[InterferometerRun], list[TotalPowerRun]]:
    """The interferometric runs, one per execution block of each configuration, and
    the total-power runs, one per configuration."""
    observation_plan = geometry.observation_plan
    corruptions = CorruptionModel.from_mapping(params.corruptions)
    interferometer_runs: list[InterferometerRun] = []
    total_power_runs: list[TotalPowerRun] = []
    for index, config in enumerate(observation_plan.configs):
        raw_noise_profile = uin.compute_channel_noise(
            uin.NoiseModelConfig(
                pwv_mm=params.pwv,
                ground_temperature_k=params.ground_temperature_k,
                atmospheric_lines=corruptions is not None and corruptions.atmospheric_tsys,
            ),
            channel_frequencies_hz,
            geometry.channel_width_hz,
            config.total_time_s,
            observation_plan.elevation_deg,
            antenna_diameter_m=config.antenna_diameter_m,
            n_antennas=_count_antennas(config.antenna_array),
        )
        ripple = np.zeros(np.size(raw_noise_profile))
        if corruptions is not None and corruptions.receiver_ripple_rms > 0:
            rng = run_rng(params.seed, "receiver", index)
            ripple = receiver_ripple(
                rng, np.size(raw_noise_profile), corruptions.receiver_ripple_rms
            )
            raw_noise_profile = np.asarray(raw_noise_profile) * (1.0 + ripple)
        noise_profile = uin.calibrate_noise_profile(
            raw_noise_profile,
            reference_noise=base_noise_reference,
        )
        # A MOUS is observed in execution blocks, each a track of its own near
        # transit on its own day, sampled alike; the samples of the blocks
        # simulated stay within one track's budget (observing.hour_angle_samples).
        n_executions = execution_count(config.total_time_s, params.n_executions)
        simulated = (
            n_executions
            if params.executions is None
            else min(max(int(params.executions), 1), n_executions)
        )
        execution_time_s = config.total_time_s / n_executions
        _, sample_time_s = hour_angle_samples(execution_time_s * simulated)
        if corruptions is not None and corruptions.radiometer_noise and config.array_type != "TP":
            # Each visibility's own noise, over the integration the simulator samples.
            noise_profile = uin.visibility_noise(
                uin.NoiseModelConfig(
                    pwv_mm=params.pwv,
                    ground_temperature_k=params.ground_temperature_k,
                    atmospheric_lines=corruptions.atmospheric_tsys,
                ),
                channel_frequencies_hz,
                geometry.noise_bandwidth_hz,
                sample_time_s,
                observation_plan.elevation_deg,
                antenna_diameter_m=config.antenna_diameter_m,
            ) * (1.0 + ripple)
        if config.array_type == "TP":
            total_power_runs.append(
                TotalPowerRun(
                    config_name=config.name,
                    array_type=config.array_type,
                    antenna_array=config.antenna_array,
                    antenna_diameter_m=config.antenna_diameter_m,
                    integration_time_s=config.total_time_s,
                    noise=noise_profile,
                )
            )
        else:
            interferometer_runs.extend(
                InterferometerRun(
                    idx=params.idx,
                    skymodel=model_cube,
                    main_dir=params.main_dir,
                    output_dir=output_dir_abs,
                    ra=params.ra * U.deg,
                    dec=params.dec * U.deg,
                    central_freq=geometry.central_freq,
                    bandwidth=geometry.band_range,
                    fov=geometry.fov.value,
                    antenna_array=config.antenna_array,
                    noise=noise_profile,
                    snr=effective_snr,
                    integration_time=execution_time_s / 3600.0,
                    observation_date=execution_date(params.obs_date, execution),
                    header=header,
                    save_mode=params.save_mode,
                    persist=persist_outputs,
                    antenna_diameter_m=config.antenna_diameter_m,
                    config_name=config.name,
                    array_type=config.array_type,
                    imaging_algorithm=params.imaging_algorithm,
                    spectral_window=geometry.spectral_window(),
                    sample_time_s=sample_time_s,
                    execution=execution,
                )
                for execution in range(simulated)
            )
    return interferometer_runs, total_power_runs
