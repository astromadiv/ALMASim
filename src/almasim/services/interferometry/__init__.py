"""Interferometry service for ALMA simulations."""

from .antenna import (
    compute_distance,
    estimate_alma_beam_size,
    generate_antenna_config_file_from_antenna_array,
    get_fov_from_band,
    get_max_baseline_from_antenna_array,
)
from .baselines import (
    generate_via_astropy,
    pairwise_baselines,
    prepare_baselines,
    set_baselines,
    set_noise,
)
from .core import Interferometer, ProgressSignal
from .frequency import freq_supp_extractor, remove_non_numeric
from .imaging import (
    _grid_uv,
    _prepare_model,
    add_thermal_noise,
    check_lfac,
    image_channel,
    image_channel_ducc0,
    observe,
    prepare_2d_arrays,
    set_beam,
    set_primary_beam,
)
from .multiconfig import combine_interferometric_results
from .noise import (
    NoiseModelConfig,
    calibrate_noise_profile,
    compute_channel_noise,
    estimate_receiver_temperature_k,
    estimate_sky_temperature_k,
    system_equivalent_flux_density,
    visibility_noise,
)
from .total_power import (
    combine_total_power_results,
    estimate_tp_beam_fwhm_arcsec,
    simulate_total_power_observation,
)
from .utils import closest_power_of_2, get_channel_wavelength, sampling_to_uv_mask

__all__ = [
    "Interferometer",
    "ProgressSignal",
    "prepare_baselines",
    "set_baselines",
    "set_noise",
    "pairwise_baselines",
    "generate_via_astropy",
    "NoiseModelConfig",
    "estimate_receiver_temperature_k",
    "estimate_sky_temperature_k",
    "compute_channel_noise",
    "calibrate_noise_profile",
    "system_equivalent_flux_density",
    "visibility_noise",
    "combine_interferometric_results",
    "estimate_tp_beam_fwhm_arcsec",
    "simulate_total_power_observation",
    "combine_total_power_results",
    "prepare_2d_arrays",
    "_grid_uv",
    "set_beam",
    "_prepare_model",
    "set_primary_beam",
    "observe",
    "image_channel",
    "image_channel_ducc0",
    "check_lfac",
    "add_thermal_noise",
    "get_channel_wavelength",
    "closest_power_of_2",
    "sampling_to_uv_mask",
    "estimate_alma_beam_size",
    "get_fov_from_band",
    "generate_antenna_config_file_from_antenna_array",
    "compute_distance",
    "get_max_baseline_from_antenna_array",
    "freq_supp_extractor",
    "remove_non_numeric",
]
