"""Physically-motivated thermal noise helpers for ALMA simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atmosphere import system_temperature

K_B = 1.380649e-23
JY = 1.0e-26


@dataclass
class NoiseModelConfig:
    """Configuration for the single-pointing thermal noise model."""

    pwv_mm: float
    ground_temperature_k: float = 270.0
    receiver_model: str = "alma"
    site: str = "ALMA"
    aperture_efficiency: float = 0.72
    n_polarizations: int = 2
    # Tsys from CASA's ATM model (atmosphere.py): lines in the signal and image
    # bands, referred above the atmosphere. Off: the smooth sky below.
    atmospheric_lines: bool = False
    sideband: str = "auto"
    # ALMA's quantisation and correlator efficiencies (eta_q eta_c): what a
    # visibility's noise carries beyond the radiometer equation.
    correlator_efficiency: float = 0.96 * 0.88


def estimate_receiver_temperature_k(
    freqs_hz: np.ndarray, receiver_model: str = "alma"
) -> np.ndarray:
    """Estimate receiver temperature with a coarse ALMA band approximation."""
    freqs_ghz = np.asarray(freqs_hz, dtype=float) / 1e9
    trx = np.full_like(freqs_ghz, 45.0, dtype=float)
    trx = np.where(freqs_ghz < 116.0, 35.0, trx)
    trx = np.where((freqs_ghz >= 116.0) & (freqs_ghz < 163.0), 45.0, trx)
    trx = np.where((freqs_ghz >= 163.0) & (freqs_ghz < 211.0), 55.0, trx)
    trx = np.where((freqs_ghz >= 211.0) & (freqs_ghz < 275.0), 65.0, trx)
    trx = np.where((freqs_ghz >= 275.0) & (freqs_ghz < 373.0), 85.0, trx)
    trx = np.where((freqs_ghz >= 373.0) & (freqs_ghz < 500.0), 110.0, trx)
    trx = np.where(freqs_ghz >= 500.0, 160.0, trx)
    if receiver_model.lower() != "alma":
        trx = trx * 1.1
    return trx


def estimate_sky_temperature_k(
    freqs_hz: np.ndarray,
    pwv_mm: float,
    elevation_deg: float,
    ground_temperature_k: float,
) -> np.ndarray:
    """Approximate sky brightness temperature from PWV and airmass."""
    freqs_ghz = np.asarray(freqs_hz, dtype=float) / 1e9
    elevation_rad = np.deg2rad(max(5.0, min(90.0, float(elevation_deg))))
    airmass = 1.0 / max(np.sin(elevation_rad), 0.1)
    tau = (0.005 + 0.0025 * pwv_mm) * airmass * (1.0 + freqs_ghz / 400.0)
    tau = np.clip(tau, 0.0, 3.0)
    return ground_temperature_k * (1.0 - np.exp(-tau))


def system_equivalent_flux_density(
    config: NoiseModelConfig,
    freqs_hz: np.ndarray,
    elevation_deg: float,
    *,
    antenna_diameter_m: float,
) -> np.ndarray:
    """Each channel's SEFD (Jy) of one antenna: 2 k Tsys / A_eff, with Tsys from
    the receiver and the sky (CASA's ATM model with ``atmospheric_lines``)."""
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    trx = estimate_receiver_temperature_k(freqs_hz, config.receiver_model)
    if config.atmospheric_lines:
        airmass = 1.0 / np.sin(np.deg2rad(max(5.0, min(90.0, float(elevation_deg)))))
        tsys = system_temperature(
            freqs_hz,
            trx,
            pwv_mm=config.pwv_mm,
            airmass=airmass,
            ground_temperature_k=config.ground_temperature_k,
            sideband=config.sideband,
        )
    else:
        tsky = estimate_sky_temperature_k(
            freqs_hz,
            pwv_mm=config.pwv_mm,
            elevation_deg=elevation_deg,
            ground_temperature_k=config.ground_temperature_k,
        )
        tsys = np.clip(trx + tsky, 1.0, None)

    area_m2 = np.pi * (antenna_diameter_m / 2.0) ** 2
    a_eff = max(area_m2 * config.aperture_efficiency, 1e-6)
    return (2.0 * K_B * tsys / a_eff) / JY


def compute_channel_noise(
    config: NoiseModelConfig,
    freqs_hz: np.ndarray,
    bandwidth_hz: float,
    integration_s: float,
    elevation_deg: float,
    *,
    antenna_diameter_m: float,
    n_antennas: int,
) -> np.ndarray:
    """Compute per-channel thermal noise in Jy using a simplified radiometer model."""
    bandwidth_hz = max(float(bandwidth_hz), 1.0)
    integration_s = max(float(integration_s), 1e-6)
    n_antennas = max(int(n_antennas), 2)
    n_baselines = max(n_antennas * (n_antennas - 1) / 2.0, 1.0)
    sefd_jy = system_equivalent_flux_density(
        config, freqs_hz, elevation_deg, antenna_diameter_m=antenna_diameter_m
    )
    denominator = np.sqrt(config.n_polarizations * n_baselines * bandwidth_hz * integration_s)
    return sefd_jy / np.clip(denominator, 1e-6, None)


def visibility_noise(
    config: NoiseModelConfig,
    freqs_hz: np.ndarray,
    channel_width_hz: float,
    sample_time_s: float,
    elevation_deg: float,
    *,
    antenna_diameter_m: float,
) -> np.ndarray:
    """Each channel's noise (Jy) of one visibility, per real and imaginary part:
    the radiometer equation for a baseline's Stokes I from ``n_polarizations``
    over ``channel_width_hz`` and ``sample_time_s``,
    SEFD / (eta sqrt(2 n_pol dnu t)). Imaged with natural weighting, N such
    visibilities give SEFD / (eta sqrt(n_pol N(N-1) dnu t_total)), ALMA's
    image sensitivity."""
    sefd_jy = system_equivalent_flux_density(
        config, freqs_hz, elevation_deg, antenna_diameter_m=antenna_diameter_m
    )
    product = (
        2.0
        * config.n_polarizations
        * max(float(channel_width_hz), 1.0)
        * max(float(sample_time_s), 1e-6)
    )
    return sefd_jy / (max(config.correlator_efficiency, 1e-6) * np.sqrt(product))


def calibrate_noise_profile(
    raw_noise_profile: np.ndarray,
    *,
    reference_noise: float,
) -> np.ndarray:
    """Scale a per-channel profile to preserve the existing noise amplitude convention."""
    raw_noise_profile = np.asarray(raw_noise_profile, dtype=float)
    reference_noise = max(float(reference_noise), 0.0)
    median = float(np.median(raw_noise_profile)) if raw_noise_profile.size else 0.0
    if median <= 0.0:
        return np.full_like(raw_noise_profile, reference_noise, dtype=float)
    return raw_noise_profile * (reference_noise / median)
