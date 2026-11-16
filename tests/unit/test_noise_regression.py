"""Monte Carlo checks of the simulated noise against the radiometer equation
and of the antenna ripple across baselines (audit F-07, F-09; plan C4): the
noise a simulated channel draws, measured as an image or a real MS is, gives
back what the model asks for."""

from __future__ import annotations

import numpy as np
import pytest

from almasim.services.interferometry.corruptions import antenna_ripple, baseline_antennas
from almasim.services.interferometry.noise import (
    NoiseModelConfig,
    system_equivalent_flux_density,
    visibility_noise,
)
from almasim.services.interferometry.simulator import ChannelImagingConfig, image_configured_channel

NANT, NH = 10, 300


def _config(**extra) -> ChannelImagingConfig:
    """A compact array (positions in km) observing a blank field."""
    npix = 64
    xy = np.random.default_rng(3).uniform(0.0, 0.1, (NANT, 2))
    return ChannelImagingConfig(
        Npix=npix,
        Nant=NANT,
        Hcov=[-0.5, 0.5],
        nH=NH,
        antPos=xy.tolist(),
        robfac=0.0,
        trlat=[np.sin(np.radians(-23.0)), np.cos(np.radians(-23.0))],
        trdec=[np.sin(np.radians(-30.0)), np.cos(np.radians(-30.0))],
        Diameters=[12.0, 0.0],
        imsize=20.0,
        Xmax=0.2,
        lfac=1.0e6,
        distmat=np.zeros((npix, npix)),
        Nphf=npix // 2,
        Np4=npix // 4,
        zooming=0,
        header={"CDELT1": -1.0 / 3600, "CDELT2": 1.0 / 3600, "NAXIS1": npix, "NAXIS2": npix},
        robust=0.0,
        scan_time_s=6.0,
        noise_entropy=(11, 2, 0),
        **extra,
    )


def _rows(config: ChannelImagingConfig, noise: float, channel: int) -> dict[str, np.ndarray]:
    """One blank channel's visibility rows (those on the uv grid)."""
    rows = image_configured_channel(config, np.zeros((64, 64)), [1e-6] * 3, noise, channel)[-1]
    valid = rows["valid"]
    return {key: np.asarray(rows[key])[valid] for key in ("data", "sigma")} | {
        "baseline": np.repeat(np.arange(NANT * (NANT - 1) // 2), NH)[valid]
    }


@pytest.mark.unit
def test_simulated_visibilities_carry_the_radiometer_noise_and_their_weight():
    """A band 6 12 m channel: each visibility's real and imaginary parts scatter
    by SEFD / (eta sqrt(2 n_pol dnu t)), its row's sigma says so (variance x
    WEIGHT = 1), and the mean of N(N-1)/2 baselines over a track, the image's
    noise at the phase centre with natural weighting, is ALMA's image
    sensitivity SEFD / (eta sqrt(n_pol N(N-1) dnu t_total))."""
    radiometer = NoiseModelConfig(pwv_mm=1.0)
    freq, width, dump = np.array([230e9]), 15.625e6, 6.0
    sigma = float(visibility_noise(radiometer, freq, width, dump, 60.0, antenna_diameter_m=12.0)[0])
    sefd = float(system_equivalent_flux_density(radiometer, freq, 60.0, antenna_diameter_m=12.0)[0])
    eta = radiometer.correlator_efficiency
    assert sigma == pytest.approx(
        sefd / (eta * np.sqrt(2 * radiometer.n_polarizations * width * dump))
    )
    config = _config()
    means, parts = [], []
    for channel in range(60):
        rows = _rows(config, sigma, channel)
        parts.append(np.r_[rows["data"].real, rows["data"].imag])
        np.testing.assert_allclose(rows["sigma"], sigma, rtol=1e-5)
        means.append(rows["data"].mean())
    measured = np.std(np.concatenate(parts))
    assert measured == pytest.approx(sigma, rel=0.01)
    assert float(np.mean(np.concatenate(parts) ** 2) / sigma**2) == pytest.approx(1.0, abs=0.01)
    # Natural weighting: the image at the phase centre is the rows' mean.
    n = rows["data"].size
    image = sefd / (eta * np.sqrt(radiometer.n_polarizations * 2 * n * width * dump))
    assert np.std(np.real(means)) == pytest.approx(image, rel=0.25)


@pytest.mark.unit
def test_the_antenna_ripple_factors_by_antenna_across_baselines():
    """Each baseline's noise variance is g_i g_j times the channel's: its log
    splits into one term per antenna with nothing left over beyond the
    sampling, and the terms are the antennas' own ripple (F-09: the real
    windows' noise factors the same way)."""
    freq = 230e9 + np.arange(4) * 15.625e6
    g = antenna_ripple(np.random.default_rng(7), NANT, freq, 0.15, 30.0)
    config = _config(antenna_noise=g)
    pairs = baseline_antennas(NANT)
    design = np.zeros((pairs.shape[0], NANT))
    design[np.arange(pairs.shape[0]), pairs[:, 0]] = 1
    design[np.arange(pairs.shape[0]), pairs[:, 1]] = 1
    for channel in range(freq.size):
        rows = _rows(config, 1.0, channel)
        z = rows["data"]
        power = np.bincount(rows["baseline"], np.abs(z) ** 2, pairs.shape[0])
        count = np.bincount(rows["baseline"], minlength=pairs.shape[0])
        seen = count > 50
        log_variance = np.log(power[seen] / count[seen] / 2)
        terms, *_ = np.linalg.lstsq(design[seen], log_variance, rcond=None)
        residual = log_variance - design[seen] @ terms
        # The sampling of a variance from 2n draws: sqrt(1 / n) in its log.
        sampling = np.sqrt(np.mean(1 / count[seen]))
        assert np.std(residual) < 1.5 * sampling
        np.testing.assert_allclose(terms, np.log(g[:, channel]), atol=4 * sampling)
