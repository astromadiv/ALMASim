"""The sky and Tsys across a window (atmosphere.py) and their use in the noise profile."""

import numpy as np
import pytest

from almasim.services.interferometry.atmosphere import (
    receiver_band,
    receiver_ripple,
    sky_spectrum,
    system_temperature,
)
from almasim.services.interferometry.corruptions import CorruptionModel
from almasim.services.interferometry.noise import (
    NoiseModelConfig,
    compute_channel_noise,
    system_equivalent_flux_density,
    visibility_noise,
)
from almasim.services.noise_model import released_model


def _window(centre_ghz, width_ghz=1.875, nchan=128):
    return np.linspace(centre_ghz - width_ghz / 2, centre_ghz + width_ghz / 2, nchan) * 1e9


def test_sky_is_brighter_in_the_water_line_and_with_more_water():
    f = np.array([183.31e9, 230e9])
    dry, _ = sky_spectrum(f, pwv_mm=0.5, airmass=1.0)
    wet, tau = sky_spectrum(f, pwv_mm=3.0, airmass=1.0)
    assert dry[0] > 5 * dry[1]
    assert np.all(wet > dry)
    assert np.all(tau > 0)
    # Opaque at the line centre: the sky is close to the air's own temperature.
    assert 200 < wet[0] < 290


def test_sky_outside_the_table_fails():
    with pytest.raises(ValueError, match="outside the atmosphere table"):
        sky_spectrum(np.array([10e9]), pwv_mm=1.0, airmass=1.0)


def test_tsys_rises_towards_an_atmospheric_line():
    """A window on the wing of the 118.75 GHz oxygen line, as the MADIV-16 CO(1-0)
    windows at 114-115 GHz show it."""
    f = _window(114.7)
    tsys = system_temperature(f, 35.0, pwv_mm=1.0, airmass=1.2)
    assert np.mean(np.diff(tsys) > 0) > 0.95  # rising, but for narrow ozone lines
    assert tsys[-1] / tsys[0] > 1.05


def test_image_band_and_sideband_choice():
    f = _window(203.8)  # Band 5: the image band of the USB tuning sits near 183 GHz
    kwargs = {"pwv_mm": 1.5, "airmass": 1.2}
    usb = system_temperature(f, 55.0, sideband="usb", **kwargs)
    lsb = system_temperature(f, 55.0, sideband="lsb", **kwargs)
    auto = system_temperature(f, 55.0, **kwargs)
    assert np.mean(usb) > np.mean(lsb)
    np.testing.assert_allclose(auto, lsb)
    # Band 1 has no image band; Band 9 is double sideband.
    assert receiver_band(43e9).image_gain == 0
    assert receiver_band(650e9).image_gain == 1
    with pytest.raises(ValueError, match="sideband"):
        system_temperature(f, 55.0, sideband="dsb", **kwargs)


def test_receiver_ripple_is_smooth_reproducible_and_of_its_rms():
    ripples = np.stack([receiver_ripple(np.random.default_rng(k), 256, 0.02) for k in range(400)])
    assert np.sqrt(np.mean(ripples**2)) == pytest.approx(0.02, rel=0.1)
    np.testing.assert_array_equal(
        receiver_ripple(np.random.default_rng(3), 256, 0.02),
        receiver_ripple(np.random.default_rng(3), 256, 0.02),
    )
    assert np.max(np.abs(np.diff(ripples, axis=1))) < 0.01  # smooth across channels
    assert not np.any(receiver_ripple(np.random.default_rng(0), 256, 0.0))


def test_channel_noise_follows_the_atmosphere_only_when_asked():
    f = _window(114.7)
    common = {
        "bandwidth_hz": 1e6,
        "integration_s": 3600.0,
        "elevation_deg": 60.0,
        "antenna_diameter_m": 12.0,
        "n_antennas": 43,
    }
    smooth = compute_channel_noise(NoiseModelConfig(pwv_mm=1.0), f, **common)
    lines = compute_channel_noise(NoiseModelConfig(pwv_mm=1.0, atmospheric_lines=True), f, **common)
    assert smooth[-1] / smooth[0] < 1.01
    assert lines[-1] / lines[0] > 1.05


def test_switches_in_the_corruption_model():
    assert CorruptionModel.from_mapping({"atmospheric_tsys": "true"}).atmospheric_tsys is True
    assert (
        CorruptionModel.from_mapping({"atmospheric_tsys": 0, "gain_amp_rms": 0.1}).atmospheric_tsys
        is False
    )
    with pytest.raises(ValueError, match="not a switch"):
        CorruptionModel.from_mapping({"atmospheric_tsys": "maybe"})
    alma = CorruptionModel.from_mapping(released_model("alma").engine_fields())
    assert alma.atmospheric_tsys and alma.receiver_ripple_rms == pytest.approx(0.02)
    assert alma.radiometer_noise and not CorruptionModel().radiometer_noise


def test_visibility_noise_follows_the_radiometer_equation():
    """One visibility's noise is SEFD / (eta sqrt(2 n_pol dnu t)); N(N-1)/2 baselines
    over a track, imaged with natural weighting, give ALMA's image sensitivity
    SEFD / (eta sqrt(n_pol N(N-1) dnu T)). A 12 m baseline at 230 GHz over
    15.6 MHz and 6 s is of order 0.1 Jy, not a fraction of a mJy."""
    config = NoiseModelConfig(pwv_mm=1.0)
    freqs = _window(230.0, nchan=8)
    dnu, t, n_ant, n_samples = 15.625e6, 6.0, 43, 600
    sigma = visibility_noise(config, freqs, dnu, t, 60.0, antenna_diameter_m=12.0)
    sefd = system_equivalent_flux_density(config, freqs, 60.0, antenna_diameter_m=12.0)
    eta = config.correlator_efficiency
    np.testing.assert_allclose(sigma, sefd / (eta * np.sqrt(2 * 2 * dnu * t)))
    n_vis = n_ant * (n_ant - 1) / 2 * n_samples
    image = sigma / np.sqrt(n_vis)
    np.testing.assert_allclose(
        image, sefd / (eta * np.sqrt(2 * n_ant * (n_ant - 1) * dnu * t * n_samples))
    )
    assert np.all((sigma > 0.02) & (sigma < 1.0))
