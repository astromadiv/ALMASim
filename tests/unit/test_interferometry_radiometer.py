"""The radiometer's receiver temperatures and aperture efficiency, calibrated on
the noise study's windows (MADIV-28)."""

from __future__ import annotations

import numpy as np
import pytest

from almasim.services.interferometry.noise import (
    NoiseModelConfig,
    aperture_efficiency,
    estimate_receiver_temperature_k,
    system_equivalent_flux_density,
)


@pytest.mark.unit
def test_each_band_has_its_measured_receiver_temperature():
    ghz = np.array([40, 100, 145, 190, 230, 345, 460, 650, 870]) * 1e9
    np.testing.assert_array_equal(
        estimate_receiver_temperature_k(ghz), [32, 39, 37, 50, 46, 63, 110, 160, 160]
    )
    assert estimate_receiver_temperature_k(np.array([230e9]), "other")[0] == pytest.approx(50.6)


@pytest.mark.unit
def test_the_dish_loses_efficiency_as_the_wavelength_shrinks():
    assert aperture_efficiency(100e9, 12.0) == pytest.approx(
        0.76 * np.exp(-((4 * np.pi * 27e-6 * 100e9 / 299_792_458.0) ** 2))
    )
    assert aperture_efficiency(100e9, 12.0) > 0.74
    assert 0.25 < aperture_efficiency(870e9, 12.0) < 0.3
    assert aperture_efficiency(345e9, 7.0) > aperture_efficiency(345e9, 12.0) * 0.9
    assert aperture_efficiency(345e9, 9.0) == aperture_efficiency(345e9, 12.0)


@pytest.mark.unit
def test_the_sefd_uses_the_dish_efficiency_unless_one_is_given():
    freqs = np.linspace(229e9, 231e9, 8)
    own = system_equivalent_flux_density(
        NoiseModelConfig(pwv_mm=1.0), freqs, 60.0, antenna_diameter_m=12.0
    )
    given = system_equivalent_flux_density(
        NoiseModelConfig(pwv_mm=1.0, aperture_efficiency=0.5), freqs, 60.0, antenna_diameter_m=12.0
    )
    np.testing.assert_allclose(given / own, aperture_efficiency(230e9, 12.0) / 0.5)
