"""Arrays whose pads hold dishes of two sizes: a 7 m array correlating ALMA's
12 m total-power dishes on their pads (T701-T704)."""

from __future__ import annotations

import numpy as np
import pytest

from almasim.services.instrument.array import pad_dish_diameter_m
from almasim.services.instrument.primary_beam import (
    baseline_beam_response,
    primary_beam_fwhm_rad,
    primary_beam_response,
    primary_beam_voltage,
)
from almasim.services.interferometry.imaging import baseline_beam_models
from almasim.services.interferometry.visibility import build_channel_visibility_rows


@pytest.mark.unit
def test_a_total_power_pad_holds_a_12m_dish():
    assert pad_dish_diameter_m("T701", 7.0) == 12.0
    assert pad_dish_diameter_m("t704", 7.0) == 12.0
    assert pad_dish_diameter_m("J501", 7.0) == 7.0
    assert pad_dish_diameter_m("A001", 12.0) == 12.0
    assert pad_dish_diameter_m("T703", 12.0) == 12.0


@pytest.mark.unit
def test_a_baseline_between_two_dish_sizes_sees_the_product_of_their_voltage_patterns():
    radius = np.linspace(0.0, 2.0 * primary_beam_fwhm_rad(7.0, 650e9), 400)
    seven = primary_beam_response(radius, 650e9, 7.0)
    twelve = primary_beam_response(radius, 650e9, 12.0)
    mixed = baseline_beam_response(radius, 650e9, 7.0, 12.0)
    assert mixed[0] == pytest.approx(1.0)
    voltage = primary_beam_voltage(radius, 650e9, 7.0) * primary_beam_voltage(radius, 650e9, 12.0)
    np.testing.assert_allclose(mixed, voltage)
    # Alike dishes: each one's power response, bit for bit.
    np.testing.assert_array_equal(baseline_beam_response(radius, 650e9, 7.0, 7.0), seven)
    # Its half-power radius lies between the two dishes' own.
    half = {
        name: radius[np.argmax(r < 0.5)] for name, r in (("7", seven), ("x", mixed), ("12", twelve))
    }
    assert half["12"] < half["x"] < half["7"]


@pytest.mark.unit
def test_each_baseline_samples_the_model_through_its_own_dishes_beam():
    npix = 32
    yy, xx = np.mgrid[:npix, :npix]
    # Minus the squared distance (arcsec^2) from the centre, 1 arcsec pixels.
    distmat = -(((xx - npix // 2) ** 2 + (yy - npix // 2) ** 2).astype(float))
    sky = np.zeros((npix, npix))
    sky[npix // 2, npix // 2 + 6] = 1.0  # a point 6 arcsec off the pointing
    wavelength_km = 299_792_458.0 / 650e9 / 1e3
    diameters = np.array([7.0, 7.0, 12.0])
    antnum = np.array([[0, 1], [0, 2], [1, 2]])
    models, which = baseline_beam_models(distmat, [0, 0, wavelength_km], diameters, antnum, sky)
    assert len(models) == 2 and list(which) == [0, 1, 1]
    rows = build_channel_visibility_rows(
        modelfft=models[0],
        gains=np.ones((3, 1)),
        noise_samples=np.zeros((3, 1)),
        u_waves=np.zeros((3, 1)),
        v_waves=np.zeros((3, 1)),
        w_waves=np.zeros((3, 1)),
        antnum=antnum,
        nH=1,
        Nphf=npix // 2,
        uv_pixsize=1.0,
        noise_std=1.0,
        mean_wavelength_m=wavelength_km * 1e3,
        beam_models=models,
        baseline_beam=which,
    )
    radius = 6.0 / 206264.80624709636
    seven = primary_beam_response(np.array(radius), 650e9, 7.0)
    mixed = baseline_beam_response(np.array(radius), 650e9, 7.0, 12.0)
    amplitude = np.abs(rows["model_data"])
    assert amplitude[0] == pytest.approx(float(seven), rel=1e-5)
    assert amplitude[1] == pytest.approx(float(mixed), rel=1e-5)
    assert amplitude[2] == pytest.approx(float(mixed), rel=1e-5)


@pytest.mark.unit
def test_a_baselines_noise_follows_its_two_dishes_effective_areas():
    """Each antenna's SEFD over the array's dish's: a 12 m dish among 7 m ones
    has (7 eta_7)^2 / (12 eta_12)^2 of it, from the effective areas when given."""
    from almasim.services.interferometry.core import Interferometer

    array = object.__new__(Interferometer)
    array.antenna_diameter_m = 7.0
    array.antenna_diameters_m = np.array([7.0, 7.0, 12.0])
    array.dish_areas_m2 = None
    assert array.mixed_dishes
    np.testing.assert_allclose(array._dish_noise(), [1.0, 1.0, (7.0 / 12.0) ** 2])
    array.dish_areas_m2 = {7.0: 20.0, 12.0: 50.0}
    np.testing.assert_allclose(array._dish_noise(), [1.0, 1.0, 0.4])
    array.antenna_diameters_m = np.array([7.0, 7.0])
    assert not array.mixed_dishes
