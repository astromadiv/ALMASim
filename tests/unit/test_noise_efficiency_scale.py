"""Per-band factors on the dishes' Ruze efficiency (MADIV-28): the model's
field, the noise it sets, and the fit from the windows' measured efficiency."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from almasim.services.interferometry import noise
from almasim.services.noise.aggregate import Measured, aggregate
from almasim.services.noise_model import NoiseModel, released_model

from .test_noise_aggregate import _window

ROWS = [
    {"diameter_m": 12.0, "upper_ghz": 116.0, "scale": 0.9, "bands": "2, 3"},
    {"diameter_m": 12.0, "upper_ghz": None, "scale": 1.0, "bands": "4-10"},
    {"diameter_m": 7.0, "upper_ghz": None, "scale": 1.05, "bands": "all"},
]


def _with_scale(rows: list[dict] | None) -> NoiseModel:
    alma = json.loads(released_model("alma", 1).model_dump_json())
    radiometer = {**alma["components"]["radiometer"], "efficiency_scale": rows}
    return NoiseModel.model_validate(
        {**alma, "name": "scaled", "components": {**alma["components"], "radiometer": radiometer}}
    )


@pytest.mark.unit
def test_a_model_scales_each_dishs_efficiency_by_band():
    model = _with_scale(ROWS)
    table = model.noise_config()["efficiency_scale"]
    assert table == {12.0: ((116.0, 0.9), (float("inf"), 1.0)), 7.0: ((float("inf"), 1.05),)}
    assert noise.efficiency_scale(100e9, 12.0, table) == 0.9
    assert noise.efficiency_scale(116e9, 12.0, table) == 1.0  # an edge belongs to the next row
    assert noise.efficiency_scale(345e9, 7.0, table) == 1.05
    assert noise.efficiency_scale(100e9, 15.0, table) == 1.0  # no row for the dish
    assert noise.efficiency_scale(100e9, 12.0, None) == 1.0
    # Unset, the field leaves the model's identity as it is without it.
    assert "efficiency_scale" not in released_model("alma", 1).canonical_json()
    assert _with_scale(None).sha256() == _with_scale(None).sha256()
    assert "efficiency_scale" not in _with_scale(None).canonical_json()
    with pytest.raises(ValidationError, match="12 m dish's edges must rise"):
        _with_scale([dict(ROWS[1]), dict(ROWS[0])])


@pytest.mark.unit
def test_a_scaled_band_is_that_much_noisier():
    """SEFD = 2 k Tsys / (eta A): a factor 0.9 on eta is 1/0.9 on the noise,
    in that band only."""
    scaled, plain = _with_scale(ROWS), released_model("alma", 1)

    def sefd(model, ghz, dish=12.0):
        config = noise.NoiseModelConfig(pwv_mm=1.0, **model.noise_config())
        return noise.system_equivalent_flux_density(
            config, np.array([ghz * 1e9]), 60.0, antenna_diameter_m=dish
        )[0]

    assert sefd(scaled, 100.0) / sefd(plain, 100.0) == pytest.approx(1 / 0.9)
    assert sefd(scaled, 230.0) == pytest.approx(sefd(plain, 230.0))
    assert sefd(scaled, 230.0, 7.0) / sefd(plain, 230.0, 7.0) == pytest.approx(1 / 1.05)


@pytest.mark.unit
def test_aggregate_keeps_each_bands_departure_from_the_curve():
    """Band 3's windows measure 0.9 of the dish's true curve, bands 6 and 7
    follow it. The fitted curve takes up part of that, and the factors the
    rest: the model's efficiency, curve times factor, is each band's measured
    one. Band 4 has two windows, too few for a factor of its own."""
    windows = [Measured(_window(k, ghz=100.0, efficiency_scale=0.9)) for k in range(6)]
    windows += [Measured(_window(10 + k, ghz=230.0)) for k in range(6)]
    windows += [Measured(_window(20 + k, ghz=345.0)) for k in range(6)]
    windows += [Measured(_window(30 + k, ghz=145.0, efficiency_scale=0.8)) for k in range(2)]
    candidate = aggregate(windows, released_model("alma", 1))
    rows = candidate.components.radiometer.efficiency_scale
    assert rows is not None and {r.diameter_m for r in rows} == {12.0}  # no 7 m windows
    scale = {r.bands: r.scale for r in rows}
    assert scale["4"] == 1.0 and scale["1"] == 1.0
    config = noise.NoiseModelConfig(pwv_mm=1.0, **candidate.noise_config())

    def model(ghz):  # the candidate's efficiency, curve times factor
        hz = ghz * 1e9
        return noise.aperture_efficiency(hz, 12.0, config.ruze) * noise.efficiency_scale(
            hz, 12.0, config.efficiency_scale
        )

    def truth(ghz):  # the windows' dish (_window), before their band's factor
        return 0.75 * np.exp(-((4 * np.pi * 26e-6 * ghz * 1e9 / 299_792_458.0) ** 2))

    assert model(100.0) / model(230.0) == pytest.approx(0.9 * truth(100.0) / truth(230.0), rel=0.02)
    assert model(345.0) / model(230.0) == pytest.approx(truth(345.0) / truth(230.0), rel=0.02)
    assert "efficiency_scale[12 m][4]: 2 window(s), kept 1" in candidate.provenance.notes
    assert candidate.provenance.values["radiometer.efficiency_scale[12 m][2, 3]"].n == 6
    # Without any band of three windows, the candidate has no factors.
    few = aggregate([Measured(_window(k)) for k in range(2)], released_model("alma", 1))
    assert few.components.radiometer.efficiency_scale is None
