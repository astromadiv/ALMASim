"""Reducing windows' measurements into a candidate noise model (MADIV-28)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from almasim.services.interferometry.atmosphere import system_temperature
from almasim.services.noise.aggregate import Measured, aggregate
from almasim.services.noise.measurements import (
    AntennaShape,
    Exposure,
    FitBin,
    PhaseFit,
    ThermalNoise,
    Tsys,
    VisibilityNoise,
    WindowFit,
    WindowInput,
    WindowNoise,
)
from almasim.services.noise_model import released_model, resolve_noise_model

C = 299_792_458.0
NCHAN = 256


def _window(
    k: int, *, ghz: float = 230.0, dish: float = 12.0, trx: float = 40.0, ripple: float = 0.02
) -> WindowNoise:
    rng = np.random.default_rng(k)
    x = np.linspace(-1, 1, NCHAN)
    tsys_shape = 1 + 0.05 * x  # Tsys^2 slope, as SYSCAL shows it
    receiver = 1 + ripple * np.sqrt(2) * np.cos(np.pi * (x + rng.uniform(0, 1)))  # rms ~ ripple
    profile = tsys_shape * receiver**2 + rng.normal(0, 0.002, NCHAN)
    width = 7.8125e6
    freq = ghz * 1e9 + np.linspace(-0.5, 0.5, 64) * NCHAN * width
    elevation, pwv = 60.0, 1.2
    airmass = 1 / np.sin(np.radians(elevation))
    tsys = float(np.median(system_temperature(freq, trx, pwv_mm=pwv, airmass=airmass)))
    eta0, eps = (0.75, 26e-6) if dish == 12 else (0.7, 24e-6)
    aperture = eta0 * np.exp(-((4 * np.pi * eps * ghz * 1e9 / C) ** 2)) * rng.normal(1, 0.02)
    return WindowNoise(
        name=f"w{k}",
        almasim_version="test",
        input=WindowInput(ms=f"w{k}.ms", spw=1, pwv_mm=pwv),
        exposure=Exposure(
            antennas=40,
            dish_diameter_m=dish,
            baselines=780,
            integrations=500,
            integration_s=6.0,
            time_on_source_s=3000.0,
            channels=NCHAN,
            channel_width_hz=width,
            noise_bandwidth_hz=width,
            frequency_ghz=ghz,
            correlations=2,
            flagged_fraction=0.01,
            elevation_deg=elevation,
        ),  # fmt: skip
        thermal=ThermalNoise(
            scale=1.0,
            rho1=0.667 + rng.normal(0, 0.005),
            rho2=0.166 + rng.normal(0, 0.005),
            channel_profile=list(profile),
            rows_sampled=10**6,
            row_step=1,
        ),  # fmt: skip
        visibility=VisibilityNoise(
            sigma_jy=1.0, unit_jy=1.0, hands_identical=False, efficiency=aperture * 0.96 * 0.88
        ),
        tsys=Tsys(median_k=tsys, profile=list(tsys_shape / tsys_shape.mean()), spectral_window=3),
        antenna_shape=AntennaShape(
            antennas_used=40,
            channels_used=NCHAN - 3,
            excess_rms=0.03 + 0.001 * k,
            half_width_channels=10,
            half_width_mhz=100.0 + k,
        ),  # fmt: skip
    )


def _fit(k: int) -> WindowFit:
    return WindowFit(
        name=f"w{k}",
        bright_groups=500,
        realizations=2,
        kappa=2.67,
        bins=[
            FitBin(
                baseline_m=100.0,
                groups=50,
                phase_rms_deg=5.0,
                thermal_phase_rms_deg=1.0,
                lnamp_mean=-0.01,
                thermal_lnamp_mean=0.0,
            )
        ],
        slow_phase=PhaseFit(
            bins=6, slope=0.4 + 0.01 * k, phase_1km_deg=20.0, path_um_1km=100.0 + 10 * k
        ),
        gain_amp_rms=0.05 + 0.01 * k,
        baseline_amp_rms=0.04,
        baseline_phase_rms_deg=3.0,
    )


@pytest.mark.unit
def test_the_candidate_takes_the_windows_medians():
    windows = [Measured(_window(k), _fit(k) if k < 4 else None) for k in range(8)]
    windows += [Measured(_window(10 + k, dish=7.0, ghz=100.0 + 60 * k)) for k in range(6)]
    windows += [Measured(_window(20 + k, ghz=100.0 + 60 * k)) for k in range(6)]
    base = released_model("alma", 1)
    candidate = aggregate(windows, base, data="/runs/x")
    assert (candidate.name, candidate.version) == ("alma-candidate", 2)
    c = candidate.components
    assert c.channel_correlation.rho1 == pytest.approx(0.667, abs=0.01)
    assert c.receiver_ripple.rms == pytest.approx(0.02, rel=0.35)
    assert c.antenna_ripple.rms == pytest.approx(
        np.median(
            [0.03 + 0.001 * k for k in range(8)]
            + [0.03 + 0.001 * k for k in list(range(10, 16)) + list(range(20, 26))]
        )
    )
    assert c.slow_phase.path_um_1km == pytest.approx(115.0)
    assert c.antenna_gains.amp_rms == pytest.approx(0.065)
    assert c.baseline_errors.phase_rms_deg == pytest.approx(3.0)
    # The receiver temperature the Tsys implies, in the 12 m windows' band (6).
    trx = {r.bands: r.trx_k for r in c.radiometer.receiver_temperature_k}
    assert trx["6"] == pytest.approx(40.0, rel=0.05)
    assert trx["9, 10"] == 160.0  # no windows there: the base's
    aperture = {a.diameter_m: a for a in c.radiometer.aperture}
    assert aperture[12.0].eta0 == pytest.approx(0.75, abs=0.03)
    assert aperture[12.0].surface_rms_m == pytest.approx(26e-6, rel=0.2)
    assert aperture[7.0].eta0 == pytest.approx(0.7, abs=0.04)
    values = candidate.provenance.values
    assert values["slow_phase.path_um_1km"].n == 4 and values["channel_correlation.rho1"].n == 20
    assert candidate.provenance.windows == 20 and candidate.provenance.data == "/runs/x"
    assert "fast_phase.path_um_1km: 0 window(s)" in candidate.provenance.notes
    assert c.fast_phase == base.components.fast_phase
    # The candidate is a model like any other: it validates and simulates.
    assert resolve_noise_model(candidate.model_dump_json()).engine_fields()[
        "gain_amp_rms"
    ] == pytest.approx(0.065)


@pytest.mark.unit
def test_too_few_windows_keep_the_base():
    base = released_model("alma")
    candidate = aggregate([Measured(_window(0))], base, name="trial", version=7)
    assert candidate.label() == "trial@7"
    assert candidate.components == base.components
    assert "kept the base's" in candidate.provenance.notes


@pytest.mark.unit
def test_noise_aggregate_writes_the_candidate(tmp_path):
    from almasim.cli.noise import noise_app

    out = tmp_path / "run"
    for k in range(5):
        d = out / "windows" / f"w{k}"
        d.mkdir(parents=True)
        (d / "window.json").write_text(_window(k).model_dump_json())
        (d / "fit.json").write_text(_fit(k).model_dump_json())
    runner = CliRunner()
    result = runner.invoke(noise_app, ["aggregate", "--out", str(out), "--from", "alma@1"])
    assert result.exit_code == 0, result.output
    assert "5 window(s), 5 fitted -> alma-candidate@2" in result.output
    assert "antenna_gains.amp_rms: 0.07" in result.output
    candidate = resolve_noise_model(str(out / "candidate.json"))
    assert candidate.components.antenna_gains.amp_rms == pytest.approx(0.07)
    empty = runner.invoke(noise_app, ["aggregate", "--out", str(tmp_path / "none")])
    assert empty.exit_code == 2 and "run noise extract first" in empty.output
    bad = runner.invoke(noise_app, ["aggregate", "--out", str(out), "--from", "stormy"])
    assert bad.exit_code == 2 and "--from" in bad.output
    assert json.loads((out / "candidate.json").read_text())["provenance"]["windows"] == 5
