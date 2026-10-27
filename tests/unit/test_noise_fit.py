"""Fitting a window's calibration residuals (MADIV-28): almasim noise fit.

A bright point source at the phase centre observed through a known per-scan
atmospheric phase (the simulator's own screens) and antenna gain errors, plus
thermal noise; the fit must find them, and none where there are none.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from astropy.io import fits

from almasim.services.interferometry.corruptions import CorruptionModel, draw_antenna_corruptions
from almasim.services.noise import extract, fit
from almasim.services.noise.measurements import (
    Exposure,
    ThermalNoise,
    VisibilityNoise,
    WindowInput,
    WindowNoise,
)

from ._fakes import FakeTable

NANT, NSCAN, PER_SCAN, NCHAN = 16, 24, 8, 8
FLUX, SIGMA = 2.0, 0.6
FREQ = 230e9 + 15.625e6 * np.arange(NCHAN)
LON, LAT = np.radians(-67.755), np.radians(-23.023)


def _itrf(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """ITRF positions of local east-north offsets (m) about the ALMA site."""
    centre = 6_378_000.0 * np.array(
        [np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)]
    )
    e = np.array([-np.sin(LON), np.cos(LON), 0.0])
    n = np.array([-np.sin(LAT) * np.cos(LON), -np.sin(LAT) * np.sin(LON), np.cos(LAT)])
    return centre + east[:, None] * e + north[:, None] * n


def _window_ms(truth: CorruptionModel, seed: int = 0) -> tuple[dict[str, FakeTable], WindowNoise]:
    rng = np.random.default_rng(seed)
    radius = np.geomspace(30, 3000, NANT)
    angle = rng.uniform(0, 2 * np.pi, NANT)
    east, north = radius * np.cos(angle), radius * np.sin(angle)
    xy = np.stack([east, north], axis=1)
    a1, a2 = np.triu_indices(NANT, 1)
    wavelength = 299_792_458.0 / FREQ.mean()
    rows = {k: [] for k in ("DATA", "TIME", "SCAN_NUMBER", "ANTENNA1", "ANTENNA2", "UVW")}
    errors_rng = np.random.default_rng(seed + 1)
    for s in range(NSCAN):
        times = 1e9 + s * 600.0 + 6.0 * np.arange(PER_SCAN)
        errors = draw_antenna_corruptions(truth, xy, times - times[0], errors_rng)
        g = errors.amplitude * np.exp(1j * (errors.phase + 2 * np.pi * errors.path_m / wavelength))
        for t, time in enumerate(times):
            gain = g[a1, t] * np.conj(g[a2, t])
            noise = SIGMA * (
                rng.standard_normal((a1.size, NCHAN, 2))
                + 1j * rng.standard_normal((a1.size, NCHAN, 2))
            )
            rows["DATA"].append(FLUX * gain[:, None, None] + noise)
            rows["TIME"].append(np.full(a1.size, time))
            rows["SCAN_NUMBER"].append(np.full(a1.size, s + 1))
            rows["ANTENNA1"].append(a1)
            rows["ANTENNA2"].append(a2)
            rows["UVW"].append(
                np.stack([east[a2] - east[a1], north[a2] - north[a1], np.zeros(a1.size)], 1)
            )
    columns = {k: np.concatenate(v) for k, v in rows.items()}
    n = len(columns["TIME"])
    columns.update(
        FLAG=np.zeros((n, NCHAN, 2), dtype=bool),
        WEIGHT=np.full((n, 2), 1.0 / SIGMA**2),
        INTERVAL=np.full(n, 6.0),
        DATA_DESC_ID=np.zeros(n, dtype=int),
        FIELD_ID=np.zeros(n, dtype=int),
    )
    tables = {
        "ms": FakeTable(columns),
        "ms/SPECTRAL_WINDOW": FakeTable(
            {
                "CHAN_FREQ": np.array([FREQ]),
                "CHAN_WIDTH": np.array([np.full(NCHAN, 15.625e6)]),
                "EFFECTIVE_BW": np.array([np.full(NCHAN, 15.625e6)]),
            }
        ),
        "ms/DATA_DESCRIPTION": FakeTable(
            {"SPECTRAL_WINDOW_ID": np.array([0]), "POLARIZATION_ID": np.array([0])}
        ),
        "ms/POLARIZATION": FakeTable({"CORR_TYPE": np.array([[9, 12]])}),
        "ms/FIELD": FakeTable({"NAME": np.array(["src"])}),
        "ms/ANTENNA": FakeTable(
            {"POSITION": _itrf(east, north), "DISH_DIAMETER": np.full(NANT, 12.0)}
        ),
    }
    window = WindowNoise(
        name="w",
        almasim_version="test",
        input=WindowInput(ms="ms", spw=0),
        exposure=Exposure(
            antennas=NANT,
            dish_diameter_m=12.0,
            baselines=int(a1.size),
            integrations=NSCAN * PER_SCAN,
            integration_s=6.0,
            time_on_source_s=6.0 * NSCAN * PER_SCAN,
            channels=NCHAN,
            channel_width_hz=15.625e6,
            noise_bandwidth_hz=15.625e6,
            frequency_ghz=230.0,
            correlations=2,
            flagged_fraction=0.0,
        ),  # fmt: skip
        thermal=ThermalNoise(
            scale=1.0, rho1=0.0, rho2=0.0, channel_profile=[1.0] * NCHAN, rows_sampled=n, row_step=1
        ),
        visibility=VisibilityNoise(sigma_jy=SIGMA / np.sqrt(2), unit_jy=1.0, hands_identical=False),
    )
    return tables, window


def _task_dir(tmp_path):
    """An imaging task with a point source at the reference pixel of an odd image."""
    task = tmp_path / "task"
    task.mkdir()
    n = 65
    header = fits.Header()
    header.update(CDELT1=-0.05 / 3600, CDELT2=0.05 / 3600, CRPIX1=n // 2 + 1, CRPIX2=n // 2 + 1)
    model = np.zeros((1, 1, n, n))
    model[0, 0, n // 2, n // 2] = FLUX
    fits.PrimaryHDU(model, header).writeto(task / "wsclean-model.fits")
    image = np.zeros((1, 1, n, n))
    image[0, 0, n // 2 - 2 : n // 2 + 3, n // 2 - 2 : n // 2 + 3] = 1.0
    fits.PrimaryHDU(image, header).writeto(task / "wsclean-image.fits")
    (task / "quality.json").write_text(
        json.dumps({"continuum": {"beam_arcsec": [0.3, 0.3], "noise_jy_beam": 0.01}})
    )
    return task


@pytest.fixture
def window_ms(monkeypatch):
    def install(truth: CorruptionModel, seed: int = 0, root: str = "ms"):
        made, window = _window_ms(truth, seed)
        tables = {root + key[2:]: table for key, table in made.items()}

        def open_table(path, ack=False):
            return tables[str(path)]

        monkeypatch.setattr(extract, "import_casacore_tables", lambda: open_table)
        monkeypatch.setattr(fit, "import_casacore_tables", lambda: open_table)
        return window

    return install


@pytest.mark.unit
def test_the_per_scan_phase_and_gains_are_found(window_ms, tmp_path):
    truth = CorruptionModel(slow_path_um_1km=150.0, slow_slope=0.5, gain_amp_rms=0.1)
    window = window_ms(truth)
    result = fit.fit_window("ms", 0, None, _task_dir(tmp_path), window, realizations=2, seed=3)
    assert result.bright_groups == NSCAN * NANT * (NANT - 1) // 2
    assert result.slow_phase is not None
    assert result.slow_phase.path_um_1km == pytest.approx(150.0, rel=0.4)
    assert 0.25 < result.slow_phase.slope < 0.75
    assert result.gain_amp_rms == pytest.approx(0.1, abs=0.03)
    assert result.baseline_amp_rms is not None and result.baseline_amp_rms < 0.03
    assert result.fast_phase is None or result.fast_phase.path_um_1km < 40
    thermal = [b.thermal_phase_rms_deg for b in result.bins if b.thermal_phase_rms_deg is not None]
    assert thermal and max(thermal) < 5


@pytest.mark.unit
def test_the_in_scan_phase_is_found(window_ms, tmp_path):
    truth = CorruptionModel(fast_path_um_1km=80.0, fast_slope=0.5, fast_coherence_s=10.0)
    window = window_ms(truth, seed=2)
    result = fit.fit_window("ms", 0, None, _task_dir(tmp_path), window, realizations=2, seed=6)
    assert result.fast_phase is not None
    assert result.fast_phase.path_um_1km == pytest.approx(80.0, rel=0.5)


@pytest.mark.unit
def test_thermal_noise_alone_fits_nothing(window_ms, tmp_path):
    window = window_ms(CorruptionModel(), seed=5)
    result = fit.fit_window("ms", 0, None, _task_dir(tmp_path), window, realizations=2, seed=4)
    assert result.slow_phase is None or result.slow_phase.path_um_1km < 20
    assert result.gain_amp_rms is not None and result.gain_amp_rms < 0.02
    assert result.baseline_amp_rms < 0.02 and result.baseline_phase_rms_deg < 1.5


@pytest.mark.unit
def test_a_faint_window_is_left_unfitted(window_ms, tmp_path, monkeypatch):
    window = window_ms(CorruptionModel())
    task = _task_dir(tmp_path)
    with fits.open(task / "wsclean-model.fits", mode="update") as hdul:
        hdul[0].data *= 1e-4
    result = fit.fit_window("ms", 0, None, task, window, realizations=1)
    assert result.slow_phase is None and result.gain_amp_rms is None
    assert any("too faint" in note for note in result.notes)


@pytest.mark.unit
def test_model_planes_are_even_and_centred():
    sky = np.zeros((1, 5, 5))
    sky[0, 2, 2] = 1.0
    even = fit._even(sky, {"CRPIX1": 3, "CRPIX2": 3})
    assert even.shape == (1, 6, 6) and even[0, 3, 3] == 1.0
    assert fit._even(np.zeros((1, 4, 4)), {"CRPIX1": 3, "CRPIX2": 3}).shape == (1, 4, 4)
    with pytest.raises(ValueError, match="not at its centre"):
        fit._even(np.zeros((1, 5, 5)), {"CRPIX1": 1, "CRPIX2": 3})


@pytest.mark.unit
def test_noise_fit_runs_after_extract_and_resumes(window_ms, tmp_path):
    from typer.testing import CliRunner

    from almasim.cli.noise import noise_app

    ms = str(tmp_path / "w.ms")
    window_ms(CorruptionModel(gain_amp_rms=0.1), root=ms)
    task, out = _task_dir(tmp_path), tmp_path / "out"
    runner = CliRunner()
    early = runner.invoke(
        noise_app, ["fit", ms, "--spw", "0", "--task-dir", str(task), "--out", str(out)]
    )
    assert early.exit_code == 1
    failures = json.loads((out / "failures-fit.json").read_text())
    assert "run noise extract first" in next(iter(failures.values()))
    extracted = runner.invoke(noise_app, ["extract", ms, "--spw", "0", "--out", str(out)])
    assert extracted.exit_code == 0, extracted.output
    args = [
        "fit",
        ms,
        "--spw",
        "0",
        "--task-dir",
        str(task),
        "--out",
        str(out),
        "--realizations",
        "1",
    ]
    fitted = runner.invoke(noise_app, args)
    assert fitted.exit_code == 0, fitted.output
    result = json.loads((out / "windows" / "w_spw0" / "fit.json").read_text())
    assert result["gain_amp_rms"] == pytest.approx(0.1, abs=0.04)
    again = runner.invoke(noise_app, args)
    assert "1 window(s) already measured; 0 to go" in again.output
    for bad, message in (
        (["fit", ms, "--spw", "0", "--out", str(out)], "--task-dir are needed"),
        (
            ["fit", ms, "--task-dir", str(task), "--out", str(out), "--line-model", "most"],
            "--line-model",
        ),
    ):
        result = runner.invoke(noise_app, bad)
        assert result.exit_code == 2 and message in result.output, result.output
