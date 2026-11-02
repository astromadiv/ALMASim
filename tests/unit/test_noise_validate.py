"""Validating a noise model against real windows (MADIV-28): image statistics,
comparison and the validate/compare commands (simulation and imaging stubbed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from typer.testing import CliRunner

from almasim.services.noise import validate as v
from almasim.services.noise.batch import WindowTask
from almasim.services.noise.compare import Criteria, Pair, compare
from almasim.services.noise.image_stats import image_noise_stats
from almasim.services.noise.measurements import (
    Exposure,
    ThermalNoise,
    VisibilityNoise,
    WindowInput,
    WindowNoise,
)
from almasim.services.noise_model import released_model

from ._fakes import FakeTable


def _header(n: int) -> fits.Header:
    header = fits.Header()
    header.update(CDELT1=-0.1 / 3600, CDELT2=0.1 / 3600, BMAJ=0.5 / 3600, BMIN=0.4 / 3600)
    return header


def _task(path: Path, sigma: float, *, seed: int = 0, rho: float = 0.3, nchan: int = 12) -> Path:
    """An imaging task whose residuals are white noise of ``sigma`` (continuum)
    and channels correlated by ``rho`` with their neighbours (cube)."""
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)
    n = 96
    fits.PrimaryHDU(rng.normal(0, sigma, (1, 1, n, n)).astype(np.float32), _header(n)).writeto(
        path / "wsclean-residual.fits"
    )
    white = rng.normal(0, 1, (nchan + 1, n, n))
    a = (1 - np.sqrt(1 - 4 * rho**2)) / (2 * rho) if rho else 0.0  # a / (1 + a^2) = rho
    cube = (white[1:] + a * white[:-1]) / np.sqrt(1 + a**2) * 5 * sigma
    fits.PrimaryHDU(cube[None].astype(np.float32), _header(n)).writeto(
        path / "wsclean-cube-residual.fits"
    )
    quality = {"continuum": {"snr": 30.0, "peak_jy_beam": 0.03}, "cube": {}}
    (path / "quality.json").write_text(json.dumps(quality))
    return path


def _window(name: str, unit: float = 2000.0, scale: float = 1.0, ghz: float = 230.0) -> WindowNoise:
    return WindowNoise(
        name=name,
        almasim_version="test",
        input=WindowInput(ms=f"/d/{name}.ms", spw=0),
        exposure=Exposure(
            antennas=40,
            dish_diameter_m=12.0,
            baselines=780,
            integrations=100,
            integration_s=6.0,
            time_on_source_s=600.0,
            channels=64,
            channel_width_hz=15.6e6,
            noise_bandwidth_hz=41.7e6,
            frequency_ghz=ghz,
            correlations=2,
            flagged_fraction=0.0,
        ),  # fmt: skip
        thermal=ThermalNoise(
            scale=scale,
            rho1=0.66,
            rho2=0.16,
            channel_profile=[1.0] * 64,
            rows_sampled=10,
            row_step=1,
        ),
        visibility=VisibilityNoise(sigma_jy=1.0, unit_jy=unit, hands_identical=False),
    )


@pytest.mark.unit
def test_image_statistics_measure_level_shape_and_correlation(tmp_path):
    stats = image_noise_stats(_task(tmp_path / "t", 1e-3, rho=0.3))
    assert stats["cont.sigma"] == pytest.approx(1e-3, rel=0.05)
    # 7,000 pixels: the MAD sigma is good to ~2%, the fourth moment to ~0.3.
    assert abs(stats["cont.kurtosis"]) < 0.4 and stats["cont.tail5"] < 1e-3
    assert stats["cube.sigma_median"] == pytest.approx(5e-3, rel=0.05)
    assert stats["cube.adjacent_corr"] == pytest.approx(0.3, abs=0.05)
    assert stats["cube.channels"] == 12 and stats["q.continuum.snr"] == 30.0
    assert stats["cont.corr_area_beams"] < 0.5  # white noise: narrower than a beam
    assert image_noise_stats(tmp_path / "missing") == {}


def _pairs(ratio: float, n: int = 8) -> list[Pair]:
    rng = np.random.default_rng(1)
    pairs = []
    for k in range(n):
        real = 1e-3 * (1 + k)
        jitter = rng.normal(1, 0.01)
        pairs.append(
            Pair(
                name=f"w{k}",
                sim_image={
                    "cont.sigma": real * ratio * jitter,
                    "cube.sigma_median": 5 * real * ratio,
                    "cont.kurtosis": 0.0,
                },
                real_image={
                    "cont.sigma": real,
                    "cube.sigma_median": 5 * real,
                    "cont.kurtosis": 0.05,
                },
                sim_window=_window(f"w{k}", unit=2000 * ratio * jitter, scale=1.0 + 0.002 * k),
                real_window=_window(f"w{k}", scale=1.3),
                band="band 6 12 m" if k < 5 else "band 3 12 m",
            )
        )
    return pairs


@pytest.mark.unit
def test_a_matching_model_passes_and_a_quiet_one_does_not():
    good = compare(_pairs(1.0), {"label": "good"})
    assert good.passed, [c for c in good.checks if not c.passed]
    assert good.metrics["cont.sigma"].median == pytest.approx(1.0, abs=0.02)
    assert good.metrics["cont.sigma"].within_10pct == 8
    assert good.metrics["cont.kurtosis"].kind == "difference"
    assert good.metrics["cont.kurtosis"].median == pytest.approx(-0.05)
    assert set(good.groups) == {"band 6 12 m", "band 3 12 m"}
    quiet = compare(_pairs(0.6), {"label": "quiet"})
    assert not quiet.passed
    failed = {c.name for c in quiet.checks if not c.passed}
    assert (
        "cont.sigma median within 5%" in failed and "cont.sigma of band 6 12 m within 10%" in failed
    )
    assert quiet.metrics["cont.sigma"].p is not None and quiet.metrics["cont.sigma"].p < 0.01
    loose = compare(
        _pairs(0.6),
        {"label": "quiet"},
        Criteria(noise_tolerance=0.5, group_tolerance=0.5, checked=()),
    )
    assert loose.passed


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """noise validate with the simulation, extraction and imaging stubbed:
    the simulated window's noise is ``ratio`` times the real one's."""
    ratio = {"value": 1.0}

    def simulate(row, window, model, run, *, seed, peak_jy=None):
        ratio["peak"] = peak_jy
        run.mkdir(parents=True, exist_ok=True)
        (run / "sim.ms").mkdir()
        return run / "sim.ms"

    def measure(ms, name, run):
        return _window(name, unit=2000 * ratio["value"])

    def image(run, imaging):
        return _task(run / "cubes" / "sim" / "SPW-0_FIELD-0", 1e-3 * ratio["value"], seed=7)

    monkeypatch.setattr(v, "simulate", simulate)
    monkeypatch.setattr(v, "measure", measure)
    monkeypatch.setattr(v, "image", image)
    measured = tmp_path / "measured"
    rows = []
    with open(tmp_path / "windows.csv", "w") as f:
        f.write("ms,spw,name,task_dir\n")
        for k in range(6):
            name = f"w{k}"
            task = _task(tmp_path / "real" / name, 1e-3, seed=k)
            f.write(f"/d/{name}.ms,0,{name},{task}\n")
            (measured / "windows" / name).mkdir(parents=True)
            (measured / "windows" / name / "window.json").write_text(
                _window(name).model_dump_json()
            )
            rows.append({"name": name, "Freq": 230.0})
    import pandas as pd

    pd.DataFrame(rows).to_csv(tmp_path / "rows.csv", index=False)
    return tmp_path, ratio


@pytest.mark.unit
def test_noise_validate_simulates_measures_and_judges(stubbed):
    from almasim.cli.noise import noise_app

    root, ratio = stubbed
    args = [
        "validate",
        "--model",
        "alma@1",
        "--windows",
        str(root / "windows.csv"),
        "--metadata",
        str(root / "rows.csv"),
        "--measured",
        str(root / "measured"),
        "--out",
        str(root / "out"),
    ]
    runner = CliRunner()
    result = runner.invoke(noise_app, args)
    assert result.exit_code == 0, result.output
    assert "PASSED" in result.output and "NOT PASSED" not in result.output
    model = released_model("alma", 1)
    base = v.validation_dir(root / "out", model)
    comparison = json.loads((base / "comparison.json").read_text())
    assert comparison["windows"] == 6 and comparison["model"]["label"] == "alma@1"
    assert ratio["peak"] == pytest.approx(0.03)  # the real image's continuum peak
    assert not (base / "windows" / "w0" / "sim" / "sim.ms").exists()  # dropped once measured
    assert json.loads((base / "model.json").read_text())["name"] == "alma"
    # Again: nothing left to simulate; compare alone judges the same.
    again = runner.invoke(noise_app, args)
    assert "0 to go" in again.output or "PASSED" in again.output
    compared = runner.invoke(
        noise_app,
        ["compare", "--model", "alma@1", "--windows", str(root / "windows.csv"),
         "--measured", str(root / "measured"), "--out", str(root / "out")],
    )  # fmt: skip
    assert compared.exit_code == 0 and "6 window(s) compared" in compared.output
    # A quieter simulation, validated as another model, does not pass.
    ratio["value"] = 0.6
    candidate = root / "quiet.json"
    candidate.write_text(model.model_copy(update={"name": "quiet", "version": 9}).model_dump_json())
    quiet = runner.invoke(noise_app, [*args[:2], str(candidate), *args[3:]])
    assert quiet.exit_code == 0 and "NOT PASSED" in quiet.output
    failing = runner.invoke(
        noise_app,
        ["compare", "--model", str(candidate), "--windows", str(root / "windows.csv"),
         "--measured", str(root / "measured"), "--out", str(root / "out")],
    )  # fmt: skip
    assert failing.exit_code == 1


@pytest.mark.unit
def test_noise_validate_checks_its_inputs_and_fans_out(stubbed, monkeypatch):
    from types import SimpleNamespace

    import almasim.services.compute as compute
    from almasim.cli.noise import noise_app

    root, _ = stubbed
    runner = CliRunner()
    base = [
        "validate",
        "--windows",
        str(root / "windows.csv"),
        "--metadata",
        str(root / "rows.csv"),
        "--measured",
        str(root / "measured"),
        "--out",
        str(root / "out"),
    ]
    for extra, message in (
        (["--model", "stormy"], "not a released model"),
        (["--model", "none"], "a model, not none"),
        (["--model", "alma", "--name", "w99"], "no window 'w99'"),
    ):
        result = runner.invoke(noise_app, [*base, *extra])
        assert result.exit_code == 2 and message in result.output, result.output
    (root / "rows.csv").write_text("name,Freq\nw0,230\n")
    result = runner.invoke(noise_app, [*base, "--model", "alma"])
    assert result.exit_code == 2 and "no row for w1" in result.output

    commands = []

    class Cluster:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit_subcommand(self, *, command, cores, timeout):
            commands.append(command)
            return SimpleNamespace(result=lambda: SimpleNamespace(returncode=0, stderr=""))

    monkeypatch.setattr(compute, "create_backend", lambda kind, **kwargs: Cluster())
    rows = "name,Freq\n" + "".join(f"w{k},230\n" for k in range(6))
    (root / "rows.csv").write_text(rows)
    result = runner.invoke(
        noise_app, [*base, "--model", "alma", "--backend", "slurm", "--wsclean-bin", "/w"]
    )
    assert len(commands) == 6, result.output
    assert commands[0][-17:-1] == [] or "--no-judge" in commands[0]
    assert "--name" in commands[0] and "--wsclean-bin" in commands[0]


@pytest.mark.unit
def test_imaging_arguments():
    imaging = v.Imaging(
        wsclean_bin="/bin/wsclean", casa_data_root="/casa", num_cores=4, extra=("--multiscale",)
    )
    args = imaging.arguments()
    assert args[:4] == ["--backend", "sync", "--cube", "--uv-contsub"]
    assert ["--wsclean-bin", "/bin/wsclean"] == args[
        args.index("--wsclean-bin") : args.index("--wsclean-bin") + 2
    ]
    assert args[-1] == "--multiscale"
    task = WindowTask(ms="/d/a.ms", spw=1, name="a")
    assert not v.done(task, Path("/nonexistent"), released_model("alma"))


@pytest.mark.unit
def test_a_window_is_simulated_from_its_row_with_the_model(monkeypatch, tmp_path):
    """One execution block, the window's frequency and channels, the model."""
    import pandas as pd

    import almasim.services.simulation as simulation

    seen = {}

    def clean_cube(params, logger=None, compute_backend=None):
        seen["params"] = params
        return "clean"

    monkeypatch.setattr(simulation, "generate_clean_cube", clean_cube)
    monkeypatch.setattr(
        simulation, "simulate_observation", lambda clean, compute_backend=None: "results"
    )
    monkeypatch.setattr(
        simulation,
        "export_results",
        lambda params, clean, results, logger=None: {"ms_path": str(tmp_path / "run" / "x.ms")},
    )
    repo = Path(__file__).resolve().parents[2]
    row = pd.read_csv(repo / "data" / "qso_metadata.csv").iloc[0]
    window = _window("w", ghz=float(row["Freq"]) + 0.5)
    model = released_model("alma")
    ms = v.simulate(row, window, model, tmp_path / "run", seed=5)
    params = seen["params"]
    assert ms == tmp_path / "run" / "x.ms"
    assert params.noise_model == model and params.executions == 1 and params.seed == 5
    assert params.freq == pytest.approx(window.exposure.frequency_ghz)
    assert params.n_channels == 64 and params.ms_export
    assert params.freq_support == row["Freq.sup."]  # no resolution measured: the row's
    assert params.int_time == pytest.approx(600.0) and params.n_executions == 1


@pytest.mark.unit
def test_a_window_is_simulated_with_its_own_channel_averaging(monkeypatch, tmp_path):
    """The row's support lists the observing unit's windows; the window's own
    MS says how its channels are averaged (window 61: the row's averaged
    channels, rho1 0.33, where the real ones are native, rho1 0.69)."""
    import astropy.units as U
    import pandas as pd

    import almasim.services.simulation as simulation
    from almasim.services.instrument.channels import channels_from_resolution
    from almasim.services.interferometry.frequency import freq_supp_extractor

    seen = {}

    def clean_cube(params, logger=None, compute_backend=None):
        seen["params"] = params
        return "clean"

    monkeypatch.setattr(simulation, "generate_clean_cube", clean_cube)
    monkeypatch.setattr(simulation, "simulate_observation", lambda clean, compute_backend=None: 0)
    monkeypatch.setattr(
        simulation,
        "export_results",
        lambda params, clean, results, logger=None: {"ms_path": str(tmp_path / "x.ms")},
    )
    repo = Path(__file__).resolve().parents[2]
    row = pd.read_csv(repo / "data" / "qso_metadata.csv").iloc[0]
    window = _window("w", ghz=float(row["Freq"]) + 0.5)
    native = window.model_copy(
        update={
            "exposure": window.exposure.model_copy(
                update={"channel_width_hz": 976562.5, "resolution_hz": 1953125.0}
            )
        }
    )

    def averaging(support: str) -> tuple[int, float]:
        band, _, _, resolution = freq_supp_extractor(support, native.exposure.frequency_ghz * U.GHz)
        channels = channels_from_resolution(resolution.to(U.Hz).value, band.to(U.Hz).value)
        return channels.averaged, channels.width_hz

    assert averaging(row["Freq.sup."])[0] >= 4  # the row's: averaged
    v.simulate(row, native, released_model("alma"), tmp_path / "run", seed=5)
    support = seen["params"].freq_support
    assert support == v.frequency_support(native)
    assert averaging(support) == (1, 976562.5)  # the window's own: native
    band, centre, _, _ = freq_supp_extractor(support, native.exposure.frequency_ghz * U.GHz)
    assert centre.value == pytest.approx(native.exposure.frequency_ghz)
    assert band.to(U.Hz).value == pytest.approx(64 * 976562.5)
    assert v.frequency_support(window) is None


@pytest.mark.unit
def test_the_simulated_source_is_as_bright_as_the_real_continuum():
    """The archive row's sensitivities make the source 100-1000 times the
    continuum's noise; scaled to the real image's peak, its cleaning residuals
    are those of the real field."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Run:
        skymodel: np.ndarray

    @dataclass
    class Stage:
        model_cube: np.ndarray
        continuum_fraction: np.ndarray | None
        interferometer_runs: list

    cube = np.zeros((4, 8, 8))
    cube[:, 4, 4] = [1.0, 1.0, 3.0, 1.0]  # a line in channel 2
    stage = Stage(cube, np.array([1.0, 1.0, 1 / 3, 1.0]), [Run(cube), Run(cube)])
    scaled = v.with_continuum(stage, 0.002)
    assert np.mean(np.sum(scaled.model_cube, axis=(1, 2)) * stage.continuum_fraction) == (
        pytest.approx(0.002)
    )
    assert all(run.skymodel is scaled.model_cube for run in scaled.interferometer_runs)
    assert stage.model_cube[2, 4, 4] == 3.0  # the stage itself untouched
    assert scaled.model_cube[2, 4, 4] / scaled.model_cube[0, 4, 4] == pytest.approx(3.0)
    for peak in (0.0, -1.0):
        assert v.with_continuum(stage, peak) is stage
    assert v.with_continuum(Stage(np.zeros_like(cube), None, []), 0.002).model_cube.max() == 0


@pytest.mark.unit
def test_a_simulated_window_is_measured_in_a_process_of_its_own(monkeypatch, tmp_path):
    """casatools wrote the MS; python-casacore reads it in another process."""
    from types import SimpleNamespace

    calls = []

    def run(command, capture_output, text):
        calls.append(command)
        out = Path(command[command.index("--out") + 1])
        (out / "windows" / "w").mkdir(parents=True)
        (out / "windows" / "w" / "window.json").write_text(_window("w").model_dump_json())
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(v.subprocess, "run", run)
    window = v.measure(tmp_path / "sim.ms", "w", tmp_path)
    assert window.name == "w"
    (command,) = calls
    assert command[command.index("noise") + 1 :][:4] == [
        "extract",
        str(tmp_path / "sim.ms"),
        "--spw",
        "0",
    ]
    monkeypatch.setattr(
        v.subprocess,
        "run",
        lambda command, capture_output, text: SimpleNamespace(
            returncode=134, stderr="free(): invalid pointer", stdout=""
        ),
    )
    with pytest.raises(RuntimeError, match="invalid pointer"):
        v.measure(tmp_path / "sim.ms", "w", tmp_path / "other")


@pytest.mark.unit
def test_windows_validated_at_once_write_through_their_own_temporary_files(tmp_path):
    from almasim.services.noise.batch import _write_atomic

    target = tmp_path / "model.json"
    _write_atomic(target, "a")
    _write_atomic(target, "b")
    assert target.read_text() == "b"
    assert [p.name for p in tmp_path.iterdir()] == ["model.json"]


@pytest.mark.unit
def test_a_simulated_window_is_imaged_as_the_real_ones(monkeypatch, tmp_path):
    from types import SimpleNamespace

    calls = []

    def run(command, capture_output, text):
        calls.append(command)
        if "image-from-ms" in command:
            (tmp_path / "cubes" / "x" / "SPW-0_FIELD-0").mkdir(parents=True)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(v.subprocess, "run", run)
    task = v.image(tmp_path, v.Imaging(wsclean_bin="/w"))
    assert task == tmp_path / "cubes" / "x" / "SPW-0_FIELD-0"
    assert [c[c.index("image") + 1] for c in calls] == [
        "compute-parameters",
        "image-from-ms",
        "clean-quality",
    ]
    assert "--uv-contsub" in calls[1] and "/w" in calls[1]

    def failing(command, capture_output, text):
        return SimpleNamespace(returncode=1, stderr="wsclean: out of memory", stdout="")

    monkeypatch.setattr(v.subprocess, "run", failing)
    with pytest.raises(RuntimeError, match="out of memory"):
        v.image(tmp_path / "other", v.Imaging())
    monkeypatch.setattr(
        v.subprocess,
        "run",
        lambda command, capture_output, text: SimpleNamespace(returncode=0, stderr="", stdout=""),
    )
    with pytest.raises(RuntimeError, match="no imaging task"):
        v.image(tmp_path / "empty", v.Imaging())
    with pytest.raises(ValueError, match="name column"):
        (tmp_path / "rows.csv").write_text("Freq\n230\n")
        v.metadata_rows(tmp_path / "rows.csv")


@pytest.mark.unit
def test_a_window_is_simulated_through_its_own_water_vapour():
    """The window's own PWV; without one, the PWV at which the model's Tsys is
    ALMA's; with neither, the row's."""
    import pandas as pd

    from almasim.services.interferometry import noise
    from almasim.services.interferometry.atmosphere import system_temperature
    from almasim.services.noise.measurements import Tsys

    model = released_model("alma")
    row = pd.Series({"PWV": 5.186})
    window = _window("w")
    measured = window.model_copy(update={"input": WindowInput(ms="/d/w.ms", spw=0, pwv_mm=0.7)})
    assert v.with_weather(row, measured, model)["PWV"] == 0.7
    assert v.with_weather(row, window, model)["PWV"] == 5.186  # no Tsys to go by
    assert row["PWV"] == 5.186, "the row itself is left alone"

    e = window.exposure
    freq = e.frequency_ghz * 1e9 + np.linspace(-0.5, 0.5, 64) * e.channels * e.channel_width_hz
    config = noise.NoiseModelConfig(pwv_mm=1.0, **model.noise_config())
    trx = noise.estimate_receiver_temperature_k(freq, "alma", config.receiver_temperatures)
    airmass = 1 / np.sin(np.radians(55.0))

    def with_tsys(kelvin):
        return window.model_copy(
            update={
                "exposure": e.model_copy(update={"elevation_deg": 55.0}),
                "tsys": Tsys(median_k=kelvin, profile=[1.0] * 64, spectral_window=1),
            }
        )

    at_1_3 = float(np.median(system_temperature(freq, trx, pwv_mm=1.3, airmass=airmass)))
    assert v.with_weather(row, with_tsys(at_1_3), model)["PWV"] == pytest.approx(1.3, abs=0.01)
    assert v.pwv_from_tsys(with_tsys(1.0), model) == 0.0  # below the driest sky


@pytest.mark.unit
def test_a_window_is_observed_as_long_and_by_as_many_antennas_as_its_ms():
    """The row's Int.Time and antennas are the observing unit's; the MS is one block."""
    import pandas as pd

    pads = [f"A{i:03d}:DA{i}" for i in range(50)]
    row = pd.Series({"Int.Time": 3000.0, "n_executions": 5, "antenna_arrays": " ".join(pads)})
    window = _window("w")  # 600 s by 40 antennas
    record = v.with_exposure(row, window, seed=3)
    antennas = record["antenna_arrays"].split()
    assert record["Int.Time"] == 600.0 and record["n_executions"] == 1
    assert len(antennas) == 40 and set(antennas) <= set(pads)
    assert antennas == v.with_exposure(row, window, seed=3)["antenna_arrays"].split()
    assert row["Int.Time"] == 3000.0, "the row itself is left alone"

    # One antenna per pad; a unit with no more antennas than the MS keeps them all.
    shared = pd.Series({"Int.Time": 600.0, "antenna_arrays": "A1:DA41 A1:DA42 A2:DV01 A3:DV02"})
    assert v.with_exposure(shared, window, seed=3)["antenna_arrays"] == "A1:DA41 A2:DV01 A3:DV02"


@pytest.mark.unit
def test_a_window_with_its_pads_is_observed_by_the_mss_own_antennas():
    """The row's antennas span the unit's configurations and its total-power
    antennas; the MS's pads are the ones it observed with."""
    import pandas as pd

    union = " ".join([f"A{i:03d}:DA{i}" for i in range(50)] + ["T701:PM01"])
    row = pd.Series({"Int.Time": 3000.0, "antenna_arrays": union})
    window = _window("w")
    own = ["A001:DA41", "W210:DV07", "P402:DA50"]
    window = window.model_copy(
        update={"exposure": window.exposure.model_copy(update={"pads": own})}
    )
    assert v.with_exposure(row, window, seed=3)["antenna_arrays"] == " ".join(own)


def _lossy(window: WindowNoise, flagged: float, high_weight: float) -> WindowNoise:
    e = window.exposure.model_copy(
        update={"flagged_fraction": flagged, "high_weight_fraction": high_weight}
    )
    return window.model_copy(update={"exposure": e})


@pytest.mark.unit
def test_a_simulated_window_misses_as_much_data_as_the_real_image():
    assert v.unimaged_fraction(_window("w0")) == 0.0
    assert v.unimaged_fraction(_lossy(_window("w0"), 0.1, 0.2)) == pytest.approx(0.3)
    assert v.unimaged_fraction(_lossy(_window("w0"), 0.6, 0.5)) == v.MAX_UNIMAGED


def _flaggable(ncross: int = 1000, nauto: int = 10, nchan: int = 8) -> FakeTable:
    """A simulated MS's flags: a tenth of its cross-correlation rows flagged,
    the next 50 rows half flagged, and autocorrelations after them."""
    flag = np.zeros((ncross + nauto, nchan, 2), dtype=bool)
    flag[:100] = True
    flag[100:150, : nchan // 2] = True
    return FakeTable(
        {
            "ANTENNA1": np.r_[np.zeros(ncross, int), np.arange(nauto)],
            "ANTENNA2": np.r_[np.ones(ncross, int), np.arange(nauto)],
            "FLAG": flag,
            "FLAG_ROW": np.zeros(ncross + nauto, dtype=bool),
        }
    )


@pytest.mark.unit
def test_rows_are_flagged_until_the_share_is_reached(monkeypatch):
    from almasim.services.imaging import ms_geometry

    tables = [_flaggable(), _flaggable()]
    opened = iter(tables)
    monkeypatch.setattr(ms_geometry, "import_casacore_tables", lambda: lambda *a, **k: next(opened))
    monkeypatch.setattr(v, "FLAG_CHUNK_ROWS", 64)
    # 0.4 of 1000 rows of data: 100 + 25 flagged already, 275 rows more.
    assert v.flag_rows("sim.ms", 0.4, seed=3) == {"flagged_rows": 275}
    flag, flag_row = tables[0].columns["FLAG"], tables[0].columns["FLAG_ROW"]
    assert flag[:1000].mean() == pytest.approx(0.4)
    assert flag_row[:1000].sum() == 275 and not flag_row[1000:].any()
    assert not flag[1000:].any()  # autocorrelations untouched
    assert not flag[100:150, 4:].any()  # rows with flags already are not drawn
    v.flag_rows("sim.ms", 0.4, seed=3)
    assert (tables[1].columns["FLAG_ROW"] == flag_row).all()  # the seed draws the rows


@pytest.mark.unit
def test_the_simulated_ms_is_flagged_in_a_process_of_its_own(monkeypatch, tmp_path):
    from almasim.services.runtime import child

    calls = []

    def call_in_child(target, kwargs, **options):
        calls.append((target, kwargs))
        return {"flagged_rows": 12}

    monkeypatch.setattr(child, "call_in_child", call_in_child)
    real = _window("w0")
    assert v.flag_as_imaged(tmp_path / "sim.ms", real, seed=5) == 0 and calls == []
    assert v.flag_as_imaged(tmp_path / "sim.ms", _lossy(real, 0.0, 0.25), seed=5) == 12
    assert calls == [
        (
            "almasim.services.noise.validate:flag_rows",
            {"ms_path": str(tmp_path / "sim.ms"), "fraction": 0.25, "seed": 5},
        )
    ]


@pytest.mark.unit
def test_each_simulated_window_is_flagged_as_its_real_image(stubbed, monkeypatch):
    from almasim.cli.noise import noise_app

    root, _ = stubbed
    flagged = []
    monkeypatch.setattr(
        v, "flag_as_imaged", lambda ms, window, *, seed: flagged.append((ms.name, window.name))
    )
    args = ["validate", "--model", "alma@1", "--windows", str(root / "windows.csv"),
            "--metadata", str(root / "rows.csv"), "--measured", str(root / "measured"),
            "--out", str(root / "out")]  # fmt: skip
    result = CliRunner().invoke(noise_app, args)
    assert result.exit_code == 0, result.output
    assert sorted(flagged) == [("sim.ms", f"w{k}") for k in range(6)]
