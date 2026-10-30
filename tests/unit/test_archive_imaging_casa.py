"""casacore and CASA steps of the imaging task, run against stand-in modules.

CI has neither python-casacore nor casatools; these fakes record what the
code asks of them, so the MS reading and writing and the CASA calls are
exercised exactly as on the cluster.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from astropy.io import fits

from almasim.cli import image as cli_image
from almasim.services.casa import environment as casa_environment
from almasim.services.casa import runtime_data as casa_runtime_data
from almasim.services.imaging import continuum as imaging_continuum
from almasim.services.imaging import cube as imaging_cube
from almasim.services.imaging import markers as imaging_markers
from almasim.services.imaging import products as imaging_products
from almasim.services.imaging import task as imaging_task
from almasim.services.imaging import uv_contsub as imaging_uv_contsub
from almasim.services.imaging.ms import chunks as chunks_module
from almasim.services.imaging.ms import chunks as imaging_ms_chunks
from almasim.services.imaging.ms import single_window as imaging_ms_single_window
from almasim.services.imaging.selfcal import casa_ops as imaging_selfcal_casa_ops
from almasim.services.imaging.selfcal import loop as imaging_selfcal_loop
from almasim.services.imaging.selfcal import noise_products as imaging_selfcal_noise_products
from almasim.services.imaging.wsclean import command as imaging_wsclean_command
from almasim.services.imaging.wsclean import runner as imaging_wsclean_runner

from ._fakes import FakeTable, FakeTask, patch_everywhere


@pytest.fixture
def fake_casacore(monkeypatch):
    """``casacore.tables`` serving the tables registered in the returned dict by path."""
    tables: dict[str, FakeTable] = {}
    package = ModuleType("casacore")
    module = ModuleType("casacore.tables")
    module.table = lambda path, **kwargs: tables[str(path)]
    module.makecoldesc = lambda name, desc: {"name": name, **desc}
    package.tables = module
    monkeypatch.setitem(sys.modules, "casacore", package)
    monkeypatch.setitem(sys.modules, "casacore.tables", module)
    return tables


def _visibilities(n_rows=40, n_channels=64, line=slice(30, 36), seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, n_channels)[None, :, None]
    continuum = (2.0 + 0.3 * x) * np.ones((n_rows, 1, 2))
    data = (continuum + 0.01 * rng.normal(size=(n_rows, n_channels, 2))).astype(np.complex64)
    data[:, line] += 1.5
    return data


def _ms(data, *, spectrum_weight=False, corrected=False):
    n_rows = data.shape[0]
    a1, a2 = np.triu_indices(5, 1)
    columns = {
        "DATA": data,
        "FLAG": np.zeros(data.shape, bool),
        "WEIGHT": np.ones((n_rows, data.shape[2])),
        "FLAG_ROW": np.zeros(n_rows, bool),
        "ANTENNA1": np.resize(a1, n_rows),
        "ANTENNA2": np.resize(a2, n_rows),
    }
    if spectrum_weight:
        columns["WEIGHT_SPECTRUM"] = np.ones(data.shape)
    if corrected:
        columns["CORRECTED_DATA"] = data.copy()
    return FakeTable(columns)


@pytest.mark.unit
@pytest.mark.parametrize("spectrum_weight", [False, True])
def test_source_spectrum_averages_every_row(spectrum_weight):
    data = _visibilities()
    ms = _ms(data, spectrum_weight=spectrum_weight)
    ms.columns["FLAG"][:, 5] = True  # a flagged channel

    spectrum, usable = imaging_ms_chunks._source_spectrum(ms, "DATA", rows_per_chunk=7)

    assert not usable[5] and usable.sum() == 63
    np.testing.assert_allclose(spectrum[usable], np.abs(data.mean(axis=(0, 2)))[usable], rtol=1e-5)
    assert imaging_ms_chunks._source_spectrum(FakeTable({"DATA": data[:0]}), "DATA", 7) is None


@pytest.mark.unit
@pytest.mark.parametrize(("spectrum_weight", "corrected"), [(False, True), (True, False)])
def test_subtract_uv_continuum_writes_continuum_and_line_columns(
    fake_casacore, spectrum_weight, corrected
):
    data = _visibilities()
    fake_casacore["x.ms"] = ms = _ms(data, spectrum_weight=spectrum_weight, corrected=corrected)
    ms.columns["FLAG"][:, 5] = True  # a flagged channel

    result = imaging_uv_contsub.subtract_uv_continuum("x.ms", rows_per_chunk=9)

    assert result["source_column"] == ("CORRECTED_DATA" if corrected else "DATA")
    assert not result.get("skipped") and ms.closed and result["usable_channels"] == 63
    ranges = [tuple(map(int, r.split("~"))) for r in result["line_ranges"].split(";")]
    assert all(any(a <= c <= b for a, b in ranges) for c in range(30, 36))
    line = ms.columns[imaging_products.LINE_COLUMN]
    assert np.abs(line[:, 30:36].real - 1.5).max() < 0.05, "the line stays in LINE_DATA"
    assert np.abs(line[:, :20]).max() < 0.05, "no continuum left"
    np.testing.assert_allclose(
        ms.columns[imaging_products.CONTINUUM_COLUMN] + line, data, atol=1e-5
    )

    # A rerun replaces the columns it made.
    assert not imaging_uv_contsub.subtract_uv_continuum("x.ms").get("skipped")


@pytest.mark.unit
def test_subtract_uv_continuum_skips_without_rows_or_line_free_channels(fake_casacore):
    fake_casacore["empty.ms"] = _ms(_visibilities()[:0])
    assert imaging_uv_contsub.subtract_uv_continuum("empty.ms")["skipped"] == "the MS has no rows"

    fake_casacore["lines.ms"] = ms = _ms(_visibilities())
    result = imaging_uv_contsub.subtract_uv_continuum("lines.ms", min_line_free_fraction=0.99)
    assert "usable channels are line-free" in result["skipped"]
    assert imaging_products.LINE_COLUMN not in ms.columns


@pytest.mark.unit
def test_line_free_channels_with_too_few_channels_to_fit():
    line_free = imaging_continuum.line_free_channels(
        np.array([1.0, 2.0]), np.array([True, False]), order=1
    )
    assert list(line_free) == [True, False], "one usable channel cannot be fitted or clipped"


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


@pytest.mark.unit
def test_subtract_uv_continuum_subprocess_returns_the_child_result(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(args=json.loads(cmd[-1]), env=kwargs["env"])
        return _Proc(stdout='{"line_channels": 3}\n', stderr="casacore note")

    monkeypatch.setattr(subprocess, "run", fake_run)
    log = tmp_path / "task.log"
    result = imaging_uv_contsub.subtract_uv_continuum_subprocess("x.ms", order=0, log_path=log)

    assert result == {"line_channels": 3} and seen["args"] == {
        "target": "almasim.services.imaging.uv_contsub:subtract_uv_continuum",
        "kwargs": {"ms_path": "x.ms", "order": 0},
    }
    assert seen["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert log.read_text() == "casacore note\n"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (_Proc(returncode=1, stderr="Traceback\nRuntimeError: lock\n"), "return code 1: .*lock"),
        (_Proc(stdout=""), "wrote no result"),
        (subprocess.TimeoutExpired("python", 5), "killed after exceeding the 5s"),
    ],
)
def test_subtract_uv_continuum_subprocess_failures(monkeypatch, outcome, message):
    def fake_run(cmd, **kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match=message):
        imaging_uv_contsub.subtract_uv_continuum_subprocess("x.ms", timeout=5)


@pytest.mark.unit
def test_drop_corrected_data(fake_casacore):
    fake_casacore["c.ms"] = ms = _ms(_visibilities(), corrected=True)
    assert imaging_uv_contsub.drop_corrected_data("c.ms") == {"dropped": True}
    assert "CORRECTED_DATA" not in ms.columns
    assert imaging_uv_contsub.drop_corrected_data("c.ms") == {"dropped": False}


@pytest.mark.unit
def test_selfcal_prepare_selects_line_free_channels_and_the_busiest_antenna(fake_casacore):
    fake_casacore["p.ms"] = ms = _ms(_visibilities())
    ms.columns["FLAG_ROW"][ms.columns["ANTENNA1"] == 0] = True  # antenna 0 mostly flagged
    fake_casacore["p.ms/ANTENNA"] = FakeTable(
        {"NAME": np.array(["DA41", "DA42", "DV01", "DV02", "PM03"])}
    )

    prepared = imaging_selfcal_casa_ops.selfcal_prepare("p.ms")

    assert prepared["spw"].startswith("0:0~") and "~63" in prepared["spw"]
    assert prepared["line_free_channels"] < 64 and prepared["refant"] != "DA41"
    with pytest.raises(RuntimeError, match="no rows"):
        fake_casacore["e.ms"] = _ms(_visibilities()[:0])
        imaging_selfcal_casa_ops.selfcal_prepare("e.ms")


@pytest.fixture
def fake_casa(monkeypatch):
    """``casatasks`` and ``casatools`` recording the calls, and a caltable to read back."""
    calls: dict[str, list] = {"gaincal": [], "applycal": [], "configure": []}
    tasks = ModuleType("casatasks")
    tasks.gaincal = lambda **kwargs: calls["gaincal"].append(kwargs)
    tasks.applycal = lambda **kwargs: calls["applycal"].append(kwargs)
    tools = ModuleType("casatools")

    class _CalTable:
        def open(self, path):
            self.path = path

        def getcol(self, name):
            gains = np.array([[[np.exp(0.2j), 1.1 * np.exp(-0.1j), 1.0]]])
            return {
                "FLAG": np.array([[[False, False, True]]]),
                "CPARAM": gains,
                "TIME": np.array([1.0, 1.0, 2.0]),
            }[name]

        def close(self):
            pass

    tools.table = _CalTable
    monkeypatch.setitem(sys.modules, "casatasks", tasks)
    monkeypatch.setitem(sys.modules, "casatools", tools)
    monkeypatch.setattr(
        casa_environment,
        "configure_casa_environment",
        lambda root, data, **kwargs: calls["configure"].append((data, kwargs["log_file"])),
    )
    return calls


@pytest.mark.unit
def test_casa_selfcal_solve_phase_and_amplitude(fake_casa):
    phase = imaging_selfcal_casa_ops.casa_selfcal_solve(
        "x.ms", "x.G", solint="inf", spw="0:0~9", refant="DA41", casa_data="/d", log_file="c.log"
    )
    (call,) = fake_casa["gaincal"]
    assert call["calmode"] == "p" and call["gaintype"] == "T" and not call["solnorm"]
    assert call["spw"] == "0:0~9" and call["gaintable"] == [] and call["interp"] == []
    assert fake_casa["configure"] == [("/d", "c.log")]
    assert phase["solutions"] == 2 and phase["possible"] == 3 and phase["intervals"] == 2
    assert phase["median_phase_deg"] == pytest.approx(8.6, abs=0.1)

    amp = imaging_selfcal_casa_ops.casa_selfcal_solve(
        "x.ms",
        "x.A",
        solint="inf",
        spw="0:0~9",
        refant="DA41",
        casa_data="/d",
        log_file="c.log",
        calmode="ap",
        pre_apply=["x.G"],
    )
    call = fake_casa["gaincal"][-1]
    assert call["calmode"] == "ap" and call["solnorm"] and call["gaintable"] == ["x.G"]
    assert call["interp"] == ["linear"] and amp["median_amp_dev"] == pytest.approx(0.05)


@pytest.mark.unit
def test_casa_selfcal_apply_both_tables(fake_casa):
    assert imaging_selfcal_casa_ops.casa_selfcal_apply(
        "x.ms", ["x.G", "x.A"], casa_data="/d", log_file="c.log"
    ) == {"applied": True}
    (call,) = fake_casa["applycal"]
    assert call["gaintable"] == ["x.G", "x.A"] and call["calwt"] == [False, False]
    assert call["applymode"] == "calonly" and call["interp"] == ["linear", "linear"]


def _fits(path: Path, data: np.ndarray) -> Path:
    fits.PrimaryHDU(data.astype(np.float32)).writeto(path)
    return path


@pytest.mark.unit
def test_image_snr_and_peak(tmp_path):
    rng = np.random.default_rng(4)
    residual = rng.normal(0, 0.1, (1, 1, 64, 64))
    image = residual.copy()
    image[0, 0, 32, 32] = 5.0
    image_path = _fits(tmp_path / "i.fits", image)
    snr = imaging_selfcal_loop.image_snr(image_path, _fits(tmp_path / "r.fits", residual))
    assert snr == pytest.approx(50, rel=0.1)
    assert imaging_selfcal_loop.image_peak(image_path) == pytest.approx(5.0)
    assert (
        imaging_selfcal_loop.image_snr(image_path, _fits(tmp_path / "z.fits", np.zeros((4, 4))))
        == 0.0
    )


@pytest.mark.unit
def test_call_in_child_runs_a_function_of_the_module(tmp_path):
    image = _fits(tmp_path / "i.fits", np.full((4, 4), 2.5))
    assert (
        imaging_selfcal_loop._call_in_child("image_peak", {"image_path": str(image)}, what="peak")
        == 2.5
    )
    assert (
        imaging_selfcal_loop._call_in_child(
            "image_peak", {"image_path": str(image)}, what="p", casa=True
        )
        == 2.5
    )


@pytest.mark.unit
def test_call_in_child_reports_failures_and_timeouts(tmp_path):
    log = tmp_path / "task.log"
    with pytest.raises(RuntimeError, match="peak exited with return code 1: .*FileNotFound"):
        imaging_selfcal_loop._call_in_child(
            "image_peak", {"image_path": str(tmp_path / "missing.fits")}, what="peak", log_path=log
        )
    assert "Traceback" in log.read_text(), "the child's stderr goes to the task log"
    with pytest.raises(RuntimeError, match="peak killed after exceeding"):
        imaging_selfcal_loop._call_in_child(
            "image_peak", {"image_path": "x"}, what="peak", timeout=0.001
        )


@pytest.mark.unit
def test_call_in_child_needs_a_result(monkeypatch):
    class _Proc:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    with pytest.raises(RuntimeError, match="wrote no result"):
        imaging_selfcal_loop._call_in_child("image_peak", {}, what="peak")


@pytest.mark.unit
def test_self_calibrate_turns_a_failed_step_into_a_task_failure(tmp_path, monkeypatch):
    patch_everywhere(monkeypatch, imaging_selfcal_loop, "image_snr", lambda image, residual: 50.0)
    patch_everywhere(monkeypatch, imaging_selfcal_loop, "image_peak", lambda image: 1.0)

    def broken(function, kwargs, **_):
        raise RuntimeError("gaincal exploded")

    patch_everywhere(monkeypatch, imaging_selfcal_loop, "_call_in_child", broken)
    with (tmp_path / "task.log").open("w") as log, pytest.raises(ValueError, match="exploded"):
        imaging_selfcal_loop.self_calibrate(
            tmp_path / "x.ms",
            ["wsclean", "-name", str(tmp_path / "w"), "ms"],
            task=FakeTask(tmp_path, run=lambda cmd: 1, fail=ValueError),
            log=log,
            solints=["inf"],
            min_snr=20,
            casa_data="/d",
        )


class _Stop(Exception):
    """Ends a task once the step under test has run."""


def _wsclean_task(tmp_path, monkeypatch, **kwargs):
    patch_everywhere(
        monkeypatch,
        imaging_ms_single_window,
        "extract_single_window_ms_subprocess",
        lambda *a, **k: {"rows": 1, "bytes": 0},
    )
    out = tmp_path / "out" / "SPW-25"
    return imaging_task.run_wsclean_task(
        command=[
            "wsclean",
            "-name",
            str(out / "wsclean"),
            imaging_products.SINGLE_WINDOW_PLACEHOLDER,
        ],
        cube_command=[
            "wsclean",
            "-name",
            str(out / "wsclean-cube"),
            imaging_products.SINGLE_WINDOW_PLACEHOLDER,
        ],
        output_dir=str(out),
        ms_path=str(tmp_path / "x.ms"),
        spw=25,
        threads=2,
        single_window=True,
        **kwargs,
    )


@pytest.mark.unit
def test_run_wsclean_task_self_calibrates_before_imaging(tmp_path, monkeypatch):
    seen = {}
    patch_everywhere(
        monkeypatch,
        imaging_wsclean_runner,
        "_run_with_retries",
        lambda cmd, *args: seen.setdefault("ran", cmd),
    )

    def fake_self_calibrate(ms, command, **kwargs):
        seen.update(kwargs, ms=ms, command=command)
        kwargs["task"].run(["wsclean", "-name", "probe"], kwargs["log"])
        raise _Stop

    patch_everywhere(monkeypatch, imaging_selfcal_loop, "self_calibrate", fake_self_calibrate)
    with pytest.raises(_Stop):
        _wsclean_task(
            tmp_path,
            monkeypatch,
            selfcal_solints=["inf"],
            casa_data="/casa",
            continuum_restart=True,
        )

    assert Path(seen["ms"]).name == "spw25.single.ms" and seen["casa_data"] == "/casa"
    assert seen["solints"] == ["inf"] and seen["command"][-1].endswith("spw25.single.ms")
    assert seen["ran"] == ["wsclean", "-name", "probe"], "rounds image through the task runner"
    assert seen["restart"] is True and "-update-model-required" in seen["command"]
    assert seen["keep_gains"] is False, "noise products (and gains) are off unless asked for"


@pytest.mark.unit
def test_run_wsclean_task_selfcal_needs_casa_data(tmp_path):
    with pytest.raises(ValueError, match="casa_data"):
        imaging_task.run_wsclean_task(
            command=["wsclean"],
            output_dir=str(tmp_path / "x"),
            ms_path="x",
            spw=1,
            threads=1,
            single_window=True,
            selfcal_solints=["inf"],
        )
    assert not (tmp_path / "x").exists(), "a rejected call must not create the output directory"


@pytest.mark.unit
def test_run_wsclean_task_fails_when_the_uv_subtraction_fails(tmp_path, monkeypatch):
    def extract(ms, spw, fields, out_path, **kwargs):
        (Path(out_path) / "table.dat").parent.mkdir(parents=True)
        (Path(out_path) / "table.dat").write_text("x")
        return {"rows": 1, "bytes": 0}

    patch_everywhere(
        monkeypatch, imaging_ms_single_window, "extract_single_window_ms_subprocess", extract
    )

    def broken(*args, **kwargs):
        raise RuntimeError("casacore lock")

    patch_everywhere(monkeypatch, imaging_uv_contsub, "subtract_uv_continuum_subprocess", broken)
    out = tmp_path / "out" / "SPW-25"
    with pytest.raises(RuntimeError, match="Could not subtract the continuum.*casacore lock"):
        imaging_task.run_wsclean_task(
            command=[
                "wsclean",
                "-name",
                str(out / "wsclean"),
                imaging_products.SINGLE_WINDOW_PLACEHOLDER,
            ],
            cube_command=[
                "wsclean",
                "-name",
                str(out / "wsclean-cube"),
                imaging_products.SINGLE_WINDOW_PLACEHOLDER,
            ],
            output_dir=str(out),
            ms_path=str(tmp_path / "x.ms"),
            spw=25,
            threads=2,
            single_window=True,
            uv_contsub_order=1,
        )
    marker = json.loads(imaging_markers.imaging_failure_marker_path(out).read_text())
    assert "casacore lock" in marker["error"]
    assert not list(out.glob("spw*.single.ms")), "the single-window copy is removed"


@pytest.mark.unit
def test_prepare_selfcal_casa_data_populates_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        casa_environment,
        "configure_casa_environment",
        lambda root, data, **kwargs: calls.append(("configure", Path(data))),
    )
    monkeypatch.setattr(
        casa_runtime_data,
        "ensure_casa_runtime_data",
        lambda data, **kwargs: calls.append(("ensure", Path(data))),
    )

    default = cli_image._prepare_selfcal_casa_data(tmp_path / "out", None)
    chosen = cli_image._prepare_selfcal_casa_data(tmp_path / "out", tmp_path / "casa")

    assert default == tmp_path / "out" / ".casa-data" and chosen == tmp_path / "casa"
    assert calls == [
        ("configure", default),
        ("ensure", default),
        ("configure", chosen),
        ("ensure", chosen),
    ]


@pytest.mark.unit
def test_subtract_uv_continuum_ignores_rows_flagged_by_flag_row(fake_casacore):
    data = _visibilities()
    data[::5, 10:20] += 100.0  # garbage in rows the observatory flagged as a whole
    fake_casacore["x.ms"] = ms = _ms(data)
    ms.columns["FLAG_ROW"] = np.zeros(data.shape[0], bool)
    ms.columns["FLAG_ROW"][::5] = True

    result = imaging_uv_contsub.subtract_uv_continuum("x.ms")

    ranges = [tuple(map(int, r.split("~"))) for r in result["line_ranges"].split(";")]
    assert not any(a <= 15 <= b for a, b in ranges), "flagged rows never make a line"
    assert any(a <= 32 <= b for a, b in ranges), "the real line is still found"


@pytest.mark.unit
def test_subtract_uv_continuum_reads_chunks_sized_by_samples(fake_casacore, monkeypatch):
    fake_casacore["x.ms"] = ms = _ms(_visibilities(n_rows=40, n_channels=64))
    reads = []
    getcol = ms.getcol
    monkeypatch.setattr(
        ms,
        "getcol",
        lambda name, startrow=0, nrow=-1: reads.append(nrow) or getcol(name, startrow, nrow),
    )
    monkeypatch.setattr(chunks_module, "_CHUNK_SAMPLES", 64 * 2 * 10)  # ten rows of 64 channels x 2

    imaging_uv_contsub.subtract_uv_continuum("x.ms")

    assert max(reads) == 10


@pytest.mark.unit
def test_rows_for_chunk():
    assert imaging_ms_chunks.rows_for_chunk(3840, 2) == 2604
    assert imaging_ms_chunks.rows_for_chunk(10**9, 4) == 1, "at least one row"


@pytest.mark.unit
def test_adopt_corrected_data_makes_the_incoming_calibration_data(fake_casacore):
    data = _visibilities()
    calibrated = data * 2  # calibrated values differ from DATA
    fake_casacore["c.ms"] = ms = _ms(data.copy(), corrected=True)
    ms.columns["CORRECTED_DATA"] = calibrated.copy()

    assert imaging_selfcal_casa_ops.adopt_corrected_data("c.ms", rows_per_chunk=7) == {
        "adopted": True
    }
    assert "CORRECTED_DATA" not in ms.columns
    np.testing.assert_array_equal(ms.columns["DATA"], calibrated)
    assert imaging_selfcal_casa_ops.adopt_corrected_data("c.ms") == {"adopted": False}


@pytest.mark.unit
def test_selfcal_prepare_skips_antennas_whose_samples_are_all_flagged(fake_casacore):
    fake_casacore["p.ms"] = ms = _ms(_visibilities())
    busiest = np.bincount(np.r_[ms.columns["ANTENNA1"], ms.columns["ANTENNA2"]]).argmax()
    rows = (ms.columns["ANTENNA1"] == busiest) | (ms.columns["ANTENNA2"] == busiest)
    ms.columns["FLAG"][rows] = True  # FLAG_ROW stays False: only the samples are flagged
    names = np.array(["DA41", "DA42", "DV01", "DV02", "PM03"])
    fake_casacore["p.ms/ANTENNA"] = FakeTable({"NAME": names})

    assert imaging_selfcal_casa_ops.selfcal_prepare("p.ms")["refant"] != names[busiest]


@pytest.mark.unit
def test_self_calibrate_logs_an_adopted_calibration(tmp_path, monkeypatch):
    def fake_call(function, kwargs, **_):
        return {"adopted": True} if function == "adopt_corrected_data" else {}

    patch_everywhere(monkeypatch, imaging_selfcal_loop, "_call_in_child", fake_call)
    patch_everywhere(monkeypatch, imaging_selfcal_loop, "image_snr", lambda image, residual: 5.0)
    patch_everywhere(monkeypatch, imaging_selfcal_loop, "image_peak", lambda image: 1.0)
    with (tmp_path / "task.log").open("w") as log:
        imaging_selfcal_loop.self_calibrate(
            tmp_path / "x.ms",
            ["wsclean", "-name", str(tmp_path / "w"), "ms"],
            task=FakeTask(tmp_path, run=lambda cmd: 1, fail=RuntimeError),
            log=log,
            solints=["inf"],
            min_snr=20,
            casa_data="/d",
        )
    assert "incoming CORRECTED_DATA adopted as DATA" in (tmp_path / "task.log").read_text()


@pytest.mark.unit
def test_default_rows_per_chunk_follows_the_cell_shape(monkeypatch):
    monkeypatch.setattr(chunks_module, "_CHUNK_SAMPLES", 640)
    assert imaging_ms_chunks._default_rows_per_chunk(_ms(_visibilities(n_channels=64)), "DATA") == 5
    assert imaging_ms_chunks._default_rows_per_chunk(_ms(_visibilities()[:0]), "DATA") == 640


@pytest.mark.unit
def test_selfcal_correction_columns_whole(fake_casacore):
    data = _visibilities()
    fake_casacore["s.ms"] = ms = _ms(data.copy(), corrected=True)
    ms.columns["CORRECTED_DATA"] = (data * 0.9).astype(data.dtype)

    result = imaging_selfcal_noise_products.selfcal_correction_columns("s.ms", rows_per_chunk=7)

    assert result["columns"] == [imaging_products.SELFCAL_COLUMN]
    np.testing.assert_allclose(
        ms.columns[imaging_products.SELFCAL_COLUMN], data * 0.1, rtol=1e-4, atol=1e-6
    )
    assert ms.closed


@pytest.mark.unit
def test_selfcal_correction_columns_split_like_the_data(fake_casacore):
    data = _visibilities()
    corrected = (data * 0.9).astype(data.dtype)
    fake_casacore["s.ms"] = ms = _ms(data.copy(), corrected=True)
    ms.columns["CORRECTED_DATA"] = corrected.copy()

    imaging_selfcal_noise_products.selfcal_correction_columns("s.ms", line_ranges="27~39", order=1)

    continuum = ms.columns[imaging_products.SELFCAL_CONTINUUM_COLUMN]
    line = ms.columns[imaging_products.SELFCAL_LINE_COLUMN]
    np.testing.assert_allclose(continuum + line, data - corrected, atol=1e-5)
    line_free = imaging_continuum._line_free_from_ranges("27~39", data.shape[1])
    expected = imaging_continuum.fit_uv_continuum(
        data - corrected, np.ones(data.shape), np.zeros(data.shape, bool), line_free, 1
    )
    np.testing.assert_allclose(continuum, expected, atol=1e-5)
    imaging_selfcal_noise_products.selfcal_correction_columns(
        "s.ms", line_ranges="", order=0
    )  # replaces the columns
    assert imaging_products.SELFCAL_LINE_COLUMN in ms.columns


@pytest.mark.unit
def test_selfcal_correction_columns_need_corrected_data(fake_casacore):
    fake_casacore["n.ms"] = _ms(_visibilities())
    with pytest.raises(RuntimeError, match="no CORRECTED_DATA"):
        imaging_selfcal_noise_products.selfcal_correction_columns("n.ms")


@pytest.mark.unit
def test_line_free_from_ranges():
    assert imaging_continuum._line_free_from_ranges("1~2;5~5", 7).tolist() == [
        True,
        False,
        False,
        True,
        True,
        False,
        True,
    ]
    assert imaging_continuum._line_free_from_ranges("", 3).all()


@pytest.mark.unit
def test_export_gain_table(fake_casacore, tmp_path):
    gains = np.array([[[np.exp(0.5j) * 1.2]], [[np.exp(-0.25j)]]])
    fake_casacore["g.cal"] = FakeTable(
        {
            "TIME": np.array([100.0, 160.0]),
            "ANTENNA1": np.array([0, 7]),
            "CPARAM": gains,
            "FLAG": np.array([[[False]], [[True]]]),
        }
    )
    fake_casacore["g.cal/ANTENNA"] = FakeTable({"NAME": np.array(["DA41", "DA42"])})

    assert imaging_selfcal_casa_ops.export_gain_table("g.cal", str(tmp_path / "g.csv")) == {
        "rows": 2
    }
    lines = (tmp_path / "g.csv").read_text().splitlines()
    assert lines[0] == "time_mjd_s,antenna,polarization,amplitude,phase_deg,flagged"
    assert lines[1] == "100.000,DA41,0,1.2,28.65,False"
    assert lines[2] == "160.000,7,0,1,-14.32,True", "an antenna id beyond the table"


@pytest.mark.unit
def test_dirty_command():
    cmd = [
        "wsclean", "-name", "/o/wsclean", "-j", "4", "-size", "256", "256", "-niter", "100000",
        "-make-psf", "-auto-mask", "3.0", "-auto-threshold", "1.0", "-update-model-required",
        "-continue", "-fits-mask", "m.fits", "-data-column", "CONTINUUM_DATA", "x.ms",
    ]  # fmt: skip
    assert imaging_wsclean_command._dirty_command(cmd, Path("/out"), "corr", "SELFCAL_DATA") == [
        "wsclean", "-name", "/out/corr", "-j", "4", "-size", "256", "256",
        "-niter", "0", "-no-update-model-required", "-data-column", "SELFCAL_DATA", "x.ms",
    ]  # fmt: skip


@pytest.mark.unit
def test_add_images(tmp_path):
    a, b = tmp_path / "a.fits", tmp_path / "b.fits"
    beams = fits.BinTableHDU.from_columns(
        [fits.Column("BMAJ", "1E", array=[1.0, 2.0])], name="BEAMS"
    )
    fits.HDUList([fits.PrimaryHDU(np.ones((1, 2, 3, 3), np.float32)), beams]).writeto(a)
    fits.PrimaryHDU(np.full((1, 2, 3, 3), 0.5, np.float32)).writeto(b)

    imaging_selfcal_noise_products.add_images(a, b, tmp_path / "sum.fits", "sum")

    with fits.open(tmp_path / "sum.fits") as hdus:
        assert np.allclose(hdus[0].data, 1.5)
        assert hdus["BEAMS"].data["BMAJ"].tolist() == [1.0, 2.0]
        assert "sum" in str(hdus[0].header["HISTORY"])
    fits.PrimaryHDU(np.zeros((1, 1, 3, 3), np.float32)).writeto(tmp_path / "c.fits")
    with pytest.raises(RuntimeError, match="differ"):
        imaging_selfcal_noise_products.add_images(
            a, tmp_path / "c.fits", tmp_path / "bad.fits", "x"
        )


def _noise_run(tmp_path, monkeypatch, *, fit, single_channel=False, write=True):
    out = tmp_path / "SPW-1"
    out.mkdir(parents=True)
    fits.PrimaryHDU(np.ones((1, 1, 4, 4), np.float32)).writeto(out / "wsclean-residual.fits")
    fits.PrimaryHDU(np.ones((1, 3, 4, 4), np.float32)).writeto(out / "wsclean-cube-residual.fits")
    children, runs = [], []
    patch_everywhere(
        monkeypatch,
        imaging_selfcal_loop,
        "_call_in_child",
        lambda f, kw, **_: children.append((f, kw)) or {},
    )

    def stack(out_dir, prefix, keep, provenance=None):
        if single_channel:
            return {}
        (out_dir / f"{prefix}-0000-dirty.fits").rename(out_dir / f"{prefix}-dirty.fits")
        return {"dirty": 3}

    patch_everywhere(monkeypatch, imaging_cube, "stack_channel_images", stack)

    def run(cmd):
        runs.append(cmd)
        name = Path(cmd[cmd.index("-name") + 1]).name
        if not write:
            return 1
        if "cube" in name and not single_channel:
            plane, target = (1, 3, 4, 4), out / f"{name}-0000-dirty.fits"
        else:
            plane, target = (
                (1, 1, 4, 4) if "cube" not in name else (1, 3, 4, 4),
                out / f"{name}-dirty.fits",
            )
        fits.PrimaryHDU(np.full(plane, 0.25, np.float32)).writeto(target)
        fits.PrimaryHDU(np.ones(plane, np.float32)).writeto(out / f"{name}-psf.fits")
        return 1

    with (tmp_path / "log").open("w") as log:
        written = imaging_selfcal_loop._selfcal_noise_products(
            Path("x.ms"),
            continuum_command=["wsclean", "-name", "w", "-niter", "9", "x.ms"],
            cube_command=["wsclean", "-name", "c", "-channels-out", "3", "x.ms"],
            fit=fit,
            task=FakeTask(out, run=run, fail=RuntimeError),
            log=log,
        )
    return out, written, children, runs, (tmp_path / "log").read_text()


@pytest.mark.unit
def test_selfcal_noise_products_with_continuum_subtraction(tmp_path, monkeypatch):
    out, written, children, runs, log = _noise_run(
        tmp_path, monkeypatch, fit={"line_ranges": "1~1", "order": 1}
    )
    assert children == [
        ("selfcal_correction_columns", {"ms_path": "x.ms", "line_ranges": "1~1", "order": 1})
    ]
    assert [r[r.index("-data-column") + 1] for r in runs] == [
        imaging_products.SELFCAL_CONTINUUM_COLUMN,
        imaging_products.SELFCAL_LINE_COLUMN,
    ]
    assert written == [
        imaging_products.SELFCAL_CORRECTION_FILENAME,
        imaging_products.TOTAL_RESIDUAL_FILENAME,
        imaging_products.CUBE_SELFCAL_CORRECTION_FILENAME,
        imaging_products.CUBE_TOTAL_RESIDUAL_FILENAME,
    ]
    assert np.allclose(fits.getdata(out / imaging_products.TOTAL_RESIDUAL_FILENAME), 1.25)
    assert np.allclose(fits.getdata(out / imaging_products.CUBE_TOTAL_RESIDUAL_FILENAME), 1.25)
    assert not list(out.glob("*-selfcal-correction-*.fits")), "WSClean's own files removed"
    assert "noise products: wsclean-selfcal-correction.fits" in log


@pytest.mark.unit
def test_selfcal_noise_products_without_subtraction_single_channel(tmp_path, monkeypatch):
    out, written, children, runs, _ = _noise_run(
        tmp_path, monkeypatch, fit=None, single_channel=True
    )
    assert children[0][1]["line_ranges"] is None
    assert {r[r.index("-data-column") + 1] for r in runs} == {imaging_products.SELFCAL_COLUMN}
    assert imaging_products.CUBE_TOTAL_RESIDUAL_FILENAME in written


@pytest.mark.unit
def test_selfcal_noise_products_failures(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="wrote no wsclean-selfcal-correction"):
        _noise_run(tmp_path, monkeypatch, fit=None, write=False)

    def broken(*args, **kwargs):
        raise RuntimeError("casacore lock")

    patch_everywhere(monkeypatch, imaging_selfcal_loop, "_call_in_child", broken)
    with (
        (tmp_path / "l").open("w") as log,
        pytest.raises(RuntimeError, match="Could not write the self-calibration correction"),
    ):
        imaging_selfcal_loop._selfcal_noise_products(
            Path("x.ms"),
            continuum_command=None,
            cube_command=None,
            fit=None,
            task=FakeTask(tmp_path, run=lambda c: 1, fail=RuntimeError),
            log=log,
        )


@pytest.mark.unit
def test_run_wsclean_task_makes_noise_products_after_an_applied_selfcal(tmp_path, monkeypatch):
    seen = {}
    patch_everywhere(
        monkeypatch, imaging_selfcal_loop, "self_calibrate", lambda *a, **k: {"applied": True}
    )
    patch_everywhere(
        monkeypatch,
        imaging_selfcal_loop,
        "_selfcal_noise_products",
        lambda ms, **kwargs: seen.update(kwargs, ms=ms) or [],
    )
    patch_everywhere(monkeypatch, imaging_wsclean_runner, "_run_with_retries", lambda cmd, *args: 1)

    def stop(*args, **kwargs):
        kwargs["after_images"]()
        raise _Stop

    patch_everywhere(monkeypatch, imaging_task, "_run_and_mark", stop)
    with pytest.raises(_Stop):
        _wsclean_task(
            tmp_path, monkeypatch, selfcal_solints=["inf"], casa_data="/casa", noise_products=True
        )
    assert Path(seen["ms"]).name == "spw25.single.ms" and seen["fit"] is None
    assert seen["continuum_command"][2].endswith("wsclean")
    assert seen["cube_command"][2].endswith("wsclean-cube")
    assert seen["task"].run(["wsclean"], None) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("continuum", "cube", "gone", "kept"),
    [
        (True, True, ["wsclean", "cube", "gains"], []),
        (False, True, ["cube"], ["wsclean", "gains"]),
        (True, False, ["wsclean", "gains"], ["cube"]),
    ],
)
def test_run_wsclean_task_clears_stale_noise_products(
    tmp_path, monkeypatch, continuum, cube, gone, kept
):
    out = tmp_path / "out" / "SPW-25"
    out.mkdir(parents=True)
    files = {
        "wsclean": [
            imaging_products.SELFCAL_CORRECTION_FILENAME,
            imaging_products.TOTAL_RESIDUAL_FILENAME,
        ],
        "cube": [
            imaging_products.CUBE_SELFCAL_CORRECTION_FILENAME,
            imaging_products.CUBE_TOTAL_RESIDUAL_FILENAME,
        ],
    }
    for names in files.values():
        for name in names:
            (out / name).write_text("previous run")
    (out / imaging_products.SELFCAL_GAINS_DIRNAME).mkdir()

    def stop(*args, **kwargs):
        raise _Stop

    patch_everywhere(monkeypatch, imaging_task, "_run_and_mark", stop)
    with pytest.raises(_Stop):
        imaging_task.run_wsclean_task(
            command=["wsclean", "-name", str(out / "wsclean"), "x.ms"] if continuum else [],
            cube_command=["wsclean", "-name", str(out / "wsclean-cube"), "x.ms"] if cube else None,
            output_dir=str(out),
            ms_path="x.ms",
            spw=25,
            threads=1,
        )

    def present(group):
        if group == "gains":
            return (out / imaging_products.SELFCAL_GAINS_DIRNAME).exists()
        return all((out / name).exists() for name in files[group])

    assert not any(present(group) for group in gone)
    assert all(present(group) for group in kept)


@pytest.mark.unit
def test_selfcal_noise_products_keep_the_cause_of_a_failure(tmp_path, monkeypatch):
    import almasim.services.imaging.selfcal.loop as loop

    def broken(*args, **kwargs):
        raise ValueError("bad planes")

    patch_everywhere(monkeypatch, imaging_selfcal_loop, "_call_in_child", lambda f, kw, **_: {})
    monkeypatch.setattr(loop, "stack_channel_images", broken)
    cube = ["wsclean", "-name", "c", "-channels-out", "3", "x.ms"]
    with (
        (tmp_path / "l").open("w") as log,
        pytest.raises(RuntimeError, match="self-calibration correction cube") as info,
    ):
        imaging_selfcal_loop._selfcal_noise_products(
            Path("x.ms"),
            continuum_command=None,
            cube_command=cube,
            fit=None,
            task=FakeTask(tmp_path, run=lambda c: 1, fail=RuntimeError),
            log=log,
        )
    assert isinstance(info.value.__cause__, ValueError)

    fits.PrimaryHDU(np.ones((1, 1, 4, 4), np.float32)).writeto(tmp_path / "wsclean-residual.fits")

    def run(cmd):
        name = Path(cmd[cmd.index("-name") + 1]).name
        fits.PrimaryHDU(np.ones((1, 1, 4, 4), np.float32)).writeto(tmp_path / f"{name}-dirty.fits")
        return 1

    monkeypatch.setattr(loop, "add_images", broken)
    continuum = ["wsclean", "-name", "w", "-niter", "9", "x.ms"]
    with (
        (tmp_path / "l2").open("w") as log,
        pytest.raises(RuntimeError, match="Could not form") as info,
    ):
        imaging_selfcal_loop._selfcal_noise_products(
            Path("x.ms"),
            continuum_command=continuum,
            cube_command=None,
            fit=None,
            task=FakeTask(tmp_path, run=run, fail=RuntimeError),
            log=log,
        )
    assert isinstance(info.value.__cause__, ValueError)
