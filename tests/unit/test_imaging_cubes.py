"""Spectral-cube imaging: channel choice, per-channel stacking, continuum subtraction."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from typer.testing import CliRunner

from almasim import cli
from almasim.cli import image as cli_image
from almasim.services.imaging import continuum as imaging_continuum
from almasim.services.imaging import cube as cube_module
from almasim.services.imaging import cube as imaging_cube
from almasim.services.imaging import markers as imaging_markers
from almasim.services.imaging import products as imaging_products
from almasim.services.imaging import task as imaging_task
from almasim.services.imaging import uv_contsub as imaging_uv_contsub
from almasim.services.imaging.ms import single_window as imaging_ms_single_window
from almasim.services.imaging.selfcal import loop as imaging_selfcal_loop
from almasim.services.imaging.wsclean import command as imaging_wsclean_command
from almasim.services.runtime import env as runtime_env

from ._fakes import FakeTask, flat_run_kwargs, patch_everywhere

runner = CliRunner()


def _row(**overrides) -> pd.Series:
    """A Hanning-smoothed FDM window: 3840 x 488 kHz channels, 977 kHz resolution."""
    values = {
        "filename": "/data/uid___A002_X1_X2.ms.split.cal",
        "spectral_window_id": 25,
        "reference_frequency": 230.0e9,
        "fov_per_frequency": 28.0,
        "synthetized_beam_size": 0.5,
        "target_field_ids": "2",
        "n_channels": 3840,
        "channel_width_hz": 488.28125e3,
        "spectral_resolution_hz": 976.5625e3,
        "total_bandwidth_hz": 1875.0e6,
    }
    values.update(overrides)
    return pd.Series(values)


@pytest.mark.unit
def test_cube_channels_nyquist_keeps_native_hanning_channels():
    options = cli_image.CubeOptions(max_cube_gb=math.inf)
    assert cli_image.cube_channels(_row(), 100, options) == 3840


@pytest.mark.unit
def test_cube_channels_spectral_sampling_one_halves_hanning_channels():
    options = cli_image.CubeOptions(spectral_sampling=1.0, max_cube_gb=math.inf)
    assert cli_image.cube_channels(_row(), 100, options) == 1920


@pytest.mark.unit
def test_cube_channels_averaged_window_is_never_upsampled():
    # Online-averaged window: resolution ~ width, so Nyquist asks for half
    # the width; the native channel is the floor.
    row = _row(n_channels=480, channel_width_hz=3906.25e3, spectral_resolution_hz=3904.3e3)
    options = cli_image.CubeOptions(max_cube_gb=math.inf)
    assert cli_image.cube_channels(row, 100, options) == 480


@pytest.mark.unit
def test_cube_channels_velocity_resolution_widens_channels():
    # 2 km/s at 230 GHz is 1.534 MHz: 3.14 native channels, so 1875 MHz /
    # 1.534 MHz = 1222 channels, rounded down to a divisor of 3840.
    options = cli_image.CubeOptions(velocity_resolution_km_s=2.0, max_cube_gb=math.inf)
    channels = cli_image.cube_channels(_row(), 100, options)
    assert channels == 960
    assert 3840 % channels == 0


@pytest.mark.unit
def test_cube_channels_size_cap_uses_fewer_channels_on_big_images():
    # 2000 x 2000 x 4 bytes = 16 MB per plane; 1 GB fits 62 planes, and the
    # largest divisor of 3840 not above 62 is 60.
    options = cli_image.CubeOptions(max_cube_gb=1.0)
    assert cli_image.cube_channels(_row(), 2000, options) == 60
    assert cli_image.cube_channels(_row(), 100, options) == 3840


@pytest.mark.unit
def test_cube_channels_needs_the_spectral_columns():
    row = _row().drop(["n_channels", "channel_width_hz"])
    with pytest.raises(ValueError, match="n_channels, channel_width_hz"):
        cli_image.cube_channels(row, 100, cli_image.CubeOptions())


@pytest.mark.unit
def test_command_args_add_channels_out_only_for_cubes():
    plain = cli_image.imaging_parameter_to_command_arg(_row(), 1.0, 4.0)
    assert "-channels-out" not in plain
    cube = cli_image.imaging_parameter_to_command_arg(
        _row(), 1.0, 4.0, cube=cli_image.CubeOptions(velocity_resolution_km_s=2.0)
    )
    assert cube[cube.index("-channels-out") + 1] == "960"
    one = cli_image.imaging_parameter_to_command_arg(
        _row(n_channels=1, channel_width_hz=1e9, spectral_resolution_hz=1e9),
        1.0,
        4.0,
        cube=cli_image.CubeOptions(),
    )
    assert "-channels-out" not in one, "a single-channel window stays one plane"


def _write_channel(path: Path, value: np.ndarray, freq: float, *, bmaj: float | None = None):
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---SIN", "DEC--SIN"
    header["CTYPE3"], header["CRVAL3"], header["CDELT3"] = "FREQ", freq, 1e6
    header["CTYPE4"], header["CRVAL4"] = "STOKES", 1.0
    if bmaj is not None:
        header["BMAJ"], header["BMIN"], header["BPA"] = bmaj, bmaj / 2, 30.0
    fits.PrimaryHDU(value.reshape(1, 1, *value.shape).astype(np.float32), header).writeto(
        path, overwrite=True
    )


@pytest.mark.unit
def test_stack_channel_images_builds_cubes_with_a_beam_table(tmp_path):
    for channel in range(3):
        plane = np.full((4, 5), float(channel))
        freq = 230e9 + channel * 1e6
        _write_channel(tmp_path / f"wsclean-cube-{channel:04d}-image.fits", plane, freq, bmaj=1e-4)
        _write_channel(tmp_path / f"wsclean-cube-{channel:04d}-psf.fits", plane, freq)
        _write_channel(tmp_path / f"wsclean-cube-{channel:04d}-model.fits", plane, freq)
    _write_channel(tmp_path / "wsclean-image.fits", np.zeros((4, 5)), 230e9)  # continuum

    written = imaging_cube.stack_channel_images(
        tmp_path, prefix="wsclean-cube", keep=("image", "psf")
    )

    assert written == {"image": 3, "psf": 3}
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "wsclean-cube-image.fits",
        "wsclean-cube-psf.fits",
        "wsclean-image.fits",
    ], "per-channel files and the unwanted model are gone; the continuum is untouched"
    with fits.open(tmp_path / "wsclean-cube-image.fits") as hdus:
        assert hdus[0].data.shape == (1, 3, 4, 5)
        assert [hdus[0].data[0, c, 0, 0] for c in range(3)] == [0.0, 1.0, 2.0]
        assert hdus[0].header["CRVAL3"] == 230e9 and hdus[0].header["CDELT3"] == 1e6
        assert hdus[0].header["CASAMBM"] is True
        assert hdus["BEAMS"].data["BMAJ"] == pytest.approx([0.36] * 3)
        assert list(hdus["BEAMS"].data["CHAN"]) == [0, 1, 2]
    with fits.open(tmp_path / "wsclean-cube-psf.fits") as hdus:
        assert len(hdus) == 1, "no beam keywords, no BEAMS table"


@pytest.mark.unit
def test_stack_channel_images_rejects_uneven_channels(tmp_path):
    for channel, freq in enumerate((230e9, 230.001e9, 230.005e9)):
        _write_channel(tmp_path / f"w-{channel:04d}-image.fits", np.zeros((2, 2)), freq)
    with pytest.raises(RuntimeError, match="not evenly spaced"):
        imaging_cube.stack_channel_images(tmp_path, prefix="w")


def _cube_file(path: Path, cube: np.ndarray) -> Path:
    header = fits.Header()
    header["CTYPE3"], header["CRVAL3"], header["CDELT3"] = "FREQ", 230e9, 1e6
    fits.PrimaryHDU(cube[None].astype(np.float32), header).writeto(path)
    return path


@pytest.mark.unit
@pytest.mark.parametrize("order", [0, 1])
def test_subtract_continuum_removes_the_continuum_and_keeps_the_line(tmp_path, order):
    rng = np.random.default_rng(1)
    n_channels = 120
    channel = np.arange(n_channels)[:, None, None]
    continuum = 5.0 + (0.02 * channel if order == 1 else 0.0)
    line = np.where((channel >= 50) & (channel < 60), 3.0, 0.0) * np.ones((1, 6, 7))
    cube = continuum + line + rng.normal(0, 0.05, (n_channels, 6, 7))
    cube[10] = 0.0  # a fully flagged channel
    source = _cube_file(tmp_path / "cube.fits", cube)

    fit = imaging_continuum.subtract_continuum(
        source, tmp_path / "contsub.fits", order=order, rows_per_chunk=4
    )

    result = fits.getdata(tmp_path / "contsub.fits")[0]
    assert np.isnan(result[10]).all(), "a blank channel stays blank"
    off_line = np.delete(result, [10, *range(50, 60)], axis=0)
    assert abs(np.nanmean(off_line)) < 0.02
    assert np.nanmean(result[50:60]) == pytest.approx(3.0, abs=0.05)
    assert 0.05 < fit["line_fraction"] < 0.15, "about the 10 line channels of 119 are clipped"
    assert "continuum subtracted" in str(fits.getheader(tmp_path / "contsub.fits")["HISTORY"])


class _FakeCubeWsclean:
    """``subprocess.Popen`` stand-in writing what WSClean writes for each command."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        name = Path(cmd[cmd.index("-name") + 1])
        if "-channels-out" in cmd:
            n = int(cmd[cmd.index("-channels-out") + 1])
            products = ["image", "residual", "psf", "model"] + ["dirty"] * ("-no-dirty" not in cmd)
            for c in range(n):
                plane = np.full((3, 3), 2.0 + 0.01 * c)
                for product in products:
                    path = name.with_name(f"{name.name}-{c:04d}-{product}.fits")
                    _write_channel(path, plane, 230e9 + c * 1e6, bmaj=1e-4)
            for product in products:
                mfs = name.with_name(f"{name.name}-MFS-{product}.fits")
                _write_channel(mfs, np.ones((3, 3)), 230e9)
        else:
            _write_channel(name.with_name(f"{name.name}-image.fits"), np.ones((3, 3)), 230e9)

        class _Proc:
            stdout = iter(["WSClean version 3.7\n"])

            @staticmethod
            def wait():
                return 0

            @staticmethod
            def kill():
                pass

        return _Proc()


def _tasks(tmp_path, **kwargs):
    parameters = pd.DataFrame([_row(n_channels=8, filename=str(tmp_path / "x.ms.split.cal"))])
    common = dict(fov_fraction=1.0, beam_sampling=4.0, num_cores=2, max_cores_per_node=4)
    kwargs.setdefault("single_window", False)
    return cli_image.build_imaging_tasks(parameters, tmp_path / "out", **common, **kwargs)


@pytest.mark.unit
def test_run_wsclean_task_makes_continuum_cube_and_contsub(tmp_path, monkeypatch):
    fake = _FakeCubeWsclean()
    monkeypatch.setattr(subprocess, "Popen", fake)
    products = ("image", "residual", "psf")
    (task,), _ = _tasks(
        tmp_path, cube=cli_image.CubeOptions(), cube_products=products, single_window=True
    )

    result = imaging_task.run_wsclean_task(
        command=task.command,
        cube_command=task.cube_command,
        output_dir=str(task.output_dir),
        ms_path=str(task.ms_path),
        spw=task.spw,
        threads=2,
        cube_products=products,
        contsub_order=0,
    )

    out = task.output_dir
    assert [Path(c[c.index("-name") + 1]).name for c in fake.calls] == ["wsclean", "wsclean-cube"]
    assert "-no-dirty" in fake.calls[1]
    assert "-no-reorder" in fake.calls[0], "the continuum keeps the contiguous reader"
    assert "-no-reorder" not in fake.calls[1], "the cube reorders"
    assert sorted(p.name for p in out.iterdir()) == [
        "wsclean-cube-image.contsub.fits",
        "wsclean-cube-image.fits",
        "wsclean-cube-psf.fits",
        "wsclean-cube-residual.fits",
        "wsclean-image.fits",
    ]
    assert result["channels"] == 8
    assert fits.getdata(out / "wsclean-cube-image.fits").shape == (1, 8, 3, 3)
    marker = json.loads(imaging_markers.imaging_marker_path(out).read_text())
    assert len(marker["commands"]) == 2 and len(marker["images"]) == 3
    assert imaging_markers.is_imaging_complete(
        out, (imaging_products.IMAGE_FILENAME, imaging_products.CUBE_IMAGE_FILENAME)
    )
    assert "stacked channel images into cubes: image (8 channels)" in (
        imaging_markers.imaging_log_path(out).read_text()
    )
    continuum_history = [str(h) for h in fits.getheader(out / "wsclean-image.fits")["HISTORY"]]
    assert continuum_history[0].startswith("ALMASim: wsclean -name wsclean -j 2 ")
    for name in ("wsclean-cube-image.fits", "wsclean-cube-image.contsub.fits"):
        header = fits.getheader(out / name)
        assert header["SPW"] == "25" and header["MSNAM01"] == "x.ms.split.cal"
        assert any("-name wsclean-cube" in str(h) for h in header["HISTORY"])


@pytest.mark.unit
def test_cube_only_run_adds_cubes_next_to_existing_continuum(tmp_path, monkeypatch):
    fake = _FakeCubeWsclean()
    monkeypatch.setattr(subprocess, "Popen", fake)
    (continuum_task,), _ = _tasks(tmp_path)
    imaging_task.run_wsclean_task(
        command=continuum_task.command,
        output_dir=str(continuum_task.output_dir),
        ms_path="x",
        spw=25,
        threads=2,
    )
    continuum_bytes = (continuum_task.output_dir / imaging_products.IMAGE_FILENAME).read_bytes()
    assert _tasks(tmp_path) == ([], 1), "the continuum task is done"

    cube = cli_image.CubeOptions()
    (task,), skipped = _tasks(tmp_path, continuum=False, cube=cube)
    assert skipped == 0 and task.command == [], "only the cube is still missing"
    imaging_task.run_wsclean_task(
        command=task.command,
        cube_command=task.cube_command,
        output_dir=str(task.output_dir),
        ms_path="x",
        spw=25,
        threads=2,
    )

    assert len(fake.calls) == 2
    assert (task.output_dir / imaging_products.IMAGE_FILENAME).read_bytes() == continuum_bytes
    assert _tasks(tmp_path, continuum=False, cube=cube) == ([], 1)
    assert _tasks(tmp_path, cube=cube) == ([], 1), "continuum and cube both present"


@pytest.mark.unit
def test_build_imaging_tasks_needs_something_to_image(tmp_path):
    with pytest.raises(ValueError, match="Nothing to image"):
        _tasks(tmp_path, continuum=False)


@pytest.mark.unit
def test_image_from_ms_plan_only_fills_spectral_columns_and_reports(tmp_path, monkeypatch):
    csv = tmp_path / "params.csv"
    pd.DataFrame(
        [
            _row().drop(
                ["n_channels", "channel_width_hz", "spectral_resolution_hz", "total_bandwidth_hz"]
            )
        ]
    ).to_csv(csv, index=False)

    class _SpectralWindow:
        def nrows(self):
            return 26

        def getcell(self, name, row):
            return np.full(3840, -488.28125e3 if name == "CHAN_WIDTH" else 976.5625e3)

        def getcol(self, name):
            return np.full(26, 3840 if name == "NUM_CHAN" else 1875e6)

    opened = []
    patch_everywhere(
        monkeypatch,
        cli_image,
        "import_casacore_tables",
        lambda: lambda name, ack=False: opened.append(name) or _SpectralWindow(),
    )

    result = runner.invoke(
        cli.app,
        [
            "image",
            "image-from-ms",
            str(csv),
            str(tmp_path / "out"),
            "--cube",
            "--velocity-resolution",
            "2",
            "--plan-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert opened == ["/data/uid___A002_X1_X2.ms.split.cal::SPECTRAL_WINDOW"]
    assert "channels per cube min 960, median 960, max 960" in result.output
    assert "image, residual, psf, model, dirty" in result.output
    assert not (tmp_path / "out").exists(), "--plan-only images nothing"


@pytest.mark.unit
@pytest.mark.parametrize(("flags", "contsub_order"), [([], None), (["--contsub"], 1)])
def test_image_from_ms_cube_keeps_every_product_and_contsub_is_opt_in(
    tmp_path, monkeypatch, flags, contsub_order
):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    seen = {}

    def _run(tasks, imaging, execution, **kwargs):
        seen.update(flat_run_kwargs(imaging, execution, **kwargs), tasks=tasks)

    monkeypatch.setattr(cli_image, "run_imaging_tasks", _run)
    result = runner.invoke(
        cli.app,
        ["image", "image-from-ms", str(csv), str(tmp_path / "out"), "--cube", *flags],
    )

    assert result.exit_code == 0, result.output
    assert seen["cube_products"] == cli_image.CUBE_PRODUCTS
    assert seen["contsub_order"] == contsub_order
    (task,) = seen["tasks"]
    assert "-no-dirty" not in task.cube_command, "the dirty cube is made by default"


@pytest.mark.unit
def test_image_from_ms_rejects_contsub_without_cube(tmp_path):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    result = runner.invoke(
        cli.app, ["image", "image-from-ms", str(csv), str(tmp_path / "out"), "--contsub"]
    )
    assert result.exit_code != 0
    # Rich styles the error box when it detects CI (GITHUB_ACTIONS/FORCE_COLOR).
    assert "need --cube" in click.unstyle(result.output)


@pytest.mark.unit
@pytest.mark.parametrize("flags", [["--auto-mask", "1"], ["--auto-threshold", "3"]])
def test_image_from_ms_rejects_a_mask_not_above_the_threshold(tmp_path, flags):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    result = runner.invoke(
        cli.app, ["image", "image-from-ms", str(csv), str(tmp_path / "out"), *flags]
    )
    assert result.exit_code != 0
    assert "--auto-mask must be above --auto-threshold" in click.unstyle(result.output)


@pytest.mark.unit
def test_command_args_join_channels_only_on_multichannel_cubes():
    joined = cli_image.imaging_parameter_to_command_arg(
        _row(), 1.0, 4.0, cube=cli_image.CubeOptions(), join_channels=True
    )
    assert "-join-channels" in joined and "-channels-out" in joined
    assert "-join-channels" not in cli_image.imaging_parameter_to_command_arg(
        _row(), 1.0, 4.0, join_channels=True
    ), "a continuum image has nothing to join"
    assert "-join-channels" not in cli_image.imaging_parameter_to_command_arg(
        _row(), 1.0, 4.0, cube=cli_image.CubeOptions()
    )


@pytest.mark.unit
def test_image_from_ms_rejects_join_channels_without_cube(tmp_path):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    result = runner.invoke(
        cli.app, ["image", "image-from-ms", str(csv), str(tmp_path / "out"), "--join-channels"]
    )
    assert result.exit_code != 0
    assert "need --cube" in click.unstyle(result.output)


def _spectrum_with_line(n=400, line=slice(180, 200), noise=0.01, seed=3):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, n)
    spectrum = 1.0 + 0.1 * x + rng.normal(0, noise, n)
    spectrum[line] += 0.5
    return spectrum


@pytest.mark.unit
def test_line_free_channels_drops_the_line_and_its_wings():
    spectrum = _spectrum_with_line()
    usable = np.ones(spectrum.size, bool)
    usable[:5] = False  # flagged band edge

    line_free = imaging_continuum.line_free_channels(spectrum, usable)

    assert not line_free[:5].any(), "unusable channels are never line-free"
    assert not line_free[176:204].any(), "the line and its wings"
    off_line = np.r_[5:150, 230:400]
    assert line_free[off_line].mean() > 0.8, "noise spikes may cost a few channels, no more"


@pytest.mark.unit
def test_line_free_channels_finds_faint_broad_wings_by_smoothing():
    rng = np.random.default_rng(2)
    n = 2048
    spectrum = 1.0 + rng.normal(0, 0.1, n)
    spectrum[1000:1100] += 0.06  # 0.6 sigma per channel, 6 sigma over the 100 channels
    usable = np.ones(n, bool)

    line_free = imaging_continuum.line_free_channels(spectrum, usable)
    assert line_free[1000:1100].mean() < 0.2, "the wing is found on the smoothed spectrum"
    single = imaging_continuum.line_free_channels(spectrum, usable, widths=(1,))
    assert single[1000:1100].mean() > 0.8, "channel by channel it is not"


@pytest.mark.unit
@pytest.mark.parametrize("order", [0, 1])
def test_fit_uv_continuum_fits_each_row_over_the_line_free_channels(order):
    rng = np.random.default_rng(5)
    n_rows, n_channels, n_corr = 6, 64, 2
    x = np.linspace(-1, 1, n_channels)[None, :, None]
    offset = rng.normal(0, 1, (n_rows, 1, n_corr)) + 1j * rng.normal(0, 1, (n_rows, 1, n_corr))
    slope = (0.3 + 0.2j) * (order == 1)
    continuum = offset + slope * x
    data = (continuum + 0j).astype(np.complex64)
    data[:, 30:34] += 5.0  # the line
    data[:, 50] = 1e6  # a flagged spike
    flag = np.zeros(data.shape, bool)
    flag[:, 50] = True
    flag[0] = True  # a fully flagged row
    weight = np.ones(data.shape)
    line_free = np.ones(n_channels, bool)
    line_free[28:36] = False

    fitted = imaging_continuum.fit_uv_continuum(data, weight, flag, line_free, order)

    assert fitted.shape == data.shape and fitted.dtype == data.dtype
    assert np.all(fitted[0] == 0), "no usable channel: no continuum"
    np.testing.assert_allclose(fitted[1:], continuum[1:].repeat(1, axis=1), atol=1e-4)
    assert np.allclose((data - fitted)[1:, 30:34], 5.0, atol=1e-4), "the line is kept"


@pytest.mark.unit
def test_fit_uv_continuum_rejects_other_orders():
    with pytest.raises(ValueError, match="order"):
        imaging_continuum.fit_uv_continuum(
            np.zeros((1, 4, 1)), np.ones((1, 4, 1)), np.zeros((1, 4, 1), bool), np.ones(4, bool), 2
        )


@pytest.mark.unit
def test_channel_ranges_lists_runs_casa_style():
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
    assert imaging_continuum._channel_ranges(mask) == "1~2;5~5;7~9"
    assert imaging_continuum._channel_ranges(np.zeros(4, bool)) == ""


def _uv_task(tmp_path, monkeypatch, fit, **kwargs):
    fake = _FakeCubeWsclean()
    monkeypatch.setattr(subprocess, "Popen", fake)
    patch_everywhere(
        monkeypatch,
        imaging_ms_single_window,
        "extract_single_window_ms_subprocess",
        lambda *a, **k: {"rows": 10, "bytes": 0},
    )
    calls = []
    patch_everywhere(
        monkeypatch,
        imaging_uv_contsub,
        "subtract_uv_continuum_subprocess",
        lambda ms, **k: calls.append((ms, k)) or fit,
    )
    (task,), _ = _tasks(tmp_path, cube=cli_image.CubeOptions(), single_window=True)
    imaging_task.run_wsclean_task(
        command=task.command,
        cube_command=task.cube_command,
        output_dir=str(task.output_dir),
        ms_path=str(task.ms_path),
        spw=task.spw,
        threads=2,
        single_window=True,
        uv_contsub_order=1,
        **kwargs,
    )
    return fake, calls, imaging_markers.imaging_log_path(task.output_dir).read_text()


@pytest.mark.unit
def test_run_wsclean_task_images_continuum_and_lines_from_their_columns(tmp_path, monkeypatch):
    fit = {
        "source_column": "DATA",
        "order": 1,
        "seconds": 1.0,
        "channels": 8,
        "usable_channels": 8,
        "line_free_channels": 6,
        "line_ranges": "3~4",
    }
    fake, calls, log = _uv_task(tmp_path, monkeypatch, fit)

    ((ms, kwargs),) = calls
    assert Path(ms).name == "spw25.single.ms" and kwargs["order"] == 1
    continuum, cube = fake.calls
    assert continuum[continuum.index("-data-column") + 1] == imaging_products.CONTINUUM_COLUMN
    assert cube[cube.index("-data-column") + 1] == imaging_products.LINE_COLUMN
    assert continuum[-1] == cube[-1] and continuum[-1].endswith("spw25.single.ms")
    assert "6 of 8 channels line-free (8 in the window), lines in channels 3~4" in log


@pytest.mark.unit
def test_run_wsclean_task_keeps_the_data_when_too_few_channels_are_line_free(tmp_path, monkeypatch):
    fake, _, log = _uv_task(
        tmp_path, monkeypatch, {"skipped": "only 1 of 8 usable channels are line-free"}
    )
    assert all("-data-column" not in c for c in fake.calls)
    assert "continuum NOT subtracted from the visibilities: only 1 of 8" in log


@pytest.mark.unit
def test_run_wsclean_task_uv_contsub_needs_a_single_window_cube():
    with pytest.raises(ValueError, match="single_window and a cube_command"):
        imaging_task.run_wsclean_task(
            command=["wsclean"],
            output_dir="x",
            ms_path="x",
            spw=1,
            threads=1,
            uv_contsub_order=1,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--uv-contsub"], "need --cube"),
        (["--cube", "--uv-contsub", "--contsub"], "drop --contsub"),
        (["--cube", "--uv-contsub", "--no-single-window-ms"], "needs --single-window-ms"),
    ],
)
def test_image_from_ms_checks_uv_contsub(tmp_path, flags, message):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    result = runner.invoke(
        cli.app, ["image", "image-from-ms", str(csv), str(tmp_path / "out"), *flags]
    )
    assert result.exit_code != 0
    assert message in click.unstyle(result.output)


@pytest.mark.unit
def test_image_from_ms_passes_uv_contsub_order(tmp_path, monkeypatch):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    seen = {}
    monkeypatch.setattr(
        cli_image,
        "run_imaging_tasks",
        lambda tasks, imaging, execution, **k: seen.update(
            flat_run_kwargs(imaging, execution, **k)
        ),
    )
    result = runner.invoke(
        cli.app,
        [
            "image",
            "image-from-ms",
            str(csv),
            str(tmp_path / "out"),
            "--cube",
            "--uv-contsub",
            "--contsub-order",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["uv_contsub_order"] == 0 and seen["contsub_order"] is None


def _run_selfcal(
    tmp_path, monkeypatch, snrs, solints=("inf", "30s"), min_snr=20.0, restart=False, **kwargs
):
    calls = []

    def fake_call(function, kwargs, *, casa=False, **_):
        assert casa == function.startswith("casa_"), "only CASA steps get the CASA environment"
        if function == "export_gain_table":
            Path(kwargs["csv_path"]).write_text("time_mjd_s,antenna\n")
            return {"rows": 1}
        if function == "casa_selfcal_solve":
            pre = tuple(Path(t).name for t in kwargs["pre_apply"])
            calls.append(("solve", Path(kwargs["caltable"]).name, kwargs["calmode"], pre))
            Path(kwargs["caltable"]).mkdir()  # a CASA caltable is a directory
            return {
                "solutions": 9,
                "possible": 10,
                "median_phase_deg": 12.0,
                "median_amp_dev": 0.05,
            }
        if function == "casa_selfcal_apply":
            calls.append(("apply", tuple(Path(t).name for t in kwargs["caltables"])))
        elif function == "drop_corrected_data":
            calls.append(("apply", ()))
        else:
            calls.append((function,))
        return {"spw": "0:0~9", "refant": "DA41", "line_free_channels": 10}

    patch_everywhere(monkeypatch, imaging_selfcal_loop, "_call_in_child", fake_call)
    remaining = list(snrs)
    patch_everywhere(
        monkeypatch, imaging_selfcal_loop, "image_snr", lambda image, residual: remaining.pop(0)
    )
    patch_everywhere(monkeypatch, imaging_selfcal_loop, "image_peak", lambda image: 1.0)
    images = []

    def fake_run(cmd):
        name = Path(cmd[cmd.index("-name") + 1])
        images.append(cmd)
        name.with_name(name.name + "-image.fits").write_text("x")
        return 1

    with (tmp_path / "task.log").open("w") as log:
        result = imaging_selfcal_loop.self_calibrate(
            tmp_path / "spw.ms",
            ["wsclean", "-name", str(tmp_path / "wsclean"), "-no-update-model-required", "ms"],
            task=FakeTask(tmp_path, run=fake_run, fail=RuntimeError),
            log=log,
            solints=solints,
            min_snr=min_snr,
            casa_data="/casa-data",
            restart=restart,
            **kwargs,
        )
    return result, calls, images, (tmp_path / "task.log").read_text()


@pytest.mark.unit
def test_self_calibrate_keeps_rounds_while_the_snr_rises(tmp_path, monkeypatch):
    result, calls, images, log = _run_selfcal(tmp_path, monkeypatch, [30.0, 45.0, 60.0])

    assert result["applied"] and result["snr"] == 60.0
    assert calls == [
        ("adopt_corrected_data",),
        ("selfcal_prepare",),
        ("solve", "selfcal-1.G", "p", ()),
        ("apply", ("selfcal-1.G",)),
        ("solve", "selfcal-2.G", "p", ()),
        ("apply", ("selfcal-2.G",)),
    ]
    assert all(
        "-update-model-required" in c and "-no-update-model-required" not in c for c in images
    )
    assert not list(tmp_path.glob("selfcal-*")), "temporary images and caltables removed"
    assert "CASA gaincal: refant DA41" in log and "S/N 45.0 -> 60.0" in log


@pytest.mark.unit
def test_self_calibrate_amplitude_round_on_top_of_the_kept_phase(tmp_path, monkeypatch):
    result, calls, _, log = _run_selfcal(
        tmp_path, monkeypatch, [30.0, 45.0, 44.0, 50.0], solints=("inf", "30s", "ap:inf")
    )
    # Round 2 (30s) is rejected, so round 1 is back in place and the rounds stop.
    assert calls[-1] == ("apply", ("selfcal-1.G",)) and len(calls) == 7

    calls.clear()
    result, calls, _, log = _run_selfcal(
        tmp_path, monkeypatch, [30.0, 45.0, 50.0], solints=("inf", "ap:inf")
    )
    assert calls[-2:] == [
        ("solve", "selfcal-2.A", "ap", ("selfcal-1.G",)),
        ("apply", ("selfcal-1.G", "selfcal-2.A")),
    ]
    assert result["applied"] and result["snr"] == 50.0
    assert "round 2 (ap, solint inf)" in log and "median |amp - 1| 0.05" in log
    assert "applied (selfcal-1.G, selfcal-2.A)" in log


@pytest.mark.unit
def test_self_calibrate_rejected_amplitude_round_keeps_the_phase(tmp_path, monkeypatch):
    result, calls, _, log = _run_selfcal(
        tmp_path, monkeypatch, [30.0, 45.0, 40.0], solints=("inf", "ap:inf")
    )
    assert calls[-1] == ("apply", ("selfcal-1.G",)) and result["snr"] == 45.0
    assert "rejected" in log and "applied (selfcal-1.G)" in log


@pytest.mark.unit
def test_self_calibrate_reverts_a_round_that_makes_it_worse(tmp_path, monkeypatch):
    result, calls, _, log = _run_selfcal(tmp_path, monkeypatch, [30.0, 45.0, 40.0])
    assert calls[-1] == ("apply", ("selfcal-1.G",)), "back to round 1"
    assert result["snr"] == 45.0 and "rejected" in log

    result, calls, _, _ = _run_selfcal(tmp_path, monkeypatch, [30.0, 25.0], solints=("inf",))
    assert calls[-1] == ("apply", ()), "round 1 rejected: CORRECTED_DATA dropped"
    assert not result["applied"]


@pytest.mark.unit
def test_self_calibrate_skips_faint_sources(tmp_path, monkeypatch):
    result, calls, images, log = _run_selfcal(tmp_path, monkeypatch, [8.0])
    assert calls == [("adopt_corrected_data",)] and len(images) == 1
    assert not result["applied"]
    assert "skipped: S/N below 20" in log and "CASA" not in log


@pytest.mark.unit
def test_parse_selfcal_rounds():
    assert imaging_selfcal_loop.parse_selfcal_rounds(["inf", "30s", "ap:inf", "ap:60s"]) == [
        ("p", "inf"),
        ("p", "30s"),
        ("ap", "inf"),
        ("ap", "60s"),
    ]
    with pytest.raises(ValueError, match="needs a phase round"):
        imaging_selfcal_loop.parse_selfcal_rounds(["ap:inf"])
    for bad in ("fast", "a:inf", "ap:"):
        with pytest.raises(ValueError, match="self-calibration rounds are"):
            imaging_selfcal_loop.parse_selfcal_rounds([bad])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--selfcal", "inf,fast"], "self-calibration rounds are"),
        (["--selfcal", "ap:inf"], "needs a phase round"),
        (["--selfcal", "inf", "--no-single-window-ms"], "--selfcal needs --continuum"),
        (["--selfcal", "inf", "--no-continuum", "--cube"], "--selfcal needs --continuum"),
    ],
)
def test_image_from_ms_checks_selfcal(tmp_path, flags, message):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    result = runner.invoke(
        cli.app, ["image", "image-from-ms", str(csv), str(tmp_path / "out"), *flags]
    )
    assert result.exit_code != 0
    assert message in " ".join(click.unstyle(result.output).replace("│", " ").split())


@pytest.mark.unit
def test_image_from_ms_passes_selfcal_solints_and_casa_data(tmp_path, monkeypatch):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    seen = {}
    monkeypatch.setattr(
        cli_image,
        "run_imaging_tasks",
        lambda tasks, imaging, execution, **k: seen.update(
            flat_run_kwargs(imaging, execution, **k)
        ),
    )
    monkeypatch.setattr(
        cli_image, "_prepare_selfcal_casa_data", lambda out, root: Path(out) / ".casa-data"
    )
    result = runner.invoke(
        cli.app,
        ["image", "image-from-ms", str(csv), str(tmp_path / "out"), "--selfcal", "inf, ap:inf"],
    )
    assert result.exit_code == 0, result.output
    assert seen["selfcal_solints"] == ["inf", "ap:inf"] and seen["selfcal_min_snr"] == 20.0
    assert seen["casa_data"].endswith("out/.casa-data")


@pytest.mark.unit
def test_casa_environment_puts_the_bundled_casatools_libraries_first():
    """One policy for every CASA child (T2.3): bundled libs, system libs, inherited."""
    base = {"LD_LIBRARY_PATH": "/opt/spack/lib", "PATH": "/bin"}
    env = runtime_env.casa_environment(base, bundled="/site/casatools/__casac__/lib")
    entries = env["LD_LIBRARY_PATH"].split(":")
    assert entries[0] == "/site/casatools/__casac__/lib" and entries[1] == "/lib64"
    assert entries[-1] == "/opt/spack/lib" and env["OMP_NUM_THREADS"] == "1"
    without = runtime_env.casa_environment({}, bundled=None)["LD_LIBRARY_PATH"]
    assert without.startswith("/lib64:/usr/lib64:")
    assert runtime_env.wsclean_environment(1, base)["LD_LIBRARY_PATH"].startswith("/lib64")


@pytest.mark.unit
def test_continuum_restart_commands_write_the_model_then_continue():
    first, second = imaging_wsclean_command.continuum_restart_commands(
        ["wsclean", "-name", "w", "-no-update-model-required", "x.ms"]
    )
    assert first == ["wsclean", "-name", "w", "-update-model-required", "x.ms"]
    assert second == ["wsclean", "-name", "w", "-update-model-required", "-continue", "x.ms"]

    first, _ = imaging_wsclean_command.continuum_restart_commands(["wsclean", "-name", "w", "x.ms"])
    assert first[-2:] == ["-update-model-required", "x.ms"]


@pytest.mark.unit
def test_run_wsclean_task_cleans_the_continuum_in_two_passes(tmp_path, monkeypatch):
    fake = _FakeCubeWsclean()
    monkeypatch.setattr(subprocess, "Popen", fake)
    patch_everywhere(
        monkeypatch,
        imaging_ms_single_window,
        "extract_single_window_ms_subprocess",
        lambda *a, **k: {"rows": 1, "bytes": 0},
    )
    (task,), _ = _tasks(tmp_path, cube=cli_image.CubeOptions(), single_window=True)

    imaging_task.run_wsclean_task(
        command=task.command,
        cube_command=task.cube_command,
        output_dir=str(task.output_dir),
        ms_path=str(task.ms_path),
        spw=task.spw,
        threads=2,
        single_window=True,
        continuum_restart=True,
    )

    names = [Path(c[c.index("-name") + 1]).name for c in fake.calls]
    assert names == ["wsclean", "wsclean", "wsclean-cube"]
    first, second, cube = fake.calls
    assert "-update-model-required" in first and "-continue" not in first
    assert second[-2] == "-continue" and second[-1] == first[-1]
    assert "-continue" not in cube and "-no-update-model-required" in cube
    marker = json.loads(imaging_markers.imaging_marker_path(task.output_dir).read_text())
    assert len(marker["commands"]) == 3


@pytest.mark.unit
def test_continuum_restart_needs_the_single_window_copy():
    with pytest.raises(ValueError, match="needs single_window"):
        imaging_task.run_wsclean_task(
            command=["wsclean", "-name", "w", "x.ms"],
            output_dir="x",
            ms_path="x",
            spw=1,
            threads=1,
            continuum_restart=True,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], True), (["--no-continuum-restart"], False), (["--no-single-window-ms"], False)],
)
def test_image_from_ms_continuum_restart_default(tmp_path, monkeypatch, flags, expected):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    seen = {}
    monkeypatch.setattr(
        cli_image,
        "run_imaging_tasks",
        lambda tasks, imaging, execution, **k: seen.update(
            flat_run_kwargs(imaging, execution, **k)
        ),
    )
    result = runner.invoke(
        cli.app, ["image", "image-from-ms", str(csv), str(tmp_path / "out"), *flags]
    )
    assert result.exit_code == 0, result.output
    assert seen["continuum_restart"] is expected


@pytest.mark.unit
def test_both_continuum_passes_read_the_fitted_continuum(tmp_path, monkeypatch):
    fit = {
        "source_column": "DATA",
        "order": 1,
        "seconds": 1.0,
        "channels": 8,
        "usable_channels": 8,
        "line_free_channels": 6,
        "line_ranges": "3~4",
    }
    fake, _, _ = _uv_task(tmp_path, monkeypatch, fit, continuum_restart=True)

    first, second, cube = fake.calls
    for continuum in (first, second):
        assert continuum[continuum.index("-data-column") + 1] == imaging_products.CONTINUUM_COLUMN
    assert (
        "-continue" in second
        and cube[cube.index("-data-column") + 1] == imaging_products.LINE_COLUMN
    )


@pytest.mark.unit
def test_self_calibrate_cleans_every_image_in_two_passes(tmp_path, monkeypatch):
    result, _, images, _ = _run_selfcal(
        tmp_path, monkeypatch, [30.0, 45.0], solints=("inf",), restart=True
    )
    names = [Path(c[c.index("-name") + 1]).name for c in images]
    assert names == ["selfcal-0", "selfcal-0", "selfcal-1", "selfcal-1"]
    assert ["-continue" in c for c in images] == [False, True, False, True]
    assert result["snr"] == 45.0


def _line_channels(out, n_channels=40, size=64, seed=8):
    """Channel images of a faint line: 1 sigma per channel, ~6 sigma summed."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    blob = np.exp(-((yy - 32) ** 2 + (xx - 20) ** 2) / (2 * 2.0**2))
    out.mkdir(parents=True, exist_ok=True)
    for c in range(n_channels):
        plane = rng.normal(0, 1, (size, size)) + 1.0 * blob
        path = out / f"wsclean-cube-{c:04d}-image.fits"
        _write_channel(path, plane, 230e9 + c * 1e6, bmaj=4 * 2.5e-4)  # 4 pixels
        with fits.open(path, mode="update") as hdus:
            hdus[0].header["CDELT1"], hdus[0].header["CDELT2"] = -2.5e-4, 2.5e-4


@pytest.mark.unit
def test_line_mask_from_channels_finds_a_line_no_channel_shows(tmp_path):
    _line_channels(tmp_path)

    pixels = imaging_cube.line_mask_from_channels(tmp_path, tmp_path / "mask.fits")

    mask = fits.getdata(tmp_path / "mask.fits").squeeze().astype(bool)
    assert pixels == mask.sum() and mask[32, 20], "the blob is in the mask"
    assert mask[32, 24] and not mask[32, 40], "grown by about one beam, no further"
    assert mask.mean() < 0.1


@pytest.mark.unit
def test_line_mask_from_channels_without_channels_or_signal(tmp_path):
    assert imaging_cube.line_mask_from_channels(tmp_path, tmp_path / "mask.fits") == 0
    for c in range(3):
        _write_channel(tmp_path / f"wsclean-cube-{c:04d}-image.fits", np.zeros((8, 8)), 230e9)
    (tmp_path / "mask.fits").write_text("stale mask from an earlier run")
    assert imaging_cube.line_mask_from_channels(tmp_path, tmp_path / "mask.fits") == 0
    assert not (tmp_path / "mask.fits").exists(), "no mask built, none left behind"


@pytest.mark.unit
def test_line_mask_pass_continues_inside_the_mask(tmp_path, monkeypatch):
    cube = ["wsclean", "-name", "c", "-auto-mask", "3.0", "-auto-threshold", "1.0", "x.ms"]
    with (tmp_path / "log").open("w") as log:
        monkeypatch.setattr(cube_module, "line_mask_from_channels", lambda out, path: 120)
        second = imaging_cube._line_mask_pass(cube, tmp_path, log)
        monkeypatch.setattr(cube_module, "line_mask_from_channels", lambda out, path: 0)
        assert imaging_cube._line_mask_pass(cube, tmp_path, log) is None
    assert second == [
        "wsclean",
        "-name",
        "c",
        "-auto-threshold",
        "1.0",
        "-fits-mask",
        str(tmp_path / imaging_products.LINE_MASK_FILENAME),
        "-continue",
        "x.ms",
    ]
    log = (tmp_path / "log").read_text()
    assert "120 pixels; second pass" in log and "nothing above 3 sigma" in log


@pytest.mark.unit
def test_run_wsclean_task_cleans_the_cube_again_inside_the_line_mask(tmp_path, monkeypatch):
    fake = _FakeCubeWsclean()
    monkeypatch.setattr(subprocess, "Popen", fake)
    patch_everywhere(
        monkeypatch,
        imaging_ms_single_window,
        "extract_single_window_ms_subprocess",
        lambda *a, **k: {"rows": 1, "bytes": 0},
    )
    monkeypatch.setattr(cube_module, "line_mask_from_channels", lambda out, path: 50)
    (task,), _ = _tasks(tmp_path, cube=cli_image.CubeOptions(), single_window=True)

    imaging_task.run_wsclean_task(
        command=task.command,
        cube_command=task.cube_command,
        output_dir=str(task.output_dir),
        ms_path=str(task.ms_path),
        spw=task.spw,
        threads=2,
        single_window=True,
        cube_line_mask=True,
    )

    continuum, cube, second = fake.calls
    assert "-no-update-model-required" in continuum
    assert "-update-model-required" in cube and "-continue" not in cube
    assert second[-2] == "-continue" and "-fits-mask" in second and "-auto-mask" not in second
    marker = json.loads(imaging_markers.imaging_marker_path(task.output_dir).read_text())
    assert len(marker["commands"]) == 3
    assert fits.getdata(task.output_dir / "wsclean-cube-image.fits").shape[1] == 8


@pytest.mark.unit
def test_cube_line_mask_needs_the_single_window_copy():
    with pytest.raises(ValueError, match="cube_line_mask writes MODEL_DATA"):
        imaging_task.run_wsclean_task(
            command=[],
            cube_command=["wsclean", "-name", "c", "x.ms"],
            output_dir="x",
            ms_path="x",
            spw=1,
            threads=1,
            cube_line_mask=True,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (["--cube"], True),
        (["--cube", "--no-cube-line-mask"], False),
        (["--cube", "--no-single-window-ms"], False),
        ([], False),
    ],
)
def test_image_from_ms_cube_line_mask_default(tmp_path, monkeypatch, flags, expected):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    seen = {}
    monkeypatch.setattr(
        cli_image,
        "run_imaging_tasks",
        lambda tasks, imaging, execution, **k: seen.update(
            flat_run_kwargs(imaging, execution, **k)
        ),
    )
    result = runner.invoke(
        cli.app, ["image", "image-from-ms", str(csv), str(tmp_path / "out"), *flags]
    )
    assert result.exit_code == 0, result.output
    assert seen["cube_line_mask"] is expected


@pytest.mark.unit
def test_line_mask_from_channels_without_a_pixel_scale_is_not_grown(tmp_path):
    rng = np.random.default_rng(3)
    for c in range(20):
        plane = rng.normal(0, 1, (32, 32))
        plane[16, 16] += 3.0
        _write_channel(tmp_path / f"wsclean-cube-{c:04d}-image.fits", plane, 230e9, bmaj=1e-3)
    pixels = imaging_cube.line_mask_from_channels(tmp_path, tmp_path / "mask.fits")
    mask = fits.getdata(tmp_path / "mask.fits").squeeze().astype(bool)
    assert mask[16, 16] and pixels <= 5, "the source pixel, plus at most a stray noise peak"


@pytest.mark.unit
def test_line_mask_pass_reads_the_line_column_too(tmp_path, monkeypatch):
    monkeypatch.setattr(cube_module, "line_mask_from_channels", lambda out, path: 50)
    fit = {
        "source_column": "DATA",
        "order": 1,
        "seconds": 1.0,
        "channels": 8,
        "usable_channels": 8,
        "line_free_channels": 6,
        "line_ranges": "3~4",
    }
    fake, _, _ = _uv_task(tmp_path, monkeypatch, fit, cube_line_mask=True)

    continuum, cube, second = fake.calls
    assert continuum[continuum.index("-data-column") + 1] == imaging_products.CONTINUUM_COLUMN
    for command in (cube, second):
        assert command[command.index("-data-column") + 1] == imaging_products.LINE_COLUMN
    assert "-continue" in second and "-fits-mask" in second


@pytest.mark.unit
def test_line_mask_pass_zeroes_the_nan_models_of_empty_channels(tmp_path, monkeypatch):
    for c, value in enumerate([1.0, np.nan]):
        plane = np.full((4, 4), value)
        _write_channel(tmp_path / f"wsclean-cube-{c:04d}-model.fits", plane, 230e9)
    monkeypatch.setattr(cube_module, "line_mask_from_channels", lambda out, path: 10)

    with (tmp_path / "log").open("w") as log:
        assert imaging_cube._line_mask_pass(["wsclean", "-name", "c", "x.ms"], tmp_path, log)

    assert np.all(fits.getdata(tmp_path / "wsclean-cube-0001-model.fits") == 0)
    assert np.all(fits.getdata(tmp_path / "wsclean-cube-0000-model.fits") == 1)
    assert "1 channel model(s) were NaN" in (tmp_path / "log").read_text()
    assert imaging_cube._zero_non_finite_models(tmp_path) == 0, "nothing left to fix"


@pytest.mark.unit
@pytest.mark.parametrize("order", [0, 1])
def test_line_free_channels_on_a_noiseless_spectrum(order):
    spectrum = np.ones(128) if order == 0 else np.linspace(1.0, 1.2, 128)
    spectrum = spectrum.copy()
    spectrum[62:66] += 4.0
    line_free = imaging_continuum.line_free_channels(spectrum, np.ones(128, bool), order=order)
    assert not line_free[62:66].any(), "zero spread must not hide the line"
    assert line_free[:50].all() and line_free[80:].all()


@pytest.mark.unit
def test_line_free_channels_on_a_short_spectrum():
    line_free = imaging_continuum.line_free_channels(
        np.array([1.0, 1.01, 1.02, 2.0]), np.ones(4, bool), order=0, grow=2
    )
    assert line_free.shape == (4,)


@pytest.mark.unit
def test_self_calibrate_keeps_the_applied_gain_tables(tmp_path, monkeypatch):
    result, _, _, log = _run_selfcal(
        tmp_path, monkeypatch, [30.0, 45.0, 50.0], solints=("inf", "ap:inf")
    )
    gains = tmp_path / imaging_products.SELFCAL_GAINS_DIRNAME
    assert result["gains"] == ["phase.cal", "phase.csv", "amplitude.cal", "amplitude.csv"]
    assert (gains / "phase.cal").is_dir() and (gains / "amplitude.csv").is_file()
    assert "gains kept in selfcal_gains/" in log
    assert not list(tmp_path.glob("selfcal-*")), "the working tables are still removed"

    (tmp_path / "rejected").mkdir()
    result, _, _, _ = _run_selfcal(
        tmp_path / "rejected", monkeypatch, [30.0, 25.0], solints=("inf",)
    )
    assert (
        result["gains"] == []
        and not (tmp_path / "rejected" / imaging_products.SELFCAL_GAINS_DIRNAME).exists()
    )


@pytest.mark.unit
@pytest.mark.parametrize(("flags", "expected"), [([], True), (["--no-noise-products"], False)])
def test_image_from_ms_noise_products_default(tmp_path, monkeypatch, flags, expected):
    csv = tmp_path / "params.csv"
    pd.DataFrame([_row()]).to_csv(csv, index=False)
    seen = {}
    monkeypatch.setattr(
        cli_image,
        "run_imaging_tasks",
        lambda tasks, imaging, execution, **k: seen.update(
            flat_run_kwargs(imaging, execution, **k)
        ),
    )
    result = runner.invoke(
        cli.app, ["image", "image-from-ms", str(csv), str(tmp_path / "out"), *flags]
    )
    assert result.exit_code == 0, result.output
    assert seen["noise_products"] is expected


@pytest.mark.unit
def test_self_calibrate_keeps_no_gains_when_asked_not_to(tmp_path, monkeypatch):
    result, _, _, log = _run_selfcal(
        tmp_path, monkeypatch, [30.0, 45.0], solints=("inf",), keep_gains=False
    )
    assert result["applied"] and result["gains"] == []
    assert (
        not (tmp_path / imaging_products.SELFCAL_GAINS_DIRNAME).exists() and "gains kept" not in log
    )
