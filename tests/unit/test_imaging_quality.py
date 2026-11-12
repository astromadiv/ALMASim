"""Cleaning-quality statistics and moment maps (almasim image clean-quality)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from typer.testing import CliRunner

from almasim import cli
from almasim.services.imaging import fits_wcs, stats
from almasim.services.imaging import products as imaging_products
from almasim.services.imaging import quality as q

runner = CliRunner()
SIZE, CELL_DEG, BEAM_PIX = 64, 1 / 3600, 4.0
SOURCE = (32, 30)


def _header(n_channels: int | None = None) -> fits.Header:
    header = fits.Header()
    header["CTYPE1"], header["CDELT1"], header["CRPIX1"] = "RA---SIN", -CELL_DEG, 33
    header["CTYPE2"], header["CDELT2"], header["CRPIX2"] = "DEC--SIN", CELL_DEG, 33
    header["BMAJ"], header["BMIN"], header["BPA"] = BEAM_PIX * CELL_DEG, BEAM_PIX * CELL_DEG, 0.0
    if n_channels:
        header["CTYPE3"], header["CRVAL3"], header["CDELT3"], header["CRPIX3"] = (
            "FREQ",
            230e9,
            1e6,
            1.0,
        )
    return header


def _blob(amplitude: float) -> np.ndarray:
    yy, xx = np.mgrid[:SIZE, :SIZE]
    sigma = BEAM_PIX / 2.355
    return amplitude * np.exp(-((yy - SOURCE[0]) ** 2 + (xx - SOURCE[1]) ** 2) / (2 * sigma**2))


def _write(path: Path, data: np.ndarray, header: fits.Header) -> None:
    fits.PrimaryHDU(data.astype(np.float32), header=header).writeto(path, overwrite=True)


def _task(root: Path, *, log_lines: bool = True, line_mask: bool = False, cube: bool = True):
    """A task: continuum 1 Jy with 5 % uncleaned; a 40-channel cube, line in channels 10-19."""
    rng = np.random.default_rng(3)
    task = root / "uid___A002_X1.ms.split" / "SPW-1"
    task.mkdir(parents=True)
    noise = rng.normal(0, 0.01, (SIZE, SIZE))
    residual = noise + _blob(0.05)
    model = np.zeros((SIZE, SIZE))
    model[SOURCE] = 0.95
    header = _header()
    _write(task / "wsclean-image.fits", (_blob(0.95) + residual)[None, None], header)
    _write(task / "wsclean-residual.fits", residual[None, None], header)
    _write(task / "wsclean-model.fits", model[None, None], header)
    log = [
        f"# /bin/wsclean -name {task}/wsclean -j 1 x.ms",
        "Estimated standard deviation of background noise: 90 mJy",
        "Estimated standard deviation of background noise: 26 mJy",
        f"# /bin/wsclean -name {task}/wsclean -j 1 -continue x.ms",
        "Estimated standard deviation of background noise: 25 mJy",
    ]
    if log_lines:
        log.append("# continuum subtracted from DATA (order 1, 1 s): lines in channels 10~19")
    (task.parent / "SPW-1.log").write_text("\n".join(log) + "\n")
    if cube:
        n = 40
        image = rng.normal(0, 0.01, (n, SIZE, SIZE))
        residual_cube = image.copy()
        cube_model = np.zeros((n, SIZE, SIZE))
        for c in range(10, 20):
            image[c] += _blob(0.5)
            residual_cube[c] += _blob(0.05)
            cube_model[c][SOURCE] = 0.45
        image[39] = residual_cube[39] = np.nan  # a fully flagged channel
        ch = _header(n)
        _write(task / "wsclean-cube-image.fits", image[None], ch)
        _write(task / "wsclean-cube-residual.fits", residual_cube[None], ch)
        _write(task / "wsclean-cube-model.fits", cube_model[None], ch)
        if line_mask:
            yy, xx = np.mgrid[:SIZE, :SIZE]
            mask = np.hypot(yy - SOURCE[0], xx - SOURCE[1]) <= 6
            _write(
                task / imaging_products.LINE_MASK_FILENAME,
                mask[None, None].astype(float),
                _header(),
            )
    return task


@pytest.mark.unit
def test_continuum_quality_measures_the_flux_left_at_the_source(tmp_path):
    task = _task(tmp_path)
    stats = q.continuum_quality(task)

    assert stats["peak_pixel"] == list(SOURCE)
    assert stats["noise_jy_beam"] == pytest.approx(0.01, rel=0.15)
    assert stats["snr"] == pytest.approx(100, rel=0.15)
    assert 0.03 < stats["residual_share_1beam"] < 0.08, "the 5 % left uncleaned"
    assert stats["residual_core_sigma"] > 3 and stats["residual_peak_sigma"] > 4
    assert set(stats["residual_rings_sigma"]) == {"0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-5"}
    assert stats["residual_symmetric_sigma"] > stats["residual_antisymmetric_sigma"]
    assert stats["model_flux_jy"] == pytest.approx(0.95) and stats["model_components"] == 1
    assert stats["noise_at_mask_creation_ratio"] == pytest.approx(90 / 26)
    assert stats["diverged"] == 0
    assert stats["beam_arcsec"] == [4.0, 4.0]


@pytest.mark.unit
def test_continuum_quality_needs_all_three_images(tmp_path):
    task = _task(tmp_path, cube=False)
    (task / "wsclean-model.fits").unlink()
    assert q.continuum_quality(task) is None


@pytest.mark.unit
def test_cube_quality_with_logged_line_channels_and_the_task_mask(tmp_path):
    task = _task(tmp_path, line_mask=True)
    stats = q.cube_quality(task)

    assert stats["channels"] == 40 and stats["usable_channels"] == 39
    assert stats["line_channels"] == 10 and stats["line_channels_from"] == "log"
    assert stats["line_mask_from"] == "task" and stats["line_mask_pixels"] > 50
    assert stats["channel_noise_mjy"] == pytest.approx(10, rel=0.2)
    assert 0.05 < stats["line_residual_share"] < 0.2, "10 % of the line left in the residual"
    assert stats["line_residual_peak_sigma"] > 3
    assert abs(stats["linefree_residual_sigma"]) < 3, "no continuum in the line-free channels"
    assert stats["model_flux_jy"] == pytest.approx(4.5, rel=1e-5)
    assert stats["diverged"] == 0
    assert stats["moments"] == [
        "cube-mom0.fits",
        "cube-mom1.fits",
        "cube-mom8.fits",
        "cube-residual-line.fits",
        "cube-residual-linefree.fits",
    ]
    mom0 = fits.getdata(task / "quality" / "cube-mom0.fits")
    mom8 = fits.getdata(task / "quality" / "cube-mom8.fits")
    assert np.unravel_index(np.nanargmax(mom0), mom0.shape) == SOURCE
    assert mom8[SOURCE] == pytest.approx(0.5, abs=0.05)
    assert fits.getheader(task / "quality" / "cube-mom0.fits")["BUNIT"] == "Jy/beam.km/s"
    mom1 = fits.getdata(task / "quality" / "cube-mom1.fits")
    assert np.isfinite(mom1[SOURCE]) and np.isnan(mom1[0, 0])


@pytest.mark.unit
def test_cube_quality_finds_the_line_on_the_spectrum_and_builds_a_mask(tmp_path):
    task = _task(tmp_path, log_lines=False)
    stats = q.cube_quality(task, write_moments=False)

    assert stats["line_channels_from"] == "source spectrum"
    assert 10 <= stats["line_channels"] <= 20, "the line, widened by the finder's margins"
    assert stats["line_mask_from"] == "moment-0" and stats["line_mask_pixels"] > 0
    assert "moments" not in stats and not (task / "quality").exists()


@pytest.mark.unit
def test_cube_quality_without_cubes(tmp_path):
    assert q.cube_quality(_task(tmp_path, cube=False)) is None


@pytest.mark.unit
def test_logged_line_channels_follow_averaged_cubes(tmp_path):
    task = tmp_path / "SPW-1"
    task.mkdir()
    log = tmp_path / "SPW-1.log"

    def logged(text, n):
        log.write_text(text + "\n")
        line = q.cube._logged_line_channels(task, n)
        return None if line is None else np.flatnonzero(line).tolist()

    head = "# continuum subtracted from DATA (order 1, 1 s): "
    # 2048 channels, 128 flagged, averaged 32 to 1: the window count, not the usable one.
    text = head + "1800 of 1920 channels line-free (2048 in the window), lines in channels 900~999"
    assert logged(text, 64) == [28, 29, 30, 31]
    assert logged(text, 2048) == list(range(900, 1000))
    assert logged(text, 100) is None, "not a whole number of native channels per cube channel"
    # Older logs without the window count: trusted only without averaging.
    assert logged(head + "60 of 80 channels line-free, lines in channels 2~19", 40) is None
    assert logged(
        head + "1700 of 1920 channels line-free, lines in channels 100~199", 2048
    ) == list(range(100, 200))
    assert logged("lines in channels 20~39;60~61", 80) == list(range(20, 40)) + [60, 61]
    assert (
        logged(head + "8 of 8 channels line-free (8 in the window), lines in channels none", 8)
        == []
    )
    assert logged("nothing about lines", 40) is None
    log.unlink()
    assert q.cube._logged_line_channels(task, 40) is None


@pytest.mark.unit
def test_cube_quality_falls_back_to_the_spectrum_when_the_mapping_is_unknown(tmp_path):
    task = _task(tmp_path, log_lines=False)
    log = task.parent / "SPW-1.log"
    log.write_text(log.read_text() + "# ...: 60 of 80 channels line-free, lines in channels 2~19\n")
    assert q.cube_quality(task, write_moments=False)["line_channels_from"] == "source spectrum"


@pytest.mark.unit
def test_noise_history_reads_quoted_paths(tmp_path):
    log = tmp_path / "x.log"
    log.write_text(
        "# /bin/wsclean -name '/data/my run/wsclean' -j 1 x.ms\n"
        "Estimated standard deviation of background noise: 50 mJy\n"
        "Estimated standard deviation of background noise: 10 mJy\n"
        "# /bin/wsclean -name 'unbalanced\n"
    )
    assert q.noise_history(log, "wsclean") == pytest.approx([0.05, 0.01])


@pytest.mark.unit
def test_noise_history_reads_the_first_run_of_a_name(tmp_path):
    log = tmp_path / "x.log"
    log.write_text(
        "# /bin/wsclean -name /o/selfcal-0 -j 1 x.ms\n"
        "Estimated standard deviation of background noise: 1 Jy\n"
        "# /bin/wsclean -name /o/wsclean -j 1 x.ms\n"
        "Estimated standard deviation of background noise: 91 mJy\n"
        "Estimated standard deviation of background noise: -nan Jy\n"
        "Estimated standard deviation of background noise: oops mJy\n"
        "Estimated standard deviation of background noise: 26 mJy\n"
        "# /bin/wsclean -name /o/wsclean -j 1 -continue x.ms\n"
        "Estimated standard deviation of background noise: 25 mJy\n"
    )
    history = q.noise_history(log, "wsclean")
    assert history[0] == pytest.approx(0.091) and history[-1] == pytest.approx(0.026)
    assert math.isnan(history[1]) and len(history) == 3
    assert q.noise_history(log, "wsclean-cube") == []


@pytest.mark.unit
def test_divergences_are_counted_over_every_run_of_a_name(tmp_path):
    log = tmp_path / "x.log"
    log.write_text(
        "# /bin/wsclean -name /o/wsclean -j 1 x.ms\n"
        "WARNING: Stopping clean because of divergence!\n"
        " ==> Initial flux density of 32.98 mJy increased to -132.92 mJy\n"
        "# /bin/wsclean -name /o/wsclean -j 1 -continue x.ms\n"
        "WARNING: Stopping clean because of divergence!\n"
        "# /bin/wsclean -name /o/wsclean-cube -j 1 x.ms\n"
        "WARNING: Stopping clean because of divergence!\n"
        "WARNING: Stopping clean because of divergence!\n"
        "WARNING: Stopping clean because of divergence!\n"
    )
    assert q.divergences(log, "wsclean") == 2
    assert q.divergences(log, "wsclean-cube") == 3
    assert q.divergences(log, "selfcal-0") == 0


@pytest.mark.unit
def test_helpers():
    assert math.isnan(stats.robust_sigma(np.array([np.nan])))
    assert math.isnan(q.continuum._ratio(1.0, 0.0))
    assert math.isnan(fits_wcs.beam_pixels(fits.Header()))
    assert not stats.moment_mask(np.zeros((8, 8)), 2.0).any()
    with pytest.raises(TypeError):
        q.report._json_default(object())
    assert q.report._json_default(np.float32(1.5)) == 1.5


@pytest.mark.unit
def test_assess_task_and_summary(tmp_path):
    task = _task(tmp_path)
    result = q.assess_task(task)

    saved = json.loads((task / q.QUALITY_FILENAME).read_text())
    assert saved["continuum"]["snr"] == pytest.approx(result["continuum"]["snr"])
    assert "cube" in saved
    assert q.find_task_dirs(tmp_path) == [task]

    rows = q.summary_rows([result, {"task": str(tmp_path)}], tmp_path)
    assert rows[0]["task"] == "uid___A002_X1.ms.split/SPW-1" and rows[1]["task"] == "."
    assert rows[1]["cube.line_residual_share"] is None
    q.write_summary(rows, tmp_path / "s.csv")
    with (tmp_path / "s.csv").open() as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["continuum.snr"] and len(written) == 2


@pytest.mark.unit
def test_clean_quality_command(tmp_path):
    task = _task(tmp_path)
    broken = tmp_path / "uid___A002_X2.ms.split" / "SPW-3"
    broken.mkdir(parents=True)
    (broken / "wsclean-image.fits").write_text("not a FITS file")
    (broken / "wsclean-residual.fits").write_text("not a FITS file")
    (broken / "wsclean-model.fits").write_text("not a FITS file")

    result = runner.invoke(cli.app, ["image", "clean-quality", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "uid___A002_X1.ms.split/SPW-1" in result.output and "cont.snr" in result.output
    assert "uid___A002_X2.ms.split/SPW-3" in result.output, "a broken task is reported"
    assert (tmp_path / q.SUMMARY_FILENAME).is_file()
    assert (task / "quality" / "cube-mom0.fits").is_file()

    custom = tmp_path / "custom.csv"
    result = runner.invoke(
        cli.app, ["image", "clean-quality", str(task), "--no-moments", "--summary", str(custom)]
    )
    assert result.exit_code == 0, result.output
    assert custom.is_file()


@pytest.mark.unit
def test_clean_quality_without_tasks(tmp_path):
    result = runner.invoke(cli.app, ["image", "clean-quality", str(tmp_path)])
    assert result.exit_code == 1 and "No imaged tasks" in result.output


@pytest.mark.unit
def test_clean_quality_when_every_task_fails(tmp_path):
    broken = tmp_path / "uid___A002_X2.ms.split" / "SPW-3"
    broken.mkdir(parents=True)
    for name in ("image", "residual", "model"):
        (broken / f"wsclean-{name}.fits").write_text("not a FITS file")
    (tmp_path / q.SUMMARY_FILENAME).write_text("an earlier summary\n")

    result = runner.invoke(cli.app, ["image", "clean-quality", str(tmp_path)])

    assert result.exit_code == 1
    assert "None of the 1 task(s)" in result.output
    assert (tmp_path / q.SUMMARY_FILENAME).read_text() == "an earlier summary\n", "left alone"


@pytest.mark.unit
def test_moment_mask_grows_by_one_beam_disk_like_the_fft_convolution():
    """Dilating by the beam disk equals the line mask's FFT disk convolution > 0.5."""
    from almasim.services.imaging.stats import moment_mask

    moment = np.zeros((40, 40))
    moment[10, 12] = moment[25, 30] = 100.0
    moment += np.random.default_rng(3).normal(0, 0.1, moment.shape)
    beam = 3.4
    mask = moment_mask(moment, beam)
    seed = moment > 3 * stats.robust_sigma(moment)
    radius = int(np.ceil(beam))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    disk = (np.hypot(yy, xx) <= beam).astype(float)
    shape = [n + 2 * radius for n in seed.shape]
    grown = np.fft.irfft2(
        np.fft.rfft2(seed.astype(float), s=shape) * np.fft.rfft2(disk, s=shape), s=shape
    )
    expected = grown[radius : radius + 40, radius : radius + 40] > 0.5
    np.testing.assert_array_equal(mask, expected)


@pytest.mark.unit
def test_frequency_axis_finds_freq_anywhere_and_falls_back():
    from almasim.services.imaging.fits_wcs import frequency_axis, velocity_axis

    header = fits.Header(
        {
            "NAXIS": 4,
            "NAXIS3": 1,
            "CTYPE3": "STOKES",
            "NAXIS4": 3,
            "CTYPE4": "FREQ",
            "CRVAL4": 1e11,
            "CDELT4": 1e6,
            "CRPIX4": 2.0,
        }
    )
    axis, freq = frequency_axis(header)
    assert axis == 4 and freq.tolist() == [1e11 - 1e6, 1e11, 1e11 + 1e6]
    with pytest.raises(ValueError):
        frequency_axis(fits.Header({"NAXIS": 2}))
    velocity, width = velocity_axis(header)
    assert velocity[1] == 0.0 and width == pytest.approx(2.99792458, rel=1e-6)
