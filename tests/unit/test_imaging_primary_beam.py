"""Unit tests for the ALMA primary-beam model and the primary-beam correction."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from typer.testing import CliRunner

from almasim import cli
from almasim.cli import image as cli_image
from almasim.services.imaging import markers as imaging_markers
from almasim.services.imaging import primary_beam as pbm
from almasim.services.imaging import products as imaging_products
from almasim.services.imaging import task as imaging_task
from almasim.services.imaging.task_run import TaskRun
from almasim.services.imaging.wsclean import runner as imaging_wsclean_runner

from ._fakes import flat_run_kwargs, patch_everywhere

runner = CliRunner()
C = pbm.SPEED_OF_LIGHT_M_S


def _write_image(
    path: Path,
    *,
    channels: int = 1,
    size: int = 65,
    cell_arcsec: float = 1.0,
    frequency: float = 100e9,
    step: float = 1e9,
    value: float = 2.0,
    beams: bool = False,
) -> None:
    """A WSClean-like image: (STOKES, FREQ, DEC, RA), reference pixel in the middle."""
    header = fits.Header()
    for axis, (ctype, _n, crpix, cdelt, crval) in enumerate(
        [
            ("RA---SIN", size, size // 2 + 1, -cell_arcsec / 3600, 10.0),
            ("DEC--SIN", size, size // 2 + 1, cell_arcsec / 3600, -30.0),
            ("FREQ", channels, 1, step, frequency),
            ("STOKES", 1, 1, 1, 1),
        ],
        start=1,
    ):
        header[f"CTYPE{axis}"] = ctype
        header[f"CRPIX{axis}"] = crpix
        header[f"CDELT{axis}"] = cdelt
        header[f"CRVAL{axis}"] = crval
    header["BUNIT"] = "JY/BEAM"
    data = np.full((1, channels, size, size), value, dtype=np.float32)
    hdus = [fits.PrimaryHDU(data, header=header)]
    if beams:
        hdus.append(
            fits.BinTableHDU.from_columns(
                [fits.Column("BMAJ", "1E", "arcsec", array=np.ones(channels))], name="BEAMS"
            )
        )
    fits.HDUList(hdus).writeto(path)


@pytest.mark.unit
def test_airy_aperture_uses_casa_alma_values_and_falls_back_to_the_dish():
    assert pbm.airy_aperture(12.0) == (10.7, 0.75)
    assert pbm.airy_aperture(7.0) == (6.25, 0.75)
    assert pbm.airy_aperture(25.0) == (25.0, 0.0)


@pytest.mark.unit
@pytest.mark.parametrize(("dish", "fwhm"), [(12.0, 1.151), (7.0, 1.143)])
def test_primary_beam_half_power_width(dish, fwhm):
    frequency = 230e9
    lam = C / frequency
    assert pbm.primary_beam_response(np.array(0.0), frequency, dish) == pytest.approx(1.0)
    width = 2 * pbm._half_power_radius(dish, frequency) / (lam / dish)
    assert width == pytest.approx(fwhm, abs=2e-3)


@pytest.mark.unit
def test_unblocked_airy_has_its_first_null_at_1_22_lambda_over_d():
    frequency, dish = 100e9, 25.0
    null = 1.2197 * C / frequency / dish
    assert pbm.primary_beam_response(np.array(null), frequency, dish) < 1e-6
    assert pbm.primary_beam_response(np.array(0.5 * null), frequency, dish) > 0.1


@pytest.mark.unit
def test_dish_diameter_from_fov_inverts_the_field_of_view():
    frequency = 345e9
    fov = 1.12 * C / frequency / 12.0 * 180 / math.pi * 3600
    assert pbm.dish_diameter_from_fov(fov, frequency, 1.12) == pytest.approx(12.0)
    with pytest.raises(ValueError):
        pbm.dish_diameter_from_fov(0.0, frequency, 1.12)


@pytest.mark.unit
@pytest.mark.parametrize(("dish", "factor"), [(12.0, 1.1508), (7.0, 1.1433), (10.0, 1.0290)])
def test_the_primary_beam_width_is_the_airy_half_power_width(dish, factor):
    """One PB/FOV factor everywhere (T2.1): the model's half-power width in lambda / D."""
    assert pbm.airy_fwhm_factor(dish) == pytest.approx(factor, abs=1e-4)
    frequency = 230e9
    radius = float(pbm.primary_beam_fwhm_rad(dish, frequency)) / 2
    assert pbm.primary_beam_response(np.array(radius), frequency, dish) == pytest.approx(
        0.5, abs=1e-6
    )


@pytest.mark.unit
@pytest.mark.parametrize("dish", [12.0, 7.0, 10.0])
def test_dish_diameter_from_fov_inverts_the_primary_beam_width(dish):
    frequency = 345e9
    fov = float(pbm.primary_beam_fwhm_rad(dish, frequency)) * 180 / math.pi * 3600
    assert pbm.dish_diameter_from_fov(fov, frequency) == pytest.approx(dish)


@pytest.mark.unit
@pytest.mark.parametrize("dish", [12.0, 7.0])
def test_parameter_files_from_before_the_airy_factor_still_give_their_dish(dish):
    """Older params CSVs store a 1.12 lambda/D field of view; pbcor must still get 12 or 7 m."""
    frequency = 107.6e9
    legacy_fov = 1.12 * C / frequency / dish * 180 / math.pi * 3600
    assert pbm.dish_diameter_from_fov(legacy_fov, frequency) == dish


@pytest.mark.unit
def test_primary_beam_correct_cube_per_channel(tmp_path):
    image = tmp_path / "cube.fits"
    # 100 and 150 GHz, 1 arcsec pixels: the 12 m beam's FWHM is ~60" at 100 GHz.
    _write_image(image, channels=2, size=129, frequency=100e9, step=50e9, beams=True)
    info = pbm.primary_beam_correct(image, tmp_path / "pb.fits", tmp_path / "pbcor.fits", 12.0)

    with fits.open(tmp_path / "pb.fits") as pb_hdul, fits.open(tmp_path / "pbcor.fits") as cor:
        pb = pb_hdul[0].data
        corrected = cor[0].data
        assert pb_hdul[0].header["PBDISH"] == 12.0 and pb_hdul[0].header["BUNIT"] == ""
        assert cor[0].header["BUNIT"] == "JY/BEAM" and cor[0].header["PBCUTOFF"] == 0.2
        assert [hdu.name for hdu in cor] == ["PRIMARY", "BEAMS"]
    centre = 64
    assert pb[0, 0, centre, centre] == pytest.approx(1.0)
    # 30" off axis, at the half-power radius at 100 GHz; lower at 150 GHz.
    off = pb[0, :, centre, centre + 30]
    assert 0.45 < off[0] < 0.55 and off[1] < off[0]
    expected = pbm.primary_beam_response(np.radians(30 / 3600), 100e9, 12.0)
    assert off[0] == pytest.approx(expected, rel=1e-3)
    assert corrected[0, 0, centre, centre + 30] == pytest.approx(2.0 / off[0], rel=1e-5)
    corner = pb[0, 0, 0, 0]
    assert corner < 0.2 and np.isnan(corrected[0, 0, 0, 0])
    assert info["channels"] == 2 and 0 < info["blanked_fraction"] < 1
    assert info["model"] == "Airy 10.7 m aperture, 0.75 m blockage"
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.unit
def test_primary_beam_correct_rejects_bad_input(tmp_path):
    image = tmp_path / "image.fits"
    _write_image(image)
    with pytest.raises(ValueError, match="cutoff"):
        pbm.primary_beam_correct(image, tmp_path / "pb.fits", tmp_path / "c.fits", 12.0, 1.5)
    no_freq = tmp_path / "nofreq.fits"
    fits.PrimaryHDU(np.zeros((1, 4, 4), np.float32)).writeto(no_freq)
    with fits.open(no_freq, mode="update") as hdul:
        hdul[0].header.update(CTYPE1="RA---SIN", CTYPE2="DEC--SIN", CTYPE3="STOKES")
        hdul[0].header.update(CRPIX1=2, CRPIX2=2, CDELT1=-1e-4, CDELT2=1e-4)
    with pytest.raises(ValueError, match="FREQ"):
        pbm.primary_beam_correct(no_freq, tmp_path / "pb.fits", tmp_path / "c.fits", 12.0)
    empty = tmp_path / "empty.fits"
    fits.PrimaryHDU().writeto(empty)
    with pytest.raises(ValueError, match="no image"):
        pbm.primary_beam_correct(empty, tmp_path / "pb.fits", tmp_path / "c.fits", 12.0)


@pytest.mark.unit
def test_pb_correct_task_corrects_what_is_there(tmp_path):
    _write_image(tmp_path / imaging_products.IMAGE_FILENAME)
    corrected = imaging_task.pb_correct_task(tmp_path, 7.0)
    assert list(corrected) == [imaging_products.IMAGE_FILENAME]
    assert (tmp_path / imaging_products.PB_FILENAME).is_file() and (
        tmp_path / imaging_products.PBCOR_FILENAME
    ).is_file()
    assert not (tmp_path / imaging_products.CUBE_PB_FILENAME).exists()
    note = imaging_task.pb_note(corrected[imaging_products.IMAGE_FILENAME])
    assert "Airy 6.25 m aperture" in note and "(7 m dish)" in note and "blanked below 0.2" in note

    _write_image(tmp_path / imaging_products.CUBE_IMAGE_FILENAME, channels=3)
    assert list(imaging_task.pb_correct_task(tmp_path, 12.0, cutoff=0.5)) == [
        imaging_products.IMAGE_FILENAME,
        imaging_products.CUBE_IMAGE_FILENAME,
    ]
    assert fits.getheader(tmp_path / imaging_products.CUBE_PBCOR_FILENAME)["PBCUTOFF"] == 0.5


@pytest.mark.unit
def test_run_wsclean_task_primary_beam_corrects_and_logs(tmp_path, monkeypatch):
    out = tmp_path / "ms" / "SPW-1"
    stale = out / imaging_products.PBCOR_FILENAME
    out.mkdir(parents=True)
    stale.write_bytes(b"old")
    seen = {}

    def _fake_run_and_mark(task, **kwargs):
        seen["stale_left"] = stale.exists()
        seen.update(kwargs)
        return {"ok": True}

    patch_everywhere(monkeypatch, imaging_task, "_run_and_mark", _fake_run_and_mark)
    imaging_task.run_wsclean_task(
        command=["wsclean", "-name", str(out / "wsclean"), "in.ms"],
        output_dir=str(out),
        ms_path="in.ms",
        spw=1,
        threads=1,
        pb_dish_diameter=12.0,
    )
    assert seen["pb_dish_diameter"] == 12.0
    assert seen["stale_left"] is False, "a rerun removes the previous run's corrected image"


def _mark_inputs(tmp_path: Path) -> dict:
    out = tmp_path / "SPW-1"
    out.mkdir()
    log = imaging_markers.imaging_log_path(out)
    log.write_text("")
    return dict(
        task=TaskRun(out=out, log_path=log, ms_path="in.ms", spw=1),
        commands=[["wsclean", "-name", str(out / "wsclean"), "in.ms"]],
        required=[imaging_products.IMAGE_FILENAME],
        started=0.0,
    )


@pytest.mark.unit
def test_run_and_mark_writes_the_primary_beam_products(tmp_path, monkeypatch):
    kwargs = _mark_inputs(tmp_path)
    out = kwargs["task"].out
    patch_everywhere(
        monkeypatch,
        imaging_wsclean_runner,
        "_run_with_retries",
        lambda cmd, out_, *a: _write_image(out_ / imaging_products.IMAGE_FILENAME) or 1,
    )
    imaging_task._run_and_mark(**kwargs, pb_dish_diameter=12.0)
    assert (out / imaging_products.PB_FILENAME).is_file() and (
        out / imaging_products.PBCOR_FILENAME
    ).is_file()
    assert imaging_markers.imaging_marker_path(out).is_file()
    log = kwargs["task"].log_path.read_text()
    assert "# primary-beam corrected wsclean-image.fits: Airy 10.7 m" in log


@pytest.mark.unit
def test_run_and_mark_fails_the_task_when_the_correction_fails(tmp_path, monkeypatch):
    kwargs = _mark_inputs(tmp_path)
    out = kwargs["task"].out
    patch_everywhere(
        monkeypatch,
        imaging_wsclean_runner,
        "_run_with_retries",
        lambda cmd, out_, *a: (out_ / imaging_products.IMAGE_FILENAME).write_bytes(b"x"),
    )
    with pytest.raises(RuntimeError, match="Could not primary-beam correct"):
        imaging_task._run_and_mark(**kwargs, pb_dish_diameter=12.0)
    assert not imaging_markers.imaging_marker_path(out).exists()


def _params(tmp_path: Path, rows: list[dict]) -> Path:
    csv = tmp_path / "params.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def _param_row(spw: int, dish: float = 12.0, **overrides) -> dict:
    frequency = 100e9
    row = {
        "filename": "/data/uid___A001_X1_X1.ms.split.cal",
        "spectral_window_id": spw,
        "reference_frequency": frequency,
        "fov_per_frequency": float(pbm.primary_beam_fwhm_rad(dish, frequency))
        * 180
        / math.pi
        * 3600,
        "synthetized_beam_size": 1.0,
        "target_field_ids": "2",
    }
    row.update(overrides)
    return row


@pytest.mark.unit
def test_row_dish_diameter_and_task_directory(tmp_path):
    assert cli_image.row_dish_diameter(pd.Series(_param_row(1, dish=7.0))) == 7.0
    assert cli_image.row_dish_diameter(pd.Series({"spectral_window_id": 1})) is None
    assert cli_image.row_dish_diameter(pd.Series(_param_row(1, fov_per_frequency=0.0))) is None
    directory, fields = cli_image.task_directory(tmp_path, pd.Series(_param_row(5)))
    assert directory == tmp_path / "uid___A001_X1_X1.ms.split" / "SPW-5_FIELD-2"
    assert fields == (2,)


@pytest.mark.unit
def test_build_imaging_tasks_carries_the_dish_diameter(tmp_path):
    tasks, _ = cli_image.build_imaging_tasks(
        pd.DataFrame([_param_row(1, dish=7.0)]),
        tmp_path,
        fov_fraction=1.5,
        beam_sampling=8,
        num_cores=2,
        max_cores_per_node=8,
    )
    assert tasks[0].dish_diameter == 7.0
    assert tasks[0].output_dir == tmp_path / "uid___A001_X1_X1.ms.split" / "SPW-1_FIELD-2"


@pytest.mark.unit
@pytest.mark.parametrize(("flags", "expected"), [([], False), (["--pbcor"], True)])
def test_image_from_ms_pbcor_is_opt_in(tmp_path, monkeypatch, flags, expected):
    csv = _params(tmp_path, [_param_row(1)])
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
    assert seen["pbcor"] is expected


@pytest.mark.unit
def test_pb_correct_command_corrects_finished_tasks(tmp_path):
    root = tmp_path / "out"
    csv = _params(
        tmp_path,
        [_param_row(1), _param_row(2), _param_row(3), _param_row(4, fov_per_frequency=-1.0)],
    )
    done, broken, missing_fov = (
        cli_image.task_directory(root, pd.Series(_param_row(spw)))[0] for spw in (1, 3, 4)
    )
    for directory in (done, broken, missing_fov):
        directory.mkdir(parents=True)
        imaging_markers.write_imaging_marker(directory, ms_path="x", spw=1)
    _write_image(done / imaging_products.IMAGE_FILENAME)
    _write_image(missing_fov / imaging_products.IMAGE_FILENAME)
    (broken / imaging_products.IMAGE_FILENAME).write_bytes(b"not a FITS file")
    imaging_markers.imaging_log_path(done).write_text("# imaged\n")

    result = runner.invoke(cli.app, ["image", "pb-correct", str(csv), str(root), "--cutoff", "0.3"])

    assert result.exit_code == 1, result.output
    assert "Primary-beam corrected 1 task(s); 1 not imaged, 2 failed." in result.output
    assert "no fov_per_frequency" in result.output
    assert fits.getheader(done / imaging_products.PBCOR_FILENAME)["PBCUTOFF"] == 0.3
    log = imaging_markers.imaging_log_path(done).read_text()
    assert log.startswith("# imaged\n") and "(pb-correct): Airy 10.7 m" in log


@pytest.mark.unit
def test_pb_correct_command_succeeds_when_everything_is_corrected(tmp_path):
    root = tmp_path / "out"
    csv = _params(tmp_path, [_param_row(1)])
    directory = cli_image.task_directory(root, pd.Series(_param_row(1)))[0]
    directory.mkdir(parents=True)
    imaging_markers.write_imaging_marker(directory, ms_path="x", spw=1)
    _write_image(directory / imaging_products.IMAGE_FILENAME)
    result = runner.invoke(cli.app, ["image", "pb-correct", str(csv), str(root)])
    assert result.exit_code == 0, result.output
    assert (directory / imaging_products.PB_FILENAME).is_file()


def _broken(*args, **kwargs):
    raise ValueError("uneven channels")


@pytest.mark.unit
def test_run_and_mark_keeps_the_cause_of_a_stacking_failure(tmp_path, monkeypatch):
    import almasim.services.imaging.task as task

    kwargs = _mark_inputs(tmp_path)
    kwargs["required"] = [imaging_products.CUBE_IMAGE_FILENAME]
    patch_everywhere(monkeypatch, imaging_wsclean_runner, "_run_with_retries", lambda *a: 1)
    monkeypatch.setattr(task, "stack_channel_images", _broken)
    with pytest.raises(RuntimeError, match="Could not stack the channel images") as info:
        imaging_task._run_and_mark(**kwargs)
    assert isinstance(info.value.__cause__, ValueError)


@pytest.mark.unit
def test_run_and_mark_keeps_the_cause_of_a_cube_continuum_failure(tmp_path, monkeypatch):
    import almasim.services.imaging.task as task

    kwargs = _mark_inputs(tmp_path)
    kwargs["required"] = [imaging_products.CUBE_IMAGE_FILENAME]
    patch_everywhere(monkeypatch, imaging_wsclean_runner, "_run_with_retries", lambda *a: 1)
    monkeypatch.setattr(
        task,
        "stack_channel_images",
        lambda out, prefix, keep, provenance=None: (
            (out / imaging_products.CUBE_IMAGE_FILENAME).write_bytes(b"x") and {}
        ),
    )
    monkeypatch.setattr(task, "subtract_continuum", _broken)
    with pytest.raises(
        RuntimeError, match="Could not subtract the continuum from the cube"
    ) as info:
        imaging_task._run_and_mark(**kwargs, contsub_order=1)
    assert isinstance(info.value.__cause__, ValueError)


@pytest.mark.unit
def test_task_run_fail_writes_the_failure_marker(tmp_path):
    log = tmp_path / "task.log"
    task = TaskRun(out=tmp_path, log_path=log, ms_path="in.ms", spw=3, retries=-2)

    error = task.fail("WSClean exited with return code 1", 1)

    assert task.retries == 0
    assert str(error) == f"WSClean exited with return code 1\nFull log: {log}"
    assert imaging_markers.imaging_failure_marker_path(tmp_path).is_file()
