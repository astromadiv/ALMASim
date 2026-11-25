"""Unit tests for the tclean-header geometry reader."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
from astropy.io import fits

from almasim.services.imaging import image_headers
from almasim.services.imaging.image_headers import (
    image_geometry,
    match_member_images,
    parse_phasecenter,
    parse_tclean_parameters,
    read_fits_header,
)
from almasim.services.imaging.wsclean_plan import (
    CubeOptions,
    _format_dec,
    _format_ra,
    cube_channels,
    imaging_parameter_to_command_arg,
)


def _header() -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 4
    header["NAXIS1"] = 128
    header["NAXIS2"] = 128
    header["NAXIS3"] = 16
    header["NAXIS4"] = 1
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CTYPE3"] = "FREQ"
    header["CDELT1"] = -1.0e-5
    header["CDELT2"] = 1.0e-5
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = 20.0
    header["CRVAL3"] = 1.0e9
    header["CDELT3"] = 1.0e6
    header["BMAJ"] = 2.0e-5
    header["BMIN"] = 1.5e-5
    header["BPA"] = 30.0
    history = [
        "CASA START LOGTABLE",
        "taskname=tclean",
        "imsize      = [128, 128]",
        "cell        = ['0.036arcsec']",
        'phasecenter = "ICRS 10:00:00.0 20:00:00.0"',
        "specmode    = cube",
        "nchan       = 16",
        'weighting   = "briggs"',
        "robust      = 0.5",
        "stokes      = I",
        "deconvolver = hogbom",
    ]
    for line in history:
        header.add_history(line)
    return header


def test_parse_tclean_parameters(tmp_path: Path) -> None:
    header = _header()
    params = parse_tclean_parameters(header)
    assert params["imsize"] == [128, 128]
    assert params["cell"] == ["0.036arcsec"]
    assert params["specmode"] == "cube"
    assert params["weighting"] == "briggs"


def test_image_geometry(tmp_path: Path) -> None:
    header = _header()
    geometry = image_geometry(header)
    assert geometry["image_n_pixels"] == 128
    assert geometry["image_cell_arcsec"] == pytest.approx(0.036, rel=1e-3)
    assert geometry["image_n_channels"] == 16
    assert geometry["phasecenter_ra_deg"] == pytest.approx(150.0, abs=1e-6)
    assert geometry["phasecenter_dec_deg"] == pytest.approx(20.0, abs=1e-6)
    assert geometry["image_weighting"] == "briggs"
    assert geometry["image_robust"] == 0.5
    assert geometry["image_stokes"] == "I"
    assert geometry["image_specmode"] == "cube"


def test_read_fits_header_round_trip(tmp_path: Path) -> None:
    header = _header()
    path = tmp_path / "test.hdr"
    text = header.tostring(padding=True)
    raw = text.encode("ascii")
    # pad to block boundary
    pad = (-len(raw)) % 2880
    path.write_bytes(raw + b" " * pad)
    read_back = read_fits_header(path)
    assert int(read_back["NAXIS1"]) == 128
    assert read_back.get("BMAJ") == pytest.approx(2.0e-5)


def test_dec_keeps_the_decimals_of_its_seconds() -> None:
    ra, dec = parse_phasecenter("ICRS 14:13:09.9473 -065.20.21.024")
    assert ra == pytest.approx((14 + 13 / 60 + 9.9473 / 3600) * 15.0, abs=1e-9)
    assert dec == pytest.approx(-(65 + 20 / 60 + 21.024 / 3600), abs=1e-9)


def test_a_moving_target_has_no_phase_centre() -> None:
    header = _header()
    del header["HISTORY"]
    header.add_history("phasecenter = 'TRACKFIELD'")
    geometry = image_geometry(header)
    assert math.isnan(geometry["phasecenter_ra_deg"])
    assert math.isnan(geometry["phasecenter_dec_deg"])


def test_image_geometry_of_a_non_square_or_gridless_header() -> None:
    header = _header()
    header["NAXIS2"] = 200
    assert image_geometry(header)["image_n_pixels"] == 200
    for card in ("NAXIS1", "NAXIS2", "CDELT1", "CDELT2"):
        del header[card]
    geometry = image_geometry(header)
    assert math.isnan(geometry["image_n_pixels"])
    assert math.isnan(geometry["image_cell_arcsec"])


def test_history_wraps_are_rejoined() -> None:
    header = fits.Header()
    header.add_history("phasecenter = 'ICRS 14:13:09.9473 -065.")
    header.add_history(">20.21.024'")
    assert parse_tclean_parameters(header)["phasecenter"] == "ICRS 14:13:09.9473 -065.20.21.024"


def test_selfcal_image_is_preferred(tmp_path: Path) -> None:
    names = [
        "member.uid___A001_X1_X2.NGC1_sci.spw25.cube.I.pbcor.fits",
        "member.uid___A001_X1_X2.NGC1_sci.spw25.cube.regcal.I.pbcor.fits",
        "member.uid___A001_X1_X2.NGC1_sci.spw25.cube.I.selfcal.pbcor.fits",
        "member.uid___A001_X1_X2.NGC1_sci.spw25_27.mfs.I.pbcor.fits",
        "member.uid___A001_X1_X2.NGC1_sci.spw27.cube.I.pbcor.fits",
    ]
    raw = _header().tostring(padding=True).encode("ascii")
    for name in names:
        (tmp_path / name).write_bytes(raw)
    images = match_member_images(tmp_path, "NGC1", 25)
    assert [(image.kind, image.filename) for image in images] == [
        ("mfs", names[3]),
        ("cube", names[2]),
    ]


def test_archive_images_are_of_the_field_and_asked_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    listing = (
        ("member.uid___A001_X1_X2.G14.11-0.57_sci.spw25.cube.I.pbcor.fits", "https://x/a"),
        ("member.uid___A001_X1_X2.NGC_1068_sci.spw25.cube.I.pbcor.fits", "https://x/b"),
        ("member.uid___A001_X1_X2.NGC_1068_sci.spw25.mfs.I.pbcor.fits", "https://x/c"),
    )

    def fake(member: str, include: tuple[str, ...], exclude: tuple[str, ...]):
        calls.append(member)
        return listing

    monkeypatch.setattr(image_headers, "_archive_science_images", fake)
    images = image_headers.archive_member_images("uid://A001/X1/X2", "NGC_1068", 25)
    assert [(image.kind, image.url) for image in images] == [
        ("mfs", "https://x/c"),
        ("cube", "https://x/b"),
    ]
    assert image_headers.archive_member_images("uid://A001/X1/X2", "G14.11-0.57", 25)[0].url == (
        "https://x/a"
    )
    assert image_headers.archive_member_images("uid://A001/X1/X2", "Other", 25) == []


def _row(**overrides: object) -> pd.Series:
    row: dict[str, object] = {
        "spectral_window_id": 25,
        "fov_per_frequency": 10.0,
        "synthetized_beam_size": 1.0,
        "reference_frequency": 230.0e9,
        "n_channels": 1920,
        "channel_width_hz": 976562.5,
        "spectral_resolution_hz": 976562.5,
        "total_bandwidth_hz": 1.875e9,
    }
    row.update(overrides)
    return pd.Series(row, dtype=object)


def test_a_row_without_image_columns_keeps_the_ms_derived_command() -> None:
    plain = imaging_parameter_to_command_arg(_row(), 1.0, 4.0)
    empty = imaging_parameter_to_command_arg(
        _row(image_n_pixels=math.nan, image_cell_arcsec=math.nan, image_weighting="",
             image_robust=math.nan, image_niter=math.nan, image_threshold_jy=math.nan),
        1.0,
        4.0,
    )  # fmt: skip
    assert plain == empty
    assert plain[:5] == ["-scale", "0.25asec", "-size", "40", "40"]
    assert plain[plain.index("-weight") + 1 :][:2] == ["briggs", "0.5"]
    assert plain[plain.index("-niter") + 1] == "100000"
    assert "-abs-threshold" not in plain and "-shift" not in plain


def test_image_columns_replace_the_ms_derived_geometry() -> None:
    args = imaging_parameter_to_command_arg(
        _row(
            image_n_pixels=250.0,
            image_cell_arcsec=0.036,
            image_weighting="briggsbwtaper",
            image_robust=2.0,
            image_niter=30000.0,
            image_threshold_jy=0.000412,
            phasecenter_ra_deg=213.2914,
            phasecenter_dec_deg=-65.3392,
            ms_phasecenter_ra_deg=213.2914,
            ms_phasecenter_dec_deg=-65.3400,
        ),
        1.0,
        4.0,
    )
    assert args[:5] == ["-scale", "0.036asec", "-size", "250", "250"]
    assert args[args.index("-weight") + 1 :][:2] == ["briggs", "2.0"]
    assert args[args.index("-niter") + 1] == "30000"
    assert args[args.index("-abs-threshold") + 1] == "0.000412Jy"
    assert args[args.index("-shift") + 1 :][:2] == ["14:13:09.936", "-65.20.21.12"]


@pytest.mark.parametrize(
    ("weighting", "robust", "expected"),
    [
        ("natural", 0.5, ["natural"]),
        ("uniform", math.nan, ["uniform"]),
        ("briggs", 0.0, ["briggs", "0.0"]),
        ("superuniform", 1.0, ["briggs", "0.5"]),
    ],
)
def test_weighting_is_one_wsclean_takes(weighting: str, robust: float, expected: list[str]) -> None:
    args = imaging_parameter_to_command_arg(
        _row(image_weighting=weighting, image_robust=robust), 1.0, 4.0
    )
    following = args[args.index("-weight") + 1 :]
    assert following[: len(expected)] == expected
    assert not following[len(expected) :] or following[len(expected)].startswith("-")


def test_an_uncleaned_image_does_not_stop_the_clean() -> None:
    args = imaging_parameter_to_command_arg(_row(image_niter=0.0), 1.0, 4.0)
    assert args[args.index("-niter") + 1] == "100000"


def test_no_shift_for_the_same_centre_across_the_ra_wrap() -> None:
    centre = {"phasecenter_dec_deg": 0.0, "ms_phasecenter_dec_deg": 0.0}
    same = _row(phasecenter_ra_deg=359.9999, ms_phasecenter_ra_deg=-0.0001, **centre)
    assert "-shift" not in imaging_parameter_to_command_arg(same, 1.0, 4.0)
    moved = _row(phasecenter_ra_deg=359.999, ms_phasecenter_ra_deg=0.001, **centre)
    args = imaging_parameter_to_command_arg(moved, 1.0, 4.0)
    assert args[args.index("-shift") + 1 :][:2] == ["23:59:59.760", "+00.00.00.00"]


def test_sexagesimal_seconds_carry() -> None:
    assert _format_ra(15.0 - 1e-7) == "01:00:00.000"
    assert _format_dec(-(10.0 - 1e-7)) == "-10.00.00.00"
    assert _format_ra(360.0 - 1e-9) == "00:00:00.000"


def test_cube_channels_take_the_averaging_of_the_pipeline_cube() -> None:
    options = CubeOptions(velocity_resolution_km_s=10.0)
    assert cube_channels(_row(), 100, options) < 480
    # tclean regrids to LSRK: 1917 of 1920 channels is the native grid, 958 is pairs.
    cube = {"image_specmode": "cube"}
    assert cube_channels(_row(image_n_channels=1917.0, **cube), 100, options) == 1920
    assert cube_channels(_row(image_n_channels=958.0, **cube), 100, options) == 960
    assert cube_channels(_row(image_n_channels=4000.0, **cube), 100, options) == 1920
    # An mfs header says nothing about a cube, and the size cap still holds.
    mfs = _row(image_n_channels=1.0, image_specmode="mfs")
    assert cube_channels(mfs, 100, options) == cube_channels(_row(), 100, options)
    capped = cube_channels(
        _row(image_n_channels=1917.0, **cube), 2000, CubeOptions(max_cube_gb=1.0)
    )
    assert capped == 60
