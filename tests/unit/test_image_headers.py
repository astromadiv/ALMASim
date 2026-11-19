"""Unit tests for the tclean-header geometry reader."""

from __future__ import annotations

from pathlib import Path

import pytest
from astropy.io import fits

from almasim.services.imaging.image_headers import (
    image_geometry,
    parse_tclean_parameters,
    read_fits_header,
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
        "weighting   = \"briggs\"",
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
