"""Unit tests for external skymodel ingestion."""

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from almasim.services.external_skymodel import (
    infer_external_cube_geometry,
    is_external_source_type,
    load_external_sky_model,
    read_cube_wcs,
)


def test_is_external_source_type():
    assert is_external_source_type("external-fits-image") is True
    assert is_external_source_type("external-fits-cube") is True
    assert is_external_source_type("external-components") is True
    assert is_external_source_type("point") is False


def test_infer_external_cube_geometry_from_fits_cube(tmp_path):
    cube = np.ones((4, 8, 8), dtype=np.float32)
    path = tmp_path / "input_cube.fits"
    fits.PrimaryHDU(data=cube).writeto(path)

    geometry = infer_external_cube_geometry(
        source_type="external-fits-cube",
        skymodel_path=str(path),
    )

    assert geometry == {
        "n_channels": 4,
        "n_pix_y": 8,
        "n_pix_x": 8,
        "n_pix": 8,
    }


def test_load_external_fits_image_aligns_to_observation_shape(tmp_path):
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    path = tmp_path / "input_image.fits"
    fits.PrimaryHDU(data=image).writeto(path)

    payload = load_external_sky_model(
        source_type="external-fits-image",
        skymodel_path=str(path),
        target_npix=8,
        target_nchannels=3,
        alignment_mode="observation",
        header_mode="observation",
        target_header=fits.Header(),
    )

    assert payload.cube.shape == (3, 8, 8)
    assert payload.metadata["input_kind"] == "fits"
    assert payload.metadata["original_shape"] == [1, 4, 4]


def test_load_external_component_table_builds_channel_cube(tmp_path):
    table_path = tmp_path / "components.ecsv"
    table = Table(
        {
            "x_pix": [2, 4],
            "y_pix": [3, 1],
            "flux_jy": [1.5, 2.0],
            "spectral_index": [0.0, 1.0],
            "reference_freq_hz": [100.0, 100.0],
        }
    )
    table.write(table_path, format="ascii.ecsv")

    payload = load_external_sky_model(
        source_type="external-components",
        component_table_path=str(table_path),
        target_npix=8,
        target_nchannels=4,
        channel_frequencies_hz=np.array([100.0, 110.0, 120.0, 130.0]),
        target_header=fits.Header(),
    )

    assert payload.cube.shape == (4, 8, 8)
    assert np.all(payload.cube[:, 3, 2] > 0.0)
    assert payload.cube[0, 1, 4] < payload.cube[-1, 1, 4]


def _wsclean_cube(path, *, channels=4, descending=False):
    """A cube as WSClean writes it: RA, Dec, FREQ, STOKES (numpy: S, F, Y, X)."""
    data = np.zeros((1, channels, 6, 10), dtype=np.float32)
    for k in range(channels):
        data[0, k] = k  # each plane its channel number
    header = fits.Header()
    for key, value in {
        "CTYPE1": "RA---SIN",
        "CDELT1": -0.5 / 3600,
        "CTYPE2": "DEC--SIN",
        "CDELT2": 0.5 / 3600,
        "CTYPE3": "FREQ",
        "CRPIX3": 1.0,
        "CRVAL3": 100e9,
        "CDELT3": -2e6 if descending else 2e6,
        "CTYPE4": "STOKES",
    }.items():
        header[key] = value
    fits.PrimaryHDU(data, header=header).writeto(path)
    return path


def test_a_cube_s_grid_comes_from_its_header(tmp_path):
    wcs = read_cube_wcs(_wsclean_cube(tmp_path / "c.fits"))
    assert (wcs.n_pix_x, wcs.n_pix_y, wcs.n_pix, wcs.n_channels) == (10, 6, 10, 4)
    assert wcs.channel_width_hz == 2e6 and not wcs.descending
    assert wcs.central_freq_hz == pytest.approx(100e9 + 1.5 * 2e6)  # the middle of 4 channels
    assert wcs.cell_arcsec == pytest.approx(0.5)


def test_a_stokes_axis_is_dropped_and_channels_run_up(tmp_path):
    path = _wsclean_cube(tmp_path / "down.fits", descending=True)
    assert read_cube_wcs(path).descending
    payload = load_external_sky_model(
        source_type="external-fits-cube",
        skymodel_path=str(path),
        target_npix=10,
        target_nchannels=4,
        alignment_mode="preserve",
    )
    # 4D in, (chan, y, x) out, reversed: the first plane is now the lowest frequency.
    assert payload.cube.shape[0] == 4
    assert [float(payload.cube[k].max()) for k in range(4)] == [3.0, 2.0, 1.0, 0.0]


def test_blanked_pixels_hold_no_emission(tmp_path):
    cube = np.ones((3, 4, 4), dtype=np.float32)
    cube[0] = np.nan  # a flagged channel, as WSClean writes it
    path = tmp_path / "blanked.fits"
    fits.PrimaryHDU(cube).writeto(path)
    payload = load_external_sky_model(
        source_type="external-fits-cube", skymodel_path=str(path), alignment_mode="preserve"
    )
    assert np.isfinite(payload.cube).all() and payload.cube[0].max() == 0.0
    assert payload.metadata["blank_pixels"] == 16
