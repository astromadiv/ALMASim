"""The primary beams of downloaded ALMA images, and the images before correction."""

from __future__ import annotations

import gzip
import os
import shutil

import numpy as np
import pytest
from astropy.io import fits

from almasim.services.download import DataProduct
from almasim.services.download.primary_beams import (
    uncorrected_image,
    with_primary_beams,
    write_uncorrected_images,
)


def _row(filename: str, path: str = "p/member.x/product") -> DataProduct:
    return DataProduct(
        access_url=f"https://example.org/{filename}",
        uid="uid://A",
        filename=filename,
        content_length=5,
        content_type="",
        product_type="fits",
        semantics="#this",
        path=path,
    )


@pytest.mark.unit
def test_each_picked_pbcor_image_brings_its_primary_beam():
    listing = [
        _row("m.cube.I.pbcor.fits"),
        _row("m.cube.I.pb.fits.gz"),
        _row("m.cube.I.mask.fits.gz"),
        _row("m.mfs.I.pbcor.fits"),
        _row("m.mfs.I.pb.fits.gz"),
        _row("m.cal.I.pbcor.fits"),  # no primary beam listed
        _row("m.cube.I.pb.fits.gz", path="other/member.y/product"),  # another member's
    ]
    picked = [listing[0], listing[4], listing[5]]  # the mfs beam already picked
    names = [(r.path, r.filename) for r in with_primary_beams(picked, listing)]
    assert names == [
        ("p/member.x/product", "m.cube.I.pbcor.fits"),
        ("p/member.x/product", "m.mfs.I.pb.fits.gz"),
        ("p/member.x/product", "m.cal.I.pbcor.fits"),
        ("p/member.x/product", "m.cube.I.pb.fits.gz"),
    ]


def _cube(seed: int = 0, shape=(1, 3, 6, 5)) -> tuple[np.ndarray, np.ndarray, fits.Header]:
    rng = np.random.default_rng(seed)
    image = rng.normal(size=shape).astype(np.float32)
    y, x = np.indices(shape[-2:])
    pb = np.exp(-((x - 2) ** 2 + (y - 3) ** 2) / 8.0).astype(np.float32)
    pb = np.broadcast_to(pb, shape).copy()
    pbcor = np.where(pb >= 0.2, image / pb, np.nan).astype(np.float32)
    header = fits.Header()
    header["BUNIT"] = "Jy/beam"
    header["OBJECT"] = "target"
    return pbcor, pb, header


def _write(path, data, header=None, extensions=()):
    fits.HDUList([fits.PrimaryHDU(data, header=header), *extensions]).writeto(path)


@pytest.mark.unit
def test_the_image_before_correction_is_pbcor_times_pb_plane_by_plane(tmp_path):
    """Gzipped beam, a cube's BEAMS table kept, blanked pixels kept blank."""
    pbcor, pb, header = _cube()
    beams = fits.BinTableHDU.from_columns(
        [fits.Column(name="BMAJ", format="E", array=np.ones(3, dtype=np.float32))],
        name="BEAMS",
    )
    _write(tmp_path / "m.cube.I.pbcor.fits", pbcor, header, [beams])
    _write(tmp_path / "beam.fits", pb)
    with (
        open(tmp_path / "beam.fits", "rb") as src,
        gzip.open(tmp_path / "m.cube.I.pb.fits.gz", "wb") as dst,
    ):
        shutil.copyfileobj(src, dst)

    out = uncorrected_image(
        tmp_path / "m.cube.I.pbcor.fits",
        tmp_path / "m.cube.I.pb.fits.gz",
        tmp_path / "m.cube.I.image.fits",
    )
    with fits.open(out) as hdul:
        assert hdul[0].data.dtype == np.dtype(">f4")
        np.testing.assert_allclose(hdul[0].data, pbcor * pb, rtol=1e-6)
        assert np.array_equal(np.isnan(hdul[0].data), np.isnan(pbcor))
        assert hdul[0].header["BUNIT"] == "Jy/beam" and hdul[0].header["OBJECT"] == "target"
        history = " ".join(" ".join(map(str, hdul[0].header["HISTORY"])).split())
        assert "the image before primary-beam correction" in history
        assert hdul["BEAMS"].data["BMAJ"].tolist() == [1.0, 1.0, 1.0]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "beam.fits",
        "m.cube.I.image.fits",
        "m.cube.I.pb.fits.gz",
        "m.cube.I.pbcor.fits",
    ]  # the unpacked beam and the .partial file are gone


@pytest.mark.unit
def test_a_beam_of_another_shape_writes_nothing(tmp_path):
    pbcor, pb, header = _cube()
    _write(tmp_path / "m.pbcor.fits", pbcor, header)
    _write(tmp_path / "m.pb.fits", pb[:, :2])
    with pytest.raises(ValueError, match="primary beam is"):
        uncorrected_image(
            tmp_path / "m.pbcor.fits", tmp_path / "m.pb.fits", tmp_path / "m.image.fits"
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.pb.fits", "m.pbcor.fits"]


@pytest.mark.unit
def test_uncorrected_images_are_written_once_and_redone_when_an_input_changes(tmp_path):
    folder = tmp_path / "p" / "product"
    folder.mkdir(parents=True)
    pbcor, pb, header = _cube()
    _write(folder / "a.mfs.I.pbcor.fits", pbcor, header)
    _write(folder / "a.mfs.I.pb.fits", pb)
    _write(folder / "b.mfs.I.pbcor.fits", pbcor, header)  # no primary beam
    _write(folder / "c.mfs.I.pbcor.fits", pbcor, header)
    _write(folder / "c.mfs.I.pb.fits", pb[:, :1])  # wrong shape
    files = [
        f"p/product/{n}" for n in ("a.mfs.I.pbcor.fits", "b.mfs.I.pbcor.fits", "c.mfs.I.pbcor.fits")
    ]
    files.append("p/product/a.mfs.I.pb.fits")  # not a pbcor image: ignored

    first = write_uncorrected_images(tmp_path, files)
    assert first.written == ["p/product/a.mfs.I.image.fits"]
    assert first.without_beam == ["p/product/b.mfs.I.pbcor.fits"]
    assert list(first.failed) == ["p/product/c.mfs.I.image.fits"]

    again = write_uncorrected_images(tmp_path, files)
    assert again.written == [] and again.present == ["p/product/a.mfs.I.image.fits"]

    image = folder / "a.mfs.I.image.fits"
    stamp = image.stat().st_mtime
    os.utime(folder / "a.mfs.I.pb.fits", (stamp + 10, stamp + 10))  # a newer primary beam
    assert write_uncorrected_images(tmp_path, files).written == ["p/product/a.mfs.I.image.fits"]
