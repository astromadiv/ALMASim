"""The primary beams of downloaded ALMA images, and the images before correction.

An ALMA delivery's product tar holds, for each cleaned image, the primary-beam
corrected image ``<stem>.pbcor.fits`` and the primary beam itself,
``<stem>.pb.fits.gz`` (the response, 1 at the pointing centre, on the image's
grid). Their product, ``<stem>.image.fits``, is the image before primary-beam
correction: its noise is flat across the field, as in the ``.image`` the ALMA
pipeline cleaned.
"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .models import DataProduct

logger = logging.getLogger(__name__)

PBCOR_SUFFIX = ".pbcor.fits"
PB_SUFFIXES = (".pb.fits.gz", ".pb.fits")
IMAGE_SUFFIX = ".image.fits"
_FITS_BLOCK = 2880


def image_stem(filename: str) -> str | None:
    """``<stem>`` of a ``<stem>.pbcor.fits`` name; None for any other file."""
    if filename.endswith(PBCOR_SUFFIX):
        return filename[: -len(PBCOR_SUFFIX)]
    return None


def with_primary_beams(
    selected: Sequence[DataProduct], listing: Iterable[DataProduct]
) -> list[DataProduct]:
    """``selected`` plus, for each ``*.pbcor.fits`` in it, its primary beam from ``listing``.

    The primary beam is the ``<stem>.pb.fits.gz`` (or ``.pb.fits``) row of the
    same folder; one already selected is not added twice.
    """
    by_place = {(row.path, row.filename): row for row in listing}
    chosen = {(row.path, row.filename) for row in selected}
    products = list(selected)
    for row in selected:
        stem = image_stem(row.filename)
        if stem is None:
            continue
        for suffix in PB_SUFFIXES:
            place = (row.path, stem + suffix)
            if place in by_place:
                if place not in chosen:
                    chosen.add(place)
                    products.append(by_place[place])
                break
    return products


def primary_beam_path(pbcor_path: Path) -> Path | None:
    """The primary beam on disk next to ``pbcor_path``, or None."""
    stem = image_stem(pbcor_path.name)
    if stem is None:
        return None
    for suffix in PB_SUFFIXES:
        candidate = pbcor_path.with_name(stem + suffix)
        if candidate.is_file():
            return candidate
    return None


def _empty_fits(path: Path, header) -> None:
    """A FITS file of ``header`` whose float32 data are all zero bytes, of full size on disk."""
    header.tofile(path, overwrite=True)
    shape = [int(header[f"NAXIS{axis}"]) for axis in range(1, int(header["NAXIS"]) + 1)]
    nbytes = int(np.prod(shape)) * 4
    padded = -(-nbytes // _FITS_BLOCK) * _FITS_BLOCK  # the data block, whole 2880-byte records
    with open(path, "rb+") as handle:
        handle.seek(len(header.tostring()) + padded - 1)
        handle.write(b"\0")


def uncorrected_image(
    pbcor_path: str | os.PathLike[str],
    pb_path: str | os.PathLike[str],
    image_path: str | os.PathLike[str],
) -> Path:
    """Write ``image_path``, the image before primary-beam correction: pbcor × pb.

    The product is taken plane by plane (one channel and Stokes at a time), so a
    cube never has to fit in memory; a gzipped primary beam is first unpacked
    next to the target and removed afterwards. The image keeps the corrected
    image's header and extensions (a cube's ``BEAMS`` table); pixels the
    correction blanked stay blank. Written through ``<image>.partial``, so a
    reader or a rerun never finds a half-written image.
    """
    from astropy.io import fits

    pbcor_path, pb_path, image_path = Path(pbcor_path), Path(pb_path), Path(image_path)
    partial = image_path.with_name(image_path.name + ".partial")
    unpacked: Path | None = None
    try:
        if pb_path.suffix == ".gz":
            unpacked = image_path.with_name(image_path.name + ".pb.partial")
            with gzip.open(pb_path, "rb") as source, open(unpacked, "wb") as target:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
        with (
            fits.open(pbcor_path, memmap=True) as corrected,
            fits.open(unpacked or pb_path, memmap=True) as beam,
        ):
            pbcor, pb = corrected[0].data, beam[0].data
            if pbcor is None or pb is None:
                raise ValueError(f"{pbcor_path.name} or {pb_path.name} holds no image")
            if pbcor.shape != pb.shape:
                raise ValueError(
                    f"{pbcor_path.name} is {pbcor.shape} but its primary beam is {pb.shape}"
                )
            header = corrected[0].header.copy()
            for key in ("BSCALE", "BZERO", "BLANK"):
                header.remove(key, ignore_missing=True)
            header["BITPIX"] = -32
            header.add_history(
                f"ALMASim: {pbcor_path.name} times {pb_path.name}, "
                "the image before primary-beam correction"
            )
            extensions = [hdu.copy() for hdu in corrected[1:]]
            _empty_fits(partial, header)
            with fits.open(partial, mode="update", memmap=True) as written:
                out = written[0].data
                # Every axis but the last two (the sky plane) is looped over.
                for index in np.ndindex(*pbcor.shape[:-2]):
                    out[index] = (pbcor[index] * pb[index]).astype(np.float32)
                written.flush()
        if extensions:
            with fits.open(partial, mode="append") as written:
                for hdu in extensions:
                    written.append(hdu)
        partial.replace(image_path)
        return image_path
    finally:
        partial.unlink(missing_ok=True)
        if unpacked is not None:
            unpacked.unlink(missing_ok=True)


@dataclass
class UncorrectedImages:
    """What :func:`write_uncorrected_images` did, as paths relative to its root."""

    written: list[str] = field(default_factory=list)
    present: list[str] = field(default_factory=list)
    """Already written, newer than both inputs: left as they are."""
    without_beam: list[str] = field(default_factory=list)
    """Corrected images with no primary beam next to them."""
    failed: dict[str, str] = field(default_factory=dict)


def write_uncorrected_images(
    root: str | os.PathLike[str],
    files: Iterable[str],
    *,
    echo: Callable[[str], None] | None = None,
) -> UncorrectedImages:
    """For each ``*.pbcor.fits`` among ``files`` (relative to ``root``) with its
    primary beam beside it, write ``<stem>.image.fits`` next to them.

    An image already newer than both its inputs is kept; one that cannot be
    written is reported in ``failed`` and the others still go ahead.
    """
    root = Path(root)
    result = UncorrectedImages()
    for relative in dict.fromkeys(files):  # once each, in order
        pbcor = root / relative
        stem = image_stem(pbcor.name)
        if stem is None or not pbcor.is_file():
            continue
        pb = primary_beam_path(pbcor)
        if pb is None:
            result.without_beam.append(relative)
            continue
        image = pbcor.with_name(stem + IMAGE_SUFFIX)
        image_relative = str(image.relative_to(root))
        if image.is_file() and image.stat().st_mtime >= max(
            pbcor.stat().st_mtime, pb.stat().st_mtime
        ):
            result.present.append(image_relative)
            continue
        try:
            uncorrected_image(pbcor, pb, image)
        except (OSError, ValueError) as exc:
            logger.warning("Could not write %s: %s", image_relative, exc)
            result.failed[image_relative] = str(exc)
            continue
        result.written.append(image_relative)
        if echo is not None:
            echo(f"Wrote {image_relative}")
    return result
