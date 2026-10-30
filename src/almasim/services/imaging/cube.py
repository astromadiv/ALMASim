"""Stack channel images into cubes; line masks from the channel images.

Moved out of archive_imaging.py (MADIV-28 module split).
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from almasim.services.imaging.products import CUBE_NAME, LINE_MASK_FILENAME
from almasim.services.imaging.wsclean.command import WSCleanCommand

if TYPE_CHECKING:
    from astropy.io import fits

_CHANNEL_FILE = re.compile(r"^(?P<prefix>.+)-(?P<channel>\d{4})-(?P<product>[a-z-]+)\.fits$")


def stack_channel_images(  # noqa: C901
    output_dir: str | os.PathLike[str],
    *,
    prefix: str = "wsclean",
    keep: Sequence[str] | None = None,
    provenance: fits.Header | None = None,
) -> dict[str, int]:
    """Stack WSClean's per-channel FITS (``-channels-out``) into one cube per product.

    WSClean writes ``<prefix>-0000-image.fits``, ``<prefix>-0001-image.fits``
    ... and no cube. Each product (image, residual, model, psf, dirty) becomes
    ``<prefix>-<product>.fits`` with a regular frequency axis, and the
    per-channel files are removed. Products not in ``keep`` (all when None)
    are removed without stacking.

    The channel beams differ (the beam scales with wavelength), so a cube
    with beam keywords also gets a CASA-style ``BEAMS`` table (one row per
    channel, BMAJ/BMIN in arcsec, BPA in degrees, ``CASAMBM = T``), which
    CASA and spectral-cube read; the primary header keeps the middle
    channel's beam.

    Each cube's header also gets the cards of ``provenance``
    (:func:`~almasim.services.imaging.provenance.copy_provenance`).

    Returns ``{product: n_channels}`` of the cubes written.
    """
    import numpy as np
    from astropy.io import fits

    from almasim.services.imaging.fits_io import ChannelBeam, read_image, write_cube
    from almasim.services.imaging.provenance import copy_provenance

    out = Path(output_dir)
    groups: dict[str, list[tuple[int, Path]]] = {}
    for path in out.iterdir():
        match = _CHANNEL_FILE.match(path.name)
        if match and match["prefix"] == prefix:
            groups.setdefault(match["product"], []).append((int(match["channel"]), path))
    written: dict[str, int] = {}
    for product, members in sorted(groups.items()):
        members.sort()
        paths = [path for _, path in members]
        if keep is not None and product not in keep:
            for path in paths:
                path.unlink(missing_ok=True)
            continue
        headers = [fits.getheader(path) for path in paths]
        first = headers[0]
        if first.get("CTYPE3", "").upper() != "FREQ":
            raise RuntimeError(f"{paths[0]} has no FREQ third axis (CTYPE3={first.get('CTYPE3')})")
        frequencies = np.array([float(h["CRVAL3"]) for h in headers])
        steps = np.diff(frequencies)
        if steps.size and not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-3):
            raise RuntimeError(f"{product} channels of {out} are not evenly spaced in frequency")
        cube = np.empty((1, len(paths), int(first["NAXIS2"]), int(first["NAXIS1"])), np.float32)
        for index, path in enumerate(paths):
            cube[0, index] = read_image(path).reshape(cube.shape[2:])
        header = first.copy()
        header["NAXIS3"] = len(paths)
        header["CRPIX3"] = 1.0
        header["CRVAL3"] = frequencies[0]
        if steps.size:
            header["CDELT3"] = float(steps[0])
        if provenance is not None:
            copy_provenance(provenance, header)
        beams = None
        if all("BMAJ" in h for h in headers):
            beams = [ChannelBeam.from_header(h) for h in headers]
        write_cube(out / f"{prefix}-{product}.fits", cube, header, beams)
        for path in paths:
            path.unlink(missing_ok=True)
        written[product] = len(paths)
    return written


def line_mask_from_channels(
    output_dir: str | os.PathLike[str],
    mask_path: str | os.PathLike[str],
    *,
    prefix: str = CUBE_NAME,
    sigma: float = 3.0,
) -> int:
    """Write the cube's moment-0 mask from WSClean's per-channel images; its pixel count.

    The channel images ``<prefix>-NNNN-image.fits`` are summed, the sum is
    thresholded at ``sigma`` times its robust noise (1.4826 x MAD) and the
    result grown by one beam (the median BMAJ of the channels). 0 (and no
    file) when the run left no channel images or nothing is above it.
    """
    import numpy as np
    from astropy.io import fits

    out = Path(output_dir)
    # A mask left by an earlier run must not survive a run that builds none.
    Path(mask_path).unlink(missing_ok=True)
    paths = sorted(
        path
        for path in out.glob(f"{prefix}-[0-9][0-9][0-9][0-9]-image.fits")
        if _CHANNEL_FILE.match(path.name)
    )
    if not paths:
        return 0
    moment = None
    beams = []
    for path in paths:
        with fits.open(path) as hdus:
            plane = np.nan_to_num(np.asarray(hdus[0].data, dtype=np.float64).squeeze())
            beams.append(float(hdus[0].header.get("BMAJ", 0.0)))
            header = hdus[0].header.copy()
        moment = plane if moment is None else moment + plane
    if moment is None:  # pragma: no cover - paths is not empty
        return 0
    from almasim.services.imaging.fits_io import write_fits_atomic
    from almasim.services.imaging.fits_wcs import beam_pixels
    from almasim.services.imaging.stats import moment_mask

    mask = moment_mask(moment, beam_pixels(header, float(np.median(beams))), sigma)
    if not mask.any():
        return 0
    write_fits_atomic(
        mask_path, [fits.PrimaryHDU(mask.astype(np.float32)[None, None], header=header)]
    )
    return int(mask.sum())


def _line_mask_pass(cube_cmd: Sequence[str], out: Path, log) -> list[str] | None:
    """The cube's second pass: ``-continue`` inside the moment-0 mask of the first.

    Each channel's ``-auto-mask`` is built from that channel alone, so the
    faint parts of a line, below 3 sigma in every channel but clear in their
    sum, are never cleaned. uid___A002_X10ded83_X6af4 SPW 16 (1121 channels
    of its line): 34 % of the line within its moment-0 mask left in the
    residual; continuing inside that mask down to the same 1 sigma left 12 %,
    with 23 % more model flux and the per-channel noise down from 135 to 107
    mJy; going below 1 sigma there diverged (2026-10-18).
    """
    mask_path = out / LINE_MASK_FILENAME
    pixels = line_mask_from_channels(out, mask_path)
    if pixels:
        blanked = _zero_non_finite_models(out)
        if blanked:
            log.write(f"# cube line mask: {blanked} channel model(s) were NaN (no data); zeroed\n")
    if not pixels:
        log.write("# cube line mask: nothing above 3 sigma in the moment-0 map; one pass\n")
        log.flush()
        return None
    log.write(f"# cube line mask: {pixels} pixels; second pass inside it\n")
    log.flush()
    # The fixed mask replaces the per-channel one.
    second = WSCleanCommand.parse(cube_cmd).without("-auto-mask")
    return second.with_option("-fits-mask", mask_path).continued().argv()


def _zero_non_finite_models(out: Path, prefix: str = CUBE_NAME) -> int:
    """Set the non-finite pixels of the per-channel model images to 0; how many files changed.

    WSClean writes a NaN model for a channel with no unflagged data, and
    ``-continue`` refuses to predict from it ("The input image contains
    non-finite values"): uid___A002_X11aaa01_X32ce SPW 16 has 128 such
    channels (2026-10-18).
    """
    import numpy as np
    from astropy.io import fits

    changed = 0
    for path in sorted(out.glob(f"{prefix}-[0-9][0-9][0-9][0-9]-model.fits")):
        with fits.open(path, mode="update") as hdus:
            data = hdus[0].data
            bad = ~np.isfinite(data)
            if bad.any():
                data[bad] = 0.0
                changed += 1
    return changed
