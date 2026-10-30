"""Cube cleaning quality: line and line-free residuals, and moment maps (MADIV-28 T4.10)."""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..continuum import line_free_channels
from ..fits_io import write_fits_atomic
from ..fits_wcs import beam_pixels, velocity_axis
from ..products import LINE_MASK_FILENAME, cube_product
from ..provenance import copy_provenance
from ..stats import moment_mask, robust_sigma
from .continuum import _plane, _ratio, _source_geometry

QUALITY_DIRNAME = "quality"
_LINE_RANGES = re.compile(
    r"(?:(\d+) of (\d+) channels line-free(?: \((\d+) in the window\))?, )?"
    r"lines in channels ([\d~;]+|none)"
)


def _logged_line_channels(task_dir: Path, n_channels: int) -> np.ndarray | None:
    """Line channels a visibility-plane continuum subtraction logged, on the cube's channels."""
    log = task_dir.parent / f"{task_dir.name}.log"
    if not log.is_file():
        return None
    matches = _LINE_RANGES.findall(log.read_text(errors="replace"))
    if not matches:
        return None
    _, usable, window, ranges = matches[-1]
    line = np.zeros(n_channels, dtype=bool)
    if ranges == "none":
        return line
    native = [tuple(map(int, part.split("~"))) for part in ranges.split(";")]
    # Cube channels average a whole number of native ones (cube_channels picks
    # divisors). The window's channel count gives that factor exactly; an
    # older log gives only the usable count, which flagged channels make
    # smaller, so it is trusted only when the cube has at least that many
    # channels (no averaging); otherwise the lines are found on the spectrum.
    if window:
        if int(window) % n_channels:
            return None
        factor = int(window) // n_channels
    elif n_channels >= max(int(usable or 0), max(end for _, end in native) + 1):
        factor = 1
    else:
        return None
    for start, end in native:
        line[start // factor : end // factor + 1] = True
    return line


@dataclass
class _ChannelSums:
    """Per-pixel sums over a cube's line and line-free channels."""

    line_image: np.ndarray
    line_residual: np.ndarray
    linefree_image: np.ndarray
    linefree_residual: np.ndarray
    peak: np.ndarray  # line channels' maximum
    weighted_velocity: np.ndarray  # line emission above 3 sigma, times velocity
    weight: np.ndarray  # line emission above 3 sigma


def _channel(planes, c: int) -> np.ndarray:
    return np.asarray(planes[c], dtype=np.float64)


def _line_channels(
    task_dir: Path, planes, residual_planes, aperture: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """The usable and line channels, and the line channels the task logged (if any).

    A channel is usable when its residual is not all zero or NaN (flagged).
    Without a logged line range, the line channels are the usable ones off the
    line-free fit of the source's spectrum (the mean inside ``aperture``).
    """
    n_channels = len(planes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN (flagged) channels
        spectrum = np.array([np.nanmean(_channel(planes, c)[aperture]) for c in range(n_channels)])
    usable = np.array(
        [bool(np.any(np.nan_to_num(_channel(residual_planes, c)))) for c in range(n_channels)]
    )
    logged = _logged_line_channels(task_dir, n_channels)
    line = (
        usable & logged
        if logged is not None
        else usable & ~line_free_channels(spectrum, usable)
        if usable.sum() > 2
        else np.zeros(n_channels, bool)
    )
    return usable, line, logged


def _channel_noise(residual_planes, usable: np.ndarray, off_source: np.ndarray) -> float:
    """Median robust noise off source of (at most about 64 of) the usable residual channels."""
    n_channels = len(residual_planes)
    noise = np.array(
        [
            robust_sigma(_channel(residual_planes, c)[off_source]) if usable[c] else math.nan
            for c in range(0, n_channels, max(n_channels // 64, 1))
        ]
    )
    return float(np.nanmedian(noise)) if np.isfinite(noise).any() else math.nan


def _sum_channels(
    planes,
    residual_planes,
    line: np.ndarray,
    linefree: np.ndarray,
    velocity: np.ndarray,
    channel_noise: float,
) -> _ChannelSums:
    shape = planes.shape[1:]
    sums = _ChannelSums(
        line_image=np.zeros(shape),
        line_residual=np.zeros(shape),
        linefree_image=np.zeros(shape),
        linefree_residual=np.zeros(shape),
        peak=np.full(shape, -np.inf),
        weighted_velocity=np.zeros(shape),
        weight=np.zeros(shape),
    )
    for c in range(len(planes)):
        if not (line[c] or linefree[c]):
            continue
        plane = np.nan_to_num(_channel(planes, c))
        residual_plane = np.nan_to_num(_channel(residual_planes, c))
        if line[c]:
            sums.line_image += plane
            sums.line_residual += residual_plane
            sums.peak = np.maximum(sums.peak, plane)
            emitting = plane > 3 * channel_noise
            sums.weighted_velocity += np.where(emitting, plane * velocity[c], 0.0)
            sums.weight += np.where(emitting, plane, 0.0)
        else:
            sums.linefree_image += plane
            sums.linefree_residual += residual_plane
    return sums


def _residual_stats(
    sums: _ChannelSums,
    line: np.ndarray,
    linefree: np.ndarray,
    line_mask: np.ndarray,
    r: np.ndarray,
) -> dict[str, float]:
    """How much of the line and line-free emission the residual still holds."""
    stats: dict[str, float] = {}
    off_source = r >= 2.5
    one = r <= 1
    if linefree.any():
        sigma = robust_sigma(sums.linefree_residual[off_source])
        stats["linefree_residual_sigma"] = _ratio(np.mean(sums.linefree_residual[r <= 0.5]), sigma)
        stats["linefree_residual_share"] = _ratio(
            np.sum(sums.linefree_residual[one]), np.sum(sums.linefree_image[one])
        )
    if line.any() and line_mask.any():
        sigma = robust_sigma(sums.line_residual[~line_mask & off_source])
        stats["line_residual_share"] = _ratio(
            np.sum(sums.line_residual[line_mask]), np.sum(sums.line_image[line_mask])
        )
        stats["line_residual_mean_sigma"] = _ratio(np.mean(sums.line_residual[line_mask]), sigma)
        stats["line_residual_peak_sigma"] = _ratio(np.max(sums.line_residual[line_mask]), sigma)
    return stats


def _model_flux(model_path: Path, n_channels: int) -> float:
    """Total flux of the model cube, summed channel by channel."""
    from astropy.io import fits

    with fits.open(model_path, memmap=True) as model_hdus:
        model = model_hdus[0].data
        return float(
            sum(
                np.nansum(np.asarray(model[..., c, :, :], dtype=np.float64))
                for c in range(n_channels)
            )
        )


def cube_quality(task_dir: Path, *, write_moments: bool = True) -> dict[str, Any] | None:
    """Cube statistics (and moment maps) of one task, or None without its cubes."""
    from astropy.io import fits

    paths = {kind: task_dir / cube_product(kind) for kind in ("image", "residual", "model")}
    if not (paths["image"].is_file() and paths["residual"].is_file()):
        return None
    with (
        fits.open(paths["image"], memmap=True) as image_hdus,
        fits.open(paths["residual"], memmap=True) as residual_hdus,
    ):
        header = image_hdus[0].header
        image = image_hdus[0].data
        residual = residual_hdus[0].data
        n_channels = image.shape[-3]
        planes = image.reshape(-1, n_channels, *image.shape[-2:])[0]
        residual_planes = residual.reshape(-1, n_channels, *residual.shape[-2:])[0]
        beam = beam_pixels(header)
        stacked_image = np.zeros(planes.shape[1:])
        for c in range(n_channels):
            stacked_image += np.nan_to_num(_channel(planes, c))
        _, _, r = _source_geometry(stacked_image, beam)
        usable, line, logged = _line_channels(task_dir, planes, residual_planes, r <= 0.5)
        linefree = usable & ~line
        channel_noise = _channel_noise(residual_planes, usable, r >= 2.5)
        velocity, width_km_s = velocity_axis(header, fallback_axis=3)
        sums = _sum_channels(planes, residual_planes, line, linefree, velocity, channel_noise)
    mask_path = task_dir / LINE_MASK_FILENAME
    if mask_path.is_file():
        line_mask = _plane(fits.getdata(mask_path)).astype(bool)
    else:
        line_mask = moment_mask(sums.line_image if line.any() else stacked_image, beam)
    stats: dict[str, Any] = {
        "channels": int(n_channels),
        "usable_channels": int(usable.sum()),
        "line_channels": int(line.sum()),
        "line_free_channels": int(linefree.sum()),
        "line_channels_from": "log" if logged is not None else "source spectrum",
        "line_mask_pixels": int(line_mask.sum()),
        "line_mask_from": "task" if mask_path.is_file() else "moment-0",
        "channel_noise_mjy": channel_noise * 1e3,
    }
    stats.update(_residual_stats(sums, line, linefree, line_mask, r))
    if paths["model"].is_file():
        stats["model_flux_jy"] = _model_flux(paths["model"], n_channels)
    if write_moments:
        weight = sums.weight
        stats["moments"] = _write_moments(
            task_dir,
            header,
            line_any=bool(line.any()),
            mom0=sums.line_image * width_km_s,
            mom1=np.where(
                weight > 0, sums.weighted_velocity / np.where(weight > 0, weight, 1.0), np.nan
            ),
            mom8=np.where(np.isfinite(sums.peak), sums.peak, np.nan),
            residual_line=sums.line_residual * width_km_s,
            residual_linefree=sums.linefree_residual * width_km_s,
        )
    return stats


def _write_moments(task_dir: Path, header, *, line_any: bool, **maps: np.ndarray) -> list[str]:
    from astropy.io import fits

    out = task_dir / QUALITY_DIRNAME
    out.mkdir(exist_ok=True)
    units = {
        "mom0": "Jy/beam.km/s",
        "mom1": "km/s",
        "mom8": "Jy/beam",
        "residual_line": "Jy/beam.km/s",
        "residual_linefree": "Jy/beam.km/s",
    }
    plane_header = fits.Header()
    for key in (
        "CTYPE1",
        "CRVAL1",
        "CDELT1",
        "CRPIX1",
        "CUNIT1",
        "CTYPE2",
        "CRVAL2",
        "CDELT2",
        "CRPIX2",
        "CUNIT2",
        "BMAJ",
        "BMIN",
        "BPA",
        "EQUINOX",
        "RADESYS",
        "OBJECT",
    ):
        if key in header:
            plane_header[key] = header[key]
    copy_provenance(header, plane_header)
    written = []
    for name, data in maps.items():
        if name in ("mom0", "mom1", "mom8", "residual_line") and not line_any:
            continue
        path = out / f"cube-{name.replace('_', '-')}.fits"
        map_header = plane_header.copy()
        map_header["BUNIT"] = units[name]
        map_header.add_history(f"ALMASim: clean-quality map {path.name}")
        write_fits_atomic(
            path, [fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=map_header)]
        )
        written.append(path.name)
    return written
