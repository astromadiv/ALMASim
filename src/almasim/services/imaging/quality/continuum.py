"""Continuum cleaning quality: S/N and the residual at the source (MADIV-28 T4.10)."""

from __future__ import annotations

import math
import re
import shlex
from pathlib import Path
from typing import Any

import numpy as np

from ..fits_io import read_image
from ..fits_wcs import beam_pixels
from ..products import IMAGE_FILENAME, continuum_product
from ..stats import robust_sigma

RINGS_BEAMS = ((0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 5))
_SIGMA = re.compile(r"Estimated standard deviation of background noise: ([-+.\w]+) (\w*Jy)")
# WSClean prints this when a clean's total flux grows instead of converging.
_DIVERGED = "WARNING: Stopping clean because of divergence"
_UNIT = {"Jy": 1.0, "mJy": 1e-3, "µJy": 1e-6, "uJy": 1e-6, "nJy": 1e-9}


def _plane(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=np.float64).reshape(data.shape[-2:])


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _source_geometry(image: np.ndarray, beam_pixels: float) -> tuple[int, int, np.ndarray]:
    """The brightest pixel in the image's inner half and every pixel's distance to it in beams."""
    ny, nx = image.shape
    yy, xx = np.mgrid[:ny, :nx]
    inner = (np.abs(yy - ny / 2) < ny / 4) & (np.abs(xx - nx / 2) < nx / 4)
    sy, sx = np.unravel_index(np.nanargmax(np.where(inner, image, -np.inf)), image.shape)
    distance = np.hypot(yy - sy, xx - sx) / beam_pixels
    return int(sy), int(sx), distance


def _runs(log_path: Path, name: str) -> list[list[str]]:
    """The log lines of each WSClean run that wrote ``name``, in order.

    The task log holds every WSClean run of a task, each after a ``# <command>``
    line; a run belongs to ``name`` when its command has ``-name .../<name>``.
    """
    runs: list[list[str]] = []
    current: list[str] | None = None
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("# ") and " -name " in line:
            # The task logs each command with shlex.join: quoted paths
            # (spaces and the like) only split right with shlex.
            try:
                words = shlex.split(line[2:])
            except ValueError:
                words = line[2:].split()
            target = words[words.index("-name") + 1] if "-name" in words[:-1] else ""
            current = [] if Path(target).name == name else None
            if current is not None:
                runs.append(current)
            continue
        if current is not None:
            current.append(line)
    return runs


def noise_history(log_path: Path, name: str) -> list[float]:
    """The noise WSClean estimated at each major cycle of the first run that
    wrote ``name`` (the continuum's first pass, before any restart)."""
    sigmas: list[float] = []
    for run in _runs(log_path, name):
        for line in run:
            match = _SIGMA.search(line)
            if match:
                try:
                    sigmas.append(float(match.group(1)) * _UNIT.get(match.group(2), 1.0))
                except ValueError:
                    continue
        if sigmas:
            break
    return sigmas


def divergences(log_path: Path, name: str) -> int:
    """How many times WSClean stopped cleaning for divergence over every run
    that wrote ``name`` (once per image it stops on: the continuum, or a cube
    channel)."""
    return sum(line.startswith(_DIVERGED) for run in _runs(log_path, name) for line in run)


def continuum_quality(task_dir: Path) -> dict[str, Any] | None:
    """Continuum statistics of one task, or None without its three images."""
    from astropy.io import fits

    paths = {kind: task_dir / continuum_product(kind) for kind in ("image", "residual", "model")}
    if not all(path.is_file() for path in paths.values()):
        return None
    header = fits.getheader(paths["image"])
    image = _plane(read_image(paths["image"]))
    residual = _plane(read_image(paths["residual"]))
    model = _plane(read_image(paths["model"]))
    beam = beam_pixels(header)
    sy, sx, r = _source_geometry(image, beam)
    sigma = robust_sigma(residual[r >= 2.5])
    one, two = r <= 1, r <= 2
    half = int(np.ceil(2 * beam)) if np.isfinite(beam) else 8
    box = residual[max(sy - half, 0) : sy + half + 1, max(sx - half, 0) : sx + half + 1]
    symmetric, antisymmetric = 0.5 * (box + box[::-1, ::-1]), 0.5 * (box - box[::-1, ::-1])
    stats: dict[str, Any] = {
        "peak_jy_beam": float(image[sy, sx]),
        "peak_pixel": [sy, sx],
        "beam_arcsec": [
            round(float(header.get("BMAJ", math.nan)) * 3600, 3),
            round(float(header.get("BMIN", math.nan)) * 3600, 3),
        ],
        "noise_jy_beam": sigma,
        "snr": _ratio(image[sy, sx], sigma),
        "residual_share_1beam": _ratio(np.nansum(residual[one]), np.nansum(image[one])),
        "residual_share_2beam": _ratio(np.nansum(residual[two]), np.nansum(image[two])),
        "residual_peak_sigma": _ratio(np.nanmax(np.abs(residual[one])), sigma),
        "residual_core_sigma": _ratio(np.nanmean(residual[r <= 0.5]), sigma),
        "residual_rings_sigma": {
            f"{lo:g}-{hi:g}": _ratio(np.nanmean(residual[(r >= lo) & (r < hi)]), sigma)
            for lo, hi in RINGS_BEAMS
        },
        "residual_symmetric_sigma": _ratio(robust_sigma(symmetric), sigma),
        "residual_antisymmetric_sigma": _ratio(robust_sigma(antisymmetric), sigma),
        "model_flux_jy": float(np.nansum(model)),
        "model_components": int(np.count_nonzero(np.nan_to_num(model))),
    }
    log = task_dir.parent / f"{task_dir.name}.log"
    if log.is_file():
        name = Path(IMAGE_FILENAME).stem.rsplit("-", 1)[0]
        history = noise_history(log, name)
        if len(history) >= 2:
            stats["noise_at_mask_creation_ratio"] = _ratio(history[0], history[-1])
        stats["diverged"] = divergences(log, name)
    return stats
