"""The noise of an imaged window, real or simulated (MADIV-28).

From an ALMASim imaging task (``image image-from-ms``, ``image
clean-quality``): the residuals' noise level, Gaussianity (skew, excess
kurtosis, 5-sigma tails), spatial correlation (the area of the
autocorrelation above half its peak, in beams), its radial trend, and in the
cube the spread of the channel noise and the correlation of neighbouring
channels. Statistics use the inner part of the image (radius below 40% of
its width), outside the cube's line mask. The metrics are
research/noise_realism's (measure.py), on which the alma model was judged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..imaging.stats import robust_sigma

INNER = 0.4


def image_noise_stats(task: str | Path) -> dict[str, float]:
    """The task's noise statistics, flat (``cont.sigma``, ``cube.adjacent_corr``,
    ``q.continuum.snr`` from its quality.json, ...)."""
    task = Path(task)
    out: dict[str, float] = {}
    quality = task / "quality.json"
    if quality.exists():
        q = json.loads(quality.read_text())
        _flatten("q", {"continuum": q.get("continuum", {}), "cube": q.get("cube", {})}, out)
    if (task / "wsclean-residual.fits").exists():
        _flatten("cont", continuum_stats(task), out)
    if (task / "wsclean-cube-residual.fits").exists():
        _flatten("cube", cube_stats(task), out)
    return out


def continuum_stats(task: Path) -> dict[str, Any]:
    residual, header = _planes(task / "wsclean-residual.fits")
    plane = residual[0]
    radius = _radius(plane.shape)
    inner = radius < INNER
    sigma = robust_sigma(plane[inner])
    rings = [robust_sigma(plane[(radius >= a) & (radius < b)]) for a, b in ((0, 0.15), (0.3, 0.45))]
    return {
        "sigma": sigma,
        **_moments(plane[inner], sigma),
        "corr_area_beams": _corr_area_beams(_centre(plane), _beam_pixels(header)),
        "radial_outer_inner": rings[1] / rings[0] if rings[0] else None,
    }


def cube_stats(task: Path) -> dict[str, Any]:
    residual, header = _planes(task / "wsclean-cube-residual.fits")
    nchan, ny, nx = residual.shape
    keep = np.broadcast_to(_radius((ny, nx)) < INNER, residual.shape).copy()
    mask_path = task / "wsclean-cube-linemask.fits"
    if mask_path.exists():
        mask, _ = _planes(mask_path)
        if mask.shape[-2:] == (ny, nx):
            keep &= ~(np.nan_to_num(mask[0] if len(mask) == 1 else mask) > 0)
    usable = [
        k for k in range(nchan) if np.isfinite(residual[k]).any() and np.nanstd(residual[k]) > 0
    ]
    sigmas = np.array([robust_sigma(residual[k][keep[k]]) for k in usable])
    adjacent = []
    for k in usable:
        if k + 1 in usable:
            both = keep[k] & keep[k + 1]
            a, b = residual[k][both], residual[k + 1][both]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 100:
                adjacent.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
    beam = _beam_pixels(header)
    areas: list[float] = []
    for k in usable[:: max(1, len(usable) // 8)]:
        area = _corr_area_beams(_centre(residual[k]), beam)
        if area is not None:
            areas.append(area)
    pooled = [
        (residual[k][keep[k]] - np.nanmedian(residual[k][keep[k]])) / s
        for k, s in zip(usable, sigmas, strict=True)
        if s > 0
    ]
    median = float(np.median(sigmas)) if sigmas.size else None
    spread = None
    if sigmas.size > 2 and median:
        q16, q84 = np.percentile(sigmas, [16, 84])
        spread = float((q84 - q16) / 2 / median)
    return {
        "channels": int(nchan),
        "sigma_median": median,
        "sigma_spread": spread,
        "adjacent_corr": float(np.median(adjacent)) if adjacent else None,
        "corr_area_beams": float(np.median(areas)) if areas else None,
        **_moments(np.concatenate(pooled) if pooled else np.array([]), 1.0),
    }


def _planes(path: Path) -> tuple[np.ndarray, Any]:
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header
    while data.ndim > 3:
        data = data[0]
    return data.reshape((-1,) + data.shape[-2:]), header


def _radius(shape: tuple[int, int]) -> np.ndarray:
    ny, nx = shape
    y, x = np.mgrid[:ny, :nx]
    return np.hypot(x - nx / 2, y - ny / 2) / max(nx, ny)


def _centre(plane: np.ndarray) -> np.ndarray:
    ny, nx = plane.shape
    half = max(16, min(ny, nx) // 4)
    return plane[ny // 2 - half : ny // 2 + half, nx // 2 - half : nx // 2 + half]


def _moments(x: np.ndarray, sigma: float) -> dict[str, float | None]:
    x = x[np.isfinite(x)]
    if not x.size or not sigma > 0:
        return {"skew": None, "kurtosis": None, "tail5": None}
    z = (x - np.median(x)) / sigma
    clipped = z[np.abs(z) < 10]  # the noise's moments, not a few bright residuals'
    return {
        "skew": float(np.mean(clipped**3)),
        "kurtosis": float(np.mean(clipped**4) - 3),
        "tail5": float(np.mean(np.abs(z) > 5)),
    }


def _corr_area_beams(plane: np.ndarray, beam_pixels: float) -> float | None:
    """The area of the plane's autocorrelation above half its peak, in beams."""
    p = np.nan_to_num(plane - np.nanmedian(plane))
    if not np.any(p) or beam_pixels <= 0:
        return None
    power = np.abs(np.fft.rfft2(p)) ** 2
    acf = np.fft.irfft2(power, s=p.shape)
    acf = np.fft.fftshift(acf) / acf.flat[0]
    return float(np.sum(acf > 0.5) / beam_pixels)


def _beam_pixels(header: Any) -> float:
    cell = abs(header.get("CDELT2", 0.0))
    bmaj, bmin = header.get("BMAJ"), header.get("BMIN")
    if not (cell and bmaj and bmin):
        return 0.0
    return float(np.pi * bmaj * bmin / (4 * np.log(2)) / cell**2)


def _flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(value, int | float) and not isinstance(value, bool):
        out[prefix] = float(value)
