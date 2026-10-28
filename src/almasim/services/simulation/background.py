"""Additive ALMA-band background skies (MADIV-28 T5.7)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# "none" adds nothing; "combined" is both the galaxies and the diffuse field.
BACKGROUND_MODES = ("none", "blank_field_dsfg", "dusty_diffuse", "combined")


def _smoothed_positive_field(
    n_pix: int,
    rng: np.random.Generator,
    *,
    correlation_scale_pix: float,
) -> np.ndarray:
    """Generate a positive correlated 2D field using FFT low-pass smoothing."""
    white = rng.normal(size=(n_pix, n_pix))
    ky = np.fft.fftfreq(n_pix)
    kx = np.fft.fftfreq(n_pix)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    sigma_k = 1.0 / max(float(correlation_scale_pix), 1.0)
    low_pass = np.exp(-0.5 * k2 / max(sigma_k**2, 1e-12))
    field = np.real(np.fft.ifft2(np.fft.fft2(white) * low_pass)).astype(np.float32)
    field -= float(field.min())
    peak = float(field.max())
    if peak > 0.0:
        field /= peak
    return field


def generate_background_cube(
    *,
    mode: str,
    n_pix: int,
    n_channels: int,
    cell_size_arcsec: float,
    channel_frequencies_hz: np.ndarray,
    cont_sens_jy: float,
    level: float = 1.0,
    seed: int | Sequence[int] | None = None,
) -> np.ndarray:
    """Generate an additive ALMA-band background cube in `(chan, y, x)` order.

    ``seed`` (an int or SeedSequence entropy) makes it reproducible.
    """
    mode = str(mode or "none").lower()
    if mode == "none" or level <= 0.0:
        return np.zeros((n_channels, n_pix, n_pix), dtype=np.float32)

    rng = np.random.default_rng(seed)
    nu0 = float(np.median(channel_frequencies_hz))
    spectral_ratio = np.asarray(channel_frequencies_hz, dtype=float) / max(nu0, 1e-12)
    cube = np.zeros((n_channels, n_pix, n_pix), dtype=np.float32)

    if mode in {"blank_field_dsfg", "combined"}:
        field_arcmin2 = ((n_pix * cell_size_arcsec) / 60.0) ** 2
        expected_sources = max(1.0, 30.0 * field_arcmin2 * max(level, 0.25))
        n_sources = int(min(250, rng.poisson(expected_sources)))
        for _ in range(n_sources):
            x = int(rng.integers(0, n_pix))
            y = int(rng.integers(0, n_pix))
            base_flux = cont_sens_jy * level * float(np.exp(rng.normal(-0.4, 0.9)))
            alpha = float(rng.uniform(2.5, 4.0))
            spectrum = base_flux * spectral_ratio**alpha
            cube[:, y, x] += spectrum.astype(np.float32)

    if mode in {"dusty_diffuse", "combined"}:
        base_field = _smoothed_positive_field(
            n_pix,
            rng,
            correlation_scale_pix=max(4.0, n_pix / 10.0),
        )
        diffuse_amp = cont_sens_jy * 0.5 * max(level, 0.1)
        beta = 3.0
        for channel in range(n_channels):
            cube[channel] += (diffuse_amp * spectral_ratio[channel] ** beta * base_field).astype(
                np.float32
            )

    return cube
