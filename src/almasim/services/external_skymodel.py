"""External sky-model ingestion helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import zoom

from ..skymodels.registry import SKYMODELS
from .imaging.fits_io import read_cube

EXTERNAL_SOURCE_TYPES = {name for name, spec in SKYMODELS.items() if spec.external}


@dataclass
class ExternalSkyModelPayload:
    """Normalized external sky-model payload."""

    cube: np.ndarray
    header: fits.Header
    metadata: dict[str, Any]


def is_external_source_type(source_type: str) -> bool:
    """Return whether the source type is backed by an external model input."""
    return str(source_type).lower() in EXTERNAL_SOURCE_TYPES


@dataclass(frozen=True)
class CubeWcs:
    """An input cube's grid, from its header alone (its data may be gigabytes)."""

    n_pix_x: int
    n_pix_y: int
    n_channels: int | None  # the spectral axis's length; None without one
    central_freq_hz: float | None  # the middle of the spectral axis
    channel_width_hz: float | None  # positive
    descending: bool  # the channels go down in frequency
    cell_arcsec: float | None

    @property
    def n_pix(self) -> int:
        return max(self.n_pix_x, self.n_pix_y)


def read_cube_wcs(path: str | Path) -> CubeWcs:
    """The grid a FITS image or cube is on: its spatial size and pixel, and
    its spectral axis (a FREQ axis, else a third axis) if it has one.
    Degenerate axes (a one-plane Stokes axis) are left out."""
    with fits.open(path, memmap=True) as hdus:
        header = next(
            (hdu.header for hdu in hdus if hdu.is_image and hdu.header.get("NAXIS", 0) >= 2),
            None,
        )
    if header is None:
        raise ValueError(f"{path} does not contain image data")
    axes = range(1, int(header["NAXIS"]) + 1)
    spectral = next(
        (k for k in axes if str(header.get(f"CTYPE{k}", "")).upper().startswith("FREQ")),
        3 if header["NAXIS"] >= 3 and header["NAXIS3"] > 1 else None,
    )
    n_channels = central = width = None
    descending = False
    if spectral is not None:
        n_channels = int(header[f"NAXIS{spectral}"])
        cdelt = float(header.get(f"CDELT{spectral}", 0.0) or 0.0)
        if cdelt and str(header.get(f"CTYPE{spectral}", "")).upper().startswith("FREQ"):
            crval = float(header.get(f"CRVAL{spectral}", 0.0))
            crpix = float(header.get(f"CRPIX{spectral}", 1.0))
            central = crval + ((n_channels + 1) / 2 - crpix) * cdelt
            width, descending = abs(cdelt), cdelt < 0
    pixel = header.get("CDELT2") or header.get("CDELT1")
    return CubeWcs(
        n_pix_x=int(header["NAXIS1"]),
        n_pix_y=int(header["NAXIS2"]),
        n_channels=n_channels,
        central_freq_hz=central,
        channel_width_hz=width,
        descending=descending,
        cell_arcsec=abs(float(pixel)) * 3600 if pixel else None,  # celestial axes are in degrees
    )


def infer_external_cube_geometry(
    *,
    source_type: str,
    skymodel_path: str | None = None,
    component_table_path: str | None = None,
) -> dict[str, int] | None:
    """Infer basic cube geometry from an external input when possible."""
    source_type = str(source_type).lower()
    if source_type == "external-components":
        if component_table_path is None:
            raise ValueError("external_component_table_path is required for external-components")
        return None

    if skymodel_path is None:
        raise ValueError("external_skymodel_path is required for external FITS inputs")

    wcs = read_cube_wcs(skymodel_path)
    channels = wcs.n_channels or 1
    return {
        "n_channels": 1 if source_type == "external-fits-image" else channels,
        "n_pix_y": wcs.n_pix_y,
        "n_pix_x": wcs.n_pix_x,
        "n_pix": wcs.n_pix,
    }


def load_external_sky_model(
    *,
    source_type: str,
    skymodel_path: str | None = None,
    component_table_path: str | None = None,
    target_npix: int | None = None,
    target_nchannels: int | None = None,
    alignment_mode: str = "observation",
    header_mode: str = "observation",
    header_overrides: Mapping[str, Any] | None = None,
    target_header: fits.Header | None = None,
    target_wcs: Any | None = None,
    channel_frequencies_hz: np.ndarray | None = None,
) -> ExternalSkyModelPayload:
    """Load an external skymodel and normalize it to `(chan, y, x)` order."""
    source_type = str(source_type).lower()
    alignment_mode = str(alignment_mode or "observation").lower()
    header_mode = str(header_mode or "observation").lower()
    if source_type not in EXTERNAL_SOURCE_TYPES:
        raise ValueError(f"Unsupported external source type: {source_type}")

    if source_type == "external-components":
        if target_npix is None or target_nchannels is None:
            raise ValueError(
                "target_npix and target_nchannels are required for external-components"
            )
        cube = _load_component_table(
            component_table_path=component_table_path,
            target_npix=int(target_npix),
            target_nchannels=int(target_nchannels),
            target_wcs=target_wcs,
            channel_frequencies_hz=channel_frequencies_hz,
        )
        header = fits.Header() if target_header is None else target_header.copy()
        metadata = {
            "input_kind": "component_table",
            "input_path": str(component_table_path),
            "alignment_mode": alignment_mode,
            "header_mode": header_mode,
            "original_shape": [int(v) for v in cube.shape],
        }
    else:
        if skymodel_path is None:
            raise ValueError("external_skymodel_path is required for external FITS inputs")
        fits_input = read_cube(skymodel_path)
        input_header = fits_input.header
        cube = _coerce_fits_to_channel_first(fits_input.data, source_type=source_type)
        if _descending(input_header):
            cube = cube[::-1]  # the simulation's channels go up in frequency
        # Blanked pixels (WSClean writes flagged channels as NaN) hold no emission.
        blank = int(np.count_nonzero(~np.isfinite(cube)))
        if blank:
            cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)
        metadata = {
            "input_kind": "fits",
            "blank_pixels": blank,
            "input_path": str(skymodel_path),
            "alignment_mode": alignment_mode,
            "header_mode": header_mode,
            "original_shape": [int(v) for v in cube.shape],
            "input_bunit": str(input_header.get("BUNIT", "")),
        }
        target_shape = _resolve_target_shape(
            cube_shape=cube.shape,
            target_npix=target_npix,
            target_nchannels=target_nchannels,
            alignment_mode=alignment_mode,
        )
        cube = _align_cube_to_shape(cube, target_shape)
        header = (
            input_header.copy()
            if header_mode == "preserve" or target_header is None
            else target_header.copy()
        )

    if header_overrides:
        for key, value in header_overrides.items():
            header[str(key).upper()] = value

    return ExternalSkyModelPayload(
        cube=np.asarray(cube, dtype=np.float32),
        header=header,
        metadata=metadata,
    )


def _coerce_fits_to_channel_first(data: np.ndarray, *, source_type: str) -> np.ndarray:
    """Normalize FITS image/cube data to `(chan, y, x)` order.

    A Stokes axis (numpy's first, FITS's fourth) keeps its first plane, Stokes I.
    """
    source_type = str(source_type).lower()
    while data.ndim > 3:
        data = data[0]
    if data.ndim == 2:
        return data[np.newaxis, :, :].astype(np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 2D image or 3D cube, got {data.ndim}D data")

    # FITS cubes are commonly stored as (chan, y, x). Keep that by default.
    if source_type == "external-fits-image":
        # Allow a single slice to be selected from a 3D cube-like image.
        return data[:1].astype(np.float32)
    return data.astype(np.float32)


def _resolve_target_shape(
    *,
    cube_shape: tuple[int, int, int],
    target_npix: int | None,
    target_nchannels: int | None,
    alignment_mode: str,
) -> tuple[int, int, int]:
    """Resolve the aligned cube shape."""
    if alignment_mode == "preserve":
        nchan = int(target_nchannels) if target_nchannels is not None else int(cube_shape[0])
        ny = int(target_npix) if target_npix is not None else int(cube_shape[1])
        nx = int(target_npix) if target_npix is not None else int(cube_shape[2])
        return (nchan, ny, nx)
    if target_npix is None or target_nchannels is None:
        raise ValueError(
            "target_npix and target_nchannels are required when alignment_mode='observation'"
        )
    return (int(target_nchannels), int(target_npix), int(target_npix))


def _align_cube_to_shape(cube: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Resize or crop/pad an external cube to the requested target shape."""
    if tuple(cube.shape) == tuple(target_shape):
        return cube.astype(np.float32, copy=False)

    factors = tuple(
        target / current if current > 0 else 1.0
        for current, target in zip(cube.shape, target_shape, strict=True)
    )
    resized = zoom(cube, zoom=factors, order=1)
    resized = np.asarray(resized, dtype=np.float32)
    return _crop_or_pad_cube(resized, target_shape)


def _crop_or_pad_cube(cube: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Center-crop or zero-pad a cube to the exact target shape."""
    out = np.zeros(target_shape, dtype=np.float32)
    src_slices = []
    dst_slices = []
    for current, target in zip(cube.shape, target_shape, strict=True):
        if current >= target:
            start = (current - target) // 2
            src_slices.append(slice(start, start + target))
            dst_slices.append(slice(None))
        else:
            offset = (target - current) // 2
            src_slices.append(slice(None))
            dst_slices.append(slice(offset, offset + current))
    out[tuple(dst_slices)] = cube[tuple(src_slices)]
    return out


def _load_component_table(
    *,
    component_table_path: str | None,
    target_npix: int,
    target_nchannels: int,
    target_wcs: Any | None,
    channel_frequencies_hz: np.ndarray | None,
) -> np.ndarray:
    """Build a cube from a simple component-list-like table."""
    if component_table_path is None:
        raise ValueError("external_component_table_path is required for external-components")

    table = Table.read(component_table_path)
    rows = table.to_pandas()
    columns = {str(col).lower(): col for col in rows.columns}
    cube = np.zeros((target_nchannels, target_npix, target_npix), dtype=np.float32)
    reference_freq = (
        float(np.median(channel_frequencies_hz))
        if channel_frequencies_hz is not None and len(channel_frequencies_hz) > 0
        else 1.0
    )

    for _, row in rows.iterrows():
        x_pix, y_pix = _resolve_component_position(row, columns, target_wcs)
        flux = float(_row_value(row, columns, ["flux_jy", "flux", "intensity"], default=0.0))
        if flux == 0.0:
            continue

        alpha = float(_row_value(row, columns, ["spectral_index", "alpha"], default=0.0))
        ref_freq = float(
            _row_value(
                row,
                columns,
                ["reference_freq_hz", "ref_freq_hz"],
                default=reference_freq,
            )
        )
        channel = _row_value(row, columns, ["channel", "chan"], default=None)

        if channel is not None:
            chan_idx = int(np.clip(int(channel), 0, target_nchannels - 1))
            cube[
                chan_idx,
                np.clip(y_pix, 0, target_npix - 1),
                np.clip(x_pix, 0, target_npix - 1),
            ] += flux
            continue

        if channel_frequencies_hz is None:
            spectrum = np.full(target_nchannels, flux, dtype=np.float32)
        else:
            nu = np.asarray(channel_frequencies_hz, dtype=float)
            spectrum = flux * (nu / max(ref_freq, 1e-12)) ** alpha
        cube[:, np.clip(y_pix, 0, target_npix - 1), np.clip(x_pix, 0, target_npix - 1)] += (
            spectrum.astype(np.float32)
        )

    return cube


def _resolve_component_position(
    row, columns: dict[str, str], target_wcs: Any | None
) -> tuple[int, int]:
    """Resolve component position from pixel or sky coordinates."""
    if "x_pix" in columns and "y_pix" in columns:
        x_pix = int(round(float(row[columns["x_pix"]])))
        y_pix = int(round(float(row[columns["y_pix"]])))
        return x_pix, y_pix

    if "ra_deg" in columns and "dec_deg" in columns and target_wcs is not None:
        ra = float(row[columns["ra_deg"]])
        dec = float(row[columns["dec_deg"]])
        x_pix, y_pix, _ = target_wcs.sub(3).wcs_world2pix(ra, dec, 1.0, 0)
        return int(round(float(x_pix))), int(round(float(y_pix)))

    raise ValueError("Component rows must provide either x_pix/y_pix or ra_deg/dec_deg columns")


def _row_value(row, columns: dict[str, str], keys: list[str], *, default: Any) -> Any:
    """Fetch a case-insensitive row value with a default."""
    for key in keys:
        actual = columns.get(key)
        if actual is None:
            continue
        value = row[actual]
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        return value
    return default


def _descending(header: fits.Header) -> bool:
    """Whether the header's FREQ axis runs down in frequency."""
    for k in range(1, int(header.get("NAXIS", 0)) + 1):
        if str(header.get(f"CTYPE{k}", "")).upper().startswith("FREQ"):
            return float(header.get(f"CDELT{k}", 0.0) or 0.0) < 0
    return False
