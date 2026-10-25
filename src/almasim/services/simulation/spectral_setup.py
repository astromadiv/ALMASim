"""Redshift, rest frequency, continuum and lines of a run (MADIV-28 T5.7)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import astropy.units as U
import numpy as np

from .. import astro as uas
from ..astro.spectral import SpectralResult, process_spectral_data
from ..external_skymodel import is_external_source_type
from .context import RunContext
from .geometry import CubeGeometry
from .params import SimulationParams, _clean


@dataclass
class SpectralSetup:
    """The source spectrum a run injects."""

    continuum: np.ndarray
    line_fluxes: Any
    line_names: Any
    line_frequency: Any
    redshift: Any
    rest_frequency: U.Quantity
    source_channel_index: Any
    fwhm_z: Any
    lum_infrared: Any

    def continuum_fraction(self, n_chan: int) -> np.ndarray | None:
        """Each channel's continuum share of the source's flux, continuum over
        continuum plus lines (with the lines' profiles as the sky models draw
        them); None when the run has no continuum spectrum (an external model)."""
        from ...skymodels.utils import gaussian

        continuum = np.asarray(self.continuum, dtype=float).reshape(-1)
        if continuum.size != n_chan:
            return None
        channels = np.arange(n_chan)
        lines = np.zeros(n_chan)
        fluxes: np.ndarray = np.atleast_1d(
            np.asarray(self.line_fluxes if self.line_fluxes is not None else [], dtype=float)
        )
        centres = np.atleast_1d(self.source_channel_index)
        widths = np.atleast_1d(self.fwhm_z)
        for flux, centre, width in zip(fluxes, centres, widths, strict=False):
            lines += gaussian(channels, float(flux), float(centre), float(width))
        total = continuum + lines
        return np.divide(continuum, total, out=np.ones(n_chan), where=total > 0).clip(0.0, 1.0)


def resolve_spectral_setup(  # noqa: C901
    params: SimulationParams,
    geometry: CubeGeometry,
    cont_sens_jy: U.Quantity,
    ctx: RunContext,
) -> SpectralSetup:
    """The redshift (or rest frequency), continuum and line fluxes of the source.

    An external sky model brings its own emission: no continuum or lines, the
    source in the central channel.
    """
    line_names = _clean(params.line_names)
    if isinstance(line_names, str):
        line_names = [
            token.strip(" \"'") for token in line_names.strip("[]").split(",") if token.strip()
        ]
    elif isinstance(line_names, np.ndarray):
        line_names = line_names.tolist()
    rest_frequency = _clean(params.rest_frequency)
    redshift = _clean(params.redshift)
    lum_infrared = _clean(params.lum_infrared)
    n_lines = _clean(params.n_lines)

    if isinstance(line_names, float) and math.isnan(line_names):
        line_names = None

    source_freq = params.freq * U.GHz
    central_freq, band_range, n_channels = (
        geometry.central_freq,
        geometry.band_range,
        geometry.n_channels,
    )
    is_external_model = is_external_source_type(params.source_type)

    if not is_external_model and redshift is None:
        if rest_frequency is None:
            # For point sources, default to redshift=0 (local universe)
            # For other source types, this should be provided
            if params.source_type == "point":
                redshift = 0.0
                ctx.log(
                    "No redshift or rest_frequency provided, "
                    "using default redshift=0 for point source"
                )
            else:
                raise ValueError(
                    "Either 'redshift' or 'rest_frequency' must be provided. "
                    "Cannot compute redshift without rest frequency "
                    f"for source type '{params.source_type}'."
                )

        if rest_frequency is not None:
            if isinstance(rest_frequency, (np.ndarray, list)):
                rest_frequency = np.sort(np.array(rest_frequency))[0]
            rest_frequency = rest_frequency * U.GHz
            redshift = uas.compute_redshift(rest_frequency, source_freq)

    # Compute rest_frequency from redshift if not already set
    # Check if rest_frequency is None or hasn't been converted to a Quantity yet
    if not is_external_model and not isinstance(rest_frequency, U.Quantity):
        rest_frequency = (
            uas.compute_rest_frequency_from_redshift(params.main_dir, source_freq.value, redshift)
            * U.GHz
        )

    ctx.status("Computing spectral lines and properties")
    if is_external_model:
        continum = np.zeros(int(n_channels), dtype=float)
        line_fluxes = np.array([], dtype=float)
        line_names = None
        line_frequency: Any = []
        source_channel_index = [int(n_channels // 2)]
        fwhm_z = 1.0
        if redshift is None:
            redshift = 0.0
        if not isinstance(rest_frequency, U.Quantity):
            rest_frequency = central_freq
    else:
        # SpectralResult(*...) also accepts a plain 12-tuple (as test doubles return).
        result = SpectralResult(
            *process_spectral_data(
                params.source_type,
                params.main_dir,
                redshift,
                central_freq.value,
                band_range.value,
                source_freq.value,
                n_channels,
                lum_infrared,
                cont_sens_jy.value,
                line_names,
                n_lines,
                bool(params.remote),
                rng=ctx.rng,
            )
        )
        # The channel count (returned unchanged), bandwidth, frequency support
        # and continuum frequencies are not used here.
        continum, line_fluxes, line_names = result.continuum, result.line_fluxes, result.line_names
        redshift, line_frequency = result.redshift, result.line_frequencies
        source_channel_index, fwhm_z = result.line_indexes, result.fwhm_z
        lum_infrared = result.lum_infrared
    return SpectralSetup(
        continuum=continum,
        line_fluxes=line_fluxes,
        line_names=line_names,
        line_frequency=line_frequency,
        redshift=redshift,
        rest_frequency=rest_frequency,
        source_channel_index=source_channel_index,
        fwhm_z=fwhm_z,
        lum_infrared=lum_infrared,
    )
