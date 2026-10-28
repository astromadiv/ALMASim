"""What ``almasim noise extract`` measures on one real window (MADIV-28).

One :class:`WindowNoise` per window, written as ``window.json``: the input it
was measured on, the observation's exposure, the thermal noise against WEIGHT
and its channel correlation, ALMA's Tsys across the window, the noise of one
visibility at unit time and bandwidth, and each antenna's noise shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WindowInput(_Record):
    """The window measured."""

    ms: str = Field(description="The calibrated MeasurementSet.")
    spw: int = Field(ge=0, description="Its spectral window (the MS's numbering).")
    field: str | None = Field(default=None, description="Field name or id; None: every field.")
    syscal_ms: str | None = Field(default=None, description="Where SYSCAL was read (default: ms).")
    line_channels: int = Field(default=0, ge=0, description="Channels left out as line emission.")
    pwv_mm: float | None = Field(
        default=None, ge=0, description="The observation's precipitable water vapour (mm)."
    )


class Exposure(_Record):
    """What the window observed (cross-correlations)."""

    antennas: int
    dish_diameter_m: float
    baselines: int
    integrations: int
    integration_s: float
    time_on_source_s: float
    channels: int
    channel_width_hz: float
    noise_bandwidth_hz: float = Field(description="EFFECTIVE_BW, the channel's noise bandwidth.")
    resolution_hz: float | None = Field(
        default=None, description="RESOLUTION, the channel's resolution as the archive gives it."
    )
    frequency_ghz: float
    correlations: int
    flagged_fraction: float
    elevation_deg: float | None = Field(default=None, description="The target's median elevation.")


class ThermalNoise(_Record):
    """The noise against WEIGHT, from channel differences (lag 3 removes the sky;
    lags 1 and 2 give the correlation of neighbouring channels)."""

    scale: float = Field(description="Noise variance x WEIGHT per component (1: WEIGHT is right).")
    rho1: float = Field(description="Correlation of channels one apart.")
    rho2: float = Field(description="Correlation of channels two apart.")
    channel_profile: list[float] = Field(description="Noise variance per channel over its mean.")
    rows_sampled: int
    row_step: int
    estimator: Literal["clipped mean"] = "clipped mean"


class Tsys(_Record):
    """ALMA's own Tsys (SYSCAL) over the window."""

    median_k: float = Field(description="Median Tsys over antennas, times, hands and channels.")
    profile: list[float] = Field(description="Tsys^2 per channel over its mean.")
    spectral_window: int = Field(description="The Tsys window read (same baseband).")


class VisibilityNoise(_Record):
    """One Stokes I visibility's noise per real and imaginary part."""

    sigma_jy: float
    unit_jy: float = Field(
        description="sigma x sqrt(integration x channel width): Jy s^0.5 Hz^0.5."
    )
    hands_identical: bool = Field(description="XX and YY carry the same data (a copied Stokes I).")
    efficiency: float | None = Field(
        default=None,
        description="The aperture x correlator efficiency the noise and Tsys imply: "
        "2 k Tsys / (2 A unit) over sqrt(noise bandwidth / width).",
    )


class AntennaShape(_Record):
    """Each antenna's own noise shape across the window, beyond the array's."""

    antennas_used: int
    channels_used: int
    excess_rms: float = Field(
        description="Fractional rms of an antenna's noise variance, beyond sampling."
    )
    half_width_channels: int = Field(description="Lag where its autocorrelation falls to half.")
    half_width_mhz: float


class WindowNoise(_Record):
    """Everything ``noise extract`` measures on one window."""

    schema_version: Literal[1] = 1
    name: str
    almasim_version: str
    input: WindowInput
    exposure: Exposure
    thermal: ThermalNoise
    visibility: VisibilityNoise
    tsys: Tsys | None = None
    antenna_shape: AntennaShape | None = None
    notes: list[str] = Field(
        default_factory=list, description="What could not be measured, and why."
    )


class PhaseFit(_Record):
    """A phase screen fitted as sigma(b) = sigma_1km (b / 1 km)^slope."""

    bins: int = Field(description="Baseline bins the fit used.")
    slope: float
    phase_1km_deg: float
    path_um_1km: float = Field(description="The rms path difference at 1 km (microns).")


class FitBin(_Record):
    """One baseline-length bin of the bright (baseline, scan) groups."""

    baseline_m: float
    groups: int
    phase_rms_deg: float | None
    thermal_phase_rms_deg: float | None
    lnamp_mean: float | None
    thermal_lnamp_mean: float | None


class WindowFit(_Record):
    """What ``noise fit`` finds in one window's calibration residuals: the
    values of the slow_phase, fast_phase, antenna_gains and baseline_errors
    components that make a simulation of it match (None: not measurable)."""

    schema_version: Literal[1] = 1
    name: str
    bright_groups: int = Field(description="(baseline, scan) groups with S/N above 10.")
    realizations: int
    kappa: float = Field(description="1 + 2 rho1 + 2 rho2: the channel-summed noise's excess.")
    bins: list[FitBin]
    slow_phase: PhaseFit | None = None
    fast_phase: PhaseFit | None = None
    gain_amp_rms: float | None = None
    baseline_amp_rms: float | None = None
    baseline_phase_rms_deg: float | None = None
    antenna_solve: dict[str, float] | None = Field(
        default=None, description="The real data's per-scan antenna solve."
    )
    notes: list[str] = Field(default_factory=list)
