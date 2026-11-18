"""What ``almasim noise extract`` measures on one real window (MADIV-28).

One :class:`WindowNoise` per window, written as ``window.json``: the input it
was measured on, the observation's exposure, the thermal noise against WEIGHT
and its channel correlation, ALMA's Tsys across the window, the noise of one
visibility at unit time and bandwidth, and each antenna's noise shape and level.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import JsonDict


def _unit(unit: str) -> JsonDict:
    """A field's unit, kept in its JSON Schema (``noise extract``'s report reads it)."""
    return {"unit": unit}


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WindowInput(_Record):
    """The window measured."""

    ms: str = Field(description="The calibrated MeasurementSet.")
    spw: int = Field(ge=0, description="Its spectral window (the MS's numbering).")
    field: str | None = Field(default=None, description="Field name or id; None: every field.")
    syscal_ms: str | None = Field(default=None, description="Where SYSCAL was read (default: ms).")
    line_channels: int = Field(
        default=0,
        ge=0,
        description="Channels left out as line emission, so only line-free channels are measured.",
        json_schema_extra=_unit("channels"),
    )
    pwv_mm: float | None = Field(
        default=None,
        ge=0,
        description="The observation's precipitable water vapour (its execution block's).",
        json_schema_extra=_unit("mm"),
    )


class Exposure(_Record):
    """What the window observed (cross-correlations)."""

    antennas: int = Field(
        description="Antennas with data the image uses (unflagged, WEIGHT within the maximum).",
        json_schema_extra=_unit("antennas"),
    )
    dish_diameter_m: float = Field(
        description="The antennas' dish diameter.", json_schema_extra=_unit("m")
    )
    baselines: int = Field(
        description="Baselines between those antennas: N (N - 1) / 2.",
        json_schema_extra=_unit("baselines"),
    )
    integrations: int = Field(
        description="Integrations (correlator dumps) on the target.",
        json_schema_extra=_unit("integrations"),
    )
    integration_s: float = Field(
        description="The length of one integration (the median INTERVAL).",
        json_schema_extra=_unit("s"),
    )
    time_on_source_s: float = Field(
        description="Integrations x their length: the time the target was observed.",
        json_schema_extra=_unit("s"),
    )
    channels: int = Field(
        description="Channels in the spectral window.", json_schema_extra=_unit("channels")
    )
    channel_width_hz: float = Field(
        description="The spacing of the channels (CHAN_WIDTH).", json_schema_extra=_unit("Hz")
    )
    noise_bandwidth_hz: float = Field(
        description="EFFECTIVE_BW, the channel's noise bandwidth: wider than the spacing "
        "when the correlator smoothed the channels.",
        json_schema_extra=_unit("Hz"),
    )
    resolution_hz: float | None = Field(
        default=None,
        description="RESOLUTION, the channel's resolution as the archive gives it.",
        json_schema_extra=_unit("Hz"),
    )
    frequency_ghz: float = Field(
        description="The window's centre frequency.", json_schema_extra=_unit("GHz")
    )
    correlations: int = Field(
        description="Polarization products in the data (2: XX and YY).",
        json_schema_extra=_unit("products"),
    )
    flagged_fraction: float = Field(
        description="The flagged share of the values on the baselines between the antennas, "
        "each value weighed by its row's integration time.",
        json_schema_extra=_unit("fraction"),
    )
    high_weight_fraction: float = Field(
        default=0.0,
        description="The share of the values, on the baselines between the antennas, that are "
        "unflagged in rows whose WEIGHT exceeds the imaging's maximum (image image-from-ms "
        "--max-weight), which the image leaves out besides the flagged ones; each value "
        "weighed by its row's integration time.",
        json_schema_extra=_unit("fraction"),
    )
    elevation_deg: float | None = Field(
        default=None,
        description="The target's median elevation.",
        json_schema_extra=_unit("deg"),
    )
    hour_angle_h: float | None = Field(
        default=None,
        description="The target's hour angle at the middle of its track.",
        json_schema_extra=_unit("h"),
    )
    track_span_s: float | None = Field(
        default=None,
        description="From the start of the first integration on the target to the end of the last.",
        json_schema_extra=_unit("s"),
    )
    scans: int | None = Field(
        default=None,
        description="The target's scans: runs of integrations without a gap longer than "
        "two integrations.",
        json_schema_extra=_unit("scans"),
    )
    pads: list[str] = Field(
        default_factory=list,
        description="The window's antennas as PAD:ANTENNA (the MS's STATION and NAME), "
        "as the archive's antenna_arrays lists them: those with data the image uses.",
    )


class ThermalNoise(_Record):
    """The noise against WEIGHT, from channel differences (lag 3 removes the sky;
    lags 1 and 2 give the correlation of neighbouring channels)."""

    scale: float = Field(
        description="The measured noise variance times WEIGHT, per real or imaginary part: "
        "1 when the data's WEIGHT column states the noise right, above 1 when the noise "
        "is larger than WEIGHT says.",
        json_schema_extra=_unit("ratio"),
    )
    rho1: float = Field(
        description="Correlation of the noise of channels one apart: 2/3 for Hanning-smoothed "
        "native channels, falling to 0.30, 0.115, 0.052 and 0.026 when 2, 4, 8 and 16 are "
        "averaged into each channel (0: independent channels).",
        json_schema_extra=_unit("correlation"),
    )
    rho2: float = Field(
        description="Correlation of the noise of channels two apart.",
        json_schema_extra=_unit("correlation"),
    )
    channel_profile: list[float] = Field(
        description="Noise variance per channel over its mean: how the noise changes "
        "across the window (1 everywhere: flat).",
        json_schema_extra=_unit("ratio per channel"),
    )
    rows_sampled: int = Field(
        description="Rows of the MS the measurement read.", json_schema_extra=_unit("rows")
    )
    row_step: int = Field(
        description="One row read in this many, evenly over the observation.",
        json_schema_extra=_unit("rows"),
    )
    estimator: Literal["clipped mean"] = Field(
        default="clipped mean", description="How the variances are averaged."
    )


class Tsys(_Record):
    """ALMA's own Tsys (SYSCAL) over the window."""

    median_k: float = Field(
        description="The system temperature ALMA measured (SYSCAL): its median over "
        "antennas, times, hands and channels.",
        json_schema_extra=_unit("K"),
    )
    profile: list[float] = Field(
        description="Tsys^2 per channel over its mean: the atmosphere's lines and the "
        "receiver's shape across the window (1 everywhere: flat).",
        json_schema_extra=_unit("ratio per channel"),
    )
    spectral_window: int = Field(description="The Tsys window read (same baseband).")


class TotalPowerNoise(_Record):
    """The 12 m total-power dishes an interferometric array correlates (on
    the T70x pads), against the array's own dishes."""

    antennas: int = Field(
        description="Total-power dishes with data.", json_schema_extra=_unit("antennas")
    )
    diameter_m: float = Field(description="Their dish diameter.", json_schema_extra=_unit("m"))
    sefd_ratio: float = Field(
        description="A total-power dish's SEFD over an own dish's: the noise variance of "
        "the baselines between the two over that of the baselines between own dishes.",
        json_schema_extra=_unit("ratio"),
    )
    tsys_ratio: float = Field(
        description="Their median SYSCAL Tsys over the own dishes' (1 without SYSCAL).",
        json_schema_extra=_unit("ratio"),
    )
    efficiency: float | None = Field(
        default=None,
        description="Their aperture x correlator efficiency: the own dishes' times their "
        "area over the total-power dish's, times tsys_ratio, over sefd_ratio.",
        json_schema_extra=_unit("fraction"),
    )


class VisibilityNoise(_Record):
    """One Stokes I visibility's noise per real and imaginary part."""

    sigma_jy: float = Field(
        description="The noise of one Stokes I visibility (one baseline, one integration, "
        "one channel), per real or imaginary part.",
        json_schema_extra=_unit("Jy"),
    )
    unit_jy: float = Field(
        description="sigma x sqrt(integration x channel width): the noise of one visibility "
        "per 1 s and 1 Hz, comparable across windows of different integration and channel "
        "width.",
        json_schema_extra=_unit("Jy s^0.5 Hz^0.5"),
    )
    hands_identical: bool = Field(description="XX and YY carry the same data (a copied Stokes I).")
    efficiency: float | None = Field(
        default=None,
        description="The aperture x correlator efficiency the noise and Tsys imply: "
        "2 k Tsys / (2 A unit) over sqrt(noise bandwidth / width), Tsys the rms of "
        "the window's Tsys spectrum.",
        json_schema_extra=_unit("fraction"),
    )
    total_power: TotalPowerNoise | None = Field(
        default=None,
        description="The total-power dishes the array correlates; None when it has none.",
    )


class AntennaShape(_Record):
    """Each antenna's own noise shape across the window, beyond the array's."""

    antennas_used: int = Field(
        description="Antennas with enough data to measure.", json_schema_extra=_unit("antennas")
    )
    channels_used: int = Field(
        description="Line-free channels measured.", json_schema_extra=_unit("channels")
    )
    excess_rms: float = Field(
        description="Fractional rms of an antenna's noise variance across the window, beyond "
        "the array's shape and beyond sampling (0: every antenna's noise has the array's shape).",
        json_schema_extra=_unit("fraction"),
    )
    half_width_channels: int = Field(
        description="Lag where that excess's autocorrelation falls to half: how wide its "
        "ripples are.",
        json_schema_extra=_unit("channels"),
    )
    half_width_mhz: float = Field(
        description="The same lag in frequency.", json_schema_extra=_unit("MHz")
    )


class AntennaNoise(_Record):
    """Each antenna's own noise level (its SEFD), from its baselines' noise
    without WEIGHT: how much the antennas' Tsys and efficiency differ."""

    antennas_used: int = Field(
        description="Antennas whose level is measured.", json_schema_extra=_unit("antennas")
    )
    baselines_used: int = Field(
        description="Baselines between them with enough values.",
        json_schema_extra=_unit("baselines"),
    )
    sefd_log_rms: float = Field(
        description="Rms of the natural log of an antenna's noise level about the array's: "
        "log of each baseline's noise variance fitted as one term per antenna.",
        json_schema_extra={"unit": "ln"},
    )


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
    antenna_noise: AntennaNoise | None = None
    notes: list[str] = Field(
        default_factory=list, description="What could not be measured, and why."
    )


class PhaseFit(_Record):
    """A phase screen fitted as sigma(b) = sigma_1km (b / 1 km)^slope."""

    bins: int = Field(description="Baseline bins the fit used.")
    slope: float
    phase_1km_deg: float
    path_um_1km: float = Field(description="The rms path difference at 1 km (microns).")
    coherence_s: float | None = Field(
        default=None, description="How long it stays coherent (s); in-scan screens only."
    )


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
