"""The noise model's components: each one thing calibrated ALMA data carry
besides the sky, with its own switch and parameters (MADIV-28).

A component is a pydantic model with an ``enabled`` switch, its parameters
(bounds and units in the field), and ``engine_fields``: the values it sets on
the simulator's :class:`~almasim.services.interferometry.corruptions.CorruptionModel`.
A component left out of a model, or disabled, sets none of them, so the
simulator applies nothing of it.

To add a component: write its class here (the parameters, ``engine_fields``),
add it as a field of :class:`Components`, and make the simulator apply the
engine fields it sets. The JSON Schema, the model files' validation, the
selection (``--noise-only``, ``--noise-without``) and ``almasim noise show``
follow from the field.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class NoiseComponent(BaseModel):
    """One component of a noise model; its subclasses add the parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(default=True, description="Apply this component.")

    def engine_fields(self) -> dict[str, Any]:
        """The CorruptionModel fields this component sets (when enabled)."""
        raise NotImplementedError


class Radiometer(NoiseComponent):
    """Each visibility's thermal noise from the radiometer equation (Tsys,
    channel noise bandwidth, integration, dish), the source keeping its flux,
    instead of a noise set from the run's target S/N."""

    def engine_fields(self) -> dict[str, Any]:
        return {"radiometer_noise": True}


class AtmosphericTsys(NoiseComponent):
    """The noise across the window follows the atmosphere's lines in Tsys,
    in the signal and the image band."""

    def engine_fields(self) -> dict[str, Any]:
        return {"atmospheric_tsys": True}


class ReceiverRipple(NoiseComponent):
    """The receiver's own smooth ripple of Tsys across the window."""

    rms: float = Field(ge=0, le=1, description="Fractional rms of Tsys.")

    def engine_fields(self) -> dict[str, Any]:
        return {"receiver_ripple_rms": self.rms}


class ChannelCorrelation(NoiseComponent):
    """Correlated noise of neighbouring channels (ALMA's correlator
    Hanning-smooths native channels)."""

    rho1: float = Field(ge=0, le=1, description="Correlation of native channels one apart.")
    rho2: float = Field(ge=0, le=0.25, description="Correlation of native channels two apart.")
    average: int = Field(
        default=0,
        ge=0,
        description="Native channels averaged into each channel, diluting the correlation; "
        "0: the window's own, from the archive's resolution.",
    )

    def engine_fields(self) -> dict[str, Any]:
        return {
            "spectral_rho1": self.rho1,
            "spectral_rho2": self.rho2,
            "spectral_average": self.average,
        }


class AntennaRipple(NoiseComponent):
    """Each antenna's own smooth ripple of its noise across the window."""

    rms: float = Field(ge=0, le=1, description="Fractional rms of the noise variance.")
    scale_mhz: float = Field(gt=0, description="Frequency scale of the ripple (MHz).")

    def engine_fields(self) -> dict[str, Any]:
        return {"antenna_ripple_rms": self.rms, "antenna_ripple_mhz": self.scale_mhz}


class AntennaGains(NoiseComponent):
    """Antenna gain errors left by calibration, per scan."""

    amp_rms: float = Field(default=0.0, ge=0, le=1, description="Fractional amplitude rms.")
    phase_rms_deg: float = Field(default=0.0, ge=0, le=180, description="Phase rms (degrees).")

    def engine_fields(self) -> dict[str, Any]:
        return {"gain_amp_rms": self.amp_rms, "gain_phase_rms_deg": self.phase_rms_deg}


class SlowPhase(NoiseComponent):
    """The atmospheric phase phase-referencing does not transfer, per scan:
    a turbulent screen growing with baseline length."""

    path_um_1km: float = Field(ge=0, description="Rms path difference at 1 km (microns).")
    slope: float = Field(
        default=5 / 6, gt=0, le=2, description="Growth with separation (Kolmogorov 5/6)."
    )

    def engine_fields(self) -> dict[str, Any]:
        return {"slow_path_um_1km": self.path_um_1km, "slow_slope": self.slope}


class FastPhase(NoiseComponent):
    """The atmospheric phase within each scan, which decorrelates the scan's average."""

    path_um_1km: float = Field(ge=0, description="Rms path difference at 1 km (microns).")
    slope: float = Field(
        default=5 / 6, gt=0, le=2, description="Growth with separation (Kolmogorov 5/6)."
    )
    coherence_s: float = Field(default=10.0, gt=0, description="Coherence time of the screen (s).")

    def engine_fields(self) -> dict[str, Any]:
        return {
            "fast_path_um_1km": self.path_um_1km,
            "fast_slope": self.slope,
            "fast_coherence_s": self.coherence_s,
        }


class BaselineErrors(NoiseComponent):
    """Errors of a baseline that do not factor into its antennas' (closure
    errors self-calibration cannot remove), per scan."""

    amp_rms: float = Field(default=0.0, ge=0, le=1, description="Fractional amplitude rms.")
    phase_rms_deg: float = Field(default=0.0, ge=0, le=180, description="Phase rms (degrees).")

    def engine_fields(self) -> dict[str, Any]:
        return {"baseline_amp_rms": self.amp_rms, "baseline_phase_rms_deg": self.phase_rms_deg}


class Components(BaseModel):
    """A noise model's components by name; a component left out is not applied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    radiometer: Radiometer | None = None
    atmospheric_tsys: AtmosphericTsys | None = None
    receiver_ripple: ReceiverRipple | None = None
    channel_correlation: ChannelCorrelation | None = None
    antenna_ripple: AntennaRipple | None = None
    antenna_gains: AntennaGains | None = None
    slow_phase: SlowPhase | None = None
    fast_phase: FastPhase | None = None
    baseline_errors: BaselineErrors | None = None

    def items(self) -> list[tuple[str, NoiseComponent]]:
        """The components present, in declaration order."""
        return [
            (name, c) for name in type(self).model_fields if (c := getattr(self, name)) is not None
        ]


COMPONENT_NAMES: tuple[str, ...] = tuple(Components.model_fields)
