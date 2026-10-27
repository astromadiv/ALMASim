"""A noise model: named, versioned, made of components (MADIV-28).

A released model is a JSON file in ``models/`` (``alma-1.json``); a candidate
is any file that validates. The simulator receives the model's engine fields
(:meth:`NoiseModel.engine_fields`), so a model changes what is simulated only
through its components and settings.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .components import (
    COMPONENT_NAMES,
    AntennaGains,
    AntennaRipple,
    AtmosphericTsys,
    BaselineErrors,
    ChannelCorrelation,
    Components,
    FastPhase,
    Radiometer,
    ReceiverRipple,
    SlowPhase,
)


class ValueStats(BaseModel):
    """How one value was reduced from the windows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n: int = Field(ge=0, description="Windows it comes from.")
    median: float
    q16: float = Field(description="16th percentile over the windows.")
    q84: float = Field(description="84th percentile over the windows.")


class Provenance(BaseModel):
    """Where a model's values come from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str = Field(description="How the values were obtained (fit, reduce, hand-set).")
    data: str | None = Field(default=None, description="The data set the values were measured on.")
    windows: int | None = Field(
        default=None, ge=0, description="How many windows the values come from."
    )
    almasim_commit: str | None = Field(
        default=None, description="The ALMASim commit that produced them."
    )
    date: str | None = Field(default=None, description="When (ISO date).")
    notes: str | None = None
    values: dict[str, ValueStats] | None = Field(
        default=None, description="Each reduced value's statistics, by its component.parameter."
    )


class Settings(BaseModel):
    """Values several components share."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scan_length_s: float = Field(
        default=0.0,
        ge=0,
        description="Scan length: gains and the slow phase hold for one (0: one scan).",
    )
    outer_scale_km: float = Field(
        default=3.0, gt=0, description="Outer scale of the turbulent screens (km)."
    )


class NoiseModel(BaseModel):
    """A named, versioned noise model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    version: int = Field(ge=1)
    description: str = ""
    provenance: Provenance | None = None
    settings: Settings = Settings()
    components: Components = Components()

    # -- what it simulates -------------------------------------------------

    def enabled(self) -> list[str]:
        """The names of the components applied."""
        return [name for name, c in self.components.items() if c.enabled]

    def engine_fields(self) -> dict[str, Any] | None:
        """The CorruptionModel fields this model sets; None when it sets nothing."""
        values: dict[str, Any] = {}
        for _, component in self.components.items():
            if component.enabled:
                values.update(component.engine_fields())
        if not values:
            return None
        return {**self.settings.model_dump(), **values}

    def noise_config(self) -> dict[str, Any]:
        """What the model sets on the thermal noise model (NoiseModelConfig):
        its radiometer's tables, when the radiometer is applied."""
        radiometer = self.components.radiometer
        if radiometer is None or not radiometer.enabled:
            return {}
        return radiometer.noise_config()

    def select(
        self, only: Iterable[str] | None = None, without: Iterable[str] | None = None
    ) -> NoiseModel:
        """This model with only the components in ``only`` (all when None)
        applied, less those in ``without``. The others are kept, disabled, so
        the selection is part of the model's identity."""
        only_set = None if only is None else set(_names(only))
        without_set = set(_names(without or ()))
        present = dict(self.components.items())
        unknown = ((only_set or set()) | without_set) - set(present)
        if unknown:
            raise ValueError(
                f"{self.label()} has no component(s) {', '.join(sorted(unknown))}; "
                f"it has {', '.join(present) or 'none'}"
            )
        changed = {
            name: c.model_copy(update={"enabled": False})
            for name, c in present.items()
            if c.enabled
            and ((only_set is not None and name not in only_set) or name in without_set)
        }
        if not changed:
            return self
        return self.model_copy(update={"components": self.components.model_copy(update=changed)})

    # -- identity ----------------------------------------------------------

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def label(self) -> str:
        """``name@version``, with the components a selection disabled."""
        disabled = [name for name, c in self.components.items() if not c.enabled]
        return f"{self.name}@{self.version}" + (
            f" without {','.join(disabled)}" if disabled else ""
        )

    def identity(self) -> dict[str, Any]:
        """What a product records about the model that made it."""
        return {
            "name": self.name,
            "version": self.version,
            "label": self.label(),
            "sha256": self.sha256(),
            "components": self.enabled(),
        }

    # -- the earlier flat form ---------------------------------------------

    @classmethod
    def from_corruption_fields(
        cls, values: Mapping[str, Any], *, name: str = "custom", version: int = 1
    ) -> NoiseModel:
        """A model from CorruptionModel fields (the flat JSON ``--corruptions``
        took before components): each component its fields switch on."""
        from ..interferometry.corruptions import CorruptionModel

        model = CorruptionModel.from_mapping(values) or CorruptionModel()
        c = asdict(model)
        parts: dict[str, Any] = {}
        if c["radiometer_noise"]:
            parts["radiometer"] = Radiometer()
        if c["atmospheric_tsys"]:
            parts["atmospheric_tsys"] = AtmosphericTsys()
        if c["receiver_ripple_rms"] > 0:
            parts["receiver_ripple"] = ReceiverRipple(rms=c["receiver_ripple_rms"])
        if c["spectral_rho1"] or c["spectral_rho2"] or c["spectral_average"]:
            parts["channel_correlation"] = ChannelCorrelation(
                rho1=c["spectral_rho1"], rho2=c["spectral_rho2"], average=c["spectral_average"]
            )
        if c["antenna_ripple_rms"] > 0:
            parts["antenna_ripple"] = AntennaRipple(
                rms=c["antenna_ripple_rms"], scale_mhz=c["antenna_ripple_mhz"]
            )
        if c["gain_amp_rms"] > 0 or c["gain_phase_rms_deg"] > 0:
            parts["antenna_gains"] = AntennaGains(
                amp_rms=c["gain_amp_rms"], phase_rms_deg=c["gain_phase_rms_deg"]
            )
        if c["slow_path_um_1km"] > 0:
            parts["slow_phase"] = SlowPhase(
                path_um_1km=c["slow_path_um_1km"], slope=c["slow_slope"]
            )
        if c["fast_path_um_1km"] > 0:
            parts["fast_phase"] = FastPhase(
                path_um_1km=c["fast_path_um_1km"],
                slope=c["fast_slope"],
                coherence_s=c["fast_coherence_s"],
            )
        if c["baseline_amp_rms"] > 0 or c["baseline_phase_rms_deg"] > 0:
            parts["baseline_errors"] = BaselineErrors(
                amp_rms=c["baseline_amp_rms"], phase_rms_deg=c["baseline_phase_rms_deg"]
            )
        return cls(
            name=name,
            version=version,
            description="From CorruptionModel fields.",
            settings=Settings(scan_length_s=c["scan_length_s"], outer_scale_km=c["outer_scale_km"]),
            components=Components(**parts),
        )


def _names(values: Iterable[str]) -> list[str]:
    """Component names from an iterable of names or comma-separated lists."""
    names = [n.strip() for v in values for n in str(v).split(",") if n.strip()]
    bad = [n for n in names if n not in COMPONENT_NAMES]
    if bad:
        raise ValueError(
            f"unknown noise component(s) {', '.join(bad)}; known: {', '.join(COMPONENT_NAMES)}"
        )
    return names
