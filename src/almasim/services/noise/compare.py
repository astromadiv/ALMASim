"""Simulated windows against their real counterparts (MADIV-28).

Each window contributes its simulated and real measurements: the image
statistics (:mod:`.image_stats`) and the visibility ones (``noise extract``'s
window.json). Per metric, across the windows: the median of the per-window
ratio (sim over real; for a scale) or difference (for a shape), its 16-84%
range, a paired Wilcoxon signed-rank p, and how many windows fall within
10%. The criteria then say whether the model passes: the median image and
visibility noise within ``noise_tolerance`` of the real, the simulated
visibilities' variance within ``weight_tolerance`` of what their WEIGHT
claims, no listed metric differing at p below ``min_p``, and no band or
array group's median noise off by more than ``group_tolerance``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .measurements import WindowNoise

# metric -> (ratio or difference)
METRICS: dict[str, Literal["ratio", "difference"]] = {
    "cont.sigma": "ratio",
    "cube.sigma_median": "ratio",
    "visibility.unit_jy": "ratio",
    "cont.corr_area_beams": "ratio",
    "cube.corr_area_beams": "ratio",
    "cont.kurtosis": "difference",
    "cube.kurtosis": "difference",
    "cube.sigma_spread": "difference",
    "cube.adjacent_corr": "difference",
    "thermal.rho1": "difference",
    "antenna_shape.excess_rms": "difference",
}
NOISE_LEVELS = ("cont.sigma", "cube.sigma_median", "visibility.unit_jy")


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Criteria(_Record):
    """When a model passes."""

    noise_tolerance: float = Field(
        default=0.05, gt=0, description="Median noise ratio within 1 +- this."
    )
    weight_tolerance: float = Field(
        default=0.01, gt=0, description="Sim variance x WEIGHT within 1 +- this."
    )
    min_p: float = Field(
        default=0.01, ge=0, le=1, description="No checked metric with Wilcoxon p below."
    )
    group_tolerance: float = Field(
        default=0.10, gt=0, description="Each band or array group within 1 +- this."
    )
    checked: tuple[str, ...] = Field(
        default=NOISE_LEVELS, description="Metrics whose p is checked."
    )


class MetricSummary(_Record):
    kind: Literal["ratio", "difference"]
    n: int
    median: float
    q16: float
    q84: float
    p: float | None = Field(description="Paired Wilcoxon signed-rank p (n >= 6).")
    within_10pct: int | None = Field(default=None, description="Windows within 10% (ratios).")


class Check(_Record):
    name: str
    passed: bool
    value: float | None = None
    detail: str = ""


class Comparison(_Record):
    """A model's simulated windows against the real ones."""

    schema_version: Literal[1] = 1
    model: dict = Field(description="The model's identity (name, version, label, sha256).")
    windows: int
    criteria: Criteria
    metrics: dict[str, MetricSummary]
    groups: dict[str, MetricSummary] = Field(description="cont.sigma by band and array.")
    checks: list[Check]
    passed: bool


@dataclass(frozen=True)
class Pair:
    """One window's measurements: simulated and real."""

    name: str
    sim_image: Mapping[str, float]
    real_image: Mapping[str, float]
    sim_window: WindowNoise | None
    real_window: WindowNoise | None
    band: str = ""


def compare(pairs: Sequence[Pair], model: dict, criteria: Criteria | None = None) -> Comparison:
    """Every metric across the windows, and the criteria's verdict."""
    criteria = criteria or Criteria()
    values = {metric: _values(pairs, metric) for metric in METRICS}
    metrics = {
        metric: summary
        for metric, (sim, real) in values.items()
        if (summary := _summary(sim, real, METRICS[metric])) is not None
    }
    groups: dict[str, MetricSummary] = {}
    for key in sorted({p.band for p in pairs if p.band}):
        members = [p for p in pairs if p.band == key]
        summary = _summary(*_values(members, "cont.sigma"), "ratio")
        if summary is not None:
            groups[key] = summary
    checks = _checks(metrics, groups, pairs, criteria)
    return Comparison(
        model=model,
        windows=len(pairs),
        criteria=criteria,
        metrics=metrics,
        groups=groups,
        checks=checks,
        passed=bool(checks) and all(c.passed for c in checks),
    )


def _values(pairs: Sequence[Pair], metric: str) -> tuple[np.ndarray, np.ndarray]:
    sim, real = [], []
    for p in pairs:
        a, b = _value(p, metric, simulated=True), _value(p, metric, simulated=False)
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
            sim.append(a)
            real.append(b)
    return np.asarray(sim, dtype=float), np.asarray(real, dtype=float)


def _value(pair: Pair, metric: str, *, simulated: bool) -> float | None:
    head = metric.split(".", 1)[0]
    if head in ("cont", "cube", "q"):
        return (pair.sim_image if simulated else pair.real_image).get(metric)
    window = pair.sim_window if simulated else pair.real_window
    if window is None:
        return None
    node: object = window
    for part in metric.split("."):
        node = getattr(node, part, None)
        if node is None:
            return None
    return float(node) if isinstance(node, int | float) else None


def _summary(sim: np.ndarray, real: np.ndarray, kind: str) -> MetricSummary | None:
    if kind == "ratio":
        ok = (sim > 0) & (real > 0)
        sim, real = sim[ok], real[ok]
        per = sim / real
        tested = np.log(per)
    else:
        per = sim - real
        tested = per
    if per.size == 0:
        return None
    p = None
    if per.size >= 6 and np.any(tested != 0):
        from scipy.stats import wilcoxon

        p = float(wilcoxon(tested).pvalue)
    q16, median, q84 = np.percentile(per, [16, 50, 84])
    return MetricSummary(
        kind=kind,  # type: ignore[arg-type]
        n=int(per.size),
        median=float(median),
        q16=float(q16),
        q84=float(q84),
        p=p,
        within_10pct=int(np.sum(np.abs(per - 1) <= 0.1)) if kind == "ratio" else None,
    )


def _checks(
    metrics: dict[str, MetricSummary],
    groups: dict[str, MetricSummary],
    pairs: Sequence[Pair],
    criteria: Criteria,
) -> list[Check]:
    checks = []
    for metric in NOISE_LEVELS:
        summary = metrics.get(metric)
        if summary is not None:
            off = abs(summary.median - 1)
            checks.append(
                Check(
                    name=f"{metric} median within {criteria.noise_tolerance:.0%}",
                    passed=off <= criteria.noise_tolerance,
                    value=summary.median,
                )
            )
    scales = [p.sim_window.thermal.scale for p in pairs if p.sim_window is not None]
    if scales:
        median = float(np.median(scales))
        checks.append(
            Check(
                name=f"simulated variance x WEIGHT within {criteria.weight_tolerance:.0%}",
                passed=abs(median - 1) <= criteria.weight_tolerance,
                value=median,
            )
        )
    for metric in criteria.checked:
        summary = metrics.get(metric)
        if summary is not None and summary.p is not None:
            checks.append(
                Check(
                    name=f"{metric} does not differ (p >= {criteria.min_p})",
                    passed=summary.p >= criteria.min_p,
                    value=summary.p,
                )
            )
    for key, summary in groups.items():
        if summary.n >= 3:
            checks.append(
                Check(
                    name=f"cont.sigma of {key} within {criteria.group_tolerance:.0%}",
                    passed=abs(summary.median - 1) <= criteria.group_tolerance,
                    value=summary.median,
                    detail=f"{summary.n} windows",
                )
            )
    return checks
