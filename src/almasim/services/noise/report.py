"""The report of what ``almasim noise extract`` measured (MADIV-28).

``noise extract`` writes one ``window.json`` per window: the measurements the
noise models are later fitted to. :func:`write_extract_report` reads every
``window.json`` under an extraction folder and writes, next to them:

- ``extract_report.md``: what each metric is (its unit and meaning, taken from
  the fields of :class:`~.measurements.WindowNoise`), its spread across the
  windows (median, 16-84% range, how many windows have it), one row of the
  main metrics per window, and what could not be measured and why;
- ``extract_measurements.csv``: every metric of every window, one row each,
  for a spreadsheet or pandas;
- ``extract_report.json``: the report as data, which the console's Noise
  models page shows.

A profile across the window (``thermal.channel_profile``, ``tsys.profile``)
is reported as its rms about 1, the fractional spread across the channels.
"""

from __future__ import annotations

import csv
import inspect
import io
import json
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

import numpy as np
from pydantic import BaseModel

from .measurements import WindowNoise

REPORT_FILE = "extract_report.md"
TABLE_FILE = "extract_measurements.csv"
DATA_FILE = "extract_report.json"

# The parts of window.json, in the report's order, and their headings.
SECTIONS: dict[str, str] = {
    "input": "The window measured",
    "exposure": "Exposure",
    "thermal": "Thermal noise against WEIGHT",
    "visibility": "One visibility's noise",
    "visibility.total_power": "Total-power dishes",
    "tsys": "ALMA's system temperature (SYSCAL)",
    "antenna_shape": "Each antenna's own noise shape",
    "antenna_noise": "Each antenna's own noise level",
}
# Bookkeeping, not measurements: in the table, not in the report's statistics.
_NOT_MEASURED = {
    "input.ms",
    "input.syscal_ms",
    "input.field",
    "input.spw",
    "thermal.rows_sampled",
    "thermal.row_step",
    "thermal.estimator",
    "tsys.spectral_window",
}
_SKIPPED = {"schema_version", "almasim_version", "name", "notes", "exposure.pads"}
# One row per window in the report: the metrics a reader looks at first.
SUMMARY_METRICS = (
    "exposure.frequency_ghz",
    "exposure.antennas",
    "exposure.time_on_source_s",
    "exposure.flagged_fraction",
    "thermal.scale",
    "thermal.rho1",
    "visibility.sigma_jy",
    "visibility.unit_jy",
    "visibility.efficiency",
    "tsys.median_k",
    "antenna_shape.excess_rms",
    "antenna_noise.sefd_log_rms",
)

Kind = Literal["number", "flag", "text", "profile"]


@dataclass(frozen=True)
class Metric:
    """One quantity of ``window.json``, as the report describes it."""

    key: str
    section: str
    unit: str
    meaning: str
    kind: Kind

    @property
    def measured(self) -> bool:
        """A measurement (not bookkeeping such as the MS's path)."""
        return self.key not in _NOT_MEASURED and self.kind in ("number", "flag", "profile")


def _model_in(annotation: Any) -> type[BaseModel] | None:
    """The pydantic model an annotation holds (``X``, ``X | None``), if any."""
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return annotation
    for arg in get_args(annotation):
        model = _model_in(arg)
        if model is not None:
            return model
    return None


def _kind(annotation: Any) -> Kind:
    base = annotation
    if get_origin(annotation) in (Union, UnionType):  # X | None
        base = next(a for a in get_args(annotation) if a is not type(None))
    if get_origin(base) is list:
        return "profile"
    if base is bool:
        return "flag"
    if base in (int, float):
        return "number"
    return "text"


def metrics(model: type[BaseModel] = WindowNoise, prefix: str = "") -> list[Metric]:
    """Every quantity of ``window.json`` with its unit and meaning, in file order."""
    found: list[Metric] = []
    for name, info in model.model_fields.items():
        key = f"{prefix}{name}"
        if key in _SKIPPED:
            continue
        nested = _model_in(info.annotation)
        if nested is not None:
            found.extend(metrics(nested, f"{key}."))
            continue
        kind = _kind(info.annotation)
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        meaning = info.description or ""
        if kind == "profile":
            meaning += " Reported as its rms about 1: the fractional spread across the channels."
        section = key.rsplit(".", 1)[0] if "." in key else ""
        found.append(Metric(key, section, str(extra.get("unit", "")), meaning, kind))
    return found


def _lookup(record: dict[str, Any], key: str) -> Any:
    value: Any = record
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def window_values(window: WindowNoise, keys: Sequence[Metric] | None = None) -> dict[str, Any]:
    """One window's value of each metric (None where it was not measured)."""
    record = window.model_dump()
    values: dict[str, Any] = {"name": window.name}
    for metric in keys if keys is not None else metrics():
        value = _lookup(record, metric.key)
        if metric.kind == "profile" and value:
            profile = np.asarray(value, dtype=float)
            value = float(np.sqrt(np.mean((profile - 1.0) ** 2)))
        values[metric.key] = value
    values["notes"] = "; ".join(window.notes)
    return values


def read_windows(out: str | os.PathLike[str]) -> list[WindowNoise]:
    """Every ``windows/<name>/window.json`` under ``out``, by name."""
    from .batch import WINDOW_FILE

    found = []
    for path in sorted(Path(out).glob(f"windows/*/{WINDOW_FILE}")):
        try:
            found.append(WindowNoise.model_validate_json(path.read_text()))
        except ValueError:
            continue  # a file of another schema: not this extraction's
    return found


@dataclass(frozen=True)
class Spread:
    """One metric across the windows."""

    windows: int
    median: float | None
    low: float | None
    high: float | None


def spread(values: Sequence[Any]) -> Spread:
    """The median and 16-84% range of the windows that have the metric."""
    numbers = [float(v) for v in values if v is not None and not _nan(v)]
    if not numbers:
        return Spread(0, None, None, None)
    low, median, high = np.percentile(numbers, [16, 50, 84])
    return Spread(len(numbers), float(median), float(low), float(high))


def _finite(value: Any) -> Any:
    """None for a NaN, so the JSON stays JSON."""
    return None if _nan(value) else value


def _nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _number(value: Any) -> str:
    if value is None or _nan(value):
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _section_intro(section: str) -> str:
    """The docstring of the part of window.json a section reports."""
    model: type[BaseModel] | None = WindowNoise
    for part in section.split("."):
        field = model.model_fields.get(part) if model else None
        model = _model_in(field.annotation) if field else None
    return inspect.cleandoc(model.__doc__ or "").replace("\n", " ") if model else ""


def _metric_entry(metric: Metric, values: Sequence[Any]) -> dict[str, Any]:
    """A metric with its spread across the windows (``windows``: how many have it)."""
    entry: dict[str, Any] = {
        "key": metric.key,
        "unit": metric.unit,
        "meaning": metric.meaning,
        "kind": metric.kind,
        "measured": metric.measured,
        "windows": sum(1 for v in values if v is not None and not _nan(v)),
    }
    if metric.kind == "flag":
        entry["yes"] = sum(1 for v in values if v)
    elif metric.measured:
        found = spread(values)
        entry.update(median=found.median, low=found.low, high=found.high)
    return entry


def report_data(
    windows: Sequence[WindowNoise], failures: dict[str, str] | None = None
) -> dict[str, Any]:
    """Everything the report shows, as data (``extract_report.json``, the console's)."""
    catalogue = metrics()
    rows = [window_values(w, catalogue) for w in windows]
    sections = []
    for section, title in SECTIONS.items():
        members = [m for m in catalogue if m.section == section]
        intro = _section_intro(section)
        sections.append(
            {
                "key": section,
                "title": title,
                "intro": intro if intro.rstrip(".") != title else "",
                "metrics": [_metric_entry(m, [row[m.key] for row in rows]) for m in members],
            }
        )
    return {
        "windows": len(windows),
        "failures": dict(sorted((failures or {}).items())),
        "sections": sections,
        "summary_metrics": list(SUMMARY_METRICS),
        "rows": [
            {"name": row["name"], **{k: _finite(row.get(k)) for k in SUMMARY_METRICS}}
            for row in rows
        ],
        "notes": {w.name: list(w.notes) for w in windows if w.notes},
    }


def _metric_line(entry: dict[str, Any]) -> str:
    """A metric's line of the report: its unit, spread across the windows and meaning."""
    if entry["kind"] == "flag":
        stats = [f"{entry['yes']} yes", ""]
    elif entry["measured"]:
        spread_text = (
            f"{_number(entry['low'])} - {_number(entry['high'])}" if entry["windows"] else "-"
        )
        stats = [_number(entry["median"]), spread_text]
    else:  # bookkeeping: which MS, which rows
        stats = ["", ""]
    return (
        f"| `{entry['key']}` | {entry['unit']} | "
        + " | ".join([*stats, str(entry["windows"])])
        + f" | {_cell(entry['meaning'])} |"
    )


def render_report(data: dict[str, Any]) -> str:
    """The Markdown report of :func:`report_data`."""
    failures: dict[str, str] = data["failures"]
    lines = [
        "# Noise measured on real windows (`almasim noise extract`)",
        "",
        f"{data['windows']} window(s) measured"
        + (f", {len(failures)} failed (see the end)." if failures else "."),
        "Every value below is in each window's `windows/<name>/window.json`, and all of "
        f"them, one row per window, in `{TABLE_FILE}`. The noise models are fitted to "
        "these measurements (`noise fit`, `noise aggregate`); this report shows the "
        "measurements themselves.",
        "",
        "For each metric: its unit, what it means, and its spread across the windows "
        "that have it (median and 16-84% range). How each is measured, and which noise "
        "model component it sets, is in the noise models guide (docs/noise_models.md, "
        "*The components, one by one*).",
    ]
    for section in data["sections"]:
        if not section["metrics"]:
            continue
        lines += ["", f"## {section['title']}", ""]
        if section["intro"]:
            lines += [section["intro"], ""]
        lines += [
            "| metric | unit | median | 16-84% | windows | meaning |",
            "|---|---|---|---|---|---|",
            *(_metric_line(entry) for entry in section["metrics"]),
        ]
    if data["rows"]:
        keys = data["summary_metrics"]
        lines += [
            "",
            "## Each window",
            "",
            "| window | " + " | ".join(f"`{k}`" for k in keys) + " |",
            "|---" * (len(keys) + 1) + "|",
        ]
        for row in data["rows"]:
            lines.append(f"| {row['name']} | " + " | ".join(_number(row[k]) for k in keys) + " |")
    if data["notes"]:
        lines += ["", "## What could not be measured", ""]
        lines += [f"- **{name}**: " + "; ".join(notes) for name, notes in data["notes"].items()]
    if failures:
        lines += ["", "## Windows that failed", ""]
        lines += [f"- **{name}**: {_cell(error)}" for name, error in failures.items()]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class ExtractReport:
    """What :func:`write_extract_report` wrote."""

    report: Path
    table: Path
    data: Path
    windows: int


def _read_failures(path: Path) -> dict[str, str]:
    try:
        found = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in found.items()} if isinstance(found, dict) else {}


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_extract_report(
    out: str | os.PathLike[str], *, echo: Callable[[str], Any] | None = None
) -> ExtractReport:
    """Write the report of the extraction in ``out`` (its ``windows/*/window.json`` and
    ``failures.json``): ``extract_report.md``, ``extract_report.json`` and
    ``extract_measurements.csv``; ``echo`` gets a short summary."""
    from .batch import FAILURES_FILE

    out = Path(out)
    windows = read_windows(out)
    catalogue = metrics()
    out.mkdir(parents=True, exist_ok=True)
    table = out / TABLE_FILE
    with io.StringIO() as buffer:
        writer = csv.DictWriter(
            buffer, ["name", *(m.key for m in catalogue), "notes"], lineterminator="\n"
        )
        writer.writeheader()
        for window in windows:
            values = window_values(window, catalogue)
            writer.writerow({k: "" if v is None else v for k, v in values.items()})
        _write_atomic(table, buffer.getvalue())
    data = report_data(windows, _read_failures(out / FAILURES_FILE))
    data_path = out / DATA_FILE
    _write_atomic(data_path, json.dumps(data, indent=1, allow_nan=False) + "\n")
    report = out / REPORT_FILE
    _write_atomic(report, render_report(data))
    if echo is not None:
        for line in summary_lines(windows):
            echo(line)
        echo(f"Report: {report}")
        echo(f"Every metric of every window: {table}")
    return ExtractReport(report, table, data_path, len(windows))


def summary_lines(windows: Sequence[WindowNoise]) -> list[str]:
    """The main metrics across the windows, one line each, for the terminal."""
    if not windows:
        return ["No window measured."]
    by_key = {m.key: m for m in metrics()}
    rows = [window_values(w) for w in windows]
    lines = [f"Measured on {len(windows)} window(s) (median, 16-84% range):"]
    for key in SUMMARY_METRICS:
        found = spread([row[key] for row in rows])
        if not found.windows:
            continue
        unit = by_key[key].unit
        lines.append(
            f"  {key:<28} {_number(found.median):>9}  "
            f"({_number(found.low)} - {_number(found.high)}) {unit}"
            + (f"  [{found.windows} windows]" if found.windows < len(windows) else "")
        )
    return lines
