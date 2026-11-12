"""Assess every imaged task and summarise the results in a CSV (MADIV-28 T4.10)."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..products import CUBE_IMAGE_FILENAME, IMAGE_FILENAME
from .continuum import continuum_quality
from .cube import cube_quality

QUALITY_FILENAME = "quality.json"
SUMMARY_FILENAME = "clean-quality.csv"


def find_task_dirs(root: Path) -> list[Path]:
    """Every directory under ``root`` (``root`` included) holding a continuum or cube image."""
    root = Path(root)
    names = (IMAGE_FILENAME, CUBE_IMAGE_FILENAME)
    found = {path.parent for name in names for path in root.rglob(name)}
    return sorted(found)


def assess_task(task_dir: Path, *, write_moments: bool = True) -> dict[str, Any]:
    """Continuum and cube statistics of one task, also written to its ``quality.json``."""
    task_dir = Path(task_dir)
    result: dict[str, Any] = {"task": str(task_dir)}
    for key, stats in (
        ("continuum", continuum_quality(task_dir)),
        ("cube", cube_quality(task_dir, write_moments=write_moments)),
    ):
        if stats is not None:
            result[key] = stats
    (task_dir / QUALITY_FILENAME).write_text(
        json.dumps(result, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


SUMMARY_COLUMNS = (
    ("continuum", "snr"),
    ("continuum", "residual_share_1beam"),
    ("continuum", "residual_share_2beam"),
    ("continuum", "residual_peak_sigma"),
    ("continuum", "residual_core_sigma"),
    ("continuum", "residual_antisymmetric_sigma"),
    ("continuum", "noise_at_mask_creation_ratio"),
    ("continuum", "model_flux_jy"),
    ("continuum", "diverged"),
    ("cube", "line_channels"),
    ("cube", "channel_noise_mjy"),
    ("cube", "linefree_residual_sigma"),
    ("cube", "line_residual_share"),
    ("cube", "line_residual_peak_sigma"),
    ("cube", "model_flux_jy"),
    ("cube", "diverged"),
)


def summary_rows(results: Iterable[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    """One flat row per task: its path relative to ``root`` and ``SUMMARY_COLUMNS``."""
    rows = []
    for result in results:
        task = Path(result["task"])
        row: dict[str, Any] = {"task": str(task.relative_to(root)) if task != root else "."}
        for section, key in SUMMARY_COLUMNS:
            row[f"{section}.{key}"] = result.get(section, {}).get(key)
        rows.append(row)
    return rows


def write_summary(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """``rows`` as CSV at ``path``."""
    columns = ["task"] + [f"{section}.{key}" for section, key in SUMMARY_COLUMNS]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
