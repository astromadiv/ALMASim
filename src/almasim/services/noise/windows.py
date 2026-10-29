"""The windows table and archive rows of an ALMASim imaging run (MADIV-28).

``noise extract``, ``fit`` and ``validate`` take a windows table (one window
per row, see :mod:`.batch`) and ``validate`` also each window's archive row
(``--metadata``, by its ``name``). An imaging run holds most of both: its
parameters CSV names each window's MS, spectral window and field, and its
output folder each window's task (the clean model ``noise fit`` needs) and log
(the line channels ``noise extract`` leaves out). The archive search the data
came from gives the rest: each observation's row, whose ``PWV`` the receiver
temperatures need. An MS names its execution block, not its observation, so
the resolved-products CSV (whose file names carry both) links the two.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .batch import window_name

WINDOW_COLUMNS = ("ms", "spw", "field", "name", "pwv", "task_log", "task_dir")
# image clean-quality's per-task summary: noise fit and validate read it.
QUALITY_FILE = "quality.json"


@dataclass
class ImagingWindows:
    """The windows of an imaging run, ready for ``noise extract``: the windows
    table, each window's archive row (with its ``name``) and the windows left
    out, with why."""

    windows: pd.DataFrame
    rows: pd.DataFrame
    skipped: list[tuple[str, str]] = field(default_factory=list)


def imaging_windows(
    parameters: pd.DataFrame,
    images: Path,
    metadata: pd.DataFrame,
    observations: dict[str, str],
    *,
    done_only: bool = True,
    science_fields: Callable[[str], Iterable[int]] | None = None,
    max_windows: int | None = None,
    names: set[str] | None = None,
) -> ImagingWindows:
    """One window per (MS, spectral window, field) row of ``parameters`` (an
    ``image compute-parameters`` CSV) imaged under ``images``.

    ``observations`` maps execution block uids to member OUS uids
    (:func:`almasim.services.simulation.input_cubes.observations_by_execution_block`);
    a window's archive row is its observation's row in ``metadata`` (an
    ``almasim metadata`` CSV), the one of its own source when the observation
    has several. With ``done_only`` a window whose imaging task has no done
    marker, or no ``quality.json`` (``image clean-quality``'s, which ``noise
    fit`` and ``validate`` read), is left out; ``science_fields(ms)``, when
    given, says which fields of an MS are science targets (others are left
    out); ``max_windows`` keeps the first that many; ``names``, when given,
    keeps only the windows so named.
    """
    from ..imaging.markers import imaging_log_path
    from ..imaging.wsclean_plan import task_directory
    from ..simulation.input_cubes import uids_in

    rows_by_ous: dict[str, list[int]] = {}
    for i, ous in enumerate(metadata.get("member_ous_uid", pd.Series(dtype=str))):
        rows_by_ous.setdefault(str(ous).lower(), []).append(i)
    science: dict[str, set[int]] = {}
    windows: list[dict[str, str]] = []
    rows: list[dict[str, object]] = []
    skipped: list[tuple[str, str]] = []
    for _, row in parameters.iterrows():
        ms = str(Path(str(row["filename"])).expanduser().absolute())
        spw = int(row["spectral_window_id"])
        task_dir, field_ids = task_directory(images, row)
        field_id = str(field_ids[0]) if len(field_ids) == 1 else None
        name = window_name(ms, spw, field_id)
        if names is not None and name not in names:
            continue
        why = _left_out(task_dir, done_only) or _not_science(ms, field_id, science_fields, science)
        if why:
            skipped.append((name, why))
            continue
        ous = {observations[uid] for uid in uids_in(ms) if uid in observations}
        matches = [i for o in sorted(ous) for i in rows_by_ous.get(o, [])]
        if not matches:
            skipped.append((name, "no archive row for its observation"))
            continue
        source = str(row.get("field_name", "") or "")
        own = [i for i in matches if str(metadata.iloc[i].get("ALMA_source_name", "")) == source]
        archive = metadata.iloc[(own or matches)[0]].to_dict()
        windows.append(
            {
                "ms": ms,
                "spw": str(spw),
                "field": field_id or "",
                "name": name,
                "pwv": _pwv(archive.get("PWV")),
                "task_log": str(imaging_log_path(task_dir)),
                "task_dir": str(task_dir),
            }
        )
        rows.append({**archive, "name": name})
        if max_windows is not None and len(windows) >= max_windows:
            break
    return ImagingWindows(
        windows=pd.DataFrame(windows, columns=list(WINDOW_COLUMNS)),
        rows=pd.DataFrame(rows),
        skipped=skipped,
    )


def _left_out(task_dir: Path, done_only: bool) -> str | None:
    """Why an imaging task is not ready to measure, or None."""
    from ..imaging.markers import imaging_marker_path

    if not done_only:
        return None
    if not imaging_marker_path(task_dir).is_file():
        return "its imaging task is not done"
    if not (task_dir / QUALITY_FILE).is_file():
        return f"no {QUALITY_FILE}: run image clean-quality on the images"
    return None


def _not_science(
    ms: str,
    field_id: str | None,
    science_fields: Callable[[str], Iterable[int]] | None,
    science: dict[str, set[int]],
) -> str | None:
    """Why a field is left out as not a science target, or None; ``science``
    caches each MS's science fields."""
    if science_fields is None or field_id is None:
        return None
    if ms not in science:
        science[ms] = {int(f) for f in science_fields(ms)}
    return None if int(field_id) in science[ms] else f"field {field_id} is not a science target"


def _pwv(value: object) -> str:
    """The archive's PWV (mm) as the windows table holds it; empty when unknown."""
    try:
        pwv = float(str(value))
    except ValueError:
        return ""
    return "" if math.isnan(pwv) or pwv <= 0 else f"{pwv:g}"
