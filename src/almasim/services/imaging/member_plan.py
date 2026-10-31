"""Member imaging: one image per window and field from every execution block of a member OUS.

ALMA images a member from all the execution blocks it calibrated for it. The
parameter rows (``compute_imaging_parameters``) are one per MS, window and
field; :func:`member_parameters` merges the rows of one member's EBs into one
per window and field, matching windows by name (an EB numbers its windows its
own way) and fields by name. The merged row keeps the earliest EB's row, with
the beam of the longest baseline and the time of all the EBs, and lists every
EB's MS, window and fields in ``member_windows``; the imaging task combines
them into one MS (``build_member_window_ms``).

Only a complete member is imaged: every EB ALMA calibrated for it has a
calibrated MS (``member_status``) and a parameter row. A mosaic, whose
pointings share the field's name, is not: its pointings cannot be told apart
by name from one EB to the next.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

from almasim.services.imaging.provenance import eb_uid_from_ms

MEMBER_WINDOWS_COLUMN = "member_windows"
WINDOW_NAME_COLUMN = "spectral_window_name"


def add_window_names(parameters: pd.DataFrame) -> pd.DataFrame:
    """Fill ``spectral_window_name`` from each MS when a CSV predates it."""
    if WINDOW_NAME_COLUMN in parameters.columns:
        return parameters
    from almasim.services.imaging.ms_geometry import import_casacore_tables
    from almasim.services.ms.tables import open_table

    opener = import_casacore_tables()
    names: dict[str, list[str]] = {}
    for filename in parameters["filename"].unique():
        with open_table(f"{filename}::SPECTRAL_WINDOW", opener=opener) as spectral_windows:
            names[filename] = [str(name) for name in spectral_windows.getcol("NAME")]
    filled = parameters.copy()
    filled[WINDOW_NAME_COLUMN] = [
        names[f][int(s)]
        for f, s in zip(filled["filename"], filled["spectral_window_id"], strict=True)
    ]
    return filled


def member_parameters(
    parameters: pd.DataFrame,
    index: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """The parameter rows merged per member, window and field; and what is left out.

    ``index`` is the delivery tree's ``ous_index.csv`` (``read_ous_index``).
    Returns ``(rows, skipped)``: one row per (member, window name, field name)
    of every complete member, and ``(member OUS UID, why)`` for each member, or
    field of a member, not imaged. A window or field missing from some of a
    member's EBs is imaged from the EBs that have it.
    """
    from almasim.services.archive.delivery import member_status

    parameters = add_window_names(parameters)
    by_eb: dict[str, pd.DataFrame] = {}
    for filename, rows in parameters.groupby("filename"):
        eb = eb_uid_from_ms(Path(str(filename)))
        if eb is not None:
            by_eb[eb] = rows
    merged: list[pd.Series] = []
    skipped: list[tuple[str, str]] = []
    for member, status in sorted(member_status(index).items()):
        ebs = [eb for eb in status.needed if eb in by_eb]
        if not ebs:
            continue  # none of its MSs is in these parameters
        if status.missing:
            skipped.append(
                (
                    member,
                    f"{len(status.missing)} of its {len(status.needed)} execution blocks have "
                    f"no calibrated MS ({', '.join(status.missing)})",
                )
            )
            continue
        without_rows = [eb for eb in status.needed if eb not in by_eb]
        if without_rows:
            skipped.append(
                (
                    member,
                    f"{len(without_rows)} of its {len(status.needed)} execution blocks have "
                    f"no parameter rows ({', '.join(without_rows)})",
                )
            )
            continue
        rows = pd.concat([by_eb[eb] for eb in ebs]).sort_values("start_mjd_s", kind="stable")
        mosaics: set[str] = set()
        for (_, field), group in rows.groupby([WINDOW_NAME_COLUMN, "field_name"], sort=False):
            if group["filename"].duplicated().any():
                mosaics.add(str(field))
                continue
            merged.append(_merge(group, member, status.member_folder))
        skipped.extend(
            (member, f"field {field} is a mosaic (several pointings of that name)")
            for field in sorted(mosaics)
        )
    if not merged:
        return parameters.iloc[0:0].assign(**{MEMBER_WINDOWS_COLUMN: []}), skipped
    return pd.DataFrame(merged).reset_index(drop=True), skipped


def _merge(group: pd.DataFrame, member: str, member_folder: str) -> pd.Series:
    """One member's rows of one window and field as one row (``group`` in time order)."""
    first = group.iloc[0].copy()
    first[MEMBER_WINDOWS_COLUMN] = json.dumps(
        [
            [str(row["filename"]), int(row["spectral_window_id"]), _field_ids(row)]
            for _, row in group.iterrows()
        ]
    )
    first["member_ous_uid"] = member
    first["member_folder"] = member_folder
    # The longest baseline sets the beam, so the cell samples every EB's.
    longest = group["max_baseline_size"].astype(float).idxmax()
    first["max_baseline_size"] = group.loc[longest, "max_baseline_size"]
    first["synthetized_beam_size"] = group.loc[longest, "synthetized_beam_size"]
    first["end_mjd_s"] = group["end_mjd_s"].max()
    first["end_datetime"] = group.loc[group["end_mjd_s"].idxmax(), "end_datetime"]
    first["duration_s"] = group["duration_s"].astype(float).sum()
    first["n_visibility_rows"] = group["n_visibility_rows"].astype(int).sum()
    # The combined MS holds the one field, renumbered 0.
    first["target_field_ids"] = "0"
    return first


def _field_ids(row: pd.Series) -> list[int]:
    value = row.get("target_field_ids")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip().removesuffix(".0")
    return [int(f) for f in text.split(",") if f.strip()]


def is_member_row(row: pd.Series) -> bool:
    value = row.get(MEMBER_WINDOWS_COLUMN) if hasattr(row, "get") else None
    return isinstance(value, str) and bool(value)


def member_windows(row: pd.Series) -> tuple[tuple[Path, int, tuple[int, ...]], ...]:
    """A merged row's (MS, window, field ids) of every EB, in time order."""
    return tuple(
        (Path(ms), int(spw), tuple(int(f) for f in fields))
        for ms, spw, fields in json.loads(row[MEMBER_WINDOWS_COLUMN])
    )


def member_task_name(row: pd.Series) -> str:
    """``SPW-<n>_<field>``: the earliest EB's window id and the field's name."""
    field = re.sub(r"[^A-Za-z0-9+.-]+", "_", str(row["field_name"])).strip("_") or "field"
    return f"SPW-{int(row['spectral_window_id'])}_{field}"
