"""Find the raw MSs and the delivered calibration of an ALMA download (MADIV-28 T4.8)."""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def find_calibration_directory(
    input_root: str | os.PathLike[str], asdm_uid: str | None = None
) -> Path:
    """Find the calibration directory matching the requested ALMA delivery."""
    input_path = Path(input_root).expanduser().resolve()
    # Walk with pruning rather than rglob: a calibration directory holds hundreds
    # of caltable archives, and rglob("calibration") walks through all of them
    # looking for nested matches that cannot exist. Over NFS that is ~100s per
    # lookup, and this runs once per UID.
    candidates: list[Path] = []
    if asdm_uid is not None:
        # Fast path for the standard ALMA delivery layout: match the calapply
        # file directly, so only directory listings are read. Falls through to
        # the general walk below for any other layout.
        candidates = [
            hit.parent
            for hit in input_path.glob(
                f"*/science_goal.*/group.*/member.*/calibration/{asdm_uid}.ms.calapply.txt"
            )
        ]
    if not candidates:
        for dirpath, dirnames, _ in os.walk(input_path):
            if "calibration" in dirnames:
                candidates.append(Path(dirpath) / "calibration")
            dirnames[:] = [name for name in dirnames if name != "calibration"]
        if asdm_uid is not None:
            candidates = [
                path for path in candidates if (path / f"{asdm_uid}.ms.calapply.txt").is_file()
            ]
    if not candidates:
        raise RuntimeError(f"No calibration directory found under {input_path}")
    if len(candidates) > 1 and asdm_uid is None:
        raise RuntimeError(
            "Found more than one calibration directory. "
            "Pass asdm_uid to select a specific execution block."
        )
    return min(candidates)


def calibration_not_delivered(input_root: str | os.PathLike[str], asdm_uid: str) -> bool:
    """True when ALMA delivered ``asdm_uid`` but never calibrated it.

    A member OUS can ship an execution block its pipeline run left out (a QA0
    SemiPass block, see :mod:`..qa0`): its raw ASDM is under ``raw/`` and the
    member has a ``calibration`` directory, but no ``<uid>.ms.calapply.txt``
    for it. There is nothing to apply, so it is skipped rather than failed. A
    member without any ``calibration`` directory (the auxiliary products were
    never downloaded) or a layout other than ALMA's is not covered here and
    still fails in :func:`find_calibration_directory`.
    """
    input_path = Path(input_root).expanduser().resolve()
    calibration_dirs = [
        raw.parent.parent / "calibration"
        for raw in input_path.glob(f"*/science_goal.*/group.*/member.*/raw/{asdm_uid}.asdm.sdm")
    ]
    calibration_dirs = [path for path in calibration_dirs if path.is_dir()]
    return bool(calibration_dirs) and not any(
        (path / f"{asdm_uid}.ms.calapply.txt").is_file() for path in calibration_dirs
    )


def find_raw_ms_directories(
    raw_ms_root: str | os.PathLike[str],
    asdm_uid: str | None = None,
) -> list[Path]:
    """Find raw imported MS directories below ``raw_ms_root``."""
    root = Path(raw_ms_root).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"Raw MS root does not exist or is not a directory: {root}")

    raw_mss = []
    for candidate in root.rglob("*.ms"):
        if not candidate.is_dir():
            continue
        if asdm_uid is not None and candidate.name != f"{asdm_uid}.ms":
            continue
        raw_mss.append(candidate)

    if not raw_mss:
        if asdm_uid is None:
            raise RuntimeError(f"No raw *.ms directories found under {root}")
        raise RuntimeError(f"No raw MS named {asdm_uid}.ms found under {root}")

    return sorted(raw_mss)


def _parse_applycal_calls(calapply_path: Path) -> list[dict[str, Any]]:
    """Parse delivered ``applycal(...)`` lines into keyword dictionaries."""
    calls: list[dict[str, Any]] = []
    tree = ast.parse(calapply_path.read_text(encoding="utf-8"), filename=str(calapply_path))
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name) or call.func.id != "applycal":
            continue
        kwargs = {
            keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords if keyword.arg
        }
        calls.append(kwargs)

    if not calls:
        raise RuntimeError(f"No applycal calls found in {calapply_path}")
    return calls
