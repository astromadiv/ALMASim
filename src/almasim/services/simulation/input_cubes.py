"""Which metadata rows an input sky-model cube belongs to.

An input path is one cube, or a folder of them. In an imaging run's folder
(``<ms>/SPW-<n>_FIELD-<m>/``, as ``almasim image image-from-ms`` writes it)
each spectral window's sky model is its clean-component cube,
``wsclean-cube-model.fits``; other folders give every FITS file.

A cube's path names its data by ALMA uid: an execution block
(``uid___A002_X113d9b7_Xb670``) or an observation (a member OUS,
``uid___A001_X362b_Xf37``). A metadata row is an observation, so an execution
block is looked up in the resolved products (``almasim products resolve``),
whose rows give each product's file name and its observation. Every
spectral window of an observation matches its row: one simulation per cube.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MODEL_CUBE = "wsclean-cube-model.fits"
FITS_SUFFIXES = (".fits", ".fit", ".fts", ".fz")
# uid___A002_X113d9b7_Xb670 in a path is uid://A002/X113d9b7/Xb670.
_PATH_UID = re.compile(r"uid___(A\d+)_(X[0-9a-z]+)_(X[0-9a-z]+)", re.IGNORECASE)


@dataclass(frozen=True)
class CubeMatch:
    """One simulation: an input cube and the metadata row it observes."""

    cube: Path
    row_idx: int
    member_ous_uid: str


def uids_in(path: str | Path) -> list[str]:
    """The ALMA uids a path names, as ``uid://A002/X113d9b7/Xb670`` (lower case)."""
    return [f"uid://{a}/{b}/{c}".lower() for a, b, c in _PATH_UID.findall(str(path))]


def find_input_cubes(path: Path) -> list[Path]:
    """The cubes an input path holds: itself, a folder's model cubes, or its FITS files."""
    path = path.expanduser().resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is neither a file nor a folder")
    models = sorted(path.rglob(MODEL_CUBE))
    if models:
        return models
    return sorted(
        p for p in path.rglob("*") if p.is_file() and p.name.lower().endswith(FITS_SUFFIXES)
    )


def products_csvs_near(metadata_path: Path) -> list[Path]:
    """The resolved-products CSVs beside a metadata CSV (``*products*.csv``)."""
    return sorted(metadata_path.expanduser().resolve().parent.glob("*products*.csv"))


def observations_by_execution_block(products: Iterable[Path]) -> dict[str, str]:
    """Execution block uid -> member OUS uid, from resolved-products CSVs.

    Their ``filename`` names each product's execution block
    (``2023.1.00494.S_uid___A002_X11a1c8d_X12c8e.asdm.sdm.tar``), their
    ``uid`` its observation.
    """
    observations: dict[str, str] = {}
    for csv in products:
        table = pd.read_csv(csv, usecols=lambda c: c in {"uid", "filename"})
        if not {"uid", "filename"} <= set(table.columns):
            continue
        for ous, filename in zip(table["uid"], table["filename"], strict=True):
            for uid in uids_in(filename):
                observations.setdefault(uid, str(ous).lower())
    return observations


def match_input_cubes(
    cubes: Iterable[Path],
    metadata: pd.DataFrame,
    observations: dict[str, str],
) -> tuple[list[CubeMatch], list[Path]]:
    """Each cube with the metadata rows of the observation its path names;
    and the cubes no row matched."""
    rows_by_ous: dict[str, list[int]] = {}
    for i, ous in enumerate(metadata.get("member_ous_uid", pd.Series(dtype=str))):
        rows_by_ous.setdefault(str(ous).lower(), []).append(i)
    matches: list[CubeMatch] = []
    unmatched: list[Path] = []
    for cube in cubes:
        named = uids_in(cube)
        found = {uid for uid in named if uid in rows_by_ous}
        found |= {observations[uid] for uid in named if observations.get(uid) in rows_by_ous}
        if not found:
            unmatched.append(cube)
            continue
        for ous in sorted(found):
            matches.extend(CubeMatch(cube, row, ous) for row in rows_by_ous[ous])
    return matches, unmatched


def cube_label(cube: Path) -> str:
    """A short name for a cube: its MeasurementSet and spectral window when an
    imaging run made it (``uid___A002_X113d9b7_Xb670_SPW-31_FIELD-2``), else its stem."""
    if cube.name == MODEL_CUBE and cube.parent.parent.name.endswith(".ms.split"):
        return f"{cube.parent.parent.name.removesuffix('.ms.split')}_{cube.parent.name}"
    return cube.stem
