"""How big a simulation will be, before it runs (web console plan, U3.6).

The cube's geometry comes from one metadata row and the size options, by
the same code a simulation uses (``estimate_simulation_footprint``); it is
arithmetic on the row, quick enough for the console to answer while the
form is being filled in. Nothing is written.

Which metadata rows a folder of input cubes simulates comes from the same
matching as `almasim simulation run --input` (``input_cubes``), each cube's
grid from its header alone.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .files import _existing, resolve

router = APIRouter(prefix="/api/v2/simulation", tags=["simulation"])

# Where `almasim simulation` finds its data by default (its --main-dir).
MAIN_DIR = Path(__file__).resolve().parents[2]


class EstimateRequest(BaseModel):
    metadata_path: str
    row_idx: int = Field(0, ge=0)
    n_pix: int | None = Field(None, ge=1)
    n_channels: int | None = Field(None, ge=1)
    source_type: str = "point"
    instrument_geometry: str = "legacy"
    # An input cube: it sets the grid (size, pixel, frequency axis) itself.
    input_path: str | None = None


class Estimate(BaseModel):
    source_name: str
    band: str
    n_pix: int
    n_channels: int
    cube_shape: list[int]
    cell_size_arcsec: float
    beam_size_arcsec: float
    raw_single_cube_gb: float
    estimated_standard_output_gb: float
    note: str


def _row(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i == index:
                return row
    raise HTTPException(status_code=400, detail=f"{path.name} has no row {index}")


@router.post("/estimate", response_model=Estimate)
def estimate(body: EstimateRequest, request: Request) -> dict[str, Any]:
    """The cube a simulation of one metadata row would make, and its raw size."""
    from almasim.services.simulation.geometry import estimate_simulation_footprint
    from almasim.services.simulation.params import SimulationParams

    profile = request.app.state.profile
    row = _row(_existing(profile, body.metadata_path), body.row_idx)
    cube = _existing(profile, body.input_path) if body.input_path else None
    unused = "/nonexistent/almasim-estimate"  # dirs the estimate never touches
    try:
        params = SimulationParams.from_metadata_row(
            row,
            idx=body.row_idx,
            main_dir=str(MAIN_DIR),
            output_dir=unused,
            tng_dir=unused,
            galaxy_zoo_dir=unused,
            hubble_dir=unused,
            project_name="estimate",
            n_pix=None if cube else body.n_pix,
            n_channels=None if cube else body.n_channels,
            source_type="external-fits-cube" if cube else body.source_type,
            external_skymodel_path=str(cube) if cube else None,
            instrument_geometry=body.instrument_geometry,
            persist=False,
        )
        footprint = estimate_simulation_footprint(params)
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"cannot estimate: {exc}") from exc
    return {
        **footprint,
        "source_name": str(params.source_name),
        "band": str(row.get("Band") or row.get("band") or ""),
    }


class CubesRequest(BaseModel):
    metadata_path: str
    input_path: str  # a cube, or a folder of them
    products_csv: str | None = None  # default: the *products*.csv next to the metadata


class InputCube(BaseModel):
    path: str
    label: str
    row_idx: int
    member_ous_uid: str
    source_name: str
    n_pix_x: int
    n_pix_y: int
    n_channels: int | None
    central_freq_ghz: float | None
    channel_width_mhz: float | None
    cell_arcsec: float | None


class CubeList(BaseModel):
    cubes: list[InputCube]  # one per (cube, metadata row): one simulation each
    unmatched: list[str]  # cubes whose path names no observation in the metadata
    products: list[str]  # the resolved-products CSVs the execution blocks were looked up in


@router.post("/cubes", response_model=CubeList)
def input_cubes(body: CubesRequest, request: Request) -> dict[str, Any]:
    """The input cubes under a path, each with the metadata row of its observation."""
    import pandas as pd

    from almasim.services.external_skymodel import read_cube_wcs
    from almasim.services.simulation.input_cubes import (
        cube_label,
        find_input_cubes,
        match_input_cubes,
        observations_by_execution_block,
        products_csvs_near,
    )

    profile = request.app.state.profile
    metadata_path = _existing(profile, body.metadata_path)
    products = (
        [_existing(profile, body.products_csv)]
        if body.products_csv
        else products_csvs_near(metadata_path)
    )
    input_path = resolve(profile, body.input_path)  # a cube or a folder
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"{body.input_path} does not exist")
    metadata = pd.read_csv(metadata_path)
    try:
        cubes = find_input_cubes(input_path)
        matches, unmatched = match_input_cubes(
            cubes, metadata, observations_by_execution_block(products)
        )
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=f"cannot match the cubes: {exc}") from exc
    names = metadata.get("ALMA_source_name", pd.Series([""] * len(metadata)))
    listed = []
    for match in matches:
        try:
            wcs = read_cube_wcs(match.cube)
        except (OSError, ValueError, KeyError):
            unmatched.append(match.cube)  # unreadable: nothing to simulate
            continue
        listed.append(
            {
                "path": str(match.cube),
                "label": cube_label(match.cube),
                "row_idx": match.row_idx,
                "member_ous_uid": match.member_ous_uid,
                "source_name": str(names.iloc[match.row_idx]),
                "n_pix_x": wcs.n_pix_x,
                "n_pix_y": wcs.n_pix_y,
                "n_channels": wcs.n_channels,
                "central_freq_ghz": wcs.central_freq_hz / 1e9 if wcs.central_freq_hz else None,
                "channel_width_mhz": wcs.channel_width_hz / 1e6 if wcs.channel_width_hz else None,
                "cell_arcsec": wcs.cell_arcsec,
            }
        )
    return {
        "cubes": listed,
        "unmatched": [str(p) for p in unmatched],
        "products": [str(p) for p in products],
    }
