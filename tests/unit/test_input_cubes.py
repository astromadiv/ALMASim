"""Input sky-model cubes and the metadata rows they observe."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from almasim.services.simulation.input_cubes import (
    cube_label,
    find_input_cubes,
    match_input_cubes,
    observations_by_execution_block,
    products_csvs_near,
    uids_in,
)

pytestmark = pytest.mark.unit

EB = "uid___A002_X11a1c8d_X12c8e"  # an execution block of observation uid://A001/X362b/Xf37
OUS = "uid://A001/X362b/Xf37"


def _cube(path, shape=(1, 4, 8, 8)):
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.zeros(shape, dtype=np.float32)).writeto(path)
    return path


@pytest.fixture
def imaging_run(tmp_path):
    """An imaging run's folder: two windows of one execution block, one of another."""
    run = tmp_path / "cubes"
    for spw in ("SPW-39_FIELD-9", "SPW-41_FIELD-9"):
        _cube(run / f"{EB}.ms.split" / spw / "wsclean-cube-model.fits")
        _cube(run / f"{EB}.ms.split" / spw / "wsclean-cube-image.fits")
    _cube(run / "uid___A002_Xaaaa_Xbbbb.ms.split" / "SPW-1_FIELD-1" / "wsclean-cube-model.fits")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    pd.DataFrame({"uid": [OUS], "filename": [f"2023.1.00494.S_{EB}.asdm.sdm.tar"]}).to_csv(
        metadata / "resolved_cycle11_products.csv", index=False
    )
    rows = pd.DataFrame({"member_ous_uid": ["uid://A001/X1/X2", OUS], "Freq": [100.0, 241.5]})
    rows.to_csv(metadata / "cycle11.csv", index=False)
    return run, metadata


def test_paths_name_their_uids():
    assert uids_in(f"/x/{EB}.ms.split/SPW-39_FIELD-9/m.fits") == ["uid://a002/x11a1c8d/x12c8e"]
    assert uids_in("model_uid___A001_X362b_Xf37.fits") == ["uid://a001/x362b/xf37"]
    assert uids_in("/data/cube.fits") == []


def test_a_folder_gives_each_window_s_model_cube(imaging_run):
    run, _ = imaging_run
    cubes = find_input_cubes(run)
    assert [c.name for c in cubes] == ["wsclean-cube-model.fits"] * 3
    assert cube_label(cubes[0]) == f"{EB}_SPW-39_FIELD-9"


def test_each_window_matches_the_row_of_its_observation(imaging_run):
    run, metadata = imaging_run
    products = products_csvs_near(metadata / "cycle11.csv")
    assert [p.name for p in products] == ["resolved_cycle11_products.csv"]
    matches, unmatched = match_input_cubes(
        find_input_cubes(run),
        pd.read_csv(metadata / "cycle11.csv"),
        observations_by_execution_block(products),
    )
    assert [(cube_label(m.cube), m.row_idx) for m in matches] == [
        (f"{EB}_SPW-39_FIELD-9", 1),
        (f"{EB}_SPW-41_FIELD-9", 1),
    ]
    assert {m.member_ous_uid for m in matches} == {OUS.lower()}
    assert [u.parent.name for u in unmatched] == ["SPW-1_FIELD-1"]


def test_a_member_ous_in_the_name_matches_without_products(tmp_path):
    cube = _cube(tmp_path / "model_uid___A001_X362b_Xf37.fits")
    metadata = pd.DataFrame({"member_ous_uid": [OUS]})
    matches, unmatched = match_input_cubes([cube], metadata, {})
    assert [(m.cube, m.row_idx) for m in matches] == [(cube, 0)] and not unmatched


def test_a_file_is_its_own_cube(tmp_path):
    cube = _cube(tmp_path / "anything.fits")
    assert find_input_cubes(cube) == [cube.resolve()]
    with pytest.raises(FileNotFoundError):
        find_input_cubes(tmp_path / "missing")
