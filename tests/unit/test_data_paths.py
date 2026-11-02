"""The bundled data layout and its cached readers (MADIV-28 T2.2)."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from almasim.services import data_paths
from almasim.services.data_paths import (
    DataLayout,
    antenna_names,
    antenna_positions,
    calibrated_lines,
    max_baseline_km,
    read_table,
)

MAIN_DIR = Path(data_paths.__file__).resolve().parents[1]
REPO_ROOT = MAIN_DIR.parents[1]


@pytest.mark.unit
def test_layout_names_the_bundled_files():
    layout = DataLayout(str(MAIN_DIR))
    assert layout.main_dir == MAIN_DIR
    for path in (layout.antenna_coordinates, layout.calibrated_lines):
        assert path.is_file()
    assert layout.sed_template("SED_low_z_type2_AGN.dat").is_file()


@pytest.mark.unit
def test_read_table_parses_once_and_hands_out_copies(tmp_path, monkeypatch):
    csv = tmp_path / "t.csv"
    csv.write_text("a,b\n1,2\n")
    calls = []
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: calls.append(a) or real(*a, **k))
    first = read_table(csv)
    first["a"] = 99
    second = read_table(csv)
    assert second["a"].tolist() == [1], "a caller's change does not leak into the cache"
    assert len(calls) == 1


@pytest.mark.unit
def test_antenna_positions_keep_table_order():
    layout = DataLayout(MAIN_DIR)
    table = read_table(layout.antenna_coordinates)
    names = table["name"].tolist()[:5]
    array = " ".join(f"{name}:DA{i}" for i, name in enumerate(reversed(names)))
    assert antenna_names(array) == list(reversed(names))
    assert antenna_positions(layout, array)["name"].tolist() == names


@pytest.mark.unit
@pytest.mark.parametrize(
    "pads, km",
    [
        # Distances between the pads' positions in the MADIV-16 MSs' ANTENNA tables.
        ("A001 P401", 3.2076),
        ("A001 S301", 3.1866),
        ("A001 W210", 3.7239),
        ("P404 W207", 5.8051),
    ],
)
def test_the_outer_pads_are_where_alma_has_them(pads, km):
    array = " ".join(f"{pad}:DA41" for pad in pads.split())
    assert max_baseline_km(DataLayout(MAIN_DIR), array) == pytest.approx(km, abs=0.001)


def _pairwise_max_km(rows) -> float:
    """The loop the simulator used before T2.2."""
    best = 0.0
    for i in range(len(rows)):
        _, x1, y1, z1 = rows[i]
        for j in range(i + 1, len(rows)):
            _, x2, y2, z2 = rows[j]
            best = max(best, math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) / 1000)
    return best


@pytest.mark.unit
def test_max_baseline_matches_the_pairwise_loop_exactly():
    layout = DataLayout(MAIN_DIR)
    arrays = pd.read_csv(REPO_ROOT / "data" / "qso_metadata.csv")["antenna_arrays"].head(5)
    for array in arrays:
        rows = antenna_positions(layout, array)[["name", "x", "y", "z"]].values
        assert max_baseline_km(layout, array) == _pairwise_max_km(rows)


@pytest.mark.unit
def test_max_baseline_of_a_single_antenna_is_zero():
    layout = DataLayout(MAIN_DIR)
    name = read_table(layout.antenna_coordinates)["name"].iloc[0]
    assert max_baseline_km(layout, f"{name}:DA41") == 0.0


@pytest.mark.unit
def test_calibrated_lines_have_names_and_frequencies():
    lines = calibrated_lines(DataLayout(MAIN_DIR))
    assert {"Line", "freq(GHz)"} <= set(lines.columns)
