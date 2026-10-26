"""One cube-geometry resolver for the estimate and the run (MADIV-28 T5.3)."""

from __future__ import annotations

from pathlib import Path

import astropy.units as U
import pandas as pd
import pytest

from almasim.services.simulation import (
    SimulationParams,
    estimate_simulation_footprint,
    resolve_cube_geometry,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = REPO_ROOT / "src" / "almasim"


def _params(tmp_path, **values):
    row = pd.read_csv(REPO_ROOT / "data" / "qso_metadata.csv").iloc[0]
    dirs = {name: tmp_path for name in ("output_dir", "tng_dir", "galaxy_zoo_dir", "hubble_dir")}
    return SimulationParams.from_metadata_row(
        row, idx=0, project_name="t", main_dir=MAIN_DIR, **dirs, **values
    )


def test_default_cell_is_a_fifth_of_the_beam(tmp_path):
    geometry = resolve_cube_geometry(_params(tmp_path))
    assert geometry.cell_size.to_value(U.arcsec) == pytest.approx(
        geometry.beam_size.to_value(U.arcsec) / 5
    )
    assert geometry.n_pix & (geometry.n_pix - 1) == 0, "a power of two"


def test_a_given_size_sets_the_cell_from_the_field_and_the_estimate_agrees(tmp_path):
    params = _params(tmp_path, n_pix=100, n_channels=8)
    geometry = resolve_cube_geometry(params)
    assert geometry.n_pix == 128 and geometry.n_channels == 8
    assert geometry.cell_size.to_value(U.arcsec) == pytest.approx(
        geometry.fov.to_value(U.arcsec) / 128
    )
    assert geometry.band_range.to_value(U.GHz) == pytest.approx(
        8 * geometry.delta_freq.to_value(U.GHz)
    )
    estimate = estimate_simulation_footprint(params)
    assert estimate["cube_shape"] == [8, 128, 128]
    assert estimate["cell_size_arcsec"] == pytest.approx(geometry.cell_size.to_value(U.arcsec))


def test_planner_geometry_uses_the_primary_beam_and_eight_cells_per_beam(tmp_path):
    """instrument_geometry="planner" shares the imaging planner's model (MADIV-28 T2.11)."""
    from almasim.services.instrument.geometry import (
        primary_beam_fov_arcsec,
        synthesized_beam_arcsec,
    )

    legacy = resolve_cube_geometry(_params(tmp_path))
    planner = resolve_cube_geometry(_params(tmp_path, instrument_geometry="planner"))
    frequency_hz = planner.central_freq.to_value(U.Hz)
    dish = min(c.antenna_diameter_m for c in planner.observation_plan.configs)
    assert planner.fov.to_value(U.arcsec) == pytest.approx(
        primary_beam_fov_arcsec(dish, frequency_hz)
    )
    assert planner.beam_size.to_value(U.arcsec) == pytest.approx(
        synthesized_beam_arcsec(frequency_hz, planner.max_baseline.to_value(U.m))
    )
    assert planner.beam_size.to_value(U.arcsec) == pytest.approx(
        legacy.beam_size.to_value(U.arcsec), rel=1e-9
    ), "the same lambda / B_max as the legacy estimate"
    assert planner.cell_size.to_value(U.arcsec) == pytest.approx(
        planner.beam_size.to_value(U.arcsec) / 8
    )
    assert planner.n_pix & (planner.n_pix - 1) == 0


def test_channels_are_the_windows_not_its_resolution(tmp_path):
    """The archive's 1938.48 kHz is the resolution of four averaged 488 kHz channels."""
    geometry = resolve_cube_geometry(_params(tmp_path))
    assert geometry.delta_freq.to_value(U.kHz) == pytest.approx(1953.125)
    assert geometry.channel_averaging == 4
    assert geometry.noise_bandwidth_hz == pytest.approx(1953.125e3 * 1.231, rel=1e-3)
    assert geometry.n_channels == int(geometry.band_range / geometry.delta_freq)
