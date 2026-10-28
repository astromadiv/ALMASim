"""A MOUS observed in execution blocks: the planner's runs, the joint visibility
table and the MS's OBSERVATION_ID (MADIV-28)."""

from __future__ import annotations

from types import SimpleNamespace

import astropy.units as U
import numpy as np
import pytest

from almasim.services.interferometry.observing import hour_angle_samples
from almasim.services.interferometry.visibility import concatenate_visibility_tables
from almasim.services.observation_plan import ObservationConfig
from almasim.services.simulation.planning import (
    EXECUTION_ANTENNA_SHARE,
    MAX_EXECUTION_TIME_S,
    execution_antennas,
    execution_count,
    execution_date,
    plan_observation_runs,
)


@pytest.mark.unit
def test_the_archive_count_of_executions_wins():
    assert execution_count(36469.44, 13) == 13
    assert execution_count(151.2, 1) == 1


@pytest.mark.unit
def test_without_a_count_the_time_is_split_into_blocks_of_at_most_the_longest():
    assert execution_count(MAX_EXECUTION_TIME_S) == 1
    assert execution_count(MAX_EXECUTION_TIME_S + 1) == 2
    assert execution_count(36469.44) == 13
    assert execution_count(0.0) == 1
    assert execution_count(5000.0, 0) == 2


@pytest.mark.unit
def test_each_execution_observes_on_its_own_day():
    assert execution_date("2026-04-21", 0) == "2026-04-21"
    assert execution_date("2026-04-30", 2) == "2026-05-02"
    assert execution_date("not a date", 3) == "not a date"


def _plan(total_time_s: float, *, n_executions=None, executions=None, configs=None):
    params = SimpleNamespace(
        pwv=1.0,
        ground_temperature_k=270.0,
        seed=1,
        corruptions=None,
        n_executions=n_executions,
        executions=executions,
        idx=0,
        main_dir="/tmp",
        ra=10.0,
        dec=-30.0,
        obs_date="2026-04-21",
        save_mode="memory",
        imaging_algorithm="legacy",
    )
    configs = configs or [
        ObservationConfig("config_0_12m", "12m", "A001:DA41 A002:DA42", total_time_s)
    ]
    geometry = SimpleNamespace(
        observation_plan=SimpleNamespace(configs=configs, elevation_deg=60.0),
        channel_width_hz=1e6,
        noise_bandwidth_hz=1e6,
        central_freq=100 * U.GHz,
        band_range=8 * U.MHz,
        fov=SimpleNamespace(value=20.0),
        spectral_window=lambda: None,
    )
    return plan_observation_runs(
        params,
        geometry,
        model_cube=np.zeros((8, 4, 4)),
        header=None,
        channel_frequencies_hz=np.linspace(99.996e9, 100.004e9, 8),
        effective_snr=10.0,
        base_noise_reference=1e-3,
        output_dir_abs="/tmp",
    )


@pytest.mark.unit
def test_a_mous_is_one_run_per_execution_of_its_share_of_the_time():
    runs, total_power = _plan(36469.44, n_executions=13)
    assert total_power == []
    assert len(runs) == 13
    assert [run.execution for run in runs] == list(range(13))
    assert all(run.integration_time * 3600.0 == pytest.approx(36469.44 / 13) for run in runs)
    assert runs[0].observation_date == "2026-04-21"
    assert runs[12].observation_date == "2026-05-03"
    # Sampled alike, within one track's budget of hour-angle samples.
    assert {run.sample_time_s for run in runs} == {hour_angle_samples(36469.44)[1]}


@pytest.mark.unit
def test_executions_simulates_only_that_many_blocks():
    runs, _ = _plan(36469.44, n_executions=13, executions=1)
    assert len(runs) == 1
    assert runs[0].integration_time * 3600.0 == pytest.approx(36469.44 / 13)
    assert runs[0].sample_time_s == hour_angle_samples(36469.44 / 13)[1]
    assert len(_plan(1000.0, executions=4)[0]) == 1


@pytest.mark.unit
def test_a_short_mous_is_one_track_as_before():
    runs, _ = _plan(1800.0)
    assert len(runs) == 1
    assert runs[0].integration_time == pytest.approx(0.5)
    assert runs[0].sample_time_s == hour_angle_samples(1800.0)[1]


@pytest.mark.unit
def test_each_configuration_is_split_into_its_executions():
    configs = [
        ObservationConfig("config_0_12m", "12m", "A001:DA41 A002:DA42", 4000.0),
        ObservationConfig(
            "config_1_7m", "7m", "J501:CM01 J502:CM02", 4000.0, antenna_diameter_m=7.0
        ),
    ]
    runs, _ = _plan(4000.0, configs=configs)
    assert [(run.array_type, run.execution) for run in runs] == [
        ("12m", 0),
        ("12m", 1),
        ("7m", 0),
        ("7m", 1),
    ]


def _table(names, *, execution=None, scans=(1, 1, 2, 2), array_type="12m"):
    n = len(scans)
    return {
        "uvw_m": np.zeros((n, 3)),
        "antenna1": np.zeros(n, np.int32),
        "antenna2": np.ones(n, np.int32),
        "time_mjd_s": np.arange(n, dtype=float),
        "interval_s": np.ones(n),
        "exposure_s": np.ones(n),
        "data": np.zeros((n, 1, 2), np.complex64),
        "model_data": np.zeros((n, 1, 2), np.complex64),
        "flag": np.zeros((n, 1, 2), bool),
        "weight": np.ones((n, 1), np.float32),
        "sigma": np.ones((n, 1), np.float32),
        "channel_freq_hz": np.array([1e11, 1.001e11]),
        "antenna_names": list(names),
        "antenna_positions_m": np.zeros((len(names), 3)),
        "source_name": "s",
        "field_ra_rad": 0.0,
        "field_dec_rad": 0.0,
        "observation_date": "2026-04-21",
        "array_type": array_type,
        "scan_number": np.asarray(scans, np.int32),
        **({} if execution is None else {"observation_id": np.full(n, execution, np.int32)}),
    }


@pytest.mark.unit
def test_the_executions_of_one_array_share_its_antennas_not_their_scans():
    joint = concatenate_visibility_tables(
        [
            _table(["A1", "A2"], execution=0),
            _table(["A1", "A2"], execution=1),
            _table(["C1", "C2"], execution=0, array_type="7m"),
        ]
    )
    assert joint["antenna_names"] == ["A1", "A2", "C1", "C2"]
    np.testing.assert_array_equal(joint["antenna2"], [1] * 8 + [3] * 4)
    np.testing.assert_array_equal(joint["scan_number"], [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    np.testing.assert_array_equal(joint["observation_id"], [0] * 4 + [1] * 4 + [0] * 4)


@pytest.mark.unit
def test_tables_without_executions_have_no_observation_ids():
    joint = concatenate_visibility_tables([_table(["A1", "A2"]), _table(["C1", "C2"])])
    assert "observation_id" not in joint


@pytest.mark.unit
def test_an_execution_is_sampled_at_its_given_sample_time():
    from astropy.io import fits

    from almasim.services.interferometry.core import Interferometer

    block = Interferometer.__new__(Interferometer)
    block.integration_time, block.header = 2805.0 / 3600.0, fits.Header()
    block.sample_time_s = 30.24
    block._get_nH()
    assert (block.nH, block.scan_time) == (92, 30.24)
    assert block.header["EPOCH"] == 92
    block.sample_time_s, block.header = None, fits.Header()
    block._get_nH()
    assert (block.nH, block.scan_time) == hour_angle_samples(2805.0)


@pytest.mark.unit
def test_each_execution_has_its_own_time_range_in_the_ms():
    from almasim.services.ms.io import _execution_ranges

    times = np.array([1.0, 2.0, 10.0, 11.0])
    np.testing.assert_array_equal(_execution_ranges({}, times), [[1.0, 11.0]])
    ids = np.array([0, 0, 2, 2])
    np.testing.assert_array_equal(
        _execution_ranges({"observation_id": ids}, times), [[1.0, 2.0], [1.0, 11.0], [10.0, 11.0]]
    )


@pytest.mark.unit
def test_a_single_block_uses_all_the_mous_antennas():
    array = "A001:DA41 A002:DA42 A002:DA43"
    assert execution_antennas(array, 1, np.random.default_rng(0)) == array


@pytest.mark.unit
def test_a_block_of_many_uses_its_share_of_the_mous_antennas_one_per_pad():
    pads = [f"A{i:03d}:DA{i:02d}" for i in range(47)] + ["A000:DV99"]
    array = " ".join(pads)
    p = EXECUTION_ANTENNA_SHARE
    two = execution_antennas(array, 2, np.random.default_rng(0)).split()
    assert len(two) == round(47 * p / (1 - (1 - p) ** 2)) == 44
    assert len({token.split(":")[0] for token in two}) == len(two)
    assert "A000:DV99" not in two
    many = execution_antennas(array, 14, np.random.default_rng(1)).split()
    assert len(many) == round(47 * p)
    assert execution_antennas(array, 14, np.random.default_rng(1)).split() == many
    assert len(execution_antennas("A001:DA41 A002:DA42", 5, np.random.default_rng(0)).split()) == 2


@pytest.mark.unit
def test_each_block_of_a_seeded_run_draws_its_own_antennas():
    array = " ".join(f"A{i:03d}:DA{i:02d}" for i in range(47))
    runs, _ = _plan(
        6000.0,
        configs=[ObservationConfig("config_0_12m", "12m", array, 6000.0)],
    )
    assert len(runs) == 2
    first, second = (set(run.antenna_array.split()) for run in runs)
    assert len(first) == len(second) == 44 and first != second
    assert (
        runs[0].antenna_array
        == _plan(6000.0, configs=[ObservationConfig("config_0_12m", "12m", array, 6000.0)])[0][
            0
        ].antenna_array
    )


@pytest.mark.unit
def test_blocks_with_different_antennas_share_one_antenna_list():
    joint = concatenate_visibility_tables(
        [_table(["A1", "A2"], execution=0), _table(["A2", "A3"], execution=1)]
    )
    assert joint["antenna_names"] == ["A1", "A2", "A3"]
    np.testing.assert_array_equal(joint["antenna1"], [0] * 4 + [1] * 4)
    np.testing.assert_array_equal(joint["antenna2"], [1] * 4 + [2] * 4)


@pytest.mark.unit
def test_the_runs_leave_their_cubes_to_the_export():
    """Every run once wrote its cubes under the same name: the last block's were kept."""
    runs, _ = _plan(36469.44, n_executions=13)
    assert len(runs) == 13 and not any(run.persist for run in runs)


@pytest.mark.unit
def test_the_export_writes_the_combined_observation(tmp_path):
    from astropy.io import fits

    from almasim.services.interferometry.multiconfig import combine_interferometric_results
    from almasim.services.simulation.export import _write_combined_observation

    def run(value: float) -> dict:
        cube = np.full((2, 8, 8), value, dtype=np.float32)
        return {
            "image_header": fits.Header([("EPOCH", int(value) + 2)]),
            "model_cube": cube,
            "dirty_cube": cube,
            "model_vis": cube.astype(np.complex64),
            "dirty_vis": cube.astype(np.complex64),
            "beam_cube": np.ones_like(cube),
            "totsampling_cube": np.ones_like(cube),
            "uv_mask_cube": np.ones_like(cube),
            "u_cube": np.zeros((2, 4)),
            "v_cube": np.zeros((2, 4)),
        }

    combined = combine_interferometric_results([run(1.0), run(3.0)], config_weights=[1.0, 1.0])
    _write_combined_observation(
        SimpleNamespace(idx=5, save_mode="fits"),
        SimpleNamespace(sim_output_dir=str(tmp_path), header=fits.Header()),
        combined,
    )
    dirty = fits.getdata(tmp_path / "dirty-cube_5.fits")
    assert dirty == pytest.approx(np.full((2, 8, 8), 2.0)), "the blocks' mean, not the last"
    assert fits.getheader(tmp_path / "dirty-cube_5.fits")["EPOCH"] == 3 + 5, "both tracks"
