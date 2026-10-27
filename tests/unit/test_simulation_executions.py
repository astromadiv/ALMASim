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
    MAX_EXECUTION_TIME_S,
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
        persist_outputs=False,
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
