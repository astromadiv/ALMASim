"""Unit tests for almasim.services.interferometry.visibility."""

from __future__ import annotations

import numpy as np
import pytest

from almasim.services.interferometry.visibility import (
    VisibilityTable,
    assemble_visibility_table,
    build_channel_visibility_rows,
    concatenate_visibility_tables,
)

# ===========================================================================
# helpers
# ===========================================================================


def _make_channel_rows(nrows: int = 6) -> dict:
    """Build minimal channel-row dict."""
    return {
        "uvw_m": np.zeros((nrows, 3), dtype=np.float64),
        "antenna1": np.zeros(nrows, dtype=np.int32),
        "antenna2": np.ones(nrows, dtype=np.int32),
        "time_index": np.arange(nrows, dtype=np.int32),
        "valid": np.ones(nrows, dtype=np.bool_),
        "model_data": np.ones(nrows, dtype=np.complex64),
        "data": np.ones(nrows, dtype=np.complex64),
        "weight": np.ones(nrows, dtype=np.float32),
        "sigma": np.ones(nrows, dtype=np.float32) * 0.1,
    }


def _make_visibility_table(nrows: int = 6, nchan: int = 4) -> VisibilityTable:
    """Build a VisibilityTable dataclass."""
    return VisibilityTable(
        uvw_m=np.zeros((nrows, 3)),
        antenna1=np.zeros(nrows, dtype=np.int32),
        antenna2=np.ones(nrows, dtype=np.int32),
        time_mjd_s=np.arange(nrows, dtype=float),
        interval_s=np.ones(nrows),
        exposure_s=np.ones(nrows),
        data=np.zeros((nrows, 1, nchan), dtype=np.complex64),
        model_data=np.zeros((nrows, 1, nchan), dtype=np.complex64),
        flag=np.ones((nrows, 1, nchan), dtype=bool),
        weight=np.ones((nrows, 1), dtype=np.float32),
        sigma=np.ones((nrows, 1), dtype=np.float32),
        channel_freq_hz=np.linspace(100e9, 104e9, nchan),
        antenna_names=["A1", "A2"],
        antenna_positions_m=np.zeros((2, 3)),
        source_name="TEST",
        field_ra_rad=0.0,
        field_dec_rad=0.0,
        observation_date="2024-01-01",
    )


# ===========================================================================
# VisibilityTable.as_dict
# ===========================================================================


@pytest.mark.unit
def test_visibility_table_as_dict_keys():
    """as_dict() returns all expected keys."""
    vt = _make_visibility_table()
    d = vt.as_dict()
    for key in (
        "uvw_m",
        "antenna1",
        "antenna2",
        "time_mjd_s",
        "data",
        "model_data",
        "flag",
        "weight",
        "sigma",
        "channel_freq_hz",
        "antenna_names",
        "source_name",
        "field_ra_rad",
        "field_dec_rad",
        "observation_date",
    ):
        assert key in d


@pytest.mark.unit
def test_the_noise_the_ms_writer_needs_reaches_it():
    """Each visibility's noise and the channels' kernel go into the dict the MS
    writer gets, and through the joining of execution blocks."""
    from dataclasses import replace

    kernel = np.array([0.25, 0.866, 0.25])
    vt = replace(
        _make_visibility_table(),
        sigma_spectrum=np.full((6, 1, 4), 2.0, np.float32),
        channel_kernel=kernel,
    )
    d = vt.as_dict()
    np.testing.assert_allclose(d["sigma_spectrum"], 2.0)
    np.testing.assert_allclose(d["channel_kernel"], kernel)
    assert "channel_kernel" not in _make_visibility_table().as_dict()
    joined = concatenate_visibility_tables([d, vt.as_dict()])
    np.testing.assert_allclose(joined["channel_kernel"], kernel)
    assert joined["sigma_spectrum"].shape == (12, 1, 4)
    other = replace(vt, channel_kernel=np.array([0.0, 1.0, 0.0])).as_dict()
    assert "channel_kernel" not in concatenate_visibility_tables([d, other])


@pytest.mark.unit
def test_visibility_table_as_dict_optional_fields():
    """as_dict() includes config_name and array_type (may be None)."""
    vt = _make_visibility_table()
    vt.config_name = "C43"
    vt.array_type = "12m"
    d = vt.as_dict()
    assert d["config_name"] == "C43"
    assert d["array_type"] == "12m"


# ===========================================================================
# assemble_visibility_table
# ===========================================================================


@pytest.mark.unit
def test_assemble_visibility_table_basic():
    """assemble_visibility_table returns a VisibilityTable."""
    rows = [_make_channel_rows(6) for _ in range(4)]
    freqs = np.linspace(100e9, 104e9, 4)
    vt = assemble_visibility_table(
        channel_rows=rows,
        channel_freq_hz=freqs,
        scan_time_s=10.0,
        observation_date="2024-01-01",
        antenna_names=["A1", "A2"],
        antenna_positions_m=np.zeros((2, 3)),
        source_name="TEST",
        field_ra_rad=0.0,
        field_dec_rad=0.0,
    )
    assert isinstance(vt, VisibilityTable)


@pytest.mark.unit
def test_assemble_visibility_table_data_shape():
    """Assembled data has shape (nrows, 1, nchan)."""
    nrows, nchan = 6, 4
    rows = [_make_channel_rows(nrows) for _ in range(nchan)]
    freqs = np.linspace(100e9, 104e9, nchan)
    vt = assemble_visibility_table(
        channel_rows=rows,
        channel_freq_hz=freqs,
        scan_time_s=10.0,
        observation_date="2024-01-01",
        antenna_names=["A1", "A2"],
        antenna_positions_m=np.zeros((2, 3)),
        source_name="TEST",
        field_ra_rad=0.0,
        field_dec_rad=0.0,
    )
    assert vt.data.shape == (nrows, 1, nchan)
    assert vt.model_data.shape == (nrows, 1, nchan)
    assert vt.flag.shape == (nrows, 1, nchan)


@pytest.mark.unit
def test_assemble_visibility_table_empty_rows_raises():
    """Empty channel_rows raises ValueError."""
    with pytest.raises(ValueError, match="channel_rows must not be empty"):
        assemble_visibility_table(
            channel_rows=[],
            channel_freq_hz=np.array([]),
            scan_time_s=10.0,
            observation_date="2024-01-01",
            antenna_names=["A1"],
            antenna_positions_m=np.zeros((1, 3)),
            source_name="TEST",
            field_ra_rad=0.0,
            field_dec_rad=0.0,
        )


@pytest.mark.unit
def test_assemble_visibility_table_config_name():
    """config_name and array_type propagate to the output VisibilityTable."""
    rows = [_make_channel_rows(4)]
    vt = assemble_visibility_table(
        channel_rows=rows,
        channel_freq_hz=np.array([100e9]),
        scan_time_s=10.0,
        observation_date="2024-01-01",
        antenna_names=["A1"],
        antenna_positions_m=np.zeros((1, 3)),
        source_name="FIELD",
        field_ra_rad=1.0,
        field_dec_rad=-0.5,
        config_name="C43",
        array_type="12m",
    )
    assert vt.config_name == "C43"
    assert vt.array_type == "12m"
    assert vt.source_name == "FIELD"


@pytest.mark.unit
def test_assemble_visibility_table_times():
    """time_mjd_s is computed from observation_date + time_index * scan_time_s."""
    nrows = 4
    rows = [_make_channel_rows(nrows)]
    vt = assemble_visibility_table(
        channel_rows=rows,
        channel_freq_hz=np.array([100e9]),
        scan_time_s=60.0,
        observation_date="2024-01-01",
        antenna_names=["A1"],
        antenna_positions_m=np.zeros((1, 3)),
        source_name="S",
        field_ra_rad=0.0,
        field_dec_rad=0.0,
    )
    # time_mjd_s should have nrows entries and be monotonically increasing
    assert vt.time_mjd_s.shape == (nrows,)
    assert np.all(np.diff(vt.time_mjd_s) >= 0)


# ===========================================================================
# concatenate_visibility_tables
# ===========================================================================


@pytest.mark.unit
def test_concatenate_empty_raises():
    """Empty list raises ValueError."""
    with pytest.raises(ValueError, match="tables must not be empty"):
        concatenate_visibility_tables([])


@pytest.mark.unit
def test_concatenate_single_returns_dict():
    """Single table (dict) is returned as dict."""
    vt = _make_visibility_table().as_dict()
    result = concatenate_visibility_tables([vt])
    assert isinstance(result, dict)
    assert "uvw_m" in result


@pytest.mark.unit
def test_concatenate_two_tables_row_count():
    """Two tables concatenated have combined row count."""
    vt1 = _make_visibility_table(nrows=4, nchan=3)
    vt2 = _make_visibility_table(nrows=6, nchan=3)
    result = concatenate_visibility_tables([vt1.as_dict(), vt2.as_dict()])
    assert result["uvw_m"].shape[0] == 10


@pytest.mark.unit
def test_concatenate_two_tables_channel_from_first():
    """channel_freq_hz comes from the first table."""
    vt1 = _make_visibility_table(nrows=4, nchan=3)
    vt2 = _make_visibility_table(nrows=4, nchan=3)
    vt1_dict = vt1.as_dict()
    vt2_dict = vt2.as_dict()
    vt1_dict["channel_freq_hz"] = np.array([100e9, 200e9, 300e9])
    vt2_dict["channel_freq_hz"] = np.array([999e9, 888e9, 777e9])
    result = concatenate_visibility_tables([vt1_dict, vt2_dict])
    np.testing.assert_array_equal(result["channel_freq_hz"], vt1_dict["channel_freq_hz"])


@pytest.mark.unit
def test_concatenate_keeps_each_configurations_antennas():
    """A second configuration's rows point at its own antennas, with their dish sizes."""
    vt12 = _make_visibility_table(nrows=4, nchan=3).as_dict()
    vt7 = _make_visibility_table(nrows=4, nchan=3).as_dict()
    vt12["antenna_diameters_m"] = np.array([12.0, 12.0])
    vt7["antenna_names"] = ["C1", "C2"]
    vt7["antenna_diameters_m"] = np.array([7.0, 7.0])
    result = concatenate_visibility_tables([vt12, vt7])
    assert result["antenna_names"] == ["A1", "A2", "C1", "C2"]
    assert result["antenna_positions_m"].shape == (4, 3)
    np.testing.assert_array_equal(result["antenna_diameters_m"], [12.0, 12.0, 7.0, 7.0])
    np.testing.assert_array_equal(result["antenna1"][4:], np.asarray(vt7["antenna1"]) + 2)
    np.testing.assert_array_equal(result["antenna2"][:4], vt12["antenna2"])


@pytest.mark.unit
def test_concatenate_assumes_12_m_dishes_without_diameters():
    vt1 = _make_visibility_table(nrows=2, nchan=1).as_dict()
    vt2 = _make_visibility_table(nrows=2, nchan=1).as_dict()
    vt2["antenna_names"] = ["C1", "C2"]
    result = concatenate_visibility_tables([vt1, vt2])
    np.testing.assert_array_equal(result["antenna_diameters_m"], np.full(4, 12.0))


@pytest.mark.unit
def test_row_weights_carry_the_gain_amplitude():
    """Calibrated data were scaled by their gains, and so was their noise: a row's
    sigma is the channel's noise times |g|, its weight 1 / sigma^2."""
    nbas, nh = 3, 2
    gains = np.full((nbas, nh), 2.0 * np.exp(0.3j), dtype=np.complex64)
    gains[0] = 1.0
    rows = build_channel_visibility_rows(
        modelfft=np.zeros((16, 16), dtype=np.complex64),
        gains=gains,
        noise_samples=np.zeros((nbas, nh), dtype=np.complex64),
        u_waves=np.zeros((nbas, nh)),
        v_waves=np.zeros((nbas, nh)),
        w_waves=np.zeros((nbas, nh)),
        antnum=np.array([[0, 1], [0, 2], [1, 2]]),
        nH=nh,
        Nphf=8,
        uv_pixsize=1.0,
        noise_std=0.5,
        mean_wavelength_m=1e-3,
    )
    np.testing.assert_allclose(rows["sigma"], np.repeat([0.5, 1.0, 1.0], nh), rtol=1e-6)
    np.testing.assert_allclose(rows["weight"], 1.0 / rows["sigma"] ** 2, rtol=1e-6)


@pytest.mark.unit
def test_row_weight_is_the_inverse_of_the_mean_variance():
    """Over a row's channels, WEIGHT is 1 / mean(sigma^2), as statwt sets it, and
    SIGMA its square root: channels with sigma 1 and 3 give 1/5, not (1 + 1/9) / 2."""
    rows = []
    for sigma in (1.0, 3.0):
        r = _make_channel_rows(4)
        r["sigma"] = np.full(4, sigma, dtype=np.float32)
        r["weight"] = np.full(4, 1 / sigma**2, dtype=np.float32)
        rows.append(r)
    vt = assemble_visibility_table(
        channel_rows=rows,
        channel_freq_hz=np.array([100e9, 101e9]),
        scan_time_s=10.0,
        observation_date="2024-01-01",
        antenna_names=["A1", "A2"],
        antenna_positions_m=np.zeros((2, 3)),
        source_name="TEST",
        field_ra_rad=0.0,
        field_dec_rad=0.0,
    )
    np.testing.assert_allclose(vt.weight, 0.2, rtol=1e-6)
    np.testing.assert_allclose(vt.sigma, np.sqrt(5.0), rtol=1e-6)
