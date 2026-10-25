"""Unit tests for MSv2 I/O utilities."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from almasim.services.ms.backends import _CasatoolsBackend, _get_backend
from almasim.services.ms.io import export_native_ms, read_native_ms
from almasim.services.ms.schema import _arr_desc, _make_tabledesc, _sca_desc, _vtype
from almasim.services.ms.subtables import (
    antenna_table,
    data_description_table,
    empty_table,
    feed_table,
    field_table,
    history_table,
    main_table,
    observation_table,
    polarization_table,
    processor_table,
    source_table,
    spectral_window_table,
    state_table,
    write_table,
)

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vtype_bool():
    assert _vtype(True) == "boolean"
    assert _vtype(False) == "boolean"


@pytest.mark.unit
def test_vtype_int():
    assert _vtype(1) == "int"
    assert _vtype(np.int32(5)) == "int"
    assert _vtype(np.int64(5)) == "int"


@pytest.mark.unit
def test_vtype_float32():
    assert _vtype(np.float32(1.0)) == "float"


@pytest.mark.unit
def test_vtype_double():
    assert _vtype(1.0) == "double"
    assert _vtype(np.float64(1.0)) == "double"


@pytest.mark.unit
def test_vtype_complex():
    assert _vtype(1 + 2j) == "complex"
    assert _vtype(np.complex64(1 + 2j)) == "complex"


@pytest.mark.unit
def test_vtype_string():
    assert _vtype("hello") == "string"
    assert _vtype(None) == "string"


@pytest.mark.unit
def test_sca_desc_structure():
    name, desc = _sca_desc("MY_COL", 3.14)
    assert name == "MY_COL"
    assert desc["valueType"] == "double"
    assert desc["dataManagerType"] == "StandardStMan"
    assert "keywords" in desc
    assert desc["option"] == 0


@pytest.mark.unit
def test_arr_desc_no_shape():
    name, desc = _arr_desc("DATA", np.complex64(0), ndim=2)
    assert name == "DATA"
    assert desc["valueType"] == "complex"
    assert desc["ndim"] == 2
    assert "shape" not in desc
    assert desc["option"] == 0


@pytest.mark.unit
def test_arr_desc_with_shape():
    name, desc = _arr_desc("UVW", 0.0, ndim=1, shape=[3])
    assert name == "UVW"
    assert desc["option"] == 5  # FixedShape
    np.testing.assert_array_equal(desc["shape"], [3])


@pytest.mark.unit
def test_make_tabledesc_structure():
    cols = [
        _sca_desc("TIME", 0.0),
        _arr_desc("DATA", np.complex64(0), 2),
    ]
    desc = _make_tabledesc(cols)
    assert "_define_hypercolumn_" in desc
    assert "_keywords_" in desc
    assert "_private_keywords_" in desc
    assert "TIME" in desc
    assert "DATA" in desc


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_backend_casatools_preferred():
    mock_ct = MagicMock()
    with patch.dict("sys.modules", {"casatools": mock_ct}):
        backend = _get_backend()
        assert backend is not None


@pytest.mark.unit
def test_get_backend_raises_when_nothing_available():
    with (
        patch(
            "almasim.services.ms.backends._CasatoolsBackend.__init__",
            side_effect=ImportError,
        ),
        patch(
            "almasim.services.ms.backends._CasacoreBackend.__init__",
            side_effect=ImportError,
        ),
    ):
        with pytest.raises(ImportError, match="MS export requires"):
            _get_backend()


# ---------------------------------------------------------------------------
# Subtable writers (mocked backend)
# ---------------------------------------------------------------------------


def _mock_backend():
    """Build a mock backend that records calls."""
    b = MagicMock()
    tb = MagicMock()
    b.create.return_value = tb
    b.open.return_value = tb
    # Array layout conversion as casatools does it (the tables' getcol results
    # below are in casatools order).
    b.put.side_effect = _CasatoolsBackend.put
    b.put_cell.side_effect = _CasatoolsBackend.put_cell
    b.get.side_effect = _CasatoolsBackend.get
    return b, tb


@pytest.mark.unit
def test_write_antenna(tmp_path):
    b, tb = _mock_backend()
    names = ["DA41", "DA42", "DV01"]
    positions = np.zeros((3, 3))
    write_table(b, str(tmp_path / "test.ms"), antenna_table(names, positions))
    assert b.create.called
    tb.putcol.assert_any_call("NAME", names)
    tb.flush.assert_called_once()
    tb.close.assert_called_once()


@pytest.mark.unit
def test_write_spectral_window_single_channel(tmp_path):
    b, tb = _mock_backend()
    freq = np.array([230e9])
    write_table(b, str(tmp_path / "test.ms"), spectral_window_table(freq, 1))
    tb.putcell.assert_any_call("NUM_CHAN", 0, 1)
    tb.flush.assert_called_once()


@pytest.mark.unit
def test_write_spectral_window_multi_channel(tmp_path):
    b, tb = _mock_backend()
    freq = np.linspace(230e9, 232e9, 32)
    write_table(b, str(tmp_path / "test.ms"), spectral_window_table(freq, 32))
    tb.putcell.assert_any_call("NUM_CHAN", 0, 32)


@pytest.mark.unit
def test_write_polarization(tmp_path):
    b, tb = _mock_backend()
    write_table(b, str(tmp_path / "test.ms"), polarization_table())
    tb.putcell.assert_any_call("NUM_CORR", 0, 2)  # XX and YY
    tb.flush.assert_called_once()


@pytest.mark.unit
def test_write_data_description(tmp_path):
    b, tb = _mock_backend()
    write_table(b, str(tmp_path / "test.ms"), data_description_table())
    tb.putcell.assert_any_call("SPECTRAL_WINDOW_ID", 0, 0)
    tb.putcell.assert_any_call("POLARIZATION_ID", 0, 0)


@pytest.mark.unit
def test_write_field(tmp_path):
    b, tb = _mock_backend()
    time_range = np.array([5e9, 5.1e9])
    write_table(b, str(tmp_path / "test.ms"), field_table("J0001+0001", 0.1, -0.2, time_range))
    tb.putcell.assert_any_call("NAME", 0, "J0001+0001")


@pytest.mark.unit
def test_write_observation(tmp_path):
    b, tb = _mock_backend()
    time_range = np.array([5e9, 5.1e9])
    write_table(
        b, str(tmp_path / "test.ms"), observation_table("ALMA", "2021.1.00001.S", time_range)
    )
    tb.putcell.assert_any_call("TELESCOPE_NAME", 0, "ALMA")
    tb.putcell.assert_any_call("PROJECT", 0, "2021.1.00001.S")


@pytest.mark.unit
def test_write_source(tmp_path):
    b, tb = _mock_backend()
    time_range = np.array([5e9, 5.1e9])
    write_table(b, str(tmp_path / "test.ms"), source_table("MySource", 1.0, -0.5, time_range))
    tb.putcell.assert_any_call("NAME", 0, "MySource")


@pytest.mark.unit
def test_write_state(tmp_path):
    b, tb = _mock_backend()
    write_table(b, str(tmp_path / "test.ms"), state_table())
    tb.putcell.assert_any_call("OBS_MODE", 0, "OBSERVE_TARGET#ON_SOURCE")


@pytest.mark.unit
def test_write_history(tmp_path):
    b, tb = _mock_backend()
    time_range = np.array([5e9, 5.1e9])
    write_table(b, str(tmp_path / "test.ms"), history_table("my_project", time_range))
    tb.putcell.assert_any_call("MESSAGE", 0, "ALMASim MSv2 export")


@pytest.mark.unit
def test_write_main(tmp_path):
    nrows, ncorr, nchan = 10, 1, 8
    vt = {
        "data": np.ones((nrows, ncorr, nchan), dtype=np.complex64),
        "model_data": np.zeros((nrows, ncorr, nchan), dtype=np.complex64),
        "flag": np.zeros((nrows, ncorr, nchan), dtype=bool),
        "uvw_m": np.zeros((nrows, 3)),
        "antenna1": np.zeros(nrows, dtype=np.int32),
        "antenna2": np.ones(nrows, dtype=np.int32),
        "time_mjd_s": np.linspace(5e9, 5.1e9, nrows),
        "interval_s": np.full(nrows, 10.0),
        "exposure_s": np.full(nrows, 10.0),
        "weight": np.ones((nrows, ncorr), dtype=np.float32),
        "sigma": np.ones((nrows, ncorr), dtype=np.float32),
    }
    b, tb = _mock_backend()
    write_table(b, str(tmp_path / "test.ms"), main_table(str(tmp_path / "test.ms"), vt))
    tb.putcol.assert_any_call("ANTENNA1", vt["antenna1"])
    tb.putcol.assert_any_call("ANTENNA2", vt["antenna2"])
    tb.putkeyword.assert_any_call("MS_VERSION", 2.0)


# ---------------------------------------------------------------------------
# export_native_ms end-to-end (fully mocked backend)
# ---------------------------------------------------------------------------


def _sample_vt(nrows=5, ncorr=1, nchan=4):
    return {
        "data": np.ones((nrows, ncorr, nchan), dtype=np.complex64),
        "model_data": np.zeros((nrows, ncorr, nchan), dtype=np.complex64),
        "flag": np.zeros((nrows, ncorr, nchan), dtype=bool),
        "uvw_m": np.zeros((nrows, 3)),
        "antenna1": np.zeros(nrows, dtype=np.int32),
        "antenna2": np.ones(nrows, dtype=np.int32),
        "time_mjd_s": np.linspace(5e9, 5.1e9, nrows),
        "interval_s": np.full(nrows, 10.0),
        "exposure_s": np.full(nrows, 10.0),
        "weight": np.ones((nrows, ncorr), dtype=np.float32),
        "sigma": np.ones((nrows, ncorr), dtype=np.float32),
        "channel_freq_hz": np.linspace(230e9, 231e9, nchan),
        "antenna_names": ["DA41", "DA42"],
        "antenna_positions_m": np.zeros((2, 3)),
        "field_ra_rad": 0.1,
        "field_dec_rad": -0.2,
    }


@pytest.mark.unit
def test_export_native_ms_calls_all_writers(tmp_path):
    vt = _sample_vt()
    b, tb = _mock_backend()

    with patch("almasim.services.ms.io._get_backend", return_value=b):
        result = export_native_ms(
            ms_path=tmp_path / "out.ms",
            visibility_table=vt,
            project_name="test_proj",
            source_name="J0000+0000",
        )

    assert result.endswith("out.ms")
    # create called once per subtable + main
    assert b.create.call_count >= 10


@pytest.mark.unit
def test_export_native_ms_default_telescope(tmp_path):
    vt = _sample_vt()
    b, tb = _mock_backend()

    with patch("almasim.services.ms.io._get_backend", return_value=b):
        export_native_ms(
            ms_path=tmp_path / "out.ms",
            visibility_table=vt,
            project_name="p",
            source_name="S",
        )

    # TELESCOPE_NAME cell should receive "ALMA"
    write_obs_calls = [c for c in tb.putcell.call_args_list if c[0][0] == "TELESCOPE_NAME"]
    assert len(write_obs_calls) == 1
    assert write_obs_calls[0][0][2] == "ALMA"


# ---------------------------------------------------------------------------
# read_native_ms (mocked backend)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_native_ms_returns_expected_keys(tmp_path):
    nrows, ncorr, nchan = 3, 1, 4
    main_tb = MagicMock()
    main_tb.getcol.side_effect = lambda col: {
        "DATA": np.ones((ncorr, nchan, nrows), dtype=np.complex64),
        "MODEL_DATA": np.zeros((ncorr, nchan, nrows), dtype=np.complex64),
        "FLAG": np.zeros((ncorr, nchan, nrows), dtype=bool),
        "UVW": np.zeros((3, nrows)),
        "ANTENNA1": np.zeros(nrows, dtype=np.int32),
        "ANTENNA2": np.ones(nrows, dtype=np.int32),
        "TIME": np.linspace(5e9, 5.1e9, nrows),
        "INTERVAL": np.full(nrows, 10.0),
        "EXPOSURE": np.full(nrows, 10.0),
        "WEIGHT": np.ones((ncorr, nrows), dtype=np.float32),
        "SIGMA": np.ones((ncorr, nrows), dtype=np.float32),
    }[col]

    spw_tb = MagicMock()
    spw_tb.getcell.return_value = np.linspace(230e9, 231e9, nchan)

    ant_tb = MagicMock()
    ant_tb.getcol.side_effect = lambda col: {
        "NAME": ["DA41", "DA42"],
        "POSITION": np.zeros((3, 2)),
        "DISH_DIAMETER": np.array([12.0, 7.0]),
    }[col]

    pol_tb = MagicMock()
    pol_tb.getcell.return_value = np.array([9])

    fld_tb = MagicMock()
    fld_tb.getcell.side_effect = lambda col, row: {
        "PHASE_DIR": np.array([[0.1, -0.2]]),
        "NAME": "J0000",
    }[col]

    obs_tb = MagicMock()
    obs_tb.getcell.side_effect = lambda col, row: {
        "TELESCOPE_NAME": "ALMA",
        "PROJECT": "2021.1.S",
    }[col]

    def fake_open(path, readonly=True):
        if path.endswith("SPECTRAL_WINDOW"):
            return spw_tb
        if path.endswith("ANTENNA"):
            return ant_tb
        if path.endswith("FIELD"):
            return fld_tb
        if path.endswith("OBSERVATION"):
            return obs_tb
        if path.endswith("POLARIZATION"):
            return pol_tb
        return main_tb

    b = MagicMock()
    b.open.side_effect = fake_open
    b.get.side_effect = _CasatoolsBackend.get  # the getcol results above are casatools-ordered

    with patch("almasim.services.ms.io._get_backend", return_value=b):
        result = read_native_ms(tmp_path / "test.ms")

    expected_keys = {
        "data",
        "model_data",
        "flag",
        "uvw_m",
        "antenna1",
        "antenna2",
        "time_mjd_s",
        "interval_s",
        "exposure_s",
        "weight",
        "sigma",
        "channel_freq_hz",
        "antenna_names",
        "antenna_diameters_m",
        "antenna_positions_m",
        "field_ra_rad",
        "field_dec_rad",
        "source_name",
        "telescope_name",
        "project_name",
    }
    assert expected_keys == set(result.keys())
    assert result["telescope_name"] == "ALMA"
    assert result["data"].shape == (nrows, ncorr, nchan)


@pytest.mark.unit
def test_read_native_ms_folds_xx_yy_back_to_stokes_i(tmp_path):
    """An ALMASim MS stores Stokes I as XX = YY; the table gets one correlation back."""
    nrows, ncorr, nchan = 3, 2, 4
    main_tb = MagicMock()
    main_tb.getcol.side_effect = lambda col: {
        "DATA": np.ones((ncorr, nchan, nrows), dtype=np.complex64),
        "MODEL_DATA": np.zeros((ncorr, nchan, nrows), dtype=np.complex64),
        "FLAG": np.zeros((ncorr, nchan, nrows), dtype=bool),
        "UVW": np.zeros((3, nrows)),
        "ANTENNA1": np.zeros(nrows, dtype=np.int32),
        "ANTENNA2": np.ones(nrows, dtype=np.int32),
        "TIME": np.linspace(5e9, 5.1e9, nrows),
        "INTERVAL": np.full(nrows, 10.0),
        "EXPOSURE": np.full(nrows, 10.0),
        "WEIGHT": np.ones((ncorr, nrows), dtype=np.float32),
        "SIGMA": np.ones((ncorr, nrows), dtype=np.float32),
    }[col]

    spw_tb = MagicMock()
    spw_tb.getcell.return_value = np.linspace(230e9, 231e9, nchan)

    ant_tb = MagicMock()
    ant_tb.getcol.side_effect = lambda col: {
        "NAME": ["DA41", "DA42"],
        "POSITION": np.zeros((3, 2)),
        "DISH_DIAMETER": np.array([12.0, 7.0]),
    }[col]

    pol_tb = MagicMock()
    pol_tb.getcell.return_value = np.array([9, 12])

    fld_tb = MagicMock()
    fld_tb.getcell.side_effect = lambda col, row: {
        "PHASE_DIR": np.array([[0.1, -0.2]]),
        "NAME": "J0000",
    }[col]

    obs_tb = MagicMock()
    obs_tb.getcell.side_effect = lambda col, row: {
        "TELESCOPE_NAME": "ALMA",
        "PROJECT": "2021.1.S",
    }[col]

    def fake_open(path, readonly=True):
        if path.endswith("SPECTRAL_WINDOW"):
            return spw_tb
        if path.endswith("ANTENNA"):
            return ant_tb
        if path.endswith("FIELD"):
            return fld_tb
        if path.endswith("OBSERVATION"):
            return obs_tb
        if path.endswith("POLARIZATION"):
            return pol_tb
        return main_tb

    b = MagicMock()
    b.open.side_effect = fake_open
    b.get.side_effect = _CasatoolsBackend.get  # the getcol results above are casatools-ordered

    with patch("almasim.services.ms.io._get_backend", return_value=b):
        result = read_native_ms(tmp_path / "test.ms")

    assert result["data"].shape == (nrows, 1, nchan)
    np.testing.assert_array_equal(result["data"], np.ones((nrows, 1, nchan)))
    assert result["weight"].shape == (nrows, 1) and result["sigma"].shape == (nrows, 1)
    assert result["flag"].shape == (nrows, 1, nchan)
    np.testing.assert_array_equal(result["antenna_diameters_m"], [12.0, 7.0])


@pytest.mark.unit
def test_every_table_spec_writes_only_declared_columns(tmp_path):
    """Each spec's values name columns of its own table (MADIV-28 T4.9)."""
    vt = _sample_vt()
    time_range = np.array([5e9, 5.1e9])
    specs = [
        main_table(str(tmp_path / "x.ms"), vt),
        antenna_table(vt["antenna_names"], vt["antenna_positions_m"]),
        spectral_window_table(vt["channel_freq_hz"], 4),
        polarization_table(),
        data_description_table(),
        field_table("S", 0.1, -0.2, time_range),
        observation_table("ALMA", "P", time_range),
        source_table("S", 0.1, -0.2, time_range),
        state_table(),
        history_table("P", time_range),
        feed_table(len(vt["antenna_names"]), time_range),
        processor_table(),
        empty_table("FLAG_CMD"),
        empty_table("POINTING"),
    ]
    for spec in specs:
        declared = {column.name for column in spec.columns}
        assert set(spec.values) <= declared, spec.name
    assert {spec.name for spec in specs[1:]} == set(specs[0].keywords) - {"MS_VERSION"}


# ---------------------------------------------------------------------------
# services.ms.tables (MADIV-28 T2.7)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_open_table_closes_and_passes_lock_options():
    from almasim.services.ms import tables

    opened = []

    class Table:
        def __init__(self, path, **kwargs):
            self.path, self.kwargs, self.closed = path, kwargs, False
            opened.append(self)

        def close(self):
            self.closed = True

    with tables.open_table("x.ms", opener=Table) as tb:
        assert tb.kwargs == {"ack": False}
    with tables.open_table("x.ms", readonly=False, shared=True, opener=Table):
        pass
    with pytest.raises(RuntimeError):
        with tables.open_table("x.ms", opener=Table):
            raise RuntimeError("boom")
    assert [t.closed for t in opened] == [True, True, True]
    assert opened[1].kwargs == {"ack": False, "readonly": False, "lockoptions": "usernoread"}
    with tables.open_table("x.ms", opener=lambda path, **kw: [path]) as rows:
        assert rows == ["x.ms"]  # an object without close() is fine


@pytest.mark.unit
def test_intent_and_subtable_helpers():
    from almasim.services.ms import tables

    mode = "CALIBRATE_PHASE#ON_SOURCE, CALIBRATE_WVR#ON_SOURCE"
    assert tables.intent_tokens(mode) == ["CALIBRATE_PHASE#ON_SOURCE", "CALIBRATE_WVR#ON_SOURCE"]
    assert tables.intent_names(mode) == {"CALIBRATE_PHASE", "CALIBRATE_WVR"}
    assert tables.intent_matches(mode, ["CALIBRATE_PHASE"])
    assert tables.intent_matches(mode, ["CALIBRATE_WVR#ON_SOURCE"])
    assert not tables.intent_matches(mode, ["CALIBRATE_PHASE#OFF_SOURCE", "OBSERVE_TARGET"])

    dd = MagicMock()
    dd.getcol.side_effect = lambda name: {"SPECTRAL_WINDOW_ID": [3, 5], "POLARIZATION_ID": [0, 1]}[
        name
    ]
    assert tables.spw_of_data_description(dd) == [3, 5]
    assert tables.polarization_of_data_description(dd) == [0, 1]
    antenna = MagicMock()
    antenna.getcol.return_value = ["DA41", "DV02"]
    assert tables.antenna_names(antenna) == ["DA41", "DV02"]


@pytest.mark.unit
def test_antenna_table_rejects_a_diameter_count_mismatch():
    from almasim.services.ms.subtables import antenna_table

    with pytest.raises(ValueError, match="1 dish diameters for 2 antennas"):
        antenna_table(["A", "B"], np.zeros((2, 3)), np.array([12.0]))


@pytest.mark.unit
def test_main_table_refuses_more_than_one_correlation(tmp_path):
    from almasim.services.ms.subtables import main_table

    vt = {"data": np.zeros((2, 2, 3)), "flag": np.zeros((2, 1, 3), dtype=bool)}
    with pytest.raises(ValueError, match="one \\(Stokes I\\) correlation"):
        main_table(str(tmp_path / "x.ms"), vt)
