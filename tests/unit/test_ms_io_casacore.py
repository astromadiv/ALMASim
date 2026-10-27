"""The MS writer through the python-casacore backend writes a correctly laid-out MS (MADIV-28 T1.5).

The casacore fallback used to transpose arrays as if python-casacore were
column-major like casatools, so it wrote and read DATA, UVW and the antenna
positions with their axes swapped. These tests write a real MS with that
backend and check the on-disk layout and the round trip.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from almasim.services.ms import backends
from almasim.services.ms import io as ms_io

tables = pytest.importorskip("casacore.tables")

# The visibility table holds Stokes I; the MS holds it as XX and YY (MADIV-28 T5.12).
NROWS, NCORR, NCHAN, NANT = 6, 1, 5, 3


def _visibility_table():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(NROWS, NCORR, NCHAN)) + 1j * rng.normal(size=(NROWS, NCORR, NCHAN))
    return {
        "data": data.astype(np.complex64),
        "model_data": (2 * data).astype(np.complex64),
        "flag": rng.random((NROWS, NCORR, NCHAN)) > 0.7,
        "uvw_m": rng.uniform(-500, 500, size=(NROWS, 3)),
        "antenna1": np.array([0, 0, 1, 0, 0, 1], dtype=np.int32),
        "antenna2": np.array([1, 2, 2, 1, 2, 2], dtype=np.int32),
        "time_mjd_s": np.array([5e9, 5e9, 5e9, 5e9 + 10, 5e9 + 10, 5e9 + 10]),
        "interval_s": np.full(NROWS, 10.0),
        "exposure_s": np.full(NROWS, 10.0),
        "weight": rng.uniform(0.5, 2.0, size=(NROWS, NCORR)).astype(np.float32),
        "sigma": rng.uniform(0.5, 2.0, size=(NROWS, NCORR)).astype(np.float32),
        "channel_freq_hz": 230e9 + 1e6 * np.arange(NCHAN),
        "antenna_names": ["DA41", "DA42", "DV01"],
        "antenna_diameters_m": np.array([12.0, 12.0, 7.0]),
        "antenna_positions_m": np.array(
            [
                [2225061.0, -5440057.0, -2481681.0],
                [2225070.0, -5440050.0, -2481690.0],
                [2225080.0, -5440040.0, -2481700.0],
            ]  # fmt: skip
        ),
        "field_ra_rad": 1.25,
        "field_dec_rad": -0.4,
    }


@pytest.fixture
def written(tmp_path):
    vt = _visibility_table()
    with patch.object(ms_io, "_get_backend", return_value=backends._CasacoreBackend()):
        path = ms_io.export_native_ms(
            ms_path=tmp_path / "sim.ms", visibility_table=vt, project_name="P", source_name="S"
        )
    return vt, path


def test_main_table_layout_matches_the_measurementset_convention(written):
    vt, path = written
    ms = tables.table(path, ack=False)
    try:
        data = ms.getcol("DATA")
        assert data.shape == (NROWS, NCHAN, 2), "MS DATA is (nrows, nchan, ncorr): XX and YY"
        for corr in range(2):  # both equal Stokes I
            np.testing.assert_array_equal(data[:, :, corr], vt["data"][:, 0, :])
            np.testing.assert_array_equal(ms.getcol("FLAG")[:, :, corr], vt["flag"][:, 0, :])
            np.testing.assert_array_equal(ms.getcol("WEIGHT")[:, corr], vt["weight"][:, 0])
        np.testing.assert_array_equal(ms.getcol("UVW"), vt["uvw_m"])
    finally:
        ms.close()


def test_subtable_layout(written):
    vt, path = written
    antenna = tables.table(f"{path}/ANTENNA", ack=False)
    np.testing.assert_array_equal(antenna.getcol("POSITION"), vt["antenna_positions_m"])
    np.testing.assert_array_equal(antenna.getcol("DISH_DIAMETER"), [12.0, 12.0, 7.0])
    antenna.close()
    state = tables.table(f"{path}/STATE", ack=False)
    assert state.getcell("OBS_MODE", 0) == "OBSERVE_TARGET#ON_SOURCE"
    state.close()
    field = tables.table(f"{path}/FIELD", ack=False)
    phase_dir = field.getcell("PHASE_DIR", 0)
    field.close()
    assert phase_dir.shape == (1, 2), "PHASE_DIR is (npoly + 1, 2) in casacore order"
    np.testing.assert_allclose(phase_dir[0], [vt["field_ra_rad"], vt["field_dec_rad"]])
    polarization = tables.table(f"{path}/POLARIZATION", ack=False)
    assert polarization.getcell("CORR_PRODUCT", 0).shape == (2, 2)
    assert list(polarization.getcell("CORR_TYPE", 0)) == [9, 12]  # XX, YY
    polarization.close()


def test_round_trip(written):
    vt, path = written
    with patch.object(ms_io, "_get_backend", return_value=backends._CasacoreBackend()):
        back = ms_io.read_native_ms(path)
    keys = ("data", "model_data", "flag", "uvw_m", "weight", "sigma", "antenna_positions_m")
    for key in (*keys, "antenna_diameters_m"):
        np.testing.assert_array_equal(back[key], vt[key], err_msg=key)
    np.testing.assert_array_equal(back["channel_freq_hz"], vt["channel_freq_hz"])
    assert back["antenna_names"] == vt["antenna_names"]
    assert back["field_ra_rad"] == pytest.approx(vt["field_ra_rad"])
    assert back["field_dec_rad"] == pytest.approx(vt["field_dec_rad"])


def test_backend_import_errors_fall_back_quietly_but_other_failures_are_logged(caplog):
    class Broken:
        def __init__(self):
            raise RuntimeError("libpmix.so: cannot open shared object file")

    class Missing:
        def __init__(self):
            raise ImportError("No module named 'casatools'")

    with patch.object(backends, "_CasatoolsBackend", Broken):
        with caplog.at_level("WARNING", logger=backends.__name__):
            assert isinstance(backends._get_backend(), backends._CasacoreBackend)
    assert "libpmix" in caplog.text

    caplog.clear()
    with patch.object(backends, "_CasatoolsBackend", Missing):
        with caplog.at_level("WARNING", logger=backends.__name__):
            assert isinstance(backends._get_backend(), backends._CasacoreBackend)
    assert caplog.text == ""


def test_each_execution_block_is_an_observation(tmp_path):
    """Rows of two execution blocks: OBSERVATION_ID per row, one OBSERVATION row each."""
    vt = _visibility_table()
    vt["time_mjd_s"] = np.array([5e9, 5e9, 5e9, 5e9 + 86400, 5e9 + 86400, 5e9 + 86400])
    vt["observation_id"] = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    with patch.object(ms_io, "_get_backend", return_value=backends._CasacoreBackend()):
        path = ms_io.export_native_ms(
            ms_path=tmp_path / "eb.ms", visibility_table=vt, project_name="P", source_name="S"
        )
    with tables.table(path, ack=False) as ms:
        np.testing.assert_array_equal(ms.getcol("OBSERVATION_ID"), vt["observation_id"])
    with tables.table(f"{path}/OBSERVATION", ack=False) as obs:
        assert obs.nrows() == 2
        np.testing.assert_array_equal(
            obs.getcol("TIME_RANGE"), [[5e9, 5e9], [5e9 + 86400, 5e9 + 86400]]
        )
        assert list(obs.getcol("TELESCOPE_NAME")) == ["ALMA", "ALMA"]
