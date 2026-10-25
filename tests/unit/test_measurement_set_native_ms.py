"""Round-trip write/read tests for the casatools/python-casacore MSv2 writer."""

import numpy as np
import pytest

try:
    import casatools
except Exception:  # noqa: BLE001 - casatools may be installed but broken
    pytest.skip("casatools not available or not functional", allow_module_level=True)

from almasim.services.ms.io import export_native_ms, read_native_ms

# ---------------------------------------------------------------------------
# Shared fixture — realistic, non-trivial sample data
# ---------------------------------------------------------------------------

NROWS = 6  # 3 baselines × 2 time samples
NCORR = 1
NCHAN = 8
NANT = 3

RNG = np.random.default_rng(42)

_DATA = (
    RNG.standard_normal((NROWS, NCORR, NCHAN)) + 1j * RNG.standard_normal((NROWS, NCORR, NCHAN))
).astype(np.complex64)
_MODEL = (
    RNG.standard_normal((NROWS, NCORR, NCHAN)) + 1j * RNG.standard_normal((NROWS, NCORR, NCHAN))
).astype(np.complex64)
_FLAG = RNG.integers(0, 2, (NROWS, NCORR, NCHAN), dtype=bool)
_UVW = RNG.standard_normal((NROWS, 3))
_TIME = np.linspace(5.0e9, 5.0e9 + 30.0, NROWS)
_WEIGHT = np.abs(RNG.standard_normal((NROWS, NCORR))).astype(np.float32) + 0.1
_SIGMA = (1.0 / np.sqrt(_WEIGHT)).astype(np.float32)
_CHAN_FREQ = np.linspace(1.0e11, 1.007e11, NCHAN)
_ANT_POS = np.array(
    [
        [0.0, 0.0, 0.0],
        [200.0, 0.0, 0.0],
        [100.0, 150.0, 0.0],
    ],
    dtype=np.float64,
)
_ANT_NAMES = ["DA41", "DA42", "DA43"]
_RA = 1.2345
_DEC = -0.6789


def _sample_vt():
    return {
        "uvw_m": _UVW.copy(),
        "antenna1": np.array([0, 0, 1, 0, 0, 1], dtype=np.int32),
        "antenna2": np.array([1, 2, 2, 1, 2, 2], dtype=np.int32),
        "time_mjd_s": _TIME.copy(),
        "interval_s": np.full(NROWS, 10.0),
        "exposure_s": np.full(NROWS, 10.0),
        "data": _DATA.copy(),
        "model_data": _MODEL.copy(),
        "flag": _FLAG.copy(),
        "weight": _WEIGHT.copy(),
        "sigma": _SIGMA.copy(),
        "channel_freq_hz": _CHAN_FREQ.copy(),
        "antenna_names": _ANT_NAMES,
        "antenna_positions_m": _ANT_POS.copy(),
        "field_ra_rad": _RA,
        "field_dec_rad": _DEC,
        "observation_date": "2024-01-15",
    }


@pytest.fixture(scope="module")
def ms(tmp_path_factory):
    p = tmp_path_factory.mktemp("ms") / "round_trip.ms"
    export_native_ms(
        ms_path=p,
        visibility_table=_sample_vt(),
        project_name="test_project",
        source_name="test_source",
        telescope_name="ALMA",
    )
    return p


# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------


def test_ms_is_directory(ms):
    assert ms.is_dir()


@pytest.mark.parametrize(
    "subtable",
    [
        "ANTENNA",
        "DATA_DESCRIPTION",
        "FIELD",
        "HISTORY",
        "OBSERVATION",
        "POLARIZATION",
        "SOURCE",
        "SPECTRAL_WINDOW",
        "STATE",
    ],
)
def test_subtable_exists(ms, subtable):
    assert (ms / subtable).is_dir()


# ---------------------------------------------------------------------------
# Main table — row count, column presence, keywords
# ---------------------------------------------------------------------------


# Run in a process of its own: python-casacore aborts when loaded beside casatools.
_MSV2_CHECK = """
import sys
from casacore.tables import required_ms_desc, table
ms = sys.argv[1]
names = ("MAIN", "ANTENNA", "DATA_DESCRIPTION", "FEED", "FIELD", "FLAG_CMD", "HISTORY",
         "OBSERVATION", "POINTING", "POLARIZATION", "PROCESSOR", "SOURCE", "SPECTRAL_WINDOW",
         "STATE")
for name in names:
    t = table(ms if name == "MAIN" else ms + "::" + name, ack=False)
    required = [c for c in required_ms_desc(name) if not c.startswith("_")]
    missing = [c for c in required if c not in t.colnames()]
    assert not missing, (name, missing)
"""


def test_ms_meets_the_msv2_definition(ms):
    """The MS has every subtable and column the MSv2 definition requires, linked
    from the main table by table keywords: casacore's MeasurementSet, and so
    WSClean and the imaging pipeline, refuse it otherwise. (casatools stores a
    "Table: <path>" keyword written before its subtable exists as plain text.)"""
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("casacore") is None:
        pytest.skip("python-casacore not installed")
    result = subprocess.run(
        [sys.executable, "-c", _MSV2_CHECK, str(ms)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr[-800:]


def test_main_nrows(ms):
    tb = casatools.table()
    tb.open(str(ms), nomodify=True)
    assert tb.nrows() == NROWS
    tb.close()


def test_main_ms_version_keyword(ms):
    tb = casatools.table()
    tb.open(str(ms), nomodify=True)
    assert abs(tb.getkeyword("MS_VERSION") - 2.0) < 1e-9
    tb.close()


@pytest.mark.parametrize(
    "col",
    [
        "TIME",
        "TIME_CENTROID",
        "INTERVAL",
        "EXPOSURE",
        "ANTENNA1",
        "ANTENNA2",
        "FEED1",
        "FEED2",
        "DATA_DESC_ID",
        "FIELD_ID",
        "ARRAY_ID",
        "OBSERVATION_ID",
        "PROCESSOR_ID",
        "SCAN_NUMBER",
        "STATE_ID",
        "FLAG_ROW",
        "UVW",
        "DATA",
        "MODEL_DATA",
        "FLAG",
        "WEIGHT",
        "SIGMA",
    ],
)
def test_main_column_exists(ms, col):
    tb = casatools.table()
    tb.open(str(ms), nomodify=True)
    assert col in tb.colnames()
    tb.close()


# ---------------------------------------------------------------------------
# Main table — data round-trip fidelity
# ---------------------------------------------------------------------------


def test_main_time(ms):
    tb = casatools.table()
    tb.open(str(ms), nomodify=True)
    np.testing.assert_allclose(tb.getcol("TIME"), _TIME, rtol=1e-10)
    tb.close()


def test_main_antenna_pairs(ms):
    tb = casatools.table()
    tb.open(str(ms), nomodify=True)
    np.testing.assert_array_equal(tb.getcol("ANTENNA1"), [0, 0, 1, 0, 0, 1])
    np.testing.assert_array_equal(tb.getcol("ANTENNA2"), [1, 2, 2, 1, 2, 2])
    tb.close()


def test_main_uvw(ms):
    tb = casatools.table()
    tb.open(str(ms), nomodify=True)
    # casatools returns (3, nrows); transpose to (nrows, 3)
    uvw_read = tb.getcol("UVW").T
    np.testing.assert_allclose(uvw_read, _UVW, rtol=1e-10)
    tb.close()


def test_main_data_roundtrip(ms):
    tb = casatools.table()
    tb.open(str(ms), nomodify=True)
    # casatools returns (ncorr, nchan, nrows); transpose to (nrows, ncorr, nchan)
    data_read = tb.getcol("DATA").transpose(2, 0, 1)
    np.testing.assert_allclose(data_read, _DATA, rtol=1e-6, atol=1e-6)
    tb.close()


def test_main_model_data_roundtrip(ms):
    tb = casatools.table()
    tb.open(str(ms), nomodify=True)
    model_read = tb.getcol("MODEL_DATA").transpose(2, 0, 1)
    np.testing.assert_allclose(model_read, _MODEL, rtol=1e-6, atol=1e-6)
    tb.close()


def test_main_flag_roundtrip(ms):
    tb = casatools.table()
    tb.open(str(ms), nomodify=True)
    flag_read = tb.getcol("FLAG").transpose(2, 0, 1)
    np.testing.assert_array_equal(flag_read, _FLAG)
    tb.close()


def test_main_weight_roundtrip(ms):
    tb = casatools.table()
    tb.open(str(ms), nomodify=True)
    # (ncorr, nrows) → (nrows, ncorr)
    weight_read = tb.getcol("WEIGHT").T
    np.testing.assert_allclose(weight_read, _WEIGHT, rtol=1e-6)
    tb.close()


def test_main_sigma_roundtrip(ms):
    tb = casatools.table()
    tb.open(str(ms), nomodify=True)
    sigma_read = tb.getcol("SIGMA").T
    np.testing.assert_allclose(sigma_read, _SIGMA, rtol=1e-6)
    tb.close()


def test_main_flag_row(ms):
    tb = casatools.table()
    tb.open(str(ms), nomodify=True)
    expected = np.all(_FLAG, axis=(1, 2))
    np.testing.assert_array_equal(tb.getcol("FLAG_ROW"), expected)
    tb.close()


def test_main_interval_exposure(ms):
    tb = casatools.table()
    tb.open(str(ms), nomodify=True)
    np.testing.assert_allclose(tb.getcol("INTERVAL"), np.full(NROWS, 10.0))
    np.testing.assert_allclose(tb.getcol("EXPOSURE"), np.full(NROWS, 10.0))
    tb.close()


def test_main_index_columns_zero(ms):
    tb = casatools.table()
    tb.open(str(ms), nomodify=True)
    for col in (
        "FEED1",
        "FEED2",
        "DATA_DESC_ID",
        "FIELD_ID",
        "ARRAY_ID",
        "OBSERVATION_ID",
        "STATE_ID",
    ):
        np.testing.assert_array_equal(tb.getcol(col), np.zeros(NROWS, dtype=np.int32), err_msg=col)
    np.testing.assert_array_equal(tb.getcol("SCAN_NUMBER"), np.ones(NROWS, dtype=np.int32))
    np.testing.assert_array_equal(tb.getcol("PROCESSOR_ID"), np.full(NROWS, -1, dtype=np.int32))
    tb.close()


# ---------------------------------------------------------------------------
# ANTENNA subtable
# ---------------------------------------------------------------------------


def test_antenna_nrows(ms):
    tb = casatools.table()
    tb.open(str(ms / "ANTENNA"), nomodify=True)
    assert tb.nrows() == NANT
    tb.close()


def test_antenna_names(ms):
    tb = casatools.table()
    tb.open(str(ms / "ANTENNA"), nomodify=True)
    assert list(tb.getcol("NAME")) == _ANT_NAMES
    assert list(tb.getcol("STATION")) == _ANT_NAMES
    tb.close()


def test_antenna_positions(ms):
    tb = casatools.table()
    tb.open(str(ms / "ANTENNA"), nomodify=True)
    # (3, nant) → (nant, 3)
    pos_read = tb.getcol("POSITION").T
    np.testing.assert_allclose(pos_read, _ANT_POS, rtol=1e-10)
    tb.close()


def test_antenna_dish_diameter(ms):
    tb = casatools.table()
    tb.open(str(ms / "ANTENNA"), nomodify=True)
    np.testing.assert_allclose(tb.getcol("DISH_DIAMETER"), np.full(NANT, 12.0))
    tb.close()


# ---------------------------------------------------------------------------
# SPECTRAL_WINDOW subtable
# ---------------------------------------------------------------------------


def test_spw_nchan(ms):
    tb = casatools.table()
    tb.open(str(ms / "SPECTRAL_WINDOW"), nomodify=True)
    assert tb.nrows() == 1
    assert tb.getcell("NUM_CHAN", 0) == NCHAN
    tb.close()


def test_spw_chan_freq(ms):
    tb = casatools.table()
    tb.open(str(ms / "SPECTRAL_WINDOW"), nomodify=True)
    np.testing.assert_allclose(tb.getcell("CHAN_FREQ", 0), _CHAN_FREQ, rtol=1e-10)
    tb.close()


def test_spw_ref_frequency(ms):
    tb = casatools.table()
    tb.open(str(ms / "SPECTRAL_WINDOW"), nomodify=True)
    assert abs(tb.getcell("REF_FREQUENCY", 0) - float(np.median(_CHAN_FREQ))) < 1.0
    tb.close()


def test_spw_total_bandwidth(ms):
    chan_width = np.gradient(_CHAN_FREQ)
    expected_bw = float(np.sum(np.abs(chan_width)))
    tb = casatools.table()
    tb.open(str(ms / "SPECTRAL_WINDOW"), nomodify=True)
    assert abs(tb.getcell("TOTAL_BANDWIDTH", 0) - expected_bw) < 1.0
    tb.close()


# ---------------------------------------------------------------------------
# POLARIZATION subtable
# ---------------------------------------------------------------------------


def test_polarization(ms):
    tb = casatools.table()
    tb.open(str(ms / "POLARIZATION"), nomodify=True)
    assert tb.nrows() == 1
    assert tb.getcell("NUM_CORR", 0) == NCORR
    np.testing.assert_array_equal(tb.getcell("CORR_TYPE", 0), [9])  # Stokes I
    tb.close()


# ---------------------------------------------------------------------------
# DATA_DESCRIPTION subtable
# ---------------------------------------------------------------------------


def test_data_description(ms):
    tb = casatools.table()
    tb.open(str(ms / "DATA_DESCRIPTION"), nomodify=True)
    assert tb.nrows() == 1
    assert tb.getcell("SPECTRAL_WINDOW_ID", 0) == 0
    assert tb.getcell("POLARIZATION_ID", 0) == 0
    tb.close()


# ---------------------------------------------------------------------------
# FIELD subtable
# ---------------------------------------------------------------------------


def test_field(ms):
    tb = casatools.table()
    tb.open(str(ms / "FIELD"), nomodify=True)
    assert tb.nrows() == 1
    assert tb.getcell("NAME", 0) == "test_source"
    direction = tb.getcell("PHASE_DIR", 0)
    assert abs(direction.flat[0] - _RA) < 1e-10
    assert abs(direction.flat[1] - _DEC) < 1e-10
    tb.close()


# ---------------------------------------------------------------------------
# OBSERVATION subtable
# ---------------------------------------------------------------------------


def test_observation(ms):
    tb = casatools.table()
    tb.open(str(ms / "OBSERVATION"), nomodify=True)
    assert tb.nrows() == 1
    assert tb.getcell("TELESCOPE_NAME", 0) == "ALMA"
    assert tb.getcell("PROJECT", 0) == "test_project"
    time_range = tb.getcell("TIME_RANGE", 0)
    assert abs(time_range[0] - _TIME.min()) < 1e-6
    assert abs(time_range[1] - _TIME.max()) < 1e-6
    tb.close()


# ---------------------------------------------------------------------------
# SOURCE subtable
# ---------------------------------------------------------------------------


def test_source(ms):
    tb = casatools.table()
    tb.open(str(ms / "SOURCE"), nomodify=True)
    assert tb.nrows() == 1
    assert tb.getcell("NAME", 0) == "test_source"
    direction = tb.getcell("DIRECTION", 0)
    assert abs(direction[0] - _RA) < 1e-10
    assert abs(direction[1] - _DEC) < 1e-10
    tb.close()


# ---------------------------------------------------------------------------
# STATE subtable
# ---------------------------------------------------------------------------


def test_state(ms):
    tb = casatools.table()
    tb.open(str(ms / "STATE"), nomodify=True)
    assert tb.nrows() == 1
    assert tb.getcell("OBS_MODE", 0) == "ON_SOURCE"
    assert tb.getcell("SIG", 0) is True
    assert tb.getcell("REF", 0) is False
    tb.close()


# ---------------------------------------------------------------------------
# HISTORY subtable
# ---------------------------------------------------------------------------


def test_history(ms):
    tb = casatools.table()
    tb.open(str(ms / "HISTORY"), nomodify=True)
    assert tb.nrows() == 1
    assert tb.getcell("APPLICATION", 0) == "ALMASim"
    assert tb.getcell("MESSAGE", 0) == "ALMASim MSv2 export"
    tb.close()


# ---------------------------------------------------------------------------
# read_native_ms round-trip — via ALMASim's own reader
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vt_roundtrip(ms):
    return read_native_ms(ms)


def test_roundtrip_nrows(vt_roundtrip):
    assert vt_roundtrip["data"].shape[0] == NROWS


def test_roundtrip_data(vt_roundtrip):
    np.testing.assert_allclose(vt_roundtrip["data"], _DATA, rtol=1e-6, atol=1e-6)


def test_roundtrip_model_data(vt_roundtrip):
    np.testing.assert_allclose(vt_roundtrip["model_data"], _MODEL, rtol=1e-6, atol=1e-6)


def test_roundtrip_flag(vt_roundtrip):
    np.testing.assert_array_equal(vt_roundtrip["flag"], _FLAG)


def test_roundtrip_uvw(vt_roundtrip):
    np.testing.assert_allclose(vt_roundtrip["uvw_m"], _UVW, rtol=1e-10)


def test_roundtrip_time(vt_roundtrip):
    np.testing.assert_allclose(vt_roundtrip["time_mjd_s"], _TIME, rtol=1e-10)


def test_roundtrip_weight(vt_roundtrip):
    np.testing.assert_allclose(vt_roundtrip["weight"], _WEIGHT, rtol=1e-6)


def test_roundtrip_sigma(vt_roundtrip):
    np.testing.assert_allclose(vt_roundtrip["sigma"], _SIGMA, rtol=1e-6)


def test_roundtrip_chan_freq(vt_roundtrip):
    np.testing.assert_allclose(vt_roundtrip["channel_freq_hz"], _CHAN_FREQ, rtol=1e-10)


def test_roundtrip_antenna_names(vt_roundtrip):
    assert vt_roundtrip["antenna_names"] == _ANT_NAMES


def test_roundtrip_antenna_positions(vt_roundtrip):
    np.testing.assert_allclose(vt_roundtrip["antenna_positions_m"], _ANT_POS, rtol=1e-10)


def test_roundtrip_field_direction(vt_roundtrip):
    assert abs(vt_roundtrip["field_ra_rad"] - _RA) < 1e-10
    assert abs(vt_roundtrip["field_dec_rad"] - _DEC) < 1e-10


def test_roundtrip_source_name(vt_roundtrip):
    assert vt_roundtrip["source_name"] == "test_source"


def test_roundtrip_telescope(vt_roundtrip):
    assert vt_roundtrip["telescope_name"] == "ALMA"


def test_roundtrip_project(vt_roundtrip):
    assert vt_roundtrip["project_name"] == "test_project"
