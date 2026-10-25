"""Column descriptors of the MeasurementSet tables ALMASim writes (MADIV-28 T4.9).

Array layout: the writers and readers use python-casacore's row-major
convention, a column is (nrows, *cell_shape) and a cell is in C order, e.g.
DATA (nrows, nchan, ncorr), UVW (nrows, 3), PHASE_DIR (npoly + 1, 2).
casatools uses the reverse (Fortran) order, e.g. DATA (ncorr, nchan, nrows),
so its backend reverses the axes on the way in and out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SUBTABLE_NAMES = (
    "ANTENNA",
    "DATA_DESCRIPTION",
    "FEED",
    "FIELD",
    "FLAG_CMD",
    "HISTORY",
    "OBSERVATION",
    "POINTING",
    "POLARIZATION",
    "PROCESSOR",
    "SOURCE",
    "SPECTRAL_WINDOW",
    "STATE",
)


_EPOCH = {"QuantumUnits": ["s"], "MEASINFO": {"type": "epoch", "Ref": "UTC"}}
_DIRECTION = {"QuantumUnits": ["rad", "rad"], "MEASINFO": {"type": "direction", "Ref": "J2000"}}
_POSITION = {"QuantumUnits": ["m", "m", "m"], "MEASINFO": {"type": "position", "Ref": "ITRF"}}
_FREQUENCY = {
    "QuantumUnits": ["Hz"],
    "MEASINFO": {
        "type": "frequency",
        "VarRefCol": "MEAS_FREQ_REF",
        "TabRefTypes": [
            "REST",
            "LSRK",
            "LSRD",
            "BARY",
            "GEO",
            "TOPO",
            "GALACTO",
            "LGROUP",
            "CMB",
            "Undefined",
        ],
        "TabRefCodes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 64],
    },
}
# Units and measure frames of the columns the MSv2 definition gives them (the
# same wherever a column name appears); casacore's MeasurementSet, and so the
# imagers, refuse an MS whose measure columns lack them.
COLUMN_KEYWORDS: dict[str, dict] = {
    "TIME": _EPOCH,
    "TIME_CENTROID": _EPOCH,
    "TIME_RANGE": _EPOCH,
    "RELEASE_DATE": _EPOCH,
    "TIME_ORIGIN": _EPOCH,
    "INTERVAL": {"QuantumUnits": ["s"]},
    "EXPOSURE": {"QuantumUnits": ["s"]},
    "UVW": {"QuantumUnits": ["m", "m", "m"], "MEASINFO": {"type": "uvw", "Ref": "J2000"}},
    "POSITION": _POSITION,
    "OFFSET": _POSITION,
    "DISH_DIAMETER": {"QuantumUnits": ["m"]},
    "BEAM_OFFSET": _DIRECTION,
    "RECEPTOR_ANGLE": {"QuantumUnits": ["rad"]},
    "DELAY_DIR": _DIRECTION,
    "PHASE_DIR": _DIRECTION,
    "REFERENCE_DIR": _DIRECTION,
    "DIRECTION": _DIRECTION,
    "TARGET": _DIRECTION,
    "PROPER_MOTION": {"QuantumUnits": ["rad/s"]},
    "CHAN_FREQ": _FREQUENCY,
    "REF_FREQUENCY": _FREQUENCY,
    "CHAN_WIDTH": {"QuantumUnits": ["Hz"]},
    "EFFECTIVE_BW": {"QuantumUnits": ["Hz"]},
    "RESOLUTION": {"QuantumUnits": ["Hz"]},
    "TOTAL_BANDWIDTH": {"QuantumUnits": ["Hz"]},
    "CAL": {"QuantumUnits": ["K"]},
    "LOAD": {"QuantumUnits": ["K"]},
}


def _keywords(name: str) -> dict:
    """``name``'s column keywords, lists as arrays (both table backends take those)."""

    def convert(value):
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return np.array(value)
        return value

    return convert(COLUMN_KEYWORDS.get(name, {}))


def _vtype(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, np.float32):
        return "float"
    if isinstance(value, (float, np.floating)):
        return "double"
    if isinstance(value, (complex, np.complexfloating)):
        return "complex"
    return "string"


def _sca_desc(name: str, value) -> tuple[str, dict]:
    return name, {
        "valueType": _vtype(value),
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "",
        "keywords": _keywords(name),
        "maxlen": 0,
        "option": 0,
        "comment": "",
    }


def _arr_desc(name: str, value, ndim: int, shape: list | None = None) -> tuple[str, dict]:
    d: dict = {
        "valueType": _vtype(value),
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "",
        "keywords": _keywords(name),
        "maxlen": 0,
        "ndim": ndim,
        "option": 0,
        "comment": "",
    }
    if shape is not None:
        d["shape"] = np.array(shape)
        d["option"] = 5  # FixedShape
    return name, d


def _make_tabledesc(coldesc_list: list[tuple[str, dict]]) -> dict:
    desc: dict = {
        "_define_hypercolumn_": {},
        "_keywords_": {},
        "_private_keywords_": {},
    }
    for name, col_desc in coldesc_list:
        desc[name] = col_desc
    return desc


@dataclass(frozen=True)
class Column:
    """One column: its name, a value of its type, and for arrays the cell rank and shape."""

    name: str
    example: Any
    ndim: int = 0
    shape: tuple[int, ...] | None = None

    def desc(self) -> tuple[str, dict]:
        if not self.ndim:
            return _sca_desc(self.name, self.example)
        shape = list(self.shape) if self.shape is not None else None
        return _arr_desc(self.name, self.example, self.ndim, shape)


def _sca(name: str, example: Any) -> Column:
    return Column(name, example)


def _arr(name: str, example: Any, ndim: int, shape: tuple[int, ...] | None = None) -> Column:
    return Column(name, example, ndim, shape)


MAIN_COLUMNS = (
    _sca("TIME", 0.0),
    _sca("TIME_CENTROID", 0.0),
    _sca("INTERVAL", 0.0),
    _sca("EXPOSURE", 0.0),
    _sca("ANTENNA1", 0),
    _sca("ANTENNA2", 0),
    _sca("FEED1", 0),
    _sca("FEED2", 0),
    _sca("DATA_DESC_ID", 0),
    _sca("FIELD_ID", 0),
    _sca("ARRAY_ID", 0),
    _sca("OBSERVATION_ID", 0),
    _sca("PROCESSOR_ID", 0),
    _sca("SCAN_NUMBER", 0),
    _sca("STATE_ID", 0),
    _sca("FLAG_ROW", False),
    _arr("UVW", 0.0, 1, (3,)),
    _arr("DATA", np.complex64(0), 2),
    _arr("MODEL_DATA", np.complex64(0), 2),
    _arr("FLAG", False, 2),
    # Required by the MSv2 definition (WSClean and the ms tool refuse an MS
    # without it); no flag categories are written, so its cells stay undefined.
    _arr("FLAG_CATEGORY", False, 3),
    _arr("WEIGHT", np.float32(0), 1),
    _arr("SIGMA", np.float32(0), 1),
)
ANTENNA_COLUMNS = (
    _sca("NAME", ""),
    _sca("STATION", ""),
    _sca("TYPE", ""),
    _sca("MOUNT", ""),
    _sca("DISH_DIAMETER", 0.0),
    _sca("FLAG_ROW", False),
    _arr("POSITION", 0.0, 1, (3,)),
    _arr("OFFSET", 0.0, 1, (3,)),
)
SPECTRAL_WINDOW_COLUMNS = (
    _sca("NUM_CHAN", 0),
    _sca("REF_FREQUENCY", 0.0),
    _sca("TOTAL_BANDWIDTH", 0.0),
    _sca("NET_SIDEBAND", 0),
    _sca("MEAS_FREQ_REF", 0),
    _sca("FLAG_ROW", False),
    _sca("NAME", ""),
    _sca("FREQ_GROUP", 0),
    _sca("FREQ_GROUP_NAME", ""),
    _sca("IF_CONV_CHAIN", 0),
    _arr("CHAN_FREQ", 0.0, 1),
    _arr("CHAN_WIDTH", 0.0, 1),
    _arr("EFFECTIVE_BW", 0.0, 1),
    _arr("RESOLUTION", 0.0, 1),
)
POLARIZATION_COLUMNS = (
    _sca("NUM_CORR", 0),
    _sca("FLAG_ROW", False),
    _arr("CORR_TYPE", 0, 1),
    _arr("CORR_PRODUCT", 0, 2),
)
DATA_DESCRIPTION_COLUMNS = (
    _sca("SPECTRAL_WINDOW_ID", 0),
    _sca("POLARIZATION_ID", 0),
    _sca("FLAG_ROW", False),
)
FIELD_COLUMNS = (
    _sca("NAME", ""),
    _sca("CODE", ""),
    _sca("NUM_POLY", 0),
    _sca("SOURCE_ID", 0),
    _sca("TIME", 0.0),
    _sca("FLAG_ROW", False),
    _arr("DELAY_DIR", 0.0, 2),
    _arr("PHASE_DIR", 0.0, 2),
    _arr("REFERENCE_DIR", 0.0, 2),
)
OBSERVATION_COLUMNS = (
    _sca("TELESCOPE_NAME", ""),
    _sca("OBSERVER", ""),
    _sca("PROJECT", ""),
    _sca("SCHEDULE_TYPE", ""),
    _sca("RELEASE_DATE", 0.0),
    _sca("FLAG_ROW", False),
    _arr("TIME_RANGE", 0.0, 1, (2,)),
    _arr("LOG", "", 1),
    _arr("SCHEDULE", "", 1),
)
SOURCE_COLUMNS = (
    _sca("SOURCE_ID", 0),
    _sca("TIME", 0.0),
    _sca("INTERVAL", 0.0),
    _sca("SPECTRAL_WINDOW_ID", 0),
    _sca("NUM_LINES", 0),
    _sca("NAME", ""),
    _sca("CODE", ""),
    _sca("CALIBRATION_GROUP", 0),
    _arr("DIRECTION", 0.0, 1, (2,)),
    _arr("PROPER_MOTION", 0.0, 1, (2,)),
)
# The MSv2 definition's other required subtables (casacore refuses an MS
# without them): one feed with two linear receptors per antenna, one processor,
# and FLAG_CMD and POINTING with their columns and no rows.
FEED_COLUMNS = (
    _sca("ANTENNA_ID", 0),
    _sca("FEED_ID", 0),
    _sca("SPECTRAL_WINDOW_ID", 0),
    _sca("TIME", 0.0),
    _sca("INTERVAL", 0.0),
    _sca("NUM_RECEPTORS", 0),
    _sca("BEAM_ID", 0),
    _arr("BEAM_OFFSET", 0.0, 2),
    _arr("POLARIZATION_TYPE", "", 1),
    _arr("POL_RESPONSE", np.complex64(0), 2),
    _arr("POSITION", 0.0, 1, (3,)),
    _arr("RECEPTOR_ANGLE", 0.0, 1),
)
FLAG_CMD_COLUMNS = (
    _sca("TIME", 0.0),
    _sca("INTERVAL", 0.0),
    _sca("TYPE", ""),
    _sca("REASON", ""),
    _sca("LEVEL", 0),
    _sca("SEVERITY", 0),
    _sca("APPLIED", False),
    _sca("COMMAND", ""),
)
POINTING_COLUMNS = (
    _sca("ANTENNA_ID", 0),
    _sca("TIME", 0.0),
    _sca("INTERVAL", 0.0),
    _sca("NAME", ""),
    _sca("NUM_POLY", 0),
    _sca("TIME_ORIGIN", 0.0),
    _sca("TRACKING", False),
    _arr("DIRECTION", 0.0, 2),
    _arr("TARGET", 0.0, 2),
)
PROCESSOR_COLUMNS = (
    _sca("TYPE", ""),
    _sca("SUB_TYPE", ""),
    _sca("TYPE_ID", 0),
    _sca("MODE_ID", 0),
    _sca("FLAG_ROW", False),
)
STATE_COLUMNS = (
    _sca("SIG", True),
    _sca("REF", False),
    _sca("CAL", 0.0),
    _sca("LOAD", 0.0),
    _sca("SUB_SCAN", 0),
    _sca("OBS_MODE", ""),
    _sca("FLAG_ROW", False),
)
HISTORY_COLUMNS = (
    _sca("TIME", 0.0),
    _sca("OBSERVATION_ID", 0),
    _sca("MESSAGE", ""),
    _sca("PRIORITY", ""),
    _sca("ORIGIN", ""),
    _sca("OBJECT_ID", 0),
    _sca("APPLICATION", ""),
    _arr("CLI_COMMAND", "", 1),
    _arr("APP_PARAMS", "", 1),
)
