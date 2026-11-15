"""ALMA array types and dish sizes from antenna names (MADIV-28 T2.11)."""

from __future__ import annotations

from .constants import DISH_DIAMETER_7M_M, DISH_DIAMETER_12M_M

# Antenna name prefixes of each array: DA/DV are 12 m dishes, CM the 7 m
# dishes of the ACA, PM the total-power antennas.
_ARRAY_PREFIXES = (("12m", ("DA", "DV")), ("7m", ("CM",)), ("TP", ("PM",)))


def derive_array_type(antennas: object) -> str:
    """The arrays present in an antenna string, '+'-joined: '12m', '7m+TP', '12m+7m+TP', ...

    Returns "" for an empty or non-string value, or when no ALMA antenna is
    recognised.
    """
    if not isinstance(antennas, str) or not antennas:
        return ""
    upper = antennas.upper()
    return "+".join(
        array for array, prefixes in _ARRAY_PREFIXES if any(p in upper for p in prefixes)
    )


def antenna_array_type(antenna: str) -> str:
    """The array one antenna name belongs to: '7m' (CM), 'TP' (PM), else '12m'."""
    upper = antenna.upper()
    if "CM" in upper:
        return "7m"
    if "PM" in upper:
        return "TP"
    return "12m"


def pad_dish_diameter_m(pad: str, array_diameter_m: float) -> float:
    """The diameter (m) of the dish on ``pad`` in an array of
    ``array_diameter_m`` dishes: 12 m on a total-power pad (T701-T704, where
    ALMA's 12 m total-power dishes stand), the array's on any other pad. A
    7 m array's MS can correlate the total-power dishes with the 7 m ones."""
    if str(pad).upper().startswith("T7"):
        return DISH_DIAMETER_12M_M
    return float(array_diameter_m)


def dish_diameter_m(array_type: str) -> float:
    """A representative dish diameter (m) for an ALMA array type: 7 m for '7m', else 12 m."""
    if (array_type or "12m").lower() == "7m":
        return DISH_DIAMETER_7M_M
    return DISH_DIAMETER_12M_M
