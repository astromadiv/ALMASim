"""Frequency support and utility functions for interferometry."""

import astropy.units as U
import numpy as np


def remove_non_numeric(text: str) -> str:
    """
    Strip characters that are not digits or decimal points.

    Parameters
    ----------
    text : str
        Input text string

    Returns
    -------
    str
        Text with only digits and decimal points
    """
    if not text:
        return ""
    numbers = "0123456789."
    return "".join(char for char in text if char in numbers)


def freq_supp_extractor(  # noqa: C901
    freq_sup: str, obs_freq: U.Quantity
) -> tuple[U.Quantity, U.Quantity, int, U.Quantity]:
    """
    Extract frequency support parameters from a formatted string.

    Parameters
    ----------
    freq_sup : str
        Frequency support string in the form
        "[freq_min..freq_maxGHz,delta_freqkHz,...] U [...]"
    obs_freq : U.Quantity
        Observed frequency to match against frequency ranges

    Returns
    -------
    tuple
        (band_range, central_freq, n_channels, freq_delta) in GHz, GHz, int, GHz.
        freq_delta is the window's resolution as the archive gives it, and
        n_channels the band over it; instrument.channels gives the channels.

    Raises
    ------
    ValueError
        If no valid frequency range is found or obs_freq doesn't match any range
    """
    if not freq_sup or not freq_sup.strip():
        raise ValueError("Frequency support string cannot be empty")

    freq_band, n_channels, freq_mins, freq_maxs, freq_ds = [], [], [], [], []
    freq_sup_parts = freq_sup.split("U")

    for part in freq_sup_parts:
        part = part.strip()
        if not part:
            continue

        # Extract content between brackets
        if not (part.startswith("[") and part.endswith("]")):
            continue

        sup = part[1:-1].split(",")
        if len(sup) < 2:
            continue

        # Parse frequency range (first element)
        freq_range_str = sup[0].split("..")
        if len(freq_range_str) < 2:
            continue

        try:
            freq_min = float(remove_non_numeric(freq_range_str[0]))
            freq_max = float(remove_non_numeric(freq_range_str[1]))
            freq_d = float(remove_non_numeric(sup[1]))
        except (ValueError, IndexError):
            continue

        freq_min = freq_min * U.GHz
        freq_max = freq_max * U.GHz
        freq_d = freq_d * U.kHz
        freq_d = freq_d.to(U.GHz)
        freq_b = freq_max - freq_min

        if freq_d.value <= 0:
            continue

        n_chan = int(freq_b / freq_d)
        if n_chan <= 0:
            continue

        freq_band.append(freq_b)
        n_channels.append(n_chan)
        freq_mins.append(freq_min)
        freq_maxs.append(freq_max)
        freq_ds.append(freq_d)

    if not freq_mins:
        raise ValueError("No valid frequency ranges found in frequency support string")

    freq_ranges = np.array(
        [[freq_mins[i].value, freq_maxs[i].value] for i in range(len(freq_mins))]
    )

    # Find matching frequency range
    matches = np.argwhere(
        (obs_freq.value >= freq_ranges[:, 0]) & (obs_freq.value <= freq_ranges[:, 1])
    )

    if len(matches) == 0:
        raise ValueError(
            f"Observed frequency {obs_freq.value} GHz does not match any frequency range"
        )

    idx_ = matches[0][0]
    freq_range = freq_ranges[idx_]
    band_range = freq_range[1] - freq_range[0]
    n_channels_val = n_channels[idx_]
    central_freq = freq_range[0] + band_range / 2
    freq_d = freq_ds[idx_]

    return band_range * U.GHz, central_freq * U.GHz, n_channels_val, freq_d
