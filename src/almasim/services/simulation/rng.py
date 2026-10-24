"""One seed for every random draw of a simulation run (MADIV-28 T5.1).

``SimulationParams.seed`` seeds independent streams, one per part of the
run, so a change in how many numbers one part draws never shifts another's.
Without a seed every stream draws fresh entropy.
"""

from __future__ import annotations

import numpy as np

STREAMS = {
    "sky": 0,
    "interferometer": 1,
    "total_power": 2,
    "background": 3,
    "metadata": 4,
    "receiver": 5,
}


def stream_entropy(seed: int | None, stream: str, *keys: int) -> tuple[int, ...] | None:
    """The entropy of ``stream`` (and ``keys``, e.g. a configuration index), or None unseeded."""
    if seed is None:
        return None
    return (int(seed), STREAMS[stream], *(int(key) for key in keys))


def run_rng(seed: int | None, stream: str, *keys: int) -> np.random.Generator:
    """The generator of one stream of a run: reproducible with a seed, fresh without."""
    return np.random.default_rng(stream_entropy(seed, stream, *keys))
