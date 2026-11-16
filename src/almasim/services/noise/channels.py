"""The channels of a real window that ``noise validate`` simulates, and how to
compare their noise with the whole window's.

A window of more than ``MAX_CHANNELS`` channels is simulated over its central
``MAX_CHANNELS``. Tsys changes across a window (an atmospheric line can sit
in it), so the noise over those channels is not the whole window's: the
comparison takes each metric over the same channels on both sides, from the
real window's Tsys spectrum (its SYSCAL, ``tsys.profile``, which follows the
noise variance and, unlike the measured noise, carries no sky). The real
window's line channels are left out of its visibility noise (``noise
extract``), so the simulated window's are too: the same channels, mapped
onto the simulated ones, which run in rising frequency while an MS may list a
window's channels falling.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .extract import line_channels_from_log, parse_channels
from .measurements import WindowNoise

BAND_FILE = "band.json"


def channel_spec(mask: np.ndarray) -> str:
    """The channels ``mask`` selects as "a~b;c" (CASA's selection syntax); "" for none."""
    index = np.flatnonzero(np.asarray(mask, dtype=bool))
    if index.size == 0:
        return ""
    breaks = np.flatnonzero(np.diff(index) > 1)
    starts = np.r_[index[0], index[breaks + 1]]
    ends = np.r_[index[breaks], index[-1]]
    return ";".join(f"{a}" if a == b else f"{a}~{b}" for a, b in zip(starts, ends, strict=True))


@dataclass(frozen=True)
class SimulatedBand:
    """Which of a real window's channels its simulation covers.

    ``first`` and ``count`` are in the MS's channel order; ``falling`` is
    whether the MS lists them in falling frequency; ``line_channels`` are the
    real window's line channels in that order ("a~b;c", "" for none)."""

    total: int
    first: int
    count: int
    falling: bool = False
    line_channels: str = ""

    def simulated(self) -> np.ndarray:
        """The simulated channels, as a mask over the real window's."""
        mask = np.zeros(self.total, dtype=bool)
        mask[self.first : self.first + self.count] = True
        return mask

    def lines(self) -> np.ndarray:
        """The real window's line channels, as a mask over its channels."""
        return parse_channels(self.line_channels, self.total)

    def simulated_line_channels(self) -> str:
        """The line channels among the simulated ones, numbered as the
        simulated MS numbers its channels (rising frequency)."""
        inside = self.lines()[self.first : self.first + self.count]
        return channel_spec(inside[::-1] if self.falling else inside)

    def scales(self, real: WindowNoise | None) -> dict[str, float]:
        """Factors that bring a simulated metric to the real window's channels,
        from the real Tsys spectrum (1 without one, or when every channel is
        simulated): ``visibility``, the mean noise variance over the line-free
        channels; ``band``, a continuum image, whose channels weigh alike (an
        MS's WEIGHT is one per row, for all its channels): its noise variance
        is the channels' mean variance over their number; ``cube``, the
        median channel."""
        profile = None if real is None or real.tsys is None else real.tsys.profile
        if profile is None or len(profile) != self.total or self.count >= self.total:
            variance = np.ones(self.total)
        else:
            variance = np.clip(np.asarray(profile, dtype=float), 1e-6, None)
        sub, free = self.simulated(), ~self.lines()
        both = sub & free
        visibility = (
            float(np.sqrt(variance[free].mean() / variance[both].mean())) if both.any() else 1.0
        )
        # Mean variance over N alike-weighted channels, divided by N: the
        # simulated channels' over the window's.
        band = float(np.sqrt(self.count / self.total * variance.mean() / variance[sub].mean()))
        cube = float(np.sqrt(np.median(variance) / np.median(variance[sub])))
        return {"visibility": visibility, "band": band, "cube": cube}

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1) + "\n"

    @classmethod
    def read(cls, path: Path) -> SimulatedBand | None:
        """The band recorded at ``path``; None when there is none."""
        if not path.is_file():
            return None
        return cls(**json.loads(path.read_text()))


def simulated_band(
    total: int,
    count: int,
    *,
    falling: bool = False,
    lines: np.ndarray | None = None,
) -> SimulatedBand:
    """The central ``count`` of a window's ``total`` channels."""
    count = min(int(count), int(total))
    mask = np.zeros(total, dtype=bool) if lines is None else np.asarray(lines, dtype=bool)
    return SimulatedBand(
        total=int(total),
        first=(int(total) - count) // 2,
        count=count,
        falling=bool(falling),
        line_channels=channel_spec(mask),
    )


def real_line_channels(
    total: int, *, line_channels: str | None = None, task_log: str | None = None
) -> np.ndarray:
    """The real window's line channels as ``noise extract`` left them out:
    its ``line_channels``, else its imaging task's log; none without either."""
    if line_channels:
        return parse_channels(line_channels, total)
    if task_log and Path(task_log).is_file():
        return line_channels_from_log(Path(task_log), total)
    return np.zeros(total, dtype=bool)


def channels_falling(ms: str, spw: int) -> dict[str, bool]:
    """``{"falling": ...}``: whether the MS lists spectral window ``spw``'s
    channels in falling frequency. Run in a process of its own (see
    ``validate.measure``)."""
    from ..imaging.ms_geometry import import_casacore_tables

    table = import_casacore_tables()
    with table(str(Path(ms) / "SPECTRAL_WINDOW"), ack=False) as spectral:
        width = np.asarray(spectral.getcell("CHAN_WIDTH", int(spw)), dtype=float)
    return {"falling": bool(width.size and width[0] < 0)}
