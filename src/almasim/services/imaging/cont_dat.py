"""The ALMA pipeline's line channels (``cont.dat``), for the continuum fit.

The pipeline's ``findcont`` writes the continuum ranges it found on each
window's image cube to ``cont.dat``, delivered in the member folder's
``calibration/*auxproducts.tgz``. The file has one block per field and
pipeline spectral window::

    Field: <name>

    SpectralWindow: 23
    339.15~339.34GHz LSRK
    339.65~340.56GHz LSRK

    SpectralWindow: 25
    ALL
    341.41~343.29GHz LSRK

A range line is continuum in the frame it names (LSRK when it names none).
``ALL`` makes the whole window continuum and ``NONE`` none of it; the range
that can follow either spans the window. Every channel outside the
continuum ranges is a pipeline line channel.

The spectral window ids are the pipeline MS's, which can differ from a split
MS's, so a window is matched by field name and frequency: among the field's
blocks with at least half their ranges' width inside the window, the one
with the most, the channel frequencies converted from the MS's frame (TOPO
for an ALMA MS) to the block's with casacore measures, at the ALMA
observatory, in the direction of the field's PHASE_DIR and at the median
TIME of its rows. An ``ALL`` or ``NONE`` block without a range carries no
frequencies: it is matched when the ranges other fields give for its
pipeline window place it over this one, or when it is the field's only such
block they place nowhere.

:func:`pipeline_line_channels` puts it together for a single-window MS: the
cont.dat of the member the source MS's execution block belongs to (its
``ous_index.csv`` row), matched to the window, as a channel mask.
"""

from __future__ import annotations

import os
import re
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "ContBlock",
    "WindowFrequencies",
    "cont_dat_log",
    "find_cont_dat",
    "line_mask",
    "match_window",
    "ms_window",
    "parse_cont_dat",
    "pipeline_line_channels",
    "read_cont_dat",
    "without_pipeline_lines",
]

AUXPRODUCTS_GLOB = "*auxproducts.tgz"
CONT_DAT_NAME = "cont.dat"
# A block is a window's only when at least this share of its ranges' width
# lies inside the window: a neighbouring window's block can touch its edge.
MIN_INSIDE = 0.5

# casacore's MFrequency reference codes (MEAS_FREQ_REF), by number.
_FREQUENCY_FRAMES = ("REST", "LSRK", "LSRD", "BARY", "GEO", "TOPO", "GALACTO", "LGROUP", "CMB")
_RANGE = re.compile(r"([\d.eE+-]+)~([\d.eE+-]+)\s*([kMGT]?Hz)\s*(\w+)?")
_HZ = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}

FrequencyRange = tuple[float, float, str]


@dataclass
class ContBlock:
    """One field's and pipeline window's block of cont.dat.

    ``ranges`` are ``(low Hz, high Hz, frame)``; ``whole`` is ``"ALL"`` or
    ``"NONE"`` when the block says so (its ranges then only place the window).
    """

    ranges: list[FrequencyRange] = field(default_factory=list)
    whole: str | None = None

    def describe(self) -> str:
        return self.whole or f"{len(self.ranges)} continuum ranges"


def parse_cont_dat(text: str) -> dict[tuple[str, int], ContBlock]:
    """``{(field, spw): ContBlock}`` from cont.dat text.

    Lines that are neither a range nor ``ALL`` / ``NONE`` are ignored.
    """
    blocks: dict[tuple[str, int], ContBlock] = {}
    name: str | None = None
    block: ContBlock | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("Field:"):
            name, block = line.split(":", 1)[1].strip(), None
        elif line.startswith("SpectralWindow:") and name is not None:
            spw = line.split(":", 1)[1].split()
            if spw and spw[0].isdigit():
                block = blocks[(name, int(spw[0]))] = ContBlock()
        elif line and block is not None:
            if line.upper() in ("ALL", "NONE"):
                block.whole = line.upper()
            elif m := _RANGE.match(line):
                scale = _HZ[m[3]]
                low, high = sorted((float(m[1]) * scale, float(m[2]) * scale))
                block.ranges.append((low, high, (m[4] or "LSRK").upper()))
    return blocks


@dataclass(frozen=True)
class WindowFrequencies:
    """A single-window MS's fields and channels.

    ``in_frame(frame)`` gives the ``channels`` frequencies (Hz) in that frame.
    """

    field_names: frozenset[str]
    channels: int
    in_frame: Callable[[str], np.ndarray]


class _Overlap:
    """How much of some frequency ranges lies inside a window's channels."""

    def __init__(self, window: WindowFrequencies):
        self.window = window
        self.spans: dict[str, tuple[float, float]] = {}

    def __call__(self, ranges: Sequence[FrequencyRange]) -> tuple[float, float]:
        """The ranges' width inside the window (Hz) and its share of their width."""
        inside = width = 0.0
        for low, high, frame in ranges:
            if frame not in self.spans:
                frequencies = self.window.in_frame(frame)
                self.spans[frame] = (float(frequencies.min()), float(frequencies.max()))
            start, stop = self.spans[frame]
            inside += max(0.0, min(high, stop) - max(low, start))
            width += high - low
        return inside, inside / width if width > 0 else 0.0


def match_window(
    blocks: Mapping[tuple[str, int], ContBlock], window: WindowFrequencies
) -> tuple[tuple[str, int] | None, str]:
    """The cont.dat block of ``window``, or None and the reason.

    Among the blocks of the window's fields with at least ``MIN_INSIDE`` of
    their ranges' width inside its channels, the one with the most. Failing
    that, an ``ALL`` or ``NONE`` block without ranges: the only one whose
    pipeline window the other fields' ranges place over this window, or else
    the only one they place nowhere.
    """
    candidates = [key for key in blocks if key[0] in window.field_names]
    fields = ", ".join(sorted(window.field_names))
    if not candidates:
        return None, f"cont.dat has no block for field {fields}"
    overlap = _Overlap(window)
    best: tuple[str, int] | None = None
    most = 0.0
    for key in candidates:
        inside, share = overlap(blocks[key].ranges)
        if share >= MIN_INSIDE and inside > most:
            best, most = key, inside
    if best is not None:
        return best, ""
    return _whole_window_block(blocks, candidates, overlap, fields)


def _whole_window_block(
    blocks: Mapping[tuple[str, int], ContBlock],
    candidates: list[tuple[str, int]],
    overlap: _Overlap,
    fields: str,
) -> tuple[tuple[str, int] | None, str]:
    """``match_window``'s ``ALL`` or ``NONE`` block, placed by the other fields' ranges."""
    # Where each pipeline window lies, from the ranges any field gives for it.
    placed: dict[int, list[FrequencyRange]] = {}
    for (_, spw), block in blocks.items():
        if block.ranges:
            placed.setdefault(spw, []).extend(block.ranges)
    whole = [key for key in candidates if blocks[key].whole and not blocks[key].ranges]
    # Those placed over this window first, then those placed nowhere.
    over = [k for k in whole if k[1] in placed and overlap(placed[k[1]])[1] >= MIN_INSIDE]
    nowhere = [k for k in whole if k[1] not in placed]
    for group in (over, nowhere):
        if len(group) == 1:
            return group[0], ""
        if group:
            windows = ", ".join(str(key[1]) for key in group)
            return None, f"cont.dat windows {windows} of field {fields} could all be this one"
    return None, f"no cont.dat window of field {fields} overlaps this one"


def line_mask(block: ContBlock, window: WindowFrequencies) -> np.ndarray:
    """The window's pipeline line channels: those outside every continuum range."""
    import numpy as np

    if block.whole:
        # ALL: every channel is continuum; NONE: none is.
        return np.full(window.channels, block.whole == "NONE", dtype=bool)
    continuum = np.zeros(window.channels, dtype=bool)
    for low, high, frame in block.ranges:
        frequencies = window.in_frame(frame)
        continuum |= (frequencies >= low) & (frequencies <= high)
    return ~continuum


def _frame_factor(
    direction: tuple[float, float, str],
    time_s: float,
    frequency_hz: float,
    source: str,
    target: str,
) -> float:
    """Frequency in ``target`` over frequency in ``source``, at ALMA, ``time_s`` (MJD seconds,
    UTC) and ``direction`` (longitude rad, latitude rad, reference)."""
    from casacore.measures import measures

    me = measures()
    me.doframe(me.observatory("ALMA"))
    me.doframe(me.epoch("UTC", f"{time_s}s"))
    me.doframe(me.direction(direction[2], f"{direction[0]}rad", f"{direction[1]}rad"))
    converted = me.measure(me.frequency(source, f"{frequency_hz}Hz"), target)
    return float(converted["m0"]["value"]) / frequency_hz


def ms_window(ms_path: str | os.PathLike[str]) -> WindowFrequencies:
    """The fields and channel frequencies of a single-window MS (its first window).

    Frequencies are converted from the window's MEAS_FREQ_REF frame for the
    field with the most rows, at the median TIME of its rows.
    """
    import numpy as np
    from casacore.tables import table

    path = str(ms_path)
    with table(f"{path}/SPECTRAL_WINDOW", ack=False) as spw:
        frequencies = np.asarray(spw.getcell("CHAN_FREQ", 0), dtype=float)
        code = int(spw.getcell("MEAS_FREQ_REF", 0))
    source = _FREQUENCY_FRAMES[code] if 0 <= code < len(_FREQUENCY_FRAMES) else "TOPO"
    with table(path, ack=False) as ms:
        field_ids = np.asarray(ms.getcol("FIELD_ID"))
        main_field = int(np.bincount(field_ids).argmax())
        with ms.query(f"FIELD_ID == {main_field}") as rows:
            time_s = float(np.median(rows.getcol("TIME")))
    with table(f"{path}/FIELD", ack=False) as fields:
        names = frozenset(str(fields.getcell("NAME", int(i))) for i in np.unique(field_ids))
        longitude, latitude = np.asarray(fields.getcell("PHASE_DIR", main_field)).reshape(-1)[:2]
        try:
            reference = str(fields.getcolkeyword("PHASE_DIR", "MEASINFO")["Ref"])
        except Exception:  # noqa: BLE001 - no MEASINFO: ALMA's default
            reference = "J2000"
    direction = (float(longitude), float(latitude), reference)
    centre = float(np.mean(frequencies))
    converted: dict[str, np.ndarray] = {source: frequencies}

    def in_frame(frame: str) -> np.ndarray:
        if frame not in converted:
            factor = _frame_factor(direction, time_s, centre, source, frame)
            converted[frame] = frequencies * factor
        return converted[frame]

    return WindowFrequencies(field_names=names, channels=int(frequencies.size), in_frame=in_frame)


def read_cont_dat(member_folder: str | os.PathLike[str]) -> tuple[str | None, str]:
    """The cont.dat text in a member folder's ``calibration/*auxproducts.tgz``, and its tarball.

    Read from the tarball without extracting it; None and the reason when no
    tarball holds one.
    """
    calibration = Path(member_folder) / "calibration"
    tarballs = sorted(calibration.glob(AUXPRODUCTS_GLOB))
    if not tarballs:
        return None, f"no {AUXPRODUCTS_GLOB} in {Path(member_folder).name}/calibration"
    for tarball in tarballs:
        with tarfile.open(tarball) as tar:
            for member in tar:
                if member.isfile() and Path(member.name).name == CONT_DAT_NAME:
                    handle = tar.extractfile(member)
                    if handle is not None:
                        return handle.read().decode(errors="replace"), tarball.name
    return None, f"no {CONT_DAT_NAME} in {', '.join(t.name for t in tarballs)}"


def find_cont_dat(
    source_ms: str | os.PathLike[str], ous_index: str | os.PathLike[str] | None = None
) -> tuple[str | None, str]:
    """The cont.dat text of the member ``source_ms``'s execution block belongs to.

    The member folder is the EB's ``member_folder`` in ``ous_index`` (default:
    the ``ous_index.csv`` nearest above the MS), relative to the index's
    folder. Returns the text and its tarball's name, or None and the reason.
    """
    from almasim.services.archive.delivery import read_ous_index
    from almasim.services.imaging.provenance import eb_uid_from_ms, find_ous_index

    eb = eb_uid_from_ms(source_ms)
    if eb is None:
        return None, f"{Path(source_ms).name} names no execution block"
    index = Path(ous_index) if ous_index else find_ous_index(source_ms)
    if index is None or not index.is_file():
        return None, f"no ous_index.csv for {Path(source_ms).name}"
    row = read_ous_index(index).get(eb)
    if not row or not row.get("member_folder"):
        return None, f"{eb} has no member folder in {index.name}"
    member = index.parent / row["member_folder"]
    try:
        return read_cont_dat(member)
    except (OSError, tarfile.TarError) as exc:
        return None, f"could not read the auxproducts tarball of {member.name}: {exc}"


def pipeline_line_channels(
    ms_path: str | os.PathLike[str],
    source_ms: str | os.PathLike[str],
    ous_index: str | os.PathLike[str] | None = None,
) -> tuple[np.ndarray | None, str]:
    """The pipeline's line channels of single-window MS ``ms_path``, and what gave them.

    ``source_ms`` is the MS whose execution block names the member (the one
    the window was extracted from). Returns the mask (True: pipeline line)
    and a note naming the tarball, field and pipeline window, or None and
    why cont.dat is not used.
    """
    text, where = find_cont_dat(source_ms, ous_index)
    if text is None:
        return None, where
    blocks = parse_cont_dat(text)
    if not blocks:
        return None, f"{CONT_DAT_NAME} in {where} has no window"
    try:
        window = ms_window(ms_path)
        key, reason = match_window(blocks, window)
        if key is None:
            return None, reason
        mask = line_mask(blocks[key], window)
    except Exception as exc:  # noqa: BLE001 - a failed match leaves the fit as it is
        return None, f"could not match {CONT_DAT_NAME} to the window: {exc}"
    return mask, f"{where}: field {key[0]}, pipeline window {key[1]} ({blocks[key].describe()})"


def without_pipeline_lines(
    line_free: np.ndarray, pipeline_line: np.ndarray | None, note: str
) -> tuple[np.ndarray, dict[str, object]]:
    """``line_free`` without the pipeline's line channels (``pipeline_line``, None when
    cont.dat is not used, ``note`` saying why), and what that did: ``cont_dat`` (the
    cont.dat window, or ``not used: <reason>``) and ``cont_dat_line_channels`` (the
    line channels it added)."""
    if pipeline_line is not None and pipeline_line.size != line_free.size:
        note = f"cont.dat gives {pipeline_line.size} channels, the window has {line_free.size}"
        pipeline_line = None
    if pipeline_line is None:
        return line_free, {"cont_dat": f"not used: {note}", "cont_dat_line_channels": 0}
    added = int((line_free & pipeline_line).sum())
    return line_free & ~pipeline_line, {"cont_dat": note, "cont_dat_line_channels": added}


def cont_dat_log(result: Mapping[str, object]) -> str:
    """What cont.dat did, for the task log, from a result carrying ``cont_dat``."""
    note = str(result["cont_dat"])
    if note.startswith("not used:"):
        return f"cont.dat {note}"
    return f"cont.dat ({note}) adds {result['cont_dat_line_channels']} line channels"
