"""Provenance in the FITS headers of the imaging products.

Every FITS an imaging task writes says, with the keywords of ALMA's own
pipeline images, which project (``PROPCODE``), science goal (``SGOAL``), group
(``GROUP``) and member ObsUnitSet (``MEMBER``) its data belong to, and its
spectral window (``SPW``) and field (``FIELD``); then the execution blocks
(``NEB``, ``EBUID01`` ...) and MeasurementSets (``NMS``, ``MSNAM01`` ...) it
was made from and the ALMASim version (``ALMASVER``). Its ``HISTORY`` lists
the task's steps in the order they ran: each WSClean command, with the task
folder's paths written relative to it, and the continuum subtraction from the
visibilities. The steps that make an image from another one (primary-beam
correction, continuum subtraction, the quality maps) copy these cards and add
their own ``HISTORY``.

The OUS UIDs come from the delivery tree's ``ous_index.csv``
(:func:`almasim.services.archive.delivery.organize_delivery`), looked up by
the execution block in the MS's name.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from astropy.io import fits

__all__ = [
    "VERSION_KEY",
    "Provenance",
    "copy_provenance",
    "eb_uid_from_ms",
    "find_ous_index",
    "provenance_header",
    "stamp_fits",
]

VERSION_KEY = "ALMASVER"
_EB_UID = re.compile(r"uid___[A-Za-z0-9]+_X[0-9A-Za-z]+_X[0-9A-Za-z]+")
_COUNTED = ("EBUID", "MSNAM")
_KEYS = ("PROPCODE", "SGOAL", "GROUP", "MEMBER", "SPW", "FIELD", "NEB", "NMS", VERSION_KEY)


def eb_uid_from_ms(ms_path: str | os.PathLike[str]) -> str | None:
    """The execution block of an ALMA MS named ``<eb>.ms[...]``: ``uid___A002_X1_X2``."""
    match = _EB_UID.match(Path(ms_path).name)
    return match.group(0) if match else None


@dataclass(frozen=True)
class Provenance:
    """What an image was made from: its OUSs, execution blocks, MSs, window and field."""

    project_code: str = ""
    science_goal_uid: str = ""
    group_ous_uid: str = ""
    member_ous_uid: str = ""
    eb_uids: tuple[str, ...] = ()
    ms_names: tuple[str, ...] = ()
    spw: int | None = None
    field: str = ""

    @classmethod
    def for_ms(
        cls,
        ms_paths: Sequence[str | os.PathLike[str]],
        index: Mapping[str, Mapping[str, str]] | None = None,
        *,
        spw: int | None = None,
        field: str = "",
    ) -> Provenance:
        """The provenance of an image of ``ms_paths``, its OUSs from ``index`` (eb_uid -> row).

        The first MS whose execution block ``index`` lists gives the OUSs; the
        member's other execution blocks share them.
        """
        ebs = tuple(eb for eb in (eb_uid_from_ms(p) for p in ms_paths) if eb)
        row: Mapping[str, str] = next((index[eb] for eb in ebs if index and eb in index), {})
        return cls(
            project_code=row.get("project_code", ""),
            science_goal_uid=row.get("science_goal_uid", ""),
            group_ous_uid=row.get("group_ous_uid", ""),
            member_ous_uid=row.get("member_ous_uid", ""),
            eb_uids=ebs,
            ms_names=tuple(Path(p).name for p in ms_paths),
            spw=spw,
            field=field,
        )

    def cards(self) -> list[tuple[str, object, str]]:
        """The header cards: ``(keyword, value, comment)``, empty values left out."""
        from almasim import __version__

        cards: list[tuple[str, object, str]] = [
            ("PROPCODE", self.project_code, "ALMA project code"),
            ("SGOAL", self.science_goal_uid, "science goal OUS"),
            ("GROUP", self.group_ous_uid, "group OUS"),
            ("MEMBER", self.member_ous_uid, "member OUS"),
            ("SPW", "" if self.spw is None else str(self.spw), "spectral window in the MS"),
            ("FIELD", self.field, "field imaged"),
        ]
        cards = [card for card in cards if card[1] != ""]
        cards.append(("NEB", len(self.eb_uids), "execution blocks imaged"))
        cards += [(f"EBUID{n:02d}", eb, "execution block") for n, eb in enumerate(self.eb_uids, 1)]
        cards.append(("NMS", len(self.ms_names), "MeasurementSets imaged"))
        cards += [
            (f"MSNAM{n:02d}", name, "MeasurementSet") for n, name in enumerate(self.ms_names, 1)
        ]
        cards.append((VERSION_KEY, __version__, "ALMASim version"))
        return cards


def provenance_header(provenance: Provenance, history: Iterable[str] = ()) -> fits.Header:
    """``provenance``'s cards and one ``HISTORY`` entry per step in ``history``."""
    header = fits.Header()
    for key, value, comment in provenance.cards():
        header[key] = (value, comment)
    for step in history:
        header.add_history(f"ALMASim: {step}")
    return header


def _is_provenance(key: str) -> bool:
    return key in _KEYS or any(
        key.startswith(prefix) and key[len(prefix) :].isdigit() for prefix in _COUNTED
    )


def copy_provenance(source: fits.Header, target: fits.Header) -> fits.Header:
    """Add ``source``'s provenance cards and ``HISTORY`` to ``target``; ``target``."""
    for card in source.cards:
        if _is_provenance(card.keyword):
            target[card.keyword] = (card.value, card.comment)
    for entry in source.get("HISTORY", []):
        target.add_history(str(entry))
    return target


def stamp_fits(path: str | os.PathLike[str], provenance: fits.Header) -> bool:
    """Add ``provenance`` to the primary header of the FITS at ``path``; False when it has one.

    A file that already carries ``ALMASVER`` (a cube stacked with it, or an
    image made from one) is left alone.
    """
    with fits.open(path, mode="update", memmap=True) as hdus:
        header = hdus[0].header
        if VERSION_KEY in header:
            return False
        copy_provenance(provenance, header)
    return True


def find_ous_index(ms_path: str | os.PathLike[str]) -> Path | None:
    """The ``ous_index.csv`` in the MS's folder or the nearest folder above it.

    ``products download`` keeps the MSs under the destination's ``archive_ms``
    and the index at the destination, so an MS it made finds its index.
    """
    from almasim.services.archive.delivery import INDEX_FILENAME

    folder = Path(ms_path).absolute().parent
    for parent in (folder, *folder.parents):
        if (parent / INDEX_FILENAME).is_file():
            return parent / INDEX_FILENAME
    return None
