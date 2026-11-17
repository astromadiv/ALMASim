"""The ALMA delivery tree, where every MeasurementSet and image of a member lives, and its index.

An ALMA delivery is a tree of member ObsUnitSets:
``<project>/science_goal.<uid>/group.<uid>/member.<uid>/``, with the member's
``product/``, ``calibration/``, ``qa/``, ``script/``, ``log/`` and ``raw/``
folders. Every execution block of a member has a ``<eb>.qa0_report.pdf`` in its
``qa/`` folder, a delivered calibration an ``<eb>.ms.calapply.txt`` in its
``calibration/`` folder, and a downloaded ASDM is ``raw/<eb>.asdm.sdm``, so the
tree says which member each EB belongs to.

The archive stages write into the tree when their output folder is its root:
the raw MS next to its ASDM, ``raw/<eb>.ms``, the calibrated MS in
``calibrated/<eb>.ms.split.cal``, where ALMA's scriptForPI puts it, and the
images in ``images/`` (:func:`eb_member_folder` finds the folder).
:func:`write_ous_index` writes ``ous_index.csv`` at the root of the tree: one row
per execution block, with its project, science goal, group and member, whether
ALMA calibrated it (it has a calapply file), and its MSs.

:func:`member_status` reads the index member by member: a member is complete
when every execution block ALMA calibrated for it has a calibrated MS, and only
a complete member is imaged as a whole.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .qa0 import index_member_directories

logger = logging.getLogger(__name__)

__all__ = [
    "CALIBRATED_FOLDER",
    "IMAGES_FOLDER",
    "RAW_FOLDER",
    "INDEX_FILENAME",
    "DeliveryIndex",
    "MemberFolder",
    "MemberStatus",
    "archive_ms_roots",
    "eb_member_folder",
    "folder_uid",
    "is_delivery_root",
    "member_folders",
    "member_status",
    "read_ous_index",
    "write_ous_index",
]

CALIBRATED_FOLDER = "calibrated"
RAW_FOLDER = "raw"
IMAGES_FOLDER = "images"
_RESCAN_SECONDS = 60.0
INDEX_FILENAME = "ous_index.csv"
_INDEX_COLUMNS = [
    "eb_uid",
    "project_code",
    "science_goal_uid",
    "group_ous_uid",
    "member_ous_uid",
    "member_folder",
    "alma_calibrated",
    "raw_ms",
    "calibrated_ms",
]


@dataclass(frozen=True)
class MemberFolder:
    """One member ObsUnitSet's folder in a delivery tree."""

    path: Path
    project_code: str
    science_goal_uid: str
    group_ous_uid: str
    member_ous_uid: str


@dataclass
class DeliveryIndex:
    """What :func:`write_ous_index` wrote."""

    index_path: Path
    execution_blocks: int
    calibrated: int
    """Execution blocks with a calibrated MS."""
    without_member: list[str]
    """Calibrated MSs outside the tree whose execution block no member folder lists."""


@dataclass(frozen=True)
class MemberStatus:
    """One member's execution blocks, as an ``ous_index.csv`` lists them."""

    member_ous_uid: str
    member_folder: str
    needed: tuple[str, ...]
    """The EBs its image is made from: those ALMA calibrated (all of them when
    ALMA calibrated none)."""
    calibrated_ms: tuple[str, ...]
    """The calibrated MSs of ``needed`` that exist, in the order of ``needed``."""
    missing: tuple[str, ...]
    """The EBs of ``needed`` without a calibrated MS."""

    @property
    def complete(self) -> bool:
        return bool(self.needed) and not self.missing


def folder_uid(folder_name: str) -> str:
    """``member.uid___A001_X1_X2`` -> ``uid://A001/X1/X2``."""
    uid = folder_name.split(".", 1)[1]
    return uid.replace("uid___", "uid://", 1).replace("_", "/")


def member_folders(root: str | os.PathLike[str]) -> dict[str, MemberFolder]:
    """Each execution block of the tree under ``root`` and its member folder.

    An EB counts as the member's when its ``qa/`` folder has the EB's QA0 report,
    its ``calibration/`` folder the EB's calapply file or its ``raw/`` folder the
    EB's ASDM.
    """
    members: dict[str, MemberFolder] = {}
    for path in sorted(index_member_directories(Path(root)).values()):
        group, goal, project = path.parent, path.parent.parent, path.parent.parent.parent
        member = MemberFolder(
            path=path,
            project_code=project.name,
            science_goal_uid=folder_uid(goal.name),
            group_ous_uid=folder_uid(group.name),
            member_ous_uid=folder_uid(path.name),
        )
        ebs = {p.name.removesuffix(".qa0_report.pdf") for p in path.glob("qa/*.qa0_report.pdf")}
        ebs |= {
            p.name.removesuffix(".ms.calapply.txt")
            for p in path.glob("calibration/*.ms.calapply.txt")
        }
        ebs |= {p.name.removesuffix(".asdm.sdm") for p in path.glob(f"{RAW_FOLDER}/*.asdm.sdm")}
        for eb in ebs:
            if eb in members and members[eb].path != path:
                logger.warning("%s is listed by %s and %s", eb, members[eb].path, path)
            members.setdefault(eb, member)
    return members


def read_ous_index(path: str | os.PathLike[str]) -> dict[str, dict[str, str]]:
    """An ``ous_index.csv`` as ``{eb_uid: row}``."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {row["eb_uid"]: row for row in csv.DictReader(handle)}


def member_status(index: dict[str, dict[str, str]]) -> dict[str, MemberStatus]:
    """Each member of an index (``read_ous_index``), by member OUS UID.

    An EB is needed when ALMA calibrated it (``alma_calibrated``); an EB ALMA
    did not calibrate, one that failed QA0 say, is left out of the member's image.
    """
    rows_by_member: dict[str, list[dict[str, str]]] = {}
    for eb in sorted(index):
        rows_by_member.setdefault(index[eb]["member_ous_uid"], []).append(index[eb])
    status: dict[str, MemberStatus] = {}
    for member, rows in rows_by_member.items():
        needed = [r for r in rows if r.get("alma_calibrated") == "yes"] or rows
        status[member] = MemberStatus(
            member_ous_uid=member,
            member_folder=rows[0]["member_folder"],
            needed=tuple(r["eb_uid"] for r in needed),
            calibrated_ms=tuple(r["calibrated_ms"] for r in needed if r["calibrated_ms"]),
            missing=tuple(r["eb_uid"] for r in needed if not r["calibrated_ms"]),
        )
    return status


def is_delivery_root(root: str | os.PathLike[str]) -> bool:
    """Whether ``root`` holds a delivery tree (``<project>/science_goal.*`` folders)."""
    return any(Path(root).expanduser().glob("*/science_goal.*"))


_members_seen: dict[Path, tuple[float, dict[str, Path]]] = {}


def eb_member_folder(root: str | os.PathLike[str], eb: str) -> Path | None:
    """The member folder of execution block ``eb`` in the delivery tree at ``root``.

    None when ``root`` holds no delivery tree or no member lists ``eb``. The tree
    is read once per process; an EB not found makes it read again, at most once a
    minute, so members extracted meanwhile are found.
    """
    root = Path(root).expanduser().resolve()
    seen = _members_seen.get(root)
    if seen is None or (eb not in seen[1] and time.monotonic() - seen[0] > _RESCAN_SECONDS):
        members = (
            {name: member.path for name, member in member_folders(root).items()}
            if is_delivery_root(root)
            else {}
        )
        seen = _members_seen[root] = (time.monotonic(), members)
    return seen[1].get(eb)


def archive_ms_roots(
    delivery_root: str | os.PathLike[str], archive_output_root: str | os.PathLike[str] | None
) -> tuple[Path, Path]:
    """The raw and calibrated MS roots of a download into ``delivery_root``.

    Without ``archive_output_root`` both are the tree itself, so each MS goes into
    its member folder; with it, ``<archive_output_root>/raw_ms`` and ``calibrated_ms``.
    """
    if archive_output_root is None:
        root = Path(delivery_root).expanduser().resolve()
        return root, root
    archive = Path(archive_output_root).expanduser().resolve()
    return archive / "raw_ms", archive / "calibrated_ms"


def _ms_by_eb(ms_root: Path | None, suffix: str) -> dict[str, Path]:
    """The MSs named ``<eb><suffix>`` in ``ms_root``, by execution block."""
    if ms_root is None or not ms_root.is_dir():
        return {}
    return {
        path.name.removesuffix(suffix): path
        for path in sorted(ms_root.glob(f"*{suffix}"))
        if path.is_dir()
    }


def write_ous_index(
    delivery_root: str | os.PathLike[str],
    calibrated_ms_root: str | os.PathLike[str] | None = None,
    raw_ms_root: str | os.PathLike[str] | None = None,
    *,
    index_path: str | os.PathLike[str] | None = None,
    echo: Callable[[str], object] = logger.info,
) -> DeliveryIndex:
    """Write the tree's ``ous_index.csv``: every execution block, its member and its MSs.

    An EB's MSs are those in its member folder (``raw/<eb>.ms``,
    ``calibrated/<eb>.ms.split.cal``) or else those in ``raw_ms_root`` and
    ``calibrated_ms_root``, for stages given a folder of their own. The index is
    rewritten whole; ``index_path`` defaults to ``<delivery_root>/ous_index.csv``.
    """
    root = Path(delivery_root)
    members = member_folders(root)
    calibrated = _ms_by_eb(
        Path(calibrated_ms_root) if calibrated_ms_root else None, ".ms.split.cal"
    )
    raw = _ms_by_eb(Path(raw_ms_root) if raw_ms_root else None, ".ms")
    without_member = [str(ms) for eb, ms in calibrated.items() if eb not in members]
    for eb, member in members.items():
        for found, folder, suffix in (
            (calibrated, CALIBRATED_FOLDER, ".ms.split.cal"),
            (raw, RAW_FOLDER, ".ms"),
        ):
            in_tree = member.path / folder / f"{eb}{suffix}"
            if in_tree.is_dir():
                found[eb] = in_tree

    index = Path(index_path) if index_path else root / INDEX_FILENAME
    index.parent.mkdir(parents=True, exist_ok=True)
    # One temporary file per process: index writes may overlap on one tree.
    temporary = index.with_name(f"{index.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_INDEX_COLUMNS)
        for eb, member in sorted(members.items(), key=lambda item: (str(item[1].path), item[0])):
            writer.writerow(
                [
                    eb,
                    member.project_code,
                    member.science_goal_uid,
                    member.group_ous_uid,
                    member.member_ous_uid,
                    os.path.relpath(member.path, index.parent),
                    "yes"
                    if (member.path / "calibration" / f"{eb}.ms.calapply.txt").is_file()
                    else "no",
                    str(raw[eb]) if eb in raw else "",
                    str(calibrated[eb]) if eb in calibrated else "",
                ]
            )
    os.replace(temporary, index)

    with_calibrated = sum(1 for eb in members if eb in calibrated)
    echo(
        f"Delivery tree {root}: {len(members)} execution blocks in "
        f"{len({m.path for m in members.values()})} members, {with_calibrated} calibrated; "
        f"index {index}"
    )
    if without_member:
        echo(
            f"{len(without_member)} calibrated MS(s) have no member folder under {root}: "
            + ", ".join(Path(p).name for p in without_member[:5])
            + (" ..." if len(without_member) > 5 else "")
        )
    return DeliveryIndex(index, len(members), with_calibrated, without_member)
