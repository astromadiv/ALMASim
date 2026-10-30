"""Keep calibrated MeasurementSets in the ALMA delivery tree, and index it.

An ALMA delivery is a tree of member ObsUnitSets:
``<project>/science_goal.<uid>/group.<uid>/member.<uid>/``, with the member's
``product/``, ``calibration/``, ``qa/``, ``script/``, ``log/`` and ``raw/``
folders. Every execution block of a member has a ``<eb>.qa0_report.pdf`` in its
``qa/`` folder, and a delivered calibration an ``<eb>.ms.calapply.txt`` in its
``calibration/`` folder, so the tree says which member each EB belongs to.

:func:`organize_delivery` links each calibrated MS into its member's
``calibrated/`` folder, as ALMA's scriptForPI does, and writes
``ous_index.csv`` at the root of the tree: one row per execution block, with
its project, science goal, group and member, and its MSs. The MSs themselves
stay where the archive stages write them; the links are relative, so the tree
and the MS folders can move together.
"""

from __future__ import annotations

import csv
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .qa0 import index_member_directories

logger = logging.getLogger(__name__)

__all__ = [
    "CALIBRATED_FOLDER",
    "INDEX_FILENAME",
    "DeliveryOrganized",
    "MemberFolder",
    "folder_uid",
    "member_folders",
    "organize_delivery",
]

CALIBRATED_FOLDER = "calibrated"
INDEX_FILENAME = "ous_index.csv"
_INDEX_COLUMNS = [
    "eb_uid",
    "project_code",
    "science_goal_uid",
    "group_ous_uid",
    "member_ous_uid",
    "member_folder",
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
class DeliveryOrganized:
    """What :func:`organize_delivery` did."""

    index_path: Path
    execution_blocks: int
    linked: int
    without_member: list[str]
    """Calibrated MSs whose execution block no member folder lists."""


def folder_uid(folder_name: str) -> str:
    """``member.uid___A001_X1_X2`` -> ``uid://A001/X1/X2``."""
    uid = folder_name.split(".", 1)[1]
    return uid.replace("uid___", "uid://", 1).replace("_", "/")


def member_folders(root: str | os.PathLike[str]) -> dict[str, MemberFolder]:
    """Each execution block of the tree under ``root`` and its member folder.

    An EB counts as the member's when its ``qa/`` folder has the EB's QA0 report
    or its ``calibration/`` folder the EB's calapply file.
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
        for eb in ebs:
            if eb in members and members[eb].path != path:
                logger.warning("%s is listed by %s and %s", eb, members[eb].path, path)
            members.setdefault(eb, member)
    return members


def _link(target: Path, link: Path) -> bool:
    """Point ``link`` at ``target`` (relative); a real file or folder there is left alone."""
    relative = os.path.relpath(target, link.parent)
    if link.is_symlink():
        if os.readlink(link) == relative:
            return False
        link.unlink()
    elif link.exists():
        logger.warning("%s exists and is not a link; it is left as it is", link)
        return False
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(relative)
    return True


def _prune_dangling(member_paths: Iterable[Path]) -> None:
    """Drop the ``calibrated/`` links whose MS is gone."""
    for member in member_paths:
        for link in (member / CALIBRATED_FOLDER).glob("*"):
            if link.is_symlink() and not link.exists():
                link.unlink()


def _ms_by_eb(ms_root: Path | None, suffix: str) -> dict[str, Path]:
    """The MSs named ``<eb><suffix>`` in ``ms_root``, by execution block."""
    if ms_root is None or not ms_root.is_dir():
        return {}
    return {
        path.name.removesuffix(suffix): path
        for path in sorted(ms_root.glob(f"*{suffix}"))
        if path.is_dir()
    }


def organize_delivery(
    delivery_root: str | os.PathLike[str],
    calibrated_ms_root: str | os.PathLike[str] | None,
    raw_ms_root: str | os.PathLike[str] | None = None,
    *,
    index_path: str | os.PathLike[str] | None = None,
    echo: Callable[[str], object] = logger.info,
) -> DeliveryOrganized:
    """Link every calibrated MS into its member's ``calibrated/`` folder and write the index.

    The calibrated MSs are the folders in ``calibrated_ms_root`` (``<eb>.ms.split.cal``),
    the raw ones those in ``raw_ms_root`` (``<eb>.ms``). It is safe to run again: a
    link already right is kept, a link whose MS is gone is dropped, and the index is
    rewritten. ``index_path`` defaults to ``<delivery_root>/ous_index.csv``.
    """
    root = Path(delivery_root)
    members = member_folders(root)
    calibrated_root = Path(calibrated_ms_root) if calibrated_ms_root else None
    calibrated = _ms_by_eb(calibrated_root, ".ms.split.cal")
    raw = _ms_by_eb(Path(raw_ms_root) if raw_ms_root else None, ".ms")

    linked = 0
    without_member: list[str] = []
    for eb, ms in calibrated.items():
        member = members.get(eb)
        if member is None:
            without_member.append(str(ms))
            continue
        linked += _link(ms, member.path / CALIBRATED_FOLDER / ms.name)
    _prune_dangling({member.path for member in members.values()})

    index = Path(index_path) if index_path else root / INDEX_FILENAME
    index.parent.mkdir(parents=True, exist_ok=True)
    temporary = index.with_name(index.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
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
                    str(raw[eb]) if eb in raw else "",
                    str(calibrated[eb]) if eb in calibrated else "",
                ]
            )
    os.replace(temporary, index)

    echo(
        f"Delivery tree {root}: {len(members)} execution blocks in "
        f"{len({m.path for m in members.values()})} members; {linked} calibrated MS link(s) "
        f"made; index {index}"
    )
    if without_member:
        echo(
            f"{len(without_member)} calibrated MS(s) have no member folder under {root}: "
            + ", ".join(Path(p).name for p in without_member[:5])
            + (" ..." if len(without_member) > 5 else "")
        )
    return DeliveryOrganized(index, len(members), linked, without_member)
