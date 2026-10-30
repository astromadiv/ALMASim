"""Read ALMA QA0 reports.

Every execution block ships a ``<eb_uid>.qa0_report.pdf`` in its member OUS
``qa/`` directory. Its ``QA0 Status`` field decides whether ALMA delivers
calibration for that EB at all: ``Pass`` blocks always carry calibration
products, ``SemiPass`` blocks never do. ALMA still ships the raw ASDM for a
SemiPass block, so resolving and downloading without consulting this field
pulls execution blocks that can never be calibrated.

Do not use ``ExecBlock Status`` for this decision -- it is a separate field and
does not discriminate: a SemiPass block frequently carries
``ExecBlock Status = SUCCESS``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "QA0FilterResult",
    "QA0Report",
    "calibration_is_delivered",
    "eb_uid_from_raw_filename",
    "entity_id_to_filename",
    "filter_products_by_qa0",
    "find_qa0_report",
    "index_member_directories",
    "parse_qa0_report_text",
    "qa0_report_path",
    "read_qa0_report",
]

# QA0 statuses for which ALMA delivers calibration products.
_DELIVERED_STATUSES = frozenset({"pass"})


@dataclass(frozen=True)
class QA0Report:
    """The fields of a QA0 report that decide whether an EB is usable, and the
    water vapour it was observed through (``pwv_mm``; None where the report has
    none, as for the 7 m array, whose antennas carry no water vapour
    radiometer)."""

    eb_uid: str
    qa0_status: str
    execblock_status: str
    execution_fraction: float | None
    comment: str
    pwv_mm: float | None = None

    @property
    def calibration_delivered(self) -> bool:
        return calibration_is_delivered(self.qa0_status)


def calibration_is_delivered(qa0_status: str) -> bool:
    """Return True when ``qa0_status`` is one ALMA delivers calibration for."""
    # Reports occasionally carry a trailing comma ("SemiPass,").
    return qa0_status.strip().rstrip(",").lower() in _DELIVERED_STATUSES


def qa0_report_path(member_dir: str | Path, eb_uid: str) -> Path:
    """Return the QA0 report path for ``eb_uid`` inside a member OUS directory."""
    return Path(member_dir) / "qa" / f"{eb_uid}.qa0_report.pdf"


def parse_qa0_report_text(text: str, eb_uid: str = "") -> QA0Report:
    """Parse the first page of a QA0 report.

    Kept separate from the PDF read so the field patterns can be tested without
    a binary fixture.
    """

    def field(pattern: str) -> str:
        # DOTALL: QA0 comments routinely wrap across several lines. It is safe
        # for the other patterns here, none of which use ".".
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    # "QA0 Status" and its value are split across lines in the delivered layout.
    fraction_text = field(r"Exec\. Fraction\s+([\d.]+)")
    pwv_text = field(r"PWV\s+([\d.]+)\s*mm")
    comment = field(r"QA0 comment\s+(.*?)(?:\nPage \d|\Z)")
    return QA0Report(
        eb_uid=eb_uid,
        qa0_status=field(r"QA0 Status\s*\n?\s*(\S+)"),
        execblock_status=field(r"ExecBlock Status\s+(\S+)"),
        execution_fraction=float(fraction_text) if fraction_text else None,
        comment=" ".join(comment.split()),
        pwv_mm=float(pwv_text) if pwv_text else None,
    )


def read_qa0_report(pdf_path: str | Path, eb_uid: str = "") -> QA0Report:
    """Read and parse the QA0 report at ``pdf_path``."""
    from pypdf import PdfReader

    path = Path(pdf_path)
    if not eb_uid:
        eb_uid = path.name.removesuffix(".qa0_report.pdf")
    text = PdfReader(str(path)).pages[0].extract_text()
    return parse_qa0_report_text(text, eb_uid)


# ``2023.1.00879.S_uid___A002_X11d61dd_X9e40.asdm.sdm.tar`` -> the EB uid.
_RAW_TAR_RE = re.compile(r"(?P<eb>uid___A002_[A-Za-z0-9_]+)\.asdm\.sdm\.tar$")


def entity_id_to_filename(entity_id: str) -> str:
    """``uid://A001/X378a/X149`` -> ``uid___A001_X378a_X149`` (ALMA's on-disk form)."""
    return entity_id.strip().replace("://", "___").replace("/", "_")


def eb_uid_from_raw_filename(filename: str) -> str | None:
    """Return the execution-block uid embedded in a raw ASDM tar name, or None."""
    match = _RAW_TAR_RE.search(Path(filename).name)
    return match.group("eb") if match else None


def index_member_directories(root: str | Path) -> dict[str, Path]:
    """Map ``member.uid___A001_...`` directory names to their paths below ``root``.

    ``root`` holds extracted ALMA deliveries in the standard layout
    ``<project>/science_goal.*/group.*/member.*``. One walk of the top four
    levels is cheap; a glob per execution block over NFS is not.
    """
    index: dict[str, Path] = {}
    for member_dir in Path(root).glob("*/science_goal.*/group.*/member.*"):
        if member_dir.is_dir():
            index.setdefault(member_dir.name, member_dir)
    return index


def find_qa0_report(
    root: str | Path,
    member_ous_uid: str,
    eb_uid: str,
    member_index: dict[str, Path] | None = None,
) -> Path | None:
    """Locate ``eb_uid``'s QA0 report inside its member's extracted delivery."""
    if member_index is None:
        member_index = index_member_directories(root)
    member_dir = member_index.get(f"member.{entity_id_to_filename(member_ous_uid)}")
    if member_dir is None:
        return None
    report = qa0_report_path(member_dir, eb_uid)
    return report if report.is_file() else None


@dataclass
class QA0FilterResult:
    """Outcome of :func:`filter_products_by_qa0`.

    ``kept`` is what to download. ``skipped`` are raw products whose QA0 report
    says SemiPass. ``unknown`` are raw products kept because no readable QA0
    report was found -- they are also in ``kept``.
    """

    kept: list[Any] = field(default_factory=list)
    skipped: list[Any] = field(default_factory=list)
    unknown: list[Any] = field(default_factory=list)

    @property
    def bytes_skipped(self) -> int:
        return sum(max(int(getattr(p, "content_length", 0) or 0), 0) for p in self.skipped)


def filter_products_by_qa0(
    products: Iterable[Any],
    report_root: str | Path,
    reader: Callable[..., QA0Report] | None = None,
) -> QA0FilterResult:
    """Drop raw ASDM products whose execution block is QA0 SemiPass.

    ALMA ships the raw ASDM of a SemiPass block but never its calibration, so
    downloading it only produces a MeasurementSet that can never be calibrated.
    The QA0 report lives in the member's *auxiliary* tar, so download and
    extract the auxiliary products under ``report_root`` first; a raw product
    without a readable report is kept (and listed as unknown), never dropped.

    Only products with ``product_type == "raw"`` are considered; everything
    else passes through untouched.
    """
    read = reader if reader is not None else read_qa0_report
    result = QA0FilterResult()
    member_index = index_member_directories(report_root)
    for product in products:
        if getattr(product, "product_type", "") != "raw":
            result.kept.append(product)
            continue
        eb_uid = eb_uid_from_raw_filename(getattr(product, "filename", ""))
        report = find_qa0_report(report_root, product.uid, eb_uid, member_index) if eb_uid else None
        if report is None:
            result.kept.append(product)
            result.unknown.append(product)
            continue
        try:
            verdict = read(report, eb_uid)
        except Exception:  # noqa: BLE001 - an unreadable report counts as unknown
            result.kept.append(product)
            result.unknown.append(product)
            continue
        if verdict.calibration_delivered:
            result.kept.append(product)
        else:
            result.skipped.append(product)
    return result
