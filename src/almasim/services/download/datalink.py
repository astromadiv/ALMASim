"""Resolve member OUS UIDs to downloadable products through ALMA DataLink. (MADIV-28 T6.9)"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from xml.etree import ElementTree

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import DataProduct

logger = logging.getLogger(__name__)


_DATALINK_MIRRORS = [
    "https://almascience.eso.org",
    "https://almascience.nrao.edu",
    "https://almascience.nao.ac.jp",
]
# ALMA recommends limiting concurrent downloads per TAP/DataLink mirror.
_VOT_NS = "http://www.ivoa.net/xml/VOTable/v1.3"
_SEMANTICS_MAP: dict[str, str] = {
    "#progenitor": "raw",
    "#derivation": "fits",
    "#auxiliary": "auxiliary",
    "alma#calibration": "calibration",
    "alma#calibratedmeasurementset": "calibration",
    "alma#pipeline_calibration": "calibration",
    "alma#script": "scripts",
    "alma#pipelinescript": "scripts",
    "alma#weblog": "weblog",
    "alma#qa": "qa_reports",
    "alma#qa2report": "qa_reports",
    "alma#imagecube": "cubes",
    "alma#spectralcube": "cubes",
    "alma#continuumimage": "continuum",
    "alma#continuum": "continuum",
    "alma#asdm": "raw",
    "alma#raw": "raw",
    "alma#auxiliary": "auxiliary",
    "alma#readme": "auxiliary",
}


def _classify_product(url: str, content_type: str, semantics: str = "") -> str:  # noqa: C901
    """Classify a product using VOTable semantics and URL patterns."""
    if semantics:
        sem_lower = semantics.lower().strip()
        if sem_lower in _SEMANTICS_MAP:
            return _SEMANTICS_MAP[sem_lower]
        for frag in ("#", "/ALMA#", "/alma#"):
            if frag in sem_lower:
                short = "alma#" + sem_lower.split(frag)[-1]
                if short in _SEMANTICS_MAP:
                    return _SEMANTICS_MAP[short]

    lower = url.lower()
    if "asdm" in lower or lower.endswith(".asdm.sdm.tar"):
        return "raw"
    if "weblog" in lower:
        return "weblog"
    if "qa" in lower and ("report" in lower or ".pdf" in lower):
        return "qa_reports"
    if "script" in lower:
        return "scripts"
    if "auxiliary" in lower or "readme" in lower or "licence" in lower:
        return "auxiliary"
    if lower.endswith(".fits"):
        return "fits"
    if "cube" in lower:
        return "cubes"
    if "continuum" in lower or "cont_" in lower:
        return "continuum"
    if lower.endswith(".tar") or lower.endswith(".tgz"):
        return "fits"
    return "other"


def _parse_votable_results(xml_bytes: bytes, uid: str) -> list[DataProduct]:
    """Parse a DataLink VOTable response into DataProduct rows."""
    products: list[DataProduct] = []
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        logger.warning("Failed to parse VOTable XML for uid=%s", uid)
        return products

    for ns_prefix in [f"{{{_VOT_NS}}}", ""]:
        table = root.find(f".//{ns_prefix}TABLEDATA")
        if table is None:
            continue

        resource = root.find(f".//{ns_prefix}RESOURCE")
        tbl = resource.find(f".//{ns_prefix}TABLE") if resource is not None else None
        fields = tbl.findall(f"{ns_prefix}FIELD") if tbl is not None else []
        col_names = [field.get("name", "").lower() for field in fields]

        def col_idx(name: str, col_names: list[str] = col_names) -> int:
            try:
                return col_names.index(name.lower())
            except ValueError:
                return -1

        i_url = col_idx("access_url")
        i_size = col_idx("content_length")
        i_type = col_idx("content_type")
        i_id = col_idx("ID")
        i_sem = col_idx("semantics")

        for tr in table.findall(f"{ns_prefix}TR"):
            tds = tr.findall(f"{ns_prefix}TD")
            if i_url < 0 or i_url >= len(tds):
                continue
            url = (tds[i_url].text or "").strip()
            if not url:
                continue

            size_str = tds[i_size].text if i_size >= 0 and i_size < len(tds) else "0"
            ctype = tds[i_type].text if i_type >= 0 and i_type < len(tds) else ""
            row_uid = tds[i_id].text if i_id >= 0 and i_id < len(tds) else uid
            semantics = (tds[i_sem].text or "").strip() if i_sem >= 0 and i_sem < len(tds) else ""

            try:
                content_length = int(size_str or 0)
            except (TypeError, ValueError):
                content_length = 0

            filename = url.rsplit("/", 1)[-1] if "/" in url else url
            products.append(
                DataProduct(
                    access_url=url,
                    uid=row_uid or uid,
                    filename=filename,
                    content_length=content_length,
                    content_type=ctype or "",
                    product_type=_classify_product(url, ctype or "", semantics),
                    semantics=semantics,
                )
            )
        break

    return products


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_datalink(uid: str, mirror_base: str = _DATALINK_MIRRORS[0]) -> list[DataProduct]:
    """Fetch DataLink results for a single member OUS UID."""
    url = f"{mirror_base}/datalink/sync?ID={uid}"
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    return _parse_votable_results(response.content, uid)


def resolve_products(
    member_ous_uids: Iterable[str],
    mirror: str = _DATALINK_MIRRORS[0],
) -> list[DataProduct]:
    """Resolve downloadable data products through ALMA DataLink."""
    products: list[DataProduct] = []
    member_uids = [str(uid) for uid in member_ous_uids]
    for uid in member_uids:
        last_error: Exception | None = None
        for base in [mirror] + [
            candidate for candidate in _DATALINK_MIRRORS if candidate != mirror
        ]:
            try:
                products.extend(_fetch_datalink(uid, base))
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - the next mirror is tried  # pragma: no cover
                last_error = exc
                logger.warning("DataLink fetch failed for %s on %s: %s", uid, base, exc)
        if last_error:
            logger.error("All DataLink mirrors failed for %s: %s", uid, last_error)
    return products


def filter_products(products: list[DataProduct], product_filter: str = "all") -> list[DataProduct]:
    """Return products matching the requested product type."""
    if product_filter == "all":
        return list(products)
    return [product for product in products if product.product_type == product_filter]


def count_executions(products: Iterable[DataProduct]) -> dict[str, int]:
    """Each member OUS's execution blocks: the distinct raw ASDMs among its products."""
    from ..archive.qa0 import eb_uid_from_raw_filename

    blocks: dict[str, set[str]] = {}
    for product in products:
        eb_uid = (
            eb_uid_from_raw_filename(product.filename) if product.product_type == "raw" else None
        )
        if eb_uid is not None:
            blocks.setdefault(product.uid, set()).add(eb_uid)
    return {uid: len(ebs) for uid, ebs in blocks.items()}
