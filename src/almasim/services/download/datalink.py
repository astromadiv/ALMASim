"""Resolve member OUS UIDs to downloadable products through ALMA DataLink. (MADIV-28 T6.9)"""

from __future__ import annotations

import io
import logging
import re
import tarfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
from urllib.parse import urlsplit, urlunsplit
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
        i_desc = col_idx("description")

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
            description = (
                (tds[i_desc].text or "").strip() if i_desc >= 0 and i_desc < len(tds) else ""
            )

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
                    path=_listing_folder(description, filename),
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


def _fetch_with_fallback(uid: str, mirror: str) -> list[DataProduct] | None:
    """DataLink rows for one ID from the first mirror that answers; None when all fail."""
    last_error: Exception | None = None
    for base in [mirror] + [candidate for candidate in _DATALINK_MIRRORS if candidate != mirror]:
        try:
            return _fetch_datalink(uid, base)
        except Exception as exc:  # noqa: BLE001 - the next mirror is tried  # pragma: no cover
            last_error = exc
            logger.warning("DataLink fetch failed for %s on %s: %s", uid, base, exc)
    logger.error("All DataLink mirrors failed for %s: %s", uid, last_error)
    return None


def resolve_products(
    member_ous_uids: Iterable[str],
    mirror: str = _DATALINK_MIRRORS[0],
) -> list[DataProduct]:
    """Resolve downloadable data products through ALMA DataLink.

    A member's tars and README land in the download destination itself.
    """
    products: list[DataProduct] = []
    for uid in member_ous_uids:
        rows = _fetch_with_fallback(str(uid), mirror) or []
        products.extend(replace(row, path="") for row in rows)
    return products


# The member's pipeline products tar (#this) and its auxiliary tar list their files
# through a DataLink of their own; the raw ASDM tars are downloaded whole.
_EXPANDABLE_SEMANTICS = {"#this", "#auxiliary"}


# The member folders of an ALMA delivery; a tar's listing names each file's in its
# description ("Download dataset of type: calibration, ..."), except the pipeline log's.
_DELIVERY_FOLDERS = {"product", "calibration", "script", "qa", "log"}
# Enough of a tar's start for its first header, a GNU long name included.
_TAR_HEAD_BYTES = 16 * 1024


def _listing_folder(description: str, filename: str) -> str:
    """The member folder a tar's listing puts a file in; empty for the member folder itself."""
    match = re.search(r"type: (\w+)", description)
    if match and match.group(1) in _DELIVERY_FOLDERS:
        return match.group(1)
    return "log" if filename.endswith(".log") else ""


def _member_dir_from_tar_head(head: bytes) -> str | None:
    """``<project>/science_goal.*/group.*/member.*`` from the first path in a tar's start."""
    try:
        with tarfile.open(fileobj=io.BytesIO(head), mode="r|") as archive:
            first = archive.next()
    except (tarfile.TarError, EOFError):
        return None
    if first is None:
        return None
    parts = PurePosixPath(first.name).parts
    for i, part in enumerate(parts):
        if part.startswith("member."):
            member_dir = parts[: i + 1]
            if ".." in member_dir or first.name.startswith("/"):
                return None
            return "/".join(member_dir)
    return None


def _tar_head(url: str) -> bytes:
    """The first bytes of a tar, from the first mirror that serves it (DataLink ignores Range)."""
    parts = urlsplit(url)
    hosts = [parts.netloc] + [
        host for host in (urlsplit(m).netloc for m in _DATALINK_MIRRORS) if host != parts.netloc
    ]
    for host in hosts:
        try:
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                with client.stream("GET", urlunsplit(parts._replace(netloc=host))) as response:
                    response.raise_for_status()
                    head = b""
                    for chunk in response.iter_bytes():
                        head += chunk
                        if len(head) >= _TAR_HEAD_BYTES:
                            break
                    return head
        except httpx.HTTPError as exc:
            logger.warning("Could not read the start of %s on %s: %s", url, host, exc)
    return b""


def _member_dir(product: DataProduct) -> str:
    """The member's folder in the delivery tree, read from its tar; named for its uid if unread."""
    member_dir = _member_dir_from_tar_head(_tar_head(product.access_url))
    if member_dir:
        return member_dir
    logger.warning("No delivery path in %s; its files go in its member folder", product.filename)
    return "member." + product.uid.replace("uid://", "uid___").replace("/", "_")


def is_expandable(product: DataProduct) -> bool:
    """Whether the archive lists the files inside this product's tar."""
    return product.semantics in _EXPANDABLE_SEMANTICS and product.filename.endswith(".tar")


def _archive_files(product: DataProduct, mirror: str, member_dir: str | None) -> list[DataProduct]:
    """The files inside one tar, typed like the tar; the tar itself if the listing fails.

    Each listed row's ``path`` is its folder in the member (``product``, ``qa`` ...); it
    becomes ``<member_dir>/<folder>``, or empty when ``member_dir`` is None (flat).
    """
    rows = _fetch_with_fallback(product.filename, mirror)
    if not rows:
        logger.warning("No file listing for %s; it is kept whole", product.filename)
        return [product]
    return [
        replace(
            row,
            uid=product.uid,
            product_type=product.product_type,
            path="/".join(p for p in (member_dir, row.path) if p) if member_dir else "",
        )
        for row in rows
    ]


LAYOUTS = ("delivery", "flat")


def expand_archives(
    products: Iterable[DataProduct],
    mirror: str = _DATALINK_MIRRORS[0],
    max_workers: int = 4,
    layout: str = "delivery",
) -> list[DataProduct]:
    """Replace each product and auxiliary tar by the files inside it, one row per file.

    The files keep the tar's ``product_type``, so ``--product-filter`` selects them as
    it selects the tar. Raw ASDM tars and other files are kept as they are. A file
    listed twice (the member README is in its auxiliary tar too) is kept once.

    ``layout="delivery"`` gives each file the folder the tar puts it in,
    ``<project>/science_goal.*/group.*/member.*/<product|calibration|qa|script|log>``,
    read from the first header of one tar per member, and puts the member README in
    the member folder. ``layout="flat"`` leaves every file in the destination itself.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, not {layout!r}")
    rows = list(products)
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        member_dirs: dict[str, str] = {}
        if layout == "delivery":
            firsts = {p.uid: p for p in reversed(rows) if is_expandable(p)}
            member_dirs = dict(zip(firsts, pool.map(_member_dir, firsts.values()), strict=True))
        listings = list(
            pool.map(
                lambda product: (
                    _archive_files(product, mirror, member_dirs.get(product.uid))
                    if is_expandable(product)
                    else [
                        replace(product, path=member_dirs[product.uid])
                        if product.semantics == "#documentation" and product.uid in member_dirs
                        else product
                    ]
                ),
                rows,
            )
        )
    expanded: list[DataProduct] = []
    seen: set[str] = set()
    for listing in listings:
        for row in listing:
            if row.filename not in seen:
                seen.add(row.filename)
                expanded.append(row)
    return expanded


def filter_products(products: list[DataProduct], product_filter: str = "all") -> list[DataProduct]:
    """Return products matching the requested product type."""
    if product_filter == "all":
        return list(products)
    return [product for product in products if product.product_type == product_filter]


def select_files(
    products: Iterable[DataProduct],
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> list[DataProduct]:
    """Products whose filename matches an ``include`` glob (any, when none) and no ``exclude`` glob.

    Globs are shell patterns (``*``, ``?``, ``[seq]``, ``[!seq]``), matched case-sensitively
    against the whole filename, for example ``*.pbcor.fits``.
    """
    include, exclude = list(include), list(exclude)
    return [
        product
        for product in products
        if (not include or any(fnmatchcase(product.filename, p) for p in include))
        and not any(fnmatchcase(product.filename, p) for p in exclude)
    ]


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
