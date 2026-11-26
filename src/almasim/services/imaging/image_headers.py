"""WSClean geometry from the FITS headers of the ALMA pipeline's tclean images.

``compute_imaging_parameters`` derives the WSClean geometry from the
MeasurementSet: the primary beam's field of view, the synthesized beam and the
window's channels. The pipeline's images already carry the geometry tclean
imaged with: the ``HISTORY`` cards repeat every tclean parameter (``imsize``,
``cell``, ``phasecenter``, ``specmode``, ``nchan``, ``weighting`` ...) and the
WCS cards describe the grid (``NAXIS``, ``CDELT``, ``CRVAL``, ``CRPIX``,
``BMAJ`` ...). This module reads those headers -- from the downloaded delivery
tree, or from the archive when they are not on disk -- and writes, for each
image, its header and a CSV of the parameter rows ``image-from-ms`` images
from, in the member's ``images/headers`` folder.

The headers are read a block at a time and stop at the ``END`` card, so a
multi-gigabyte cube costs a few tens of kilobytes. The ALMA data portal answers
``Range`` requests with 416, so a remote header is read with a stream that stops
at the same card.
"""

from __future__ import annotations

import ast
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import httpx
import pandas as pd
from astropy.io import fits
from tenacity import retry, stop_after_attempt, wait_exponential

from almasim.services.archive.delivery import IMAGES_FOLDER
from almasim.services.imaging.ms_geometry import compute_imaging_parameters
from almasim.services.imaging.provenance import eb_uid_from_ms

logger = logging.getLogger(__name__)

__all__ = [
    "IMAGE_COLUMNS",
    "MemberImage",
    "collect_headers",
    "image_geometry",
    "match_member_images",
    "parse_headers",
    "read_parameters",
    "parse_tclean_parameters",
    "read_fits_header",
    "read_fits_header_bytes",
    "read_remote_fits_header_bytes",
]

FITS_BLOCK_BYTES = 2880
FITS_CARD_BYTES = 80
END_CARD = b"END"
# A header is never this large; the cap turns a missing END card into an error
# instead of an endless read.
MAX_HEADER_BYTES = 64 * 1024 * 1024

# A member's headers and parameter CSVs: <member>/images/headers/.
HEADERS_FOLDER = "headers"
FAILED_FILENAME = "headers.failed.tsv"
# One parameter row is of one MS, window and field.
_ROW_KEY = ("filename", "spectral_window_id", "field_name")

# The parameter columns ``collect_headers`` adds to the MS-derived rows. The
# command builder prefers them over the MS-derived geometry when they are set.
IMAGE_COLUMNS = (
    "image_n_pixels",
    "image_cell_arcsec",
    "image_n_channels",
    "image_channel_width_hz",
    "image_reference_frequency",
    "phasecenter_ra_deg",
    "phasecenter_dec_deg",
    "ms_phasecenter_ra_deg",
    "ms_phasecenter_dec_deg",
    "image_weighting",
    "image_robust",
    "image_stokes",
    "image_specmode",
    "image_deconvolver",
    "image_niter",
    "image_threshold",
    "image_threshold_jy",
    "image_imsize",
    "image_cell",
    "image_phasecenter",
    "image_path",
    "image_filename",
    "image_kind",
    "header_path",
)

_IMAGE_KINDS = ("mfs", "cube")


def _block_has_end(block: bytes) -> bool:
    """Whether a 2880-byte FITS block holds the header's ``END`` card."""
    for start in range(0, len(block) - FITS_CARD_BYTES + 1, FITS_CARD_BYTES):
        if block[start : start + FITS_CARD_BYTES].rstrip() == END_CARD:
            return True
    return False


def read_fits_header_bytes(path: str | Path) -> bytes:
    """The FITS header blocks of ``path``, up to and including its ``END`` card."""
    source = Path(path)
    blocks: list[bytes] = []
    total = 0
    with source.open("rb") as handle:
        while True:
            block = handle.read(FITS_BLOCK_BYTES)
            if not block:
                raise ValueError(f"{source}: no END card in its FITS header")
            blocks.append(block)
            total += len(block)
            if _block_has_end(block):
                return b"".join(blocks)
            if total > MAX_HEADER_BYTES:
                raise ValueError(f"{source}: FITS header larger than {MAX_HEADER_BYTES} bytes")


def read_fits_header(path: str | Path) -> fits.Header:
    """The parsed primary header of a local FITS file, read only up to ``END``."""
    raw = read_fits_header_bytes(path)
    return fits.Header.fromstring(raw.decode("ascii", "replace"))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def read_remote_fits_header_bytes(url: str, *, timeout: float = 120.0) -> bytes:
    """The FITS header blocks of a remote file, read up to its ``END`` card.

    The ALMA portal rejects ``Range`` requests, so the file is streamed and the
    read stops as soon as a block holds the ``END`` card.
    """
    buffer = b""
    scanned = 0
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                buffer += chunk
                whole = len(buffer) // FITS_BLOCK_BYTES
                while scanned < whole:
                    block = buffer[scanned * FITS_BLOCK_BYTES : (scanned + 1) * FITS_BLOCK_BYTES]
                    scanned += 1
                    if _block_has_end(block):
                        return buffer[: scanned * FITS_BLOCK_BYTES]
                if len(buffer) > MAX_HEADER_BYTES:
                    raise ValueError(f"{url}: FITS header larger than {MAX_HEADER_BYTES} bytes")
    raise ValueError(f"{url}: no END card in its FITS header")


def read_remote_fits_header(url: str) -> fits.Header:
    """The parsed primary header of a remote FITS file."""
    raw = read_remote_fits_header_bytes(url)
    return fits.Header.fromstring(raw.decode("ascii", "replace"))


_TcleanLine = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_FRAME_NAMES = {"ICRS", "J2000", "B1950", "FK5", "GALACTIC"}


def _history_lines(header: fits.Header) -> list[str]:
    """The header's ``HISTORY`` entries with CASA's 72-column wraps rejoined."""
    lines: list[str] = []
    for entry in header.get("HISTORY", []):
        text = str(entry)
        if text.startswith(">"):
            # CASA wraps a long card at 72 characters and prefixes the next one.
            if lines:
                lines[-1] += text[1:]
            else:
                lines.append(text[1:])
        else:
            lines.append(text)
    return lines


def _literal(text: str) -> object:
    """A tclean value as a Python literal, or the raw string when it is not one."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_tclean_parameters(header: fits.Header) -> dict[str, object]:
    """The tclean parameters CASA wrote into the header's ``HISTORY`` cards.

    Each card after the ``SRCCODE`` banner is a ``key = value`` line, e.g.
    ``imsize = [6250, 6250]`` or ``cell = ['0.0047arcsec']``.
    """
    parameters: dict[str, object] = {}
    for line in _history_lines(header):
        match = _TcleanLine.match(line.strip())
        if match is None:
            continue
        key, raw = match.group(1), match.group(2).strip()
        if raw:
            parameters[key] = _literal(raw)
    return parameters


def _sexagesimal_deg(text: str, *, ra: bool) -> float:
    """``14:13:09.9473`` (RA) or ``-065.20.21.024`` (Dec) as decimal degrees.

    A declination with at most one dot is already in degrees. A right
    ascension that is not ``h:m:s`` could be hours or degrees and is refused.
    """
    sign = -1.0 if text.startswith("-") else 1.0
    body = text.lstrip("+-")
    if ":" in body:
        parts = body.split(":")
    elif ra:
        raise ValueError(f"cannot read a right ascension from {text!r}")
    elif body.count(".") <= 1:
        return sign * float(body)
    else:
        parts = body.split(".")
    if len(parts) < 3:
        raise ValueError(f"cannot read a sexagesimal angle from {text!r}")
    # ``-065.20.21.024``: the dots after the third field are the seconds' decimals.
    first, minutes = float(parts[0]), float(parts[1])
    seconds = float(".".join(parts[2:])) if ":" not in body else float(parts[2])
    value = first + minutes / 60.0 + seconds / 3600.0
    return sign * value * (15.0 if ra else 1.0)


def parse_phasecenter(text: str) -> tuple[float, float]:
    """``"ICRS 14:13:09.9473 -065.20.21.024"`` as ``(ra_deg, dec_deg)``."""
    fields = text.replace(",", " ").split()
    if len(fields) > 2 and fields[0].upper() in _FRAME_NAMES:
        fields = fields[1:]
    if len(fields) < 2:
        raise ValueError(f"cannot read a phase centre from {text!r}")
    return _sexagesimal_deg(fields[0], ra=True), _sexagesimal_deg(fields[1], ra=False)


def _phasecenter_deg(
    parameters: dict[str, object], header: fits.Header
) -> tuple[float | None, float | None]:
    """The tclean phase centre, falling back to the WCS reference coordinates."""
    raw = parameters.get("phasecenter")
    if isinstance(raw, str) and raw.strip().upper() == "TRACKFIELD":
        # A moving target (an ephemeris field): the image follows it, so it has
        # no fixed phase centre to shift the MS to.
        return None, None
    if isinstance(raw, str) and raw.strip():
        try:
            ra, dec = parse_phasecenter(raw)
        except ValueError:
            logger.warning("Cannot read the tclean phase centre %r; using CRVAL", raw)
        else:
            return ra, dec
    try:
        return float(header["CRVAL1"]), float(header["CRVAL2"])
    except (KeyError, TypeError, ValueError):
        return None, None


def image_geometry(header: fits.Header) -> dict[str, object]:
    """The WSClean-relevant geometry of an image header, as CSV columns.

    ``image_n_pixels`` and ``image_cell_arcsec`` are the ``-size``/``-scale``
    tclean imaged with; ``image_n_channels`` is the cube's ``NAXIS3`` (1 for an
    mfs image). The ``image_*`` strings are tclean's own values, kept for the
    report.
    """
    parameters = parse_tclean_parameters(header)
    naxis1 = int(header.get("NAXIS1", 0) or 0)
    naxis2 = int(header.get("NAXIS2", 0) or 0)
    cell = abs(float(header.get("CDELT2", 0.0) or 0.0))
    if cell == 0.0:
        cell = abs(float(header.get("CDELT1", 0.0) or 0.0))
    ra, dec = _phasecenter_deg(parameters, header)
    n_channels = int(header.get("NAXIS3", 1) or 1)
    channel_width = header.get("CDELT3")
    reference = header.get("CRVAL3")
    imsize = parameters.get("imsize")
    tclean_cell = parameters.get("cell")
    niter = parameters.get("niter")
    threshold = parameters.get("threshold")
    return {
        # WSClean images a square: the longer side covers a non-square image.
        # A header without the grid leaves both unset: the MS-derived ones apply.
        "image_n_pixels": max(naxis1, naxis2) or math.nan,
        "image_cell_arcsec": round(cell * 3600.0, 6) or math.nan,
        "image_n_channels": n_channels,
        "image_channel_width_hz": float(channel_width) if channel_width is not None else math.nan,
        "image_reference_frequency": float(reference) if reference is not None else math.nan,
        "phasecenter_ra_deg": ra if ra is not None else math.nan,
        "phasecenter_dec_deg": dec if dec is not None else math.nan,
        "image_weighting": parameters.get("weighting", ""),
        "image_robust": parameters.get("robust", math.nan),
        "image_stokes": parameters.get("stokes", ""),
        "image_specmode": parameters.get("specmode", ""),
        "image_deconvolver": parameters.get("deconvolver", ""),
        "image_niter": (
            int(niter)
            if isinstance(niter, (int, float)) and not isinstance(niter, bool)
            else (niter if niter is not None else math.nan)
        ),
        "image_threshold": threshold if isinstance(threshold, str) else _format_value(threshold),
        "image_threshold_jy": _threshold_to_jy(threshold),
        "image_imsize": imsize if isinstance(imsize, str) else _format_value(imsize),
        "image_cell": tclean_cell if isinstance(tclean_cell, str) else _format_value(tclean_cell),
        "image_phasecenter": parameters.get("phasecenter", ""),
    }


def _format_value(value: object) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


def _threshold_to_jy(value: object) -> float:
    """A tclean ``threshold`` string such as ``0.000412Jy`` or ``1.2mJy`` as Jy."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        return math.nan
    match = re.search(r"([0-9.eE+-]+)\s*(Jy|mJy|uJy|nJy)?", value.strip(), re.IGNORECASE)
    if match is None:
        return math.nan
    number = float(match.group(1))
    unit = (match.group(2) or "Jy").lower()
    scale = {"jy": 1.0, "mjy": 1e-3, "ujy": 1e-6, "njy": 1e-9}[unit]
    return number * scale


@dataclass(frozen=True)
class MemberImage:
    """One pipeline image of a member: where it is and what it is."""

    filename: str
    kind: str
    path: Path | None = None
    url: str | None = None

    @property
    def source(self) -> str:
        return str(self.path) if self.path is not None else (self.url or "")

    def read_header_bytes(self) -> bytes:
        """The image's FITS header blocks, read up to its ``END`` card."""
        if self.path is not None:
            return read_fits_header_bytes(self.path)
        if self.url is not None:
            return read_remote_fits_header_bytes(self.url)
        raise ValueError(f"{self.filename}: no local path or archive URL")

    def read_header(self) -> fits.Header:
        """The image's parsed primary header."""
        return fits.Header.fromstring(self.read_header_bytes().decode("ascii", "replace"))


def _normalize_field(value: str) -> str:
    """A field name without separators, lower case: ``G14.11-0.57`` -> ``g1411057``."""
    return re.sub(r"[^A-Za-z0-9]", "", str(value)).lower()


def _image_field_token(filename: str) -> str:
    """The field part of an ALMA science image name (``...<field>_sci.spw25...``)."""
    match = re.search(r"uid___[A-Za-z0-9_]+?\.(.+)_sci\.spw[\d_]+\.", filename)
    return match.group(1) if match else ""


def _names_field(filename: str, field_name: str) -> bool:
    """Whether an image's name is of ``field_name``, which the name may truncate."""
    token = _normalize_field(_image_field_token(filename))
    field = _normalize_field(field_name)
    return bool(token) and bool(field) and (field.startswith(token) or token.startswith(field))


def _spw_list(filename: str) -> list[int]:
    """The spectral windows an image covers, parsed from its name (``spw13_15_17_19``)."""
    match = re.search(r"spw([\d_]+)\.", filename)
    if match is None:
        return []
    return [int(part) for part in match.group(1).split("_") if part.isdigit()]


def _image_preference(name: str) -> tuple[int, int, int]:
    """Lower is better: selfcal, then regcal, then Stokes I, then shorter name."""
    calibration = 0 if ".selfcal." in name else 1 if ".regcal." in name else 2
    stokes_i = 0 if (".I." in name or ".I_" in name or name.endswith(".I.pbcor.fits")) else 1
    return (calibration, stokes_i, len(name))


def match_member_images(
    product_dir: Path,
    field_name: str,
    spw: int,
    kinds: Sequence[str] = _IMAGE_KINDS,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> list[MemberImage]:
    """The member's local images of one field and spectral window.

    ALMA names images ``...<field>_sci.spw<list>.<mfs|cube>...pbcor.fits`` where
    ``<list>`` may be a single spw (``spw25``) or several (``spw13_15_17_19``);
    the field is matched on the header's ``OBJECT`` card, since the file name
    truncates it. The self-calibrated (or regcal) Stokes-I image is preferred.
    """
    images: list[MemberImage] = []
    for kind in kinds:
        candidates = sorted(product_dir.glob(f"*.spw*.{kind}.*.pbcor.fits"))
        if include:
            candidates = [p for p in candidates if any(p.match(glob) for glob in include)]
        if exclude:
            candidates = [p for p in candidates if not any(p.match(glob) for glob in exclude)]
        field_matches: list[Path] = []
        for path in candidates:
            if spw not in _spw_list(path.name):
                continue
            try:
                obj = str(read_fits_header(path).get("OBJECT", "")).strip()
            except Exception:  # noqa: BLE001 - unreadable candidate is skipped
                obj = ""
            if not obj or obj == field_name:
                field_matches.append(path)
        if not field_matches:
            continue
        field_matches.sort(key=lambda p: _image_preference(p.name))
        images.append(MemberImage(field_matches[0].name, kind, path=field_matches[0]))
    return images


@lru_cache(maxsize=256)
def _archive_science_images(
    member_ous_uid: str, include: tuple[str, ...], exclude: tuple[str, ...]
) -> tuple[tuple[str, str], ...]:
    """A member's science images in the archive as ``(filename, url)``, asked once."""
    from almasim.services.download.datalink import (
        expand_archives,
        resolve_products,
        select_files,
    )

    products = resolve_products([member_ous_uid])
    products = expand_archives(products, layout="flat")
    products = select_files(products, include=list(include) or ["*.pbcor.fits"], exclude=exclude)
    return tuple(
        (product.filename, product.access_url)
        for product in products
        if "_sci." in product.filename
    )


def archive_member_images(
    member_ous_uid: str,
    field_name: str,
    spw: int,
    kinds: Sequence[str] = _IMAGE_KINDS,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> list[MemberImage]:
    """The member's images of one field and spectral window, resolved from the archive.

    The archive is asked once per member. The field is matched on the image
    name, which may truncate it; a member whose images are all of one field
    needs no match.
    """
    products = _archive_science_images(member_ous_uid, tuple(include), tuple(exclude))
    if len({_image_field_token(filename) for filename, _ in products}) > 1:
        products = tuple(item for item in products if _names_field(item[0], field_name))
    images: list[MemberImage] = []
    for kind in kinds:
        selected = sorted(
            (
                (filename, url)
                for filename, url in products
                if f".{kind}." in filename and spw in _spw_list(filename)
            ),
            key=lambda item: _image_preference(item[0]),
        )
        if selected:
            filename, url = selected[0]
            images.append(MemberImage(filename, kind, url=url))
    return images


def headers_folder(
    member_folder: str,
    member_ous_uid: str,
    *,
    delivery_root: Path | None,
    output: Path | None = None,
) -> Path:
    """Where a member's headers and parameter CSVs go: its own
    ``images/headers`` folder in the delivery tree, or ``<output>/<member>``."""
    if output is not None:
        member = member_ous_uid.replace("://", "___").replace("/", "_") or "member"
        return Path(output) / member
    if delivery_root is None or not member_folder:
        raise ValueError("an MS outside a delivery tree needs an output folder (--output)")
    return Path(delivery_root) / member_folder / IMAGES_FOLDER / HEADERS_FOLDER


def product_paths(folder: Path, image: MemberImage) -> tuple[Path, Path]:
    """One delivered image's saved header and its parameter CSV."""
    return folder / f"{image.filename}.hdr", folder / f"{image.filename}.csv"


def listed_mss(folder: Path) -> set[str]:
    """The MSs the parameter CSVs of one member's folder have rows of."""
    listed: set[str] = set()
    for path in sorted(Path(folder).glob("*.fits.csv")):
        try:
            listed |= {str(v) for v in pd.read_csv(path, usecols=["filename"])["filename"]}
        except Exception:  # noqa: BLE001 - an unreadable CSV lists nothing
            continue
    return listed


def _geometry(header_path: object, kind: object) -> dict[str, object]:
    """The image columns of one saved header (empty ones when it cannot be read)."""
    try:
        return image_geometry(read_fits_header(str(header_path)))
    except Exception as exc:  # noqa: BLE001 - reported in the row
        logger.warning("Could not parse %s: %s", header_path, exc)
        geometry: dict[str, object] = {
            column: math.nan for column in IMAGE_COLUMNS if column != "image_kind"
        }
        geometry["image_kind"] = kind
        return geometry


def _checked(record: dict[str, object], grid: tuple[float, float, float]) -> dict[str, object]:
    """``record`` with the MS-derived size and cell and how the image's compare."""
    from almasim.services.imaging.ms_geometry import image_pixels

    fov_fraction, beam_sampling, min_beams_across = grid
    # Without the image's own size, which image_pixels would hand back.
    from_ms = {key: record.get(key) for key in _MS_COLUMNS}
    record["ms_n_pixels"] = image_pixels(from_ms, fov_fraction, beam_sampling, min_beams_across)
    record["ms_cell_arcsec"] = float(str(record["synthetized_beam_size"])) / beam_sampling
    record["image_check"] = _geometry_check(record)
    return record


def _write_rows(csv_path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Put ``rows`` in one image's parameter CSV, replacing those of the same
    (MS, window, field): an image of several execution blocks or windows has
    a row for each."""
    frame = pd.DataFrame(list(rows))
    if csv_path.is_file():
        try:
            old = pd.read_csv(csv_path)
        except Exception:  # noqa: BLE001 - an unreadable CSV is written afresh
            old = pd.DataFrame()
        if not old.empty:
            keys = set(zip(*(frame[k].astype(str) for k in _ROW_KEY), strict=True))
            kept = [
                tuple(str(v) for v in key) not in keys
                for key in zip(*(old[k] for k in _ROW_KEY), strict=True)
            ]
            frame = pd.concat([old[kept], frame], ignore_index=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)


def collect_headers(
    mss: Sequence[Path],
    *,
    index: dict[str, dict[str, str]],
    delivery_root: Path | None,
    output: Path | None = None,
    kinds: Sequence[str] = _IMAGE_KINDS,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    source: str = "local",
    overwrite: bool = False,
    fov_fraction: float = 1.5,
    beam_sampling: float = 8.0,
    min_beams_across: float = 32.0,
) -> tuple[list[dict[str, object]], list[tuple[Path, str]]]:
    """Save the header of every (MS, window, field)'s tclean image and write
    its parameter CSV beside it; the rows written and what failed.

    ``source`` is ``local`` (the delivery tree's ``product/`` folders),
    ``archive`` (the ALMA data portal), or ``auto`` (local first, then archive).
    Each image has ``<image>.hdr`` and ``<image>.csv`` in its member's
    ``headers_folder``; the CSV has one row per (MS, window, field) the image
    is of, with the MS-derived columns, the image's and their comparison.
    """
    grid = (fov_fraction, beam_sampling, min_beams_across)
    rows: list[dict[str, object]] = []
    failures: list[tuple[Path, str]] = []
    for ms in mss:
        eb = eb_uid_from_ms(ms) or ""
        member = index.get(eb, {})
        member_uid = str(member.get("member_ous_uid", ""))
        member_folder = str(member.get("member_folder", ""))
        try:
            folder = headers_folder(
                member_folder, member_uid, delivery_root=delivery_root, output=output
            )
            ms_rows = compute_imaging_parameters(ms)
        except Exception as exc:  # noqa: BLE001 - recorded and reported
            failures.append((ms, f"{type(exc).__name__}: {exc}"))
            continue
        by_image: dict[Path, list[dict[str, object]]] = {}
        for row in ms_rows.to_dict("records"):
            field_name = str(row.get("field_name", ""))
            spw = int(row["spectral_window_id"])
            images = _images_for(
                source,
                delivery_root,
                member_folder,
                member_uid,
                field_name,
                spw,
                kinds,
                include,
                exclude,
            )
            if not images:
                failures.append(
                    (ms, f"no tclean image for spw {spw} field {field_name} ({member_uid})")
                )
                continue
            ms_ra, ms_dec = ms_phase_center(ms, row.get("target_field_ids"))
            for image in images:
                header_path, csv_path = product_paths(folder, image)
                status, error = "ok", ""
                if header_path.is_file() and not overwrite:
                    logger.info("Keeping saved header %s", header_path)
                else:
                    try:
                        raw = image.read_header_bytes()
                    except Exception as exc:  # noqa: BLE001 - recorded and reported
                        status, error = "failed", f"{type(exc).__name__}: {exc}"
                        logger.warning("Could not read %s: %s", image.source, error)
                    else:
                        header_path.parent.mkdir(parents=True, exist_ok=True)
                        header_path.write_bytes(_pad_header_bytes(raw))
                record = {
                    **{key: row.get(key, "") for key in _MS_COLUMNS},
                    "ms_phasecenter_ra_deg": ms_ra if ms_ra is not None else math.nan,
                    "ms_phasecenter_dec_deg": ms_dec if ms_dec is not None else math.nan,
                    **_geometry(header_path, image.kind),
                    "image_path": "" if image.path is None else str(image.path),
                    "image_url": "" if image.url is None else str(image.url),
                    "image_filename": image.filename,
                    "image_kind": image.kind,
                    "header_path": str(header_path),
                    "member_ous_uid": member_uid,
                    "member_folder": member_folder,
                    "status": status,
                    "error": error,
                }
                by_image.setdefault(csv_path, []).append(_checked(record, grid))
        for csv_path, image_rows in by_image.items():
            _write_rows(csv_path, image_rows)
            rows.extend(image_rows)
    return rows, failures


def _images_for(
    source: str,
    delivery_root: Path | None,
    member_folder: str,
    member_uid: str,
    field_name: str,
    spw: int,
    kinds: Sequence[str],
    include: Sequence[str],
    exclude: Sequence[str],
) -> list[MemberImage]:
    if source not in {"local", "archive", "auto"}:
        raise ValueError(f"source must be local, archive or auto, not {source!r}")
    if source in {"local", "auto"} and delivery_root is not None and member_folder:
        product_dir = Path(delivery_root) / member_folder / "product"
        local = match_member_images(product_dir, field_name, spw, kinds, include, exclude)
        if local:
            return local
        if source == "local":
            return []
    if source in {"archive", "auto"} and member_uid:
        return archive_member_images(member_uid, field_name, spw, kinds, include, exclude)
    return []


def _pad_header_bytes(raw: bytes) -> bytes:
    remainder = len(raw) % FITS_BLOCK_BYTES
    if remainder:
        raw += b" " * (FITS_BLOCK_BYTES - remainder)
    return raw


def _geometry_check(record: dict[str, object]) -> str:
    """A short note on how the image's grid compares with the MS-derived one."""
    notes: list[str] = []
    n_pixels = record.get("image_n_pixels")
    if isinstance(n_pixels, (int, float)) and not pd.isna(n_pixels):
        ms_pixels = record.get("ms_n_pixels")
        if isinstance(ms_pixels, (int, float)) and not pd.isna(ms_pixels) and int(ms_pixels) > 0:
            ratio = int(n_pixels) / int(ms_pixels)
            if abs(ratio - 1.0) > 0.02:
                notes.append(f"size {int(n_pixels)} vs MS {int(ms_pixels)}")
    cell = record.get("image_cell_arcsec")
    if isinstance(cell, (int, float)) and not pd.isna(cell):
        ms_cell = record.get("ms_cell_arcsec")
        if isinstance(ms_cell, (int, float)) and not pd.isna(ms_cell) and float(ms_cell) > 0:
            ratio = float(cell) / float(ms_cell)
            if abs(ratio - 1.0) > 0.02:
                notes.append(f"cell {float(cell):.4g} vs MS {float(ms_cell):.4g} asec")
    return "; ".join(notes) if notes else "ok"


def parameter_csvs(root: Path) -> list[Path]:
    """The per-image parameter CSVs under ``root``: a delivery tree (each
    member's ``images/headers``), one member's folder, or an output folder of
    ``fetch-headers``. An MS is never descended into."""
    from almasim.services.archive.delivery import is_delivery_root

    root = Path(root)
    inside = f"{IMAGES_FOLDER}/{HEADERS_FOLDER}/*.fits.csv"
    if is_delivery_root(root):
        return sorted(root.glob(f"*/science_goal.*/group.*/member.*/{inside}"))
    found: list[Path] = []
    for pattern in ("*.fits.csv", "*/*.fits.csv", f"{HEADERS_FOLDER}/*.fits.csv", inside):
        found += root.glob(pattern)
    return sorted(set(found))


def parse_headers(
    root: Path,
    *,
    fov_fraction: float = 1.5,
    beam_sampling: float = 8.0,
    min_beams_across: float = 32.0,
) -> pd.DataFrame:
    """Read each saved header under ``root`` again and rewrite the image
    columns and the comparison of its parameter CSV; every row rewritten."""
    grid = (fov_fraction, beam_sampling, min_beams_across)
    frames: list[pd.DataFrame] = []
    for csv_path in parameter_csvs(root):
        records = []
        for row in pd.read_csv(csv_path).to_dict("records"):
            header_path = csv_path.with_suffix(".hdr")
            row.update(_geometry(header_path, row.get("image_kind", "")))
            row["header_path"] = str(header_path)
            records.append(_checked(row, grid))
        frame = pd.DataFrame(records)
        frame.to_csv(csv_path, index=False)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_parameters(root: Path) -> pd.DataFrame:
    """The imaging parameter rows of the per-image CSVs under ``root``: one
    per (MS, window, field), from its cube image's CSV, or its ``mfs``
    image's when there is no cube."""
    paths = parameter_csvs(root)
    if not paths:
        raise ValueError(f"{root}: no per-image parameter CSV; run image fetch-headers first")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    cube_first = (frame["image_kind"] != "cube").astype(int)
    frame = frame.assign(_order=cube_first).sort_values("_order", kind="stable")
    frame = frame.drop_duplicates(list(_ROW_KEY), keep="first").drop(columns="_order")
    return frame.sort_values(list(_ROW_KEY), kind="stable").reset_index(drop=True)


def ms_phase_center(ms: str | Path, field_id: object) -> tuple[float | None, float | None]:
    """The MS field's phase centre as ``(ra_deg, dec_deg)``, or ``(None, None)``.

    Read from the MS ``FIELD`` table's ``PHASE_DIR`` column (radians), with the
    right ascension between 0 and 360 degrees.
    """
    try:
        from almasim.services.imaging.ms_geometry import import_casacore_tables
        from almasim.services.ms.tables import open_table

        opener = import_casacore_tables()
        with open_table(f"{ms}::FIELD", opener=opener) as field:
            direction = field.getcell("PHASE_DIR", int(str(field_id)))
        # casacore gives RA in (-180, 180]; the image headers count from 0 to 360.
        ra_deg = math.degrees(float(direction[0, 0])) % 360.0
        return ra_deg, math.degrees(float(direction[0, 1]))
    except Exception:  # noqa: BLE001 - missing optional dependency or table
        return None, None


_MS_COLUMNS = (
    "filename",
    "spectral_window_id",
    "spectral_window_name",
    "reference_frequency",
    "fov_per_frequency",
    "max_baseline_size",
    "synthetized_beam_size",
    "n_channels",
    "channel_width_hz",
    "spectral_resolution_hz",
    "total_bandwidth_hz",
    "start_mjd_s",
    "end_mjd_s",
    "duration_s",
    "start_datetime",
    "end_datetime",
    "n_visibility_rows",
    "target_field_ids",
    "field_name",
)
