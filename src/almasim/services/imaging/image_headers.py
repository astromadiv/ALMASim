"""WSClean geometry from the FITS headers of the ALMA pipeline's tclean images.

``compute_imaging_parameters`` derives the WSClean geometry from the
MeasurementSet: the primary beam's field of view, the synthesized beam and the
window's channels. The pipeline's images already carry the geometry tclean
imaged with: the ``HISTORY`` cards repeat every tclean parameter (``imsize``,
``cell``, ``phasecenter``, ``specmode``, ``nchan``, ``weighting`` ...) and the
WCS cards describe the grid (``NAXIS``, ``CDELT``, ``CRVAL``, ``CRPIX``,
``BMAJ`` ...). This module reads those headers -- from the downloaded delivery
tree, or from the archive when they are not on disk -- and turns them into the
parameter rows ``image-from-ms`` images from.

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
from pathlib import Path

import httpx
import pandas as pd
from astropy.io import fits
from tenacity import retry, stop_after_attempt, wait_exponential

from almasim.services.imaging.ms_geometry import compute_imaging_parameters
from almasim.services.imaging.provenance import eb_uid_from_ms

logger = logging.getLogger(__name__)

__all__ = [
    "IMAGE_COLUMNS",
    "MemberImage",
    "collect_headers",
    "image_geometry",
    "match_member_images",
    "parse_manifest",
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

MANIFEST_FILENAME = "headers.csv"

# The parameter columns ``parse_manifest`` adds to the MS-derived rows. The
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
    """``14:13:09.9473`` (RA) or ``-065.20.21.024`` (Dec) as decimal degrees."""
    sign = -1.0 if text.startswith("-") else 1.0
    body = text.lstrip("+-")
    separator = ":" if ":" in body else "."
    parts = body.split(separator)
    if len(parts) == 1:
        value = float(parts[0])
        return sign * value * (15.0 if ra else 1.0)
    hours, minutes, seconds = (float(part) for part in parts[:3])
    value = hours + minutes / 60.0 + seconds / 3600.0
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
        "image_n_pixels": naxis1 or naxis2,
        "image_cell_arcsec": round(cell * 3600.0, 6),
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
    """The ``<field>_<intent>`` token of an ALMA image name."""
    match = re.search(r"\.([^.]+)\.spw\d+\.", filename)
    return match.group(1) if match else ""


def _spw_list(filename: str) -> list[int]:
    """The spectral windows an image covers, parsed from its name (``spw13_15_17_19``)."""
    match = re.search(r"spw([\d_]+)\.", filename)
    if match is None:
        return []
    return [int(part) for part in match.group(1).split("_") if part.isdigit()]


def _image_preference(name: str) -> tuple[int, int, int]:
    """Lower is better: selfcal/regcal/manual, then Stokes I, then shorter name."""
    selfcal = 0 if ".selfcal." in name else 1
    regcal = 0 if ".regcal." in name else 1
    stokes_i = 0 if (".I." in name or ".I_" in name or name.endswith(".I.pbcor.fits")) else 1
    return (selfcal + regcal, stokes_i, len(name))


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


def archive_member_images(
    member_ous_uid: str,
    field_name: str,
    spw: int,
    kinds: Sequence[str] = _IMAGE_KINDS,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> list[MemberImage]:
    """The member's images of one field and spectral window, resolved from the archive."""
    from almasim.services.download.datalink import (
        expand_archives,
        resolve_products,
        select_files,
    )

    products = resolve_products([member_ous_uid])
    products = expand_archives(products, layout="flat")
    globs = list(include) or ["*.pbcor.fits"]
    products = select_files(products, include=globs, exclude=exclude)
    images: list[MemberImage] = []
    for kind in kinds:
        selected = [
            product
            for product in products
            if f".{kind}." in product.filename
            and spw in _spw_list(product.filename)
            and ("_sci" in product.filename)
        ]
        selfcal = [product for product in selected if ".selfcal." in product.filename]
        regcal = [product for product in selected if ".regcal." in product.filename]
        chosen = selfcal or regcal or selected[:1]
        for product in chosen[:1]:
            images.append(MemberImage(product.filename, kind, url=product.access_url))
    return images


def _header_path(header_folder: Path, member_ous_uid: str, image: MemberImage) -> Path:
    """The local path a header is cached under, one per unique image."""
    member = member_ous_uid.replace("://", "___").replace("/", "_") or "member"
    return header_folder / member / f"{image.filename}.hdr"


def collect_headers(
    mss: Sequence[Path],
    header_folder: Path,
    *,
    index: dict[str, dict[str, str]],
    delivery_root: Path | None,
    kinds: Sequence[str] = _IMAGE_KINDS,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    source: str = "local",
    overwrite: bool = False,
) -> tuple[list[dict[str, object]], list[tuple[Path, str]]]:
    """Read the tclean header of every (MS, window, field) row; the manifest rows.

    ``source`` is ``local`` (the delivery tree's ``product/`` folders),
    ``archive`` (the ALMA data portal), or ``auto`` (local first, then archive).
    Each unique image's header is read once and cached under
    ``header_folder/<member>/<image>.hdr``; the manifest has one row per
    (MS, window, field, image kind).
    """
    header_folder = Path(header_folder)
    header_folder.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[tuple[Path, str]] = []
    for ms in mss:
        try:
            ms_rows = compute_imaging_parameters(ms)
        except Exception as exc:  # noqa: BLE001 - recorded and reported
            failures.append((ms, f"{type(exc).__name__}: {exc}"))
            continue
        for row in ms_rows.to_dict("records"):
            eb = eb_uid_from_ms(ms) or ""
            member = index.get(eb, {})
            member_uid = str(member.get("member_ous_uid", ""))
            member_folder = str(member.get("member_folder", ""))
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
            for image in images:
                header_path = _header_path(header_folder, member_uid, image)
                status, error = "ok", ""
                if header_path.is_file() and not overwrite:
                    logger.info("Keeping cached header %s", header_path)
                else:
                    try:
                        raw = image.read_header_bytes()
                    except Exception as exc:  # noqa: BLE001 - recorded and reported
                        status, error = "failed", f"{type(exc).__name__}: {exc}"
                        logger.warning("Could not read %s: %s", image.source, error)
                    else:
                        header_path.parent.mkdir(parents=True, exist_ok=True)
                        header_path.write_bytes(_pad_header_bytes(raw))
                ms_ra, ms_dec = ms_phase_center(ms, int(row.get("target_field_ids") or 0))
                rows.append(
                    {
                        **row,
                        "member_ous_uid": member_uid,
                        "member_folder": member_folder,
                        "ms_phasecenter_ra_deg": ms_ra if ms_ra is not None else math.nan,
                        "ms_phasecenter_dec_deg": ms_dec if ms_dec is not None else math.nan,
                        "image_filename": image.filename,
                        "image_kind": image.kind,
                        "image_path": "" if image.path is None else str(image.path),
                        "image_url": "" if image.url is None else str(image.url),
                        "header_path": str(header_path),
                        "status": status,
                        "error": error,
                    }
                )
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


def write_manifest(rows: Sequence[dict[str, object]], path: Path) -> Path:
    """Write the header manifest CSV; the path."""
    frame = pd.DataFrame(list(rows))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _first_present(row: dict[str, object], key: str) -> object:
    value = row.get(key)
    if value is None:
        return math.nan
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    return value


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


def parse_manifest(
    manifest: Path,
    output_csv: Path,
    *,
    fov_fraction: float = 1.5,
    beam_sampling: float = 8.0,
    min_beams_across: float = 32.0,
) -> pd.DataFrame:
    """Turn the header manifest into the imaging parameter CSV.

    One output row per (MS, window, field): the MS-derived columns of the
    manifest, plus the image geometry of the cube header (the mfs header when
    there is no cube), plus a comparison with the MS-derived geometry.
    """
    from almasim.services.imaging.ms_geometry import image_pixels

    frame = pd.read_csv(manifest)
    if frame.empty:
        raise ValueError(f"{manifest}: no header rows")
    groups = frame.groupby(
        ["filename", "spectral_window_id", "field_name"], sort=False, dropna=False
    )
    records: list[dict[str, object]] = []
    for _, group in groups:
        base = dict(group.iloc[0])
        cube = group[group["image_kind"] == "cube"]
        chosen = cube.iloc[0] if len(cube) else group.iloc[0]
        header_path = chosen.get("header_path")
        try:
            header = read_fits_header(header_path)
            geometry = image_geometry(header)
        except Exception as exc:  # noqa: BLE001 - reported in the row
            logger.warning("Could not parse %s: %s", header_path, exc)
            geometry = {column: math.nan for column in IMAGE_COLUMNS if column != "image_kind"}
            geometry["image_kind"] = chosen.get("image_kind", "")
        record = {
            **{key: base.get(key, "") for key in _MS_COLUMNS},
            "ms_phasecenter_ra_deg": base.get("ms_phasecenter_ra_deg", math.nan),
            "ms_phasecenter_dec_deg": base.get("ms_phasecenter_dec_deg", math.nan),
            **geometry,
            "image_path": chosen.get("image_path", ""),
            "image_filename": chosen.get("image_filename", ""),
            "image_kind": chosen.get("image_kind", ""),
            "header_path": chosen.get("header_path", ""),
            "member_ous_uid": base.get("member_ous_uid", ""),
            "member_folder": base.get("member_folder", ""),
        }
        record["ms_n_pixels"] = image_pixels(
            base, fov_fraction, beam_sampling, min_beams_across
        )
        record["ms_cell_arcsec"] = float(base["synthetized_beam_size"]) / beam_sampling
        record["image_check"] = _geometry_check(record)
        records.append(record)
    result = pd.DataFrame(records)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def ms_phase_center(ms: str | Path, field_id: int) -> tuple[float | None, float | None]:
    """The MS field's phase centre as ``(ra_deg, dec_deg)``, or ``(None, None)``.

    Read from the MS ``FIELD`` table's ``PHASE_DIR`` column (radians).
    """
    try:
        from almasim.services.imaging.ms_geometry import import_casacore_tables
        from almasim.services.ms.tables import open_table

        opener = import_casacore_tables()
        with open_table(f"{ms}::FIELD", opener=opener) as field:
            direction = field.getcell("PHASE_DIR", int(field_id))
        return float(direction[0, 0]) * 180.0 / math.pi, float(direction[0, 1]) * 180.0 / math.pi
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


def read_manifest(path: Path) -> pd.DataFrame:
    """The header manifest CSV."""
    return pd.read_csv(path)
