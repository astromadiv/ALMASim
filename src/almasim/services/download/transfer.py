"""Download single files with retries, spread over the ALMA mirrors. (MADIV-28 T6.9)"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tarfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

from .datalink import _DATALINK_MIRRORS
from .models import FileDownloadStatus

logger = logging.getLogger(__name__)


MAX_PARALLEL_PER_MIRROR = 10
MAX_PARALLEL_TOTAL = MAX_PARALLEL_PER_MIRROR * len(_DATALINK_MIRRORS)
_RATE_LIMIT_SEC = 0.5
# ESO drops long tar streams mid-body; each retry resumes from the .part, so
# attempts are cheap. Waits between them: 1, 4, 16, 64, 256 s.
_MAX_FILE_ATTEMPTS = 6
_RETRY_BACKOFF_BASE = 4
_CHUNK_SIZE = 256 * 1024


def _sha256_file(path: Path) -> str:
    hsh = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hsh.update(chunk)
    return hsh.hexdigest()


def _starts_like_html(path: Path, filename: str) -> bool:
    """True when a file meant to be data begins with an HTML page."""
    if filename.lower().endswith((".html", ".htm", ".xml")):
        return False
    with open(path, "rb") as handle:
        return handle.read(64).lstrip().startswith(b"<")


def _has_tar_header(path: Path) -> bool:
    """True when the first 512-byte block is a tar header with a valid checksum."""
    with open(path, "rb") as handle:
        block = handle.read(tarfile.BLOCKSIZE)
    try:
        tarfile.TarInfo.frombuf(block, tarfile.ENCODING, "surrogateescape")
    except tarfile.HeaderError:
        return False
    return True


def _download_single_attempt(  # noqa: C901
    file_status: FileDownloadStatus,
    part_path: Path,
    *,
    should_cancel: Callable[[], bool] | None = None,
    update_callback: Callable[[FileDownloadStatus], None] | None = None,
) -> bool:
    existing_bytes = part_path.stat().st_size if part_path.exists() else 0
    if existing_bytes > 0 and _starts_like_html(part_path, file_status.filename):
        # A .part left by a portal maintenance notice (EA, 2026-09-29): resuming
        # it appends the real bytes after the page and yields a right-sized tar
        # whose first header is HTML. Start the file over instead.
        logger.warning("Discarding %s: it begins with an HTML page", part_path.name)
        part_path.unlink()
        existing_bytes = 0
    headers = {}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    if should_cancel is not None and should_cancel():
        file_status.status = "cancelled"
        if update_callback is not None:
            update_callback(file_status)
        return False

    def _stream(response: httpx.Response, resume_from: int) -> bool:
        response.raise_for_status()
        # A portal under maintenance answers 200 with an HTML notice (the EA
        # mirror on 2026-09-29); without this it only shows up as a short download.
        content_type = getattr(response, "headers", {}).get("content-type", "")
        if "text/html" in content_type and not file_status.filename.endswith((".html", ".htm")):
            page = b"".join(response.iter_bytes(chunk_size=_CHUNK_SIZE))[:4096]
            text = " ".join(re.sub(rb"<[^>]+>", b" ", page).decode("utf-8", "replace").split())
            raise RuntimeError(f"Server sent an HTML page instead of the file: {text[:300]}")
        file_status.bytes_downloaded = resume_from
        if update_callback is not None:
            update_callback(file_status)
        with open(part_path, "ab" if resume_from > 0 else "wb") as handle:
            for chunk in response.iter_bytes(chunk_size=_CHUNK_SIZE):
                if should_cancel is not None and should_cancel():
                    file_status.status = "cancelled"
                    if update_callback is not None:
                        update_callback(file_status)
                    return False
                handle.write(chunk)
                file_status.bytes_downloaded += len(chunk)
                if update_callback is not None:
                    update_callback(file_status)
        return True

    with httpx.Client(timeout=300, follow_redirects=True) as client:
        range_rejected = False
        with client.stream("GET", file_status.access_url, headers=headers) as response:
            if existing_bytes > 0 and response.status_code == 416:
                # The server will not resume (the ALMA data portal answers 416
                # to every ranged GET). Start over rather than fail the file.
                range_rejected = True
            else:
                # A 200 to a ranged request means the server ignored the range
                # and is sending the whole file: overwrite, do not append.
                resume_from = 0 if response.status_code == 200 else existing_bytes
                if not _stream(response, resume_from):
                    return False
        if range_rejected:
            part_path.unlink(missing_ok=True)
            with client.stream("GET", file_status.access_url) as response:
                if not _stream(response, 0):
                    return False

    # A stream that ends early is not an error to httpx, and a tar truncated
    # on a block boundary is not an error to tarfile either: it extracts as a
    # smaller, valid-looking ASDM. The declared size is the only thing that
    # catches it, so a short file is a failed attempt, never a completed one.
    received = part_path.stat().st_size if part_path.exists() else 0
    if file_status.content_length > 0 and received < file_status.content_length:
        raise RuntimeError(
            f"Short download: received {received} of {file_status.content_length} bytes"
        )
    return True


def _download_single_file(  # noqa: C901
    file_status: FileDownloadStatus,
    destination: Path,
    *,
    should_cancel: Callable[[], bool] | None = None,
    update_callback: Callable[[FileDownloadStatus], None] | None = None,
    host_semaphores: dict[str, threading.BoundedSemaphore] | None = None,
) -> None:
    destination_path = destination / file_status.filename
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination_path.with_suffix(destination_path.suffix + ".part")

    if (
        destination_path.exists()
        and destination_path.suffix == ".tar"
        and not _has_tar_header(destination_path)
    ):
        # Right size is not enough: an HTML-prefixed tar passes the size check
        # and then fails extraction on every rerun. Fetch it again.
        logger.warning("Re-downloading %s: it has no tar header", destination_path.name)
        destination_path.unlink()

    if destination_path.exists():
        existing_size = destination_path.stat().st_size
        if file_status.content_length <= 0 or existing_size >= file_status.content_length:
            file_status.bytes_downloaded = existing_size
            file_status.status = "completed"
            try:
                file_status.sha256 = _sha256_file(destination_path)
            except OSError:
                file_status.sha256 = None
            if update_callback is not None:
                update_callback(file_status)
            return

    file_status.status = "downloading"
    if update_callback is not None:
        update_callback(file_status)
    last_error: Exception | None = None
    host = urlsplit(file_status.access_url).netloc
    semaphore = host_semaphores.get(host) if host_semaphores else None
    if semaphore is not None:
        semaphore.acquire()
    try:
        for attempt in range(1, _MAX_FILE_ATTEMPTS + 1):
            try:
                completed = _download_single_attempt(
                    file_status,
                    part_path,
                    should_cancel=should_cancel,
                    update_callback=update_callback,
                )
                if not completed:
                    return
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - retried, then raised
                last_error = exc
                if attempt < _MAX_FILE_ATTEMPTS:
                    time.sleep(_RETRY_BACKOFF_BASE ** (attempt - 1))
    finally:
        if semaphore is not None:
            semaphore.release()

    if last_error is not None:
        file_status.status = "failed"
        file_status.error = str(last_error)
        if update_callback is not None:
            update_callback(file_status)
        return

    try:
        os.replace(part_path, destination_path)
    except OSError as exc:
        file_status.status = "failed"
        file_status.error = f"Rename failed: {exc}"
        if update_callback is not None:
            update_callback(file_status)
        return

    try:
        file_status.sha256 = _sha256_file(destination_path)
    except OSError:
        file_status.sha256 = None
    file_status.status = "completed"
    if update_callback is not None:
        update_callback(file_status)


def _mirror_hosts() -> list[str]:
    """Return the netloc of each known DataLink/TAP mirror."""
    return [urlsplit(base).netloc for base in _DATALINK_MIRRORS]


def _distribute_across_mirrors(
    statuses: list[FileDownloadStatus],
) -> None:
    """Round-robin rewrite each ``access_url`` host across the known ALMA mirrors.

    The ALMA archive is replicated on three DataLink/TAP mirrors (ESO, NRAO and
    NAOJ). Rewriting hostnames lets us spread concurrent transfers across all
    three endpoints instead of hammering whichever mirror DataLink originally
    pointed at.
    """
    hosts = _mirror_hosts()
    if not hosts:
        return
    for index, status in enumerate(statuses):
        parts = urlsplit(status.access_url)
        if not parts.netloc:
            continue
        new_host = hosts[index % len(hosts)]
        if parts.netloc == new_host:
            continue
        status.access_url = urlunsplit(
            (parts.scheme or "https", new_host, parts.path, parts.query, parts.fragment)
        )
