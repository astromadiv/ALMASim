"""Download a list of products, then extract, unpack, calibrate and clean up. (MADIV-28 T6.9)"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from .models import DataProduct, DownloadSummary, FileDownloadStatus, format_bytes
from .transfer import (
    _RATE_LIMIT_SEC,
    MAX_PARALLEL_PER_MIRROR,
    MAX_PARALLEL_TOTAL,
    _distribute_across_mirrors,
    _download_single_file,
    _mirror_hosts,
)

logger = logging.getLogger(__name__)


def _extract_tar(tar_path: Path, destination: Path) -> list[str]:
    """Extract a downloaded tar under ``destination`` and delete the tar.

    The extraction is atomic at the directory level (see
    :mod:`almasim.services.extraction`): an ASDM never appears under its final
    name until every member is on disk, so an unpack job running alongside the
    download cannot pick up a half-written ``<uid>.asdm.sdm``. Returns the
    extracted member names relative to ``destination``; raises on failure with
    the tar left in place for a retry.
    """
    from almasim.services.extraction import extract_tar_atomically

    paths = extract_tar_atomically(
        tar_path,
        destination,
        flat_archive_subdir=False,
        on_skipped_member=lambda name: logger.warning("Skipping unsafe tar member: %s", name),
    )
    tar_path.unlink(missing_ok=True)
    return [str(path.relative_to(destination)) for path in paths]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _cleanup_download_inputs(  # noqa: C901
    destination: Path,
    statuses: Iterable[FileDownloadStatus],
    extracted_files: Iterable[str],
    *,
    protected_roots: Iterable[Path],
) -> None:
    """Remove downloaded archive/source products while preserving generated outputs."""
    protected = [root.resolve() for root in protected_roots if root.exists()]

    def is_protected(path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved == root or _is_relative_to(resolved, root) for root in protected)

    for status in statuses:
        for path in (
            destination / status.filename,
            destination / (status.filename + ".part"),
        ):
            if is_protected(path):
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.is_file():
                    path.unlink()
            except OSError:
                logger.warning("Failed to remove downloaded input %s", path)

    extracted_paths = sorted(
        ((destination / relative).resolve() for relative in extracted_files),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in extracted_paths:
        if is_protected(path):
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        except OSError:
            logger.warning("Failed to remove extracted input %s", path)

    for path in extracted_paths:
        parent = path.parent
        while parent != destination and _is_relative_to(parent, destination):
            if is_protected(parent):
                break
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def download_products(  # noqa: C901
    products: list[DataProduct],
    destination: Path | str,
    *,
    max_parallel: int = 3,
    rate_limit_sec: float = _RATE_LIMIT_SEC,
    extract_tar: bool = False,
    unpack_ms: bool = False,
    generate_calibrated_visibilities: bool = False,
    clean_intermediate_files: bool = False,
    archive_output_root: Path | str | None = None,
    casa_data_root: Path | str | None = None,
    skip_casa_data_update: bool = False,
    logger_fn: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    update_callback: Callable[[FileDownloadStatus], None] | None = None,
) -> DownloadSummary:
    """Download resolved products into a local directory."""
    destination_path = Path(destination).expanduser().resolve()
    destination_path.mkdir(parents=True, exist_ok=True)

    statuses = [
        FileDownloadStatus(
            filename=product.relative_path,
            access_url=product.access_url,
            content_length=product.content_length,
        )
        for product in products
    ]
    _distribute_across_mirrors(statuses)
    extracted_files: list[str] = []
    extraction_failed: list[str] = []
    raw_measurement_sets: list[str] = []
    calibrated_measurement_sets: list[str] = []

    # Spread requests across the ALMA mirrors but never exceed the per-mirror
    # cap recommended by the archive (10 concurrent transfers per TAP host).
    effective_parallel = max(1, min(int(max_parallel), MAX_PARALLEL_TOTAL))
    host_semaphores: dict[str, threading.BoundedSemaphore] = {
        host: threading.BoundedSemaphore(MAX_PARALLEL_PER_MIRROR) for host in _mirror_hosts()
    }

    with ThreadPoolExecutor(max_workers=effective_parallel) as pool:
        futures = []
        for index, status in enumerate(statuses):
            if should_cancel is not None and should_cancel():
                status.status = "cancelled"
                if update_callback is not None:
                    update_callback(status)
                continue
            futures.append(
                pool.submit(
                    _download_single_file,
                    status,
                    destination_path,
                    should_cancel=should_cancel,
                    update_callback=update_callback,
                    host_semaphores=host_semaphores,
                )
            )
            if index < len(statuses) - 1 and rate_limit_sec > 0:
                time.sleep(rate_limit_sec)

        for future in as_completed(futures):
            future.result()

    # The per-file error otherwise lives only in the manifest's "files" list.
    for status in statuses:
        if status.status != "failed":
            continue
        message = (
            f"Download failed: {status.filename} from {urlsplit(status.access_url).netloc} "
            f"after {status.bytes_downloaded} of {status.content_length} bytes: "
            f"{(status.error or 'unknown error').splitlines()[0]}"
        )
        logger.error(message)
        if logger_fn is not None:
            logger_fn(message)

    if extract_tar:
        for status in statuses:
            if status.status != "completed":
                continue
            if not (status.filename.endswith(".tar") or status.filename.endswith(".tgz")):
                continue
            tar_path = destination_path / status.filename
            if not tar_path.exists():
                continue
            try:
                extracted_files.extend(_extract_tar(tar_path, destination_path))
            except Exception as exc:  # noqa: BLE001 - logged; the other tars go on
                # Keep going: the other tars are fine and the failed one is
                # still on disk. But say so loudly — a download that "completed"
                # with an unextracted tar is exactly the kind of silent gap the
                # unpack stage then trips over.
                logger.error("Failed to extract %s: %s", tar_path.name, exc)
                if logger_fn is not None:
                    logger_fn(f"Failed to extract {tar_path.name}: {exc}")
                extraction_failed.append(str(tar_path))

    archive_failures: list[str] = []
    if unpack_ms or generate_calibrated_visibilities:
        from almasim.services.archive import cleanup_intermediate_calibration_data
        from almasim.services.archive.pipeline import run_archive_stages

        archive_root = (
            Path(archive_output_root).expanduser().resolve()
            if archive_output_root
            else destination_path / "archive_ms"
        )
        raw_ms_root = archive_root / "raw_ms"
        calibrated_ms_root = archive_root / "calibrated_ms"

        def _echo(message: str, *, err: bool = False) -> None:
            if logger_fn is not None:
                logger_fn(message)
            else:
                (logger.error if err else logger.info)(message)

        # One subprocess per UID (MADIV-28 T3.5): a CASA crash on one
        # execution block is a recorded failure, not the end of this process.
        stages = run_archive_stages(
            destination_path,
            archive_root,
            unpack_ms=True,
            calibrate=generate_calibrated_visibilities,
            casa_data_root=Path(casa_data_root) if casa_data_root else None,
            skip_casa_data_update=skip_casa_data_update,
            echo=_echo,
        )
        raw_measurement_sets = list(stages.raw_ms)
        calibrated_measurement_sets = list(stages.calibrated_ms)
        archive_failures = [f"{f.label}: {f.error}" for f in stages.failures]
        if clean_intermediate_files and generate_calibrated_visibilities and stages.ok:
            cleanup_intermediate_calibration_data(
                raw_ms_root,
                calibrated_ms_root,
                [Path(path) for path in calibrated_measurement_sets],
                logger_fn=logger_fn,
            )

    files_completed = sum(1 for status in statuses if status.status == "completed")
    files_failed = sum(1 for status in statuses if status.status == "failed")
    total_bytes = sum(product.content_length for product in products)
    manifest_path = destination_path / "download_manifest.json"

    manifest_path.write_text(
        json.dumps(
            {
                "destination": str(destination_path),
                "products": [asdict(product) for product in products],
                "files": [asdict(status) for status in statuses],
                "extracted_files": extracted_files,
                "extraction_failed": extraction_failed,
                "raw_measurement_sets": raw_measurement_sets,
                "calibrated_measurement_sets": calibrated_measurement_sets,
                "archive_failures": archive_failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if archive_failures:
        raise RuntimeError(
            f"Archive post-processing failed for {len(archive_failures)} UID(s): "
            + "; ".join(archive_failures)
            + f" (manifest: {manifest_path})"
        )

    if clean_intermediate_files and generate_calibrated_visibilities:
        _cleanup_download_inputs(
            destination_path,
            statuses,
            extracted_files,
            protected_roots=[
                archive_root,
                *(Path(path) for path in calibrated_measurement_sets),
            ],
        )

    if logger_fn is not None:
        logger_fn(
            f"Downloaded {files_completed}/{len(statuses)} files "
            f"to {destination_path} ({format_bytes(total_bytes)})"
        )

    return DownloadSummary(
        destination=str(destination_path),
        total_files=len(products),
        total_bytes=total_bytes,
        files_completed=files_completed,
        files_failed=files_failed,
        files=statuses,
        extracted_files=extracted_files,
        extraction_failed=extraction_failed,
        raw_measurement_sets=raw_measurement_sets,
        calibrated_measurement_sets=calibrated_measurement_sets,
        manifest_path=str(manifest_path),
    )
