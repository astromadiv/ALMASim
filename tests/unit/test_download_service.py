"""Unit tests for the library-first download service."""

import io
import tarfile

from almasim.services.download import (
    DataProduct,
    datalink,
    download_products,
    expand_archives,
    filter_products,
    load_products_csv,
    save_products_csv,
    select_files,
)


def test_filter_products():
    """Filtering should preserve only matching product types."""
    products = [
        DataProduct(
            access_url="https://example.org/a.fits",
            uid="uid://A001/X1",
            filename="a.fits",
            content_length=10,
            content_type="application/fits",
            product_type="fits",
        ),
        DataProduct(
            access_url="https://example.org/b.tar",
            uid="uid://A001/X1",
            filename="b.tar",
            content_length=20,
            content_type="application/x-tar",
            product_type="raw",
        ),
    ]

    filtered = filter_products(products, "fits")

    assert len(filtered) == 1
    assert filtered[0].filename == "a.fits"


def _row(filename, semantics="#this", product_type="fits", uid="uid://A001/X1/X1"):
    return DataProduct(
        access_url=f"https://example.org/{filename}",
        uid=uid,
        filename=filename,
        content_length=10,
        content_type="application/octet-stream",
        product_type=product_type,
        semantics=semantics,
    )


def test_select_files_matches_globs_on_the_filename():
    """Include globs keep any match (all files when none); exclude globs drop matches."""
    products = [
        _row("m.spw23.cube.I.pbcor.fits"),
        _row("m.spw23.cube.I.pb.fits.gz"),
        _row("m.spw23.mfs.I.pbcor.fits"),
        _row("m.spw23.mfs.I.mask.fits.gz"),
    ]
    names = lambda rows: [row.filename for row in rows]  # noqa: E731

    assert names(select_files(products, ["*.pbcor.fits"])) == [
        "m.spw23.cube.I.pbcor.fits",
        "m.spw23.mfs.I.pbcor.fits",
    ]
    assert names(select_files(products, ["*.pbcor.fits"], ["*.cube.*"])) == [
        "m.spw23.mfs.I.pbcor.fits"
    ]
    assert names(select_files(products, exclude=["*.gz"])) == names(products)[::2]
    assert names(select_files(products, ["*.PBCOR.fits"])) == []  # case-sensitive
    assert names(select_files(products)) == names(products)


def test_expand_archives_lists_the_files_inside_the_tars(monkeypatch):
    """Product and auxiliary tars become their files, typed like the tar and kept on the member."""
    listings = {
        "P_001_of_001.tar": [
            _row("m.cube.I.pbcor.fits", uid="P_001_of_001.tar", product_type="other"),
            _row("m.cube.I.pb.fits.gz", uid="P_001_of_001.tar", product_type="cubes"),
        ],
        "P_auxiliary.tar": [
            _row("m.s1.caltables.tgz", "#auxiliary", "calibration", "P_auxiliary.tar"),
            _row("m.README.txt", "#documentation", "auxiliary", "P_auxiliary.tar"),
        ],
    }
    requested: list[str] = []

    def fake_fetch(uid, mirror_base):
        requested.append(uid)
        return listings[uid]

    monkeypatch.setattr(datalink, "_fetch_datalink", fake_fetch)
    products = [
        _row("m.README.txt", "#documentation", "auxiliary"),
        _row("P_001_of_001.tar"),
        _row("P_auxiliary.tar", "#auxiliary", "auxiliary"),
        _row("P_uid___A002_X1_X1.asdm.sdm.tar", "#progenitor", "raw"),
    ]

    expanded = expand_archives(products, layout="flat")

    assert sorted(requested) == ["P_001_of_001.tar", "P_auxiliary.tar"]
    assert [(row.filename, row.product_type) for row in expanded] == [
        ("m.README.txt", "auxiliary"),
        ("m.cube.I.pbcor.fits", "fits"),
        ("m.cube.I.pb.fits.gz", "fits"),
        ("m.s1.caltables.tgz", "auxiliary"),
        ("P_uid___A002_X1_X1.asdm.sdm.tar", "raw"),
    ]
    assert {row.uid for row in expanded} == {"uid://A001/X1/X1"}
    assert {row.path for row in expanded} == {""}


_MEMBER = (
    "2023.1.00882.S/science_goal.uid___A001_X3621_X1f9b/group.uid___A001_X3621_X1f9c/"
    "member.uid___A001_X3621_X1f9d"
)


def _tar_start(first_name: str) -> bytes:
    """The start of a tar whose first file is ``first_name`` (a GNU long name when long)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as archive:
        info = tarfile.TarInfo(first_name)
        info.size = 4096
        archive.addfile(info, io.BytesIO(b"x" * 4096))
    return buffer.getvalue()[:2048]  # the transfer stops early: the file data is cut off


def test_member_dir_is_read_from_the_tar_start():
    """The member folder comes from the tar's first path, long names and cut-off data included."""
    first = f"{_MEMBER}/product/member.uid___A001_X3621_X1f9d.J1037-2934_bp.spw23.mfs.I.pb.fits.gz"
    assert len(first) > 100
    assert datalink._member_dir_from_tar_head(_tar_start(first)) == _MEMBER
    assert datalink._member_dir_from_tar_head(_tar_start("../member.x/a.fits")) is None
    assert datalink._member_dir_from_tar_head(_tar_start("README.txt")) is None
    assert datalink._member_dir_from_tar_head(b"<html>maintenance</html>") is None


def test_listing_descriptions_name_the_delivery_folders():
    """A listed file's folder is the type its description names; logs go in log/."""
    folder = datalink._listing_folder
    assert folder("Download dataset of type: product, and class: SCIENCE.", "a.fits") == "product"
    assert folder("Download dataset of type: calibration, and class: CALIBRATION.", "a.tgz") == (
        "calibration"
    )
    assert folder("Download dataset of type: qa, and class: QA0.", "a.pdf") == "qa"
    assert folder("Download dataset of type: script, and class: PPR.", "a.xml") == "script"
    assert folder("", "member.x.hifa_calimage.casa_commands.log") == "log"
    assert folder("Download dataset of type: null, and class: N/A.", "x.tar") == ""


def test_expand_archives_puts_the_files_in_the_delivery_tree(monkeypatch):
    """layout="delivery": each file in its member folder, read once per member from a tar."""
    pbcor = _row("m.cube.I.pbcor.fits", uid="P_001_of_001.tar")
    qa2 = _row("m.qa2_report.pdf", "#auxiliary", "auxiliary", "P_auxiliary.tar")
    log = _row("m.casa_commands.log", "#documentation", "auxiliary", "P_auxiliary.tar")
    pbcor.path, qa2.path, log.path = "product", "qa", "log"
    listings = {"P_001_of_001.tar": [pbcor], "P_auxiliary.tar": [qa2, log]}
    heads: list[str] = []

    def fake_head(url):
        heads.append(url)
        return _tar_start(f"{_MEMBER}/product/m.cube.I.pbcor.fits")

    monkeypatch.setattr(datalink, "_fetch_datalink", lambda uid, mirror_base: listings[uid])
    monkeypatch.setattr(datalink, "_tar_head", fake_head)
    products = [
        _row("m.README.txt", "#documentation", "auxiliary"),
        _row("P_001_of_001.tar"),
        _row("P_auxiliary.tar", "#auxiliary", "auxiliary"),
        _row("P.asdm.sdm.tar", "#progenitor", "raw"),
    ]

    expanded = expand_archives(products)

    assert heads == ["https://example.org/P_001_of_001.tar"]
    assert [row.relative_path for row in expanded] == [
        f"{_MEMBER}/m.README.txt",
        f"{_MEMBER}/product/m.cube.I.pbcor.fits",
        f"{_MEMBER}/qa/m.qa2_report.pdf",
        f"{_MEMBER}/log/m.casa_commands.log",
        "P.asdm.sdm.tar",
    ]


def test_expand_archives_names_the_member_folder_when_the_tar_is_unreadable(monkeypatch):
    """Without a readable tar start, the files still go in a folder named after their member."""
    listed = _row("m.pbcor.fits", uid="P_001_of_001.tar")
    listed.path = "product"
    monkeypatch.setattr(datalink, "_fetch_datalink", lambda uid, mirror_base: [listed])
    monkeypatch.setattr(datalink, "_tar_head", lambda url: b"")

    (row,) = expand_archives([_row("P_001_of_001.tar")])

    assert row.relative_path == "member.uid___A001_X1_X1/product/m.pbcor.fits"


def test_resolved_tars_land_in_the_destination(monkeypatch):
    """A member's own DataLink rows (its tars, its README) keep no folder."""
    listed = _row("P_001_of_001.tar")
    listed.path = "product"
    monkeypatch.setattr(datalink, "_fetch_datalink", lambda uid, mirror_base: [listed])

    assert [row.path for row in datalink.resolve_products(["uid://A001/X1/X1"])] == [""]


def test_expand_archives_keeps_a_tar_it_cannot_list(monkeypatch):
    """A tar whose listing fails on every mirror is kept whole, so nothing is lost."""

    def failing_fetch(uid, mirror_base):
        raise OSError("mirror down")

    monkeypatch.setattr(datalink, "_fetch_datalink", failing_fetch)
    tar = _row("P_001_of_001.tar")

    assert expand_archives([tar], layout="flat") == [tar]


def test_products_csv_roundtrip(tmp_path):
    """Resolved products should round-trip through CSV."""
    products = [
        DataProduct(
            access_url="https://example.org/a.fits",
            uid="uid://A001/X1",
            filename="a.fits",
            content_length=10,
            content_type="application/fits",
            product_type="fits",
            semantics="alma#imagecube",
            path="P/member.uid___A001_X1_X1/product",
        )
    ]
    csv_path = tmp_path / "products.csv"

    save_products_csv(products, csv_path)
    loaded = load_products_csv(csv_path)

    assert len(loaded) == 1
    assert loaded[0].access_url == products[0].access_url
    assert loaded[0].uid == products[0].uid
    assert loaded[0].filename == products[0].filename
    assert loaded[0].product_type == products[0].product_type
    assert loaded[0].semantics == products[0].semantics
    assert loaded[0].path == products[0].path


def test_download_products_skips_existing_file(tmp_path):
    """Existing files should be treated as completed without network access."""
    destination = tmp_path / "downloads"
    destination.mkdir()
    file_path = destination / "existing.txt"
    file_path.write_bytes(b"already here")

    products = [
        DataProduct(
            access_url="https://example.org/existing.txt",
            uid="uid://A001/X1",
            filename="existing.txt",
            content_length=file_path.stat().st_size,
            content_type="text/plain",
            product_type="auxiliary",
        )
    ]

    summary = download_products(products, destination)

    assert summary.total_files == 1
    assert summary.files_completed == 1
    assert summary.files_failed == 0
    assert summary.files[0].status == "completed"
    assert summary.files[0].bytes_downloaded == file_path.stat().st_size


def test_download_products_puts_a_file_in_its_folder(tmp_path):
    """A product with a path lands in that folder under the destination."""
    folder = tmp_path / "downloads" / "P" / "member.uid___A001_X1_X1" / "product"
    folder.mkdir(parents=True)
    (folder / "m.pbcor.fits").write_bytes(b"image")
    product = _row("m.pbcor.fits")
    product.path = "P/member.uid___A001_X1_X1/product"
    product.content_length = 5

    summary = download_products([product], tmp_path / "downloads")

    assert summary.files_completed == 1
    assert summary.files[0].filename == "P/member.uid___A001_X1_X1/product/m.pbcor.fits"


# ---------------------------------------------------------------------------
# Truncated / short downloads must never be reported as completed
# ---------------------------------------------------------------------------

from contextlib import contextmanager  # noqa: E402

import pytest  # noqa: E402

from almasim.services.download import transfer as download_module  # noqa: E402


class _FakeResponse:
    def __init__(self, status_code: int, body: bytes, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_bytes(self, chunk_size: int):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


class _FakeClient:
    """Plays back a script of ``headers -> (status, body)`` per GET."""

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @contextmanager
    def stream(self, method, url, headers=None):
        self.requests.append(dict(headers or {}))
        status, body, *response_headers = self.script.pop(0)(headers or {})
        yield _FakeResponse(status, body, *response_headers)


def _install_fake_client(monkeypatch, script) -> _FakeClient:
    fake = _FakeClient(script)
    monkeypatch.setattr(download_module.httpx, "Client", lambda **kwargs: fake)
    monkeypatch.setattr(download_module.time, "sleep", lambda *_: None)
    return fake


def _asdm_tar_bytes(uid: str = "uid___A002_X1_X1") -> bytes:
    asdm = (
        f"2023.1.00001.S/science_goal.uid___A001_X1_X1/member.uid___A001_X1_X3/raw/{uid}.asdm.sdm"
    )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in (("ASDM.xml", b"<ASDM/>"), ("Main.xml", b"<Main/>")):
            info = tarfile.TarInfo(f"{asdm}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _product(filename: str, body: bytes) -> DataProduct:
    return DataProduct(
        access_url=f"https://example.org/{filename}",
        uid="uid://A001/X1",
        filename=filename,
        content_length=len(body),
        content_type="application/x-tar",
        product_type="raw",
    )


@pytest.mark.unit
def test_short_download_is_failed_not_completed(tmp_path, monkeypatch):
    """The server closes early; without a size check this became a truncated tar."""
    body = _asdm_tar_bytes()
    half = body[: len(body) // 2]
    monkeypatch.setattr(download_module, "_MAX_FILE_ATTEMPTS", 1)
    _install_fake_client(monkeypatch, [lambda h: (200, half)])

    summary = download_products(
        [_product("a.asdm.sdm.tar", body)], tmp_path, rate_limit_sec=0, extract_tar=True
    )

    assert summary.files_failed == 1
    assert summary.files_completed == 0
    assert "Short download" in (summary.files[0].error or "")
    assert not (tmp_path / "a.asdm.sdm.tar").exists()
    assert not list(tmp_path.rglob("*.asdm.sdm"))
    assert summary.extraction_failed == []


@pytest.mark.unit
def test_maintenance_page_is_reported_and_logged(tmp_path, monkeypatch):
    """The EA portal answered 200 with an HTML notice during its 2026-09-29 maintenance."""
    body = _asdm_tar_bytes()
    page = b"<H1>ALMA Science Portal Maintenance</H1>\nPortal is temporarily unavailable."
    monkeypatch.setattr(download_module, "_MAX_FILE_ATTEMPTS", 1)
    _install_fake_client(
        monkeypatch, [lambda h: (200, page, {"content-type": "text/html; charset=utf-8"})]
    )
    messages: list[str] = []

    summary = download_products(
        [_product("a.asdm.sdm.tar", body)],
        tmp_path,
        rate_limit_sec=0,
        extract_tar=True,
        logger_fn=messages.append,
    )

    assert summary.files_failed == 1
    error = summary.files[0].error or ""
    assert "HTML page" in error and "ALMA Science Portal Maintenance" in error
    assert not (tmp_path / "a.asdm.sdm.tar.part").exists(), "the page is not saved as data"
    failed = [m for m in messages if m.startswith("Download failed: a.asdm.sdm.tar")]
    assert len(failed) == 1
    assert " from almascience." in failed[0] and "Maintenance" in failed[0]


@pytest.mark.unit
def test_short_download_restarts_when_server_rejects_range(tmp_path, monkeypatch):
    """Attempt 1 is short; the resume gets a 416 (the ALMA portal); a fresh GET completes."""
    body = _asdm_tar_bytes()
    half = body[: len(body) // 2]

    def first(headers):
        assert "Range" not in headers
        return 200, half

    def resume(headers):
        assert headers.get("Range") == f"bytes={len(half)}-"
        return 416, b""

    def fresh(headers):
        assert "Range" not in headers
        return 200, body

    fake = _install_fake_client(monkeypatch, [first, resume, fresh])

    summary = download_products(
        [_product("a.asdm.sdm.tar", body)], tmp_path, rate_limit_sec=0, extract_tar=True
    )

    assert len(fake.requests) == 3
    assert summary.files_completed == 1
    assert summary.files[0].bytes_downloaded == len(body)
    asdms = list(tmp_path.rglob("*.asdm.sdm"))
    assert len(asdms) == 1 and (asdms[0] / "Main.xml").is_file()
    assert not (tmp_path / "a.asdm.sdm.tar").exists(), "tar is deleted after extraction"
    assert summary.extraction_failed == []


@pytest.mark.unit
def test_unextractable_tar_is_reported_and_kept(tmp_path, monkeypatch):
    """A full-size but corrupt tar: download completes, extraction failure is loud."""
    body = _asdm_tar_bytes()
    garbage = b"\xab" * len(body)  # right length, not a tar (all-zero *would* be one)
    _install_fake_client(monkeypatch, [lambda h: (200, garbage)])
    messages: list[str] = []

    summary = download_products(
        [_product("a.asdm.sdm.tar", body)],
        tmp_path,
        rate_limit_sec=0,
        extract_tar=True,
        logger_fn=messages.append,
    )

    assert summary.files_completed == 1
    assert summary.extraction_failed == [str(tmp_path / "a.asdm.sdm.tar")]
    assert (tmp_path / "a.asdm.sdm.tar").is_file(), "kept for a manual retry"
    assert not list(tmp_path.rglob("*.asdm.sdm"))
    assert not any(p.name.startswith(".extracting-") for p in tmp_path.iterdir())
    assert any("Failed to extract a.asdm.sdm.tar" in m for m in messages)
    import json

    manifest = json.loads((tmp_path / "download_manifest.json").read_text())
    assert manifest["extraction_failed"] == summary.extraction_failed


_MAINTENANCE_PAGE = b"<H1>ALMA Science Portal Maintenance</H1>\nWe regret the inconvenience.\n"


@pytest.mark.unit
def test_html_part_is_discarded_not_resumed(tmp_path, monkeypatch):
    """A .part holding a maintenance page (job 4029, 2026-09-29) must not be resumed."""
    body = _asdm_tar_bytes()
    (tmp_path / "a.asdm.sdm.tar.part").write_bytes(_MAINTENANCE_PAGE)

    def fresh(headers):
        assert "Range" not in headers, "resuming would append the tar after the page"
        return 200, body

    fake = _install_fake_client(monkeypatch, [fresh])

    summary = download_products(
        [_product("a.asdm.sdm.tar", body)], tmp_path, rate_limit_sec=0, extract_tar=True
    )

    assert len(fake.requests) == 1
    assert summary.files_completed == 1
    assert summary.extraction_failed == []
    assert len(list(tmp_path.rglob("*.asdm.sdm"))) == 1


@pytest.mark.unit
def test_right_sized_tar_without_header_is_downloaded_again(tmp_path, monkeypatch):
    """The page overwrote the first bytes of a full-size tar: size alone passes it."""
    body = _asdm_tar_bytes()
    corrupt = _MAINTENANCE_PAGE + body[len(_MAINTENANCE_PAGE) :]
    (tmp_path / "a.asdm.sdm.tar").write_bytes(corrupt)
    fake = _install_fake_client(monkeypatch, [lambda h: (200, body)])

    summary = download_products(
        [_product("a.asdm.sdm.tar", body)], tmp_path, rate_limit_sec=0, extract_tar=True
    )

    assert len(fake.requests) == 1
    assert summary.files_completed == 1
    assert summary.extraction_failed == []
    assert len(list(tmp_path.rglob("*.asdm.sdm"))) == 1


@pytest.mark.unit
def test_valid_existing_tar_is_not_downloaded_again(tmp_path, monkeypatch):
    body = _asdm_tar_bytes()
    (tmp_path / "a.asdm.sdm.tar").write_bytes(body)
    fake = _install_fake_client(monkeypatch, [])

    summary = download_products(
        [_product("a.asdm.sdm.tar", body)], tmp_path, rate_limit_sec=0, extract_tar=True
    )

    assert fake.requests == []
    assert summary.files_completed == 1
    assert len(list(tmp_path.rglob("*.asdm.sdm"))) == 1


def _existing_product(destination):
    destination.mkdir(exist_ok=True)
    path = destination / "existing.txt"
    path.write_bytes(b"already here")
    return [
        DataProduct(
            access_url="https://example.org/existing.txt",
            uid="uid://A001/X1",
            filename="existing.txt",
            content_length=path.stat().st_size,
            content_type="text/plain",
            product_type="auxiliary",
        )
    ]


def test_archive_stages_run_one_process_per_uid(tmp_path, monkeypatch):
    """Unpack and calibration go through run_archive_stages (MADIV-28 T3.5)."""
    import json

    from almasim.services.archive import pipeline

    seen = {}

    def _stages(download_root, archive_root, **kwargs):
        seen.update(kwargs, download_root=download_root, archive_root=archive_root)
        kwargs["echo"]("calibrating uid___A")
        return pipeline.ArchiveStageResult(["/raw/a.ms"], ["/cal/a.ms.split.cal"])

    monkeypatch.setattr(pipeline, "run_archive_stages", _stages)
    destination = tmp_path / "downloads"
    messages: list[str] = []
    summary = download_products(
        _existing_product(destination),
        destination,
        generate_calibrated_visibilities=True,
        logger_fn=messages.append,
    )
    assert "calibrating uid___A" in messages

    assert seen["unpack_ms"] is True and seen["calibrate"] is True
    assert seen["archive_root"] == destination / "archive_ms"
    assert summary.raw_measurement_sets == ["/raw/a.ms"]
    assert summary.calibrated_measurement_sets == ["/cal/a.ms.split.cal"]
    manifest = json.loads((destination / "download_manifest.json").read_text())
    assert manifest["archive_failures"] == []


def test_failed_archive_uids_fail_the_download_after_the_manifest(tmp_path, monkeypatch):
    import json

    import pytest

    from almasim.services.archive import pipeline
    from almasim.services.compute.progress import TaskFailure

    monkeypatch.setattr(
        pipeline,
        "run_archive_stages",
        lambda *a, **k: pipeline.ArchiveStageResult(
            ["/raw/a.ms"], [], [TaskFailure("uid___B", "Worker exited with return code -6", None)]
        ),
    )
    destination = tmp_path / "downloads"
    with pytest.raises(RuntimeError, match="failed for 1 UID"):
        download_products(_existing_product(destination), destination, unpack_ms=True)
    manifest = json.loads((destination / "download_manifest.json").read_text())
    assert manifest["archive_failures"] == ["uid___B: Worker exited with return code -6"]
    assert manifest["raw_measurement_sets"] == ["/raw/a.ms"]


def test_archive_stage_messages_and_cleanup(tmp_path, monkeypatch, caplog):
    """Without a logger_fn the stages log; a fully calibrated run cleans its intermediates."""
    import logging

    from almasim.services import archive
    from almasim.services.archive import pipeline

    def _stages(download_root, archive_root, **kwargs):
        kwargs["echo"]("unpacking")
        kwargs["echo"]("one UID failed", err=True)
        return pipeline.ArchiveStageResult(["/raw/a.ms"], ["/cal/a.ms.split.cal"])

    cleaned = {}
    monkeypatch.setattr(pipeline, "run_archive_stages", _stages)
    monkeypatch.setattr(
        archive,
        "cleanup_intermediate_calibration_data",
        lambda raw, cal, outputs, logger_fn=None: cleaned.update(raw=raw, outputs=outputs),
    )
    destination = tmp_path / "downloads"
    with caplog.at_level(logging.INFO, logger="almasim"):
        download_products(
            _existing_product(destination),
            destination,
            generate_calibrated_visibilities=True,
            clean_intermediate_files=True,
        )
    assert "unpacking" in caplog.text and "one UID failed" in caplog.text
    assert cleaned["raw"] == destination / "archive_ms" / "raw_ms"
    assert [str(p) for p in cleaned["outputs"]] == ["/cal/a.ms.split.cal"]
