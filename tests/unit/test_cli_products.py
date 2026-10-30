"""The products commands (MADIV-28 T8.2: split from test_cli.py)."""

from __future__ import annotations

import re
import tarfile
from types import SimpleNamespace

import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from almasim import cli
from almasim.cli import products as cli_products
from almasim.services.archive import pipeline as archive_pipeline

runner = CliRunner()


def test_products_resolve_extracts_member_uids(tmp_path, monkeypatch):
    """Resolve should extract member_ous_uid values and write the list file."""
    metadata_csv = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "member_ous_uid": ["uid://A", "uid://B", "uid://A", None],
            "Band": [6, 7, 6, 3],
        }
    ).to_csv(metadata_csv, index=False)

    captured: dict[str, object] = {}

    def _fake_resolve(**kwargs):
        captured.update(kwargs)
        return [SimpleNamespace(), SimpleNamespace()]

    monkeypatch.setattr(cli_products, "_resolve_products_from_inputs", _fake_resolve)

    uid_list_path = tmp_path / "uids.txt"
    products_csv = tmp_path / "products.csv"
    result = runner.invoke(
        cli.app,
        [
            "products",
            "resolve",
            "--metadata-csv",
            str(metadata_csv),
            "--save-member-ous-uid-list",
            str(uid_list_path),
            "--save-products-csv",
            str(products_csv),
        ],
    )

    assert result.exit_code == 0
    assert uid_list_path.read_text(encoding="utf-8") == "uid://A\nuid://B\n"
    assert captured["member_ous_uid"] == ["uid://A", "uid://B"]


def test_products_download_invalid_filter_exits():
    """Invalid product filter should fail fast before resolution/download calls."""
    result = runner.invoke(
        cli.app,
        [
            "products",
            "download",
            "--member-ous-uid",
            "uid://A",
            "--product-filter",
            "not-a-type",
        ],
    )

    assert result.exit_code == 2
    assert "'not-a-type' is not one of 'all'" in " ".join(
        re.sub(r"\x1b\[[0-9;]*m", "", result.output).replace("│", " ").split()
    )


def test_products_download_slurm_postprocess_path(tmp_path, monkeypatch):
    """Slurm mode should run download first, then post-process archive jobs."""
    products = [SimpleNamespace(content_length=10)]

    monkeypatch.setattr(cli_products, "_resolve_products_from_inputs", lambda **kwargs: products)
    monkeypatch.setattr(cli_products, "filter_products", lambda products, product_filter: products)

    destination = tmp_path / "downloads"
    captured: dict[str, object] = {}

    def _fake_download_products(*args, **kwargs):
        captured.update(kwargs)
        assert callable(kwargs["update_callback"])
        destination.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            destination=str(destination),
            files_completed=1,
            files_failed=0,
            manifest_path=str(destination / "download_manifest.json"),
        )

    monkeypatch.setattr(cli_products, "download_products", _fake_download_products)
    monkeypatch.setattr(
        cli_products,
        "_run_parallel_archive_jobs",
        lambda **kwargs: (["/tmp/raw.ms"], ["/tmp/cal.ms"]),
    )

    result = runner.invoke(
        cli.app,
        [
            "products",
            "download",
            "--member-ous-uid",
            "uid://A",
            "--yes",
            "--postprocess-backend",
            "slurm",
            "--unpack-ms",
            "--generate-calibrated-visibilities",
        ],
    )

    assert result.exit_code == 0
    assert callable(captured["update_callback"])
    assert "Raw MS products:" in result.output
    assert "Calibrated MS products:" in result.output


def test_products_download_sync_path(tmp_path, monkeypatch):
    """Sync mode should download first, then run per-UID archive stages like Slurm does."""
    products = [SimpleNamespace(content_length=12)]
    monkeypatch.setattr(cli_products, "_resolve_products_from_inputs", lambda **kwargs: products)
    monkeypatch.setattr(cli_products, "filter_products", lambda products, product_filter: products)

    captured: dict[str, object] = {}
    stage_kwargs: dict[str, object] = {}

    def _fake_download_products(*args, **kwargs):
        captured.update(kwargs)
        assert callable(kwargs["update_callback"])
        return SimpleNamespace(
            destination=str(tmp_path / "downloads"),
            files_completed=1,
            files_failed=0,
            manifest_path=None,
            raw_measurement_sets=[],
            calibrated_measurement_sets=[],
        )

    def _fake_stages(**kwargs):
        stage_kwargs.update(kwargs)
        return (["/tmp/raw-a.ms"], [])

    monkeypatch.setattr(cli_products, "download_products", _fake_download_products)
    monkeypatch.setattr(cli_products, "_run_parallel_archive_jobs", _fake_stages)
    organized: list[tuple] = []
    monkeypatch.setattr(cli_products, "_organize_delivery", lambda *roots: organized.append(roots))

    result = runner.invoke(
        cli.app,
        [
            "products",
            "download",
            "--member-ous-uid",
            "uid://A",
            "--yes",
            "--postprocess-backend",
            "sync",
            "--unpack-ms",
        ],
    )

    assert result.exit_code == 0
    assert callable(captured["update_callback"])
    # In-process unpack/calibrate inside download_products is no longer used.
    assert captured["unpack_ms"] is False
    assert captured["generate_calibrated_visibilities"] is False
    assert stage_kwargs["postprocess_backend"] == "sync"
    assert stage_kwargs["postprocess_backend_kwargs"] == {}
    assert stage_kwargs["unpack_ms"] is True
    assert stage_kwargs["continue_on_error"] is True
    assert "Raw MS products:" in result.output
    ((delivery_root, calibrated_root, raw_root),) = organized  # the download is the tree
    assert delivery_root == tmp_path / "downloads"
    assert (calibrated_root.name, raw_root.name) == ("calibrated_ms", "raw_ms")


def test_products_download_sync_without_postprocess_keeps_single_call(tmp_path, monkeypatch):
    """Plain downloads (no unpack/calibrate) still go through download_products alone."""
    products = [SimpleNamespace(content_length=12)]
    monkeypatch.setattr(cli_products, "_resolve_products_from_inputs", lambda **kwargs: products)
    monkeypatch.setattr(cli_products, "filter_products", lambda products, product_filter: products)
    monkeypatch.setattr(
        cli_products,
        "_run_parallel_archive_jobs",
        lambda **kwargs: pytest.fail("no archive stages expected"),
    )
    monkeypatch.setattr(
        cli_products,
        "download_products",
        lambda *args, **kwargs: SimpleNamespace(
            destination=str(tmp_path / "downloads"),
            files_completed=1,
            files_failed=0,
            manifest_path=None,
            raw_measurement_sets=[],
            calibrated_measurement_sets=[],
        ),
    )

    result = runner.invoke(
        cli.app, ["products", "download", "--member-ous-uid", "uid://A", "--yes"]
    )
    assert result.exit_code == 0


def test_products_download_picks_files_inside_the_archives(tmp_path, monkeypatch):
    """--expand-archives lists the filtered tars' files; --include/--exclude pick among them."""
    from almasim.services.download import DataProduct

    def product(filename, semantics, product_type):
        return DataProduct(
            access_url=f"https://example.org/{filename}",
            uid="uid://A",
            filename=filename,
            content_length=5,
            content_type="",
            product_type=product_type,
            semantics=semantics,
        )

    tars = [
        product("P_001_of_001.tar", "#this", "fits"),
        product("P_auxiliary.tar", "#auxiliary", "auxiliary"),
    ]
    files = [
        product(name, "#this", "fits")
        for name in ("m.cube.I.pbcor.fits", "m.mfs.I.pbcor.fits", "m.cube.I.pb.fits.gz")
    ]
    expanded: list[list[str]] = []

    def fake_expand(products, layout):
        assert layout == "delivery"
        expanded.append([p.filename for p in products])
        return files

    downloaded: list[str] = []

    def fake_download(products, destination, **kwargs):
        downloaded.extend(p.filename for p in products)
        return SimpleNamespace(
            destination=str(destination),
            files_completed=len(products),
            files_failed=0,
            manifest_path=None,
            raw_measurement_sets=[],
            calibrated_measurement_sets=[],
        )

    monkeypatch.setattr(cli_products, "_resolve_products_from_inputs", lambda **kwargs: tars)
    monkeypatch.setattr(cli_products, "expand_archives", fake_expand)
    monkeypatch.setattr(cli_products, "download_products", fake_download)

    result = runner.invoke(
        cli.app,
        [
            "products", "download", "--member-ous-uid", "uid://A",
            "--product-filter", "fits", "--expand-archives",
            "--include", "*.pbcor.fits", "--exclude", "*.cube.*",
            "--destination", str(tmp_path), "--yes",
        ],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert expanded == [["P_001_of_001.tar"]]  # only the tars the filter kept
    assert downloaded == ["m.mfs.I.pbcor.fits"]
    assert "Listing the files inside 1 archive(s)" in result.output

    result = runner.invoke(
        cli.app,
        [
            "products", "download", "--member-ous-uid", "uid://A",
            "--include", "*.png", "--destination", str(tmp_path), "--yes",
        ],
    )  # fmt: skip
    assert result.exit_code != 0
    assert "--include=*.png" in result.output


def test_products_resolve_saves_the_expanded_listing(tmp_path, monkeypatch):
    """resolve --expand-archives saves one row per file inside the tars."""
    from almasim.services.download import DataProduct

    tar = DataProduct("https://example.org/P.tar", "uid://A", "P.tar", 9, "", "fits", "#this")
    inner = DataProduct("https://example.org/m.fits", "uid://A", "m.pbcor.fits", 4, "", "fits")
    monkeypatch.setattr(cli_products, "resolve_products", lambda uids: [tar])
    monkeypatch.setattr(cli_products, "expand_archives", lambda products, layout: [inner])
    products_csv = tmp_path / "products.csv"

    result = runner.invoke(
        cli.app,
        [
            "products", "resolve", "--member-ous-uid", "uid://A",
            "--expand-archives", "--save-products-csv", str(products_csv),
        ],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert pd.read_csv(products_csv)["filename"].tolist() == ["m.pbcor.fits"]


def test_products_download_skip_qa0_semipass_filters_raw_products(tmp_path, monkeypatch):
    """--skip-qa0-semipass drops SemiPass raw ASDMs and reports what it skipped."""
    from almasim.services.archive import qa0

    member = "uid://A001/X378a/X149"
    member_dir = (
        tmp_path
        / "downloads"
        / "2023.1.00879.S"
        / "science_goal.uid___A001_X378a_X147"
        / "group.uid___A001_X378a_X148"
        / "member.uid___A001_X378a_X149"
        / "qa"
    )
    member_dir.mkdir(parents=True)
    for eb in ("uid___A002_X1_Xpass", "uid___A002_X1_Xsemi"):
        (member_dir / f"{eb}.qa0_report.pdf").write_bytes(b"%PDF-1.4")

    def fake_reader(path, eb_uid=""):
        status = "SemiPass" if eb_uid.endswith("Xsemi") else "Pass"
        return qa0.QA0Report(eb_uid, status, "SUCCESS", 1.0, "")

    monkeypatch.setattr(qa0, "read_qa0_report", fake_reader)

    raw_pass = SimpleNamespace(
        product_type="raw",
        uid=member,
        filename="2023.1.00879.S_uid___A002_X1_Xpass.asdm.sdm.tar",
        content_length=10,
    )
    raw_semi = SimpleNamespace(
        product_type="raw",
        uid=member,
        filename="2023.1.00879.S_uid___A002_X1_Xsemi.asdm.sdm.tar",
        content_length=2048,
    )
    monkeypatch.setattr(
        cli_products, "_resolve_products_from_inputs", lambda **kwargs: [raw_pass, raw_semi]
    )
    captured: dict[str, object] = {}

    def fake_download(products, *args, **kwargs):
        captured["products"] = list(products)
        return SimpleNamespace(
            destination=str(tmp_path / "downloads"),
            files_completed=1,
            files_failed=0,
            manifest_path=None,
            raw_measurement_sets=[],
            calibrated_measurement_sets=[],
        )

    monkeypatch.setattr(cli_products, "download_products", fake_download)

    result = runner.invoke(
        cli.app,
        [
            "products",
            "download",
            "--member-ous-uid",
            member,
            "--product-filter",
            "raw",
            "--destination",
            str(tmp_path / "downloads"),
            "--skip-qa0-semipass",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "QA0 filter: skipped 1 SemiPass raw product(s)" in result.output
    assert captured["products"] == [raw_pass]


def test_products_download_sync_cleanup_only_when_no_failures(tmp_path, monkeypatch):
    """--clean-intermediate-files runs after a fully successful sync post-processing run."""
    products = [SimpleNamespace(content_length=12)]
    monkeypatch.setattr(cli_products, "_resolve_products_from_inputs", lambda **kwargs: products)
    monkeypatch.setattr(cli_products, "filter_products", lambda products, product_filter: products)
    manifest = tmp_path / "downloads" / "download_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"raw_measurement_sets": [], "calibrated_measurement_sets": []}')
    monkeypatch.setattr(
        cli_products,
        "download_products",
        lambda *args, **kwargs: SimpleNamespace(
            destination=str(tmp_path / "downloads"),
            files_completed=1,
            files_failed=0,
            manifest_path=str(manifest),
            files=[],
            extracted_files=[],
        ),
    )
    cleaned: list[str] = []
    monkeypatch.setattr(
        cli_products, "_cleanup_after_download_postprocess", lambda **kwargs: cleaned.append("yes")
    )

    def _stages_ok(**kwargs):
        return (["/raw/a.ms"], ["/cal/a.ms.split.cal"])

    def _stages_failing(**kwargs):
        kwargs["failures"].append(cli_products.TaskFailure("uid___B", "boom"))
        return (["/raw/a.ms"], ["/cal/a.ms.split.cal"])

    args = [
        "products",
        "download",
        "--member-ous-uid",
        "uid://A",
        "--yes",
        "--unpack-ms",
        "--generate-calibrated-visibilities",
        "--clean-intermediate-files",
    ]

    monkeypatch.setattr(cli_products, "_run_parallel_archive_jobs", _stages_ok)
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0
    assert cleaned == ["yes"]
    import json

    assert json.loads(manifest.read_text())["calibrated_measurement_sets"] == [
        "/cal/a.ms.split.cal"
    ]

    monkeypatch.setattr(cli_products, "_run_parallel_archive_jobs", _stages_failing)
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1
    assert cleaned == ["yes"]  # not called a second time
    assert "Skipping --clean-intermediate-files" in result.output


def test_products_cli_uses_the_download_service(monkeypatch):
    """The CLI uses the download service's limits and functions, not copies of them."""
    import almasim.services.compute as compute_mod
    import almasim.services.download as download

    monkeypatch.setattr(compute_mod, "create_backend", lambda *args, **kwargs: (args, kwargs))
    assert cli_products.create_backend("slurm", n_workers=2) == (("slurm",), {"n_workers": 2})
    assert cli_products.MAX_PARALLEL_TOTAL == download.MAX_PARALLEL_TOTAL
    assert cli_products.MAX_PARALLEL_PER_MIRROR == download.MAX_PARALLEL_PER_MIRROR
    assert cli_products.PRODUCT_TYPES is download.PRODUCT_TYPES
    for name in (
        "download_products",
        "filter_products",
        "format_bytes",
        "load_products_csv",
        "resolve_products",
        "save_products_csv",
    ):
        assert getattr(cli_products, name) is getattr(download, name)


def test_products_extract_standalone(tmp_path):
    """Standalone extract should unpack tar archives from disk."""
    source_root = tmp_path / "downloads"
    source_root.mkdir(parents=True)
    payload = source_root / "data.txt"
    payload.write_text("hello", encoding="utf-8")
    archive_path = source_root / "bundle.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(payload, arcname="nested/data.txt")
    payload.unlink()

    result = runner.invoke(
        cli.app,
        [
            "products",
            "extract",
            "--source-root",
            str(source_root),
            "--no-recursive",
        ],
    )

    assert result.exit_code == 0
    assert (source_root / "nested" / "data.txt").is_file()
    assert "Extracted files: 1" in result.output


def test_products_unpack_standalone(monkeypatch):
    """Standalone unpack should delegate to unpack job runner and print outputs."""
    monkeypatch.setattr(
        cli_products,
        "_run_unpack_jobs",
        lambda **kwargs: ["/tmp/raw-a.ms", "/tmp/raw-b.ms"],
    )

    result = runner.invoke(
        cli.app,
        [
            "products",
            "unpack",
            "--asdm-uid",
            "uid://A,uid://B",
        ],
    )

    assert result.exit_code == 0
    assert "Raw MS products: 2" in result.output


def test_products_calibrate_standalone(monkeypatch):
    """Standalone calibrate should delegate to calibrate job runner and print outputs."""
    monkeypatch.setattr(
        cli_products,
        "_run_calibrate_jobs",
        lambda **kwargs: ["/tmp/cal-a.ms"],
    )

    result = runner.invoke(
        cli.app,
        [
            "products",
            "calibrate",
            "--asdm-uid",
            "uid://A",
        ],
    )

    assert result.exit_code == 0
    assert "Calibrated MS products: 1" in result.output


def test_products_calibrate_exits_nonzero_with_failure_summary(monkeypatch):
    """The command should list outputs, print skipped UIDs, and exit 1."""

    def _fake_run(**kwargs):
        kwargs["failures"].append(
            cli_products.TaskFailure("uid___B", "Calibration failed for uid___B.", "/logs/b.log")
        )
        assert kwargs["continue_on_error"] is True
        return ["/tmp/cal-a.ms"]

    monkeypatch.setattr(cli_products, "_run_calibrate_jobs", _fake_run)

    result = runner.invoke(cli.app, ["products", "calibrate", "--asdm-uid", "uid___A,uid___B"])

    assert result.exit_code == 1
    assert "Calibrated MS products: 1" in result.output
    assert "1 UID(s) failed and were skipped" in result.output
    assert "uid___B" in result.output
    assert "/logs/b.log" in result.output


def test_products_calibrate_fail_fast_flag_passthrough(monkeypatch):
    """--fail-fast should disable continue_on_error."""
    captured: dict[str, object] = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli_products, "_run_calibrate_jobs", _fake_run)
    result = runner.invoke(cli.app, ["products", "calibrate", "--fail-fast"])

    assert result.exit_code == 0
    assert captured["continue_on_error"] is False


def test_products_calibrate_keep_working_copies_passthrough(monkeypatch):
    """--keep-working-copies reaches the job runner."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_products, "_run_calibrate_jobs", lambda **kwargs: captured.update(kwargs) or []
    )

    assert runner.invoke(cli.app, ["products", "calibrate"]).exit_code == 0
    assert captured["keep_working_copies"] is False
    assert runner.invoke(cli.app, ["products", "calibrate", "--keep-working-copies"]).exit_code == 0
    assert captured["keep_working_copies"] is True


def test_stage_errors_map_to_exit_codes():
    """StageError exits 1 and StageUsageError exits 2, with their messages."""
    with pytest.raises(typer.Exit) as info, cli_products._stage_errors():
        raise archive_pipeline.StageError("nothing to do")
    assert info.value.exit_code == 1
    with pytest.raises(typer.Exit) as info, cli_products._stage_errors():
        raise archive_pipeline.StageUsageError("bad combination")
    assert info.value.exit_code == 2
