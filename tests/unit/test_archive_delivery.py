"""The ALMA delivery tree: the MSs in their members' folders, and its index."""

from __future__ import annotations

import csv
import os

from typer.testing import CliRunner

from almasim import cli
from almasim.cli import products as cli_products
from almasim.services.archive.calibration.cleanup import cleanup_intermediate_calibration_data
from almasim.services.archive.calibration.discovery import find_raw_ms_directories
from almasim.services.archive.calibration.split import (
    calibrated_output_path,
    calibration_marker_path,
)
from almasim.services.archive.delivery import (
    archive_ms_roots,
    eb_member_folder,
    folder_uid,
    is_delivery_root,
    member_folders,
    write_ous_index,
)
from almasim.services.archive.unpack import raw_ms_marker_path, raw_ms_path

runner = CliRunner()

GOAL = "science_goal.uid___A001_X1_X1"
GROUP = "group.uid___A001_X1_X2"


def _member(root, member, qa_ebs=(), calapply_ebs=()):
    path = root / "2023.1.00001.S" / GOAL / GROUP / member
    (path / "qa").mkdir(parents=True)
    (path / "calibration").mkdir()
    for eb in qa_ebs:
        (path / "qa" / f"{eb}.qa0_report.pdf").write_bytes(b"%PDF")
    for eb in calapply_ebs:
        (path / "calibration" / f"{eb}.ms.calapply.txt").write_text("applycal()")
    return path


def _ms(folder, name):
    (folder / name).mkdir(parents=True)
    return folder / name


def _tree(tmp_path):
    root = tmp_path / "deliveries"
    first = _member(root, "member.uid___A001_X1_X3", ["uid___A002_X9_X1"], ["uid___A002_X9_X2"])
    second = _member(root, "member.uid___A001_X1_X4", ["uid___A002_X9_X3"])
    calibrated = tmp_path / "archive_ms" / "calibrated_ms"
    for eb in ("uid___A002_X9_X1", "uid___A002_X9_X2", "uid___A002_X9_X3", "uid___A002_X9_X8"):
        _ms(calibrated, f"{eb}.ms.split.cal")
    (calibrated / "uid___A002_X9_X1.ms.split.cal.done").write_text("ok")
    _ms(calibrated, "uid___A002_X9_X1.ms.split.cal.working")  # neither is an MS to link
    (calibrated / "logs").mkdir()
    raw = tmp_path / "archive_ms" / "raw_ms"
    _ms(raw, "uid___A002_X9_X1.ms")
    return root, first, second, calibrated, raw


def test_folder_names_give_the_ous_uids():
    assert folder_uid("member.uid___A001_X3621_X1f9d") == "uid://A001/X3621/X1f9d"
    assert folder_uid("science_goal.uid___A001_X3621_X1f9b") == "uid://A001/X3621/X1f9b"


def test_each_execution_block_is_placed_by_its_qa0_report_or_calapply_file(tmp_path):
    root, first, second, _, _ = _tree(tmp_path)

    members = member_folders(root)

    assert {eb: m.path for eb, m in members.items()} == {
        "uid___A002_X9_X1": first,
        "uid___A002_X9_X2": first,
        "uid___A002_X9_X3": second,
    }
    member = members["uid___A002_X9_X1"]
    assert (member.project_code, member.science_goal_uid, member.group_ous_uid) == (
        "2023.1.00001.S",
        "uid://A001/X1/X1",
        "uid://A001/X1/X2",
    )
    assert member.member_ous_uid == "uid://A001/X1/X3"


def test_an_asdm_in_raw_places_its_execution_block_too(tmp_path):
    root, first, _, _, _ = _tree(tmp_path)
    _ms(first / "raw", "uid___A002_X9_X5.asdm.sdm")
    assert member_folders(root)["uid___A002_X9_X5"].path == first


def test_the_stages_write_each_ms_into_its_member_folder(tmp_path):
    root, first, second, _, _ = _tree(tmp_path)
    assert is_delivery_root(root) and not is_delivery_root(tmp_path / "archive_ms")

    assert raw_ms_path(root, "uid___A002_X9_X1") == first / "raw" / "uid___A002_X9_X1.ms"
    assert (
        raw_ms_marker_path(root, "uid___A002_X9_X1") == first / "raw" / "uid___A002_X9_X1.ms.done"
    )
    assert calibrated_output_path(root, "uid___A002_X9_X3") == (
        second / "calibrated" / "uid___A002_X9_X3.ms.split.cal"
    )
    assert calibration_marker_path(root, "uid___A002_X9_X3").parent == second / "calibrated"
    # An EB no member lists, and any folder that is not a tree, keep the flat paths.
    assert (
        raw_ms_path(root, "uid___A002_X9_X8") == root.resolve() / "working" / "uid___A002_X9_X8.ms"
    )
    flat = tmp_path / "archive_ms" / "calibrated_ms"
    assert calibrated_output_path(flat, "uid___A002_X9_X1") == (
        flat.resolve() / "uid___A002_X9_X1.ms.split.cal"
    )

    assert archive_ms_roots(root, None) == (root.resolve(), root.resolve())
    archive = tmp_path / "ms"
    assert archive_ms_roots(root, archive) == (archive / "raw_ms", archive / "calibrated_ms")


def test_a_member_extracted_later_is_found(tmp_path, monkeypatch):
    root, _, _, _, _ = _tree(tmp_path)
    assert eb_member_folder(root, "uid___A002_X9_X1") is not None
    later = _member(root, "member.uid___A001_X1_X7", ["uid___A002_X9_X7"])
    monkeypatch.setattr("almasim.services.archive.delivery._RESCAN_SECONDS", 0.0)
    assert eb_member_folder(root, "uid___A002_X9_X7") == later.resolve()


def test_the_raw_mss_of_a_tree_are_found_in_the_members_raw_folders(tmp_path):
    root, first, second, _, _ = _tree(tmp_path)
    _ms(first / "raw", "uid___A002_X9_X1.ms")
    _ms(second / "raw", "uid___A002_X9_X3.ms")
    _ms(first / "calibrated", "uid___A002_X9_X2.ms")  # not a raw MS's place
    found = find_raw_ms_directories(root)
    assert [p.name for p in found] == ["uid___A002_X9_X1.ms", "uid___A002_X9_X3.ms"]


def test_the_index_lists_the_mss_in_the_tree_and_in_flat_folders(tmp_path):
    root, first, second, calibrated, raw = _tree(tmp_path)
    in_tree = _ms(second / "calibrated", "uid___A002_X9_X3.ms.split.cal")
    raw_in_tree = _ms(second / "raw", "uid___A002_X9_X3.ms")

    done = write_ous_index(root, calibrated, raw, echo=lambda message: None)

    assert done.execution_blocks == 3 and done.calibrated == 3
    assert [os.path.basename(p) for p in done.without_member] == ["uid___A002_X9_X8.ms.split.cal"]
    with done.index_path.open() as handle:
        rows = {row["eb_uid"]: row for row in csv.DictReader(handle)}
    assert done.index_path == root / "ous_index.csv"
    assert b"\r" not in done.index_path.read_bytes()
    assert [p.name for p in root.glob("ous_index.csv*")] == ["ous_index.csv"]
    assert set(rows) == {"uid___A002_X9_X1", "uid___A002_X9_X2", "uid___A002_X9_X3"}
    assert rows["uid___A002_X9_X1"]["member_ous_uid"] == "uid://A001/X1/X3"
    assert rows["uid___A002_X9_X1"]["member_folder"] == (
        f"2023.1.00001.S/{GOAL}/{GROUP}/member.uid___A001_X1_X3"
    )
    assert rows["uid___A002_X9_X1"]["raw_ms"] == str(raw / "uid___A002_X9_X1.ms")
    assert rows["uid___A002_X9_X2"]["raw_ms"] == ""
    assert rows["uid___A002_X9_X2"]["alma_calibrated"] == "yes", "it has a calapply file"
    assert rows["uid___A002_X9_X1"]["alma_calibrated"] == "no", "only a QA0 report"
    assert rows["uid___A002_X9_X3"]["calibrated_ms"] == str(in_tree), "the tree's MS first"
    assert rows["uid___A002_X9_X3"]["raw_ms"] == str(raw_in_tree)
    assert not (first / "calibrated").exists(), "nothing is linked or moved"


def test_cleanup_in_a_tree_removes_only_the_raw_mss(tmp_path):
    root, first, _, _, _ = _tree(tmp_path)
    raw = _ms(first / "raw", "uid___A002_X9_X1.ms")
    _ms(first / "raw", "uid___A002_X9_X1.ms.flagversions")
    (first / "raw" / "uid___A002_X9_X1.ms.done").write_text("{}")
    asdm = _ms(first / "raw", "uid___A002_X9_X1.asdm.sdm")
    calibrated = _ms(first / "calibrated", "uid___A002_X9_X1.ms.split.cal")
    _ms(root / "working", "uid___A002_X9_X1.calibration")

    cleanup_intermediate_calibration_data(root, root, [calibrated])

    assert not raw.exists() and not (first / "raw" / "uid___A002_X9_X1.ms.flagversions").exists()
    assert not (first / "raw" / "uid___A002_X9_X1.ms.done").exists()
    assert not (root / "working").exists()
    assert calibrated.is_dir() and asdm.is_dir() and (first / "qa").is_dir()


def test_calibrate_indexes_the_tree_and_writes_into_it_by_default(tmp_path, monkeypatch):
    root, _, _, _, _ = _tree(tmp_path)
    seen = {}

    def _calibrate(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(cli_products, "_run_calibrate_jobs", _calibrate)
    result = runner.invoke(cli.app, ["visibilities", "calibrate", "--input-root", str(root)])
    assert result.exit_code == 0, result.output
    assert seen["raw_ms_root"] == seen["output_root"] == root.resolve()
    assert (root / "ous_index.csv").is_file()

    (root / "ous_index.csv").unlink()
    result = runner.invoke(
        cli.app,
        ["visibilities", "calibrate", "--input-root", str(root), "--no-index-delivery"],
    )
    assert result.exit_code == 0, result.output
    assert not (root / "ous_index.csv").exists()


def test_unpack_writes_next_to_the_asdms_by_default(tmp_path, monkeypatch):
    root, _, _, _, _ = _tree(tmp_path)
    seen = {}

    def _unpack(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(cli_products, "_run_unpack_jobs", _unpack)
    result = runner.invoke(cli.app, ["visibilities", "unpack", "--input-root", str(root)])
    assert result.exit_code == 0, result.output
    assert seen["output_root"] == root.resolve()
