"""The ALMA delivery tree: calibrated MSs linked into their members, and its index."""

from __future__ import annotations

import csv
import os

from typer.testing import CliRunner

from almasim import cli
from almasim.cli import products as cli_products
from almasim.services.archive.delivery import (
    folder_uid,
    member_folders,
    organize_delivery,
)

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


def test_calibrated_mss_are_linked_into_their_members_and_indexed(tmp_path):
    root, first, second, calibrated, raw = _tree(tmp_path)

    done = organize_delivery(root, calibrated, raw, echo=lambda message: None)

    link = first / "calibrated" / "uid___A002_X9_X1.ms.split.cal"
    assert link.is_symlink() and not os.path.isabs(os.readlink(link))
    assert link.resolve() == (calibrated / "uid___A002_X9_X1.ms.split.cal").resolve()
    assert (second / "calibrated" / "uid___A002_X9_X3.ms.split.cal").is_symlink()
    assert sorted(p.name for p in (first / "calibrated").iterdir()) == [
        "uid___A002_X9_X1.ms.split.cal",
        "uid___A002_X9_X2.ms.split.cal",
    ]
    assert done.linked == 3
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
    assert rows["uid___A002_X9_X3"]["calibrated_ms"] == str(
        calibrated / "uid___A002_X9_X3.ms.split.cal"
    )


def test_organizing_again_keeps_good_links_and_drops_dangling_ones(tmp_path):
    root, first, _, calibrated, raw = _tree(tmp_path)
    organize_delivery(root, calibrated, raw, echo=lambda message: None)
    (calibrated / "uid___A002_X9_X2.ms.split.cal").rmdir()
    real = first / "calibrated" / "uid___A002_X9_X1.ms.split.cal"
    real.unlink()
    real.mkdir()  # a real MS someone put there is never replaced

    again = organize_delivery(root, calibrated, raw, echo=lambda message: None)

    assert again.linked == 0
    assert real.is_dir() and not real.is_symlink()
    assert not (first / "calibrated" / "uid___A002_X9_X2.ms.split.cal").is_symlink()


def test_visibilities_organize_builds_the_tree(tmp_path):
    root, first, _, calibrated, raw = _tree(tmp_path)
    index = tmp_path / "index.csv"

    result = runner.invoke(
        cli.app,
        [
            "visibilities", "organize", "--delivery-root", str(root),
            "--calibrated-ms-root", str(calibrated), "--raw-ms-root", str(raw),
            "--index-csv", str(index),
        ],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert "3 calibrated MS link(s) made" in result.output
    assert "1 calibrated MS(s) have no member folder" in result.output
    assert (first / "calibrated" / "uid___A002_X9_X1.ms.split.cal").is_symlink()
    assert index.is_file()


def test_calibrate_organizes_the_tree_unless_told_not_to(tmp_path, monkeypatch):
    root, first, _, calibrated, raw = _tree(tmp_path)
    monkeypatch.setattr(cli_products, "_run_calibrate_jobs", lambda **kwargs: [])
    args = [
        "visibilities", "calibrate", "--input-root", str(root),
        "--raw-ms-root", str(raw), "--output-dir", str(calibrated),
    ]  # fmt: skip

    result = runner.invoke(cli.app, [*args, "--no-organize-delivery"])
    assert result.exit_code == 0, result.output
    assert not (first / "calibrated").exists()

    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert (first / "calibrated" / "uid___A002_X9_X1.ms.split.cal").is_symlink()
    assert (root / "ous_index.csv").is_file()
