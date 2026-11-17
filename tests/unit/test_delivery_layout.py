"""Images written into the members' folders of an ALMA delivery tree."""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from almasim.services.archive.delivery import write_ous_index
from almasim.services.imaging.quality.report import find_task_dirs
from almasim.services.imaging.wsclean_plan import _task_label, task_directory
from almasim.services.ms.remove import remove_measurement_set

MEMBER = (
    "2023.1.00001.S/science_goal.uid___A001_X1_X1/group.uid___A001_X1_X2/member.uid___A001_X1_X3"
)
EB = "uid___A002_X9_X1"


def _delivery(tmp_path):
    """A tree of one member with one EB (from its QA0 report), indexed."""
    root = tmp_path / "tree"
    (root / MEMBER / "qa").mkdir(parents=True)
    (root / MEMBER / "qa" / f"{EB}.qa0_report.pdf").write_bytes(b"%PDF")
    write_ous_index(root, echo=lambda message: None)
    return root


def _row(filename, **extra):
    return pd.Series(
        {"filename": filename, "spectral_window_id": 25, "target_field_ids": "2", **extra}
    )


@pytest.mark.unit
def test_a_delivery_root_puts_each_task_in_its_member_images_folder(tmp_path):
    root = _delivery(tmp_path)
    ms = f"/data/{EB}.ms.split.cal"

    directory, fields = task_directory(root, _row(ms))
    assert directory == root / MEMBER / "images" / f"{EB}.ms.split" / "SPW-25_FIELD-2"
    assert fields == (2,)

    member_row = _row(
        ms,
        field_name="NGC 253",
        member_folder=MEMBER,
        member_windows=json.dumps([[ms, 25, [2]]]),
    )
    directory, fields = task_directory(root, member_row)
    assert directory == root / MEMBER / "images" / "SPW-25_NGC_253"
    assert fields == (0,)
    assert _task_label(directory, root, 25, None, True) == "member.uid___A001_X1_X3_SPW-25_NGC_253"

    other = task_directory(root, _row("/data/uid___A002_X9_X7.ms.split.cal"))[0]
    assert other == root / "uid___A002_X9_X7.ms.split" / "SPW-25_FIELD-2", "not in the index"


@pytest.mark.unit
def test_any_other_output_folder_keeps_the_flat_layout(tmp_path):
    ms = f"/data/{EB}.ms.split.cal"
    member_row = _row(
        ms, field_name="NGC253", member_folder=MEMBER, member_windows=json.dumps([[ms, 25, [2]]])
    )
    assert task_directory(tmp_path, member_row)[0] == (
        tmp_path / "member.uid___A001_X1_X3" / "SPW-25_NGC253"
    )
    assert task_directory(tmp_path, _row(ms))[0] == tmp_path / f"{EB}.ms.split" / "SPW-25_FIELD-2"


@pytest.mark.unit
def test_quality_searches_only_the_members_images_in_a_delivery_tree(tmp_path):
    root = _delivery(tmp_path)
    task = root / MEMBER / "images" / "SPW-25_NGC253"
    task.mkdir(parents=True)
    (task / "wsclean-image.fits").write_bytes(b"")
    stray = root / MEMBER / "product" / "elsewhere"
    stray.mkdir(parents=True)
    (stray / "wsclean-image.fits").write_bytes(b"")

    assert find_task_dirs(root) == [task]


@pytest.mark.unit
def test_removing_a_linked_ms_removes_the_ms_it_links_to(tmp_path):
    placed = tmp_path / "tree" / "calibrated" / f"{EB}.ms.split.cal"
    placed.mkdir(parents=True)
    (placed / "table.dat").write_text("rows")
    link = tmp_path / "calibrated_ms" / placed.name
    link.parent.mkdir()
    link.symlink_to(os.path.relpath(placed, link.parent))

    remove_measurement_set(link)
    assert not link.is_symlink() and not placed.exists()

    plain = tmp_path / "plain.ms"
    plain.mkdir()
    remove_measurement_set(plain)
    remove_measurement_set(plain)  # already gone: nothing to do
    assert not plain.exists()
