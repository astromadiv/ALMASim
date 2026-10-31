"""Member imaging: a member's execution blocks merged per window and field, and combined."""

from __future__ import annotations

import contextlib
import csv
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

from almasim import cli
from almasim.cli import image as cli_image
from almasim.services.archive.delivery import member_status
from almasim.services.imaging import markers as imaging_markers
from almasim.services.imaging import ms_geometry
from almasim.services.imaging import products as imaging_products
from almasim.services.imaging import task as imaging_task
from almasim.services.imaging.member_plan import (
    MEMBER_WINDOWS_COLUMN,
    WINDOW_NAME_COLUMN,
    member_parameters,
    member_windows,
)
from almasim.services.imaging.ms import member_window
from almasim.services.imaging.wsclean_plan import build_imaging_tasks
from almasim.services.ms import tables as ms_tables

from ._fakes import patch_everywhere

MEMBER = "uid://A001/X1/X3"
FOLDER = (
    "2023.1.00001.S/science_goal.uid___A001_X1_X1/group.uid___A001_X1_X2/member.uid___A001_X1_X3"
)
WINDOW = "X1#ALMA_RB_03#BB_1#SW-01#FULL_RES"
EBS = ("uid___A002_X9_X1", "uid___A002_X9_X2", "uid___A002_X9_X3")

runner = CliRunner()


def _index(calibrated=EBS, alma=EBS, extra=()):
    rows = {}
    for eb in (*alma, *extra):
        rows[eb] = {
            "eb_uid": eb,
            "project_code": "2023.1.00001.S",
            "science_goal_uid": "uid://A001/X1/X1",
            "group_ous_uid": "uid://A001/X1/X2",
            "member_ous_uid": MEMBER,
            "member_folder": FOLDER,
            "alma_calibrated": "yes" if eb in alma else "no",
            "raw_ms": "",
            "calibrated_ms": f"/cal/{eb}.ms.split.cal" if eb in calibrated else "",
        }
    return rows


def _row(eb, spw, start, *, field="Spiderweb", field_id=2, baseline=300.0, window=WINDOW):
    return {
        "filename": f"/cal/{eb}.ms.split.cal",
        "spectral_window_id": spw,
        "spectral_window_name": window,
        "reference_frequency": 100.0e9,
        "fov_per_frequency": 60.0,
        "max_baseline_size": baseline,
        "synthetized_beam_size": 2.0 * 300.0 / baseline,
        "n_channels": 480,
        "channel_width_hz": 1.953125e6,
        "spectral_resolution_hz": 3.90625e6,
        "total_bandwidth_hz": 937.5e6,
        "start_mjd_s": start,
        "end_mjd_s": start + 3000.0,
        "duration_s": 3000.0,
        "start_datetime": f"t{start}",
        "end_datetime": f"t{start + 3000}",
        "n_visibility_rows": 100,
        "target_field_ids": str(field_id),
        "field_name": field,
    }


def _parameters():
    # The EBs number the window differently; the earliest EB is listed last.
    return pd.DataFrame(
        [
            _row(EBS[0], 25, 2000.0, field_id=2),
            _row(EBS[1], 17, 3000.0, field_id=1, baseline=900.0),
            _row(EBS[2], 25, 1000.0, field_id=2),
        ]
    )


@pytest.mark.unit
def test_a_member_is_complete_when_every_eb_alma_calibrated_has_a_calibrated_ms():
    complete = member_status(_index(extra=("uid___A002_X9_X8",)))[MEMBER]
    assert complete.complete
    assert complete.needed == EBS, "an EB ALMA did not calibrate is not needed"

    waiting = member_status(_index(calibrated=EBS[:2]))[MEMBER]
    assert not waiting.complete and waiting.missing == (EBS[2],)


@pytest.mark.unit
def test_member_rows_merge_the_ebs_of_one_window_and_field_by_name():
    rows, skipped = member_parameters(_parameters(), _index())

    assert skipped == []
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["member_ous_uid"] == MEMBER and row["member_folder"] == FOLDER
    assert [(p.name, s, f) for p, s, f in member_windows(row)] == [
        (f"{EBS[2]}.ms.split.cal", 25, (2,)),
        (f"{EBS[0]}.ms.split.cal", 25, (2,)),
        (f"{EBS[1]}.ms.split.cal", 17, (1,)),
    ], "every EB's own window and field ids, earliest first"
    assert row["filename"] == f"/cal/{EBS[2]}.ms.split.cal", "the earliest EB's row"
    assert row["max_baseline_size"] == 900.0
    assert row["synthetized_beam_size"] == pytest.approx(2.0 / 3.0), "the longest baseline's"
    assert row["duration_s"] == 9000.0 and row["n_visibility_rows"] == 300
    assert row["target_field_ids"] == "0", "the combined MS holds the one field"


@pytest.mark.unit
def test_an_incomplete_member_is_not_imaged():
    rows, skipped = member_parameters(_parameters(), _index(calibrated=EBS[:2]))

    assert rows.empty
    assert skipped == [(MEMBER, f"1 of its 3 execution blocks have no calibrated MS ({EBS[2]})")]


@pytest.mark.unit
def test_a_member_whose_ebs_lack_parameter_rows_is_not_imaged():
    rows, skipped = member_parameters(_parameters().iloc[:2], _index())

    assert rows.empty
    assert skipped == [(MEMBER, f"1 of its 3 execution blocks have no parameter rows ({EBS[2]})")]


@pytest.mark.unit
def test_window_names_come_from_each_ms_when_the_csv_has_none(monkeypatch):
    # Each EB's SPECTRAL_WINDOW table, as its own numbering lists the windows.
    names = {f"/cal/{eb}.ms.split.cal": ["other"] * 26 for eb in EBS}
    for eb, spw in zip(EBS, (25, 17, 25), strict=True):
        names[f"/cal/{eb}.ms.split.cal"][spw] = WINDOW
    opened = []

    @contextlib.contextmanager
    def fake_open_table(path, **kwargs):
        filename = path.removesuffix("::SPECTRAL_WINDOW")
        opened.append(filename)
        yield SimpleNamespace(getcol=lambda column: names[filename])

    monkeypatch.setattr(ms_geometry, "import_casacore_tables", lambda: object())
    monkeypatch.setattr(ms_tables, "open_table", fake_open_table)

    rows, skipped = member_parameters(_parameters().drop(columns=WINDOW_NAME_COLUMN), _index())

    assert skipped == [] and len(rows) == 1
    assert rows.iloc[0][WINDOW_NAME_COLUMN] == WINDOW
    assert sorted(opened) == sorted(names), "each MS read once"


@pytest.mark.unit
def test_an_ms_not_named_after_its_eb_and_a_member_without_rows_are_left_out_quietly():
    parameters = pd.concat([_parameters(), pd.DataFrame([_row(EBS[0], 25, 500.0)])])
    parameters.iloc[-1, parameters.columns.get_loc("filename")] = "/cal/simulated.ms"
    index = _index()
    other = "uid___A002_X9_X9"
    index[other] = {**index[EBS[0]], "eb_uid": other, "member_ous_uid": "uid://A001/X1/X4"}

    rows, skipped = member_parameters(parameters, index)

    assert skipped == [] and len(rows) == 1
    assert len(member_windows(rows.iloc[0])) == 3


@pytest.mark.unit
def test_an_eb_row_without_target_fields_gives_its_window_no_field_ids():
    parameters = _parameters()
    parameters["target_field_ids"] = parameters["target_field_ids"].astype(object)
    parameters.loc[1, "target_field_ids"] = float("nan")

    rows, _ = member_parameters(parameters, _index())

    assert [f for _, _, f in member_windows(rows.iloc[0])] == [(2,), (2,), ()]


@pytest.mark.unit
def test_a_mosaic_field_is_left_out_and_said_so():
    parameters = pd.concat(
        [
            _parameters(),
            pd.DataFrame(
                [_row(eb, 25, 1000.0 * i, field="M31", field_id=5) for i, eb in enumerate(EBS)]
            ),
            pd.DataFrame([_row(EBS[0], 25, 2000.0, field="M31", field_id=6)]),
        ]
    )

    rows, skipped = member_parameters(parameters, _index())

    assert list(rows["field_name"]) == ["Spiderweb"]
    assert skipped == [(MEMBER, "field M31 is a mosaic (several pointings of that name)")]


@pytest.mark.unit
def test_member_tasks_image_the_combined_ms_in_the_member_folder(tmp_path):
    rows, _ = member_parameters(_parameters(), _index())
    index = tmp_path / "ous_index.csv"
    pd.DataFrame(_index().values()).to_csv(index, index=False)

    tasks, skipped = build_imaging_tasks(
        rows,
        tmp_path / "out",
        fov_fraction=1.0,
        beam_sampling=5,
        num_cores=4,
        max_cores_per_node=96,
        ous_index_csv=index,
    )

    assert skipped == 0 and len(tasks) == 1
    task = tasks[0]
    assert task.output_dir == tmp_path / "out" / "member.uid___A001_X1_X3" / "SPW-25_Spiderweb"
    assert task.label == "member.uid___A001_X1_X3_SPW-25_Spiderweb"
    assert task.spw == 25 and task.field_ids == (0,)
    assert [s for _, s, _ in task.member_windows] == [25, 25, 17]
    assert task.command[task.command.index("-field") + 1] == "0"
    assert task.command[-1] == imaging_products.SINGLE_WINDOW_PLACEHOLDER
    assert task.provenance.eb_uids == (EBS[2], EBS[0], EBS[1])
    assert task.provenance.member_ous_uid == MEMBER


@pytest.mark.unit
def test_member_tasks_need_the_single_window_ms(tmp_path):
    rows, _ = member_parameters(_parameters(), _index())

    with pytest.raises(ValueError, match="single-window"):
        build_imaging_tasks(
            rows,
            tmp_path,
            fov_fraction=1.0,
            beam_sampling=5,
            num_cores=4,
            max_cores_per_node=96,
            single_window=False,
        )


@pytest.mark.unit
def test_the_combined_ms_is_built_from_each_ebs_window_and_the_copies_removed(
    tmp_path, monkeypatch
):
    extracted = []

    def fake_extract(ms, spw, fields, out, *, timeout=None, log_path=None, max_weight=None):
        extracted.append((Path(ms).name, spw, fields, Path(out).name))
        Path(out).mkdir()
        return {"rows": 10, "dropped_rows": 1}

    combined = {}

    def fake_child(target, kwargs, **_):
        combined.update(target=target, **kwargs)
        out = Path(kwargs["out_path"])
        out.mkdir()
        (out / "table.dat").write_bytes(b"x" * 8)
        for copy in kwargs["window_mss"]:
            Path(copy.removesuffix(".ms") + ".lsrk.ms").mkdir()
        return {
            "channels": 476,
            "start_hz": 1e11,
            "width_hz": 1.95e6,
            "shifts_hz": [0.0, 1.5e6],
            "channels_out": None,
        }

    monkeypatch.setattr(member_window, "extract_single_window_ms_subprocess", fake_extract)
    monkeypatch.setattr(member_window, "call_in_child", fake_child)
    out = tmp_path / "spw25.single.ms"

    result = member_window.build_member_window_ms(
        [(f"/cal/{EBS[0]}.ms.split.cal", 25, [2]), (f"/cal/{EBS[1]}.ms.split.cal", 17, [1])],
        out,
        casa_data="/casa",
        max_weight=100.0,
    )

    assert extracted == [
        (f"{EBS[0]}.ms.split.cal", 25, [2], f"{EBS[0]}.spw25.single.ms"),
        (f"{EBS[1]}.ms.split.cal", 17, [1], f"{EBS[1]}.spw25.single.ms"),
    ]
    assert combined["target"].endswith("member_window:regrid_and_concat")
    assert combined["out_path"] == str(out) and combined["casa_data"] == "/casa"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["spw25.single.ms"], "copies removed"
    assert result["rows"] == 20 and result["dropped_rows"] == 2
    assert result["execution_blocks"] == 2 and result["channels"] == 476
    assert combined["channels_out"] is None
    assert result["bytes"] == 8 and result["max_weight"] == 100.0


@pytest.mark.unit
def test_a_member_task_images_the_combined_ms(tmp_path, monkeypatch):
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append(list(cmd))
        (Path(cmd[cmd.index("-name") + 1]).parent / imaging_products.IMAGE_FILENAME).write_bytes(
            b"SIMPLE"
        )

        class _Proc:
            stdout = iter(["WSClean version 3.7\n"])

            @staticmethod
            def wait():
                return 0

        return _Proc()

    built = []

    def fake_build(
        windows, out, *, casa_data, timeout=None, log_path=None, max_weight=None, channels_out=None
    ):
        built.append((list(windows), Path(out).name, casa_data, channels_out))
        Path(out).mkdir()
        return {
            "rows": 30,
            "bytes": 2e9,
            "dropped_rows": 0,
            "max_weight": max_weight,
            "seconds": 5.0,
            "channels": 480,
            "start_hz": 92.5e9,
            "width_hz": 1953.125e3,
            "shifts_hz": [0.0, 1.5e6, -2.0e6],
            "channels_out": None,
        }

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    patch_everywhere(monkeypatch, member_window, "build_member_window_ms", fake_build)
    outdir = tmp_path / "member.uid___A001_X1_X3" / "SPW-25_Spiderweb"
    windows = [(f"/cal/{eb}.ms.split.cal", 25, [2]) for eb in EBS]

    imaging_task.run_wsclean_task(
        command=["wsclean", "-name", str(outdir / "wsclean"), "-field", "0",
                 imaging_products.SINGLE_WINDOW_PLACEHOLDER],
        output_dir=str(outdir),
        ms_path=windows[0][0],
        spw=25,
        threads=2,
        single_window=True,
        field_ids=[0],
        casa_data="/casa",
        member_windows=windows,
    )  # fmt: skip

    assert built == [(windows, "spw25.single.ms", "/casa", None)], "no cube, no channel count"
    assert calls[0][-1] == str(outdir / "spw25.single.ms")
    assert imaging_markers.is_imaging_complete(outdir)
    log = imaging_markers.imaging_log_path(outdir).read_text().splitlines()
    assert log[0] == (
        "# combined spectral window 25 of 3 execution blocks into spw25.single.ms: 30 rows, "
        "dropped 0 rows (0.0 %) with WEIGHT > 10000, on an LSRK grid of 480 channels from "
        "92.500000 GHz, 1953.125 kHz wide, 2.0 GB, 5.0 s"
    )
    assert log[1] == "# the execution blocks' own LSRK starts, in kHz: +0.0, +1500.0, -2000.0"


@pytest.mark.unit
def test_a_member_task_needs_casa_data(tmp_path):
    with pytest.raises(ValueError, match="member_windows needs single_window and casa_data"):
        imaging_task.run_wsclean_task(
            command=["wsclean", "-name", str(tmp_path / "wsclean"), "x"],
            output_dir=str(tmp_path),
            ms_path="/cal/a.ms",
            spw=25,
            threads=2,
            single_window=True,
            member_windows=[("/cal/a.ms", 25, [2])],
        )


@pytest.mark.unit
def test_member_windows_round_trip_through_the_csv(tmp_path):
    rows, _ = member_parameters(_parameters(), _index())
    rows.to_csv(tmp_path / "members.csv", index=False)

    again = pd.read_csv(tmp_path / "members.csv")

    assert member_windows(again.iloc[0]) == member_windows(rows.iloc[0])
    last = json.loads(again.iloc[0][MEMBER_WINDOWS_COLUMN])[2]
    assert last == [f"/cal/{EBS[1]}.ms.split.cal", 17, [1]]


@pytest.mark.unit
def test_a_member_cube_takes_its_channel_count_from_the_grid(tmp_path, monkeypatch):
    commands = []

    def fake_popen(cmd, **kwargs):
        commands.append(list(cmd))

        class _Proc:
            stdout = iter(["WSClean version 3.7\n"])

            @staticmethod
            def wait():
                return 0

        return _Proc()

    asked = []

    def fake_build(windows, out, *, channels_out=None, **_):
        asked.append(channels_out)
        Path(out).mkdir()
        return {
            "rows": 30,
            "bytes": 1,
            "dropped_rows": 0,
            "max_weight": None,
            "channels": 476,
            "start_hz": 92.5e9,
            "width_hz": 1953.125e3,
            "shifts_hz": [0.0, 1.5e6],
            "channels_out": 238,
        }

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    patch_everywhere(monkeypatch, member_window, "build_member_window_ms", fake_build)
    outdir = tmp_path / "SPW-25_Spiderweb"
    cube = ["wsclean", "-name", str(outdir / imaging_products.CUBE_NAME), "-channels-out", "240",
            imaging_products.SINGLE_WINDOW_PLACEHOLDER]  # fmt: skip

    with pytest.raises(RuntimeError):  # the fake WSClean writes no cube
        imaging_task.run_wsclean_task(
            command=[],
            cube_command=cube,
            output_dir=str(outdir),
            ms_path="/cal/a.ms",
            spw=25,
            threads=2,
            single_window=True,
            casa_data="/casa",
            member_windows=[("/cal/a.ms", 25, [2]), ("/cal/b.ms", 25, [2])],
        )

    assert asked == [240], "the plan's count, for the first EB's channels"
    assert commands[0][commands[0].index("-channels-out") + 1] == "238"
    log = imaging_markers.imaging_log_path(outdir).read_text()
    assert "; the cube has 238 channels" in log


def _delivery(tmp_path, *, with_index=True):
    """A parameter CSV of the member's three EBs plus one EB of a member missing a row,
    their MSs under ``archive_ms`` and the index above them."""
    other = "uid___A002_X9_X9"
    index = _index()
    index[other] = {**index[EBS[0]], "eb_uid": other, "member_ous_uid": "uid://A001/X1/X4"}
    archive = tmp_path / "archive_ms"
    parameters = pd.concat([_parameters(), pd.DataFrame([_row(other, 25, 5000.0)])])
    parameters["filename"] = [str(archive / Path(f).name) for f in parameters["filename"]]
    csv_path = tmp_path / "params.csv"
    parameters.to_csv(csv_path, index=False)
    if with_index:
        # The other member's second EB has a calibrated MS but no row in the CSV.
        index["uid___A002_X9_XA"] = {**index[other], "eb_uid": "uid___A002_X9_XA"}
        with (tmp_path / "ous_index.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(index[EBS[0]]))
            writer.writeheader()
            writer.writerows(index.values())
    return csv_path


def _output(result) -> str:
    """The CLI's output without Rich's colours and panel borders."""
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", result.output).replace("\u2502", " ").split())


@pytest.mark.unit
def test_image_from_ms_by_member_images_each_complete_member(tmp_path, monkeypatch):
    csv_path = _delivery(tmp_path)
    runs = []

    def fake_run(tasks, imaging, execution, *, casa_data, failures, **kwargs):
        runs.append((tasks, casa_data))
        return [{"ok": 1}]

    monkeypatch.setattr(cli_image, "run_imaging_tasks", fake_run)
    monkeypatch.setattr(
        cli_image, "_prepare_selfcal_casa_data", lambda out, root: tmp_path / "casa-data"
    )

    result = runner.invoke(
        cli.app,
        ["image", "image-from-ms", str(csv_path), str(tmp_path / "out"), "--by-member"],
    )

    assert result.exit_code == 0, result.output
    output = _output(result)
    assert "Not imaging member uid://A001/X1/X4: 1 of its 2 execution blocks have no " in output
    assert "1 complete member(s), 1 (window, field) task(s)" in output
    [(tasks, casa_data)] = runs
    assert casa_data == str(tmp_path / "casa-data"), "the regridding needs CASA's data"
    assert [s for _, s, _ in tasks[0].member_windows] == [25, 25, 17]


@pytest.mark.unit
def test_image_from_ms_by_member_needs_the_single_window_ms(tmp_path):
    csv_path = _delivery(tmp_path)

    result = runner.invoke(
        cli.app,
        [
            "image",
            "image-from-ms",
            str(csv_path),
            str(tmp_path / "out"),
            "--by-member",
            "--no-single-window-ms",
        ],
    )

    assert result.exit_code == 2
    assert "--by-member needs --single-window-ms" in _output(result)


@pytest.mark.unit
def test_image_from_ms_by_member_needs_an_ous_index(tmp_path):
    csv_path = _delivery(tmp_path, with_index=False)

    result = runner.invoke(
        cli.app, ["image", "image-from-ms", str(csv_path), str(tmp_path / "out"), "--by-member"]
    )

    assert result.exit_code == 2
    assert "pass --ous-index" in _output(result)
