"""An imaging run's windows table and archive rows (MADIV-28): almasim noise windows.

The run is a folder of imaging tasks as image-from-ms leaves them (task
folders, .done markers, logs) with its parameters CSV; the archive search and
the resolved products link each MS's execution block to its observation's row.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from almasim.cli.noise import noise_app
from almasim.services.noise import batch
from almasim.services.noise import windows as noise_windows
from almasim.services.noise.windows import imaging_windows

EB1, EB2 = "uid___A002_X10aa712_X5ddf", "uid___A002_X10aa712_X6000"
OUS1, OUS2 = "uid://A001/X35f5/Xab7", "uid://A001/X35f5/Xab9"


def _run(tmp_path: Path) -> dict[str, Path]:
    """Four parameter rows: EB1 SPW 13 fields 2 and 3 (done), EB1 SPW 15 field 2
    (not done), EB2 SPW 13 field 0 (done, an observation the search lacks)."""
    calibrated = tmp_path / "calibrated_ms"
    images = tmp_path / "images"
    rows = [
        (EB1, 13, "2", "RGALX346m", True),
        (EB1, 13, "3", "RGALX345p", True),
        (EB1, 15, "2", "RGALX346m", False),
        (EB2, 13, "0", "Other", True),
    ]
    parameters = []
    for eb, spw, field, source, done in rows:
        ms = calibrated / f"{eb}.ms.split.cal"
        parameters.append(
            {
                "filename": str(ms),
                "spectral_window_id": spw,
                "target_field_ids": field,
                "field_name": source,
            }
        )
        task = images / f"{eb}.ms.split" / f"SPW-{spw}_FIELD-{field}"
        task.mkdir(parents=True)
        task.with_name(task.name + ".log").write_text("lines in channels 10~20\n")
        if done:
            task.with_name(task.name + ".done").write_text("{}")
            (task / "quality.json").write_text("{}")
    parameters_csv = tmp_path / "imaging_parameters.csv"
    pd.DataFrame(parameters).to_csv(parameters_csv, index=False)
    metadata_csv = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {"ALMA_source_name": "RGALX345p", "member_ous_uid": OUS1, "PWV": "5.186"},
            {"ALMA_source_name": "RGALX346m", "member_ous_uid": OUS1, "PWV": "nan"},
            {"ALMA_source_name": "Elsewhere", "member_ous_uid": "uid://A001/X1/X1", "PWV": "1"},
        ]
    ).to_csv(metadata_csv, index=False)
    products_csv = tmp_path / "products.csv"
    pd.DataFrame(
        [
            {"uid": OUS1, "filename": f"2022.1.01515.S_{EB1}.asdm.sdm.tar"},
            {"uid": OUS2, "filename": f"2022.1.01515.S_{EB2}.asdm.sdm.tar"},
            {"uid": OUS1, "filename": "member.uid___A001_X35f5_Xab7.README.txt"},
        ]
    ).to_csv(products_csv, index=False)
    return {
        "parameters": parameters_csv,
        "images": images,
        "metadata": metadata_csv,
        "products": products_csv,
    }


def test_the_windows_of_an_imaging_run(tmp_path):
    paths = _run(tmp_path)
    runner = CliRunner()
    out = tmp_path / "noise"
    result = runner.invoke(
        noise_app,
        [
            "windows",
            str(paths["parameters"]),
            "--images",
            str(paths["images"]),
            "--metadata",
            str(paths["metadata"]),
            "--products",
            str(paths["products"]),
            "--out",
            str(out),
            "--all-fields",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "2 window(s)" in result.output
    assert f"left out {EB1}_spw15_field2: its imaging task is not done" in result.output
    assert f"left out {EB2}_spw13_field0: no archive row for its observation" in result.output

    # The table noise extract, fit and validate read, every task found.
    tasks = batch.read_windows(out / "windows.csv")
    assert [t.name for t in tasks] == [f"{EB1}_spw13_field2", f"{EB1}_spw13_field3"]
    first, second = tasks
    task_dir = paths["images"] / f"{EB1}.ms.split" / "SPW-13_FIELD-2"
    assert first.ms == str(tmp_path / "calibrated_ms" / f"{EB1}.ms.split.cal")
    assert (first.spw, first.field) == (13, "2")
    assert first.task_dir == str(task_dir)
    assert first.task_log == str(task_dir.with_name("SPW-13_FIELD-2.log"))
    # Each window's own source's archive row: a PWV only where the archive has one.
    assert first.pwv is None and second.pwv == "5.186"

    # Archive rows by window name, as noise validate --metadata reads them.
    rows = pd.read_csv(out / "rows.csv")
    assert list(rows["name"]) == [t.name for t in tasks]
    assert list(rows["ALMA_source_name"]) == ["RGALX346m", "RGALX345p"]


def test_a_window_takes_its_execution_blocks_pwv(tmp_path, monkeypatch):
    """With the auxiliary deliveries, each window's PWV is its execution block's
    (QA0 report); without, the archive row's, but for a 7 m array's."""
    from almasim.services.archive import qa0

    paths = _run(tmp_path)
    metadata = pd.read_csv(paths["metadata"], dtype=str, keep_default_na=False)
    observations = {"uid://a002/x10aa712/x5ddf": OUS1.lower()}

    def found(**kwargs):
        windows = imaging_windows(
            pd.read_csv(paths["parameters"]), paths["images"], metadata, observations, **kwargs
        ).windows
        return dict(zip(windows["name"], windows["pwv"], strict=True))

    assert found()[f"{EB1}_spw13_field3"] == "5.186"
    seven = metadata.assign(Array_type="7m")
    windows = imaging_windows(
        pd.read_csv(paths["parameters"]), paths["images"], seven, observations
    ).windows
    assert set(windows["pwv"]) == {""}

    root = tmp_path / "auxiliary"
    qa = (
        root
        / "2022.1.01515.S"
        / "science_goal.uid___A001_X35f5_Xab5"
        / "group.uid___A001_X35f5_Xab6"
    )
    qa = qa / "member.uid___A001_X35f5_Xab7" / "qa"
    qa.mkdir(parents=True)
    (qa / f"{EB1}.qa0_report.pdf").write_bytes(b"%PDF")
    read: list[Path] = []

    def report(path, eb_uid=""):
        read.append(Path(path))
        return qa0.QA0Report(EB1, "Pass", "SUCCESS", 1.0, "", pwv_mm=0.72)

    monkeypatch.setattr(qa0, "read_qa0_report", report)
    assert set(found(qa0_root=root).values()) == {"0.72"}
    assert read[0] == qa / f"{EB1}.qa0_report.pdf"
    (qa / f"{EB1}.qa0_report.pdf").unlink()  # no report: the archive row's
    assert found(qa0_root=root)[f"{EB1}_spw13_field3"] == "5.186"


def test_science_fields_and_a_cap(tmp_path):
    paths = _run(tmp_path)
    looked: list[str] = []

    def science(ms: str) -> list[int]:
        looked.append(ms)
        return [3]

    found = imaging_windows(
        pd.read_csv(paths["parameters"]),
        paths["images"],
        pd.read_csv(paths["metadata"], dtype=str, keep_default_na=False),
        {"uid://a002/x10aa712/x5ddf": OUS1.lower()},
        done_only=False,
        science_fields=science,
    )
    assert list(found.windows["name"]) == [f"{EB1}_spw13_field3"]
    assert (f"{EB1}_spw13_field2", "field 2 is not a science target") in found.skipped
    # Each MS's fields are read once.
    assert looked.count(str(tmp_path / "calibrated_ms" / f"{EB1}.ms.split.cal")) == 1

    capped = imaging_windows(
        pd.read_csv(paths["parameters"]),
        paths["images"],
        pd.read_csv(paths["metadata"], dtype=str, keep_default_na=False),
        {"uid://a002/x10aa712/x5ddf": OUS1.lower()},
        max_windows=1,
    )
    assert len(capped.windows) == 1 and len(capped.rows) == 1


def test_products_beside_the_search_and_none_at_all(tmp_path):
    paths = _run(tmp_path)
    runner = CliRunner()
    args = [
        "windows",
        str(paths["parameters"]),
        "--images",
        str(paths["images"]),
        "--metadata",
        str(paths["metadata"]),
        "--out",
        str(tmp_path / "noise"),
        "--all-fields",
    ]
    # products.csv sits beside results.csv: found without --products.
    assert runner.invoke(noise_app, args).exit_code == 0
    paths["products"].unlink()
    result = runner.invoke(noise_app, args)
    assert result.exit_code == 2
    assert "--products" in result.output


def test_nothing_to_measure_fails(tmp_path):
    paths = _run(tmp_path)
    for marker in paths["images"].rglob("*.done"):
        marker.unlink()
    result = CliRunner().invoke(
        noise_app,
        [
            "windows",
            str(paths["parameters"]),
            "--images",
            str(paths["images"]),
            "--metadata",
            str(paths["metadata"]),
            "--products",
            str(paths["products"]),
            "--out",
            str(tmp_path / "noise"),
            "--all-fields",
        ],
    )
    assert result.exit_code == 1
    assert "No window to measure" in result.output
    assert not (tmp_path / "noise" / "windows.csv").exists()


@pytest.mark.parametrize(
    ("value", "expected"), [("0.79", "0.79"), ("nan", ""), ("", ""), ("-1", ""), (None, "")]
)
def test_pwv(value, expected):
    assert noise_windows._pwv({"PWV": value}, {}, None) == expected
    assert noise_windows._pwv({"PWV": value, "Array_type": "7m"}, {}, None) == ""


def test_only_the_named_windows(tmp_path):
    paths = _run(tmp_path)
    found = imaging_windows(
        pd.read_csv(paths["parameters"]),
        paths["images"],
        pd.read_csv(paths["metadata"], dtype=str, keep_default_na=False),
        {"uid://a002/x10aa712/x5ddf": OUS1.lower()},
        names={f"{EB1}_spw13_field3", "not-there"},
    )
    assert list(found.windows["name"]) == [f"{EB1}_spw13_field3"]
    # Windows not asked for are not reported as left out.
    assert found.skipped == []


def test_windows_need_their_clean_quality(tmp_path):
    paths = _run(tmp_path)
    (paths["images"] / f"{EB1}.ms.split" / "SPW-13_FIELD-3" / "quality.json").unlink()
    found = imaging_windows(
        pd.read_csv(paths["parameters"]),
        paths["images"],
        pd.read_csv(paths["metadata"], dtype=str, keep_default_na=False),
        {"uid://a002/x10aa712/x5ddf": OUS1.lower()},
    )
    assert list(found.windows["name"]) == [f"{EB1}_spw13_field2"]
    assert (
        f"{EB1}_spw13_field3",
        "no quality.json: run image clean-quality on the images",
    ) in found.skipped


@pytest.mark.parametrize(
    ("failures", "allow_partial", "failed"),
    [
        ({}, False, False),
        ({"a": "x"}, False, True),
        ({"a": "x"}, True, False),
        ({"a": "x", "b": "y"}, True, True),  # all of them
    ],
)
def test_a_step_fails(failures, allow_partial, failed):
    from almasim.cli.noise import _failed

    assert _failed(failures, ["a", "b"], allow_partial) is failed
