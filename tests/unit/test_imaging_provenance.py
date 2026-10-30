"""Provenance in the imaging products' FITS headers: OUSs, execution blocks, commands."""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from almasim import __version__
from almasim.services.imaging.provenance import (
    Provenance,
    copy_provenance,
    eb_uid_from_ms,
    find_ous_index,
    provenance_header,
    stamp_fits,
)
from almasim.services.imaging.task_run import TaskRun
from almasim.services.imaging.wsclean_plan import row_provenance

EB = "uid___A002_X11b5555_X3f2"
ROW = {
    "eb_uid": EB,
    "project_code": "2023.1.00882.S",
    "science_goal_uid": "uid://A001/X3621/X1f9b",
    "group_ous_uid": "uid://A001/X3621/X1f9c",
    "member_ous_uid": "uid://A001/X3621/X1f9d",
}


def _index(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*ROW, "member_folder"])
        writer.writeheader()
        writer.writerow({**ROW, "member_folder": "x"})
    return path


def _image(path, **cards):
    header = fits.Header()
    for key, value in cards.items():
        header[key] = value
    fits.PrimaryHDU(np.zeros((1, 1, 2, 2), np.float32), header=header).writeto(path)
    return path


@pytest.mark.unit
def test_the_execution_block_comes_from_the_ms_name():
    assert eb_uid_from_ms(f"/data/{EB}.ms.split.cal") == EB
    assert eb_uid_from_ms(f"{EB}.ms") == EB
    assert eb_uid_from_ms("/data/spw25.single.ms") is None


@pytest.mark.unit
def test_the_header_names_the_ouss_as_alma_does_and_lists_the_steps():
    provenance = Provenance.for_ms(
        [f"/data/{EB}.ms.split.cal"], {EB: ROW}, spw=29, field="Spiderweb"
    )

    header = provenance_header(provenance, ["wsclean -name wsclean x.ms"])

    assert header["PROPCODE"] == "2023.1.00882.S"
    assert (header["SGOAL"], header["GROUP"], header["MEMBER"]) == (
        "uid://A001/X3621/X1f9b",
        "uid://A001/X3621/X1f9c",
        "uid://A001/X3621/X1f9d",
    )
    assert (header["SPW"], header["FIELD"]) == ("29", "Spiderweb")
    assert (header["NEB"], header["EBUID01"]) == (1, EB)
    assert (header["NMS"], header["MSNAM01"]) == (1, f"{EB}.ms.split.cal")
    assert header["ALMASVER"] == __version__
    assert list(header["HISTORY"]) == ["ALMASim: wsclean -name wsclean x.ms"]


@pytest.mark.unit
def test_an_ms_no_index_lists_keeps_its_execution_block_only():
    header = provenance_header(Provenance.for_ms([f"{EB}.ms"], {}, spw=5))
    assert "PROPCODE" not in header and "MEMBER" not in header
    assert header["EBUID01"] == EB and header["SPW"] == "5"


@pytest.mark.unit
def test_stamping_adds_the_cards_once_and_derived_images_copy_them(tmp_path):
    image = _image(tmp_path / "wsclean-image.fits", BUNIT="Jy/beam")
    provenance = provenance_header(
        Provenance.for_ms([f"{EB}.ms"], {EB: ROW}), ["a very long command " * 10]
    )

    assert stamp_fits(image, provenance)
    assert not stamp_fits(image, provenance), "a stamped file is left alone"

    header = fits.getheader(image)
    assert header["MEMBER"] == ROW["member_ous_uid"] and header["BUNIT"] == "Jy/beam"
    assert (
        "".join(str(h) for h in header["HISTORY"])
        .replace("ALMASim: ", "")
        .startswith("a very long command a very long command")
    )
    derived = copy_provenance(header, fits.Header())
    assert derived["EBUID01"] == EB and len(derived["HISTORY"]) == len(header["HISTORY"])


@pytest.mark.unit
def test_a_parameter_row_finds_the_index_above_its_ms(tmp_path):
    index = _index(tmp_path / "downloads" / "ous_index.csv")
    ms = tmp_path / "downloads" / "archive_ms" / "calibrated_ms" / f"{EB}.ms.split.cal"
    ms.mkdir(parents=True)
    row = pd.Series({"filename": str(ms), "spectral_window_id": 29, "field_name": "Spiderweb"})

    assert find_ous_index(ms) == index
    found = row_provenance(row)
    assert found.member_ous_uid == ROW["member_ous_uid"] and found.field == "Spiderweb"

    elsewhere = tmp_path / "elsewhere" / "ous_index.csv"
    _index(elsewhere)
    indexes: dict = {}
    assert row_provenance(row, elsewhere, indexes).project_code == ROW["project_code"]
    assert list(indexes) == [elsewhere], "each index is read once per run"


@pytest.mark.unit
def test_a_task_records_each_command_relative_to_its_folder(tmp_path):
    out = tmp_path / "SPW-29"
    out.mkdir()
    task = TaskRun(out=out, log_path=out / "SPW-29.log", ms_path=f"/data/{EB}.ms", spw=29)
    with task.log_path.open("w") as log:
        task.run(["true", "-name", f"{out}/wsclean", "-temp-dir", str(out), "/data/x.ms"], log)
    task.note("continuum subtracted")

    header = task.header()
    assert list(header["HISTORY"]) == [
        "ALMASim: true -name wsclean -temp-dir . /data/x.ms",
        "ALMASim: continuum subtracted",
    ]
    assert header["EBUID01"] == EB and header["SPW"] == "29"
