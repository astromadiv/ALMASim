"""Unit tests for almasim.services.archive.qa0."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from almasim.services.archive.qa0 import (
    QA0Report,
    calibration_is_delivered,
    eb_uid_from_raw_filename,
    entity_id_to_filename,
    filter_products_by_qa0,
    find_qa0_report,
    index_member_directories,
    parse_qa0_report_text,
    qa0_report_path,
)

# Text as pypdf extracts it from a delivered report, including the line break
# between "QA0 Status" and its value.
SEMIPASS_TEXT = """QA0 Report

Execution Block Summary
Project Code 2023.1.01253.S SchedBlock A2744-20_a_04_TM1
ExecBlock uid://A002/X10d9399/Xae5e ExecBlock Status FAIL
QA0 Status
 SemiPass Exec. Fraction 0.00
Repr. frequency 128.639 GHz (Sky) Band ALMA_RB_04
QA0 comment Failed just after initial calibration scans have finished,
and thus almost no target data.
Page 1 of
 10
"""

PASS_TEXT = """QA0 Report

Execution Block Summary
Project Code 2023.1.00868.S SchedBlock X_a_03_TM1
ExecBlock uid://A002/X10e2702/X5c84 ExecBlock Status SUCCESS
QA0 Status
 Pass Exec. Fraction 1.34
QA0 comment No severe issue was found.
Page 1 of
 12
"""


@pytest.mark.unit
def test_parse_semipass_report():
    report = parse_qa0_report_text(SEMIPASS_TEXT, "uid___A002_X10d9399_Xae5e")
    assert report.qa0_status == "SemiPass"
    assert report.execblock_status == "FAIL"
    assert report.execution_fraction == 0.0
    # The comment is rejoined across its line break.
    assert report.comment.startswith("Failed just after initial calibration")
    assert "almost no target data." in report.comment
    assert report.calibration_delivered is False


@pytest.mark.unit
def test_parse_pass_report():
    report = parse_qa0_report_text(PASS_TEXT, "uid___A002_X10e2702_X5c84")
    assert report.qa0_status == "Pass"
    assert report.execution_fraction == 1.34
    assert report.calibration_delivered is True


@pytest.mark.unit
def test_the_weather_the_block_was_observed_in():
    weather = "Weather PWV 0.72 mm; Wind 4.89 m/s; Humidity 29.46 %; Pressure 556.44 hPa;\n"
    report = parse_qa0_report_text(PASS_TEXT.replace("QA0 comment", weather + "QA0 comment"))
    assert report.pwv_mm == 0.72
    # A 7 m array's antennas carry no water vapour radiometer.
    seven = PASS_TEXT.replace("QA0 comment", weather.replace("0.72", "N/A") + "QA0 comment")
    assert parse_qa0_report_text(seven).pwv_mm is None
    assert parse_qa0_report_text(PASS_TEXT).pwv_mm is None


@pytest.mark.unit
def test_execblock_status_does_not_decide_delivery():
    """A SemiPass EB often reports ExecBlock Status SUCCESS; only QA0 Status counts."""
    text = SEMIPASS_TEXT.replace("ExecBlock Status FAIL", "ExecBlock Status SUCCESS")
    report = parse_qa0_report_text(text)
    assert report.execblock_status == "SUCCESS"
    assert report.calibration_delivered is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "status,expected",
    [
        ("Pass", True),
        ("pass", True),
        ("SemiPass", False),
        ("SemiPass,", False),  # trailing comma occurs in real reports
        (" Pass ", True),
        ("", False),
    ],
)
def test_calibration_is_delivered(status, expected):
    assert calibration_is_delivered(status) is expected


@pytest.mark.unit
def test_missing_fields_do_not_raise():
    report = parse_qa0_report_text("QA0 Report\nnothing useful here\n")
    assert report == QA0Report(
        eb_uid="", qa0_status="", execblock_status="", execution_fraction=None, comment=""
    )
    assert report.calibration_delivered is False


@pytest.mark.unit
def test_qa0_report_path():
    path = qa0_report_path("/data/member.uid___A001_X1_X2", "uid___A002_X3_X4")
    assert path == Path("/data/member.uid___A001_X1_X2/qa/uid___A002_X3_X4.qa0_report.pdf")


# ---------------------------------------------------------------------------
# Product filter: drop raw ASDMs of SemiPass execution blocks before download.


def _report(eb_uid: str, status: str) -> QA0Report:
    return QA0Report(
        eb_uid=eb_uid,
        qa0_status=status,
        execblock_status="SUCCESS",
        execution_fraction=1.0,
        comment="",
    )


def _raw(member: str, eb: str, size: int = 100) -> SimpleNamespace:
    project = "2023.1.00879.S"
    return SimpleNamespace(
        product_type="raw",
        uid=member,
        filename=f"{project}_{eb}.asdm.sdm.tar",
        content_length=size,
    )


def _member_dir(root: Path, member: str) -> Path:
    member_dir = (
        root
        / "2023.1.00879.S"
        / "science_goal.uid___A001_X378a_X147"
        / "group.uid___A001_X378a_X148"
        / f"member.{entity_id_to_filename(member)}"
    )
    (member_dir / "qa").mkdir(parents=True)
    return member_dir


def test_entity_id_to_filename():
    assert entity_id_to_filename("uid://A001/X378a/X149") == "uid___A001_X378a_X149"
    assert entity_id_to_filename(" uid://A002/X11b5555/X43a9 ") == "uid___A002_X11b5555_X43a9"


def test_eb_uid_from_raw_filename():
    name = "2023.1.00879.S_uid___A002_X11d61dd_X9e40.asdm.sdm.tar"
    assert eb_uid_from_raw_filename(name) == "uid___A002_X11d61dd_X9e40"
    assert eb_uid_from_raw_filename("/some/dir/" + name) == "uid___A002_X11d61dd_X9e40"
    assert eb_uid_from_raw_filename("2023.1.00879.S_uid___A001_X378a_X149_auxiliary.tar") is None
    assert eb_uid_from_raw_filename("") is None


def test_index_and_find_qa0_report(tmp_path):
    member = "uid://A001/X378a/X149"
    member_dir = _member_dir(tmp_path, member)
    report = member_dir / "qa" / "uid___A002_X11d61dd_X9e40.qa0_report.pdf"
    report.write_bytes(b"%PDF-1.4")

    index = index_member_directories(tmp_path)
    assert index == {"member.uid___A001_X378a_X149": member_dir}
    assert find_qa0_report(tmp_path, member, "uid___A002_X11d61dd_X9e40") == report
    assert find_qa0_report(tmp_path, member, "uid___A002_X11d61dd_X0000") is None
    assert find_qa0_report(tmp_path, "uid://A001/X378a/X999", "uid___A002_X11d61dd_X9e40") is None


def test_filter_products_by_qa0_drops_semipass_keeps_pass_and_unknown(tmp_path):
    member = "uid://A001/X378a/X149"
    member_dir = _member_dir(tmp_path, member)
    statuses = {
        "uid___A002_X1_Xpass": "Pass",
        "uid___A002_X1_Xsemi": "SemiPass,",  # trailing comma as in some reports
        "uid___A002_X1_Xbroken": "Pass",
    }
    for eb in statuses:
        (member_dir / "qa" / f"{eb}.qa0_report.pdf").write_bytes(b"%PDF-1.4")

    def fake_reader(path, eb_uid=""):
        if eb_uid.endswith("Xbroken"):
            raise ValueError("cannot parse")
        return _report(eb_uid, statuses[eb_uid])

    raw_pass = _raw(member, "uid___A002_X1_Xpass", 10)
    raw_semi = _raw(member, "uid___A002_X1_Xsemi", 1000)
    raw_broken = _raw(member, "uid___A002_X1_Xbroken", 20)
    raw_noreport = _raw(member, "uid___A002_X1_Xnoreport", 30)
    raw_nomember = _raw("uid://A001/X378a/X999", "uid___A002_X2_X1", 40)
    aux = SimpleNamespace(
        product_type="auxiliary",
        uid=member,
        filename="2023.1.00879.S_uid___A001_X378a_X149_auxiliary.tar",
        content_length=5,
    )
    products = [raw_pass, raw_semi, raw_broken, raw_noreport, raw_nomember, aux]

    result = filter_products_by_qa0(products, tmp_path, reader=fake_reader)

    assert result.skipped == [raw_semi]
    assert result.bytes_skipped == 1000
    assert result.kept == [raw_pass, raw_broken, raw_noreport, raw_nomember, aux]
    # Unreadable or missing reports never drop a product; they are reported instead.
    assert result.unknown == [raw_broken, raw_noreport, raw_nomember]


def test_filter_products_by_qa0_without_deliveries_keeps_everything(tmp_path):
    products = [_raw("uid://A001/X378a/X149", "uid___A002_X1_X1")]
    result = filter_products_by_qa0(products, tmp_path / "empty", reader=lambda *a, **k: None)
    assert result.kept == products
    assert result.unknown == products
    assert result.skipped == []
