"""The ALMA pipeline's cont.dat: parsing, finding, matching to a window, the mask."""

from __future__ import annotations

import csv
import io
import sys
import tarfile
from types import ModuleType

import numpy as np
import pytest

from almasim.services.imaging import cont_dat

from ._fakes import FakeTable

CONT_DAT = """Field: NGC1

SpectralWindow: 23
100.0~100.2GHz LSRK
100.5~100.9GHz LSRK

SpectralWindow: 25
ALL

SpectralWindow: 27
NONE

SpectralWindow: 29
NONE

Field: NGC2

SpectralWindow: 23
100100~100300MHz
SpectralWindow: 25
ALL
103.0~103.5GHz LSRK
SpectralWindow: 27
102.0~102.4GHz LSRK
"""

EB = "uid___A002_X1_X2"
MEMBER = (
    "2023.1.00001.S/science_goal.uid___A001_X1_X1/group.uid___A001_X1_X2/member.uid___A001_X1_X3"
)
# Channel centres between cont.dat's range edges: 99.95, 100.05, ... 101.05 GHz.
FREQ = np.linspace(99.95e9, 101.05e9, 12)
Block = cont_dat.ContBlock


def _window(freq_hz, names=("NGC1",), shift=1.0):
    """A window whose LSRK frequencies are ``shift`` times the given ones."""
    freq = np.asarray(freq_hz, dtype=float)
    return cont_dat.WindowFrequencies(
        field_names=frozenset(names),
        channels=freq.size,
        in_frame=lambda frame: freq * (shift if frame == "LSRK" else 1.0),
    )


@pytest.mark.unit
def test_parse_cont_dat_reads_every_field_and_window():
    blocks = cont_dat.parse_cont_dat(CONT_DAT)

    assert blocks[("NGC1", 23)] == Block([(100.0e9, 100.2e9, "LSRK"), (100.5e9, 100.9e9, "LSRK")])
    assert blocks[("NGC1", 25)] == Block(whole="ALL") and blocks[("NGC1", 27)].whole == "NONE"
    assert blocks[("NGC2", 23)] == Block([(100.1e9, 100.3e9, "LSRK")]), "MHz; LSRK when unnamed"
    assert blocks[("NGC2", 25)] == Block([(103.0e9, 103.5e9, "LSRK")], "ALL"), "ALL and its span"
    assert blocks[("NGC2", 27)] == Block([(102.0e9, 102.4e9, "LSRK")])
    assert len(blocks) == 7
    assert cont_dat.parse_cont_dat("SpectralWindow: 1\n1~2GHz\n") == {}, "no field, no block"
    assert blocks[("NGC2", 25)].describe() == "ALL"
    assert blocks[("NGC1", 23)].describe() == "2 continuum ranges"


@pytest.mark.unit
def test_match_window_picks_the_field_block_overlapping_the_window_most():
    blocks = cont_dat.parse_cont_dat(CONT_DAT)
    # 99.95-101.05 GHz: NGC1 spw 23 has 0.6 GHz of continuum there, 27 is NONE.
    window = _window(FREQ)
    assert cont_dat.match_window(blocks, window) == (("NGC1", 23), "")
    # NGC2: spw 23 overlaps; spw 27 (102 GHz) does not.
    ngc2 = _window(FREQ, ("NGC2",))
    assert cont_dat.match_window(blocks, ngc2)[0] == ("NGC2", 23)
    # An ALL block's range places it: NGC2's spw 25 spans 103-103.5 GHz.
    assert cont_dat.match_window(blocks, _window(np.linspace(102.95e9, 103.55e9, 7), ("NGC2",)))[
        0
    ] == ("NGC2", 25)
    key, reason = cont_dat.match_window(blocks, _window(FREQ, ("Other",)))
    assert key is None and "no block for field Other" in reason


@pytest.mark.unit
def test_match_window_leaves_a_neighbouring_window_that_touches_the_edge():
    # Block 1 is a neighbour whose last range reaches 0.1 GHz into the window;
    # block 2 is the window's own, all ALL.
    blocks = {
        ("F", 1): Block([(99.0e9, 99.5e9, "LSRK"), (99.6e9, 100.05e9, "LSRK")]),
        ("F", 2): Block(whole="ALL"),
    }
    assert cont_dat.match_window(blocks, _window(FREQ, ("F",))) == (("F", 2), "")


@pytest.mark.unit
def test_match_window_compares_in_the_block_frame():
    blocks = {
        ("F", 1): Block([(101.0e9, 101.5e9, "LSRK")]),
        ("F", 2): Block([(100.0e9, 100.4e9, "LSRK")]),
    }
    topo = np.linspace(100.0e9, 100.5e9, 8)
    assert cont_dat.match_window(blocks, _window(topo, ("F",)))[0] == ("F", 2)
    # Shifted by 1 % into LSRK the window sits on block 1.
    assert cont_dat.match_window(blocks, _window(topo, ("F",), shift=1.01))[0] == ("F", 1)


@pytest.mark.unit
def test_match_window_places_all_and_none_blocks_by_the_other_fields():
    blocks = cont_dat.parse_cont_dat(CONT_DAT)
    # 102.1-102.3 GHz: NGC1 has no range there; NGC2 places spw 27 there, NGC1's 27 is NONE.
    window = _window(np.linspace(101.95e9, 102.45e9, 6))
    assert cont_dat.match_window(blocks, window) == (("NGC1", 27), "")
    # 103.2 GHz: NGC2's ALL range places NGC1's spw 25 there.
    assert cont_dat.match_window(blocks, _window(np.linspace(102.95e9, 103.55e9, 7)))[0] == (
        "NGC1",
        25,
    )
    # 104 GHz: spw 29 (NONE) is the only window placed nowhere.
    assert cont_dat.match_window(blocks, _window(np.linspace(104e9, 104.2e9, 5)))[0] == (
        "NGC1",
        29,
    )
    # Two unplaced ALL/NONE windows: no telling which.
    two = {("F", 1): Block(whole="ALL"), ("F", 2): Block(whole="NONE")}
    key, reason = cont_dat.match_window(two, _window(np.linspace(1e9, 2e9, 3), ("F",)))
    assert key is None and "1, 2 of field F could all be this one" in reason
    placed_elsewhere = {("F", 1): Block([(5e9, 6e9, "LSRK")])}
    key, reason = cont_dat.match_window(placed_elsewhere, _window(np.linspace(1e9, 2e9, 3), ("F",)))
    assert key is None and "overlaps" in reason


@pytest.mark.unit
def test_line_mask_is_every_channel_outside_the_continuum_ranges():
    window = _window(FREQ)
    block = Block([(100.0e9, 100.2e9, "LSRK"), (100.5e9, 100.9e9, "LSRK")])
    assert list(np.flatnonzero(cont_dat.line_mask(block, window))) == [0, 3, 4, 5, 10, 11]
    assert not cont_dat.line_mask(Block([(1.0, 2.0, "LSRK")], "ALL"), window).any()
    none = cont_dat.line_mask(Block(whole="NONE"), window)
    assert none.all() and none.size == 12


def _tarball(path, text=CONT_DAT, name="pipeline/products/cont.dat"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        other = tarfile.TarInfo("pipeline/products/notes.txt")
        other.size = 2
        tar.addfile(other, io.BytesIO(b"hi"))
        data = text.encode()
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return path


def _delivery(tmp_path, with_tarball=True, text=CONT_DAT):
    root = tmp_path / "tree"
    folder = root / MEMBER
    (folder / "calibrated").mkdir(parents=True)
    if with_tarball:
        _tarball(folder / "calibration" / "member.uid___A001_X1_X3.auxproducts.tgz", text)
    with (root / "ous_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["eb_uid", "member_folder"])
        writer.writeheader()
        writer.writerow({"eb_uid": EB, "member_folder": MEMBER})
    return root, folder / "calibrated" / f"{EB}.ms.split.cal"


@pytest.mark.unit
def test_read_cont_dat_reads_the_tarball_without_extracting(tmp_path):
    member = tmp_path / "member"
    _tarball(member / "calibration" / "m.auxproducts.tgz")

    text, where = cont_dat.read_cont_dat(member)

    assert text == CONT_DAT and where == "m.auxproducts.tgz"
    assert not (member / "calibration" / "pipeline").exists()
    assert cont_dat.read_cont_dat(tmp_path / "none") == (
        None,
        "no *auxproducts.tgz in none/calibration",
    )
    _tarball(tmp_path / "bare" / "calibration" / "b.auxproducts.tgz", name="other.dat")
    assert cont_dat.read_cont_dat(tmp_path / "bare") == (None, "no cont.dat in b.auxproducts.tgz")


@pytest.mark.unit
def test_find_cont_dat_follows_the_ous_index_to_the_member(tmp_path):
    root, ms = _delivery(tmp_path)

    assert cont_dat.find_cont_dat(ms)[0] == CONT_DAT, "the index nearest above the MS"
    assert cont_dat.find_cont_dat(tmp_path / f"{EB}.ms", root / "ous_index.csv")[0] == CONT_DAT
    assert cont_dat.find_cont_dat(tmp_path / f"{EB}.ms") == (None, f"no ous_index.csv for {EB}.ms")
    assert "names no execution block" in cont_dat.find_cont_dat(ms.with_name("x.ms"))[1]
    other = ms.with_name("uid___A002_X9_X9.ms")
    assert "has no member folder" in cont_dat.find_cont_dat(other)[1]

    root, ms = _delivery(tmp_path / "bare", with_tarball=False)
    assert "no *auxproducts.tgz" in cont_dat.find_cont_dat(ms)[1]


@pytest.fixture
def fake_tables(monkeypatch):
    """``casacore.tables`` serving the registered tables by path."""
    tables: dict[str, FakeTable] = {}
    package = ModuleType("casacore")
    module = ModuleType("casacore.tables")
    module.table = lambda path, **kwargs: tables[str(path)]
    package.tables = module
    monkeypatch.setitem(sys.modules, "casacore", package)
    monkeypatch.setitem(sys.modules, "casacore.tables", module)
    return tables


def _single_window_ms(tables, path="w.ms", freq=None, frame=5):
    freq = FREQ if freq is None else freq
    tables[f"{path}/SPECTRAL_WINDOW"] = FakeTable(
        {"CHAN_FREQ": np.array([freq]), "MEAS_FREQ_REF": np.array([frame])}
    )
    tables[path] = FakeTable(
        {"FIELD_ID": np.array([1, 1, 1, 2]), "TIME": np.array([10.0, 20.0, 30.0, 99.0])}
    )
    tables[f"{path}/FIELD"] = FakeTable(
        {
            "NAME": np.array(["phase cal", "NGC1", "NGC1"]),
            "PHASE_DIR": np.array([[[0.0, 0.0]], [[1.0, -0.5]], [[1.1, -0.5]]]),
        }
    )
    return freq


@pytest.mark.unit
def test_ms_window_converts_from_the_window_frame_at_the_main_field(fake_tables, monkeypatch):
    freq = _single_window_ms(fake_tables)
    seen = []

    def factor(direction, time_s, frequency_hz, source, target):
        seen.append((direction, time_s, source, target))
        return 1.001

    monkeypatch.setattr(cont_dat, "_frame_factor", factor)
    window = cont_dat.ms_window("w.ms")

    assert window.field_names == {"NGC1"} and window.channels == 12
    np.testing.assert_array_equal(window.in_frame("TOPO"), freq)
    np.testing.assert_allclose(window.in_frame("LSRK"), freq * 1.001)
    window.in_frame("LSRK")
    assert seen == [((1.0, -0.5, "J2000"), 20.0, "TOPO", "LSRK")], "field 1, its median time, once"


@pytest.mark.unit
def test_pipeline_line_channels_matches_the_window(fake_tables, tmp_path, monkeypatch):
    root, ms = _delivery(tmp_path)
    _single_window_ms(fake_tables, frame=1)  # already LSRK: nothing to convert
    monkeypatch.setattr(
        cont_dat, "_frame_factor", lambda *a: pytest.fail("an LSRK window needs no conversion")
    )

    mask, note = cont_dat.pipeline_line_channels("w.ms", ms)

    assert note.endswith("auxproducts.tgz: field NGC1, pipeline window 23 (2 continuum ranges)")
    assert list(np.flatnonzero(mask)) == [0, 3, 4, 5, 10, 11]


@pytest.mark.unit
def test_pipeline_line_channels_gives_the_reason_it_is_not_used(fake_tables, tmp_path, monkeypatch):
    root, ms = _delivery(tmp_path, text="Field: NGC1\nSpectralWindow: 23\n100.0~100.2GHz LSRK\n")
    _single_window_ms(fake_tables, freq=np.linspace(200e9, 201e9, 4))
    monkeypatch.setattr(cont_dat, "_frame_factor", lambda *a: 1.0)
    assert cont_dat.pipeline_line_channels("w.ms", ms) == (
        None,
        "no cont.dat window of field NGC1 overlaps this one",
    )

    def broken(*a):
        raise RuntimeError("no measures data")

    monkeypatch.setattr(cont_dat, "_frame_factor", broken)
    mask, note = cont_dat.pipeline_line_channels("w.ms", ms)
    assert mask is None and "no measures data" in note
    assert cont_dat.pipeline_line_channels("w.ms", tmp_path / "x.ms")[0] is None


@pytest.mark.unit
def test_without_pipeline_lines_keeps_channels_both_call_continuum():
    ours = np.array([1, 1, 0, 1, 1, 1], bool)
    pipeline = np.array([0, 1, 1, 0, 0, 1], bool)

    line_free, info = cont_dat.without_pipeline_lines(ours, pipeline, "a.tgz: field F")

    assert list(line_free) == [True, False, False, True, True, False]
    assert info == {"cont_dat": "a.tgz: field F", "cont_dat_line_channels": 2}
    assert cont_dat.cont_dat_log(info) == "cont.dat (a.tgz: field F) adds 2 line channels"
    same, info = cont_dat.without_pipeline_lines(ours, None, "no ous_index.csv for x.ms")
    assert same is ours and info["cont_dat"] == "not used: no ous_index.csv for x.ms"
    assert cont_dat.cont_dat_log(info) == "cont.dat not used: no ous_index.csv for x.ms"
    _, info = cont_dat.without_pipeline_lines(ours, pipeline[:4], "x")
    assert info["cont_dat"] == "not used: cont.dat gives 4 channels, the window has 6"
