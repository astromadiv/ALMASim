"""Unit tests for almasim.services.archive.calibration.

CASA dependencies (casatools, casatasks) are mocked at the function level
using patch.dict(sys.modules) so that the module-level sys.modules state is
never permanently polluted. This prevents cross-test interference with
tests that require the *real* casatools (e.g. test_measurement_set_native_ms).
"""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import only the pure-Python helpers that don't require CASA at import time.
# Functions that call `from casatools import ...` inside their body are tested
# with explicit patch.dict(sys.modules) in each test.
from almasim.services.archive.calibration import (
    apply_delivered_calibration,
    calibration_not_delivered,
    create_calibrated_measurement_sets,
    find_calibration_directory,
    find_raw_ms_directories,
    science_spws,
    split_calibrated_science_ms,
)
from almasim.services.archive.calibration.apply import (
    _prepare_working_directory,
    _safe_extract_tar,
)
from almasim.services.archive.calibration.cleanup import (
    _is_relative_to,
    _remove_tree_after_success,
)
from almasim.services.archive.calibration.discovery import _parse_applycal_calls
from almasim.services.archive.calibration.run import _prepare_casa_environment


@pytest.fixture(autouse=True)
def _skip_raw_ms_row_check():
    """Neutralise the raw-MS row check for tests that fabricate empty MS directories.

    ``create_calibrated_measurement_sets`` verifies each raw MS has rows before
    touching CASA, which needs a real table. Tests that care about that check
    patch ``verify_measurement_set`` themselves; their patch applies inside this
    one and wins.
    """
    with patch("almasim.services.archive.calibration.run.verify_measurement_set", return_value=1):
        yield


# ===========================================================================
# helpers
# ===========================================================================


def _make_flagversions_tgz(cal_dir: Path, uid: str, versions=("Pipeline_Final",)) -> Path:
    """Create ``<uid>.ms.flagversions.tgz`` as ALMA delivers it next to the calapply file."""
    src = cal_dir / "_build" / f"{uid}.ms.flagversions"
    src.mkdir(parents=True, exist_ok=True)
    (src / "FLAG_VERSION_LIST").write_text(
        "".join(f"{name} : {name} flags\n" for name in versions), encoding="utf-8"
    )
    for name in versions:
        (src / f"flags.{name}").mkdir(exist_ok=True)
        (src / f"flags.{name}" / "table.dat").write_bytes(b"\x00")
    tar_path = cal_dir / f"{uid}.ms.flagversions.tgz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(src, arcname=src.name)
    return tar_path


def _make_calibration_dir(
    tmp_path: Path, uid: str = "uid___A001_X1_X1", with_flags: bool = True
) -> Path:
    """Create a fake calibration directory with a .calapply.txt file (and flag versions)."""
    cal_dir = tmp_path / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / f"{uid}.ms.calapply.txt").write_text(
        f'applycal(vis="{uid}.ms", gaintable=["{uid}.ms.gcal"], intent="TARGET")\n'
    )
    if with_flags:
        _make_flagversions_tgz(cal_dir, uid)
    return cal_dir


def _make_raw_ms(tmp_path: Path, uid: str = "uid___A001_X1_X1") -> Path:
    """Create a fake raw MS directory."""
    ms_dir = tmp_path / f"{uid}.ms"
    ms_dir.mkdir(parents=True, exist_ok=True)
    return ms_dir


def _mock_casatools(spws=(0, 1, 2), nchan=lambda spw: 128 if spw in [0, 2] else 2):
    """Return a MagicMock that looks like casatools with msmetadata."""
    mock_msmd = MagicMock()
    mock_msmd.spwsforintent.return_value = list(spws)
    mock_msmd.nchan.side_effect = nchan
    mock_msmd.intents.return_value = ["OBSERVE_TARGET#ON_SOURCE"]
    mock_casatools = MagicMock()
    mock_casatools.msmetadata.return_value = mock_msmd
    return mock_casatools, mock_msmd


def _fake_casatasks(applycal):
    """A casatasks module whose applycal is *applycal*."""
    fake_casatasks = MagicMock()
    fake_casatasks.applycal = applycal
    fake_casatasks.clearcal = MagicMock()
    fake_casatasks.mstransform = MagicMock()
    return fake_casatasks


# ===========================================================================
# _safe_extract_tar
# ===========================================================================


@pytest.mark.unit
def test_safe_extract_tar_normal_file(tmp_path):
    """Normal tar member is extracted and returned in list."""
    src_file = tmp_path / "data.txt"
    src_file.write_text("content")
    tar_path = tmp_path / "archive.tgz"
    dest = tmp_path / "dest"
    dest.mkdir()
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(src_file, arcname="data.txt")
    extracted = _safe_extract_tar(tar_path, dest)
    assert len(extracted) == 1
    assert (dest / "data.txt").is_file()


@pytest.mark.unit
def test_safe_extract_tar_skips_parent_traversal(tmp_path):
    """Tar members with .. in path are skipped."""
    src_file = tmp_path / "x.txt"
    src_file.write_text("x")
    tar_path = tmp_path / "traversal.tgz"
    dest = tmp_path / "dest"
    dest.mkdir()
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(src_file, arcname="x.txt")

    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()

    members[0].name = "../outside.txt"
    bad_tar = tmp_path / "bad_traversal.tgz"
    with tarfile.open(bad_tar, "w:gz") as tf2:
        for m in members:
            tf2.addfile(m, src_file.open("rb"))

    _safe_extract_tar(bad_tar, dest)
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.unit
def test_safe_extract_tar_multiple_files(tmp_path):
    """Multiple safe files are all extracted."""
    dest = tmp_path / "dest"
    dest.mkdir()
    tar_path = tmp_path / "multi.tgz"
    file1 = tmp_path / "a.txt"
    file2 = tmp_path / "b.txt"
    file1.write_text("a")
    file2.write_text("b")
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(file1, arcname="a.txt")
        tf.add(file2, arcname="b.txt")
    extracted = _safe_extract_tar(tar_path, dest)
    assert len(extracted) == 2


# ===========================================================================
# find_calibration_directory
# ===========================================================================


def _make_member(root: Path, raw_uids: list[str], calibrated_uids: list[str] | None) -> Path:
    """An ALMA member OUS: raw ASDMs, and calibration for some of them (None: no directory)."""
    member = root / "2022.1.0.S" / "science_goal.uid___S" / "group.uid___G" / "member.uid___M"
    for uid in raw_uids:
        (member / "raw" / f"{uid}.asdm.sdm").mkdir(parents=True)
    if calibrated_uids is not None:
        (member / "calibration").mkdir(parents=True)
        for uid in calibrated_uids:
            (member / "calibration" / f"{uid}.ms.calapply.txt").write_text("")
    return member


@pytest.mark.unit
def test_calibration_not_delivered_for_a_block_the_member_left_out(tmp_path):
    """A raw ASDM without a calapply in its member's calibration directory was never calibrated."""
    _make_member(tmp_path, ["uid___A", "uid___B"], calibrated_uids=["uid___A"])
    assert calibration_not_delivered(tmp_path, "uid___B")
    assert not calibration_not_delivered(tmp_path, "uid___A")


@pytest.mark.unit
def test_calibration_not_delivered_needs_the_calibration_directory(tmp_path):
    """Without any calibration directory (auxiliary not downloaded) the UID still fails later."""
    _make_member(tmp_path, ["uid___A"], calibrated_uids=None)
    assert not calibration_not_delivered(tmp_path, "uid___A")
    assert not calibration_not_delivered(tmp_path, "uid___UNKNOWN")


@pytest.mark.unit
def test_find_calibration_directory_finds_dir(tmp_path):
    """find_calibration_directory returns the calibration directory."""
    uid = "uid___A001_X1_X1"
    cal_dir = _make_calibration_dir(tmp_path, uid)
    result = find_calibration_directory(tmp_path, asdm_uid=uid)
    assert result == cal_dir.resolve()


@pytest.mark.unit
def test_find_calibration_directory_no_match_raises(tmp_path):
    """No matching calibration directory raises RuntimeError."""
    with pytest.raises(RuntimeError, match="No calibration directory found"):
        find_calibration_directory(tmp_path, asdm_uid="uid___MISSING")


@pytest.mark.unit
def test_find_calibration_directory_no_uid_single_ok(tmp_path):
    """A single calibration dir without asdm_uid is returned."""
    _make_calibration_dir(tmp_path, "uid___A001_X1_X1")
    result = find_calibration_directory(tmp_path)
    assert result.name == "calibration"


@pytest.mark.unit
def test_find_calibration_directory_no_uid_multiple_raises(tmp_path):
    """Multiple calibration dirs without asdm_uid raises RuntimeError."""
    (tmp_path / "A" / "calibration").mkdir(parents=True)
    (tmp_path / "B" / "calibration").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Found more than one"):
        find_calibration_directory(tmp_path)


# ===========================================================================
# find_raw_ms_directories
# ===========================================================================


@pytest.mark.unit
def test_find_raw_ms_directories_finds_ms(tmp_path):
    """find_raw_ms_directories discovers *.ms directories."""
    ms = _make_raw_ms(tmp_path)
    results = find_raw_ms_directories(tmp_path)
    assert ms.resolve() in results


@pytest.mark.unit
def test_find_raw_ms_directories_specific_uid(tmp_path):
    """find_raw_ms_directories filters by UID."""
    _make_raw_ms(tmp_path, "uid___A001_X1_X1")
    target = _make_raw_ms(tmp_path, "uid___A001_X1_X2")
    results = find_raw_ms_directories(tmp_path, asdm_uid="uid___A001_X1_X2")
    assert results == [target.resolve()]


@pytest.mark.unit
def test_find_raw_ms_directories_missing_root_raises(tmp_path):
    """Non-existent root raises RuntimeError."""
    with pytest.raises(RuntimeError, match="does not exist or is not a directory"):
        find_raw_ms_directories(tmp_path / "nope")


@pytest.mark.unit
def test_find_raw_ms_directories_empty_raises(tmp_path):
    """No MS directories found raises RuntimeError."""
    with pytest.raises(RuntimeError, match="No raw \\*.ms directories found"):
        find_raw_ms_directories(tmp_path)


@pytest.mark.unit
def test_find_raw_ms_directories_missing_uid_raises(tmp_path):
    """Specific UID not found raises RuntimeError."""
    _make_raw_ms(tmp_path, "uid___A001_X1_X1")
    with pytest.raises(RuntimeError, match="No raw MS named"):
        find_raw_ms_directories(tmp_path, asdm_uid="uid___MISSING")


# ===========================================================================
# _parse_applycal_calls
# ===========================================================================


@pytest.mark.unit
def test_parse_applycal_calls_basic(tmp_path):
    """_parse_applycal_calls parses a simple applycal call."""
    calapply = tmp_path / "test.ms.calapply.txt"
    calapply.write_text('applycal(vis="test.ms", gaintable=["test.gcal"], intent="TARGET")\n')
    calls = _parse_applycal_calls(calapply)
    assert len(calls) == 1
    assert calls[0]["vis"] == "test.ms"
    assert calls[0]["gaintable"] == ["test.gcal"]
    assert calls[0]["intent"] == "TARGET"


@pytest.mark.unit
def test_parse_applycal_calls_multiple(tmp_path):
    """_parse_applycal_calls handles multiple applycal blocks."""
    calapply = tmp_path / "multi.calapply.txt"
    calapply.write_text(
        'applycal(vis="a.ms", gaintable=["a.gcal"])\napplycal(vis="b.ms", gaintable=["b.gcal"])\n'
    )
    calls = _parse_applycal_calls(calapply)
    assert len(calls) == 2


@pytest.mark.unit
def test_parse_applycal_calls_no_applycal_raises(tmp_path):
    """No applycal() calls in file raises RuntimeError."""
    calapply = tmp_path / "empty.calapply.txt"
    calapply.write_text("# This file has no applycal calls\nx = 1\n")
    with pytest.raises(RuntimeError, match="No applycal calls found"):
        _parse_applycal_calls(calapply)


@pytest.mark.unit
def test_parse_applycal_calls_ignores_non_applycal(tmp_path):
    """Non-applycal function calls in file are ignored."""
    calapply = tmp_path / "mixed.calapply.txt"
    calapply.write_text('flagcmd(vis="test.ms")\napplycal(vis="test.ms", gaintable=["g.gcal"])\n')
    calls = _parse_applycal_calls(calapply)
    assert len(calls) == 1


@pytest.mark.unit
def test_find_input_casa_data_returns_populated_runtime_dir(tmp_path):
    """find_input_casa_data should return the first populated .casa-data directory."""
    from almasim.services.casa import find_input_casa_data

    input_root = tmp_path / "input"
    candidate = input_root / "nested" / ".casa-data"
    candidate.mkdir(parents=True)
    (candidate / "readme.txt").write_text("ok")
    (candidate / "geodetic").mkdir()

    result = find_input_casa_data(input_root)

    assert result == candidate.resolve()


@pytest.mark.unit
@patch("almasim.services.archive.calibration.run.tempfile.mkdtemp")
def test_prepare_casa_environment_honors_explicit_casa_data_root(mock_mkdtemp, tmp_path):
    """Explicit casa_data_root should be used directly with an isolated workspace."""
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    explicit = tmp_path / "explicit-casa"
    input_root.mkdir()
    output_root.mkdir()

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mock_mkdtemp.return_value = str(workspace)

    casa_data, workspace_root = _prepare_casa_environment(input_root, output_root, explicit)

    assert casa_data == explicit.resolve()
    assert workspace_root == workspace


@pytest.mark.unit
@patch("almasim.services.archive.calibration.run.tempfile.mkdtemp")
def test_prepare_casa_environment_falls_back_to_input_runtime_data(
    mock_mkdtemp,
    tmp_path,
):
    """When output/runtime data is missing, calibration should reuse populated input data."""
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()

    input_casa = input_root / "delivery" / ".casa-data"
    input_casa.mkdir(parents=True)
    (input_casa / "readme.txt").write_text("ok")
    (input_casa / "geodetic").mkdir()

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mock_mkdtemp.return_value = str(workspace)

    casa_data, workspace_root = _prepare_casa_environment(input_root, output_root, None)

    assert casa_data == input_casa.resolve()
    assert workspace_root == workspace


# ===========================================================================
# _is_relative_to
# ===========================================================================


@pytest.mark.unit
def test_is_relative_to_true():
    """Child is relative to parent."""
    assert _is_relative_to(Path("/a/b/c"), Path("/a/b")) is True


@pytest.mark.unit
def test_is_relative_to_false():
    """Unrelated path is not relative."""
    assert _is_relative_to(Path("/x/y"), Path("/a/b")) is False


@pytest.mark.unit
def test_is_relative_to_same_path():
    """A path is relative to itself."""
    assert _is_relative_to(Path("/a/b"), Path("/a/b")) is True


# ===========================================================================
# _remove_tree_after_success
# ===========================================================================


@pytest.mark.unit
def test_remove_tree_after_success_removes(tmp_path):
    """_remove_tree_after_success removes the target when safe."""
    target = tmp_path / "intermediate"
    target.mkdir()
    (target / "file.txt").write_text("junk")
    protected = [tmp_path / "output.ms"]
    _remove_tree_after_success(target, protected)
    assert not target.exists()


@pytest.mark.unit
def test_remove_tree_after_success_skips_protected(tmp_path):
    """Protected output inside target prevents deletion."""
    target = tmp_path / "working"
    target.mkdir()
    protected_path = target / "output.ms"
    protected_path.mkdir()
    _remove_tree_after_success(target, [protected_path])
    assert target.exists()


@pytest.mark.unit
def test_remove_tree_after_success_nonexistent_is_noop(tmp_path):
    """Non-existent target is silently ignored."""
    _remove_tree_after_success(tmp_path / "nope", [])


@pytest.mark.unit
def test_remove_tree_after_success_emits_log(tmp_path):
    """Log message is emitted after removal."""
    target = tmp_path / "to_remove"
    target.mkdir()
    messages = []
    _remove_tree_after_success(target, [], logger_fn=messages.append)
    assert any("Removed intermediate data" in m for m in messages)


# ===========================================================================
# _prepare_working_directory
# ===========================================================================


@pytest.mark.unit
def test_prepare_working_directory_copies_ms(tmp_path):
    """_prepare_working_directory copies the raw MS into working_dir."""
    raw_ms = _make_raw_ms(tmp_path / "raw")
    cal_dir = _make_calibration_dir(tmp_path)
    working_dir = tmp_path / "working"
    result = _prepare_working_directory(raw_ms, cal_dir, working_dir)
    assert result.is_dir()
    assert result.name == raw_ms.name


@pytest.mark.unit
def test_prepare_working_directory_raises_when_exists_no_overwrite(tmp_path):
    """RuntimeError when working MS exists and overwrite=False."""
    raw_ms = _make_raw_ms(tmp_path / "raw")
    cal_dir = _make_calibration_dir(tmp_path)
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    (working_dir / raw_ms.name).mkdir()
    with pytest.raises(RuntimeError, match="Working MS already exists"):
        _prepare_working_directory(raw_ms, cal_dir, working_dir, overwrite=False)


@pytest.mark.unit
def test_prepare_working_directory_overwrite_removes_old(tmp_path):
    """overwrite=True replaces existing working MS."""
    raw_ms = _make_raw_ms(tmp_path / "raw")
    cal_dir = _make_calibration_dir(tmp_path)
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    (working_dir / raw_ms.name).mkdir()
    result = _prepare_working_directory(raw_ms, cal_dir, working_dir, overwrite=True)
    assert result.is_dir()


# ===========================================================================
# apply_delivered_calibration
# ===========================================================================


@pytest.mark.unit
def test_apply_delivered_calibration_calls_applycal(tmp_path):
    """apply_delivered_calibration calls the applycal callable for each block."""
    uid = "uid___A001_X1_X1"
    raw_ms = _make_raw_ms(tmp_path / "raw", uid)
    cal_dir = _make_calibration_dir(tmp_path / "cal", uid)
    working_dir = tmp_path / "working"

    manager = MagicMock()
    mock_applycal = manager.applycal
    mock_clearcal = manager.clearcal
    mock_flagmanager = manager.flagmanager

    with patch(
        "almasim.services.archive.calibration.apply._normalize_intent",
        side_effect=lambda intent, ms_path: intent,
    ):
        result = apply_delivered_calibration(
            mock_applycal, mock_clearcal, raw_ms, cal_dir, working_dir, flagmanager=mock_flagmanager
        )

    mock_clearcal.assert_called_once()
    mock_applycal.assert_called_once()
    assert mock_applycal.call_args.kwargs["flagbackup"] is False
    assert result.name == f"{uid}.ms"
    # The pipeline's final flags are restored before the tables are applied.
    mock_flagmanager.assert_called_once_with(
        vis=str(working_dir / f"{uid}.ms"),
        mode="restore",
        versionname="Pipeline_Final",
        merge="replace",
    )
    order = [call[0] for call in manager.mock_calls if call[0] in {"flagmanager", "applycal"}]
    assert order == ["flagmanager", "applycal"]
    assert (working_dir / f"{uid}.ms.flagversions" / "FLAG_VERSION_LIST").is_file(), (
        "the flag versions are extracted next to the working MS, where flagmanager looks"
    )


@pytest.mark.unit
def test_apply_delivered_calibration_requires_flagmanager(tmp_path):
    """Forgetting the flag restore is an error, not a silent repeat of Cycle 11."""
    uid = "uid___A001_X1_X1"
    raw_ms = _make_raw_ms(tmp_path / "raw", uid)
    cal_dir = _make_calibration_dir(tmp_path / "cal", uid)
    with pytest.raises(RuntimeError, match="flagmanager"):
        apply_delivered_calibration(MagicMock(), MagicMock(), raw_ms, cal_dir, tmp_path / "w")


@pytest.mark.unit
def test_apply_delivered_calibration_fails_without_delivered_flags(tmp_path):
    """A delivery without <uid>.ms.flagversions.tgz is a recorded failure by default."""
    uid = "uid___A001_X1_X1"
    raw_ms = _make_raw_ms(tmp_path / "raw", uid)
    cal_dir = _make_calibration_dir(tmp_path / "cal", uid, with_flags=False)
    mock_applycal = MagicMock()
    with pytest.raises(RuntimeError, match="No delivered flag versions"):
        apply_delivered_calibration(
            mock_applycal, MagicMock(), raw_ms, cal_dir, tmp_path / "w", flagmanager=MagicMock()
        )
    mock_applycal.assert_not_called()


@pytest.mark.unit
def test_apply_delivered_calibration_can_skip_missing_flags_when_told(tmp_path):
    uid = "uid___A001_X1_X1"
    raw_ms = _make_raw_ms(tmp_path / "raw", uid)
    cal_dir = _make_calibration_dir(tmp_path / "cal", uid, with_flags=False)
    mock_applycal = MagicMock()
    mock_flagmanager = MagicMock()
    messages = []
    with patch(
        "almasim.services.archive.calibration.apply._normalize_intent",
        side_effect=lambda intent, ms_path: intent,
    ):
        apply_delivered_calibration(
            mock_applycal,
            MagicMock(),
            raw_ms,
            cal_dir,
            tmp_path / "w",
            logger_fn=messages.append,
            flagmanager=mock_flagmanager,
            require_pipeline_flags=False,
        )
    mock_flagmanager.assert_not_called()
    mock_applycal.assert_called_once()
    assert any("WARNING" in m and "flag versions" in m for m in messages)


@pytest.mark.unit
def test_restore_pipeline_flags_rejects_a_tarball_without_the_version(tmp_path):
    from almasim.services.archive.calibration import restore_pipeline_flags

    uid = "uid___A001_X1_X1"
    cal_dir = tmp_path / "calibration"
    cal_dir.mkdir()
    _make_flagversions_tgz(cal_dir, uid, versions=("after_deterministic_flagging",))
    working_dir = tmp_path / "w"
    working_dir.mkdir()
    with pytest.raises(RuntimeError, match="Pipeline_Final.*not in"):
        restore_pipeline_flags(MagicMock(), working_dir / f"{uid}.ms", cal_dir, working_dir)


@pytest.mark.unit
def test_apply_delivered_calibration_missing_calapply_raises(tmp_path):
    """Missing .calapply.txt raises RuntimeError."""
    uid = "uid___A001_X1_X1"
    raw_ms = _make_raw_ms(tmp_path / "raw", uid)
    cal_dir = tmp_path / "calibration"
    cal_dir.mkdir()
    working_dir = tmp_path / "working"
    with pytest.raises(RuntimeError, match="Cannot find calibration apply file"):
        apply_delivered_calibration(
            MagicMock(), MagicMock(), raw_ms, cal_dir, working_dir, flagmanager=MagicMock()
        )


@pytest.mark.unit
def test_apply_delivered_calibration_emits_log(tmp_path):
    """apply_delivered_calibration emits log messages."""
    uid = "uid___A001_X1_X1"
    raw_ms = _make_raw_ms(tmp_path / "raw", uid)
    cal_dir = _make_calibration_dir(tmp_path / "cal", uid)
    working_dir = tmp_path / "working"
    messages = []
    mock_applycal = MagicMock()
    mock_clearcal = MagicMock()

    with patch(
        "almasim.services.archive.calibration.apply._normalize_intent",
        side_effect=lambda intent, ms_path: intent,
    ):
        apply_delivered_calibration(
            mock_applycal,
            mock_clearcal,
            raw_ms,
            cal_dir,
            working_dir,
            logger_fn=messages.append,
            flagmanager=MagicMock(),
        )

    assert any("Applying" in m for m in messages)
    assert any("Restoring flag version 'Pipeline_Final'" in m for m in messages)


@pytest.mark.unit
def test_apply_delivered_calibration_resolves_relative_callib(tmp_path):
    """Relative callib paths in delivered applycal should be resolved under working_dir."""
    uid = "uid___A001_X1_X1"
    raw_ms = _make_raw_ms(tmp_path / "raw", uid)
    cal_dir = tmp_path / "calibration"
    cal_dir.mkdir(parents=True)
    (cal_dir / f"{uid}.ms.calapply.txt").write_text(
        'applycal(vis="x.ms", gaintable=["g.gcal"], callib="cal/chain.callib")\n'
    )
    _make_flagversions_tgz(cal_dir, uid)
    working_dir = tmp_path / "working"

    mock_applycal = MagicMock()
    with patch(
        "almasim.services.archive.calibration.apply._normalize_intent",
        side_effect=lambda intent, ms_path: intent,
    ):
        apply_delivered_calibration(
            mock_applycal,
            MagicMock(),
            raw_ms,
            cal_dir,
            working_dir,
            flagmanager=MagicMock(),
        )

    kwargs = mock_applycal.call_args.kwargs
    assert kwargs["callib"] == str(working_dir / "cal/chain.callib")


# ===========================================================================
# science_spws — uses casatools internally
# ===========================================================================


@pytest.mark.unit
def test_science_spws_returns_comma_separated_string(tmp_path):
    """science_spws returns SPW IDs as a comma-separated string."""
    ms_path = tmp_path / "test.ms"
    ms_path.mkdir()

    mock_casatools, mock_msmd = _mock_casatools()

    with patch.dict(sys.modules, {"casatools": mock_casatools}):
        result = science_spws(ms_path)

    assert "0" in result
    assert "2" in result
    assert "1" not in result  # nchan == 2 <= 4


@pytest.mark.unit
def test_science_spws_no_science_spws_raises(tmp_path):
    """RuntimeError when no science SPWs with >4 channels exist."""
    ms_path = tmp_path / "test.ms"
    ms_path.mkdir()

    mock_casatools, _ = _mock_casatools(spws=(0, 1), nchan=lambda spw: 1)  # all <= 4

    with patch.dict(sys.modules, {"casatools": mock_casatools}):
        with pytest.raises(RuntimeError, match="No science SPWs found"):
            science_spws(ms_path)


# ===========================================================================
# split_calibrated_science_ms
# ===========================================================================


@pytest.mark.unit
def test_split_calibrated_science_ms_calls_mstransform_and_returns_path(tmp_path):
    """split_calibrated_science_ms calls mstransform and returns the output path."""
    uid = "uid___A001_X1_X1"
    calibrated_ms = tmp_path / "working" / f"{uid}.ms"
    calibrated_ms.mkdir(parents=True)
    output_root = tmp_path / "output"

    def fake_mstransform(vis, outputvis, spw, reindex):
        Path(outputvis).mkdir(parents=True)

    with patch("almasim.services.archive.calibration.split.science_spws", return_value="0,1,2"):
        result = split_calibrated_science_ms(fake_mstransform, calibrated_ms, output_root)

    expected = output_root / f"{uid}.ms.split.cal"
    assert result == expected


@pytest.mark.unit
def test_split_calibrated_science_ms_raises_if_output_not_created(tmp_path):
    """RuntimeError if mstransform does not create the output MS."""
    uid = "uid___A001_X1_X1"
    calibrated_ms = tmp_path / "working" / f"{uid}.ms"
    calibrated_ms.mkdir(parents=True)
    output_root = tmp_path / "output"

    def bad_mstransform(vis, outputvis, spw, reindex):
        pass  # intentionally does NOT create outputvis

    with patch("almasim.services.archive.calibration.split.science_spws", return_value="0,1"):
        with pytest.raises(RuntimeError, match="Expected calibrated MS was not created"):
            split_calibrated_science_ms(bad_mstransform, calibrated_ms, output_root)


@pytest.mark.unit
def test_split_calibrated_science_ms_raises_when_exists_no_overwrite(tmp_path):
    """RuntimeError when output MS already exists and overwrite=False."""
    uid = "uid___A001_X1_X1"
    calibrated_ms = tmp_path / "working" / f"{uid}.ms"
    calibrated_ms.mkdir(parents=True)
    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / f"{uid}.ms.split.cal").mkdir()

    with patch("almasim.services.archive.calibration.split.science_spws", return_value="0"):
        with pytest.raises(RuntimeError, match="Calibrated output MS already exists"):
            split_calibrated_science_ms(MagicMock(), calibrated_ms, output_root, overwrite=False)


@pytest.mark.unit
def test_split_calibrated_science_ms_overwrite_removes_old(tmp_path):
    """overwrite=True replaces an existing output MS."""
    uid = "uid___A001_X1_X1"
    calibrated_ms = tmp_path / "working" / f"{uid}.ms"
    calibrated_ms.mkdir(parents=True)
    output_root = tmp_path / "output"
    output_root.mkdir()
    existing_out = output_root / f"{uid}.ms.split.cal"
    existing_out.mkdir()

    def fake_mstransform(vis, outputvis, spw, reindex):
        Path(outputvis).mkdir(parents=True, exist_ok=True)

    with patch("almasim.services.archive.calibration.split.science_spws", return_value="0"):
        result = split_calibrated_science_ms(
            fake_mstransform, calibrated_ms, output_root, overwrite=True
        )
    assert result == existing_out


@pytest.mark.unit
@patch("almasim.services.archive.calibration.run.tempfile.mkdtemp")
@patch("almasim.services.archive.calibration.run.configure_casa_environment")
@patch("almasim.services.archive.calibration.run.ensure_casa_runtime_data")
@patch("almasim.services.archive.calibration.run.find_calibration_directory")
@patch("almasim.services.archive.calibration.run.find_raw_ms_directories")
@patch("almasim.services.archive.calibration.run.split_calibrated_science_ms")
@patch("almasim.services.archive.calibration.run.apply_delivered_calibration")
def test_create_calibrated_measurement_sets_uses_temp_workspace_when_no_casa_root(
    mock_apply,
    mock_split,
    mock_find_raw,
    mock_find_cal,
    mock_ensure_casa,
    mock_configure_casa,
    mock_mkdtemp,
    tmp_path,
):
    """Calibration should avoid writing CASA scratch under the shared output tree by default."""
    input_root = tmp_path / "input"
    raw_ms_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    input_root.mkdir()
    raw_ms_root.mkdir()
    output_root.mkdir()

    raw_ms = raw_ms_root / "uid___A001_X1_X1.ms"
    raw_ms.mkdir()
    cal_dir = input_root / "calibration"
    cal_dir.mkdir()
    (cal_dir / f"{raw_ms.name}.calapply.txt").write_text('applycal(vis="x")\n')

    mock_find_raw.return_value = [raw_ms]
    mock_find_cal.return_value = cal_dir
    mock_apply.return_value = raw_ms
    mock_split.return_value = output_root / f"{raw_ms.name}.split.cal"

    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    mock_mkdtemp.return_value = str(scratch_root)

    fake_casatasks = _fake_casatasks(mock_apply)

    with patch.dict(sys.modules, {"casatasks": fake_casatasks}):
        results = create_calibrated_measurement_sets(
            input_root,
            raw_ms_root,
            output_root,
            skip_casa_data_update=True,
        )

    mock_configure_casa.assert_called_once()
    assert mock_configure_casa.call_args.kwargs["workspace_root"] == scratch_root
    assert results == [output_root / f"{raw_ms.name}.split.cal"]


@pytest.mark.unit
@patch("almasim.services.archive.calibration.run.tempfile.mkdtemp")
@patch("almasim.services.archive.calibration.run.configure_casa_environment")
@patch("almasim.services.archive.calibration.run.ensure_casa_runtime_data")
@patch("almasim.services.archive.calibration.run.find_calibration_directory")
@patch("almasim.services.archive.calibration.run.find_raw_ms_directories")
@patch("almasim.services.archive.calibration.run.split_calibrated_science_ms")
@patch("almasim.services.archive.calibration.run.apply_delivered_calibration")
def test_create_calibrated_measurement_sets_reuses_preflighted_output_casa_data(
    mock_apply,
    mock_split,
    mock_find_raw,
    mock_find_cal,
    mock_ensure_casa,
    mock_configure_casa,
    mock_mkdtemp,
    tmp_path,
):
    """Slurm preflighted CASA data under the output root should be reused, not recreated."""
    input_root = tmp_path / "input"
    raw_ms_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    input_root.mkdir()
    raw_ms_root.mkdir()
    output_root.mkdir()
    (output_root / ".casa-data").mkdir()
    (output_root / ".casa-data" / "readme.txt").write_text("data")
    (output_root / ".casa-data" / "geodetic").mkdir()

    raw_ms = raw_ms_root / "uid___A001_X1_X1.ms"
    raw_ms.mkdir()
    cal_dir = input_root / "calibration"
    cal_dir.mkdir()
    (cal_dir / f"{raw_ms.name}.calapply.txt").write_text('applycal(vis="x")\n')

    mock_find_raw.return_value = [raw_ms]
    mock_find_cal.return_value = cal_dir
    mock_apply.return_value = raw_ms
    mock_split.return_value = output_root / f"{raw_ms.name}.split.cal"

    fake_casatasks = _fake_casatasks(mock_apply)

    scratch_root = tmp_path / "scratch"
    mock_mkdtemp.return_value = str(scratch_root)

    with patch.dict(sys.modules, {"casatasks": fake_casatasks}):
        results = create_calibrated_measurement_sets(
            input_root,
            raw_ms_root,
            output_root,
            skip_casa_data_update=True,
        )

    mock_mkdtemp.assert_called_once_with(prefix="almasim-casa-")
    mock_configure_casa.assert_called_once()
    assert mock_configure_casa.call_args.kwargs["workspace_root"] == scratch_root
    assert mock_ensure_casa.call_args.args[0] == output_root / ".casa-data"
    assert results == [output_root / f"{raw_ms.name}.split.cal"]


@pytest.mark.unit
@patch("almasim.services.archive.calibration.run.shutil.rmtree")
@patch("almasim.services.archive.calibration.run.tempfile.mkdtemp")
@patch("almasim.services.archive.calibration.run.configure_casa_environment")
@patch("almasim.services.archive.calibration.run.ensure_casa_runtime_data")
@patch("almasim.services.archive.calibration.run.find_raw_ms_directories")
def test_create_calibrated_measurement_sets_cleans_workspace_on_failure(
    mock_find_raw,
    mock_ensure_casa,
    mock_configure_casa,
    mock_mkdtemp,
    mock_rmtree,
    tmp_path,
):
    """Temporary workspace should be removed in finally even when setup fails."""
    input_root = tmp_path / "input"
    raw_ms_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    input_root.mkdir()
    raw_ms_root.mkdir()
    output_root.mkdir()

    mock_find_raw.return_value = []
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    mock_mkdtemp.return_value = str(scratch_root)
    mock_ensure_casa.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        create_calibrated_measurement_sets(
            input_root,
            raw_ms_root,
            output_root,
            skip_casa_data_update=False,
        )

    mock_configure_casa.assert_called_once()
    mock_rmtree.assert_any_call(scratch_root, ignore_errors=True)


@pytest.mark.unit
@patch("almasim.services.archive.calibration.run.tempfile.mkdtemp")
@patch("almasim.services.archive.calibration.cleanup._remove_tree_after_success")
@patch("almasim.services.archive.calibration.run.configure_casa_environment")
@patch("almasim.services.archive.calibration.run.ensure_casa_runtime_data")
@patch("almasim.services.archive.calibration.run.find_calibration_directory")
@patch("almasim.services.archive.calibration.run.find_raw_ms_directories")
@patch("almasim.services.archive.calibration.run.split_calibrated_science_ms")
@patch("almasim.services.archive.calibration.run.apply_delivered_calibration")
def test_create_calibrated_measurement_sets_cleans_intermediate_paths(
    mock_apply,
    mock_split,
    mock_find_raw,
    mock_find_cal,
    mock_ensure_casa,
    mock_configure_casa,
    mock_remove_tree,
    mock_mkdtemp,
    tmp_path,
):
    """Successful clean_intermediate mode should request cleanup of raw/working/original roots."""
    input_root = tmp_path / "input"
    raw_ms_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    original_data_root = tmp_path / "original"
    input_root.mkdir()
    raw_ms_root.mkdir()
    output_root.mkdir()
    original_data_root.mkdir()

    raw_ms = raw_ms_root / "uid___A001_X1_X1.ms"
    raw_ms.mkdir()
    cal_dir = input_root / "calibration"
    cal_dir.mkdir()
    (cal_dir / f"{raw_ms.name}.calapply.txt").write_text('applycal(vis="x")\n')

    mock_find_raw.return_value = [raw_ms]
    mock_find_cal.return_value = cal_dir
    mock_apply.return_value = raw_ms
    split_output = output_root / f"{raw_ms.name}.split.cal"
    mock_split.return_value = split_output

    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    mock_mkdtemp.return_value = str(scratch_root)

    fake_casatasks = _fake_casatasks(mock_apply)

    with patch.dict(sys.modules, {"casatasks": fake_casatasks}):
        results = create_calibrated_measurement_sets(
            input_root,
            raw_ms_root,
            output_root,
            clean_intermediate=True,
            original_data_root=original_data_root,
            skip_casa_data_update=True,
        )

    assert results == [split_output]
    cleanup_targets = [call.args[0] for call in mock_remove_tree.call_args_list]
    assert raw_ms_root in cleanup_targets
    assert output_root / "working" in cleanup_targets
    assert original_data_root in cleanup_targets


@pytest.mark.unit
@patch("almasim.services.archive.calibration.run.tempfile.mkdtemp")
@patch("almasim.services.archive.calibration.run.configure_casa_environment")
@patch("almasim.services.archive.calibration.run.ensure_casa_runtime_data")
@patch("almasim.services.archive.calibration.run.find_calibration_directory")
@patch("almasim.services.archive.calibration.run.find_raw_ms_directories")
@patch("almasim.services.archive.calibration.run.split_calibrated_science_ms")
@patch("almasim.services.archive.calibration.run.apply_delivered_calibration")
def test_create_calibrated_measurement_sets_continue_on_error_skips_failed_uid(
    mock_apply,
    mock_split,
    mock_find_raw,
    mock_find_cal,
    mock_ensure_casa,
    mock_configure_casa,
    mock_mkdtemp,
    tmp_path,
):
    """With continue_on_error the good UIDs are produced and the bad one is reported at the end."""
    raw_ms_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    raw_a = raw_ms_root / "uid___A.ms"
    raw_b = raw_ms_root / "uid___B.ms"
    for path in (raw_a, raw_b):
        path.mkdir(parents=True)
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    mock_mkdtemp.return_value = str(scratch_root)

    mock_find_raw.return_value = [raw_a, raw_b]
    mock_find_cal.return_value = tmp_path / "calibration"

    def _apply(applycal, clearcal, raw_ms, *args, **kwargs):
        if raw_ms.name == "uid___B.ms":
            raise RuntimeError("Array has no elements")
        return raw_ms

    mock_apply.side_effect = _apply
    mock_split.side_effect = lambda mstransform, working_ms, *a, **k: (
        output_root / f"{working_ms.name}.split.cal"
    )

    messages: list[str] = []
    with patch.dict(sys.modules, {"casatasks": MagicMock()}):
        with pytest.raises(RuntimeError, match="failed for 1 of 2 UID") as exc:
            create_calibrated_measurement_sets(
                tmp_path / "input",
                raw_ms_root,
                output_root,
                skip_casa_data_update=True,
                continue_on_error=True,
                logger_fn=messages.append,
            )

    assert "uid___B" in str(exc.value)
    # Both UIDs were attempted; the good one was split.
    assert mock_apply.call_count == 2
    assert mock_split.call_count == 1
    assert any("Calibration FAILED for uid___B" in message for message in messages)
    assert any("Calibration finished for uid___A" in message for message in messages)
    # CASA log routing is requested for the whole run.
    configure_kwargs = mock_configure_casa.call_args.kwargs
    assert configure_kwargs["log_to_terminal"] is True
    assert configure_kwargs["log_file"] == output_root.resolve() / "logs" / "casa-calibrate-all.log"


@pytest.mark.unit
@patch("almasim.services.archive.calibration.run.tempfile.mkdtemp")
@patch("almasim.services.archive.calibration.run.configure_casa_environment")
@patch("almasim.services.archive.calibration.run.ensure_casa_runtime_data")
@patch("almasim.services.archive.calibration.run.find_raw_ms_directories")
@patch("almasim.services.archive.calibration.run.split_calibrated_science_ms")
@patch("almasim.services.archive.calibration.run.apply_delivered_calibration")
def test_create_calibrated_measurement_sets_skips_blocks_delivered_without_calibration(
    mock_apply,
    mock_split,
    mock_find_raw,
    mock_ensure_casa,
    mock_configure_casa,
    mock_mkdtemp,
    tmp_path,
):
    """A block ALMA never calibrated is skipped, not failed, and its old failure marker goes."""
    input_root = tmp_path / "input"
    member = _make_member(input_root, ["uid___A", "uid___B"], calibrated_uids=["uid___A"])
    raw_ms_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    raw_a = raw_ms_root / "uid___A.ms"
    raw_b = raw_ms_root / "uid___B.ms"
    for path in (raw_a, raw_b):
        path.mkdir(parents=True)
    output_root.mkdir()
    stale_marker = output_root / "uid___B.ms.split.cal.failed"
    stale_marker.write_text("{}")
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    mock_mkdtemp.return_value = str(scratch_root)
    mock_find_raw.return_value = [raw_a, raw_b]
    mock_apply.side_effect = lambda applycal, clearcal, raw_ms, *a, **k: raw_ms
    mock_split.side_effect = lambda mstransform, working_ms, *a, **k: (
        output_root / f"{working_ms.name}.split.cal"
    )

    messages: list[str] = []
    with patch.dict(sys.modules, {"casatasks": MagicMock()}):
        outputs = create_calibrated_measurement_sets(
            input_root,
            raw_ms_root,
            output_root,
            skip_casa_data_update=True,
            logger_fn=messages.append,
        )

    assert outputs == [output_root / "uid___A.ms.split.cal"]
    assert [call.args[2] for call in mock_apply.call_args_list] == [raw_a]
    assert mock_apply.call_args.args[3] == (member / "calibration").resolve()
    assert any("uid___B was delivered without calibration" in message for message in messages)
    assert not stale_marker.exists()


@pytest.mark.unit
@patch("almasim.services.archive.calibration.run.tempfile.mkdtemp")
@patch("almasim.services.archive.calibration.run.configure_casa_environment")
@patch("almasim.services.archive.calibration.run.ensure_casa_runtime_data")
@patch("almasim.services.archive.calibration.run.find_calibration_directory")
@patch("almasim.services.archive.calibration.run.find_raw_ms_directories")
@patch("almasim.services.archive.calibration.run.apply_delivered_calibration")
def test_create_calibrated_measurement_sets_fail_fast_by_default(
    mock_apply,
    mock_find_raw,
    mock_find_cal,
    mock_ensure_casa,
    mock_configure_casa,
    mock_mkdtemp,
    tmp_path,
):
    """Without continue_on_error the first failure propagates unchanged."""
    raw_ms_root = tmp_path / "raw"
    raw_a = raw_ms_root / "uid___A.ms"
    raw_a.mkdir(parents=True)
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    mock_mkdtemp.return_value = str(scratch_root)
    mock_find_raw.return_value = [raw_a]
    mock_find_cal.return_value = tmp_path / "calibration"
    mock_apply.side_effect = RuntimeError("Array has no elements")

    with patch.dict(sys.modules, {"casatasks": MagicMock()}):
        with pytest.raises(RuntimeError, match="Array has no elements"):
            create_calibrated_measurement_sets(
                tmp_path / "input",
                raw_ms_root,
                tmp_path / "output",
                skip_casa_data_update=True,
            )


@pytest.mark.unit
def test_calibration_marker_helpers(tmp_path):
    """Completion requires both the output directory and the marker; empty markers count."""
    from almasim.services.archive.calibration import (
        calibration_marker_path,
        is_calibration_complete,
        write_calibration_marker,
    )

    output_root = tmp_path / "cal"
    output_root.mkdir()
    assert is_calibration_complete(output_root, "uid___A") is False

    (output_root / "uid___A.ms.split.cal").mkdir()
    assert is_calibration_complete(output_root, "uid___A") is False

    marker = write_calibration_marker(output_root, "uid___A")
    assert marker == output_root / "uid___A.ms.split.cal.done"
    assert '"uid": "uid___A"' in marker.read_text(encoding="utf-8")
    assert is_calibration_complete(output_root, "uid___A") is True

    # A manually created empty marker is enough to trust a pre-existing output.
    (output_root / "uid___B.ms.split.cal").mkdir()
    calibration_marker_path(output_root, "uid___B").touch()
    assert is_calibration_complete(output_root, "uid___B") is True


@pytest.mark.unit
def test_split_calibrated_science_ms_writes_marker(tmp_path):
    """A successful split leaves a completion marker next to the output."""
    from almasim.services.archive.calibration import split_calibrated_science_ms

    working_ms = tmp_path / "working" / "uid___A.ms"
    working_ms.mkdir(parents=True)
    output_root = tmp_path / "cal"

    def _mstransform(vis, outputvis, spw, reindex):
        Path(outputvis).mkdir(parents=True)

    with patch("almasim.services.archive.calibration.split.science_spws", return_value="0,1"):
        result = split_calibrated_science_ms(_mstransform, working_ms, output_root)

    assert result == output_root / "uid___A.ms.split.cal"
    assert (output_root / "uid___A.ms.split.cal.done").is_file()


@pytest.mark.unit
def test_split_calibrated_science_ms_clears_stale_marker_on_overwrite(tmp_path):
    """Overwriting removes the old marker before the new split runs."""
    from almasim.services.archive.calibration import split_calibrated_science_ms

    working_ms = tmp_path / "working" / "uid___A.ms"
    working_ms.mkdir(parents=True)
    output_root = tmp_path / "cal"
    (output_root / "uid___A.ms.split.cal").mkdir(parents=True)
    (output_root / "uid___A.ms.split.cal.done").write_text("old")

    def _failing_mstransform(vis, outputvis, spw, reindex):
        raise RuntimeError("mstransform crashed")

    with patch("almasim.services.archive.calibration.split.science_spws", return_value="0"):
        with pytest.raises(RuntimeError, match="mstransform crashed"):
            split_calibrated_science_ms(
                _failing_mstransform, working_ms, output_root, overwrite=True
            )

    # The stale marker must not survive a failed redo.
    assert not (output_root / "uid___A.ms.split.cal.done").exists()


@pytest.mark.unit
@patch("almasim.services.archive.calibration.run.tempfile.mkdtemp")
@patch("almasim.services.archive.calibration.run.configure_casa_environment")
@patch("almasim.services.archive.calibration.run.ensure_casa_runtime_data")
@patch("almasim.services.archive.calibration.run.find_calibration_directory")
@patch("almasim.services.archive.calibration.run.find_raw_ms_directories")
@patch("almasim.services.archive.calibration.run.split_calibrated_science_ms")
@patch("almasim.services.archive.calibration.run.apply_delivered_calibration")
def test_create_calibrated_measurement_sets_skips_complete_and_redoes_stale(
    mock_apply,
    mock_split,
    mock_find_raw,
    mock_find_cal,
    mock_ensure_casa,
    mock_configure_casa,
    mock_mkdtemp,
    tmp_path,
):
    """Marked outputs are skipped; unmarked leftovers are redone with overwrite semantics."""
    from almasim.services.archive.calibration import write_calibration_marker

    raw_ms_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    raw_done = raw_ms_root / "uid___DONE.ms"
    raw_stale = raw_ms_root / "uid___STALE.ms"
    for path in (raw_done, raw_stale):
        path.mkdir(parents=True)
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    mock_mkdtemp.return_value = str(scratch_root)

    # DONE: output + marker.  STALE: partial output and working copy, no marker.
    (output_root / "uid___DONE.ms.split.cal").mkdir(parents=True)
    write_calibration_marker(output_root, "uid___DONE")
    (output_root / "uid___STALE.ms.split.cal").mkdir()
    stale_working = output_root / "working" / "uid___STALE.calibration"
    (stale_working / "uid___STALE.ms").mkdir(parents=True)

    mock_find_raw.return_value = [raw_done, raw_stale]
    mock_find_cal.return_value = tmp_path / "calibration"
    mock_apply.side_effect = lambda applycal, clearcal, raw_ms, *a, **k: raw_ms
    mock_split.side_effect = lambda mstransform, working_ms, *a, **k: (
        output_root / f"{working_ms.name}.split.cal"
    )

    messages: list[str] = []
    with patch.dict(sys.modules, {"casatasks": MagicMock()}):
        results = create_calibrated_measurement_sets(
            tmp_path / "input",
            raw_ms_root,
            output_root,
            skip_casa_data_update=True,
            logger_fn=messages.append,
        )

    assert results == [
        output_root / "uid___DONE.ms.split.cal",
        output_root / "uid___STALE.ms.split.cal",
    ]
    assert mock_apply.call_count == 1
    assert mock_apply.call_args.args[2] == raw_stale
    assert mock_apply.call_args.kwargs["overwrite"] is True
    assert mock_split.call_args.kwargs["overwrite"] is True
    assert any("already calibrated, skipping" in m for m in messages)
    assert any("Replacing incomplete output" in m for m in messages)
    assert any("Replacing stale working copy" in m for m in messages)
    # Working copy removed after success by default.
    assert not stale_working.exists()
    assert any("Removed working copy" in m for m in messages)


@pytest.mark.unit
@patch("almasim.services.archive.calibration.run.tempfile.mkdtemp")
@patch("almasim.services.archive.calibration.run.configure_casa_environment")
@patch("almasim.services.archive.calibration.run.ensure_casa_runtime_data")
@patch("almasim.services.archive.calibration.run.find_calibration_directory")
@patch("almasim.services.archive.calibration.run.find_raw_ms_directories")
@patch("almasim.services.archive.calibration.run.split_calibrated_science_ms")
@patch("almasim.services.archive.calibration.run.apply_delivered_calibration")
def test_create_calibrated_measurement_sets_keeps_working_copy_when_asked(
    mock_apply,
    mock_split,
    mock_find_raw,
    mock_find_cal,
    mock_ensure_casa,
    mock_configure_casa,
    mock_mkdtemp,
    tmp_path,
):
    """remove_working_copy=False leaves the per-UID working directory in place."""
    raw_ms_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    raw_a = raw_ms_root / "uid___A.ms"
    raw_a.mkdir(parents=True)
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    mock_mkdtemp.return_value = str(scratch_root)
    mock_find_raw.return_value = [raw_a]
    mock_find_cal.return_value = tmp_path / "calibration"

    def _apply(applycal, clearcal, raw_ms, calibration_dir, working_dir, **kwargs):
        (working_dir / raw_ms.name).mkdir(parents=True)
        return working_dir / raw_ms.name

    mock_apply.side_effect = _apply
    mock_split.side_effect = lambda mstransform, working_ms, *a, **k: (
        output_root / f"{working_ms.name}.split.cal"
    )

    with patch.dict(sys.modules, {"casatasks": MagicMock()}):
        create_calibrated_measurement_sets(
            tmp_path / "input",
            raw_ms_root,
            output_root,
            skip_casa_data_update=True,
            remove_working_copy=False,
        )

    assert (output_root / "working" / "uid___A.calibration" / "uid___A.ms").is_dir()


@pytest.mark.unit
def test_calibration_failure_marker_helpers(tmp_path):
    """The failure marker records the error and is cleared once the UID succeeds."""
    from almasim.services.archive.calibration import (
        calibration_failure_marker_path,
        has_calibration_failed,
        write_calibration_failure_marker,
        write_calibration_marker,
    )

    output_root = tmp_path / "cal"
    output_root.mkdir()

    assert has_calibration_failed(output_root, "uid___A") is False

    marker = write_calibration_failure_marker(
        output_root, "uid___A", "RuntimeError: boom", log_path="/logs/uid___A.calibrate.log"
    )
    assert marker == calibration_failure_marker_path(output_root, "uid___A")
    assert marker.name == "uid___A.ms.split.cal.failed"
    assert has_calibration_failed(output_root, "uid___A") is True

    payload = json.loads(marker.read_text())
    assert payload["uid"] == "uid___A"
    assert payload["error"] == "RuntimeError: boom"
    assert payload["log"] == "/logs/uid___A.calibrate.log"
    assert payload["failed_at"]

    # A later success must not leave the UID looking failed.
    write_calibration_marker(output_root, "uid___A")
    assert has_calibration_failed(output_root, "uid___A") is False


@pytest.mark.unit
@patch("almasim.services.archive.calibration.run.tempfile.mkdtemp")
@patch("almasim.services.archive.calibration.run.configure_casa_environment")
@patch("almasim.services.archive.calibration.run.ensure_casa_runtime_data")
@patch("almasim.services.archive.calibration.run.find_calibration_directory")
@patch("almasim.services.archive.calibration.run.find_raw_ms_directories")
@patch("almasim.services.archive.calibration.run.apply_delivered_calibration")
def test_failed_uid_writes_marker_and_reclaims_working_copy(
    mock_apply,
    mock_find_raw,
    mock_find_cal,
    mock_ensure_casa,
    mock_configure_casa,
    mock_mkdtemp,
    tmp_path,
):
    """A failed UID leaves a .failed marker and no multi-GB working copy behind."""
    from almasim.services.archive.calibration import has_calibration_failed

    raw_ms_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    raw_a = raw_ms_root / "uid___A.ms"
    raw_a.mkdir(parents=True)
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    mock_mkdtemp.return_value = str(scratch_root)
    mock_find_raw.return_value = [raw_a]
    mock_find_cal.return_value = tmp_path / "calibration"

    def _apply(applycal, clearcal, raw_ms, calibration_dir, working_dir, **kwargs):
        # The raw copy is made before the failure, exactly as in the real run.
        (working_dir / raw_ms.name).mkdir(parents=True)
        raise RuntimeError("No calibration directory found under /input")

    mock_apply.side_effect = _apply

    with patch.dict(sys.modules, {"casatasks": MagicMock()}):
        with pytest.raises(RuntimeError, match="failed for 1 of 1 UID"):
            create_calibrated_measurement_sets(
                tmp_path / "input",
                raw_ms_root,
                output_root,
                skip_casa_data_update=True,
                continue_on_error=True,
            )

    assert has_calibration_failed(output_root, "uid___A") is True
    payload = json.loads((output_root / "uid___A.ms.split.cal.failed").read_text())
    assert "No calibration directory found" in payload["error"]
    # The working copy is gone: this is what was stranding TBs of disk.
    assert not (output_root / "working" / "uid___A.calibration").exists()


@pytest.mark.unit
@patch("almasim.services.archive.calibration.run.tempfile.mkdtemp")
@patch("almasim.services.archive.calibration.run.configure_casa_environment")
@patch("almasim.services.archive.calibration.run.ensure_casa_runtime_data")
@patch("almasim.services.archive.calibration.run.find_calibration_directory")
@patch("almasim.services.archive.calibration.run.find_raw_ms_directories")
@patch("almasim.services.archive.calibration.run.split_calibrated_science_ms")
@patch("almasim.services.archive.calibration.run.apply_delivered_calibration")
def test_stale_failure_marker_cleared_when_uid_is_retried(
    mock_apply,
    mock_split,
    mock_find_raw,
    mock_find_cal,
    mock_ensure_casa,
    mock_configure_casa,
    mock_mkdtemp,
    tmp_path,
):
    """A retry that succeeds removes the marker left by the previous attempt."""
    from almasim.services.archive.calibration import (
        has_calibration_failed,
        write_calibration_failure_marker,
    )

    raw_ms_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    raw_a = raw_ms_root / "uid___A.ms"
    raw_a.mkdir(parents=True)
    output_root.mkdir(parents=True)
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    mock_mkdtemp.return_value = str(scratch_root)
    mock_find_raw.return_value = [raw_a]
    mock_find_cal.return_value = tmp_path / "calibration"
    mock_apply.side_effect = lambda applycal, clearcal, raw_ms, *a, **k: raw_ms
    mock_split.side_effect = lambda mstransform, working_ms, *a, **k: (
        output_root / f"{working_ms.name}.split.cal"
    )

    write_calibration_failure_marker(output_root, "uid___A", "RuntimeError: previous run")
    assert has_calibration_failed(output_root, "uid___A") is True

    with patch.dict(sys.modules, {"casatasks": MagicMock()}):
        create_calibrated_measurement_sets(
            tmp_path / "input",
            raw_ms_root,
            output_root,
            skip_casa_data_update=True,
        )

    assert has_calibration_failed(output_root, "uid___A") is False


@pytest.mark.unit
@patch("almasim.services.archive.calibration.run.tempfile.mkdtemp")
@patch("almasim.services.archive.calibration.run.configure_casa_environment")
@patch("almasim.services.archive.calibration.run.ensure_casa_runtime_data")
@patch("almasim.services.archive.calibration.run.verify_measurement_set")
@patch("almasim.services.archive.calibration.run.find_calibration_directory")
@patch("almasim.services.archive.calibration.run.find_raw_ms_directories")
@patch("almasim.services.archive.calibration.run.apply_delivered_calibration")
def test_truncated_raw_ms_fails_cleanly_instead_of_aborting_casa(
    mock_apply,
    mock_find_raw,
    mock_find_cal,
    mock_verify,
    mock_ensure_casa,
    mock_configure_casa,
    mock_mkdtemp,
    tmp_path,
):
    """A 0-row raw MS is rejected before clearcal can abort the whole process."""
    from almasim.services.archive.calibration import has_calibration_failed

    raw_ms_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    raw_a = raw_ms_root / "uid___A.ms"
    raw_a.mkdir(parents=True)
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    mock_mkdtemp.return_value = str(scratch_root)
    mock_find_raw.return_value = [raw_a]
    mock_find_cal.return_value = tmp_path / "calibration"
    mock_verify.side_effect = RuntimeError(
        "MeasurementSet has no rows, the import did not finish: uid___A.ms"
    )

    with patch.dict(sys.modules, {"casatasks": MagicMock()}):
        with pytest.raises(RuntimeError, match="failed for 1 of 1 UID"):
            create_calibrated_measurement_sets(
                tmp_path / "input",
                raw_ms_root,
                output_root,
                skip_casa_data_update=True,
                continue_on_error=True,
            )

    # The MS is checked before any CASA work is attempted on it.
    mock_verify.assert_called_once_with(raw_a)
    mock_apply.assert_not_called()
    assert has_calibration_failed(output_root, "uid___A") is True
    payload = json.loads((output_root / "uid___A.ms.split.cal.failed").read_text())
    assert "has no rows" in payload["error"]
