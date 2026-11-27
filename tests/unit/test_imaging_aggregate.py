"""The aggregate continuum: all the windows of one field in one image."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from almasim.services.imaging import aggregate, markers
from almasim.services.imaging.options import ImagingOptions
from almasim.services.imaging.products import (
    CONTINUUM_COLUMN,
    IMAGE_FILENAME,
    SINGLE_WINDOW_PLACEHOLDER,
)
from almasim.services.imaging.wsclean_plan import build_imaging_tasks

pytestmark = pytest.mark.unit

COMMON = dict(fov_fraction=1.5, beam_sampling=8, num_cores=4, max_cores_per_node=8)


def _parameters(windows: dict[int, tuple[float, float]], field: int = 4) -> pd.DataFrame:
    """One MS's rows: window id -> (beam arcsec, field of view arcsec)."""
    return pd.DataFrame(
        {
            "filename": ["/data/uid___A002_Xa_Xb.ms.split.cal"] * len(windows),
            "spectral_window_id": list(windows),
            "reference_frequency": [3.4e11] * len(windows),
            "fov_per_frequency": [fov for _, fov in windows.values()],
            "max_baseline_size": [500.0] * len(windows),
            "synthetized_beam_size": [beam for beam, _ in windows.values()],
            "target_field_ids": [field] * len(windows),
        }
    )


def test_output_channels_are_split_in_the_gaps_between_the_windows():
    ghz = 1e9
    edges = [(347 * ghz, 349 * ghz), (335 * ghz, 337 * ghz), (337.5 * ghz, 339.5 * ghz)]
    assert aggregate.division_frequencies(edges) == [337.25 * ghz, 343.25 * ghz]
    # Overlapping windows share an output channel.
    overlapping = [(335 * ghz, 337 * ghz), (336.5 * ghz, 338 * ghz), (347 * ghz, 349 * ghz)]
    assert aggregate.division_frequencies(overlapping) == [342.5 * ghz]
    assert aggregate.division_frequencies([(1.0, 2.0)]) == []


def test_the_joint_command_takes_every_ms_after_its_options():
    command = ["wsclean", "-name", "x", "-no-reorder", "-niter", "10", SINGLE_WINDOW_PLACEHOLDER]
    (one,) = aggregate.joint_commands(command, ["a.ms", "b.ms"], [3.4e11])
    assert one[-2:] == ["a.ms", "b.ms"] and SINGLE_WINDOW_PLACEHOLDER not in one
    assert "-no-reorder" not in one and "-join-channels" in one
    assert one[one.index("-channels-out") + 1] == "2"
    assert one[one.index("-channel-division-frequencies") + 1] == "340000000000.0"
    assert one[one.index("-fit-spectral-pol") + 1] == "2"

    first, second = aggregate.joint_commands(
        command, ["a.ms", "b.ms", "c.ms"], [1.0, 2.0], data_column=CONTINUUM_COLUMN, restart=True
    )
    assert "-update-model-required" in first and "-continue" not in first
    assert second.index("-continue") < second.index("a.ms") and second[-3:] == first[-3:]
    assert first[first.index("-data-column") + 1] == CONTINUUM_COLUMN
    # One band (overlapping windows, or one window left): a plain image.
    (plain,) = aggregate.joint_commands(command, ["a.ms", "b.ms"], [])
    assert "-join-channels" not in plain and "-channels-out" not in plain


def test_a_field_of_several_windows_gets_an_aggregate_task(tmp_path):
    parameters = _parameters({13: (0.60, 17.0), 19: (0.56, 16.5)})
    tasks, skipped = build_imaging_tasks(parameters, tmp_path, aggregate_continuum=True, **COMMON)
    assert skipped == 0 and [bool(t.aggregate) for t in tasks] == [False, False, True]
    joint = tasks[-1]
    folder = tmp_path / "uid___A002_Xa_Xb.ms.split"
    assert joint.output_dir == folder / "CONT_FIELD-4"
    assert joint.label == "uid___A002_Xa_Xb.ms.split_CONT_FIELD-4"
    assert [w["spw"] for w in joint.aggregate] == [13, 19]
    assert joint.aggregate[0]["command"] == tasks[0].command
    command = joint.command
    # The finest cell (0.56 / 8) and the widest image (window 13's) on it.
    assert command[command.index("-scale") + 1] == "0.07asec"
    widest = int(tasks[0].command[tasks[0].command.index("-size") + 1]) * 0.075
    pixels = int(command[command.index("-size") + 1])
    assert pixels % 2 == 0 and 0 <= pixels * 0.07 - widest < 0.15
    assert command[command.index("-name") + 1] == str(joint.output_dir / "wsclean")
    assert command[-1] == SINGLE_WINDOW_PLACEHOLDER

    # Done windows are still in the aggregate; a done aggregate is skipped.
    for task in tasks:
        task.output_dir.mkdir(parents=True)
        (task.output_dir / IMAGE_FILENAME).write_bytes(b"")
        markers.write_imaging_marker(task.output_dir, ms_path="x", spw=task.spw)
    markers.imaging_marker_path(joint.output_dir).unlink()
    (only,), skipped = build_imaging_tasks(parameters, tmp_path, aggregate_continuum=True, **COMMON)
    assert skipped == 2 and [w["spw"] for w in only.aggregate] == [13, 19]
    markers.write_imaging_marker(joint.output_dir, ms_path="x", spw=13)
    assert build_imaging_tasks(parameters, tmp_path, aggregate_continuum=True, **COMMON) == (
        [],
        3,
    )


def test_the_aggregate_is_restored_with_the_beam_of_the_delivered_one(tmp_path):
    parameters = _parameters({13: (0.60, 17.0), 19: (0.56, 16.5)})
    for column, values in {
        "continuum_bmaj_arcsec": [0.61, 0.57],
        "continuum_bmin_arcsec": [0.51, 0.48],
        "continuum_bpa_deg": [-72.0, -70.0],
        "aggregate_bmaj_arcsec": [0.59, 0.59],
        "aggregate_bmin_arcsec": [0.5, 0.5],
        "aggregate_bpa_deg": [-71.0, -71.0],
    }.items():
        parameters[column] = values
    build = dict(aggregate_continuum=True, header_beam=True, **COMMON)

    def beam(command):
        return command[command.index("-beam-shape") + 1 :][:3]

    first, second, joint = build_imaging_tasks(parameters, tmp_path, **build)[0]
    assert beam(first.command) == ["0.61asec", "0.51asec", "-72deg"]
    assert beam(second.command) == ["0.57asec", "0.48asec", "-70deg"]
    assert beam(joint.command) == ["0.59asec", "0.5asec", "-71deg"]
    assert joint.command.count("-beam-shape") == 1
    # Without an image of the windows together: the first window's mfs beam.
    alone = parameters.drop(columns=[c for c in parameters if c.startswith("aggregate_")])
    assert beam(build_imaging_tasks(alone, tmp_path, **build)[0][-1].command) == beam(first.command)
    plain = build_imaging_tasks(parameters, tmp_path, aggregate_continuum=True, **COMMON)[0]
    assert not any("-beam-shape" in task.command for task in plain)


def test_one_window_or_no_flag_gives_no_aggregate_task(tmp_path):
    two = _parameters({13: (0.6, 17.0), 19: (0.56, 16.5)})
    assert not any(t.aggregate for t in build_imaging_tasks(two, tmp_path, **COMMON)[0])
    one = _parameters({13: (0.6, 17.0)})
    tasks, _ = build_imaging_tasks(one, tmp_path, aggregate_continuum=True, **COMMON)
    assert len(tasks) == 1 and not tasks[0].aggregate
    with pytest.raises(ValueError, match="single-window"):
        build_imaging_tasks(two, tmp_path, aggregate_continuum=True, single_window=False, **COMMON)
    with pytest.raises(ValueError, match="--aggregate-continuum needs"):
        ImagingOptions(aggregate_continuum=True, single_window=False).validate()
    ImagingOptions(aggregate_continuum=True).validate()
    # The aggregate's own continuum fit can take the pipeline's line channels.
    ImagingOptions(aggregate_continuum=True, pipeline_line_free=True).validate()
    with pytest.raises(ValueError, match="--pipeline-line-free needs"):
        ImagingOptions(
            aggregate_continuum=True, aggregate_line_free=False, pipeline_line_free=True
        ).validate()


def test_the_aggregate_task_fits_the_continuum_unless_told_not_to(tmp_path):
    from almasim.services.imaging.batch_runner import _task_call

    parameters = _parameters({13: (0.60, 17.0), 19: (0.56, 16.5)})
    *windows, joint = build_imaging_tasks(parameters, tmp_path, aggregate_continuum=True, **COMMON)[
        0
    ]
    shared = {"threads": 2}
    run, kwargs = _task_call(joint, ImagingOptions(aggregate_continuum=True), shared)
    assert run is aggregate.run_aggregate_continuum_task
    assert kwargs["fit_order"] == 1 and len(kwargs["windows"]) == 2
    assert "ms_path" not in kwargs and kwargs["threads"] == 2
    _, kwargs = _task_call(
        joint, ImagingOptions(aggregate_continuum=True, aggregate_line_free=False), shared
    )
    assert kwargs["fit_order"] is None
    run, kwargs = _task_call(windows[0], ImagingOptions(aggregate_continuum=True), shared)
    assert run.__name__ == "run_wsclean_task" and "fit_order" not in kwargs


def test_clean_quality_finds_the_aggregate_folder(tmp_path):
    from almasim.services.imaging.quality.report import find_task_dirs

    for name in ("SPW-13_FIELD-4", "CONT_FIELD-4"):
        (tmp_path / "ms" / name).mkdir(parents=True)
        (tmp_path / "ms" / name / IMAGE_FILENAME).write_bytes(b"")
    assert [p.name for p in find_task_dirs(tmp_path)] == ["CONT_FIELD-4", "SPW-13_FIELD-4"]


@pytest.fixture
def stubbed(monkeypatch):
    """The MS building, the fit and WSClean replaced by stand-ins; what was run."""
    ran: list[list[str]] = []
    state = {"skipped": set(), "corrected": set(), "copied": [], "frames": set()}

    def extract(ms_path, spw, field_ids, out_path, **kwargs):
        Path(out_path).mkdir(parents=True)
        return {"rows": 10}

    def fit(ms_path, **kwargs):
        spw = int(Path(ms_path).name.split("spw")[1].split(".")[0])
        if spw in state["skipped"]:
            return {"skipped": "2 of 100 channels are line-free"}
        return {"order": 1, "line_free_channels": 90, "usable_channels": 100, "line_ranges": ""}

    def run(self, cmd, log):
        ran.append(list(cmd))
        name = Path(cmd[cmd.index("-name") + 1])
        joined = "-join-channels" in cmd
        for product in ("image", "residual", "model", "psf"):
            (name.parent / f"wsclean-{'MFS-' if joined else ''}{product}.fits").write_bytes(b"")
            if joined:
                (name.parent / f"wsclean-0000-{product}.fits").write_bytes(b"")
        self.history.append("wsclean")
        return 1

    def has_corrected(ms):
        return int(Path(ms).name.split("spw")[1].split(".")[0]) in state["corrected"]

    monkeypatch.setattr(aggregate, "extract_single_window_ms_subprocess", extract)
    monkeypatch.setattr(aggregate, "subtract_uv_continuum_subprocess", fit)
    monkeypatch.setattr(aggregate.TaskRun, "run", run)
    monkeypatch.setattr(
        aggregate,
        "stamp_fits",
        lambda path, header, spectral_frame=None: state["frames"].add(spectral_frame),
    )
    monkeypatch.setattr(aggregate, "has_corrected_data", has_corrected)
    monkeypatch.setattr(
        aggregate, "call_in_child", lambda target, kwargs, **kw: state["copied"].append(kwargs)
    )
    edges = {13: (335e9, 337e9), 15: (337.5e9, 339.5e9), 19: (349e9, 351e9)}
    monkeypatch.setattr(
        aggregate,
        "band_edges",
        lambda ms: edges[int(Path(ms).name.split("spw")[1].split(".")[0])],
    )
    return ran, state


def _windows(*spws: int) -> list[dict]:
    return [
        {
            "ms_path": "/data/in.ms",
            "spw": spw,
            "field_ids": [4],
            "command": ["wsclean", "-name", "w", SINGLE_WINDOW_PLACEHOLDER],
            "member_windows": [],
        }
        for spw in spws
    ]


def _run(out: Path, windows: list[dict], **kwargs):
    command = ["wsclean", "-name", str(out / "wsclean"), "-no-reorder", SINGLE_WINDOW_PLACEHOLDER]
    return aggregate.run_aggregate_continuum_task(
        command=command, output_dir=str(out), windows=windows, threads=2, **kwargs
    )


def test_the_task_images_the_windows_together_and_names_the_products(tmp_path, stubbed):
    ran, state = stubbed
    out = tmp_path / "CONT_FIELD-4"
    result = _run(out, _windows(13, 15, 19), cube_command=["ignored"], noise_products=True)
    (cmd,) = ran
    assert [Path(ms).name for ms in cmd[-3:]] == [
        "w0.spw13.single.ms",
        "w1.spw15.single.ms",
        "w2.spw19.single.ms",
    ]
    assert cmd[cmd.index("-channels-out") + 1] == "3" and "-data-column" not in cmd
    assert sorted(p.name for p in out.glob("*.fits")) == [
        "wsclean-image.fits",
        "wsclean-model.fits",
        "wsclean-psf.fits",
        "wsclean-residual.fits",
    ]
    assert not list(out.glob("*.single.ms")), "the single-window MSs are removed"
    assert markers.is_imaging_complete(out, (IMAGE_FILENAME,))
    assert result["imaged_spws"] == [13, 15, 19] and result["channels"] == 3
    assert state["frames"] == {None}, "single EBs' windows stay topocentric, as WSClean writes"
    payload = json.loads(markers.imaging_marker_path(out).read_text())
    assert payload["spw"] == 13
    log = markers.imaging_log_path(out).read_text()
    assert "aggregate continuum of windows 13, 15, 19 (3 output channel(s))" in log
    assert not state["copied"]


def test_a_window_without_a_continuum_fit_is_left_out(tmp_path, stubbed):
    ran, state = stubbed
    state["skipped"] = {15}
    out = tmp_path / "CONT"
    result = _run(out, _windows(13, 15, 19), fit_order=1, continuum_restart=True)
    first, second = ran
    assert [Path(ms).name for ms in first[-2:]] == ["w0.spw13.single.ms", "w2.spw19.single.ms"]
    assert first[first.index("-data-column") + 1] == CONTINUUM_COLUMN and "-continue" in second
    assert result["imaged_spws"] == [13, 19] and result["spws"] == [13, 15, 19]
    assert "window 15 left out of the aggregate continuum" in (
        markers.imaging_log_path(out).read_text()
    )

    state["skipped"] = {13, 15, 19}
    with pytest.raises(RuntimeError, match="every fit was skipped"):
        _run(out, _windows(13, 15, 19), uv_contsub_order=1)
    assert markers.imaging_failure_marker_path(out).is_file()


def test_windows_imaged_together_read_one_data_column(tmp_path, stubbed):
    ran, state = stubbed
    state["corrected"] = {13}  # gains applied to window 13 only
    _run(tmp_path / "CONT", _windows(13, 19))
    assert [Path(k["ms_path"]).name for k in state["copied"]] == ["w1.spw19.single.ms"]


def test_the_task_refuses_what_it_cannot_do(tmp_path, stubbed):
    out = tmp_path / "CONT"
    with pytest.raises(ValueError, match="at least two windows"):
        _run(out, _windows(13))
    with pytest.raises(ValueError, match="selfcal_solints needs casa_data"):
        _run(out, _windows(13, 19), selfcal_solints=["inf"])
    members = _windows(13, 19)
    members[0]["member_windows"] = [("/data/a.ms", 13, [4])]
    with pytest.raises(ValueError, match="member windows need casa_data"):
        _run(out, members)
    assert not out.exists(), "a refused call leaves no folder"
    assert aggregate.aggregate_directory(Path("/x/SPW-13_NGC1"), 13) == Path("/x/CONT_NGC1")
    assert aggregate.aggregate_directory(Path("/x/SPW-3"), 3) == Path("/x/CONT")


def test_selfcal_members_scratch_and_the_primary_beam_are_the_window_tasks(
    tmp_path, stubbed, monkeypatch
):
    ran, state = stubbed
    calibrated: list[tuple[str, list[str]]] = []
    built: list[list] = []

    def selfcal(ms, command, **kwargs):
        assert kwargs["keep_gains"] is False and kwargs["solints"] == ["inf"]
        calibrated.append((Path(ms).name, list(command)))
        return {"applied": True}

    def member_ms(members, out_path, **kwargs):
        built.append(list(members))
        Path(out_path).mkdir(parents=True)
        return {"rows": 5}

    from almasim.services.imaging import task as imaging_task

    monkeypatch.setattr(aggregate, "self_calibrate", selfcal)
    monkeypatch.setattr(aggregate, "build_member_window_ms", member_ms)
    monkeypatch.setattr(
        imaging_task, "pb_correct_task", lambda out, diameter: {IMAGE_FILENAME: {"d": diameter}}
    )
    monkeypatch.setattr(imaging_task, "pb_note", lambda info: f"dish {info['d']} m")
    windows = _windows(13, 19)
    windows[0]["member_windows"] = [("/data/a.ms", 13, [4]), ("/data/b.ms", 21, [2])]
    windows[0]["command"] = ["wsclean", "-temp-dir", "__SCRATCH__", SINGLE_WINDOW_PLACEHOLDER]
    out = tmp_path / "CONT_NGC1"
    command = [
        "wsclean", "-name", str(out / "wsclean"), "-temp-dir", "__SCRATCH__",
        SINGLE_WINDOW_PLACEHOLDER,
    ]  # fmt: skip
    from almasim.services.imaging.provenance import Provenance

    aggregate.run_aggregate_continuum_task(
        command=command,
        output_dir=str(out),
        windows=windows,
        threads=2,
        scratch_root=str(tmp_path / "scratch"),
        selfcal_solints=["inf"],
        casa_data="/casa",
        pipeline_line_free=True,
        pb_dish_diameter=12.0,
        provenance=Provenance(spw=13, field="NGC1"),
    )
    assert built == [windows[0]["member_windows"]]
    assert [name for name, _ in calibrated] == ["w0.spw13.single.ms", "w1.spw19.single.ms"]
    scratch = calibrated[0][1][calibrated[0][1].index("-temp-dir") + 1]
    assert Path(scratch).parent == tmp_path / "scratch" and calibrated[0][1][-1].endswith(
        "w0.spw13.single.ms"
    )
    assert not list((tmp_path / "scratch").iterdir()), "the scratch folder is removed"
    (cmd,) = ran
    assert cmd[cmd.index("-temp-dir") + 1] == scratch
    log = markers.imaging_log_path(out).read_text()
    assert "primary-beam corrected wsclean-image.fits: dish 12.0 m" in log
    assert "window 13: self-calibration on its own continuum" in log


class _Table:
    """A casacore table stand-in with the calls the column helpers make."""

    def __init__(self, columns):
        self.columns = columns

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def colnames(self):
        return list(self.columns)

    def nrows(self):
        return len(self.columns["DATA"])

    def getcell(self, name, row):
        return self.columns[name][row]

    def getcoldesc(self, name):
        return {"of": name}

    def addcols(self, description):
        self.columns[description["name"]] = [None] * self.nrows()

    def getcol(self, name, start, count):
        return self.columns[name][start : start + count]

    def putcol(self, name, values, start, count):
        self.columns[name][start : start + count] = values

    def close(self):
        self.closed = True


def test_the_column_helpers_read_and_fill_the_ms(monkeypatch):
    import sys
    from types import ModuleType

    import numpy as np

    from almasim.services.imaging import ms_geometry

    data = _Table({"DATA": list(range(5))})
    window = _Table(
        {"CHAN_FREQ": [np.array([100.0, 102.0, 104.0])], "CHAN_WIDTH": [np.array([-2.0] * 3)]}
    )
    tables = {"x.ms": data, "x.ms::SPECTRAL_WINDOW": window}
    module = ModuleType("casacore.tables")
    module.table = lambda path, **kwargs: tables[str(path)]
    module.makecoldesc = lambda name, desc: {"name": name, **desc}
    package = ModuleType("casacore")
    package.tables = module
    monkeypatch.setitem(sys.modules, "casacore", package)
    monkeypatch.setitem(sys.modules, "casacore.tables", module)
    monkeypatch.setattr(ms_geometry, "import_casacore_tables", lambda: module.table)

    assert aggregate.band_edges("x.ms") == (99.0, 105.0)
    assert aggregate.has_corrected_data("x.ms") is False
    assert aggregate.copy_data_to_corrected("x.ms", rows_per_chunk=2) == {"copied": True, "rows": 5}
    assert data.columns["CORRECTED_DATA"] == data.columns["DATA"] and data.closed
    assert aggregate.has_corrected_data("x.ms") is True
    assert aggregate.copy_data_to_corrected("x.ms") == {"copied": False}


def test_a_window_that_cannot_be_built_or_fitted_fails_the_task(tmp_path, stubbed, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    out = tmp_path / "CONT"
    monkeypatch.setattr(aggregate, "subtract_uv_continuum_subprocess", broken)
    with pytest.raises(RuntimeError, match="Could not fit the continuum of window 13"):
        _run(out, _windows(13, 19), fit_order=1)
    monkeypatch.setattr(aggregate, "extract_single_window_ms_subprocess", broken)
    with pytest.raises(RuntimeError, match="Could not build the MS of window 13: disk full"):
        _run(out, _windows(13, 19))
    assert markers.imaging_failure_marker_path(out).is_file()


@pytest.mark.unit
def test_member_windows_are_labelled_lsrk(tmp_path, stubbed, monkeypatch):
    """A member's windows are regridded onto an LSRK grid; WSClean writes TOPOCENT."""
    _, state = stubbed

    def member_ms(members, out_path, **kwargs):
        Path(out_path).mkdir(parents=True)
        return {"rows": 5}

    monkeypatch.setattr(aggregate, "build_member_window_ms", member_ms)
    windows = _windows(13, 19)
    for window in windows:
        window["member_windows"] = [("/data/a.ms", window["spw"], [4])]
    out = tmp_path / "CONT_NGC1"
    aggregate.run_aggregate_continuum_task(
        command=["wsclean", "-name", str(out / "wsclean"), SINGLE_WINDOW_PLACEHOLDER],
        output_dir=str(out),
        windows=windows,
        threads=2,
        casa_data="/casa",
    )
    assert state["frames"] == {"LSRK"}
