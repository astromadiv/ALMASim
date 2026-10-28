"""Archive stages: per-UID workers, progress and the Slurm/sync drivers (MADIV-28 T8.2).

The code lives in services/archive/pipeline; these tests moved from test_cli.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from almasim.cli import products as cli_products
from almasim.services.archive import pipeline as archive_pipeline

from ._fakes import fake_popen, patch_everywhere

runner = CliRunner()


class _FakeFuture:
    def __init__(self, *, done_after: int = 1, status: str = "finished"):
        self._calls = 0
        self._done_after = done_after
        self.status = status

    def done(self):
        self._calls += 1
        return self._calls >= self._done_after


class _FakeAsyncBackend:
    def __init__(self, futures, gathered):
        self._futures = futures
        self._gathered = gathered
        self.compute_sync_values = []
        self.gather_called = False

    def compute(self, jobs, sync=True):
        self.compute_sync_values.append(sync)
        return self._futures

    def gather(self, futures):
        self.gather_called = True
        assert futures is self._futures
        return self._gathered


class _FakeFailingFuture(_FakeFuture):
    def __init__(self, message: str):
        super().__init__(done_after=1, status="error")
        self._message = message

    def exception(self):
        return RuntimeError(self._message)


class _FakePerFutureBackend:
    """Async backend whose gather() works on single-future lists."""

    def __init__(self, futures, results):
        self._futures = futures
        self._results = results

    def compute(self, jobs, sync=True):
        return self._futures

    def gather(self, futures):
        return [self._results[self._futures.index(future)] for future in futures]


class _CrashingProcess:
    """Child that prints a casacore abort and dies with SIGABRT (-6)."""

    def __init__(self, lines: list[str] | None = None, return_code: int = -6):
        self.stdout = iter(
            lines
            or [
                "Extracting calibration tables\n",
                "terminate called after throwing an instance of 'casacore::ArrayError'\n",
                "  what():  minMax - Array has no elements\n",
            ]
        )
        self._rc = return_code

    def wait(self):
        return self._rc


def test_compute_jobs_with_progress_uses_async_and_gather(monkeypatch):
    """Slurm progress helper should compute async and gather futures."""
    backend = _FakeAsyncBackend(
        futures=[_FakeFuture(done_after=1), _FakeFuture(done_after=2)],
        gathered=[["raw1.ms"], ["raw2.ms"]],
    )

    monkeypatch.setattr(archive_pipeline, "sleep", lambda *_args, **_kwargs: None)

    results = cli_products._compute_jobs_with_progress(
        backend=backend,
        jobs=[object(), object()],
        job_uids=["uid://A", "uid://B"],
        stage_label="Slurm unpack",
    )

    assert backend.compute_sync_values == [False]
    assert backend.gather_called is True
    assert results == [["raw1.ms"], ["raw2.ms"]]


def test_preflight_casa_data_uses_resolved_path(monkeypatch, tmp_path):
    """Preflight should resolve CASA data path and pass skip flag through."""
    captured: dict[str, object] = {}

    def _fake_find_existing(input_root, output_root, casa_data_root):
        del input_root, output_root, casa_data_root
        return tmp_path / "resolved-casa"

    def _fake_ensure(path, skip_update=False, logger_fn=None):
        del logger_fn
        captured["path"] = path
        captured["skip_update"] = skip_update

    import types

    fake_unpack_mod = types.SimpleNamespace(
        ensure_casa_runtime_data=_fake_ensure,
        find_existing_casa_data=_fake_find_existing,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "almasim.services.casa.runtime_data",
        fake_unpack_mod,
    )

    cli_products._preflight_casa_data(
        output_root=tmp_path,
        casa_data_root=None,
        skip_casa_data_update=True,
    )

    assert captured["path"] == tmp_path / "resolved-casa"
    assert captured["skip_update"] is True


def test_run_unpack_jobs_slurm_exits_when_no_uids(monkeypatch, tmp_path):
    """Slurm unpack should fail early when no ASDM inputs can be discovered."""
    patch_everywhere(
        monkeypatch, cli_products, "_extract_asdm_uids_from_download_root", lambda *_: []
    )

    with pytest.raises(archive_pipeline.StageError, match="No ASDM directories"):
        cli_products._run_unpack_jobs(
            input_root=tmp_path / "input",
            output_root=tmp_path / "out",
            asdm_uids=[],
            postprocess_backend="slurm",
            postprocess_backend_kwargs={},
            casa_data_root=None,
            skip_casa_data_update=False,
            overwrite_outputs=False,
        )


def test_run_calibrate_jobs_slurm_rejects_clean_intermediate(tmp_path):
    """Slurm calibrate should reject clean-intermediate mode."""
    with pytest.raises(archive_pipeline.StageUsageError, match="clean-intermediate"):
        cli_products._run_calibrate_jobs(
            input_root=tmp_path / "input",
            raw_ms_root=tmp_path / "raw",
            output_root=tmp_path / "out",
            asdm_uids=["uid://A"],
            postprocess_backend="slurm",
            postprocess_backend_kwargs={},
            casa_data_root=None,
            skip_casa_data_update=False,
            overwrite_outputs=False,
            clean_intermediate=True,
        )


def test_run_unpack_jobs_slurm_preflights_and_submits(monkeypatch, tmp_path):
    """Slurm unpack should preflight CASA data and submit one job per UID."""
    captured: dict[str, object] = {}

    class _Backend:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

        def delayed(self, fn):
            return lambda **kwargs: kwargs

    patch_everywhere(
        monkeypatch, cli_products, "_preflight_casa_data", lambda *args, **kwargs: None
    )
    patch_everywhere(
        monkeypatch, cli_products, "create_backend", lambda *args, **kwargs: _Backend()
    )

    def _fake_compute_jobs_with_progress(**kwargs):
        captured.update(kwargs)
        return [["/tmp/raw-a.ms"]]

    patch_everywhere(
        monkeypatch, cli_products, "_compute_jobs_with_progress", _fake_compute_jobs_with_progress
    )

    outputs = cli_products._run_unpack_jobs(
        input_root=tmp_path / "input",
        output_root=tmp_path / "out",
        asdm_uids=["uid://A"],
        postprocess_backend="slurm",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=False,
        overwrite_outputs=False,
    )

    assert outputs == ["/tmp/raw-a.ms"]
    assert captured["job_uids"] == ["uid://A"]


def test_run_calibrate_jobs_slurm_preflights_and_submits(monkeypatch, tmp_path):
    """Slurm calibrate should preflight CASA data and submit one job per UID."""
    captured: dict[str, object] = {}

    class _Backend:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

        def delayed(self, fn):
            return lambda **kwargs: kwargs

    patch_everywhere(
        monkeypatch, cli_products, "_preflight_casa_data", lambda *args, **kwargs: None
    )
    patch_everywhere(
        monkeypatch, cli_products, "create_backend", lambda *args, **kwargs: _Backend()
    )

    def _fake_compute_jobs_with_progress(**kwargs):
        captured.update(kwargs)
        return [["/tmp/cal-a.ms"]]

    patch_everywhere(
        monkeypatch, cli_products, "_compute_jobs_with_progress", _fake_compute_jobs_with_progress
    )

    outputs = cli_products._run_calibrate_jobs(
        input_root=tmp_path / "input",
        raw_ms_root=tmp_path / "raw",
        output_root=tmp_path / "out",
        asdm_uids=["uid://A"],
        postprocess_backend="slurm",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=False,
        overwrite_outputs=False,
        clean_intermediate=False,
    )

    assert outputs == ["/tmp/cal-a.ms"]
    assert captured["job_uids"] == ["uid://A"]


def test_calibrate_single_uid_streams_logs_and_returns_expected_output(tmp_path, monkeypatch):
    """Calibration worker should stream subprocess logs and return the expected split.cal path."""
    calls = fake_popen(monkeypatch, ["first line\n", "second line\n"])
    monkeypatch.setenv("PYTHONPATH", "existing-path")

    uid = "uid___A001_X1_X1"
    output_root = tmp_path / "calibrated"
    expected = output_root / f"{uid}.ms.split.cal"
    expected.mkdir(parents=True)

    outputs = cli_products._calibrate_single_uid(
        input_root="/input",
        raw_ms_root="/raw",
        calibrated_output_root=str(output_root),
        asdm_uid=uid,
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite=True,
        clean_intermediate=False,
    )

    assert outputs == [str(expected)]
    cmd, kwargs = calls[0]
    assert isinstance(cmd, list)
    assert "--overwrite-outputs" in cmd
    assert "--skip-casa-data-update" in cmd

    assert kwargs["cwd"].endswith("ALMASim")
    assert "PYTHONPATH" in kwargs["env"]
    assert kwargs["env"]["PYTHONPATH"].endswith(":existing-path")


def test_calibrate_single_uid_uses_fallback_glob_when_expected_missing(tmp_path, monkeypatch):
    """Calibration worker should discover split.cal outputs via glob
    when canonical path is absent.
    """

    fake_popen(monkeypatch, ["ok\n"])

    uid = "uid___A001_X1_X2"
    output_root = tmp_path / "calibrated"
    fallback = output_root / f"{uid}_extra.ms.split.cal"
    fallback.mkdir(parents=True)

    outputs = cli_products._calibrate_single_uid(
        input_root="/input",
        raw_ms_root="/raw",
        calibrated_output_root=str(output_root),
        asdm_uid=uid,
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite=False,
        clean_intermediate=False,
    )

    assert outputs == [str(fallback)]


def test_calibrate_single_uid_raises_with_bounded_log_tail(tmp_path, monkeypatch):
    """Calibration worker should include only the recent bounded tail on subprocess failure."""

    fake_popen(monkeypatch, [f"line-{idx}\n" for idx in range(205)], returncode=7)

    uid = "uid___A001_X1_X3"
    output_root = tmp_path / "calibrated"
    output_root.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Calibration failed") as exc:
        cli_products._calibrate_single_uid(
            input_root="/input",
            raw_ms_root="/raw",
            calibrated_output_root=str(output_root),
            asdm_uid=uid,
            casa_data_root=None,
            skip_casa_data_update=True,
            overwrite=False,
            clean_intermediate=False,
        )

    msg = str(exc.value)
    assert "Return code: 7" in msg
    assert "Last 200 log lines" in msg
    assert "line-0" not in msg
    assert "line-204" in msg


def test_compute_jobs_with_progress_continue_on_error_skips_failed(monkeypatch):
    """A failed future should become a None placeholder and a recorded StageFailure."""
    futures = [_FakeFuture(done_after=1), _FakeFailingFuture("boom on B"), _FakeFuture()]
    backend = _FakePerFutureBackend(futures, results=[["a.ms"], None, ["c.ms"]])
    monkeypatch.setattr(archive_pipeline, "sleep", lambda *_args, **_kwargs: None)
    failures: list[cli_products.TaskFailure] = []

    results = cli_products._compute_jobs_with_progress(
        backend=backend,
        jobs=[object(), object(), object()],
        job_uids=["uid://A", "uid://B", "uid://C"],
        stage_label="Slurm calibrate",
        continue_on_error=True,
        log_paths=[Path("/logs/a.log"), Path("/logs/b.log"), Path("/logs/c.log")],
        failures=failures,
    )

    assert results == [["a.ms"], None, ["c.ms"]]
    assert [failure.label for failure in failures] == ["uid://B"]
    assert "boom on B" in failures[0].error
    assert failures[0].log_path == "/logs/b.log"


def test_compute_jobs_with_progress_fail_fast_still_raises(monkeypatch):
    """Without continue_on_error the helper defers to backend.gather, which raises."""

    class _RaisingBackend(_FakePerFutureBackend):
        def gather(self, futures):
            raise RuntimeError("gather failed")

    backend = _RaisingBackend([_FakeFailingFuture("boom")], results=[None])
    monkeypatch.setattr(archive_pipeline, "sleep", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="gather failed"):
        cli_products._compute_jobs_with_progress(
            backend=backend,
            jobs=[object()],
            job_uids=["uid://A"],
            stage_label="Slurm calibrate",
        )


def test_compute_jobs_with_progress_heartbeat_prints_log_tail(monkeypatch, tmp_path):
    """While jobs run, the last line of each UID log should be echoed periodically."""
    log_path = tmp_path / "uid_A.calibrate.log"
    log_path.write_text("starting\napplycal 2/5 running\n", encoding="utf-8")
    future = _FakeFuture(done_after=3)
    backend = _FakePerFutureBackend([future], results=[["a.ms"]])
    written: list[str] = []

    class _Bar:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def set_postfix_str(self, *_args):
            pass

        def update(self, *_args):
            pass

        def write(self, text):
            written.append(text)

    patch_everywhere(monkeypatch, cli_products, "tqdm", _Bar)
    # The worker "writes" to the log while the stage is running.
    monkeypatch.setattr(
        archive_pipeline, "sleep", lambda *_args, **_kwargs: os.utime(log_path, None)
    )

    cli_products._compute_jobs_with_progress(
        backend=backend,
        jobs=[object()],
        job_uids=["uid://A"],
        stage_label="Slurm calibrate",
        log_paths=[log_path],
        heartbeat_interval=0.0,
    )

    assert any("applycal 2/5 running" in line for line in written)
    assert any("1 running" in line for line in written)


def test_run_uid_stage_sync_continues_after_failure(monkeypatch, tmp_path):
    """Sync stage runner should skip the failing UID and keep the others."""
    calls: list[str] = []

    def _task(*, asdm_uid: str) -> list[str]:
        calls.append(asdm_uid)
        if asdm_uid == "uid://B":
            raise RuntimeError("Calibration failed for uid://B.\nReturn code: 134")
        return [f"/out/{asdm_uid}.ms.split.cal"]

    failures: list[cli_products.TaskFailure] = []
    uids = ["uid://A", "uid://B", "uid://C"]
    results = cli_products._run_uid_stage(
        backend=None,
        postprocess_backend="sync",
        task_fn=_task,
        job_kwargs=[{"asdm_uid": uid} for uid in uids],
        job_uids=uids,
        stage_label="Calibrate",
        log_paths=[tmp_path / f"{i}.log" for i in range(3)],
        continue_on_error=True,
        failures=failures,
    )

    assert calls == uids
    assert results == [["/out/uid://A.ms.split.cal"], [], ["/out/uid://C.ms.split.cal"]]
    assert [failure.label for failure in failures] == ["uid://B"]
    assert failures[0].log_path == str(tmp_path / "1.log")


def test_run_uid_stage_sync_fail_fast_raises():
    """With continue_on_error disabled the first failure propagates."""

    def _task(*, asdm_uid: str) -> list[str]:
        raise RuntimeError(f"failed {asdm_uid}")

    with pytest.raises(RuntimeError, match="failed uid://A"):
        cli_products._run_uid_stage(
            backend=None,
            postprocess_backend="sync",
            task_fn=_task,
            job_kwargs=[{"asdm_uid": "uid://A"}],
            job_uids=["uid://A"],
            stage_label="Calibrate",
            log_paths=[],
            continue_on_error=False,
            failures=[],
        )


def test_run_calibrate_jobs_sync_multi_uid_isolates_each_uid(monkeypatch, tmp_path):
    """Sync calibrate over several UIDs should run one subprocess wrapper per UID."""
    seen: list[dict[str, object]] = []

    def _fake_single(**kwargs):
        seen.append(kwargs)
        if kwargs["asdm_uid"] == "uid___B":
            raise RuntimeError("casacore::ArrayError")
        return [f"{kwargs['calibrated_output_root']}/{kwargs['asdm_uid']}.ms.split.cal"]

    patch_everywhere(monkeypatch, cli_products, "_calibrate_single_uid", _fake_single)
    patch_everywhere(
        monkeypatch,
        cli_products,
        "_preflight_casa_data",
        lambda *a, **k: tmp_path / "out" / ".casa-data",
    )
    patch_everywhere(
        monkeypatch,
        cli_products,
        "_extract_uids_from_raw_ms_root",
        lambda *_: ["uid___A", "uid___B", "uid___C"],
    )

    failures: list[cli_products.TaskFailure] = []
    outputs = cli_products._run_calibrate_jobs(
        input_root=tmp_path / "input",
        raw_ms_root=tmp_path / "raw",
        output_root=tmp_path / "out",
        asdm_uids=[],
        postprocess_backend="sync",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=False,
        overwrite_outputs=False,
        clean_intermediate=False,
        failures=failures,
    )

    assert [entry["asdm_uid"] for entry in seen] == ["uid___A", "uid___B", "uid___C"]
    assert all(entry["casa_data_root"] == str(tmp_path / "out" / ".casa-data") for entry in seen)
    assert all(entry["skip_casa_data_update"] is True for entry in seen)
    assert outputs == [
        str(tmp_path / "out" / "uid___A.ms.split.cal"),
        str(tmp_path / "out" / "uid___C.ms.split.cal"),
    ]
    assert [failure.label for failure in failures] == ["uid___B"]


def test_run_calibrate_jobs_sync_single_uid_runs_in_process(monkeypatch, tmp_path):
    """A single UID (the per-UID worker entry point) must not spawn another subprocess."""
    import almasim.services.archive as archive_mod

    captured: dict[str, object] = {}

    def _fake_create(**kwargs):
        captured.update(kwargs)
        return [tmp_path / "out" / "uid___A.ms.split.cal"]

    monkeypatch.setattr(archive_mod, "create_calibrated_measurement_sets", _fake_create)
    patch_everywhere(
        monkeypatch,
        cli_products,
        "_calibrate_single_uid",
        lambda **kwargs: pytest.fail("subprocess wrapper must not be used for one UID"),
    )

    outputs = cli_products._run_calibrate_jobs(
        input_root=tmp_path / "input",
        raw_ms_root=tmp_path / "raw",
        output_root=tmp_path / "out",
        asdm_uids=["uid___A"],
        postprocess_backend="sync",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite_outputs=False,
        clean_intermediate=False,
    )

    assert outputs == [str(tmp_path / "out" / "uid___A.ms.split.cal")]
    assert captured["asdm_uid"] == "uid___A"
    assert callable(captured["logger_fn"])


def test_run_calibrate_jobs_sync_skips_cleanup_when_failures(monkeypatch, tmp_path):
    """Intermediate cleanup must not run when any UID failed."""
    import almasim.services.archive as archive_mod

    patch_everywhere(
        monkeypatch,
        cli_products,
        "_calibrate_single_uid",
        lambda **kwargs: (
            (_ for _ in ()).throw(RuntimeError("boom"))
            if kwargs["asdm_uid"] == "uid___B"
            else ["/out/ok.ms.split.cal"]
        ),
    )
    patch_everywhere(
        monkeypatch, cli_products, "_preflight_casa_data", lambda *a, **k: tmp_path / "casa"
    )
    monkeypatch.setattr(
        archive_mod,
        "cleanup_intermediate_calibration_data",
        lambda *a, **k: pytest.fail("cleanup must be skipped after failures"),
    )

    failures: list[cli_products.TaskFailure] = []
    cli_products._run_calibrate_jobs(
        input_root=tmp_path / "input",
        raw_ms_root=tmp_path / "raw",
        output_root=tmp_path / "out",
        asdm_uids=["uid___A", "uid___B"],
        postprocess_backend="sync",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite_outputs=False,
        clean_intermediate=True,
        failures=failures,
    )
    assert [failure.label for failure in failures] == ["uid___B"]


def test_calibrate_single_uid_writes_per_uid_log_file(tmp_path, monkeypatch):
    """The subprocess wrapper should mirror child output into <output>/logs/<uid>.calibrate.log."""

    calls = fake_popen(monkeypatch, ["applying\n", "splitting\n"])

    uid = "uid___A001_X1_X9"
    output_root = tmp_path / "calibrated"
    (output_root / f"{uid}.ms.split.cal").mkdir(parents=True)

    cli_products._calibrate_single_uid(
        input_root="/input",
        raw_ms_root="/raw",
        calibrated_output_root=str(output_root),
        asdm_uid=uid,
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite=False,
        clean_intermediate=False,
    )

    log_path = output_root / "logs" / f"{uid}.calibrate.log"
    content = log_path.read_text(encoding="utf-8")
    assert content.startswith("# ")
    assert "applying\nsplitting\n" in content
    assert calls[0][1]["env"]["PYTHONUNBUFFERED"] == "1"


def test_run_parallel_archive_jobs_skips_calibration_for_failed_unpack(monkeypatch, tmp_path):
    """UIDs whose unpack failed must not be submitted to the calibrate stage."""

    class _Backend:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    patch_everywhere(monkeypatch, cli_products, "create_backend", lambda *a, **k: _Backend())
    patch_everywhere(
        monkeypatch, cli_products, "_preflight_casa_data", lambda *a, **k: tmp_path / "casa"
    )
    patch_everywhere(
        monkeypatch,
        cli_products,
        "_extract_asdm_uids_from_download_root",
        lambda *_: ["uid___A", "uid___B"],
    )

    stages: list[tuple[str, list[str]]] = []

    def _fake_stage(**kwargs):
        stages.append((kwargs["stage_label"], list(kwargs["job_uids"])))
        if kwargs["stage_label"].endswith("unpack"):
            kwargs["failures"].append(cli_products.TaskFailure("uid___B", "importasdm crashed"))
            return [["/raw/uid___A.ms"], []]
        return [["/cal/uid___A.ms.split.cal"]]

    patch_everywhere(monkeypatch, cli_products, "_run_uid_stage", _fake_stage)

    failures: list[cli_products.TaskFailure] = []
    raw, cal = cli_products._run_parallel_archive_jobs(
        download_root=tmp_path / "dl",
        archive_output_root=tmp_path / "archive",
        unpack_ms=True,
        generate_calibrated_visibilities=True,
        postprocess_backend="slurm",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=False,
        overwrite_archive_outputs=False,
        failures=failures,
    )

    assert raw == ["/raw/uid___A.ms"]
    assert cal == ["/cal/uid___A.ms.split.cal"]
    assert stages == [("Slurm unpack", ["uid___A", "uid___B"]), ("Slurm calibrate", ["uid___A"])]
    assert [failure.label for failure in failures] == ["uid___B"]


def _member_with_left_out_block(download_root: Path) -> None:
    """A member OUS whose pipeline run calibrated uid___A and left uid___B out."""
    member = download_root / "P" / "science_goal.uid___S" / "group.uid___G" / "member.uid___M"
    for uid in ("uid___A", "uid___B"):
        (member / "raw" / f"{uid}.asdm.sdm").mkdir(parents=True)
    (member / "calibration").mkdir()
    (member / "calibration" / "uid___A.ms.calapply.txt").write_text("")


def test_run_parallel_archive_jobs_skips_blocks_delivered_without_calibration(
    monkeypatch, tmp_path
):
    """A block ALMA never calibrated is unpacked but neither calibrated nor counted as failed."""

    class _Backend:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    download_root = tmp_path / "dl"
    _member_with_left_out_block(download_root)
    stale_marker = tmp_path / "archive" / "calibrated_ms" / "uid___B.ms.split.cal.failed"
    stale_marker.parent.mkdir(parents=True)
    stale_marker.write_text("{}")
    patch_everywhere(monkeypatch, cli_products, "create_backend", lambda *a, **k: _Backend())
    patch_everywhere(
        monkeypatch, cli_products, "_preflight_casa_data", lambda *a, **k: tmp_path / "casa"
    )
    patch_everywhere(
        monkeypatch,
        cli_products,
        "_extract_asdm_uids_from_download_root",
        lambda *_: ["uid___A", "uid___B"],
    )

    stages: list[tuple[str, list[str]]] = []

    def _fake_stage(**kwargs):
        stages.append((kwargs["stage_label"], list(kwargs["job_uids"])))
        if kwargs["stage_label"].endswith("unpack"):
            return [["/raw/uid___A.ms"], ["/raw/uid___B.ms"]]
        return [["/cal/uid___A.ms.split.cal"]]

    patch_everywhere(monkeypatch, cli_products, "_run_uid_stage", _fake_stage)

    messages: list[str] = []
    failures: list[cli_products.TaskFailure] = []
    raw, cal = cli_products._run_parallel_archive_jobs(
        download_root=download_root,
        archive_output_root=tmp_path / "archive",
        unpack_ms=True,
        generate_calibrated_visibilities=True,
        postprocess_backend="slurm",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=False,
        overwrite_archive_outputs=False,
        failures=failures,
        echo=lambda message, err=False: messages.append(message),
    )

    assert raw == ["/raw/uid___A.ms", "/raw/uid___B.ms"]
    assert cal == ["/cal/uid___A.ms.split.cal"]
    assert stages == [("Slurm unpack", ["uid___A", "uid___B"]), ("Slurm calibrate", ["uid___A"])]
    assert failures == []
    assert any("delivered without calibration" in m and "uid___B" in m for m in messages)
    assert not stale_marker.exists()


def test_run_calibrate_jobs_with_only_uncalibrated_blocks_does_nothing(monkeypatch, tmp_path):
    """When ALMA calibrated none of the requested blocks nothing is preflighted or submitted."""
    download_root = tmp_path / "dl"
    _member_with_left_out_block(download_root)
    patch_everywhere(
        monkeypatch,
        cli_products,
        "_preflight_casa_data",
        lambda *a, **k: pytest.fail("no preflight"),
    )
    patch_everywhere(
        monkeypatch,
        cli_products,
        "create_backend",
        lambda *a, **k: pytest.fail("no backend expected"),
    )

    outputs = cli_products._run_calibrate_jobs(
        input_root=download_root,
        raw_ms_root=tmp_path / "raw",
        output_root=tmp_path / "cal",
        asdm_uids=["uid___B"],
        postprocess_backend="slurm",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite_outputs=False,
        clean_intermediate=False,
    )
    assert outputs == []


def test_tail_log_line_ignores_stale_logs(tmp_path):
    """Logs older than the stage start are not reported as running jobs."""
    from time import time

    log_path = tmp_path / "old.log"
    log_path.write_text("first\nlast line\n", encoding="utf-8")
    os.utime(log_path, (time() - 3600, time() - 3600))

    from almasim.services.compute.progress import tail_line

    assert tail_line(log_path) == "last line"
    assert tail_line(log_path, modified_since=time() - 60) is None
    assert tail_line(tmp_path / "missing.log") is None


def test_partition_completed_calibrations_skips_marked_outputs(tmp_path):
    """Only outputs with a completion marker are treated as done."""
    from almasim.services.archive import write_calibration_marker

    output_root = tmp_path / "cal"
    (output_root / "uid___A.ms.split.cal").mkdir(parents=True)
    write_calibration_marker(output_root, "uid___A")
    (output_root / "uid___B.ms.split.cal").mkdir()  # partial: no marker
    (output_root / "uid___C.ms.split.cal.done").touch()  # marker without output

    todo, done = cli_products._partition_completed_calibrations(
        output_root, ["uid___A", "uid___B", "uid___C", "uid___D"], overwrite=False
    )

    assert todo == ["uid___B", "uid___C", "uid___D"]
    assert done == [str(output_root / "uid___A.ms.split.cal")]

    todo, done = cli_products._partition_completed_calibrations(
        output_root, ["uid___A", "uid___B"], overwrite=True
    )
    assert todo == ["uid___A", "uid___B"]
    assert done == []


def test_run_calibrate_jobs_skips_completed_uids_before_submission(monkeypatch, tmp_path):
    """Completed UIDs are neither preflighted nor submitted, but appear in the outputs."""
    from almasim.services.archive import write_calibration_marker

    output_root = tmp_path / "cal"
    (output_root / "uid___A.ms.split.cal").mkdir(parents=True)
    write_calibration_marker(output_root, "uid___A")

    submitted: list[str] = []

    def _fake_single(**kwargs):
        submitted.append(kwargs["asdm_uid"])
        assert kwargs["keep_working_copies"] is False
        return [f"{kwargs['calibrated_output_root']}/{kwargs['asdm_uid']}.ms.split.cal"]

    patch_everywhere(monkeypatch, cli_products, "_calibrate_single_uid", _fake_single)
    patch_everywhere(
        monkeypatch, cli_products, "_preflight_casa_data", lambda *a, **k: tmp_path / "casa"
    )

    outputs = cli_products._run_calibrate_jobs(
        input_root=tmp_path / "input",
        raw_ms_root=tmp_path / "raw",
        output_root=output_root,
        asdm_uids=["uid___A", "uid___B", "uid___C"],
        postprocess_backend="sync",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite_outputs=False,
        clean_intermediate=False,
    )

    assert submitted == ["uid___B", "uid___C"]
    assert outputs[0] == str(output_root / "uid___A.ms.split.cal")
    assert len(outputs) == 3


def test_run_calibrate_jobs_all_completed_does_nothing(monkeypatch, tmp_path):
    """When every UID is done the runner returns the existing outputs without a backend."""
    from almasim.services.archive import write_calibration_marker

    output_root = tmp_path / "cal"
    for uid in ("uid___A", "uid___B"):
        (output_root / f"{uid}.ms.split.cal").mkdir(parents=True)
        write_calibration_marker(output_root, uid)
    patch_everywhere(
        monkeypatch,
        cli_products,
        "_preflight_casa_data",
        lambda *a, **k: pytest.fail("no preflight"),
    )
    patch_everywhere(
        monkeypatch,
        cli_products,
        "create_backend",
        lambda *a, **k: pytest.fail("no backend expected"),
    )

    outputs = cli_products._run_calibrate_jobs(
        input_root=tmp_path / "input",
        raw_ms_root=tmp_path / "raw",
        output_root=output_root,
        asdm_uids=["uid___A", "uid___B"],
        postprocess_backend="slurm",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite_outputs=False,
        clean_intermediate=False,
    )
    assert len(outputs) == 2


def test_calibrate_single_uid_forwards_keep_working_copies(tmp_path, monkeypatch):
    """The worker wrapper should pass --keep-working-copies to the child only when asked."""
    calls = fake_popen(monkeypatch)
    output_root = tmp_path / "cal"
    (output_root / "uid___A.ms.split.cal").mkdir(parents=True)
    common = dict(
        input_root="/input",
        raw_ms_root="/raw",
        calibrated_output_root=str(output_root),
        asdm_uid="uid___A",
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite=False,
        clean_intermediate=False,
    )
    cli_products._calibrate_single_uid(**common)
    cli_products._calibrate_single_uid(**common, keep_working_copies=True)

    assert "--keep-working-copies" not in calls[0][0]
    assert "--keep-working-copies" in calls[1][0]


def test_casa_bundled_lib_dir_precedes_system_paths(tmp_path, monkeypatch):
    """Worker env must search casatools' bundled libs before /usr/lib64.

    The system libpmix otherwise shadows the copy bundled with casatools and the
    CASA extension modules fail with "undefined symbol: pmix_framework_names".
    """

    calls = fake_popen(monkeypatch)
    bundled = tmp_path / "site-packages" / "casatools" / "__casac__" / "lib"
    bundled.mkdir(parents=True)
    patch_everywhere(monkeypatch, cli_products, "_casa_bundled_lib_dir", lambda: str(bundled))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/spack/lib")

    output_root = tmp_path / "cal"
    (output_root / "uid___A.ms.split.cal").mkdir(parents=True)
    cli_products._calibrate_single_uid(
        input_root="/input",
        raw_ms_root="/raw",
        calibrated_output_root=str(output_root),
        asdm_uid="uid___A",
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite=False,
        clean_intermediate=False,
    )

    entries = calls[-1][1]["env"]["LD_LIBRARY_PATH"].split(":")
    assert entries[0] == str(bundled)
    assert entries.index(str(bundled)) < entries.index("/usr/lib64")
    # Inherited (spack) paths still come last.
    assert entries[-1] == "/opt/spack/lib"


def test_casa_bundled_lib_dir_absent_keeps_system_paths(tmp_path, monkeypatch):
    """Without casatools installed the previous system-first ordering is unchanged."""

    calls = fake_popen(monkeypatch)
    patch_everywhere(monkeypatch, cli_products, "_casa_bundled_lib_dir", lambda: None)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    output_root = tmp_path / "raw"
    (output_root / "working").mkdir(parents=True)
    cli_products._unpack_single_uid(
        input_root="/input",
        raw_output_root=str(output_root),
        asdm_uid="uid___A",
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite=False,
    )

    assert calls[-1][1]["env"]["LD_LIBRARY_PATH"].startswith("/lib64:/usr/lib64:")


def test_casa_bundled_lib_dir_returns_none_when_not_installed(monkeypatch):
    """A missing casatools package resolves to None rather than raising."""
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert cli_products._casa_bundled_lib_dir() is None


def test_one_uid_per_worker_limits_task_slots():
    """Slurm workers get a single task slot so --slurm-workers == UIDs in flight."""
    assert cli_products._one_uid_per_worker({"n_workers": 10}) == {
        "n_workers": 10,
        "worker_extra_args": ["--nthreads", "1"],
    }
    # An explicit caller setting wins.
    explicit = {"worker_extra_args": ["--nthreads", "4"]}
    assert cli_products._one_uid_per_worker(explicit) == explicit


def test_run_calibrate_jobs_slurm_pins_one_task_per_worker(monkeypatch, tmp_path):
    """The slurm calibrate path must not let a worker run many UIDs at once."""
    captured: dict[str, object] = {}

    class _Backend:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    patch_everywhere(
        monkeypatch,
        cli_products,
        "create_backend",
        lambda backend, **kwargs: captured.update(kwargs) or _Backend(),
    )
    patch_everywhere(
        monkeypatch, cli_products, "_preflight_casa_data", lambda *a, **k: tmp_path / "casa"
    )
    patch_everywhere(
        monkeypatch, cli_products, "_run_uid_stage", lambda **kwargs: [["/cal/a.ms.split.cal"]]
    )

    cli_products._run_calibrate_jobs(
        input_root=tmp_path / "input",
        raw_ms_root=tmp_path / "raw",
        output_root=tmp_path / "out",
        asdm_uids=["uid___A"],
        postprocess_backend="slurm",
        postprocess_backend_kwargs={"n_workers": 10},
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite_outputs=False,
        clean_intermediate=False,
    )

    assert captured["worker_extra_args"] == ["--nthreads", "1"]


def test_run_calibrate_jobs_sync_does_not_add_worker_args(monkeypatch, tmp_path):
    """The sync path has no Dask workers, so no worker arguments are injected."""
    captured: dict[str, object] = {}
    patch_everywhere(
        monkeypatch, cli_products, "_preflight_casa_data", lambda *a, **k: tmp_path / "casa"
    )
    patch_everywhere(
        monkeypatch,
        cli_products,
        "_run_uid_stage",
        lambda **kwargs: captured.update(kwargs) or [[], []],
    )

    cli_products._run_calibrate_jobs(
        input_root=tmp_path / "input",
        raw_ms_root=tmp_path / "raw",
        output_root=tmp_path / "out",
        asdm_uids=["uid___A", "uid___B"],
        postprocess_backend="sync",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite_outputs=False,
        clean_intermediate=False,
    )
    assert captured["postprocess_backend"] == "sync"


def test_calibrate_single_uid_hard_abort_writes_failure_marker(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: _CrashingProcess())
    uid = "uid___A001_X1_X9"
    output_root = tmp_path / "calibrated"
    working_dir = output_root / "working" / f"{uid}.calibration"
    working_dir.mkdir(parents=True)
    (working_dir / f"{uid}.ms").mkdir()

    with pytest.raises(RuntimeError, match="Return code: -6"):
        cli_products._calibrate_single_uid(
            input_root="/input",
            raw_ms_root="/raw",
            calibrated_output_root=str(output_root),
            asdm_uid=uid,
            casa_data_root=None,
            skip_casa_data_update=True,
            overwrite=False,
            clean_intermediate=False,
        )

    marker = output_root / f"{uid}.ms.split.cal.failed"
    assert marker.is_file()
    payload = json.loads(marker.read_text())
    assert payload["uid"] == uid
    assert "return code -6" in payload["error"]
    assert "minMax - Array has no elements" in payload["error"]
    assert Path(payload["log"]).is_file()
    assert not (output_root / f"{uid}.ms.split.cal.done").exists()
    assert not working_dir.exists(), "the multi-GB working copy is reclaimed"


def test_calibrate_single_uid_hard_abort_keeps_working_copy_when_asked(tmp_path, monkeypatch):
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: _CrashingProcess())
    uid = "uid___A001_X1_Xa"
    output_root = tmp_path / "calibrated"
    working_dir = output_root / "working" / f"{uid}.calibration"
    working_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError):
        cli_products._calibrate_single_uid(
            input_root="/input",
            raw_ms_root="/raw",
            calibrated_output_root=str(output_root),
            asdm_uid=uid,
            casa_data_root=None,
            skip_casa_data_update=True,
            overwrite=False,
            clean_intermediate=False,
            keep_working_copies=True,
        )

    assert (output_root / f"{uid}.ms.split.cal.failed").is_file()
    assert working_dir.is_dir()


def test_unpack_single_uid_hard_abort_writes_failure_marker(tmp_path, monkeypatch):
    import json

    lines = ["importasdm::::casa\tstarting\n", "Killed\n"]
    monkeypatch.setattr(
        "subprocess.Popen", lambda *args, **kwargs: _CrashingProcess(lines, return_code=-9)
    )
    uid = "uid___A001_X1_Xb"
    output_root = tmp_path / "raw"
    (output_root / "working").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Return code: -9"):
        cli_products._unpack_single_uid(
            input_root="/input",
            raw_output_root=str(output_root),
            asdm_uid=uid,
            casa_data_root=None,
            skip_casa_data_update=True,
            overwrite=False,
        )

    marker = output_root / "working" / f"{uid}.ms.failed"
    assert marker.is_file()
    payload = json.loads(marker.read_text())
    assert "return code -9" in payload["error"]
    assert not (output_root / "working" / f"{uid}.ms.done").exists()


def test_run_archive_stages_collects_failures(monkeypatch, tmp_path):
    """run_archive_stages returns the outputs and the failed UIDs instead of raising."""

    def _fake_jobs(**kwargs):
        kwargs["failures"].append(archive_pipeline.TaskFailure("uid___B", "boom", None))
        assert kwargs["continue_on_error"] is True
        return ["/raw/uid___A.ms"], ["/cal/uid___A.ms.split.cal"]

    monkeypatch.setattr(archive_pipeline, "_run_parallel_archive_jobs", _fake_jobs)
    result = archive_pipeline.run_archive_stages(tmp_path / "dl", tmp_path / "out")
    assert result.raw_ms == ["/raw/uid___A.ms"]
    assert result.calibrated_ms == ["/cal/uid___A.ms.split.cal"]
    assert not result.ok and result.failures[0].label == "uid___B"
