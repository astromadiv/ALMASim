"""The console relays a running job's Dask dashboard, websockets included."""

from __future__ import annotations

import socket
import sys
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

import uvicorn
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from almasim.jobs import Job, JobState, create_run_dir, new_job_id, write_manifest
from almasim.jobs import rundir as rundir_mod
from almasim.jobs.profile import Profile
from almasim.jobs.rundir import (
    DASHBOARD_NAME,
    RUN_DIR_ENV,
    DashboardAddress,
    RunStatus,
    clear_dashboard,
    read_dashboard,
    write_dashboard,
    write_status,
)
from almasim.scheduling import cluster as cluster_mod
from almasim.web.app import COOKIE, create_app

pytestmark = pytest.mark.unit

TOKEN = "dashboard-test-token"


# A stand-in for a Dask scheduler's dashboard: the page, a redirect, a websocket.
async def _status(request):
    return HTMLResponse(f"<script src='static/js/bokeh.min.js'></script>{request.url.query}")


async def _headers(request):
    return JSONResponse({k: v for k, v in request.headers.items()})


async def _root(request):
    return RedirectResponse("/status", status_code=301)


async def _ws(websocket: WebSocket):
    await websocket.accept(subprotocol="bokeh")
    try:
        while True:
            await websocket.send_text("echo " + await websocket.receive_text())
    except WebSocketDisconnect:
        pass


@pytest.fixture(scope="module")
def upstream():
    """A dashboard-like server on a free local port."""
    app = Starlette(
        routes=[
            Route("/", _root),
            Route("/status", _status),
            Route("/headers", _headers),
            WebSocketRoute("/status/ws", _ws),
        ]
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    yield DashboardAddress(host="127.0.0.1", port=port)
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def profile(tmp_path) -> Profile:
    return Profile(runs_root=tmp_path / "runs", python=sys.executable, partition="zwickyA-CPU")


def _running_run(profile: Profile):
    job = Job(
        id=new_job_id(), owner="astro", argv=["image", "clean-quality", "/o"], executor="slurm"
    )
    run_dir = create_run_dir(profile.runs_root, job)
    write_manifest(run_dir, job)
    write_status(run_dir, RunStatus(started_at=time.time(), host="compute-node47"))
    return job.id, run_dir


def _client(profile, tmp_path, base_path=""):
    app = create_app(TOKEN, static_dir=tmp_path / "none", profile=profile, base_path=base_path)
    return TestClient(app)


def test_the_note_is_read_back_and_only_from_this_user(tmp_path, monkeypatch):
    assert read_dashboard(tmp_path) is None
    write_dashboard(tmp_path, DashboardAddress(host="10.20.25.47", port=40123))
    assert read_dashboard(tmp_path) == DashboardAddress(host="10.20.25.47", port=40123)
    # Someone else's note in a group-writable runs root is not followed.
    monkeypatch.setattr(rundir_mod.os, "getuid", lambda: 4242)
    assert read_dashboard(tmp_path) is None
    monkeypatch.undo()
    (tmp_path / DASHBOARD_NAME).write_text("{not json")
    assert read_dashboard(tmp_path) is None
    clear_dashboard(tmp_path)
    clear_dashboard(tmp_path)  # already gone: fine
    assert not (tmp_path / DASHBOARD_NAME).exists()


def test_the_job_page_learns_when_there_is_a_dashboard(profile, tmp_path, upstream):
    run_id, run_dir = _running_run(profile)
    with _client(profile, tmp_path) as client:
        client.cookies.set(COOKIE, TOKEN)
        assert client.get(f"/api/v2/jobs/{run_id}").json()["dashboard"] is False
        write_dashboard(run_dir, upstream)
        assert client.get(f"/api/v2/jobs/{run_id}").json()["dashboard"] is True
        write_status(run_dir, RunStatus(started_at=1.0, finished_at=2.0, exit_code=0))
        assert client.get(f"/api/v2/jobs/{run_id}").json()["dashboard"] is False


def test_pages_are_relayed_without_the_consoles_token(profile, tmp_path, upstream):
    run_id, run_dir = _running_run(profile)
    write_dashboard(run_dir, upstream)
    here = f"/api/v2/jobs/{run_id}/dashboard"
    with _client(profile, tmp_path) as client:
        assert client.get(f"{here}/status").status_code == 401  # no token, no relay
        client.cookies.set(COOKIE, TOKEN)
        page = client.get(f"{here}/status", params={"a": "1"})
        assert page.status_code == 200 and "static/js/bokeh.min.js" in page.text
        assert page.text.endswith("a=1")
        seen = client.get(f"{here}/headers").json()
        assert "cookie" not in seen and "authorization" not in seen
        # The bare route and the dashboard's own redirect both stay under the route.
        bare = client.get(here, follow_redirects=False)
        assert bare.status_code == 307 and bare.headers["location"] == f"{here}/status"
        root = client.get(f"{here}/", follow_redirects=False)
        assert root.status_code == 301 and root.headers["location"] == f"{here}/status"


def test_the_websocket_is_relayed_with_its_subprotocol(profile, tmp_path, upstream):
    run_id, run_dir = _running_run(profile)
    write_dashboard(run_dir, upstream)
    path = f"/console/api/v2/jobs/{run_id}/dashboard/status/ws"
    with _client(profile, tmp_path, base_path="/console") as client:
        with client.websocket_connect(
            path, subprotocols=["bokeh", "session-token"], headers={"cookie": f"{COOKIE}={TOKEN}"}
        ) as ws:
            assert ws.accepted_subprotocol == "bokeh"
            ws.send_text("hello")
            assert ws.receive_text() == "echo hello"
        with pytest.raises(WebSocketDisconnect):  # without the token
            with client.websocket_connect(path) as ws:
                ws.receive_text()
        with pytest.raises(WebSocketDisconnect):  # outside the console's prefix
            with client.websocket_connect(path.removeprefix("/console")) as ws:
                ws.receive_text()


def test_a_run_without_a_live_dashboard_says_why(profile, tmp_path):
    run_id, run_dir = _running_run(profile)
    here = f"/api/v2/jobs/{run_id}/dashboard/status"
    with _client(profile, tmp_path) as client:
        client.cookies.set(COOKIE, TOKEN)
        missing = client.get(here)
        assert missing.status_code == 404 and "no Dask dashboard" in missing.json()["detail"]
        assert client.get("/api/v2/jobs/ZZZZZZZZ/dashboard/status").status_code == 404
        with socket.socket() as s:  # a port nothing listens on
            s.bind(("127.0.0.1", 0))
            closed = s.getsockname()[1]
        write_dashboard(run_dir, DashboardAddress(host="127.0.0.1", port=closed))
        gone = client.get(here)
        assert gone.status_code == 502 and "does not answer" in gone.json()["detail"]
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                f"/api/v2/jobs/{run_id}/dashboard/status/ws",
                headers={"cookie": f"{COOKIE}={TOKEN}"},
            ) as ws:
                ws.receive_text()
        write_status(run_dir, RunStatus(started_at=1.0, finished_at=2.0, exit_code=1))
        ended = client.get(here)
        assert ended.status_code == 409 and "finished" in ended.json()["detail"]
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                f"/api/v2/jobs/{run_id}/dashboard/status/ws",
                headers={"cookie": f"{COOKIE}={TOKEN}"},
            ) as ws:
                ws.receive_text()


def test_a_console_job_notes_its_dashboard_and_removes_it(tmp_path, monkeypatch):
    singleton = cluster_mod.SlurmDaskClusterSingleton
    manager = singleton.__new__(singleton)
    manager.cluster = SimpleNamespace(
        scheduler_info={"services": {"dashboard": 40123}},
        scheduler_address="tcp://10.20.25.47:8786",
        close=lambda: None,
    )
    manager.client = SimpleNamespace(close=lambda: None)

    manager._record_dashboard()  # not in a console job: nothing to note
    assert not (tmp_path / DASHBOARD_NAME).exists()

    monkeypatch.setenv(RUN_DIR_ENV, str(tmp_path))
    manager._record_dashboard()
    assert read_dashboard(tmp_path) == DashboardAddress(host="10.20.25.47", port=40123)

    monkeypatch.setattr(singleton, "_instance", manager)
    singleton.close_instance()
    assert not (tmp_path / DASHBOARD_NAME).exists()

    manager.cluster.scheduler_info = {"services": {}}  # no bokeh, no dashboard
    manager._record_dashboard()
    assert not (tmp_path / DASHBOARD_NAME).exists()


def test_a_note_that_cannot_be_written_is_only_logged(tmp_path, monkeypatch, caplog):
    singleton = cluster_mod.SlurmDaskClusterSingleton
    manager = singleton.__new__(singleton)
    manager.cluster = SimpleNamespace(
        scheduler_info={"services": {"dashboard": 40123}},
        scheduler_address="tcp://10.20.25.47:8786",
    )
    monkeypatch.setenv(RUN_DIR_ENV, str(tmp_path / "gone"))
    manager._record_dashboard()
    assert "Could not record the Dask dashboard" in caplog.text


def test_the_scheduler_asks_for_a_free_dashboard_port(monkeypatch):
    seen = {}

    class Cluster:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def scale(self, n):
            pass

        def close(self):
            pass

    monkeypatch.setattr(cluster_mod, "SLURM_DASK_AVAILABLE", True)
    monkeypatch.setattr(cluster_mod, "SLURMCluster", Cluster)
    monkeypatch.setattr(cluster_mod, "Client", lambda cluster: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cluster_mod, "_partition_node_address", lambda queue: None)
    monkeypatch.delenv(RUN_DIR_ENV, raising=False)
    try:
        cluster_mod.SlurmDaskClusterSingleton.get_instance(node_cores=8, n_jobs=1)
    finally:
        cluster_mod.SlurmDaskClusterSingleton.close_instance()
    assert seen["scheduler_options"]["dashboard_address"] == ":0"


def test_jobs_are_told_their_run_folder(tmp_path):
    from almasim.jobs.wrapper import STDOUT_NAME, run_job

    job = Job(id=new_job_id(), owner="astro", argv=["x"], executor="slurm", state=JobState.QUEUED)
    run_dir = create_run_dir(tmp_path, job)
    write_manifest(run_dir, job)
    code = run_job(
        run_dir, [sys.executable, "-c", f"import os; print(os.environ[{RUN_DIR_ENV!r}])"]
    )
    assert code == 0
    assert (run_dir / STDOUT_NAME).read_text().strip() == str(run_dir.absolute())
