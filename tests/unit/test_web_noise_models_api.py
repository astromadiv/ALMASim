"""API v2: the noise models the Simulate page offers."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from almasim.jobs.profile import load_profile
from almasim.services.noise_model import COMPONENT_NAMES, Provenance, released_model
from almasim.web.app import COOKIE, create_app

pytestmark = pytest.mark.unit

TOKEN = "noise-models-test-token"


def _client(tmp_path, monkeypatch, folder):
    monkeypatch.delenv("ALMASIM_NOISE_MODELS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    site = tmp_path / "site.toml"
    site.write_text(f'runs_root = "{tmp_path}/runs"\n[tools]\nnoise_models = "{folder}"\n')
    profile = load_profile(site=site, environ={})
    app = create_app(TOKEN, static_dir=tmp_path / "none", profile=profile, events_interval=60)
    client = TestClient(app)
    client.cookies.set(COOKIE, TOKEN)
    return client


def test_package_and_site_models_with_their_components(tmp_path, monkeypatch):
    folder = tmp_path / "noise-models"
    folder.mkdir()
    alma = released_model("alma", 1)
    for version in (1, 2):
        notes = "Validated on 4 windows: x (sha256 0123456789ab) passed every check."
        team = alma.model_copy(
            update={
                "name": "team",
                "version": version,
                "provenance": Provenance(method="reduce", windows=4, notes=notes),
            }
        )
        (folder / f"team-{version}.json").write_text(team.model_dump_json())
    got = _client(tmp_path, monkeypatch, folder).get("/api/v2/noise-models").json()
    assert got["site_folder"] == str(folder)
    specs = [(m["spec"], m["source"], m["newest"]) for m in got["models"]]
    assert specs[0] == ("alma@1", "package", True)
    assert specs[-2:] == [("team@2", "site", True), ("team@1", "site", False)]
    first = got["models"][0]
    assert first["sha256"] == alma.sha256()
    assert [c["name"] for c in first["components"]] == list(COMPONENT_NAMES)
    assert all(c["enabled"] and c["description"] for c in first["components"])
    assert first["provenance"]["windows"] == 100 and first["provenance"]["validated"] is None
    assert got["models"][-1]["provenance"]["validated"] == "0123456789ab"


def test_without_a_site_folder_only_the_package_models(tmp_path, monkeypatch):
    got = _client(tmp_path, monkeypatch, tmp_path / "missing").get("/api/v2/noise-models").json()
    assert {m["source"] for m in got["models"]} == {"package"}


def test_the_environment_names_the_folder_too(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, tmp_path / "missing")
    monkeypatch.setenv("ALMASIM_NOISE_MODELS", str(tmp_path))
    assert client.get("/api/v2/noise-models").json()["site_folder"] == str(tmp_path)
