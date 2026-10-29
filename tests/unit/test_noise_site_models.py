"""The team's own noise models, in the site folder beside the package's."""

from __future__ import annotations

import logging

import pytest
from typer.testing import CliRunner

from almasim import cli
from almasim.services.noise_model import (
    model_source,
    packaged,
    released,
    released_model,
    resolve_noise_model,
    site_models,
    site_models_dir,
)

pytestmark = pytest.mark.unit


def _release(folder, name, version, **changes):
    model = released_model("alma", 1).model_copy(
        update={"name": name, "version": version, **changes}
    )
    path = folder / f"{name}-{version}.json"
    path.write_text(model.model_dump_json(indent=2))
    return model


@pytest.fixture
def folder(tmp_path, monkeypatch):
    folder = tmp_path / "noise-models"
    folder.mkdir()
    monkeypatch.setenv("ALMASIM_NOISE_MODELS", str(folder))
    return folder


def test_site_models_resolve_by_name_and_version(folder):
    first = _release(folder, "zwicky-alma", 1)
    second = _release(folder, "zwicky-alma", 2, description="second")
    assert resolve_noise_model("zwicky-alma") == second
    assert resolve_noise_model("zwicky-alma@1") == first
    assert set(released()) == {"alma", "zwicky-alma"}
    assert model_source("zwicky-alma") == "site" and model_source("alma") == "package"
    assert "zwicky-alma" not in packaged()


def test_a_model_released_while_running_shows_at_once(folder):
    assert "team" not in released()
    _release(folder, "team", 1)
    assert resolve_noise_model("team@1").name == "team"


def test_files_that_do_not_belong_are_skipped_with_a_warning(folder, caplog):
    _release(folder, "alma", 9)  # a package name stays the package's
    (folder / "broken-1.json").write_text("{not json")
    _release(folder, "named", 1).model_dump_json()
    (folder / "named-1.json").rename(folder / "other-3.json")  # holds named@1
    (folder / "notes.txt").write_text("not a model")
    good = _release(folder, "good", 1)
    with caplog.at_level(logging.WARNING):
        found = site_models()
    assert found == {"good": {1: good}}
    assert 9 not in released()["alma"]
    warnings = caplog.text
    assert "'alma' is a package model" in warnings
    assert "broken-1.json ignored" in warnings and "holds named@1" in warnings


def test_the_folder_comes_from_the_profile_unless_the_environment_names_one(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # no user profile
    site = tmp_path / "site.toml"
    site.write_text(f'[tools]\nnoise_models = "{tmp_path}/team"\n')
    env = {"ALMASIM_SITE_PROFILE": str(site)}
    assert site_models_dir(env) == tmp_path / "team"
    assert site_models_dir({**env, "ALMASIM_NOISE_MODELS": "/elsewhere"}).as_posix() == "/elsewhere"
    assert site_models_dir({}) is None
    site.write_text("[tools]\nunknown = 1\n")
    assert site_models_dir(env) is None  # a broken profile hides only the site's models
    assert site_models(tmp_path / "missing") == {}


def test_noise_show_lists_where_each_model_comes_from(folder):
    _release(folder, "zwicky-alma", 1)
    result = CliRunner().invoke(cli.app, ["noise", "show"])
    assert result.exit_code == 0, result.output
    assert "alma (package): versions 1" in result.output
    assert "zwicky-alma (site): versions 1" in result.output
    assert f"site models: {folder}" in result.output
