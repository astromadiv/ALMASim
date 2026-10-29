"""Site and user profiles (web console plan, J1.3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from almasim.jobs import Resources
from almasim.jobs.profile import (
    DEFAULT_USER_PROFILE,
    SITE_ENV,
    USER_ENV,
    ProfileError,
    load_profile,
    profile_paths,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

SITE = """
runs_root = "/shared/runs"
roots = ["/shared", "~"]

[python]
executable = "/opt/py/bin/python"
setup = ["module load gcc"]

[tools]
wsclean = "/opt/wsclean"

[slurm]
partition = "cpu"
account = "ska"

[slurm.partitions.cpu]
cpus = 96
memory = "770G"

[commands."image image-from-ms"]
cpus = 4
memory = "16G"
time = "2-00:00:00"
args = ["--backend", "slurm"]
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_without_files_the_defaults_apply(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.config/almasim/almasim.toml
    profile = load_profile(environ={})
    assert profile.sources == ()
    assert profile.python == sys.executable
    assert profile.resources_for("metadata") == Resources()


def test_the_site_profile_sets_shared_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # no user profile
    site = _write(tmp_path / "site.toml", SITE)
    profile = load_profile(site=site, environ={})
    assert profile.sources == (site,)
    assert profile.runs_root == Path("/shared/runs")
    assert profile.roots == (Path("/shared"), Path("~").expanduser())
    assert profile.python == "/opt/py/bin/python"
    assert profile.setup == ("module load gcc",)
    assert profile.partitions["cpu"].cpus == 96
    assert profile.policy("image image-from-ms").args == ("--backend", "slurm")
    assert profile.resources_for("image image-from-ms") == Resources(
        partition="cpu", account="ska", cpus=4, memory="16G", time="2-00:00:00"
    )
    assert profile.resources_for("image image-from-ms", cpus=8, time=None).cpus == 8


def test_the_user_profile_overrides_key_by_key(tmp_path):
    site = _write(tmp_path / "site.toml", SITE)
    user = _write(
        tmp_path / "me.toml",
        '[slurm]\naccount = ""\n\n[commands."image image-from-ms"]\nmemory = "64G"\n',
    )
    profile = load_profile(site=site, user=user, environ={})
    assert profile.sources == (site, user)
    assert profile.partition == "cpu"  # kept from the site
    assert profile.account is None  # "" clears it
    policy = profile.policy("image image-from-ms")
    assert policy.memory == "64G" and policy.cpus == 4


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("runs_rot = '/x'", "unknown key"),
        ("[tools]\nwsclen = '/x'", r"\[tools\]: unknown key"),
        ("[commands.metadata]\nexecutor = 'kubernetes'", "executor must be one of"),
        ("executor = 'local'", "executor must be one of"),
        ("[commands.metadata]\nmemory = '16 GB'", "memory"),
        ("[slurm]\npartition = 'a b'", "partition"),
        ("[commands.metadata]\nargs = '--x'", "list of strings"),
        ("not = toml = at all", "toml|Expected|Invalid"),
    ],
)
def test_mistakes_are_errors_that_name_the_file(tmp_path, text, message):
    user = _write(tmp_path / "me.toml", text)
    with pytest.raises(ProfileError, match=message) as info:
        load_profile(user=user, environ={})
    assert str(user) in str(info.value)


def test_a_named_profile_must_exist(tmp_path):
    with pytest.raises(ProfileError, match="missing.toml"):
        load_profile(site=tmp_path / "missing.toml", environ={})
    with pytest.raises(ProfileError):
        load_profile(environ={USER_ENV: str(tmp_path / "missing.toml")})


def test_the_environment_names_the_files(tmp_path):
    assert profile_paths({}) == (None, DEFAULT_USER_PROFILE.expanduser())
    site, user = profile_paths({SITE_ENV: "/etc/almasim.toml", USER_ENV: str(tmp_path / "u")})
    assert site == Path("/etc/almasim.toml") and user == tmp_path / "u"


def test_the_zwicky_example_profile_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    profile = load_profile(site=ROOT / "examples" / "site-profiles" / "zwicky.toml", environ={})
    assert profile.partition == "zwickyA-CPU"
    assert profile.resources_for("image image-from-ms").time == "2-00:00:00"


def test_the_laptop_example_profile_runs_everything_on_the_server(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    profile = load_profile(site=ROOT / "examples" / "site-profiles" / "laptop.toml", environ={})
    assert profile.executor == "server"
    assert profile.runs_root == Path("/app/runs")


def test_the_profile_files_are_kept_as_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "site.toml", SITE)
    _write(tmp_path / "me.toml", "")
    profile = load_profile(site="site.toml", user="me.toml", environ={})
    assert profile.site_file == tmp_path / "site.toml"
    assert profile.user_file == tmp_path / "me.toml"


def test_the_site_names_the_teams_noise_models_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # no user profile
    site = _write(tmp_path / "site.toml", '[tools]\nnoise_models = "~/team-models"\n')
    assert load_profile(site=site, environ={}).noise_models == tmp_path / "team-models"
    assert load_profile(environ={}).noise_models is None
