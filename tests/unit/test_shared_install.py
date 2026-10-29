"""almasim update-shared: update the shared install from a git ref."""

from __future__ import annotations

import subprocess

import pytest
from typer.testing import CliRunner

from almasim import cli
from almasim.cli import update_shared as update_mod
from almasim.jobs import service
from almasim.jobs.profile import Profile
from almasim.jobs.shared_install import (
    SharedInstallError,
    Source,
    find_prefix,
    job_script,
    read_source,
    resolve_source,
    sbatch_command,
)

pytestmark = pytest.mark.unit

REPO = "https://github.com/astromadiv/ALMASim"


def _install(tmp_path, record: str | None = None):
    prefix = tmp_path / "almasim"
    (prefix / "env" / "bin").mkdir(parents=True)
    (prefix / "almasim.sh").write_text("export PATH=...\n")
    (prefix / "site.toml").write_text(f'runs_root = "{tmp_path / "runs"}"\n')
    if record is not None:
        (prefix / "install.env").write_text(record)
    return prefix


def test_the_install_is_found_from_its_environment(tmp_path):
    prefix = _install(tmp_path)
    assert find_prefix(str(prefix / "env")) == prefix
    assert find_prefix(str(tmp_path / "elsewhere" / "env")) is None


def test_the_record_gives_the_source_and_the_command_line_overrides_it(tmp_path):
    prefix = _install(
        tmp_path, f"repo={REPO}\ncommit=e05ad09a\ntemplate=examples/site-profiles/zwicky.toml\n"
    )
    recorded = read_source(prefix)
    assert recorded == Source(REPO, "examples/site-profiles/zwicky.toml", "e05ad09a")
    assert resolve_source(recorded, None, None) == recorded
    assert resolve_source(recorded, "/srv/fork.git", None).repo == "/srv/fork.git"


def test_an_install_without_a_record_needs_repo_and_template(tmp_path):
    assert read_source(_install(tmp_path)) is None
    with pytest.raises(SharedInstallError, match="--repo and --template"):
        resolve_source(None, None, None)
    assert resolve_source(None, REPO, "site.toml") == Source(REPO, "site.toml")


def test_the_job_clones_the_ref_installs_with_its_script_and_checks(tmp_path):
    prefix = tmp_path / "almasim"
    work = prefix / "updates" / "1"
    text = job_script(prefix, work, Source(REPO, "examples/site-profiles/zwicky.toml"), "main")
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)
    assert f'git clone --quiet --no-checkout {REPO} "$src"' in text
    assert "ref=main" in text
    assert (
        f'"$src/scripts/install_shared.sh" {prefix} "$src"/examples/site-profiles/zwicky.toml'
        in text
    )
    assert f"ALMASIM_SITE_PROFILE={prefix}/site.toml {prefix}/env/bin/almasim doctor" in text
    # An absolute template is used as it is, and values are quoted.
    text = job_script(prefix, work, Source(REPO, "/etc/almasim site.toml"), "v2.2.0; rm")
    assert "'/etc/almasim site.toml'" in text
    assert "ref='v2.2.0; rm'" in text


def test_sbatch_waits_and_uses_the_profile_partition(tmp_path):
    cmd = sbatch_command(Profile(partition="p", account="a"), tmp_path / "u.sh", tmp_path / "u.log")
    assert cmd[:3] == ["sbatch", "--wait", "--parsable"]
    assert "--partition=p" in cmd and "--account=a" in cmd
    assert cmd[-1] == str(tmp_path / "u.sh")


@pytest.fixture
def tools(monkeypatch):
    monkeypatch.setattr(update_mod.shutil, "which", lambda tool: f"/usr/bin/{tool}")


def test_unfinished_runs_stop_the_update(tmp_path, monkeypatch, tools):
    prefix = _install(tmp_path, f"repo={REPO}\ntemplate=t.toml\n")
    busy = type("Run", (), {})()
    busy.state = service.JobState.RUNNING
    busy.job = type(
        "Job", (), {"id": "0" * 16 + "ABCDEFGHIJ", "owner": "someone", "command": "simulation"}
    )()
    monkeypatch.setattr(service, "list_runs", lambda profile: [busy])
    submitted = []
    monkeypatch.setattr(update_mod.subprocess, "run", lambda *a, **k: submitted.append(a))

    result = CliRunner().invoke(cli.app, ["update-shared", "--prefix", str(prefix)])
    assert result.exit_code == 1
    assert "ABCDEFGHIJ" in result.output and "1 run(s)" in result.output
    assert not submitted


def test_the_update_submits_the_job_and_shows_its_log(tmp_path, monkeypatch, tools):
    prefix = _install(tmp_path, f"repo={REPO}\ntemplate=t.toml\n")
    monkeypatch.setattr(service, "list_runs", lambda profile: [])

    def sbatch(cmd, **kwargs):
        log = next(c for c in cmd if c.startswith("--output=")).removeprefix("--output=")
        with open(log, "w") as f:
            f.write("installed almasim 2.1.17\nok  profile\n")
        return subprocess.CompletedProcess(cmd, 0, "7600\n", "")

    monkeypatch.setattr(update_mod.subprocess, "run", sbatch)
    result = CliRunner().invoke(
        cli.app, ["update-shared", "--prefix", str(prefix), "--ref", "v2.1.17"]
    )
    assert result.exit_code == 0, result.output
    assert "installed almasim 2.1.17" in result.output
    (script,) = (prefix / "updates").glob("*/update.sh")
    assert "ref=v2.1.17" in script.read_text()


def test_the_install_script_installs_every_extra_jobs_need():
    """Jobs run the shared Python: image compute-parameters needs python-casacore,
    and a Dask scheduler serves the dashboard the console relays only with bokeh."""
    from pathlib import Path

    script = (Path(__file__).parents[2] / "scripts" / "install_shared.sh").read_text()
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    for extra in ("web", "casa", "ms-casacore", "dashboard"):
        assert f"--extra {extra}" in script
    assert '"$wheel[web,casa,ms-casacore,dashboard]"' in script


def test_the_install_makes_the_teams_noise_models_folder(tmp_path):
    """The site profile points at <prefix>/noise-models, made group-writable and
    setgid; run with a stand-in uv, whose environment's Python only says its version."""
    import os
    import stat
    from pathlib import Path

    repo = Path(__file__).parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = venv ]; then mkdir -p "${@: -1}/bin"; '
        "printf '#!/bin/bash\\necho 0.0\\n' > \"${@: -1}/bin/python\"; "
        'chmod +x "${@: -1}/bin/python"; fi\n'
    )
    uv.chmod(0o755)
    prefix = tmp_path / "almasim"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    template = repo / "examples" / "site-profiles" / "zwicky.toml"
    for _ in range(2):  # an update leaves the folder as it is
        subprocess.run(
            [
                "bash",
                str(repo / "scripts" / "install_shared.sh"),
                str(prefix),
                str(template),
                "w.whl",
            ],
            env=env,
            check=True,
            capture_output=True,
        )
    folder = prefix / "noise-models"
    mode = folder.stat().st_mode
    assert folder.is_dir() and mode & stat.S_ISGID and mode & stat.S_IWGRP
    site = (prefix / "site.toml").read_text()
    assert f'noise_models = "{folder}"' in site
    assert f'executable = "{prefix}/env/bin/python"' in site
