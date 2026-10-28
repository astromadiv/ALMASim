"""almasim doctor (web console plan, J1.3)."""

from __future__ import annotations

import subprocess
import sys

import pytest
from typer.testing import CliRunner

from almasim import cli
from almasim.jobs import doctor as doctor_mod
from almasim.jobs.doctor import Check, format_report, passed, run_checks, srun_command
from almasim.jobs.profile import Profile

pytestmark = pytest.mark.unit


def _fake_run(outputs: dict[str, tuple[int, str, str]]):
    """A runner answering by the first word of the code or tool it runs."""

    def run(cmd: list[str], env=None) -> subprocess.CompletedProcess[str]:
        key = next((k for k in outputs if any(k in part for part in cmd)), None)
        code, out, err = outputs.get(key, (127, "", "not found"))
        return subprocess.CompletedProcess(cmd, code, out, err)

    return run


def test_a_working_profile_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    wsclean = tmp_path / "wsclean"
    casa_data = tmp_path / "casa-data"
    casa_data.mkdir()
    profile = Profile(
        runs_root=tmp_path / "runs",
        roots=(tmp_path,),
        python=sys.executable,
        wsclean=str(wsclean),
        casa_data=casa_data,
        partition="zwickyA-CPU",
    )
    run = _fake_run(
        {
            "import almasim": (0, "/x/almasim/__init__.py 3.12.13\n", ""),
            "casatools": (0, "6.7.5\n", ""),
            "import casacore": (0, "3.7.1\n", ""),
            "wsclean": (0, "\nWSClean version 3.7 (2026-04-24)\n", ""),
            "sinfo": (0, "zwickyA-CPU*\nzwickyB-CPU\n", ""),
        }
    )
    checks = run_checks(profile, run)
    assert passed(checks), format_report(checks)
    names = [c.name for c in checks]
    assert names == [
        "profile",
        "runs root",
        "roots",
        "python",
        "casatools",
        "casacore",
        "wsclean",
        "casa data",
        "slurm",
    ]
    assert "WSClean version 3.7" in format_report(checks)
    assert (tmp_path / "runs").is_dir()


def test_missing_optional_tools_warn_and_required_ones_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda tool: None)
    profile = Profile(runs_root=tmp_path / "runs", roots=(tmp_path / "gone",), python="/no/python")
    checks = {c.name: c for c in run_checks(profile, _fake_run({}))}
    assert not checks["roots"].ok and checks["roots"].required
    assert not checks["python"].ok
    assert not checks["wsclean"].ok and not checks["wsclean"].required
    assert not checks["slurm"].ok
    assert not passed(list(checks.values()))
    report = format_report(list(checks.values()))
    assert "FAIL  python" in report and "warn  wsclean" in report


def test_a_python_without_casacore_warns(tmp_path, monkeypatch):
    """image compute-parameters needs python-casacore: its absence is shown, not fatal."""
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    profile = Profile(runs_root=tmp_path / "runs", roots=(tmp_path,), python=sys.executable)
    run = _fake_run(
        {
            "import almasim": (0, "/x/almasim/__init__.py 3.12.13\n", ""),
            "casatools": (0, "6.7.5\n", ""),
            "import casacore": (1, "", "ModuleNotFoundError: No module named 'casacore'\n"),
            "sinfo": (0, "zwickyA-CPU*\n", ""),
        }
    )
    checks = run_checks(profile, run)
    report = format_report(checks)
    assert "warn  casacore" in report and "ms-casacore extra" in report
    assert "No module named 'casacore'" in report


def test_partitions_the_profile_names_must_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    profile = Profile(runs_root=tmp_path, roots=(tmp_path,), partition="gpu-x")
    checks = run_checks(profile, _fake_run({"sinfo": (0, "zwickyA-CPU*\n", "")}))
    assert [c for c in checks if c.name == "partitions" and "gpu-x" in c.detail and not c.ok]


def test_doctor_on_slurm_uses_srun_with_the_profile_python():
    profile = Profile(python="/opt/py", partition="zwickyA-CPU", account="ska")
    cmd = srun_command(profile, ["doctor"])
    assert cmd[0] == "srun"
    assert "--partition=zwickyA-CPU" in cmd and "--account=ska" in cmd
    assert cmd[-4:] == ["/opt/py", "-m", "almasim.cli", "doctor"]


def test_the_cli_reports_and_exits_with_the_result(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ALMASIM_SITE_PROFILE", raising=False)
    monkeypatch.delenv("ALMASIM_PROFILE", raising=False)
    monkeypatch.setattr(
        doctor_mod, "run_checks", lambda profile: [Check("runs root", True, "fine")]
    )
    result = CliRunner().invoke(cli.app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "ok    runs root  fine" in result.output

    monkeypatch.setattr(
        doctor_mod, "run_checks", lambda profile: [Check("python", False, "missing")]
    )
    assert CliRunner().invoke(cli.app, ["doctor"]).exit_code == 1

    bad = tmp_path / "bad.toml"
    bad.write_text("runs_rot = 1")
    result = CliRunner().invoke(cli.app, ["doctor", "--profile", str(bad)])
    assert result.exit_code == 1
