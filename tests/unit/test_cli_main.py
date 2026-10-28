"""The almasim entry point, the clean passthrough and the cli_* aliases (MADIV-28 T8.2)."""

from __future__ import annotations

from types import SimpleNamespace

import click
import pytest
import typer
from typer.testing import CliRunner

from almasim import cli
from almasim.cli import clean as cli_clean

runner = CliRunner()


def test_clean_passthrough_runs_wsclean(monkeypatch):
    """Top-level clean command should forward unknown options to WSClean."""
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli_clean.shutil, "which", lambda executable: f"/usr/bin/{executable}")

    def _fake_run(command, cwd=None, check=False):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli_clean.subprocess, "run", _fake_run)

    result = runner.invoke(
        cli.app,
        [
            "clean",
            "--wsclean-bin",
            "wsclean",
            "--",
            "-name",
            "img",
            "-niter",
            "1000",
            "input.ms",
        ],
    )

    assert result.exit_code == 0
    assert captured["command"] == [
        "/usr/bin/wsclean",
        "-name",
        "img",
        "-niter",
        "1000",
        "input.ms",
    ]


def test_invoke_click_command_raises_typer_exit_for_nonzero_int_result():
    """Click command return codes should map to Typer exit codes."""

    class _FakeClickCommand:
        def main(self, **kwargs):
            del kwargs
            return 9

    with pytest.raises(typer.Exit) as exc:
        cli._invoke_click_command(_FakeClickCommand(), args=[], prog_name="almasim")

    assert exc.value.exit_code == 9


def test_invoke_click_command_maps_click_exit_to_typer_exit():
    """Click Exit exceptions should be mapped to Typer Exit with same code."""

    class _FakeClickCommand:
        def main(self, **kwargs):
            del kwargs
            raise click.exceptions.Exit(3)

    with pytest.raises(typer.Exit) as exc:
        cli._invoke_click_command(_FakeClickCommand(), args=[], prog_name="almasim")

    assert exc.value.exit_code == 3


def test_the_old_cli_module_aliases_are_gone():
    """``almasim.cli_<group>`` were removed in MADIV-28 T8.4: use ``almasim.cli.<group>``."""
    import importlib

    for old in ("cli_image", "cli_products", "cli_shared", "cli_simulation"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"almasim.{old}")


def test_child_process_targets_name_real_modules():
    """Every function a child process runs lives where its target string says."""
    import importlib

    from almasim.services.imaging.selfcal import loop

    for function, module in loop._CHILD_FUNCTIONS.items():
        assert callable(getattr(importlib.import_module(module), function)), function


def test_simulation_main_dir_defaults_to_the_package_data():
    """The default ``--main-dir`` is ``src/almasim``, where the data directories live."""
    import inspect

    from almasim.cli import simulation

    default = inspect.signature(simulation.simulation_run).parameters["main_dir"].default.default
    assert (default / "cli").is_dir() and default.name == "almasim"


def test_python_m_almasim_cli_runs_the_app(monkeypatch, capsys):
    """``python -m almasim.cli --help`` prints the command groups."""
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["almasim", "--help"])
    with pytest.raises(SystemExit) as info:
        runpy.run_module("almasim.cli", run_name="__main__")
    assert info.value.code == 0
    assert "visibilities" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        ((1024, 131072), [(131072, 131072)]),  # the console's inherited limit
        ((131072, 131072), []),  # already at the hard limit
        ((1024, -1), [(1 << 20, -1)]),  # unlimited hard limit: a finite soft one
    ],
)
def test_main_lifts_the_soft_open_file_limit(monkeypatch, limits, expected):
    """A 2048-channel WSClean cube opens ~6000 files; 1024 made it fail."""
    resource = pytest.importorskip("resource")
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(resource, "RLIM_INFINITY", -1)
    monkeypatch.setattr(resource, "getrlimit", lambda _: limits)
    monkeypatch.setattr(resource, "setrlimit", lambda _, value: calls.append(value))
    monkeypatch.setattr(cli, "app", lambda: None)

    cli.main()

    assert calls == expected


def test_main_starts_when_the_limit_cannot_be_raised(monkeypatch):
    """A refused raise keeps the inherited limit instead of failing the command."""
    resource = pytest.importorskip("resource")
    ran: list[bool] = []

    def _refuse(_, value):
        raise ValueError("not allowed")

    monkeypatch.setattr(resource, "getrlimit", lambda _: (1024, 4096))
    monkeypatch.setattr(resource, "setrlimit", _refuse)
    monkeypatch.setattr(cli, "app", lambda: ran.append(True))

    cli.main()

    assert ran == [True]
