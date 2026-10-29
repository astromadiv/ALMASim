"""The command policy (web console plan, J1.6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from almasim.jobs import Resources
from almasim.jobs.policy import KNOWN_COMMANDS, PolicyError, has_option, plan_job
from almasim.jobs.profile import CommandPolicy, Profile

pytestmark = pytest.mark.unit

ZWICKY = Profile(
    partition="zwickyA-CPU",
    account="ska",
    wsclean="/opt/wsclean",
    casa_data=Path("/data/casa"),
    commands={"image image-from-ms": CommandPolicy(cpus=4, memory="16G", time="2-00:00:00")},
)


def test_every_command_runs_on_slurm_by_default():
    """zwicky's compute nodes reach the ALMA archive: queries and downloads go there too."""
    for argv in (
        ["metadata", "--band", "3"],
        ["products", "resolve", "--member-ous-uid", "x"],
        ["products", "download", "--products-csv", "p.csv"],
        ["image", "clean-quality", "/out"],
    ):
        assert plan_job(argv, ZWICKY).executor == "slurm", argv


def test_a_site_without_internet_on_compute_nodes_moves_them_to_the_server():
    profile = Profile(commands={"metadata": CommandPolicy(executor="server")})
    assert plan_job(["metadata"], profile).executor == "server"
    assert plan_job(["image", "clean-quality", "/out"], profile).executor == "slurm"


def test_a_machine_without_slurm_runs_everything_on_the_server():
    laptop = Profile(executor="server", commands={"metadata": CommandPolicy(cpus=2)})
    for argv in (["metadata"], ["image", "clean-quality", "/out"], ["doctor"]):
        assert plan_job(argv, laptop).executor == "server", argv
    # A command's own table still wins.
    mixed = Profile(executor="server", commands={"predict": CommandPolicy(executor="slurm")})
    assert plan_job(["predict", "ms", "model"], mixed).executor == "slurm"


def test_a_server_job_keeps_the_commands_own_backend():
    """Workers go to Slurm only from a Slurm job; on a laptop there is no Slurm."""
    laptop = Profile(executor="server", partition="p", account="a")
    argv = plan_job(["image", "image-from-ms", "p.csv", "out"], laptop).argv
    assert "--backend" not in argv and "--slurm-queue" not in argv
    calibrate = plan_job(["visibilities", "calibrate", "raw", "out"], laptop).argv
    assert "--backend" not in calibrate and "--skip-casa-data-update" not in calibrate


def test_commands_that_ask_for_confirmation_get_yes():
    assert plan_job(["metadata", "--band", "3"], ZWICKY).argv[-1] == "--yes"
    assert "--yes" not in plan_job(["image", "clean-quality", "/o"], ZWICKY).argv
    assert plan_job(["metadata", "-y"], ZWICKY).argv.count("--yes") == 0


def test_imaging_becomes_a_slurm_driver_with_the_profiles_tools():
    plan = plan_job(["image", "image-from-ms", "p.csv", "out", "--cube"], ZWICKY)
    assert plan.command == "image image-from-ms"
    assert plan.argv == [
        "image", "image-from-ms", "p.csv", "out", "--cube",
        "--backend", "slurm",
        "--slurm-queue", "zwickyA-CPU",
        "--slurm-project", "ska",
        "--casa-data-root", "/data/casa",
        "--wsclean-bin", "/opt/wsclean",
    ]  # fmt: skip
    assert plan.resources == Resources(
        partition="zwickyA-CPU", account="ska", cpus=4, memory="16G", time="2-00:00:00"
    )


def test_options_the_caller_gave_are_kept():
    argv = [
        "image", "image-from-ms", "p.csv", "out",
        "--backend=sync", "--wsclean-bin", "/my/wsclean", "--casa-data-root", "/mine",
    ]  # fmt: skip
    assert plan_job(argv, ZWICKY).argv == argv


def test_a_simulation_that_writes_an_ms_gets_the_profiles_casa_data():
    """casatools writes the MS only with its runtime data; a simulation
    without --ms-export needs none."""
    argv = ["simulation", "--metadata-path", "m.csv", "--ms-export"]
    planned = plan_job(argv, ZWICKY).argv
    assert planned[planned.index("--casa-data-root") + 1] == "/data/casa"
    assert "--casa-data-root" not in plan_job(argv[:-1], ZWICKY).argv
    mine = plan_job([*argv, "--casa-data-root", "/mine"], ZWICKY).argv
    assert mine.count("--casa-data-root") == 1


def test_calibration_on_slurm_does_not_download_casa_data():
    plan = plan_job(["visibilities", "calibrate", "--input-root", "/in"], ZWICKY)
    assert "--skip-casa-data-update" in plan.argv
    assert plan.argv[plan.argv.index("--backend") + 1] == "slurm"


def test_a_download_fans_out_only_for_its_post_processing():
    plain = plan_job(["products", "download", "--products-csv", "p.csv"], ZWICKY)
    assert plain.executor == "slurm"
    assert "--backend" not in plain.argv

    unpack = plan_job(["products", "download", "--products-csv", "p.csv", "--unpack-ms"], ZWICKY)
    assert unpack.argv[unpack.argv.index("--backend") + 1] == "slurm"
    # Jobs share one CASA data folder: they must not update it concurrently.
    assert "--skip-casa-data-update" in unpack.argv

    on_server = Profile(commands={"products download": CommandPolicy(executor="server")})
    served = plan_job(["products", "download", "--products-csv", "p.csv"], on_server)
    assert "--skip-casa-data-update" not in served.argv


def test_predict_gets_the_partition_and_clean_the_wsclean_binary():
    assert plan_job(["predict", "ms.ms"], ZWICKY).argv[-2:] == ["--slurm-queue", "zwickyA-CPU"]
    assert plan_job(["clean", "-size", "512", "512", "ms.ms"], ZWICKY).argv[-2:] == [
        "--wsclean-bin",
        "/opt/wsclean",
    ]


def test_extras_go_before_a_double_dash():
    plan = plan_job(["clean", "--", "-name", "x", "ms.ms"], ZWICKY)
    assert plan.argv == ["clean", "--wsclean-bin", "/opt/wsclean", "--", "-name", "x", "ms.ms"]


def test_the_profiles_extra_args_come_last_and_overrides_win():
    profile = Profile(commands={"simulation": CommandPolicy(args=("--verbose",), cpus=16)})
    plan = plan_job(["simulation", "--metadata-path", "m.csv"], profile, {"cpus": 32})
    assert plan.argv[-1] == "--verbose"
    assert plan.resources.cpus == 32


def test_the_console_releases_noise_models_into_the_site_folder_only():
    argv = ["noise", "promote", "c.json", "--comparison", "cmp.json", "--name", "zwicky-alma"]
    assert plan_job(argv, ZWICKY).argv == [*argv, "--site"]
    assert plan_job([*argv, "--site"], ZWICKY).argv.count("--site") == 1
    with pytest.raises(PolicyError, match="site folder only"):
        plan_job([*argv, "--models-dir", "/env/lib/models"], ZWICKY)


def test_unknown_commands_are_refused():
    with pytest.raises(PolicyError, match="unknown command"):
        plan_job(["rm", "-rf"], ZWICKY)
    with pytest.raises(PolicyError):
        plan_job([], ZWICKY)
    with pytest.raises(PolicyError):
        plan_job(["--help"], ZWICKY)


def test_has_option():
    assert has_option(["--backend=slurm"], "--backend")
    assert has_option(["x", "--postprocess-backend", "sync"], "--backend", "--postprocess-backend")
    assert not has_option(["--backends"], "--backend")


def test_every_known_command_exists_in_the_cli():
    """The policy's command list follows the CLI (docs/cli.md is generated from it)."""
    reference = (Path(__file__).resolve().parents[2] / "docs" / "cli.md").read_text()
    for command in KNOWN_COMMANDS:
        assert f"`almasim {command}`" in reference, command


def test_option_value():
    from almasim.jobs.policy import option_value

    assert option_value(["--backend=sync"], "--backend") == "sync"
    assert (
        option_value(["x", "--postprocess-backend", "slurm"], "--backend", "--postprocess-backend")
        == "slurm"
    )
    assert option_value(["--backend"], "--backend") is None
    assert option_value([], "--backend") is None
