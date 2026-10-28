"""The Open OnDemand app bundle (ood/almasim) agrees with the CLI (Open OnDemand plan, O4).

The portal cannot run in CI; these checks keep the bundle's files in step
with what `almasim serve` takes and with each other.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

APP = Path(__file__).resolve().parents[2] / "ood" / "almasim"


def _read(name: str) -> str:
    return (APP / name).read_text(encoding="utf-8")


def test_the_bundle_has_the_files_a_batch_connect_app_needs():
    for name in [
        "manifest.yml",
        "form.yml",
        "submit.yml.erb",
        "template/before.sh.erb",
        "template/script.sh.erb",
        "view.html.erb",
    ]:
        assert (APP / name).is_file(), name
    assert yaml.safe_load(_read("manifest.yml"))["role"] == "batch_connect"


def test_every_form_field_is_defined_and_used():
    form = yaml.safe_load(_read("form.yml"))
    custom = [f for f in form["form"] if not f.startswith("bc_")]
    assert set(custom) <= set(form["attributes"])
    used = _read("submit.yml.erb") + _read("template/script.sh.erb")
    for field in custom:
        assert field in used, f"{field} is asked for but never used"


def test_the_start_script_uses_only_options_serve_has():
    import typer

    from almasim.cli import app as cli

    # From the command itself, not its help (which colour codes break up).
    serve = typer.main.get_command(cli).commands["serve"]
    options = {name for param in serve.params for name in param.opts}
    script = _read("template/script.sh.erb")
    assert "almasim serve" in script
    for option in re.findall(r"(--[a-z][a-z-]+)", script):
        assert option in options, option


def test_the_console_and_the_connect_button_agree_on_the_path_and_token():
    before = _read("template/before.sh.erb")
    view = _read("view.html.erb")
    # The console serves under the path the button opens, with the token it shows.
    assert 'ALMASIM_BASE_PATH="/node/${host}/${port}"' in before
    assert 'action="/node/<%= host %>/<%= port %>/"' in view
    assert 'ALMASIM_TOKEN="${password}"' in before
    assert 'name="token" value="<%= password %>"' in view
    # A token the console accepts (at least 16 characters).
    length = int(re.search(r"create_passwd (\d+)", before).group(1))
    assert length >= 16


def test_the_rehearsal_script_fills_in_what_the_templates_ask_for():
    rehearse = _read("try-without-ondemand.sh")
    # It runs the app's own scripts, not copies of them...
    assert "template/before.sh.erb" in rehearse and "template/script.sh.erb" in rehearse
    # ...replacing the one ERB tag in script.sh, which must still be there.
    tag = "<%= context.almasim_setup %>"
    assert tag in rehearse and tag in _read("template/script.sh.erb")
    # It provides OnDemand's helpers that before.sh calls.
    for helper in re.findall(r"\$\((\w+) ", _read("template/before.sh.erb")):
        assert f"{helper}()" in rehearse, helper


def test_the_rehearsal_script_keeps_one_console_per_user():
    rehearse = _read("try-without-ondemand.sh")
    # It looks for this user's console by the job name it submits it under...
    name = re.search(r"sbatch --parsable -J (\S+)", rehearse).group(1)
    assert re.search(rf'squeue -h -u "\$USER" -n {name} ', rehearse)
    # ...and --replace, which must come before the sbatch options, stops it.
    assert 'if [ "${1:-}" = "--replace" ]; then replace=true; shift; fi' in rehearse
    assert "scancel $(" in rehearse
