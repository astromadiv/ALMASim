"""Releasing a validated candidate as a model's next version (MADIV-28)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from almasim.services.noise.compare import Check, Comparison, Criteria
from almasim.services.noise.promote import promote
from almasim.services.noise_model import NoiseModel, released_model


def _candidate() -> NoiseModel:
    alma = released_model("alma", 1)
    components = alma.components.model_copy(
        update={
            "antenna_gains": alma.components.antenna_gains.model_copy(update={"amp_rms": 0.065})
        }
    )
    return alma.model_copy(
        update={"name": "alma-candidate", "version": 2, "components": components}
    )


def _comparison(model: NoiseModel, passed: bool = True) -> Comparison:
    return Comparison(
        model=model.identity(),
        windows=40,
        criteria=Criteria(),
        metrics={},
        groups={},
        checks=[
            Check(name="cont.sigma median within 5%", passed=passed, value=1.02 if passed else 0.8)
        ],
        passed=passed,
    )


@pytest.mark.unit
def test_a_passing_candidate_becomes_the_next_version(tmp_path):
    (tmp_path / "alma-1.json").write_text(released_model("alma", 1).model_dump_json())
    candidate = _candidate()
    path = promote(candidate, _comparison(candidate), models_dir=tmp_path)
    assert path == tmp_path / "alma-2.json"
    released = NoiseModel.model_validate_json(path.read_text())
    assert released.label() == "alma@2"
    assert released.components.antenna_gains.amp_rms == 0.065
    assert "Validated on 40 windows" in released.provenance.notes
    again = promote(candidate, _comparison(candidate), name="alma", models_dir=tmp_path)
    assert again.name == "alma-3.json"


@pytest.mark.unit
def test_only_a_passed_comparison_of_that_model_promotes(tmp_path):
    candidate = _candidate()
    with pytest.raises(ValueError, match="did not pass"):
        promote(candidate, _comparison(candidate, passed=False), models_dir=tmp_path)
    with pytest.raises(ValueError, match="the comparison is of alma@1"):
        promote(candidate, _comparison(released_model("alma", 1)), models_dir=tmp_path)
    with pytest.raises(ValueError, match="not a model name"):
        promote(candidate, _comparison(candidate), name="Alma 2", models_dir=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.unit
def test_noise_promote_command(tmp_path):
    from almasim.cli.noise import noise_app

    candidate = _candidate()
    model_path, comparison_path = tmp_path / "candidate.json", tmp_path / "comparison.json"
    model_path.write_text(candidate.model_dump_json())
    comparison_path.write_text(_comparison(candidate).model_dump_json())
    models = tmp_path / "models"
    runner = CliRunner()
    args = [
        "promote",
        str(model_path),
        "--comparison",
        str(comparison_path),
        "--models-dir",
        str(models),
    ]
    result = runner.invoke(noise_app, args)
    assert result.exit_code == 0, result.output
    assert "alma@1" in result.output and (models / "alma-1.json").is_file()
    comparison_path.write_text(_comparison(candidate, passed=False).model_dump_json())
    refused = runner.invoke(noise_app, args)
    assert refused.exit_code == 1 and "did not pass" in refused.output
    assert json.loads((models / "alma-1.json").read_text())["version"] == 1


@pytest.mark.unit
def test_a_site_release_takes_its_own_name_and_records_who(tmp_path, monkeypatch):
    monkeypatch.setattr("getpass.getuser", lambda: "astro")
    candidate = _candidate()
    with pytest.raises(ValueError, match="'alma' is a model ALMASim releases"):
        promote(candidate, _comparison(candidate), models_dir=tmp_path, site=True)
    path = promote(
        candidate, _comparison(candidate), name="zwicky-alma", models_dir=tmp_path, site=True
    )
    assert path == tmp_path / "zwicky-alma-1.json"
    notes = NoiseModel.model_validate_json(path.read_text()).provenance.notes
    assert "passed every check" in notes and "Released to the site by astro on " in notes


@pytest.mark.unit
def test_a_release_never_overwrites_a_version(tmp_path, monkeypatch):
    candidate = _candidate()
    real_open = Path.open

    def raced(self, mode="r", *args, **kwargs):
        if mode == "x":  # another release wrote this version first
            self.write_text("{}")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", raced)
    with pytest.raises(ValueError, match="alma-1.json was written meanwhile"):
        promote(candidate, _comparison(candidate), models_dir=tmp_path)
    assert (tmp_path / "alma-1.json").read_text() == "{}"


@pytest.mark.unit
def test_noise_promote_into_the_site_folder(tmp_path, monkeypatch):
    from almasim.cli.noise import noise_app

    candidate = _candidate()
    model_path, comparison_path = tmp_path / "candidate.json", tmp_path / "comparison.json"
    model_path.write_text(candidate.model_dump_json())
    comparison_path.write_text(_comparison(candidate).model_dump_json())
    site = tmp_path / "noise-models"
    monkeypatch.setenv("ALMASIM_NOISE_MODELS", str(site))
    runner = CliRunner()
    args = ["promote", str(model_path), "--comparison", str(comparison_path), "--site"]
    missing = runner.invoke(noise_app, [*args, "--name", "zwicky-alma"])
    assert missing.exit_code == 1 and "almasim update-shared makes it" in missing.output
    site.mkdir()
    both = runner.invoke(noise_app, [*args, "--models-dir", str(site)])
    assert both.exit_code == 2 and "not both" in both.output
    reused = runner.invoke(noise_app, args)
    assert reused.exit_code == 1 and "a name of its own" in reused.output
    result = runner.invoke(noise_app, [*args, "--name", "zwicky-alma"])
    assert result.exit_code == 0, result.output
    assert "--noise-model zwicky-alma@1" in result.output and "pull request" not in result.output
    assert (site / "zwicky-alma-1.json").is_file()
