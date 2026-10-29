"""Noise models as versioned data: components, selection, files (MADIV-28)."""

import json
from dataclasses import fields
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from almasim.services.interferometry.corruptions import CorruptionModel
from almasim.services.noise_model import (
    COMPONENT_NAMES,
    Components,
    NoiseComponent,
    NoiseModel,
    released,
    released_model,
    resolve_noise_model,
)
from almasim.services.noise_model.schema import noise_model_schema

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_alma_is_released_and_named_by_version():
    assert 1 in released()["alma"]
    alma = released_model("alma")
    assert resolve_noise_model("alma") == alma
    assert resolve_noise_model("alma@1") == released_model("alma", 1)
    assert resolve_noise_model("none") is None and resolve_noise_model(None) is None
    with pytest.raises(ValueError, match="no version 99"):
        resolve_noise_model("alma@99")
    assert set(alma.enabled()) == set(COMPONENT_NAMES)
    assert alma.provenance is not None and alma.provenance.windows == 100


@pytest.mark.unit
def test_every_component_sets_only_simulator_fields():
    """A new component must map onto CorruptionModel fields, and every
    component's fields are its own (selecting one never sets another's)."""
    engine = {f.name for f in fields(CorruptionModel)}
    alma = released_model("alma")
    seen: dict[str, str] = {}
    for name, component in alma.components.items():
        for key in component.engine_fields():
            assert key in engine, (name, key)
            assert key not in seen, (name, key, seen.get(key))
            seen[key] = name
    assert set(Components.model_fields) == set(COMPONENT_NAMES)
    for info in Components.model_fields.values():
        assert any(
            isinstance(a, type) and issubclass(a, NoiseComponent) for a in info.annotation.__args__
        )


@pytest.mark.unit
def test_components_are_selected_and_separable():
    alma = released_model("alma")
    only = alma.select(only=["radiometer,channel_correlation"])
    assert only.enabled() == ["radiometer", "channel_correlation"]
    model = CorruptionModel.from_mapping(only.engine_fields())
    assert model.radiometer_noise and model.correlates_channels
    assert not model.has_antenna_errors and not model.atmospheric_tsys
    assert model.antenna_ripple_rms == 0 and model.receiver_ripple_rms == 0

    without = alma.select(without=["fast_phase", "baseline_errors"])
    model = CorruptionModel.from_mapping(without.engine_fields())
    assert model.fast_path_um_1km == 0 and model.baseline_amp_rms == 0
    assert model.slow_path_um_1km == 80 and model.gain_amp_rms == pytest.approx(0.08)

    # The selection is part of the model's identity; selecting nothing away is the model.
    assert without.sha256() != alma.sha256()
    assert without.label() == "alma@1 without fast_phase,baseline_errors"
    assert alma.select() is alma
    assert alma.select(only=[]).engine_fields() is None
    with pytest.raises(ValueError, match="unknown noise component"):
        alma.select(without=["wind"])
    partial = alma.select(only=["radiometer"])
    with pytest.raises(ValueError, match="has no component"):
        NoiseModel(
            name="thermal",
            version=1,
            components=Components(radiometer=partial.components.radiometer),
        ).select(only=["slow_phase"])


@pytest.mark.unit
def test_models_are_checked_against_the_schema():
    base = json.loads(released_model("alma").model_dump_json())
    with pytest.raises(ValidationError):
        NoiseModel.model_validate({**base, "components": {"wind": {}}})
    bad = json.loads(json.dumps(base))
    bad["components"]["channel_correlation"]["rho2"] = 0.3
    with pytest.raises(ValidationError):
        NoiseModel.model_validate(bad)
    with pytest.raises(ValidationError):
        NoiseModel.model_validate({**base, "colour": "blue"})
    candidate = {**base, "name": "alma-candidate", "version": 2}
    candidate["components"]["slow_phase"]["slope"] = 0.2
    model = resolve_noise_model(json.dumps(candidate))
    assert model.name == "alma-candidate" and model.components.slow_phase.slope == 0.2


@pytest.mark.unit
def test_the_flat_corruption_fields_still_name_a_model(tmp_path):
    flat = {"gain_amp_rms": "0.1", "atmospheric_tsys": "true", "scan_length_s": 60}
    model = resolve_noise_model(flat)
    assert model.enabled() == ["atmospheric_tsys", "antenna_gains"]
    assert CorruptionModel.from_mapping(model.engine_fields()) == CorruptionModel(
        gain_amp_rms=0.1, atmospheric_tsys=True, scan_length_s=60.0
    )
    path = tmp_path / "Stormy Night.json"
    path.write_text(json.dumps(flat))
    assert resolve_noise_model(str(path)).name == "stormy-night"
    assert resolve_noise_model({"scan_length_s": 60}) is None
    with pytest.raises(ValueError, match="unknown corruption parameter"):
        resolve_noise_model({"gain_amplitude": 0.1})


def _params(tmp_path, **values):
    import pandas as pd

    from almasim.services.simulation.params import SimulationParams

    row = pd.read_csv(REPO / "data" / "qso_metadata.csv").iloc[0]
    dirs = {name: tmp_path for name in ("output_dir", "tng_dir", "galaxy_zoo_dir", "hubble_dir")}
    return SimulationParams.from_metadata_row(
        row, idx=0, project_name="t", main_dir=REPO / "src" / "almasim", **dirs, **values
    )


@pytest.mark.unit
def test_simulation_parameters_hold_the_model_and_its_fields(tmp_path):
    params = _params(tmp_path, noise_model="alma@1")
    assert params.noise_model == released_model("alma", 1)
    assert params.corruptions == released_model("alma").engine_fields()
    # The earlier keyword names the model as noise_model does.
    assert _params(tmp_path, corruptions="alma").noise_model == params.noise_model
    assert _params(tmp_path, noise_model=float("nan")).noise_model is None
    assert _params(tmp_path).corruptions is None


@pytest.mark.unit
def test_products_record_the_model_that_made_them(tmp_path):
    import numpy as np

    from almasim.services.ms.subtables import history_table
    from almasim.services.simulation.metadata import _noise_model_line

    params = _params(tmp_path, noise_model="alma")
    line = _noise_model_line(params)
    assert line == f"alma@1 (sha256 {released_model('alma').sha256()})"
    assert _noise_model_line(_params(tmp_path)) is None
    identity = released_model("alma").identity()
    assert identity["label"] == "alma@1" and identity["components"] == list(COMPONENT_NAMES)
    spec = history_table("p", np.array([0.0, 1.0]), (f"noise_model={json.dumps(identity)}",))
    assert list(spec.values["APP_PARAMS"])[1].startswith("noise_model=")
    assert list(history_table("p", np.array([0.0, 1.0])).values["APP_PARAMS"]) == ["p"]


@pytest.mark.unit
def test_the_schema_file_is_current():
    path = REPO / "docs" / "noise-model.schema.json"
    assert json.loads(path.read_text()) == noise_model_schema(), (
        "run scripts/generate_noise_model_schema.py"
    )


@pytest.mark.unit
def test_noise_commands_show_models_components_and_schema(tmp_path):
    from almasim.cli.noise import noise_app

    runner = CliRunner()
    result = runner.invoke(noise_app, ["show"])
    assert result.exit_code == 0 and "alma (package): versions 1" in result.output
    result = runner.invoke(noise_app, ["show", "alma", "--noise-without", "fast_phase"])
    assert result.exit_code == 0
    assert "[off] fast_phase" in result.output and "[on ] radiometer" in result.output
    result = runner.invoke(noise_app, ["show", "alma@1", "--json"])
    assert NoiseModel.model_validate_json(result.output) == released_model("alma", 1)
    result = runner.invoke(noise_app, ["show", "stormy"])
    assert result.exit_code == 1 and "not a released model" in result.output
    result = runner.invoke(noise_app, ["components"])
    assert all(f"{name}:" in result.output for name in COMPONENT_NAMES)
    out = tmp_path / "schema.json"
    assert runner.invoke(noise_app, ["schema", "-o", str(out)]).exit_code == 0
    assert json.loads(out.read_text()) == noise_model_schema()


@pytest.mark.unit
def test_alma_states_the_instrument_radiometer():
    """alma@1 sets ALMA's radiometer as ALMASim models it, value for value."""
    from almasim.services.interferometry import noise

    config = released_model("alma", 1).noise_config()
    assert config["receiver_temperatures"] == noise._RECEIVER_TEMPERATURE_K
    assert config["ruze"] == dict(noise._RUZE)
    assert (
        config["correlator_efficiency"] == noise.NoiseModelConfig(pwv_mm=1.0).correlator_efficiency
    )
    assert released_model("alma").select(without=["radiometer"]).noise_config() == {}


@pytest.mark.unit
def test_a_candidate_radiometer_changes_tsys_where_it_differs():
    import numpy as np

    from almasim.services.interferometry.noise import (
        NoiseModelConfig,
        system_equivalent_flux_density,
    )

    alma = released_model("alma")
    rows = [r.model_dump() for r in alma.components.radiometer.receiver_temperature_k]
    rows[4]["trx_k"] = 92.0  # band 6 twice as hot
    candidate = NoiseModel.model_validate(
        {
            **json.loads(alma.model_dump_json()),
            "name": "alma-hot-b6",
            "components": {"radiometer": {"receiver_temperature_k": rows}},
        }
    )

    def sefd(model, ghz):
        config = NoiseModelConfig(pwv_mm=1.0, **model.noise_config())
        return system_equivalent_flux_density(
            config, np.array([ghz * 1e9]), 60.0, antenna_diameter_m=12.0
        )[0]

    assert sefd(candidate, 230.0) > 1.3 * sefd(alma, 230.0)
    assert sefd(candidate, 100.0) == sefd(alma, 100.0)
    rows[0]["upper_ghz"] = 400.0
    with pytest.raises(ValidationError, match="edges must rise"):
        NoiseModel.model_validate(
            {
                **json.loads(candidate.model_dump_json()),
                "components": {"radiometer": {"receiver_temperature_k": rows}},
            }
        )
