"""The JSON Schema of a noise model file (docs/noise-model.schema.json)."""

from __future__ import annotations

from typing import Any

from .model import NoiseModel

SCHEMA_ID = "https://github.com/astromadiv/ALMASim/blob/main/docs/noise-model.schema.json"


def noise_model_schema() -> dict[str, Any]:
    """The schema every noise model file validates against."""
    schema = NoiseModel.model_json_schema()
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": SCHEMA_ID, **schema}
