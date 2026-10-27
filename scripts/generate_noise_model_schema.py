"""Write the JSON Schema of a noise model file (MADIV-28).

python scripts/generate_noise_model_schema.py           # writes docs/noise-model.schema.json
python scripts/generate_noise_model_schema.py --check   # fails if it is stale

The schema follows services.noise_model; regenerate it when a component or
its parameters change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from almasim.services.noise_model.schema import noise_model_schema

TARGET = Path(__file__).resolve().parents[1] / "docs" / "noise-model.schema.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the file is stale")
    args = parser.parse_args()
    text = json.dumps(noise_model_schema(), indent=2) + "\n"
    if args.check:
        if not TARGET.exists() or TARGET.read_text() != text:
            print(f"{TARGET} is stale: run scripts/generate_noise_model_schema.py", file=sys.stderr)
            return 1
        return 0
    TARGET.write_text(text)
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
