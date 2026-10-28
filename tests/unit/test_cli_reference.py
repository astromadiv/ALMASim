"""docs/cli.md matches the command definitions (MADIV-28 T8.3)."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_the_cli_reference_is_current():
    generated = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "generate_cli_reference.py"), "--stdout"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    ).stdout
    current = (REPO / "docs" / "cli.md").read_text(encoding="utf-8")
    assert generated == current, "run: python scripts/generate_cli_reference.py"
    assert "--install-completion" not in generated
    assert "\n## `almasim image`\n" in generated, "commands sit under the page title"
    assert "\n### `almasim image image-from-ms`\n" in generated
    assert "`almasim image batch-image`" not in generated, "hidden commands are left out"
