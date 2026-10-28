"""Each docs page has one title, so the Pages sidebar lists each page once."""

import re
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "docs" / "source"
INCLUDE = re.compile(r"^```\{include\} (?P<path>\S+)\n(?P<options>(?::[^\n]*\n)*)", re.M)


def _titles(text: str) -> int:
    """The ``#`` headings in ``text`` outside code blocks."""
    count = 0
    in_code = False
    for line in text.splitlines():
        if line.startswith(("```", "~~~")):
            in_code = not in_code
        elif not in_code and line.startswith("# "):
            count += 1
    return count


@pytest.mark.unit
@pytest.mark.parametrize("page", sorted(SOURCE.glob("*.md")), ids=lambda page: page.name)
def test_a_page_has_one_title_with_its_includes(page):
    text = page.read_text(encoding="utf-8")
    titles = _titles(INCLUDE.sub("", text))
    for include in INCLUDE.finditer(text):
        target = page.parent / include["path"]
        assert target.is_file(), f"{page.name} includes {include['path']}, which is missing"
        start = re.search(r":start-line: (\d+)", include["options"])
        lines = target.read_text(encoding="utf-8").splitlines()[int(start[1]) if start else 0 :]
        titles += _titles("\n".join(lines))
    assert titles == 1, f"{page.name} has {titles} '# ' titles; demote or skip the extra ones"
