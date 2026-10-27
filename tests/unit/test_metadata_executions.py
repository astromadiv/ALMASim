"""Each MOUS's execution blocks from DataLink: n_executions (MADIV-28)."""

from __future__ import annotations

import pandas as pd
import pytest
from typer.testing import CliRunner

from almasim import cli
from almasim.services.download.datalink import count_executions
from almasim.services.download.models import DataProduct
from almasim.services.metadata import executions

runner = CliRunner()


def _product(uid: str, filename: str, product_type: str = "raw") -> DataProduct:
    return DataProduct(f"https://x/{filename}", uid, filename, 1, "application/x-tar", product_type)


def _products() -> list[DataProduct]:
    return [
        _product("uid://A001/X1/X1", "2023.1.00882.S_uid___A002_X11adad7_X304ff.asdm.sdm.tar"),
        _product("uid://A001/X1/X1", "2023.1.00882.S_uid___A002_X11adad7_X3767c.asdm.sdm.tar"),
        # The same block listed twice still counts once.
        _product("uid://A001/X1/X1", "2023.1.00882.S_uid___A002_X11adad7_X3767c.asdm.sdm.tar"),
        _product("uid://A001/X1/X1", "member.uid___A001_X1_X1.README.txt", "auxiliary"),
        _product("uid://A001/X2/X2", "2023.1.01732.S_uid___A002_X127a8ae_X286e2.asdm.sdm.tar"),
        _product("uid://A001/X3/X3", "2023.1.01732.S_uid___A001_X3_X3_001_of_001.tar", "fits"),
    ]


@pytest.mark.unit
def test_a_mous_has_one_execution_per_raw_asdm():
    assert count_executions(_products()) == {"uid://A001/X1/X1": 2, "uid://A001/X2/X2": 1}


@pytest.mark.unit
def test_each_row_gets_its_mous_count_and_none_without_raw_data():
    metadata = pd.DataFrame(
        {"member_ous_uid": ["uid://A001/X1/X1", "uid://A001/X3/X3", "uid://A001/X1/X1", None]}
    )
    asked = []

    def resolve(uids):
        asked.append(list(uids))
        return _products()

    counted = executions.add_execution_counts(metadata, resolve=resolve)
    assert asked == [["uid://A001/X1/X1", "uid://A001/X3/X3"]]
    assert counted["n_executions"].tolist() == [2, pd.NA, 2, pd.NA]
    assert "n_executions" not in metadata.columns


@pytest.mark.unit
def test_metadata_without_mous_uids_is_refused():
    with pytest.raises(KeyError):
        executions.add_execution_counts(pd.DataFrame({"x": [1]}), resolve=lambda uids: [])


@pytest.mark.unit
def test_the_executions_command_adds_the_column(tmp_path, monkeypatch):
    path = tmp_path / "rows.csv"
    pd.DataFrame({"member_ous_uid": ["uid://A001/X1/X1", "uid://A001/X3/X3"]}).to_csv(path)
    monkeypatch.setattr(
        executions, "add_execution_counts", lambda m: m.assign(n_executions=[2, None])
    )
    out = tmp_path / "out.csv"
    result = runner.invoke(
        cli.app, ["products", "executions", "--metadata-path", str(path), "--save-csv", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "n_executions left empty" in result.output
    assert pd.read_csv(out)["n_executions"].tolist()[0] == 2


@pytest.mark.unit
def test_query_counts_executions_when_asked(tmp_path, monkeypatch):
    from almasim.cli import metadata as cli_metadata

    rows = pd.DataFrame({"member_ous_uid": ["uid://A001/X1/X1"], "Band": [6], "Freq": [230.0]})
    monkeypatch.setattr(cli_metadata, "query_metadata_by_science", lambda **kwargs: rows)
    monkeypatch.setattr(executions, "add_execution_counts", lambda m: m.assign(n_executions=[13]))
    out = tmp_path / "q.csv"
    result = runner.invoke(
        cli.app, ["metadata", "query", "--yes", "--count-executions", "--save-csv", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert pd.read_csv(out)["n_executions"].tolist() == [13]
