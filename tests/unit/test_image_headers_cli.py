"""``image fetch-headers`` and ``image parse-headers`` on a small delivery tree."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest
from astropy.io import fits
from typer.testing import CliRunner

from almasim import cli
from almasim.services.imaging import image_headers

runner = CliRunner()

MEMBER = "uid://A001/X1/X2"
MEMBER_FOLDER = (
    "2023.1.00001.S/science_goal.uid___A001_X1_X0/group.uid___A001_X1_X1/member.uid___A001_X1_X2"
)
EB = "uid___A002_Xaa_Xbb"
PREFIX = "member.uid___A001_X1_X2"


def _header_bytes(kind: str, obj: str = "NGC1", pixels: int = 128) -> bytes:
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = -32
    header["NAXIS"] = 4
    header["NAXIS1"] = pixels
    header["NAXIS2"] = pixels
    header["NAXIS3"] = 958 if kind == "cube" else 1
    header["NAXIS4"] = 1
    header["CDELT1"] = -1.0e-5
    header["CDELT2"] = 1.0e-5
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 20.0
    header["CRVAL3"] = 230.0e9
    header["CDELT3"] = 1.0e6
    header["OBJECT"] = obj
    for line in (
        "imsize      = [128, 128]",
        "cell        = ['0.036arcsec']",
        "phasecenter = 'ICRS 10:00:00.0 +020.00.00.000'",
        f"specmode    = '{kind}'",
        "weighting   = 'briggsbwtaper'",
        "robust      = 0.5",
        "niter       = 30000",
        "threshold   = '1.2mJy'",
    ):
        header.add_history(line)
    return header.tostring(padding=True).encode("ascii")


def _tree(tmp_path: Path, windows: tuple[int, ...] = (25,)) -> tuple[Path, Path]:
    """A delivery tree with one member, one MS and its images; (root, ms)."""
    root = tmp_path / "tree"
    member = root / MEMBER_FOLDER
    ms = member / "calibrated" / f"{EB}.ms.split.cal"
    ms.mkdir(parents=True)
    ms.with_name(ms.name + ".done").write_text("{}")
    product = member / "product"
    product.mkdir()
    for spw in windows:
        for kind in ("mfs", "cube"):
            name = f"{PREFIX}.NGC1_sci.spw{spw}.{kind}.I.pbcor.fits"
            (product / name).write_bytes(_header_bytes(kind) + b"\0" * 2880)
    (root / "ous_index.csv").write_text(
        f"eb_uid,member_ous_uid,member_folder\n{EB},{MEMBER},{MEMBER_FOLDER}\n"
    )
    return root, ms


def _fake_parameters(windows: tuple[int, ...]):
    def compute(ms: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "filename": [str(Path(ms).resolve())] * len(windows),
                "spectral_window_id": list(windows),
                "reference_frequency": [230.0e9] * len(windows),
                "fov_per_frequency": [10.0] * len(windows),
                "max_baseline_size": [500.0] * len(windows),
                "synthetized_beam_size": [1.0] * len(windows),
                "n_channels": [1920] * len(windows),
                "target_field_ids": ["3"] * len(windows),
                "field_name": ["NGC1"] * len(windows),
            }
        )

    return compute


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch):
    def patch(windows: tuple[int, ...] = (25,)) -> None:
        monkeypatch.setattr(image_headers, "compute_imaging_parameters", _fake_parameters(windows))
        monkeypatch.setattr(image_headers, "ms_phase_center", lambda ms, field: (150.0, 20.01))

    return patch


def test_fetch_then_parse_headers(tmp_path: Path, patched) -> None:
    root, ms = _tree(tmp_path)
    patched()
    headers = tmp_path / "headers"
    result = runner.invoke(cli.app, ["image", "fetch-headers", str(root), str(headers)])
    assert result.exit_code == 0, result.output
    manifest = pd.read_csv(headers / "headers.csv")
    assert sorted(manifest["image_kind"]) == ["cube", "mfs"]
    assert set(manifest["status"]) == {"ok"}
    saved = headers / "uid___A001_X1_X2" / f"{PREFIX}.NGC1_sci.spw25.cube.I.pbcor.fits.hdr"
    assert saved.stat().st_size % 2880 == 0
    assert image_headers.read_fits_header(saved)["OBJECT"] == "NGC1"
    assert len(image_headers.read_manifest(headers / "headers.csv")) == 2

    again = runner.invoke(cli.app, ["image", "fetch-headers", str(root), str(headers)])
    assert again.exit_code == 0 and "Nothing to do" in again.output

    out = tmp_path / "params.csv"
    parsed = runner.invoke(cli.app, ["image", "parse-headers", str(headers), str(out)])
    assert parsed.exit_code == 0, parsed.output
    row = pd.read_csv(out).iloc[0]
    assert row["image_kind"] == "cube" and row["image_n_channels"] == 958
    assert row["image_n_pixels"] == 128 and row["image_cell_arcsec"] == pytest.approx(0.036)
    assert row["image_weighting"] == "briggsbwtaper" and row["image_niter"] == 30000
    assert row["image_threshold_jy"] == pytest.approx(1.2e-3)
    assert row["phasecenter_ra_deg"] == pytest.approx(150.0)
    assert row["ms_phasecenter_dec_deg"] == pytest.approx(20.01)
    assert row["member_ous_uid"] == MEMBER
    # 10" at 0.125"/pixel is 80 pixels, widened to 32 beams of 8 pixels: 256.
    assert row["ms_n_pixels"] == 256
    assert row["image_check"] == "size 128 vs MS 256; cell 0.036 vs MS 0.125 asec"


def test_a_window_without_an_image_is_listed_and_the_rest_kept(tmp_path: Path, patched) -> None:
    root, ms = _tree(tmp_path)
    patched((25, 27))
    headers = tmp_path / "headers"
    result = runner.invoke(
        cli.app,
        ["image", "fetch-headers", str(root), str(headers), "--source", "local", "--kinds", "mfs"],
    )
    assert result.exit_code == 1
    assert list(pd.read_csv(headers / "headers.csv")["spectral_window_id"]) == [25]
    failed = (headers / "headers.csv.failed.tsv").read_text()
    assert "no tclean image for spw 27 field NGC1" in failed


def test_an_unreadable_ms_is_listed_or_stops_the_scan(tmp_path: Path, monkeypatch) -> None:
    root, ms = _tree(tmp_path)

    def broken(ms: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        raise RuntimeError("no table")

    monkeypatch.setattr(image_headers, "compute_imaging_parameters", broken)
    headers = tmp_path / "headers"
    result = runner.invoke(cli.app, ["image", "fetch-headers", str(root), str(headers)])
    assert result.exit_code == 1
    assert "RuntimeError: no table" in (headers / "headers.csv.failed.tsv").read_text()

    def stop(*args: object, **kwargs: object):
        raise RuntimeError("stop here")

    monkeypatch.setattr(cli.image, "collect_headers", stop)
    kept = runner.invoke(cli.app, ["image", "fetch-headers", str(root), str(headers)])
    assert (
        kept.exit_code == 1
        and "RuntimeError: stop here" in (headers / "headers.csv.failed.tsv").read_text()
    )
    fast = runner.invoke(
        cli.app, ["image", "fetch-headers", str(root), str(headers), "--fail-fast"]
    )
    assert isinstance(fast.exception, RuntimeError)


def test_fetch_headers_refuses_what_it_cannot_do(tmp_path: Path, patched) -> None:
    root, ms = _tree(tmp_path)
    patched()
    out = str(tmp_path / "headers")
    assert runner.invoke(cli.app, ["image", "fetch-headers", str(tmp_path / "none"), out]).exit_code
    for option, value in (("--source", "tape"), ("--kinds", "mfs,dirty")):
        result = runner.invoke(cli.app, ["image", "fetch-headers", str(root), out, option, value])
        assert result.exit_code == 2 and option in result.output
    (root / "ous_index.csv").unlink()
    lone = tmp_path / "lone"
    (lone / f"{EB}.ms.split.cal").mkdir(parents=True)
    result = runner.invoke(cli.app, ["image", "fetch-headers", str(lone), out])
    assert result.exit_code == 2 and "ous_index.csv" in result.output
    missing = runner.invoke(cli.app, ["image", "parse-headers", out, str(tmp_path / "p.csv")])
    assert missing.exit_code == 1 and "fetch-headers" in missing.output


def test_headers_come_from_the_archive_when_the_tree_has_none(
    tmp_path: Path, patched, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, ms = _tree(tmp_path)
    for image in (root / MEMBER_FOLDER / "product").iterdir():
        image.unlink()
    patched()
    name = f"{PREFIX}.NGC1_sci.spw25.cube.I.pbcor.fits"
    products = [
        SimpleNamespace(filename=name, access_url="https://archive.test/cube"),
        SimpleNamespace(filename=f"{PREFIX}.J1_ph.spw25.mfs.I.pbcor.fits", access_url="https://x"),
    ]
    from almasim.services.download import datalink

    asked: list[list[str]] = []
    monkeypatch.setattr(datalink, "resolve_products", lambda uids: asked.append(uids) or products)
    monkeypatch.setattr(datalink, "expand_archives", lambda found, layout: found)
    image_headers._archive_science_images.cache_clear()

    body = _header_bytes("cube") + b"\0" * 28800
    client = httpx.Client

    def stream_only(**kwargs: object) -> httpx.Client:
        return client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
        )

    monkeypatch.setattr(image_headers.httpx, "Client", stream_only)
    headers = tmp_path / "headers"
    result = runner.invoke(cli.app, ["image", "fetch-headers", str(root), str(headers)])
    assert result.exit_code == 0, result.output
    manifest = pd.read_csv(headers / "headers.csv")
    assert list(manifest["image_url"]) == ["https://archive.test/cube"]
    assert asked == [[MEMBER]]
    saved = image_headers.read_fits_header(manifest["header_path"][0])
    assert saved["NAXIS3"] == 958
    assert image_headers.read_remote_fits_header("https://archive.test/cube")["OBJECT"] == "NGC1"
    image_headers._archive_science_images.cache_clear()


def test_a_header_that_cannot_be_read_is_a_failed_row(tmp_path: Path, patched) -> None:
    root, ms = _tree(tmp_path)
    patched()
    cube = root / MEMBER_FOLDER / "product" / f"{PREFIX}.NGC1_sci.spw25.cube.I.pbcor.fits"
    cube.write_bytes(b" " * 2880)  # no END card
    with pytest.raises(ValueError, match="no END card"):
        image_headers.read_fits_header_bytes(cube)
    headers = tmp_path / "headers"
    assert (
        runner.invoke(cli.app, ["image", "fetch-headers", str(root), str(headers)]).exit_code == 0
    )
    manifest = pd.read_csv(headers / "headers.csv").set_index("image_kind")
    assert manifest.loc["cube", "status"] == "failed"
    assert "no END card" in manifest.loc["cube", "error"]
    out = tmp_path / "params.csv"
    assert runner.invoke(cli.app, ["image", "parse-headers", str(headers), str(out)]).exit_code == 0
    row = pd.read_csv(out).iloc[0]
    assert row["image_kind"] == "cube" and math.isnan(row["image_n_pixels"])


def test_small_readers() -> None:
    to_jy = image_headers._threshold_to_jy
    assert to_jy("0.000412Jy") == pytest.approx(4.12e-4)
    assert to_jy("3uJy") == pytest.approx(3e-6) and to_jy(0.5) == 0.5
    assert math.isnan(to_jy(None)) and math.isnan(to_jy("Jy"))
    assert image_headers.parse_phasecenter("J2000 12:00:00 -30:30:00") == (180.0, -30.5)
    assert image_headers.parse_phasecenter("12:00:00, -30.5") == (180.0, -30.5)
    with pytest.raises(ValueError, match="right ascension"):
        image_headers.parse_phasecenter("180.0 -30.5")
    with pytest.raises(ValueError, match="phase centre"):
        image_headers.parse_phasecenter("ICRS")
    header = fits.Header()
    header["CRVAL1"], header["CRVAL2"] = 10.0, -5.0
    assert image_headers._phasecenter_deg({"phasecenter": "ICRS 1h2m3s +4d5m6s"}, header) == (
        10.0,
        -5.0,
    )
    assert image_headers._phasecenter_deg({}, fits.Header()) == (None, None)
    assert image_headers.ms_phase_center("/no/such.ms", "x") == (None, None)
    image = image_headers.MemberImage("a.fits", "mfs")
    assert image.source == ""
    with pytest.raises(ValueError, match="no local path"):
        image.read_header_bytes()
    with pytest.raises(ValueError, match="source must be"):
        image_headers._images_for("tape", None, "", "", "", 1, ("mfs",), (), ())
