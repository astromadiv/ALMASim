"""``image fetch-headers`` and ``image parse-headers`` on a small delivery tree:
a header and a parameter CSV per delivered image, in its member's folder."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest
import typer
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
        f"niter       = {30000 if kind == 'cube' else 9000}",
        f"threshold   = '{'1.2mJy' if kind == 'cube' else '0.04mJy'}'",
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


HEADERS = Path(MEMBER_FOLDER) / "images" / "headers"
CUBE = f"{PREFIX}.NGC1_sci.spw25.cube.I.pbcor.fits"
MFS = f"{PREFIX}.NGC1_sci.spw25.mfs.I.pbcor.fits"


def test_each_image_gets_its_header_and_parameter_csv_in_its_member(
    tmp_path: Path, patched
) -> None:
    root, ms = _tree(tmp_path)
    patched()
    result = runner.invoke(cli.app, ["image", "fetch-headers", str(root)])
    assert result.exit_code == 0, result.output
    folder = root / HEADERS
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        f"{name}{suffix}" for name in (CUBE, MFS) for suffix in (".csv", ".hdr")
    )
    saved = folder / f"{CUBE}.hdr"
    assert saved.stat().st_size % 2880 == 0
    assert image_headers.read_fits_header(saved)["OBJECT"] == "NGC1"
    assert pd.read_csv(folder / f"{MFS}.csv")["image_kind"].tolist() == ["mfs"]

    row = pd.read_csv(folder / f"{CUBE}.csv").iloc[0]
    assert row["status"] == "ok" and row["header_path"] == str(saved)
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

    again = runner.invoke(cli.app, ["image", "fetch-headers", str(root)])
    assert again.exit_code == 0 and "Nothing to do" in again.output

    # image-from-ms reads the tree, or one member: the cube's row of each window.
    for place in (root, root / MEMBER_FOLDER):
        rows = cli.image._read_parameters(place)
        assert rows["image_kind"].tolist() == ["cube"]
        # ... with the clean limits of the window's mfs image beside the cube's.
        assert rows["image_niter"].tolist() == [30000]
        assert rows["continuum_niter"].tolist() == [9000]
        assert rows["continuum_threshold_jy"].tolist() == [pytest.approx(4.0e-5)]
        assert rows["continuum_weighting"].tolist() == ["briggsbwtaper"]
    assert len(cli.image._read_parameters(folder / f"{MFS}.csv")) == 1


def test_parse_headers_rewrites_the_csvs_from_the_saved_headers(tmp_path: Path, patched) -> None:
    root, ms = _tree(tmp_path)
    patched()
    assert runner.invoke(cli.app, ["image", "fetch-headers", str(root)]).exit_code == 0
    csv = root / HEADERS / f"{CUBE}.csv"
    spoiled = pd.read_csv(csv)
    spoiled["image_n_pixels"] = 7
    spoiled.to_csv(csv, index=False)
    parsed = runner.invoke(
        cli.app, ["image", "parse-headers", str(root), "--min-beams-across", "0"]
    )
    assert parsed.exit_code == 0, parsed.output
    assert "2 image(s), 2 row(s)" in parsed.output
    row = pd.read_csv(csv).iloc[0]
    # 10" x 1.5 at 0.125"/pixel, without the 32-beam floor.
    assert row["image_n_pixels"] == 128 and row["ms_n_pixels"] == 120


def test_an_output_folder_holds_a_sub_folder_per_member(tmp_path: Path, patched) -> None:
    root, ms = _tree(tmp_path)
    patched()
    out = tmp_path / "headers"
    result = runner.invoke(cli.app, ["image", "fetch-headers", str(root), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "uid___A001_X1_X2" / f"{CUBE}.hdr").is_file()
    assert not (root / MEMBER_FOLDER / "images").exists()
    assert cli.image._read_parameters(out)["image_kind"].tolist() == ["cube"]
    assert runner.invoke(cli.app, ["image", "parse-headers", str(out)]).exit_code == 0


def test_an_image_of_two_execution_blocks_has_a_row_of_each(tmp_path: Path, patched) -> None:
    root, ms = _tree(tmp_path)
    second = ms.with_name("uid___A002_Xaa_Xcc.ms.split.cal")
    second.mkdir()
    second.with_name(second.name + ".done").write_text("{}")
    with (root / "ous_index.csv").open("a") as index:
        index.write(f"uid___A002_Xaa_Xcc,{MEMBER},{MEMBER_FOLDER}\n")
    patched()
    assert runner.invoke(cli.app, ["image", "fetch-headers", str(root)]).exit_code == 0
    rows = pd.read_csv(root / HEADERS / f"{CUBE}.csv")
    assert sorted(Path(name).name for name in rows["filename"]) == [ms.name, second.name]
    # A rerun of one MS replaces its row and keeps the other's.
    again = runner.invoke(cli.app, ["image", "fetch-headers", str(root), "--no-resume"])
    assert again.exit_code == 0
    assert len(pd.read_csv(root / HEADERS / f"{CUBE}.csv")) == 2
    assert len(cli.image._read_parameters(root)) == 2


def test_a_window_without_an_image_is_listed_and_the_rest_kept(tmp_path: Path, patched) -> None:
    root, ms = _tree(tmp_path)
    patched((25, 27))
    result = runner.invoke(
        cli.app,
        ["image", "fetch-headers", str(root), "--source", "local", "--kinds", "mfs"],
    )
    assert result.exit_code == 1
    assert list(pd.read_csv(root / HEADERS / f"{MFS}.csv")["spectral_window_id"]) == [25]
    assert not (root / HEADERS / f"{CUBE}.csv").exists()
    failed = (root / "headers.failed.tsv").read_text()
    assert "no tclean image for spw 27 field NGC1" in failed


def test_an_unreadable_ms_is_listed_or_stops_the_scan(tmp_path: Path, monkeypatch) -> None:
    root, ms = _tree(tmp_path)

    def broken(ms: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        raise RuntimeError("no table")

    monkeypatch.setattr(image_headers, "compute_imaging_parameters", broken)
    result = runner.invoke(cli.app, ["image", "fetch-headers", str(root)])
    assert result.exit_code == 1
    assert "RuntimeError: no table" in (root / "headers.failed.tsv").read_text()

    def stop(*args: object, **kwargs: object):
        raise RuntimeError("stop here")

    monkeypatch.setattr(cli.image, "collect_headers", stop)
    kept = runner.invoke(cli.app, ["image", "fetch-headers", str(root)])
    assert (
        kept.exit_code == 1
        and "RuntimeError: stop here" in (root / "headers.failed.tsv").read_text()
    )
    fast = runner.invoke(cli.app, ["image", "fetch-headers", str(root), "--fail-fast"])
    assert isinstance(fast.exception, RuntimeError)


def test_fetch_headers_refuses_what_it_cannot_do(tmp_path: Path, patched) -> None:
    root, ms = _tree(tmp_path)
    patched()
    assert runner.invoke(cli.app, ["image", "fetch-headers", str(tmp_path / "none")]).exit_code
    for option, value in (("--source", "tape"), ("--kinds", "mfs,dirty")):
        result = runner.invoke(cli.app, ["image", "fetch-headers", str(root), option, value])
        assert result.exit_code == 2  # the message is styled on a colour terminal
    (root / "ous_index.csv").unlink()
    lone = tmp_path / "lone"
    (lone / f"{EB}.ms.split.cal").mkdir(parents=True)
    result = runner.invoke(cli.app, ["image", "fetch-headers", str(lone)])
    assert result.exit_code == 2
    missing = runner.invoke(cli.app, ["image", "parse-headers", str(lone)])
    assert missing.exit_code == 1
    with pytest.raises(ValueError, match="needs an output folder"):
        image_headers.headers_folder("", MEMBER, delivery_root=None)
    with pytest.raises(typer.BadParameter, match="no per-image parameter CSV"):
        cli.image._read_parameters(lone)


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
    result = runner.invoke(cli.app, ["image", "fetch-headers", str(root)])
    assert result.exit_code == 0, result.output
    manifest = pd.read_csv(root / HEADERS / f"{CUBE}.csv")
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
    assert runner.invoke(cli.app, ["image", "fetch-headers", str(root)]).exit_code == 0
    failed = pd.read_csv(root / HEADERS / f"{CUBE}.csv").iloc[0]
    assert failed["status"] == "failed" and "no END card" in failed["error"]
    assert runner.invoke(cli.app, ["image", "parse-headers", str(root)]).exit_code == 0
    # The cube's row stands for the window, with the MS-derived grid.
    row = cli.image._read_parameters(root).iloc[0]
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
