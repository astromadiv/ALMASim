"""API v2: a simulation's size before it runs (web console plan, U3.6)."""

from __future__ import annotations

import csv

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from almasim.jobs.profile import Profile
from almasim.web.app import COOKIE, create_app

pytestmark = pytest.mark.unit

TOKEN = "simulation-test-token"

# One row of `almasim metadata` output (public ALMA metadata, abstract cut).
ROW = {
    "ALMA_source_name": "Cloverleaf",
    "Band": "4",
    "Array_type": "12m",
    "antenna_arrays": "A001:DV12 A002:DA45 A003:DV07 A004:DA41 A005:DV05 A006:DA59 "
    "A007:DV20 A008:DV17 A009:DV09 A010:DV06 A011:DV01 A013:DV18 A015:DA52",
    "Ang.res.": "2.2268121773845264",
    "Obs.date": "2025-03-18",
    "Project_abstract": "The progenitors of today's early-type ma",
    "science_keyword": "Galaxy structure & evolution;null",
    "scientific_category": "Galaxy evolution",
    "QA2_status": "T",
    "Type": "S",
    "PWV": "4.8605385",
    "SB_name": "Cloverle_a_04_TM1",
    "Vel.res.": "16865.72790867241",
    "RA": "213.94266666669853",
    "Dec": "11.495466666668452",
    "FOV": "0.012171129238001002",
    "Int.Time": "2872.8",
    "Cont_sens_mJybeam": "0.013916438931011077",
    "Line_sens_10kms_mJybeam": "0.586731828389117",
    "Bandwidth": "1875000000.0",
    "Freq": "126.01075860894431",
    "Freq.sup.": "[125.08..126.94GHz,7812.01kHz,586.7uJy/beam@10km/s,27.8uJy/beam@native, XX YY]",
    "proposal_id": "2023.1.00133.S",
    "member_ous_uid": "uid://A001/X3660/Xe",
    "group_ous_uid": "uid://A001/X3660/Xd",
}


@pytest.fixture
def area(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    with (data / "meta.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ROW))
        writer.writeheader()
        writer.writerow(ROW)
    return tmp_path


@pytest.fixture
def client(area):
    profile = Profile(runs_root=area / "runs", roots=(area / "data",))
    app = create_app(TOKEN, static_dir=area / "none", profile=profile, events_interval=60)
    c = TestClient(app)
    c.cookies.set(COOKIE, TOKEN)
    return c


def test_the_cube_of_a_metadata_row(client, area):
    path = str(area / "data" / "meta.csv")
    got = client.post(
        "/api/v2/simulation/estimate",
        json={"metadata_path": path, "n_pix": 64, "n_channels": 32},
    )
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["source_name"] == "Cloverleaf" and body["band"] == "4"
    assert body["cube_shape"] == [32, 64, 64]
    expected_gb = 32 * 64 * 64 * 4 / 1024**3
    assert body["raw_single_cube_gb"] == pytest.approx(expected_gb)
    assert body["beam_size_arcsec"] > 0 and body["cell_size_arcsec"] > 0
    # Left to the row, the size comes from the observation.
    auto = client.post("/api/v2/simulation/estimate", json={"metadata_path": path}).json()
    assert auto["n_pix"] > 0 and auto["n_channels"] > 0


def test_a_missing_row_or_file_is_an_error(client, area):
    path = str(area / "data" / "meta.csv")
    missing_row = client.post(
        "/api/v2/simulation/estimate", json={"metadata_path": path, "row_idx": 5}
    )
    assert missing_row.status_code == 400 and "no row 5" in missing_row.json()["detail"]
    outside = client.post("/api/v2/simulation/estimate", json={"metadata_path": "/etc/passwd"})
    assert outside.status_code == 403
    nothing = client.post(
        "/api/v2/simulation/estimate", json={"metadata_path": str(area / "data" / "nope.csv")}
    )
    assert nothing.status_code == 404


EB = "uid___A002_X11a1c8d_X12c8e"  # an execution block of the Cloverleaf observation


def _model_cube(path, *, centre_hz=126e9):
    import numpy as np
    from astropy.io import fits

    header = fits.Header()
    header.update(
        CTYPE1="RA---SIN", CDELT1=-0.4 / 3600, CTYPE2="DEC--SIN", CDELT2=0.4 / 3600,
        CTYPE3="FREQ", CRPIX3=1.0, CRVAL3=centre_hz - 1.5 * 4e6, CDELT3=4e6, CTYPE4="STOKES",
    )  # fmt: skip
    path.parent.mkdir(parents=True)
    fits.PrimaryHDU(np.zeros((1, 4, 24, 24), dtype=np.float32), header=header).writeto(path)
    return path


@pytest.fixture
def imaging_run(area):
    """Two windows of the Cloverleaf observation, and a products CSV that says so."""
    run = area / "data" / "cubes"
    for spw in ("SPW-25_FIELD-2", "SPW-27_FIELD-2"):
        _model_cube(run / f"{EB}.ms.split" / spw / "wsclean-cube-model.fits")
    with (area / "data" / "resolved_products.csv").open("w", newline="") as f:
        csv.writer(f).writerows(
            [["uid", "filename"], [ROW["member_ous_uid"], f"2023.1.00133.S_{EB}.asdm.sdm.tar"]]
        )
    return run


def test_a_folder_of_cubes_lists_each_window_with_its_row(client, area, imaging_run):
    got = client.post(
        "/api/v2/simulation/cubes",
        json={"metadata_path": str(area / "data" / "meta.csv"), "input_path": str(imaging_run)},
    )
    assert got.status_code == 200, got.text
    body = got.json()
    assert [(c["label"], c["row_idx"], c["source_name"]) for c in body["cubes"]] == [
        (f"{EB}_SPW-25_FIELD-2", 0, "Cloverleaf"),
        (f"{EB}_SPW-27_FIELD-2", 0, "Cloverleaf"),
    ]
    cube = body["cubes"][0]
    assert (cube["n_pix_x"], cube["n_pix_y"], cube["n_channels"]) == (24, 24, 4)
    assert cube["central_freq_ghz"] == pytest.approx(126.0)
    assert cube["channel_width_mhz"] == pytest.approx(4.0)
    assert cube["cell_arcsec"] == pytest.approx(0.4)
    assert body["unmatched"] == [] and body["products"][0].endswith("resolved_products.csv")


def test_an_input_cube_sizes_the_estimate(client, area, imaging_run):
    cube = next(imaging_run.rglob("wsclean-cube-model.fits"))
    got = client.post(
        "/api/v2/simulation/estimate",
        json={
            "metadata_path": str(area / "data" / "meta.csv"),
            "input_path": str(cube),
            "n_pix": 512,  # ignored: the cube sets its own grid
        },
    )
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["cube_shape"] == [4, 24, 24]
    assert body["cell_size_arcsec"] == pytest.approx(0.4)
