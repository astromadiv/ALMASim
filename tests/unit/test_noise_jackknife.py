"""The jackknifed continuum noise of an imaged window (MADIV-28): the command
it reuses, the sign flip, and the measurement (extraction and WSClean stubbed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from almasim.services.noise import jackknife as jk

from ._fakes import FakeTable

CONTINUUM = (
    "/opt/wsclean -name {task}/wsclean -j 10 -scale 0.68asec -size 256 256 -mgain 0.85 "
    "-niter 100000 -pol I -make-psf -weight briggs 0.5 -auto-mask 3.0 -auto-threshold 1.0 "
    "-abs-mem 36 -temp-dir {task} -update-model-required -field 5 -no-reorder "
    "-data-column CONTINUUM_DATA {task}/spw10.single.ms"
)


def _log(task: Path) -> None:
    """An imaging log as image-from-ms writes it: self-calibration's model
    image, both continuum passes and the cube."""
    lines = [
        "# extracted spectral window 10 fields [5] into spw10.single.ms: 5850 rows",
        f"# /opt/wsclean -name {task}/selfcal-0 -niter 100 -field 5 {task}/spw10.single.ms",
        "# " + CONTINUUM.format(task=task),
        "# " + CONTINUUM.format(task=task).replace(" -data-column", " -continue -data-column"),
        f"# /opt/wsclean -name {task}/wsclean-cube -channels-out 128 {task}/spw10.single.ms",
        "WSClean version 3.7 ...",
    ]
    task.with_name(task.name + ".log").write_text("\n".join(lines) + "\n")


@pytest.mark.unit
def test_the_continuum_command_is_the_first_pass_the_log_records(tmp_path):
    task = tmp_path / "SPW-10_FIELD-5"
    assert jk.continuum_command(task) is None  # no log
    _log(task)
    command = jk.continuum_command(task)
    assert command == CONTINUUM.format(task=task).split()


@pytest.mark.unit
def test_the_jackknife_image_is_the_continuum_dirty_of_the_flipped_copy(tmp_path):
    task = tmp_path / "SPW-10_FIELD-5"
    work = tmp_path / "jk"
    argv = jk.jackknife_command(
        CONTINUUM.format(task=task).split(), work, work / "jackknife.ms", "DATA", threads=4
    )
    joined = " ".join(argv)
    assert argv[0] == "/opt/wsclean" and argv[-1] == str(work / "jackknife.ms")
    for kept in ("-scale 0.68asec", "-size 256 256", "-weight briggs 0.5", "-field 5"):
        assert kept in joined  # the image's own grid, weighting and field
    for gone in ("-auto-mask", "-auto-threshold", "-make-psf", "-update-model-required"):
        assert gone not in argv
    assert f"-name {work}/jackknife" in joined and "-niter 0" in joined
    assert "-data-column DATA" in joined and f"-temp-dir {work}" in joined and "-j 4" in joined
    other = jk.jackknife_command(
        CONTINUUM.format(task=task).split(), work, work / "j.ms", "DATA", threads=1,
        wsclean_bin="/usr/bin/wsclean",
    )  # fmt: skip
    assert other[0] == "/usr/bin/wsclean"


@pytest.mark.unit
def test_signs_alternate_along_each_baselines_integrations():
    # Two baselines, rows out of time order and interleaved.
    a1 = np.array([0, 0, 1, 0, 1, 0, 1])
    a2 = np.array([1, 1, 2, 1, 2, 1, 2])
    t = np.array([3.0, 1.0, 1.0, 2.0, 2.0, 4.0, 3.0])
    signs = jk.alternating_signs(a1, a2, t)
    for baseline in ((0, 1), (1, 2)):
        rows = np.flatnonzero((a1 == baseline[0]) & (a2 == baseline[1]))
        in_time = signs[rows[np.argsort(t[rows])]]
        assert list(in_time) == [1.0, -1.0, 1.0, -1.0][: rows.size]


@pytest.mark.unit
def test_the_flip_cancels_a_constant_sky_and_keeps_the_noise():
    rng = np.random.default_rng(0)
    n_bl, n_t, nchan = 45, 40, 16
    a1 = np.repeat(np.arange(n_bl), n_t)
    a2 = a1 + 1
    t = np.tile(np.arange(n_t, dtype=float), n_bl)
    sky = 50.0  # constant on every baseline
    noise = rng.normal(size=(a1.size, nchan, 2)) + 1j * rng.normal(size=(a1.size, nchan, 2))
    table = FakeTable(
        {"ANTENNA1": a1, "ANTENNA2": a2, "TIME": t,
         "DATA": (sky + noise).astype(np.complex64)}
    )  # fmt: skip
    result = jk.flip_alternate_integrations("x.ms", opener=lambda *a, **k: table)
    assert result == {"column": "DATA", "rows": a1.size}
    flipped = table.columns["DATA"]
    # Summed over each baseline's (even number of) integrations, the sky cancels
    # and the noise adds up as it would unflipped: sqrt(n_t) times one row's.
    sums = flipped.reshape(n_bl, n_t, nchan, 2).sum(axis=1)
    assert np.abs(sums.mean()) < 1.0  # the sky (50 per row, 2000 per baseline) cancels
    assert np.std(sums.real) == pytest.approx(np.sqrt(n_t) * np.std(noise.real), rel=0.05)


class StrictTable(FakeTable):
    """casacore's putcol: without a row count it writes every row from startrow on."""

    def putcol(self, name, value, startrow=0, nrow=-1):
        rows = self.nrows() - startrow if nrow < 0 else nrow
        if len(value) != rows:
            raise RuntimeError(f"column {name} has {rows}, array has {len(value)} rows")
        super().putcol(name, value, startrow, nrow)


@pytest.mark.unit
def test_the_flip_writes_a_long_ms_chunk_by_chunk(monkeypatch):
    monkeypatch.setattr(jk, "CHUNK_ROWS", 3)  # 8 rows: chunks of 3, 3 and 2
    a = np.zeros(8, dtype=int)
    table = StrictTable(
        {"ANTENNA1": a, "ANTENNA2": a + 1, "TIME": np.arange(8.0),
         "DATA": np.ones((8, 2, 2), np.complex64)}
    )  # fmt: skip
    jk.flip_alternate_integrations("x.ms", opener=lambda *a, **k: table)
    assert list(table.columns["DATA"][:, 0, 0].real) == [1, -1] * 4


@pytest.mark.unit
def test_the_corrected_column_is_flipped_and_read_when_present():
    a = np.zeros(4, dtype=int)
    table = FakeTable(
        {"ANTENNA1": a, "ANTENNA2": a + 1, "TIME": np.arange(4.0),
         "DATA": np.ones((4, 2, 2), np.complex64),
         "CORRECTED_DATA": np.ones((4, 2, 2), np.complex64)}
    )  # fmt: skip
    result = jk.flip_alternate_integrations("x.ms", opener=lambda *a, **k: table)
    assert result["column"] == "CORRECTED_DATA"
    for column in ("DATA", "CORRECTED_DATA"):
        assert list(table.columns[column][:, 0, 0].real) == [1, -1, 1, -1]


@pytest.mark.unit
def test_jackknife_noise_measures_the_inner_image(monkeypatch, tmp_path):
    task = tmp_path / "SPW-10_FIELD-5"
    work = tmp_path / "jk"
    assert jk.jackknife_noise(task, "/d/x.ms", 10, [5], work) == {}  # no log, no command
    _log(task)
    calls: dict[str, object] = {}

    def extract(ms, spw, fields, out, **kwargs):
        calls["extract"] = (ms, spw, fields, kwargs["max_weight"])
        Path(out).mkdir(parents=True)
        return {"rows": 10}

    def flip(ms, **kwargs):
        calls["flip"] = ms
        return {"column": "DATA", "rows": 10}

    class Done:
        returncode, stdout, stderr = 0, "", ""

    def run(argv, **kwargs):
        calls["argv"] = argv
        rng = np.random.default_rng(1)
        plane = rng.normal(0, 2e-3, (64, 64))
        plane[:4] = 1.0  # the image's edge, outside the inner part measured
        fits.PrimaryHDU(plane[None, None].astype(np.float32)).writeto(work / "jackknife-dirty.fits")
        return Done()

    monkeypatch.setattr(jk, "extract_single_window_ms_subprocess", extract)
    monkeypatch.setattr(jk, "flip_in_child", flip)
    monkeypatch.setattr(jk.subprocess, "run", run)
    noise = jk.jackknife_noise(task, "/d/x.ms", 10, [5], work, threads=2)
    assert noise["jk.sigma"] == pytest.approx(2e-3, rel=0.1)
    assert calls["extract"] == ("/d/x.ms", 10, [5], 1e4)  # rows over WEIGHT 1e4 left out
    assert calls["flip"] == work / "jackknife.ms"
    assert calls["argv"][-1] == str(work / "jackknife.ms")
    assert not list(work.glob("jackknife*"))  # the copy and the image are removed

    class Failed:
        returncode, stdout, stderr = 1, "", "wsclean: no data"

    monkeypatch.setattr(jk.subprocess, "run", lambda argv, **kw: Failed())
    with pytest.raises(RuntimeError, match="no data"):
        jk.jackknife_noise(task, "/d/x.ms", 10, [5], work)
