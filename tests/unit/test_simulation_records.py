"""The simulation record types (MADIV-28 T5.8)."""

from unittest.mock import patch

import numpy as np

from almasim.services.astro.spectral import SpectralResult
from almasim.services.simulation.metadata import SimParamsRecord
from almasim.skymodels import serendipitous
from almasim.skymodels.serendipitous import SerendipitousConfig


def _record(path):
    return SimParamsRecord(
        path=str(path),
        source_name="J1234+5678",
        member_ouid="uid://A001/X123/X456",
        ra=188.5,
        dec=-5.2,
        ang_res=0.1,
        vel_res=10.0,
        int_time=3600.0,
        band=6.0,
        band_range=1.875,
        central_freq=250.0,
        redshift=0.5,
        line_fluxes=np.array([1.0, 0.5]),
        line_names=["CO(3-2)", "CII"],
        line_frequencies=np.array([250.0, 350.0]),
        continuum=np.array([0.1, 0.2, 0.15]),
        fov=10.0,
        beam_size=0.05,
        cell_size=0.01,
        n_pix=64,
        n_channels=32,
        snapshot=None,
        subhalo=None,
        lum_infrared=1e10,
        fwhm_z=np.array([2.0, 3.0]),
        source_type="gaussian",
        fwhm_x=3.0,
        fwhm_y=4.0,
        angle=30.0,
    )


def test_sim_params_record_writes_its_file(tmp_path):
    path = tmp_path / "sim_params_0.txt"
    record = _record(path)

    assert record.write() == str(path)
    text = path.read_text()
    assert "J1234+5678" in text
    assert "CO(3-2)" in text


def test_sim_params_record_passes_every_field():
    record = _record("p.txt")
    with patch("almasim.services.simulation.metadata.write_sim_parameters") as write:
        record.write()
    kwargs = write.call_args.kwargs
    assert kwargs["path"] == "p.txt"
    assert kwargs["continum"] is record.continuum
    assert kwargs["angle"] == 30.0
    assert "continuum" not in kwargs


def test_serendipitous_config_insert_forwards_arguments():
    config = SerendipitousConfig(
        terminal=None,
        client=None,
        update_progress=None,
        datacube="cube",
        continum=np.zeros(4),
        cont_sens=0.1,
        line_fluxes=np.ones(1),
        line_names=["CO"],
        line_frequencies=np.ones(1),
        freq_sup=0.1,
        pos_zs=[2],
        fwhm_x=3,
        fwhm_y=4,
        fwhm_zs=[2.0],
        n_px=16,
        n_chan=4,
    )
    with patch.object(serendipitous, "insert_serendipitous", return_value="out") as insert:
        assert config.insert() == "out"
    kwargs = insert.call_args.kwargs
    assert kwargs["datacube"] == "cube"
    assert kwargs["n_chan"] == 4
    assert kwargs["sim_params_path"] is None


def test_spectral_result_is_a_tuple_with_names():
    result = SpectralResult(*range(12))
    assert result.continuum == 0
    assert result.lum_infrared == 11
    assert tuple(result) == tuple(range(12))


def test_interferometer_run_gives_the_interferometer_keywords():
    import astropy.units as U

    from almasim.services.simulation.planning import InterferometerRun

    run = InterferometerRun(
        idx=0, skymodel=np.zeros((2, 4, 4)), main_dir="/m", output_dir="/o", ra=1 * U.deg,
        dec=2 * U.deg, central_freq=100 * U.GHz, bandwidth=1 * U.GHz, fov=10.0,
        antenna_array="A001 A002", noise=0.1, snr=5.0, integration_time=1.0,
        observation_date="2024-01-01", header=None, save_mode="npz", persist=False,
        antenna_diameter_m=7.0, config_name="7m", array_type="7m",
        imaging_algorithm="legacy", spectral_window=None,
    )  # fmt: skip
    kwargs = run.interferometer_kwargs()
    assert kwargs["antenna_diameter_m"] == 7.0 and kwargs["config_name"] == "7m"
    assert len(kwargs) == 24
    assert kwargs["sample_time_s"] is None and kwargs["execution"] is None
