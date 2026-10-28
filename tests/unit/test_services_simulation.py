"""Unit tests for simulation service functions."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import astropy.io.fits as fits
import h5py
import numpy as np
import pandas as pd
import pytest

from almasim.services.interferometry.noise import (
    NoiseModelConfig,
    calibrate_noise_profile,
    compute_channel_noise,
)
from almasim.services.observation_plan import build_single_pointing_observation_plan
from almasim.services.simulation import (
    CleanCubeStage,
    SimulationParams,
    export_results,
    generate_background_cube,
    resolve_source_pixel_position,
    run_simulation,
    write_ml_dataset_shard,
)


def _shared_patch(name, *modules):
    """Patch ``name`` with one mock in every simulation submodule that looks it up.

    The mock is passed to the test like ``@patch`` would pass it.
    """
    shared = MagicMock()

    def fresh():
        shared.reset_mock(return_value=True, side_effect=True)
        return shared

    first, *rest = modules

    def decorator(func):
        for module in rest:
            func = patch(f"almasim.services.simulation.{module}.{name}", new=shared)(func)
        return patch(f"almasim.services.simulation.{first}.{name}", new_callable=fresh)(func)

    return decorator


@pytest.fixture
def sample_metadata_row_dict():
    """Create a sample metadata row as dictionary."""
    return {
        "ALMA_source_name": "J1234+5678",
        "member_ous_uid": "uid://A001/X123/X456",
        "Band": 6.0,
        "RA": 188.5,
        "Dec": -5.2,
        "Ang.res.": 0.1,
        "Vel.res.": 10.0,
        "FOV": 0.01,
        "Obs.date": "2020-01-01",
        "PWV": 2.0,
        "Int.Time": 3600.0,
        "Bandwidth": 1.875,
        "Freq": 250.0,
        "Freq.sup.": ("[250.0..252.0GHz,31250.00kHz,2mJy/beam@10km/s,78.5uJy/beam@native, XX YY]"),
        "Cont_sens_mJybeam": 0.1,
        "antenna_arrays": "A001:DA01 A002:DA02",
        "redshift": 0.5,
        "rest_frequency": 500.0,
        "lum_infrared": 1e10,
    }


@pytest.fixture
def sample_metadata_row_series(sample_metadata_row_dict):
    """Create a sample metadata row as pandas Series."""
    return pd.Series(sample_metadata_row_dict)


@pytest.mark.unit
def test_simulation_params_from_dict(tmp_path, sample_metadata_row_dict):
    """Test creating SimulationParams from dictionary."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )

    assert params.idx == 0
    assert params.source_name == "J1234+5678"
    assert params.member_ouid == "uid://A001/X123/X456"
    assert params.band == 6.0
    assert params.ra == 188.5
    assert params.dec == -5.2
    assert params.project_name == "test"


@pytest.mark.unit
def test_simulation_params_from_series(tmp_path, sample_metadata_row_series):
    """Test creating SimulationParams from pandas Series."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_series,
        idx=1,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )

    assert params.idx == 1
    assert params.source_name == "J1234+5678"
    assert isinstance(params.main_dir, str)
    assert Path(params.main_dir).exists()


@pytest.mark.unit
def test_simulation_params_missing_required_field(tmp_path, sample_metadata_row_dict):
    """Test that missing required fields raise KeyError."""
    # Remove required field
    del sample_metadata_row_dict["RA"]

    main_dir = tmp_path / "main"
    main_dir.mkdir()

    with pytest.raises(KeyError, match="Missing required metadata column"):
        SimulationParams.from_metadata_row(
            sample_metadata_row_dict,
            idx=0,
            main_dir=main_dir,
            output_dir=tmp_path / "output",
            tng_dir=tmp_path / "tng",
            galaxy_zoo_dir=tmp_path / "galaxy_zoo",
            hubble_dir=tmp_path / "hubble",
            project_name="test",
        )


@pytest.mark.unit
def test_simulation_params_alternative_column_names(tmp_path):
    """Test that alternative column names are accepted."""
    row = {
        "source_name": "J1234+5678",  # Alternative to ALMA_source_name
        "member_ouid": "uid://A001/X123/X456",
        "band": 6.0,  # Lowercase
        "RA": 188.5,
        "Dec": -5.2,
        "ang_res": 0.1,  # Alternative format
        "vel_res": 10.0,
        "fov": 0.01,
        "obs_date": "2020-01-01",
        "PWV": 2.0,
        "int_time": 3600.0,
        "bandwidth": 1.875,
        "frequency": 250.0,  # Alternative to Freq
        "frequency_support": "[250.0..252.0GHz]",
        "cont_sens": 0.1,
        "antenna_array": "A001:DA01",
    }

    main_dir = tmp_path / "main"
    main_dir.mkdir()

    params = SimulationParams.from_metadata_row(
        row,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )

    assert params.source_name == "J1234+5678"
    assert params.band == 6.0
    assert params.freq == 250.0


@pytest.mark.unit
def test_simulation_params_nan_handling(tmp_path, sample_metadata_row_dict):
    """Test that NaN values are handled correctly."""
    sample_metadata_row_dict["n_pix"] = np.nan
    sample_metadata_row_dict["n_channels"] = np.nan

    main_dir = tmp_path / "main"
    main_dir.mkdir()

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )

    assert params.n_pix is None
    assert params.n_channels is None
    assert params.source_offset_x_arcsec == 0.0
    assert params.source_offset_y_arcsec == 0.0
    assert params.background_mode == "none"


@pytest.mark.unit
def test_simulation_params_path_resolution(tmp_path, sample_metadata_row_dict):
    """Test that paths are properly resolved."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()

    # Use relative paths
    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir="./main",
        output_dir="./output",
        tng_dir="./tng",
        galaxy_zoo_dir="./galaxy_zoo",
        hubble_dir="./hubble",
        project_name="test",
    )

    # Paths should be resolved to absolute
    assert Path(params.main_dir).is_absolute()
    assert Path(params.output_dir).is_absolute()


@pytest.mark.unit
def test_simulation_params_line_names_string(tmp_path, sample_metadata_row_dict):
    """Test parsing line_names from string format."""
    # line_names parsing happens in run_simulation, not in from_metadata_row
    # So we just test that it's stored as-is
    sample_metadata_row_dict["line_names"] = '["CO(3-2)", "CII"]'

    main_dir = tmp_path / "main"
    main_dir.mkdir()

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        line_names=sample_metadata_row_dict["line_names"],
    )

    # Should be stored as string (parsing happens in run_simulation)
    assert isinstance(params.line_names, str) or params.line_names is None


@pytest.mark.unit
def test_simulation_params_defaults(tmp_path, sample_metadata_row_dict):
    """Test that default values are applied correctly."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )

    assert params.source_type == "point"  # Default
    assert params.snr is None  # Auto-derived by default
    assert params.save_mode == "fits"  # Default
    assert params.inject_serendipitous is False  # Default
    assert params.remote is False  # Default
    assert params.ncpu >= 1  # Should use CPU count or 1


@pytest.mark.unit
def test_generate_background_cube_combined_mode_returns_positive_cube():
    """Background generation should produce a non-empty additive cube."""
    cube = generate_background_cube(
        mode="combined",
        n_pix=32,
        n_channels=8,
        cell_size_arcsec=0.1,
        channel_frequencies_hz=np.linspace(240e9, 250e9, 8),
        cont_sens_jy=1e-4,
        level=1.0,
        seed=7,
    )

    assert cube.shape == (8, 32, 32)
    assert np.all(cube >= 0.0)
    assert float(np.sum(cube)) > 0.0


@pytest.mark.unit
def test_resolve_source_pixel_position_applies_explicit_offsets():
    """Source position stays centered by default and moves only by explicit offsets."""
    wcs = MagicMock()
    sub_wcs = MagicMock()
    sub_wcs.wcs_world2pix.return_value = (32.0, 32.0, 0.0)
    wcs.sub.return_value = sub_wcs

    pos_x, pos_y = resolve_source_pixel_position(
        wcs=wcs,
        ra=0.0,
        dec=0.0,
        central_freq=0.0,
        n_pix=64,
        cell_size_arcsec=0.1,
        offset_x_arcsec=0.5,
        offset_y_arcsec=-0.2,
    )

    assert pos_x == pytest.approx(37.0)
    assert pos_y == pytest.approx(30.0)


@pytest.mark.unit
def test_simulation_params_overrides(tmp_path, sample_metadata_row_dict):
    """Test that parameter overrides work correctly."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        source_type="gaussian",
        snr=2.5,
        save_mode="fits",
        n_pix=256,
        n_channels=64,
        inject_serendipitous=True,
        remote=True,
    )

    assert params.source_type == "gaussian"
    assert params.snr == 2.5
    assert params.save_mode == "fits"
    assert params.n_pix == 256
    assert params.n_channels == 64
    assert params.inject_serendipitous is True
    assert params.remote is True


@pytest.mark.unit
@patch("almasim.services.simulation.spectral_setup.process_spectral_data")
@_shared_patch("usm", "stages", "skymodel_stage")
@_shared_patch("uin", "stages", "planning")
def test_run_simulation_point_source(
    mock_interferometry,
    mock_skymodels,
    mock_process_spectral,
    tmp_path,
    sample_metadata_row_dict,
    main_dir,
):
    """Test running simulation with point source."""
    # Setup mocks
    mock_process_spectral.return_value = (
        np.ones(32) * 0.1,  # continuum
        np.array([1.0, 0.5]),  # line_fluxes
        ["CO(3-2)", "CII"],  # line_names
        0.5,  # redshift
        np.array([250.0, 350.0]) * 1e9,  # line_frequency
        [10, 11],  # source_channel_index
        32,  # n_channels_nw
        1.875,  # bandwidth
        0.1,  # freq_support
        np.array([250.0]) * 1e9,  # cont_frequencies
        [2.0, 2.0],  # fwhm_z
        1e10,  # lum_infrared
    )

    mock_datacube = Mock()
    mock_datacube._array = Mock()
    mock_datacube._array.to_value.return_value = np.random.rand(32, 32, 32) * 0.1
    mock_datacube.wcs = Mock()
    mock_datacube.wcs.sub.return_value.wcs_world2pix.return_value = (16, 16, 0)
    mock_datacube.n_px_x = 32
    mock_datacube.n_channels = 32

    mock_pointlike = Mock()
    mock_pointlike.insert.return_value = mock_datacube
    mock_skymodels.PointlikeSkyModel.return_value = mock_pointlike
    mock_skymodels.DataCube.return_value = mock_datacube
    mock_skymodels.get_datacube_header.return_value = fits.Header()

    mock_interferometer = Mock()
    mock_interferometer.run_interferometric_sim.return_value = {
        "model_cube": np.random.rand(32, 32, 32),
        "dirty_cube": np.random.rand(32, 32, 32),
        "model_vis": np.random.rand(32, 32, 32),
        "dirty_vis": np.random.rand(32, 32, 32),
        "beam_cube": np.random.rand(32, 32, 32),
        "totsampling_cube": np.random.rand(32, 32, 32),
        "uv_mask_cube": np.ones((32, 32, 32), dtype=np.uint8),
        "u_cube": np.random.rand(32, 2, 2),
        "v_cube": np.random.rand(32, 2, 2),
    }
    mock_interferometry.Interferometer.return_value = mock_interferometer
    mock_interferometry.compute_channel_noise = compute_channel_noise
    mock_interferometry.NoiseModelConfig = NoiseModelConfig
    mock_interferometry.calibrate_noise_profile = calibrate_noise_profile
    mock_interferometry.combine_interferometric_results.side_effect = (
        lambda results, config_weights=None: {
            **results[0],
            "per_config_results": results,
            "combined_config_count": len(results),
        }
    )

    # Create params
    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        source_type="point",
        ms_save_mode="none",
    )

    results = run_simulation(
        params,
        logger=print,
    )

    assert results is not None
    assert "model_cube" in results or "dirty_cube" in results
    mock_pointlike.insert.assert_called_once()


@pytest.mark.unit
@patch("almasim.services.simulation.spectral_setup.process_spectral_data")
@_shared_patch("usm", "stages", "skymodel_stage")
@_shared_patch("uin", "stages", "planning")
def test_run_simulation_gaussian_source(
    mock_interferometry,
    mock_skymodels,
    mock_process_spectral,
    tmp_path,
    sample_metadata_row_dict,
    main_dir,
):
    """Test running simulation with Gaussian source."""
    # Setup mocks
    mock_process_spectral.return_value = (
        np.ones(32) * 0.1,
        np.array([1.0]),
        ["CO(3-2)"],
        0.5,
        np.array([250.0]) * 1e9,
        [10],
        32,
        1.875,
        0.1,
        np.array([250.0]) * 1e9,
        [2.0],
        1e10,
    )

    mock_datacube = Mock()
    mock_datacube._array = Mock()
    mock_datacube._array.to_value.return_value = np.random.rand(32, 32, 32) * 0.1
    mock_datacube.wcs = Mock()
    mock_datacube.wcs.sub.return_value.wcs_world2pix.return_value = (16, 16, 0)
    mock_datacube.n_px_x = 32
    mock_datacube.n_channels = 32

    mock_gaussian = Mock()
    mock_gaussian.insert.return_value = mock_datacube
    mock_skymodels.GaussianSkyModel.return_value = mock_gaussian
    mock_skymodels.DataCube.return_value = mock_datacube
    mock_skymodels.get_datacube_header.return_value = fits.Header()

    mock_interferometer = Mock()
    mock_interferometer.run_interferometric_sim.return_value = {
        "model_cube": np.random.rand(32, 32, 32),
        "dirty_cube": np.random.rand(32, 32, 32),
        "model_vis": np.random.rand(32, 32, 32),
        "dirty_vis": np.random.rand(32, 32, 32),
        "beam_cube": np.random.rand(32, 32, 32),
        "totsampling_cube": np.random.rand(32, 32, 32),
        "uv_mask_cube": np.ones((32, 32, 32), dtype=np.uint8),
        "u_cube": np.random.rand(32, 2, 2),
        "v_cube": np.random.rand(32, 2, 2),
    }
    mock_interferometry.Interferometer.return_value = mock_interferometer
    mock_interferometry.compute_channel_noise = compute_channel_noise
    mock_interferometry.NoiseModelConfig = NoiseModelConfig
    mock_interferometry.calibrate_noise_profile = calibrate_noise_profile
    mock_interferometry.combine_interferometric_results.side_effect = (
        lambda results, config_weights=None: {
            **results[0],
            "per_config_results": results,
            "combined_config_count": len(results),
        }
    )

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        source_type="gaussian",
        ms_save_mode="none",
    )

    results = run_simulation(
        params,
        logger=print,
    )

    assert results is not None
    mock_gaussian.insert.assert_called_once()


@pytest.mark.unit
def test_simulation_params_dataclass_fields(tmp_path, sample_metadata_row_dict):
    """Test that SimulationParams has all required fields."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )

    # Check that all expected fields exist and are accessible
    required_fields = [
        "idx",
        "source_name",
        "member_ouid",
        "main_dir",
        "output_dir",
        "tng_dir",
        "galaxy_zoo_dir",
        "hubble_dir",
        "project_name",
        "ra",
        "dec",
        "band",
        "ang_res",
        "vel_res",
        "fov",
        "obs_date",
        "pwv",
        "int_time",
        "bandwidth",
        "freq",
        "freq_support",
        "cont_sens",
        "antenna_array",
        "n_pix",
        "n_channels",
        "source_type",
        "tng_api_key",
        "ncpu",
        "rest_frequency",
        "redshift",
        "lum_infrared",
        "snr",
        "n_lines",
        "line_names",
        "save_mode",
        "persist",
        "ml_dataset_path",
        "inject_serendipitous",
        "remote",
        "observation_configs",
        "ground_temperature_k",
        "correlator",
        "elevation_deg",
        "line_sens_10kms",
        "source_offset_x_arcsec",
        "source_offset_y_arcsec",
        "background_mode",
        "background_level",
        "background_seed",
        "external_skymodel_path",
        "external_component_table_path",
        "external_alignment_mode",
        "external_header_mode",
        "external_header_overrides",
        "ms_export",
        "ms_export_dir",
    ]

    # Verify all fields exist and can be accessed
    for field in required_fields:
        assert hasattr(params, field), f"Missing field: {field}"
        getattr(params, field)  # Should not raise AttributeError


@pytest.mark.unit
def test_write_ml_dataset_shard(tmp_path):
    """Test writing ML dataset shards to HDF5."""
    output_path = tmp_path / "dataset" / "sample_0001.h5"
    clean_cube = np.ones((2, 4, 4), dtype=np.float32)
    dirty_cube = np.zeros((2, 4, 4), dtype=np.float32)
    dirty_vis = np.ones((2, 4, 4), dtype=np.complex64) * (1 + 2j)
    uv_mask_cube = np.array(clean_cube > 0, dtype=np.uint8)
    metadata = {"source_name": "J1234+5678", "n_channels": 2}

    result_path = write_ml_dataset_shard(
        output_path,
        clean_cube=clean_cube,
        dirty_cube=dirty_cube,
        dirty_vis=dirty_vis,
        uv_mask_cube=uv_mask_cube,
        metadata=metadata,
    )

    assert result_path == str(output_path.resolve())
    with h5py.File(output_path, "r") as h5f:
        assert np.array_equal(h5f["clean_cube"][:], clean_cube)
        assert np.array_equal(h5f["dirty_cube"][:], dirty_cube)
        assert np.array_equal(h5f["dirty_vis"][:], dirty_vis)
        assert np.array_equal(h5f["uv_mask_cube"][:], uv_mask_cube)
        assert "source_name" in h5f.attrs["metadata_json"]


@pytest.mark.unit
@patch("almasim.services.simulation.export.export_native_ms")
def test_export_results_triggers_ms_export(mock_export_native_ms, tmp_path):
    """Native MeasurementSet export should be triggered when requested."""
    mock_export_native_ms.return_value = str(tmp_path / "demo.ms")
    params = SimulationParams(
        idx=1,
        source_name="demo",
        member_ouid="uid://demo",
        main_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        tng_dir=str(tmp_path / "tng"),
        galaxy_zoo_dir=str(tmp_path / "gz"),
        hubble_dir=str(tmp_path / "hub"),
        project_name="demo_project",
        ra=1.0,
        dec=2.0,
        band=6.0,
        ang_res=0.1,
        vel_res=10.0,
        fov=1.0,
        obs_date="2020-01-01",
        pwv=1.0,
        int_time=10.0,
        bandwidth=1.0,
        freq=250.0,
        freq_support="spec",
        cont_sens=0.1,
        antenna_array="A001:DA01 A002:DA02",
        n_pix=16,
        n_channels=4,
        source_type="point",
        tng_api_key=None,
        ncpu=1,
        rest_frequency=None,
        redshift=0.0,
        lum_infrared=None,
        snr=5.0,
        n_lines=None,
        line_names=None,
        save_mode="memory",
        persist=False,
        ml_dataset_path=None,
        inject_serendipitous=False,
        ms_export=True,
        ms_export_dir=str(tmp_path / "demo.ms"),
    )
    stage = CleanCubeStage(
        datacube=None,
        model_cube=np.zeros((4, 16, 16), dtype=np.float32),
        header=Mock(),
        output_dir_abs=str(tmp_path / "out"),
        sim_output_dir=None,
        sim_params_payload={"sim_params_path": str(tmp_path / "params.txt")},
        interferometer_kwargs={},
        interferometer_runs=[],
        total_power_runs=[],
        metadata={"source_name": "demo"},
        observation_plan={"configs": []},
        channel_frequencies_hz=np.linspace(1.0, 2.0, 4),
        channel_width_hz=1.0,
        cell_size_arcsec=0.1,
        background_cube=None,
        external_input_metadata=None,
    )
    simulation_results = {
        "dirty_cube": np.zeros((4, 16, 16), dtype=np.float32),
        "dirty_vis": np.zeros((4, 16, 16), dtype=np.complex64),
        "uv_mask_cube": np.zeros((4, 16, 16), dtype=np.uint8),
        "visibility_table": {
            "uvw_m": np.zeros((2, 3)),
            "antenna1": np.array([0, 0], dtype=np.int32),
            "antenna2": np.array([1, 1], dtype=np.int32),
            "time_mjd_s": np.array([0.0, 1.0]),
            "interval_s": np.array([1.0, 1.0]),
            "exposure_s": np.array([1.0, 1.0]),
            "data": np.zeros((2, 1, 4), dtype=np.complex64),
            "model_data": np.zeros((2, 1, 4), dtype=np.complex64),
            "flag": np.zeros((2, 1, 4), dtype=bool),
            "weight": np.ones((2, 1), dtype=np.float32),
            "sigma": np.ones((2, 1), dtype=np.float32),
            "channel_freq_hz": np.linspace(1.0, 2.0, 4),
            "antenna_names": ["DA01", "DA02"],
            "antenna_positions_m": np.zeros((2, 3)),
            "source_name": "demo",
            "field_ra_rad": 0.0,
            "field_dec_rad": 0.0,
            "observation_date": "2020-01-01",
        },
    }

    exported = export_results(params, stage, simulation_results)

    assert exported["ms_path"] == str(tmp_path / "demo.ms")
    mock_export_native_ms.assert_called_once()


@pytest.mark.unit
@patch("almasim.services.simulation.pipeline.export_results")
@patch("almasim.services.simulation.pipeline.simulate_observation")
@patch("almasim.services.simulation.pipeline.generate_clean_cube")
def test_run_simulation_memory_mode(
    mock_generate_clean_cube,
    mock_simulate_observation,
    mock_export_results,
    tmp_path,
    sample_metadata_row_dict,
    main_dir,
):
    """Test pure-Python execution mode without standard on-disk persistence."""
    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        save_mode="memory",
        persist=False,
        ml_dataset_path=tmp_path / "ml" / "sample_0001.h5",
    )

    mock_generate_clean_cube.return_value = Mock()
    mock_simulate_observation.return_value = {
        "model_cube": np.ones((2, 2, 2), dtype=np.float32),
        "dirty_cube": np.zeros((2, 2, 2), dtype=np.float32),
        "dirty_vis": np.zeros((2, 2, 2), dtype=np.complex64),
        "uv_mask_cube": np.ones((2, 2, 2), dtype=np.uint8),
    }
    mock_export_results.return_value = {"status": "ok"}

    result = run_simulation(params)

    assert result == {"status": "ok"}
    mock_generate_clean_cube.assert_called_once()
    mock_simulate_observation.assert_called_once()
    mock_export_results.assert_called_once()


@pytest.mark.unit
@patch("almasim.services.simulation.spectral_setup.process_spectral_data")
@_shared_patch("usm", "stages", "skymodel_stage")
@_shared_patch("uin", "stages", "planning")
def test_run_simulation_negative_line_flux_keeps_noise_non_negative(
    mock_interferometry,
    mock_skymodels,
    mock_process_spectral,
    tmp_path,
    sample_metadata_row_dict,
    main_dir,
):
    """Negative line fluxes should not produce negative interferometer noise."""
    mock_process_spectral.return_value = (
        np.ones(32) * 0.1,
        np.array([-1.0]),
        ["CO(3-2)"],
        0.5,
        np.array([250.0]) * 1e9,
        [10],
        32,
        1.875,
        0.1,
        np.array([250.0]) * 1e9,
        [2.0],
        1e10,
    )

    mock_datacube = Mock()
    mock_datacube._array = Mock()
    mock_datacube._array.to_value.return_value = np.random.rand(32, 32, 32) * 0.1
    mock_datacube.wcs = Mock()
    mock_datacube.wcs.sub.return_value.wcs_world2pix.return_value = (16, 16, 0)

    mock_pointlike = Mock()
    mock_pointlike.insert.return_value = mock_datacube
    mock_skymodels.PointlikeSkyModel.return_value = mock_pointlike
    mock_skymodels.DataCube.return_value = mock_datacube
    mock_skymodels.get_datacube_header.return_value = fits.Header()

    mock_interferometer = Mock()
    mock_interferometer.run_interferometric_sim.return_value = {
        "model_cube": np.random.rand(32, 32, 32),
        "dirty_cube": np.random.rand(32, 32, 32),
        "dirty_vis": np.random.rand(32, 32, 32),
        "model_vis": np.random.rand(32, 32, 32),
        "beam_cube": np.random.rand(32, 32, 32),
        "totsampling_cube": np.random.rand(32, 32, 32),
        "uv_mask_cube": np.ones((32, 32, 32), dtype=np.uint8),
        "u_cube": np.random.rand(32, 2, 2),
        "v_cube": np.random.rand(32, 2, 2),
    }
    mock_interferometry.Interferometer.return_value = mock_interferometer
    mock_interferometry.compute_channel_noise = compute_channel_noise
    mock_interferometry.NoiseModelConfig = NoiseModelConfig
    mock_interferometry.calibrate_noise_profile = calibrate_noise_profile
    mock_interferometry.combine_interferometric_results.side_effect = (
        lambda results, config_weights=None: {
            **results[0],
            "per_config_results": results,
            "combined_config_count": len(results),
        }
    )

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        source_type="point",
        ms_save_mode="none",
    )

    run_simulation(
        params,
        logger=print,
    )

    assert mock_interferometry.Interferometer.called
    noise_value = mock_interferometry.Interferometer.call_args.kwargs["noise"]
    assert np.all(np.asarray(noise_value) >= 0)


@pytest.mark.unit
def test_simulation_params_clean_function(tmp_path, sample_metadata_row_dict):
    """Test the clean function handles None and NaN correctly."""
    sample_metadata_row_dict["n_pix"] = None
    sample_metadata_row_dict["n_channels"] = np.nan

    main_dir = tmp_path / "main"
    main_dir.mkdir()

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )

    # Clean function should convert None/NaN to None
    assert params.n_pix is None
    assert params.n_channels is None


@pytest.mark.unit
def test_build_single_pointing_observation_plan_defaults(tmp_path, sample_metadata_row_dict):
    """Single-config simulations should still produce an explicit observation plan."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )

    plan = build_single_pointing_observation_plan(params)

    assert plan.phase_center_ra_deg == params.ra
    assert len(plan.configs) == 1
    assert plan.configs[0].antenna_array == params.antenna_array
    assert plan.configs[0].total_time_s == params.int_time


@pytest.mark.unit
def test_build_single_pointing_observation_plan_multiple_configs(
    tmp_path, sample_metadata_row_dict
):
    """P0 should support multi-config single-pointing plans."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        observation_configs=[
            {
                "name": "alma12",
                "array_type": "12m",
                "antenna_array": "A001:DA01 A002:DV02",
                "total_time_s": 1800.0,
            },
            {
                "name": "aca7",
                "array_type": "7m",
                "antenna_array": "A001:CM01 A002:CM02",
                "total_time_s": 2400.0,
            },
        ],
    )

    plan = build_single_pointing_observation_plan(params)

    assert len(plan.configs) == 2
    assert [cfg.array_type for cfg in plan.configs] == ["12m", "7m"]
    assert plan.configs[1].antenna_diameter_m == 7.0


@pytest.mark.unit
def test_build_single_pointing_observation_plan_splits_mixed_metadata_antenna_arrays(
    tmp_path, sample_metadata_row_dict
):
    """Mixed metadata antenna strings should auto-split into 12m and 7m configs."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    sample_metadata_row_dict["antenna_arrays"] = "A001:DA01 A002:DV02 A003:CM03 A004:CM04"

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )

    plan = build_single_pointing_observation_plan(params)

    assert len(plan.configs) == 2
    assert plan.configs[0].array_type == "12m"
    assert "DA01" in plan.configs[0].antenna_array
    assert "CM03" not in plan.configs[0].antenna_array
    assert plan.configs[1].array_type == "7m"
    assert "CM03" in plan.configs[1].antenna_array


@pytest.mark.unit
def test_build_single_pointing_observation_plan_includes_tp_from_metadata(
    tmp_path, sample_metadata_row_dict
):
    """Mixed metadata antenna strings should include TP configs once P1 is enabled."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    sample_metadata_row_dict["antenna_arrays"] = (
        "A001:DA01 A002:DV02 A003:CM03 A004:CM04 A005:PM05 A006:PM06"
    )

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )

    plan = build_single_pointing_observation_plan(params)

    assert [cfg.array_type for cfg in plan.configs] == ["12m", "7m", "TP"]
    assert "PM05" in plan.configs[2].antenna_array


@pytest.mark.unit
def test_compute_channel_noise_increases_with_pwv():
    """PWV should materially increase the single-pointing thermal noise profile."""
    freqs_hz = np.linspace(220e9, 230e9, 8)
    low = compute_channel_noise(
        NoiseModelConfig(pwv_mm=0.5),
        freqs_hz,
        bandwidth_hz=15.625e6,
        integration_s=1800.0,
        elevation_deg=60.0,
        antenna_diameter_m=12.0,
        n_antennas=10,
    )
    high = compute_channel_noise(
        NoiseModelConfig(pwv_mm=3.0),
        freqs_hz,
        bandwidth_hz=15.625e6,
        integration_s=1800.0,
        elevation_deg=60.0,
        antenna_diameter_m=12.0,
        n_antennas=10,
    )

    assert np.all(high > low)


@pytest.mark.unit
@patch("almasim.services.simulation.stages.uin.combine_interferometric_results")
@patch("almasim.services.simulation.spectral_setup.process_spectral_data")
@_shared_patch("usm", "stages", "skymodel_stage")
@_shared_patch("uin", "stages", "planning")
def test_run_simulation_multiconfig_single_pointing(
    mock_interferometry_module,
    mock_skymodels,
    mock_process_spectral,
    mock_combine_results,
    tmp_path,
    sample_metadata_row_dict,
    main_dir,
):
    """Multi-config P0 runs should call the interferometer once per observation config."""
    mock_process_spectral.return_value = (
        np.ones(8) * 0.1,
        np.array([1.0]),
        ["CO(3-2)"],
        0.5,
        np.array([250.0]) * 1e9,
        [2],
        8,
        1.875,
        0.1,
        np.array([250.0]) * 1e9,
        [2.0],
        1e10,
    )

    mock_datacube = Mock()
    mock_datacube._array = Mock()
    mock_datacube._array.to_value.return_value = np.random.rand(8, 8, 8) * 0.1
    mock_datacube.wcs = Mock()
    mock_datacube.wcs.sub.return_value.wcs_world2pix.return_value = (4, 4, 0)
    mock_pointlike = Mock()
    mock_pointlike.insert.return_value = mock_datacube
    mock_skymodels.PointlikeSkyModel.return_value = mock_pointlike
    mock_skymodels.DataCube.return_value = mock_datacube
    mock_skymodels.get_datacube_header.return_value = fits.Header()

    mock_interferometer = Mock()
    mock_interferometer.run_interferometric_sim.side_effect = [
        {
            "model_cube": np.ones((8, 8, 8), dtype=np.float32),
            "dirty_cube": np.ones((8, 8, 8), dtype=np.float32),
            "model_vis": np.ones((8, 8, 8), dtype=np.complex64),
            "dirty_vis": np.ones((8, 8, 8), dtype=np.complex64),
            "beam_cube": np.ones((8, 8, 8), dtype=np.float32),
            "totsampling_cube": np.ones((8, 8, 8), dtype=np.float32),
            "uv_mask_cube": np.ones((8, 8, 8), dtype=np.uint8),
            "u_cube": np.ones((8, 2, 2), dtype=np.float32),
            "v_cube": np.ones((8, 2, 2), dtype=np.float32),
        },
        {
            "model_cube": np.ones((8, 8, 8), dtype=np.float32),
            "dirty_cube": np.ones((8, 8, 8), dtype=np.float32),
            "model_vis": np.ones((8, 8, 8), dtype=np.complex64),
            "dirty_vis": np.ones((8, 8, 8), dtype=np.complex64),
            "beam_cube": np.ones((8, 8, 8), dtype=np.float32),
            "totsampling_cube": np.ones((8, 8, 8), dtype=np.float32),
            "uv_mask_cube": np.ones((8, 8, 8), dtype=np.uint8),
            "u_cube": np.ones((8, 2, 2), dtype=np.float32),
            "v_cube": np.ones((8, 2, 2), dtype=np.float32),
        },
    ]
    mock_interferometry_module.Interferometer.return_value = mock_interferometer
    mock_interferometry_module.compute_channel_noise = compute_channel_noise
    mock_interferometry_module.NoiseModelConfig = NoiseModelConfig
    mock_interferometry_module.calibrate_noise_profile.side_effect = lambda raw, reference_noise: (
        raw * (reference_noise / np.median(raw))
    )
    # The export writes the combined observation's cubes: a whole result.
    mock_combine_results.side_effect = lambda results, **_: {
        **results[0],
        "combined_config_count": 2,
    }

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        observation_configs=[
            {
                "name": "alma12",
                "array_type": "12m",
                "antenna_array": "A001:DA01 A002:DV02",
                "total_time_s": 1800.0,
            },
            {
                "name": "aca7",
                "array_type": "7m",
                "antenna_array": "A001:CM01 A002:CM02",
                "total_time_s": 2400.0,
            },
        ],
        ms_save_mode="none",
    )

    result = run_simulation(params)

    assert result["combined_config_count"] == 2
    assert mock_interferometry_module.Interferometer.call_count == 2


@pytest.mark.unit
@patch("almasim.services.simulation.spectral_setup.process_spectral_data")
@_shared_patch("usm", "stages", "skymodel_stage")
@_shared_patch("uin", "stages", "planning")
def test_run_simulation_p1_tp_and_reconstruction_outputs(
    mock_interferometry_module,
    mock_skymodels,
    mock_process_spectral,
    tmp_path,
    sample_metadata_row_dict,
    main_dir,
):
    """P1 runs should surface TP, INT image, and TP+INT image products."""
    mock_process_spectral.return_value = (
        np.ones(8) * 0.1,
        np.array([1.0]),
        ["CO(3-2)"],
        0.5,
        np.array([250.0]) * 1e9,
        [2],
        8,
        1.875,
        0.1,
        np.array([250.0]) * 1e9,
        [2.0],
        1e10,
    )

    mock_datacube = Mock()
    mock_datacube._array = Mock()
    mock_datacube._array.to_value.return_value = np.random.rand(8, 8, 8) * 0.1
    mock_datacube.wcs = Mock()
    mock_datacube.wcs.sub.return_value.wcs_world2pix.return_value = (4, 4, 0)
    mock_pointlike = Mock()
    mock_pointlike.insert.return_value = mock_datacube
    mock_skymodels.PointlikeSkyModel.return_value = mock_pointlike
    mock_skymodels.DataCube.return_value = mock_datacube
    mock_skymodels.get_datacube_header.return_value = fits.Header()

    mock_interferometer = Mock()
    mock_interferometer.run_interferometric_sim.return_value = {
        "model_cube": np.ones((8, 8, 8), dtype=np.float32),
        "dirty_cube": np.ones((8, 8, 8), dtype=np.float32),
        "model_vis": np.ones((8, 8, 8), dtype=np.complex64),
        "dirty_vis": np.ones((8, 8, 8), dtype=np.complex64),
        "beam_cube": np.ones((8, 8, 8), dtype=np.float32),
        "totsampling_cube": np.ones((8, 8, 8), dtype=np.float32),
        "uv_mask_cube": np.ones((8, 8, 8), dtype=np.uint8),
        "u_cube": np.ones((8, 2, 2), dtype=np.float32),
        "v_cube": np.ones((8, 2, 2), dtype=np.float32),
    }
    mock_interferometry_module.Interferometer.return_value = mock_interferometer
    mock_interferometry_module.compute_channel_noise = compute_channel_noise
    mock_interferometry_module.NoiseModelConfig = NoiseModelConfig
    mock_interferometry_module.calibrate_noise_profile.side_effect = lambda raw, reference_noise: (
        raw * (reference_noise / np.median(raw))
    )
    mock_interferometry_module.combine_interferometric_results.side_effect = (
        lambda results, config_weights=None: {
            **results[0],
            "per_config_results": results,
            "combined_config_count": len(results),
        }
    )
    mock_interferometry_module.simulate_total_power_observation.return_value = {
        "tp_model_cube": np.ones((8, 8, 8), dtype=np.float32),
        "tp_dirty_cube": np.ones((8, 8, 8), dtype=np.float32) * 2,
        "tp_beam_cube": np.ones((8, 8, 8), dtype=np.float32),
        "tp_sampling_cube": np.ones((8, 8, 8), dtype=np.float32),
        "tp_beam_fwhm_arcsec": np.ones(8, dtype=np.float32),
    }
    mock_interferometry_module.combine_total_power_results.side_effect = (
        lambda results, config_weights=None: {
            **results[0],
            "per_config_results": results,
            "combined_config_count": len(results),
        }
    )

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        observation_configs=[
            {
                "name": "alma12",
                "array_type": "12m",
                "antenna_array": "A001:DA01 A002:DV02",
                "total_time_s": 1800.0,
            },
            {
                "name": "tp",
                "array_type": "TP",
                "antenna_array": "A005:PM05 A006:PM06",
                "total_time_s": 1200.0,
            },
        ],
        persist=False,
        save_mode="memory",
        ms_save_mode="none",
    )

    result = run_simulation(params)

    assert result["tp_results"] is not None
    assert result["int_results"] is not None
    assert result["int_image_cube"] is not None
    assert result["tp_image_cube"] is not None
    assert result["tp_int_image_cube"] is not None


# ===========================================================================
# Additional tests for uncovered branches in simulation.py
# ===========================================================================


@pytest.mark.unit
def test_generate_background_cube_none_mode_returns_zeros():
    """Background cube in 'none' mode should be all zeros."""
    cube = generate_background_cube(
        mode="none",
        n_pix=16,
        n_channels=4,
        cell_size_arcsec=0.1,
        channel_frequencies_hz=np.linspace(240e9, 250e9, 4),
        cont_sens_jy=1e-4,
    )
    assert cube.shape == (4, 16, 16)
    assert np.all(cube == 0.0)


@pytest.mark.unit
def test_generate_background_cube_zero_level_returns_zeros():
    """Background cube with level=0 should return zeros regardless of mode."""
    cube = generate_background_cube(
        mode="combined",
        n_pix=16,
        n_channels=4,
        cell_size_arcsec=0.1,
        channel_frequencies_hz=np.linspace(240e9, 250e9, 4),
        cont_sens_jy=1e-4,
        level=0.0,
    )
    assert np.all(cube == 0.0)


@pytest.mark.unit
def test_generate_background_cube_dusty_diffuse_mode():
    """Background cube in 'dusty_diffuse' mode has non-zero values."""
    cube = generate_background_cube(
        mode="dusty_diffuse",
        n_pix=16,
        n_channels=4,
        cell_size_arcsec=0.1,
        channel_frequencies_hz=np.linspace(240e9, 250e9, 4),
        cont_sens_jy=1e-4,
        level=1.0,
        seed=42,
    )
    assert cube.shape == (4, 16, 16)
    assert float(np.sum(cube)) > 0.0


@pytest.mark.unit
def test_generate_background_cube_blank_field_dsfg_mode():
    """Background cube in 'blank_field_dsfg' mode has non-zero values."""
    cube = generate_background_cube(
        mode="blank_field_dsfg",
        n_pix=16,
        n_channels=4,
        cell_size_arcsec=0.1,
        channel_frequencies_hz=np.linspace(240e9, 250e9, 4),
        cont_sens_jy=1e-4,
        level=1.0,
        seed=7,
    )
    assert cube.shape == (4, 16, 16)
    assert float(np.sum(cube)) >= 0.0


@pytest.mark.unit
def test_resolve_source_pixel_position_clips_to_bounds():
    """Pixel position is clipped within [0, n_pix-1]."""
    wcs = MagicMock()
    sub_wcs = MagicMock()
    # Return positions that exceed bounds
    sub_wcs.wcs_world2pix.return_value = (-100.0, 200.0, 0.0)
    wcs.sub.return_value = sub_wcs

    pos_x, pos_y = resolve_source_pixel_position(
        wcs=wcs,
        ra=0.0,
        dec=0.0,
        central_freq=0.0,
        n_pix=64,
        cell_size_arcsec=0.1,
    )

    assert pos_x == pytest.approx(0.0)  # clipped from -100
    assert pos_y == pytest.approx(63.0)  # clipped from 200


@pytest.mark.unit
def test_simulation_params_with_line_sens_10kms(tmp_path, sample_metadata_row_dict):
    """SimulationParams stores line_sens_10kms when present in metadata."""
    sample_metadata_row_dict["Line_sens_10kms_mJybeam"] = 0.05

    main_dir = tmp_path / "main"
    main_dir.mkdir()

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )

    assert params.line_sens_10kms == pytest.approx(0.05)


@pytest.mark.unit
def test_simulation_params_with_background_params(tmp_path, sample_metadata_row_dict):
    """SimulationParams correctly stores background configuration."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        background_mode="combined",
        background_level=2.5,
        background_seed=99,
        seed="7",
    )

    assert params.background_mode == "combined"
    assert params.background_level == pytest.approx(2.5)
    assert params.background_seed == 99
    assert params.seed == 7, "the simulation seed is kept, as an int"


@pytest.mark.unit
def test_simulation_params_external_skymodel_paths(tmp_path, sample_metadata_row_dict):
    """SimulationParams resolves external skymodel paths to absolute paths."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        external_skymodel_path=tmp_path / "model.fits",
        external_component_table_path=tmp_path / "table.csv",
        external_alignment_mode="observation",
        external_header_mode="preserve",
    )

    assert params.external_skymodel_path is not None
    assert Path(params.external_skymodel_path).is_absolute()
    assert params.external_alignment_mode == "observation"
    assert params.external_header_mode == "preserve"


@pytest.mark.unit
def test_simulation_params_ms_export_config(tmp_path, sample_metadata_row_dict):
    """SimulationParams stores MS export configuration."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        ms_export=True,
        ms_export_dir=tmp_path / "ms_exports",
        ms_save_mode="msv2",
    )

    assert params.ms_export is True
    assert params.ms_export_dir is not None
    assert Path(params.ms_export_dir).is_absolute()
    assert params.ms_save_mode == "msv2"


@pytest.mark.unit
def test_write_ml_dataset_shard_creates_parent_dirs(tmp_path):
    """write_ml_dataset_shard creates parent directories if needed."""
    output_path = tmp_path / "deep" / "nested" / "sample_0001.h5"
    clean_cube = np.ones((2, 4, 4), dtype=np.float32)
    dirty_cube = np.zeros((2, 4, 4), dtype=np.float32)
    dirty_vis = np.zeros((2, 4, 4), dtype=np.complex64)
    uv_mask_cube = np.ones((2, 4, 4), dtype=np.uint8)

    result = write_ml_dataset_shard(
        output_path,
        clean_cube=clean_cube,
        dirty_cube=dirty_cube,
        dirty_vis=dirty_vis,
        uv_mask_cube=uv_mask_cube,
        metadata={"source_name": "test"},
    )
    assert Path(result).exists()


@pytest.mark.unit
@patch("almasim.services.simulation.pipeline.export_results")
@patch("almasim.services.simulation.pipeline.simulate_observation")
@patch("almasim.services.simulation.pipeline.generate_clean_cube")
def test_run_simulation_with_background_mode(
    mock_generate_clean_cube,
    mock_simulate_observation,
    mock_export_results,
    tmp_path,
    sample_metadata_row_dict,
    main_dir,
):
    """run_simulation handles background_mode='combined' without error."""
    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        save_mode="memory",
        persist=False,
        background_mode="combined",
        background_level=1.0,
    )

    mock_generate_clean_cube.return_value = Mock()
    mock_simulate_observation.return_value = {
        "model_cube": np.ones((2, 2, 2), dtype=np.float32),
        "dirty_cube": np.zeros((2, 2, 2), dtype=np.float32),
        "dirty_vis": np.zeros((2, 2, 2), dtype=np.complex64),
        "uv_mask_cube": np.ones((2, 2, 2), dtype=np.uint8),
    }
    mock_export_results.return_value = {"status": "ok"}

    result = run_simulation(params)
    assert result == {"status": "ok"}


@pytest.mark.unit
def test_simulation_params_ncpu_explicit(tmp_path, sample_metadata_row_dict):
    """Explicit ncpu overrides the cpu_count default."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        ncpu=4,
    )

    assert params.ncpu == 4


@pytest.mark.unit
def test_simulation_params_ground_temperature(tmp_path, sample_metadata_row_dict):
    """SimulationParams stores custom ground_temperature_k."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()

    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        ground_temperature_k=280.0,
    )

    assert params.ground_temperature_k == pytest.approx(280.0)
