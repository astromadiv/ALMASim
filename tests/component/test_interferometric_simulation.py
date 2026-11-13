"""Component tests for interferometric simulation."""

import astropy.units as U
import numpy as np
import pandas as pd
import pytest

from almasim.services import interferometry


class InlineBackend:
    """Minimal synchronous backend for component tests."""

    def scatter(self, data, broadcast: bool = False):
        return data

    def compute(self, tasks, sync: bool = True):
        if isinstance(tasks, list):
            return [
                (task() if callable(task) else task.compute() if hasattr(task, "compute") else task)
                for task in tasks
            ]
        return (
            tasks() if callable(tasks) else tasks.compute() if hasattr(tasks, "compute") else tasks
        )

    def gather(self, futures):
        return futures if isinstance(futures, list) else [futures]

    def delayed(self, func):
        def delayed_decorator(*args, **kwargs):
            return lambda: func(*args, **kwargs)

        return delayed_decorator

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


@pytest.fixture
def sample_skymodel():
    """Create a sample sky model array for testing."""
    # Create a simple 3D sky model: (n_channels, n_pix, n_pix)
    n_channels = 16
    n_pix = 32
    model = np.random.rand(n_channels, n_pix, n_pix) * 0.1  # Jy
    return model


@pytest.fixture
def sample_fits_header():
    """Create a sample FITS header for testing."""
    from astropy.io import fits

    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = 32
    header["NAXIS2"] = 32
    header["NAXIS3"] = 16
    header["DATE-OBS"] = "2020-01-01T00:00:00"
    return header


@pytest.fixture
def sample_antenna_array(test_data_dir):
    """Load a realistic antenna array string from sample metadata."""
    metadata = pd.read_csv(test_data_dir / "qso_metadata.csv")
    return metadata["antenna_arrays"].iloc[0]


@pytest.fixture
def compact_antenna_array():
    """Compact 12 m-like antenna subset for UV/beam regression tests."""
    return "A001:DA01 A002:DA02 A003:DA03 A004:DA04"


@pytest.fixture
def extended_antenna_array():
    """More extended 12 m-like antenna subset for UV/beam regression tests."""
    return "A001:DA01 A020:DV20 A040:DV40 A060:DV60"


@pytest.fixture
def point_source_skymodel():
    """Create a deterministic centered point-source cube."""
    n_channels = 4
    n_pix = 16
    model = np.zeros((n_channels, n_pix, n_pix), dtype=np.float32)
    model[:, n_pix // 2, n_pix // 2] = 1.0
    return model


def _beam_core_width(beam: np.ndarray) -> int:
    """Count central-row pixels above half maximum as a simple beam-width proxy."""
    center = beam.shape[0] // 2
    row = beam[center]
    return int(np.count_nonzero(row >= 0.5))


@pytest.mark.component
def test_interferometer_initialization(
    main_dir, tmp_path, sample_skymodel, sample_fits_header, sample_antenna_array
):
    """Test initializing the Interferometer class."""
    with InlineBackend() as backend:
        interferometer = interferometry.Interferometer(
            idx=0,
            skymodel=sample_skymodel,
            backend=backend,
            main_dir=str(main_dir),
            output_dir=str(tmp_path),
            ra=0.0 * U.deg,
            dec=0.0 * U.deg,
            central_freq=100.0 * U.GHz,
            bandwidth=10.0 * U.GHz,
            fov=10.0,
            antenna_array=sample_antenna_array,
            noise=0.01,
            snr=1.3,
            integration_time=1.0,
            observation_date="2020-01-01",
            header=sample_fits_header,
            save_mode="npz",
            robust=0.0,
        )

        assert interferometer.idx == 0
        assert interferometer.skymodel.shape == sample_skymodel.shape
        assert interferometer.main_dir == str(main_dir)
        assert interferometer.output_dir == str(tmp_path)
        assert interferometer.antenna_array == sample_antenna_array


@pytest.mark.component
def test_interferometer_baseline_preparation(
    main_dir, tmp_path, sample_skymodel, sample_fits_header, sample_antenna_array
):
    """Test baseline preparation in interferometer."""
    with InlineBackend() as backend:
        interferometer = interferometry.Interferometer(
            idx=0,
            skymodel=sample_skymodel,
            backend=backend,
            main_dir=str(main_dir),
            output_dir=str(tmp_path),
            ra=0.0 * U.deg,
            dec=0.0 * U.deg,
            central_freq=100.0 * U.GHz,
            bandwidth=10.0 * U.GHz,
            fov=10.0,
            antenna_array=sample_antenna_array,
            noise=0.01,
            snr=1.3,
            integration_time=1.0,
            observation_date="2020-01-01",
            header=sample_fits_header,
            save_mode="npz",
            robust=0.0,
        )

        # Check that baseline-related attributes are set
        assert hasattr(interferometer, "Nant")
        assert hasattr(interferometer, "antPos")
        assert interferometer.Nant > 0
        assert len(interferometer.antPos) > 0


@pytest.mark.component
@pytest.mark.slow
def test_interferometer_simulation_run(
    main_dir, tmp_path, sample_skymodel, sample_fits_header, sample_antenna_array
):
    """Test running a full interferometric simulation."""
    with InlineBackend() as backend:
        interferometer = interferometry.Interferometer(
            idx=0,
            skymodel=sample_skymodel,
            backend=backend,
            main_dir=str(main_dir),
            output_dir=str(tmp_path),
            ra=0.0 * U.deg,
            dec=0.0 * U.deg,
            central_freq=100.0 * U.GHz,
            bandwidth=10.0 * U.GHz,
            fov=10.0,
            antenna_array=sample_antenna_array,
            noise=0.01,
            snr=1.3,
            integration_time=0.1,  # Short integration for testing
            observation_date="2020-01-01",
            header=sample_fits_header,
            save_mode="npz",
            robust=0.0,
        )

        results = interferometer.run_interferometric_sim()

        assert results is not None
        # Check for snake_case keys (actual return format)
        assert "model_cube" in results or "modelCube" in results
        assert "dirty_cube" in results or "dirtyCube" in results
        assert "model_vis" in results or "modelVis" in results
        assert "dirty_vis" in results or "dirtyVis" in results

        # Check output files were created
        output_files = list(tmp_path.glob("**/*.npz"))
        assert len(output_files) > 0


@pytest.mark.component
def test_interferometer_progress_signal(
    main_dir, tmp_path, sample_skymodel, sample_fits_header, sample_antenna_array
):
    """Test interferometer progress signal."""
    progress_values = []

    def progress_callback(value):
        progress_values.append(value)

    with InlineBackend() as backend:
        interferometer = interferometry.Interferometer(
            idx=0,
            skymodel=sample_skymodel,
            backend=backend,
            main_dir=str(main_dir),
            output_dir=str(tmp_path),
            ra=0.0 * U.deg,
            dec=0.0 * U.deg,
            central_freq=100.0 * U.GHz,
            bandwidth=10.0 * U.GHz,
            fov=10.0,
            antenna_array=sample_antenna_array,
            noise=0.01,
            snr=1.3,
            integration_time=1.0,
            observation_date="2020-01-01",
            header=sample_fits_header,
            save_mode="npz",
            robust=0.0,
        )

        interferometer.progress_signal.connect(progress_callback)
        interferometer.progress_signal.emit(50)
        interferometer.progress_signal.emit(100)

        assert len(progress_values) == 2
        assert progress_values == [50, 100]


@pytest.mark.component
def test_interferometer_save_modes(
    main_dir, tmp_path, sample_skymodel, sample_fits_header, sample_antenna_array
):
    """Test different save modes for interferometer."""
    save_modes = ["npz", "fits"]

    for save_mode in save_modes:
        with InlineBackend() as backend:
            interferometer = interferometry.Interferometer(
                idx=0,
                skymodel=sample_skymodel,
                backend=backend,
                main_dir=str(main_dir),
                output_dir=str(tmp_path / save_mode),
                ra=0.0 * U.deg,
                dec=0.0 * U.deg,
                central_freq=100.0 * U.GHz,
                bandwidth=10.0 * U.GHz,
                fov=10.0,
                antenna_array=sample_antenna_array,
                noise=0.01,
                snr=1.3,
                integration_time=0.05,  # Very short for testing
                observation_date="2020-01-01",
                header=sample_fits_header,
                save_mode=save_mode,
                robust=0.0,
            )

            # Just verify it can be initialized with different save modes
            assert interferometer.save_mode == save_mode


@pytest.mark.component
@pytest.mark.slow
def test_interferometer_config_changes_uv_coverage_and_dirty_beam(
    main_dir,
    tmp_path,
    point_source_skymodel,
    compact_antenna_array,
    extended_antenna_array,
):
    """Different antenna configurations should change UV support and beam width."""
    from astropy.io import fits

    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = point_source_skymodel.shape[2]
    header["NAXIS2"] = point_source_skymodel.shape[1]
    header["NAXIS3"] = point_source_skymodel.shape[0]
    common_kwargs = dict(
        idx=0,
        skymodel=point_source_skymodel,
        main_dir=str(main_dir),
        output_dir=str(tmp_path),
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        central_freq=100.0 * U.GHz,
        bandwidth=1.0 * U.GHz,
        fov=10.0,
        noise=0.0,
        snr=1.3,
        integration_time=0.02,
        observation_date="2020-01-01",
        header=header,
        save_mode="memory",
        robust=0.0,
        persist=False,
    )

    with InlineBackend() as backend:
        compact = interferometry.Interferometer(
            backend=backend,
            antenna_array=compact_antenna_array,
            **common_kwargs,
        ).run_interferometric_sim()
        extended = interferometry.Interferometer(
            backend=backend,
            antenna_array=extended_antenna_array,
            **common_kwargs,
        ).run_interferometric_sim()

    compact_uv_support = int(np.count_nonzero(compact["uv_mask_cube"]))
    extended_uv_support = int(np.count_nonzero(extended["uv_mask_cube"]))
    compact_beam_width = _beam_core_width(compact["beam"])
    extended_beam_width = _beam_core_width(extended["beam"])

    assert extended_uv_support > compact_uv_support
    assert extended_beam_width < compact_beam_width


@pytest.mark.component
def test_a_7m_array_with_a_total_power_dish_simulates_each_baselines_own_dishes(
    main_dir, tmp_path, point_source_skymodel
):
    """A 7 m array correlating a 12 m dish on a total-power pad: the MS lists
    its 12 m dish, and a baseline with it carries the noise of its two
    dishes' effective areas; the others the array's."""
    from astropy.io import fits

    header = fits.Header()
    header["DATE-OBS"] = "2020-01-01T00:00:00"
    with InlineBackend() as backend:
        results = interferometry.Interferometer(
            idx=0,
            skymodel=point_source_skymodel,
            backend=backend,
            main_dir=str(main_dir),
            output_dir=str(tmp_path),
            ra=0.0 * U.deg,
            dec=-30.0 * U.deg,
            central_freq=650.0 * U.GHz,
            bandwidth=1.0 * U.GHz,
            fov=10.0,
            antenna_array="J501:CM01 J502:CM02 J503:CM03 T701",
            noise=0.01,
            snr=1.3,
            integration_time=0.02,
            observation_date="2020-01-01",
            header=header,
            save_mode="memory",
            robust=0.0,
            persist=False,
            antenna_diameter_m=7.0,
            dish_areas_m2={7.0: 20.0, 12.0: 50.0},
            noise_entropy=(1, 2, 3),
        ).run_interferometric_sim()

    table = results["visibility_table"]
    names = list(table["antenna_names"])
    assert sorted(table["antenna_diameters_m"]) == [7.0, 7.0, 7.0, 12.0]
    big = names.index("T701")
    with_big = (table["antenna1"] == big) | (table["antenna2"] == big)
    sigma = np.asarray(table["sigma"])
    sigma = sigma.reshape(sigma.shape[0], -1)[:, 0]
    np.testing.assert_allclose(sigma[~with_big], sigma[~with_big][0])
    # sqrt(1 x 20/50): the 7 m dish's SEFD times the 12 m one's, over two 7 m's.
    np.testing.assert_allclose(sigma[with_big], sigma[~with_big][0] * np.sqrt(0.4), rtol=1e-5)
