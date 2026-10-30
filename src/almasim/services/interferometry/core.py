"""Core interferometry classes and functionality."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import astropy.units as U
import numpy as np
from astropy.io import fits

if TYPE_CHECKING:
    from ..compute.base import ComputationBackend

from ..compute.progress import watch_futures
from ..instrument.constants import ALMA_LATITUDE_DEG, SPEED_OF_LIGHT_M_S
from ..instrument.spectral_window import SpectralWindow
from ..products.cube_export import write_interferometry_products
from .antenna import read_array_layout
from .corruptions import (
    CorruptionModel,
    antenna_ripple,
    draw_antenna_corruptions,
    spectral_kernel,
)
from .observing import alma_location, clip_to_horizon, hour_angle_coverage, hour_angle_samples
from .simulator import ChannelImagingConfig, image_channels, picklable_header, stack_channels
from .utils import get_channel_wavelength
from .visibility import assemble_visibility_table

_logger = logging.getLogger(__name__)

LogFn = Callable[[str], None] | None


class ProgressSignal:
    """Simple progress callback system (not PyQt-specific)."""

    def __init__(self):
        self._subscribers: list[Callable[[int], None]] = []

    def connect(self, callback: Callable[[int], None] | None) -> None:
        """Connect a callback function to receive progress updates."""
        if callback is not None:
            self._subscribers.append(callback)

    def emit(self, value: int) -> None:
        """Emit progress value to all connected callbacks."""
        for callback in self._subscribers:
            callback(value)


class Interferometer:
    """ALMA interferometric observation simulator."""

    def __init__(
        self,
        idx: int,
        skymodel: np.ndarray,
        backend: ComputationBackend,
        main_dir: str,
        output_dir: str,
        ra: U.Quantity,
        dec: U.Quantity,
        central_freq: U.Quantity,
        bandwidth: U.Quantity,
        fov: float,
        antenna_array: str,
        noise: float,
        snr: float,
        integration_time: float,
        observation_date: str,
        header: fits.Header,
        save_mode: str,
        robust: float,
        imaging_algorithm: str = "legacy",
        persist: bool = True,
        antenna_diameter_m: float = 12.0,
        config_name: str | None = None,
        array_type: str | None = None,
        logger: LogFn = None,
        spectral_window: SpectralWindow | None = None,
        noise_entropy: tuple[int, ...] | None = None,
        corruptions: CorruptionModel | None = None,
        continuum_fraction: np.ndarray | None = None,
        sample_time_s: float | None = None,
        execution: int | None = None,
    ):
        """
        Initialize interferometer.

        Parameters
        ----------
        idx : int
            Simulation index
        skymodel : np.ndarray
            Sky model cube
        backend : ComputationBackend
            Computation backend for parallel processing
        main_dir : str
            Main data directory
        output_dir : str
            Output directory
        ra : U.Quantity
            Right ascension
        dec : U.Quantity
            Declination
        central_freq : U.Quantity
            Central frequency
        bandwidth : U.Quantity
            Bandwidth
        fov : float
            Field of view in arcsec
        antenna_array : str
            Antenna array configuration
        noise : float
            Noise level
        snr : float
            Signal-to-noise ratio
        integration_time : float
            Integration time in hours
        observation_date : str
            Observation date
        header : fits.Header
            FITS header
        save_mode : str
            Save mode ('npz', 'h5', or 'fits')
        persist : bool
            Whether to persist generated cubes to disk
        robust : float
            Robustness parameter
        logger : LogFn, optional
            Logger callback function
        spectral_window : SpectralWindow, optional
            The cube's channel grid, which the uv sampling and the visibility
            table use too; by default the band split into the cube's channels
        noise_entropy : tuple of int, optional
            Seeds the visibility noise, one stream per channel, and the
            corruptions (MADIV-28 T5.1); fresh entropy when None
        corruptions : CorruptionModel, optional
            What the calibrated visibilities carry besides white noise
            (correlated channels, antenna gains, atmospheric phase); none when None
        sample_time_s : float, optional
            Seconds per hour-angle sample; by default from the integration time
            (``observing.hour_angle_samples``)
        execution : int, optional
            The execution block this run is (its rows' OBSERVATION_ID); 0 when None
        """
        self.noise_entropy = noise_entropy
        self.corruptions = corruptions
        self.continuum_fraction = continuum_fraction
        self.idx = idx
        self.skymodel = skymodel
        self.backend = backend
        self.main_dir = main_dir
        self.output_dir = output_dir
        self.plot_dir = os.path.join(output_dir, "plots")
        self.ra = ra
        self.dec = dec
        self.central_freq = central_freq
        self.bandwidth = bandwidth
        self.fov = fov
        self.antenna_array = antenna_array
        self.noise = noise
        self.snr = snr
        self.integration_time = integration_time
        self.observation_date = observation_date
        self.sample_time_s = sample_time_s
        self.execution = execution
        # Each configuration gets its own copy: _get_nH appends EPOCH, which every
        # configuration used to add to one shared header (MADIV-28 T1.14).
        self.header = header.copy() if hasattr(header, "copy") else header
        self.save_mode = save_mode
        self.persist = persist
        self.robust = robust
        self.imaging_algorithm = str(imaging_algorithm).lower()
        if self.imaging_algorithm == "ducc0":
            # MADIV-28 T1.3: the ducc0 channel imager reads the local (x, y)
            # antenna offsets in km as (lat, lon) in degrees, so its uv distances
            # are ~1e5 too large; it also images one snapshot instead of the
            # hour-angle track. Disabled until the shared predictor (T3.7).
            raise ValueError(
                "imaging_algorithm='ducc0' is disabled: its uv geometry is wrong "
                "(see MADIV-28 T1.3); use 'legacy'"
            )
        self.antenna_diameter_m = float(antenna_diameter_m)
        self.config_name = config_name
        self.array_type = array_type
        self.logger = logger
        self.spectral_window = spectral_window
        self.progress_signal = ProgressSignal()

        # Initialize variables
        self._init_variables()
        # Get the observing location
        self._get_observing_location()
        # Get coverage and antennas
        self._get_Hcov()
        self._read_antennas()
        # Get the observing wavelengths for each channel
        self._get_wavelengths()

        msg = f"Performing {self.nH} scans with a scan time of {self.scan_time} seconds"
        self._log(msg)

    def _log(self, message: str) -> None:
        """Log a message using the logger callback or print."""
        if self.logger is not None:
            self.logger(message)
        else:
            _logger.info(message)

    def _hz_to_m(self, freq: float) -> float:
        """Convert frequency in Hz to wavelength in meters."""
        return self.c_ms / freq

    def _init_variables(self) -> None:
        """Initialize calculation variables."""
        self.deg2rad = np.pi / 180.0
        self.rad2deg = 180.0 / np.pi
        self.curzoom = [0, 0, 0, 0]
        self.gamma = 0.5
        self.lfac = 1.0e6
        self._get_nH()
        self.lat = ALMA_LATITUDE_DEG * self.deg2rad
        self.trlat = [np.sin(self.lat), np.cos(self.lat)]
        self.Diameters = [self.antenna_diameter_m, 0]
        self.ra = self.ra.value * self.deg2rad
        self.dec = self.dec.value * self.deg2rad
        self.trdec = [np.sin(self.dec), np.cos(self.dec)]
        self.central_freq = self.central_freq.to(U.Hz).value
        self.bandwidth = self.bandwidth.to(U.Hz).value
        self.imsize = 3 * self.fov
        # The input sky model IS the science field. The interferometric FFT needs
        # a 2x zero-padded grid (guard band) around it: the model is placed in the
        # central Nphf x Nphf box of an Npix = 2*Nphf grid. Previously Npix was set
        # to the model size and Nphf = Npix//2, which forced the model to be
        # downsampled by 2x into the central quarter (the "central square"
        # artefact for extended sources). Deriving Nphf from the model size and
        # padding to Npix = 2*Nphf keeps the model at full resolution.
        self.Nphf = self.skymodel.shape[1]
        self.Npix = 2 * self.Nphf
        self.Nchan = self.skymodel.shape[0]
        self.Np4 = self.Npix // 4
        self.pixsize = self.imsize / self.Npix
        self.UVpixsize = 2.0 / (self.imsize * np.pi / 180.0 / 3600.0)
        self.Xaxmax = self.imsize / 2
        self.c_ms = SPEED_OF_LIGHT_M_S
        self.xx = np.linspace(-self.Xaxmax, self.Xaxmax, self.Npix)
        self.yy = np.ones(self.Npix, dtype=np.float32)
        # Minus the squared distance (arcsec^2) from the pointing centre: xx is
        # already in arcsec. It used to be multiplied by pixsize^2 as well, which
        # made the primary beam's width scale with the pixel size (MADIV-28 T4.5).
        self.distmat = -np.outer(self.xx**2.0, self.yy) - np.outer(self.yy, self.xx**2.0)
        self.robfac = 0.0
        self.zooming = 0
        self._log(f"Number of Epochs: {self.nH}")

    def _get_observing_location(self) -> None:
        """Get ALMA observing location."""
        self.observing_location = alma_location()

    def _get_Hcov(self) -> None:
        """Hour-angle coverage [start, end] in radians, centred on the track's middle."""
        self.Hcov = hour_angle_coverage(
            self.observation_date,
            float(self.integration_time) * 3600.0,  # the integration time is in hours
            self.ra,
            self.observing_location,
        )

    def _get_nH(self) -> None:
        """Number of hour-angle samples (one per scan) and the scan time in seconds:
        the run's own sample time when it has one (an execution block sampled as
        all the MOUS's blocks are)."""
        duration_s = float(self.integration_time) * 3600.0
        if self.sample_time_s:
            self.scan_time = float(self.sample_time_s)
            self.nH = max(1, int(duration_s / self.scan_time))
        else:
            self.nH, self.scan_time = hour_angle_samples(duration_s)
        self.header.append(("EPOCH", self.nH))

    def _read_antennas(self) -> None:
        """Read antenna positions from configuration."""
        layout = read_array_layout(self.main_dir, self.antenna_array)
        self.antPos = layout.positions_km
        self.antNames = layout.names
        self.antPos3D = layout.positions_m
        clip_to_horizon(self.Hcov, self.lat, self.dec)
        self.Xmax = layout.max_extent_km * 1.5
        self.Nant = len(self.antPos)

    def _get_wavelengths(self) -> None:
        """Each channel's wavelength range, on the cube's channel grid (T5.4)."""
        if self.spectral_window is None:
            self.spectral_window = SpectralWindow.from_band(
                self.central_freq, self.bandwidth, self.Nchan
            )
        self.obs_wavelengths = self.spectral_window.wavelength_edges_m(self.c_ms)
        self.w_min = float(self.obs_wavelengths.min())
        self.w_max = float(self.obs_wavelengths.max())

    def get_channel_wavelength(self, channel: int) -> tuple:
        """Get wavelength and format string for a channel."""
        wavelength = get_channel_wavelength(self.obs_wavelengths, channel)
        fmtB1 = r"$\lambda = $ %4.1fmm  " % (wavelength[2] * 1.0e6)
        fmtB = (
            fmtB1
            + "\n"
            + r"% 4.2f Jy/beam"
            + "\n"
            + r"$\Delta\alpha = $ % 4.2f / $\Delta\delta = $ % 4.2f "
        )
        return wavelength, fmtB

    def _corruption_fields(self) -> dict:
        """The corruption model, its antenna errors (drawn once, for every
        channel) and the entropy correlated noise is keyed by."""
        model = self.corruptions
        if model is None:
            return {}
        # Streams of their own, apart from the per-channel noise streams.
        if self.noise_entropy is not None:
            entropy = (*self.noise_entropy, 1_000_000)
        else:
            entropy = tuple(int(x) for x in np.random.SeedSequence().generate_state(4))
        fields = {"corruptions": model, "corruption_entropy": (*entropy, 2)}
        fraction = self.continuum_fraction
        if fraction is not None and np.size(fraction) == self.Nchan:
            fields["continuum_fraction"] = np.asarray(fraction, dtype=np.float64)
        if model.has_antenna_errors:
            times_s = np.arange(self.nH) * float(self.scan_time)
            antenna_xy_m = np.asarray(self.antPos, dtype=np.float64)[:, :2] * 1e3
            fields["antenna_corruptions"] = draw_antenna_corruptions(
                model, antenna_xy_m, times_s, np.random.default_rng((*entropy, 1))
            )
        if model.antenna_ripple_rms > 0:
            fields["antenna_noise"] = antenna_ripple(
                np.random.default_rng((*entropy, 3)),
                self.Nant,
                self.spectral_window.frequencies_hz(),
                model.antenna_ripple_rms,
                model.antenna_ripple_mhz,
            )
        return fields

    def run_interferometric_sim(self) -> dict:
        """Run interferometric simulation."""
        config = ChannelImagingConfig(
            Npix=self.Npix,
            Nant=self.Nant,
            Hcov=self.Hcov,
            nH=self.nH,
            antPos=self.antPos,
            robfac=self.robfac,
            trlat=self.trlat,
            trdec=self.trdec,
            Diameters=self.Diameters,
            imsize=self.imsize,
            Xmax=self.Xmax,
            lfac=self.lfac,
            distmat=self.distmat,
            Nphf=self.Nphf,
            Np4=self.Np4,
            zooming=self.zooming,
            header=picklable_header(self.header),
            robust=self.robust,
            scan_time_s=self.scan_time,
            noise_entropy=self.noise_entropy,
            **self._corruption_fields(),
        )
        wavelengths = [self.get_channel_wavelength(i)[0] for i in range(self.Nchan)]
        results_per_channel = image_channels(
            self.backend, config, self.skymodel, wavelengths, self.noise, self.track_progress
        )
        cubes, raw_visibility_rows = stack_channels(
            results_per_channel, start=self.Np4, size=self.Nphf
        )
        modelCube, dirtyCube, beam = cubes["model"], cubes["dirty"], cubes["beam"]
        modelVis, dirtyVis = cubes["model_vis"], cubes["dirty_vis"]
        totsampling, uv_mask = cubes["totsampling"], cubes["uv_mask"]
        u, v = cubes["u"], cubes["v"]
        visibility_table = assemble_visibility_table(
            channel_rows=raw_visibility_rows,
            channel_freq_hz=self.spectral_window.frequencies_hz(),
            scan_time_s=float(self.scan_time),
            observation_date=self.observation_date,
            antenna_names=self.antNames,
            antenna_positions_m=self.antPos3D,
            source_name=str(self.header.get("OBJECT", "ALMASimSource")),
            field_ra_rad=float(self.ra),
            field_dec_rad=float(self.dec),
            config_name=self.config_name,
            array_type=self.array_type,
            antenna_diameters_m=np.full(len(self.antNames), self.antenna_diameter_m),
            scan_length_s=self.corruptions.scan_length_s if self.corruptions is not None else 0.0,
            observation_id=self.execution,
            hour_angle_start_rad=float(self.Hcov[0]),
            channel_kernel=(
                spectral_kernel(*self.corruptions.channel_correlation())
                if self.corruptions is not None and self.corruptions.correlates_channels
                else None
            ),
        ).as_dict()

        # Save the results only when explicitly requested.
        if self.persist and self.save_mode != "memory":
            write_interferometry_products(
                self.output_dir,
                idx=self.idx,
                save_mode=self.save_mode,
                header=self.header if isinstance(self.header, fits.Header) else None,
                model_cube=modelCube,
                vis_cube=modelVis,
                dirty_cube=dirtyCube,
                dirty_vis_cube=dirtyVis,
                beam_cube=beam,
                totsampling_cube=totsampling,
                uv_mask_cube=uv_mask,
                u_cube=u,
                v_cube=v,
            )
            self._log(f"Total Flux detected in model cube: {round(np.sum(modelCube), 2)} Jy")
            self._log(f"Total Flux detected in dirty cube: {round(np.sum(dirtyCube), 2)} Jy")

        # Get wavelength and format for middle channel
        self.s_wavelength, self.s_fmtB = self.get_channel_wavelength(self.Nchan // 2)

        # Prepare the simulation results dictionary
        simulation_results = {
            "model_cube": modelCube,
            "model_vis": modelVis,
            "dirty_cube": dirtyCube,
            "dirty_vis": dirtyVis,
            "beam_cube": beam,
            "totsampling_cube": totsampling,
            "uv_mask_cube": uv_mask,
            "u_cube": u,
            "v_cube": v,
            "beam": beam[self.Nchan // 2],
            "totsampling": totsampling[self.Nchan // 2],
            "uv_mask": uv_mask[self.Nchan // 2],
            "u": u[self.Nchan // 2],
            "v": v[self.Nchan // 2],
            "Npix": self.Npix,
            "Np4": self.Np4,
            "Nchan": self.Nchan,
            "gamma": self.gamma,
            "Xaxmax": self.Xaxmax,
            "lfac": self.lfac,
            "UVpixsize": self.UVpixsize,
            "w_min": self.w_min,
            "w_max": self.w_max,
            "plot_dir": self.plot_dir,
            "idx": self.idx,
            "fmtB": self.s_fmtB,
            "wavelength": self.s_wavelength,
            "curzoom": self.curzoom,
            "Nphf": self.Nphf,
            "Xmax": self.Xmax,
            "antPos": self.antPos,
            "Nant": self.Nant,
            "config_name": self.config_name,
            "array_type": self.array_type,
            "antenna_diameter_m": self.antenna_diameter_m,
            "visibility_table": visibility_table,
            # Its image header, with the EPOCH (hour-angle samples) of its track.
            "image_header": self.header,
        }

        return simulation_results

    def track_progress(self, futures: list) -> None:
        """Emit the percentage of the channel futures done until all are."""
        watch_futures(futures, self.progress_signal.emit)
