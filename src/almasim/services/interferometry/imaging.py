"""Image processing functions for interferometry."""

import numbers

import astropy.units as u
import ducc0 as _ducc0
import numpy as np
from astropy.io import fits
from astropy.time import Time
from scipy.ndimage import zoom

from ..instrument.constants import ARCSEC_PER_RAD, SPEED_OF_LIGHT_M_S
from ..instrument.primary_beam import primary_beam_response
from .baselines import generate_via_astropy, prepare_baselines, set_baselines, set_noise
from .predict import ALMASIM_NTHREADS, _ducc0_with_wgridder_fallback, predict_visibilities
from .visibility import build_channel_visibility_rows


def prepare_2d_arrays(Npix: int) -> tuple:
    """Prepare 2D arrays for interferometric processing."""
    beam = np.zeros((Npix, Npix), dtype=np.float32)
    totsampling = np.zeros((Npix, Npix), dtype=np.float32)
    dirtymap = np.zeros((Npix, Npix), dtype=np.float32)
    noisemap = np.zeros((Npix, Npix), dtype=np.complex64)
    robustsamp = np.zeros((Npix, Npix), dtype=np.float32)
    Gsampling = np.zeros((Npix, Npix), dtype=np.complex64)
    Grobustsamp = np.zeros((Npix, Npix), dtype=np.complex64)
    GrobustNoise = np.zeros((Npix, Npix), dtype=np.complex64)
    return (
        beam,
        totsampling,
        dirtymap,
        noisemap,
        robustsamp,
        Gsampling,
        Grobustsamp,
        GrobustNoise,
    )


def _grid_uv(
    Nbas: int,
    totsampling: np.ndarray,
    Gsampling: np.ndarray,
    noisemap: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    Nphf: int,
    Gains: np.ndarray,
    Noise: np.ndarray,
    robust: float,
    nH: int,
    imsize: float,
) -> tuple:
    """Grid visibility measurements onto a discrete UV plane."""
    bas2change = range(Nbas)
    pixpos = [[] for nb in bas2change]
    totsampling[:] = 0.0
    Gsampling[:] = 0.0
    noisemap[:] = 0.0
    UVpixsize = 2.0 / (imsize * np.pi / 180.0 / 3600.0)
    baseline_phases = {}

    for nb in bas2change:
        pixU = np.rint(u[nb] / UVpixsize).flatten().astype(np.int32)
        pixV = np.rint(v[nb] / UVpixsize).flatten().astype(np.int32)
        goodpix = np.where(np.logical_and(np.abs(pixU) < Nphf, np.abs(pixV) < Nphf))[0]

        baseline_phases[nb] = np.angle(Gains[nb, goodpix])

        pU = pixU[goodpix] + Nphf
        pV = pixV[goodpix] + Nphf
        mU = -pixU[goodpix] + Nphf
        mV = -pixV[goodpix] + Nphf
        pixpos[nb] = [np.copy(pU), np.copy(pV), np.copy(mU), np.copy(mV)]

        for pi, gp in enumerate(goodpix):
            gabs = np.abs(Gains[nb, gp])
            pVi = pV[pi]
            mUi = mU[pi]
            mVi = mV[pi]
            pUi = pU[pi]
            totsampling[pVi, mUi] += 1.0
            totsampling[mVi, pUi] += 1.0
            Gsampling[pVi, mUi] += Gains[nb, gp]
            Gsampling[mVi, pUi] += np.conjugate(Gains[nb, gp])
            noisemap[pVi, mUi] += Noise[nb, gp] * gabs
            noisemap[mVi, pUi] += np.conjugate(Noise[nb, gp]) * gabs

    robfac = (5.0 * 10.0 ** (-robust)) ** 2.0 * (2.0 * Nbas * nH) / np.sum(totsampling**2.0)
    return (
        pixpos,
        totsampling,
        Gsampling,
        noisemap,
        UVpixsize,
        baseline_phases,
        bas2change,
        robfac,
    )


def set_beam(
    robfac: float,
    totsampling: np.ndarray,
    robustsamp: np.ndarray,
    Gsampling: np.ndarray,
    GrobustNoise: np.ndarray,
    Grobustsamp: np.ndarray,
    noisemap: np.ndarray,
    beam: np.ndarray,
    Nphf: int,
) -> tuple:
    """Calculate dirty beam with robust weighting."""
    denom = 1.0 + robfac * totsampling
    robustsamp[:] = totsampling / denom
    Grobustsamp[:] = Gsampling / denom
    GrobustNoise[:] = noisemap / denom
    beam[:] = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(robustsamp))).real
    beamScale = np.max(beam[Nphf : Nphf + 1, Nphf : Nphf + 1])
    beam[:] /= beamScale
    return beam, beamScale, robustsamp, GrobustNoise, Grobustsamp


def check_lfac(Xmax: float, wavelength: list[float], lfac: float) -> tuple:
    """Check and adjust length factor for UV coordinates."""
    mw = 2.0 * Xmax / wavelength[2] / lfac
    if mw < 0.1 and lfac == 1.0e6:
        lfac = 1.0e3
    elif mw >= 100.0 and lfac == 1.0e3:
        lfac = 1.0e6
    ulab = r"U (k$\lambda$)"
    vlab = r"V (k$\lambda$)"
    return lfac, ulab, vlab


def _prepare_model(Npix: int, img: np.ndarray, Nphf: int, Np4: int, zooming: int) -> tuple:
    """Prepare model image for processing."""
    modelim = [np.zeros((Npix, Npix), dtype=np.float32) for i in [0, 1]]
    modelimTrue = np.zeros((Npix, Npix), dtype=np.float32)
    dims = np.shape(img)
    d1 = img.shape[0]
    if d1 == Nphf:
        sh0 = (Nphf - dims[0]) // 2
        sh1 = (Nphf - dims[1]) // 2
        zoomimg = img
        modelimTrue[
            sh0 + Np4 : sh0 + Np4 + dims[0],
            sh1 + Np4 : sh1 + Np4 + dims[1],
        ] += zoomimg
    else:
        zoomimg = zoom(img, float(Nphf) / d1)
        zdims = np.shape(zoomimg)
        zd0 = min(zdims[0], Nphf)
        zd1 = min(zdims[1], Nphf)
        sh0 = (Nphf - zdims[0]) // 2
        sh1 = (Nphf - zdims[1]) // 2
        modelimTrue[
            sh0 + Np4 : sh0 + Np4 + zd0,
            sh1 + Np4 : sh1 + Np4 + zd1,
        ] += zoomimg[:zd0, :zd1]
    modelimTrue[modelimTrue < 0.0] = 0.0
    return modelim, modelimTrue


def add_thermal_noise(
    img: np.ndarray, noise: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Add thermal noise to image."""
    rng = rng if rng is not None else np.random.default_rng()
    return img + rng.normal(loc=0.0, scale=noise, size=np.shape(img))


def set_primary_beam(
    header: fits.Header,
    distmat: np.ndarray,
    wavelength: list[float],
    Diameters: list[float],
    modelim: list,
    modelimTrue: np.ndarray,
) -> tuple:
    """Apply the primary beam to the model and return the model FFT.

    ``distmat`` is minus the squared distance (arcsec^2) of every pixel from
    the pointing centre; the response is the instrument's Airy model at the
    channel's wavelength (``wavelength[2]``, in km) for ``Diameters[0]`` m.
    """
    radius_rad = np.sqrt(np.maximum(-distmat, 0.0)) / ARCSEC_PER_RAD
    frequency_hz = SPEED_OF_LIGHT_M_S / (wavelength[2] * 1e3)  # wavelengths are in km here
    beamImg = primary_beam_response(radius_rad, frequency_hz, Diameters[0])
    modelim[0][:] = modelimTrue * beamImg
    modelfft = np.fft.fft2(np.fft.fftshift(modelim[0]))
    return modelfft, modelim


def observe(
    dirtymap: np.ndarray,
    GrobustNoise: np.ndarray,
    modelfft: np.ndarray,
    Grobustsamp: np.ndarray,
    beamScale: float,
) -> tuple:
    """Observe model through interferometer and create dirty map."""
    dirtymap[:] = (
        np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(GrobustNoise) + modelfft * np.fft.ifftshift(Grobustsamp))
        )
    ).real
    dirtymap /= beamScale
    modelvis = np.fft.fftshift(modelfft)
    dirtyvis = np.fft.fftshift(
        np.fft.ifftshift(GrobustNoise) + modelfft * np.fft.ifftshift(Grobustsamp)
    )
    return dirtymap, modelvis, dirtyvis


def image_channel(
    img: np.ndarray,
    wavelength: list[float],
    Npix: int,
    Nant: int,
    Hcov: list[float],
    nH: int,
    noise: float,
    antPos: list,
    robfac: float,
    trlat: list[float],
    trdec: list[float],
    Diameters: list[float],
    imsize: float,
    Xmax: float,
    lfac: float,
    distmat: np.ndarray,
    Nphf: int,
    Np4: int,
    zooming: int,
    header: fits.Header | dict,
    robust: float,
    scan_time_s: float | None = None,
    rng: np.random.Generator | None = None,
    gains: np.ndarray | None = None,
    noise_samples: np.ndarray | None = None,
    noise_scale: np.ndarray | None = None,
) -> tuple:
    """Process a single channel through interferometric simulation.

    ``rng`` draws the visibility noise (fresh entropy when None).
    ``gains`` (baselines x integrations): what the calibration left on each
    visibility (all ones when None); ``noise_samples``: this channel's noise,
    when drawn outside (correlated with its neighbours), instead of white
    noise from ``rng``; ``noise_scale`` (baselines): each baseline's noise
    over the channel's (antennas whose noise differs), all ones when None.
    """
    rng = rng if rng is not None else np.random.default_rng()
    # Convert header dict back to FITS Header if needed (for pickling compatibility)
    if isinstance(header, dict):
        header = fits.Header(header)

    (
        beam,
        totsampling,
        dirtymap,
        noisemap,
        robustsamp,
        Gsampling,
        Grobustsamp,
        GrobustNoise,
    ) = prepare_2d_arrays(Npix)
    Nbas, B, basnum, basidx, antnum, Gains, Noise, H, u, v, ravelDims = prepare_baselines(
        Nant, nH, Hcov
    )
    if noise_samples is not None:
        Noise[:] = noise_samples
    else:
        Noise = set_noise(noise, Noise, rng)
    if noise_scale is not None:
        Noise *= np.asarray(noise_scale, dtype=np.float32)[:, None]
    if gains is not None:
        Gains[:] = gains
    B, u, v = set_baselines(Nbas, antnum, B, u, v, antPos, trlat, trdec, H, wavelength)
    (
        pixpos,
        totsampling,
        Gsampling,
        noisemap,
        UVpixsize,
        baseline_phases,
        bas2change,
        robfac,
    ) = _grid_uv(
        Nbas,
        totsampling,
        Gsampling,
        noisemap,
        u,
        v,
        Nphf,
        Gains,
        Noise,
        robust,
        nH,
        imsize,
    )
    beam, beamScale, robustsamp, GrobustNoise, Grobustsamp = set_beam(
        robfac,
        totsampling,
        robustsamp,
        Gsampling,
        GrobustNoise,
        Grobustsamp,
        noisemap,
        beam,
        Nphf,
    )
    lfac, ulab, vlab = check_lfac(Xmax, wavelength, lfac)
    modelim, modelimTrue = _prepare_model(Npix, img, Nphf, Np4, zooming)
    modelfft, modelim = set_primary_beam(
        header, distmat, wavelength, Diameters, modelim, modelimTrue
    )
    w = (
        B[:, 0][:, None] * trdec[1] * H[1][None, :]
        - B[:, 1][:, None] * trdec[1] * H[0][None, :]
        + trdec[0] * B[:, 2][:, None]
    )
    raw_visibility = build_channel_visibility_rows(
        modelfft=modelfft,
        gains=Gains,
        noise_samples=Noise,
        u_waves=u,
        v_waves=v,
        w_waves=w,
        antnum=antnum,
        nH=nH,
        Nphf=Nphf,
        uv_pixsize=UVpixsize,
        noise_std=float(noise),
        mean_wavelength_m=float(wavelength[2] * 1e3),
        noise_scale=noise_scale,
    )
    dirtymap, modelvis, dirtyvis = observe(dirtymap, GrobustNoise, modelfft, Grobustsamp, beamScale)
    del noisemap, robustsamp, Gsampling, Grobustsamp, GrobustNoise
    del Nbas, B, basnum, basidx, antnum, Gains, Noise, H, ravelDims
    del pixpos, baseline_phases, bas2change
    return (
        modelim[0],
        dirtymap,
        modelvis,
        dirtyvis,
        u,
        v,
        beam,
        totsampling,
        raw_visibility,
    )


def _get_xy_from_single_or_multi(couple_or_value):
    if isinstance(couple_or_value, numbers.Number):
        return couple_or_value, couple_or_value
    elif hasattr(couple_or_value, "__iter__") and len(couple_or_value) == 2:
        return couple_or_value
    else:
        raise ValueError(f"{couple_or_value} is neither a couple or a value")


def exact_weighs(uvw, n_pix, pixsize, wavelengths, nthreads=ALMASIM_NTHREADS):
    """
    Compute the most accurate weights possible for the problem
    However this is expensive and an approximation should be used

    ex. briggs or uniform weighting
    """
    # We assume here that the uvw first axis length
    # is (n_baselines * n_times, 3)
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    n_pix_x, n_pix_y = _get_xy_from_single_or_multi(n_pix)
    pixsize_x, pixsize_y = _get_xy_from_single_or_multi(pixsize)
    n_times_baselines = uvw.shape[0]
    n_frequencies = len(wavelengths)
    frequencies = SPEED_OF_LIGHT_M_S / wavelengths
    # Sort frequencies in ascending order as required by ducc0
    freq_sort_idx = np.argsort(frequencies)
    frequencies = frequencies[freq_sort_idx]
    ones_visibilities = np.ones((n_times_baselines, n_frequencies), dtype=np.complex128)

    dirty_image = _ducc0_with_wgridder_fallback(
        _ducc0.wgridder.vis2dirty,
        uvw=uvw,
        freq=frequencies,
        vis=ones_visibilities,
        npix_x=n_pix_x,
        npix_y=n_pix_y,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        epsilon=1.0e-8,
        do_wgridding=True,
        nthreads=nthreads,
    )
    weights = _ducc0_with_wgridder_fallback(
        _ducc0.wgridder.dirty2vis,
        uvw=uvw,
        freq=frequencies,
        dirty=dirty_image,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        epsilon=1.0e-8,
        do_wgridding=True,
        nthreads=nthreads,
    )
    # Convert to float if it's complex (take magnitude)
    if np.iscomplexobj(weights):
        weights = np.abs(weights).astype(np.float32)
    else:
        weights = weights.astype(np.float32)
    return weights


def image_based_predict(
    uvw, pix_size, image, wavelenghts, weight_func=exact_weighs, n_threads=ALMASIM_NTHREADS
):
    n_pix = image.shape[:2]
    pixsize_x, pixsize_y = _get_xy_from_single_or_multi(pix_size)

    weights = weight_func(uvw, n_pix, pix_size, wavelenghts, nthreads=n_threads)
    wavelenghts = np.asarray(wavelenghts, dtype=np.float64)
    frequencies = SPEED_OF_LIGHT_M_S / wavelenghts
    # Sort frequencies in ascending order as required by ducc0
    freq_sort_idx = np.argsort(frequencies)
    frequencies = frequencies[freq_sort_idx]
    visibilities = predict_visibilities(
        uvw, frequencies, image, pixsize_x, pixsize_y, weights=weights, nthreads=n_threads
    )
    return visibilities, weights


def image_channel_ducc0(
    img: np.ndarray,
    wavelengths: list[float],
    Npix: int,
    Nant: int,
    Hcov: list[float],
    nH: int,
    noise: float,
    antPos: list,
    robfac: float,
    trlat: list[float],
    trdec: list[float],
    Diameters: list[float],
    imsize: float,
    Xmax: float,
    lfac: float,
    distmat: np.ndarray,
    Nphf: int,
    Np4: int,
    zooming: int,
    header: fits.Header | dict,
    robust: float,
    scan_time_s: float | None = None,
):
    """Process a single channel through interferometric simulation."""
    # Convert header dict back to FITS Header if needed (for pickling compatibility)
    if isinstance(header, dict):
        header = fits.Header(header)
    date_obs = header.get("DATE-OBS", header.get("DATEOBS"))
    obs_time = Time(date_obs, format="isot", scale="utc") if date_obs else Time.now()

    ra_deg = header.get("OBSRA", header.get("CRVAL1", header.get("RA")))
    dec_deg = header.get("OBSDEC", header.get("CRVAL2", header.get("DEC")))
    if ra_deg is None or dec_deg is None:
        raise ValueError(
            "Header must contain pointing coordinates in OBSRA/OBSDEC or CRVAL1/CRVAL2"
        )

    antenna_positions = np.asarray(antPos, dtype=np.float64)
    if antenna_positions.ndim != 2 or antenna_positions.shape[1] not in (2, 3):
        raise ValueError("antPos must be a 2D array-like with shape (N, 2) or (N, 3)")

    uvw = generate_via_astropy(
        antenna_positions,
        float(ra_deg) * u.deg,
        float(dec_deg) * u.deg,
        obs_time,
    )
    modelim, modelimTrue = _prepare_model(Npix, img, Nphf, Np4, zooming)
    modelfft, modelim = set_primary_beam(
        header, distmat, wavelengths, Diameters, modelim, modelimTrue
    )
    pixel_scale_arcsec = float(imsize) / float(Npix)
    pixel_scale_rad = np.deg2rad(pixel_scale_arcsec / 3600.0)
    pixsize_x = pixel_scale_rad
    pixsize_y = pixel_scale_rad
    frequencies = SPEED_OF_LIGHT_M_S / np.asarray(wavelengths, dtype=np.float64)
    nthreads = ALMASIM_NTHREADS

    visibilities, weights = image_based_predict(
        uvw,
        (pixsize_x, pixsize_y),
        modelim[0],
        wavelengths,
        weight_func=exact_weighs,
        n_threads=nthreads,
    )

    frequency_order = np.argsort(frequencies)
    frequencies = frequencies[frequency_order]
    visibilities = np.asarray(visibilities)[..., frequency_order]
    weights = np.asarray(weights)[..., frequency_order]
    center_index = len(frequencies) // 2

    dirtymap = _ducc0_with_wgridder_fallback(
        _ducc0.wgridder.vis2dirty,
        uvw=uvw,
        freq=frequencies,
        vis=visibilities,
        npix_x=Npix,
        npix_y=Npix,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        epsilon=1.0e-6,
        do_wgridding=True,
        nthreads=nthreads,
    )

    # Extract u, v from uvw (ducc0 path doesn't compute these separately in classic way)
    u_vals = uvw[:, 0]
    v_vals = uvw[:, 1]

    # Initialize standard output arrays to match image_channel signature
    beam = np.zeros((Npix, Npix), dtype=np.float32)
    totsampling = np.ones((Npix, Npix), dtype=np.float32)

    modelvis = visibilities[..., center_index]
    dirtyvis = np.fft.fft2(dirtymap)
    antenna_pairs = np.array([(i, j) for i in range(len(antenna_positions)) for j in range(i)])
    nrows = uvw.shape[0]
    if antenna_pairs.shape[0] != nrows:
        raise ValueError("UVW rows do not match the expected number of antenna baselines")
    raw_visibility = {
        "uvw_m": uvw,
        "antenna1": antenna_pairs[:, 0].astype(np.int32),
        "antenna2": antenna_pairs[:, 1].astype(np.int32),
        "time_index": np.zeros(nrows, dtype=np.int32),
        "valid": np.ones(nrows, dtype=np.bool_),
        "model_data": np.asarray(visibilities[:, center_index], dtype=np.complex64),
        "data": np.asarray(visibilities[:, center_index], dtype=np.complex64),
        "weight": np.asarray(weights[:, center_index], dtype=np.float32),
        "sigma": np.full(nrows, max(float(noise), 1e-12), dtype=np.float32),
    }

    return (
        modelim[0],
        dirtymap,
        modelvis,
        dirtyvis,
        u_vals,
        v_vals,
        beam,
        totsampling,
        raw_visibility,
    )
