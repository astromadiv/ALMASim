"""Baseline preparation and management for interferometry."""

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time


def pairwise_baselines(positions_xyz_m: np.ndarray) -> np.ndarray:
    """Compute all unique antenna-pair baselines from Cartesian positions."""
    n_antennas = positions_xyz_m.shape[0]
    baselines = np.zeros((n_antennas * (n_antennas - 1) // 2, 3), dtype=np.float64)
    count = 0
    for i in range(n_antennas):
        for j in range(i):
            baselines[count, :] = positions_xyz_m[i] - positions_xyz_m[j]
            count += 1
    return baselines


def _reference_location_for_positions(positions: np.ndarray) -> EarthLocation:
    """Build a local reference EarthLocation from the provided antenna positions."""
    if positions.shape[1] == 3:
        return EarthLocation(
            x=np.mean(positions[:, 0]) * u.m,
            y=np.mean(positions[:, 1]) * u.m,
            z=np.mean(positions[:, 2]) * u.m,
        )

    return EarthLocation.from_geodetic(
        lon=np.mean(positions[:, 1]) * u.deg,
        lat=np.mean(positions[:, 0]) * u.deg,
        height=0.0 * u.m,
    )


def generate_via_astropy(
    antenna_positions_m: np.ndarray,
    ra: u.Quantity,
    dec: u.Quantity,
    time: Time,
) -> np.ndarray:
    """Project antenna positions into UVW coordinates for a phase center.

    Parameters
    ----------
    antenna_positions_m : np.ndarray
        Either geocentric positions with shape (N, 3) as [x, y, z] in meters,
        or geodetic positions with shape (N, 2) as [lat_deg, lon_deg].
    """
    positions = np.asarray(antenna_positions_m, dtype=np.float64)
    if positions.ndim != 2:
        raise ValueError("antenna_positions_m must be a 2D array")

    if positions.shape[1] == 3:
        antpos = EarthLocation(
            x=positions[:, 0] * u.m,
            y=positions[:, 1] * u.m,
            z=positions[:, 2] * u.m,
        )
    elif positions.shape[1] == 2:
        lat_deg = positions[:, 0]
        lon_deg = positions[:, 1]
        if np.any(lat_deg < -90.0) or np.any(lat_deg > 90.0):
            raise ValueError("Latitude must be within [-90, 90] degrees")
        if np.any(lon_deg < -360.0) or np.any(lon_deg > 360.0):
            raise ValueError("Longitude must be within [-360, 360] degrees")
        antpos = EarthLocation.from_geodetic(
            lon=lon_deg * u.deg,
            lat=lat_deg * u.deg,
            height=np.zeros_like(lat_deg) * u.m,
        )
    else:
        raise ValueError("antenna_positions_m must have shape (N, 3) or (N, 2)")

    reference_location = _reference_location_for_positions(positions)

    alma_p, alma_v = reference_location.get_gcrs_posvel(time)
    antpos_gcrs = coord.GCRS(
        antpos.get_gcrs_posvel(time)[0],
        obstime=time,
        obsgeoloc=alma_p,
        obsgeovel=alma_v,
    )

    pnt = coord.SkyCoord(ra, dec, frame="icrs")
    frame_uvw = pnt.transform_to(antpos_gcrs).skyoffset_frame()
    antpos_uvw_xyz = antpos_gcrs.transform_to(frame_uvw).cartesian.xyz.to_value(u.m).T
    return pairwise_baselines(antpos_uvw_xyz)


def prepare_baselines(
    Nant: int,
    nH: int,
    Hcov: list[float],
    hour_angles: np.ndarray | None = None,
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list,
    np.ndarray,
    np.ndarray,
    tuple[int, int],
]:
    """
    Prepare baselines for an interferometer array.

    Parameters
    ----------
    Nant : int
        Number of antennas
    nH : int
        Number of hour angle samples
    Hcov : list[float]
        Hour angle coverage [start, end]
    hour_angles : np.ndarray, optional
        Each sample's hour angle (rad, ``nH`` of them); by default ``nH``
        evenly spaced over ``Hcov``

    Returns
    -------
    tuple
        (Nbas, B, basnum, basidx, antnum, Gains, Noise, H, u, v, ravelDims)
        Baseline parameters
    """
    if Nant < 2:
        raise ValueError("Number of antennas must be at least 2")
    if nH < 1:
        raise ValueError("Number of hour angle samples must be at least 1")
    if len(Hcov) < 2:
        raise ValueError("Hour angle coverage must have [start, end]")

    Nbas = Nant * (Nant - 1) // 2
    NBmax = Nbas

    # B stores 3D baseline vector components (x, y, z) for each baseline
    B = np.zeros((NBmax, 3), dtype=np.float32)
    # Use int16 to support more antennas (int8 max is 127 baselines)
    basnum = np.zeros((Nant, Nant - 1), dtype=np.int16)
    basidx = np.zeros((Nant, Nant), dtype=np.int16)
    antnum = np.zeros((NBmax, 2), dtype=np.int16)
    Gains = np.ones((Nbas, nH), dtype=np.complex64)
    Noise = np.zeros((Nbas, nH), dtype=np.complex64)

    if hour_angles is not None:
        H_angles = np.asarray(hour_angles, dtype=np.float64)
        if H_angles.shape != (nH,):
            raise ValueError(f"hour_angles must hold nH={nH} values, not {H_angles.shape}")
    else:
        H_angles = np.linspace(Hcov[0], Hcov[1], nH)
    H = [np.sin(H_angles), np.cos(H_angles)]

    # Generate baseline pairs
    bi = 0
    nii = np.zeros(Nant, dtype=np.int16)

    for n1 in range(Nant - 1):
        for n2 in range(n1 + 1, Nant):
            basnum[n1, nii[n1]] = bi
            basnum[n2, nii[n2]] = bi
            basidx[n1, n2] = bi
            antnum[bi] = [n1, n2]
            nii[n1] += 1
            nii[n2] += 1
            bi += 1

    u = np.zeros((NBmax, nH))
    v = np.zeros((NBmax, nH))
    ravelDims = (NBmax, nH)
    return Nbas, B, basnum, basidx, antnum, Gains, Noise, H, u, v, ravelDims


def set_noise(
    noise: float, Noise: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Set noise values for baselines.

    Parameters
    ----------
    noise : float
        Noise standard deviation
    Noise : np.ndarray
        Array to fill with noise values (complex)

    Returns
    -------
    np.ndarray
        Noise array with random complex noise
    """
    if noise < 0:
        raise ValueError("Noise must be non-negative")
    rng = rng if rng is not None else np.random.default_rng()
    shape = Noise.shape
    Noise[:] = rng.normal(loc=0.0, scale=noise, size=shape) + 1.0j * rng.normal(
        loc=0.0, scale=noise, size=shape
    )
    return Noise


def set_baselines(
    Nbas: int,
    antnum: np.ndarray,
    B: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    antPos: list,
    trlat: list[float],
    trdec: list[float],
    H: list,
    wavelength: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Set baseline vectors and u-v coordinates.

    Parameters
    ----------
    Nbas : int
        Number of baselines
    antnum : np.ndarray
        Antenna number pairs for each baseline
    B : np.ndarray
        Baseline vector array (output)
    u, v : np.ndarray
        UV coordinates (output)
    antPos : list
        Antenna positions [[x0, y0], [x1, y1], ...]
    trlat : list[float]
        Transformation latitude [sin, cos]
    trdec : list[float]
        Transformation declination [sin, cos]
    H : list
        Hour angle [sin(H), cos(H)]
    wavelength : list[float]
        Wavelengths [min, max, mean]

    Returns
    -------
    tuple
        (B, u, v) baseline vectors and UV coordinates
    """
    if len(wavelength) < 3:
        raise ValueError("wavelength must have at least 3 elements")
    if len(trlat) < 2 or len(trdec) < 2:
        raise ValueError("trlat and trdec must have at least 2 elements")

    mean_wavelength = wavelength[2]

    for currBas in range(Nbas):
        n1, n2 = antnum[currBas]

        # Calculate baseline vector components
        dx = antPos[n2][0] - antPos[n1][0]
        dy = antPos[n2][1] - antPos[n1][1]

        B[currBas, 0] = -dy * trlat[0] / mean_wavelength
        B[currBas, 1] = dx / mean_wavelength
        B[currBas, 2] = dy * trlat[1] / mean_wavelength

        # Calculate UV coordinates
        u[currBas, :] = -(B[currBas, 0] * H[0] + B[currBas, 1] * H[1])
        v[currBas, :] = (
            -B[currBas, 0] * trdec[0] * H[1]
            + B[currBas, 1] * trdec[0] * H[0]
            + trdec[1] * B[currBas, 2]
        )
    return B, u, v
