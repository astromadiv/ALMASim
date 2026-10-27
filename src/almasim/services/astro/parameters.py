"""Simulation parameter writing functions."""

from pathlib import Path

import numpy as np


def write_sim_parameters(
    path: str | Path,
    source_name: str,
    member_ouid: str,
    ra: float,
    dec: float,
    ang_res: float,
    vel_res: float,
    int_time: float,
    band: float,
    band_range: float,
    central_freq: float,
    redshift: float,
    line_fluxes: np.ndarray,
    line_names: list,
    line_frequencies: np.ndarray,
    continum: np.ndarray,
    fov: float,
    beam_size: float,
    cell_size: float,
    n_pix: int,
    n_channels: int,
    snapshot: int | None,
    subhalo: int | None,
    lum_infrared: float,
    fwhm_z: np.ndarray,
    source_type: str,
    fwhm_x: float | None = None,
    fwhm_y: float | None = None,
    angle: float | None = None,
    noise_model: str | None = None,
) -> None:
    """
    Write simulation parameters to a text file.

    Parameters
    ----------
    path : str or Path
        Output file path
    source_name : str
        Source name
    member_ouid : str
        Member OUID
    ra, dec : float
        Right ascension and declination
    ang_res, vel_res : float
        Angular and velocity resolution
    int_time : float
        Integration time
    band : float
        Band number
    band_range, central_freq : float
        Bandwidth and central frequency
    redshift : float
        Source redshift
    line_fluxes : np.ndarray
        Line flux values
    line_names : list
        Line names
    line_frequencies : np.ndarray
        Line frequencies in GHz
    continum : np.ndarray
        Continuum flux array
    fov : float
        Field of view
    beam_size, cell_size : float
        Beam and pixel sizes
    n_pix, n_channels : int
        Cube dimensions
    snapshot, subhalo : int or None
        TNG snapshot and subhalo IDs
    lum_infrared : float
        Infrared luminosity
    fwhm_z : np.ndarray
        FWHM in z direction per line
    source_type : str
        Source type (point, gaussian, extended, etc.)
    fwhm_x, fwhm_y : float or None
        FWHM in x and y directions (for gaussian sources)
    angle : float or None
        Projection angle (for gaussian/extended sources)
    noise_model : str or None
        The noise model's label and SHA-256, when the run used one
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write("Simulation Parameters:\n")
        f.write(f"Source Name: {source_name}\n")
        f.write(f'Member OUID: "{member_ouid}"\n')
        f.write(f"RA: {ra}\n")
        f.write(f"DEC: {dec}\n")
        f.write(f"Band: {band}\n")
        f.write(f"Bandwidth {band_range}\n")
        f.write(f"Band Central Frequency: {central_freq}\n")
        f.write(f"Pixel size: {cell_size}\n")
        f.write(f"Beam Size: {beam_size}\n")
        f.write(f"Fov: {fov}\n")
        f.write(f"Angular Resolution: {ang_res}\n")
        f.write(f"Velocity Resolution: {vel_res}\n")
        f.write(f"Redshift: {redshift}\n")
        f.write(f"Integration Time: {int_time}\n")
        f.write(f"Cube Size: {n_pix} x {n_pix} x {n_channels} pixels\n")
        f.write(f"Mean Continum Flux: {np.mean(continum)}\n")
        f.write(f"Infrared Luminosity: {lum_infrared}\n")

        if source_type == "gaussian":
            if fwhm_x is not None:
                f.write(f"FWHM_x (pixels): {fwhm_x}\n")
            if fwhm_y is not None:
                f.write(f"FWHM_y (pixels): {fwhm_y}\n")

        if source_type in ("gaussian", "extended") and angle is not None:
            f.write(f"Projection Angle: {angle}\n")

        for i, (line_name, line_freq, line_flux) in enumerate(
            zip(line_names, line_frequencies, line_fluxes, strict=False)
        ):
            f.write(
                f"Line: {line_name} - Frequency: {line_freq} GHz "
                f"- Flux: {line_flux} Jy  - Width (Channels): {fwhm_z[i]}\n"
            )

        if snapshot is not None:
            f.write(f"TNG Snapshot ID: {snapshot}\n")
        if subhalo is not None:
            f.write(f"TNG Subhalo ID: {subhalo}\n")
        if noise_model is not None:
            f.write(f"Noise Model: {noise_model}\n")
