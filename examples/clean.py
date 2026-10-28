from pathlib import Path

import numpy as np
import radler
from astropy.io import fits


def _read_float32_plane(path: str) -> np.ndarray:
    data = fits.getdata(path, ext=0, memmap=False)
    # Radler expects dense 2D float32 arrays.
    return np.asarray(np.squeeze(data), dtype=np.float32, order="C")


def run_clean() -> bool:
    dirty_path = "./wsclean_like_image.fits"
    psf_path = "./psf.fits"

    header = fits.getheader(dirty_path, ext=0)

    residual = _read_float32_plane(dirty_path)
    psf = _read_float32_plane(psf_path)
    model = np.zeros_like(residual, dtype=np.float32, order="C")

    if residual.ndim != 2 or psf.ndim != 2:
        raise ValueError(f"Expected 2D images, got residual={residual.shape}, psf={psf.shape}")
    if residual.shape != psf.shape:
        raise ValueError(f"Residual and PSF shapes must match: {residual.shape} != {psf.shape}")

    settings = radler.Settings()
    settings.trimmed_image_width, settings.trimmed_image_height = psf.shape
    # CDELT is in degrees; Radler takes the pixel scale in radians.
    settings.pixel_scale.x = np.deg2rad(abs(float(header.get("CDELT1", 0.0))))
    settings.pixel_scale.y = np.deg2rad(abs(float(header.get("CDELT2", 0.0))))
    settings.minor_iteration_count = 50000
    settings.major_iteration_count = 1
    settings.auto_threshold_sigma = 1.0
    # major_loop_gain < 1 lets perform() return True when the in-iteration
    # peak drops by this fraction, signalling a new prediction-gridding round.
    # Without it the algorithm always runs to full completion → returns False.
    settings.major_loop_gain = 0.9

    arcsec_to_radians = np.pi / 3600.0 / 180.0
    radler_object = radler.Radler(
        settings,
        psf,
        residual,
        model,
        10.0 * arcsec_to_radians,
        radler.Polarization.stokes_i,
    )

    reached_major_threshold = radler_object.perform(1)
    print(f"perform() returned: {reached_major_threshold}")
    print(f"Total model flux: {float(model.sum())}")

    # Save the model image, reusing the dirty-image header for WCS.
    model_path = Path(dirty_path).with_name(Path(dirty_path).stem + "_model.fits")
    model_header = header.copy()
    model_header["BTYPE"] = "Model"
    model_header["HISTORY"] = "Produced by radler deconvolution (clean.py)"
    fits.writeto(str(model_path), model, header=model_header, overwrite=True)
    print(f"Model saved to: {model_path}")

    # Save the updated residual image with the same WCS information.
    residual_path = Path(dirty_path).with_name(Path(dirty_path).stem + "_residual.fits")
    residual_header = header.copy()
    residual_header["BTYPE"] = "Residual"
    residual_header["HISTORY"] = "Residual after radler deconvolution (clean.py)"
    fits.writeto(str(residual_path), residual, header=residual_header, overwrite=True)
    print(f"Residual saved to: {residual_path}")

    return reached_major_threshold


if __name__ == "__main__":
    run_clean()
