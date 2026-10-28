import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import ducc0
    import radler
    import xarray

    return ducc0, radler, xarray


@app.cell
def _(mo):
    file_browser = mo.ui.file_browser(
        initial_path="/Data", multiple=False, selection_mode="directory"
    )
    return (file_browser,)


@app.cell
def _(file_browser):
    file_browser
    return


@app.cell
def _(file_browser, xarray):
    dataset = file_browser.path(index=0)
    dset = xarray.open_datatree(dataset)
    return (dset,)


@app.cell
def _(dset, mo):
    dset_selector = mo.ui.dropdown(options=dset.keys())
    dset_selector
    return (dset_selector,)


@app.cell
def _(dset, dset_selector):
    chunked_dset = dset[dset_selector.value].chunk({"time": 50})
    chunked_uvw = chunked_dset["VISIBILITY"]
    chunked_data = chunked_dset["VISIBILITY"]
    chunked_I = chunked_data.sel(polarization="XX") + chunked_data.sel(polarization="YY")
    chunked_frequency = chunked_I.frequency
    return (chunked_dset,)


@app.cell
def _(dset, dset_selector):
    def read_data():
        uvw = dset[dset_selector.value]["UVW"]
        frequencies = dset[dset_selector.value]["frequency"]
        data = dset[dset_selector.value]["VISIBILITY"]
        flag = dset[dset_selector.value]["FLAG"]

        I = data.sel(polarization="XX") + data.sel(polarization="YY")
        return uvw, frequencies, I

    uvw, frequencies, I = read_data()
    return I, frequencies, uvw


@app.cell
def _(I, frequencies, uvw):
    import numpy as np

    arcsec_to_radians = np.pi / 180 / 3600
    pix_scale_rad = np.array((10, 10), dtype=np.float64) * arcsec_to_radians
    image_size = (1024, 1024)
    epsilon = 1.0e-6

    uvw_array = np.array(uvw.compute().data)

    uvw_array = uvw_array.reshape((uvw_array.shape[0] * uvw_array.shape[1], uvw_array.shape[2]))
    vis_array = np.array(I.compute().data)[:, :, :50]

    vis_array = vis_array.reshape((vis_array.shape[0] * vis_array.shape[1], vis_array.shape[2]))
    freq_array = np.array(frequencies)[:50]

    return (
        arcsec_to_radians,
        epsilon,
        freq_array,
        image_size,
        np,
        pix_scale_rad,
        uvw_array,
        vis_array,
    )


@app.cell
def _(
    ducc0,
    epsilon,
    freq_array,
    image_size,
    np,
    pix_scale_rad,
    uvw_array,
    vis_array,
):
    # Computes weights
    def compute_weights():
        ones = np.ones_like(vis_array)
        psf_unweighted = ducc0.wgridder.vis2dirty(
            uvw=uvw_array,
            vis=ones,
            freq=freq_array,
            npix_x=image_size[0],
            npix_y=image_size[1],
            pixsize_x=pix_scale_rad[0],
            pixsize_y=pix_scale_rad[1],
            do_wgridding=True,
            epsilon=epsilon,
            nthreads=10,
        )
        weights = ducc0.wgridder.dirty2vis(
            uvw=uvw_array,
            dirty=psf_unweighted,
            freq=freq_array,
            pixsize_x=pix_scale_rad[0],
            pixsize_y=pix_scale_rad[1],
            do_wgridding=True,
            epsilon=epsilon,
            nthreads=10,
        )

        weights = 1.0 / np.abs(weights)
        psf = ducc0.wgridder.vis2dirty(
            uvw=uvw_array,
            vis=ones,
            wgt=weights,
            freq=freq_array,
            npix_x=image_size[0],
            npix_y=image_size[1],
            pixsize_x=pix_scale_rad[0],
            pixsize_y=pix_scale_rad[1],
            do_wgridding=True,
            epsilon=epsilon,
            nthreads=10,
        )
        return weights, psf

    weights, psf = compute_weights()
    return psf, weights


@app.cell
def _():
    return


@app.cell
def _(
    ducc0,
    epsilon,
    freq_array,
    image_size,
    pix_scale_rad,
    uvw_array,
    vis_array,
    weights,
):
    dirty_image = ducc0.wgridder.vis2dirty(
        uvw=uvw_array,
        vis=vis_array,
        wgt=weights,
        freq=freq_array,
        npix_x=image_size[0],
        npix_y=image_size[1],
        pixsize_x=pix_scale_rad[0],
        pixsize_y=pix_scale_rad[1],
        do_wgridding=True,
        epsilon=epsilon,
        nthreads=10,
    )
    return (dirty_image,)


@app.cell
def _(chunked_dset):
    (ra, dec) = chunked_dset["field_and_source_base_xds"]["FIELD_PHASE_CENTER_DIRECTION"][0]
    return dec, ra


@app.cell
def _(dec, dirty_image, image_size, np, pix_scale_rad, psf, ra):
    from astropy.io import fits
    from astropy.wcs import WCS

    # --- user / WSClean-like inputs ---

    # Image size
    nx, ny = image_size  # NAXIS1, NAXIS2

    # Phase centre in degrees (from MS field centre or -phasecentre)
    ra_deg = ra / np.pi * 180  # CRVAL1
    dec_deg = dec / np.pi * 180  # CRVAL2

    # Pixel scale in degrees, as CDELT holds it (pix_scale_rad is in radians)
    pixscale_deg = np.degrees(pix_scale_rad)

    # --- construct WCS like WSClean ---

    w = WCS(naxis=2)

    # Reference pixel at image centre (WSClean default)
    w.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]

    # World coordinate at reference pixel = phase centre, freq centre
    w.wcs.crval = [ra_deg, dec_deg]

    # Pixel increments:
    #  - RA: negative so RA increases to the left
    #  - Dec: positive, frequency: positive
    w.wcs.cdelt = np.array(
        [
            -pixscale_deg[0],  # CDELT1
            pixscale_deg[1],  # CDELT2
        ]
    )

    # Axis types and units
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.cunit = ["deg", "deg"]

    # --- write to FITS header ---

    header = w.to_header()
    header["NAXIS"] = 2
    header["NAXIS1"] = nx
    header["NAXIS2"] = ny

    hdu = fits.PrimaryHDU(header=header, data=dirty_image)
    hdu.writeto("wsclean_like_image.fits", overwrite=True)
    psf_hdu = fits.PrimaryHDU(data=psf)
    psf_hdu.writeto("psf.fits", overwrite=True)
    return


@app.cell
def _(dirty_image, mo, psf):
    import holoviews as hv
    import matplotlib.pyplot as plt
    from astropy.visualization import ImageNormalize, SqrtStretch, ZScaleInterval

    # Create an ImageNormalize object
    norm_img = ImageNormalize(dirty_image, interval=ZScaleInterval(), stretch=SqrtStretch())
    norm_psf = ImageNormalize(psf, interval=ZScaleInterval(), stretch=SqrtStretch())

    dirty_image_show = hv.Image(dirty_image).options(
        cmap="viridis",
        colorbar=True,
        frame_width=500,
        frame_height=500,
    )

    d = plt.figure("d", figsize=(10, 10))
    plt.imshow(dirty_image, norm=norm_img)
    p = plt.figure("p", figsize=(10, 10))
    plt.imshow(psf, norm=norm_psf)
    mo.vstack([d, p])
    return


@app.cell
def _(arcsec_to_radians, dirty_image, np, pix_scale_rad, psf, radler):
    residual = np.array(dirty_image)
    model = np.zeros_like(residual)
    settings = radler.Settings()

    settings.trimmed_image_width, settings.trimmed_image_height = psf.shape
    # Radler takes the pixel scale in radians.
    settings.pixel_scale.x = pix_scale_rad[0]
    settings.pixel_scale.y = pix_scale_rad[1]
    settings.minor_iteration_count = 50000
    settings.major_iteration_count = 1
    settings.auto_threshold_sigma = 1.0
    # major_loop_gain < 1 lets perform() return True when the in-iteration
    # peak drops by this fraction, signalling a new prediction-gridding round.
    # Without it the algorithm always runs to full completion → returns False.
    settings.major_loop_gain = 0.9

    radler_object = radler.Radler(
        settings,  # Configuration parameters
        psf,  # Point spread function
        residual,  # Residual image data
        model,  # Sky model
        10 * arcsec_to_radians,  # Synthesized beam size
        radler.Polarization.stokes_i,  # Polarization type
    )
    return (radler_object,)


@app.cell
def _(radler_object):
    radler_object.perform(1)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
