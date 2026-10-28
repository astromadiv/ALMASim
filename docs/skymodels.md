# Skymodels

This page describes ALMASim's sky-model layer: the source a simulation puts in its model cube.

## Location

Sky-model implementations live in [`src/almasim/skymodels`](../src/almasim/skymodels). The families ALMASim can simulate are listed in one table, `SKYMODELS`, in [`registry.py`](../src/almasim/skymodels/registry.py); the simulation builds the chosen model in [`services/simulation/skymodel_stage.py`](../src/almasim/services/simulation/skymodel_stage.py).

## Supported Source Types

`SimulationParams.source_type` (`--source-type` on the command line) is one of:

| Source type | Model | Command line |
|---|---|---|
| `point` | `PointlikeSkyModel` | yes |
| `gaussian` | `GaussianSkyModel` | yes |
| `extended` | `ExtendedSkyModel` (IllustrisTNG) | no, Python only |
| `diffuse` | `DiffuseSkyModel` | yes |
| `galaxy-zoo` | `GalaxyZooSkyModel` | yes |
| `molecular` | `MolecularCloudSkyModel` | yes |
| `hubble-100` | `HubbleSkyModel` | yes |
| `external-fits-cube` | an external FITS cube | yes, with `--input` |
| `external-fits-image` | an external FITS image | yes, with `--input` |
| `external-components` | an external component table | yes, with `--input` |

The default is `point`.

## Common Structure

The built-in models except `extended` derive from `SkyModel` in [`base.py`](../src/almasim/skymodels/base.py).

They receive:

- a MARTINI `DataCube` to fill
- the continuum flux per channel
- the line fluxes
- the line positions and widths, in channels
- the cube size in pixels and channels
- spatial parameters when needed
- a random generator (`rng`), drawn from the run's seed

Each model adds its source's spectrum, continuum plus Gaussian lines, spread over its spatial shape. The spectrum comes from the spectral setup: an SED template chosen by the source type (a type-2 AGN for `point` and `gaussian`, a star-forming galaxy for the others), at the source's redshift.

## Point and Gaussian Models

For compact sources:

- `PointlikeSkyModel` puts a point source in one pixel
- `GaussianSkyModel` puts in an elliptical 2D Gaussian with a random FWHM of 3–9 pixels on each axis and a random position angle

These are useful for:

- debugging
- PSF and deconvolution validation
- controlled synthetic datasets

## Physically Richer Models

### `extended`

Uses IllustrisTNG galaxies, inserted with MARTINI, for realistic extended systems. The subhalo is chosen from the TNG snapshot at the source's redshift and rotated at random. It needs the TNG data (`tng_dir`) and, to download it, a TNG API key (`tng_api_key`). The command line does not offer it.

### `galaxy-zoo`

Uses a randomly chosen Galaxy Zoo 2 image as the spatial morphology.

### `hubble-100`

Uses a randomly chosen Hubble Top-100 image as the spatial morphology.

Both image-based models download their images from Kaggle when they are missing (into `galaxy_zoo_dir` or `hubble_dir`), with the credentials given by `kaggle_token` (`--kaggle-token`) or Kaggle's default locations.

### `molecular`

Generates molecular-cloud-like structured emission: a power-law field with random slope, ellipticity and orientation.

### `diffuse`

Builds a diffuse emission field with NIFTy.

## External Sky Models

The external source types take their emission from a file instead of a model ([`services/external_skymodel.py`](../src/almasim/services/external_skymodel.py)). The file brings its own spectrum: ALMASim adds no continuum or lines.

- `external-fits-cube`: a FITS cube, read as `(channel, y, x)`. A Stokes axis keeps Stokes I, channels that run down in frequency are reversed, and blanked (NaN) pixels hold no emission (WSClean writes flagged channels as NaN)
- `external-fits-image`: a 2D FITS image (or the first plane of a cube), the same in every channel
- `external-components`: a table of components, each with `x_pix`/`y_pix` or `ra_deg`/`dec_deg`, a flux (`flux_jy`, `flux` or `intensity`), and either a `channel` or a power-law spectrum (`spectral_index` or `alpha`, at `reference_freq_hz`)

In Python, the FITS inputs are given by `external_skymodel_path` and the component table by `external_component_table_path`. A FITS input sets the run's grid from its header (`resolve_cube_geometry`): its size, pixel and frequency axis, with `n_pix` and `n_channels` ignored. A 2D image sets the size and pixel only. The products take the header built from the metadata on that grid; with `external_header_mode="preserve"` the run keeps the input's header instead. `external_header_overrides` sets individual header keywords.

On the command line, `--input` gives a sky-model cube, or a folder of them, and each cube is simulated on its own grid: its size, pixel and frequency axis (centre, channel width, channels) come from its header, so `--n-pix` and `--n-channels` are refused with it. A 2D image sets the size and pixel; the metadata gives the channels. A folder from an imaging run (`<ms>/SPW-<n>_FIELD-<m>/`) gives each spectral window's clean-component cube, `wsclean-cube-model.fits`; other folders give every FITS file.

Each cube is simulated with the metadata row of the observation its path names: a member OUS uid matches directly, and an execution block uid is looked up in the resolved products (`--products-csv`; by default the `*products*.csv` files next to `--metadata-path`), which give its member OUS. Every spectral window of an observation matches that row, so there is one run per cube, named after its MeasurementSet and window. A single cube whose path names no observation is used for the chosen rows. With `--input`, a non-external `--source-type` becomes `external-fits-cube`.

With `--source-type external-components`, `--input` is one component table (any format astropy's `Table.read` recognises, such as `.ecsv` or `.csv`), used for every row; a table has no grid, so `--n-pix` and `--n-channels` apply (128 and 32 by default).

## Source Placement

The science target of the built-in models is:

- centered at the phase center by default
- optionally shifted by explicit `source_offset_x_arcsec` and `source_offset_y_arcsec` (in Python; the command line has no option for them)

The offsets apply to the models placed at a position, `point` and `gaussian`; the image-based, `molecular` and `diffuse` models fill the whole field, and `extended` is placed by MARTINI at the pointing.

## Serendipitous Sources

With `inject_serendipitous` (`--inject-serendipitous`), ALMASim adds one to four Gaussian sources within `n_pix / 4` pixels of the phase center, each with its own continuum and its own lines (one, or up to two when the target has several).

They are separate from the [background sky](background.md):

- serendipitous sources are a few discrete extra sources near the target
- background sky is an additive field-level component
