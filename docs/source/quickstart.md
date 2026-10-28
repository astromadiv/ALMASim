# Quick Start

Install the package (Python 3.12) from PyPI:

```bash
pip install almasim
almasim --help
```

or, from a clone of the repository, into a uv environment (`uv sync --group dev`,
then `uv run almasim ...`) or editable (`pip install -e .`). The `[web]` extra adds
the web console, `[casa]` and `[ms-casacore]` the MeasurementSet tools.

The `almasim` command is the main entry point. A first simulation, from an
archive query:

```bash
# Band 6 galaxy observations from the ALMA archive, saved as a metadata CSV
almasim metadata --science-keyword Galaxies --band 6 --limit 10 \
  --save-csv output/metadata.csv --yes

# Simulate the first row (a point source by default) into an ML-ready HDF5 shard
almasim simulation --metadata-path output/metadata.csv --row-idx 0 \
  --project-name demo --output-dir output/simulation --ml-shard-path output/demo.h5
```

The [command-line reference](cli.md) lists every command and option. On a
cluster, the [web console](console.md) runs the same commands as Slurm jobs
from the browser.

The staged Python API does the same from code:

```python
from almasim import (
    SimulationParams,
    generate_clean_cube,
    simulate_observation,
    image_products,
    export_results,
    run_simulation,
)
from almasim.services.metadata.tap import InclusionFilters, query_metadata_by_science
```

[Simulation](simulation.md) walks through the stages and their parameters.

Example scripts in `examples/`:

```bash
python examples/query_metadata_cli.py --help
python examples/staged_pipeline_cli.py --help
python examples/download_products_cli.py --help
python examples/archive_ms_cli.py --help
```

For the subsystem docs, continue with:

- [Simulation](simulation.md)
- [Interferometry](interferometry.md)
- [Imaging and Combination](imaging.md)
- [Metadata](metadata.md)
