# Nozzle Analysis

Python tools and Jupyter notebooks for processing nozzle interferometry data. The repository reads camera data stored in HDF5 files, identifies reference/experiment image pairs, extracts phase maps using Fourier filtering and phase unwrapping, and derives nozzle depth and gas-density profiles.

> **Status:** This project is research-oriented and is currently driven primarily through the notebooks in `notebooks/`. APIs and processing parameters may change as the analysis develops.

## Repository layout

```text
.
├── config.py                         # Data-directory configuration
├── notebooks/
│   ├── nozzle_153.ipynb              # Notebook analysis workflow
│   └── nozzle_155.ipynb              # Notebook analysis workflow
└── src/
    ├── calibration.py                # Calibration helpers
    ├── data_analysis.py              # HDF5 file discovery and access facade
    ├── nozzle_interferometry_analysis.py  # Legacy; reference only
    ├── phase_extraction.py            # Fourier-based phase extraction functions
    ├── phase_map_generator.py         # Current phase-map generation workflow
    ├── reading_hdf5.py               # HDF5 reader and camera-image decoding
    └── slit_analysis.py               # Phase-map cropping, depth, and density analysis
```

## Processing workflow

The current workflow is centered on `src/phase_map_generator.py`:

1. Place interferometry HDF5 files in a data directory.
2. Read image data and acquisition metadata with `reading_hdf5.py` and `data_analysis.py`.
3. Match reference and experiment images by experiment-trigger delay and Parker-valve state.
4. Detect the nozzle edge and nozzle x-range.
5. Fourier-filter the interferograms and demodulate them.
6. Unwrap the phase difference and generate a calibrated phase map.
7. Export phase maps as `.npy` files with accompanying JSON metadata.
8. Use `slit_analysis.py` to estimate a depth profile and convert phase into number density.

The lower-level functions in `phase_extraction.py` expose an alternative, smaller Fourier-processing workflow for selecting ROIs and extracting wrapped phase differences.

## Requirements

The code is Python-based and uses scientific-computing, image-processing, and notebook packages, including:

- Python 3.10 or newer recommended
- NumPy
- SciPy
- Matplotlib
- scikit-image
- h5py
- OpenCV (`opencv-python`)
- Jupyter Notebook or JupyterLab

Create an environment and install the dependencies, for example:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib scikit-image h5py opencv-python jupyter
```

## Configure data directories

`config.py` defines the locations used for raw files and generated results. By default, it points to a Windows-specific local path. Override that path with the `NOZZLE_DATA_DIR` environment variable before importing the project:

```bash
# macOS/Linux
export NOZZLE_DATA_DIR=/path/to/nozzle-data

# Windows PowerShell
$env:NOZZLE_DATA_DIR = "D:\\path\\to\\nozzle-data"
```

The following subdirectories are created automatically beneath the configured data directory:

- `files/` — raw HDF5 files
- `phase_maps/` — generated phase maps
- `density_measurements/` — density-profile results
- `density_maps/` — generated density maps

For best results, retain the date-based directory structure expected by the phase-map export methods (`YYYY/MM/DD/<interferometry-name>`), because exported phase maps use the date components when constructing output paths.

## Running the notebooks

Launch Jupyter from the repository root so that imports such as `from src...` resolve correctly:

```bash
jupyter lab
# or
jupyter notebook
```

Open one of the notebooks in `notebooks/` and update its input paths and calibration values for the dataset being processed. The notebooks contain the exploratory, dataset-specific workflow and visual checks.

## Example: generate phase maps from Python

```python
from src.phase_map_generator import PhaseMapGenerator

# The folder should contain paired HDF5 reference and experiment scans.
generator = PhaseMapGenerator(r"/path/to/YYYY/MM/DD/interferometry_scan")

output_dir, saved, failed = generator.export_phase_maps_npy_by_delay(
    out_root=r"/path/to/nozzle-data/phase_maps",
    interferometry_name="interferometry_scan",
    overwrite_existing=False,
)

print(f"Output: {output_dir}")
print(f"Saved: {len(saved)}, failed: {len(failed)}")
```

The phase-map generator requires physical scaling. Supply either `px_per_mm` or a physical `nozzle_width_mm` when calling `get_interferometry_phase`, depending on the calibration information available for the dataset.

## Example: estimate depth and density

```python
import numpy as np
from src.slit_analysis import (
    crop_and_format_phase_map,
    estimate_depth_profile,
    phase_to_density,
)

phase = np.load("/path/to/phase_map.npy")
px_to_mm = 1.0 / 100.0  # replace with the dataset calibration

cropped = crop_and_format_phase_map(
    phase,
    nozzle_edge_y=phase.shape[0],
    px_to_mm=px_to_mm,
    max_height_mm=6.0,
)

y_mm, depth_mm, _ = estimate_depth_profile(
    cropped,
    px_to_mm=px_to_mm,
    nozzle_opening=6.0,  # replace with the physical nozzle opening
)

density, x_mm, y_density_mm = phase_to_density(
    cropped,
    px_to_mm=px_to_mm,
    depth_y_mm=y_mm,
    depth_profile_mm=depth_mm,
    gas="argon",
)
```

The density conversion currently supports `argon` and `helium`. Verify the wavelength, gas constants, nozzle dimensions, and calibration values before using results for quantitative analysis.

## Data format

Input scans are expected to be HDF5 files containing:

- camera data under the `cameras` group;
- experiment trigger metadata under `pulse_generators/qc_a/pulse_generator_channels/experiment_image_trigger`;
- Parker-valve state under `pulse_generators/qc_a/pulse_generator_channels/parker_valve_trigger`;
- acquisition time data under `main/time`.

`reading_hdf5.py` decodes the camera payload into NumPy arrays and validates the expected image format.

## Legacy file

`src/nozzle_interferometry_analysis.py` is a **legacy implementation**. It is no longer used by the current workflow and is retained only as a historical/reference record. New analysis should use `phase_map_generator.py`, `phase_extraction.py`, `data_analysis.py`, `reading_hdf5.py`, and `slit_analysis.py` as appropriate.

## Notes and limitations

- This repository does not currently provide a packaged command-line interface or an automated test suite.
- Several workflows use interactive Matplotlib plots and Tk folder-selection dialogs, so a graphical session may be required.
- The default paths and some notebook parameters are machine- and dataset-specific; update them before running the analysis.
- Large HDF5 files and generated NumPy phase maps are data artifacts and should generally remain outside version control.
