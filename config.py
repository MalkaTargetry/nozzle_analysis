import os
from pathlib import Path

# Identify the root directory of this repository
PROJECT_ROOT = Path(__file__).resolve().parent

# Set the base data directory to your standard local path.
DEFAULT_DATA_PATH = r"C:\Users\vmlab\Documents\data"
BASE_DATA_DIR = Path(os.getenv("NOZZLE_DATA_DIR", DEFAULT_DATA_PATH))

# Define specific sub-directories based on your structure
RAW_FILES_DIR = BASE_DATA_DIR / "files"           # For raw HDF5 files
PHASE_MAPS_DIR = BASE_DATA_DIR / "phase_maps"     # For generated phase maps
DENSITY_MEASUREMENTS_DIR = BASE_DATA_DIR / "density_measurements"
DENSITY_MAPS_DIR = BASE_DATA_DIR / "density_maps"

# Automatically create these folders if they don't exist yet
for d in [RAW_FILES_DIR, PHASE_MAPS_DIR, DENSITY_MEASUREMENTS_DIR, DENSITY_MAPS_DIR]:
    d.mkdir(parents=True, exist_ok=True)