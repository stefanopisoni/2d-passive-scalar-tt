"""Central configuration: filesystem paths and external-dependency locations.

Repo outputs (cached statistics and figures) are resolved relative to this file,
so the figure scripts run straight from a fresh clone. The large raw/compressed
data and the `tensnet` library live OUTSIDE the repo; override their locations
with environment variables if they sit elsewhere on your machine.

    PSCALAR_DATA_ROOT   root holding data/, standard_mps_data/, hybrid_mps_data/
    TENSNET_SRC         path to the tensnet `src` directory
"""
import os

REPO_DIR  = os.path.dirname(os.path.abspath(__file__))
STATS_DIR = os.path.join(REPO_DIR, "statistics")   # committed .pt/.csv caches
FIG_DIR   = os.path.join(REPO_DIR, "figures")       # committed final paper figures

# Raw DNS snapshots + precompressed TT fields (NOT shipped in the repo; large).
# Needed only to recompute the caches (PLOT_ONLY=False) or run generate_data.py.
DATA_ROOT = os.environ.get("PSCALAR_DATA_ROOT", "/Volumes/QAlg_SSD/2D_passive_scalar")
DATA_DIR  = os.path.join(DATA_ROOT, "data")
STD_DIR   = os.path.join(DATA_ROOT, "standard_mps_data")
HYB_DIR   = os.path.join(DATA_ROOT, "hybrid_mps_data")

# tensnet source tree (needed only by generate_data.py / hybrid_MPS.py).
TENSNET_SRC = os.environ.get(
    "TENSNET_SRC", os.path.join(REPO_DIR, "..", "tensnet", "src")
)
