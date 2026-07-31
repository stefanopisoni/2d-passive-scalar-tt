# Multiscale passive scalar turbulence in a compressed subspace via tensor trains

Code and cached statistics reproducing the figures of

> S. Pisoni, E. Tiunov, C. Calascibetta,
> *Multiscale passive scalar turbulence in a compressed subspace via tensor trains.*

We compress a highly intermittent 2D passive scalar field (4096² grid) with four
strategies — Galerkin (spectral) truncation, wavelet (Daubechies db32) truncation,
a standard Tensor Train (TT), and a **hybrid TT** that keeps the coarsest scales
exact and compresses only the finer ones — and compare how faithfully each
preserves the multiscale/intermittent statistics (energy spectrum, structure
functions, flatness, and increment PDFs) at a fixed compression ratio.

## Layout

```
config.py            Central paths (repo dirs + data/tensnet locations, env-overridable)
hybrid_MPS.py        Hybrid TT construction (dense head + compressed tail)
spectral_tools.py    Galerkin / wavelet truncations and isotropic spectrum
generate_data.py     Pipeline: DNS snapshots -> compressed TT fields (needs data + tensnet)
figure1.py           Fig. 1  E(k), S4(r), flatness comparison
figure2.py           Fig. 2  increment PDFs
figure4.py           Fig. 4  flatness vs compression ratio rho
figure5.py           Fig. 5  dependence on the hybrid head size L_h  (Appendix B)
statistics/          Cached averaged statistics (.pt) + CSVs — enough to replot everything
figures/             Final paper figures (PNG + PDF)
```

(Paper Fig. 3 is a schematic and has no script; there is intentionally no `figure3.py`.)

## Reproducing the figures

### Tier 1 — replot from cached statistics (no raw data, no tensnet)

```bash
pip install -r requirements.txt
python figure1.py    # -> figures/figure1.{png,pdf}
python figure2.py
python figure4.py
python figure5.py
```

Each script runs with `PLOT_ONLY = True` and reads the small cached arrays in
`statistics/`, so anyone can regenerate the paper figures from a fresh clone.
Only `torch`, `numpy`, `matplotlib`, `pandas`, `PyWavelets` are required.

### Tier 2 — recompute the statistics from the full dataset

Requires the raw DNS snapshots and the precompressed TT fields (not shipped —
they are large), plus the `tensnet` library. Point the code at them via
environment variables:

```bash
export PSCALAR_DATA_ROOT=/path/to/2D_passive_scalar   # holds data/, standard_mps_data/, hybrid_mps_data/
export TENSNET_SRC=/path/to/tensnet/src
```

Then set `PLOT_ONLY = False` at the top of a figure script to rebuild its
`statistics/*.pt` cache, or run `generate_data.py` to rebuild the compressed
TT fields from the DNS snapshots. The DNS dataset is the *TURB-scalar* database
(Calascibetta et al.); see the paper's references for access.

## Notes

- The Morton (Z-order) interleaving of the two spatial coordinates is what makes
  successive TT physical indices correspond to successive spatial scales.
- The hybrid TT's dense head partitions the domain into 2^{L_h} × 2^{L_h} blocks;
  the compression-induced discontinuities at those interfaces are removed with a
  local linear interpolation (see `generate_data.py`).
