# Time evolution in the hybrid TT representation

**Follow-on work, not part of the published paper.** Nothing in this directory is
needed to reproduce the paper's figures, and nothing in the repository root
imports from here. The dependency runs one way only: these modules read
`config.py` (for `TENSNET_SRC`) and `hybrid_MPS.py` (for the legacy encoder, used
as a reference in one test) from the repository root.

The question here is different from the paper's. The paper asks how faithfully
the hybrid TT *represents* a given snapshot. This asks whether a field can be
*evolved* in that representation — whether the rank stays bounded under a
cascade, and whether truncation error accumulates.

## Layout

```
hybrid_ops.py              Hybrid TT algebra: construction, per-bond compression,
                           exact arithmetic, MPO application (un-fused and fused
                           head), mass projection, pointwise evaluation
hybrid_operators.py        MPO builders: stencils, derivatives, Laplacian,
                           translation, advection-diffusion, isotropic seam smoother
hybrid_evolve.py           Explicit RK steppers, truncation policy, variable-coefficient
                           transport, dense/spectral references, diagnostics
run_decaying_cellular.py   Experiment: decaying scalar in a cellular flow, dense
                           ground truth plus hybrid runs at several rank caps
plot_decaying_cellular.py  Figures from those runs
tests/                     ~117 assertions, all against dense references
results/                   .npz written by the runs (git-ignored, regenerable)
figures/                   committed .png
recap_time_evolution.tex   Status note: equations, what is checked, findings, outlook
recap_time_evolution.pdf
```

## Running

Every test is a standalone script and can be run from anywhere:

```bash
python time_evolution/tests/test_hybrid_ops.py
python time_evolution/tests/test_hybrid_operators.py
python time_evolution/tests/test_advection.py
python time_evolution/tests/test_scaleup_L8_N6.py
```

Needs `tensnet` (see `config.py` / the root README). No raw DNS data is required —
the tests build their own fields and check against dense references computed in
place.

The cellular-flow experiment runs the same way. The dense ground truth takes
about 20 s; each hybrid run is 20-50 min at `L=8`, so they are meant to be run in
parallel and then plotted:

```bash
cd time_evolution
python run_decaying_cellular.py --dense --label dense
python run_decaying_cellular.py --eps 1e-6 --bd 128 --label adaptive &
python run_decaying_cellular.py --bd 64 --label bd64 &
python run_decaying_cellular.py --bd 32 --label bd32 &
python run_decaying_cellular.py --bd 16 --label bd16 &
wait
python plot_decaying_cellular.py
```

`results/` is git-ignored: the `.npz` files are a few MB each and regenerate from
the scripts.

## Conventions

`L` = bits per axis (grid `2^L x 2^L`); `2L` interleaved Morton legs with even legs
carrying rows (`y`) and odd legs columns (`x`); `N` = legs held by the dense head,
spanning `2^(N/2) x 2^(N/2)` coarse blocks. The seam period is `2^(L - N/2)`.

Everything is built on top of `tensnet` through its public API. No `tensnet` source
is modified, and no `tensnet` global state is read or written — several of its
defaults silently corrupt results rather than raising, which the module docstrings
record.

See `recap_time_evolution.pdf` for the equations being solved, what has been
validated, the findings so far, and the planned next steps.
