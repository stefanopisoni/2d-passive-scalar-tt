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
plot_hybrid_vs_standard.py Hybrid vs standard TT under evolution, at matched cost
sweep_head_size.py         Best head size N, and how its advantage scales with L
sweep_resolution.py        Budget needed for a fixed accuracy, L=7..10
plot_scaleup.py            Hybrid vs standard under evolution at L=8 and L=9
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

The hybrid-versus-standard comparison sweeps head size against bond cap. `N=0`
means *no head* and is exactly the standard MPS, so both encodings run through
one code path and the head size is the only variable:

```bash
for N in 0 4 6 8 10; do for bd in 8 12 16 24 32; do
  python run_decaying_cellular.py --N $N --bd $bd --T 8 --label cmp_N${N}_bd${bd} &
done; done; wait
python plot_hybrid_vs_standard.py
python sweep_head_size.py          # representation only, ~5 min
```

Scaling to higher resolution. Holding rho fixed as L grows needs `chi ~ 2^L`, so
evolution cost grows like `8^L`; the evolution comparison is therefore run at
L=8 and L=9 only, and the resolution trend is measured by representation, which
`plot_scaleup.py` first checks is a faithful stand-in. All of it uses
`kappa ~ dx^2` so that `ell_B/dx` is fixed and a finer grid adds scales rather
than over-resolving the ones already there.

```bash
for N in 0 8 10; do for bd in 12 16 24 32; do
  python run_decaying_cellular.py --L 9 --N $N --bd $bd --kappa 1.25e-3 \
    --T 8 --label L9_N${N}_bd${bd} &
done; done; wait
python plot_scaleup.py
python sweep_resolution.py --L 7 8 9 10
```

Read the gain as **parameters saved at a fixed accuracy** (~2-3x). The error
ratio at a fixed budget describes the same gap down a much steeper axis and
comes out ten times larger; both are printed, and only the first answers "how
much does the head buy me".

`results/` is git-ignored: the `.npz` files are a few MB each and regenerate from
the scripts.

## Running on the cluster

`cluster/submit_ladder.slurm` runs the L=10/11/12 ladder as a Slurm array on
`login.qrccluster.com`. The code is **rsynced, not cloned** — `origin/main` is
far behind the local tree, so a clone gets none of this directory.

Run **single-threaded**. This is not a tuning preference: a step is thousands of
small contractions and SVDs, so asking for 4 threads instead of 1 measured 29x
*slower* per step on a 256-core node (and 43x on the hot einsum alone).
`run_decaying_cellular.py` pins `torch.set_num_threads` itself; set
`PSCALAR_THREADS` to override. One core per task, wide array.

## Conventions

`L` = bits per axis (grid `2^L x 2^L`); `2L` interleaved Morton legs with even legs
carrying rows (`y`) and odd legs columns (`x`); `N` = legs held by the dense head,
spanning `2^(N/2) x 2^(N/2)` coarse blocks. The seam period is `2^(L - N/2)`.
`N = 0` is permitted and means no head at all: a plain QTT, verified to
reproduce `tt.vector_to_mps(vec, 2, 2L)` bond-for-bond (up to tensnet's
renormalisation factor), which is the paper's standard MPS.

Everything is built on top of `tensnet` through its public API. No `tensnet` source
is modified, and no `tensnet` global state is read or written — several of its
defaults silently corrupt results rather than raising, which the module docstrings
record.

See `recap_time_evolution.pdf` for the equations being solved, what has been
validated, the findings so far, and the planned next steps.
