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
tests/                     ~117 assertions, all against dense references
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
