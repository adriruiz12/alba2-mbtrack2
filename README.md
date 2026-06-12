# ALBA II longitudinal tracking with mbtrack2

Longitudinal tracking of the ALBA II storage ring with a double RF system
(main + harmonic cavity), built on **mbtrack2** and parallelised over a small
number of MPI ranks (16 by default) for the 448 filled bunches.

The goal is to reproduce tracking simulations for the ALBA II lattice in 
mbtrack2, rather than in ELEGANT.

We validate simulation behaviour across a set of defined operating
scenarios, with success measured by accurate bunch-length reproduction and
stability of the results across mbtrack2 versions. Some additional experiments
are carried out.

## Why a custom MPI layer

mbtrack2's built-in `Beam(..., mpi=True)` mode maps one non-empty bunch to one
MPI rank. For a uniformly filled ALBA II ring (448 bunches) that would require
448 ranks. From the mbtrack2 documentation:

> MPI can be used to speed up the tracking when using a Beam object by
> distributing the different Bunch objects in different cores. To be able to 
> use this feature, the python code must be run with as many cores as there 
> are Bunch objects in the Beam.

This code instead distributes the 448 bunches over a small number of ranks
(28 bunches/rank for 16 ranks), uses `mpi4py` to exchange the longitudinal
profiles every turn, and applies the cavity kicks only to the bunches owned 
by each rank. This keeps the ALBA II parameters and the physics model while 
making the run possible on 16 CPU cores. Note that no time improvement is 
observed beyond 8 ranks due to an unstudied bottleneck.

mbtrack2's CavityMonitor class could be used to store detailed cavity data
during tracking, but many of the variables it tracks are not not relevant to
this script. Instead, the code manually stores the specific cavity quantities 
needed for the final analysis, such as the beam-induced phasor records of the
main and harmonic cavities.

## Module layout

- **`constants.py`**: static physical, machine and RF constants (never change
  between runs). Physics notes on U0 and the RF cosine/sine convention are in
  the docstring here.
- **`config.py`**: run knobs (turns, macro-particles, binning), the validation
  case table and selection, and the case-dependent HC voltage and detuning.
  The full description of each validation case is in the docstring here.
- **`helpers.py`**: pure helpers: relativistic factors, sigma_z/sigma_t
  conversions, RF detuning, last-N averaging.
- **`machine.py`**: builders for the mbtrack2 objects (`build_ring`,
  `build_cavities`, `build_local_bunches`) and low-level cavity-state
  operations.
- **`parallel.py`**: custom MPI layer: bunch distribution and per-turn profile
  and statistics exchange.
- **`tracking.py`**: core physics: equilibrium pre-check, distributed cavity
  tracker, the tracking loop and final data collection.
- **`diagnostics.py`**: plots and the plain-text machine summary (rank 0 only).
- **`soasim.py`**: entry point: wires the modules together and runs
  `main()`.

Each module only imports from modules listed above it, so there are no circular
dependencies.

## Repository structure

```
alba2-mbtrack2/
├── README.md
├── .gitignore
├── requirements.txt
├── submit_soa_base_cpar.sl
├── replot_all.sl
├── paramsalba/                 ← analytical reference params + figures
│   ├── paramsalba.py
│   ├── paramsalba_<case>.txt
│   └── figures/
├── src/
│   ├── simulation/             ← tracking modules + replot.py
│   └── fs_vs_I/                ← fs_analytical.py, plot_fs_comparison.py,
│                                 exp_*.csv (tracked inputs)
└── results/                    
```

## How to run

All simulation files live in `src/simulation/`. Launch from the repo root:

    sbatch submit_soa_base_cpar.sl

Follow the output live:

    tail -f results/mbxlogs/soa_base_<arrayjobid>_<taskid>.txt

The `results/` directory is created automatically on the first run. Each case
writes its output (PDFs, NPZ, summary TXT) into
`results/<STUDY>/<STUDY>_<VALIDATION_CASE>/`.

The number of MPI ranks must divide 448 exactly
(valid: 8, 16, 28, 32, 56, 64).

### Choosing which cases to run

To run only a subset, edit **both** the `CASES` array and the `--array` bound
in the `USER CONFIGURATION` block of `submit_soa_base_cpar.sl`:

```bash
  # default — all four cases
  CASES=(no_hc hc_100kv goal flat_potential)
  #SBATCH --array=0-3

  # example — only the two bunch-lengthening cases
  CASES=(goal flat_potential)
  #SBATCH --array=0-1
```

`#SBATCH` directives are parsed by the scheduler before the shell runs, so the
array bound **cannot** be set dynamically from `${#CASES[@]}`. A guard in the
script exits cleanly for any task ID that exceeds `${#CASES[@]} - 1`, so
launching more tasks than cases is safe but wasteful.

### Regenerating plots without re-running the tracking

`replot.py` reads the saved `history.npz` and `rep_profiles.npz` from a completed
run and regenerates all diagnostic PDFs without repeating the tracking. It stubs
out `mpi4py` so it can be run as a plain Python script.

    VALIDATION_CASE=goal python src/simulation/replot.py

`VALIDATION_CASE` must match the original run; it controls which output
directory is read from (`results/soa_base/soa_base_<VALIDATION_CASE>/`) and
which target values are overlaid on the plots. `STUDY` can also be set if the
original run used a non-default study name.
