"""
config.py - Run-time configuration and validation-case resolution.

This module centralises everything that changes between runs:
  * USER CONFIGURATION : numerical knobs (turns, macro-particles, binning...).
  * VALIDATION CASE    : the CASE_CONFIG table and the case selected through
                         the VALIDATION_CASE environment variable.
  * Case-dependent RF  : the HC voltage and the analytical HC detuning, which
                         depend on the selected validation case and on
                         TOTAL_CURRENT_A (constants.py), which is env-overridable
                         to allow current sweeps without editing source files.

Resolving these values once, at import time, keeps the original "module-level
globals" style of the script while giving every other module a single, clear
place to import them from.

VALIDATION CASES
----------------
The case is set via the VALIDATION_CASE environment variable (default: "goal").
The SLURM script runs the four cases sequentially.

    Case                HC voltage/cav   Target sigma_z   Target sigma_t
    -------             --------------   --------------   --------------
    no_hc               —                 2.463 mm          8.216 ps
    hc_100kv            100 kV            3.731 mm         12.445 ps
    goal                170 kV            9.630 mm         32.150 ps
    flat_potential      ~176 kV          11.897 mm         39.685 ps
    goal_elegant_freq   170 kV            9.630 mm         32.150 ps  (*)

    (*) Same operating point as 'goal' but the HC resonance frequency is taken
    directly from the Elegant template (track_ALBA_II.ele line 150) instead of
    the analytical paramsalba detuning. Isolates the ~5.9 kHz FREQ_HC mismatch
    between the two codes.

All numerical targets come from the reference paramsalba results.
The "goal" case is the nominal ALBA II operating point.
The "flat_potential" case uses the exact voltage for a flat RF potential
(xi = 1 condition).

To run a subset of cases, edit the CASES array in submit_soa_base_cpar.sl.
Go to the README.md doc for further information
"""

import os
from pathlib import Path

import numpy as np

from constants import (
    ALPHA_C,
    ALPHA_C2,
    ALPHA_C3,
    BETA_COUPLER_HC,
    FREQ_HC_DRIVE_HZ,
    N_HC,
    QL_HC,
    RS_HC_PER_CAVITY_OHM,
    TOTAL_CURRENT_A,
    phi_s_hc,
)


# =============================================================================
# USER CONFIGURATION
# =============================================================================

DEBUG = False
SEED = 12345 # For reproducibility
N_TURNS = 5_000 if DEBUG else int(os.environ.get("N_TURNS", 50_000))

# Macro-particles per bunch. Each mp represents a lot of real electrons.
# More mp => A smoother profile and better statistics, though => More
# computing time.
MP_PER_BUNCH = 1_000 if DEBUG else 10_000

SAVE_EVERY = 50 if DEBUG else 100

# Profile binning: converts the macro-particle cloud into a histogram for
# diagnostics and final plots. Independent of the cavity calculation. 201 is a
# practical starting point; increase to 401 or 1001 if finer profile resolution 
# is needed, or reduce to speed up runs where profile shape is not the focus.
PROFILE_NBIN = 101 if DEBUG else 201

# Cavity binning: used by CavityResonator to compute the beam-induced phasor
# each turn. Set to 1001 to match Elegant's RFMODE n_bins (MC and HC both use 
# n_bins=1001 in track_ALBA_II.ele). Expensive because all bunch profiles are
# exchanged across MPI ranks every turn, so reduce to 201 if in DEBUG mode.
CAVITY_NBIN = 201 if DEBUG else int(os.environ.get("CAVITY_NBIN", 1001))

# Momentum compaction factor order. When USE_LINEAR_MCF is set to "1" via the
# environment, only the linear term alpha_c is used and the higher-order terms
# alpha_c2 and alpha_c3 are zeroed. Default is the full non-linear polynomial.
# In DEBUG mode the linear variant is never forced: the flag is ignored.
USE_LINEAR_MCF = False if DEBUG else (os.environ.get("USE_LINEAR_MCF", "0") == "1")

# mbtrack2 expects the coefficients in decreasing polynomial order:
#     mcf(delta) = ALPHA_C3 * delta**2 + ALPHA_C2 * delta + ALPHA_C
if USE_LINEAR_MCF:
    MCF_ORDER = np.array([0.0, 0.0, ALPHA_C], dtype=float)
else:
    MCF_ORDER = np.array([ALPHA_C3, ALPHA_C2, ALPHA_C], dtype=float)

# BeamLoadingEquilibrium numerical grid.
# BLE solves the Haissinski equation numerically: it evaluates
#   rho(z) ∝ exp(-u(z))
# on a discrete grid of EQ_NBIN points in z (metres), where u(z) is a
# dimensionless scaled potential combining the U0 energy loss, generator
# voltage and beam-loading terms. It then integrates over that grid
# (trapezoidal rule) to get sigma_z via std_rho().The grid spans 
#   [-B_HALF_M, +B_HALF_M] = [-c*EQ_HALF_WINDOW_S, +c*EQ_HALF_WINDOW_S].
# The window must be wide enough to capture the full distribution tails 
# (or std_rho() underestimates sigma_z), but narrow enough to exclude 
# neighbouring RF buckets. 5e-10 s = 500 ps <=> c*500 ps ≈ 150 mm: much larger
# than the expected bunch lengths in all validation cases, and well within the
# RF bucket spacing (~2000 ps at 499.6 MHz).
EQ_HALF_WINDOW_S = 5e-10
EQ_NBIN = 4000

# Number of saved points used to estimate the final averaged value. This is useful 
# when results still shows a residual coherent oscillation in sigma_t/sigma_z.
LAST_AVG_N = 100

# mbtrack2 can track which macro-particles are lost and exclude them from further
# calculations. Disabled here because nothing in the tracking loop
# (LongitudinalMap, SynchrotronRadiation, the custom cavity tracker) applies
# any aperture or loss criterion, so no particles are ever marked as dead.
TRACK_ALIVE = False

# =============================================================================
# VALIDATION CASE
# =============================================================================
# The validation case is selected from the environment variable VALIDATION_CASE. 
# This allows the SLURM script to run the four cases sequentially:
#   no_hc -> hc_100kv -> goal -> flat_potential
# If the environment variable is not defined, the default case is goal, defined
# as the ALBA II case. The values for each case come from the paramsalba results.

VALIDATION_CASE = os.environ.get("VALIDATION_CASE", "goal")

CASE_CONFIG = {
    "no_hc": {
        "use_hc": False,
        "V_HC_PER_CAVITY_V": 0.0,
        "TARGET_SIGMA_Z_MM": 2.462967837425364,
        "TARGET_SIGMA_T_PS": 8.215576501110004,
    },
    "hc_100kv": {
        "use_hc": True,
        "V_HC_PER_CAVITY_V": 100e3,
        "TARGET_SIGMA_Z_MM": 3.7309883551485274,
        "TARGET_SIGMA_T_PS": 12.445237729338462,
    },
    "goal": {
        "use_hc": True,
        "V_HC_PER_CAVITY_V": 170e3,
        "TARGET_SIGMA_Z_MM": 9.630000,
        "TARGET_SIGMA_T_PS": 32.150000,
    },
    "goal_elegant_freq": {
    # Same operating point as 'goal' but the HC resonance frequency is taken directly
    # from the Elegant template (track_ALBA_II.ele), not from the analytical
    # paramsalba detuning. This isolates the impact of the ~66 kHz FREQ_HC mismatch
    # between the two codes for the Elegant cross-check.
        "use_hc": True,
        "V_HC_PER_CAVITY_V": 170e3,
        "TARGET_SIGMA_Z_MM": 9.630000,
        "TARGET_SIGMA_T_PS": 32.150000,
        "FREQ_HC_RESONANCE_HZ_OVERRIDE": 1.499176140487614e9,
    },
    "flat_potential": {
        "use_hc": True,
        "V_HC_PER_CAVITY_V": 176087.95063073197,
        "TARGET_SIGMA_Z_MM": 11.897167681695016,
        "TARGET_SIGMA_T_PS": 39.6846803073452,
    },
}

if VALIDATION_CASE not in CASE_CONFIG:
    raise ValueError(f"Unknown VALIDATION_CASE: {VALIDATION_CASE}")

CASE = CASE_CONFIG[VALIDATION_CASE]
USE_HC = CASE["use_hc"]

TARGET_SIGMA_Z_MM = CASE["TARGET_SIGMA_Z_MM"]
TARGET_SIGMA_T_PS = CASE["TARGET_SIGMA_T_PS"]

# Simulation study label, used to group results under results/<STUDY>/.
STUDY = os.environ.get("STUDY", "soa_base")
# Absolute path to the repository root (src/core/ -> src/ -> repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Output directory for this run. Defaults to results/<STUDY>/<STUDY>_<CASE>/
# and can be overridden via the OUTDIR environment variable.
OUTDIR = Path(os.environ.get(
    "OUTDIR",
    str(_REPO_ROOT / "results" / STUDY / f"{STUDY}_{VALIDATION_CASE}")
))

# =============================================================================
# HARMONIC CAVITY VOLTAGE (case-dependent)
# =============================================================================

V_HC_PER_CAVITY_V = CASE["V_HC_PER_CAVITY_V"]
V_HC_TOTAL_V = N_HC * V_HC_PER_CAVITY_V


# =============================================================================
# HARMONIC CAVITY DETUNING (case-dependent)
# =============================================================================

# Compute the HC detuning only when the harmonic cavity is enabled. In the no_hc
# validation case, V_HC_PER_CAVITY_V = 0, so this branch also avoids a division 
# by zero.
if USE_HC:
    # Detuning angle of the harmonic cavity. It estimates the cavity detuning 
    # needed to sustain the selected HC voltage in the presence of beam loading.
    # The angle increases with beam current and shunt impedance, and decreases
    # with HC voltage and coupling.
    psi_d_hc = np.arctan((2.0 * TOTAL_CURRENT_A * RS_HC_PER_CAVITY_OHM) / 
                         (V_HC_PER_CAVITY_V * (1.0 + BETA_COUPLER_HC)) * np.cos(phi_s_hc))

    # Convert the detuning angle into the HC resonance frequency. FREQ_HC_DRIVE_HZ
    # is the ELEGANT drive frequency, kept as the RF reference. FREQ_HC_RESONANCE_HZ
    # is the actual resonant frequency later used by mbtrack2 to compute:
    #     detune = resonance_frequency - m * ring.f1
    # The relation tan(psi_d)/(2*QL) gives the relative frequency shift from the
    # drive frequency.
    FREQ_HC_RESONANCE_HZ = FREQ_HC_DRIVE_HZ * (1.0 + np.tan(psi_d_hc) /
                                               (2.0 * QL_HC))


    # Optional override: a validation case may specify FREQ_HC_RESONANCE_HZ_OVERRIDE
    # to bypass the analytical value. Used for the Elegant cross-check 
    # ('goal_elegant_freq'), where we want mbtrack2 to use exactly the same HC
    # resonance frequency that Elegant has hardcoded in track_ALBA_II.ele. The
    # analytical psi_d_hc above is kept untouched: it is reported in the summary
    # as the value that would have been used without override, which is useful for
    # documenting the detuning gap.
    if "FREQ_HC_RESONANCE_HZ_OVERRIDE" in CASE:
        FREQ_HC_RESONANCE_HZ_ANALYTICAL = FREQ_HC_RESONANCE_HZ
        FREQ_HC_RESONANCE_HZ = float(CASE["FREQ_HC_RESONANCE_HZ_OVERRIDE"])
    else:
        FREQ_HC_RESONANCE_HZ_ANALYTICAL = FREQ_HC_RESONANCE_HZ



else:
    psi_d_hc = np.nan
    FREQ_HC_RESONANCE_HZ = np.nan
    FREQ_HC_RESONANCE_HZ_ANALYTICAL = np.nan