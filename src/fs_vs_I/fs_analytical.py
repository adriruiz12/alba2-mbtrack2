"""
fs_analytical.py - Analytical coherent synchrotron frequency f_s versus stored
current I_sr, for ALBA-I and ALBA-II. PART 1 of the f_s validation task.

Standalone: no MPI, no mbtrack2, no tracking. Produces the reference curves
against which the tracking-measured f_s (PART 2, fs_measure.py) is compared.

PHYSICS MODEL  -  f_s_shifted_coherent formula
-------------------------------------------------------
f_s is the beam-loading-shifted coherent synchrotron frequency. It is computed
with the formula given to apply explicitly:

    f_s_shifted_coherent =
        f_s0_with_hhc * sqrt( 1 - 2*I_sr*(N_mc*Zi_mc + m*N_hc*Zi_hc)
                                  / (V_t*cos(phi_s) + m*V_th*cos(phi_h)) )

with the zero-current frequency

    f_s0_with_hhc = f_rf * sqrt( eta_c*(V_t*cos(phi_s) + m*V_th*cos(phi_h))
                                 / (2*pi*h*E0) )

Without a harmonic cavity the HC terms drop out and this reduces to

    f_s_shifted_coherent = f_s0 * sqrt( 1 - 2*I_sr*N_mc*Zi_mc
                                            / (V_t*cos(phi_s)) )

Zi = imag(Z_loaded) is the imaginary part of the loaded cavity impedance, per
cavity.

Sign / convention note. This module uses the sine phase convention: phi_s is
the obtuse, above-transition synchronous phase, so cos(phi_s) < 0. With those
phases V_t*cos(phi_s) + m*V_th*cos(phi_h) comes out negative,
so it is stored here as a positive quantity

    D = -( V_t*cos(phi_s) + m*V_th*cos(phi_h) )           (D > 0)

and the formula is applied as f_s = f_s0*sqrt(1 - 2*I_sr*(...)/D). This is
exactly the given formula; D is just his denominator written with the explicit
sign that makes it positive in this convention. It is also the algebraic twin
of the Jacob & Serriere DC-Robinson K' expression (22nd ESLS RF Meeting,
SOLEIL 2018): D is K'_0 and -2*I_sr*(...) is the beam-loading part of K'.

IMPEDANCES DEPEND ON THE STORED CURRENT
---------------------------------------
Zi is NOT a constant of the machine. For every stored current the cavity is
re-tuned for minimum generator power (zero load angle):

    tan(psi) = 2*I_sr*RL*cos(phi) / V_per_cav        RL = Rs/(1 + beta)
    Zi(I_sr) = -(RL/2) * sin( 2*psi(I_sr) )

so psi = psi(I_sr) and therefore Zi = Zi(I_sr). The sweep below varies BOTH
I_sr explicitly AND the impedances through psi(I_sr): each current is a
different phasor diagram, as stressed. At I_sr = 0, psi = 0 and Zi = 0,
so f_s(0) = f_s0 exactly.

VALIDATION ANCHORS
------------------
  ALBA-I, no HC : f_s0 = 8.54 kHz,  f_s(250 mA) = 6.11 kHz   (ESLS plots)
  ALBA-II + HC  : f_s0 = 0.48 kHz                            (paramsalba)

Usage:
    python fs_analytical.py
Produces fs_analytical.pdf and one fs_analytical_<case>.txt per case
(two columns: I_sr[mA]  f_s[Hz]).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# Generated artifacts go to results/fs_vs_I/ (src/fs_vs_I -> src -> repo root).
OUTDIR = SCRIPT_DIR.parents[1] / "results" / "fs_vs_I"
OUTDIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

C_LIGHT = 299792458.0
TWO_PI = 2.0 * np.pi


# =============================================================================
# CASE DEFINITIONS
# =============================================================================
# ALBA-II values mirror constants.py / config.py (CASE_CONFIG["goal"]).
# ALBA-I values come from the reference paramsalba file for ALBA-I.
#
# Each case dict carries:
#   E0        beam energy [eV]
#   alpha_c   linear momentum compaction factor (eta_c ~ alpha_c at 3 GeV)
#   h         harmonic number of the ring
#   L         ring circumference [m]   (used only for f_rf = h*c/L)
#   U0        energy loss per turn [eV]
#   mc        main-cavity parameters
#   hc        harmonic-cavity parameters or None
#   I_max     upper end of the I_sr sweep [A]  (0.35 A = 350 mA, wanted range)
#   phi_s_mc_deg : fixed MC synchronous phase (sine convention). If None it is
#                  computed from the MC-only energy balance V_mc*sin(phi_s)=U0.
#   phi_hc_deg   : fixed HC phase (sine convention), None when no HC.

CASES = {
    "alba1_no_hc": {
        "label": "ALBA-I, no HC",
        "E0": 3.0e9,
        "alpha_c": 8.880344803809678e-4,
        "h": 448,
        "L": 268.8,
        "U0": 1.024058e6 + 0.1098775e6,
        "mc": {"N": 6, "V_total": 3.0e6, "beta": 2.6, "Rs_per_cav": 3.3e6},
        "hc": None,
        "I_max": 0.35,
        "phi_s_mc_deg": None,    # computed from MC-only energy balance
        "phi_hc_deg": None,
    },
    "alba2_no_hc": {
        "label": "ALBA-II, no HC",
        "E0": 3.0e9,
        "alpha_c": 1.0433261800610089e-4,
        "h": 448,
        "L": 268.79999999999967,
        "U0": 0.9352696554914384e6 + 1.376275e5,
        "mc": {"N": 6, "V_total": 2.4e6, "beta": 3.5, "Rs_per_cav": 3.3e6},
        "hc": None,
        "I_max": 0.35,
        "phi_s_mc_deg": None,    # computed from MC-only energy balance
        "phi_hc_deg": None,
    },
    "alba2_hc": {
        "label": "ALBA-II, with HC (goal, 170 kV/cav)",
        "E0": 3.0e9,
        "alpha_c": 1.0433261800610089e-4,
        "h": 448,
        "L": 268.79999999999967,
        "U0": 0.9352696554914384e6 + 1.376275e5,
        "mc": {"N": 6, "V_total": 2.4e6, "beta": 3.5, "Rs_per_cav": 3.3e6},
        "hc": {"N": 4, "V_total": 4 * 170e3, "beta": 0.7,
               "Rs_per_cav": 1.1e6, "m": 3},
        "I_max": 0.35,
        # With the HC active the design phases are fixed by paramsalba.
        "phi_s_mc_deg": 149.9345744486530,
        "phi_hc_deg": 349.0235762560818,    # equivalently -10.98 deg
    },
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def f_rf(case):
    """Main RF frequency f_rf = h * c / L  [Hz]."""
    return case["h"] * C_LIGHT / case["L"]


def mc_synchronous_phase(case):
    """
    MC synchronous phase (sine convention) in radians.

    If the case provides an explicit phi_s_mc_deg it is used directly (HC
    cases, where the phase is fixed by the paramsalba design). Otherwise the
    phase is obtained from the MC-only energy balance V_mc*sin(phi_s) = U0,
    taking the obtuse (above-transition, stable) solution.
    """
    if case["phi_s_mc_deg"] is not None:
        return np.deg2rad(case["phi_s_mc_deg"])
    sin_phi = case["U0"] / case["mc"]["V_total"]
    return np.pi - np.arcsin(sin_phi)


def psi_tuning(I0, cav, phi):
    """
    Current-dependent optimum cavity tuning angle psi [rad].

        tan(psi) = 2*I0*RL*cos(phi) / V_per_cav ,   RL = Rs/(1 + beta)

    This is the minimum-generator-power (zero load angle) detuning. Because it
    depends on I0, psi - and hence the loaded impedance below - is recomputed
    for every stored current: each current is a different phasor diagram.
    I0 may be a scalar or a NumPy array.
    """
    RL = cav["Rs_per_cav"] / (1.0 + cav["beta"])
    V_per_cav = cav["V_total"] / cav["N"]
    return np.arctan(2.0 * I0 * RL * np.cos(phi) / V_per_cav)


def Z_imag_loaded(I0, cav, phi):
    """
    Imaginary part of the loaded cavity impedance, per cavity [Ohm].

        Zi = imag(Z_loaded) = -(RL/2) * sin(2*psi(I0))

    Zi(I0) is the quantity called Z_i_0 / Z_i_0_hhc in the
    f_s_shifted_coherent formula. It is current-dependent through psi(I0):
    at I0 = 0, psi = 0 and Zi = 0; as I0 grows the cavity detunes and Zi
    grows in magnitude. I0 may be a scalar or a NumPy array.
    """
    RL = cav["Rs_per_cav"] / (1.0 + cav["beta"])
    psi = psi_tuning(I0, cav, phi)
    return -0.5 * RL * np.sin(2.0 * psi)


def restoring_denominator(case):
    """
    D = -( V_t*cos(phi_s) + m*V_th*cos(phi_h) )   [V].

    Denominator written as a positive quantity in this module's sine
    phase convention (phi_s obtuse => cos(phi_s) < 0). It is the zero-current
    restoring coefficient K'_0. The HC term is included only when the case has
    a harmonic cavity.
    """
    phi_s = mc_synchronous_phase(case)
    D = -case["mc"]["V_total"] * np.cos(phi_s)
    if case["hc"] is not None:
        m = case["hc"]["m"]
        phi_h = np.deg2rad(case["phi_hc_deg"])
        D += -m * case["hc"]["V_total"] * np.cos(phi_h)
    return D


def fs0_coherent(case):
    """
    Zero-current coherent synchrotron frequency f_s0 [Hz]:

        f_s0 = f_rf * sqrt( alpha_c * D / (2*pi*h*E0) )

    (given f_s0_with_hhc; eta_c ~ alpha_c at 3 GeV.)
    """
    D = restoring_denominator(case)
    arg = case["alpha_c"] * D / (TWO_PI * case["h"] * case["E0"])
    return f_rf(case) * np.sqrt(arg) if arg > 0.0 else np.nan


def coherent_fs(I0, case):
    """
    Coherent (beam-loading-shifted) synchrotron frequency f_s [Hz] at stored
    current I0 [A]. This is the given f_s_shifted_coherent formula:

        f_s(I0) = f_s0 * sqrt( 1 - 2*I0*(N_mc*Zi_mc + m*N_hc*Zi_hc) / D )

    Both I0 (explicit) and the impedances Zi (through psi(I0)) vary with the
    stored current. I0 may be a scalar or a NumPy array. Returns NaN where the
    bracket is <= 0 (DC-Robinson unstable: f_s would be imaginary).
    """
    I0 = np.asarray(I0, dtype=float)
    phi_s = mc_synchronous_phase(case)
    mc = case["mc"]
    D = restoring_denominator(case)

    # Numerator of the shift: 2*I0*(N_mc*Zi_mc + m*N_hc*Zi_hc).
    # Zi_mc and Zi_hc are current-dependent (psi = psi(I0)).
    sum_NZ = mc["N"] * Z_imag_loaded(I0, mc, phi_s)
    if case["hc"] is not None:
        hc = case["hc"]
        m = hc["m"]
        phi_h = np.deg2rad(case["phi_hc_deg"])
        sum_NZ = sum_NZ + m * hc["N"] * Z_imag_loaded(I0, hc, phi_h)
    shift_numerator = 2.0 * I0 * sum_NZ

    f_s0 = fs0_coherent(case)
    bracket = 1.0 - shift_numerator / D
    bracket = np.where(bracket > 0.0, bracket, np.nan)
    return f_s0 * np.sqrt(bracket)


def sweep(case, n_points=71):
    """
    Returns (I_sr_array [A], f_s_array [Hz]) over 0 .. I_max for the case.
    71 points over 0..0.35 A is one sample every 5 mA.
    """
    I = np.linspace(0.0, case["I_max"], n_points)
    fs = coherent_fs(I, case)
    return I, fs


# =============================================================================
# SANITY CHECKS
# =============================================================================

def report(case_key):
    """Print f_s0, a checkpoint, and the current-dependence of the impedance."""
    case = CASES[case_key]
    fs0 = float(coherent_fs(np.array([0.0]), case)[0])
    phi_s = mc_synchronous_phase(case)
    mc = case["mc"]

    print(f"\n[{case_key}]  {case['label']}")
    print(f"  f_rf            = {f_rf(case)/1e6:.4f} MHz")
    print(f"  phi_s (MC)      = {np.rad2deg(phi_s):.3f} deg")
    print(f"  D (K'_0)        = {restoring_denominator(case):.6e} V")
    print(f"  f_s0  (I=0)     = {fs0/1e3:.4f} kHz")

    # Show explicitly that the loaded impedance changes with current.
    for I_test in (0.0, 0.150, case["I_max"]):
        zi = float(Z_imag_loaded(I_test, mc, phi_s))
        psi = float(psi_tuning(I_test, mc, phi_s))
        print(f"  Zi_mc({I_test*1e3:6.1f} mA) = {zi:11.4e} Ohm   "
              f"(psi = {np.rad2deg(psi):7.3f} deg)")

    if case_key == "alba1_no_hc":
        fs_250 = float(coherent_fs(np.array([0.250]), case)[0])
        print(f"  f_s   (250 mA)  = {fs_250/1e3:.4f} kHz")
        print(f"  reference       : f_s0 = 8.54 kHz, f_s(250 mA) = 6.11 kHz")
    elif case_key == "alba2_hc":
        print(f"  reference       : f_s0 = 0.48 kHz")
    elif case_key == "alba2_no_hc":
        print(f"  (no published reference value for this configuration)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ANALYTICAL COHERENT SYNCHROTRON FREQUENCY  f_s(I_sr)")
    print("Given f_s_shifted_coherent formula, impedances vary with current")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    colors = {"alba1_no_hc": "tab:blue",
              "alba2_no_hc": "tab:green",
              "alba2_hc": "tab:red"}

    for key, case in CASES.items():
        report(key)
        I, fs = sweep(case)

        # Save the swept data for later overlay with the tracking points.
        out_txt = OUTDIR / f"fs_analytical_{key}.txt"
        np.savetxt(
            out_txt,
            np.column_stack([I * 1e3, fs]),
            header="I_sr[mA]    f_s[Hz]",
            fmt="%.6e",
        )
        print(f"  data saved      -> {out_txt}")

        ax.plot(I * 1e3, fs / 1e3, "-", color=colors[key], lw=2,
                label=case["label"])
        ax.plot(0.0, fs[0] / 1e3, "o", color=colors[key], ms=6)  # mark f_s0

    ax.set_xlabel("Stored current  $I_{sr}$  [mA]")
    ax.set_ylabel("Coherent synchrotron frequency  $f_s$  [kHz]")
    ax.set_title("Analytical $f_s$ vs $I_{sr}$  (f_s_shifted_coherent)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    out_fig = OUTDIR / "fs_analytical.pdf"
    fig.savefig(out_fig, dpi=150)
    print(f"\nFigure saved -> {out_fig}")
    print("=" * 70)


if __name__ == "__main__":
    main()
