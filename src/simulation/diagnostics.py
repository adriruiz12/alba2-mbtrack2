"""
diagnostics.py - Output side of the simulation: plots and the plain-text
machine summary file. Everything here runs on rank 0 only.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI

from constants import (
    ALPHA_C,
    C_LIGHT,
    CIRCUMFERENCE_M,
    E0_EV,
    EMIT_X_MRAD,
    EMIT_Y_MRAD,
    FREQ_HC_DRIVE_HZ,
    FREQ_MC_DRIVE_HZ,
    FREQ_MC_RESONANCE_HZ,
    H,
    N_HC,
    N_MC,
    RS_HC_PER_CAVITY_OHM,
    SIGMA_DELTA,
    THETA_HC_COS_RAD,
    THETA_MC_COS_RAD,
    TOTAL_CURRENT_A,
    U0_EV,
    V_MC_TOTAL_V,
    m_hc,
)
from config import (
    CASE,
    CAVITY_NBIN,
    FREQ_HC_RESONANCE_HZ,
    FREQ_HC_RESONANCE_HZ_ANALYTICAL,
    LAST_AVG_N,
    MP_PER_BUNCH,
    N_TURNS,
    OUTDIR,
    PROFILE_NBIN,
    TARGET_SIGMA_T_PS,
    TARGET_SIGMA_Z_MM,
    USE_HC,
    VALIDATION_CASE,
    V_HC_TOTAL_V,
)
from helpers import (
    analytical_equilibrium_profile,
    relativistic_factors,
    safe_last_mean,
)


# Colour palette — mirrors paramsalba.py exactly
C_MC  = "#2A5FA5"   # main cavity / reference / no-HC baseline
C_HC  = "#E86B3E"   # harmonic cavity / user / current setting
C_OPT = "#3A9E6F"   # optimal / target / total signal
C_SEC = "#7A8C99"   # secondary / losses / auxiliary quantity

_BBOX = dict(boxstyle="round", facecolor="white", alpha=0.85, lw=0.5)


def write_summary_file(path: Path, lines):
    """
    Writes a list of strings to a plain-text summary file, one line per entry.
    """

    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def save_run_data(history, final_data):
    """
    Persists history and the rep_profiles subset of final_data to OUTDIR so
    that make_plots() can be re-run without repeating the tracking.
    Writes:
      history.npz           — all history arrays
      rep_profiles.npz      — tau and profile arrays for first/middle/last bunch
    """
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTDIR / "history.npz", **history)

    rep = final_data["rep_profiles"]
    flat = {}
    for label, (tau_s, prof) in rep.items():
        flat[f"{label}_tau"] = tau_s
        flat[f"{label}_prof"] = prof
    np.savez(OUTDIR / "rep_profiles.npz", **flat)


def load_run_data():
    """
    Reloads history and rep_profiles from OUTDIR.
    Returns (history, final_data) with the same structure make_plots() expects.
    """
    hist_npz = np.load(OUTDIR / "history.npz", allow_pickle=False)
    history = {k: hist_npz[k] for k in hist_npz.files}
    # wall_clock_s is saved as a 0-d array — restore as plain float
    if history["wall_clock_s"].ndim == 0:
        history["wall_clock_s"] = float(history["wall_clock_s"])

    prof_npz = np.load(OUTDIR / "rep_profiles.npz", allow_pickle=False)
    rep_profiles = {}
    for label in ("first", "middle", "last"):
        tau_key, prof_key = f"{label}_tau", f"{label}_prof"
        if tau_key in prof_npz.files:
            rep_profiles[label] = (prof_npz[tau_key], prof_npz[prof_key])

    final_data = {"rep_profiles": rep_profiles}
    return history, final_data


def make_plots(ring, history, final_data, eq_cm_s=None):
    """
    Generate and save every diagnostic plot of the run into OUTDIR. Rank 0 only;
    writes PDF files, returns nothing. eq_cm_s is accepted for signature symmetry
    with the precheck but unused (discarded below).

    Plots written (file name -> content):
      V_beam_MC_vs_pass.pdf              beam-induced voltage per MC cavity
      V_beam_HC_vs_pass.pdf              beam-induced voltage per HC cavity (USE_HC only)
      V_c_MC_vs_pass.pdf                 total (gen+beam) voltage per MC cavity
      V_c_HC_vs_pass.pdf                 total voltage per HC cavity (USE_HC only)
      E_vs_pass.pdf                      mean beam kinetic energy
      ecx_vs_pass.pdf                    horizontal emittance + equilibrium line
      ecy_vs_pass.pdf                    vertical emittance + equilibrium line
      el_vs_pass.pdf                     longitudinal emittance + target line
      bunch_profiles_current.pdf         current profiles of first/middle/last bunch
      bunch_length_per_bunch_vs_turn.pdf per-bunch RMS length + target + last-N avg
      MC_voltage_and_Ig_vs_turn.pdf      MC voltage and generator current (twin axes)
      HC_voltage_and_Ig_vs_turn.pdf      HC voltage and generator current (USE_HC only)

    The HC-specific plots are written only when USE_HC is True.
    """

    del eq_cm_s
    OUTDIR.mkdir(parents=True, exist_ok=True)
    turn = history["turn"]

    def _tx(ax):
        t = ax.secondary_xaxis("top",
            functions=(lambda t: t * ring.T0, lambda s: s / ring.T0))
        t.set_xlabel("time [s]")
        return t

    # -------------------------------------------------------------------------
    # Plot 1: beam-induced voltage per MC cavity vs pass
    # (ELEGANT: V_beam_MC_vs_pass)
    # mbtrack2 source: mc.beam_phasor_record, averaged over
    # the train and divided by N_MC
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(turn, history["Vbeam_MC_kV"] * 1e3, color=C_MC)
    ax.set_xlabel("Pass")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)
    _tx(ax)
    fig.savefig(OUTDIR / "V_beam_MC_vs_pass.pdf")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 2: beam-induced voltage per HC cavity vs pass
    # (ELEGANT: V_beam_HC_vs_pass)
    # Only written when USE_HC is True
    # -------------------------------------------------------------------------
    if USE_HC:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.plot(turn, history["Vbeam_HC_kV"] * 1e3, color=C_HC)
        ax.set_xlabel("Pass")
        ax.set_ylabel("Voltage (V)")
        ax.grid(True, alpha=0.3)
        _tx(ax)
        fig.savefig(OUTDIR / "V_beam_HC_vs_pass.pdf")
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 3: total cavity voltage per MC cavity vs pass
    # (ELEGANT: V_c_MC_vs_pass)
    # mbtrack2 source: mc.cavity_phasor_record = generator + beam, averaged
    # over train
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(turn, history["V_MC_kV"] * 1e3, color=C_MC)
    ax.set_xlabel("Pass")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)
    _tx(ax)
    fig.savefig(OUTDIR / "V_c_MC_vs_pass.pdf")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 4: total cavity voltage per HC cavity vs pass
    # (ELEGANT: V_c_HC_vs_pass)
    # Only written when USE_HC is True
    # -------------------------------------------------------------------------
    if USE_HC:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.plot(turn, history["V_HC_kV"] * 1e3, color=C_HC)
        ax.set_xlabel("Pass")
        ax.set_ylabel("Voltage (V)")
        ax.grid(True, alpha=0.3)
        _tx(ax)
        fig.savefig(OUTDIR / "V_c_HC_vs_pass.pdf")
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 5: mean beam kinetic energy vs pass  (ELEGANT: E_vs_pass)
    # K = E0 * (1 + <delta>) in MeV, averaged over all bunches
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(turn, history["K_MeV"], color="k")
    ax.set_xlabel("Pass")
    ax.set_ylabel("Mean energy (MeV)")
    ax.grid(True, alpha=0.3)
    _tx(ax)
    fig.savefig(OUTDIR / "E_vs_pass.pdf")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 6: horizontal emittance vs pass  (ELEGANT: ecx_vs_pass)
    # Computed from 6D coordinates as eps_x = sqrt(<x^2><x'^2> - <x x'>^2),
    # train average
    # No dispersive correction needed because DISPERSION_LOCAL = [0, 0, 0, 0]
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(turn, history["eps_x_m"], color=C_MC)
    ax.axhline(EMIT_X_MRAD, linestyle="--", color=C_OPT,
               label=f"Equilibrium ({EMIT_X_MRAD:.3e} m)")
    ax.legend()
    ax.set_xlabel("Pass")
    ax.set_ylabel(r"$\varepsilon_{x,c}$ (m)")
    ax.grid(True, alpha=0.3)
    _tx(ax)
    fig.savefig(OUTDIR / "ecx_vs_pass.pdf")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 7: vertical emittance vs pass  (ELEGANT: ecy_vs_pass)
    # Same formula as eps_x but using y, y' coordinates, train average
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(turn, history["eps_y_m"], color=C_MC)
    ax.axhline(EMIT_Y_MRAD, linestyle="--", color=C_OPT,
               label=f"Equilibrium ({EMIT_Y_MRAD:.3e} m)")
    ax.legend()
    ax.set_xlabel("Pass")
    ax.set_ylabel(r"$\varepsilon_{y,c}$ (m)")
    ax.grid(True, alpha=0.3)
    _tx(ax)
    fig.savefig(OUTDIR / "ecy_vs_pass.pdf")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 8: longitudinal emittance vs pass  (ELEGANT: el_vs_pass)
    # eps_l = sigma_t * sigma_delta  [units: s], train average
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(turn, history["eps_l_s"], color=C_MC)
    target_el_s = (TARGET_SIGMA_T_PS * 1e-12) * SIGMA_DELTA
    ax.axhline(target_el_s, linestyle="--", color=C_OPT,
               label=f"Target ({target_el_s:.3e} s)")
    ax.legend()
    ax.set_xlabel("Pass")
    ax.set_ylabel(r"$\varepsilon_l$ (s)")
    ax.grid(True, alpha=0.3)
    _tx(ax)
    fig.savefig(OUTDIR / "el_vs_pass.pdf")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 9: Bunch current profiles (first / middle / last bunch)
    # -------------------------------------------------------------------------
    rep_profiles = final_data["rep_profiles"]
    rep_idx = {"first": 0, "middle": H // 2, "last": H - 1}
    I_bunch = TOTAL_CURRENT_A / H
    Q_bunch = I_bunch * ring.T0

    _profile_colors = {"first": C_MC, "middle": C_SEC, "last": C_HC}
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for label in ["first", "middle", "last"]:
        if label not in rep_profiles:
            continue
        tau_s, prof = rep_profiles[label]
        if prof.sum() <= 0:
            continue
        dt = float(tau_s[1] - tau_s[0])
        I_t = prof * Q_bunch / (dt * float(prof.sum()))
        ax.plot(tau_s * 1e12, I_t, color=_profile_colors[label], label=f"Bunch {rep_idx[label]}")
    
    # Analytical equilibrium profile (paramsalba 'Current profile' case),
    # same charge normalization as the simulated curves (int I dt = Q_bunch).
    _, beta_rel = relativistic_factors(E0_EV)
    omega_rf = 2.0 * np.pi * H * (C_LIGHT * beta_rel / CIRCUMFERENCE_M)
    t_ps_th, rho_t_th = analytical_equilibrium_profile(
        omega_rf=omega_rf, Vrf=V_MC_TOTAL_V, V_hc_total=V_HC_TOTAL_V,
        U0_eV=U0_EV, n=m_hc, alpha_c=ALPHA_C, sigma_e=SIGMA_DELTA,
        E0_eV=E0_EV, C=CIRCUMFERENCE_M,
    )
    I_th = rho_t_th * Q_bunch
    # Restrict to the simulated window so the x-range stays tied to the data.
    xlo, xhi = ax.get_xlim()
    win = (t_ps_th >= xlo) & (t_ps_th <= xhi)
    ax.plot(t_ps_th[win], I_th[win], color=C_OPT, ls="--", lw=2.0,
            label="Theoretical (paramsalba)")
    ax.set_xlim(xlo, xhi)
    
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Bunch current [A]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(OUTDIR / "bunch_profiles_current.pdf")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 10: Per-bunch RMS bunch length vs turn + target
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(turn, history["sigma_t_first_ps"], color="k",
            label=f"Bunch {rep_idx['first']}")
    ax.plot(turn, history["sigma_t_middle_ps"], color=C_SEC,
            label=f"Bunch {rep_idx['middle']}")
    ax.plot(turn, history["sigma_t_last_ps"], color=C_HC,
            label=f"Bunch {rep_idx['last']}")
    ax.axhline(TARGET_SIGMA_T_PS, linestyle="--", color=C_OPT,
               label=f"Target ({TARGET_SIGMA_T_PS:.2f} ps)")
    mean_final = safe_last_mean(history["sigma_t_ps"])
    ax.axhline(mean_final, linestyle=":", color=C_SEC,
               label=f"Last {LAST_AVG_N} pts avg = {mean_final:.2f} ps")
    ax.text(0.05, 0.05,
            f"All beam avg RMS: {mean_final:.2f} ps",
            transform=ax.transAxes, bbox=_BBOX)
    ax.set_xlabel("Turn number")
    ax.set_ylabel("Bunch length (RMS) [ps]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _tx(ax)
    fig.savefig(OUTDIR / "bunch_length_per_bunch_vs_turn.pdf")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 11: MC voltage and I_g vs turn (twin y-axes)
    # -------------------------------------------------------------------------
    fig, ax_v = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax_v.plot(turn, history["V_MC_kV"], color=C_MC)
    ax_v.set_xlabel("Turn")
    ax_v.set_ylabel("Voltage [kV]", color=C_MC)
    ax_v.tick_params(axis="y", labelcolor=C_MC)
    ax_v.grid(True, alpha=0.3)
    ax_i = ax_v.twinx()
    ax_i.plot(turn, history["Ig_MC_A"], color=C_OPT)
    ax_i.set_ylabel(r"$I_g$ [A]", color=C_OPT)
    ax_i.tick_params(axis="y", labelcolor=C_OPT)
    _tx(ax_v)
    fig.savefig(OUTDIR / "MC_voltage_and_Ig_vs_turn.pdf")
    plt.close(fig)


    # -------------------------------------------------------------------------
    # Plot 12: HC voltage and I_g vs turn (twin y-axes)
    # -------------------------------------------------------------------------
    if USE_HC:
        fig, ax_v = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax_v.plot(turn, history["V_HC_kV"], color=C_HC)
        ax_v.set_xlabel("Turn")
        ax_v.set_ylabel("Voltage [kV]", color=C_HC)
        ax_v.tick_params(axis="y", labelcolor=C_HC)
        ax_v.grid(True, alpha=0.3)

        ax_i = ax_v.twinx()
        ax_i.plot(turn, history["Ig_HC_A"], color=C_OPT)
        ax_i.set_ylabel(r"$I_g$ [A]", color=C_OPT)
        ax_i.tick_params(axis="y", labelcolor=C_OPT)

        _tx(ax_v)

        fig.savefig(OUTDIR / "HC_voltage_and_Ig_vs_turn.pdf")
        plt.close(fig)


def save_machine_summary(ring, mc, hc, eq_sigma_z_m, eq_sigma_t_s, eq_cm_s,
                         history):
    """
    Write the plain-text run summary to OUTDIR/summary_<VALIDATION_CASE>.txt.
    Rank 0 only; returns nothing.

    Collects in one file: run configuration (case, turns, current, MPI layout,
    n_bin); MC and HC parameters (voltages, theta, frequencies, detuning used);
    equilibrium pre-check results with relative errors vs the targets; final
    tracking results (last point and last-N average); final cavity diagnostics
    (voltages, generator currents, mean beam energy); wall-clock time and beta.
    """

    _, beta_rel = relativistic_factors(E0_EV)

    final_sigma_t_ps = float(history["sigma_t_ps"][-1])
    final_sigma_z_mm = float(history["sigma_z_mm"][-1])

    final_sigma_t_avg_ps = safe_last_mean(history["sigma_t_ps"])
    final_sigma_z_avg_mm = safe_last_mean(history["sigma_z_mm"])

    mc_detune_used_hz = getattr(mc, "detune_used_hz", np.nan)
    hc_detune_used_hz = getattr(hc, "detune_used_hz", np.nan) if hc is not None else np.nan
    
    mpi_size = MPI.COMM_WORLD.Get_size()

    lines = [
        "ALBA II longitudinal tracking with mbtrack2",
        "==========================================",
        "",
        f"Validation case                 : {VALIDATION_CASE}",
        f"USE_HC                          : {USE_HC}",
        "",
        f"Turns                          : {N_TURNS}",
        f"Macro-particles per bunch      : {MP_PER_BUNCH}",
        f"Total beam current             : {TOTAL_CURRENT_A:.6f} A",
        f"Harmonic number                : {H}",
        f"Circumference                  : {CIRCUMFERENCE_M:.12f} m",
        f"Main RF frequency              : {ring.f1:.12e} Hz",
        f"MPI ranks                      : {mpi_size}",
        f"MPI bunch partition            : {H // mpi_size} bunches per rank",
        f"Cavity n_bin                   : {CAVITY_NBIN}",
        f"Profile n_bin                  : {PROFILE_NBIN}",
        "",
        "Main cavity:",
        f"  Ncav                         : {N_MC}",
        f"  Vc(total)                    : {V_MC_TOTAL_V:.6e} V",
        f"  Vc(per cavity)               : {V_MC_TOTAL_V / N_MC:.6e} V",
        f"  theta(cos convention)        : {np.rad2deg(THETA_MC_COS_RAD):.6f} deg",
        f"  resonance frequency          : {FREQ_MC_RESONANCE_HZ:.12e} Hz",
        f"  Elegant drive_frequency      : {FREQ_MC_DRIVE_HZ:.12e} Hz",
        f"  detune used in mbtrack2      : {mc_detune_used_hz:.6f} Hz",
    ]

    if hc is None:
        lines += [
            "",
            "Harmonic cavity:",
            "  not used in this validation case",
        ]
    else:
        lines += [
            "",
            "Harmonic cavity:",
            f"  Ncav                         : {N_HC}",
            f"  Vc(total)                    : {V_HC_TOTAL_V:.6e} V",
            f"  Vc(per cavity)               : {V_HC_TOTAL_V / N_HC:.6e} V",
            f"  theta(cos convention)        : {np.rad2deg(THETA_HC_COS_RAD):.6f} deg",
            f"  resonance frequency          : {FREQ_HC_RESONANCE_HZ:.12e} Hz",
            f"  Elegant drive_frequency      : {FREQ_HC_DRIVE_HZ:.12e} Hz",
            f"  detune used in mbtrack2      : {hc_detune_used_hz:.6f} Hz",
            f"  Rs(per cavity, storage ring) : {RS_HC_PER_CAVITY_OHM:.6e} Ohm",
        ]


        if "FREQ_HC_RESONANCE_HZ_OVERRIDE" in CASE:
            lines += [
                f"  resonance freq (analytical)  : {FREQ_HC_RESONANCE_HZ_ANALYTICAL:.12e} Hz  (NOT used: override active)",
                f"  override delta vs analytical : {(FREQ_HC_RESONANCE_HZ - FREQ_HC_RESONANCE_HZ_ANALYTICAL):.3f} Hz",
            ]

    lines += [
        "",
        "Equilibrium pre-check:",
        f"  sigma_z(eq)                  : {eq_sigma_z_m * 1e3:.6f} mm",
        f"  sigma_t(eq)                  : {eq_sigma_t_s * 1e12:.6f} ps",
        f"  sigma_z target               : {TARGET_SIGMA_Z_MM:.6f} mm",
        f"  sigma_t target               : {TARGET_SIGMA_T_PS:.6f} ps",
        f"  rel. error sigma_z(eq)       : {(eq_sigma_z_m * 1e3 - TARGET_SIGMA_Z_MM) / TARGET_SIGMA_Z_MM:.6%}",
        f"  rel. error sigma_t(eq)       : {(eq_sigma_t_s * 1e12 - TARGET_SIGMA_T_PS) / TARGET_SIGMA_T_PS:.6%}",
        f"  center of mass(eq)           : {eq_cm_s * 1e12:.6f} ps",
        "",
        "Final tracking result, last saved point:",
        f"  sigma_z(final)               : {final_sigma_z_mm:.6f} mm",
        f"  sigma_t(final)               : {final_sigma_t_ps:.6f} ps",
        f"  sigma_z target               : {TARGET_SIGMA_Z_MM:.6f} mm",
        f"  sigma_t target               : {TARGET_SIGMA_T_PS:.6f} ps",
        f"  rel. error sigma_z           : {(final_sigma_z_mm - TARGET_SIGMA_Z_MM) / TARGET_SIGMA_Z_MM:.6%}",
        f"  rel. error sigma_t           : {(final_sigma_t_ps - TARGET_SIGMA_T_PS) / TARGET_SIGMA_T_PS:.6%}",
        "",
        f"Final tracking result, average of last {LAST_AVG_N} saved points:",
        f"  sigma_z(avg)                 : {final_sigma_z_avg_mm:.6f} mm",
        f"  sigma_t(avg)                 : {final_sigma_t_avg_ps:.6f} ps",
        f"  rel. error sigma_z(avg)      : {(final_sigma_z_avg_mm - TARGET_SIGMA_Z_MM) / TARGET_SIGMA_Z_MM:.6%}",
        f"  rel. error sigma_t(avg)      : {(final_sigma_t_avg_ps - TARGET_SIGMA_T_PS) / TARGET_SIGMA_T_PS:.6%}",
        "",
        "Final cavity diagnostics, average of last saved points:",
        f"  V_MC per cavity              : {safe_last_mean(history['V_MC_kV']):.3f} kV",
        f"  I_g (MC)                     : {safe_last_mean(history['Ig_MC_A']):.4f} A",
    ]

    if hc is None:
        lines += [
            "  V_HC per cavity              : not used",
            "  I_g (HC)                     : not used",
        ]
    else:
        lines += [
            f"  V_HC per cavity              : {safe_last_mean(history['V_HC_kV']):.3f} kV",
            f"  I_g (HC)                     : {safe_last_mean(history['Ig_HC_A']):.4f} A",
        ]

    lines += [
        f"  Mean beam energy             : {safe_last_mean(history['K_MeV']):.3f} MeV",
        "",
        f"Wall-clock time                : {history['wall_clock_s']:.6f} s",
        f"Relativistic beta              : {beta_rel:.12f}",
    ]

    write_summary_file(OUTDIR / f"summary_{VALIDATION_CASE}.txt", lines)