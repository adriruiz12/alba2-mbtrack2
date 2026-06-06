"""
ALBA II longitudinal tracking with double RF (main + harmonic) in mbtrack2,
parallelised over a small number of MPI ranks (16 by default) for 448 filled
bunches.

This file is only the entry point: it wires together the modules and runs
main(). See README.md for the physics background, the custom-MPI rationale and
the U0 / RF-convention notes.

Module layout
-------------
    constants.py    static physical / machine / RF constants
    config.py       run knobs + validation-case resolution + case-dependent RF
    helpers.py  pure helpers (unit conversions, relativistic factors)
    machine.py      builders for ring / cavities / bunches + cavity ops
    parallel.py     custom MPI layer (bunch distribution + profile exchange)
    tracking.py     equilibrium pre-check + cavity tracker + tracking loop
    diagnostics.py  plots + machine summary (rank 0 only)

Execute with   sbatch submit_soa_base_cpar.sl
Track with   tail -f results/mbxlogs/soa_base_<arrayjobid>_<taskid>.txt
"""

import numpy as np
from mpi4py import MPI

from constants import N_HC, N_MC, TOTAL_CURRENT_A
from config import (
    LAST_AVG_N,
    SEED,
    TARGET_SIGMA_T_PS,
    TARGET_SIGMA_Z_MM,
    USE_HC,
    V_HC_PER_CAVITY_V,
    V_HC_TOTAL_V,
    VALIDATION_CASE,
)
from helpers import safe_last_mean
from machine import build_cavities, build_local_bunches, build_ring, reset_cavity_state
from parallel import local_bunch_indices, validate_parallel_layout
from tracking import collect_final_data, equilibrium_precheck, run_tracking
from diagnostics import make_plots, save_machine_summary, save_run_data


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size, bunches_per_rank = validate_parallel_layout(comm)
    my_indices = local_bunch_indices(rank, size)

    # Per-bunch deterministic seeds for initialization were already used.
    # Here we set a rank seed for turn-by-turn stochastic radiation kicks.
    np.random.seed(SEED + 1000 * rank)

    ring = build_ring()
    mc, hc = build_cavities(ring)

    eq_sigma_z_m, eq_sigma_t_s, eq_cm_s = equilibrium_precheck(ring, mc, hc, rank)

    # Start tracking from a clean cavity state after the static equilibrium check.
    mc.set_generator(TOTAL_CURRENT_A)
    reset_cavity_state(mc)

    if hc is not None:
        hc.set_generator(TOTAL_CURRENT_A)
        reset_cavity_state(hc)

    if comm.rank == 0:
        print(f"\n[VALIDATION CASE] {VALIDATION_CASE}")
        print(f"[VALIDATION CASE] USE_HC = {USE_HC}")
        print(f"[VALIDATION CASE] Target sigma_z = {TARGET_SIGMA_Z_MM:.6f} mm")
        print(f"[VALIDATION CASE] Target sigma_t = {TARGET_SIGMA_T_PS:.6f} ps")
        print(f"[VALIDATION CASE] V_HC per cavity = {V_HC_PER_CAVITY_V/1e3:.6f} kV")
        print(f"[VALIDATION CASE] V_HC total      = {V_HC_TOTAL_V/1e3:.6f} kV")
        print(f"\n[DIAGNOSTIC] ring.ac      = {ring.ac:.6e}")
        print(f"[DIAGNOSTIC] ring.sigma_delta  = {ring.sigma_delta:.6e}")
        print(f"[DIAGNOSTIC] ring.U0           = {ring.U0:.6e} eV")
        print(f"[DIAGNOSTIC] ring.tau_z        = {ring.tau[2]:.6e} s  ({ring.tau[2] / ring.T0:.1f} turns)")

    local_bunches = build_local_bunches(ring, my_indices)

    history = run_tracking(comm, ring, my_indices, local_bunches, mc, hc)
    final_data = collect_final_data(comm, my_indices, local_bunches, mc, hc) # Studying this function with the debugging mode is difficult because all the tracking would have to run before

    if rank == 0:
        save_run_data(history, final_data)
        make_plots(ring, history, final_data, eq_cm_s)
        save_machine_summary(ring, mc, hc, eq_sigma_z_m, eq_sigma_t_s, eq_cm_s, history)

        final_sigma_t_ps = float(history["sigma_t_ps"][-1])
        final_sigma_z_mm = float(history["sigma_z_mm"][-1])

        final_sigma_t_avg_ps = safe_last_mean(history["sigma_t_ps"])
        final_sigma_z_avg_mm = safe_last_mean(history["sigma_z_mm"])

        mc_total_per_cavity_kv = np.mean(np.abs(final_data["mc_total_phasor_record"])) / N_MC / 1e3
        mc_beam_per_cavity_kv = np.mean(np.abs(final_data["mc_beam_phasor_record"])) / N_MC / 1e3

        if hc is not None:
            hc_total_per_cavity_kv = np.mean(np.abs(final_data["hc_total_phasor_record"])) / N_HC / 1e3
            hc_beam_per_cavity_kv = np.mean(np.abs(final_data["hc_beam_phasor_record"])) / N_HC / 1e3
        else:
            hc_total_per_cavity_kv = 0.0
            hc_beam_per_cavity_kv = 0.0

        print("\n=== FINAL COMPARISON ===")
        print(f"Final sigma_t, last point        : {final_sigma_t_ps:.3f} ps   | target {TARGET_SIGMA_T_PS:.3f} ps")
        print(f"Final sigma_z, last point        : {final_sigma_z_mm:.3f} mm   | target {TARGET_SIGMA_Z_MM:.3f} mm")
        print(f"Final sigma_t, last {LAST_AVG_N} avg : {final_sigma_t_avg_ps:.3f} ps   | target {TARGET_SIGMA_T_PS:.3f} ps")
        print(f"Final sigma_z, last {LAST_AVG_N} avg : {final_sigma_z_avg_mm:.3f} mm   | target {TARGET_SIGMA_Z_MM:.3f} mm")

        print("\n=== CAVITY DIAGNOSTICS ===")
        print(f"MC total voltage / cavity        : {mc_total_per_cavity_kv:.3f} kV")
        print(f"MC beam-loading voltage / cavity : {mc_beam_per_cavity_kv:.3f} kV")
        print(f"MC detune used in mbtrack2       : {getattr(mc, 'detune_used_hz', np.nan):.6f} Hz")

        if hc is not None:
            print(f"HC total voltage / cavity        : {hc_total_per_cavity_kv:.3f} kV")
            print(f"HC beam-loading voltage / cavity : {hc_beam_per_cavity_kv:.3f} kV")
            print(f"HC detune used in mbtrack2       : {getattr(hc, 'detune_used_hz', np.nan):.6f} Hz")

        else:
            print("HC total voltage / cavity        : not used")
            print("HC beam-loading voltage / cavity : not used")
            print("HC detune used in mbtrack2       : not used")

    # -------------------------------------------------------------------------
    # Clean ending for all MPI ranks.
    # -------------------------------------------------------------------------
    # At this point rank 0 has finished plotting and saving files.

    # As all ranks run concurrently, Ranks 1..N-1 skip the plotting and summary
    # block and reach this point before rank 0. comm.Barrier() holds them here
    # until rank 0 finishes its output work and joins, so all ranks exit cleanly
    # together.
    comm.Barrier()
    if rank == 0:
        print("\n=== MPI PROGRAM FINISHED CLEANLY ===", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc: # Abort all ranks on any unhandled exception
        comm = MPI.COMM_WORLD
        print(f"[rank {comm.Get_rank()}] ERROR: {exc}", flush=True)
        comm.Abort(1)