"""
tracking.py - Core physics of the simulation.

Contains the static equilibrium pre-check, the custom distributed cavity
tracker (the many-bunches-per-rank version of mbtrack2's CavityResonator.track),
the main turn-by-turn tracking loop, and the final data collection.
"""

import time

import numpy as np
from mpi4py import MPI

from mbtrack2.utilities import BeamLoadingEquilibrium
from mbtrack2.tracking import LongitudinalMap, SynchrotronRadiation

from constants import C_LIGHT, E0_EV, H, N_HC, N_MC, TOTAL_CURRENT_A
from config import (
    EQ_HALF_WINDOW_S,
    EQ_NBIN,
    MP_PER_BUNCH,
    N_TURNS,
    PROFILE_NBIN,
    SAVE_EVERY,
    TARGET_SIGMA_T_PS,
    TARGET_SIGMA_Z_MM,
)
from helpers import (
    relativistic_factors,
    sigma_t_to_sigma_z,
    sigma_z_to_sigma_t,
)
from parallel import (
    allgather_profiles,
    compute_local_profiles,
    gather_checkpoint_stats,
)
from machine import enforce_active_cavity_setpoint, total_cavity_phasor_record


def representative_indices():
    """
    Returns indices of three representative bunches (first, middle, last)
    used for diagnostic plots of longitudinal profiles along the train.
    """

    return {
        "first": 0,
        "middle": H // 2,
        "last": H - 1,
    }


def bunch_profile_tau(bunch, n_bin=PROFILE_NBIN):
    """
    Returns bin centres and peak-normalised longitudinal profile of a bunch
    in the tau dimension, binned into n_bin intervals.
    """

    _, _, profile, center = bunch.binning(dimension="tau", n_bin=n_bin)
    profile = profile.astype(float)
    if profile.max() > 0:
        profile /= profile.max()
    return center, profile


def equilibrium_precheck(ring, mc, hc, rank):
    """
    Paramsalba-like static precheck.

    This precheck intentionally does NOT call beam_equilibrium(), because
    beam_equilibrium() solves the self-consistent form factors F and PHI.
    That is useful for a passive/self-consistent beam-loading equilibrium,
    but it is not the same model as paramsalba, where the RF potential is
    built from fixed active-cavity voltage setpoints. Here we keep:
        F   = [1, 1]
        PHI = [0, 0],
    where F and PHI are the bunch form factor and its phase for each cavity.
    F is the normalised magnitude of the beam's Fourier component at the
    cavity harmonic (1 for a point bunch, <1 for a finite-length bunch).
    Setting F=1, PHI=0 fixes the cavity voltage at its nominal active setpoint,
    matching the fixed-potential model of paramsalba, rather than solving
    a self-consistent passive equilibrium. So the precheck uses the fixed RF
    potential corresponding to the active MC and active HC setpoints.

    NOTE: BeamLoadingEquilibrium assumes linear momentum compaction (ring.ac);
    higher-order terms are only exercised by the actual tracking map. The
    precheck validates the fixed RF potential; it does not guarantee that
    the active beam-loading tracking converges to the same value.

    Returns sigma_z [m], sigma_t [s], and centre-of-mass [s] at equilibrium.
    """

    B_HALF_M = C_LIGHT * EQ_HALF_WINDOW_S

    cavity_list = [mc] if hc is None else [mc, hc]

    # This computes everything (and saves it in 'eq')
    eq = BeamLoadingEquilibrium(
        ring=ring,
        cavity_list=cavity_list,
        I0=TOTAL_CURRENT_A,
        auto_set_MC_theta=False,
        F=np.ones(len(cavity_list)),
        PHI=np.zeros(len(cavity_list)),
        B1=-B_HALF_M,
        B2=+B_HALF_M,
        N=EQ_NBIN,
    )

    eq.update_potentials()
    eq.update_rho()

    sigma_z_eq_m = float(eq.std_rho())
    sigma_t_eq_s = sigma_z_to_sigma_t(sigma_z_eq_m, relativistic_factors(E0_EV)[1])
    cm_eq_s = float(eq.center_of_mass())

    if rank == 0:
        sigma_z_eq_mm = sigma_z_eq_m * 1e3
        sigma_t_eq_ps = sigma_t_eq_s * 1e12
        rel_err_eq = (sigma_t_eq_ps - TARGET_SIGMA_T_PS) / TARGET_SIGMA_T_PS

        print("\n=== PARAMSALBA-LIKE EQUILIBRIUM PRE-CHECK ===")
        print("Model              : fixed active RF potential")
        print(f"beam_equilibrium() : NOT used")
        print(f"F                  : {eq.F}")
        print(f"PHI                : {eq.PHI}")
        print(f"Equilibrium sigma_z = {sigma_z_eq_mm:.3f} mm")
        print(f"Equilibrium sigma_t = {sigma_t_eq_ps:.3f} ps")
        print(f"Target sigma_z      = {TARGET_SIGMA_Z_MM:.3f} mm")
        print(f"Target sigma_t      = {TARGET_SIGMA_T_PS:.3f} ps")
        print(f"Relative error      = {rel_err_eq:.3%}")
        print(f"Center of mass      = {eq.center_of_mass() * 1e12:.3f} ps")
        print(f"Energy balance      = {eq.energy_balance():.6e}")

        eb = eq.energy_balance()
        if abs(eb) > 100:   # reasonable threshold in V
            print(f"NOTE: energy balance = {eb:.1f} V >> 0.")
            print("      The synchronous phase is not at tau=0 for this case.")
            print("      This is expected near the flat-potential condition (xi~1).")
            print("      Pre-check sigma_z may differ from the paramsalba target.")

        # Extra local checks around tau = 0.
        print("")
        print("Local RF-potential diagnostics at z=0:")
        print(f"V_RF(0)   = {float(eq.voltage(np.array([0.0]))[0]):.6e} V")
        print(f"dV_RF(0)  = {float(eq.dV(np.array([0.0]))[0]):.6e}")
        print(f"ddV_RF(0) = {float(eq.ddV(np.array([0.0]))[0]):.6e}")

        print("")
        print("Cavity diagnostics:")
        print(f"mc theta deg   = {np.rad2deg(mc.theta):.6f}")
        print(f"mc theta_g deg = {np.rad2deg(mc.theta_g):.6f}")
        print(f"mc psi deg     = {np.rad2deg(mc.psi):.6f}")
        print(f"mc Vg          = {mc.Vg:.6e} V")
        print(f"mc Vb          = {mc.Vb(TOTAL_CURRENT_A):.6e} V")

        if hc is not None:
            print(f"hc theta deg   = {np.rad2deg(hc.theta):.6f}")
            print(f"hc theta_g deg = {np.rad2deg(hc.theta_g):.6f}")
            print(f"hc psi deg     = {np.rad2deg(hc.psi):.6f}")
            print(f"hc Vg          = {hc.Vg:.6e} V")
            print(f"hc Vb          = {hc.Vb(TOTAL_CURRENT_A):.6e} V")
        else:
            print("hc             = not used in this validation case")

        if abs(rel_err_eq) > 0.05:
            print("")
            print("WARNING: fixed-potential precheck is still far from paramsalba.")
            print("If this happens, the remaining issue is not beam_equilibrium(),")
            print("but a phase, voltage, U0, Rs, detuning, or convention mismatch.")

    return sigma_z_eq_m, sigma_t_eq_s, cm_eq_s


# In this simulation, the dominant collective effect included explicitly is the
# beam loading of the RF cavities. This can be interpreted as the beam exciting
# a resonant electromagnetic wake in the cavity, which is then represented through
# the cavity beam phasor and applied back to the bunches as an energy kick.

def track_cavity_distributed(
    cavity, local_indices_arr, local_bunches, local_slot_of_global, local_sorted_index,
    global_centers, global_profiles, global_bin_lengths, global_charge_per_mp,
):
    """
    Track one CavityResonator through the full bunch train using the custom
    many-bunches-per-rank MPI layout.

    "Track one CavityResonator" means a single pass for one cavity that does
    two things at once: it advances the cavity beam phasor around the whole
    ring, and applies the resulting voltage kick to the local particles. It is
    called per-cavity because run_tracking invokes it once for the MC and once
    for the HC.

    Bunches interact with each other only through the cavity: each bunch
    deposits charge that updates cavity.beam_phasor, which then decays and
    affects the following bunches. This is the cavity beam-loading wake. All
    other collective effects (resistive wall, broadband wakes, etc.) are not
    modelled.

    All MPI ranks execute this function simultaneously, each with its own copy
    of the cavity object, following these steps:

      1. Every rank already holds the profiles of all H bunches from the
         preceding allgather_profiles call.
      2. Each rank loops over all H buckets in global order (bunch_idx = 0..H-1).
      3. For a bucket this rank does NOT own (slot < 0): only the cavity beam
         phasor is advanced through that bunch. No particles are touched.
      4. For a bucket this rank DOES own (slot >= 0): the function walks the
         bunch bin by bin, computes the total voltage (generator + beam-loading
         - self-loss), updates the beam phasor with the charge just deposited,
         and applies the per-bin kick to the local particles' delta.
      5. After each bunch, the beam phasor is saved to the record and decayed
         by one bucket spacing so the next bunch sees the correctly aged wake.

    Because all ranks have all profiles, the phasor evolution in step 2 is
    identical on every rank, keeping the cavity state consistent without extra
    communication. The only rank-specific work is the particle kick in step 4.
    """

    for bunch_idx in range(H):

        center          = global_centers[bunch_idx]
        profile         = global_profiles[bunch_idx]
        bin_length      = float(global_bin_lengths[bunch_idx])
        charge_per_mp   = float(global_charge_per_mp[bunch_idx])
        slot            = local_slot_of_global[bunch_idx]

        if slot >= 0:
            bunch        = local_bunches[bunch_idx]
            sorted_index = local_sorted_index[slot]
            bin_kick     = np.zeros_like(center, dtype=np.float64)
        else:
            bunch = sorted_index = bin_kick = None

        cavity.phasor_decay(center[0] - bin_length / 2.0, ref_frame="beam")
        if slot < 0:
            cavity.phasor_evol(profile, bin_length, charge_per_mp, ref_frame="beam")
            cavity.phasor_decay(-(center[-1] + bin_length / 2.0), ref_frame="beam") # Return phasor to t=0 of current bunch before saving record
        else:
            for ibin, center0 in enumerate(center):
                mp_per_bin = int(profile[ibin])
                if mp_per_bin == 0:
                    cavity.phasor_decay(bin_length, ref_frame="beam")
                    continue
                # Full phase at bin centre, including the bunch timing offset in
                # the ring. (omega1*T1 = 2*pi exactly, so the bunch_idx + h*nturn
                # term is an integer multiple of 2*pi for any cavity harmonic m
                # and drops out of the complex exponential. Kept here for clarity)
                phase  = cavity.m * cavity.ring.omega1 * (
                    center0 + cavity.ring.T1 * (bunch_idx + cavity.ring.h *
                                                cavity.nturn)
                )
                v_g = np.real(cavity.generator_phasor_record[bunch_idx] *
                              np.exp(1j * phase))
                v_beam = np.real(cavity.beam_phasor)   # phasor_decay rotates bin to bin
                v_tot  = v_g + v_beam - charge_per_mp * cavity.loss_factor * mp_per_bin
                bin_kick[ibin] = v_tot / cavity.ring.E0
                cavity.beam_phasor -= 2.0 * charge_per_mp * cavity.loss_factor * mp_per_bin # beam-loading update
                                
                cavity.phasor_decay(bin_length, ref_frame="beam")
            cavity.phasor_decay(-(center[-1] + bin_length / 2.0), ref_frame="beam")
            bunch["delta"] += bin_kick[sorted_index]
        cavity.beam_phasor_record[bunch_idx] = cavity.beam_phasor
        cavity.phasor_decay(cavity.ring.T1, ref_frame="beam")
    # for fb in cavity.feedback:
    #     fb.track() # currently replaced by enforce_active_cavity_setpoint
    cavity.nturn += 1


def run_tracking(comm, ring, local_indices_arr, local_bunches, mc, hc):
    """
    Main distributed tracking loop. This function advances the local bunches for
    N_TURNS turns and, on rank 0, builds a history dictionary containing the
    evolution of the main diagnostics: mean bunch length, mean bunch duration,
    bunch centroid, mean beam energy, cavity voltages and generator currents.

    Tracking order per turn:
        1. Compute local longitudinal profiles.
        2. Exchange the profiles across all MPI ranks with Allgather.
        3. Enforce active-cavity setpoint on MC (pre-kick).
        4. Apply the main-cavity kick (track_cavity_distributed).
        5. Enforce active-cavity setpoint on MC (post-kick).
        6. If HC enabled: enforce setpoint, apply HC kick, enforce setpoint again.
        7. Apply the longitudinal map.
        8. Apply synchrotron radiation.

    The enforce_active_cavity_setpoint() calls before and after each cavity
    kick keep the generator phasor consistent with the beam-loading state.

    The cavity kicks are applied before the longitudinal map and radiation,
    matching the intended element ordering used for comparison with the Elegant
    setup (RFMODE before the ILMATRIX / radiation elements in the ring).

    Parallelization:
        Bunches are distributed across MPI ranks. Each rank owns and modifies only
        its local Bunch objects, but every rank needs the longitudinal profiles of
        the full bunch train in order to evolve the cavity beam phasor consistently.
        Therefore, each turn, local profiles are computed first and then shared
        with all ranks before calling the distributed cavity tracker.

    Active-cavity setpoint:
        Before and after each cavity kick, the generator phasor is reset bucket
        by bucket so that
            generator_phasor_record + beam_phasor_record = target cavity phasor
        at tau = 0 for each bucket. This is the simplified active-LLRF model used
        here to keep the MC and HC at their fixed voltage and phase setpoints,
        consistently with the fixed-potential paramsalba reference.

    The phase arrays and target phasors used for this correction are precomputed
    once before the tracking loop because they depend only on fixed ring and cavity
    parameters.

    Radiation damping:
        The longitudinal damping time is taken from the Synchrotron object through
        ring.tau[2]. With the current paramsalba value of U0, tau_z is about
        6.06 ms, corresponding to about 6.8e3 turns. This sets the natural scale
        on which longitudinal oscillations are damped. The active-cavity setpoint
        correction additionally suppresses residual beam-cavity voltage drifts by
        keeping the total cavity phasor at the desired setpoint.

    Every SAVE_EVERY turns (and on the first turn), rank 0 gathers bunch
    statistics from all ranks and appends them to the history arrays. The
    history dictionary is returned on rank 0; all other ranks return None.
    """

    rank = comm.rank
    size = comm.Get_size()
    _, beta_rel = relativistic_factors(E0_EV)

    # One-turn longitudinal map: subtracts the energy loss U0/E0 from delta and
    # converts the energy offset delta into an arrival-time shift tau.
    longitudinal_map = LongitudinalMap(ring)

    # Applies radiation damping and quantum excitation turn by turn.
    radiation = SynchrotronRadiation(ring)

    local_slot_of_global = -np.ones(H, dtype=int)
    local_slot_of_global[local_indices_arr] = np.arange(len(local_indices_arr),
                                                        dtype=int)

    history_turn = []
    history_sigma_z_mm = []
    history_sigma_t_ps = []
    history_mean_tau_ps = []
    history_K_MeV = []
    history_V_MC_kV = []
    history_V_HC_kV = []
    history_Ig_MC_A = []
    history_Ig_HC_A = []
    history_Vbeam_MC_kV = []
    history_Vbeam_HC_kV = []
    history_sigma_delta  = []
    history_eps_x_m      = []
    history_eps_y_m      = []
    history_eps_l_s      = []
    history_sigma_t_first_ps  = []
    history_sigma_t_middle_ps = []
    history_sigma_t_last_ps   = []

    t0 = time.time()
    t_last = t0

    if rank == 0:
        print("\n=== TRACKING STARTED ===")
        print(f"MPI ranks          : {size}")
        print(f"Bunches/rank       : {len(local_indices_arr)}")
        print(f"Turns              : {N_TURNS}")
        print(f"Macro/bunch        : {MP_PER_BUNCH}")
        print(f"Filled bunches     : {H}")
        print(f"Current per bunch  : {TOTAL_CURRENT_A / H * 1e3:.6f} mA")
        print("")

    # Pre-compute phasors per bucket (constant, only calculated once)
    mc_phase_arr = mc.m * ring.omega1 * ring.T1 * np.arange(H)
    mc_exp_neg_phase = np.exp(-1j * mc_phase_arr)
    mc_setpoint = mc.Vc * np.exp(1j * mc.theta)

    # Apply the active setpoint before the first turn. Otherwise, the first kick
    # would use the generator phasors produced by CavityResonator.set_generator(),
    # not the bucket-by-bucket active LLRF setpoint. For each bucket, the generator
    # phasor is adjusted so that:
    #     generator phasor + beam-induced phasor = target cavity phasor
    # This ensures that the beam sees the desired total cavity voltage from the
    # first turn, instead of using only the generator phasor produced by
    # set_generator().
    enforce_active_cavity_setpoint(mc, mc_setpoint, mc_exp_neg_phase)

    if hc is not None:
        hc_phase_arr = hc.m * ring.omega1 * ring.T1 * np.arange(H)
        hc_exp_neg_phase = np.exp(-1j * hc_phase_arr)
        hc_setpoint = hc.Vc * np.exp(1j * hc.theta)

        enforce_active_cavity_setpoint(hc, hc_setpoint, hc_exp_neg_phase)
    else:
        hc_exp_neg_phase = None
        hc_setpoint = None

    for turn in range(N_TURNS):

        # Step 1: compute local profiles (pre-kick positions)
        (local_centers, local_profiles, local_bin_lengths,
         local_charge_per_mp, local_sorted_index) = compute_local_profiles(local_indices_arr, local_bunches)

        # Step 2: exchange profiles across all MPI ranks
        (global_centers, global_profiles, global_bin_lengths,
         global_charge_per_mp) = allgather_profiles(comm, local_centers, local_profiles,
                                                    local_bin_lengths, local_charge_per_mp)

        # Step 3: enforce MC setpoint (pre-kick)
        enforce_active_cavity_setpoint(mc, mc_setpoint, mc_exp_neg_phase)

        # Step 4: MC cavity kick
        track_cavity_distributed(mc, local_indices_arr, local_bunches, local_slot_of_global,
                                 local_sorted_index, global_centers, global_profiles,
                                 global_bin_lengths, global_charge_per_mp)

        # Step 5: enforce MC setpoint (post-kick)
        enforce_active_cavity_setpoint(mc, mc_setpoint, mc_exp_neg_phase)

        # Step 6: HC enforce setpoint + kick + enforce setpoint (skipped when hc is None)
        if hc is not None:
            enforce_active_cavity_setpoint(hc, hc_setpoint, hc_exp_neg_phase)

            track_cavity_distributed(hc, local_indices_arr, local_bunches, local_slot_of_global,
                                     local_sorted_index, global_centers, global_profiles,
                                     global_bin_lengths, global_charge_per_mp)

            # Enforce HC setpoint after the beam phasor has changed during the kick.
            enforce_active_cavity_setpoint(hc, hc_setpoint, hc_exp_neg_phase)
        
        # Steps 7-8: longitudinal map + synchrotron radiation
        for bunch_idx in local_indices_arr:
            bunch = local_bunches[bunch_idx]
            longitudinal_map.track(bunch)
            radiation.track(bunch)

        if (turn + 1) % SAVE_EVERY == 0 or turn == 0:
            (global_sigma_t, global_mean_tau, global_mean_delta,
             global_sigma_delta, global_eps_x, global_eps_y) = gather_checkpoint_stats(comm,
                                                                local_indices_arr, local_bunches)
            if rank == 0:
                sigma_t_s = float(np.mean(global_sigma_t))
                sigma_z_m = sigma_t_to_sigma_z(sigma_t_s, beta_rel)
                mean_tau_s = float(np.mean(global_mean_tau))

                history_turn.append(turn + 1)
                history_sigma_t_ps.append(sigma_t_s * 1e12)
                history_sigma_z_mm.append(sigma_z_m * 1e3)
                history_mean_tau_ps.append(mean_tau_s * 1e12)

                # Mean beam energy <E>
                mean_delta = float(np.mean(global_mean_delta))
                K_MeV = E0_EV * (1.0 + mean_delta) / 1e6
                history_K_MeV.append(K_MeV)

                # Cavity total voltage per cavity (kV).
                # cavity_phasor_record = generator_phasor_record + beam_phasor_record
                v_mc_per_cav_kV = float(np.mean(np.abs(mc.cavity_phasor_record))) / N_MC / 1e3

                if hc is not None:
                    v_hc_per_cav_kV = float(np.mean(np.abs(hc.cavity_phasor_record))) / N_HC / 1e3
                else:
                    v_hc_per_cav_kV = 0.0

                history_V_MC_kV.append(v_mc_per_cav_kV)
                history_V_HC_kV.append(v_hc_per_cav_kV)


                # Beam-induced voltage per cavity → V_beam_*_vs_pass
                vbeam_mc = float(np.mean(np.abs(mc.beam_phasor_record))) / N_MC / 1e3
                vbeam_hc = (float(np.mean(np.abs(hc.beam_phasor_record))) / N_HC / 1e3
                            if hc is not None else 0.0)
                history_Vbeam_MC_kV.append(vbeam_mc)
                history_Vbeam_HC_kV.append(vbeam_hc)

                # Emittances → ecx / ecy / el vs pass
                sd   = float(np.mean(global_sigma_delta))
                ex   = float(np.mean(global_eps_x))
                ey   = float(np.mean(global_eps_y))
                el_s = sigma_t_s * sd
                history_sigma_delta.append(sd)
                history_eps_x_m.append(ex)
                history_eps_y_m.append(ey)
                history_eps_l_s.append(el_s)

                history_sigma_t_first_ps.append( float(global_sigma_t[0]) * 1e12)
                history_sigma_t_middle_ps.append(float(global_sigma_t[H // 2]) * 1e12)
                history_sigma_t_last_ps.append(  float(global_sigma_t[H - 1]) * 1e12)

                # Active-cavity setpoint errors. If the simplified LLRF is working,
                # these should remain close to numerical zero.
                mc_target_record = mc_setpoint * mc_exp_neg_phase

                mc_setpoint_error = (np.mean(np.abs(mc.cavity_phasor_record - mc_target_record)) / mc.Vc)

                if hc is not None:
                    hc_target_record = hc_setpoint * hc_exp_neg_phase
                    hc_setpoint_error = (
                        np.mean(np.abs(hc.cavity_phasor_record - hc_target_record)) / hc.Vc
                    )
                else:
                    hc_setpoint_error = np.nan

                # Generator current per cavity (A).
                # |I_g| = |Vg_per_cav| / (RL_per_cav * |cos(psi)|) = |gen_phasor| / (mc.RL * |cos(psi)|) [Ncav cancels]
                ig_mc_A = (float(np.mean(np.abs(mc.generator_phasor_record))) /
                           (mc.RL * abs(np.cos(mc.psi))))

                if hc is not None:
                    ig_hc_A = (float(np.mean(np.abs(hc.generator_phasor_record))) /
                               (hc.RL * abs(np.cos(hc.psi))))
                else:
                    ig_hc_A = 0.0

                history_Ig_MC_A.append(ig_mc_A)
                history_Ig_HC_A.append(ig_hc_A)

                now = time.time()
                elapsed_total = now - t0
                elapsed_block = now - t_last
                avg_per_turn = elapsed_total / (turn + 1)

                err_hc_txt = f"{hc_setpoint_error:.2e}" if hc is not None else "n/a"

                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"Turn {turn + 1:6d}/{N_TURNS} | "
                    f"sigma_t = {sigma_t_s * 1e12:8.3f} ps | "
                    f"sigma_z = {sigma_z_m * 1e3:8.3f} mm | "
                    f"tau = {mean_tau_s * 1e12:8.3f} ps | "
                    # f"V_MC = {v_mc_per_cav_kV:8.3f} kV/cav | "
                    # f"V_HC = {v_hc_per_cav_kV:8.3f} kV/cav | " 
                    f"err_MC = {mc_setpoint_error:.2e} | "
                    f"err_HC = {err_hc_txt} | "
                    f"block = {elapsed_block:8.2f} s | "
                    f"total = {elapsed_total:8.2f} s | "
                    f"avg/turn = {avg_per_turn:8.4f} s"
                )
                t_last = now

    local_wall = time.time() - t0
    wall_clock_s = comm.allreduce(local_wall, op=MPI.MAX)

    if rank == 0:
        print("\n=== TRACKING FINISHED ===")
        print(f"Wall-clock time: {wall_clock_s:.2f} s")

    history = None
    if rank == 0:
        history = {
            "turn": np.array(history_turn, dtype=int),
            "sigma_z_mm": np.array(history_sigma_z_mm, dtype=float),
            "sigma_t_ps": np.array(history_sigma_t_ps, dtype=float),
            "mean_tau_ps": np.array(history_mean_tau_ps, dtype=float),
            "K_MeV": np.array(history_K_MeV, dtype=float),
            "V_MC_kV": np.array(history_V_MC_kV, dtype=float),
            "V_HC_kV": np.array(history_V_HC_kV, dtype=float),
            "Ig_MC_A": np.array(history_Ig_MC_A, dtype=float),
            "Ig_HC_A": np.array(history_Ig_HC_A, dtype=float),
            "Vbeam_MC_kV": np.array(history_Vbeam_MC_kV, dtype=float),
            "Vbeam_HC_kV": np.array(history_Vbeam_HC_kV, dtype=float),
            "sigma_delta": np.array(history_sigma_delta, dtype=float),
            "eps_x_m": np.array(history_eps_x_m, dtype=float),
            "eps_y_m": np.array(history_eps_y_m, dtype=float),
            "eps_l_s": np.array(history_eps_l_s, dtype=float),
            "sigma_t_first_ps":  np.array(history_sigma_t_first_ps, dtype=float),
            "sigma_t_middle_ps": np.array(history_sigma_t_middle_ps, dtype=float),
            "sigma_t_last_ps":   np.array(history_sigma_t_last_ps, dtype=float),
            "wall_clock_s": wall_clock_s,
        }

    return history


def collect_final_data(comm, local_indices_arr, local_bunches, mc, hc):
    """
    Collect the end-of-run snapshot used by make_plots and the final comparison
    in main(). Called once after the tracking loop. All ranks contribute, but
    only rank 0 returns data; all others return None.

    Gathers:
      * per-bunch sigma_t and mean_tau for the full train (via
        gather_checkpoint_stats).
      * normalised longitudinal profiles of the three representative bunches
        (first, middle, last); each rank computes its own and rank 0 merges them.
      * MC and HC phasor records, split into beam-induced, generator, and total
        (generator + beam). When hc is None, HC records are returned as zero
        arrays so downstream code can treat both cavities uniformly.
    """    
    
    rank = comm.rank
    rep_idx = representative_indices()

    # gather_checkpoint_stats returns (sigma_t, mean_tau, mean_delta,
    # sigma_delta, eps_x, eps_y). Only sigma_t and mean_tau are needed here
    # for the end-of-run snapshot; the rest are tracked turn-by-turn in the
    # history dict via the SAVE_EVERY checkpoint inside run_tracking.
    global_sigma_t, global_mean_tau, *_ = gather_checkpoint_stats(
        comm, local_indices_arr, local_bunches
    )

    local_rep = {}
    for label, bunch_idx in rep_idx.items():
        if bunch_idx in local_bunches:
            tau_s, prof = bunch_profile_tau(
                local_bunches[bunch_idx],
                n_bin=PROFILE_NBIN,
            )
            local_rep[label] = (tau_s, prof)

    gathered_rep = comm.gather(local_rep, root=0)

    if rank != 0:
        return None

    rep_profiles = {}
    for piece in gathered_rep:
        rep_profiles.update(piece)

    mc_total_phasor_record = total_cavity_phasor_record(mc)

    if hc is not None:
        hc_beam_phasor_record = hc.beam_phasor_record.copy()
        hc_generator_phasor_record = hc.generator_phasor_record.copy()
        hc_total_phasor_record = total_cavity_phasor_record(hc).copy()
    else:
        hc_beam_phasor_record = np.zeros(H, dtype=complex)
        hc_generator_phasor_record = np.zeros(H, dtype=complex)
        hc_total_phasor_record = np.zeros(H, dtype=complex)

    return {
        "sigma_t_each_s": global_sigma_t,
        "mean_tau_each_s": global_mean_tau,
        "rep_profiles": rep_profiles,

        # Beam-induced phasor only.
        "mc_beam_phasor_record": mc.beam_phasor_record.copy(),
        "hc_beam_phasor_record": hc_beam_phasor_record,

        # Generator phasor only.
        "mc_generator_phasor_record": mc.generator_phasor_record.copy(),
        "hc_generator_phasor_record": hc_generator_phasor_record,

        # Total cavity phasor = generator + beam.
        "mc_total_phasor_record": mc_total_phasor_record.copy(),
        "hc_total_phasor_record": hc_total_phasor_record,
    }