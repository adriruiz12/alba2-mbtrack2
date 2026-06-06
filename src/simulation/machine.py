"""
machine.py - Builders for the mbtrack2 objects (ring, cavities, bunches) and
the low-level cavity-state operations.

OOP note: these functions create objects from mbtrack2 Classes and then use
their methods; no custom class hierarchy is introduced.
"""

import numpy as np

from mbtrack2.utilities import Optics
from mbtrack2.tracking import Electron, Bunch, Synchrotron, CavityResonator

from constants import (
    ALPHA_C,
    ALPHA_X,
    ALPHA_Y,
    BETA_X_M,
    BETA_Y_M,
    CIRCUMFERENCE_M,
    DISPERSION_LOCAL,
    E0_EV,
    EMIT_X_MRAD,
    EMIT_Y_MRAD,
    FREQ_MC_RESONANCE_HZ,
    H,
    N_HC,
    N_MC,
    Q0_HC,
    Q0_MC,
    QL_HC,
    QL_MC,
    RS_HC_PER_CAVITY_OHM,
    RS_MC_PER_CAVITY_OHM,
    SIGMA_DELTA,
    SIGMA_Z0_M,
    TAU_X_S,
    TAU_Y_S,
    TAU_Z_S,
    THETA_HC_COS_RAD,
    THETA_MC_COS_RAD,
    TOTAL_CURRENT_A,
    U0_EV,
    V_MC_TOTAL_V,
)
from config import (
    CAVITY_NBIN,
    FREQ_HC_RESONANCE_HZ,
    MCF_ORDER,
    MP_PER_BUNCH,
    SEED,
    TRACK_ALIVE,
    USE_HC,
    V_HC_TOTAL_V,
)
from helpers import relativistic_factors, rf_detune_from_ring, sigma_z_to_sigma_t


def total_cavity_phasor_record(cavity):
    """
    Total cavity phasor = generator phasor + beam-induced phasor.
    This is the quantity that should be compared with a total cavity voltage.
    The beam_phasor_record alone is only the beam-loading contribution.
    """

    return cavity.generator_phasor_record + cavity.beam_phasor_record


def reset_cavity_state(cavity):
    """
    Reset the dynamic state of a CavityResonator before tracking. Clears the
    beam-induced phasor, reinitialises per-bucket records, resets the turn
    counter, and restores bunch indexing arrays. Call this after
    BeamLoadingEquilibrium, which modifies internal phasor state.
    """

    cavity.beam_phasor = 0.0 + 0.0j
    cavity.beam_phasor_record = np.zeros(H, dtype=complex)
    cavity.generator_phasor_record = np.ones(H, dtype=complex) * cavity.generator_phasor
    cavity.tracking = True
    cavity.nturn = 0
    cavity.valid_bunch_index = np.arange(H, dtype=int)
    cavity.distance = np.ones(H, dtype=int)


def enforce_active_cavity_setpoint(cavity, setpoint, exp_neg_phase):
    """
    Enforce an active-cavity voltage setpoint bucket by bucket.
    Desired condition at tau = 0, i.e. when the beam arrives at each cavity:
        total cavity phasor = generator phasor + beam phasor = Vc * exp(i theta)
    Therefore:
        generator phasor = desired total phasor - beam phasor
    This is the simplified LLRF model needed to keep the HC at its active
    voltage setpoint. Without this, the HC effective voltage can drift away
    from the theoretical 680 kV total value. This function does not directly
    apply a kick to the bunch. Instead, it updatesthe generator phasor so
    that, when the beam enters the cavity, it sees the intended total cavity
    voltage, even in the presence of beam loading.

    mbtrack2's CavityResonator is not limited to passive cavities: it supports
    active cavities through set_generator() plus a set of LLRF/feedback classes
    (proportional and integral feedback, etc.) stored in cavity.feedback and
    applied inside CavityResonator.track(). The reason we do not use them here
    is that this script replaces CavityResonator.track() with its own custom
    many-bunches-per-rank distributed tracker (track_cavity_distributed in
    tracking.py). Because the built-in feedback objects are only driven from
    inside the native track() method, they are never advanced in this layout.
    enforce_active_cavity_setpoint is therefore a deliberately minimal,
    deterministic stand-in for that feedback: a direct bucket-by-bucket
    re-anchoring of the generator phasor that fits the custom tracking loop.
    """

    # [:] ensures in-place update of the existing array rather than rebinding the
    # variable to a new object, so mbtrack2's internal references stay valid.
    cavity.generator_phasor_record[:] = (
        setpoint * exp_neg_phase - cavity.beam_phasor_record
    )


def build_ring():
    """
    Constructs and returns the mbtrack2 Synchrotron object representing the
    ALBA II storage ring, using the lattice and beam parameters extracted
    from the Elegant twiss output.
    """

    _, beta_rel = relativistic_factors(E0_EV)
    sigma_0_s = sigma_z_to_sigma_t(SIGMA_Z0_M, beta_rel)

    particle = Electron()
    optics = Optics(
        local_beta=np.array([BETA_X_M, BETA_Y_M]),
        local_alpha=np.array([ALPHA_X, ALPHA_Y]),
        local_dispersion=DISPERSION_LOCAL,
    )

    ring = Synchrotron(
        H,
        optics,
        particle,
        tau=np.array([TAU_X_S, TAU_Y_S, TAU_Z_S]),
        sigma_delta=SIGMA_DELTA,
        sigma_0=sigma_0_s,
        emit=np.array([EMIT_X_MRAD, EMIT_Y_MRAD]),
        L=CIRCUMFERENCE_M,
        E0=E0_EV,

        # ac: first-order MCF for ring-level scalars (synchrotron tune, sigma_0).
        # Higher-order terms belong to mcf_order below, which drives the
        # turn-by-turn tau <- tau + mcf(delta)*L/c mapping.
        ac=ALPHA_C, 

        # mcf_order: full momentum compaction polynomial [alpha_c3, alpha_c2, alpha_c]
        # used by LongitudinalMap each turn for the actual tau <- tau + mcf(delta)*L/c
        # mapping. Higher-order terms are active here, unlike in ac above.
        mcf_order=MCF_ORDER,
        U0=U0_EV,
    )
    return ring


def build_cavities(ring):
    """
    Construct the main cavity (mc) and, if enabled, the harmonic cavity (hc)
    as mbtrack2 CavityResonator objects. mbtrack2 uses the detuning convention:
        detune = resonance_frequency - m * ring.f1
    where:
        - Resonance_frequency is the physical resonant frequency of the cavity.
        - m is the cavity harmonic with respect to the main RF frequency.
        - ring.f1 is the main RF frequency of the ring.
    In this script:
        - For the main cavity, resonance_frequency = FREQ_MC_RESONANCE_HZ.
        - For the harmonic cavity, resonance_frequency = FREQ_HC_RESONANCE_HZ.
    The Elegant drive frequencies are kept only as reference inputs. They are
    not passed directly as mbtrack2 detuning values.

    Validation cases:
        - no_hc          : returns (mc, None)
        - hc_100kv       : returns (mc, hc) with 100 kV per HC cavity
        - goal           : returns (mc, hc) with 170 kV per HC cavity
        - flat_potential : returns (mc, hc) with the flat-potential HC voltage
        - goal_elegant_freq : returns (mc, hc) with 170 kV per HC cavity;
                              FREQ_HC_RESONANCE_HZ is overridden to the Elegant
                              value instead of the analytical paramsalba detuning.
    """

    detune_mc_hz = rf_detune_from_ring(FREQ_MC_RESONANCE_HZ, 1, ring)

    mc = CavityResonator(
        ring=ring,
        m=1, # Fundamental harmonic
        Rs=RS_MC_PER_CAVITY_OHM,
        Q=Q0_MC,
        QL=QL_MC,
        detune=detune_mc_hz,
        Ncav=N_MC,
        Vc=V_MC_TOTAL_V,
        theta=THETA_MC_COS_RAD,
        n_bin=CAVITY_NBIN,
    )
    mc.detune_used_hz = detune_mc_hz
    mc.set_generator(TOTAL_CURRENT_A)

    if not USE_HC:
        return mc, None
    detune_hc_hz = rf_detune_from_ring(FREQ_HC_RESONANCE_HZ, 3, ring)

    hc = CavityResonator(
        ring=ring,
        m=3, # Third harmonic
        Rs=RS_HC_PER_CAVITY_OHM,
        Q=Q0_HC,
        QL=QL_HC,
        detune=detune_hc_hz,
        Ncav=N_HC,
        Vc=V_HC_TOTAL_V,
        theta=THETA_HC_COS_RAD,
        n_bin=CAVITY_NBIN,
    )

    # Store the actual detuning used, so the summary does not report the old
    # Elegant-drive-frequency detuning.
    hc.detune_used_hz = detune_hc_hz

    # Both cavities are active in the Elegant input: MC and HC both have
    # drive_frequency, v_setpoint, phase_setpoint and feedback records.
    # Therefore we keep both generators active in mbtrack2.
    hc.set_generator(TOTAL_CURRENT_A)

    return mc, hc


def build_local_bunches(ring, indices):
    """
    Creates and returns the dict of Bunch objects assigned to this MPI rank.
    Each bunch is initialised with a Gaussian distribution matching the
    Synchrotron equilibrium parameters, with a unique random seed per bunch
    to avoid identical macro-particle distributions across the train.
    """
    current_per_bunch = TOTAL_CURRENT_A / H
    local = {}
    for bunch_idx in indices:
        np.random.seed(SEED + bunch_idx)

        bunch = Bunch(
            ring,
            mp_number=MP_PER_BUNCH,
            current=current_per_bunch,
            track_alive=TRACK_ALIVE,
        )

        # Gaussian distribution from Synchrotron equilibrium parameters.
        bunch.init_gaussian()

        local[bunch_idx] = bunch
    return local