"""
constants.py - Static physical, machine and RF constants for the ALBA II
longitudinal tracking simulation.

Everything in this module is fixed: it does NOT depend on the validation case
or on any run-time knob. Case-dependent values (HC voltage, HC detuning,
targets, output directory, run knobs) live in config.py instead.

PHYSICS NOTE on U0
------------------
The energy loss per turn is

    U0 = 0.9352696554914384 MeV + 0.1376275 MeV = 1.0728971554914384 MeV

The first term is the synchrotron-radiation loss obtained from the Elegant twiss
output (twiss_ALBA_II.twi), computed from the lattice radiation integrals. The
second term is NOT an ad-hoc correction: it is taken from the reference
paramsalba file, where the total energy loss is defined as
    U_0 = 0.9352696554914384e6 + 1.376275e5   [eV]
corresponding to "bending + IDs". Using the paramsalba value of U0 keeps the
longitudinal-equilibrium calculation consistent with the reference RF
calculations that define the target bunch lengths for each validation case.

RF CONVENTION NOTE
------------------
mbtrack2 uses the cosine convention for RF voltage; ELEGANT uses sine.
Conversion:  theta_cos = phi_sine - 90 deg

With the design phases:
    theta_MC = 149.934574 deg - 90 deg =  59.934574 deg
    theta_HC = 349.023576 deg - 90 deg = 259.023576 deg

CavityResonator expects the detuning as:
    detune = resonance_frequency - m * ring.f1
where m * ring.f1 is mbtrack2's equivalent of the ELEGANT drive frequency.
The ELEGANT drive frequencies are kept as reference inputs only: the value
passed to mbtrack2 is the difference (resonance - drive), not the drive itself.
"""

import numpy as np
import os


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

C_LIGHT = 299792458.0
E_CHARGE = 1.602176634e-19
M_E = 9.10938356e-31

# =============================================================================
# ALBA II MACHINE / BEAM PARAMETERS
# =============================================================================

E0_EV = 3.0e9
U0_EV = 0.9352696554914384e6 + 1.376275e5

ALPHA_C = 1.0433261800610089e-4
ALPHA_C2 = 8.620599833413995e-4
ALPHA_C3 = 6.630633197338763e-4
SIGMA_DELTA = 0.0012728536680170711
H = 448
CIRCUMFERENCE_M = 268.79999999999967

# Total stored beam current [A]. Nominal ALBA II operating point is 0.300 A.
# Overridable via TOTAL_CURRENT_A env var; lives here (not config.py) so that
# all modules that need it can import from a single place without circular deps.
TOTAL_CURRENT_A = float(os.environ.get("TOTAL_CURRENT_A", "0.300"))

SIGMA_Z0_M = 0.002462010883796751

BETA_X_M = 3.680811297008652
BETA_Y_M = 3.888776256698545
ALPHA_X = 0.0
ALPHA_Y = 0.0
DISPERSION_LOCAL = np.array([0.0, 0.0, 0.0, 0.0])

EMIT_X_MRAD = 1.671614243984046e-10
EMIT_Y_MRAD = 8.358071219920230e-11

# Radiation integrals from the elegant twiss file (twiss_ALBA_II.twi), the
# optics-calculation output that lists the lattice functions and the global
# synchrotron-radiation integrals I1-I5.
# I2 and I4 fix the damping partition numbers J_x, J_y, J_z (dimensionless
# factors that set how the radiation damping rate is distributed among the
# horizontal, vertical and longitudinal planes, with Jx + Jy + J_z = 4),
# and from them the radiation damping times below, which the mbtrack2
# SynchrotronRadiation element needs to model radiative damping.
# Extracted with: 
#   module load GSL/2.7-GCC-11.3.0
#   sdds2stream twiss_ALBA_II.twi -parameter=I1,I2,I3,I4,I5
I2   = 8.201088141795471e-01 # radiation integral I2 from elegant twiss output [m⁻¹]
I4   = -9.620305861296242e-01 # radiation integral I4 from elegant twiss output [m⁻¹]
T0   = CIRCUMFERENCE_M / C_LIGHT  # revolution period [s], derived from elegant lattice
J_x  = 1.0 - I4 / I2 # horizontal partition number = 2.17305
J_y  = 1.0 # vertical partition number (no x-y coupling)
J_z  = 2.0 + I4 / I2 # longitudinal partition number = 0.82695
fac  = 2.0 * E0_EV * T0 / U0_EV   # common damping-time prefactor ≈ 5.014e-3 s
TAU_X_S = fac / J_x # ≈ 2.307e-3 s (~2573 turns) (from elegant radiation integrals)
TAU_Y_S = fac / J_y # ≈ 5.014e-3 s (~5592 turns) (from elegant radiation integrals)
TAU_Z_S = fac / J_z # ≈ 6.064e-3 s (~6763 turns) (from elegant radiation integrals)


# =============================================================================
# RF SYSTEM PARAMETERS
# =============================================================================

N_MC = 6
V_MC_TOTAL_V = 2.4e6
BETA_COUPLER_MC = 3.5
RS_MC_PER_CAVITY_OHM = 3.3e6
Q0_MC = 29500.0
QL_MC = Q0_MC / (1.0 + BETA_COUPLER_MC)

# Frequency convention:
#   - drive_frequency is the frequency at which the external RF generator feeds
#     the cavity.
#   - resonance_frequency is the natural resonant frequency of the cavity mode, 
#     determined by the cavity geometry and tuning.
# In a cavity with beam loading, the resonant frequency is often not chosen to
# be exactly equal to the beam/generator RF frequency. The cavity can be detuned
# to compensate the phase and power effects introduced by the beam. In mbtrack2,
# CavityResonator does not take the Elegant drive_frequency directly as the
# detuning parameter. Instead, it uses:
#     detune = resonance_frequency - m * ring.f1
# Therefore, the Elegant drive frequencies are kept below as reference inputs,
# while the resonance frequencies are used to compute the detuning passed to
# mbtrack2.

FREQ_MC_RESONANCE_HZ = 4.996178107947031e8
FREQ_MC_DRIVE_HZ = 4.996540894183540e8  # Elegant drive_frequency, kept for reference.


PHI_MC_SINE_DEG = 149.9345744486530
THETA_MC_COS_RAD = np.deg2rad(PHI_MC_SINE_DEG - 90.0) # cosine convention

# As mbtrack2 CavityResonator expects detune = fr - m*ring.f1,  the actual 
# detuning must be computed after the Synchrotron object exists.

N_HC = 4
BETA_COUPLER_HC = 0.7
RS_HC_PER_CAVITY_OHM = 1.1e6
Q0_HC = 13000.0
QL_HC = Q0_HC / (1.0 + BETA_COUPLER_HC)

FREQ_HC_DRIVE_HZ = 1.498962268255062e9  # Elegant drive_frequency

m_hc = 3 # Third harmonic

# Dimensionless ratio between the energy loss per turn and the total main-cavity voltage.
x_hc = U0_EV / V_MC_TOTAL_V


# Auxiliary synchronous phase used only for the analytical HC detuning estimate.
# It comes from the flat-potential / harmonic-cavity RF conditions for a cavity
# of harmonic order m. This is not the phase directly passed to mbtrack2; the
# actual HC phase is defined below from PHI_HC_SINE_DEG.
phi_s_hc = np.arctan(-(m_hc * x_hc) /
                     np.sqrt((m_hc**2 - 1)**2 - (m_hc**2 * x_hc)**2)) 


PHI_HC_SINE_DEG = 349.0235762560818
THETA_HC_COS_RAD = np.deg2rad(PHI_HC_SINE_DEG - 90.0)
