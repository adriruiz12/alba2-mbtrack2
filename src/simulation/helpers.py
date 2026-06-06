"""
helpers.py - Pure helper functions: unit conversions, relativistic factors,
RF detuning and a small statistics helper.

These functions have no MPI dependency and almost no direct mbtrack2 dependency.
For example, rf_detune_from_ring does not require a real mbtrack2 Synchrotron
object: it only needs an object with the attributes it uses, such as ring.f1.
This duck-typing style makes the functions easier to test in isolation.
"""

import numpy as np

from constants import C_LIGHT, E_CHARGE, M_E
from config import LAST_AVG_N


def relativistic_factors(E0_eV: float):
    """
    Computes the Lorentz factor gamma and relativistic beta from the beam energy
    in eV. Used to convert between spatial and temporal bunch length
    (sigma_z <-> sigma_t).
    """

    gamma = E0_eV / (M_E * C_LIGHT**2 / E_CHARGE)
    beta_rel = np.sqrt(1.0 - 1.0 / gamma**2)
    return gamma, beta_rel


# sigma_z (metres) and sigma_t (seconds) are two equivalent ways to express the
# longitudinal bunch size, related by z = beta * c * t (beta ≈ 1). Both conversions
# are needed because Elegant outputs sigma_z while mbtrack2 works internally with
# sigma_t.

def sigma_z_to_sigma_t(sigma_z_m: float, beta_rel: float):
    """
    Converts RMS bunch length from metres to seconds.
    """

    return sigma_z_m / (beta_rel * C_LIGHT)


def sigma_t_to_sigma_z(sigma_t_s: float, beta_rel: float):
    """
    Converts RMS bunch duration from seconds to metres.
    """

    return beta_rel * C_LIGHT * sigma_t_s


def rf_detune_from_ring(resonance_hz: float, harmonic: int, ring):
    """
    mbtrack2 convention:
        detune = resonance_frequency - harmonic * ring.f1
    Do not use Elegant drive_frequency here. Elegant's drive_frequency is kept only
    as a reference to compare inputs.
    The detune is how much the real cavity resonance frequency differs from
    the exact beam RF frequency.
    """

    return resonance_hz - harmonic * ring.f1


def safe_last_mean(values, n=LAST_AVG_N):
    """
    Returns the mean of the last n elements of an array. Used to estimate the
    converged steady-state value of sigma_t or sigma_z by averaging over the final
    LAST_AVG_N saved points, reducing the impact of residual coherent oscillations.
    Returns NaN if the array is empty.
    """

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    n_used = min(n, arr.size)

    return float(np.mean(arr[-n_used:]))


def analytical_equilibrium_profile(omega_rf, Vrf, V_hc_total, U0_eV, n,
                                   alpha_c, sigma_e, E0_eV, C, n_phi=20001):
    """
    Analytical equilibrium longitudinal profile for the active HC voltage,
    mirroring the 'Current profile' case of paramsalba.py (Exercise 5).
    Returns (t_ps, rho_t) with rho_t a time density normalized to
    int rho_t dt = 1 [1/s]; t = phi / omega_rf over one RF bucket.
    Reduces to the single-RF case when V_hc_total = 0.
    """
    phi = np.linspace(-np.pi, np.pi, n_phi)

    k = V_hc_total / Vrf
    x = U0_eV / Vrf
    # Flat-potential harmonic phase (sin convention), same as phi_s_hc.
    phi_h = np.arctan(-(n * x) / np.sqrt((n**2 - 1.0)**2 - (n**2 * x)**2))
    # Synchronous phase from energy balance at phi=0, stable branch.
    phi_s = np.pi - np.arcsin(np.clip(x - k * np.sin(phi_h), -1.0, 1.0))

    # Closed-form integral of (V_t(phi) - U0) from 0 to phi.
    integral_eV = (
        Vrf * (np.cos(phi_s) - np.cos(phi + phi_s))
        + (Vrf * k / n) * (np.cos(phi_h) - np.cos(n * phi + phi_h))
        - U0_eV * phi
    )
    pref = (C_LIGHT * alpha_c) / (E0_eV * C * omega_rf)
    Phi = -pref * integral_eV

    expo = -Phi / (alpha_c**2 * sigma_e**2)
    expo -= np.max(expo)  # numerical stability
    rho_phi = np.exp(expo)

    t = phi / omega_rf
    rho_t = rho_phi * omega_rf
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    rho_t /= trapz(rho_t, t)

    return t * 1e12, rho_t