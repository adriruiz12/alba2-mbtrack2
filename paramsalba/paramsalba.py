'''
Longitudinal particle tracking simulations for ALBA II
Initial exercises: basic parameters of the ALBA II RF system

Author: Adrián Ruiz Doblas
Supervisor: Dr. Ignasi Bellafont (ALBA / CELLS)

Goal of these exercises:
  - Gain practical understanding of RF basics (main cavities + harmonic cavities).
  - Compute the key RF/beam parameters used as analytical reference for mbtrack2
    longitudinal tracking simulations.

Structure:
  - Exercise 2: steady-state phasor diagrams (MC and HC).
  - Exercise 3: power balance per cavity and totals.
  - Exercise 4: total RF voltage signal for three HC scenarios.
  - Exercise 5: RF potential well and equilibrium bunch profiles.
  - Exercise 6: bunch lengthening ratio u = sigma_z / sigma_z0 vs HC voltage.

Validation cases (set via VALIDATION_CASE):
  "no_hc"          V_HC = 0       u ~ 1.00   sigma_z ~ 2.46 mm  (natural)
  "hc_100kv"       V_HC = 100 kV  u ~ 1.51
  "goal"           V_HC = 170 kV  u ~ 3.91   sigma_z ~ 9.63 mm  (mbtrack2 target)
  "flat_potential" V_HC ~ 176 kV  u ~ 4.83   sigma_z ~ 11.9 mm

Notes:
  - MC and HC share the same RFCavity class; two instances (mc, hc) encapsulate
    the loaded parameters, detuning, phasors and power balance for each type.
  - Watch out for conventions in the literature.
  - The synchronous phase phi_s is measured from the zero crossing of the main RF
    signal; for a stable electron ring above transition it lies between 90° and 180°.
  - Run as: python paramsalba.py > paramsalba_<case>.txt  to save the output.
'''


# -----------------------------
# Imports
# -----------------------------

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.patches import Arc
from scipy.integrate import cumulative_trapezoid


SAVE_PLOTS = True
SHOW_PLOTS = False
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def savefig_pdf(fig, stem):
    if not SAVE_PLOTS:
        return
    out = FIG_DIR / f"{stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"[saved] {out}")


# VALIDATION_CASE = "no_hc"
# VALIDATION_CASE = "hc_100kv"
VALIDATION_CASE = "goal"
# VALIDATION_CASE = "flat_potential"

# DEPENDING ON THE EXECUTION:

# python paramsalba.py > paramsalba_no_hc.txt
# python paramsalba.py > paramsalba_hc_100kv.txt
# python paramsalba.py > paramsalba_goal.txt
# python paramsalba.py > paramsalba_flat_potential.txt

# Inputs:

V_t_mc = 2.4e6  # Total main voltage [V]
V_c_hc = 170e+03  # 170e3 nominal. Change to 'Optimal for flat potential' if desired to automatically achieve this condition.

N_mc = 6  # Number of main cavities
N_hc = 4  # Number of harmonic cavities
beta_mc = 3.5  # Coupling factor
beta_hc = 0.7  # Coupling factor
R_s_mc = 3.3e6  # Shunt impedance
R_s_hc = 1.1e6
Q_0_mc = 29500
Q_0_hc = 13000
E_0 = 3e9  # [eV] Nominal beam energy
U_0 = 0.9352696554914384e6 + 1.376275e5  # energy losses, in [eV], bending + IDs
I_sr = 0.3  # Storage ring current [A]
sigma_e = 0.0012728536680170711  # Energy spread
alpha_c = 1.0433261800610089E-4  # Momentum compaction factor
h_n = 448  # Harmonic number of the storage ring, the number of RF buckets it has
length_sr = 268.79999999999967  # [m], given by Elegant

m = 3  # Harmonic order of the harmonic cavities


# -----------------------------
# Physical constants
# -----------------------------
e = 1.602176634e-19
c = 299792458.0
m_e = 9.10938356e-31


# -----------------------------
# EXERCISE 2
# -----------------------------


# -----------------------------
# Helper functions
# -----------------------------

def loaded_params(R_s, Q_0, beta):
    """Loaded parameters (per cavity)."""

    Q_L = Q_0 / (1.0 + beta)
    R_L = R_s / (1.0 + beta)
    return R_L, Q_L


def flat_potential_phi_h(U_loss, V_rf, n):
    """Flat-potential harmonic synchronous phase phi_h (in the *sin* convention).

    Uses the closed-form expression from the double-RF flat-potential condition.
    """

    x = U_loss / V_rf  # e cancels in eV/V
    denom = np.sqrt((n**2 - 1.0)**2 - (n**2 * x)**2)
    return np.arctan(-(n * x) / denom)


def flat_potential_k(U_loss, V_rf, n):
    """Flat-potential voltage ratio k = |V_h|/|V_rf|."""

    x = U_loss / V_rf
    return np.sqrt(1.0 / n**2 - (x**2) / (n**2 - 1.0))


def psi_d_optimal(I_sr, R_s, V_c_mag, beta, phi_s):
    """Detuning angle psi_d (rad) from the 'optimal' (matched) condition."""

    return np.arctan((2.0 * I_sr * R_s / (abs(V_c_mag) * (1.0 + beta))) * np.cos(phi_s))


def Z_from_psi_d(R_L, psi_d):
    """Loaded cavity impedance expressed via detuning angle psi_d.

    We use the convention where:
      |Z| = R_L * cos(psi_d)
      arg(Z) = psi_d
    which is equivalent to:  Z = R_L / (1 - i tan(psi_d)).
    """

    return R_L / (1.0 - 1j * np.tan(psi_d))


def arrow(ax, z, label, color=None, label_offset_pts=0):
    """Arrow from origin to z, with label offset radially from the tip."""

    ax.annotate(
        "",
        xy=(z.real, z.imag),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", lw=2.0, color=color, shrinkA=0, shrinkB=0), zorder=3)

    angle = np.angle(z) if abs(z) > 1e-30 else 0.0
    dx = label_offset_pts * np.cos(angle)
    dy = label_offset_pts * np.sin(angle)
    ha = "left"  if np.cos(angle) >= 0 else "right"
    va = "bottom" if np.sin(angle) >= 0 else "top"

    ax.annotate(
        label,
        xy=(z.real, z.imag),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha, va=va,
        fontsize=11,
        color=color,
        zorder=4,
    )

def unit(z):
    """Unit phasor with the angle of z."""

    ang = np.angle(z) if abs(z) > 1e-30 else 0.0
    return np.exp(1j * ang)


def draw_angle_arc(ax, theta1_deg, theta2_deg, radius, label, color="0.25",
                   lw=1.4, text_scale=1.10, text_offset_pts=(0, 0)):
    """
    Draw the shortest arc between theta1 and theta2 (degrees), centered at origin.
    """

    a1 = theta1_deg % 360.0
    a2 = theta2_deg % 360.0
    d = (a2 - a1 + 180.0) % 360.0 - 180.0

    if d >= 0:
        t1 = a1
        t2 = a1 + d
    else:
        t1 = a1 + d
        t2 = a1

    arc = Arc((0.0, 0.0), 2.0 * radius, 2.0 * radius, angle=0.0, theta1=t1,
              theta2=t2, color=color, lw=lw, zorder=2)
    ax.add_patch(arc)

    amid = np.radians(a1 + 0.5 * d)
    rt = radius * text_scale
    x = rt * np.cos(amid)
    y = rt * np.sin(amid)

    ax.annotate(label, xy=(x, y), xytext=text_offset_pts, textcoords="offset points",
                color=color, fontsize=11, ha="center", va="center", zorder=4)

def format_axes_power(ax, power=5):
    """
    Scale both axes by 10**power and place ×10^power at the end of each drawn axis.

    Important: call this after setting xlim and ylim.
    """

    scale = 10 ** power

    # Scale tick labels
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/scale:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y/scale:g}"))

    # Hide automatic offset text
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)

    # Remove previous labels if the function is called again
    if hasattr(ax, "_power_texts"):
        for txt in ax._power_texts:
            txt.remove()

    txt_x = ax.text(
        1.01, 0.0, rf"$\times 10^{{{power}}}$",
        transform=ax.get_yaxis_transform(),   # x in axes coords, y in data coords
        ha="left", va="center", fontsize=10
    )

    txt_y = ax.text(
        0.0, 1.01, rf"$\times 10^{{{power}}}$",
        transform=ax.get_xaxis_transform(),   # x in data coords, y in axes coords
        ha="center", va="bottom", fontsize=10
    )

    ax._power_texts = [txt_x, txt_y]


def phasor_plot(I_b, I_g, V_b, V_g, V_c, info_text=None, currents_text=None,
                psi_d=None, phi_s=None, draw_phi_s=False, force_limits=None,
                ig_label_offset_pts=(-10, 10)):
    """Steady-state cavity phasor diagram (Exercise 2).

    Draws voltage phasors V_b, V_g, V_c = V_g + V_b to scale on a centred
    equal-aspect axes, plus current phasors I_b and I_g rescaled to a fixed
    fraction of the plot radius (direction only).  Optionally overlays angle
    arcs for the detuning angle psi_d and the synchronous phase phi_s.

    Parameters
    ----------
    title : str
        Figure title.
    I_b : complex
        Beam current phasor [A].
    I_g : complex
        Generator current phasor [A].
    V_b : complex
        Beam-loading voltage phasor [V].
    V_g : complex
        Generator voltage phasor [V].
    V_c : complex
        Cavity voltage phasor [V]; should satisfy V_c = V_g + V_b.
    info_text : str or None
        Optional physics-quantities text box placed in the top-left corner.
    currents_text : str or None
        Optional current-values text box placed in the bottom-right corner.
    psi_d : float or None
        Detuning angle [rad].  If given, an arc is drawn between -I_b and V_b.
    phi_s : float or None
        Synchronous phase [rad].  Used only when draw_phi_s is True.
    draw_phi_s : bool
        If True and phi_s is not None, draw the phi_s arc between -Im axis
        and V_c.
    force_limits : tuple(float, float) or None
        Override automatic axis half-widths as (Lx, Ly) [V].
    ig_label_offset_pts : tuple(float, float)
        Point offset (dx, dy) for the I_g label annotation.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """

    fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)

    ax.set_aspect("equal", adjustable="box")

    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    frame = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
        fill=False, edgecolor="black", linewidth=2.0, zorder=1)
    ax.add_patch(frame)

    scale = 1e5

    # Major ticks every 0.5 × 10^5, minor ticks every 0.1 × 10^5
    ax.xaxis.set_major_locator(MultipleLocator(0.5 * scale))
    ax.yaxis.set_major_locator(MultipleLocator(0.5 * scale))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1 * scale))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1 * scale))

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    ax.tick_params(axis="both", which="major", direction="in", top=False,
        right=False, labelsize=10, length=6, width=1.0, colors="black")

    ax.tick_params(axis="both", which="minor", direction="in", top=False,
        right=False, length=3, width=0.8, colors="black")

    # ---- limits first
    if force_limits is not None:
        Lx, Ly = force_limits
    else:
        allz = np.array([V_b, V_g, V_c], dtype=complex)
        mx = np.max(np.abs(np.concatenate([allz.real, allz.imag]))) + 1e-9
        pad = 0.25
        Lx = Ly = (1.0 + pad) * mx

    ax.set_xlim(-Lx, Lx)
    ax.set_ylim(-Ly, Ly)

    ax.set_xlabel("")
    ax.set_ylabel("")

    # Re [V]: anchored to the positive end of the horizontal axis
    ax.annotate("Re [V]", xy=(ax.get_xlim()[1], 0.0), xytext=(-8, -12), textcoords="offset points",
        ha="right", va="top", fontsize=11, color="black", zorder=5)

    # Im [V]: anchored to the positive end of the vertical axis
    ax.annotate("Im [V]", xy=(0.0, ax.get_ylim()[1]), xytext=(-12, -6), textcoords="offset points",
        ha="right", va="top", rotation=90, fontsize=11, color="black", zorder=5)

    # ---- visible current phasors (scaled, only to show direction)
    I_vis_len = 0.18 * min(Lx, Ly)
    I_b_vis = I_vis_len * unit(I_b)
    I_g_vis = I_vis_len * unit(I_g)

    arrow(ax, I_b_vis, r"$\tilde I_b$ (beam, 0°)", color="#3A9E6F", label_offset_pts=4)

    # Arrow for I_g
    ax.annotate("", xy=(I_g_vis.real, I_g_vis.imag), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", lw=2.0, color="#3A9E6F", shrinkA=0, shrinkB=0),
        zorder=3,
    )

    # Label for I_g
    ax.annotate(r"$\tilde I_g$", xy=(I_g_vis.real, I_g_vis.imag), xytext=ig_label_offset_pts,
        textcoords="offset points", ha="right", va="bottom", fontsize=11, color="#3A9E6F",
        zorder=4,
    )

    # ---- voltages
    arrow(ax, V_b, r"$\tilde V_b$", color="#2A5FA5", label_offset_pts=4)
    arrow(ax, V_g, r"$\tilde V_g$", color="#E86B3E", label_offset_pts=4)
    arrow(ax, V_c, r"$\tilde V_c=\tilde V_g+\tilde V_b$", color="k", label_offset_pts=4)

    # Vector-sum construction: exact translated copy of V_b starting at the tip of V_g
    z0 = V_g
    z1 = V_g + V_b   # natural triangle closure; should coincide with V_c

    # Exact dashed line between both points
    ax.plot([z0.real, z1.real], [z0.imag, z1.imag], ls="--", lw=1.5,
        color="#2A5FA5", alpha=0.7, zorder=2)

    # Single arrowhead at the end
    head_frac = 0.10
    z_head = z1 - head_frac * (z1 - z0)

    ax.annotate("", xy=(z1.real, z1.imag), xytext=(z_head.real, z_head.imag),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#2A5FA5",
                                alpha=0.7, shrinkA=0, shrinkB=0),
                zorder=3,
    )

    # ---- angle arcs
    # psi_d: between -I_b direction and V_b direction (this matches the reference plots)
    if psi_d is not None:
        th_minus_ib = np.degrees(np.angle(-I_b))
        th_vb = np.degrees(np.angle(V_b))
        draw_angle_arc(ax, th_minus_ib, th_vb, radius=0.12 * min(Lx, Ly),
            label=r"$\psi_d$", text_offset_pts=(-6, -2))

    # phi_s: between -Im axis (-90°) and V_c phasor angle
    # because phi_c = phi_s - 90°
    if draw_phi_s and (phi_s is not None):
        th_minus_im = -90.0
        th_vc = np.degrees(np.angle(V_c))
        draw_angle_arc(ax, th_minus_im, th_vc, radius=0.18 * min(Lx, Ly),
            label=r"$\phi_s$", text_offset_pts=(0, -2),
        )

    if info_text:
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, va="top",
            ha="left", fontsize=10,bbox=dict(boxstyle="round", facecolor="white",
                                             alpha=0.85, lw=0.5),
        )

    if currents_text:
        ax.text(0.98, 0.06, currents_text, transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, lw=0.5),
        )

    format_axes_power(ax, power=5)
    return fig, ax


def deg(x):
    return float(np.degrees(x))

def pol(z):
    return abs(z), float(np.degrees(np.angle(z)))

def beta_optimal(V_c, I_sr, phi_s, R_s):
    """
    Optimal coupling from the usual steady-state relation:
        beta_opt = 1 + P_b / P_c
    with
        P_b = |V_c| I_sr sin(phi_s)
        P_c = |V_c|^2 / (2 R_s)
    """

    P_b = abs(V_c) * I_sr * np.sin(phi_s)
    P_c = abs(V_c) ** 2 / (2.0 * R_s)
    return 1.0 + P_b / P_c


# -----------------------------
# RF cavity class (MC / HC)
# -----------------------------

class RFCavity:
    """Steady-state RF cavity model (Exercises 2 and 3).
 
    One instance is created per cavity type (MC / HC). It groups the
    loaded parameters, detuning, steady-state phasors and power balance
    so the formulas are written once instead of being duplicated.
 
    Parameters
    ----------
    name     : human-readable label, e.g. "Main Cavity (MC)".
    N        : number of cavities of this type in the ring.
    R_s      : shunt impedance [Ohm] (storage-ring convention).
    Q_0      : unloaded quality factor.
    beta     : input coupling factor.
    V_c_cav  : per-cavity accelerating voltage magnitude [V].
    phi_s    : synchronous phase [rad] in the sin convention.
    I_sr     : average storage-ring current [A].
    I_b      : beam current phasor (RF Fourier component).
    """
 
    def __init__(self, name, N, R_s, Q_0, beta, V_c_cav, phi_s, I_sr, I_b):
        self.name = name
        self.N = N
        self.R_s = R_s
        self.Q_0 = Q_0
        self.beta = beta
        self.V_c_cav = V_c_cav
        self.phi_s = phi_s
        self.I_sr = I_sr
        self.I_b = I_b
 
        # Loaded parameters (per cavity)
        self.R_L, self.Q_L = loaded_params(R_s, Q_0, beta)
 
        # Detuning angle and loaded impedance
        self.psi_d = psi_d_optimal(I_sr, R_s, V_c_cav, beta, phi_s)
        self.Z_L = Z_from_psi_d(self.R_L, self.psi_d)
 
        # Cavity voltage phasor (cos convention: angle = phi_s - 90 deg)
        self.V_c = V_c_cav * np.exp(1j * (phi_s - np.pi / 2.0))
 
        # Generator / beam-loading phasors (V_c = V_g + V_b)
        self.V_b = self.Z_L * I_b
        self.V_g = self.V_c - self.V_b
        self.I_g = self.V_g / self.Z_L
 
        # Optimal coupling for zero reflected power
        self.beta_opt = beta_optimal(self.V_c, I_sr, phi_s, R_s)
 
    def power_balance(self):
        """Steady-state power balance dict (P_g, P_b, P_c, P_r) per cavity."""
        return power_balance(self.V_c, self.I_g, self.I_sr,
                             self.phi_s, self.R_s, self.beta)
 
 
# -----------------------------
# Frequencies from ring length
# (computed once here; reused by Exercises 5 and 6)
# -----------------------------

# Relativistic factors
gamma = E_0 / (m_e * c**2 / e)
beta_rel = np.sqrt(1.0 - 1.0 / gamma**2)

f_rev = c * beta_rel / length_sr
f_rf_mc = h_n * f_rev
f_rf_hc = m * f_rf_mc
omega_rf_mc = 2.0 * np.pi * f_rf_mc

# ----------------------------------------------
# Beam current phasor (beam is the 0° reference)
# ----------------------------------------------
# Short-bunch approximation: RF Fourier component is 2 * I_sr
I_b = - (2.0 * I_sr) + 0j


# -----------------------------
# Per-cavity voltages
# -----------------------------
V_c_mc_mag = V_t_mc / N_mc


if VALIDATION_CASE == "no_hc":
    USE_HC = False
    V_c_hc = 0.0

elif VALIDATION_CASE == "hc_100kv":
    USE_HC = True
    V_c_hc = 100e3  # per HC cavity

elif VALIDATION_CASE == "goal":
    USE_HC = True
    V_c_hc = 170e3 # per HC cavity

elif VALIDATION_CASE == "flat_potential":
    USE_HC = True
    k_fp_case = flat_potential_k(U_0, V_t_mc, n=m)
    V_c_hc = k_fp_case * V_t_mc / N_hc  # per HC cavity

else:
    raise ValueError(f"Unknown validation case: {VALIDATION_CASE}")


V_t_hc_total = N_hc * V_c_hc


# -----------------------------
# Phases in the *sin* convention
# -----------------------------
# Harmonic synchronous phase (flat-potential formula)
phi_s_hc = flat_potential_phi_h(U_0, V_t_mc, n=m)

# Energy loss in harmonic cavities (positive number): U_hc = - V_hc * sin(phi_s_hc)
U_hc = -V_t_hc_total * np.sin(phi_s_hc)

# Total energy loss per turn that MC must compensate
U_t = U_0 + U_hc

# Main cavity synchronous phase (electrons above transition => between 90° and 180°)
phi_s_mc = np.pi - np.arcsin(U_t / V_t_mc)


# -----------------------------
# Cavity objects (Exercises 2 & 3)
# -----------------------------
# The RFCavity class holds the detuning / impedance / phasor / power
# chain. Instantiating it once per cavity type keeps the MC and HC
# formulas in a single place (V_c phasor: cos convention, angle phi_s-90).
mc = RFCavity("Main Cavity (MC)", N_mc, R_s_mc, Q_0_mc, beta_mc,
              V_c_mc_mag, phi_s_mc, I_sr, I_b)

hc = None
if USE_HC:
    hc = RFCavity("Harmonic Cavity (HC)", N_hc, R_s_hc, Q_0_hc, beta_hc,
                  V_c_hc, phi_s_hc, I_sr, I_b)


# -----------------------------
# Phasor diagrams (Exercise 2)
# -----------------------------
psi_gr_mc = deg(np.angle(mc.I_g))
psi_r = 0.0  # beam current reference

info_mc = (
    f"$V_c$ = {abs(mc.V_c)/1e3:.1f} kV\n"
    f"$V_g$ = {abs(mc.V_g)/1e3:.1f} kV\n"
    f"$V_b$ = {abs(mc.V_b)/1e3:.1f} kV\n\n"
    f"$\\psi_d$ = {deg(mc.psi_d):.1f}°\n"
    f"$\\psi_{{gr}}$ = {psi_gr_mc:.1f}°\n"
    f"$\\psi_r$ = {psi_r:.0f}°\n\n"
    f"$V_t$ = {V_t_mc/1e6:.1f} MV\n"
    f"$N_{{mc}}$ = {mc.N}\n"
    f"$\\beta$ = {mc.beta:.1f}\n"
    f"$\\beta_{{opt}}$ = {mc.beta_opt:.2f}\n\n"
    f"$U_t$ = {U_t/1e6:.3f} MeV\n"
    f"$\\phi_s$ = {deg(mc.phi_s):.2f}°"
)

currents_mc = (
    f"$I_{{sr}}$ = {I_sr:.1f} A\n"
    f"$I_b$ = {abs(I_b):.1f} A\n"
    f"$I_g$ = {abs(mc.I_g):.3f} A"
)

fig_ph_mc, _ = phasor_plot(I_b=I_b, I_g=mc.I_g, V_b=mc.V_b, V_g=mc.V_g, V_c=mc.V_c, info_text=info_mc,
    currents_text=currents_mc, psi_d=mc.psi_d, phi_s=mc.phi_s, draw_phi_s=False,
    ig_label_offset_pts=(-4, 6),
)
savefig_pdf(fig_ph_mc, "mc_phasor")

if USE_HC:
    info_hc = (
        f"$V_c$ = {abs(hc.V_c)/1e3:.0f} kV\n"
        f"$V_g$ = {abs(hc.V_g)/1e3:.2f} kV\n"
        f"$V_b$ = {abs(hc.V_b)/1e3:.2f} kV\n\n"
        f"$\\psi_d$ = {deg(hc.psi_d):.3f}°\n"
        f"$\\psi_r$ = {psi_r:.0f}°\n"
        f"$\\phi_{{s,hc}}$ = {deg(hc.phi_s):.2f}°\n\n"
        f"$V_t$ = {V_t_mc/1e6:.1f} MV\n"
        f"$V_{{t,hc}}$ = {V_t_hc_total/1e3:.0f} kV\n"
        f"$\\beta_{{hc}}$ = {hc.beta:.1f}\n"
        f"$\\beta_{{hc,opt}}$ = {hc.beta_opt:.2f}\n\n"
        f"$U_t$ = {U_t/1e6:.3f} MeV\n"
        f"$\\phi_{{s,hc,opt}}$ = {deg(hc.phi_s):.2f}°"
    )
 
    currents_hc = (
        f"$I_{{sr}}$ = {I_sr:.1f} A\n"
        f"$I_b$ = {abs(I_b):.1f} A\n"
        f"$I_g$ = {abs(hc.I_g):.3f} A"
    )

    fig_ph_hc, _ = phasor_plot(I_b=I_b, I_g=hc.I_g, V_b=hc.V_b, V_g=hc.V_g, V_c=hc.V_c, info_text=info_hc,
        currents_text=currents_hc, psi_d=hc.psi_d, phi_s=hc.phi_s, draw_phi_s=True,
        ig_label_offset_pts=(22, -2),
    )
    savefig_pdf(fig_ph_hc, "hc_phasor")

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")


def power_balance(V_c, I_g, I_sr, phi_s, R_s, beta):
    """Per-cavity steady-state power balance: P_g, P_b, P_c, P_r [W]."""

    Pg = R_s * (abs(I_g) ** 2) / (8.0 * beta)
    Pb = abs(V_c) * I_sr * np.sin(phi_s)
    Pc = (abs(V_c) ** 2) / (2.0 * R_s)
    Pr = Pg - Pb - Pc
    return {"P_g": Pg, "P_b": Pb, "P_c": Pc, "P_r": Pr}


def plot_power_balance_mc_pie(pb):
    """Pie chart of MC power balance: P_g = P_b + P_c + P_r."""

    # (normally Pb,Pc,Pr >= 0 in MC case)
    Pb, Pc, Pr, Pg = pb["P_b"], pb["P_c"], pb["P_r"], pb["P_g"]

    # Use positive parts for pie (if something comes negative,
    # we still plot magnitudes and warn in text)
    vals = [max(Pb, 0.0), max(Pc, 0.0), max(Pr, 0.0)]
    labels = [f"$P_b$ = {Pb/1e3:.2f} kW", f"$P_c$ = {Pc/1e3:.2f} kW", f"$P_r$ = {Pr/1e3:.2f} kW"]
    pie_colors = ["#2A5FA5", "#7A8C99", "#3A9E6F"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90, colors=pie_colors)
    ax.text(
        0.2, -0.12,
        f"$P_g = P_b + P_c + P_r$ = {Pg/1e3:.2f} kW",
        transform=ax.transAxes
    )
    plt.tight_layout()
    return fig, ax


def plot_power_balance_hc_bars(pb):
    """Stacked bar chart of HC power balance.

    Uses bars instead of pie because P_b < 0 in the HC case (beam drives
    the cavity).
    """

    # Stacked bars:
    #   left bar : Pg - Pb  (if Pb is negative -> Pg + |Pb|)
    #   right bar: Pc + Pr
    Pb, Pc, Pr, Pg = pb["P_b"], pb["P_c"], pb["P_r"], pb["P_g"]

    # Build LHS components to stack positively
    # If Pb < 0, then Pg - Pb = Pg + |Pb| (stack Pg and -Pb)
    comp_left_1 = max(Pg, 0.0)
    comp_left_2 = (-Pb) if Pb < 0 else 0.0

    # RHS components (Pc and Pr) as stacked positives
    comp_right_1 = max(Pc, 0.0)
    comp_right_2 = max(Pr, 0.0)

    C_Pg  = "#E86B3E"
    C_Pb  = "#2A5FA5"
    C_Pc  = "#7A8C99"
    C_Pr  = "#3A9E6F"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_ylabel("Power [kW]")

    x0, x1 = 0, 1
    ax.bar(x0, comp_left_1/1e3, label="$P_g$", color=C_Pg)
    if comp_left_2 > 0:
        ax.bar(x0, comp_left_2/1e3, bottom=comp_left_1/1e3, label="$-P_b$ (beam→cavity)", color=C_Pb)

    ax.bar(x1, comp_right_1/1e3, label="$P_c$", color=C_Pc)
    if comp_right_2 > 0:
        ax.bar(x1, comp_right_2/1e3, bottom=comp_right_1/1e3, label="$P_r$", color=C_Pr)

    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["$P_g - P_b$", "$P_c + P_r$"])
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    # Numeric annotation
    txt = (
        f"$P_g$ = {Pg/1e3:.2f} kW\n"
        f"$P_b$ = {Pb/1e3:.2f} kW\n"
        f"$P_c$ = {Pc/1e3:.2f} kW\n"
        f"$P_r$ = {Pr/1e3:.2f} kW"
    )
    ax.text(1.05, 0.5, txt, transform=ax.transAxes, va="center")

    plt.tight_layout()
    return fig, ax


# -----------------------------
# Exercise 3: Power balance plots (MC and HC)
# -----------------------------

def print_power_balance(name, pb, N_cav):
    print(f"\n--- Power balance: {name} ---")
    print("Per cavity:")
    print(f"  P_g     = {pb['P_g']/1e3:.3f} kW")
    print(f"  P_b     = {pb['P_b']/1e3:.3f} kW")
    print(f"  P_c     = {pb['P_c']/1e3:.3f} kW")
    print(f"  P_r     = {pb['P_r']/1e3:.3f} kW")
    print(f"  P_b/P_c = {pb['P_b']/pb['P_c']:.4f}")

    print(f"Total ({N_cav} cavities):")
    print(f"  P_g_tot = {N_cav * pb['P_g']/1e3:.3f} kW")
    print(f"  P_b_tot = {N_cav * pb['P_b']/1e3:.3f} kW")
    print(f"  P_c_tot = {N_cav * pb['P_c']/1e3:.3f} kW")
    print(f"  P_r_tot = {N_cav * pb['P_r']/1e3:.3f} kW")

# -----------------------------
# MC - pie chart
# -----------------------------
pb_mc3 = mc.power_balance()

print_power_balance(mc.name, pb_mc3, mc.N)

fig_pb_mc, _ = plot_power_balance_mc_pie(pb_mc3)
savefig_pdf(fig_pb_mc, "mc_power_balance")


# -----------------------------
# HC - stacked bar chart
# -----------------------------
pb_hc3 = None
fig_pb_hc = None

if USE_HC:
    pb_hc3 = hc.power_balance()

    print_power_balance(hc.name, pb_hc3, hc.N)

    fig_pb_hc, _ = plot_power_balance_hc_bars(pb_hc3)

    savefig_pdf(fig_pb_hc, "hc_power_balance")


# -----------------------------
# Show / close
# -----------------------------
if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")

# -----------------------------
# EXERCISE 4
# -----------------------------
# Total RF voltage signals (main + harmonic) and "flat potential" comparison
#
# We follow the double-RF voltage model:
#   V_t(φ) = V_rf [ sin(φ + φ_s) + k sin(n φ + φ_h) ]
# where:
#   - φ is the phase offset from the synchronous particle (bunch center at φ=0),
#   - V_rf is the *total* main RF voltage (here V_t_mc),
#   - k = |V_h|/|V_rf| is the harmonic-to-main voltage ratio,
#   - n is the harmonic number (here n = m = 3),
#   - φ_s is the synchronous phase w.r.t. the main RF,
#   - φ_h is the harmonic phase at the bunch center.
#
# The "flat potential" (cancellation) condition imposes V_t'(0)=V_t''(0)=0,
# yielding closed forms for k_fp and φ_h_fp (see Byrd & Georgsson, PRST-AB 4, 030701).
# In Exercise 4 we compare:
#   (i)  No HC  (k=0)
#   (ii) User HC voltage (k set by your V_c_hc input), with φ_h fixed to the flat-potential value
#   (iii) Optimal flat potential (k_fp, φ_h_fp, and the corresponding φ_s_fp)

def _phi_s_single_rf(U0, Vrf):
    """Synchronous phase for a *single* RF system (electrons above transition)."""

    x = U0 / Vrf
    if abs(x) > 1.0:
        raise ValueError(f"U0/Vrf = {x:.4f} is not in [-1,1]. Increase Vrf or check units.")
    return np.pi - np.arcsin(x)

def _phi_s_double_rf(U0, Vrf, k, phi_h):
    """Synchronous phase for a *double* RF system from energy balance at φ=0:

         sin(phi_s) + k sin(phi_h) = U0/Vrf

       (electrons above transition -> choose pi - arcsin(.)).
    """

    rhs = (U0 / Vrf) - k * np.sin(phi_h)
    if abs(rhs) > 1.0:
        raise ValueError(
            f"sin(phi_s) = {rhs:.4f} is not in [-1,1]. "
            "Your (U0, Vrf, k, phi_h) combination is inconsistent."
        )
    return np.pi - np.arcsin(rhs)

def _phi_s_flat_potential(U0, Vrf, n):
    """Closed-form synchronous phase under flat-potential (cancellation) condition."""

    x = U0 / Vrf
    rhs = (n**2 / (n**2 - 1.0)) * x
    if abs(rhs) > 1.0:
        raise ValueError(f"Flat-potential arcsin argument {rhs:.4f} not in [-1,1].")
    return np.pi - np.arcsin(rhs)

def voltage_signals(phi, Vrf, k, n, phi_s, phi_h):
    """Return V_mc(phi), V_hc(phi), V_tot(phi) for the double RF model."""

    V_mc = Vrf * np.sin(phi + phi_s)
    V_hc = Vrf * k * np.sin(n * phi + phi_h)
    return V_mc, V_hc, V_mc + V_hc

def _dV_dphi_at0(Vrf, k, n, phi_s, phi_h):
    return Vrf * np.cos(phi_s) + Vrf * k * n * np.cos(phi_h)

def _d2V_dphi2_at0(Vrf, k, n, phi_s, phi_h):
    return -Vrf * np.sin(phi_s) - Vrf * k * (n**2) * np.sin(phi_h)

def plot_voltage_signals_cases(Vrf, U0, n, k_user, phi_h_fp, k_fp, phi_s0,
                               phi_s_user, phi_s_fp):
    """Three-panel figure of V_mc, V_hc, V_tot vs φ for the three Exercise 4
    cases: no HC, user voltage, and flat-potential optimum."""    

    phi = np.linspace(-np.pi, np.pi, 2000)

    cases = [
        ("(i) No HC", 0.0, 0.0, phi_s0),
        ("(ii) User $V_hc$", k_user, phi_h_fp, phi_s_user),
        ("(iii) Flat potential (optimal)", k_fp, phi_h_fp, phi_s_fp),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.5), sharex=True,
                             sharey=True, constrained_layout=True)
    for ax, (title, k, phi_h, phi_s) in zip(axes, cases):
        V_mc, V_hc, V_tot = voltage_signals(phi, Vrf, k, n, phi_s, phi_h)

        # Plot in kV to keep axes clean (similar to the reference plot)
        ax.plot(phi, V_mc / 1e3, label="Main cavity voltage", color="#2A5FA5")
        ax.plot(phi, V_hc / 1e3, label="Harmonic cavity voltage", color="#E86B3E")
        ax.plot(phi, V_tot / 1e3, label="Total voltage", lw=2.0, color="#3A9E6F")

        # Reference lines: bunch center (φ=0) and U0 (as voltage)
        # Vertical axis: φ = 0
        ax.axvline(0.0, lw=1.0, ls="--", alpha=0.7, color="black")
        ax.axhline(0.0, lw=1.0, ls="--", alpha=0.7, color="black")
        ax.text(0.48, 0.64, "Synchronous phase",
                transform=ax.transAxes, rotation=90,
                va="top", ha="left", fontsize=8, color="black", alpha=0.7)

        # U_total line (always visible)
        ax.axhline(U0 / 1e3, lw=1.0, ls="--", alpha=0.7, color="black")
        ax.text(np.pi * 0.55, U0 * 1.15/ 1e3, r"$U_\mathrm{total}$",
                va="bottom", ha="left", fontsize=8, color="black", alpha=0.7)

        # U_w/o hc line (only when there is HC)
        if k > 0:
            U_wo_hc = Vrf * np.sin(phi_s) / 1e3
            ax.axhline(U_wo_hc, lw=1.0, ls="-.", alpha=0.6, color="black")
            ax.text(np.pi * 0.55, U_wo_hc * 0.85, r"$U_\mathrm{without\,hc}$",
                    va="top", ha="left", fontsize=8, color="black", alpha=0.6)
    
        # Simple "bunch" marker (Gaussian around φ=0), just for visualization
        sigma_phi = 0.15  # rad (purely illustrative)
        bunch = np.exp(-0.5 * (phi / sigma_phi) ** 2)
        bunch_scale = 0.08 * np.max(np.abs(V_tot / 1e3))
        ax.fill_between(phi, 0.0, bunch * bunch_scale, alpha=0.20,
                        label="Bunch (illustrative)")

        d1 = _dV_dphi_at0(Vrf, k, n, phi_s, phi_h)
        d2 = _d2V_dphi2_at0(Vrf, k, n, phi_s, phi_h)

        info = "\n".join([
            rf"$k$ = {k:.4f}",
            rf"$\phi_s$ = {deg(phi_s):.2f}°",
            rf"$\phi_h$ = {deg(phi_h):.2f}°",
            rf"$V'(0)$ = {d1/1e6:.3e} MV/rad",
            rf"$V''(0)$ = {d2/1e6:.3e} MV/rad²",
        ])
        ax.text(
            0.02, 0.98, info,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, lw=0.5),
        )
        ax.set_ylabel("Voltage [kV]")

    axes[-1].set_xlabel("Phase offset from synchronous particle  $\\phi$  [rad]")
    # Legend only once (top plot)
    axes[0].legend(loc="upper right")
    return fig, axes


# --- Compute the three cases (no HC / user / optimal flat potential) ---
n = m
Vrf = V_t_mc
U0 = U_0

# Flat-potential closed-form settings (k_fp, phi_h_fp)
phi_h_fp = flat_potential_phi_h(U0, Vrf, n=n)
k_fp = flat_potential_k(U0, Vrf, n=n)

# Case (i): no HC
phi_s0 = _phi_s_single_rf(U0, Vrf)

# Case (ii): user harmonic voltage (total), expressed as k = V_h_total / V_rf
k_user = (V_t_hc_total / Vrf)
phi_s_user = _phi_s_double_rf(U0, Vrf, k_user, phi_h_fp)

# Case (iii): optimal flat potential
phi_s_fp = _phi_s_flat_potential(U0, Vrf, n=n)

fig_vs, _ = plot_voltage_signals_cases(Vrf, U0, n, k_user, phi_h_fp,
                                       k_fp, phi_s0, phi_s_user, phi_s_fp)
savefig_pdf(fig_vs, "voltage_signals")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")


# -----------------------------
# EXERCISE 5
# -----------------------------
# Bunch longitudinal profile (homogeneous filling pattern)
#
# Key changes vs previous version:
#  - Use a robust trapezoidal integrator (_trapz) to avoid NumPy version issues.
#  - DO NOT normalize each rho(t) to peak=1 individually.
#    Instead, scale ALL profiles by the peak of the "no HC" case (so stretched
#    bunches have lower peak).
#  - Use a COMMON scaling for the potential well (normalized to the no-HC case
#    in the shown window).

def _trapz(y, x):
    """Trapezoidal integration compatible with old/new NumPy."""

    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def rf_potential_Phi(phi, Vrf, k, n, phi_s, phi_h, U0_eV, alpha_c,
                     E0_eV, C, omega_rf):
    """
    Dimensionless RF potential Φ(φ):

        Φ(φ) = -(c α_c)/(E C ω_rf) ∫_0^φ ( e V_t(φ') - U0 ) dφ'

    Voltage model:
        V_t(φ) = Vrf [ sin(φ + φ_s) + k sin(n φ + φ_h) ].

    We work in eV/Volts consistently (1 V ↔ 1 eV for an electron),
    so we can subtract U0_eV directly from V_t in Volts.

    The prefactor affects overall scaling; the *shape* controls rho.
    """

    # Closed-form integral of (V_t(φ) - U0) from 0 to φ
    integral_eV = (
        Vrf * (np.cos(phi_s) - np.cos(phi + phi_s))
        + (Vrf * k / n) * (np.cos(phi_h) - np.cos(n * phi + phi_h))
        - U0_eV * phi
    )

    pref = (c * alpha_c) / (E0_eV * C * omega_rf)
    return -pref * integral_eV


def rho_phi_from_Phi(phi, Phi, alpha_c, sigma_e):
    """
    Equilibrium density in phase:
        ρ(φ) ∝ exp( - Φ(φ) / (α_c^2 σ_e^2) ),
    normalized so that ∫ ρ(φ) dφ = 1 over the given grid.
    """

    expo = -Phi / (alpha_c**2 * sigma_e**2)
    expo -= np.max(expo)  # numerical stability
    w = np.exp(expo)
    w /= _trapz(w, phi)
    return w


def bunch_profile_case(phi, Vrf, k, n, phi_s, phi_h, U0_eV, alpha_c, sigma_e,
                       E0_eV, C, omega_rf):
    """
    Returns:
      t_ps      : time axis in ps (t = φ/ω_rf)
      Phi       : Φ(φ)
      rho_t     : density in time, normalized to ∫ rho_t dt = 1
      sigma_t_ps: RMS bunch length in time [ps]
    """

    Phi = rf_potential_Phi(phi, Vrf, k, n, phi_s, phi_h, U0_eV, alpha_c,
                           E0_eV, C, omega_rf)
    rho_phi = rho_phi_from_Phi(phi, Phi, alpha_c, sigma_e)

    # Map φ -> t
    t = phi / omega_rf           # [s]
    t_ps = t * 1e12              # [ps]

    # Convert density: rho(t) dt = rho(φ) dφ => rho(t) = rho(φ) * ω_rf
    rho_t = rho_phi * omega_rf
    rho_t /= _trapz(rho_t, t) # safety renormalization (should already be consistent)

    # RMS bunch length in time
    t_mean = _trapz(t * rho_t, t)
    sigma_t = np.sqrt(_trapz((t - t_mean) ** 2 * rho_t, t))  # [s]
    sigma_t_ps = sigma_t * 1e12

    return t_ps, Phi, rho_t, sigma_t_ps


def plot_bunch_profiles():
    """Two-panel figure of the RF potential well and equilibrium bunch profiles
    for the no-HC, user-voltage, and flat-potential cases (Exercise 5)."""

    # Full bucket grid
    phi = np.linspace(-np.pi, np.pi, 20001)

    omega_rf = 2.0 * np.pi * f_rf_mc
    Vrf = V_t_mc
    n = m

    # Cases (phi_s0, phi_s_user, phi_s_fp, k_user, k_fp, phi_h_fp come from Exercise 4)
    cases = [
        ("Profile without HC", 0.0,      0.0,      phi_s0),
        ("Current profile", k_user,  phi_h_fp, phi_s_user),
        ("Optimal profile", k_fp,    phi_h_fp, phi_s_fp),
    ]

    # Compute curves first
    curves = []
    for label, k, phi_h, phi_s in cases:
        t_ps, Phi, rho_t, sigma_t_ps = bunch_profile_case(
            phi, Vrf, k, n, phi_s, phi_h,
            U0_eV=U_0, alpha_c=alpha_c, sigma_e=sigma_e,
            E0_eV=E_0, C=length_sr, omega_rf=omega_rf
        )
        curves.append((label, t_ps, Phi, rho_t, sigma_t_ps))

    # Plot window (as in the reference)
    tlim = 100.0  # [ps]
    mask = np.abs(curves[0][1]) <= tlim

    # --- Common scaling for potential: normalize to no-HC (in the shown window)
    Phi0_win = curves[0][2][mask]
    Phi0_shift = Phi0_win - np.min(Phi0_win)
    Phi_scale = np.max(Phi0_shift) if np.max(Phi0_shift) > 0 else 1.0

    # --- Common scaling for density: normalize to peak of no-HC (so others have lower peak)
    rho0_peak = np.max(curves[0][3]) if np.max(curves[0][3]) > 0 else 1.0

    fig, (axP, axR) = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)

    sig_t = {}

    style = {
        "Profile without HC":  dict(lw=2.2, ls="-",  color="#2A5FA5"),
        "Current profile": dict(lw=2.2, ls="-",  color="#E86B3E"),
        "Optimal profile": dict(lw=2.2, ls=":",  color="#3A9E6F"),
    }

    for label, t_ps, Phi, rho_t, sigma_t_ps in curves:
        sig_t[label] = sigma_t_ps

        # Potential (scaled common)
        Phi_win = Phi[mask]
        Phi_plot = (Phi_win - np.min(Phi_win)) / Phi_scale
        axP.plot(t_ps[mask], Phi_plot, label=label, **style.get(label, {}))

        # Density (scaled to no-HC peak)
        rho_plot = rho_t / rho0_peak
        axR.plot(t_ps, rho_plot, label=label, **style.get(label, {}))

    # ---- Formatting: potential
    axP.set_xlabel("t  [ps]")
    axP.set_ylabel("Potential  [arbitrary units]")
    axP.grid(False)
    axP.legend(loc="upper right")
    axP.set_xlim(-tlim, tlim)

    # ---- Formatting: density
    axR.set_xlabel("t  [ps]")
    axR.set_ylabel(r"$e^{-\Phi}\;$ (normalized to no-HC peak)")
    axR.grid(False)
    axR.legend(loc="upper right")
    axR.set_xlim(-tlim, tlim)
    axR.set_ylim(0.0, 1.05)

    # ---- Info box (like the reference)
    s0 = sig_t["Profile without HC"]
    u_cur = sig_t["Current profile"] / s0
    u_opt = sig_t["Optimal profile"] / s0
    xi = k_user / k_fp
    T = np.nan if np.isclose(xi, 0.0) else u_cur / xi

    Vt_MV = V_t_mc / 1e6
    Vhh_cur_kV = (Vrf * k_user) / 1e3
    Vhh_opt_kV = (Vrf * k_fp) / 1e3

    info = (
        f"$V_t$ = {Vt_MV:.2f} MV\n"
        f"$V_\\mathrm{{hc}}$ = {Vhh_cur_kV:.1f} kV\n"
        f"$V_\\mathrm{{hc,opt}}$ = {Vhh_opt_kV:.1f} kV\n\n"
        f"$\\sigma_t$ no hc = {sig_t['Profile without HC']:.2f} ps\n"
        f"$\\sigma_t$ current = {sig_t['Current profile']:.2f} ps\n"
        f"$\\sigma_t$ optimal = {sig_t['Optimal profile']:.2f} ps\n\n"
        f"$u$ = {u_cur:.2f}\n"
        f"$u_\\mathrm{{opt}}$ = {u_opt:.2f}\n"
        f"$\\xi$ = {xi:.3f}\n"
        f"$T$ = {T:.2f}"
    )
    axR.text(
        0.03, 0.97, info,
        transform=axR.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, lw=0.5),
    )

    return fig, (axP, axR)


# --- Exercise 5 plot ---
fig_bp, _ = plot_bunch_profiles()
savefig_pdf(fig_bp, f"potential_profiles_{VALIDATION_CASE}")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")


# ============================================================
# EXERCISE 6: Bunch lengthening ratio u vs harmonic voltage
# u(V_HC) = sigma_z(with HC at that voltage) / sigma_z0(no HC)
# RF frequency: omega_rf_mc is computed once near the top of the file.
# ============================================================

def flat_potential_settings(Vrf, U0, n):
    """
    Returns:
      k_opt     = V_HC_opt / Vrf
      phi_s_opt = main RF phase offset (stable branch)
      phi_h_opt = harmonic phase offset (argument offset in sin(n*phi + phi_h))
    """

    s = U0 / Vrf  # (eV)/(V) -> treat 1 V = 1 eV for electrons
    # k_opt from cancellation-condition model (matches classic result form).
    k_opt = np.sqrt(1.0 / n**2 - (s**2) / (n**2 - 1.0))

    # stable synchronous phase (electron storage ring convention)
    phi_s_opt = np.pi - np.arcsin((n**2 / (n**2 - 1.0)) * s)

    # recover phi_h from derivative-cancellation relations:
    # cos(phi_s) = -k n cos(phi_h),  sin(phi_s) = -k n^2 sin(phi_h)
    sin_phi_h = -np.sin(phi_s_opt) / (k_opt * n**2)
    cos_phi_h = -np.cos(phi_s_opt) / (k_opt * n)
    phi_h_opt = np.arctan2(sin_phi_h, cos_phi_h)

    return k_opt, phi_s_opt, phi_h_opt


def sigma_z_from_voltage(Vrf, U0, n, Vhc, phi_h,
                         alpha_c, sigma_e, C, omega_rf,
                         Nphi=20001):
    """
    Returns RMS bunch length sigma_z [m] for given harmonic voltage Vhc [V].
    """

    phi = np.linspace(-np.pi, np.pi, Nphi)
    i0 = np.argmin(np.abs(phi))

    k = Vhc / Vrf
    s0 = U0 / Vrf

    # enforce energy balance at bunch center phi=0:
    # sin(phi_s) + k sin(phi_h) = U0/Vrf  ->  sin(phi_s) = s0 - k sin(phi_h)
    arg = np.clip(s0 - k * np.sin(phi_h), -1.0, 1.0)
    phi_s = np.pi - np.arcsin(arg)  # stable branch

    Vt = Vrf * (np.sin(phi + phi_s) + k * np.sin(n * phi + phi_h))  # [V] ~ [eV]
    g = Vt - U0  # integrand (eV per rad in this convention)

    # Potential-like function Phi(phi) from integral of (eV - U0).
    # IMPORTANT: build integral from 0 to phi with correct sign for negative phi.
    I = np.zeros_like(phi)
    I[i0:] = cumulative_trapezoid(g[i0:], phi[i0:], initial=0.0)
    I[:i0+1] = cumulative_trapezoid(g[:i0+1][::-1], phi[:i0+1][::-1], initial=0.0)[::-1]

    K = (c * alpha_c) / (E_0 * C * omega_rf)
    Phi = -K * I

    expo = -Phi / (alpha_c**2 * sigma_e**2)
    expo -= np.max(expo)  # numerical stability
    rho = np.exp(expo)
    rho /= _trapz(rho, phi)

    phi_bar = _trapz(phi * rho, phi)
    sigma_phi = np.sqrt(_trapz((phi - phi_bar)**2 * rho, phi))

    sigma_z = c * sigma_phi / omega_rf
    return sigma_z


# -----------------------------
# Compute curve u(V_HC)
# -----------------------------
Vrf = V_t_mc               # total main RF voltage [V]
Uloss = U_0                # energy loss/turn [eV]
n = m                      # harmonic order (e.g. 3)

k_opt, phi_s_opt, phi_h_opt = flat_potential_settings(Vrf, Uloss, n)
Vhc_opt_total = k_opt * Vrf

# User setting: V_c_hc is per HC cavity, total harmonic voltage is N_hc * V_c_hc
Vhc_user_total = N_hc * V_c_hc

# Baseline: no harmonic voltage
sigma_z0 = sigma_z_from_voltage(
    Vrf=Vrf, U0=Uloss, n=n, Vhc=0.0, phi_h=phi_h_opt,
    alpha_c=alpha_c, sigma_e=sigma_e, C=length_sr, omega_rf=omega_rf_mc
)

# Scan harmonic voltage around optimum (adjust range if you want)
Vhc_scan = np.linspace(0.0, 1.02 * Vhc_opt_total, 220)
u_scan = np.zeros_like(Vhc_scan)

for i, Vhc in enumerate(Vhc_scan):
    sig = sigma_z_from_voltage(
        Vrf=Vrf, U0=Uloss, n=n, Vhc=Vhc, phi_h=phi_h_opt,
        alpha_c=alpha_c, sigma_e=sigma_e, C=length_sr, omega_rf=omega_rf_mc
    )
    u_scan[i] = sig / sigma_z0

# Values at user and optimal settings (for annotation)
u_user = np.interp(Vhc_user_total, Vhc_scan, u_scan)
u_opt = np.interp(Vhc_opt_total, Vhc_scan, u_scan)

print(f"[Ex6] sigma_z0 (no HC)        = {sigma_z0*1e3:.3f} mm")
print(f"[Ex6] V_HC,opt (total)       = {Vhc_opt_total/1e3:.2f} kV  -> u_opt ~ {u_opt:.2f}")
print(f"[Ex6] V_HC,user (total)      = {Vhc_user_total/1e3:.2f} kV -> u_user ~ {u_user:.2f}")

# -----------------------------
# Plot: u vs V_HC
# -----------------------------
fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.plot(Vhc_scan/1e3, u_scan, lw=2.0, color="#2A5FA5", label=r"$u=\sigma_z/\sigma_{z,0}$")

ax.axvline(Vhc_opt_total/1e3, ls="--", lw=1.5, color="#3A9E6F",
           label=fr"$V_{{HC,opt}}$ = {Vhc_opt_total/1e3:.1f} kV")
ax.axvline(Vhc_user_total/1e3, ls=":", lw=2.0, color="#E86B3E",
           label=fr"$V_{{HC,user}}$ = {Vhc_user_total/1e3:.1f} kV")

ax.set_xlabel(r"Total harmonic voltage $V_{HC}$  [kV]")
ax.set_ylabel(r"Bunch lengthening ratio $u=\sigma_z/\sigma_{z,0}$")
ax.legend()

# Optional: top axis normalized to V_opt
Vopt_kV = Vhc_opt_total/1e3
secax = ax.secondary_xaxis(
    "top",
    functions=(lambda x_kV: x_kV / Vopt_kV,
               lambda x_norm: x_norm * Vopt_kV)
)
secax.set_xlabel(r"$V_{HC}/V_{HC,opt}$")

plt.tight_layout()
savefig_pdf(fig, "lengthening_ratio_vs_vhc")

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")