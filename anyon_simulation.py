"""
Simulation of a nu = 1/3 Fabry-Perot interferometer following Supplementary
Discussion 2 of Nakamura et al., "Direct observation of anyonic braiding
statistics", Nat. Phys. 16, 931 (2020), arXiv:2006.14115.

Thermal expectation value of the conductance oscillation (Supp. Eqs. 5-6):

    <dG> = (1/Z) sum_N exp(-E(N)/kT) cos(2 pi (e*/e) A_I B / Phi0 + N theta_anyon)
    Z    =       sum_N exp(-E(N)/kT)

with the device energy (Supp. Eq. 7, constant offset E0 dropped)

    E(N) = (e^2 / 2C) (dq_net / e)^2 + Delta_qp |N|

and the net charge (Supp. Eq. 3), with q_donor = e n A_I:

    dq_net / e = nu A_I B / Phi0 + (e*/e) N - n A_I - alpha_bulk dVg

Units used throughout: B in T, dVg in mV, energies and kBT in K,
e_star in units of the electron charge e (e.g. 1/3).
"""

import numpy as np

# Physical constants (SI)
E_CHARGE = 1.602176634e-19       
H_PLANCK = 6.62607015e-34        
K_B = 1.380649e-23               
EPS0 = 8.8541878128e-12          
PHI0 = H_PLANCK / E_CHARGE       

# Device / model parameters (paper values)
NU = 1 / 3                       
THETA_ANYON = 2 * np.pi / 3      
DELTA_QP = 5.5 / 2               
N_DENSITY = 0.7e11 * 1e4         
A0 = 0.1e-12                     
DA_DVG = 0.167e-15    
A_REAL = 0.38e-12
# bulk lever arm measured on the real device (0.07 / mV), scaled to the smaller simulated area
ALPHA_BULK = 0.07 * A0 / A_REAL
EPS_R = 12.9                     
D_SCREEN = 48e-9                 


def area(dVg):
    return A0 + DA_DVG * np.asarray(dVg, dtype=float)


def charging_energy(A):
    C = 2 * EPS_R * EPS0 * A / D_SCREEN
    return E_CHARGE**2 / (2 * C) / K_B


def _n_axis(N_min, N_max):
    return np.arange(N_min, N_max + 1)


def net_charge(N, dVg, B, e_star):
    A = area(dVg)
    return NU * A * B / PHI0 + e_star * N - N_DENSITY * A - ALPHA_BULK * dVg


def energy(N, dVg, B, e_star):
    Ec = charging_energy(area(dVg))
    return Ec * net_charge(N, dVg, B, e_star) ** 2 + DELTA_QP * np.abs(N)


def phase(N, dVg, B, e_star):
    return 2 * np.pi * e_star * area(dVg) * B / PHI0 + N * THETA_ANYON


def shifted_weights(dVg, B, e_star, N_min, N_max, kBT):
    dVg = np.asarray(dVg, dtype=float)[..., None]
    B = np.asarray(B, dtype=float)[..., None]
    N = _n_axis(N_min, N_max)
    E = energy(N, dVg, B, e_star)
    E_min = E.min(axis=-1, keepdims=True)
    return N, dVg, B, np.exp(-(E - E_min) / kBT), E_min[..., 0]


def partition_function(dVg, B, e_star, N_min, N_max, kBT, shifted=False):
    _, _, _, w, E_min = shifted_weights(dVg, B, e_star, N_min, N_max, kBT)
    Z_shifted = w.sum(axis=-1)
    if shifted:
        return Z_shifted
    return Z_shifted * np.exp(-E_min / kBT)


def expectation_value(dVg, B, e_star, N_min, N_max, kBT):
    N, dVg_, B_, w, _ = shifted_weights(dVg, B, e_star, N_min, N_max, kBT)
    cos_theta = np.cos(phase(N, dVg_, B_, e_star))
    return (w * cos_theta).sum(axis=-1) / w.sum(axis=-1)


def expectation_N(dVg, B, e_star, N_min, N_max, kBT):
    N, _, _, w, _ = shifted_weights(dVg, B, e_star, N_min, N_max, kBT)
    return (w * N).sum(axis=-1) / w.sum(axis=-1)


def simulate_grid(B_range=(8.2, 9.1), dVg_range=(-30.0, 0.0), nB=600, nV=500,
                  e_star=1 / 3, N_min=-20, N_max=20, kBT=None):
    if kBT is None:
        kBT = 0.002 * charging_energy(A0)
    B_grid, dVg_grid = np.meshgrid(np.linspace(*B_range, nB),
                                   np.linspace(*dVg_range, nV))
    dG = expectation_value(dVg_grid, B_grid, e_star, N_min, N_max, kBT)
    return B_grid, dVg_grid, dG


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Ec0 = charging_energy(A0)
    ratios = (0.1,0.01,0.001)
    print(f"E_c(A0) = {Ec0:.3f} K; centre of state B = {N_DENSITY   * PHI0 / NU:.3f} T")

    fig, axes = plt.subplots(1, 3, figsize=(22, 4.5), layout="constrained")
    for ax, r in zip(axes, ratios):
        B_grid, dVg_grid, dG = simulate_grid(kBT=r * Ec0)
        mesh = ax.pcolormesh(B_grid, dVg_grid, dG, cmap="RdBu_r", vmin=-1, vmax=1,
                             shading="auto")
        ax.set_title(rf"$k_BT/E_c = {r}$")
        ax.set_xlabel("B (T)")
        ax.set_ylabel(r"$\delta V_g$ (mV)")
    fig.colorbar(mesh, ax=axes[:3], label=r"$\langle\delta G\rangle / a_0$")

    fig.savefig("supp_fig5_simulation.png", dpi=150, bbox_inches="tight")
    plt.show()
