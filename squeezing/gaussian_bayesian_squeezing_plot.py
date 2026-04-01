"""
Generates a single figure with two subplots (shared x-axis):
  (a) vacuum probe state
  (b) thermal probe state
for squeezing estimation MSL ratio L_R vs prior variance sigma^2.

Paste this alongside your original script or run standalone — it imports
nothing from the original file; all needed functions are duplicated here.
"""

import os, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------------------------------------------ #
#  Operator / state helpers (same as original)
# ------------------------------------------------------------------ #

def _make_ops(N):
    a = np.zeros((N, N), dtype=complex)
    for n in range(1, N):
        a[n-1, n] = np.sqrt(n)
    adag = a.conj().T
    I    = np.eye(N, dtype=complex)
    x    = (a + adag) / np.sqrt(2)
    p    = (a - adag) / (1j * np.sqrt(2))
    return a, adag, I, x, p


def squeeze_op(r, phi, a, adag):
    G = 0.5 * (np.exp(-2j*phi) * a @ a - np.exp(2j*phi) * adag @ adag)
    return la.expm(r * G)


def displace_op(alpha, a, adag):
    return la.expm(alpha * adag - np.conj(alpha) * a)


def thermal_state(n_bar, N):
    rho = np.zeros((N, N), dtype=complex)
    for n in range(N):
        rho[n, n] = (n_bar**n / (1 + n_bar)**(n+1)) if n_bar > 0 else (1.0 if n == 0 else 0.0)
    return rho


def reference_state(state_type, N, ops, x0=0.0, p0=0.0, alpha=1.0,
                    n_th=0.2, r=0.4, phi=0.0):
    a, adag, I, x, p = ops
    if state_type == 'vacuum':
        rho = np.zeros((N, N), dtype=complex); rho[0, 0] = 1.0
    elif state_type == 'thermal':
        rho = thermal_state(n_th, N)
    elif state_type == 'coherent':
        D = displace_op(alpha, a, adag)
        vac = np.zeros((N, N), dtype=complex); vac[0, 0] = 1.0
        rho = D @ vac @ D.conj().T
    elif state_type == 'squeezed_vacuum':
        vac = np.zeros((N, N), dtype=complex); vac[0, 0] = 1.0
        S = squeeze_op(r, phi, a, adag)
        rho = S @ vac @ S.conj().T
    else:
        raise ValueError(state_type)
    if abs(x0) > 1e-10 or abs(p0) > 1e-10:
        alpha_d = (x0 + 1j*p0) / np.sqrt(2)
        D = displace_op(alpha_d, a, adag)
        rho = D @ rho @ D.conj().T
    rho = 0.5*(rho + rho.conj().T)
    return rho / np.trace(rho)


def get_prior(theta_grid, theta0, theta_sigma, theta_max):
    prior = np.exp(-0.5*((theta_grid - theta0)/theta_sigma)**2)
    dt = theta_grid[1] - theta_grid[0]
    return prior / (np.sum(prior)*dt)


def HS(A, B):
    return np.real(np.trace(A.conj().T @ B))


def get_optimal_coefficients(rho0, rho1, B):
    m = len(B)
    G = np.array([[0.5*HS(B[i], rho0 @ B[j] + B[j] @ rho0) for j in range(m)] for i in range(m)])
    b = np.array([HS(B[i], rho1) for i in range(m)])
    alpha_opt, *_ = la.lstsq(G, b)
    return alpha_opt, G, b


def msl_bayes_for_pvm(S_op, rho0, rho1, lambda_val, theta0, I):
    eigvals, eigvecs = la.eigh(S_op)
    if np.linalg.norm(S_op - theta0*I) < 1e-10:
        return lambda_val - theta0**2
    gain = 0.0
    for k in range(len(eigvals)):
        ket = eigvecs[:, k:k+1]
        Pk  = ket @ ket.conj().T
        pk  = np.real(np.trace(Pk @ rho0))
        mk  = np.real(np.trace(Pk @ rho1))
        if pk > 1e-10:
            gain += mk**2 / pk
    return lambda_val - gain


# ------------------------------------------------------------------ #
#  Design parameters (same logic as original)
# ------------------------------------------------------------------ #

def design_parameters(N, ref_state_type, n_th=0.2, r=0.4, theta0=0.5,
                      sigma_pts=10, safety_factor=3):
    n0 = {'vacuum': 0, 'thermal': n_th,
          'squeezed_vacuum': np.sinh(r)**2}[ref_state_type]
    n_budget = N / safety_factor - n0
    if n_budget < 1:
        raise ValueError("Insufficient Fock truncation")
    if ref_state_type == 'vacuum':
        theta_max = 0.5 * np.log(4 * n_budget)
    elif ref_state_type == 'thermal':
        theta_max = 0.5 * np.log(2 * n_budget / (n_th + 0.5))
    elif ref_state_type == 'squeezed_vacuum':
        theta_max = 0.5 * np.log(4 * n_budget) - r

    theta_min   = 2*theta0 - theta_max
    sigma_max   = 2*(theta_max - abs(theta0)) / 3
    dtheta      = (theta_max - theta_min) / 1000
    sigma_min   = 10 * dtheta
    sigma_vals  = np.logspace(-1.2, np.log10(sigma_max), sigma_pts)
    return theta_min, theta_max, sigma_vals


# ------------------------------------------------------------------ #
#  Per-sigma computation (self-contained, no globals)
# ------------------------------------------------------------------ #

def compute_one_sigma(args):
    (theta_sigma, theta0, theta_min, theta_max, theta_pts,
     N, ref_state_type, n_th, r_sq, phi_homodyne) = args

    a, adag, I, x, p = _make_ops(N)
    ops = (a, adag, I, x, p)

    theta_grid = np.linspace(theta_min, theta_max, theta_pts)
    dt = theta_grid[1] - theta_grid[0]

    prior = get_prior(theta_grid, theta0, theta_sigma, theta_max)
    prior_var = np.sum((theta_grid - np.sum(theta_grid*prior*dt))**2 * prior * dt)

    rho_ref = reference_state(ref_state_type, N, ops, n_th=n_th, r=r_sq)

    rho0 = np.zeros((N,N), dtype=complex)
    rho1 = np.zeros((N,N), dtype=complex)
    for i, th in enumerate(theta_grid):
        S    = squeeze_op(th, 0, a, adag)
        rho_th = S @ rho_ref @ S.conj().T
        rho_th = 0.5*(rho_th + rho_th.conj().T) / np.trace(rho_th)
        rho0 += prior[i] * rho_th * dt
        rho1 += prior[i] * th * rho_th * dt
    rho0 = 0.5*(rho0 + rho0.conj().T)
    rho1 = 0.5*(rho1 + rho1.conj().T)
    lambda_val = np.sum(prior * theta_grid**2 * dt)

    # Bayes-optimal S
    dim = N*N
    lhs = np.kron(np.eye(N), rho0) + np.kron(rho0.T, np.eye(N))
    S_bayes = la.pinv(lhs) @ (2.0 * rho1.reshape(dim, order='F'))
    S_bayes = S_bayes.reshape((N,N), order='F')
    S_bayes = 0.5*(S_bayes + S_bayes.conj().T)
    msl_bayes = lambda_val - np.real(np.trace(rho0 @ S_bayes @ S_bayes))

    def ratio(msl):
        return msl / msl_bayes - 1.0

    # Prior-only baseline: B = {I}
    aP, G_P, b_P = get_optimal_coefficients(rho0, rho1, [I])
    msl_prior = lambda_val - b_P @ la.pinv(G_P) @ b_P

    # Quadratic basis: {I, x_phi, x_phi^2}  (phi=pi/2 → p)
    xphi = x*np.cos(np.pi/2) + p*np.sin(np.pi/2)   # = p
    B_quad = [0.5*(M+M.conj().T) for M in [I, xphi, xphi@xphi]]
    aQ, G_Q, b_Q = get_optimal_coefficients(rho0, rho1, B_quad)
    msl_quad = lambda_val - b_Q @ la.pinv(G_Q) @ b_Q
    S_quad = sum(aQ[i]*B_quad[i] for i in range(len(B_quad)))
    S_quad = 0.5*(S_quad + S_quad.conj().T)
    msl_quad_pm = msl_bayes_for_pvm(S_quad, rho0, rho1, lambda_val, theta0, I)

    # Cubic basis: {I, x^2, p^2}
    B_cub = [0.5*(M+M.conj().T) for M in [I, x@x, p@p]]
    aC, G_C, b_C = get_optimal_coefficients(rho0, rho1, B_cub)
    msl_cub = lambda_val - b_C @ la.pinv(G_C) @ b_C
    S_cub = sum(aC[i]*B_cub[i] for i in range(len(B_cub)))
    S_cub = 0.5*(S_cub + S_cub.conj().T)
    msl_cub_pm = msl_bayes_for_pvm(S_cub, rho0, rho1, lambda_val, theta0, I)

    return (prior_var,
            ratio(msl_prior),
            ratio(msl_quad),  ratio(msl_quad_pm),
            ratio(msl_cub),   ratio(msl_cub_pm))


# ------------------------------------------------------------------ #
#  Run for one state type and return ratio arrays
# ------------------------------------------------------------------ #

def run_case(ref_state_type, N=30, theta0=0.1, theta_pts=1000,
             sigma_pts=10, safety_factor=5, n_th=0.02, r_sq=0.4):

    theta_min, theta_max, sigma_vals = design_parameters(
        N, ref_state_type, n_th=n_th, r=r_sq,
        theta0=theta0, sigma_pts=sigma_pts, safety_factor=safety_factor)

    args_list = [
        (sig, theta0, theta_min, theta_max, theta_pts,
         N, ref_state_type, n_th, r_sq, 0.0)
        for sig in sigma_vals
    ]

    label = ref_state_type
    with ProcessPoolExecutor() as ex:
        results = list(tqdm(ex.map(compute_one_sigma, args_list),
                            total=len(sigma_vals), desc=label))

    (pv, rP, rQ, rQ_pm, rC, rC_pm) = zip(*results)
    return (np.array(pv),
            np.array(rP),
            np.array(rQ),  np.array(rQ_pm),
            np.array(rC),  np.array(rC_pm))


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

if __name__ == '__main__':

    cases = [
        ('vacuum',  '(a)'),
        ('thermal', '(b)'),
    ]

    all_results = {}
    for state, _ in cases:
        print(f"\n--- {state} ---")
        all_results[state] = run_case(state)

    # ---- Plotting ------------------------------------------------- #
    lw = 4
    C_gray  = "#959ba0"
    C_green = "#2ca02c"
    C_black = "#000000"

    fig, axes = plt.subplots(2, 1, figsize=(7, 10),
                             sharex=True,
                             gridspec_kw={'hspace': 0.08})

    for ax, (state, panel_label) in zip(axes, cases):
        pv, rP, rQ, rQ_pm, rC, rC_pm = all_results[state]

        ax.loglog(pv, rP,     ':', lw=lw,   color=C_gray,  label='Prior')
        ax.loglog(pv, rQ,     '--', lw=lw,  color=C_green, label='Quadratic')
        ax.loglog(pv, rQ_pm,  '-',  lw=lw,  color=C_green, label='Quadratic (PM)')
        ax.loglog(pv, rC,     '--', lw=lw,  color=C_black, label='Cubic')
        ax.loglog(pv, rC_pm,  '-',  lw=lw,  color=C_black, label='Cubic (PM)')

        ax.set_ylabel('$\\mathcal{L}_R$', fontsize=26)
        ax.tick_params(axis='both', which='major', length=8, width=1.8, labelsize=17)
        ax.tick_params(axis='both', which='minor', length=4, width=1.2)
        ax.grid(False)

        # panel label
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        ax.text(0.04, 0.92, panel_label, transform=ax.transAxes,
                fontsize=26, va='top')

        # ---- inset: zoomed large-sigma region ---- #
        axins = ax.inset_axes([0.55, 0.08, 0.40, 0.33])
        zm = pv >= 0.15
        axins.loglog(pv[zm], rP[zm],    ':',  lw=lw-2, color=C_gray)
        axins.loglog(pv[zm], rQ[zm],    '--', lw=lw-2, color=C_green)
        axins.loglog(pv[zm], rQ_pm[zm], '-',  lw=lw-2, color=C_green)
        axins.loglog(pv[zm], rC[zm],    '--', lw=lw-2, color=C_black)
        axins.loglog(pv[zm], rC_pm[zm], '-',  lw=lw-2, color=C_black)
        axins.set_xlim(pv[zm][0], pv[zm][-1])
        axins.set_ylim(top=1)
        axins.tick_params(which='both', left=False, labelleft=False,
                          bottom=False, labelbottom=False)
        axins.grid(False)
        ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.7)

    axes[-1].set_xlabel('$\\sigma^2_0$', fontsize=26)

    # Shared legend under the bottom panel
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=16,
               ncol=3, bbox_to_anchor=(0.5, -0.04), frameon=True)

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    output_dir = Path(__file__).parent / "figs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "squeezing_LR_combined.png", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / "squeezing_LR_combined.pdf", bbox_inches='tight')
    print(f"\nSaved to {output_dir}/squeezing_LR_combined.{{png,pdf}}")

    # Log-log slope fits (small-sigma region)
    print("\nLog-log slopes (sigma^2 < 0.01):")
    for state, _ in cases:
        pv, rP, rQ, rQ_pm, rC, rC_pm = all_results[state]
        fm = pv < 0.01
        print(f"  [{state}]")
        for lbl, arr in [('Green constrained', rQ), ('Green PM', rQ_pm),
                         ('Black constrained', rC), ('Black PM', rC_pm)]:
            c = np.polyfit(np.log(pv[fm]), np.log(arr[fm]), 1)
            print(f"    {lbl}: slope = {c[0]:.3f}")

    plt.show()