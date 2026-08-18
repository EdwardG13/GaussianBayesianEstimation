"""
Computes MSL vs prior width for squeezing estimation across constrained bases
and the Bayes-optimal bound, then saves everything to disk.

The encoded state is
    rho(theta) = S(theta) rho_in S^dagger(theta),   S(theta) = exp(i theta {q,p} / 2),
so theta is a location parameter (S(t1)S(t2) = S(t1+t2)) and f(theta) = theta.

Companion to plot_squeezing.py, which reads the files written here.

Outputs:
    data/squeezing_{ref_state_type}_{prior_type}.npz   (all numerical arrays)
    data/squeezing_{ref_state_type}_{prior_type}.json  (scalar/string metadata)
"""

import json
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.optimize import brentq
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Unicode values for some characters
sigma_unicode = '\u03C3'
sigma2_unicode = '\u03C3\u00B2'
alpha_unicode = '\u03B1'
theta_unicode = '\u03B8'
nbar_unicode = '\u0304n'
phi_unicode = '\u03D5'

# ------------------------------------------------- Functions -------------------------------------------------

def probe_moments(ref_state_type, x0=0.0, p0=0.0, alpha=1.0, n_th=0.1, r=0.4,phi_squeeze=0):
    """
    First and second moments (qbar, pbar, Vqq, Vpp, Vqp) of the probe state, before the squeezing is applied.
    The photon budget and the analytic MSL formulas are functions of these alone.
    """
    if ref_state_type == 'vacuum':
        qbar, pbar, Vqq, Vpp = x0, p0, 0.5, 0.5
    elif ref_state_type == 'coherent':
        qbar = np.sqrt(2)*np.real(alpha) + x0
        pbar = np.sqrt(2)*np.imag(alpha) + p0
        Vqq, Vpp = 0.5, 0.5
    elif ref_state_type == 'thermal':
        qbar, pbar, Vqq, Vpp = x0, p0, n_th + 0.5, n_th + 0.5
    elif ref_state_type in ('squeezed_vacuum', 'squeezed_thermal'):
        qbar, pbar = x0, p0
        V0 = 0.5 if ref_state_type == 'squeezed_vacuum' else n_th + 0.5
        c, s = np.cos(phi_squeeze)**2, np.sin(phi_squeeze)**2
        Vqq = V0*(np.exp(-2*r)*c + np.exp(2*r)*s)
        Vpp = V0*(np.exp(-2*r)*s + np.exp(2*r)*c)
        Vqp = V0*np.cos(phi_squeeze)*np.sin(phi_squeeze)*(np.exp(-2*r) - np.exp(2*r))
    else:
        raise ValueError(f"Unknown state type: {ref_state_type}")
    return qbar, pbar, Vqq, Vpp, 0.0

def photon_number_at(theta, moments):
    """
    Mean photon number of the squeezed probe.
    Squeezing acts as r -> Sigma r and V -> Sigma V Sigma^T with Sigma = diag(e^-theta, e^theta), so

    <n>(theta) = [ (Vqq + qbar^2) e^{-2theta} + (Vpp + pbar^2) e^{2theta} - 1 ] / 2.
    """
    qbar, pbar, Vqq, Vpp, _ = moments
    return 0.5*((Vqq + qbar**2)*np.exp(-2*theta) + (Vpp + pbar**2)*np.exp(2*theta) - 1.0)

def theta_range_supported(N, moments, safety_factor):
    """
    Interval of theta the Fock truncation supports, i.e. where <n>(theta) <= N/safety_factor.

    Unlike the displacement case, <n> grows exponentially in theta and is asymmetric whenever
    Vqq + qbar^2 != Vpp + pbar^2, so the two endpoints are found numerically.
    """
    n_max = N / safety_factor
    if photon_number_at(0.0, moments) >= n_max:
        raise ValueError("Insufficient Fock truncation: the probe alone exceeds the budget")
    f = lambda t: photon_number_at(t, moments) - n_max
    hi = brentq(f, 0.0, 50.0) # Find the theta where <n>(theta)=n_max=N/safety_factor
    lo = brentq(f, -50.0, 0.0)
    return lo, hi

def theta_scale_from_variance(prior_var, prior_type):
    """
    Convert a target prior variance sigma_0^2 into the width parameter of each prior family:
    the standard deviation for 'gaussian', the half-width for 'uniform' (variance h^2/3).
    """
    if prior_type in ('gaussian', 'two_gaussian'):
        return np.sqrt(prior_var)
    if prior_type == 'uniform':
        return np.sqrt(3 * prior_var)
    raise ValueError(f"Unknown prior type: {prior_type}")

def grid_half_width(theta_scale, prior_type, grid_sigmas):
    # How far from theta0 the grid must reach for this prior to be faithfully represented
    return theta_scale if prior_type == 'uniform' else grid_sigmas * theta_scale

def squeeze_op(theta):
    """
    Squeezing operator S(theta) = exp(i theta {q,p}/2) = exp(theta (a^2 - a^dag^2)/2).

    We can simplify the matrix exponential by using the eigendecomposition of the generator
    (which is anti-Hermitian, so i*G is Hermitian).

    """
    return (sq_eigvecs * np.exp(-1j * theta * sq_eigvals)) @ sq_eigvecs.conj().T

def displace_op(alpha):
    # Displacement operator D(alpha) = exp(alpha a^dagger - alpha* a)
    return la.expm(alpha * adag - np.conj(alpha) * a)

def static_squeeze_op(r, phi):
    # Squeezing at an arbitrary angle, used only to prepare squeezed probe states
    G = 0.5 * (np.exp(-2j*phi) * a @ a - np.exp(2j*phi) * adag @ adag)
    return la.expm(r * G)

def thermal_state(n_bar):
    # Thermal state with mean photon number n_bar
    n_bar = max(n_bar, 0.0)
    rho_th = np.zeros((N, N), dtype=complex)
    for n in range(N):
        rho_th[n, n] = (n_bar**n) / ((1 + n_bar)**(n+1)) if n_bar > 0 else (1.0 if n == 0 else 0.0)
    return rho_th / np.trace(rho_th)

def reference_state(state_type, x0=0.0, p0=0.0, alpha=1.0, n_th=0.1, r=0.4, phi=0.0):
    # Create the probe/reference state rho_in, before the unknown squeezing
    vac = np.zeros((N, N), dtype=complex)
    vac[0, 0] = 1.0

    if state_type == 'vacuum':
        rho = vac
    elif state_type == 'coherent':
        D = displace_op(alpha)
        rho = D @ vac @ D.conj().T
    elif state_type == 'thermal':
        rho = thermal_state(n_th)
    elif state_type == 'squeezed_vacuum':
        S = static_squeeze_op(r, phi)
        rho = S @ vac @ S.conj().T
    elif state_type == 'squeezed_thermal':
        S = static_squeeze_op(r, phi)
        rho = S @ thermal_state(n_th) @ S.conj().T
    else:
        raise ValueError(f"Unknown state type: {state_type}")

    # Apply an additional known displacement if x0 or p0 is non-zero
    if abs(x0) > 1e-10 or abs(p0) > 1e-10:
        D = displace_op((x0 + 1j * p0) / np.sqrt(2))
        rho = D @ rho @ D.conj().T

    rho = 0.5 * (rho + rho.conj().T)
    return rho / np.trace(rho)

def get_prior(theta_grid, prior_type, theta0, theta_scale):
    # Priors on the squeezing grid, with theta_scale from theta_scale_from_variance
    if prior_type == 'gaussian':
        prior_unnorm = np.exp(-0.5 * ((theta_grid - theta0) / theta_scale)**2)
    elif prior_type == 'two_gaussian':
        prior_unnorm = (np.exp(-0.5 * ((theta_grid - theta0) / theta_scale)**2)
                        + 2*np.exp(-0.5 * ((theta_grid - 2*theta0) / theta_scale)**2))
    elif prior_type == 'uniform':
        prior_unnorm = np.zeros_like(theta_grid)
        idx = (theta_grid >= theta0 - theta_scale) & (theta_grid <= theta0 + theta_scale)
        prior_unnorm[idx] = 1.0
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

    dtheta = theta_grid[1] - theta_grid[0]
    total = np.sum(prior_unnorm) * dtheta
    if total <= 1e-100:
        raise ValueError(f"Prior normalisation failed for {prior_type}")
    return prior_unnorm / total

def HS(Aop, Bop):
    # Hilbert-Schmidt inner product between operators A and B
    return np.real(np.trace(Aop.conj().T @ Bop))

def rho0_inner_product(rho0,Aop,Bop):
    # rho0-weighted inner product between operators A and B
    return 0.5*np.real(np.trace(rho0 @ (Aop @ Bop +Bop @ Aop )))

def get_optimal_coefficients(rho0, rho1, B):
    """
    Solve the Gram system G alpha = b of Eq. (24), with G_ij = 1/2 Tr(rho0 {B_i, B_j}) and b_i = Tr(rhobar B_i).
    """
    m = len(B)
    G = np.zeros((m, m), dtype=float)
    b = np.zeros(m, dtype=float)

    for i in range(m):
        for j in range(i, m):
            #G[i, j] = G[j, i] = 0.5 * HS(B[i], rho0 @ B[j] + B[j] @ rho0)
            G[i, j] = G[j, i] = rho0_inner_product(rho0,B[i],B[j])
        b[i] = HS(B[i], rho1)

    alpha_opt, *_ = la.lstsq(G, b)
    return alpha_opt, G, b

def solve_spm(rho0, rho1, cutoff=1e-13):
    """
    Solve the Lyapunov equation S rho0 + rho0 S = 2 rhobar [Eq. (16)] in the eigenbasis of
    rho0, where it reduces to <i|S|j> = 2 <i|rhobar|j> / (lam_i + lam_j) as in Appendix B.

    rho0 is generically rank-deficient here (for a vacuum probe every rho(theta) is a
    squeezed vacuum, so rho0 is supported entirely on the even-parity sector), and the
    kernel is dropped to give the minimum-norm solution on the support of rho0.
    """
    lam, U = la.eigh(rho0)
    rhobar_eig = U.conj().T @ rho1 @ U
    keep = lam > cutoff * lam.max()
    lam_sum = np.add.outer(lam, lam)
    S_eig = np.where(np.outer(keep, keep),
                     2.0 * rhobar_eig / np.where(lam_sum == 0, 1.0, lam_sum), 0.0)
    S = U @ S_eig @ U.conj().T
    return 0.5 * (S + S.conj().T)

def msl_bayes_for_pvm(S_op, rho0, rho1, lambda_val, theta_mean, tol=1e-10):
    """
    MSL for the PVM given by the spectral decomposition of S_op, post-processed with the
    posterior mean estimator. This is given by

    L = lambda - sum_k Tr(P_k rhobar)^2 / Tr(P_k rho0).

    If S_op is proportional to the identity, the measurement carries no information and its
    eigenvectors are an arbitrary basis, so the prior MSL =lambda - <theta>^2 is returned (rather than whatever eigh gives).
    This case is reached whenever b vanishes on the whole subspace, which for squeezing happens for every linear basis [Eq. (58)].
    """
    dim = S_op.shape[0]
    S_traceless = S_op - (np.trace(S_op) / dim) * np.eye(dim)
    if np.linalg.norm(S_traceless) <= tol * max(np.linalg.norm(S_op), 1.0):
        return lambda_val - theta_mean**2

    _, eigvecs = la.eigh(S_op)
    msl_gain = 0.0
    for k in range(eigvecs.shape[1]):
        ket = eigvecs[:, [k]]
        Pk = ket @ ket.conj().T
        pk = np.real(np.trace(Pk @ rho0))
        mk = np.real(np.trace(Pk @ rho1))
        if pk > 1e-12:
            msl_gain += (mk**2) / pk
    return lambda_val - msl_gain

def msl_homodyne_func(phi, rho0, rho1, lambda_val, theta_mean):
    """
    MSL for homodyne detection at angle phi with the posterior mean estimator.
    Homodyne is the PVM in the eigenbasis of q_phi = q cos(phi) + p sin(phi).

    Note this is not the eigenbasis of S_Vhom, which is a function of q_phi^2 and so has a
    doubly degenerate spectrum. Under truncation eigh splits that degeneracy into parity
    eigenstates ~ (|q> +/- |-q>)/sqrt2, which is a different measurement. 
    (The two coincide whenever the posterior is even in q_phi, e.g. for undisplaced probes)
    """
    q_phi = q * np.cos(phi) + p * np.sin(phi)
    return msl_bayes_for_pvm(q_phi, rho0, rho1, lambda_val, theta_mean)

# ---- Analytic MSL formulas, for a Gaussian prior only ----

def msl_quad_analytic(prior_var, moments):
    # Eqs. 62b and 63: full quadratic subspace, using lambda - mu0^2 = sigma_0^2 and b_Qminus = -4 sigma_0^2
    _, _, Vqq, Vpp, Vqp = moments
    s2 = prior_var
    norm_Qminus = (1 + 2*(3*np.exp(8*s2) - 1)*Vqq*Vpp - 4*Vqp**2) / (np.exp(4*s2)*Vqq*Vpp)
    return s2 - (4*s2)**2 / norm_Qminus

def msl_hom_analytic(prior_var):
    # Eq. 68: V_hom at phi = 0 or pi/2 with Vqp = 0. Independent of the probe state.
    s2 = prior_var
    return s2 - 4*s2**2 / (3*np.exp(4*s2) - 1)

def compute_msl_for_prior_width(prior_var_target):
    """
    Compute, for one prior width:
      - the global optimum L(S) from the Lyapunov equation,
      - constrained optima L(S_V) for the prior / linear / homodyne-quadratic / quadratic
        subspaces of Section IV B,
      - the corresponding posterior mean strategies,
      - diagnostics (achieved prior variance, excluded prior tail, truncation leakage).
    """
    theta_scale = theta_scale_from_variance(prior_var_target, prior_type)
    half = grid_half_width(theta_scale, prior_type, grid_sigmas)

    theta_grid = np.linspace(theta0 - half, theta0 + half, theta_pts)
    dtheta = theta_grid[1] - theta_grid[0]

    prior = get_prior(theta_grid, prior_type, theta0, theta_scale)

    # Prior mean and variance on the grid.
    # For the symmetric priors on the symmetric grid built above, the mean equals theta0 to machine precision.
    # (it is computed rather than assumed so that asymmetric priors (e.g. the two_gaussian) stay correct)
    prior_mean = np.sum(theta_grid * prior * dtheta)
    prior_var = np.sum((theta_grid - prior_mean)**2 * prior * dtheta)

    # Reference state (same for all theta)
    rho_ref = reference_state(ref_state_type, x0=x0, p0=p0, alpha=alpha_coherent,
                              n_th=n_thermal, r=r_squeeze, phi=phi_squeeze)

    # Prior-averaged state moments rho_0 and rhobar  [Eq. (13)]
    rho0 = np.zeros((N, N), dtype=complex)
    rho1 = np.zeros((N, N), dtype=complex)
    for i, theta in enumerate(theta_grid):
        S_th = squeeze_op(theta)
        rho_theta = S_th @ rho_ref @ S_th.conj().T
        rho_theta = 0.5 * (rho_theta + rho_theta.conj().T)
        rho_theta = rho_theta / np.trace(rho_theta)
        rho0 += prior[i] * rho_theta * dtheta
        rho1 += prior[i] * theta * rho_theta * dtheta
    rho0 = 0.5 * (rho0 + rho0.conj().T)
    rho1 = 0.5 * (rho1 + rho1.conj().T)

    lambda_val = np.sum(prior * theta_grid**2 * dtheta)

    # Truncation diagnostics
    leak = float(np.real(np.sum(np.diag(rho0)[-3:])))
    n_mean = float(np.real(np.trace(rho0 @ adag @ a)))

    # ---------------- Exact Bayes S (Fock basis) ---------------
    # rho0 can fail to be full rank when truncated, so the Lyapunov equation is solved on the support of rho0.
    S_bayes = solve_spm(rho0, rho1)

    msl_bayes = lambda_val - np.real(np.trace(rho0 @ (S_bayes @ S_bayes)))

    # ---------------- Operator bases ---------------------------
    # Centred second-order elements. Centring leaves each subspace unchanged but makes the
    # Gram system diagonal, as in Section IV B.
    def centred(M):
        return M - HS(rho0, M) * I

    Dq2 = centred(q @ q)
    Dp2 = centred(p @ p)
    DK = centred(0.5 * (q @ p + p @ q))

    # Homodyne-quadratic subspace V_hom = span(1, q_phi^2) at phi = 0 and pi/2, keeping the better of the two.
    # For Vqp = 0 the two angles give the same MSL, so phi = 0 is preferred on a tie to keep the reported angle stable across widths.
    hom_options = {}
    for name, phi_val, Dq2_phi in (('0', 0.0, Dq2), ('pi/2', np.pi/2, Dp2)):
        B = [I, Dq2_phi]
        a_opt, G_mat, b_vec = get_optimal_coefficients(rho0, rho1, B)
        hom_options[name] = (lambda_val - float(b_vec @ a_opt), phi_val)
    L0, L90 = hom_options['0'][0], hom_options['pi/2'][0]
    hom_name = '0' if L0 <= L90 * (1 + 1e-12) else 'pi/2'
    msl_quad_hom, phi_hom = hom_options[hom_name]

    bases = {
        'prior':  [I],                                  # a priori MSL, no measurement
        'linear': [I, q, p],                            # b vanishes identically [Eq. (58)]
        'quad':   [I, Dq2, Dp2],                        # contains Q_-, so equals V_quad
        'quad_full': [I, q, p, Dq2, DK, Dp2],           # full V_quad of Section IV B
    }

    msl, alpha, S_ops = {}, {}, {}
    for name, B in bases.items():
        B = [0.5 * (M + M.conj().T) for M in B]
        a_opt, G_mat, b_vec = get_optimal_coefficients(rho0, rho1, B)
        msl[name] = lambda_val - float(b_vec @ a_opt)   # L(S_V) = lambda - b.alpha  [Eq. (28)]
        alpha[name] = a_opt
        S_V = sum(a_opt[i] * B[i] for i in range(len(B)))
        S_ops[name] = 0.5 * (S_V + S_V.conj().T)

    # Magnitude of b along the linear and correlation directions. Should be zero for an zero-mean Gaussian state.
    b_linear_max = max(abs(HS(q, rho1)), abs(HS(p, rho1)), abs(HS(DK, rho1)))

    # Excess MSL as the rho0-weighted distance to S [Eq. (36)], and the orthogonality of S - S_V to the subspace.
    S_diff = S_ops['quad'] - S_bayes
    excess_norm = rho0_inner_product(rho0, S_diff, S_diff)
    B_quad = [I, Dq2, Dp2]
    ortho_max = max(abs(rho0_inner_product(rho0, S_diff, Bi)) for Bi in B_quad)
    ortho_max /= max(np.sqrt(rho0_inner_product(rho0, S_bayes, S_bayes)), 1e-300)

    # ---------------- Posterior mean strategies ---------------
    # Quadratic subspace: PVM of S_Vquad (non-Gaussian, scattering states) plus PM estimator
    msl_quad_bayes = msl_bayes_for_pvm(S_ops['quad'], rho0, rho1, lambda_val, prior_mean)

    # Homodyne at the selected angle plus PM estimator: the directly implementable strategy
    msl_homodyne = msl_homodyne_func(phi_hom, rho0, rho1, lambda_val, prior_mean)

    return (prior_var, prior_mean, lambda_val, msl_bayes,
            msl['prior'], msl['linear'], msl_quad_hom, msl['quad'], msl['quad_full'],
            msl_quad_bayes, msl_homodyne, phi_hom, alpha['quad'],
            b_linear_max, excess_norm, ortho_max, leak, n_mean)

######### -------------------------------------------------------------- Main program --------------------------------------------------------------#########


### Note: having a displaced probe: i.e. using a coherent state probe or having non-zero x0,p0 means, the quadratic subspaces will be suboptimal. I intend to add this change soon...

# -------------------------- User parameters --------------------------
N = 180  # Fock truncation

# Reference state parameters
ref_state_type = 'vacuum'     # Options: 'vacuum', 'coherent', 'thermal', 'squeezed_vacuum', or 'squeezed_thermal'
x0, p0 = 0.0, 0.0             # Additional known displacement of the probe
alpha_coherent = 0.1 + 0.5j   # Coherent state amplitude (if coherent)
n_thermal = 0.1               # Thermal photons (if thermal)
r_squeeze = 0.4               # Squeezing parameter (if squeezed probe)
phi_squeeze = 0.0             # Squeezing angle of the probe (0 for q-squeezed)

# Prior settings
prior_type = 'uniform'       # Options: 'gaussian', 'two_gaussian', or 'uniform'
theta0 = 0.0                  # Prior mean/centre for theta
theta_pts = 2000              # Number of grid points for theta
grid_sigmas = 5             # Gaussian priors are represented out to +/- this many sigma

# Prior widths, given directly as the target prior variances sigma_0^2 to sweep.
sigma_pts = 25
prior_var_requested = np.logspace(-3.0, 0.25, sigma_pts)

safety_factor = 5            # Ensures the Fock truncation is enough (5 is mostly safe)

# ---- Feasibility: keep only the widths whose theta grid fits inside the truncation ----
moments = probe_moments(ref_state_type, x0, p0, alpha_coherent, n_thermal, r_squeeze,phi_squeeze)
theta_lo_sup, theta_hi_sup = theta_range_supported(N, moments, safety_factor)
theta_reach = min(theta_hi_sup - theta0, theta0 - theta_lo_sup)
if theta_reach <= 0:
    raise ValueError(f"theta0 = {theta0} lies outside the supported range "
                     f"[{theta_lo_sup:.2f}, {theta_hi_sup:.2f}]")

# Grid half-width scales as sqrt(sigma_0^2), so one evaluation fixes the proportionality
unit_half = grid_half_width(theta_scale_from_variance(1.0, prior_type), prior_type, grid_sigmas)
prior_var_max = (theta_reach / unit_half)**2

feasible = prior_var_requested <= prior_var_max
prior_var_values = prior_var_requested[feasible]
N_for_widest = int(np.ceil(safety_factor * photon_number_at(abs(theta0) + unit_half*np.sqrt(prior_var_requested[-1]), moments)))
if not feasible.any():
    raise ValueError(
        f"No requested prior width fits at N = {N} (largest supported {sigma2_unicode} = "
        f"{prior_var_max:.3g}). Lower the width range, raise N (N >~ {N_for_widest} covers "
        f"the full range requested)."
    )

# Ladder operators in truncated Fock basis
a = np.zeros((N, N), dtype=complex)
for n in range(1, N):
    a[n-1, n] = np.sqrt(n)
adag = a.conj().T
I = np.eye(N, dtype=complex)

# Quadratures
q = (a + adag) / np.sqrt(2)
p = (a - adag) / (1j * np.sqrt(2))

# Cached spectral decomposition of the squeezing generator, used by squeeze_op
sq_eigvals, sq_eigvecs = la.eigh(1j * 0.5 * (a @ a - adag @ adag))

if __name__ == '__main__':
    print("=" * 70)
    print(f"Squeezing estimation  |  ref state: {ref_state_type}")
    if ref_state_type == 'coherent':
        print(f"  {alpha_unicode} = {alpha_coherent}")
    elif ref_state_type == 'thermal':
        print(f"  {nbar_unicode} = {n_thermal}")
    elif ref_state_type in ('squeezed_vacuum', 'squeezed_thermal'):
        print(f"  r = {r_squeeze}, {phi_unicode} = {phi_squeeze}")

    print(f"Prior type: {prior_type}   |   prior centre: {theta_unicode} = {theta0}")

    print(f"Fock truncation N = {N} supports {theta_unicode} in "
          f"[{theta_lo_sup:.2f}, {theta_hi_sup:.2f}], i.e. {sigma2_unicode} up to "
          f"{prior_var_max:.3g}")

    if not feasible.all():
        print(f"  Note: dropped {int((~feasible).sum())} width(s) above {sigma2_unicode} = "
                f"{prior_var_max:.3g}. N >~ {N_for_widest} would cover the full range requested.")
    print(f"Sweeping {len(prior_var_values)} widths, {sigma2_unicode} in "
          f"[{prior_var_values[0]:.3g}, {prior_var_values[-1]:.3g}]")
    print("=" * 70)

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(compute_msl_for_prior_width, prior_var_values),
            total=len(prior_var_values),
            desc="Computing MSL",
            unit="sigma"
        ))

    keys = ['prior_variance', 'prior_mean', 'lambda', 'msl_bayes',
            'msl_prior', 'msl_linear', 'msl_quad_hom', 'msl_quad', 'msl_quad_full',
            'msl_quad_bayes', 'msl_homodyne', 'phi_hom', 'alpha_quad',
            'b_linear_max', 'excess_norm', 'ortho_max', 'trunc_leak', 'n_mean']

    data = {k: [] for k in keys}
    for res in results:
        for k, v in zip(keys, res):
            data[k].append(v)
    arrays = {k: np.array(v) for k, v in data.items()}
    arrays['prior_variance_target'] = np.array(prior_var_values)

    # ---------------- Consistency checks ----------------
    prior_var = arrays['prior_variance']
    print("\nConsistency checks")
    print("-" * 70)

    print(f"  max |prior mean - {theta_unicode}0|                         = "
          f"{np.max(np.abs(arrays['prior_mean'] - theta0)):.2e}")
    print(f"  max |{sigma2_unicode} achieved / {sigma2_unicode} target - 1|             = "
          f"{np.max(np.abs(prior_var/prior_var_values - 1)):.2e}")

    # b vanishes along every linear direction, and along the correlation {q,p}/2 [Eq. (58)]
    print(f"  max |b| on linear and {{q,p}}/2 directions     = "
          f"{np.max(arrays['b_linear_max']):.2e}   (Eq. 58)")
    print(f"  max |L(linear)/L(prior) - 1|                 = "
          f"{np.max(np.abs(arrays['msl_linear']/arrays['msl_prior'] - 1)):.2e}   (Eq. 58)")

    # span(1, q^2, p^2) already contains Q_-, so it attains the full quadratic optimum
    print(f"  max |L(quad)/L(quad_full) - 1|               = "
          f"{np.max(np.abs(arrays['msl_quad']/arrays['msl_quad_full'] - 1)):.2e}")

    # Excess MSL equals the squared rho0-distance to S [Eq. (36)], and S - S_V is orthogonal
    # to the subspace [Eq. (C9)]
    print(f"  max |[L(quad) - L(S)]/||S_V - S||^2 - 1|     = "
          f"{np.max(np.abs((arrays['msl_quad'] - arrays['msl_bayes'])/arrays['excess_norm'] - 1)):.2e}   (Eq. 36)")
    print(f"  max <S - S_V, B_i> / ||S||                   = "
          f"{np.max(arrays['ortho_max']):.2e}   (Eq. C9)")

    # Ordering of Eq. (40), together with V_hom being a subspace of V_quad
    atol = 1e-7 * np.abs(arrays['msl_bayes'])
    ord_ok = (np.all(arrays['msl_quad_hom'] >= arrays['msl_quad'] - atol)
              and np.all(arrays['msl_quad'] >= arrays['msl_quad_bayes'] - atol)
              and np.all(arrays['msl_quad_bayes'] >= arrays['msl_bayes'] - atol)
              and np.all(arrays['msl_quad_hom'] >= arrays['msl_homodyne'] - atol))
    print(f"  ordering L(hom) >= L(quad) >= L(quad,PM) >= L(S): {ord_ok}")

    print(f"  homodyne angles selected                     = "
          f"{sorted(set(np.round(arrays['phi_hom'], 6)))}")

    # For a Gaussian prior, compare against the closed-form MSLs
    if prior_type == 'gaussian':
        hom_ana = msl_hom_analytic(prior_var)
        quad_ana = msl_quad_analytic(prior_var, moments)
        arrays['msl_hom_analytic'] = hom_ana
        arrays['msl_quad_analytic'] = quad_ana
        print(f"  max |L(V_hom)/Eq.68 - 1|                     = "
              f"{np.max(np.abs(arrays['msl_quad_hom']/hom_ana - 1)):.2e}")
        print(f"  max |L(V_quad)/Eq.62b - 1|                   = "
              f"{np.max(np.abs(arrays['msl_quad']/quad_ana - 1)):.2e}")

    print(f"  max Fock-truncation leakage (top 3 levels)   = {np.max(arrays['trunc_leak']):.2e}")
    print(f"  max <n> under rho0                           = {np.max(arrays['n_mean']):.2f}  (N = {N})")

    # ---------------- Save ----------------
    out_dir = Path(__file__).parent / 'data'
    out_dir.mkdir(exist_ok=True)
    stem = f'squeezing_{ref_state_type}_{prior_type}'

    np.savez(out_dir / f'{stem}.npz', **arrays)
    print(f"\nSaved arrays  \u2192 data/{stem}.npz")

    metadata = dict(
        task='squeezing',
        ref_state_type=ref_state_type,
        alpha_coherent_re=float(np.real(alpha_coherent)),
        alpha_coherent_im=float(np.imag(alpha_coherent)),
        n_thermal=n_thermal,
        r_squeeze=r_squeeze,
        phi_squeeze=phi_squeeze,
        x0=x0,
        p0=p0,
        theta0=theta0,
        prior_type=prior_type,
        N=N,
        theta_pts=theta_pts,
        grid_sigmas=grid_sigmas,
        safety_factor=safety_factor,
        theta_lo_supported=float(theta_lo_sup),
        theta_hi_supported=float(theta_hi_sup),
        prior_var_max_supported=float(prior_var_max),
        prior_var_min_used=float(prior_var_values[0]),
        prior_var_max_used=float(prior_var_values[-1]),
        sigma_pts=int(len(prior_var_values)),
        Vqq=float(moments[2]),
        Vpp=float(moments[3]),
    )
    with open(out_dir / f'{stem}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata \u2192 data/{stem}.json")