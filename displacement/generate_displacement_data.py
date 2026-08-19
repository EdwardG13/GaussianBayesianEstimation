"""
Computes MSL vs prior width for displacement estimation across constrained bases
and the Bayes-optimal bound, then saves everything to disk.

The encoded state is
    rho(theta) = D_x(theta) rho_in D_x^dagger(theta),   D_x(theta) = exp(-i theta p),
so theta is a location parameter and f(theta) = theta (square loss).

Companion to plot_displacement.py, which reads the files written here.

Outputs:
    data/displacement_{ref_state_type}_{prior_type}.npz   (all numerical arrays)
    data/displacement_{ref_state_type}_{prior_type}.json  (scalar/string metadata)
"""

import json
from pathlib import Path
import numpy as np
import scipy.linalg as la
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

def probe_photon_number(ref_state_type, alpha=1.0, n_th=0.2, r=0.4):
    # Mean photon number of the probe state, before the unknown displacement
    if ref_state_type == 'vacuum':
        return 0.0
    if ref_state_type == 'coherent':
        return abs(alpha)**2
    if ref_state_type == 'thermal':
        return n_th
    if ref_state_type == 'squeezed_vacuum':
        return np.sinh(r)**2
    if ref_state_type == 'squeezed_thermal':
        return (n_th + 0.5) * np.cosh(2*r) - 0.5
    raise ValueError(f"Unknown state type: {ref_state_type}")

def probe_covariance(ref_state_type, n_th=0.2, r=0.4):
    """
    Diagonal covariance elements (Vxx, Vpp) of the probe. Only used for the check of the analytic
    MSL formula Eq.52 (here displacement leaves the covariance matrix unchanged, so the probe's first moments never enter).
    """
    if ref_state_type in ('vacuum', 'coherent'):
        return 0.5, 0.5
    if ref_state_type == 'thermal':
        return n_th + 0.5, n_th + 0.5
    if ref_state_type == 'squeezed_vacuum':
        return 0.5*np.exp(-2*r), 0.5*np.exp(2*r)
    if ref_state_type == 'squeezed_thermal':
        return (n_th + 0.5)*np.exp(-2*r), (n_th + 0.5)*np.exp(2*r)
    raise ValueError(f"Unknown state type: {ref_state_type}")

def theta_max_supported(N, ref_state_type, alpha, n_th, r, safety_factor):
    """
    Largest |theta| the Fock truncation supports.

    A probe displaced along x has <n>(theta) = n0 + theta^2/2. For truncation N the largest
    mean photon number we can represent is <n>_max ~ N/safety_factor, which inverts to
    theta_max = sqrt(2 (N/safety_factor - n0)).
    """
    n_budget = N / safety_factor - probe_photon_number(ref_state_type, alpha, n_th, r)
    if n_budget < 1:
        raise ValueError("Insufficient Fock truncation for this probe")
    return np.sqrt(2 * n_budget)

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
    """
    How far from theta0 the grid must reach for this prior to be faithfully represented.

    Composed with theta_scale_from_variance, half(sigma_0^2) = c * sqrt(sigma_0^2), where c=grid_sigmas for Gaussians (i.e. 3-5 prior widths in grid) or sqrt(3) for uniform.

    The idea is that theta_max-theta_0 >= c * sqrt(sigma_0^2_max) for some constant c which depends on the prior type. This allows us to invert for sigma_0
    """
    return theta_scale if prior_type == 'uniform' else grid_sigmas * theta_scale

def squeeze_op(r, phi):
    # Squeezing operator
    G = 0.5 * (np.exp(-2j*phi) * a @ a - np.exp(2j*phi) * adag @ adag)
    return la.expm(r * G)

def displace_op(alpha):
    # Displacement operator D(alpha) = exp(alpha a^dagger - alpha* a)
    return la.expm(alpha * adag - np.conj(alpha) * a)

def displacement_x(theta):
    """
    Displacement along x: D_x(theta) = exp(-i theta p), which shifts x -> x + theta.

    We can simplfy the matrix exponential by using the eigendecomposition of the momentum operator.

    """
    return (p_eigvecs * np.exp(-1j * theta * p_eigvals)) @ p_eigvecs.conj().T

def thermal_state(n_bar):
    # Thermal state with mean photon number n_bar
    n_bar = max(n_bar, 0.0)
    rho_th = np.zeros((N, N), dtype=complex)
    for n in range(N):
        rho_th[n, n] = (n_bar**n) / ((1 + n_bar)**(n+1)) if n_bar > 0 else (1.0 if n == 0 else 0.0)
    return rho_th / np.trace(rho_th)

def reference_state(state_type, x0=0.0, p0=0.0, alpha=1.0, n_th=0.2, r=0.4, phi=0.0):
    # Create the probe/reference state rho_in, before the unknown displacement
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
        S = squeeze_op(r, phi)
        rho = S @ vac @ S.conj().T
    elif state_type == 'squeezed_thermal':
        S = squeeze_op(r, phi)
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
    # Priors on the displacement grid, with theta_scale from theta_scale_from_variance
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
    Solve the Gram system G alpha = b of Eq. (24), with
        G_ij = Tr(rho0 {B_i, B_j}) / 2,   b_i = Tr(rhobar B_i).
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

def msl_homodyne_func(phi_homodyne, rho0, rho1, lambda_val, theta_mean):
    """
    MSL for homodyne detection at angle phi with the posterior mean estimator.
    Homodyne is the PVM in the eigenbasis of x_phi = x cos(phi) + p sin(phi).
    """
    x_phi = x * np.cos(phi_homodyne) + p * np.sin(phi_homodyne)
    return msl_bayes_for_pvm(x_phi, rho0, rho1, lambda_val, theta_mean)

def compute_msl_for_prior_width(prior_var_target):
    """
    Compute, for one prior width:
      - the global optimum L(S) from the Lyapunov equation,
      - constrained optima L(S_V) for the prior / linear / quadratic / cubic / quintic
        subspaces in the centred quadrature Delta x = x - <x>_rho0,
      - the same measurements post-processed with the posterior mean estimator,
      - x-homodyne with the posterior mean estimator,
      - diagnostics (achieved prior variance, truncation leakage, Gram conditioning).

    Note the prior grid is created fresh for every target prior variance, since it is unfeasable for the same grid to serve large and small prior variances.
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
        D_x = displacement_x(theta)
        rho_theta = D_x @ rho_ref @ D_x.conj().T
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
    # rho0 can failto be full rank when truncated, so the Lyapunov equation is solved on the support of rho0.
    S_bayes = solve_spm(rho0, rho1)

    # lyapunov_lhs = np.kron(np.eye(N), rho0) + np.kron(rho0.T, np.eye(N))
    # vecrho1 = rho1.reshape(N*N, order='F')
    # S_bayes = (la.pinv(lyapunov_lhs) @ (2.0 * vecrho1)).reshape((N, N), order='F')
    # S_bayes = 0.5 * (S_bayes + S_bayes.conj().T)
    

    msl_bayes = lambda_val - np.real(np.trace(rho0 @ (S_bayes @ S_bayes)))

    # ---------------- Operator bases ---------------------------
    # Centred quadrature Delta x = x - <x>_rho0, as used in Section IV A.
    Dx = x - HS(rho0, x) * I

    bases = {
        'prior':   [I],                                    # a priori MSL, no measurement
        'linear':  [I, Dx],                                # x-homodyne, linear estimator
        'quad':    [I, Dx, Dx @ Dx],                       # even order adds nothing [Eq. (54)]
        'cubic':   [I, Dx, Dx @ Dx @ Dx],                  # x-homodyne, cubic estimator
        'quintic': [I, Dx, Dx @ Dx @ Dx, Dx @ Dx @ Dx @ Dx @ Dx],
    }

    msl, alpha, cond, S_ops = {}, {}, {}, {}
    for name, B in bases.items():
        B = [0.5 * (M + M.conj().T) for M in B]
        a_opt, G_mat, b_vec = get_optimal_coefficients(rho0, rho1, B)
        msl[name] = lambda_val - float(b_vec @ a_opt)      # L(S_V) = lambda - b.alpha  [Eq. (28)]
        alpha[name] = a_opt
        cond[name] = float(np.linalg.cond(G_mat))
        S_V = sum(a_opt[i] * B[i] for i in range(len(B)))
        S_ops[name] = 0.5 * (S_V + S_V.conj().T)

    # ---------------- Constrained PVM + posterior mean estimator ---------------
    msl_linear_bayes = msl_bayes_for_pvm(S_ops['linear'], rho0, rho1, lambda_val, prior_mean)
    msl_cubic_bayes = msl_bayes_for_pvm(S_ops['cubic'], rho0, rho1, lambda_val, prior_mean)

    # x-homodyne with the posterior mean estimator (the all-orders limit of the above)
    msl_homodyne = msl_homodyne_func(phi_homodyne, rho0, rho1, lambda_val, prior_mean)

    return (prior_var, prior_mean, msl_bayes,
            msl['prior'], msl['linear'], msl['quad'], msl['cubic'], msl['quintic'],
            msl_linear_bayes, msl_cubic_bayes, msl_homodyne,
            alpha['linear'], alpha['cubic'], cond['quintic'], leak, n_mean)

######### -------------------------------------------------------------- Main program --------------------------------------------------------------#########

# -------------------------- User parameters --------------------------
N = 40  # Fock truncation

# Reference state parameters
ref_state_type = 'vacuum'   # Options: 'vacuum', 'coherent', 'thermal', 'squeezed_vacuum', or 'squeezed_thermal'
x0, p0 = 0.0, 0.0             # Additional known displacement of the probe
alpha_coherent = 0.5 + 0.5j   # Coherent state amplitude (if coherent)
n_thermal = 0.2               # Thermal photons (if thermal)
r_squeeze = 0.4               # Squeezing parameter (if squeezed)
phi_squeeze = 0.0             # Squeezing angle (0 for x-squeezed)

# Prior settings
prior_type = 'uniform'        # Options: 'gaussian', 'two_gaussian', or 'uniform'
theta0 = 0.0                  # Prior mean for theta
theta_pts = 2000              # Number of grid points for theta
grid_sigmas = 5.0             # Gaussian priors are represented out to +/- this many sigma

phi_homodyne = 0.0            # Homodyne angle: 0 and pi/2 are the x and p quadratures

# Prior variances sigma_0^2. Any variance the Fock truncation cannot support are later dropped, with a notice saying what N would allow.
sigma_pts = 25
prior_var_requested = np.logspace(-2.5, 1, sigma_pts)

safety_factor = 10             # Ensures the Fock truncation is enough (I found 5-10 to be safe)

# ---- Feasibility: keep only the widths whose theta grid fits inside the truncation ----
theta_max = theta_max_supported(N, ref_state_type, alpha_coherent, n_thermal, r_squeeze, safety_factor) # This is largest theta (unknown displacement) allowed within the truncation (given the safety factor)
theta_reach = theta_max - abs(theta0) # Distance from prior mean
if theta_reach <= 0:
    raise ValueError(f"theta0 = {theta0} already exceeds the supported range {theta_max:.2f}")

"""
The largest prior variance is calculated from the theta grid by theta_reach >= c * sqrt(sigma_0^2_max) for some constant c which depends on the prior type.
This allows us to invert for sigma_0. Here, grid_half_width(sigma_0)=c * sqrt(sigma_0), where c=grid_sigmas for Gaussians (i.e. 3-5 prior widths in grid) or sqrt(3) for uniform.
c (called unit_half here) can then be calculated by grid_half_width(1).
"""
unit_half = grid_half_width(theta_scale_from_variance(1.0, prior_type), prior_type, grid_sigmas) # 
prior_var_max = (theta_reach / unit_half)**2

feasible = prior_var_requested <= prior_var_max
prior_var_values = prior_var_requested[feasible]
n0_probe = probe_photon_number(ref_state_type, alpha_coherent, n_thermal, r_squeeze)
N_for_widest = int(np.ceil(safety_factor *(n0_probe + (unit_half*np.sqrt(prior_var_requested[-1]))**2 / 2)))
if not feasible.any():
    raise ValueError(
        f"No requested prior width fits at N = {N} (largest supported {sigma2_unicode} = "
        f"{prior_var_max:.3g}). Lower the width range, or raise N (N >~ {N_for_widest} "
        f"covers the full range requested)."
    )

# Ladder operators in truncated Fock basis
a = np.zeros((N, N), dtype=complex)
for n in range(1, N):
    a[n-1, n] = np.sqrt(n)
adag = a.conj().T
I = np.eye(N, dtype=complex)

# Quadratures
x = (a + adag) / np.sqrt(2)
p = (a - adag) / (1j * np.sqrt(2))

# Cached spectral decomposition of p, used by displacement_x
p_eigvals, p_eigvecs = la.eigh(p)

if __name__ == '__main__':
    print("=" * 70)
    print(f"Displacement estimation along x  |  ref state: {ref_state_type}")
    if ref_state_type == 'coherent':
        print(f"  {alpha_unicode} = {alpha_coherent}")
    elif ref_state_type == 'thermal':
        print(f"  {nbar_unicode} = {n_thermal}")
    elif ref_state_type in ('squeezed_vacuum', 'squeezed_thermal'):
        print(f"  r = {r_squeeze}, {phi_unicode} = {phi_squeeze}")

    print(f"Prior type: {prior_type}   |   prior centre: {theta_unicode} = {theta0}")

    print(f"Fock truncation N = {N} supports |{theta_unicode}| up to {theta_max:.2f}, "
          f"i.e. {sigma2_unicode} up to {prior_var_max:.3g}")
    
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

    keys = ['prior_variance', 'prior_mean', 'msl_bayes',
            'msl_prior', 'msl_linear', 'msl_quad', 'msl_cubic', 'msl_quintic',
            'msl_linear_bayes', 'msl_cubic_bayes', 'msl_homodyne',
            'alpha_linear', 'alpha_cubic', 'cond_quintic', 'trunc_leak', 'n_mean']

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

    # Even-order basis elements do not contribute for a prior symmetric about theta0 [Eq. (54)]
    print(f"  max |L(quad)/L(linear) - 1|                   = "
          f"{np.max(np.abs(arrays['msl_quad']/arrays['msl_linear'] - 1)):.2e}   (Eq. 54)")

    # The PVM of S_linear is x-homodyne, so the two PM curves must agree
    print(f"  max |L(linear PVM, PM)/L(homodyne, PM) - 1|   = "
          f"{np.max(np.abs(arrays['msl_linear_bayes']/arrays['msl_homodyne'] - 1)):.2e}")

    # Ordering of Eq. (40): L(linear) >= L(cubic) >= L(PM) >= L(S), up to quadrature noise
    atol = 1e-7 * np.abs(arrays['msl_bayes'])
    ord_ok = (np.all(arrays['msl_linear'] >= arrays['msl_cubic'] - atol)
              and np.all(arrays['msl_cubic'] >= arrays['msl_cubic_bayes'] - atol)
              and np.all(arrays['msl_cubic_bayes'] >= arrays['msl_bayes'] - atol))
    print(f"  ordering L(linear) >= L(cubic) >= L(PM) >= L(S): {ord_ok}")

    # For a Gaussian prior, L(S) = (1/Vxx + 1/sigma_0^2)^-1 exactly [Eq. (52)], and Theorem 1
    # makes the linear subspace exactly optimal, so L_R(linear) should be numerical zero.
    if prior_type == 'gaussian':
        Vxx, _ = probe_covariance(ref_state_type, n_thermal, r_squeeze)
        msl_analytic = 1.0 / (1.0/Vxx + 1.0/prior_var)
        print(f"  max |L(S)/Eq.(52) - 1|                        = "
              f"{np.max(np.abs(arrays['msl_bayes']/msl_analytic - 1)):.2e}")
        print(f"  max |L(linear)/L(S) - 1|                      = "
              f"{np.max(np.abs(arrays['msl_linear']/arrays['msl_bayes'] - 1)):.2e}   (Theorem 1)")
        arrays['msl_bayes_analytic'] = msl_analytic

    print(f"  max Fock-truncation leakage (top 3 levels)    = {np.max(arrays['trunc_leak']):.2e}")
    print(f"  max <n> under rho0                            = {np.max(arrays['n_mean']):.2f}  (N = {N})")
    print(f"  max Gram condition number (quintic basis)     = {np.max(arrays['cond_quintic']):.2e}")

    # ---------------- Save ----------------
    out_dir = Path(__file__).parent / 'data'
    out_dir.mkdir(exist_ok=True)
    stem = f'displacement_{ref_state_type}_{prior_type}'

    np.savez(out_dir / f'{stem}.npz', **arrays)
    print(f"\nSaved arrays  \u2192 data/{stem}.npz")

    metadata = dict(
        task='displacement',
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
        phi_homodyne=phi_homodyne,
        N=N,
        theta_pts=theta_pts,
        grid_sigmas=grid_sigmas,
        safety_factor=safety_factor,
        theta_max_supported=float(theta_max),
        prior_var_max_supported=float(prior_var_max),
        prior_var_min_used=float(prior_var_values[0]),
        prior_var_max_used=float(prior_var_values[-1]),
        sigma_pts=int(len(prior_var_values)),
    )
    with open(out_dir / f'{stem}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata \u2192 data/{stem}.json")