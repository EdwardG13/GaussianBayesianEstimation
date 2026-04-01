"""
Computes MSL vs prior width for squeezing estimation across constrained bases
and the Bayes-optimal bound, then saves everything to disk.

Outputs:
data/squeezing_{ref_state_type}.npz   (all numerical arrays)
data/squeezing_{ref_state_type}.json  (scalar/string metadata)
"""

import os
import math
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

def design_parameters(N, ref_state_type, alpha=1.0, n_th=0.2, r=0.4, theta0=0.5, sigma_pts=10,safety_factor=3):
    """
    Parameter designer to ensure Fock truncation and prior range is sufficient.

    A squeezed reference state has <n>(theta) which a function depending on the reference state.
    For a given Fock truncation, the maximum average photon number we can resolve is <n>_max~N/safety factor.
    One can then invert for the largest grid size theta_max allowed.
    """
    
    # Initial photon number depending on probe state
    if ref_state_type == 'vacuum':
        n0 = 0
    elif ref_state_type == 'coherent':
        n0 = abs(alpha)**2
    elif ref_state_type == 'thermal':
        n0 = n_th
    elif ref_state_type == 'squeezed_vacuum':
        n0 = np.sinh(r)**2
    elif ref_state_type == 'squeezed_thermal':
        n0 = (n_th + 0.5) * np.cosh(2*r) - 0.5
    
    # Photon budget
    n_budget = N / safety_factor - n0
    
    if n_budget < 1:
        raise ValueError("Insufficient Fock truncation")
    
    # Reference state-specific limit
    if ref_state_type =='vacuum':
        # Here <n>(theta_max) = sinh^2(theta_max) ~ e^(2 theta_max)/4
        theta_max = 0.5 * np.log(4 * n_budget)
        #theta_max = np.sqrt(np.arcsinh(n_budget))
    elif ref_state_type == 'thermal':
        # <n>(theta) = (n_th + 1/2)cosh(2 theta) - 1/2 ~ 1/2(n_th + 1/2)e^(2 theta)
        theta_max = 0.5 * np.log(2*n_budget / (n_th + 0.5))
    elif ref_state_type == 'coherent':
        # <n>_max ~ max(|Re(alpha)|^2,|Im(alpha)|^2) e^(2 theta_max)
        alpha_max = max(abs(np.real(alpha)), abs(np.imag(alpha)))
        A = max(alpha_max**2, 1 / 4)  # vacuum term fallback
        theta_max = 0.5 * np.log(n_budget / A)
            
    elif ref_state_type == 'squeezed_vacuum':
        # <n>(theta) = sinh^2(r+theta) ~ 1/4 e^2(r+theta)
        theta_max = 0.5 * np.log(4 * n_budget) - r
            
    elif ref_state_type == 'squeezed_thermal':
        # <n>(theta) = (n_th + 1/2)cosh(2 (r+theta)) - 1/2 ~ 1/2(n_th + 1/2)e^(2(r+theta))
        theta_max = 0.5 * np.log(2 * n_budget / (n_th + 0.5)) - r

    theta_max = 0.5 * np.log(4 * n_budget) # fix this for the plots to have the same axes
    
    # Prior and grid
    sigma_max = 2*(theta_max - abs(theta0)) / 3 # The maximum standard deviation of a Gaussian prior that ensures the prior doesn't put significant weight beyond theta_max.
    theta_min = 2*theta0 -  theta_max

    dtheta = (theta_max - theta_min) / theta_pts
    sigma_min = 10*dtheta  # prior must cover at least 10 grid spacings

    #theta_sigma_values = np.logspace(np.log10(sigma_min), np.log10(sigma_max), sigma_pts) # Create a prior grid with sigma_pts
    theta_sigma_values = np.logspace(-1.2, 0.5, sigma_pts) # Fixed grid for now
    
    return {
        'theta_min': theta_min,
        'theta_max': theta_max,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'theta_sigma_values': theta_sigma_values,

    }

def squeeze_op(r, phi):
    # Squeezing operator 
    G = 0.5 * (np.exp(-2j*phi) * a @ a - np.exp(2j*phi) * adag @ adag)
    return la.expm(r * G)

def displace_op(alpha):
    # Displacement operator D(alpha) = exp(alpha a^dagger - alpha* a)
    return la.expm(alpha * adag - np.conj(alpha) * a)

def thermal_state(n_bar):
    # Thermal state with mean photon number n_bar
    if n_bar < 0:
        n_bar = 0

    rho_th = np.zeros((N, N), dtype=complex)
    for n in range(N):
        if n_bar > 0:
            rho_th[n, n] = (n_bar**n) / ((1 + n_bar)**(n+1))
        else:
            rho_th[n, n] = 1.0 if n == 0 else 0.0
    return rho_th

def reference_state(state_type, x0=0.0, p0=0.0, alpha=1.0, n_th=0.2, r=0.8, phi=0.0):
    # Create probe/reference state rho (before displacement)
    if state_type == 'vacuum':
        vac = np.zeros((N, N), dtype=complex)
        vac[0, 0] = 1.0
        rho = vac
    elif state_type == 'coherent':
        D = displace_op(alpha)
        vac = np.zeros((N, N), dtype=complex)
        vac[0, 0] = 1.0
        rho = D @ vac @ D.conj().T
    elif state_type == 'thermal':
        rho = thermal_state(n_th)
    elif state_type == 'squeezed_vacuum':
        vac = np.zeros((N, N), dtype=complex)
        vac[0, 0] = 1.0
        S = squeeze_op(r, phi)
        rho = S @ vac @ S.conj().T
    elif state_type == 'squeezed_thermal':
        rho = thermal_state(n_th)
        S = squeeze_op(r, phi)
        rho = S @ rho @ S.conj().T
    else:
        raise ValueError(f"Unknown state type: {state_type}")
    
    # Apply displacement if x0 or p0 is non-zero
    if abs(x0) > 1e-10 or abs(p0) > 1e-10:
        alpha_disp = (x0 + 1j * p0) / np.sqrt(2)
        D = displace_op(alpha_disp)
        rho = D @ rho @ D.conj().T
    
    rho = 0.5 * (rho + rho.conj().T)
    rho = rho / np.trace(rho)
    return rho

def get_prior(theta_grid, prior_type, theta0, theta_sigma, theta_min, theta_max):
    # Generate different types of priors on the displacement grid
    if prior_type == 'gaussian':
        prior_unnorm = np.exp(-0.5 * ((theta_grid - theta0) / theta_sigma)**2)

    elif prior_type == 'two_gaussian':
        prior_unnorm = np.exp(-0.5 * ((theta_grid - theta0) / theta_sigma)**2) + 2*np.exp(-0.5 * ((theta_grid - 2*theta0) / theta_sigma)**2)
    elif prior_type == 'uniform':
        prior_unnorm = np.zeros_like(theta_grid)
        # find indices within the uniform window
        idx = (theta_grid >= theta0 - theta_sigma) & (theta_grid <= theta0 + theta_sigma)
        prior_unnorm[idx] = 1
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")
    
    dtheta = theta_grid[1] - theta_grid[0]
    total = np.sum(prior_unnorm) * dtheta
    if total > 1e-100:
        prior = prior_unnorm / total
    else:
        print(f"Warning: Prior normalization failed for {prior_type}, using uniform")
        prior = np.ones_like(theta_grid) / (theta_max - theta_min)
    
    return prior

def get_prior_variance(theta_grid, prior):
    # Compute the variance of the prior distribution
    dtheta = theta_grid[1] - theta_grid[0]
    mean = np.sum(theta_grid * prior * dtheta)
    variance = np.sum((theta_grid - mean)**2 * prior * dtheta)
    return variance

def HS(Aop, Bop):
        # Define the Hilbert-Schmidt inner product between operators A and B
        return np.real(np.trace(Aop.conj().T @ Bop))

def get_optimal_coefficients(rho0, rho1, B):
    # Compute optimal coefficients alpha^opt for constrained basis
    
    m = len(B)
    G = np.zeros((m, m), dtype=float)
    b = np.zeros(m, dtype=float)
    
    for i in range(m):
        for j in range(m):
            G[i, j] = 0.5 * HS(B[i], rho0 @ B[j] + B[j] @ rho0)
        b[i] = HS(B[i], rho1)
    
    alpha_opt, *_ = la.lstsq(G, b)
    
    return alpha_opt, G, b

def msl_bayes_for_pvm(S_op, rho0, rho1, lambda_val):
    """
    Compute Bayes MSL for the PVM defined by the spectral decomposition of S_op and the (corresponding optimal) posterior mean estimator.
    """
    # Eigen-decomposition of S_op
    eigvals, eigvecs = la.eigh(S_op)

    msl_gain = 0.0
    for k in range(len(eigvals)):
        ket = eigvecs[:, k].reshape(-1, 1)
        Pk = ket @ ket.conj().T

        pk = np.real(np.trace(Pk @ rho0))
        mk = np.real(np.trace(Pk @ rho1))

        if pk > 1e-10:
            msl_gain += (mk**2) / pk
    
    # If eigenvalues are identical, the eigenvectors returned are arbitrary orthonormal basis vectors. Treat separately.
    if np.linalg.norm(S_op - theta0 * I) < 1e-10: 
        return lambda_val - theta0**2
    else:
        return lambda_val - msl_gain
    
def msl_homodyne_func(phi_homodyne, rho0, rho1, lambda_val):
    """
    Compute MSL for homodyne measurement at angle phi with posterior mean estimator.
    
    Homodyne at angle phi measures the quadrature: x_phi = x*cos(phi) + p*sin(phi)
    """
    # Construct rotated quadrature operator
    x_phi = x * np.cos(phi_homodyne) + p * np.sin(phi_homodyne)
    
    # Homodyne is a PVM in the eigenbasis of x_phi
    return msl_bayes_for_pvm(x_phi, rho0, rho1, lambda_val)

def build_M1_pvm(S_op, rho0, rho1, lambda_val):
    """
    Build M_1 = sum_k theta_hat_k P_k for a PVM defined by eigenbasis of S_op,
    where theta_hat_k is the posterior mean for outcome k.
    """
    eigvals, eigvecs = la.eigh(S_op)

    # Compute normalisation: p_k and m_k = Tr(P_k rho1)
    M1 = np.zeros((N, N), dtype=complex)

    for k in range(len(eigvals)):
        ket = eigvecs[:, k].reshape(-1, 1)
        Pk = ket @ ket.conj().T
        pk = np.real(np.trace(Pk @ rho0))
        mk = np.real(np.trace(Pk @ rho1))

        if pk > 1e-10:
            theta_hat_k = mk / pk   # posterior mean for outcome k
        else:
            theta_hat_k = 0.0       # uninformative outcome

        M1 += theta_hat_k * Pk

    M1 = 0.5 * (M1 + M1.conj().T)
    return M1


def weighted_norm_sq(A, rho0):
    """
    Compute ||A||^2_{rho0} = Tr(rho0 A^2) = HS(A, rho0 @ A + A @ rho0) / 2
    which equals Tr(A rho0 A) for Hermitian A.
    """
    return np.real(np.trace(A @ rho0 @ A))


def relative_msl_via_norm(S_op_basis, S_bayes, rho0, rho1, lambda_val):
    """
    Compute L_R = ||S - M1||^2_{rho0} / L(S)
    where M1 is the PVM+PM operator built from S_op_basis.
    """
    M1 = build_M1_pvm(S_op_basis, rho0, rho1, lambda_val)
    diff = S_bayes - M1
    norm_sq = weighted_norm_sq(diff, rho0)
    msl_bayes = lambda_val - np.real(np.trace(rho0 @ S_bayes @ S_bayes))
    return norm_sq / msl_bayes, M1


def compute_msl_for_prior_width(theta_sigma, theta0=0.0, prior_type='gaussian'):
    """
    Compute MSL for Bayes-optimal, linear homodyne, quadratic homodyne, and quadratic (Fock) basis.
    
    Returns:
    - msl_bayes, msl_linear, msl_quad, msl_quad
    - alpha_linear, alpha_quad, alpha_quad (coefficients)
    - prior_variance
    """
    
    theta_grid = np.linspace(theta_min, theta_max, theta_pts)
    dtheta = theta_grid[1] - theta_grid[0]
    
    prior = get_prior(theta_grid, prior_type, theta0, theta_sigma, theta_min, theta_max)
    prior_var = get_prior_variance(theta_grid, prior)
    
    # Reference state (same for all theta)
    rho_ref = reference_state(ref_state_type, x0=x0, p0=p0, alpha=alpha_coherent,
                              n_th=n_thermal, r=r_squeeze, phi=phi_squeeze)
    
    # Build rho(theta) list - states after squeezing
    rho_list = []
    for theta in theta_grid:
        #D_x = displacement_x(theta)
        #rho_theta = thermal_state_varying(theta)
        #rho_theta = D_x @ rho_ref @ D_x.conj().T
        S = squeeze_op(theta,0)
        rho_theta = S @ rho_ref @ S.conj().T
        rho_theta = 0.5 * (rho_theta + rho_theta.conj().T)
        rho_theta = rho_theta / np.trace(rho_theta)
        rho_list.append(rho_theta)
    
    # Compute rho_0 and rho_1
    rho0 = np.zeros((N, N), dtype=complex)
    rho1 = np.zeros((N, N), dtype=complex)
    for i, theta in enumerate(theta_grid):
        rho0 += prior[i] * rho_list[i] * dtheta
        #rho1 += prior[i] * theta * rho_list[i] * dtheta
        rho1 += prior[i] * theta * rho_list[i] * dtheta
    rho0 = 0.5 * (rho0 + rho0.conj().T)
    rho1 = 0.5 * (rho1 + rho1.conj().T)
    
    lambda_val = np.sum(prior * theta_grid**2 * dtheta)
    
    # ---------------- Exact Bayes S (Fock basis) ---------------
    dim = N * N
    lyapunov_rhs = np.kron(np.eye(N), rho0) + np.kron(rho0.T, np.eye(N))
    vecrho1 = rho1.reshape(dim, order='F')
    vecS_bayes = la.pinv(lyapunov_rhs) @ (2.0 * vecrho1)
    S_bayes = vecS_bayes.reshape((N, N), order='F')
    S_bayes = 0.5 * (S_bayes + S_bayes.conj().T)
    
    msl_bayes = lambda_val - np.real(np.trace(rho0 @ (S_bayes @ S_bayes)))
    
    # ---------------- Linear basis {I, x_phi} -----------------------
    phi=-np.arctan(np.imag(alpha_coherent)/np.real(alpha_coherent)*np.exp(-2*theta0))
    #phi=np.pi/2

    xphi=x*np.cos(phi) +p*np.sin(phi) # Rotated quadrature operator
    B_linear = [I,xphi]
    #B_linear = [I, x, p] # Optimised over phi, the 1,x,p basis has the same MSL.
    B_linear = [0.5 * (M + M.conj().T) for M in B_linear]
    
    alpha_opt_linear, G_mat_linear, b_vec_linear = get_optimal_coefficients(rho0, rho1, B_linear)
    msl_linear = lambda_val - b_vec_linear @ la.pinv(G_mat_linear) @ b_vec_linear

    
    # ---------------- Quadratic homodyne basis {I, x_phi,x_phi^2} -----------------------
    phi=np.pi/2
    xphi=x*np.cos(phi) +p*np.sin(phi)
    B_quad_hom = [I,xphi,xphi@xphi]
    B_quad_hom = [0.5 * (M + M.conj().T) for M in B_quad_hom]

    
    alpha_opt_quad_hom, G_mat_quad_hom, b_vec_quad_hom = get_optimal_coefficients(rho0, rho1, B_quad_hom)
    msl_quad_hom = lambda_val - b_vec_quad_hom @ la.pinv(G_mat_quad_hom) @ b_vec_quad_hom
    
    # ---------------- Quadratic basis -----------------------

    B_quad = [I, x @ x,p@p]
    B_quad = [0.5 * (M + M.conj().T) for M in B_quad]
    
    alpha_opt_quad, G_mat_quad, b_vec_quad = get_optimal_coefficients(rho0, rho1, B_quad)
    msl_quad = lambda_val - b_vec_quad @ la.pinv(G_mat_quad) @ b_vec_quad

    # ---------------- Prior information -----------------------
    B_prior =[I]
    alpha_opt_prior, G_mat_prior, b_vec_prior = get_optimal_coefficients(rho0, rho1, B_prior)
    msl_prior = lambda_val - b_vec_prior @ la.pinv(G_mat_prior) @ b_vec_prior


    # ---------------- Constrained PVM + Posterioir mean MSL ---------------
    # Linear basis 
    S_linear = sum(alpha_opt_linear[i] * B_linear[i] for i in range(len(B_linear)))
    S_linear = 0.5 * (S_linear + S_linear.conj().T)

    msl_linear_bayes = msl_bayes_for_pvm(S_linear, rho0, rho1, lambda_val)

    # Quadratic homodyne basis 
    S_quad_hom = sum(alpha_opt_quad_hom[i] * B_quad_hom[i] for i in range(len(B_quad_hom)))
    S_quad_hom = 0.5 * (S_quad_hom + S_quad_hom.conj().T)
    msl_quad_hom_bayes = msl_bayes_for_pvm(S_quad_hom, rho0, rho1, lambda_val)

    # Quadratic basis 
    S_quad = sum(alpha_opt_quad[i] * B_quad[i] for i in range(len(B_quad)))
    S_quad = 0.5 * (S_quad + S_quad.conj().T)
    msl_quad_bayes = msl_bayes_for_pvm(S_quad, rho0, rho1, lambda_val)


    # Homodyne measurement at an angle phi + PM estimator
    # Angle of homodyne measurement. 0 and pi/2 corresponds to x and p quadratures respectively
    if ref_state_type == 'coherent':
        phi_homodyne=math.atan2(np.imag(alpha_coherent),np.real(alpha_coherent)) # For a coherent probe state, take the homodyne in the direction of displacement
        #phi_homodyne=0
    if ref_state_type == 'vacuum' or 'thermal':
        phi_homodyne=np.pi/2
    else:
        phi_homodyne=0

    msl_homodyne = msl_homodyne_func(phi_homodyne, rho0, rho1, lambda_val)

    
    #return (msl_bayes, msl_linear, msl_quad_hom, msl_quad, alpha_opt_linear, alpha_opt_quad_hom, alpha_opt_quad, prior_var)
    return (msl_bayes, msl_linear, msl_quad_hom, msl_quad, alpha_opt_linear, alpha_opt_quad_hom, alpha_opt_quad,
             prior_var,alpha_opt_prior,msl_prior,msl_linear_bayes,msl_quad_hom_bayes,msl_quad_bayes,msl_homodyne)


def compute_sigma(theta_sigma):
    # Function used for parallel loop
    return compute_msl_for_prior_width(theta_sigma, theta0=theta0, prior_type=prior_type)


######### -------------------------------------------------------------- Main program --------------------------------------------------------------#########

# -------------------------- User parameters --------------------------
N = 30 # Fock truncation 

# Reference state parameters
ref_state_type = 'coherent'  # Options: 'vacuum', 'coherent', 'thermal', 'squeezed_vacuum', or 'squeezed_thermal'
x0, p0 = 0.0, 0.0  # Initial mean position
alpha_coherent = 0.1+0.5j # Coherent state amplitude (if coherent)
n_thermal = 0.2  # Thermal photons (if thermal)
r_squeeze = 0.3  # Squeezing parameter (if squeezed)
phi_squeeze = 0.0  # Squeezing angle (0 for x-squeezed)

# Prior settings
prior_type = 'gaussian'  # Options: 'gaussian', 'two_gaussian', or 'uniform'
theta0 = 0.1     # Prior mean for theta
theta_pts = 1000    # Number of grid points for theta
sigma_pts = 10 # Number of prior standard deviation grid points


safety_factor=5 # Ensures Fock truncation is enough (5 is safe)
params = design_parameters(N, ref_state_type,alpha_coherent,n_thermal,r_squeeze,theta0,sigma_pts,safety_factor)

theta_min=params['theta_min'] # Prior range
theta_max=params['theta_max']
sigma_max=params['sigma_max'] # Max standard deviation of a Gaussian prior which is 3 sigma contained in the prior range
sigma_min=params['sigma_min'] # Min standard deviation such that the prior is not smaller than the theta_grid spacing
theta_sigma_values=params['theta_sigma_values'] # Range of prior widths to test

# Ladder operators in truncated Fock basis (module-level)
a = np.zeros((N, N), dtype=complex)
for n in range(1, N):
    a[n-1, n] = np.sqrt(n)
adag = a.conj().T
I = np.eye(N, dtype=complex)

# Quadratures
x = (a + adag) / np.sqrt(2)
p = (a - adag) / (1j * np.sqrt(2))


if __name__ == '__main__':
    print("=" * 60)
    print(f"Squeezing estimation  |  ref state: {ref_state_type}")
    if ref_state_type == 'coherent':
        print(f"  {alpha_unicode} = {alpha_coherent}")
    elif ref_state_type == 'thermal':
        print(f"  {nbar_unicode} = {n_thermal}")
    elif ref_state_type == 'squeezed_vacuum':
        print(f"  r = {r_squeeze}, phi = {phi_squeeze}")
    print(f"Prior type: {prior_type}")
    print("=" * 60)

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(compute_sigma, theta_sigma_values),
            total=len(theta_sigma_values),
            desc="Computing MSL",
            unit="sigma"
        ))

    keys = ['msl_bayes', 'msl_linear', 'msl_quad_hom', 'msl_quad',
            'alpha_linear', 'alpha_quad_hom', 'alpha_quad',
            'prior_variance', 'alpha_prior', 'msl_prior',
            'msl_linear_bayes', 'msl_quad_hom_bayes', 'msl_quad_bayes', 'msl_homodyne']

    data = {k: [] for k in keys}
    for res in results:
        for k, v in zip(keys, res):
            data[k].append(v)

    arrays = {k: np.array(v) for k, v in data.items()}

    # Save data
    out_dir = Path(__file__).parent / 'data'
    out_dir.mkdir(exist_ok=True)
    stem = f'squeezing_{ref_state_type}'

    np.savez(out_dir / f'{stem}.npz', **arrays)
    print(f"Saved arrays  → data/{stem}.npz")

    # Save metadata
    metadata = dict(
        ref_state_type = ref_state_type,
        alpha_coherent_re = float(np.real(alpha_coherent)),
        alpha_coherent_im = float(np.imag(alpha_coherent)),
        n_thermal    = n_thermal,
        r_squeeze    = r_squeeze,
        phi_squeeze  = phi_squeeze,
        x0           = x0,
        p0           = p0,
        theta0       = theta0,
        prior_type   = prior_type,
        N            = N,
    )
    with open(out_dir / f'{stem}.json', 'w') as f: json.dump(metadata, f, indent=2)
    print(f"Saved metadata → data/{stem}.json")