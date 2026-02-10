"""
Here we compare the minimum MSL vs prior width for the Bayes-optimal (solution to the Lyapunov equation)
and constrained ansatzes (linear, quadratic, cubic) for estimating 
displacement parameter theta along the x quadrature.

The encoded state is:
rho(theta) = D_x(theta) rho D_x^dagger(theta)
where D_x(theta) = exp(-i theta p)

We estimate theta with different priors and probe states rho.
Since rho(theta) has a translation symmetry, we use f(theta) = theta leading to the mean square loss.
"""
import os
from pathlib import Path
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Unicode values for some characters
sigma_unicode = '\u03C3'
sigma2_unicode = '\u03C3\u00B2'
alpha_unicode = '\u03B1'
theta_unicode = '\u03B8'
nbar_unicode = '\u0304n' 

# ------------------------------------------------- Functions -------------------------------------------------

def design_parameters(N, ref_state_type, alpha=1.0, n_th=0.2, r=0.4, theta0=0.5, sigma_pts=10,safety_factor=3):
    """
    Parameter designer to ensure Fock truncation and prior range is sufficient.

    A reference state displaced along x has <n>(theta) = n0 + 1/2 \theta^2, where n0=|alpha|^2 + n_th.
    For a given Fock truncation, the maximum average photon number we can resolve is <n>_max~N/safety factor, where <n>_max ~ n0 + 1/2 theta_max^2.
    One can then invert for the largest grid size theta_max allowed.

    
    Inputs:
    -----------
    int N : Fock truncation
    str ref_state_type : 'vacuum', 'coherent', 'thermal', 'squeezed_vacuum', 'squeezed_thermal'
    complex or float alpha : Coherent state amplitude
    float n_th : Thermal photon number
    float : Initial squeezing parameter
    float theta0 : Prior center
    float sigma_pts : Number of prior standard deviation grid points
    float safety_factor : Safety margin (3-5 recommended)
    
    Returns:
    --------
    dict with theta_min, theta_max, sigma_max, theta_sigma_values
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
    else:
        raise ValueError("Unknown ref_state_type")
    
    # Photon budget
    n_budget = N / safety_factor - n0
    
    if n_budget < 1:
        raise ValueError("Insufficient Fock truncation")
    
    # For displacements, 1/2theta_max^2 ~ <n>_max - n0 
    theta_max = np.sqrt(2 * n_budget)
    
    #  Prior and grid
    sigma_max = (theta_max - abs(theta0)) / 3 # The maximum standard deviation of a Gaussian prior that ensures the prior doesn't put significant weight beyond theta_max.
    theta_min = theta0 - 3 * sigma_max
    
    theta_sigma_values = np.logspace(-1.5, np.log10(sigma_max), sigma_pts) # Create a prior grid with sigma_pts
    
    return {
        'theta_min': theta_min,
        'theta_max': theta_max,
        'sigma_max': sigma_max,
        'theta_sigma_values': theta_sigma_values,
        'n0': n0,
        'n_budget': n_budget,
    }

def squeeze_op(r, phi):
    # Squeezing operator 
    G = 0.5 * (np.exp(-2j*phi) * adag @ adag - np.exp(2j*phi) * a @ a)
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
    """Create reference state rho (before displacement)"""
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

def displacement_x(theta):
    # Displacement operator along x: D_x(theta) = exp(-i theta p). This shifts x by theta: x -> x + theta
    
    return la.expm(-1j * theta * p)

def get_prior(theta_grid, prior_type, theta0, theta_sigma, theta_min, theta_max):
    # Generate different types of priors on the displacement grid
    if prior_type == 'gaussian':
        prior_unnorm = np.exp(-0.5 * ((theta_grid - theta0) / theta_sigma)**2)

    elif prior_type == 'two_gaussian':
        prior_unnorm = np.exp(-0.5 * ((theta_grid - theta0) / theta_sigma)**2) + 2*np.exp(-0.5 * ((theta_grid - 2*theta0) / theta_sigma)**2)

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

def get_optimal_coefficients(rho0, rho1, B):
    # Compute optimal coefficients alpha^opt for constrained ansatz
    def HS(Aop, Bop):
        # Define the Hilbert-Schmidt inner product between operators A and B
        return np.real(np.trace(Aop.conj().T @ Bop))
    
    m = len(B)
    G = np.zeros((m, m), dtype=float)
    b = np.zeros(m, dtype=float)
    
    for i in range(m):
        for j in range(m):
            G[i, j] = 0.5 * HS(B[i], rho0 @ B[j] + B[j] @ rho0)
        b[i] = HS(B[i], rho1)
    
    alpha_opt, *_ = la.lstsq(G, b)
    
    return alpha_opt, G, b

# Thermal state with varying mean photon number. Not currently used.
"""
def thermal_state_varying(n_bar):
    rho_th = np.zeros((N, N), dtype=complex)
    for n in range(N):
        if n_bar > 0:
            rho_th[n, n] = (n_bar**n) / ((1 + n_bar)**(n+1))
        else:
            rho_th[n, n] = 1.0 if n == 0 else 0.0
    return rho_th
"""

def msl_bayes_for_pvm(S_op, rho0, rho1, lambda_val):
    """
    Compute Bayes MSL for the PVM defined by the spectral decomposition of S_op.
    Uses the posterior mean estimator.
    """
    # Eigen-decomposition of S_op
    eigvals, eigvecs = la.eigh(S_op)

    msl_gain = 0.0
    for k in range(len(eigvals)):
        ket = eigvecs[:, k].reshape(-1, 1)
        Pk = ket @ ket.conj().T

        pk = np.real(np.trace(Pk @ rho0))
        mk = np.real(np.trace(Pk @ rho1))

        if pk > 1e-14:
            msl_gain += (mk**2) / pk

    return lambda_val - msl_gain


def compute_msl_for_prior_width(theta_sigma, theta0=0.0, prior_type='gaussian'):
    """
    Compute MSL for Bayes-optimal, linear, quadratic, and cubic ansatzes.
    
    Returns:
    - msl_bayes, msl_linear, msl_quad, msl_cubic
    - alpha_linear, alpha_quad, alpha_cubic (coefficients)
    - prior_variance
    """
    
    theta_grid = np.linspace(theta_min, theta_max, theta_pts)
    dtheta = theta_grid[1] - theta_grid[0]
    
    prior = get_prior(theta_grid, prior_type, theta0, theta_sigma, theta_min, theta_max)
    prior_var = get_prior_variance(theta_grid, prior)
    
    # Reference state (same for all theta)
    rho_ref = reference_state(ref_state_type, x0=x0, p0=p0, alpha=alpha_coherent,
                              n_th=n_thermal, r=r_squeeze, phi=phi_squeeze)
    
    # Build rho(theta) list - states after displacement
    rho_list = []
    for theta in theta_grid:
        D_x = displacement_x(theta)
        #rho_theta = thermal_state_varying(theta)
        rho_theta = D_x @ rho_ref @ D_x.conj().T
        rho_theta = 0.5 * (rho_theta + rho_theta.conj().T)
        rho_theta = rho_theta / np.trace(rho_theta)
        rho_list.append(rho_theta)
    
    # Compute rho_0 and rho_1
    rho0 = np.zeros((N, N), dtype=complex)
    rho1 = np.zeros((N, N), dtype=complex)
    for i, theta in enumerate(theta_grid):
        rho0 += prior[i] * rho_list[i] * dtheta
        rho1 += prior[i] * theta * rho_list[i] * dtheta
    rho0 = 0.5 * (rho0 + rho0.conj().T)
    rho1 = 0.5 * (rho1 + rho1.conj().T)
    
    lambda_val = np.sum(prior * theta_grid**2 * dtheta)
    
    # ---------------- Exact Bayes S (Fock basis) ---------------
    dim = N * N
    A_big = np.kron(np.eye(N), rho0) + np.kron(rho0.T, np.eye(N))
    vecrho1 = rho1.reshape(dim, order='F')
    vecS_bayes = la.pinv(A_big) @ (2.0 * vecrho1)
    S_bayes = vecS_bayes.reshape((N, N), order='F')
    S_bayes = 0.5 * (S_bayes + S_bayes.conj().T)
    
    msl_bayes = lambda_val - np.real(np.trace(rho0 @ (S_bayes @ S_bayes)))
    
    # Linear ansatz {I, x, p}
    B_linear = [I, x, p]
    B_linear = [0.5 * (M + M.conj().T) for M in B_linear]
    
    alpha_opt_linear, G_mat_linear, b_vec_linear = get_optimal_coefficients(rho0, rho1, B_linear)
    msl_linear = lambda_val - b_vec_linear @ la.pinv(G_mat_linear) @ b_vec_linear
    
    # Quadratic ansatz
    B_quad = [I, x, p, x @ x, 0.5 * (x @ p + p @ x), p @ p]
    B_quad = [0.5 * (M + M.conj().T) for M in B_quad]
    
    alpha_opt_quad, G_mat_quad, b_vec_quad = get_optimal_coefficients(rho0, rho1, B_quad)
    msl_quad = lambda_val - b_vec_quad @ la.pinv(G_mat_quad) @ b_vec_quad
    
    # Cubic ansatz
    B_cubic = B_quad.copy()
    B_cubic.append(x @ x @ x)
    B_cubic.append(x @ x @ p)
    B_cubic.append(x @ p @ p)
    B_cubic.append(p @ p @ p)
    B_cubic.append(0.5 * (x @ x @ p + p @ x @ x))
    B_cubic.append(0.5 * (x @ p @ x + p @ x @ p))
    B_cubic = [0.5 * (M + M.conj().T) for M in B_cubic]
    
    alpha_opt_cubic, G_mat_cubic, b_vec_cubic = get_optimal_coefficients(rho0, rho1, B_cubic)
    msl_cubic = lambda_val - b_vec_cubic @ la.pinv(G_mat_cubic) @ b_vec_cubic


    # ---------------- Constrained PVM + Posterioir mean MSL ---------------
    # Linear ansatz 
    S_linear = sum(alpha_opt_linear[i] * B_linear[i] for i in range(len(B_linear)))
    S_linear = 0.5 * (S_linear + S_linear.conj().T)

    msl_linear_bayes = msl_bayes_for_pvm(S_linear, rho0, rho1, lambda_val)

    # Quadratic ansatz 
    S_quad = sum(alpha_opt_quad[i] * B_quad[i] for i in range(len(B_quad)))
    S_quad = 0.5 * (S_quad + S_quad.conj().T)
    msl_quad_bayes = msl_bayes_for_pvm(S_quad, rho0, rho1, lambda_val)

    # Cubic ansatz 
    S_cubic = sum(alpha_opt_cubic[i] * B_cubic[i] for i in range(len(B_cubic)))
    S_cubic = 0.5 * (S_cubic + S_cubic.conj().T)
    msl_cubic_bayes = msl_bayes_for_pvm(S_cubic, rho0, rho1, lambda_val)

    
    return (msl_bayes, msl_linear, msl_quad, msl_cubic, 
            alpha_opt_linear, alpha_opt_quad, alpha_opt_cubic, prior_var,msl_linear_bayes,msl_quad_bayes,msl_cubic_bayes)


def compute_sigma(theta_sigma):
    return compute_msl_for_prior_width(theta_sigma, theta0=theta0, prior_type=prior_type)

# ------------------------------------------------- Main program -------------------------------------------------

# --------------------- User parameters ---------------------

N = 20 # Fock truncation 

# Reference state parameters (before displacement)
ref_state_type = 'coherent'  # Options: 'vacuum', 'coherent', 'thermal', 'squeezed_vacuum', or 'squeezed_thermal'
x0, p0 = 0.0, 0.0  # Initial mean position
alpha_coherent = 0.5+0.4j  # coherent state amplitude (if coherent)
n_thermal = 0.2  # thermal photons (if thermal)
r_squeeze = 0.4  # squeezing parameter
phi_squeeze = 0.0  # squeezing angle (0 for x-squeezed)

# Prior settings
prior_type = 'gaussian'  # Options: 'gaussian' or 'two_gaussian'
theta0 = 0.1     # prior mean/center for theta
theta_pts = 2000    # number of grid points for theta
sigma_pts = 10 # Number of prior standard deviation grid points

safety_factor=5 # Ensures Fock truncation is enough
params = design_parameters(N, ref_state_type,alpha_coherent,n_thermal,r_squeeze,theta0,sigma_pts,safety_factor)


theta_min=params['theta_min'] # Prior range
theta_max=params['theta_max']
sigma_max=params['sigma_max'] # Max standard deviation of a Gaussian prior which is 99.7% contained in the prior range
theta_sigma_values=params['theta_sigma_values'] # Range of prior widths to test

# Ladder operators in truncated Fock basis
a = np.zeros((N, N), dtype=complex)
for n in range(1, N):
    a[n-1, n] = np.sqrt(n)
adag = a.conj().T
I = np.eye(N, dtype=complex)

# Quadratures
x = (a + adag) / np.sqrt(2)
p = (a - adag) / (1j * np.sqrt(2))

if __name__ == '__main__':

    # Initialise lists 
    msl_bayes_list = []
    msl_linear_list = []
    msl_quad_list = []
    msl_cubic_list = []
    alpha_opt_linear_list = []
    alpha_opt_quad_list = []
    alpha_opt_cubic_list = []
    prior_variance_list = []
    msl_linear_bayes_list = []
    msl_quad_bayes_list = []
    msl_cubic_bayes_list = []

    print("="*70)
    print(f"Estimating displacement theta along x quadrature")
    print(f"Reference state: {ref_state_type}")
    if ref_state_type == 'coherent':
        print(f"  {alpha_unicode} = {alpha_coherent}")
    elif ref_state_type == 'thermal':
        print(f"  {nbar_unicode} = {n_thermal}")
    elif ref_state_type == 'squeezed_vacuum':
        print(f"  r = {r_squeeze}, phi = {phi_squeeze}")
    print(f"Prior type: {prior_type}")
    print(f"Prior center: theta = {theta0}")
    print(f"{theta_unicode} range: [{params['theta_min']:.2f}, {params['theta_max']:.2f}]")
    print(f"{sigma_unicode}_max: {params['sigma_max']:.2f}")
    print("="*70)

    # Loop over prior widths
    # for i, theta_sigma in enumerate(theta_sigma_values):
    #     print(f"Progress: {i+1}/{len(theta_sigma_values)}, {sigma_unicode} = {theta_sigma:.4f}")
    #     result = compute_msl_for_prior_width(theta_sigma, theta0=theta0, prior_type=prior_type)
    #     msl_b, msl_l, msl_q, msl_c, alpha_l, alpha_q, alpha_c, prior_var,msl_bayes_l,msl_bayes_q,msl_bayes_c = result
        
    #     msl_bayes_list.append(msl_b)
    #     msl_linear_list.append(msl_l)
    #     msl_quad_list.append(msl_q)
    #     msl_cubic_list.append(msl_c)
    #     alpha_opt_linear_list.append(alpha_l)
    #     alpha_opt_quad_list.append(alpha_q)
    #     alpha_opt_cubic_list.append(alpha_c)
    #     prior_variance_list.append(prior_var)
    #     msl_linear_bayes_list.append(msl_bayes_l)
    #     msl_quad_bayes_list.append(msl_bayes_q)
    #     msl_cubic_bayes_list.append(msl_bayes_c)

    #     print(f" -> Bayes={msl_b:.4e}, Linear={msl_l:.4e}, Quad={msl_q:.4e}, Cubic={msl_c:.4e}")

    results = []

    # Parallelize with progress bar
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(compute_sigma, theta_sigma_values),
            total=len(theta_sigma_values),
            desc="Computing MSL",
            unit="sigma"
        ))
    for res in results:
        msl_b, msl_l, msl_q, msl_c, alpha_l, alpha_q, alpha_c, prior_var,msl_bayes_l,msl_bayes_q,msl_bayes_c= res
        msl_bayes_list.append(msl_b)
        msl_linear_list.append(msl_l)
        msl_quad_list.append(msl_q)
        msl_cubic_list.append(msl_c)
        alpha_opt_linear_list.append(alpha_l)
        alpha_opt_quad_list.append(alpha_q)
        alpha_opt_cubic_list.append(alpha_c)
        prior_variance_list.append(prior_var)
        # alpha_opt_prior_list.append(alpha_prior)
        # msl_prior_list.append(msl_prior)
        msl_linear_bayes_list.append(msl_bayes_l)
        msl_quad_bayes_list.append(msl_bayes_q)
        msl_cubic_bayes_list.append(msl_bayes_c)

    # Convert to arrays
    prior_variance_list = np.array(prior_variance_list)
    msl_bayes_arr = np.array(msl_bayes_list)
    msl_linear_arr = np.array(msl_linear_list)
    msl_quad_arr = np.array(msl_quad_list)
    msl_cubic_arr = np.array(msl_cubic_list)
    msl_linear_bayes_arr = np.array(msl_linear_bayes_list)
    msl_quad_bayes_arr = np.array(msl_quad_bayes_list)
    msl_cubic_bayes_arr = np.array(msl_cubic_bayes_list)

    """
    4x4 set of plots. Each as a function of the prior width.
    Top left: MSL. Top right: ratio MSL to optimum.
    Bottom left: linear alpha. Bottom right: quadratic alpha 
    """

    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(2, 2)

    # Plot 1: MSL vs prior variance
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.loglog(prior_variance_list, msl_bayes_arr, 'o-', linewidth=2.5, 
            markersize=8, label='Bayes-optimal', color='C0')
    ax1.loglog(prior_variance_list, msl_linear_arr, 'd--', linewidth=2, 
            markersize=7, label='Linear SPM', color='C3')
    ax1.loglog(prior_variance_list, msl_quad_arr, 's--', linewidth=2, 
            markersize=7, label='Quadratic SPM', color='C1')
    ax1.loglog(prior_variance_list, msl_cubic_arr, '^:', linewidth=2, 
            markersize=7, label='Cubic SPM', color='C2')

    # ax1.loglog(prior_variance_list, msl_linear_bayes_arr, '^:', linewidth=2, 
    #            markersize=7, label='Linear Bayes SPM', color='C4')
    # ax1.loglog(prior_variance_list, msl_quad_bayes_arr, '^:', linewidth=2, 
    #            markersize=7, label='Quad Bayes SPM', color='C5')
    ax1.set_xlabel('Prior variance $\\sigma^2$', fontsize=11)
    ax1.set_ylabel('Minimum MSL (MSE)', fontsize=11)
    #ax1.set_title(f'MSE vs Prior Variance ({prior_type} prior)', fontsize=12)
    ax1.legend(fontsize=9)
    #ax1.grid(False, which='both', alpha=0.3)

    # Plot 2: Ratio to Bayes-optimal
    ax2 = fig.add_subplot(gs[0, 1])
    ratio_linear = msl_linear_arr / msl_bayes_arr
    ratio_quad = msl_quad_arr / msl_bayes_arr
    ratio_cubic = msl_cubic_arr / msl_bayes_arr
    ratio_linear_bayes = msl_linear_bayes_arr / msl_bayes_arr
    ratio_quad_bayes = msl_quad_bayes_arr / msl_bayes_arr

    ax2.axhline(y=1, color='C0', linestyle='-', linewidth=2, alpha=0.5, label='Bayes (ratio=1)')
    ax2.semilogx(prior_variance_list, ratio_linear, 'd--', linewidth=2, 
                markersize=7, label='Linear / Bayes', color='C3')
    ax2.semilogx(prior_variance_list, ratio_quad, 's--', linewidth=2, 
                markersize=7, label='Quadratic / Bayes', color='C1')
    ax2.semilogx(prior_variance_list, ratio_cubic, '^:', linewidth=2, 
                markersize=7, label='Cubic / Bayes', color='C2')

    # ax2.semilogx(prior_variance_list, ratio_linear_bayes, '^:', linewidth=2, 
    #              markersize=7, label='Linear SPM / Bayes', color='C4')
    # ax2.semilogx(prior_variance_list, ratio_quad_bayes, '^:', linewidth=2, 
    #              markersize=7, label='Quad SPM / Bayes', color='C5')
    ax2.set_xlabel('Prior variance $\\sigma^2$', fontsize=11)
    ax2.set_ylabel('MSL Ratio', fontsize=11)
    #ax2.set_title('Performance Ratio vs Bayes-Optimal', fontsize=12)
    ax2.legend(fontsize=9)
    #ax2.grid(True, which='both', alpha=0.3)
    ax2.grid(False)

    #Plot 3: Linear coefficients vs prior variance
    ax3 = fig.add_subplot(gs[1, 0])
    linear_labels = ['I', 'x', 'p']
    alpha_linear_array = np.array(alpha_opt_linear_list)
    for i in range(alpha_linear_array.shape[1]):
        ax3.semilogx(prior_variance_list, alpha_linear_array[:, i], 'o-', 
                    linewidth=2, markersize=5, label=linear_labels[i])
    ax3.set_xlabel('Prior variance $\\sigma^2$', fontsize=11)
    ax3.set_ylabel('Optimal coefficient $\\alpha$', fontsize=11)
    #ax3.set_title('Linear SPM vs Prior Variance', fontsize=12)
    ax3.legend(fontsize=9)
    #ax3.grid(True, which='both', alpha=0.3)
    ax3.grid(False)

    # Plot 4: Quadratic coefficients vs prior variance
    ax4 = fig.add_subplot(gs[1, 1])
    quad_labels = ['I', 'x', 'p', 'x²', '(xp+px)/2', 'p²']
    alpha_quad_array = np.array(alpha_opt_quad_list)
    for i in range(alpha_quad_array.shape[1]):
        ax4.semilogx(prior_variance_list, alpha_quad_array[:, i], 'o-', 
                    linewidth=2, markersize=5, label=quad_labels[i])
    ax4.set_xlabel('Prior variance $\\sigma^2$', fontsize=11)
    ax4.set_ylabel('Optimal coefficient $\\alpha$', fontsize=11)
    #ax4.set_title('Quadratic SPM vs Prior Variance', fontsize=12)
    ax4.legend(fontsize=9, ncol=2)
    #ax4.grid(True, which='both', alpha=0.3)
    ax4.grid(False)

    plt.suptitle(f'Displacement Estimation: {ref_state_type} state, {prior_type} prior', fontsize=14, y=0.995)



    """
    Two individual plots of MSL ratio MSL to optimum as a function of the prior width. Then save to folder.
    """

    # Create output directory if it doesn't exist
    # output_dir = Path(__file__).parent / "figs"
    # output_dir.mkdir(exist_ok=True)
    # os.makedirs(output_dir, exist_ok=True)
    # # Plot 1: MSL vs prior variance
    # fig1, ax1 = plt.subplots(figsize=(8, 6))
    # ax1.loglog(prior_variance_list, msl_bayes_arr, 'o-', linewidth=3.5, markersize=10, label='Bayes-optimal', color='C0')
    # ax1.loglog(prior_variance_list, msl_linear_arr, 'd--', linewidth=3, markersize=9, label='Linear SPM', color='C3')
    # ax1.loglog(prior_variance_list, msl_quad_arr, 's--', linewidth=3, markersize=9, label='Quadratic SPM', color='C1')
    # ax1.loglog(prior_variance_list, msl_cubic_arr, '^:', linewidth=3, markersize=9, label='Cubic SPM', color='C2')
    # ax1.set_xlabel('Prior variance $\\sigma^2$', fontsize=20)
    # ax1.set_ylabel('Minimum MSL (MSE)', fontsize=20)
    # ax1.legend(fontsize=20)
    # ax1.tick_params(axis='both', which='major',length=10, width=2, labelsize=20)
    # ax1.tick_params(axis='both', which='minor', length=6, width=1.5)
    # ax1.grid(False)
    # fig1.tight_layout()
    # fig1.savefig(f'{output_dir}/msl_vs_variance_{ref_state_type}.png', dpi=300, bbox_inches='tight')
    # fig1.savefig(f'{output_dir}/msl_vs_variance_{ref_state_type}.pdf', bbox_inches='tight')
    # print(f"Saved: {output_dir}/msl_vs_variance_{ref_state_type}.png")


    # Plot 2: Ratio to Bayes-optimal
    # fig2, ax2 = plt.subplots(figsize=(8, 6))
    # ratio_linear = msl_linear_arr / msl_bayes_arr
    # ratio_quad = msl_quad_arr / msl_bayes_arr
    # ratio_cubic = msl_cubic_arr / msl_bayes_arr

    # ax2.axhline(y=1, color='C0', linestyle='-', linewidth=3, alpha=0.6, label='Bayes (ratio=1)')
    # ax2.semilogx(prior_variance_list, ratio_linear, 'd--', linewidth=3, markersize=9, label='Linear / Bayes', color='C3')
    # ax2.semilogx(prior_variance_list, ratio_quad, 's--', linewidth=3, markersize=9, label='Quadratic / Bayes', color='C1')
    # ax2.semilogx(prior_variance_list, ratio_cubic, '^:', linewidth=3, markersize=9, label='Cubic / Bayes', color='C2')
    # ax2.set_xlabel('Prior variance $\\sigma^2$', fontsize=20)
    # ax2.set_ylabel('MSL Ratio', fontsize=20)
    # ax2.legend(fontsize=20)
    # ax2.tick_params(axis='both', which='major',length=10, width=2, labelsize=20)
    # ax2.tick_params(axis='both', which='minor', length=6, width=1.5)
    # ax2.ticklabel_format(axis='y', style='plain', useOffset=False) # Stops Matplotlib from factoring out +1 from the Bayes ratio on the y axis.
    # ax2.grid(False)
    # fig2.tight_layout()
    # fig2.savefig(f'{output_dir}/ratio_vs_variance_{ref_state_type}.png', dpi=300, bbox_inches='tight')
    # fig2.savefig(f'{output_dir}/ratio_vs_variance_{ref_state_type}.pdf', bbox_inches='tight')
    # print(f"Saved: {output_dir}/ratio_vs_variance_{ref_state_type}.png")


    plt.show()


    # Print summary
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(f"Reference state: {ref_state_type}")
    print(f"Prior type: {prior_type}")
    print(f"Prior center: theta = {theta0}")
    print(f"Prior variance range: {sigma2_unicode} ∈ [{prior_variance_list[0]:.3f}, {prior_variance_list[-1]:.3f}]")
    print(f"\nFinal MSL values (at {sigma2_unicode} = {prior_variance_list[-1]:.3f}):")
    print(f"  Bayes-optimal: {msl_bayes_list[-1]:.6e}")
    print(f"  Linear:        {msl_linear_list[-1]:.6e} (ratio: {ratio_linear[-1]:.4f})")
    print(f"  Quadratic:     {msl_quad_list[-1]:.6e} (ratio: {ratio_quad[-1]:.4f})")
    print(f"  Cubic:         {msl_cubic_list[-1]:.6e} (ratio: {ratio_cubic[-1]:.4f})")

    print(f"\nFinal optimal coefficients $\\alpha$:")
    basis_labels_quad = ['I', 'x', 'p', 'x²', '(xp+px)/2', 'p²']
    for i, label in enumerate(basis_labels_quad):
        print(f"  {alpha_unicode}[{label}] = {alpha_opt_quad_list[-1][i]:+.6f}")

