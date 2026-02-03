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

# Unicode values for some 
sigma_unicode = '\u03C3'
sigma2_unicode = '\u03C3\u00B2'
alpha_unicode = '\u03B1'
theta_unicode = '\u03B8'

# ---------------- User parameters ----------------
N = 20           # Fock truncation 
theta_min, theta_max = -10.0, 10.0  # Displacement range
theta_pts = 1000    # number of grid points for theta

# Reference state parameters (before displacement)
ref_state_type = 'coherent'  # Options: 'vacuum', 'coherent', 'thermal', 'squeezed_vacuum', or 'squeezed_thermal'
x0, p0 = 0.0, 0.0  # Initial mean position
alpha_coherent = 1.0  # coherent state amplitude (if coherent)
n_thermal = 0.2  # thermal photons (if thermal)
r_squeeze = 0.4  # squeezing parameter
phi_squeeze = 0.0  # squeezing angle (0 for x-squeezed)

# Prior settings
prior_type = 'gaussian'  # Options: 'gaussian' or 'two_gaussian'
theta0 = 1     # prior mean/center for theta

# Range of prior widths to test
theta_sigma_values = np.logspace(-1.5, 0.1, 10) 
# -------------------------------------------------

# Ladder operators in truncated Fock basis
a = np.zeros((N, N), dtype=complex)
for n in range(1, N):
    a[n-1, n] = np.sqrt(n)
adag = a.conj().T
I = np.eye(N, dtype=complex)

# Quadratures
x = (a + adag) / np.sqrt(2)
p = (a - adag) / (1j * np.sqrt(2))

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
    A_big = np.kron(np.eye(N), rho0) + np.kron(rho0.conj().T, np.eye(N))
    vecrho1 = rho1.reshape(dim, order='F')
    vecS_bayes = la.pinv(A_big) @ (2.0 * vecrho1)
    S_bayes = vecS_bayes.reshape((N, N), order='F')
    S_bayes = 0.5 * (S_bayes + S_bayes.conj().T)
    
    msl_bayes = lambda_val - np.real(np.trace(rho0 @ (S_bayes @ S_bayes)))
    
    # ---------------- Linear ansatz: {I, x, p} -----------------------
    B_linear = [I, x, p]
    B_linear = [0.5 * (M + M.conj().T) for M in B_linear]
    
    alpha_opt_linear, G_mat_linear, b_vec_linear = get_optimal_coefficients(rho0, rho1, B_linear)
    msl_linear = lambda_val - b_vec_linear @ la.pinv(G_mat_linear) @ b_vec_linear
    
    # ---------------- Quadratic ansatz -----------------------
    B_quad = [I, x, p, x @ x, 0.5 * (x @ p + p @ x), p @ p]
    B_quad = [0.5 * (M + M.conj().T) for M in B_quad]
    
    alpha_opt_quad, G_mat_quad, b_vec_quad = get_optimal_coefficients(rho0, rho1, B_quad)
    msl_quad = lambda_val - b_vec_quad @ la.pinv(G_mat_quad) @ b_vec_quad
    
    # ---------------- Cubic ansatz -----------------------
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
    
    return (msl_bayes, msl_linear, msl_quad, msl_cubic, 
            alpha_opt_linear, alpha_opt_quad, alpha_opt_cubic, prior_var)

# Compute MSL for each prior width
msl_bayes_list = []
msl_linear_list = []
msl_quad_list = []
msl_cubic_list = []
alpha_opt_linear_list = []
alpha_opt_quad_list = []
alpha_opt_cubic_list = []
prior_variance_list = []

print("="*70)
print(f"Estimating displacement theta along x quadrature")
print(f"Reference state: {ref_state_type}")
if ref_state_type == 'coherent':
    print(f"  {alpha_unicode} = {alpha_coherent}")
elif ref_state_type == 'thermal':
    print(f"  n = {n_thermal}")
elif ref_state_type == 'squeezed_vacuum':
    print(f"  r = {r_squeeze}, phi = {phi_squeeze}")
print(f"Prior type: {prior_type}")
print(f"Prior center: theta = {theta0}")
print("="*70)

for i, theta_sigma in enumerate(theta_sigma_values):
    print(f"Progress: {i+1}/{len(theta_sigma_values)}, {sigma_unicode} = {theta_sigma:.4f}", end='')
    result = compute_msl_for_prior_width(theta_sigma, theta0=theta0, prior_type=prior_type)
    msl_b, msl_l, msl_q, msl_c, alpha_l, alpha_q, alpha_c, prior_var = result
    
    msl_bayes_list.append(msl_b)
    msl_linear_list.append(msl_l)
    msl_quad_list.append(msl_q)
    msl_cubic_list.append(msl_c)
    alpha_opt_linear_list.append(alpha_l)
    alpha_opt_quad_list.append(alpha_q)
    alpha_opt_cubic_list.append(alpha_c)
    prior_variance_list.append(prior_var)
    print(f" -> Bayes={msl_b:.4e}, Linear={msl_l:.4e}, Quad={msl_q:.4e}, Cubic={msl_c:.4e}")

# Convert to arrays
prior_variance_list = np.array(prior_variance_list)
msl_bayes_arr = np.array(msl_bayes_list)
msl_linear_arr = np.array(msl_linear_list)
msl_quad_arr = np.array(msl_quad_list)
msl_cubic_arr = np.array(msl_cubic_list)

"""
4x4 set of plots. Each as a function of the prior width.
Top left: MSL. Top right: ratio MSL to optimum.
Bottom left: linear alpha. Bottom right: quadratic alpha 
"""

# fig = plt.figure(figsize=(16, 16))
# gs = fig.add_gridspec(2, 2)

# # Plot 1: MSL vs prior variance
# ax1 = fig.add_subplot(gs[0, 0])
# ax1.loglog(prior_variance_list, msl_bayes_arr, 'o-', linewidth=2.5, 
#            markersize=8, label='Bayes-optimal', color='C0')
# ax1.loglog(prior_variance_list, msl_linear_arr, 'd--', linewidth=2, 
#            markersize=7, label='Linear SPM', color='C3')
# ax1.loglog(prior_variance_list, msl_quad_arr, 's--', linewidth=2, 
#            markersize=7, label='Quadratic SPM', color='C1')
# ax1.loglog(prior_variance_list, msl_cubic_arr, '^:', linewidth=2, 
#            markersize=7, label='Cubic SPM', color='C2')
# ax1.set_xlabel('Prior variance $\\sigma^2$', fontsize=11)
# ax1.set_ylabel('Minimum MSL (MSE)', fontsize=11)
# #ax1.set_title(f'MSE vs Prior Variance ({prior_type} prior)', fontsize=12)
# ax1.legend(fontsize=9)
# #ax1.grid(False, which='both', alpha=0.3)

# # Plot 2: Ratio to Bayes-optimal
# ax2 = fig.add_subplot(gs[0, 1])
# ratio_linear = msl_linear_arr / msl_bayes_arr
# ratio_quad = msl_quad_arr / msl_bayes_arr
# ratio_cubic = msl_cubic_arr / msl_bayes_arr

# ax2.axhline(y=1, color='C0', linestyle='-', linewidth=2, alpha=0.5, label='Bayes (ratio=1)')
# ax2.semilogx(prior_variance_list, ratio_linear, 'd--', linewidth=2, 
#              markersize=7, label='Linear / Bayes', color='C3')
# ax2.semilogx(prior_variance_list, ratio_quad, 's--', linewidth=2, 
#              markersize=7, label='Quadratic / Bayes', color='C1')
# ax2.semilogx(prior_variance_list, ratio_cubic, '^:', linewidth=2, 
#              markersize=7, label='Cubic / Bayes', color='C2')
# ax2.set_xlabel('Prior variance $\\sigma^2$', fontsize=11)
# ax2.set_ylabel('MSL Ratio', fontsize=11)
# #ax2.set_title('Performance Ratio vs Bayes-Optimal', fontsize=12)
# ax2.legend(fontsize=9)
# #ax2.grid(True, which='both', alpha=0.3)
# ax2.grid(False)

# #Plot 3: Linear coefficients vs prior variance
# ax3 = fig.add_subplot(gs[1, 0])
# linear_labels = ['I', 'x', 'p']
# alpha_linear_array = np.array(alpha_opt_linear_list)
# for i in range(alpha_linear_array.shape[1]):
#     ax3.semilogx(prior_variance_list, alpha_linear_array[:, i], 'o-', 
#                  linewidth=2, markersize=5, label=linear_labels[i])
# ax3.set_xlabel('Prior variance $\\sigma^2$', fontsize=11)
# ax3.set_ylabel('Optimal coefficient $\\alpha$', fontsize=11)
# #ax3.set_title('Linear SPM vs Prior Variance', fontsize=12)
# ax3.legend(fontsize=9)
# #ax3.grid(True, which='both', alpha=0.3)
# ax3.grid(False)

# # Plot 4: Quadratic coefficients vs prior variance
# ax4 = fig.add_subplot(gs[1, 1])
# quad_labels = ['I', 'x', 'p', 'x²', '(xp+px)/2', 'p²']
# alpha_quad_array = np.array(alpha_opt_quad_list)
# for i in range(alpha_quad_array.shape[1]):
#     ax4.semilogx(prior_variance_list, alpha_quad_array[:, i], 'o-', 
#                  linewidth=2, markersize=5, label=quad_labels[i])
# ax4.set_xlabel('Prior variance $\\sigma^2$', fontsize=11)
# ax4.set_ylabel('Optimal coefficient $\\alpha$', fontsize=11)
# #ax4.set_title('Quadratic SPM vs Prior Variance', fontsize=12)
# ax4.legend(fontsize=9, ncol=2)
# #ax4.grid(True, which='both', alpha=0.3)
# ax4.grid(False)

# plt.suptitle(f'Displacement Estimation: {ref_state_type} state, {prior_type} prior', fontsize=14, y=0.995)


"""
Two individual plots of MSL ratio MSL to optimum as a function of the prior width. Then save to folder.
"""

# Create output directory if it doesn't exist
output_dir = Path(__file__).parent / "figs"
output_dir.mkdir(exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
# Plot 1: MSL vs prior variance
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.loglog(prior_variance_list, msl_bayes_arr, 'o-', linewidth=3.5, markersize=10, label='Bayes-optimal', color='C0')
ax1.loglog(prior_variance_list, msl_linear_arr, 'd--', linewidth=3, markersize=9, label='Linear SPM', color='C3')
ax1.loglog(prior_variance_list, msl_quad_arr, 's--', linewidth=3, markersize=9, label='Quadratic SPM', color='C1')
ax1.loglog(prior_variance_list, msl_cubic_arr, '^:', linewidth=3, markersize=9, label='Cubic SPM', color='C2')
ax1.set_xlabel('Prior variance $\\sigma^2$', fontsize=20)
ax1.set_ylabel('Minimum MSL (MSE)', fontsize=20)
ax1.legend(fontsize=20)
ax1.tick_params(axis='both', which='major',length=10, width=2, labelsize=20)
ax1.tick_params(axis='both', which='minor', length=6, width=1.5)
ax1.grid(False)
fig1.tight_layout()
# fig1.savefig(f'{output_dir}/msl_vs_variance_{ref_state_type}.png', dpi=300, bbox_inches='tight')
# fig1.savefig(f'{output_dir}/msl_vs_variance_{ref_state_type}.pdf', bbox_inches='tight')
# print(f"Saved: {output_dir}/msl_vs_variance_{ref_state_type}.png")

# Plot 2: Ratio to Bayes-optimal
fig2, ax2 = plt.subplots(figsize=(8, 6))
ratio_linear = msl_linear_arr / msl_bayes_arr
ratio_quad = msl_quad_arr / msl_bayes_arr
ratio_cubic = msl_cubic_arr / msl_bayes_arr

ax2.axhline(y=1, color='C0', linestyle='-', linewidth=3, alpha=0.6, label='Bayes (ratio=1)')
ax2.semilogx(prior_variance_list, ratio_linear, 'd--', linewidth=3, markersize=9, label='Linear / Bayes', color='C3')
ax2.semilogx(prior_variance_list, ratio_quad, 's--', linewidth=3, markersize=9, label='Quadratic / Bayes', color='C1')
ax2.semilogx(prior_variance_list, ratio_cubic, '^:', linewidth=3, markersize=9, label='Cubic / Bayes', color='C2')
ax2.set_xlabel('Prior variance $\\sigma^2$', fontsize=20)
ax2.set_ylabel('MSL Ratio', fontsize=20)
ax2.legend(fontsize=20)
ax2.tick_params(axis='both', which='major',length=10, width=2, labelsize=20)
ax2.tick_params(axis='both', which='minor', length=6, width=1.5)
ax2.ticklabel_format(axis='y', style='plain', useOffset=False) # Stops Matplotlib from factoring out +1 from the Bayes ratio on the y axis.
ax2.grid(False)
fig2.tight_layout()
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

