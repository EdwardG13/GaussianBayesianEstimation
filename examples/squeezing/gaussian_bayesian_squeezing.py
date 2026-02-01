"""
Compare minimum MSL vs prior width for Bayes-optimal S (Lyapunov) 
and quadratic ansatz S for estimating displacement.
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# ---------------- User parameters ----------------
N = 20            # Fock truncation
r0 = 0.5          # prior mean for r (keep in middle of range)
r_min, r_max = 0.0, 1.0
r_pts = 41        # number of grid points for r

# Range of prior widths to test
r_sigma_values = np.logspace(-2, 1, 20)  # from 0.01 to 10
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

# Squeezing operator: S(r) = exp[r * (a^\dagger^2 - a^2)/2]
def squeeze_op(r):
    G = 0.5 * (adag @ adag - a @ a)
    return la.expm(r * G)

# Vacuum state
vac = np.zeros((N, N), dtype=complex)
vac[0, 0] = 1.0

def compute_msl_for_prior_width(r_sigma):
    """Compute MSL for both methods given a prior width"""
    
    r_grid = np.linspace(r_min, r_max, r_pts)
    dr = r_grid[1] - r_grid[0]
    
    # Truncated Gaussian prior on [r_min, r_max]
    prior_unnorm = np.exp(-0.5 * ((r_grid - r0) / r_sigma)**2)
    prior = prior_unnorm / (np.sum(prior_unnorm) * dr)
    
    # Build rho(r) list
    rho_list = []
    for r in r_grid:
        S = squeeze_op(r)
        rho = S @ vac @ S.conj().T
        rho = 0.5 * (rho + rho.conj().T)
        rho = rho / np.trace(rho)
        rho_list.append(rho)
    
    # Compute bar-rho (rho_0) and W (bar-rho or rho_1)
    barrho = np.zeros((N, N), dtype=complex)
    W = np.zeros((N, N), dtype=complex)
    for i, r in enumerate(r_grid):
        barrho += prior[i] * rho_list[i] * dr
        W += prior[i] * r * rho_list[i] * dr
    barrho = 0.5 * (barrho + barrho.conj().T)
    W = 0.5 * (W + W.conj().T)
    
    # Compute lambda (second moment of prior * f(theta))
    lambda_val = np.sum(prior * r_grid**2 * dr)
    
    # ---------------- Exact Bayes S (Fock basis) ---------------
    dim = N * N
    A_big = np.kron(np.eye(N), barrho) + np.kron(barrho.conj().T, np.eye(N))
    vecW = W.reshape(dim, order='F')
    vecS_bayes = la.pinv(A_big) @ (2.0 * vecW)
    S_bayes = vecS_bayes.reshape((N, N), order='F')
    S_bayes = 0.5 * (S_bayes + S_bayes.conj().T)
    
    # MSL for Bayes: L_min = lambda - Tr(rho_0 S^2)
    msl_bayes = lambda_val - np.real(np.trace(barrho @ (S_bayes @ S_bayes)))
    
    # ---------------- Quadratic ansatz -----------------------
    B = []
    B.append(I)
    B.append(x)
    B.append(p)
    B.append(x @ x)
    B.append(0.5 * (x @ p + p @ x))
    B.append(p @ p)
    B = [0.5 * (M + M.conj().T) for M in B]
    
    def HS(A, Bop):
        return np.real(np.trace(A.conj().T @ Bop))
    
    m = len(B)
    Mmat = np.zeros((m, m), dtype=float)
    bvec = np.zeros(m, dtype=float)
    for i, Bi in enumerate(B):
        for j, Bj in enumerate(B):
            Mmat[i, j] = HS(Bi, barrho @ Bj + Bj @ barrho)
        bvec[i] = 2.0 * HS(B[i], W)
    
    c_coeffs, *_ = la.lstsq(Mmat, bvec)
    S_quad = sum(c_coeffs[k] * B[k] for k in range(m))
    S_quad = 0.5 * (S_quad + S_quad.conj().T)
    
    # MSL for quadratic: L = lambda - b^T G^{-1} b
    # where b_i = Tr(rho_1 B_i) and G_ij = (1/2)Tr(rho_0 {B_i, B_j})
    b_vec = np.array([HS(Bi, W) for Bi in B])
    G_mat = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            G_mat[i, j] = 0.5 * HS(B[i], barrho @ B[j] + B[j] @ barrho)
    
    msl_quad = lambda_val - b_vec @ la.pinv(G_mat) @ b_vec
    
    return msl_bayes, msl_quad

# Compute MSL for each prior width
msl_bayes_list = []
msl_quad_list = []

print("Computing MSL for different prior widths...")
for i, r_sigma in enumerate(r_sigma_values):
    print(f"Progress: {i+1}/{len(r_sigma_values)}, sigma = {r_sigma:.4f}")
    msl_b, msl_q = compute_msl_for_prior_width(r_sigma)
    msl_bayes_list.append(msl_b)
    msl_quad_list.append(msl_q)

# Plotting
plt.figure(figsize=(10, 6))
plt.loglog(r_sigma_values, msl_bayes_list, 'o-', linewidth=2, 
           markersize=6, label='Full Bayes-optimal S (Lyapunov)')
plt.loglog(r_sigma_values, msl_quad_list, 's--', linewidth=2, 
           markersize=6, label='Constrained S (Quadratic ansatz)')

plt.xlabel('Prior width σ', fontsize=12)
plt.ylabel('Minimum MSL', fontsize=12)
plt.title('Minimum Mean Squared Loss vs Prior Width', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()

# Print summary
print("\n" + "="*60)
print("Summary:")
print("="*60)
print(f"Prior mean: r₀ = {r0}")
print(f"Prior width range: σ ∈ [{r_sigma_values[0]:.3f}, {r_sigma_values[-1]:.3f}]")
print(f"\nFinal MSL values (at σ = {r_sigma_values[-1]:.3f}):")
print(f"  Bayes-optimal: {msl_bayes_list[-1]:.6e}")
print(f"  Quadratic:     {msl_quad_list[-1]:.6e}")
print(f"  Ratio:         {msl_quad_list[-1]/msl_bayes_list[-1]:.4f}")


def compute_msl_for_prior_width(r_sigma):
    """Compute MSL for both methods given a prior width"""
    
    r_grid = np.linspace(r_min, r_max, r_pts)
    dr = r_grid[1] - r_grid[0]
    
    # Truncated Gaussian prior on [r_min, r_max]
    prior_unnorm = np.exp(-0.5 * ((r_grid - r0) / r_sigma)**2)
    prior = prior_unnorm / (np.sum(prior_unnorm) * dr)
    
    # Build rho(r) list
    rho_list = []
    for r in r_grid:
        S = squeeze_op(r)
        rho = S @ vac @ S.conj().T
        rho = 0.5 * (rho + rho.conj().T)
        rho = rho / np.trace(rho)
        rho_list.append(rho)
    
    # Compute bar-rho (rho_0) and W (bar-rho or rho_1)
    barrho = np.zeros((N, N), dtype=complex)
    W = np.zeros((N, N), dtype=complex)
    for i, r in enumerate(r_grid):
        barrho += prior[i] * rho_list[i] * dr
        W += prior[i] * r * rho_list[i] * dr
    barrho = 0.5 * (barrho + barrho.conj().T)
    W = 0.5 * (W + W.conj().T)
    
    # Compute lambda (second moment of prior * f(theta))
    lambda_val = np.sum(prior * r_grid**2 * dr)
    
    # ---------------- Exact Bayes S (Fock basis) ---------------
    dim = N * N
    A_big = np.kron(np.eye(N), barrho) + np.kron(barrho.conj().T, np.eye(N))
    vecW = W.reshape(dim, order='F')
    vecS_bayes = la.pinv(A_big) @ (2.0 * vecW)
    S_bayes = vecS_bayes.reshape((N, N), order='F')
    S_bayes = 0.5 * (S_bayes + S_bayes.conj().T)
    
    # MSL for Bayes: L_min = lambda - Tr(rho_0 S^2)
    msl_bayes = lambda_val - np.real(np.trace(barrho @ (S_bayes @ S_bayes)))
    
    # ---------------- Quadratic ansatz -----------------------
    B = []
    B.append(I)
    B.append(x)
    B.append(p)
    B.append(x @ x)
    B.append(0.5 * (x @ p + p @ x))
    B.append(p @ p)
    B = [0.5 * (M + M.conj().T) for M in B]
    
    def HS(A, Bop):
        return np.real(np.trace(A.conj().T @ Bop))
    
    m = len(B)
    Mmat = np.zeros((m, m), dtype=float)
    bvec = np.zeros(m, dtype=float)
    for i, Bi in enumerate(B):
        for j, Bj in enumerate(B):
            Mmat[i, j] = HS(Bi, barrho @ Bj + Bj @ barrho)
        bvec[i] = 2.0 * HS(B[i], W)
    
    c_coeffs, *_ = la.lstsq(Mmat, bvec)
    S_quad = sum(c_coeffs[k] * B[k] for k in range(m))
    S_quad = 0.5 * (S_quad + S_quad.conj().T)
    
    # MSL for quadratic: L = lambda - b^T G^{-1} b
    # where b_i = Tr(rho_1 B_i) and G_ij = (1/2)Tr(rho_0 {B_i, B_j})
    b_vec = np.array([HS(Bi, W) for Bi in B])
    G_mat = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            G_mat[i, j] = 0.5 * HS(B[i], barrho @ B[j] + B[j] @ barrho)
    
    msl_quad = lambda_val - b_vec @ la.pinv(G_mat) @ b_vec
    
    return msl_bayes, msl_quad