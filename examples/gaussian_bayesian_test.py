"""
Compare Bayes-optimal S (Lyapunov), quadratic ansatz S
for estimating squeezing parameter r of a single-mode squeezed vacuum.
"""

import numpy as np
import scipy.linalg as la
import math
import matplotlib.pyplot as plt

# ---------------- User parameters ----------------
N = 20            # Fock truncation (increase to check convergence; decreases speed)
r0 = 2          # prior mean for r
r_sigma = 10   # prior std
r_min, r_max = 0.0, 1.0
r_pts = 41        # number of grid points for r (increase for accuracy)
# -------------------------------------------------

r_grid = np.linspace(r_min, r_max, r_pts)
dr = r_grid[1] - r_grid[0]

# Truncated Gaussian prior on [r_min, r_max]
prior_unnorm = np.exp(-0.5 * ((r_grid - r0) / r_sigma)**2)
prior = prior_unnorm / (np.sum(prior_unnorm) * dr)

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
# Construct generator G = 0.5*(adag^2 - a^2) so that S = exp(r G)
def squeeze_op(r):
    G = 0.5 * (adag @ adag - a @ a)
    return la.expm(r * G)

# Build rho(r) list
vac = np.zeros((N, N), dtype=complex)
vac[0, 0] = 1.0
rho_list = []
for r in r_grid:
    S = squeeze_op(r)
    rho = S @ vac @ S.conj().T
    rho = 0.5 * (rho + rho.conj().T)           # enforce Hermiticity
    rho = rho / np.trace(rho)                 # normalize numerically
    rho_list.append(rho)

# Compute bar-rho and W = ∫ prior r rho(r) dr
barrho = np.zeros((N, N), dtype=complex)
W = np.zeros((N, N), dtype=complex)
for i, r in enumerate(r_grid):
    barrho += prior[i] * rho_list[i] * dr
    W += prior[i] * r * rho_list[i] * dr
barrho = 0.5 * (barrho + barrho.conj().T)
W = 0.5 * (W + W.conj().T)

# ---------------- Exact Bayes S (Fock basis) ---------------
dim = N * N
A_big = np.kron(np.eye(N), barrho) + np.kron(barrho.conj().T, np.eye(N))
vecW = W.reshape(dim, order='F')
# Use pseudo-inverse for stability; if barrho is full-rank you can la.solve
vecS_bayes = la.pinv(A_big) @ (2.0 * vecW)
S_bayes = vecS_bayes.reshape((N, N), order='F')
S_bayes = 0.5 * (S_bayes + S_bayes.conj().T)

# ---------------- Quadratic ansatz -----------------------
# basis: [I, x, p, x^2, (xp+px)/2, p^2]
B = []
B.append(I)
B.append(x)
B.append(p)
B.append(x @ x)
B.append(0.5 * (x @ p + p @ x))
B.append(p @ p)
# hermitize basis
B = [0.5 * (M + M.conj().T) for M in B]

def HS(A, Bop):
    "Hilbert-Schmidt inner product (real part)"
    return np.real(np.trace(A.conj().T @ Bop))

m = len(B)
Mmat = np.zeros((m, m), dtype=float)
bvec = np.zeros(m, dtype=float)
for i, Bi in enumerate(B):
    for j, Bj in enumerate(B):
        Mmat[i, j] = HS(Bi, barrho @ Bj + Bj @ barrho)
    bvec[i] = 2.0 * HS(B[i], W)

# solve for coefficients (least squares in case of near-singular)
c_coeffs, *_ = la.lstsq(Mmat, bvec)
S_quad = sum(c_coeffs[k] * B[k] for k in range(m))
S_quad = 0.5 * (S_quad + S_quad.conj().T)

# ---------------- SLD at prior mean (local) ----------------
# finite-difference derivative at r0
eps = 1e-6
rho_plus = squeeze_op(r0 + eps) @ vac @ squeeze_op(r0 + eps).conj().T
rho_minus = squeeze_op(r0 - eps) @ vac @ squeeze_op(r0 - eps).conj().T
drho = (rho_plus - rho_minus) / (2.0 * eps)
drho = 0.5 * (drho + drho.conj().T)

# Solve rho L + L rho = 2 drho at mean (use rho at r0)
rho_r0 = squeeze_op(r0) @ vac @ squeeze_op(r0).conj().T
A_big_r0 = np.kron(np.eye(N), rho_r0) + np.kron(rho_r0.conj().T, np.eye(N))
vec_drho = drho.reshape(dim, order='F')
vecL = la.pinv(A_big_r0) @ (2.0 * vec_drho)
L_SLD = vecL.reshape((N, N), order='F')
L_SLD = 0.5 * (L_SLD + L_SLD.conj().T)

# ---------------- Compute Bayes risks ---------------------
def bayes_risk(M_op):
    risk = 0.0
    for i, r in enumerate(r_grid):
        rho = rho_list[i]
        diff = M_op - r * np.eye(N)
        val = np.real(np.trace(rho @ (diff @ diff)))
        risk += prior[i] * val * dr
    return float(risk)

risk_bayes = bayes_risk(S_bayes)
risk_quad = bayes_risk(S_quad)
risk_sld = bayes_risk(L_SLD)

# operator distances
hs_norm = np.linalg.norm(S_quad - S_bayes, 'fro')
op_norm = la.norm(S_quad - S_bayes, 2)

# ---------------- Output & simple plots -------------------
print("Parameters: N=", N, ", r grid pts=", r_pts)
print("Bayes risk (exact Bayes operator): {:.6e}".format(risk_bayes))
print("Bayes risk (quadratic ansatz):       {:.6e}".format(risk_quad))
print("Bayes risk (SLD at mean):            {:.6e}".format(risk_sld))
print("||S_quad - S_bayes||_HS = {:.3e}, op-norm = {:.3e}".format(hs_norm, op_norm))

# quick visual checks
# plt.figure(figsize=(10,4))
# plt.subplot(1,2,1)
# plt.plot(np.real(np.diag(S_bayes)), 'o-', label='S_bayes diag')
# plt.plot(np.real(np.diag(S_quad)), 'x--', label='S_quad diag')
# plt.plot(np.real(np.diag(L_SLD)), 's:', label='SLD diag')
# plt.xlabel('Fock level n'); plt.ylabel('Diagonal value'); plt.legend(); plt.grid(True)

#plt.subplot(1,2,2)
instant_bayes = [np.real(np.trace(rho_list[i] @ (S_bayes - r_grid[i]*np.eye(N)) @ (S_bayes - r_grid[i]*np.eye(N)))) for i in range(len(r_grid))]
instant_quad = [np.real(np.trace(rho_list[i] @ (S_quad - r_grid[i]*np.eye(N)) @ (S_quad - r_grid[i]*np.eye(N)))) for i in range(len(r_grid))]
#instant_sld = [np.real(np.trace(rho_list[i] @ (L_SLD - r_grid[i]*np.eye(N)) @ (L_SLD - r_grid[i]*np.eye(N)))) for i in range(len(r_grid))]
plt.plot(r_grid, instant_bayes, label='inst risk Bayes'); plt.plot(r_grid, instant_quad, label='inst risk quad');# plt.plot(r_grid, instant_sld, label='inst risk SLD')
plt.xlabel('r'); plt.ylabel('MSE (for that r)'); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.show()
