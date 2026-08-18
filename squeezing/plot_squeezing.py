"""
Loads data produced by generate_squeezing_data.py and produces the MSL ratio figure.

"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Must match the run being plotted
ref_state_type = 'vacuum'  # Options: 'vacuum', 'coherent', 'thermal', 'squeezed_vacuum', or 'squeezed_thermal'
prior_type = 'gaussian' # Options: 'gaussian', 'two_gaussian', or 'uniform'
stem = f'squeezing_{ref_state_type}_{prior_type}'

# Overlay the analytic formulas as thin dotted lines.
# Only useful as a visual check, at small widths the analytic MSL sits below the numerical L(S) by less than the error, so its ratio goes negative and hits the eps floor.
# (To fix, increase grid_sigmas. This would then reduce the max sigma_0^2)
show_analytic = False

inset = 'msl' # 'msl' for absolute MSL vs prior width, 'zoom' for a zoomed view of the wide end, or None
zoom_fraction = 0.4 # Fraction of the sweep shown in the 'zoom' inset

# Find paths
data_dir = Path(__file__).parent / 'data'
out_dir = Path(__file__).parent / 'figs'
out_dir.mkdir(exist_ok=True)

# Load data
npz = np.load(data_dir / f'{stem}.npz', allow_pickle=True)
with open(data_dir / f'{stem}.json') as f:
    meta = json.load(f)

# Unpack arrays
prior_var = npz['prior_variance']
msl_bayes = npz['msl_bayes']
msl_prior = npz['msl_prior']
msl_quad_hom = npz['msl_quad_hom']
msl_quad = npz['msl_quad']
msl_quad_bayes = npz['msl_quad_bayes']
msl_homodyne = npz['msl_homodyne']
#prior_tail = npz['prior_tail']

has_analytic = 'msl_hom_analytic' in npz.files
if has_analytic:
    msl_hom_analytic = npz['msl_hom_analytic']
    msl_quad_analytic = npz['msl_quad_analytic']

# Ratios relative to Bayes-optimal (Floored at eps so that near-zero excesses stay plottable on log axes).
eps = 1e-12
ratio = lambda arr: np.maximum((arr - msl_bayes) / np.abs(msl_bayes), eps)
r_prior = ratio(msl_prior)
r_quad_hom = ratio(msl_quad_hom)
r_quad = ratio(msl_quad)
r_quad_bayes = ratio(msl_quad_bayes)
r_homodyne = ratio(msl_homodyne)

# Styles and colours

lw = 8
fs = 40
fs_t = 30

C_blue = "#070f8b"
C_green = "#2ca02c"
C_red = "#A30000"
C_black = "#000000"
C_grey = "#959ba0"

# Figure 2 plot

fig, ax = plt.subplots(figsize=(14, 9))

ax.loglog(prior_var, r_prior, linestyle='-.', lw=lw, color=C_grey, label='Prior')

# V_quad: constrained estimator [Eq. (62b)] and PM estimator [Eq. (19)], same measurement
ax.loglog(prior_var, r_quad, linestyle='--', lw=lw, color=C_green, label='Quadratic')
ax.loglog(prior_var, r_quad_bayes, linestyle='-', lw=lw, color=C_green, label='Quadratic (PM)')

# V_hom: constrained quadratic estimator [Eq. (68)] and homodyne with the PM estimator.
# The solid curve uses the PVM of q_phi, which is what a homodyne experiment records; the
# PVM of the projected SPM operator itself is a function of q_phi^2 and is degenerate.
ax.loglog(prior_var, r_quad_hom, linestyle='--', lw=lw, color=C_blue, label='Quadratic homodyne')
ax.loglog(prior_var, r_homodyne, linestyle='-', lw=lw, color=C_blue, label='Homodyne (PM)')

if show_analytic and has_analytic:
    ax.loglog(prior_var, ratio(msl_hom_analytic), linestyle=':', lw=3, color=C_black)
    ax.loglog(prior_var, ratio(msl_quad_analytic), linestyle=':', lw=3, color=C_black)

ax.set_xlabel(r'$\sigma^2_0$', fontsize=fs)
ax.set_ylabel(r'$\mathcal{L}_R$', fontsize=fs)
ax.tick_params(axis='both', which='major', length=20, width=3, labelsize=fs_t)
ax.tick_params(axis='both', which='minor', length=12, width=2)
ax.grid(False)

### The inset location and size have been chosen for the vacuum and thermal probes.
### For others, it might not be aligned.

if inset == 'msl':
    # Inset: absolute MSL vs prior width, with the global optimum for reference
    axins = ax.inset_axes([0.55, 0.15, 0.42, 0.35])
    lw_inset = lw - 2

    axins.plot(prior_var, msl_prior, ':', lw=lw_inset, color=C_grey)
    axins.plot(prior_var, msl_quad_hom, '--', lw=lw_inset, color=C_blue)
    axins.plot(prior_var, msl_homodyne, '-', lw=lw_inset, color=C_blue)
    axins.plot(prior_var, msl_quad, '--', lw=lw_inset, color=C_green)
    axins.plot(prior_var, msl_quad_bayes, '-', lw=lw_inset, color=C_green)
    axins.plot(prior_var, msl_bayes, '-', lw=lw_inset, color=C_red)

    axins.set_xlim(prior_var[0], prior_var[-1])
    axins.set_ylim(bottom=0)
    axins.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune='upper'))
    axins.set_ylabel(r'$\mathcal{L}$', fontsize=fs-4)
    axins.tick_params(axis='both', which='major', length=14, width=3.5, labelsize=fs_t)
    axins.grid(False)

elif inset == 'zoom':
    # Inset: zoomed view over the widest priors. The window is a fraction of the sweep rather
    # than a hard-coded sigma_0^2 threshold, so it follows whatever width range was generated.
    n_zoom = max(int(np.ceil(zoom_fraction * len(prior_var))), 3)
    if len(prior_var) >= 6:
        sl = slice(len(prior_var) - n_zoom, None)
        axins = ax.inset_axes([0.55, 0.08, 0.4, 0.35])

        for r_arr, ls, c in [(r_quad_hom, '--', C_blue), (r_homodyne, '-', C_blue),
                             (r_quad, '--', C_green), (r_quad_bayes, '-', C_green)]:
            axins.loglog(prior_var[sl], r_arr[sl], ls, lw=lw-2, color=c)

        axins.set_xlim(prior_var[sl][0], prior_var[sl][-1])
        axins.set_ylim(top=1.3*r_quad_hom[sl].max())
        axins.tick_params(which='both', left=False, labelleft=False,
                          bottom=False, labelbottom=False)
        axins.grid(False)
        ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.75)

fig.tight_layout()
fig.savefig(out_dir / f'squeezing_LR_vs_variance_{ref_state_type}_{prior_type}.png', dpi=300, bbox_inches='tight')
fig.savefig(out_dir / f'squeezing_LR_vs_variance_{ref_state_type}_{prior_type}.pdf', bbox_inches='tight')
print(f"Saved: figs/squeezing_LR_vs_variance_{ref_state_type}_{prior_type}.png")

# Fit log-log slopes in the small prior width region
print("\n" + "="*70)
print("Prior width scaling")
print("="*70)
fit_mask = (prior_var <= 10*prior_var[0]) & (r_quad_hom > 10*eps)
if fit_mask.sum() >= 4:
    print(f"fitting {fit_mask.sum()} points over sigma_0^2 in "
          f"[{prior_var[fit_mask][0]:.2e}, {prior_var[fit_mask][-1]:.2e}]")
    for label, r_arr in [('Quadratic homodyne [I,Dq_phi^2]', r_quad_hom),
                         ('Homodyne (PM)', r_homodyne),
                         ('Quadratic [I,Dq^2,Dp^2]', r_quad),
                         ('Quadratic (PM)', r_quad_bayes)]:
        slope = np.polyfit(np.log(prior_var[fit_mask]), np.log(r_arr[fit_mask]), 1)[0]
        print(f"  {label:<33}: slope = {slope:.3f}")
else:
    print(f"  too few points ({fit_mask.sum()}) for a slope fit; increase sigma_pts "
          f"or widen the range")

# Widths where the grid clipped the prior are not the prior the analytic results assume
# clipped = prior_tail > 1e-6
# if clipped.any():
#     print(f"\nWARNING: {int(clipped.sum())} width(s) use a truncated prior "
#           f"(from sigma_0^2 = {prior_var[clipped][0]:.3g}); up to "
#           f"{np.max(prior_tail):.1%} of the prior weight was cut. Eqs. (62b) and (68) do "
#           f"not describe those points.")

print(f"\nRelative MSL at the largest prior width (sigma_0^2 = {prior_var[-1]:.4f}):")
for label, r_arr in [('Prior', r_prior), ('Quadratic homodyne', r_quad_hom),
                     ('Homodyne (PM)', r_homodyne), ('Quadratic', r_quad),
                     ('Quadratic (PM)', r_quad_bayes)]:
    print(f"  {label:>20}: L_R = {r_arr[-1]:.3e}")

plt.show()