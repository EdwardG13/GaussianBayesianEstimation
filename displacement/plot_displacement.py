"""
Loads data produced by generate_displacement_data.py and produces the MSL ratio figure.

"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Must match the run being plotted
ref_state_type = 'squeezed_thermal'  # Options: 'vacuum', 'coherent', 'thermal', 'squeezed_vacuum', or 'squeezed_thermal'
prior_type = 'uniform'       # Options: 'gaussian', 'two_gaussian', or 'uniform'
stem = f'displacement_{ref_state_type}_{prior_type}'

show_inset = True            # Zoomed view over the widest priors
zoom_fraction = 0.4          # Fraction of the sweep (from the wide end) shown in the inset

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
msl_linear = npz['msl_linear']
msl_cubic = npz['msl_cubic']
msl_quintic = npz['msl_quintic']
msl_linear_bayes = npz['msl_linear_bayes']
msl_cubic_bayes = npz['msl_cubic_bayes']
msl_homodyne = npz['msl_homodyne']

# Ratios relative to Bayes-optimal (Floored at eps so that near-zero excesses stay plottable on log axes).
eps = 1e-12
ratio = lambda arr: np.maximum((arr - msl_bayes) / np.abs(msl_bayes), eps)
r_prior = ratio(msl_prior)
r_linear = ratio(msl_linear)
r_cubic = ratio(msl_cubic)
r_quintic = ratio(msl_quintic)
r_linear_bayes = ratio(msl_linear_bayes)
r_cubic_bayes = ratio(msl_cubic_bayes)
r_homodyne = ratio(msl_homodyne)

# Styles and colours

lw = 8
fs = 40
fs_t = 30

C_blue = "#070f8b"
C_green = "#2ca02c"
C_black = "#000000"
C_grey = "#959ba0"

# Figure 1 plot

fig, ax = plt.subplots(figsize=(14, 9))

ax.loglog(prior_var, r_prior, linestyle='-.', lw=lw, color=C_grey, label='Prior')
ax.loglog(prior_var, r_linear, linestyle=':', lw=lw, color=C_black, label='Linear')
ax.loglog(prior_var, r_cubic, linestyle='--', lw=lw, color=C_black, label='Cubic')
#ax.loglog(prior_var, r_quintic, linestyle='--', lw=lw, color=C_green, label='Quintic')
#ax.loglog(prior_var, r_cubic_bayes, linestyle='-', lw=lw, color=C_green, label='Cubic (PM)')

# Homodyne with the PM estimator. The PVM of the linear projected SPM operator is exactly
# x-homodyne, so r_linear_bayes and r_homodyne coincide; either may be plotted.
ax.loglog(prior_var, r_homodyne, linestyle='-', lw=lw, color=C_black, label='Homodyne (PM)')

ax.set_xlabel(r'$\sigma^2_0$', fontsize=fs)
ax.set_ylabel(r'$\mathcal{L}_R$', fontsize=fs)
ax.tick_params(axis='both', which='major', length=20, width=3, labelsize=fs_t)
ax.tick_params(axis='both', which='minor', length=12, width=2)
ax.grid(False)

# Inset: zoomed view over the widest priors. The window is a fraction of the sweep rather
# than a hard-coded sigma_0^2 threshold, so it follows whatever width range was generated.
n_zoom = max(int(np.ceil(zoom_fraction * len(prior_var))), 3)
if show_inset and len(prior_var) >= 6:
    sl = slice(len(prior_var) - n_zoom, None)
    axins = ax.inset_axes([0.55, 0.08, 0.4, 0.35])

    for r_arr, ls in [(r_linear, ':'), (r_cubic, '--'), (r_homodyne, '-')]:
        axins.loglog(prior_var[sl], r_arr[sl], ls, lw=lw-2, color=C_black)

    axins.set_xlim(prior_var[sl][0], prior_var[sl][-1])
    axins.set_ylim(top=1.3*max(r_linear[sl].max(), r_cubic[sl].max(), r_homodyne[sl].max()))
    axins.tick_params(which='both', left=False, labelleft=False, bottom=False, labelbottom=False)
    axins.grid(False)
    ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.75)

fig.tight_layout()
#fig.savefig(out_dir / f'displacement_LR_vs_variance_{ref_state_type}_{prior_type}.png', dpi=300, bbox_inches='tight')
#fig.savefig(out_dir / f'displacement_LR_vs_variance_{ref_state_type}_{prior_type}.pdf', bbox_inches='tight')
#print(f"Saved: figs/displacement_LR_vs_variance_{ref_state_type}_{prior_type}.png")

# Fit log-log slopes in the small prior width region
print("\n" + "="*70)
print("Prior width scaling")
print("="*70)
fit_mask = (prior_var <= 10*prior_var[0]) & (r_linear > 10*eps)
if fit_mask.sum() >= 4:
    print(f"fitting {fit_mask.sum()} points over "
          f"{'sigma_0^2'} in [{prior_var[fit_mask][0]:.2e}, {prior_var[fit_mask][-1]:.2e}]")
    for label, r_arr in [('Linear   [I,Dx]', r_linear),
                         ('Cubic    [I,Dx,Dx^3]', r_cubic),
                         ('Quintic  [I,Dx,Dx^3,Dx^5]', r_quintic),
                         ('Homodyne (PM)', r_homodyne)]:
        slope = np.polyfit(np.log(prior_var[fit_mask]), np.log(r_arr[fit_mask]), 1)[0]
        print(f"  {label:<26}: slope = {slope:.3f}")
else:
    print(f"  too few points ({fit_mask.sum()}) for a slope fit; increase sigma_pts "
          f"or widen the range")

# L_R is non-monotonic: it peaks and then falls as the prior broadens towards the flat limit,
# where homodyne becomes optimal for displacement. Report where the peak sits.
print(f"\nPeak of L_R(homodyne, PM): sigma_0^2 = {prior_var[np.argmax(r_homodyne)]:.3f} "
      f"(sweep covers [{prior_var[0]:.3g}, {prior_var[-1]:.3g}])")

print(f"\nRelative MSL at the largest prior width (sigma_0^2 = {prior_var[-1]:.3f}):")
for label, r_arr in [('Prior', r_prior), ('Linear', r_linear), ('Cubic', r_cubic),
                     ('Quintic', r_quintic), ('Homodyne (PM)', r_homodyne)]:
    print(f"  {label:>14}: L_R = {r_arr[-1]:.3e}")

plt.show()