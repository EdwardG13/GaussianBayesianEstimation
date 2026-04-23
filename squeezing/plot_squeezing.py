"""
Loads data produced by generate_squeezing_data.py and produces the MSL ratio figure.

"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Reference state parameters
ref_state_type = 'thermal'  # Options: 'vacuum', 'coherent', 'thermal', 'squeezed_vacuum', or 'squeezed_thermal'
#ref_state_type = sys.argv[1] if len(sys.argv) > 1 else 'coherent'
stem     = f'squeezing_{ref_state_type}'

# Find paths
data_dir = Path(__file__).parent / 'data'
out_dir  = Path(__file__).parent / 'figs'
out_dir.mkdir(exist_ok=True)

# Load data
npz  = np.load(data_dir / f'{stem}.npz', allow_pickle=True)
with open(data_dir / f'{stem}.json') as f:
    meta = json.load(f)

# Unpack arrays
prior_var = npz['prior_variance']
msl_bayes = npz['msl_bayes']
msl_linear = npz['msl_linear']
msl_quad_hom = npz['msl_quad_hom']
msl_quad = npz['msl_quad']
msl_prior = npz['msl_prior']
msl_linear_bayes = npz['msl_linear_bayes']
msl_quad_hom_bayes = npz['msl_quad_hom_bayes']
msl_quad_bayes = npz['msl_quad_bayes']
msl_homodyne = npz['msl_homodyne']

# Ratios relative to Bayes-optimal
ratio = lambda arr: arr / msl_bayes - 1
r_prior = ratio(msl_prior)
r_linear = ratio(msl_linear)
r_linear_bayes = ratio(msl_linear_bayes)
r_quad_hom  = ratio(msl_quad_hom)
r_quad_hom_bayes = ratio(msl_quad_hom_bayes)
r_quad = ratio(msl_quad)
r_quad_bayes = ratio(msl_quad_bayes)
r_homodyne = ratio(msl_homodyne)

# Styles and colours

lw   = 8
fs   = 40
fs_t = 30

C_blue  = "#070f8b"
C_green = "#2ca02c"
C_black = "#000000"
C_grey  = "#959ba0"

# Figure 2 plot

fig, ax = plt.subplots(figsize=(14, 9))

ax.loglog(prior_var, r_prior, linestyle=':',  lw=lw,   color=C_grey,  label='Prior')
#ax.loglog(prior_var, r_linear,  linestyle='--', lw=lw,   color=C_blue,  label='Linear') # Remove the linear homodyne MSL for zero-mean probes (like vacuum and thermal) since these offer no improvement over the prior.
#ax.loglog(prior_var, r_linear_bayes, linestyle='-',  lw=lw,   color=C_blue,  label='Linear (PM)')
ax.loglog(prior_var, r_quad_hom, linestyle='--', lw=lw,   color=C_green, label='Quadratic homodyne')
#ax.loglog(prior_var, r_quad_hom_bayes, linestyle='-',  lw=lw,   color=C_green, label='Quadratic homodyne (PM)')
ax.loglog(prior_var, r_quad,  linestyle='--', lw=lw,   color=C_blue, label='Quadratic')
ax.loglog(prior_var, r_quad_bayes, linestyle='-',  lw=lw,   color=C_blue, label='Quadratic (PM)')

ax.loglog(prior_var, r_homodyne, linestyle='-',  lw=lw,   color=C_green, label='Homodyne (PM)') # Use this for Quadratic homodyne (PM) for a thermal probe (the numerical one is unstable).

ax.set_xlabel(r'$\sigma^2_0$',   fontsize=fs)
ax.set_ylabel(r'$\mathcal{L}_R$', fontsize=fs)
ax.tick_params(axis='both', which='major', length=20, width=3,   labelsize=fs_t)
ax.tick_params(axis='both', which='minor', length=12, width=2)
ax.grid(False)

### The insets locations and sizes have been chosen for the vacuum and thermal probe states. For others, it might not be aligned. 

# Inset: zoomed view for large sigma_0^2
"""
zoom_mask = prior_var >= 0.1
axins = ax.inset_axes([0.55, 0.08, 0.4, 0.35])

for r_arr, ls, c in [
    (r_prior,        ':',  C_grey),
    (r_linear,       '--', C_blue),
    (r_linear_bayes, '-',  C_blue),
    (r_quad_hom,         '--', C_green),
    (r_quad_hom_bayes,   '-',  C_green),
    (r_quad,        '--', C_black),
    (r_quad_bayes,  '-',  C_black),
]:
    axins.loglog(prior_var[zoom_mask], r_arr[zoom_mask], ls, lw=lw-2, color=c)

axins.set_xlim(prior_var[zoom_mask][0], prior_var[zoom_mask][-1])
axins.set_ylim(top=r_prior[zoom_mask][-1])
axins.tick_params(which='both', left=False, labelleft=False, bottom=False, labelbottom=False)
axins.grid(False)
ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.75)
"""


# Inset: MSL vs prior width
axins = ax.inset_axes([0.55, 0.15, 0.42, 0.35])
lw_inset = lw - 2

axins.plot(prior_var, msl_bayes, '-',  lw=lw_inset, color=C_black)
axins.plot(prior_var, msl_quad_hom, '--', lw=lw_inset, color=C_green)
axins.plot(prior_var, msl_quad,  '--', lw=lw_inset, color=C_blue)
axins.plot(prior_var, msl_quad_bayes, '-',  lw=lw_inset, color=C_blue)
axins.plot(prior_var, msl_homodyne, '-',  lw=lw_inset, color=C_green)
axins.plot(prior_var, msl_prior, ':',  lw=lw_inset, color=C_grey)

axins.set_xlim(prior_var[0], prior_var[-1])
axins.set_ylim(bottom=0)
axins.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune='upper'))
axins.set_ylabel(r'$\mathcal{L}$', fontsize=fs-4)
axins.tick_params(axis='both', which='major', length=14, width=3.5, labelsize=fs_t)
axins.grid(False)

fig.tight_layout()
fig.savefig(out_dir / f'squeezing_LR_vs_variance_{ref_state_type}.png', dpi=300, bbox_inches='tight')
fig.savefig(out_dir / f'squeezing_LR_vs_variance_{ref_state_type}.pdf', bbox_inches='tight')
print(f"Saved: figs/squeezing_LR_vs_variance_{ref_state_type}.png")

# Fit log-log slopes in the small sigma region
print("\n" + "="*70)
print("Prior width scaling")
print("="*70)
fit_mask = prior_var < 0.02  # small prior width region

for label, ratio in [('Quad homodyne [I,x_phi]', r_quad_hom), 
                    ('Quad homodyne [I,x_phi] (PM)', r_quad_hom_bayes), 
                    ('Quadratic [I,x²,p²]', r_quad),
                     ('Quadratic [I,x²,p²] (PM)', r_quad_bayes)]:
    coeffs = np.polyfit(np.log(prior_var[fit_mask]), 
                        np.log(ratio[fit_mask]), 1)
    print(f"{label}: slope = {coeffs[0]:.3f}")
plt.show()