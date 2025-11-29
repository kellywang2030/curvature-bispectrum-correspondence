# ============================================================================
# APACHE LICENSE 2.0 HEADER TEMPLATE
# ============================================================================
# Add this header to the top of each Python script in the repository.
# ============================================================================

# Copyright 2025 Kelly Wang (Kelly.wang@ieee.org)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================================
# MODULE-SPECIFIC DOCSTRINGS
# ============================================================================
# Add the appropriate docstring after the license header for each module.
# ============================================================================

"""Paper Figure: Training Dynamics 3-Panel Visualization

Generates Figure 1 for the camera-ready paper, showing the evolution of
curvature-bispectrum correspondence through 10,000 training steps.

Panel (a): Correspondence stability (bound validity percentage)
Panel (b): Slope constant evolution (log-scale c_ell estimates)
Panel (c): Invariant magnitudes (curvature vs bispectral energy, log-scale)

Data source: Table 2 from Section 7.2 (Training Dynamics Analysis)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Training checkpoint data from NEW experimental results (Table 2)
checkpoint_steps = np.array([0, 100, 500, 1000, 2500, 5000, 10000])

# Correspondence rates from Table 2
correspondence_rates = np.array([100.0, 100.0, 96.7, 100.0, 100.0, 96.7, 93.3])

# Estimated slope constants (c_ell) from Table 2 - LOG SCALE needed
# Values span from 1.52e-10 to 2.87e-4
c_ell_values = np.array([1.52e-10, 8.21e-10, 2.81e-4, 2.39e-4, 1.46e-4, 2.87e-8, 4.94e-6])

# Euclidean curvature evolution from Table 2 (stays near 1e-4, then ~0)
# Using approximate values; "~0" interpreted as 1e-6 for log plotting
curvature_values = np.array([2e-4, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])

# Bispectral energy evolution from Table 2
energy_values = np.array([3.56e5, 2.82e4, 9.43e-2, 2.98e-2, 3.63e-2, 1.21e2, 1.06])

# Loss values from Table 2 (for reference, not plotted)
loss_values = np.array([np.nan, 468.0, 21.3, 8.3, 7.3, 7.0, 6.9])

# Create figure with three panels
fig = plt.figure(figsize=(14, 4.5))
gs = GridSpec(1, 3, figure=fig, wspace=0.3)

# Common styling
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Left Panel: Bound Validity
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(checkpoint_steps, correspondence_rates, 'o-', linewidth=2, 
         markersize=6, color='#2E86AB', label='Correspondence Rate')
ax1.axhline(y=90, color='red', linestyle='--', alpha=0.5, linewidth=1.5, 
            label='90% threshold')
ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Bound Validity (%)')
ax1.set_title('(a) Correspondence Stability')
ax1.set_ylim([85, 105])
ax1.set_xlim([-200, 10500])
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.legend(loc='lower right', framealpha=0.95)

# Format x-axis for thousands
ax1.set_xticks([0, 2500, 5000, 7500, 10000])
ax1.set_xticklabels(['0', '2.5k', '5k', '7.5k', '10k'])

# Middle Panel: c_ell Variation (LOG SCALE - varies by many orders of magnitude)
ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogy(checkpoint_steps, c_ell_values, 'o-', linewidth=2, 
             markersize=6, color='#A23B72')
ax2.set_xlabel('Training Steps')
ax2.set_ylabel('$\\hat{c}_\\ell$')
ax2.set_title('(b) Slope Constant Evolution')
ax2.set_xlim([-200, 10500])
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='both')

# Format x-axis for thousands
ax2.set_xticks([0, 2500, 5000, 7500, 10000])
ax2.set_xticklabels(['0', '2.5k', '5k', '7.5k', '10k'])

# Right Panel: Invariant Magnitudes (LOG SCALE for both)
ax3 = fig.add_subplot(gs[0, 2])

# Plot curvature (solid line)
ax3.semilogy(checkpoint_steps, curvature_values, 'o-', linewidth=2, 
             markersize=6, color='#F18F01', label='Curvature $\\|\\Omega\\|_F^2$')

# Plot bispectral energy (dashed line)
ax3.semilogy(checkpoint_steps, energy_values, 's--', linewidth=2, 
             markersize=6, color='#006494', label='Bispectral Energy $\\mathcal{E}$')

ax3.set_xlabel('Training Steps')
ax3.set_ylabel('Magnitude (log scale)')
ax3.set_title('(c) Invariant Magnitudes')
ax3.set_xlim([-200, 10500])
ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='both')
ax3.legend(loc='center right', framealpha=0.95)

# Format x-axis for thousands
ax3.set_xticks([0, 2500, 5000, 7500, 10000])
ax3.set_xticklabels(['0', '2.5k', '5k', '7.5k', '10k'])

# Overall title
fig.suptitle('Evolution of Correspondence Through Training (Euclidean Gauge-Orthogonal Curvature)', 
             fontsize=13, fontweight='bold', y=1.02)

# Save figure
plt.tight_layout()
plt.savefig('3-panel.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('3-panel.png', dpi=150, bbox_inches='tight', format='png')

plt.show()

# Print summary statistics for verification
print("\nTraining Stability Summary (New Data):")
print(f"Correspondence rate range: {np.min(correspondence_rates):.1f}% - {np.max(correspondence_rates):.1f}%")
print(f"c_ell range: {np.min(c_ell_values):.2e} to {np.max(c_ell_values):.2e}")
print(f"Energy decrease: {energy_values[0]:.2e} -> {energy_values[-1]:.2e} ({energy_values[0]/energy_values[-1]:.0f}x)")
print(f"Curvature: stays near {curvature_values[0]:.0e}")
