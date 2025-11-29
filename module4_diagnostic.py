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

# --- module4_diagnostic.py ---
"""Module 4: Numerical Diagnostics

Implements Richardson ratio computation and other numerical diagnostics
for validating the linearized regime.
"""

#!/usr/bin/env python3
"""
Module 4 Diagnostic: Understand why curvature is zero.

This script investigates:
1. Whether vertical directions have non-trivial FR inner products with tangents
2. Whether the mechanical connection produces non-zero Γ for non-horizontal directions
3. Whether the Jacobian J_θ of the MHA function is non-degenerate
4. How the horizontal subspace changes with θ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import sys

# Import from previous modules
from module1_gauge_algebra import (
    ReferenceConfig, HeadParams, MHAParams, HeadLieAlgebra, GaugeDirection,
    HeadTangent, MHATangent, vertical_tangent, vertical_tangent_transpose,
    random_mha_params, random_gauge_direction, random_tangent,
    mha_forward
)


def tangent_norm(xi: MHATangent) -> float:
    """Compute Euclidean norm of a tangent vector."""
    total = 0.0
    for head in xi.heads:
        total += torch.sum(head.dW_Q ** 2).item()
        total += torch.sum(head.dW_K ** 2).item()
        total += torch.sum(head.dW_V ** 2).item()
        total += torch.sum(head.dW_O ** 2).item()
    return total ** 0.5

from module2_fisher_rao_metric import (
    FisherRaoMetric, compute_fr_adjoint_J, compute_M_operator
)

from module3_mechanical_connection import (
    MechanicalConnection, lie_algebra_norm, lie_algebra_inner
)


def diagnose_geometry(config: ReferenceConfig, device: torch.device, dtype: torch.dtype):
    """Run comprehensive diagnostics on the geometric structure."""
    
    print("="*70)
    print("GEOMETRIC STRUCTURE DIAGNOSTICS")
    print("="*70)
    
    # Create random parameter point
    theta = random_mha_params(config, device, dtype)
    
    # Create Fisher-Rao metric
    batch_size = 64
    seq_len = 32
    torch.manual_seed(42)
    eval_batch = torch.randn(batch_size, seq_len, config.d_model, device=device, dtype=dtype)
    fr_metric = FisherRaoMetric(config, eval_batch)
    
    # Create mechanical connection
    connection = MechanicalConnection(config, fr_metric)
    
    # =================================================================
    # Diagnostic 1: Vertical direction norms and FR inner products
    # =================================================================
    print("\n[Diagnostic 1] Vertical direction structure")
    print("-" * 50)
    
    # Generate several random gauge directions
    n_samples = 5
    vertical_euclidean_norms = []
    vertical_fr_norms = []
    
    for i in range(n_samples):
        eta = random_gauge_direction(config, device, dtype)
        v_tangent = vertical_tangent(theta, eta)
        
        # Euclidean norm
        euc_norm = tangent_norm(v_tangent)
        vertical_euclidean_norms.append(euc_norm)
        
        # FR norm
        fr_norm = fr_metric.norm(theta, v_tangent).item()
        vertical_fr_norms.append(fr_norm)
    
    print(f"  Vertical directions (n={n_samples}):")
    print(f"    Euclidean norms: {[f'{x:.4f}' for x in vertical_euclidean_norms]}")
    print(f"    FR norms:        {[f'{x:.4f}' for x in vertical_fr_norms]}")
    print(f"    Ratio FR/Euc:    {[f'{fr/euc:.4f}' for fr, euc in zip(vertical_fr_norms, vertical_euclidean_norms)]}")
    
    # =================================================================
    # Diagnostic 2: Random tangent decomposition
    # =================================================================
    print("\n[Diagnostic 2] Random tangent decomposition")
    print("-" * 50)
    
    for i in range(3):
        xi = random_tangent(config, device, dtype)
        xi_euc_norm = tangent_norm(xi)
        
        # Decompose
        xi_hor, xi_vert, gamma, cg_result = connection.decompose(theta, xi)
        
        xi_hor_euc = tangent_norm(xi_hor)
        xi_vert_euc = tangent_norm(xi_vert)
        gamma_norm = lie_algebra_norm(gamma)
        
        print(f"  Sample {i+1}:")
        print(f"    ||ξ||_Euc = {xi_euc_norm:.6f}")
        print(f"    ||ξ_hor||_Euc = {xi_hor_euc:.6f} ({100*xi_hor_euc/xi_euc_norm:.1f}%)")
        print(f"    ||ξ_vert||_Euc = {xi_vert_euc:.6e} ({100*xi_vert_euc/xi_euc_norm:.4f}%)")
        print(f"    ||Γ(ξ)||_Lie = {gamma_norm:.6e}")
        print(f"    CG iterations: {cg_result.iterations}, residual: {cg_result.final_residual:.2e}")
    
    # =================================================================
    # Diagnostic 3: Decomposition of a VERTICAL direction
    # =================================================================
    print("\n[Diagnostic 3] Decomposition of pure vertical direction")
    print("-" * 50)
    
    eta = random_gauge_direction(config, device, dtype)
    v_tangent = vertical_tangent(theta, eta)
    v_euc_norm = tangent_norm(v_tangent)
    
    v_hor, v_vert, gamma_v, cg_result = connection.decompose(theta, v_tangent)
    
    v_hor_euc = tangent_norm(v_hor)
    v_vert_euc = tangent_norm(v_vert)
    gamma_v_norm = lie_algebra_norm(gamma_v)
    
    # Compare recovered gamma with original eta
    eta_norm = lie_algebra_norm(eta)
    
    print(f"  Input: vertical direction J(η) with ||η||_Lie = {eta_norm:.6f}")
    print(f"  ||J(η)||_Euc = {v_euc_norm:.6f}")
    print(f"  After decomposition:")
    print(f"    ||hor||_Euc = {v_hor_euc:.6e} (should be ~0)")
    print(f"    ||vert||_Euc = {v_vert_euc:.6f} (should equal input)")
    print(f"    ||Γ||_Lie = {gamma_v_norm:.6f} (should equal ||η||)")
    print(f"    Relative error in Γ recovery: {abs(gamma_v_norm - eta_norm)/eta_norm:.6e}")
    print(f"    CG iterations: {cg_result.iterations}, residual: {cg_result.final_residual:.2e}")
    
    # =================================================================
    # Diagnostic 4: M operator eigenvalues
    # =================================================================
    print("\n[Diagnostic 4] M operator structure (M = J* G J)")
    print("-" * 50)
    
    # Compute M·η for several random η and check magnitudes
    M_norms = []
    eta_norms = []
    for i in range(5):
        eta = random_gauge_direction(config, device, dtype)
        M_eta = compute_M_operator(theta, eta, fr_metric)
        
        eta_norms.append(lie_algebra_norm(eta))
        M_norms.append(lie_algebra_norm(M_eta))
    
    print(f"  ||η||_Lie:   {[f'{x:.4f}' for x in eta_norms]}")
    print(f"  ||Mη||_Lie:  {[f'{x:.4f}' for x in M_norms]}")
    print(f"  Ratio ||Mη||/||η||: {[f'{m/e:.4f}' for m, e in zip(M_norms, eta_norms)]}")
    
    # =================================================================
    # Diagnostic 5: How does horizontal subspace change with θ?
    # =================================================================
    print("\n[Diagnostic 5] Horizontal subspace variation with θ")
    print("-" * 50)
    
    # Create a tangent direction
    xi = random_tangent(config, device, dtype)
    xi_hor_base, _, _, _ = connection.decompose(theta, xi)
    
    # Perturb theta and re-decompose the SAME xi
    u = random_tangent(config, device, dtype)
    u_hor, _, _, _ = connection.decompose(theta, u)
    u_hor_normalized = u_hor.scale(1.0 / tangent_norm(u_hor))
    
    epsilons = [1e-4, 1e-3, 1e-2, 1e-1]
    print(f"  Base horizontal component ||ξ_hor||: {tangent_norm(xi_hor_base):.6f}")
    
    for eps in epsilons:
        # Perturb theta
        theta_pert = perturb_params(theta, u_hor_normalized, eps)
        
        # Create new metric and connection at perturbed point
        fr_metric_pert = FisherRaoMetric(config, eval_batch)
        connection_pert = MechanicalConnection(config, fr_metric_pert)
        
        # Decompose the SAME tangent xi at the new point
        xi_hor_pert, xi_vert_pert, gamma_pert, _ = connection_pert.decompose(theta_pert, xi)
        
        # Measure how much the horizontal component changed
        diff = xi_hor_pert.subtract(xi_hor_base)
        diff_norm = tangent_norm(diff)
        
        gamma_diff_norm = lie_algebra_norm(gamma_pert)
        
        print(f"  ε={eps:.0e}: ||Γ_{θ+εu}(ξ)||_Lie = {gamma_diff_norm:.6e}, ||ξ_hor change|| = {diff_norm:.6e}")
    
    # =================================================================
    # Diagnostic 6: Curvature via explicit formula
    # =================================================================
    print("\n[Diagnostic 6] Curvature via connection variation")
    print("-" * 50)
    
    # Generate two horizontal directions at base point
    u = random_tangent(config, device, dtype)
    v = random_tangent(config, device, dtype)
    
    u_hor, _, _, _ = connection.decompose(theta, u)
    v_hor, _, _, _ = connection.decompose(theta, v)
    
    # Normalize
    u_hor = u_hor.scale(1.0 / tangent_norm(u_hor))
    v_hor = v_hor.scale(1.0 / tangent_norm(v_hor))
    
    # At base point, Γ_θ(u_hor) and Γ_θ(v_hor) should be ~0
    _, _, gamma_u_base, _ = connection.decompose(theta, u_hor)
    _, _, gamma_v_base, _ = connection.decompose(theta, v_hor)
    
    print(f"  At base θ:")
    print(f"    ||Γ_θ(u_hor)||_Lie = {lie_algebra_norm(gamma_u_base):.6e} (should be ~0)")
    print(f"    ||Γ_θ(v_hor)||_Lie = {lie_algebra_norm(gamma_v_base):.6e} (should be ~0)")
    
    # Perturb in direction u and compute Γ_{θ+εu}(v_hor)
    eps = 1e-3
    theta_u = perturb_params(theta, u_hor, eps)
    fr_metric_u = FisherRaoMetric(config, eval_batch)
    connection_u = MechanicalConnection(config, fr_metric_u)
    
    _, _, gamma_v_at_u, cg1 = connection_u.decompose(theta_u, v_hor)
    
    # Perturb in direction v and compute Γ_{θ+εv}(u_hor)
    theta_v = perturb_params(theta, v_hor, eps)
    fr_metric_v = FisherRaoMetric(config, eval_batch)
    connection_v = MechanicalConnection(config, fr_metric_v)
    
    _, _, gamma_u_at_v, cg2 = connection_v.decompose(theta_v, u_hor)
    
    print(f"\n  At perturbed points (ε={eps}):")
    print(f"    ||Γ_{{θ+εu}}(v_hor)||_Lie = {lie_algebra_norm(gamma_v_at_u):.6e}")
    print(f"    ||Γ_{{θ+εv}}(u_hor)||_Lie = {lie_algebra_norm(gamma_u_at_v):.6e}")
    print(f"    CG iterations: {cg1.iterations}, {cg2.iterations}")
    
    # The curvature estimate
    diff = gamma_v_at_u.add(gamma_u_at_v.scale(-1.0))
    curvature_estimate = lie_algebra_norm(diff) / eps
    
    print(f"\n  Curvature estimate ||Ω(u,v)|| ≈ {curvature_estimate:.6e}")
    
    # =================================================================
    # Diagnostic 7: Check if we're in a flat region
    # =================================================================
    print("\n[Diagnostic 7] Testing with non-horizontal directions")
    print("-" * 50)
    
    # Use a mix of horizontal and vertical
    eta1 = random_gauge_direction(config, device, dtype)
    eta2 = random_gauge_direction(config, device, dtype)
    
    # Create directions with vertical components
    mixed1 = u_hor.add(vertical_tangent(theta, eta1).scale(0.5))
    mixed2 = v_hor.add(vertical_tangent(theta, eta2).scale(0.5))
    
    # Decompose at perturbed points
    theta_m1 = perturb_params(theta, mixed1, eps)
    fr_m1 = FisherRaoMetric(config, eval_batch)
    conn_m1 = MechanicalConnection(config, fr_m1)
    _, _, gamma_m2_at_m1, _ = conn_m1.decompose(theta_m1, mixed2)
    
    theta_m2 = perturb_params(theta, mixed2, eps)
    fr_m2 = FisherRaoMetric(config, eval_batch)
    conn_m2 = MechanicalConnection(config, fr_m2)
    _, _, gamma_m1_at_m2, _ = conn_m2.decompose(theta_m2, mixed1)
    
    print(f"  Using mixed directions (hor + 0.5*vert):")
    print(f"    ||Γ_{{θ+εξ1}}(ξ2)||_Lie = {lie_algebra_norm(gamma_m2_at_m1):.6e}")
    print(f"    ||Γ_{{θ+εξ2}}(ξ1)||_Lie = {lie_algebra_norm(gamma_m1_at_m2):.6e}")
    
    diff_mixed = gamma_m2_at_m1.add(gamma_m1_at_m2.scale(-1.0))
    print(f"    ||difference||/ε = {lie_algebra_norm(diff_mixed)/eps:.6e}")


def perturb_params(theta: MHAParams, xi: MHATangent, epsilon: float) -> MHAParams:
    """Create θ + ε·ξ."""
    new_heads = []
    for i, (head, head_tangent) in enumerate(zip(theta.heads, xi.heads)):
        new_head = HeadParams(
            W_Q=head.W_Q + epsilon * head_tangent.dW_Q,
            W_K=head.W_K + epsilon * head_tangent.dW_K,
            W_V=head.W_V + epsilon * head_tangent.dW_V,
            W_O=head.W_O + epsilon * head_tangent.dW_O
        )
        new_heads.append(new_head)
    return MHAParams(heads=new_heads)


if __name__ == "__main__":
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    config = ReferenceConfig()
    
    print(f"Using device: {device}")
    
    torch.manual_seed(12345)
    if device.type == 'cuda':
        torch.cuda.manual_seed(12345)
    
    diagnose_geometry(config, device, dtype)
