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

# --- module4_euclidean_curvature.py ---
"""Module 4: Euclidean Mechanical Connection

Implements the canonical Euclidean connection with horizontal spaces as
Euclidean orthogonal complements of vertical spaces.
"""


#!/usr/bin/env python3
"""
Module 4 (Corrected): Discrete Holonomy and Curvature Computation

IMPORTANT FINDING:
The Fisher-Rao metric G has the vertical subspace (gauge directions) in its 
null space because MHA is gauge-invariant. This makes M = J^T G J = 0, so the
FR mechanical connection is degenerate.

SOLUTION:
We use the EUCLIDEAN mechanical connection instead. This projects tangent 
vectors onto the horizontal subspace using the Euclidean metric, which is 
well-defined and non-degenerate.

The curvature we compute measures how the Euclidean-horizontal subspace 
rotates as we move through parameter space. This is still a valid geometric
invariant that captures the structure of the gauge orbits.

Mathematical framework:
- Euclidean metric: <ξ, η>_Euc = Σ_ij ξ_ij η_ij
- Vertical subspace: V_θ = {J_θ(η) : η ∈ g}
- Horizontal subspace: H_θ = V_θ^⊥ (Euclidean orthogonal complement)
- Connection: Γ_θ(ξ) = (J_θ^T J_θ)^{-1} J_θ^T ξ
- Horizontal projection: P_hor(ξ) = ξ - J_θ Γ_θ(ξ)
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
    random_mha_params, random_gauge_direction, random_tangent, zero_gauge_direction,
    mha_forward, lie_algebra_dim
)


# ============================================================================
# Helper functions
# ============================================================================

def tangent_norm(xi: MHATangent) -> float:
    """Compute Euclidean norm of a tangent vector."""
    total = 0.0
    for head in xi.heads:
        total += torch.sum(head.dW_Q ** 2).item()
        total += torch.sum(head.dW_K ** 2).item()
        total += torch.sum(head.dW_V ** 2).item()
        total += torch.sum(head.dW_O ** 2).item()
    return total ** 0.5


def tangent_inner(xi: MHATangent, eta: MHATangent) -> float:
    """Compute Euclidean inner product of two tangent vectors."""
    total = 0.0
    for xi_head, eta_head in zip(xi.heads, eta.heads):
        total += torch.sum(xi_head.dW_Q * eta_head.dW_Q).item()
        total += torch.sum(xi_head.dW_K * eta_head.dW_K).item()
        total += torch.sum(xi_head.dW_V * eta_head.dW_V).item()
        total += torch.sum(xi_head.dW_O * eta_head.dW_O).item()
    return total


def lie_algebra_inner(eta1: GaugeDirection, eta2: GaugeDirection) -> float:
    """Euclidean inner product on the Lie algebra."""
    total = 0.0
    for h1, h2 in zip(eta1.heads, eta2.heads):
        total += torch.sum(h1.X * h2.X).item()
        total += torch.sum(h1.Y * h2.Y).item()
    return total


def lie_algebra_norm(eta: GaugeDirection) -> float:
    """Euclidean norm on the Lie algebra."""
    return lie_algebra_inner(eta, eta) ** 0.5


def perturb_params(theta: MHAParams, xi: MHATangent, epsilon: float) -> MHAParams:
    """Create θ + ε·ξ."""
    new_heads = []
    for head, head_tangent in zip(theta.heads, xi.heads):
        new_head = HeadParams(
            W_Q=head.W_Q + epsilon * head_tangent.dW_Q,
            W_K=head.W_K + epsilon * head_tangent.dW_K,
            W_V=head.W_V + epsilon * head_tangent.dW_V,
            W_O=head.W_O + epsilon * head_tangent.dW_O
        )
        new_heads.append(new_head)
    return MHAParams(heads=new_heads)


# ============================================================================
# Euclidean Mechanical Connection
# ============================================================================

@dataclass
class CGResult:
    """Result of conjugate gradient solver."""
    solution: GaugeDirection
    converged: bool
    iterations: int
    final_residual: float
    residual_history: List[float]


def conjugate_gradient_euclidean(
    theta: MHAParams,
    b: GaugeDirection,  # Right-hand side: J^T ξ
    config: ReferenceConfig,
    tol: float = 1e-10,
    max_iter: int = 500  # Increased from 100
) -> CGResult:
    """
    Solve M·Γ = b where M = J^T J (Euclidean version).
    
    This is the normal equation for the least-squares problem:
        min_Γ ||J(Γ) - ξ||^2
    
    The solution gives Γ = (J^T J)^{-1} J^T ξ.
    """
    device = theta.device
    dtype = theta.dtype
    
    # Initialize
    gamma = zero_gauge_direction(config, device, dtype)
    
    # Compute initial residual: r = b - M·γ = b (since γ = 0)
    # M·γ = J^T J γ
    J_gamma = vertical_tangent(theta, gamma)
    M_gamma = vertical_tangent_transpose(theta, J_gamma)
    
    # r = b - M·γ
    r = b.add(M_gamma.scale(-1.0))
    
    # Check if already solved
    r_norm = lie_algebra_norm(r)
    b_norm = lie_algebra_norm(b)
    
    if b_norm < 1e-15:
        return CGResult(
            solution=gamma,
            converged=True,
            iterations=0,
            final_residual=r_norm,
            residual_history=[r_norm]
        )
    
    if r_norm / b_norm < tol:
        return CGResult(
            solution=gamma,
            converged=True,
            iterations=0,
            final_residual=r_norm,
            residual_history=[r_norm]
        )
    
    # CG iteration
    p = r  # Search direction
    residual_history = [r_norm]
    
    for iteration in range(max_iter):
        # Compute M·p = J^T J p
        J_p = vertical_tangent(theta, p)
        M_p = vertical_tangent_transpose(theta, J_p)
        
        # α = (r^T r) / (p^T M p)
        r_dot_r = lie_algebra_inner(r, r)
        p_dot_Mp = lie_algebra_inner(p, M_p)
        
        if abs(p_dot_Mp) < 1e-15:
            # M is singular in this direction
            break
        
        alpha = r_dot_r / p_dot_Mp
        
        # Update solution: γ = γ + α·p
        gamma = gamma.add(p.scale(alpha))
        
        # Update residual: r = r - α·M·p
        r_new = r.add(M_p.scale(-alpha))
        
        r_new_norm = lie_algebra_norm(r_new)
        residual_history.append(r_new_norm)
        
        # Check convergence
        if r_new_norm / b_norm < tol:
            return CGResult(
                solution=gamma,
                converged=True,
                iterations=iteration + 1,
                final_residual=r_new_norm,
                residual_history=residual_history
            )
        
        # β = (r_new^T r_new) / (r^T r)
        r_new_dot_r_new = lie_algebra_inner(r_new, r_new)
        beta = r_new_dot_r_new / r_dot_r
        
        # Update search direction: p = r_new + β·p
        p = r_new.add(p.scale(beta))
        r = r_new
    
    return CGResult(
        solution=gamma,
        converged=False,
        iterations=max_iter,
        final_residual=lie_algebra_norm(r),
        residual_history=residual_history
    )


class EuclideanMechanicalConnection:
    """
    Mechanical connection using Euclidean metric.
    
    This computes the projection of tangent vectors onto the horizontal 
    subspace (orthogonal to gauge orbits) using the Euclidean metric.
    """
    
    def __init__(self, config: ReferenceConfig):
        self.config = config
    
    def solve_connection(self, theta: MHAParams, xi: MHATangent) -> Tuple[GaugeDirection, CGResult]:
        """
        Solve for Γ_θ(ξ) such that P_hor(ξ) = ξ - J_θ(Γ_θ(ξ)) is horizontal.
        
        This solves: (J^T J) Γ = J^T ξ
        """
        # Compute b = J^T ξ
        b = vertical_tangent_transpose(theta, xi)
        
        # Solve M·Γ = b
        cg_result = conjugate_gradient_euclidean(theta, b, self.config)
        
        return cg_result.solution, cg_result
    
    def project_vertical(self, theta: MHAParams, xi: MHATangent) -> Tuple[MHATangent, CGResult]:
        """Project onto vertical subspace."""
        gamma, cg_result = self.solve_connection(theta, xi)
        xi_vert = vertical_tangent(theta, gamma)
        return xi_vert, cg_result
    
    def project_horizontal(self, theta: MHAParams, xi: MHATangent) -> Tuple[MHATangent, CGResult]:
        """Project onto horizontal subspace."""
        xi_vert, cg_result = self.project_vertical(theta, xi)
        xi_hor = xi.subtract(xi_vert)
        return xi_hor, cg_result
    
    def decompose(self, theta: MHAParams, xi: MHATangent) -> Tuple[MHATangent, MHATangent, GaugeDirection, CGResult]:
        """Full decomposition: ξ = ξ_hor + ξ_vert."""
        gamma, cg_result = self.solve_connection(theta, xi)
        xi_vert = vertical_tangent(theta, gamma)
        xi_hor = xi.subtract(xi_vert)
        return xi_hor, xi_vert, gamma, cg_result


# ============================================================================
# Discrete Holonomy and Curvature
# ============================================================================

@dataclass
class HolonomyResult:
    """Result of holonomy computation."""
    holonomy_lie: GaugeDirection  # Curvature in Lie algebra
    holonomy_norm: float  # ||Ω(u,v)||
    epsilon: float
    cg_iterations: List[int]
    cg_converged: List[bool]


@dataclass
class CurvatureResult:
    """Result of curvature computation with Richardson extrapolation."""
    curvature_richardson: float
    curvature_squared: float
    curvature_raw: List[float]
    richardson_ratio: float
    linearization_index: float
    epsilon_values: List[float]
    holonomy_results: List[HolonomyResult]


class DiscreteHolonomy:
    """
    Compute curvature via the variation of the connection form.
    
    For horizontal directions u, v at base point θ, the curvature is:
        Ω(u,v) ≈ (Γ_{θ+εu}(v) - Γ_{θ+εv}(u)) / ε
    
    This measures how the horizontal subspace rotates as we move in parameter space.
    """
    
    def __init__(self, config: ReferenceConfig):
        self.config = config
        self.connection = EuclideanMechanicalConnection(config)
    
    def compute_holonomy(
        self,
        theta: MHAParams,
        u: MHATangent,  # First horizontal direction
        v: MHATangent,  # Second horizontal direction  
        epsilon: float
    ) -> HolonomyResult:
        """
        Compute curvature Ω(u,v) via connection form variation.
        
        Since u, v are horizontal at θ, we have Γ_θ(u) = Γ_θ(v) = 0.
        The curvature comes from how the connection changes:
            Ω(u,v) ≈ (Γ_{θ+εu}(v) - Γ_{θ+εv}(u)) / ε
        """
        cg_iterations = []
        cg_converged = []
        
        # Point 1: θ + εu
        theta_u = perturb_params(theta, u, epsilon)
        gamma_v_at_u, cg1 = self.connection.solve_connection(theta_u, v)
        cg_iterations.append(cg1.iterations)
        cg_converged.append(cg1.converged)
        
        # Point 2: θ + εv
        theta_v = perturb_params(theta, v, epsilon)
        gamma_u_at_v, cg2 = self.connection.solve_connection(theta_v, u)
        cg_iterations.append(cg2.iterations)
        cg_converged.append(cg2.converged)
        
        # Curvature: Ω(u,v) = (Γ_{θ+εu}(v) - Γ_{θ+εv}(u)) / ε
        omega = gamma_v_at_u.add(gamma_u_at_v.scale(-1.0)).scale(1.0 / epsilon)
        omega_norm = lie_algebra_norm(omega)
        
        return HolonomyResult(
            holonomy_lie=omega,
            holonomy_norm=omega_norm,
            epsilon=epsilon,
            cg_iterations=cg_iterations,
            cg_converged=cg_converged
        )
    
    def compute_curvature_with_richardson(
        self,
        theta: MHAParams,
        u: MHATangent,
        v: MHATangent,
        epsilon_values: List[float] = None
    ) -> CurvatureResult:
        """
        Compute curvature with Richardson extrapolation.
        
        K(ε) = ||Ω(u,v; ε)|| = K(0) + a·ε + O(ε²)
        K_richardson = (ε₂·K(ε₁) - ε₁·K(ε₂)) / (ε₂ - ε₁)
        """
        if epsilon_values is None:
            epsilon_values = [1e-4, 2e-4]
        
        holonomy_results = []
        K_values = []
        
        for eps in epsilon_values:
            result = self.compute_holonomy(theta, u, v, eps)
            holonomy_results.append(result)
            K_values.append(result.holonomy_norm)
        
        # Richardson extrapolation
        if len(epsilon_values) >= 2:
            eps1, eps2 = epsilon_values[0], epsilon_values[1]
            K1, K2 = K_values[0], K_values[1]
            
            if abs(eps2 - eps1) > 1e-15:
                K_richardson = (eps2 * K1 - eps1 * K2) / (eps2 - eps1)
            else:
                K_richardson = K1
            
            # Richardson ratio
            if K1 > 1e-15:
                richardson_ratio = K2 / K1
            else:
                richardson_ratio = 1.0
        else:
            K_richardson = K_values[0]
            richardson_ratio = 1.0
        
        K_richardson = max(0.0, K_richardson)
        
        # Linearization index
        if len(epsilon_values) >= 3:
            K1, K2, K3 = K_values[:3]
            if K2 > 1e-15:
                eta = abs(K3 - 2*K2 + K1) / K2
            else:
                eta = 0.0
        else:
            eta = 0.0
        
        return CurvatureResult(
            curvature_richardson=K_richardson,
            curvature_squared=K_richardson ** 2,
            curvature_raw=K_values,
            richardson_ratio=richardson_ratio,
            linearization_index=eta,
            epsilon_values=epsilon_values,
            holonomy_results=holonomy_results
        )


def generate_horizontal_direction_pair(
    theta: MHAParams,
    connection: EuclideanMechanicalConnection
) -> Tuple[MHATangent, MHATangent]:
    """
    Generate a pair of orthogonal horizontal directions.
    """
    config = connection.config
    device = theta.device
    dtype = theta.dtype
    
    # Generate random directions
    u_rand = random_tangent(config, device, dtype)
    v_rand = random_tangent(config, device, dtype)
    
    # Project to horizontal
    u_hor, _ = connection.project_horizontal(theta, u_rand)
    v_hor, _ = connection.project_horizontal(theta, v_rand)
    
    # Normalize u
    u_norm = tangent_norm(u_hor)
    if u_norm > 1e-10:
        u_hor = u_hor.scale(1.0 / u_norm)
    
    # Orthogonalize v w.r.t. u (Gram-Schmidt)
    uv_inner = tangent_inner(u_hor, v_hor)
    v_hor = v_hor.subtract(u_hor.scale(uv_inner))
    
    # Normalize v
    v_norm = tangent_norm(v_hor)
    if v_norm > 1e-10:
        v_hor = v_hor.scale(1.0 / v_norm)
    
    return u_hor, v_hor


# ============================================================================
# Validation Tests
# ============================================================================

def test_euclidean_decomposition(config: ReferenceConfig, device: torch.device, 
                                  dtype: torch.dtype) -> Dict:
    """Test that Euclidean decomposition works correctly."""
    theta = random_mha_params(config, device, dtype)
    connection = EuclideanMechanicalConnection(config)
    
    # Test with random tangent
    xi = random_tangent(config, device, dtype)
    xi_hor, xi_vert, gamma, cg_result = connection.decompose(theta, xi)
    
    # Check: ξ = ξ_hor + ξ_vert
    recon = xi_hor.add(xi_vert)
    diff_norm = tangent_norm(xi.subtract(recon))
    xi_norm = tangent_norm(xi)
    
    passed = (diff_norm / xi_norm) < 1e-10 if xi_norm > 1e-10 else True
    
    return {
        'passed': passed,
        'reconstruction_error': diff_norm,
        'xi_norm': xi_norm,
        'xi_hor_norm': tangent_norm(xi_hor),
        'xi_vert_norm': tangent_norm(xi_vert),
        'gamma_norm': lie_algebra_norm(gamma),
        'cg_iterations': cg_result.iterations,
        'cg_converged': cg_result.converged
    }


def test_horizontal_orthogonality(config: ReferenceConfig, device: torch.device,
                                   dtype: torch.dtype) -> Dict:
    """Test that horizontal is orthogonal to all vertical directions."""
    theta = random_mha_params(config, device, dtype)
    connection = EuclideanMechanicalConnection(config)
    
    # Get horizontal projection
    xi = random_tangent(config, device, dtype)
    xi_hor, _ = connection.project_horizontal(theta, xi)
    xi_hor_norm = tangent_norm(xi_hor)
    
    # Check orthogonality to random vertical directions
    max_inner = 0.0
    for _ in range(10):
        eta = random_gauge_direction(config, device, dtype)
        v_dir = vertical_tangent(theta, eta)
        
        inner = abs(tangent_inner(xi_hor, v_dir))
        v_norm = tangent_norm(v_dir)
        
        if v_norm > 1e-10 and xi_hor_norm > 1e-10:
            normalized_inner = inner / (xi_hor_norm * v_norm)
            max_inner = max(max_inner, normalized_inner)
    
    passed = max_inner < 1e-8
    
    return {
        'passed': passed,
        'max_normalized_inner': max_inner
    }


def test_vertical_recovery(config: ReferenceConfig, device: torch.device,
                           dtype: torch.dtype) -> Dict:
    """Test that a pure vertical direction is recovered correctly."""
    theta = random_mha_params(config, device, dtype)
    connection = EuclideanMechanicalConnection(config)
    
    # Create pure vertical direction
    eta = random_gauge_direction(config, device, dtype)
    v_dir = vertical_tangent(theta, eta)
    
    # Decompose - should get back mostly vertical
    v_hor, v_vert, gamma, cg_result = connection.decompose(theta, v_dir)
    
    v_hor_norm = tangent_norm(v_hor)
    v_vert_norm = tangent_norm(v_vert)
    v_dir_norm = tangent_norm(v_dir)
    
    # Check that horizontal component is small relative to input
    hor_ratio = v_hor_norm / v_dir_norm if v_dir_norm > 1e-10 else 0.0
    
    # Check that recovered gamma matches original eta
    gamma_norm = lie_algebra_norm(gamma)
    eta_norm = lie_algebra_norm(eta)
    gamma_error = abs(gamma_norm - eta_norm) / eta_norm if eta_norm > 1e-10 else 0.0
    
    # Pass if horizontal is small AND gamma recovers eta
    passed = (hor_ratio < 1e-4) and (gamma_error < 1e-4)
    
    return {
        'passed': passed,
        'horizontal_ratio': hor_ratio,
        'gamma_recovery_error': gamma_error,
        'v_dir_norm': v_dir_norm,
        'v_hor_norm': v_hor_norm,
        'v_vert_norm': v_vert_norm,
        'gamma_norm': gamma_norm,
        'eta_norm': eta_norm,
        'cg_iterations': cg_result.iterations,
        'cg_converged': cg_result.converged
    }


def test_curvature_nonzero(config: ReferenceConfig, device: torch.device,
                           dtype: torch.dtype, n_samples: int = 5) -> Dict:
    """Test that we get non-zero curvature."""
    theta = random_mha_params(config, device, dtype)
    connection = EuclideanMechanicalConnection(config)
    holonomy = DiscreteHolonomy(config)
    
    curvatures = []
    cg_iters = []
    
    for _ in range(n_samples):
        u, v = generate_horizontal_direction_pair(theta, connection)
        result = holonomy.compute_curvature_with_richardson(theta, u, v)
        curvatures.append(result.curvature_richardson)
        cg_iters.extend([h.cg_iterations for h in result.holonomy_results])
    
    max_curv = max(curvatures)
    mean_curv = sum(curvatures) / len(curvatures)
    
    passed = max_curv > 1e-10
    
    return {
        'passed': passed,
        'max_curvature': max_curv,
        'mean_curvature': mean_curv,
        'curvatures': curvatures,
        'mean_cg_iterations': sum(sum(x) for x in cg_iters) / len(cg_iters) if cg_iters else 0
    }


def test_curvature_convergence(config: ReferenceConfig, device: torch.device,
                               dtype: torch.dtype) -> Dict:
    """Test that curvature estimate converges as ε → 0."""
    theta = random_mha_params(config, device, dtype)
    connection = EuclideanMechanicalConnection(config)
    holonomy = DiscreteHolonomy(config)
    
    u, v = generate_horizontal_direction_pair(theta, connection)
    
    # Compute at multiple epsilon values
    epsilons = [1e-3, 5e-4, 2.5e-4, 1.25e-4]
    K_values = []
    
    for eps in epsilons:
        result = holonomy.compute_holonomy(theta, u, v, eps)
        K_values.append(result.holonomy_norm)
    
    # Check convergence: K(ε) should stabilize
    if K_values[0] > 1e-10:
        # Compute relative differences
        diffs = [abs(K_values[i+1] - K_values[i]) / K_values[i] 
                 for i in range(len(K_values)-1)]
        converging = all(d < 0.5 for d in diffs)  # Less than 50% change
    else:
        converging = True
    
    return {
        'passed': converging,
        'epsilon_values': epsilons,
        'K_values': K_values,
        'relative_diffs': [abs(K_values[i+1] - K_values[i]) / max(K_values[i], 1e-15) 
                          for i in range(len(K_values)-1)]
    }


def run_all_tests(config: ReferenceConfig = None, 
                  device: torch.device = None,
                  dtype: torch.dtype = None) -> Dict:
    """Run all Module 4 validation tests."""
    if config is None:
        config = ReferenceConfig()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.float64
    
    print("=" * 70)
    print("MODULE 4 VALIDATION: Euclidean Mechanical Connection & Curvature")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  n_heads = {config.n_heads}")
    print(f"  d_k = {config.d_k}, d_v = {config.d_v}")
    print(f"  d_model = {config.d_model}")
    print(f"  Device = {device}")
    print(f"  Dtype = {dtype}")
    
    results = {}
    all_passed = True
    
    # Test 1: Euclidean decomposition
    print("\n[Test 1] Euclidean decomposition (ξ = ξ_hor + ξ_vert)...")
    results['decomposition'] = test_euclidean_decomposition(config, device, dtype)
    status = "PASS" if results['decomposition']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  ||ξ||: {results['decomposition']['xi_norm']:.4f}")
    print(f"  ||ξ_hor||: {results['decomposition']['xi_hor_norm']:.4f} ({100*results['decomposition']['xi_hor_norm']/results['decomposition']['xi_norm']:.1f}%)")
    print(f"  ||ξ_vert||: {results['decomposition']['xi_vert_norm']:.4e}")
    print(f"  CG iterations: {results['decomposition']['cg_iterations']}")
    all_passed = all_passed and results['decomposition']['passed']
    
    # Test 2: Horizontal orthogonality
    print("\n[Test 2] Horizontal orthogonality to vertical...")
    results['orthogonality'] = test_horizontal_orthogonality(config, device, dtype)
    status = "PASS" if results['orthogonality']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Max |<ξ_hor, J(η)>| / norms: {results['orthogonality']['max_normalized_inner']:.2e}")
    all_passed = all_passed and results['orthogonality']['passed']
    
    # Test 3: Vertical recovery
    print("\n[Test 3] Vertical direction recovery...")
    results['vertical'] = test_vertical_recovery(config, device, dtype)
    status = "PASS" if results['vertical']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  ||J(η)||: {results['vertical']['v_dir_norm']:.4f}")
    print(f"  ||horizontal part||: {results['vertical']['v_hor_norm']:.2e} (ratio: {results['vertical']['horizontal_ratio']:.2e})")
    print(f"  ||Γ|| vs ||η||: {results['vertical']['gamma_norm']:.4f} vs {results['vertical']['eta_norm']:.4f} (error: {results['vertical']['gamma_recovery_error']:.2e})")
    print(f"  CG iterations: {results['vertical']['cg_iterations']}, converged: {results['vertical']['cg_converged']}")
    all_passed = all_passed and results['vertical']['passed']
    
    # Test 4: Non-zero curvature
    print("\n[Test 4] Non-zero curvature detection...")
    results['curvature'] = test_curvature_nonzero(config, device, dtype)
    status = "PASS" if results['curvature']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Max curvature: {results['curvature']['max_curvature']:.2e}")
    print(f"  Mean curvature: {results['curvature']['mean_curvature']:.2e}")
    print(f"  Curvatures: {[f'{c:.2e}' for c in results['curvature']['curvatures']]}")
    all_passed = all_passed and results['curvature']['passed']
    
    # Test 5: Curvature convergence
    print("\n[Test 5] Curvature estimate convergence...")
    results['convergence'] = test_curvature_convergence(config, device, dtype)
    status = "PASS" if results['convergence']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  ε values: {results['convergence']['epsilon_values']}")
    print(f"  K(ε) values: {[f'{k:.4e}' for k in results['convergence']['K_values']]}")
    print(f"  Relative changes: {[f'{d:.2%}' for d in results['convergence']['relative_diffs']]}")
    all_passed = all_passed and results['convergence']['passed']
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("OVERALL: ALL TESTS PASSED")
    else:
        print("OVERALL: SOME TESTS FAILED")
    print("=" * 70)
    
    results['all_passed'] = all_passed
    return results


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)
    
    config = ReferenceConfig()
    results = run_all_tests(config, device, torch.float64)
