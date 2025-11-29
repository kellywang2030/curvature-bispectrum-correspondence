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

# --- module3_mechanical_connection.py ---
"""Module 3: Mechanical Connection (Fisher-Rao)

Implements the Fisher-Rao mechanical connection via normal equations.
Note: Included for theoretical completeness; empirical results use Module 4.
"""


#!/usr/bin/env python3
"""
Module 3: Mechanical Connection Solver

This module implements the Fisher-Rao mechanical connection, which provides
the horizontal-vertical decomposition of tangent vectors essential for
computing curvature on the quotient manifold.

Mathematical Background:
    The mechanical connection Γ_θ : T_θΘ → g_max decomposes any tangent vector
    ξ into vertical and horizontal components:
    
        ξ = ξ_vert + ξ_hor = J_θ(Γ_θ(ξ)) + P_hor(ξ)
    
    where:
        - ξ_vert = J_θ(Γ_θ(ξ)) is tangent to the gauge orbit
        - ξ_hor = P_hor(ξ) is FR-orthogonal to the orbit
    
    The connection 1-form Γ_θ(ξ) is determined by requiring ξ_hor to be
    FR-orthogonal to all vertical directions:
    
        ⟨ξ - J_θ(Γ_θ(ξ)), J_θ(η)⟩_FR = 0  for all η ∈ g_max
    
    This gives the mechanical connection equation:
        M_θ · Γ_θ(ξ) = b_θ(ξ)
    
    where:
        M_θ = J_θ^* G_θ J_θ  (pulled-back metric on Lie algebra)
        b_θ(ξ) = J_θ^* G_θ ξ  (projection of ξ onto vertical via FR metric)
    
    We solve this via conjugate gradient, which only requires M_θ-vector products.

Key Operations:
    1. solve_connection(θ, ξ) → Γ_θ(ξ)  (solve for gauge component)
    2. project_horizontal(θ, ξ) → ξ_hor  (project to horizontal)
    3. project_vertical(θ, ξ) → ξ_vert  (project to vertical)

Reference Configuration:
    h = 4 heads, d_k = d_v = 8, d_model = 32
    Lie algebra dimension: 512
    CG tolerance: 1e-10, max iterations: 100

Dependencies:
    - Module 1: Data structures, vertical_tangent, vertical_tangent_transpose
    - Module 2: FisherRaoMetric, compute_M_operator, compute_fr_adjoint_J

Author: Research validation implementation
Date: 2025
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable
import numpy as np

# Import from Modules 1 and 2
from module1_gauge_algebra import (
    ReferenceConfig, DTYPE,
    HeadParams, MHAParams, HeadTangent, MHATangent,
    GaugeDirection, HeadLieAlgebra,
    vertical_tangent, vertical_tangent_transpose,
    random_mha_params, random_tangent, random_gauge_direction,
    zero_tangent, zero_gauge_direction, mha_forward
)

from module2_fisher_rao_metric import (
    FisherRaoMetric, create_eval_batch,
    compute_M_operator, compute_fr_adjoint_J
)


# =============================================================================
# Conjugate Gradient Solver for Lie Algebra
# =============================================================================

@dataclass
class CGResult:
    """Result of conjugate gradient solve."""
    solution: GaugeDirection
    converged: bool
    iterations: int
    final_residual: float
    residual_history: List[float]


def lie_algebra_inner(g1: GaugeDirection, g2: GaugeDirection) -> float:
    """Compute Euclidean inner product on Lie algebra."""
    total = 0.0
    for h1, h2 in zip(g1.heads, g2.heads):
        total += torch.sum(h1.X * h2.X).item()
        total += torch.sum(h1.Y * h2.Y).item()
    return total


def lie_algebra_norm(g: GaugeDirection) -> float:
    """Compute Euclidean norm on Lie algebra."""
    return np.sqrt(lie_algebra_inner(g, g))


def conjugate_gradient_solve(
    M_apply: Callable[[GaugeDirection], GaugeDirection],
    b: GaugeDirection,
    config: ReferenceConfig,
    x0: Optional[GaugeDirection] = None,
    tol: float = 1e-10,
    max_iter: int = 100,
    verbose: bool = False
) -> CGResult:
    """
    Solve M·x = b using conjugate gradient method.
    
    Args:
        M_apply: Function that computes M·x for any x
        b: Right-hand side
        config: Reference configuration
        x0: Initial guess (zero if None)
        tol: Convergence tolerance (relative residual)
        max_iter: Maximum iterations
        verbose: Print progress
    
    Returns:
        CGResult with solution and convergence info
    """
    device = b.heads[0].X.device
    dtype = b.heads[0].X.dtype
    
    # Initial guess
    if x0 is None:
        x = zero_gauge_direction(config, device, dtype)
    else:
        x = x0
    
    # Initial residual: r = b - M·x
    Mx = M_apply(x)
    r = b.add(Mx.scale(-1.0))
    
    # Initial search direction
    p = GaugeDirection(heads=[
        HeadLieAlgebra(X=h.X.clone(), Y=h.Y.clone()) for h in r.heads
    ])
    
    # Initial residual norm
    r_norm_sq = lie_algebra_inner(r, r)
    b_norm = lie_algebra_norm(b)
    
    if b_norm < 1e-15:
        # b is zero, solution is zero
        return CGResult(
            solution=x,
            converged=True,
            iterations=0,
            final_residual=0.0,
            residual_history=[0.0]
        )
    
    residual_history = [np.sqrt(r_norm_sq) / b_norm]
    
    if verbose:
        print(f"  CG iter 0: rel_residual = {residual_history[0]:.2e}")
    
    for k in range(max_iter):
        # Check convergence
        if residual_history[-1] < tol:
            return CGResult(
                solution=x,
                converged=True,
                iterations=k,
                final_residual=residual_history[-1],
                residual_history=residual_history
            )
        
        # Compute M·p
        Mp = M_apply(p)
        
        # Step size: α = (r·r) / (p·M·p)
        pMp = lie_algebra_inner(p, Mp)
        if abs(pMp) < 1e-30:
            # M is singular or p is in null space
            break
        alpha = r_norm_sq / pMp
        
        # Update solution: x = x + α·p
        x = x.add(p.scale(alpha))
        
        # Update residual: r = r - α·M·p
        r = r.add(Mp.scale(-alpha))
        
        # New residual norm
        r_norm_sq_new = lie_algebra_inner(r, r)
        
        # Update search direction: p = r + β·p
        beta = r_norm_sq_new / r_norm_sq
        p = r.add(p.scale(beta))
        
        r_norm_sq = r_norm_sq_new
        rel_residual = np.sqrt(r_norm_sq) / b_norm
        residual_history.append(rel_residual)
        
        if verbose and (k + 1) % 10 == 0:
            print(f"  CG iter {k+1}: rel_residual = {rel_residual:.2e}")
    
    return CGResult(
        solution=x,
        converged=residual_history[-1] < tol,
        iterations=len(residual_history) - 1,
        final_residual=residual_history[-1],
        residual_history=residual_history
    )


# =============================================================================
# Mechanical Connection
# =============================================================================

class MechanicalConnection:
    """
    Fisher-Rao mechanical connection for MHA parameter space.
    
    Provides the horizontal-vertical decomposition:
        ξ = J_θ(Γ_θ(ξ)) + P_hor(ξ)
    
    where Γ_θ(ξ) solves M_θ·Γ_θ(ξ) = b_θ(ξ).
    """
    
    def __init__(self, config: ReferenceConfig, fr_metric: FisherRaoMetric,
                 cg_tol: float = 1e-10, cg_max_iter: int = 100):
        """
        Initialize the mechanical connection.
        
        Args:
            config: Reference configuration
            fr_metric: Fisher-Rao metric instance
            cg_tol: CG convergence tolerance
            cg_max_iter: CG maximum iterations
        """
        self.config = config
        self.fr_metric = fr_metric
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
    
    def solve_connection(self, theta: MHAParams, xi: MHATangent,
                         verbose: bool = False) -> Tuple[GaugeDirection, CGResult]:
        """
        Solve for the connection 1-form: Γ_θ(ξ).
        
        Solves M_θ·Γ = b_θ(ξ) where:
            M_θ = J^* G J
            b_θ(ξ) = J^* G ξ
        
        Args:
            theta: MHA parameters (base point)
            xi: Tangent vector to decompose
            verbose: Print CG progress
        
        Returns:
            Tuple of (Γ_θ(ξ), CGResult)
        """
        # Compute right-hand side: b = J^* G ξ
        b = compute_fr_adjoint_J(theta, xi, self.fr_metric)
        
        # Define M operator
        def M_apply(eta: GaugeDirection) -> GaugeDirection:
            return compute_M_operator(theta, eta, self.fr_metric)
        
        # Solve via CG
        result = conjugate_gradient_solve(
            M_apply, b, self.config,
            tol=self.cg_tol, max_iter=self.cg_max_iter,
            verbose=verbose
        )
        
        return result.solution, result
    
    def project_vertical(self, theta: MHAParams, xi: MHATangent,
                         verbose: bool = False) -> Tuple[MHATangent, CGResult]:
        """
        Project tangent vector to vertical subspace: ξ_vert = J_θ(Γ_θ(ξ))
        
        Args:
            theta: MHA parameters (base point)
            xi: Tangent vector to project
            verbose: Print CG progress
        
        Returns:
            Tuple of (ξ_vert, CGResult)
        """
        gamma, cg_result = self.solve_connection(theta, xi, verbose)
        xi_vert = vertical_tangent(theta, gamma)
        return xi_vert, cg_result
    
    def project_horizontal(self, theta: MHAParams, xi: MHATangent,
                           verbose: bool = False) -> Tuple[MHATangent, CGResult]:
        """
        Project tangent vector to horizontal subspace: ξ_hor = ξ - J_θ(Γ_θ(ξ))
        
        Args:
            theta: MHA parameters (base point)
            xi: Tangent vector to project
            verbose: Print CG progress
        
        Returns:
            Tuple of (ξ_hor, CGResult)
        """
        xi_vert, cg_result = self.project_vertical(theta, xi, verbose)
        xi_hor = xi.subtract(xi_vert)
        return xi_hor, cg_result
    
    def decompose(self, theta: MHAParams, xi: MHATangent,
                  verbose: bool = False) -> Tuple[MHATangent, MHATangent, GaugeDirection, CGResult]:
        """
        Full horizontal-vertical decomposition.
        
        Args:
            theta: MHA parameters (base point)
            xi: Tangent vector to decompose
            verbose: Print CG progress
        
        Returns:
            Tuple of (ξ_hor, ξ_vert, Γ_θ(ξ), CGResult)
        """
        gamma, cg_result = self.solve_connection(theta, xi, verbose)
        xi_vert = vertical_tangent(theta, gamma)
        xi_hor = xi.subtract(xi_vert)
        return xi_hor, xi_vert, gamma, cg_result


# =============================================================================
# Validation Tests
# =============================================================================

def test_decomposition_sum(config: ReferenceConfig, connection: MechanicalConnection,
                           device: torch.device, dtype: torch.dtype,
                           tol: float = 1e-8) -> Dict:
    """
    Test that ξ = ξ_hor + ξ_vert (decomposition is complete).
    """
    theta = random_mha_params(config, device, dtype)
    xi = random_tangent(config, device, dtype)
    
    xi_hor, xi_vert, _, cg_result = connection.decompose(theta, xi)
    
    # Check ξ = ξ_hor + ξ_vert
    reconstructed = xi_hor.add(xi_vert)
    diff = xi.subtract(reconstructed)
    error = diff.norm_euclidean().item()
    xi_norm = xi.norm_euclidean().item()
    rel_error = error / xi_norm if xi_norm > 1e-15 else error
    
    passed = rel_error < tol and cg_result.converged
    
    return {
        'passed': passed,
        'cg_converged': cg_result.converged,
        'cg_iterations': cg_result.iterations,
        'cg_residual': cg_result.final_residual,
        'reconstruction_error': rel_error,
        'tolerance': tol
    }


def test_horizontal_orthogonality(config: ReferenceConfig, connection: MechanicalConnection,
                                   device: torch.device, dtype: torch.dtype,
                                   n_directions: int = 5, tol: float = 1e-6) -> Dict:
    """
    Test that ξ_hor is FR-orthogonal to all vertical directions.
    
    This is the defining property of the mechanical connection:
        ⟨ξ_hor, J_θ(η)⟩_FR = 0 for all η ∈ g_max
    """
    theta = random_mha_params(config, device, dtype)
    xi = random_tangent(config, device, dtype)
    
    xi_hor, _, _, cg_result = connection.decompose(theta, xi)
    
    # Test orthogonality against several random vertical directions
    inner_products = []
    for _ in range(n_directions):
        eta = random_gauge_direction(config, device, dtype)
        v_eta = vertical_tangent(theta, eta)
        
        # Compute ⟨ξ_hor, J(η)⟩_FR
        inner = connection.fr_metric.inner_product(theta, xi_hor, v_eta).item()
        inner_products.append(abs(inner))
    
    max_inner = max(inner_products)
    
    # Normalize by the norm of ξ_hor
    xi_hor_norm = connection.fr_metric.norm(theta, xi_hor).item()
    rel_error = max_inner / xi_hor_norm if xi_hor_norm > 1e-15 else max_inner
    
    passed = rel_error < tol and cg_result.converged
    
    return {
        'passed': passed,
        'cg_converged': cg_result.converged,
        'max_inner_product': max_inner,
        'xi_hor_norm_FR': xi_hor_norm,
        'relative_error': rel_error,
        'tolerance': tol
    }


def test_vertical_is_vertical(config: ReferenceConfig, connection: MechanicalConnection,
                               device: torch.device, dtype: torch.dtype,
                               tol: float = 1e-8) -> Dict:
    """
    Test that ξ_vert is actually a vertical direction (in image of J_θ).
    
    By construction ξ_vert = J_θ(Γ), so this should hold exactly.
    We verify by checking that projecting ξ_vert gives back ξ_vert.
    """
    theta = random_mha_params(config, device, dtype)
    xi = random_tangent(config, device, dtype)
    
    _, xi_vert, gamma, cg_result1 = connection.decompose(theta, xi)
    
    # Project xi_vert - should give back xi_vert (up to numerical precision)
    xi_vert_hor, xi_vert_vert, _, cg_result2 = connection.decompose(theta, xi_vert)
    
    # xi_vert should have zero horizontal component
    hor_norm = xi_vert_hor.norm_euclidean().item()
    vert_norm = xi_vert.norm_euclidean().item()
    rel_error = hor_norm / vert_norm if vert_norm > 1e-15 else hor_norm
    
    passed = rel_error < tol and cg_result1.converged and cg_result2.converged
    
    return {
        'passed': passed,
        'horizontal_component_norm': hor_norm,
        'vertical_norm': vert_norm,
        'relative_error': rel_error,
        'tolerance': tol
    }


def test_horizontal_is_horizontal(config: ReferenceConfig, connection: MechanicalConnection,
                                   device: torch.device, dtype: torch.dtype,
                                   tol: float = 1e-8) -> Dict:
    """
    Test that ξ_hor stays horizontal under re-projection (idempotence).
    
    P_hor(P_hor(ξ)) = P_hor(ξ)
    """
    theta = random_mha_params(config, device, dtype)
    xi = random_tangent(config, device, dtype)
    
    xi_hor, _, _, cg_result1 = connection.decompose(theta, xi)
    
    # Project xi_hor again
    xi_hor_hor, xi_hor_vert, _, cg_result2 = connection.decompose(theta, xi_hor)
    
    # xi_hor should equal xi_hor_hor
    diff = xi_hor.subtract(xi_hor_hor)
    error = diff.norm_euclidean().item()
    hor_norm = xi_hor.norm_euclidean().item()
    rel_error = error / hor_norm if hor_norm > 1e-15 else error
    
    passed = rel_error < tol and cg_result1.converged and cg_result2.converged
    
    return {
        'passed': passed,
        'absolute_error': error,
        'relative_error': rel_error,
        'tolerance': tol
    }


def test_cg_convergence(config: ReferenceConfig, connection: MechanicalConnection,
                        device: torch.device, dtype: torch.dtype,
                        n_samples: int = 5) -> Dict:
    """
    Test that CG consistently converges across multiple random inputs.
    """
    theta = random_mha_params(config, device, dtype)
    
    results = []
    for _ in range(n_samples):
        xi = random_tangent(config, device, dtype)
        _, cg_result = connection.project_horizontal(theta, xi)
        results.append({
            'converged': cg_result.converged,
            'iterations': cg_result.iterations,
            'final_residual': cg_result.final_residual
        })
    
    all_converged = all(r['converged'] for r in results)
    avg_iterations = np.mean([r['iterations'] for r in results])
    max_residual = max(r['final_residual'] for r in results)
    
    return {
        'passed': all_converged,
        'all_converged': all_converged,
        'average_iterations': avg_iterations,
        'max_final_residual': max_residual,
        'per_sample_results': results
    }


def run_all_tests(config: ReferenceConfig = None,
                  device: torch.device = None,
                  dtype: torch.dtype = DTYPE) -> Dict:
    """
    Run all validation tests for Module 3.
    """
    if config is None:
        config = ReferenceConfig()
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("MODULE 3 VALIDATION: Mechanical Connection Solver")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  n_heads = {config.n_heads}")
    print(f"  d_k = {config.d_k}, d_v = {config.d_v}")
    print(f"  d_model = {config.d_model}")
    print(f"  Lie algebra dimension = {config.lie_algebra_dim}")
    print(f"  Device = {device}")
    print(f"  Dtype = {dtype}")
    
    # Create evaluation batch and metric
    print("\nSetting up Fisher-Rao metric...")
    eval_batch = create_eval_batch(config, batch_size=64, seq_len=32,
                                    device=device, dtype=dtype)
    fr_metric = FisherRaoMetric(config, eval_batch)
    
    # Create mechanical connection
    connection = MechanicalConnection(config, fr_metric, cg_tol=1e-10, cg_max_iter=100)
    
    results = {}
    
    # Test 1: Decomposition completeness
    print("\n[Test 1] Decomposition sum (ξ = ξ_hor + ξ_vert)...")
    results['decomposition_sum'] = test_decomposition_sum(config, connection, device, dtype)
    status = "PASS" if results['decomposition_sum']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  CG iterations: {results['decomposition_sum']['cg_iterations']}")
    print(f"  Reconstruction error: {results['decomposition_sum']['reconstruction_error']:.2e}")
    
    # Test 2: Horizontal orthogonality
    print("\n[Test 2] Horizontal orthogonality (⟨ξ_hor, J(η)⟩_FR = 0)...")
    results['orthogonality'] = test_horizontal_orthogonality(config, connection, device, dtype)
    status = "PASS" if results['orthogonality']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Max |⟨ξ_hor, J(η)⟩| / ||ξ_hor||: {results['orthogonality']['relative_error']:.2e}")
    
    # Test 3: Vertical projection is vertical
    print("\n[Test 3] Vertical is vertical (P_hor(ξ_vert) = 0)...")
    results['vertical_is_vertical'] = test_vertical_is_vertical(config, connection, device, dtype)
    status = "PASS" if results['vertical_is_vertical']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Relative error: {results['vertical_is_vertical']['relative_error']:.2e}")
    
    # Test 4: Horizontal projection is idempotent
    print("\n[Test 4] Horizontal idempotence (P_hor(P_hor(ξ)) = P_hor(ξ))...")
    results['horizontal_idempotent'] = test_horizontal_is_horizontal(config, connection, device, dtype)
    status = "PASS" if results['horizontal_idempotent']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Relative error: {results['horizontal_idempotent']['relative_error']:.2e}")
    
    # Test 5: CG convergence reliability
    print("\n[Test 5] CG convergence reliability...")
    results['cg_convergence'] = test_cg_convergence(config, connection, device, dtype)
    status = "PASS" if results['cg_convergence']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Average iterations: {results['cg_convergence']['average_iterations']:.1f}")
    print(f"  Max final residual: {results['cg_convergence']['max_final_residual']:.2e}")
    
    # Summary
    print("\n" + "=" * 70)
    all_passed = all(r['passed'] for r in results.values())
    print(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 70)
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = ReferenceConfig()
    results = run_all_tests(config, device, DTYPE)
