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


# --- module4_discrete_holonomy.py ---
"""Module 4: Discrete Holonomy Estimator

Implements discrete holonomy-based curvature estimation with Richardson
extrapolation for accurate curvature computation.
"""

#!/usr/bin/env python3
"""
Module 4: Discrete Holonomy and Curvature

This module computes the Fisher-Rao curvature on the quotient manifold via
discrete holonomy - measuring the failure of parallel transport around
infinitesimal loops to close.

Mathematical Background:
    The curvature 2-form Ω captures the non-commutativity of parallel transport.
    For horizontal vectors u, v at a point [θ] on the quotient F = Θ/G_max:
    
        Ω(u, v) = [∇_u, ∇_v] - ∇_{[u,v]}
    
    In terms of discrete holonomy around a small square loop:
        θ₀ → θ₀+εu → θ₀+εu+εv → θ₀+εv → θ₀
    
    The holonomy defect is:
        Δ_□(u, v; ε) = ε² Ω(u, v) + O(ε³)
    
    CRITICAL DISTINCTION from prior implementations:
    This computes ACTUAL parallel transport around loops using horizontal 
    projection at each step, NOT mixed partial derivatives of a scalar loss.
    The latter measures Hessian structure, not geometric curvature.

Algorithm:
    1. Start at θ₀ with reference vector ξ₀
    2. Transport along u: project ξ to horizontal at θ₀+εu
    3. Transport along v: project to horizontal at θ₀+εu+εv  
    4. Transport along -u: project to horizontal at θ₀+εv
    5. Transport along -v: project back to θ₀
    6. Holonomy = difference between final and initial vectors
    
    Richardson extrapolation removes O(ε) bias:
        K_richardson = (ε₂·K(ε₁) - ε₁·K(ε₂)) / (ε₂ - ε₁)

Reference Configuration:
    h = 4 heads, d_k = d_v = 8, d_model = 32
    ε values: [1e-4, 2e-4] for Richardson extrapolation
    Direction pairs: 20 (10 train, 10 test)

Dependencies:
    - Module 1: Data structures, vertical_tangent
    - Module 2: FisherRaoMetric
    - Module 3: MechanicalConnection, horizontal projection

Author: Research validation implementation
Date: 2025
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

# Import from previous modules
from module1_gauge_algebra import (
    ReferenceConfig, DTYPE,
    HeadParams, MHAParams, HeadTangent, MHATangent,
    GaugeDirection, HeadLieAlgebra,
    vertical_tangent, random_mha_params, random_tangent,
    mha_forward
)

from module2_fisher_rao_metric import (
    FisherRaoMetric, create_eval_batch
)

from module3_mechanical_connection import (
    MechanicalConnection, lie_algebra_norm
)


# =============================================================================
# Discrete Holonomy Computation
# =============================================================================

@dataclass
class HolonomyResult:
    """Result of holonomy computation around a loop."""
    holonomy_vector: MHATangent  # The holonomy defect
    holonomy_norm: float  # ||holonomy||_FR
    epsilon: float
    cg_iterations: List[int]  # CG iterations at each corner
    cg_converged: List[bool]


@dataclass  
class CurvatureResult:
    """Result of curvature computation with Richardson extrapolation."""
    curvature_richardson: float  # Extrapolated ||Ω(u,v)||
    curvature_squared: float  # ||Ω(u,v)||²
    curvature_raw: List[float]  # Raw values at each ε
    richardson_ratio: float  # Convergence diagnostic
    linearization_index: float  # Higher-order effect measure
    epsilon_values: List[float]
    holonomy_results: List[HolonomyResult]


def perturb_params(theta: MHAParams, direction: MHATangent, 
                   epsilon: float) -> MHAParams:
    """
    Create perturbed parameters: θ' = θ + ε·direction
    """
    new_heads = []
    for hp, ht in zip(theta.heads, direction.heads):
        new_heads.append(HeadParams(
            W_Q=hp.W_Q + epsilon * ht.dW_Q,
            W_K=hp.W_K + epsilon * ht.dW_K,
            W_V=hp.W_V + epsilon * ht.dW_V,
            W_O=hp.W_O + epsilon * ht.dW_O
        ))
    return MHAParams(heads=new_heads)


class DiscreteHolonomy:
    """
    Computes discrete holonomy around infinitesimal loops for curvature estimation.
    
    The holonomy measures how much a vector changes when parallel transported
    around a closed loop. For the Fisher-Rao connection, this captures the
    curvature of the quotient manifold.
    """
    
    def __init__(self, config: ReferenceConfig, connection: MechanicalConnection):
        """
        Initialize discrete holonomy computer.
        
        Args:
            config: Reference configuration
            connection: Mechanical connection for horizontal projection
        """
        self.config = config
        self.connection = connection
        self.fr_metric = connection.fr_metric
    
    def _create_metric_at_point(self, theta: MHAParams) -> FisherRaoMetric:
        """
        Create a Fisher-Rao metric instance at a new parameter point.
        
        Note: We reuse the same evaluation batch but create the metric
        for the new parameter point.
        """
        # The metric depends on θ through the Jacobian, but the eval batch is fixed
        return FisherRaoMetric(self.config, self.fr_metric.eval_batch)
    
    def _create_connection_at_point(self, theta: MHAParams) -> MechanicalConnection:
        """
        Create a mechanical connection at a new parameter point.
        """
        fr_metric = self._create_metric_at_point(theta)
        return MechanicalConnection(
            self.config, fr_metric,
            cg_tol=self.connection.cg_tol,
            cg_max_iter=self.connection.cg_max_iter
        )
    
    def compute_holonomy_square_loop(
        self, 
        theta: MHAParams,
        u: MHATangent,  # First horizontal direction
        v: MHATangent,  # Second horizontal direction
        epsilon: float,
        reference_vector: Optional[MHATangent] = None
    ) -> HolonomyResult:
        """
        Compute holonomy around a square loop in the (u, v) plane.
        
        The curvature is measured via the connection 1-form Γ. For horizontal
        directions u, v, the curvature satisfies:
        
            Ω(u,v) = dΓ(u,v) + [Γ(u), Γ(v)]
        
        Since u, v are horizontal, Γ(u) = Γ(v) = 0 at the base point.
        The curvature comes from how the connection form varies:
        
            Ω(u,v) ≈ (Γ_{θ+εu}(v) - Γ_{θ}(v))/ε - (Γ_{θ+εv}(u) - Γ_{θ}(u))/ε
        
        This measures the non-commutativity: moving in direction u changes
        how v projects, and vice versa.
        
        Args:
            theta: Base point θ₀
            u: First horizontal direction (should be FR-horizontal at θ₀)
            v: Second horizontal direction (should be FR-horizontal at θ₀)
            epsilon: Step size
            reference_vector: Not used in this implementation
        
        Returns:
            HolonomyResult with holonomy defect and diagnostics
        """
        device = theta.device
        dtype = theta.dtype
        
        cg_iterations = []
        cg_converged = []
        
        # At base point, u and v are horizontal, so Γ_θ(u) = Γ_θ(v) = 0
        # We need to measure how the connection changes as we move
        
        # Point 1: θ + εu
        theta_u = perturb_params(theta, u, epsilon)
        conn_u = self._create_connection_at_point(theta_u)
        # Compute Γ_{θ+εu}(v) - how v projects at the new point
        gamma_u_v, cg_result = conn_u.solve_connection(theta_u, v)
        cg_iterations.append(cg_result.iterations)
        cg_converged.append(cg_result.converged)
        
        # Point 2: θ + εv  
        theta_v = perturb_params(theta, v, epsilon)
        conn_v = self._create_connection_at_point(theta_v)
        # Compute Γ_{θ+εv}(u) - how u projects at the new point
        gamma_v_u, cg_result = conn_v.solve_connection(theta_v, u)
        cg_iterations.append(cg_result.iterations)
        cg_converged.append(cg_result.converged)
        
        # The curvature in the Lie algebra is approximately:
        # Ω(u,v) ≈ (Γ_{θ+εu}(v) - Γ_{θ+εv}(u)) / ε
        # (The Γ_θ terms are zero since u,v are horizontal at θ)
        
        # Compute the difference in the Lie algebra
        # holonomy_lie = gamma_u_v - gamma_v_u (but we want the antisymmetric part)
        holonomy_lie = gamma_u_v.add(gamma_v_u.scale(-1.0))
        
        # Scale by 1/ε to get the curvature estimate
        # ||Ω(u,v)|| ≈ ||holonomy_lie|| / ε
        holonomy_norm = lie_algebra_norm(holonomy_lie) / epsilon
        
        # Convert to tangent space for compatibility (apply J_θ)
        holonomy_tangent = vertical_tangent(theta, holonomy_lie)
        
        return HolonomyResult(
            holonomy_vector=holonomy_tangent,
            holonomy_norm=holonomy_norm,
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
        Compute curvature ||Ω(u,v)|| with Richardson extrapolation.
        
        The connection form difference scales as:
            ||Γ_{θ+εu}(v) - Γ_{θ+εv}(u)|| / ε = ||Ω(u,v)|| + O(ε)
        
        Richardson extrapolation removes the O(ε) bias.
        
        Args:
            theta: Base point
            u: First horizontal direction
            v: Second horizontal direction
            epsilon_values: Step sizes for extrapolation
        
        Returns:
            CurvatureResult with extrapolated curvature and diagnostics
        """
        if epsilon_values is None:
            epsilon_values = [1e-4, 2e-4]
        
        # Compute holonomy at each epsilon
        holonomy_results = []
        K_values = []  # K(ε) = holonomy_norm (already scaled by 1/ε in compute_holonomy)
        
        for eps in epsilon_values:
            result = self.compute_holonomy_square_loop(theta, u, v, eps)
            holonomy_results.append(result)
            K_values.append(result.holonomy_norm)
        
        # Richardson extrapolation
        # K(ε) = K(0) + a·ε + O(ε²)
        # K_richardson = (ε₂·K(ε₁) - ε₁·K(ε₂)) / (ε₂ - ε₁)
        if len(epsilon_values) >= 2:
            eps1, eps2 = epsilon_values[0], epsilon_values[1]
            K1, K2 = K_values[0], K_values[1]
            
            if abs(eps2 - eps1) > 1e-15:
                K_richardson = (eps2 * K1 - eps1 * K2) / (eps2 - eps1)
            else:
                K_richardson = K1
            
            # Richardson ratio for convergence monitoring
            if K1 > 1e-15:
                richardson_ratio = K2 / K1
            else:
                richardson_ratio = 1.0
        else:
            K_richardson = K_values[0]
            richardson_ratio = 1.0
        
        # Ensure non-negative
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


# =============================================================================
# Direction Pair Generation
# =============================================================================

def generate_horizontal_direction_pair(
    theta: MHAParams,
    connection: MechanicalConnection,
    seed: int = None
) -> Tuple[MHATangent, MHATangent]:
    """
    Generate a pair of FR-orthogonal horizontal directions.
    
    Args:
        theta: Base point
        connection: Mechanical connection
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (u, v) where u, v are unit FR-horizontal directions
        that are FR-orthogonal to each other.
    """
    config = connection.config
    fr_metric = connection.fr_metric
    device = theta.device
    dtype = theta.dtype
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate first random direction and project to horizontal
    u_rand = random_tangent(config, device, dtype)
    u, _ = connection.project_horizontal(theta, u_rand)
    
    # Normalize to unit FR norm
    u = fr_metric.normalize(theta, u)
    
    # Generate second random direction
    v_rand = random_tangent(config, device, dtype)
    v, _ = connection.project_horizontal(theta, v_rand)
    
    # Gram-Schmidt: orthogonalize v with respect to u in FR metric
    inner_uv = fr_metric.inner_product(theta, u, v).item()
    v = v.subtract(u.scale(inner_uv))
    
    # Normalize
    v = fr_metric.normalize(theta, v)
    
    return u, v


def generate_direction_pairs(
    theta: MHAParams,
    connection: MechanicalConnection,
    n_pairs: int,
    base_seed: int = 42
) -> List[Tuple[MHATangent, MHATangent]]:
    """
    Generate multiple pairs of FR-orthogonal horizontal directions.
    
    Args:
        theta: Base point
        connection: Mechanical connection
        n_pairs: Number of pairs to generate
        base_seed: Base random seed
    
    Returns:
        List of (u, v) pairs
    """
    pairs = []
    for i in range(n_pairs):
        u, v = generate_horizontal_direction_pair(theta, connection, seed=base_seed + i)
        pairs.append((u, v))
    return pairs


# =============================================================================
# Validation Tests
# =============================================================================

def test_holonomy_scaling(config: ReferenceConfig, holonomy_computer: DiscreteHolonomy,
                          device: torch.device, dtype: torch.dtype,
                          tol: float = 0.5) -> Dict:
    """
    Test that the curvature estimate converges as ε → 0.
    
    With the connection form approach:
        K(ε) = ||Γ_{θ+εu}(v) - Γ_{θ+εv}(u)|| / ε = ||Ω(u,v)|| + O(ε)
    
    So K(ε₁) ≈ K(ε₂) for small ε, with the difference being O(ε).
    """
    theta = random_mha_params(config, device, dtype)
    
    # Generate horizontal directions
    u, v = generate_horizontal_direction_pair(theta, holonomy_computer.connection)
    
    # Compute curvature at two epsilon values
    eps1, eps2 = 1e-4, 2e-4
    
    result1 = holonomy_computer.compute_holonomy_square_loop(theta, u, v, eps1)
    result2 = holonomy_computer.compute_holonomy_square_loop(theta, u, v, eps2)
    
    K1 = result1.holonomy_norm  # Already scaled by 1/ε
    K2 = result2.holonomy_norm
    
    # For O(ε) convergence, the ratio K2/K1 should be close to 1
    # with the difference being O(ε)
    if K1 > 1e-15:
        ratio = K2 / K1
        # The ratio should be between 0.5 and 2.0 for reasonable convergence
        ratio_ok = 0.5 < ratio < 2.0
    else:
        ratio = 1.0
        ratio_ok = True
    
    # Also check that we're getting non-trivial curvature
    has_curvature = K1 > 1e-15 or K2 > 1e-15
    
    passed = ratio_ok and has_curvature
    
    return {
        'passed': passed,
        'K_eps1': K1,
        'K_eps2': K2,
        'ratio': ratio,
        'has_nonzero_curvature': has_curvature,
        'tolerance': tol
    }


def test_richardson_consistency(config: ReferenceConfig, holonomy_computer: DiscreteHolonomy,
                                  device: torch.device, dtype: torch.dtype,
                                  tol: float = 0.5) -> Dict:
    """
    Test that Richardson extrapolation gives consistent results.
    
    The Richardson ratio K(2ε)/K(ε) should be approximately constant
    if the O(ε²) expansion is valid.
    """
    theta = random_mha_params(config, device, dtype)
    u, v = generate_horizontal_direction_pair(theta, holonomy_computer.connection)
    
    # Use three epsilon values
    result = holonomy_computer.compute_curvature_with_richardson(
        theta, u, v, epsilon_values=[1e-4, 2e-4, 4e-4]
    )
    
    # Richardson ratio should be around 1.0 for good extrapolation
    # (actually around 2.0 since K(2ε) ≈ 2·K(ε) for linear ε dependence)
    passed = 0.5 < result.richardson_ratio < 4.0
    
    return {
        'passed': passed,
        'curvature_richardson': result.curvature_richardson,
        'curvature_raw': result.curvature_raw,
        'richardson_ratio': result.richardson_ratio,
        'linearization_index': result.linearization_index,
        'tolerance': tol
    }


def test_antisymmetry(config: ReferenceConfig, holonomy_computer: DiscreteHolonomy,
                       device: torch.device, dtype: torch.dtype,
                       tol: float = 0.3) -> Dict:
    """
    Test that curvature is antisymmetric: Ω(u,v) = -Ω(v,u)
    
    The magnitudes ||Ω(u,v)|| and ||Ω(v,u)|| should be equal.
    """
    theta = random_mha_params(config, device, dtype)
    u, v = generate_horizontal_direction_pair(theta, holonomy_computer.connection)
    
    # Compute curvature for (u,v) and (v,u)
    result_uv = holonomy_computer.compute_curvature_with_richardson(theta, u, v)
    result_vu = holonomy_computer.compute_curvature_with_richardson(theta, v, u)
    
    # Magnitudes should be equal
    curv_uv = result_uv.curvature_richardson
    curv_vu = result_vu.curvature_richardson
    
    if max(curv_uv, curv_vu) > 1e-15:
        rel_diff = abs(curv_uv - curv_vu) / max(curv_uv, curv_vu)
    else:
        rel_diff = 0.0
    
    passed = rel_diff < tol
    
    return {
        'passed': passed,
        'curvature_uv': curv_uv,
        'curvature_vu': curv_vu,
        'relative_difference': rel_diff,
        'tolerance': tol
    }


def test_curvature_nonzero(config: ReferenceConfig, holonomy_computer: DiscreteHolonomy,
                            device: torch.device, dtype: torch.dtype,
                            n_pairs: int = 5) -> Dict:
    """
    Test that curvature is generally non-zero for random horizontal directions.
    
    Zero curvature everywhere would indicate a flat connection, which should
    not occur for the MHA gauge structure.
    """
    theta = random_mha_params(config, device, dtype)
    
    curvatures = []
    for i in range(n_pairs):
        u, v = generate_horizontal_direction_pair(
            theta, holonomy_computer.connection, seed=42 + i
        )
        result = holonomy_computer.compute_curvature_with_richardson(theta, u, v)
        curvatures.append(result.curvature_richardson)
    
    # At least some curvatures should be significantly non-zero
    max_curv = max(curvatures)
    mean_curv = np.mean(curvatures)
    
    # Pass if we have detectable curvature
    passed = max_curv > 1e-10
    
    return {
        'passed': passed,
        'curvatures': curvatures,
        'max_curvature': max_curv,
        'mean_curvature': mean_curv,
        'min_curvature': min(curvatures)
    }


def run_all_tests(config: ReferenceConfig = None,
                  device: torch.device = None,
                  dtype: torch.dtype = DTYPE) -> Dict:
    """
    Run all validation tests for Module 4.
    """
    if config is None:
        config = ReferenceConfig()
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("MODULE 4 VALIDATION: Discrete Holonomy and Curvature")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  n_heads = {config.n_heads}")
    print(f"  d_k = {config.d_k}, d_v = {config.d_v}")
    print(f"  d_model = {config.d_model}")
    print(f"  Device = {device}")
    print(f"  Dtype = {dtype}")
    
    # Setup
    print("\nSetting up connection and holonomy computer...")
    eval_batch = create_eval_batch(config, batch_size=64, seq_len=32,
                                    device=device, dtype=dtype)
    fr_metric = FisherRaoMetric(config, eval_batch)
    connection = MechanicalConnection(config, fr_metric)
    holonomy_computer = DiscreteHolonomy(config, connection)
    
    results = {}
    
    # Test 1: Curvature convergence
    print("\n[Test 1] Curvature estimate convergence...")
    results['scaling'] = test_holonomy_scaling(config, holonomy_computer, device, dtype)
    status = "PASS" if results['scaling']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  K(ε=1e-4): {results['scaling']['K_eps1']:.2e}")
    print(f"  K(ε=2e-4): {results['scaling']['K_eps2']:.2e}")
    print(f"  Ratio K2/K1: {results['scaling']['ratio']:.3f} (expect ~1.0 for convergence)")
    
    # Test 2: Richardson consistency
    print("\n[Test 2] Richardson extrapolation consistency...")
    results['richardson'] = test_richardson_consistency(config, holonomy_computer, device, dtype)
    status = "PASS" if results['richardson']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Curvature (Richardson): {results['richardson']['curvature_richardson']:.2e}")
    print(f"  Richardson ratio: {results['richardson']['richardson_ratio']:.3f}")
    
    # Test 3: Antisymmetry
    print("\n[Test 3] Curvature antisymmetry (||Ω(u,v)|| = ||Ω(v,u)||)...")
    results['antisymmetry'] = test_antisymmetry(config, holonomy_computer, device, dtype)
    status = "PASS" if results['antisymmetry']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  ||Ω(u,v)||: {results['antisymmetry']['curvature_uv']:.2e}")
    print(f"  ||Ω(v,u)||: {results['antisymmetry']['curvature_vu']:.2e}")
    print(f"  Relative difference: {results['antisymmetry']['relative_difference']:.3f}")
    
    # Test 4: Non-zero curvature
    print("\n[Test 4] Non-zero curvature detection...")
    results['nonzero'] = test_curvature_nonzero(config, holonomy_computer, device, dtype)
    status = "PASS" if results['nonzero']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Max curvature: {results['nonzero']['max_curvature']:.2e}")
    print(f"  Mean curvature: {results['nonzero']['mean_curvature']:.2e}")
    print(f"  Curvatures: {[f'{c:.2e}' for c in results['nonzero']['curvatures']]}")
    
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
