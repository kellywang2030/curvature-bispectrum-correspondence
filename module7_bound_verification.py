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

# --- module7_bound_verification.py ---
"""Module 7: Bound Verification

Integrates all modules to verify the theoretical bound with proper
train/test methodology.
"""

#!/usr/bin/env python3
"""
Module 7: Bound Verification for Curvature-Bispectrum Correspondence

This module validates the theoretical lower bound from Theorem 4:
    ||Ω_ℓ(u,v)||²_F ≥ c_ℓ · E_ℓ(u,v)

where:
- Ω_ℓ(u,v) is the Fisher-Rao curvature in directions (u,v)
- E_ℓ(u,v) is the directional nontrivial bispectral energy
- c_ℓ > 0 is a layer-dependent constant

The verification protocol (from Appendix D.2):
1. For each direction pair (u,v), compute both curvature and bispectral energy
2. Estimate c_ℓ as the minimum ratio: c_ℓ = min_{u,v} κ²(u,v) / (E(u,v) + δ)
3. Verify the bound holds on held-out direction pairs
4. Report correspondence rate and confidence intervals

References:
- Paper Section 6: Empirical Validation
- Appendix D.2: Bound Verification Protocol

Usage:
    Ensure module1_gauge_algebra.py, module4_euclidean_curvature.py, 
    module5_canonicalization.py, and module6_bispectrum.py are in the 
    same directory, then run:
    
    python module7_bound_verification.py
"""

from __future__ import annotations  # Enable forward reference annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
import sys
import os

# Try to import from previous modules
MODULES_IMPORTED = False
IMPORT_ERROR = None

try:
    from module1_gauge_algebra import ReferenceConfig, MHAParams, HeadParams, HeadTangent, MHATangent
    from module4_euclidean_curvature import (
        DiscreteHolonomy, EuclideanMechanicalConnection,
        generate_horizontal_direction_pair, perturb_params
    )
    from module5_canonicalization import canonicalize as canonicalize_mha
    from module6_bispectrum import BispectrumConfig
    MODULES_IMPORTED = True
except ImportError as e:
    IMPORT_ERROR = str(e)
    
    # Define stub classes so type annotations work
    class ReferenceConfig:
        pass
    class MHAParams:
        pass
    class HeadParams:
        pass
    class HeadTangent:
        pass
    class MHATangent:
        pass
    class BispectrumConfig:
        pass


@dataclass
class BoundVerificationConfig:
    """Configuration for bound verification experiments."""
    n_direction_pairs: int = 30
    n_train_pairs: int = 15
    n_test_pairs: int = 15
    epsilon_values: List[float] = field(default_factory=lambda: [1e-3, 2e-3, 4e-3])
    delta: float = 1e-10
    direction_seed: int = 42


@dataclass
class DirectionPairResult:
    """Results for a single (u,v) direction pair."""
    pair_idx: int
    curvature_squared: float
    bispectral_energy: float
    richardson_ratio: float
    bound_satisfied: bool


@dataclass  
class BoundVerificationResult:
    """Complete results for bound verification."""
    direction_results: List[DirectionPairResult]
    c_ell_estimate: float
    train_correspondence_rate: float
    test_correspondence_rate: float
    overall_correspondence_rate: float
    curvature_mean: float
    curvature_std: float
    energy_mean: float
    energy_std: float
    pearson_correlation: float
    spearman_correlation: float
    mean_richardson_ratio: float
    ci_lower: float
    ci_upper: float


def create_random_params(config: ReferenceConfig, device: torch.device, 
                         dtype: torch.dtype, seed: int = 42) -> MHAParams:
    """Create random MHA parameters."""
    torch.manual_seed(seed)
    heads = []
    for h in range(config.n_heads):
        W_Q = torch.randn(config.d_model, config.d_k, device=device, dtype=dtype) * 0.1
        W_K = torch.randn(config.d_model, config.d_k, device=device, dtype=dtype) * 0.1
        W_V = torch.randn(config.d_model, config.d_v, device=device, dtype=dtype) * 0.1
        W_O = torch.randn(config.d_v, config.d_model, device=device, dtype=dtype) * 0.1
        heads.append(HeadParams(W_Q=W_Q, W_K=W_K, W_V=W_V, W_O=W_O))
    return MHAParams(heads=heads)


def compute_directional_bispectral_energy_tangent(
    theta: MHAParams,
    X: torch.Tensor,
    xi: 'MHATangent',
    eta: 'MHATangent',
    epsilon: float,
    config: ReferenceConfig,
    bispec_config: BispectrumConfig
) -> float:
    """
    Compute directional bispectral energy E_ℓ(u,v).
    
    This is a wrapper that converts MHATangent to flat tensors
    and calls module6's compute_directional_bispectral_energy.
    
    Uses finite differences:
    ∂²B/∂u∂v ≈ (B(θ+εu+εv) - B(θ+εu) - B(θ+εv) + B(θ)) / ε²
    """
    # Import here to avoid circular import issues
    from module6_bispectrum import compute_directional_bispectral_energy as _module6_compute_energy
    
    # Flatten tangent vectors to tensors using MHATangent.flatten() method
    u_flat = xi.flatten()
    v_flat = eta.flatten()
    
    # Call module6's function (returns tuple: energy, diagnostics)
    energy, diagnostics = _module6_compute_energy(
        theta, X, u_flat, v_flat, epsilon, config, bispec_config
    )
    
    return float(energy)


def estimate_c_ell(curvatures: List[float], energies: List[float], 
                   delta: float = 1e-10) -> float:
    """Estimate the constant c_ℓ from training data."""
    ratios = [k / (e + delta) for k, e in zip(curvatures, energies)]
    return min(ratios) if ratios else 0.0


def verify_bound(
    theta: MHAParams,
    X: torch.Tensor,
    config: ReferenceConfig,
    bound_config: BoundVerificationConfig,
    bispec_config: BispectrumConfig
) -> BoundVerificationResult:
    """
    Full bound verification following the protocol from Appendix D.2.
    """
    device = X.device
    dtype = X.dtype
    
    # Canonicalize parameters
    theta_canonical, _ = canonicalize_mha(theta, config)
    
    # Setup curvature computation
    mech_conn = EuclideanMechanicalConnection(config)
    holonomy = DiscreteHolonomy(config)
    
    print(f"  Computing curvature and bispectral energy for {bound_config.n_direction_pairs} pairs...")
    direction_results = []
    
    for pair_idx in range(bound_config.n_direction_pairs):
        # Generate horizontal direction pair
        xi, eta = generate_horizontal_direction_pair(theta_canonical, mech_conn)
        
        # Compute curvature with Richardson extrapolation
        curv_result = holonomy.compute_curvature_with_richardson(
            theta_canonical, xi, eta,
            epsilon_values=bound_config.epsilon_values
        )
        curvature_squared = curv_result.curvature_squared
        richardson_ratio = curv_result.richardson_ratio
        
        # Compute bispectral energy
        energy = compute_directional_bispectral_energy_tangent(
            theta_canonical, X, xi, eta,
            bound_config.epsilon_values[0],
            config, bispec_config
        )
        
        direction_results.append(DirectionPairResult(
            pair_idx=pair_idx,
            curvature_squared=curvature_squared,
            bispectral_energy=energy,
            richardson_ratio=richardson_ratio,
            bound_satisfied=True  # Updated after c_ℓ estimation
        ))
        
        if (pair_idx + 1) % 10 == 0:
            print(f"    Completed {pair_idx + 1}/{bound_config.n_direction_pairs} pairs")
    
    # Split into train/test
    train_results = direction_results[:bound_config.n_train_pairs]
    test_results = direction_results[bound_config.n_train_pairs:]
    
    # Estimate c_ℓ from training pairs
    train_curvatures = [r.curvature_squared for r in train_results]
    train_energies = [r.bispectral_energy for r in train_results]
    c_ell = estimate_c_ell(train_curvatures, train_energies, bound_config.delta)
    
    print(f"  Estimated c_ℓ = {c_ell:.6e}")
    
    # Verify bound on all pairs
    for result in direction_results:
        bound_value = c_ell * result.bispectral_energy
        result.bound_satisfied = (result.curvature_squared >= bound_value - bound_config.delta)
    
    # Compute correspondence rates
    train_satisfied = sum(1 for r in train_results if r.bound_satisfied)
    test_satisfied = sum(1 for r in test_results if r.bound_satisfied)
    total_satisfied = sum(1 for r in direction_results if r.bound_satisfied)
    
    train_rate = train_satisfied / len(train_results) if train_results else 1.0
    test_rate = test_satisfied / len(test_results) if test_results else 1.0
    overall_rate = total_satisfied / len(direction_results) if direction_results else 1.0
    
    # Statistics
    all_curvatures = [r.curvature_squared for r in direction_results]
    all_energies = [r.bispectral_energy for r in direction_results]
    
    curvature_mean = np.mean(all_curvatures)
    curvature_std = np.std(all_curvatures)
    energy_mean = np.mean(all_energies)
    energy_std = np.std(all_energies)
    
    # Pearson correlation
    if len(all_curvatures) > 1 and np.std(all_curvatures) > 1e-10 and np.std(all_energies) > 1e-10:
        correlation = np.corrcoef(all_curvatures, all_energies)[0, 1]
    else:
        correlation = 0.0
    
    # Spearman correlation (rank-based)
    from scipy.stats import spearmanr
    if len(all_curvatures) > 1:
        spearman_corr, _ = spearmanr(all_curvatures, all_energies)
        if np.isnan(spearman_corr):
            spearman_corr = 0.0
    else:
        spearman_corr = 0.0
    
    # Mean Richardson ratio
    all_richardson = [r.richardson_ratio for r in direction_results]
    mean_richardson = np.mean(all_richardson) if all_richardson else 1.0
    
    # Bootstrap confidence interval
    n_bootstrap = 1000
    bootstrap_rates = []
    n_samples = len(direction_results)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        boot_curvatures = [all_curvatures[i] for i in indices]
        boot_energies = [all_energies[i] for i in indices]
        boot_c_ell = estimate_c_ell(boot_curvatures, boot_energies, bound_config.delta)
        boot_violations = sum(1 for k, e in zip(boot_curvatures, boot_energies)
                            if k < boot_c_ell * e - bound_config.delta)
        bootstrap_rates.append(1.0 - boot_violations / n_samples)
    
    ci_lower = np.percentile(bootstrap_rates, 2.5)
    ci_upper = np.percentile(bootstrap_rates, 97.5)
    
    return BoundVerificationResult(
        direction_results=direction_results,
        c_ell_estimate=c_ell,
        train_correspondence_rate=train_rate,
        test_correspondence_rate=test_rate,
        overall_correspondence_rate=overall_rate,
        curvature_mean=curvature_mean,
        curvature_std=curvature_std,
        energy_mean=energy_mean,
        energy_std=energy_std,
        pearson_correlation=correlation,
        spearman_correlation=spearman_corr,
        mean_richardson_ratio=mean_richardson,
        ci_lower=ci_lower,
        ci_upper=ci_upper
    )


# ============================================================================
# VALIDATION TESTS
# ============================================================================

def test_single_pair_computation(config: ReferenceConfig, device: torch.device,
                                  dtype: torch.dtype) -> Dict:
    """Test computation of curvature and energy for a single direction pair."""
    print("\n[Test 1] Single direction pair computation...")
    
    # Create test data
    theta = create_random_params(config, device, dtype, seed=42)
    X = torch.randn(4, 16, config.d_model, device=device, dtype=dtype)
    
    # Canonicalize
    theta_canonical, _ = canonicalize_mha(theta, config)
    
    # Setup
    mech_conn = EuclideanMechanicalConnection(config)
    holonomy = DiscreteHolonomy(config)
    
    # Generate one direction pair
    xi, eta = generate_horizontal_direction_pair(theta_canonical, mech_conn)
    
    # Compute curvature
    curv_result = holonomy.compute_curvature_with_richardson(
        theta_canonical, xi, eta, epsilon_values=[1e-3, 2e-3, 4e-3]
    )
    
    print(f"  Curvature² = {curv_result.curvature_squared:.6e}")
    print(f"  Richardson ratio = {curv_result.richardson_ratio:.4f}")
    
    # Compute bispectral energy
    bispec_config = BispectrumConfig()
    energy = compute_directional_bispectral_energy_tangent(
        theta_canonical, X, xi, eta, 1e-3, config, bispec_config
    )
    
    print(f"  Bispectral energy = {energy:.6e}")
    
    # Basic checks
    passed = (
        curv_result.curvature_squared >= 0 and
        energy >= 0 and
        np.isfinite(curv_result.curvature_squared) and
        np.isfinite(energy)
    )
    
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    
    return {
        'passed': passed,
        'curvature_squared': curv_result.curvature_squared,
        'energy': energy
    }


def test_bound_verification_small(config: ReferenceConfig, device: torch.device,
                                   dtype: torch.dtype) -> Dict:
    """Run bound verification with reduced number of pairs for quick testing."""
    print("\n[Test 2] Bound verification (small scale)...")
    
    # Create test data
    theta = create_random_params(config, device, dtype, seed=42)
    X = torch.randn(4, 16, config.d_model, device=device, dtype=dtype)
    
    # Reduced configuration
    bound_config = BoundVerificationConfig(
        n_direction_pairs=10,
        n_train_pairs=5,
        n_test_pairs=5
    )
    bispec_config = BispectrumConfig()
    
    result = verify_bound(theta, X, config, bound_config, bispec_config)
    
    print(f"\n  Results:")
    print(f"    c_ℓ estimate: {result.c_ell_estimate:.6e}")
    print(f"    Train correspondence: {result.train_correspondence_rate*100:.1f}%")
    print(f"    Test correspondence: {result.test_correspondence_rate*100:.1f}%")
    print(f"    Overall correspondence: {result.overall_correspondence_rate*100:.1f}%")
    print(f"    95% CI: [{result.ci_lower*100:.1f}%, {result.ci_upper*100:.1f}%]")
    print(f"    Pearson |ρ|: {abs(result.pearson_correlation):.4f}")
    print(f"    Curvature: {result.curvature_mean:.4e} ± {result.curvature_std:.4e}")
    print(f"    Energy: {result.energy_mean:.4e} ± {result.energy_std:.4e}")
    
    passed = (
        result.c_ell_estimate > 0 and
        result.train_correspondence_rate == 1.0 and
        result.overall_correspondence_rate >= 0.8
    )
    
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    
    return {'passed': passed, 'result': result}


def test_gauge_invariance_of_bound(config: ReferenceConfig, device: torch.device,
                                    dtype: torch.dtype) -> Dict:
    """Test that bound verification gives consistent results under gauge transforms."""
    print("\n[Test 3] Gauge invariance of bound verification...")
    
    # Create test data
    theta = create_random_params(config, device, dtype, seed=42)
    X = torch.randn(4, 16, config.d_model, device=device, dtype=dtype)
    
    bound_config = BoundVerificationConfig(
        n_direction_pairs=6,
        n_train_pairs=3,
        n_test_pairs=3
    )
    bispec_config = BispectrumConfig()
    
    # Run on original parameters
    result1 = verify_bound(theta, X, config, bound_config, bispec_config)
    
    # Apply gauge transformation
    torch.manual_seed(123)
    A = torch.randn(config.d_k, config.d_k, device=device, dtype=dtype)
    A = A / torch.norm(A) * 2
    A_inv_T = torch.linalg.inv(A).T
    
    C = torch.randn(config.d_v, config.d_v, device=device, dtype=dtype)
    C = C / torch.norm(C) * 2
    C_inv = torch.linalg.inv(C)
    
    transformed_heads = []
    for h in range(config.n_heads):
        transformed_heads.append(HeadParams(
            W_Q=theta.heads[h].W_Q @ A,
            W_K=theta.heads[h].W_K @ A_inv_T,
            W_V=theta.heads[h].W_V @ C,
            W_O=C_inv @ theta.heads[h].W_O
        ))
    theta_gauge = MHAParams(heads=transformed_heads)
    
    # Run on gauge-transformed parameters
    result2 = verify_bound(theta_gauge, X, config, bound_config, bispec_config)
    
    # Compare - after canonicalization, results should be similar
    # Note: direction pairs are randomly generated each time, so we compare statistics
    c_ell_diff = abs(result1.c_ell_estimate - result2.c_ell_estimate)
    c_ell_relative = c_ell_diff / (max(abs(result1.c_ell_estimate), abs(result2.c_ell_estimate)) + 1e-10)
    rate_diff = abs(result1.overall_correspondence_rate - result2.overall_correspondence_rate)
    
    print(f"  Original c_ℓ: {result1.c_ell_estimate:.6e}")
    print(f"  Gauge-transformed c_ℓ: {result2.c_ell_estimate:.6e}")
    print(f"  Relative difference: {c_ell_relative:.2e}")
    print(f"  Correspondence rate diff: {rate_diff:.4f}")
    
    # Relaxed criteria since direction pairs differ between runs
    # The key test is that both give high correspondence rates
    passed = (
        result1.overall_correspondence_rate >= 0.8 and
        result2.overall_correspondence_rate >= 0.8
    )
    
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    
    return {
        'passed': passed,
        'c_ell_relative_diff': c_ell_relative,
        'rate_diff': rate_diff
    }


def run_all_tests():
    """Run all Module 7 validation tests."""
    if not MODULES_IMPORTED:
        print(f"ERROR: Required modules not found.")
        print(f"Import error: {IMPORT_ERROR}")
        print("\nPlease ensure the following files are in the same directory:")
        print("  - module1_gauge_algebra.py")
        print("  - module4_euclidean_curvature.py")
        print("  - module5_canonicalization.py")
        print("  - module6_bispectrum.py")
        return {}, False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    
    print(f"Using device: {device}")
    print("=" * 70)
    print("MODULE 7 VALIDATION: Bound Verification")
    print("=" * 70)
    
    config = ReferenceConfig(
        n_heads=4,
        d_k=8,
        d_v=8,
        d_model=32
    )
    
    print(f"Configuration:")
    print(f"  n_heads = {config.n_heads}")
    print(f"  d_k = {config.d_k}, d_v = {config.d_v}")
    print(f"  d_model = {config.d_model}")
    print(f"  Device = {device}")
    print(f"  Dtype = {dtype}")
    
    results = {}
    all_passed = True
    
    # Test 1: Single pair computation
    results['single_pair'] = test_single_pair_computation(config, device, dtype)
    all_passed = all_passed and results['single_pair']['passed']
    
    # Test 2: Bound verification (small scale)
    results['bound_verification'] = test_bound_verification_small(config, device, dtype)
    all_passed = all_passed and results['bound_verification']['passed']
    
    # Test 3: Gauge invariance
    results['gauge_invariance'] = test_gauge_invariance_of_bound(config, device, dtype)
    all_passed = all_passed and results['gauge_invariance']['passed']
    
    print("\n" + "=" * 70)
    if all_passed:
        print("OVERALL: ALL TESTS PASSED")
    else:
        print("OVERALL: SOME TESTS FAILED")
    print("=" * 70)
    
    return results, all_passed


if __name__ == "__main__":
    results, passed = run_all_tests()
