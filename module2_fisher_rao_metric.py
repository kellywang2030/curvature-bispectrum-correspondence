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

# --- module2_fisher_rao_metric.py ---
"""Module 2: Fisher-Rao Metric

Implements the empirical Fisher-Rao metric on parameter space via
automatic differentiation.
"""


#!/usr/bin/env python3
"""
Module 2: Fisher-Rao Metric Application

This module implements the Fisher-Rao metric G_θ on the MHA parameter space,
providing the geometric structure needed for the mechanical connection.

Mathematical Background:
    The Fisher-Rao metric at θ measures the distinguishability of model outputs:
    
        G_θ(ξ, η) = E_{x∈B}[⟨∇_θ log p_θ(x), ξ⟩ · ⟨∇_θ log p_θ(x), η⟩]
    
    For neural networks without explicit probabilistic output, we use the
    empirical Fisher-Rao metric based on the Jacobian of the network function:
    
        G_θ(ξ, η) = (1/|B|) Σ_{x∈B} ⟨J_f(θ)·ξ, J_f(θ)·η⟩
    
    where J_f(θ) is the Jacobian of the function f_θ with respect to parameters.
    
    Key operations:
    1. fisher_apply(θ, ξ, batch) → G_θ·ξ  (metric application)
    2. fisher_inner(θ, ξ, η, batch) → ⟨ξ, η⟩_FR  (inner product)
    3. fisher_norm(θ, ξ, batch) → ||ξ||_FR  (norm)

Implementation Strategy:
    We avoid materializing the full metric tensor (which would be p×p for p parameters).
    Instead, we compute G_θ·ξ via:
    1. Forward pass to get f(θ; x)
    2. Compute directional derivative ∂f/∂θ · ξ via forward-mode AD or finite differences
    3. Backward pass to accumulate J^T · (J · ξ) = G_θ · ξ

Reference Configuration:
    h = 4 heads, d_k = d_v = 8, d_model = 32
    Batch: 64 sequences × 32 tokens

Dependencies:
    - Module 1: MHAParams, MHATangent, HeadParams, HeadTangent, etc.

Author: Research validation implementation
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable
import numpy as np

# Import from Module 1
from module1_gauge_algebra import (
    ReferenceConfig, DTYPE,
    HeadParams, MHAParams, HeadTangent, MHATangent,
    GaugeDirection, HeadLieAlgebra,
    vertical_tangent, vertical_tangent_transpose,
    random_mha_params, random_tangent, random_gauge_direction,
    zero_tangent, mha_forward
)


# =============================================================================
# Fisher-Rao Metric Implementation
# =============================================================================

class FisherRaoMetric:
    """
    Fisher-Rao metric on MHA parameter space.
    
    The metric is defined empirically over a fixed evaluation batch:
        G_θ(ξ, η) = (1/N) Σ_i ||J_f(θ; x_i) · ξ||² when ξ = η
    
    where J_f is the Jacobian of the MHA output with respect to parameters.
    
    We implement this without materializing the full Jacobian by using
    vector-Jacobian products (VJPs) and Jacobian-vector products (JVPs).
    """
    
    def __init__(self, config: ReferenceConfig, eval_batch: torch.Tensor):
        """
        Initialize the Fisher-Rao metric with a fixed evaluation batch.
        
        Args:
            config: Reference configuration
            eval_batch: Input tensor [batch_size, seq_len, d_model]
        """
        self.config = config
        self.eval_batch = eval_batch
        self.batch_size = eval_batch.shape[0]
        self.seq_len = eval_batch.shape[1]
        
        # Output dimension for normalization
        self.output_dim = self.batch_size * self.seq_len * config.d_model
    
    def _params_to_tensor_list(self, theta: MHAParams) -> List[torch.Tensor]:
        """Convert MHAParams to a flat list of tensors for autograd."""
        tensors = []
        for head in theta.heads:
            tensors.extend([head.W_Q, head.W_K, head.W_V, head.W_O])
        return tensors
    
    def _tangent_to_tensor_list(self, xi: MHATangent) -> List[torch.Tensor]:
        """Convert MHATangent to a flat list of tensors."""
        tensors = []
        for head in xi.heads:
            tensors.extend([head.dW_Q, head.dW_K, head.dW_V, head.dW_O])
        return tensors
    
    def _tensor_list_to_tangent(self, tensors: List[torch.Tensor], 
                                 config: ReferenceConfig) -> MHATangent:
        """Convert flat list of tensors back to MHATangent."""
        heads = []
        idx = 0
        for _ in range(config.n_heads):
            heads.append(HeadTangent(
                dW_Q=tensors[idx],
                dW_K=tensors[idx + 1],
                dW_V=tensors[idx + 2],
                dW_O=tensors[idx + 3]
            ))
            idx += 4
        return MHATangent(heads=heads)
    
    def _mha_forward_functional(self, param_tensors: List[torch.Tensor], 
                                 X: torch.Tensor) -> torch.Tensor:
        """
        Functional version of MHA forward pass for autograd.
        
        Args:
            param_tensors: Flat list [W_Q_0, W_K_0, W_V_0, W_O_0, W_Q_1, ...]
            X: Input tensor [batch, seq, d_model]
        
        Returns:
            Output tensor [batch, seq, d_model]
        """
        batch_size, seq_len, d_model = X.shape
        d_k = self.config.d_k
        n_heads = self.config.n_heads
        
        output = torch.zeros_like(X)
        
        for i in range(n_heads):
            W_Q = param_tensors[4*i]
            W_K = param_tensors[4*i + 1]
            W_V = param_tensors[4*i + 2]
            W_O = param_tensors[4*i + 3]
            
            # Q, K, V projections
            Q = X @ W_Q  # [batch, seq, d_k]
            K = X @ W_K  # [batch, seq, d_k]
            V = X @ W_V  # [batch, seq, d_v]
            
            # Attention scores
            scores = (Q @ K.transpose(-2, -1)) / np.sqrt(d_k)
            attn_weights = F.softmax(scores, dim=-1)
            
            # Attention output
            head_out = attn_weights @ V  # [batch, seq, d_v]
            
            # Output projection
            output = output + head_out @ W_O
        
        return output
    
    def compute_jvp(self, theta: MHAParams, xi: MHATangent) -> torch.Tensor:
        """
        Compute Jacobian-vector product: J_f(θ) · ξ
        
        This gives the directional derivative of f in direction ξ.
        
        Args:
            theta: MHA parameters (base point)
            xi: Tangent vector (direction)
        
        Returns:
            Directional derivative [batch, seq, d_model]
        """
        param_tensors = self._params_to_tensor_list(theta)
        tangent_tensors = self._tangent_to_tensor_list(xi)
        
        # Use torch.func.jvp for forward-mode AD
        # Create primals and tangents
        def forward_fn(*params):
            return self._mha_forward_functional(list(params), self.eval_batch)
        
        # Compute JVP using torch.autograd.functional
        primals = tuple(param_tensors)
        tangents = tuple(tangent_tensors)
        
        _, jvp_result = torch.autograd.functional.jvp(forward_fn, primals, tangents)
        
        return jvp_result
    
    def compute_vjp(self, theta: MHAParams, v: torch.Tensor) -> MHATangent:
        """
        Compute vector-Jacobian product: J_f(θ)^T · v
        
        This is the adjoint of the Jacobian, giving the gradient of ⟨f, v⟩.
        
        Args:
            theta: MHA parameters (base point)
            v: Vector in output space [batch, seq, d_model]
        
        Returns:
            MHATangent representing J^T · v
        """
        param_tensors = self._params_to_tensor_list(theta)
        
        # Make copies that require gradients
        param_tensors_grad = [p.clone().detach().requires_grad_(True) for p in param_tensors]
        
        # Forward pass
        output = self._mha_forward_functional(param_tensors_grad, self.eval_batch)
        
        # Backward pass with v as the upstream gradient
        grads = torch.autograd.grad(output, param_tensors_grad, grad_outputs=v,
                                     retain_graph=False, create_graph=False)
        
        return self._tensor_list_to_tangent(list(grads), self.config)
    
    def apply(self, theta: MHAParams, xi: MHATangent) -> MHATangent:
        """
        Apply the Fisher-Rao metric: G_θ · ξ = J^T · J · ξ
        
        This computes G_θ(ξ) without materializing the full metric tensor.
        
        The computation is:
        1. Compute J · ξ (JVP) - directional derivative of f
        2. Compute J^T · (J · ξ) (VJP) - gradient of ⟨f, J·ξ⟩
        3. Normalize by batch size
        
        Args:
            theta: MHA parameters (base point)
            xi: Tangent vector to transform
        
        Returns:
            G_θ · ξ as MHATangent
        """
        # Step 1: Compute J · ξ
        jvp = self.compute_jvp(theta, xi)
        
        # Step 2: Compute J^T · (J · ξ)
        vjp = self.compute_vjp(theta, jvp)
        
        # Step 3: Normalize by number of output elements
        # The Fisher-Rao metric is an average over the batch
        scale = 1.0 / self.output_dim
        
        return vjp.scale(scale)
    
    def inner_product(self, theta: MHAParams, xi: MHATangent, 
                      eta: MHATangent) -> torch.Tensor:
        """
        Compute Fisher-Rao inner product: ⟨ξ, η⟩_FR = ⟨ξ, G_θ·η⟩_Euclidean
        
        Args:
            theta: MHA parameters (base point)
            xi: First tangent vector
            eta: Second tangent vector
        
        Returns:
            Scalar inner product value
        """
        G_eta = self.apply(theta, eta)
        return xi.inner_product_euclidean(G_eta)
    
    def norm(self, theta: MHAParams, xi: MHATangent) -> torch.Tensor:
        """
        Compute Fisher-Rao norm: ||ξ||_FR = sqrt(⟨ξ, ξ⟩_FR)
        
        Args:
            theta: MHA parameters (base point)
            xi: Tangent vector
        
        Returns:
            Scalar norm value
        """
        return torch.sqrt(self.inner_product(theta, xi, xi))
    
    def normalize(self, theta: MHAParams, xi: MHATangent) -> MHATangent:
        """
        Normalize a tangent vector to unit FR norm.
        
        Args:
            theta: MHA parameters (base point)
            xi: Tangent vector to normalize
        
        Returns:
            Normalized tangent vector with ||·||_FR = 1
        """
        norm = self.norm(theta, xi)
        if norm.item() < 1e-15:
            return xi
        return xi.scale(1.0 / norm.item())


# =============================================================================
# Fisher-Rao Adjoint for Mechanical Connection
# =============================================================================

def compute_fr_adjoint_J(theta: MHAParams, xi: MHATangent, 
                         fr_metric: FisherRaoMetric) -> GaugeDirection:
    """
    Compute J_θ^* · ξ = J_θ^T · G_θ · ξ, the Fisher-Rao adjoint of J_θ.
    
    This is needed for the mechanical connection equation:
        M_θ · Γ_θ(ξ) = b_θ(ξ)
    where:
        M_θ = J_θ^* · G_θ · J_θ (pulled-back metric on Lie algebra)
        b_θ(ξ) = J_θ^* · G_θ · ξ = J_θ^T · G_θ · ξ
    
    Args:
        theta: MHA parameters (base point)
        xi: Tangent vector
        fr_metric: Fisher-Rao metric instance
    
    Returns:
        GaugeDirection representing J^* · ξ (pullback to Lie algebra)
    """
    # First apply G_θ
    G_xi = fr_metric.apply(theta, xi)
    
    # Then apply J^T (Euclidean adjoint)
    return vertical_tangent_transpose(theta, G_xi)


def compute_M_operator(theta: MHAParams, eta: GaugeDirection,
                       fr_metric: FisherRaoMetric) -> GaugeDirection:
    """
    Apply the M_θ operator: M_θ · η = J_θ^* · G_θ · J_θ · η
    
    This is the pulled-back Fisher-Rao metric on the Lie algebra.
    
    Args:
        theta: MHA parameters (base point)
        eta: Lie algebra element
        fr_metric: Fisher-Rao metric instance
    
    Returns:
        M_θ · η as GaugeDirection
    """
    # J_θ · η (vertical tangent)
    J_eta = vertical_tangent(theta, eta)
    
    # G_θ · J_θ · η
    G_J_eta = fr_metric.apply(theta, J_eta)
    
    # J_θ^T · G_θ · J_θ · η
    return vertical_tangent_transpose(theta, G_J_eta)


# =============================================================================
# Utility Functions
# =============================================================================

def create_eval_batch(config: ReferenceConfig, 
                      batch_size: int = 64,
                      seq_len: int = 32,
                      device: torch.device = None,
                      dtype: torch.dtype = DTYPE,
                      seed: int = 42) -> torch.Tensor:
    """
    Create a fixed evaluation batch for Fisher-Rao metric computation.
    
    Args:
        config: Reference configuration
        batch_size: Number of sequences
        seq_len: Sequence length
        device: Target device
        dtype: Target dtype
        seed: Random seed for reproducibility
    
    Returns:
        Input tensor [batch_size, seq_len, d_model]
    """
    torch.manual_seed(seed)
    return torch.randn(batch_size, seq_len, config.d_model, 
                       device=device, dtype=dtype)


# =============================================================================
# Validation Tests
# =============================================================================

def test_metric_symmetry(config: ReferenceConfig, fr_metric: FisherRaoMetric,
                         device: torch.device, dtype: torch.dtype,
                         tol: float = 1e-8) -> Dict:
    """
    Test that the Fisher-Rao metric is symmetric: ⟨ξ, η⟩_FR = ⟨η, ξ⟩_FR
    """
    theta = random_mha_params(config, device, dtype)
    xi = random_tangent(config, device, dtype)
    eta = random_tangent(config, device, dtype)
    
    inner_xi_eta = fr_metric.inner_product(theta, xi, eta).item()
    inner_eta_xi = fr_metric.inner_product(theta, eta, xi).item()
    
    error = abs(inner_xi_eta - inner_eta_xi)
    rel_error = error / max(abs(inner_xi_eta), abs(inner_eta_xi), 1e-15)
    
    passed = rel_error < tol
    
    return {
        'passed': passed,
        'inner_xi_eta': inner_xi_eta,
        'inner_eta_xi': inner_eta_xi,
        'absolute_error': error,
        'relative_error': rel_error,
        'tolerance': tol
    }


def test_metric_positive_definite(config: ReferenceConfig, fr_metric: FisherRaoMetric,
                                   device: torch.device, dtype: torch.dtype,
                                   n_samples: int = 10) -> Dict:
    """
    Test that the Fisher-Rao metric is positive definite: ⟨ξ, ξ⟩_FR > 0 for ξ ≠ 0
    """
    theta = random_mha_params(config, device, dtype)
    
    norms = []
    for _ in range(n_samples):
        xi = random_tangent(config, device, dtype)
        norm_sq = fr_metric.inner_product(theta, xi, xi).item()
        norms.append(norm_sq)
    
    all_positive = all(n > 0 for n in norms)
    min_norm = min(norms)
    
    return {
        'passed': all_positive,
        'all_positive': all_positive,
        'min_norm_squared': min_norm,
        'norms_squared': norms
    }


def test_metric_linearity(config: ReferenceConfig, fr_metric: FisherRaoMetric,
                          device: torch.device, dtype: torch.dtype,
                          tol: float = 1e-8) -> Dict:
    """
    Test that G_θ is linear: G_θ(a·ξ + b·η) = a·G_θ(ξ) + b·G_θ(η)
    """
    theta = random_mha_params(config, device, dtype)
    xi = random_tangent(config, device, dtype)
    eta = random_tangent(config, device, dtype)
    
    a, b = 2.3, -1.7
    
    # Left side: G_θ(a·ξ + b·η)
    combined = xi.scale(a).add(eta.scale(b))
    lhs = fr_metric.apply(theta, combined)
    
    # Right side: a·G_θ(ξ) + b·G_θ(η)
    G_xi = fr_metric.apply(theta, xi)
    G_eta = fr_metric.apply(theta, eta)
    rhs = G_xi.scale(a).add(G_eta.scale(b))
    
    # Compare
    diff = lhs.subtract(rhs)
    error = diff.norm_euclidean().item()
    rhs_norm = rhs.norm_euclidean().item()
    rel_error = error / rhs_norm if rhs_norm > 1e-15 else error
    
    passed = rel_error < tol
    
    return {
        'passed': passed,
        'absolute_error': error,
        'relative_error': rel_error,
        'tolerance': tol
    }


def test_M_operator_symmetry(config: ReferenceConfig, fr_metric: FisherRaoMetric,
                             device: torch.device, dtype: torch.dtype,
                             tol: float = 1e-8) -> Dict:
    """
    Test that M_θ is symmetric: ⟨η₁, M_θ·η₂⟩ = ⟨η₂, M_θ·η₁⟩
    
    This is crucial for the mechanical connection solver.
    """
    theta = random_mha_params(config, device, dtype)
    eta1 = random_gauge_direction(config, device, dtype)
    eta2 = random_gauge_direction(config, device, dtype)
    
    M_eta1 = compute_M_operator(theta, eta1, fr_metric)
    M_eta2 = compute_M_operator(theta, eta2, fr_metric)
    
    # Inner product in Lie algebra (Euclidean)
    def lie_inner(g1: GaugeDirection, g2: GaugeDirection) -> float:
        total = 0.0
        for h1, h2 in zip(g1.heads, g2.heads):
            total += torch.sum(h1.X * h2.X).item()
            total += torch.sum(h1.Y * h2.Y).item()
        return total
    
    inner_1_M2 = lie_inner(eta1, M_eta2)
    inner_2_M1 = lie_inner(eta2, M_eta1)
    
    error = abs(inner_1_M2 - inner_2_M1)
    rel_error = error / max(abs(inner_1_M2), abs(inner_2_M1), 1e-15)
    
    passed = rel_error < tol
    
    return {
        'passed': passed,
        'inner_1_M2': inner_1_M2,
        'inner_2_M1': inner_2_M1,
        'absolute_error': error,
        'relative_error': rel_error,
        'tolerance': tol
    }


def test_adjoint_consistency(config: ReferenceConfig, fr_metric: FisherRaoMetric,
                             device: torch.device, dtype: torch.dtype,
                             tol: float = 1e-4) -> Dict:
    """
    Test that J^* is the FR-adjoint of J: ⟨J·η, ξ⟩_FR = ⟨η, J^*·ξ⟩_Lie
    
    Note: This test involves two levels of autodiff (JVP then VJP), so we
    expect precision around 1e-5 to 1e-6 rather than machine precision.
    The tolerance is set accordingly.
    """
    theta = random_mha_params(config, device, dtype)
    eta = random_gauge_direction(config, device, dtype)
    xi = random_tangent(config, device, dtype)
    
    # Left side: ⟨J·η, ξ⟩_FR
    J_eta = vertical_tangent(theta, eta)
    lhs = fr_metric.inner_product(theta, J_eta, xi).item()
    
    # Right side: ⟨η, J^*·ξ⟩_Lie (Euclidean inner product on Lie algebra)
    J_star_xi = compute_fr_adjoint_J(theta, xi, fr_metric)
    
    rhs = 0.0
    for h_eta, h_adj in zip(eta.heads, J_star_xi.heads):
        rhs += torch.sum(h_eta.X * h_adj.X).item()
        rhs += torch.sum(h_eta.Y * h_adj.Y).item()
    
    error = abs(lhs - rhs)
    rel_error = error / max(abs(lhs), abs(rhs), 1e-15)
    
    passed = rel_error < tol
    
    return {
        'passed': passed,
        'lhs': lhs,
        'rhs': rhs,
        'absolute_error': error,
        'relative_error': rel_error,
        'tolerance': tol
    }


def run_all_tests(config: ReferenceConfig = None,
                  device: torch.device = None,
                  dtype: torch.dtype = DTYPE) -> Dict:
    """
    Run all validation tests for Module 2.
    """
    if config is None:
        config = ReferenceConfig()
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("MODULE 2 VALIDATION: Fisher-Rao Metric")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  n_heads = {config.n_heads}")
    print(f"  d_k = {config.d_k}, d_v = {config.d_v}")
    print(f"  d_model = {config.d_model}")
    print(f"  Device = {device}")
    print(f"  Dtype = {dtype}")
    
    # Create evaluation batch
    print("\nCreating evaluation batch...")
    eval_batch = create_eval_batch(config, batch_size=64, seq_len=32,
                                    device=device, dtype=dtype)
    print(f"  Batch shape: {eval_batch.shape}")
    
    # Create Fisher-Rao metric
    fr_metric = FisherRaoMetric(config, eval_batch)
    
    results = {}
    
    # Test 1: Symmetry
    print("\n[Test 1] Metric symmetry (⟨ξ,η⟩ = ⟨η,ξ⟩)...")
    results['symmetry'] = test_metric_symmetry(config, fr_metric, device, dtype)
    status = "PASS" if results['symmetry']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Relative error: {results['symmetry']['relative_error']:.2e}")
    
    # Test 2: Positive definiteness
    print("\n[Test 2] Positive definiteness (⟨ξ,ξ⟩ > 0)...")
    results['positive_definite'] = test_metric_positive_definite(config, fr_metric, device, dtype)
    status = "PASS" if results['positive_definite']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Min ||ξ||²_FR: {results['positive_definite']['min_norm_squared']:.2e}")
    
    # Test 3: Linearity
    print("\n[Test 3] Metric linearity (G(aξ+bη) = aG(ξ)+bG(η))...")
    results['linearity'] = test_metric_linearity(config, fr_metric, device, dtype)
    status = "PASS" if results['linearity']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Relative error: {results['linearity']['relative_error']:.2e}")
    
    # Test 4: M operator symmetry
    print("\n[Test 4] M_θ symmetry (⟨η₁,M·η₂⟩ = ⟨η₂,M·η₁⟩)...")
    results['M_symmetry'] = test_M_operator_symmetry(config, fr_metric, device, dtype)
    status = "PASS" if results['M_symmetry']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Relative error: {results['M_symmetry']['relative_error']:.2e}")
    
    # Test 5: FR adjoint consistency
    print("\n[Test 5] FR adjoint (⟨J·η,ξ⟩_FR = ⟨η,J*·ξ⟩_Lie)...")
    print("  (Note: involves double autodiff, expect ~1e-5 precision)")
    results['adjoint'] = test_adjoint_consistency(config, fr_metric, device, dtype)
    status = "PASS" if results['adjoint']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Relative error: {results['adjoint']['relative_error']:.2e}")
    
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
