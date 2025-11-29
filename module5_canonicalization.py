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


# --- module5_canonicalization.py ---
"""Module 5: Canonicalization

Removes continuous gauge freedom through deterministic normalization,
reducing symmetry from G_max to discrete S_h.
"""



#!/usr/bin/env python3
"""
Module 5: Canonicalization (SVD of Gauge-Invariant Products)

This module removes continuous gauge freedom from MHA parameters, leaving only
the residual discrete symmetry S_h (head permutations).

Key insight: Both Q/K and V/O sectors have gauge-invariant products that we
can use to define unique canonical forms via SVD.

Q/K Sector:
- Gauge transform: W_Q -> W_Q @ A, W_K -> W_K @ A^{-T}
- Gauge-invariant: M_QK = W_Q @ W_K^T (since (W_Q@A) @ (W_K@A^{-T})^T = M_QK)
- SVD: M_QK = U @ Σ @ V^T
- Canonical form: W_Q = U @ sqrt(Σ), W_K = V @ sqrt(Σ)
- Verification: W_Q @ W_K^T = U @ sqrt(Σ) @ sqrt(Σ) @ V^T = U @ Σ @ V^T = M_QK ✓

V/O Sector:
- Gauge transform: W_V -> W_V @ C, W_O -> C^{-1} @ W_O
- Gauge-invariant: M_VO = W_V @ W_O (since (W_V@C) @ (C^{-1}@W_O) = M_VO)
- SVD: M_VO = U @ Σ @ V^T
- Canonical form: W_V = U @ sqrt(Σ), W_O = sqrt(Σ) @ V^T
- Verification: W_V @ W_O = U @ sqrt(Σ) @ sqrt(Σ) @ V^T = U @ Σ @ V^T = M_VO ✓

After canonicalization:
- W_Q^T W_Q = W_K^T W_K = diag(Σ_QK) (symmetric form)
- W_V^T W_V = diag(Σ_VO), W_O @ W_O^T = diag(Σ_VO) (symmetric form)
- All gauge-invariant products preserved exactly
- Heads are sorted deterministically

The residual symmetry is exactly S_h acting by head permutation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import sys

# Import from Module 1
from module1_gauge_algebra import (
    ReferenceConfig, HeadParams, MHAParams,
    random_mha_params, mha_forward
)


# ============================================================================
# Canonicalization Metrics
# ============================================================================

@dataclass
class HeadCanonMetrics:
    """Metrics for a single head's canonicalization."""
    V_orthonormality_before: float  # ||W_V^T W_V - I||_F
    V_orthonormality_after: float
    Q_singular_values: List[float]  # Singular values of W_Q
    K_singular_values: List[float]  # Singular values of W_K


@dataclass
class CanonMetrics:
    """Complete canonicalization metrics."""
    head_metrics: List[HeadCanonMetrics]
    head_permutation: List[int]     # Applied permutation [new_idx -> old_idx]
    head_sort_scores: List[float]   # Scores used for sorting


# ============================================================================
# Per-Head Canonicalization
# ============================================================================

def canonicalize_head(
    W_Q: torch.Tensor,  # [d_model, d_k]
    W_K: torch.Tensor,  # [d_model, d_k]
    W_V: torch.Tensor,  # [d_model, d_v]
    W_O: torch.Tensor,  # [d_v, d_model]
    config: ReferenceConfig
) -> Tuple[HeadParams, HeadCanonMetrics]:
    """
    Canonicalize a single attention head.
    
    Both Q/K and V/O sectors use SVD of their gauge-invariant products:
    
    Q/K Sector:
    - Gauge-invariant: M_QK = W_Q @ W_K^T  (since (W_Q@A) @ (W_K@A^{-T})^T = M_QK)
    - SVD: M_QK = U @ Σ @ V^T
    - Canonical: W_Q = U @ sqrt(Σ), W_K = V @ sqrt(Σ)
    
    V/O Sector:
    - Gauge-invariant: M_VO = W_V @ W_O  (since (W_V@C) @ (C^{-1}@W_O) = M_VO)
    - SVD: M_VO = U @ Σ @ V^T
    - Canonical: W_V = U @ sqrt(Σ), W_O = sqrt(Σ) @ V^T
    
    This gives unique canonical forms (up to sign conventions which we fix).
    """
    device = W_Q.device
    dtype = W_Q.dtype
    d_v = config.d_v
    d_k = config.d_k
    d_model = config.d_model
    
    # ========================================
    # (A) V/O Canonicalization via SVD of W_V @ W_O
    # ========================================
    
    # Before metrics
    V_gram_before = W_V.T @ W_V  # [d_v, d_v]
    I_v = torch.eye(d_v, dtype=dtype, device=device)
    V_orth_before = torch.norm(V_gram_before - I_v, p='fro').item()
    
    # Compute the gauge-invariant product
    M_VO = W_V @ W_O  # [d_model, d_model]
    
    # SVD: M_VO = U @ Σ @ V^T
    U_vo, S_vo, Vh_vo = torch.linalg.svd(M_VO, full_matrices=False)
    
    # Take only top d_v components (rank of M_VO <= d_v)
    U_v = U_vo[:, :d_v]      # [d_model, d_v]
    S_v = S_vo[:d_v]         # [d_v]
    Vh_v = Vh_vo[:d_v, :]    # [d_v, d_model]
    
    # Ensure deterministic sign convention: make first nonzero element of each
    # column of U_v positive
    for j in range(d_v):
        col = U_v[:, j]
        nonzero_mask = torch.abs(col) > 1e-10
        if nonzero_mask.any():
            first_nonzero_idx = torch.where(nonzero_mask)[0][0]
            if col[first_nonzero_idx] < 0:
                U_v[:, j] = -U_v[:, j]
                Vh_v[j, :] = -Vh_v[j, :]  # Maintain M = U @ S @ V^T
    
    # Clamp small singular values for numerical stability
    S_v = S_v.clamp(min=1e-12)
    sqrt_S_v = torch.sqrt(S_v)
    
    # Canonical forms: W_V = U @ sqrt(S), W_O = sqrt(S) @ V^T
    W_V_new = U_v @ torch.diag(sqrt_S_v)  # [d_model, d_v]
    W_O_new = torch.diag(sqrt_S_v) @ Vh_v  # [d_v, d_model]
    
    # After metrics for V
    V_gram_after = W_V_new.T @ W_V_new
    V_orth_after = torch.norm(V_gram_after - I_v, p='fro').item()
    
    # ========================================
    # (B) Q/K Canonicalization via SVD of W_Q @ W_K^T
    # ========================================
    
    # Compute the gauge-invariant product
    M_QK = W_Q @ W_K.T  # [d_model, d_model]
    
    # SVD: M_QK = U @ Σ @ V^T
    U_qk, S_qk, Vh_qk = torch.linalg.svd(M_QK, full_matrices=False)
    
    # Take only top d_k components since rank(M_QK) <= d_k
    U_k = U_qk[:, :d_k]      # [d_model, d_k]
    S_k = S_qk[:d_k]         # [d_k]
    V_k = Vh_qk[:d_k, :].T   # [d_model, d_k]
    
    # Ensure deterministic sign convention: make first nonzero element of each
    # column of U_k positive
    for j in range(d_k):
        col = U_k[:, j]
        nonzero_mask = torch.abs(col) > 1e-10
        if nonzero_mask.any():
            first_nonzero_idx = torch.where(nonzero_mask)[0][0]
            if col[first_nonzero_idx] < 0:
                U_k[:, j] = -U_k[:, j]
                V_k[:, j] = -V_k[:, j]  # Maintain M = U @ S @ V^T
    
    # Clamp small singular values for numerical stability
    S_k = S_k.clamp(min=1e-12)
    sqrt_S_k = torch.sqrt(S_k)
    
    # Canonical forms: W_Q = U @ sqrt(S), W_K = V @ sqrt(S)
    W_Q_new = U_k @ torch.diag(sqrt_S_k)  # [d_model, d_k]
    W_K_new = V_k @ torch.diag(sqrt_S_k)  # [d_model, d_k]
    
    # Singular values for metrics
    S_Q_singular = sqrt_S_k
    S_K_singular = sqrt_S_k
    
    # Create canonical head
    canonical_head = HeadParams(
        W_Q=W_Q_new,
        W_K=W_K_new,
        W_V=W_V_new,
        W_O=W_O_new
    )
    
    metrics = HeadCanonMetrics(
        V_orthonormality_before=V_orth_before,
        V_orthonormality_after=V_orth_after,
        Q_singular_values=S_Q_singular.tolist(),
        K_singular_values=S_K_singular.tolist()
    )
    
    return canonical_head, metrics


# ============================================================================
# Head Sorting
# ============================================================================

def compute_head_sort_score(head: HeadParams) -> Tuple[float, ...]:
    """
    Compute a deterministic score tuple for head sorting.
    
    We use multiple criteria to ensure stable ordering:
    1. Frobenius norm of W_V (primary)
    2. Frobenius norm of W_Q (secondary)
    3. Frobenius norm of W_O (tertiary)
    4. Sum of absolute values of first row of W_V (quaternary)
    """
    v_norm = torch.norm(head.W_V, p='fro').item()
    q_norm = torch.norm(head.W_Q, p='fro').item()
    o_norm = torch.norm(head.W_O, p='fro').item()
    v_first_row_sum = torch.sum(torch.abs(head.W_V[0, :])).item()
    
    return (v_norm, q_norm, o_norm, v_first_row_sum)


def sort_heads_deterministic(
    heads: List[HeadParams],
    tau_sort: float = 1e-9
) -> Tuple[List[HeadParams], List[int], List[float]]:
    """
    Sort heads with stable deterministic ordering.
    
    Args:
        heads: List of HeadParams to sort
        tau_sort: Tolerance for rounding scores (for numerical stability)
    
    Returns:
        sorted_heads: Heads in canonical order
        permutation: Mapping new_idx -> old_idx
        scores: Primary sort scores for each head
    """
    n_heads = len(heads)
    
    # Compute score tuples
    score_tuples = [compute_head_sort_score(h) for h in heads]
    
    # Create indexed list for sorting
    indexed = [(i, score_tuples[i]) for i in range(n_heads)]
    
    # Sort by score tuple (Python tuple comparison gives lexicographic order)
    def sort_key(item):
        idx, scores = item
        # Round to tau_sort for stability
        rounded = tuple(round(s / tau_sort) * tau_sort for s in scores)
        return (rounded, idx)  # idx as final tiebreaker
    
    sorted_indexed = sorted(indexed, key=sort_key)
    
    # Extract results
    permutation = [item[0] for item in sorted_indexed]
    sorted_heads = [heads[i] for i in permutation]
    primary_scores = [score_tuples[i][0] for i in permutation]  # V norms
    
    return sorted_heads, permutation, primary_scores


# ============================================================================
# Full Canonicalization
# ============================================================================

def canonicalize(theta: MHAParams, config: ReferenceConfig) -> Tuple[MHAParams, CanonMetrics]:
    """
    Apply full canonicalization to MHA parameters.
    
    Steps:
    1. Canonicalize each head independently (V/O via QR, Q/K via SVD)
    2. Sort heads deterministically
    
    Args:
        theta: Original MHA parameters
        config: Reference configuration
    
    Returns:
        canonical_theta: Canonicalized parameters
        metrics: Canonicalization metrics
    """
    # Step 1: Canonicalize each head
    canonical_heads = []
    head_metrics = []
    
    for head in theta.heads:
        canon_head, metrics = canonicalize_head(
            head.W_Q, head.W_K, head.W_V, head.W_O, config
        )
        canonical_heads.append(canon_head)
        head_metrics.append(metrics)
    
    # Step 2: Sort heads
    sorted_heads, permutation, scores = sort_heads_deterministic(canonical_heads)
    
    # Reorder metrics to match sorted heads
    sorted_metrics = [head_metrics[i] for i in permutation]
    
    # Create canonical MHAParams
    canonical_theta = MHAParams(heads=sorted_heads)
    
    canon_metrics = CanonMetrics(
        head_metrics=sorted_metrics,
        head_permutation=permutation,
        head_sort_scores=scores
    )
    
    return canonical_theta, canon_metrics


# ============================================================================
# Gauge Transform Application (for testing)
# ============================================================================

def apply_gauge_transform(
    theta: MHAParams,
    A_list: List[torch.Tensor],  # [n_heads] list of [d_k, d_k] matrices
    C_list: List[torch.Tensor],  # [n_heads] list of [d_v, d_v] matrices
    sigma: Optional[List[int]] = None  # Head permutation (new_idx -> old_idx mapping)
) -> MHAParams:
    """
    Apply a gauge transformation g ∈ G_max to parameters.
    
    For each head i:
        W_Q -> W_Q @ A_i
        W_K -> W_K @ (A_i^{-1})^T
        W_V -> W_V @ C_i
        W_O -> C_i^{-1} @ W_O
    
    Then permute heads by σ.
    """
    n_heads = len(theta.heads)
    
    # Apply per-head transforms
    transformed_heads = []
    for i, head in enumerate(theta.heads):
        A = A_list[i]
        C = C_list[i]
        A_inv_T = torch.linalg.inv(A).T
        C_inv = torch.linalg.inv(C)
        
        new_head = HeadParams(
            W_Q=head.W_Q @ A,
            W_K=head.W_K @ A_inv_T,
            W_V=head.W_V @ C,
            W_O=C_inv @ head.W_O
        )
        transformed_heads.append(new_head)
    
    # Apply permutation: sigma[i] is the OLD index that goes to NEW position i
    if sigma is not None:
        transformed_heads = [transformed_heads[sigma[i]] for i in range(n_heads)]
    
    return MHAParams(heads=transformed_heads)


def random_gauge_transform(
    config: ReferenceConfig,
    device: torch.device,
    dtype: torch.dtype,
    scale: float = 0.3
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
    """
    Generate a random gauge transformation.
    
    Returns:
        A_list: [n_heads] list of GL(d_k) matrices (invertible)
        C_list: [n_heads] list of GL(d_v) matrices (invertible)
        sigma: Random head permutation
    """
    n_heads = config.n_heads
    d_k = config.d_k
    d_v = config.d_v
    
    A_list = []
    C_list = []
    
    for _ in range(n_heads):
        # Random A: use random orthogonal + scaling to ensure invertible
        # A = Q @ diag(1 + scale * uniform) where Q is random orthogonal
        Q_A, _ = torch.linalg.qr(torch.randn(d_k, d_k, dtype=dtype, device=device))
        scales_A = 1.0 + scale * torch.rand(d_k, dtype=dtype, device=device)
        A = Q_A @ torch.diag(scales_A)
        A_list.append(A)
        
        # Random C: similar construction
        Q_C, _ = torch.linalg.qr(torch.randn(d_v, d_v, dtype=dtype, device=device))
        scales_C = 1.0 + scale * torch.rand(d_v, dtype=dtype, device=device)
        C = Q_C @ torch.diag(scales_C)
        C_list.append(C)
    
    # Random permutation
    sigma = torch.randperm(n_heads).tolist()
    
    return A_list, C_list, sigma


# ============================================================================
# Validation Tests
# ============================================================================

def test_idempotence(config: ReferenceConfig, device: torch.device, 
                     dtype: torch.dtype, tol: float = 1e-10) -> Dict:
    """
    Test that canon(canon(θ)) = canon(θ).
    
    Canonicalization should be idempotent: applying it twice gives the same result.
    """
    theta = random_mha_params(config, device, dtype)
    
    # First canonicalization
    canon1, _ = canonicalize(theta, config)
    
    # Second canonicalization
    canon2, _ = canonicalize(canon1, config)
    
    # Compare
    max_diff = 0.0
    for h1, h2 in zip(canon1.heads, canon2.heads):
        max_diff = max(max_diff, torch.max(torch.abs(h1.W_Q - h2.W_Q)).item())
        max_diff = max(max_diff, torch.max(torch.abs(h1.W_K - h2.W_K)).item())
        max_diff = max(max_diff, torch.max(torch.abs(h1.W_V - h2.W_V)).item())
        max_diff = max(max_diff, torch.max(torch.abs(h1.W_O - h2.W_O)).item())
    
    passed = max_diff < tol
    
    return {
        'passed': passed,
        'max_diff': max_diff,
        'tolerance': tol
    }


def test_gauge_invariance(config: ReferenceConfig, device: torch.device,
                          dtype: torch.dtype, n_transforms: int = 10,
                          tol: float = 1e-8) -> Dict:
    """
    Test that canon(g·θ) = canon(θ) for random gauge transforms g.
    
    The canonical form should be invariant under gauge transformations.
    """
    theta = random_mha_params(config, device, dtype)
    
    # Canonical form of original
    canon_orig, _ = canonicalize(theta, config)
    
    max_diffs = []
    diagnostics = []
    
    for t in range(n_transforms):
        # Random gauge transform
        A_list, C_list, sigma = random_gauge_transform(config, device, dtype, scale=0.5)
        
        # Apply transform
        theta_transformed = apply_gauge_transform(theta, A_list, C_list, sigma)
        
        # Canonicalize transformed
        canon_transformed, _ = canonicalize(theta_transformed, config)
        
        # Compare individual weight matrices
        Q_diff = max(torch.max(torch.abs(h1.W_Q - h2.W_Q)).item()
                     for h1, h2 in zip(canon_orig.heads, canon_transformed.heads))
        K_diff = max(torch.max(torch.abs(h1.W_K - h2.W_K)).item()
                     for h1, h2 in zip(canon_orig.heads, canon_transformed.heads))
        V_diff = max(torch.max(torch.abs(h1.W_V - h2.W_V)).item()
                     for h1, h2 in zip(canon_orig.heads, canon_transformed.heads))
        O_diff = max(torch.max(torch.abs(h1.W_O - h2.W_O)).item()
                     for h1, h2 in zip(canon_orig.heads, canon_transformed.heads))
        
        # Check gauge-invariant quantities
        M_diff = max(torch.norm(h1.W_Q @ h1.W_K.T - h2.W_Q @ h2.W_K.T).item()
                     for h1, h2 in zip(canon_orig.heads, canon_transformed.heads))
        VO_diff = max(torch.norm(h1.W_V @ h1.W_O - h2.W_V @ h2.W_O).item()
                      for h1, h2 in zip(canon_orig.heads, canon_transformed.heads))
        
        max_diff = max(Q_diff, K_diff, V_diff, O_diff)
        max_diffs.append(max_diff)
        
        diagnostics.append({
            'Q': Q_diff, 'K': K_diff, 'V': V_diff, 'O': O_diff,
            'M': M_diff, 'VO': VO_diff
        })
    
    overall_max = max(max_diffs)
    passed = overall_max < tol
    
    return {
        'passed': passed,
        'max_diff': overall_max,
        'all_diffs': max_diffs,
        'diagnostics': diagnostics,
        'tolerance': tol
    }


def test_function_preservation(config: ReferenceConfig, device: torch.device,
                               dtype: torch.dtype, tol: float = 1e-10) -> Dict:
    """
    Test that MHA(X; θ) = MHA(X; canon(θ)) for random inputs.
    
    Canonicalization should not change the function.
    """
    theta = random_mha_params(config, device, dtype)
    canon_theta, _ = canonicalize(theta, config)
    
    # Random input
    batch_size = 16
    seq_len = 32
    X = torch.randn(batch_size, seq_len, config.d_model, dtype=dtype, device=device)
    
    # Compute outputs
    out_orig = mha_forward(theta, X)
    out_canon = mha_forward(canon_theta, X)
    
    # Compare
    max_diff = torch.max(torch.abs(out_orig - out_canon)).item()
    rel_diff = max_diff / (torch.max(torch.abs(out_orig)).item() + 1e-10)
    
    passed = max_diff < tol
    
    return {
        'passed': passed,
        'max_diff': max_diff,
        'relative_diff': rel_diff,
        'tolerance': tol
    }


def test_v_orthonormality(config: ReferenceConfig, device: torch.device,
                          dtype: torch.dtype, tol: float = 1e-10) -> Dict:
    """
    Test that after canonicalization, W_V and W_O have symmetric form:
    W_V^T W_V = W_O @ W_O^T (both equal diag(Σ) from SVD).
    """
    theta = random_mha_params(config, device, dtype)
    canon_theta, metrics = canonicalize(theta, config)
    
    max_asymmetry = 0.0
    asymmetries = []
    
    for head in canon_theta.heads:
        V_gram = head.W_V.T @ head.W_V  # [d_v, d_v]
        O_gram = head.W_O @ head.W_O.T  # [d_v, d_v]
        
        # Both should be equal (diagonal matrices with singular values squared)
        diff = torch.norm(V_gram - O_gram, p='fro').item()
        norm = (torch.norm(V_gram, p='fro').item() + 1e-10)
        relative_diff = diff / norm
        
        asymmetries.append(relative_diff)
        max_asymmetry = max(max_asymmetry, relative_diff)
    
    passed = max_asymmetry < tol
    
    return {
        'passed': passed,
        'max_error': max_asymmetry,
        'per_head_errors': asymmetries,
        'tolerance': tol
    }


def test_q_canonical_form(config: ReferenceConfig, device: torch.device,
                          dtype: torch.dtype, tol: float = 1e-10) -> Dict:
    """
    Test that after canonicalization, W_Q and W_K have the symmetric SVD form.
    
    The canonical form has W_Q = U @ sqrt(S), W_K = V @ sqrt(S).
    Key properties:
    1. W_Q @ W_K^T should be preserved (gauge-invariant)
    2. W_Q^T W_Q should equal W_K^T W_K (both equal diag(S))
    """
    theta = random_mha_params(config, device, dtype)
    canon_theta, metrics = canonicalize(theta, config)
    
    # Test that W_Q^T W_Q = W_K^T W_K (symmetric form)
    max_asymmetry = 0.0
    asymmetry_errors = []
    
    for head in canon_theta.heads:
        Q_gram = head.W_Q.T @ head.W_Q  # Should equal S (diagonal)
        K_gram = head.W_K.T @ head.W_K  # Should also equal S
        
        diff = torch.norm(Q_gram - K_gram, p='fro').item()
        norm = (torch.norm(Q_gram, p='fro').item() + 1e-10)
        relative_diff = diff / norm
        
        asymmetry_errors.append(relative_diff)
        max_asymmetry = max(max_asymmetry, relative_diff)
    
    passed = max_asymmetry < tol
    
    return {
        'passed': passed,
        'max_asymmetry': max_asymmetry,
        'per_head_asymmetry': asymmetry_errors,
        'tolerance': tol
    }


def test_permutation_equivariance(config: ReferenceConfig, device: torch.device,
                                   dtype: torch.dtype, tol: float = 1e-10) -> Dict:
    """
    Test that head permutation in input leads to corresponding permutation in output.
    
    If we permute heads of θ by σ before canonicalization, the result should be
    related to canon(θ) by some permutation (possibly different from σ due to sorting).
    
    More specifically: the SET of canonical heads should be the same.
    """
    theta = random_mha_params(config, device, dtype)
    
    # Canonical form of original
    canon_orig, _ = canonicalize(theta, config)
    
    # Apply a head permutation to original
    sigma = [2, 0, 3, 1]  # A fixed permutation for h=4
    permuted_heads = [theta.heads[sigma[i]] for i in range(config.n_heads)]
    theta_permuted = MHAParams(heads=permuted_heads)
    
    # Canonicalize the permuted version
    canon_permuted, _ = canonicalize(theta_permuted, config)
    
    # The canonical forms should be identical (same heads in same order)
    max_diff = 0.0
    for h1, h2 in zip(canon_orig.heads, canon_permuted.heads):
        max_diff = max(max_diff, torch.max(torch.abs(h1.W_Q - h2.W_Q)).item())
        max_diff = max(max_diff, torch.max(torch.abs(h1.W_K - h2.W_K)).item())
        max_diff = max(max_diff, torch.max(torch.abs(h1.W_V - h2.W_V)).item())
        max_diff = max(max_diff, torch.max(torch.abs(h1.W_O - h2.W_O)).item())
    
    passed = max_diff < tol
    
    return {
        'passed': passed,
        'max_diff': max_diff,
        'tolerance': tol
    }


def run_all_tests(config: ReferenceConfig = None,
                  device: torch.device = None,
                  dtype: torch.dtype = None) -> Dict:
    """Run all Module 5 validation tests."""
    if config is None:
        config = ReferenceConfig()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.float64
    
    print("=" * 70)
    print("MODULE 5 VALIDATION: Canonicalization (SVD-based)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  n_heads = {config.n_heads}")
    print(f"  d_k = {config.d_k}, d_v = {config.d_v}")
    print(f"  d_model = {config.d_model}")
    print(f"  Device = {device}")
    print(f"  Dtype = {dtype}")
    
    results = {}
    all_passed = True
    
    # Test 1: Idempotence
    print("\n[Test 1] Idempotence: canon(canon(θ)) = canon(θ)...")
    results['idempotence'] = test_idempotence(config, device, dtype)
    status = "PASS" if results['idempotence']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Max diff: {results['idempotence']['max_diff']:.2e}")
    all_passed = all_passed and results['idempotence']['passed']
    
    # Test 2: Gauge invariance
    print("\n[Test 2] Gauge invariance: canon(g·θ) = canon(θ)...")
    results['gauge_invariance'] = test_gauge_invariance(config, device, dtype)
    status = "PASS" if results['gauge_invariance']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Max diff across 10 transforms: {results['gauge_invariance']['max_diff']:.2e}")
    if not results['gauge_invariance']['passed']:
        print(f"  All diffs: {[f'{d:.2e}' for d in results['gauge_invariance']['all_diffs']]}")
        if 'diagnostics' in results['gauge_invariance']:
            diag = results['gauge_invariance']['diagnostics'][0]
            print(f"  First transform breakdown: Q={diag['Q']:.2e}, K={diag['K']:.2e}, V={diag['V']:.2e}, O={diag['O']:.2e}")
            print(f"  Gauge-invariant products: M={diag['M']:.2e}, VO={diag['VO']:.2e}")
    all_passed = all_passed and results['gauge_invariance']['passed']
    
    # Test 3: Function preservation
    print("\n[Test 3] Function preservation: MHA(X;θ) = MHA(X;canon(θ))...")
    results['function_preservation'] = test_function_preservation(config, device, dtype)
    status = "PASS" if results['function_preservation']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Max diff: {results['function_preservation']['max_diff']:.2e}")
    print(f"  Relative diff: {results['function_preservation']['relative_diff']:.2e}")
    all_passed = all_passed and results['function_preservation']['passed']
    
    # Test 4: V/O symmetric form
    print("\n[Test 4] V/O symmetric form: W_V^T W_V = W_O @ W_O^T...")
    results['v_orthonormality'] = test_v_orthonormality(config, device, dtype)
    status = "PASS" if results['v_orthonormality']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Max asymmetry: {results['v_orthonormality']['max_error']:.2e}")
    all_passed = all_passed and results['v_orthonormality']['passed']
    
    # Test 5: Q/K symmetric canonical form
    print("\n[Test 5] Q/K symmetric form: W_Q^T W_Q = W_K^T W_K...")
    results['q_canonical'] = test_q_canonical_form(config, device, dtype)
    status = "PASS" if results['q_canonical']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Max asymmetry: {results['q_canonical']['max_asymmetry']:.2e}")
    all_passed = all_passed and results['q_canonical']['passed']
    
    # Test 6: Permutation equivariance
    print("\n[Test 6] Permutation equivariance: canon(σ·θ) = canon(θ)...")
    results['permutation_equivariance'] = test_permutation_equivariance(config, device, dtype)
    status = "PASS" if results['permutation_equivariance']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Max diff: {results['permutation_equivariance']['max_diff']:.2e}")
    all_passed = all_passed and results['permutation_equivariance']['passed']
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("OVERALL: ALL TESTS PASSED")
    else:
        print("OVERALL: SOME TESTS FAILED")
    print("=" * 70)
    
    # Print sample canonicalization metrics
    print("\n[Sample Canonicalization Metrics]")
    theta = random_mha_params(config, device, dtype)
    _, metrics = canonicalize(theta, config)
    
    print(f"  Head permutation: {metrics.head_permutation}")
    print(f"  Sort scores (V norms): {[f'{s:.4f}' for s in metrics.head_sort_scores]}")
    print("  Per-head metrics:")
    for i, hm in enumerate(metrics.head_metrics):
        print(f"    Head {i}: V_orth {hm.V_orthonormality_before:.4f} -> {hm.V_orthonormality_after:.2e}")
        print(f"             Q singular values: {[f'{s:.4f}' for s in hm.Q_singular_values]}")
    
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
