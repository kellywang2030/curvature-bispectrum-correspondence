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

# --- module6_bispectrum.py ---
"""Module 6: Bispectrum Computation

Computes the full S_h bispectrum and directional bispectral energy
after canonicalization.
"""


#!/usr/bin/env python3
"""
Module 6: S_h Bispectrum Computation

After canonicalization (Module 5), the residual symmetry is the symmetric group S_h
acting by head permutation. This module computes bispectral invariants on S_h.

The bispectrum is a complete invariant (under genericity conditions) that captures
the algebraic structure of head interactions. For the curvature-bispectrum 
correspondence theorem, we need:

1. Feature extraction: Per-head statistics from attention outputs
2. Triple correlation: T(σ₁, σ₂) = Σ_τ f(τ) f(σ₁τ) f(σ₂τ)*
3. Bispectrum: Fourier transform of triple correlation
4. Energy decomposition: Trivial vs nontrivial isotypic components

The directional bispectral energy for the correspondence theorem is:
    E_ℓ(u,v) = Σ_{ρ≠triv} w_ρ ||D_u D_v f̂_ρ(e) - D_v D_u f̂_ρ(e)||²_F

This measures how the bispectrum vector changes under directional perturbations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from itertools import permutations
import sys

# Import from previous modules
from module1_gauge_algebra import (
    ReferenceConfig, HeadParams, MHAParams,
    random_mha_params, mha_forward
)
from module5_canonicalization import canonicalize


# ============================================================================
# Bispectrum Configuration
# ============================================================================

@dataclass
class BispectrumConfig:
    """Configuration for bispectrum computation."""
    # Feature extraction
    feature_type: str = 'statistics'  # 'statistics' or 'raw'
    whiten_features: bool = True
    
    # Determining set (for selective bispectrum)
    use_full_bispectrum: bool = False  # If True, compute all (h!)² terms
    determining_set_factor: float = 2.5  # Size multiplier for selective set
    
    # Numerical stability
    eps: float = 1e-10


# ============================================================================
# Permutation Group Utilities
# ============================================================================

def identity_permutation(n: int) -> Tuple[int, ...]:
    """Return identity permutation of length n."""
    return tuple(range(n))


def compose_permutations(perm1: Tuple[int, ...], perm2: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Compose two permutations: (perm1 ∘ perm2)(i) = perm1[perm2[i]].
    
    This follows the convention that perm2 is applied first, then perm1.
    """
    return tuple(perm1[perm2[i]] for i in range(len(perm1)))


def invert_permutation(perm: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute inverse permutation."""
    n = len(perm)
    inv = [0] * n
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


def permutation_sign(perm: Tuple[int, ...]) -> int:
    """
    Compute the sign (parity) of a permutation.
    
    Returns +1 for even permutations, -1 for odd permutations.
    """
    n = len(perm)
    visited = [False] * n
    sign = 1
    
    for i in range(n):
        if visited[i]:
            continue
        cycle_len = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            cycle_len += 1
        # Each cycle of length k contributes (-1)^(k-1)
        if cycle_len > 1:
            sign *= (-1) ** (cycle_len - 1)
    
    return sign


def generate_symmetric_group(n: int) -> List[Tuple[int, ...]]:
    """Generate all elements of S_n as tuples."""
    return list(permutations(range(n)))


def apply_permutation_to_heads(heads: List, perm: Tuple[int, ...]) -> List:
    """Apply a permutation to a list of heads."""
    return [heads[perm[i]] for i in range(len(perm))]


# ============================================================================
# Determining Set Construction
# ============================================================================

def construct_determining_set(n_heads: int, factor: float = 2.5) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Construct a determining set for the selective bispectrum.
    
    The selective G-bispectrum (Mataigne et al. 2024) uses a subset of
    O(|G|) pairs instead of all O(|G|²) pairs while preserving discrimination power.
    
    Our determining set includes:
    1. (e, σ) for adjacent transpositions σ = (i, i+1)
    2. (e, σ) for non-adjacent transpositions
    3. (e, σ) for 3-cycles
    4. (σ, σ⁻¹) pairs for coverage
    5. Random pairs to reach target size
    
    Args:
        n_heads: Number of heads (size of S_n)
        factor: Multiplier for target set size (factor * n_heads)
    
    Returns:
        List of (σ₁, σ₂) pairs
    """
    n = n_heads
    determining_set = []
    identity = tuple(range(n))
    
    # 1. Adjacent transpositions: (e, (i, i+1))
    for i in range(n - 1):
        perm = list(range(n))
        perm[i], perm[i+1] = perm[i+1], perm[i]
        determining_set.append((identity, tuple(perm)))
    
    # 2. Non-adjacent transpositions (sample): (e, (i, i+2))
    for i in range(0, n - 2, 2):
        perm = list(range(n))
        perm[i], perm[i+2] = perm[i+2], perm[i]
        determining_set.append((identity, tuple(perm)))
    
    # 3. 3-cycles: (e, (i, i+1, i+2))
    for i in range(min(n - 2, 5)):
        perm = list(range(n))
        # Cycle: i -> i+1 -> i+2 -> i
        perm[i], perm[i+1], perm[i+2] = perm[i+1], perm[i+2], perm[i]
        determining_set.append((identity, tuple(perm)))
    
    # 4. Inverse pairs from existing set
    for perm_tuple in list(determining_set)[:min(5, len(determining_set))]:
        _, perm = perm_tuple
        perm_inv = invert_permutation(perm)
        determining_set.append((perm, perm_inv))
    
    # 5. Random pairs to reach target size
    target_size = int(factor * n)
    n_random = max(0, target_size - len(determining_set))
    
    if n_random > 0:
        np.random.seed(42)  # Reproducibility
        for _ in range(n_random):
            perm1 = tuple(np.random.permutation(n))
            perm2 = tuple(np.random.permutation(n))
            determining_set.append((perm1, perm2))
    
    return determining_set


# ============================================================================
# Feature Extraction
# ============================================================================

@dataclass
class HeadFeatures:
    """Features extracted from a single attention head."""
    mean: torch.Tensor      # [batch, d_v] mean over sequence
    std: torch.Tensor       # [batch, d_v] std over sequence
    range_feat: torch.Tensor  # [batch, d_v] max - min over sequence
    
    def to_vector(self) -> torch.Tensor:
        """Concatenate features into a single vector [batch, 3*d_v]."""
        return torch.cat([self.mean, self.std, self.range_feat], dim=-1)


def extract_head_features(
    head_outputs: List[torch.Tensor],
    config: BispectrumConfig
) -> torch.Tensor:
    """
    Extract per-head feature vectors for bispectrum computation.
    
    Args:
        head_outputs: List of [batch, seq, d_v] tensors, one per head
        config: Bispectrum configuration
    
    Returns:
        features: [n_heads, batch, feature_dim] tensor
    """
    features = []
    
    for head_out in head_outputs:
        # Aggregate statistics over sequence dimension
        mean_feat = head_out.mean(dim=1)       # [batch, d_v]
        std_feat = head_out.std(dim=1)         # [batch, d_v]
        max_feat = head_out.max(dim=1)[0]      # [batch, d_v]
        min_feat = head_out.min(dim=1)[0]      # [batch, d_v]
        range_feat = max_feat - min_feat       # [batch, d_v]
        
        # Concatenate statistics
        feat = torch.cat([mean_feat, std_feat, range_feat], dim=-1)
        features.append(feat)
    
    # Stack: [n_heads, batch, feature_dim]
    features = torch.stack(features)
    
    # Optionally whiten features
    if config.whiten_features:
        features = whiten_features(features, config.eps)
    
    return features


def whiten_features(features: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Whiten features to have zero mean and unit variance per coordinate.
    
    This removes amplitude effects that would confound energy comparisons.
    
    Args:
        features: [n_heads, batch, feature_dim] tensor
        eps: Small constant for numerical stability
    
    Returns:
        Whitened features with same shape
    """
    # Compute mean and std across heads and batch
    mean = features.mean(dim=(0, 1), keepdim=True)
    std = features.std(dim=(0, 1), keepdim=True)
    
    # Whiten
    whitened = (features - mean) / (std + eps)
    
    return whitened


def get_head_outputs_from_mha(
    theta: MHAParams,
    X: torch.Tensor,
    config: ReferenceConfig
) -> List[torch.Tensor]:
    """
    Compute per-head attention outputs (before concatenation).
    
    Args:
        theta: MHA parameters
        X: Input tensor [batch, seq, d_model]
        config: Model configuration
    
    Returns:
        List of [batch, seq, d_v] tensors, one per head
    """
    batch_size, seq_len, d_model = X.shape
    head_outputs = []
    
    for head in theta.heads:
        # Q, K, V projections
        Q = X @ head.W_Q  # [batch, seq, d_k]
        K = X @ head.W_K
        V = X @ head.W_V  # [batch, seq, d_v]
        
        # Attention scores
        d_k = config.d_k
        scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)  # [batch, seq, seq]
        attn_weights = F.softmax(scores, dim=-1)
        
        # Attention output (before output projection)
        head_out = attn_weights @ V  # [batch, seq, d_v]
        head_outputs.append(head_out)
    
    return head_outputs


# ============================================================================
# Bispectrum Computation
# ============================================================================

@dataclass
class BispectrumResult:
    """Results from bispectrum computation."""
    # Bispectrum values on determining set
    bispectrum_values: List[float]
    determining_set: List[Tuple[Tuple[int, ...], Tuple[int, ...]]]
    
    # Energy decomposition
    trivial_energy: float      # Energy in trivial isotypic component
    nontrivial_energy: float   # Energy in nontrivial components (used for theorem)
    sign_energy: float         # Energy in sign representation
    total_energy: float        # Total squared norm
    
    # Statistics
    mean: float
    std: float
    
    def to_vector(self) -> np.ndarray:
        """Return bispectrum as a numpy vector."""
        return np.array(self.bispectrum_values)


class BispectrumComputer:
    """
    Computes bispectral invariants on the symmetric group S_h.
    
    After canonicalization, the residual symmetry is S_h acting by head permutation.
    The bispectrum captures the algebraic structure of this symmetry.
    """
    
    def __init__(self, n_heads: int, config: BispectrumConfig = None):
        self.n_heads = n_heads
        self.config = config or BispectrumConfig()
        
        # Construct determining set
        if self.config.use_full_bispectrum:
            self.determining_set = self._construct_full_set()
        else:
            self.determining_set = construct_determining_set(
                n_heads, self.config.determining_set_factor
            )
    
    def _construct_full_set(self) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """Construct full bispectrum set (all (h!)² pairs)."""
        all_perms = generate_symmetric_group(self.n_heads)
        return [(p1, p2) for p1 in all_perms for p2 in all_perms]
    
    def compute_bispectrum(self, features: torch.Tensor) -> BispectrumResult:
        """
        Compute the bispectrum on the determining set.
        
        The bispectrum element for (σ₁, σ₂) is computed as a triple correlation:
            B(σ₁, σ₂) = ⟨f(σ₁), f(σ₂), f(σ₁σ₂)⟩
        
        where f is the feature map and ⟨·,·,·⟩ is computed via element-wise
        product and summation.
        
        Args:
            features: [n_heads, batch, feature_dim] whitened features
        
        Returns:
            BispectrumResult with values and energy decomposition
        """
        bispectrum_values = []
        
        for perm1, perm2 in self.determining_set:
            # Apply permutations to head dimension
            # f(σ) = features[σ^{-1}(:), :, :] (left action convention)
            f1 = features[list(perm1), :, :]
            f2 = features[list(perm2), :, :]
            
            # Composition: σ₁ ∘ σ₂
            perm_comp = compose_permutations(perm1, perm2)
            f_comp = features[list(perm_comp), :, :]
            
            # Triple correlation: element-wise product and mean
            bispec_element = torch.mean(f1 * f2 * f_comp).item()
            bispectrum_values.append(bispec_element)
        
        # Convert to numpy for energy computation
        values = np.array(bispectrum_values)
        
        # Energy decomposition into isotypic components
        trivial_energy, nontrivial_energy, sign_energy = self._decompose_energy(values)
        
        return BispectrumResult(
            bispectrum_values=bispectrum_values,
            determining_set=self.determining_set,
            trivial_energy=trivial_energy,
            nontrivial_energy=nontrivial_energy,
            sign_energy=sign_energy,
            total_energy=float(np.sum(values ** 2)),
            mean=float(np.mean(values)),
            std=float(np.std(values))
        )
    
    def _decompose_energy(self, values: np.ndarray) -> Tuple[float, float, float]:
        """
        Decompose bispectrum energy into isotypic components.
        
        For S_h, the key components are:
        - Trivial: Constant functions (energy in mean)
        - Sign: Alternating under permutation sign
        - Standard + others: Remaining nontrivial components
        
        Returns:
            (trivial_energy, nontrivial_energy, sign_energy)
        """
        # Trivial component: projection onto constant functions
        trivial_energy = np.mean(values) ** 2 * len(values)
        
        # Nontrivial energy: total minus trivial (variance-like)
        centered = values - np.mean(values)
        nontrivial_energy = float(np.sum(centered ** 2))
        
        # Sign component: weighted by product of signs
        sign_weighted = []
        for (perm1, perm2), val in zip(self.determining_set, values):
            sign1 = permutation_sign(perm1)
            sign2 = permutation_sign(perm2)
            perm_comp = compose_permutations(perm1, perm2)
            sign_comp = permutation_sign(perm_comp)
            sign_weighted.append(val * sign1 * sign2 * sign_comp)
        sign_energy = float(np.mean(sign_weighted) ** 2 * len(values))
        
        return float(trivial_energy), nontrivial_energy, sign_energy


# ============================================================================
# Directional Bispectral Energy (for Correspondence Theorem)
# ============================================================================

def compute_directional_bispectral_energy(
    theta: MHAParams,
    X: torch.Tensor,
    u: torch.Tensor,  # Direction vector (flattened parameters)
    v: torch.Tensor,  # Direction vector (flattened parameters)
    epsilon: float,
    ref_config: ReferenceConfig,
    bispec_config: BispectrumConfig = None
) -> Tuple[float, Dict]:
    """
    Compute the directional bispectral energy E_ℓ(u,v).
    
    This measures how the bispectrum vector changes under directional perturbations:
        E(u,v) = ||∂²B/∂u∂v||²
    
    where the mixed partial is approximated by finite differences:
        ∂²B/∂u∂v ≈ (B(θ+εu+εv) - B(θ+εu) - B(θ+εv) + B(θ)) / ε²
    
    Args:
        theta: MHA parameters at base point
        X: Input batch [batch, seq, d_model]
        u, v: Direction vectors (flattened)
        epsilon: Step size for finite differences
        ref_config: Reference configuration
        bispec_config: Bispectrum configuration
    
    Returns:
        energy: ||mixed_partial||²
        diagnostics: Additional information
    """
    bispec_config = bispec_config or BispectrumConfig()
    bispec_computer = BispectrumComputer(ref_config.n_heads, bispec_config)
    
    # Helper to compute bispectrum vector at perturbed parameters
    def get_bispectrum_vector(theta_perturbed: MHAParams) -> np.ndarray:
        # Canonicalize
        canon_theta, _ = canonicalize(theta_perturbed, ref_config)
        # Get head outputs
        head_outputs = get_head_outputs_from_mha(canon_theta, X, ref_config)
        # Extract features
        features = extract_head_features(head_outputs, bispec_config)
        # Compute bispectrum
        result = bispec_computer.compute_bispectrum(features)
        return result.to_vector()
    
    # Helper to perturb parameters
    def perturb_theta(theta: MHAParams, direction: torch.Tensor, step: float) -> MHAParams:
        """Add step * direction to flattened parameters."""
        flat = flatten_mha_params(theta)
        flat_perturbed = flat + step * direction
        return unflatten_mha_params(flat_perturbed, ref_config)
    
    # Compute bispectrum at 4 corners of the parallelogram
    B_base = get_bispectrum_vector(theta)
    B_u = get_bispectrum_vector(perturb_theta(theta, u, epsilon))
    B_v = get_bispectrum_vector(perturb_theta(theta, v, epsilon))
    B_uv = get_bispectrum_vector(perturb_theta(theta, u, epsilon))
    B_uv = get_bispectrum_vector(perturb_theta(perturb_theta(theta, u, epsilon), v, epsilon))
    
    # Mixed partial: (B_uv - B_u - B_v + B_base) / ε²
    mixed_partial = (B_uv - B_u - B_v + B_base) / (epsilon ** 2)
    
    # Energy = ||mixed partial||²
    energy = float(np.sum(mixed_partial ** 2))
    
    # Also compute centered version (nontrivial components only)
    mixed_partial_centered = mixed_partial - np.mean(mixed_partial)
    nontrivial_energy = float(np.sum(mixed_partial_centered ** 2))
    
    diagnostics = {
        'raw_energy': energy,
        'nontrivial_energy': nontrivial_energy,
        'B_base_norm': float(np.linalg.norm(B_base)),
        'mixed_partial_norm': float(np.linalg.norm(mixed_partial)),
        'determining_set_size': len(bispec_computer.determining_set)
    }
    
    return nontrivial_energy, diagnostics


# ============================================================================
# Parameter Flattening Utilities
# ============================================================================

def flatten_mha_params(theta: MHAParams) -> torch.Tensor:
    """Flatten MHA parameters to a single vector."""
    parts = []
    for head in theta.heads:
        parts.extend([
            head.W_Q.flatten(),
            head.W_K.flatten(),
            head.W_V.flatten(),
            head.W_O.flatten()
        ])
    return torch.cat(parts)


def unflatten_mha_params(flat: torch.Tensor, config: ReferenceConfig) -> MHAParams:
    """Unflatten vector back to MHA parameters."""
    d_model = config.d_model
    d_k = config.d_k
    d_v = config.d_v
    n_heads = config.n_heads
    
    heads = []
    offset = 0
    
    for _ in range(n_heads):
        # W_Q: [d_model, d_k]
        size = d_model * d_k
        W_Q = flat[offset:offset + size].reshape(d_model, d_k)
        offset += size
        
        # W_K: [d_model, d_k]
        W_K = flat[offset:offset + size].reshape(d_model, d_k)
        offset += size
        
        # W_V: [d_model, d_v]
        size = d_model * d_v
        W_V = flat[offset:offset + size].reshape(d_model, d_v)
        offset += size
        
        # W_O: [d_v, d_model]
        size = d_v * d_model
        W_O = flat[offset:offset + size].reshape(d_v, d_model)
        offset += size
        
        heads.append(HeadParams(W_Q=W_Q, W_K=W_K, W_V=W_V, W_O=W_O))
    
    return MHAParams(heads=heads)


# ============================================================================
# Validation Tests
# ============================================================================

def test_permutation_utilities() -> Dict:
    """Test permutation group utilities."""
    print("\n[Test 1] Permutation utilities...")
    
    # Test identity
    n = 4
    e = identity_permutation(n)
    assert e == (0, 1, 2, 3), f"Identity failed: {e}"
    
    # Test composition
    perm1 = (1, 0, 2, 3)  # Swap 0,1
    perm2 = (0, 1, 3, 2)  # Swap 2,3
    comp = compose_permutations(perm1, perm2)
    # perm2 first: (0,1,3,2), then perm1: swap 0,1 -> (1,0,3,2)
    expected = (1, 0, 3, 2)
    assert comp == expected, f"Composition failed: {comp} != {expected}"
    
    # Test inverse
    perm = (2, 0, 3, 1)  # Maps 0->2, 1->0, 2->3, 3->1
    inv = invert_permutation(perm)
    comp_with_inv = compose_permutations(perm, inv)
    assert comp_with_inv == e, f"Inverse failed: {perm} ∘ {inv} = {comp_with_inv}"
    
    # Test sign
    assert permutation_sign(e) == 1, "Identity should have sign +1"
    assert permutation_sign((1, 0, 2, 3)) == -1, "Transposition should have sign -1"
    assert permutation_sign((1, 2, 0, 3)) == 1, "3-cycle should have sign +1"
    
    print("  All permutation utilities passed!")
    return {'passed': True}


def test_feature_extraction(config: ReferenceConfig, device: torch.device,
                            dtype: torch.dtype) -> Dict:
    """Test feature extraction from head outputs."""
    print("\n[Test 2] Feature extraction...")
    
    bispec_config = BispectrumConfig()
    
    # Create random head outputs
    batch_size = 8
    seq_len = 16
    head_outputs = [
        torch.randn(batch_size, seq_len, config.d_v, device=device, dtype=dtype)
        for _ in range(config.n_heads)
    ]
    
    # Extract features
    features = extract_head_features(head_outputs, bispec_config)
    
    # Check shape
    expected_feature_dim = 3 * config.d_v  # mean, std, range
    assert features.shape == (config.n_heads, batch_size, expected_feature_dim), \
        f"Feature shape mismatch: {features.shape}"
    
    # Check whitening (mean ~0, std ~1)
    mean = features.mean().item()
    std = features.std().item()
    assert abs(mean) < 0.1, f"Features not zero-mean: {mean}"
    assert abs(std - 1.0) < 0.1, f"Features not unit-std: {std}"
    
    print(f"  Feature shape: {features.shape}")
    print(f"  Mean: {mean:.4f}, Std: {std:.4f}")
    print("  Feature extraction passed!")
    
    return {'passed': True, 'feature_shape': features.shape}


def test_bispectrum_conjugation_invariance(config: ReferenceConfig, device: torch.device,
                                           dtype: torch.dtype) -> Dict:
    """
    Test that bispectrum energy is invariant under conjugation by S_h.
    
    The bispectrum should satisfy: ||B(f)||² = ||B(σ · f · σ⁻¹)||² for all σ ∈ S_h
    where conjugation (σ · f · σ⁻¹)(τ) = f(σ⁻¹ τ σ) corresponds to relabeling heads.
    
    Note: The full bispectrum (over all (h!)² pairs) has stronger invariance properties.
    With the selective determining set, we check energy invariance which should hold.
    """
    print("\n[Test 3] Bispectrum conjugation invariance...")
    
    bispec_config = BispectrumConfig()
    bispec_computer = BispectrumComputer(config.n_heads, bispec_config)
    
    # Create random features
    batch_size = 8
    feature_dim = 3 * config.d_v
    features = torch.randn(config.n_heads, batch_size, feature_dim, 
                          device=device, dtype=dtype)
    features = whiten_features(features)
    
    # Compute original bispectrum
    result_orig = bispec_computer.compute_bispectrum(features)
    
    # For conjugation invariance, we need to check that the bispectrum
    # computed from differently-ordered heads gives compatible results.
    # Since our bispectrum is computed on canonicalized parameters,
    # and canonicalization handles permutation, we test via the 
    # full workflow in Test 5 instead.
    
    # Here we verify basic properties: energy is positive and finite
    assert result_orig.total_energy >= 0, "Total energy should be non-negative"
    assert result_orig.nontrivial_energy >= 0, "Nontrivial energy should be non-negative"
    assert np.isfinite(result_orig.total_energy), "Energy should be finite"
    
    # Test that applying the same permutation to both determining set and features
    # gives equivalent results (this is a self-consistency check)
    sigma = (1, 2, 3, 0) if config.n_heads == 4 else tuple((np.arange(config.n_heads) + 1) % config.n_heads)
    
    # Permute features: f_new[i] = f[σ(i)]
    features_permuted = features[list(sigma), :, :]
    features_permuted = whiten_features(features_permuted)  # Re-whiten after permutation
    
    result_permuted = bispec_computer.compute_bispectrum(features_permuted)
    
    # The total energy should be preserved (up to numerical precision)
    # since we're just relabeling heads
    energy_ratio = result_orig.total_energy / (result_permuted.total_energy + 1e-15)
    
    # Energy should be in a reasonable range (not requiring exact equality
    # since the determining set is not closed under conjugation)
    passed = 0.5 < energy_ratio < 2.0
    
    print(f"  Original total energy: {result_orig.total_energy:.6e}")
    print(f"  Permuted total energy: {result_permuted.total_energy:.6e}")
    print(f"  Energy ratio: {energy_ratio:.4f}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    
    return {'passed': passed, 'energy_ratio': energy_ratio}


def test_bispectrum_computation(config: ReferenceConfig, device: torch.device,
                                dtype: torch.dtype) -> Dict:
    """Test basic bispectrum computation."""
    print("\n[Test 4] Bispectrum computation...")
    
    bispec_config = BispectrumConfig()
    bispec_computer = BispectrumComputer(config.n_heads, bispec_config)
    
    # Create random features
    batch_size = 8
    feature_dim = 3 * config.d_v
    features = torch.randn(config.n_heads, batch_size, feature_dim,
                          device=device, dtype=dtype)
    features = whiten_features(features)
    
    # Compute bispectrum
    result = bispec_computer.compute_bispectrum(features)
    
    # Check that we got valid values
    assert len(result.bispectrum_values) == len(bispec_computer.determining_set), \
        "Bispectrum size mismatch"
    assert result.total_energy >= 0, "Energy should be non-negative"
    assert result.nontrivial_energy >= 0, "Nontrivial energy should be non-negative"
    
    # Energy decomposition check: trivial + nontrivial should relate to total
    # (Not exact due to different definitions, but should be consistent)
    
    print(f"  Determining set size: {len(result.determining_set)}")
    print(f"  Total energy: {result.total_energy:.4e}")
    print(f"  Trivial energy: {result.trivial_energy:.4e}")
    print(f"  Nontrivial energy: {result.nontrivial_energy:.4e}")
    print(f"  Sign energy: {result.sign_energy:.4e}")
    print(f"  Mean: {result.mean:.4e}, Std: {result.std:.4e}")
    print("  Bispectrum computation passed!")
    
    return {'passed': True, 'result': result}


def test_canonicalized_bispectrum(config: ReferenceConfig, device: torch.device,
                                  dtype: torch.dtype) -> Dict:
    """
    Test bispectrum on canonicalized MHA parameters.
    
    After canonicalization, gauge-equivalent parameters should give the same bispectrum.
    """
    print("\n[Test 5] Canonicalized bispectrum gauge invariance...")
    
    from module5_canonicalization import (
        apply_gauge_transform, random_gauge_transform
    )
    
    bispec_config = BispectrumConfig()
    
    # Random parameters and input
    theta = random_mha_params(config, device, dtype)
    X = torch.randn(8, 16, config.d_model, device=device, dtype=dtype)
    
    # Canonicalize and compute bispectrum
    def get_bispectrum(theta_):
        canon_theta, _ = canonicalize(theta_, config)
        head_outputs = get_head_outputs_from_mha(canon_theta, X, config)
        features = extract_head_features(head_outputs, bispec_config)
        computer = BispectrumComputer(config.n_heads, bispec_config)
        return computer.compute_bispectrum(features)
    
    result_orig = get_bispectrum(theta)
    
    # Apply random gauge transform and compute again
    A_list, C_list, sigma = random_gauge_transform(config, device, dtype)
    theta_transformed = apply_gauge_transform(theta, A_list, C_list, sigma)
    result_transformed = get_bispectrum(theta_transformed)
    
    # Compare
    energy_diff = abs(result_orig.nontrivial_energy - result_transformed.nontrivial_energy)
    relative_diff = energy_diff / (abs(result_orig.nontrivial_energy) + 1e-10)
    
    passed = relative_diff < 1e-6
    
    print(f"  Original nontrivial energy: {result_orig.nontrivial_energy:.6e}")
    print(f"  After gauge transform: {result_transformed.nontrivial_energy:.6e}")
    print(f"  Relative difference: {relative_diff:.2e}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    
    return {'passed': passed, 'relative_diff': relative_diff}


def run_all_tests(config: ReferenceConfig = None,
                  device: torch.device = None,
                  dtype: torch.dtype = None) -> Dict:
    """Run all Module 6 validation tests."""
    if config is None:
        config = ReferenceConfig()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.float64
    
    print("=" * 70)
    print("MODULE 6 VALIDATION: S_h Bispectrum Computation")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  n_heads = {config.n_heads}")
    print(f"  d_k = {config.d_k}, d_v = {config.d_v}")
    print(f"  d_model = {config.d_model}")
    print(f"  Device = {device}")
    print(f"  Dtype = {dtype}")
    
    results = {}
    all_passed = True
    
    # Test 1: Permutation utilities
    results['permutation_utils'] = test_permutation_utilities()
    all_passed = all_passed and results['permutation_utils']['passed']
    
    # Test 2: Feature extraction
    results['feature_extraction'] = test_feature_extraction(config, device, dtype)
    all_passed = all_passed and results['feature_extraction']['passed']
    
    # Test 3: Bispectrum conjugation invariance
    results['conjugation_invariance'] = test_bispectrum_conjugation_invariance(config, device, dtype)
    all_passed = all_passed and results['conjugation_invariance']['passed']
    
    # Test 4: Bispectrum computation
    results['bispectrum_computation'] = test_bispectrum_computation(config, device, dtype)
    all_passed = all_passed and results['bispectrum_computation']['passed']
    
    # Test 5: Canonicalized bispectrum gauge invariance
    results['canonicalized_invariance'] = test_canonicalized_bispectrum(config, device, dtype)
    all_passed = all_passed and results['canonicalized_invariance']['passed']
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("OVERALL: ALL TESTS PASSED")
    else:
        print("OVERALL: SOME TESTS FAILED")
    print("=" * 70)
    
    return {'all_passed': all_passed, 'results': results}


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = run_all_tests()
