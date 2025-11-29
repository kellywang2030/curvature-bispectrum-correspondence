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

# --- module1_gauge_algebra.py ---
"""Module 1: Gauge Algebra Structures

Implements the Lie algebra of the maximal gauge group G_max and the
infinitesimal gauge action on multi-head attention parameters.
"""


#!/usr/bin/env python3
"""
Module 1: Gauge Algebra and Vertical Directions

This module implements the foundational data structures and the vertical tangent map
J_θ : g_max → T_θ Θ for the multi-head attention gauge group.

Mathematical Background:
    The MHA gauge group on the generic stratum is:
        G_max = ((GL(d_k))^h × (GL(d_v))^h) ⋊ S_h
    
    acting by:
        (W_Q^(i), W_K^(i)) ↦ (W_Q^(i) A_i, W_K^(i) A_i^{-T})
        (W_V^(i), W_O^(i)) ↦ (W_V^(i) C_i, C_i^{-1} W_O^(i))
    
    The Lie algebra g_max consists of tuples (X_1,...,X_h, Y_1,...,Y_h) where:
        X_i ∈ gl(d_k)  (infinitesimal GL(d_k) for Q-K sector)
        Y_i ∈ gl(d_v)  (infinitesimal GL(d_v) for V-O sector)
    
    The vertical tangent directions (infinitesimal gauge action) are:
        δW_Q^(i) = W_Q^(i) X_i
        δW_K^(i) = -W_K^(i) X_i^T
        δW_V^(i) = W_V^(i) Y_i
        δW_O^(i) = -Y_i W_O^(i)

Reference Configuration:
    h = 4 heads, d_k = d_v = 8, d_model = 32
    Lie algebra dimension: h × (d_k² + d_v²) = 4 × (64 + 64) = 512

Author: Research validation implementation
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

DTYPE = torch.float64  # Use float64 for numerical precision in geometric computations


@dataclass
class ReferenceConfig:
    """Configuration for the reference experiment."""
    n_heads: int = 4
    d_k: int = 8
    d_v: int = 8
    d_model: int = 32  # Should equal n_heads * d_v for standard MHA
    
    def __post_init__(self):
        assert self.d_model == self.n_heads * self.d_v, \
            f"d_model ({self.d_model}) must equal n_heads * d_v ({self.n_heads * self.d_v})"
    
    @property
    def lie_algebra_dim(self) -> int:
        """Dimension of the gauge Lie algebra g_max."""
        return self.n_heads * (self.d_k ** 2 + self.d_v ** 2)
    
    @property
    def param_dim_per_head(self) -> int:
        """Number of parameters per attention head."""
        return (self.d_model * self.d_k +   # W_Q
                self.d_model * self.d_k +   # W_K
                self.d_model * self.d_v +   # W_V
                self.d_v * self.d_model)    # W_O
    
    @property
    def total_param_dim(self) -> int:
        """Total number of attention parameters."""
        return self.n_heads * self.param_dim_per_head


# =============================================================================
# Data Structures for Parameters and Tangent Spaces
# =============================================================================

@dataclass
class HeadParams:
    """
    Parameters for a single attention head.
    
    Attributes:
        W_Q: Query projection matrix [d_model, d_k]
        W_K: Key projection matrix [d_model, d_k]
        W_V: Value projection matrix [d_model, d_v]
        W_O: Output projection matrix [d_v, d_model]
    """
    W_Q: torch.Tensor  # [d_model, d_k]
    W_K: torch.Tensor  # [d_model, d_k]
    W_V: torch.Tensor  # [d_model, d_v]
    W_O: torch.Tensor  # [d_v, d_model]
    
    def validate_shapes(self, d_model: int, d_k: int, d_v: int) -> None:
        """Validate tensor shapes match expected dimensions."""
        assert self.W_Q.shape == (d_model, d_k), \
            f"W_Q shape {self.W_Q.shape} != expected ({d_model}, {d_k})"
        assert self.W_K.shape == (d_model, d_k), \
            f"W_K shape {self.W_K.shape} != expected ({d_model}, {d_k})"
        assert self.W_V.shape == (d_model, d_v), \
            f"W_V shape {self.W_V.shape} != expected ({d_model}, {d_v})"
        assert self.W_O.shape == (d_v, d_model), \
            f"W_O shape {self.W_O.shape} != expected ({d_v}, {d_model})"
    
    def clone(self) -> 'HeadParams':
        """Create a deep copy of this HeadParams."""
        return HeadParams(
            W_Q=self.W_Q.clone(),
            W_K=self.W_K.clone(),
            W_V=self.W_V.clone(),
            W_O=self.W_O.clone()
        )
    
    def to(self, device: torch.device, dtype: torch.dtype = None) -> 'HeadParams':
        """Move tensors to specified device and dtype."""
        return HeadParams(
            W_Q=self.W_Q.to(device=device, dtype=dtype or self.W_Q.dtype),
            W_K=self.W_K.to(device=device, dtype=dtype or self.W_K.dtype),
            W_V=self.W_V.to(device=device, dtype=dtype or self.W_V.dtype),
            W_O=self.W_O.to(device=device, dtype=dtype or self.W_O.dtype)
        )


@dataclass
class MHAParams:
    """
    Parameters for multi-head attention (all heads).
    
    Represents a point θ in the parameter space Θ.
    """
    heads: List[HeadParams]
    
    @property
    def n_heads(self) -> int:
        return len(self.heads)
    
    @property
    def device(self) -> torch.device:
        return self.heads[0].W_Q.device
    
    @property
    def dtype(self) -> torch.dtype:
        return self.heads[0].W_Q.dtype
    
    def validate(self, config: ReferenceConfig) -> None:
        """Validate all head parameters against configuration."""
        assert len(self.heads) == config.n_heads, \
            f"Number of heads {len(self.heads)} != config {config.n_heads}"
        for i, head in enumerate(self.heads):
            head.validate_shapes(config.d_model, config.d_k, config.d_v)
    
    def clone(self) -> 'MHAParams':
        """Create a deep copy of all parameters."""
        return MHAParams(heads=[h.clone() for h in self.heads])
    
    def to(self, device: torch.device, dtype: torch.dtype = None) -> 'MHAParams':
        """Move all tensors to specified device and dtype."""
        return MHAParams(heads=[h.to(device, dtype) for h in self.heads])


@dataclass
class HeadLieAlgebra:
    """
    Lie algebra element for a single head.
    
    Represents an element of gl(d_k) × gl(d_v), the continuous gauge
    directions for one attention head.
    
    Attributes:
        X: Infinitesimal GL(d_k) element for Q-K sector [d_k, d_k]
        Y: Infinitesimal GL(d_v) element for V-O sector [d_v, d_v]
    """
    X: torch.Tensor  # [d_k, d_k]
    Y: torch.Tensor  # [d_v, d_v]
    
    def validate_shapes(self, d_k: int, d_v: int) -> None:
        """Validate tensor shapes."""
        assert self.X.shape == (d_k, d_k), \
            f"X shape {self.X.shape} != expected ({d_k}, {d_k})"
        assert self.Y.shape == (d_v, d_v), \
            f"Y shape {self.Y.shape} != expected ({d_v}, {d_v})"
    
    def scale(self, alpha: float) -> 'HeadLieAlgebra':
        """Scale by a scalar."""
        return HeadLieAlgebra(X=alpha * self.X, Y=alpha * self.Y)
    
    def add(self, other: 'HeadLieAlgebra') -> 'HeadLieAlgebra':
        """Add two Lie algebra elements."""
        return HeadLieAlgebra(X=self.X + other.X, Y=self.Y + other.Y)


@dataclass
class GaugeDirection:
    """
    Full Lie algebra element for all heads.
    
    Represents an element η ∈ g_max = ⊕_i (gl(d_k) ⊕ gl(d_v)).
    This is the input to the vertical tangent map J_θ.
    """
    heads: List[HeadLieAlgebra]
    
    @property
    def n_heads(self) -> int:
        return len(self.heads)
    
    def validate(self, config: ReferenceConfig) -> None:
        """Validate against configuration."""
        assert len(self.heads) == config.n_heads
        for head in self.heads:
            head.validate_shapes(config.d_k, config.d_v)
    
    def scale(self, alpha: float) -> 'GaugeDirection':
        """Scale by a scalar."""
        return GaugeDirection(heads=[h.scale(alpha) for h in self.heads])
    
    def add(self, other: 'GaugeDirection') -> 'GaugeDirection':
        """Add two gauge directions."""
        return GaugeDirection(
            heads=[h1.add(h2) for h1, h2 in zip(self.heads, other.heads)]
        )
    
    def flatten(self) -> torch.Tensor:
        """Flatten to a single vector for linear algebra operations."""
        parts = []
        for h in self.heads:
            parts.append(h.X.flatten())
            parts.append(h.Y.flatten())
        return torch.cat(parts)
    
    @staticmethod
    def unflatten(vec: torch.Tensor, config: ReferenceConfig, 
                  device: torch.device = None) -> 'GaugeDirection':
        """Unflatten from vector form."""
        heads = []
        idx = 0
        x_size = config.d_k ** 2
        y_size = config.d_v ** 2
        
        for _ in range(config.n_heads):
            X = vec[idx:idx + x_size].view(config.d_k, config.d_k)
            idx += x_size
            Y = vec[idx:idx + y_size].view(config.d_v, config.d_v)
            idx += y_size
            heads.append(HeadLieAlgebra(X=X, Y=Y))
        
        return GaugeDirection(heads=heads)


@dataclass
class HeadTangent:
    """
    Tangent vector for a single head's parameters.
    
    Represents an element of T_{θ_i} (parameter space for head i).
    """
    dW_Q: torch.Tensor  # [d_model, d_k]
    dW_K: torch.Tensor  # [d_model, d_k]
    dW_V: torch.Tensor  # [d_model, d_v]
    dW_O: torch.Tensor  # [d_v, d_model]
    
    def validate_shapes(self, d_model: int, d_k: int, d_v: int) -> None:
        """Validate tensor shapes."""
        assert self.dW_Q.shape == (d_model, d_k)
        assert self.dW_K.shape == (d_model, d_k)
        assert self.dW_V.shape == (d_model, d_v)
        assert self.dW_O.shape == (d_v, d_model)
    
    def scale(self, alpha: float) -> 'HeadTangent':
        """Scale by a scalar."""
        return HeadTangent(
            dW_Q=alpha * self.dW_Q,
            dW_K=alpha * self.dW_K,
            dW_V=alpha * self.dW_V,
            dW_O=alpha * self.dW_O
        )
    
    def add(self, other: 'HeadTangent') -> 'HeadTangent':
        """Add two tangent vectors."""
        return HeadTangent(
            dW_Q=self.dW_Q + other.dW_Q,
            dW_K=self.dW_K + other.dW_K,
            dW_V=self.dW_V + other.dW_V,
            dW_O=self.dW_O + other.dW_O
        )
    
    def inner_product_euclidean(self, other: 'HeadTangent') -> torch.Tensor:
        """Compute Euclidean inner product with another tangent."""
        return (torch.sum(self.dW_Q * other.dW_Q) +
                torch.sum(self.dW_K * other.dW_K) +
                torch.sum(self.dW_V * other.dW_V) +
                torch.sum(self.dW_O * other.dW_O))
    
    def norm_euclidean(self) -> torch.Tensor:
        """Compute Euclidean norm."""
        return torch.sqrt(self.inner_product_euclidean(self))


@dataclass
class MHATangent:
    """
    Full tangent vector for multi-head attention parameters.
    
    Represents an element ξ ∈ T_θ Θ (tangent space at θ).
    """
    heads: List[HeadTangent]
    
    @property
    def n_heads(self) -> int:
        return len(self.heads)
    
    def validate(self, config: ReferenceConfig) -> None:
        """Validate against configuration."""
        assert len(self.heads) == config.n_heads
        for head in self.heads:
            head.validate_shapes(config.d_model, config.d_k, config.d_v)
    
    def scale(self, alpha: float) -> 'MHATangent':
        """Scale by a scalar."""
        return MHATangent(heads=[h.scale(alpha) for h in self.heads])
    
    def add(self, other: 'MHATangent') -> 'MHATangent':
        """Add two tangent vectors."""
        return MHATangent(
            heads=[h1.add(h2) for h1, h2 in zip(self.heads, other.heads)]
        )
    
    def subtract(self, other: 'MHATangent') -> 'MHATangent':
        """Subtract another tangent vector."""
        return self.add(other.scale(-1.0))
    
    def inner_product_euclidean(self, other: 'MHATangent') -> torch.Tensor:
        """Compute Euclidean inner product."""
        return sum(h1.inner_product_euclidean(h2) 
                   for h1, h2 in zip(self.heads, other.heads))
    
    def norm_euclidean(self) -> torch.Tensor:
        """Compute Euclidean norm."""
        return torch.sqrt(self.inner_product_euclidean(self))
    
    def flatten(self) -> torch.Tensor:
        """Flatten to a single vector."""
        parts = []
        for h in self.heads:
            parts.append(h.dW_Q.flatten())
            parts.append(h.dW_K.flatten())
            parts.append(h.dW_V.flatten())
            parts.append(h.dW_O.flatten())
        return torch.cat(parts)
    
    @staticmethod
    def unflatten(vec: torch.Tensor, config: ReferenceConfig) -> 'MHATangent':
        """Unflatten from vector form."""
        heads = []
        idx = 0
        q_size = config.d_model * config.d_k
        k_size = config.d_model * config.d_k
        v_size = config.d_model * config.d_v
        o_size = config.d_v * config.d_model
        
        for _ in range(config.n_heads):
            dW_Q = vec[idx:idx + q_size].view(config.d_model, config.d_k)
            idx += q_size
            dW_K = vec[idx:idx + k_size].view(config.d_model, config.d_k)
            idx += k_size
            dW_V = vec[idx:idx + v_size].view(config.d_model, config.d_v)
            idx += v_size
            dW_O = vec[idx:idx + o_size].view(config.d_v, config.d_model)
            idx += o_size
            heads.append(HeadTangent(dW_Q=dW_Q, dW_K=dW_K, dW_V=dW_V, dW_O=dW_O))
        
        return MHATangent(heads=heads)


# =============================================================================
# Vertical Tangent Map: J_θ : g_max → T_θ Θ
# =============================================================================

def vertical_tangent(theta: MHAParams, eta: GaugeDirection) -> MHATangent:
    """
    Compute the vertical tangent direction corresponding to a Lie algebra element.
    
    This implements J_θ : g_max → T_θ Θ, the infinitesimal gauge action.
    
    For each head i, given Lie algebra elements X_i ∈ gl(d_k) and Y_i ∈ gl(d_v):
        δW_Q^(i) = W_Q^(i) @ X_i
        δW_K^(i) = -W_K^(i) @ X_i^T
        δW_V^(i) = W_V^(i) @ Y_i
        δW_O^(i) = -Y_i @ W_O^(i)
    
    Args:
        theta: MHA parameters (the base point in parameter space)
        eta: Lie algebra element (direction in gauge algebra)
    
    Returns:
        MHATangent: The vertical tangent vector at theta
    """
    tangents = []
    
    for head_params, head_dir in zip(theta.heads, eta.heads):
        X = head_dir.X  # [d_k, d_k]
        Y = head_dir.Y  # [d_v, d_v]
        
        # Infinitesimal gauge action formulas
        dW_Q = head_params.W_Q @ X           # [d_model, d_k]
        dW_K = -head_params.W_K @ X.t()      # [d_model, d_k]
        dW_V = head_params.W_V @ Y           # [d_model, d_v]
        dW_O = -Y @ head_params.W_O          # [d_v, d_model]
        
        tangents.append(HeadTangent(
            dW_Q=dW_Q,
            dW_K=dW_K,
            dW_V=dW_V,
            dW_O=dW_O
        ))
    
    return MHATangent(heads=tangents)


def vertical_tangent_transpose(theta: MHAParams, xi: MHATangent) -> GaugeDirection:
    """
    Compute the transpose of the vertical tangent map: J_θ^T : T_θ Θ → g_max.
    
    This is the adjoint with respect to Euclidean inner products on both spaces.
    For the full Fisher-Rao adjoint, we need J_θ^* = J_θ^T G_θ, which requires
    the metric (implemented in Module 2).
    
    The Euclidean adjoint satisfies:
        ⟨J_θ(η), ξ⟩_Euclidean = ⟨η, J_θ^T(ξ)⟩_Euclidean
    
    Args:
        theta: MHA parameters (base point)
        xi: Tangent vector at theta
    
    Returns:
        GaugeDirection: The pullback to the Lie algebra
    """
    heads = []
    
    for head_params, head_tangent in zip(theta.heads, xi.heads):
        # Adjoint computation from the vertical tangent formulas:
        # ⟨W_Q @ X, dW_Q⟩ = ⟨X, W_Q^T @ dW_Q⟩  →  contribution to X
        # ⟨-W_K @ X^T, dW_K⟩ = ⟨X, -dW_K^T @ W_K⟩  →  contribution to X
        # ⟨W_V @ Y, dW_V⟩ = ⟨Y, W_V^T @ dW_V⟩  →  contribution to Y
        # ⟨-Y @ W_O, dW_O⟩ = ⟨Y, -dW_O @ W_O^T⟩  →  contribution to Y
        
        X = (head_params.W_Q.t() @ head_tangent.dW_Q - 
             head_tangent.dW_K.t() @ head_params.W_K)  # [d_k, d_k]
        
        Y = (head_params.W_V.t() @ head_tangent.dW_V - 
             head_tangent.dW_O @ head_params.W_O.t())  # [d_v, d_v]
        
        heads.append(HeadLieAlgebra(X=X, Y=Y))
    
    return GaugeDirection(heads=heads)


# =============================================================================
# Utility Functions
# =============================================================================

def lie_algebra_dim(config: ReferenceConfig) -> int:
    """
    Compute the dimension of the gauge Lie algebra g_max.
    
    dim(g_max) = h × (d_k² + d_v²)
    
    For the reference configuration (h=4, d_k=d_v=8): 4 × (64 + 64) = 512
    """
    return config.n_heads * (config.d_k ** 2 + config.d_v ** 2)


def random_gauge_direction(config: ReferenceConfig, 
                           device: torch.device = None,
                           dtype: torch.dtype = DTYPE) -> GaugeDirection:
    """
    Generate a random Lie algebra element for testing.
    
    Args:
        config: Reference configuration
        device: Target device
        dtype: Target dtype
    
    Returns:
        GaugeDirection: Random element of g_max
    """
    heads = []
    for _ in range(config.n_heads):
        X = torch.randn(config.d_k, config.d_k, device=device, dtype=dtype)
        Y = torch.randn(config.d_v, config.d_v, device=device, dtype=dtype)
        heads.append(HeadLieAlgebra(X=X, Y=Y))
    return GaugeDirection(heads=heads)


def random_tangent(config: ReferenceConfig,
                   device: torch.device = None,
                   dtype: torch.dtype = DTYPE) -> MHATangent:
    """
    Generate a random tangent vector for testing.
    
    Args:
        config: Reference configuration
        device: Target device
        dtype: Target dtype
    
    Returns:
        MHATangent: Random element of T_θ Θ
    """
    heads = []
    for _ in range(config.n_heads):
        dW_Q = torch.randn(config.d_model, config.d_k, device=device, dtype=dtype)
        dW_K = torch.randn(config.d_model, config.d_k, device=device, dtype=dtype)
        dW_V = torch.randn(config.d_model, config.d_v, device=device, dtype=dtype)
        dW_O = torch.randn(config.d_v, config.d_model, device=device, dtype=dtype)
        heads.append(HeadTangent(dW_Q=dW_Q, dW_K=dW_K, dW_V=dW_V, dW_O=dW_O))
    return MHATangent(heads=heads)


def zero_tangent(config: ReferenceConfig,
                 device: torch.device = None,
                 dtype: torch.dtype = DTYPE) -> MHATangent:
    """Create a zero tangent vector."""
    heads = []
    for _ in range(config.n_heads):
        heads.append(HeadTangent(
            dW_Q=torch.zeros(config.d_model, config.d_k, device=device, dtype=dtype),
            dW_K=torch.zeros(config.d_model, config.d_k, device=device, dtype=dtype),
            dW_V=torch.zeros(config.d_model, config.d_v, device=device, dtype=dtype),
            dW_O=torch.zeros(config.d_v, config.d_model, device=device, dtype=dtype)
        ))
    return MHATangent(heads=heads)


def zero_gauge_direction(config: ReferenceConfig,
                         device: torch.device = None,
                         dtype: torch.dtype = DTYPE) -> GaugeDirection:
    """Create a zero gauge direction."""
    heads = []
    for _ in range(config.n_heads):
        heads.append(HeadLieAlgebra(
            X=torch.zeros(config.d_k, config.d_k, device=device, dtype=dtype),
            Y=torch.zeros(config.d_v, config.d_v, device=device, dtype=dtype)
        ))
    return GaugeDirection(heads=heads)


def random_mha_params(config: ReferenceConfig,
                      device: torch.device = None,
                      dtype: torch.dtype = DTYPE,
                      scale: float = 0.02) -> MHAParams:
    """
    Generate random MHA parameters for testing.
    
    Uses standard initialization scale (0.02) matching GPT-2.
    
    Args:
        config: Reference configuration
        device: Target device
        dtype: Target dtype
        scale: Standard deviation for initialization
    
    Returns:
        MHAParams: Random parameters
    """
    heads = []
    for _ in range(config.n_heads):
        heads.append(HeadParams(
            W_Q=torch.randn(config.d_model, config.d_k, device=device, dtype=dtype) * scale,
            W_K=torch.randn(config.d_model, config.d_k, device=device, dtype=dtype) * scale,
            W_V=torch.randn(config.d_model, config.d_v, device=device, dtype=dtype) * scale,
            W_O=torch.randn(config.d_v, config.d_model, device=device, dtype=dtype) * scale
        ))
    return MHAParams(heads=heads)


# =============================================================================
# MHA Forward Pass (for validation)
# =============================================================================

def mha_forward(theta: MHAParams, X: torch.Tensor) -> torch.Tensor:
    """
    Compute multi-head attention output for given parameters.
    
    This implements the standard MHA function:
        MHA(X; θ) = Σ_i softmax(X W_Q^(i) (W_K^(i))^T X^T / √d_k) X W_V^(i) W_O^(i)
    
    Args:
        theta: MHA parameters
        X: Input tensor [batch, seq_len, d_model]
    
    Returns:
        Output tensor [batch, seq_len, d_model]
    """
    batch_size, seq_len, d_model = X.shape
    n_heads = theta.n_heads
    d_k = theta.heads[0].W_Q.shape[1]
    
    output = torch.zeros_like(X)
    
    for head in theta.heads:
        # Q, K, V projections
        Q = X @ head.W_Q  # [batch, seq, d_k]
        K = X @ head.W_K  # [batch, seq, d_k]
        V = X @ head.W_V  # [batch, seq, d_v]
        
        # Attention scores
        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(d_k)  # [batch, seq, seq]
        attn_weights = F.softmax(scores, dim=-1)
        
        # Attention output
        head_out = attn_weights @ V  # [batch, seq, d_v]
        
        # Output projection
        output = output + head_out @ head.W_O  # [batch, seq, d_model]
    
    return output


# =============================================================================
# Validation Tests
# =============================================================================

def test_linearity(config: ReferenceConfig, device: torch.device = None,
                   dtype: torch.dtype = DTYPE, tol: float = 1e-10) -> Dict[str, float]:
    """
    Test that J_θ is linear in the Lie algebra argument.
    
    Verifies: J_θ(a·η₁ + b·η₂) = a·J_θ(η₁) + b·J_θ(η₂)
    
    Returns:
        Dictionary with test results and error metrics
    """
    theta = random_mha_params(config, device, dtype)
    eta1 = random_gauge_direction(config, device, dtype)
    eta2 = random_gauge_direction(config, device, dtype)
    
    a, b = 2.5, -1.3
    
    # Left side: J_θ(a·η₁ + b·η₂)
    combined_eta = eta1.scale(a).add(eta2.scale(b))
    lhs = vertical_tangent(theta, combined_eta)
    
    # Right side: a·J_θ(η₁) + b·J_θ(η₂)
    v1 = vertical_tangent(theta, eta1)
    v2 = vertical_tangent(theta, eta2)
    rhs = v1.scale(a).add(v2.scale(b))
    
    # Compute difference
    diff = lhs.subtract(rhs)
    error = diff.norm_euclidean().item()
    
    # Relative error
    rhs_norm = rhs.norm_euclidean().item()
    rel_error = error / rhs_norm if rhs_norm > 1e-15 else error
    
    passed = rel_error < tol
    
    return {
        'passed': passed,
        'absolute_error': error,
        'relative_error': rel_error,
        'tolerance': tol
    }


def test_adjoint_consistency(config: ReferenceConfig, device: torch.device = None,
                             dtype: torch.dtype = DTYPE, tol: float = 1e-10) -> Dict[str, float]:
    """
    Test that J_θ^T is the Euclidean adjoint of J_θ.
    
    Verifies: ⟨J_θ(η), ξ⟩ = ⟨η, J_θ^T(ξ)⟩
    
    Returns:
        Dictionary with test results and error metrics
    """
    theta = random_mha_params(config, device, dtype)
    eta = random_gauge_direction(config, device, dtype)
    xi = random_tangent(config, device, dtype)
    
    # Left side: ⟨J_θ(η), ξ⟩
    v_eta = vertical_tangent(theta, eta)
    lhs = v_eta.inner_product_euclidean(xi).item()
    
    # Right side: ⟨η, J_θ^T(ξ)⟩
    eta_pullback = vertical_tangent_transpose(theta, xi)
    
    # Compute inner product in Lie algebra (Euclidean)
    rhs = 0.0
    for h_eta, h_pullback in zip(eta.heads, eta_pullback.heads):
        rhs += torch.sum(h_eta.X * h_pullback.X).item()
        rhs += torch.sum(h_eta.Y * h_pullback.Y).item()
    
    # Compute difference
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


def test_gauge_invariance_first_order(config: ReferenceConfig, 
                                       device: torch.device = None,
                                       dtype: torch.dtype = DTYPE,
                                       t_values: List[float] = None,
                                       tol: float = 0.5) -> Dict[str, float]:
    """
    Test that vertical directions preserve the MHA function to first order.
    
    For small t and gauge direction η, the gauge-transformed parameters
    θ' = θ + t·J_θ(η) should satisfy:
        MHA(X; θ') ≈ MHA(X; θ) up to O(t²)
    
    We verify this by checking that ||f(θ') - f(θ)|| / t² is approximately
    constant as t → 0 (quadratic convergence), rather than ||f(θ') - f(θ)|| / t
    being constant (which would indicate first-order, non-gauge behavior).
    
    The key test is: if we halve t, the error should decrease by factor ~4 (quadratic).
    
    Returns:
        Dictionary with test results and error metrics
    """
    if t_values is None:
        t_values = [1e-3, 5e-4, 2.5e-4, 1.25e-4]
    
    theta = random_mha_params(config, device, dtype)
    eta = random_gauge_direction(config, device, dtype)
    
    # Generate test input
    batch_size, seq_len = 4, 8
    X = torch.randn(batch_size, seq_len, config.d_model, device=device, dtype=dtype)
    
    # Compute MHA at base point
    f_base = mha_forward(theta, X)
    f_base_norm = torch.norm(f_base).item()
    
    # Compute vertical tangent
    v_eta = vertical_tangent(theta, eta)
    
    errors = []
    for t in t_values:
        # Create perturbed parameters θ' = θ + t·v_η
        theta_perturbed = MHAParams(heads=[])
        for hp, ht in zip(theta.heads, v_eta.heads):
            theta_perturbed.heads.append(HeadParams(
                W_Q=hp.W_Q + t * ht.dW_Q,
                W_K=hp.W_K + t * ht.dW_K,
                W_V=hp.W_V + t * ht.dW_V,
                W_O=hp.W_O + t * ht.dW_O
            ))
        
        # Compute MHA at perturbed point
        f_perturbed = mha_forward(theta_perturbed, X)
        
        # The difference should be O(t²) for vertical directions
        diff = f_perturbed - f_base
        diff_norm = torch.norm(diff).item()
        errors.append(diff_norm)
    
    # Check quadratic convergence: error(t/2) / error(t) should be ~0.25
    # We compute the convergence ratios between consecutive t values
    convergence_ratios = []
    for i in range(len(errors) - 1):
        t_ratio = t_values[i+1] / t_values[i]  # should be 0.5
        error_ratio = errors[i+1] / errors[i] if errors[i] > 1e-15 else 0
        # For O(t²) behavior: error_ratio ≈ t_ratio²
        expected_ratio = t_ratio ** 2  # ≈ 0.25 for halving t
        convergence_ratios.append(error_ratio / expected_ratio if expected_ratio > 0 else 0)
    
    # The convergence ratios should be close to 1.0 for quadratic behavior
    mean_convergence = np.mean(convergence_ratios) if convergence_ratios else 0
    
    # Pass if convergence is approximately quadratic (ratio near 1.0)
    # Allow some tolerance since softmax introduces mild nonlinearity
    passed = 0.5 < mean_convergence < 2.0
    
    # Also compute the "second-order coefficient" for reference
    # ||f(θ+tη) - f(θ)|| ≈ C·t², so C ≈ error/t²
    second_order_coeffs = [e / (t**2) for e, t in zip(errors, t_values)]
    mean_second_order = np.mean(second_order_coeffs)
    
    return {
        'passed': passed,
        't_values': t_values,
        'errors': errors,
        'convergence_ratios': convergence_ratios,
        'mean_convergence_ratio': mean_convergence,
        'second_order_coeffs': second_order_coeffs,
        'mean_second_order_coeff': mean_second_order,
        'f_base_norm': f_base_norm,
        'tolerance': tol
    }


def run_all_tests(config: ReferenceConfig = None,
                  device: torch.device = None,
                  dtype: torch.dtype = DTYPE) -> Dict[str, Dict]:
    """
    Run all validation tests for Module 1.
    
    Returns:
        Dictionary with results for each test
    """
    if config is None:
        config = ReferenceConfig()
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("MODULE 1 VALIDATION: Gauge Algebra and Vertical Directions")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  n_heads = {config.n_heads}")
    print(f"  d_k = {config.d_k}, d_v = {config.d_v}")
    print(f"  d_model = {config.d_model}")
    print(f"  Lie algebra dimension = {config.lie_algebra_dim}")
    print(f"  Device = {device}")
    print(f"  Dtype = {dtype}")
    
    results = {}
    
    # Test 1: Linearity
    print("\n[Test 1] Linearity of J_θ...")
    results['linearity'] = test_linearity(config, device, dtype)
    status = "PASS" if results['linearity']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Relative error: {results['linearity']['relative_error']:.2e}")
    
    # Test 2: Adjoint consistency
    print("\n[Test 2] Adjoint consistency (J_θ^T)...")
    results['adjoint'] = test_adjoint_consistency(config, device, dtype)
    status = "PASS" if results['adjoint']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Relative error: {results['adjoint']['relative_error']:.2e}")
    
    # Test 3: Gauge invariance (quadratic convergence)
    print("\n[Test 3] Gauge invariance (quadratic convergence)...")
    results['gauge_invariance'] = test_gauge_invariance_first_order(config, device, dtype)
    status = "PASS" if results['gauge_invariance']['passed'] else "FAIL"
    print(f"  Status: {status}")
    print(f"  Mean convergence ratio: {results['gauge_invariance']['mean_convergence_ratio']:.3f}")
    print(f"    (should be ~1.0 for O(t²) behavior; range 0.5-2.0 passes)")
    print(f"  Errors at t = {results['gauge_invariance']['t_values']}:")
    for t, e in zip(results['gauge_invariance']['t_values'], results['gauge_invariance']['errors']):
        print(f"    t={t:.2e}: ||Δf|| = {e:.2e}")
    print(f"  ||f_base||: {results['gauge_invariance']['f_base_norm']:.2e}")
    
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
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run validation tests
    config = ReferenceConfig()
    results = run_all_tests(config, device, DTYPE)
