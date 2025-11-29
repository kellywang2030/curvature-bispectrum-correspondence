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

# --- validation_track2_training.py ---
"""Track 2: Training Dynamics Validation

Verifies that the correspondence persists through optimization and is
not merely an initialization artifact.
"""


#!/usr/bin/env python3
"""
Validation Track 2: Training Dynamics Analysis

This script reproduces Figure 1 from the paper by tracking the curvature-bispectrum
correspondence through training. The paper shows:

Panel (a): Bound validity starting at 100%, dipping to 90% at step 5000, 
           recovering to 100% by step 10000
Panel (b): c_ℓ stability between 1.15e-3 and 1.21e-3 (5% variation)
Panel (c): Curvature growing from 0.284 to 0.418 (47% increase)
           Energy decreasing from 7.31 to 6.89 (6% reduction)

Experimental protocol:
- 12-head attention layer (d_model = 768 to match paper)
- 10,000 training steps with AdamW optimizer
- Checkpoints at steps: 0, 100, 500, 1000, 2500, 5000, 10000
- 30 direction pairs evaluated at each checkpoint
- Track curvature, energy, c_ℓ, and correspondence rate
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Import our modules
from module1_gauge_algebra import (
    ReferenceConfig, MHAParams, HeadParams, MHATangent,
    random_mha_params, mha_forward
)
from module4_euclidean_curvature import (
    DiscreteHolonomy, EuclideanMechanicalConnection,
    generate_horizontal_direction_pair
)
from module5_canonicalization import canonicalize
from module6_bispectrum import BispectrumConfig, compute_directional_bispectral_energy
from module7_bound_verification import (
    BoundVerificationConfig, verify_bound, BoundVerificationResult
)


@dataclass
class TrainingConfig:
    """Configuration for training dynamics experiment."""
    # Model architecture (paper uses 12 heads)
    n_heads: int = 12
    d_k: int = 64
    d_v: int = 64
    
    @property
    def d_model(self) -> int:
        return self.n_heads * self.d_v
    
    # Training parameters (from paper Appendix E.4)
    total_steps: int = 10000
    batch_size: int = 256
    seq_length: int = 128
    learning_rate: float = 5e-4
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    
    # Checkpoint schedule (from paper)
    checkpoint_steps: List[int] = field(
        default_factory=lambda: [0, 100, 500, 1000, 2500, 5000, 10000]
    )
    
    # Validation config
    n_direction_pairs: int = 30
    n_train_pairs: int = 15
    n_test_pairs: int = 15
    epsilon_values: List[float] = field(default_factory=lambda: [1e-3, 2e-3, 4e-3])
    
    # Evaluation batch (fixed across checkpoints)
    eval_batch_size: int = 64
    eval_seq_length: int = 32
    
    # Output
    output_dir: str = "results/track2_training"
    
    def to_reference_config(self) -> ReferenceConfig:
        return ReferenceConfig(
            n_heads=self.n_heads,
            d_k=self.d_k,
            d_v=self.d_v,
            d_model=self.d_model
        )


class TrainableMHA(nn.Module):
    """
    Trainable Multi-Head Attention module.
    
    This wraps our MHAParams structure in an nn.Module for training.
    """
    
    def __init__(self, config: ReferenceConfig):
        super().__init__()
        self.config = config
        
        # Create parameters for each head
        self.W_Q = nn.ParameterList([
            nn.Parameter(torch.randn(config.d_model, config.d_k) * 0.02)
            for _ in range(config.n_heads)
        ])
        self.W_K = nn.ParameterList([
            nn.Parameter(torch.randn(config.d_model, config.d_k) * 0.02)
            for _ in range(config.n_heads)
        ])
        self.W_V = nn.ParameterList([
            nn.Parameter(torch.randn(config.d_model, config.d_v) * 0.02)
            for _ in range(config.n_heads)
        ])
        self.W_O = nn.ParameterList([
            nn.Parameter(torch.randn(config.d_v, config.d_model) * 0.02)
            for _ in range(config.n_heads)
        ])
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Standard MHA forward pass."""
        batch, seq, _ = X.shape
        output = torch.zeros_like(X)
        
        for i in range(self.config.n_heads):
            Q = X @ self.W_Q[i]
            K = X @ self.W_K[i]
            V = X @ self.W_V[i]
            
            scores = Q @ K.transpose(-2, -1) / np.sqrt(self.config.d_k)
            attn = torch.softmax(scores, dim=-1)
            head_out = attn @ V
            output = output + head_out @ self.W_O[i]
        
        return output
    
    def to_mha_params(self) -> MHAParams:
        """Convert to MHAParams for validation."""
        heads = []
        for i in range(self.config.n_heads):
            heads.append(HeadParams(
                W_Q=self.W_Q[i].data.clone(),
                W_K=self.W_K[i].data.clone(),
                W_V=self.W_V[i].data.clone(),
                W_O=self.W_O[i].data.clone()
            ))
        return MHAParams(heads=heads)
    
    def load_from_mha_params(self, params: MHAParams):
        """Load weights from MHAParams."""
        for i, head in enumerate(params.heads):
            self.W_Q[i].data.copy_(head.W_Q)
            self.W_K[i].data.copy_(head.W_K)
            self.W_V[i].data.copy_(head.W_V)
            self.W_O[i].data.copy_(head.W_O)


class SimpleLanguageModel(nn.Module):
    """
    Simple language model for training the MHA layer.
    
    Architecture: Embedding -> MHA -> LayerNorm -> FFN -> Output
    """
    
    def __init__(self, config: ReferenceConfig, vocab_size: int = 50257):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.attention = TrainableMHA(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model)
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, vocab_size, bias=False)
        
        # Tie embeddings
        self.output_proj.weight = self.embedding.weight
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        x = self.embedding(input_ids)
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return self.output_proj(x)


@dataclass
class CheckpointResult:
    """Validation results at a training checkpoint."""
    step: int
    
    # Bound verification results
    c_ell_estimate: float
    train_correspondence_rate: float
    test_correspondence_rate: float
    overall_correspondence_rate: float
    
    # Invariant magnitudes
    curvature_mean: float
    curvature_std: float
    energy_mean: float
    energy_std: float
    
    # Correlation
    pearson_correlation: float
    spearman_correlation: float
    
    # Numerical diagnostics
    mean_richardson_ratio: float
    
    # Training metrics
    train_loss: float
    
    # Timing
    validation_time_seconds: float


@dataclass
class TrainingResult:
    """Complete results from training dynamics experiment."""
    config: Dict
    checkpoint_results: List[CheckpointResult]
    total_training_time_seconds: float
    total_validation_time_seconds: float


def get_lr_schedule(step: int, warmup_steps: int, total_steps: int, 
                    peak_lr: float, min_lr: float = 1e-5) -> float:
    """Compute learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1 + np.cos(np.pi * progress))


def validate_at_checkpoint(
    model: SimpleLanguageModel,
    eval_batch: torch.Tensor,
    ref_config: ReferenceConfig,
    training_config: TrainingConfig,
    step: int,
    train_loss: float
) -> CheckpointResult:
    """Run full validation at a training checkpoint."""
    
    start_time = time.time()
    
    # Extract MHA parameters
    theta = model.attention.to_mha_params()
    
    # Setup validation
    bound_config = BoundVerificationConfig(
        n_direction_pairs=training_config.n_direction_pairs,
        n_train_pairs=training_config.n_train_pairs,
        n_test_pairs=training_config.n_test_pairs,
        epsilon_values=training_config.epsilon_values
    )
    bispec_config = BispectrumConfig()
    
    # Run bound verification
    result = verify_bound(theta, eval_batch, ref_config, bound_config, bispec_config)
    
    elapsed = time.time() - start_time
    
    return CheckpointResult(
        step=step,
        c_ell_estimate=result.c_ell_estimate,
        train_correspondence_rate=result.train_correspondence_rate,
        test_correspondence_rate=result.test_correspondence_rate,
        overall_correspondence_rate=result.overall_correspondence_rate,
        curvature_mean=result.curvature_mean,
        curvature_std=result.curvature_std,
        energy_mean=result.energy_mean,
        energy_std=result.energy_std,
        pearson_correlation=result.pearson_correlation,
        spearman_correlation=result.spearman_correlation,
        mean_richardson_ratio=result.mean_richardson_ratio,
        train_loss=train_loss,
        validation_time_seconds=elapsed
    )


def run_training_experiment(config: TrainingConfig) -> TrainingResult:
    """Run the complete training dynamics experiment."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    
    print("="*70)
    print("VALIDATION TRACK 2: TRAINING DYNAMICS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {config.n_heads} heads, d_model={config.d_model}")
    print(f"  Training: {config.total_steps} steps, lr={config.learning_rate}")
    print(f"  Checkpoints: {config.checkpoint_steps}")
    print(f"  Device: {device}")
    
    # Create model
    ref_config = config.to_reference_config()
    model = SimpleLanguageModel(ref_config).to(device).to(dtype)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create fixed evaluation batch for consistent validation
    torch.manual_seed(42)
    eval_batch = torch.randn(
        config.eval_batch_size, config.eval_seq_length, config.d_model,
        device=device, dtype=dtype
    )
    
    # Training loop
    checkpoint_results = []
    training_start = time.time()
    total_validation_time = 0.0
    
    current_loss = 0.0
    checkpoint_idx = 0
    
    print(f"\nStarting training...")
    
    for step in range(config.total_steps + 1):
        # Check if we need to validate at this step
        if checkpoint_idx < len(config.checkpoint_steps) and step == config.checkpoint_steps[checkpoint_idx]:
            print(f"\n  Checkpoint at step {step}...")
            
            model.eval()
            with torch.no_grad():
                result = validate_at_checkpoint(
                    model, eval_batch, ref_config, config, step, current_loss
                )
            checkpoint_results.append(result)
            total_validation_time += result.validation_time_seconds
            
            print(f"    Correspondence: {result.overall_correspondence_rate*100:.1f}%")
            print(f"    c_ℓ: {result.c_ell_estimate:.4e}")
            print(f"    Curvature: {result.curvature_mean:.4f}")
            print(f"    Energy: {result.energy_mean:.4f}")
            print(f"    Loss: {result.train_loss:.4f}")
            
            checkpoint_idx += 1
            model.train()
        
        if step >= config.total_steps:
            break
        
        # Training step
        # Generate random input (simulating language modeling)
        torch.manual_seed(step)
        input_ids = torch.randint(0, 1000, (config.batch_size, config.seq_length), device=device)
        target_ids = torch.randint(0, 1000, (config.batch_size, config.seq_length), device=device)
        
        # Forward pass
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, model.vocab_size),
            target_ids.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        
        # Learning rate schedule
        lr = get_lr_schedule(step, config.warmup_steps, config.total_steps, config.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.step()
        current_loss = loss.item()
        
        # Progress update
        if step > 0 and step % 1000 == 0:
            print(f"  Step {step}/{config.total_steps}, Loss: {current_loss:.4f}, LR: {lr:.2e}")
    
    training_elapsed = time.time() - training_start
    
    # Compile results
    result = TrainingResult(
        config=asdict(config),
        checkpoint_results=checkpoint_results,
        total_training_time_seconds=training_elapsed,
        total_validation_time_seconds=total_validation_time
    )
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING DYNAMICS SUMMARY")
    print("="*70)
    print("\nCheckpoint Results (compare with paper Figure 1):")
    print(f"{'Step':>8} | {'Corr Rate':>10} | {'c_ℓ':>12} | {'Curvature':>10} | {'Energy':>10} | {'Loss':>8}")
    print("-" * 70)
    
    for cr in checkpoint_results:
        print(f"{cr.step:>8} | {cr.overall_correspondence_rate*100:>9.1f}% | "
              f"{cr.c_ell_estimate:>12.4e} | {cr.curvature_mean:>10.4f} | "
              f"{cr.energy_mean:>10.4f} | {cr.train_loss:>8.4f}")
    
    # Paper comparisons
    print("\nPaper claims (Figure 1):")
    print("  Panel (a): 100% → 90% (step 5000) → 100% (step 10000)")
    print("  Panel (b): c_ℓ between 1.15e-3 and 1.21e-3")
    print("  Panel (c): Curvature 0.284 → 0.418, Energy 7.31 → 6.89")
    
    if len(checkpoint_results) >= 2:
        first = checkpoint_results[0]
        last = checkpoint_results[-1]
        print(f"\nOur results:")
        print(f"  Correspondence: {first.overall_correspondence_rate*100:.1f}% → {last.overall_correspondence_rate*100:.1f}%")
        print(f"  c_ℓ range: {min(cr.c_ell_estimate for cr in checkpoint_results):.4e} to "
              f"{max(cr.c_ell_estimate for cr in checkpoint_results):.4e}")
        print(f"  Curvature: {first.curvature_mean:.4f} → {last.curvature_mean:.4f}")
        print(f"  Energy: {first.energy_mean:.4f} → {last.energy_mean:.4f}")
    
    print(f"\nTotal training time: {training_elapsed/60:.1f} minutes")
    print(f"Total validation time: {total_validation_time/60:.1f} minutes")
    
    return result


def save_results(result: TrainingResult, output_dir: str) -> Path:
    """Save results to JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"track2_results_{timestamp}.json"
    
    # Convert to serializable format
    data = {
        'config': result.config,
        'checkpoint_results': [asdict(cr) for cr in result.checkpoint_results],
        'total_training_time_seconds': result.total_training_time_seconds,
        'total_validation_time_seconds': result.total_validation_time_seconds
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    """Main entry point for Track 2 validation."""
    
    config = TrainingConfig()
    
    # For quick testing:
    # config.total_steps = 1000
    # config.checkpoint_steps = [0, 100, 500, 1000]
    # config.n_direction_pairs = 10
    # config.n_train_pairs = 5
    # config.n_test_pairs = 5
    
    result = run_training_experiment(config)
    save_results(result, config.output_dir)
    
    return result


if __name__ == "__main__":
    main()
