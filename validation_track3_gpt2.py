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

# --- validation_track3_gpt2.py ---
"""Track 3: Pretrained GPT-2 Validation

Validates the correspondence on production-scale pretrained transformers
(GPT-2 124M and GPT-2 Medium 355M).
"""


#!/usr/bin/env python3
"""
Validation Track 3: GPT-2 Pretrained Model Validation

This script validates the curvature-bispectrum correspondence on real pretrained
GPT-2 models, bridging from our synthetic experiments to production-scale models.

Models to test:
- gpt2 (124M parameters): 12 layers, 12 heads, d_model=768
- gpt2-medium (355M): 24 layers, 16 heads, d_model=1024
- gpt2-large (774M): 36 layers, 20 heads, d_model=1280
- gpt2-xl (1.5B): 48 layers, 25 heads, d_model=1600

For each model, we:
1. Extract attention weights from selected layers (early, middle, late)
2. Apply our canonicalization procedure
3. Compute curvature and bispectral energy
4. Verify the theoretical bound

This validates that our theoretical framework applies to real-world models
trained on natural language data.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: transformers not installed. Install with: pip install transformers")

# Import our modules
from module1_gauge_algebra import ReferenceConfig, MHAParams, HeadParams
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
class GPT2ModelSpec:
    """Specification for a GPT-2 model variant."""
    name: str
    n_layers: int
    n_heads: int
    d_model: int
    d_k: int  # = d_model // n_heads
    n_params: str
    
    @property
    def d_v(self) -> int:
        return self.d_k
    
    def layers_to_analyze(self) -> List[int]:
        """Return early, middle, late layer indices."""
        return [0, self.n_layers // 2, self.n_layers - 1]


GPT2_MODELS = {
    'gpt2': GPT2ModelSpec('gpt2', 12, 12, 768, 64, '124M'),
    'gpt2-medium': GPT2ModelSpec('gpt2-medium', 24, 16, 1024, 64, '355M'),
    'gpt2-large': GPT2ModelSpec('gpt2-large', 36, 20, 1280, 64, '774M'),
    'gpt2-xl': GPT2ModelSpec('gpt2-xl', 48, 25, 1600, 64, '1.5B'),
}


@dataclass
class ValidationConfig:
    """Configuration for GPT-2 validation."""
    # Models to test
    model_names: List[str] = field(default_factory=lambda: ['gpt2', 'gpt2-medium'])
    
    # Validation parameters
    n_direction_pairs: int = 30
    n_train_pairs: int = 15
    n_test_pairs: int = 15
    epsilon_values: List[float] = field(default_factory=lambda: [1e-3, 2e-3, 4e-3])
    
    # Evaluation data
    batch_size: int = 16
    seq_length: int = 64
    
    # Output
    output_dir: str = "results/track3_gpt2"


class GPT2AttentionExtractor:
    """Extract attention weights from GPT-2 in our MHAParams format."""
    
    def __init__(self, model: 'GPT2LMHeadModel', spec: GPT2ModelSpec):
        self.model = model
        self.spec = spec
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
    
    def extract_layer_weights(self, layer_idx: int) -> MHAParams:
        """
        Extract attention weights for a specific layer.
        
        GPT-2 stores attention as:
        - c_attn.weight: [d_model, 3*d_model] containing Q, K, V stacked
        - c_proj.weight: [d_model, d_model] for output projection
        """
        block = self.model.transformer.h[layer_idx]
        
        # Get fused QKV projection
        c_attn = block.attn.c_attn.weight.data  # [d_model, 3*d_model] or transposed
        c_proj = block.attn.c_proj.weight.data  # [d_model, d_model] or transposed
        
        d_model = self.spec.d_model
        n_heads = self.spec.n_heads
        d_k = self.spec.d_k
        d_v = self.spec.d_v
        
        # GPT-2 uses Conv1D which stores [in_features, out_features]
        # c_attn: [d_model, 3*d_model]
        if c_attn.shape[0] == d_model and c_attn.shape[1] == 3 * d_model:
            # Shape is [d_model, 3*d_model], split on dim 1
            W_Q_full, W_K_full, W_V_full = c_attn.split(d_model, dim=1)
        else:
            # Shape is [3*d_model, d_model], need to transpose
            W_Q_full, W_K_full, W_V_full = c_attn.T.split(d_model, dim=1)
        
        # W_Q_full is [d_model, d_model], reshape to per-head [n_heads, d_model, d_k]
        # GPT-2 interleaves heads: first d_k columns are head 0, next d_k are head 1, etc.
        W_Q_heads = W_Q_full.view(d_model, n_heads, d_k).permute(1, 0, 2)
        W_K_heads = W_K_full.view(d_model, n_heads, d_k).permute(1, 0, 2)
        W_V_heads = W_V_full.view(d_model, n_heads, d_v).permute(1, 0, 2)
        
        # Output projection: [d_model, d_model] -> [n_heads, d_v, d_model]
        if c_proj.shape[0] == d_model:
            W_O_full = c_proj
        else:
            W_O_full = c_proj.T
        
        # Split output projection by input dimension (concatenated heads)
        W_O_heads = W_O_full.view(n_heads, d_v, d_model)
        
        # Create MHAParams
        heads = []
        for h in range(n_heads):
            heads.append(HeadParams(
                W_Q=W_Q_heads[h].clone().to(torch.float64),
                W_K=W_K_heads[h].clone().to(torch.float64),
                W_V=W_V_heads[h].clone().to(torch.float64),
                W_O=W_O_heads[h].clone().to(torch.float64)
            ))
        
        return MHAParams(heads=heads)


@dataclass
class LayerResult:
    """Validation results for a single layer."""
    layer_idx: int
    layer_position: str  # 'early', 'middle', 'late'
    
    c_ell_estimate: float
    train_correspondence_rate: float
    test_correspondence_rate: float
    overall_correspondence_rate: float
    
    curvature_mean: float
    curvature_std: float
    energy_mean: float
    energy_std: float
    
    pearson_correlation: float
    mean_richardson_ratio: float
    
    computation_time_seconds: float


@dataclass
class ModelResult:
    """Validation results for a complete GPT-2 model."""
    model_name: str
    n_params: str
    n_layers: int
    n_heads: int
    d_model: int
    
    layer_results: List[LayerResult]
    
    # Aggregate statistics
    mean_correspondence_rate: float = 0.0
    mean_c_ell: float = 0.0
    mean_pearson: float = 0.0
    total_time_seconds: float = 0.0
    
    def compute_aggregates(self):
        if not self.layer_results:
            return
        
        corr_rates = [r.overall_correspondence_rate for r in self.layer_results]
        c_ells = [r.c_ell_estimate for r in self.layer_results]
        pearsons = [abs(r.pearson_correlation) for r in self.layer_results]
        times = [r.computation_time_seconds for r in self.layer_results]
        
        self.mean_correspondence_rate = float(np.mean(corr_rates))
        self.mean_c_ell = float(np.mean(c_ells))
        self.mean_pearson = float(np.mean(pearsons))
        self.total_time_seconds = float(sum(times))


def create_evaluation_batch(
    tokenizer: 'GPT2Tokenizer',
    model: 'GPT2LMHeadModel',
    spec: GPT2ModelSpec,
    batch_size: int,
    seq_length: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create evaluation batch by getting embeddings from sample text.
    
    We use the model's own embeddings to create realistic activation patterns.
    """
    # Sample texts for evaluation
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can learn complex patterns from data.",
        "In mathematics, a group is an algebraic structure.",
        "The attention mechanism allows models to focus on relevant parts.",
        "Transformers have revolutionized natural language processing.",
        "Neural networks are composed of layers of interconnected nodes.",
        "The gradient descent algorithm optimizes model parameters.",
        "Language models predict the probability of word sequences.",
    ] * (batch_size // 8 + 1)
    
    texts = texts[:batch_size]
    
    # Tokenize
    encoded = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=seq_length,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    
    # Get embeddings
    with torch.no_grad():
        embeddings = model.transformer.wte(input_ids)
        positions = torch.arange(seq_length, device=device).unsqueeze(0)
        pos_embeddings = model.transformer.wpe(positions)
        X = (embeddings + pos_embeddings).to(torch.float64)
    
    return X


def validate_layer(
    theta: MHAParams,
    X: torch.Tensor,
    ref_config: ReferenceConfig,
    val_config: ValidationConfig,
    layer_idx: int,
    layer_position: str
) -> LayerResult:
    """Validate a single layer."""
    
    start_time = time.time()
    
    bound_config = BoundVerificationConfig(
        n_direction_pairs=val_config.n_direction_pairs,
        n_train_pairs=val_config.n_train_pairs,
        n_test_pairs=val_config.n_test_pairs,
        epsilon_values=val_config.epsilon_values
    )
    bispec_config = BispectrumConfig()
    
    result = verify_bound(theta, X, ref_config, bound_config, bispec_config)
    
    elapsed = time.time() - start_time
    
    return LayerResult(
        layer_idx=layer_idx,
        layer_position=layer_position,
        c_ell_estimate=result.c_ell_estimate,
        train_correspondence_rate=result.train_correspondence_rate,
        test_correspondence_rate=result.test_correspondence_rate,
        overall_correspondence_rate=result.overall_correspondence_rate,
        curvature_mean=result.curvature_mean,
        curvature_std=result.curvature_std,
        energy_mean=result.energy_mean,
        energy_std=result.energy_std,
        pearson_correlation=result.pearson_correlation,
        mean_richardson_ratio=result.mean_richardson_ratio,
        computation_time_seconds=elapsed
    )


def validate_gpt2_model(
    model_name: str,
    val_config: ValidationConfig,
    device: torch.device
) -> ModelResult:
    """Validate a complete GPT-2 model."""
    
    spec = GPT2_MODELS[model_name]
    
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name} ({spec.n_params} parameters)")
    print(f"{'='*70}")
    print(f"  Architecture: {spec.n_layers} layers, {spec.n_heads} heads, d_model={spec.d_model}")
    
    # Load model
    print(f"  Loading model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    # Create extractor
    extractor = GPT2AttentionExtractor(model, spec)
    
    # Create evaluation batch
    print(f"  Creating evaluation batch...")
    X = create_evaluation_batch(
        tokenizer, model, spec,
        val_config.batch_size, val_config.seq_length, device
    )
    
    # Validate selected layers
    layers_to_analyze = spec.layers_to_analyze()
    layer_positions = ['early', 'middle', 'late']
    
    layer_results = []
    
    for layer_idx, position in zip(layers_to_analyze, layer_positions):
        print(f"\n  Layer {layer_idx} ({position})...")
        
        # Extract weights
        theta = extractor.extract_layer_weights(layer_idx)
        
        # Create reference config for this model
        ref_config = ReferenceConfig(
            n_heads=spec.n_heads,
            d_k=spec.d_k,
            d_v=spec.d_v,
            d_model=spec.d_model
        )
        
        # Validate
        try:
            result = validate_layer(theta, X, ref_config, val_config, layer_idx, position)
            layer_results.append(result)
            
            print(f"    Correspondence: {result.overall_correspondence_rate*100:.1f}%")
            print(f"    c_ℓ: {result.c_ell_estimate:.4e}")
            print(f"    |ρ|: {abs(result.pearson_correlation):.4f}")
            print(f"    Time: {result.computation_time_seconds:.1f}s")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Create model result
    model_result = ModelResult(
        model_name=model_name,
        n_params=spec.n_params,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        d_model=spec.d_model,
        layer_results=layer_results
    )
    model_result.compute_aggregates()
    
    print(f"\n  Model Summary:")
    print(f"    Mean correspondence: {model_result.mean_correspondence_rate*100:.1f}%")
    print(f"    Mean c_ℓ: {model_result.mean_c_ell:.4e}")
    print(f"    Mean |ρ|: {model_result.mean_pearson:.4f}")
    print(f"    Total time: {model_result.total_time_seconds:.1f}s")
    
    # Clear GPU memory
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return model_result


def run_gpt2_validation(config: ValidationConfig) -> Dict:
    """Run validation across all specified GPT-2 models."""
    
    if not HAS_TRANSFORMERS:
        print("ERROR: transformers library not installed")
        return {}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("VALIDATION TRACK 3: GPT-2 PRETRAINED MODELS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Models: {config.model_names}")
    print(f"  Direction pairs: {config.n_direction_pairs}")
    print(f"  Device: {device}")
    
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    model_results = []
    campaign_start = time.time()
    
    for model_name in config.model_names:
        if model_name not in GPT2_MODELS:
            print(f"\nWARNING: Unknown model {model_name}, skipping")
            continue
        
        try:
            result = validate_gpt2_model(model_name, config, device)
            model_results.append(result)
        except Exception as e:
            print(f"\nERROR validating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    campaign_elapsed = time.time() - campaign_start
    
    # Compile results
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'total_time_seconds': campaign_elapsed,
            'config': asdict(config)
        },
        'model_results': [],
        'summary': {}
    }
    
    for mr in model_results:
        mr_dict = {
            'model_name': mr.model_name,
            'n_params': mr.n_params,
            'n_layers': mr.n_layers,
            'n_heads': mr.n_heads,
            'd_model': mr.d_model,
            'mean_correspondence_rate': mr.mean_correspondence_rate,
            'mean_c_ell': mr.mean_c_ell,
            'mean_pearson': mr.mean_pearson,
            'total_time_seconds': mr.total_time_seconds,
            'layer_results': [asdict(lr) for lr in mr.layer_results]
        }
        results['model_results'].append(mr_dict)
    
    # Overall summary
    if model_results:
        all_corr = [mr.mean_correspondence_rate for mr in model_results]
        all_c_ell = [mr.mean_c_ell for mr in model_results]
        all_pearson = [mr.mean_pearson for mr in model_results]
        
        results['summary'] = {
            'mean_correspondence_rate': float(np.mean(all_corr)),
            'min_correspondence_rate': float(np.min(all_corr)),
            'max_correspondence_rate': float(np.max(all_corr)),
            'mean_c_ell': float(np.mean(all_c_ell)),
            'mean_pearson': float(np.mean(all_pearson))
        }
    
    # Print final summary
    print("\n" + "="*70)
    print("GPT-2 VALIDATION SUMMARY")
    print("="*70)
    print(f"\n{'Model':>15} | {'Params':>8} | {'Corr Rate':>10} | {'c_ℓ':>12} | {'|ρ|':>8}")
    print("-" * 60)
    
    for mr in model_results:
        print(f"{mr.model_name:>15} | {mr.n_params:>8} | "
              f"{mr.mean_correspondence_rate*100:>9.1f}% | "
              f"{mr.mean_c_ell:>12.4e} | {mr.mean_pearson:>8.4f}")
    
    if results['summary']:
        print("-" * 60)
        print(f"{'OVERALL':>15} | {'':>8} | "
              f"{results['summary']['mean_correspondence_rate']*100:>9.1f}% | "
              f"{results['summary']['mean_c_ell']:>12.4e} | "
              f"{results['summary']['mean_pearson']:>8.4f}")
    
    print(f"\nTotal campaign time: {campaign_elapsed/60:.1f} minutes")
    
    return results


def save_results(results: Dict, output_dir: str) -> Path:
    """Save results to JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"track3_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    """Main entry point for Track 3 validation."""
    
    config = ValidationConfig()
    
    # For quick testing:
    # config.model_names = ['gpt2']
    # config.n_direction_pairs = 10
    # config.n_train_pairs = 5
    # config.n_test_pairs = 5
    
    results = run_gpt2_validation(config)
    if results:
        save_results(results, config.output_dir)
    
    return results


if __name__ == "__main__":
    main()
