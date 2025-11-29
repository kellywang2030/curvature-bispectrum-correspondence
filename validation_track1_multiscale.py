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

# --- validation_track1_multiscale.py ---
"""Track 1: Multi-Scale Correspondence Validation

Validates the curvature-bispectrum bound across varying model scales
(h in {4, 6, 8, 12, 16, 24}).
"""


#!/usr/bin/env python3
"""
Validation Track 1: Multi-Scale Correspondence Validation

This script reproduces Table 1 from the paper by running the bound verification
protocol across head counts h ∈ {4, 6, 8, 12, 16, 24} with d_model = 64h.

Target metrics to reproduce:
- Correspondence rates (paper claims 98.9% mean, range 96.7% to 100%)
- c_ℓ estimates (paper: 2.1e-3 for h=4 down to 0.9e-3 for h=24)
- Pearson |ρ| < 0.35 between curvature and energy
- Richardson ratios near 0.97 confirming numerical stability

Experimental protocol:
- 30 direction pairs per configuration (15 train, 15 test)
- 5 random seeds per configuration for stability analysis
- Bootstrap confidence intervals with 1000 resamples
- All computations in float64 for numerical precision
"""

import torch
import numpy as np
import json
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

# Import our modules
from module1_gauge_algebra import ReferenceConfig, MHAParams, HeadParams, random_mha_params
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
class ScaleConfig:
    """Configuration for a specific model scale."""
    n_heads: int
    d_k: int = 64
    d_v: int = 64
    
    @property
    def d_model(self) -> int:
        return self.n_heads * self.d_v
    
    def to_reference_config(self) -> ReferenceConfig:
        """Convert to ReferenceConfig for module compatibility."""
        return ReferenceConfig(
            n_heads=self.n_heads,
            d_k=self.d_k,
            d_v=self.d_v,
            d_model=self.d_model
        )


@dataclass
class SeedResult:
    """Results for a single seed at a given scale."""
    seed: int
    c_ell_estimate: float
    train_correspondence_rate: float
    test_correspondence_rate: float
    overall_correspondence_rate: float
    pearson_correlation: float
    spearman_correlation: float
    curvature_mean: float
    curvature_std: float
    energy_mean: float
    energy_std: float
    mean_richardson_ratio: float
    computation_time_seconds: float


@dataclass
class ScaleResult:
    """Aggregated results for a model scale across all seeds."""
    n_heads: int
    d_model: int
    
    # Per-seed results
    seed_results: List[SeedResult]
    
    # Aggregated statistics
    c_ell_mean: float = 0.0
    c_ell_std: float = 0.0
    correspondence_mean: float = 0.0
    correspondence_std: float = 0.0
    test_correspondence_mean: float = 0.0
    test_correspondence_std: float = 0.0
    pearson_mean: float = 0.0
    pearson_std: float = 0.0
    richardson_mean: float = 0.0
    
    # Confidence intervals (from bootstrap)
    correspondence_ci_lower: float = 0.0
    correspondence_ci_upper: float = 0.0
    
    # Timing
    total_time_seconds: float = 0.0
    
    def compute_aggregates(self):
        """Compute aggregate statistics from seed results."""
        if not self.seed_results:
            return
        
        c_ells = [r.c_ell_estimate for r in self.seed_results]
        corr_rates = [r.overall_correspondence_rate for r in self.seed_results]
        test_rates = [r.test_correspondence_rate for r in self.seed_results]
        pearsons = [abs(r.pearson_correlation) for r in self.seed_results]
        richardsons = [r.mean_richardson_ratio for r in self.seed_results]
        times = [r.computation_time_seconds for r in self.seed_results]
        
        self.c_ell_mean = float(np.mean(c_ells))
        self.c_ell_std = float(np.std(c_ells))
        self.correspondence_mean = float(np.mean(corr_rates))
        self.correspondence_std = float(np.std(corr_rates))
        self.test_correspondence_mean = float(np.mean(test_rates))
        self.test_correspondence_std = float(np.std(test_rates))
        self.pearson_mean = float(np.mean(pearsons))
        self.pearson_std = float(np.std(pearsons))
        self.richardson_mean = float(np.mean(richardsons))
        self.total_time_seconds = float(sum(times))
        
        # Bootstrap CI for correspondence rate
        all_corr = []
        for r in self.seed_results:
            all_corr.append(r.overall_correspondence_rate)
        
        if len(all_corr) >= 2:
            bootstrap_means = []
            for _ in range(1000):
                sample = np.random.choice(all_corr, size=len(all_corr), replace=True)
                bootstrap_means.append(np.mean(sample))
            self.correspondence_ci_lower = float(np.percentile(bootstrap_means, 2.5))
            self.correspondence_ci_upper = float(np.percentile(bootstrap_means, 97.5))
        else:
            self.correspondence_ci_lower = self.correspondence_mean
            self.correspondence_ci_upper = self.correspondence_mean


@dataclass 
class CampaignConfig:
    """Configuration for the full validation campaign."""
    # Model scales to test (paper uses h ∈ {4, 6, 8, 12, 16, 24})
    head_counts: List[int] = field(default_factory=lambda: [4, 6, 8, 12, 16, 24])
    
    # Fixed dimensions (paper uses d_k = d_v = 64)
    d_k: int = 64
    d_v: int = 64
    
    # Direction pairs (paper uses 30 total, 15 train / 15 test)
    n_direction_pairs: int = 30
    n_train_pairs: int = 15
    n_test_pairs: int = 15
    
    # Seeds for stability analysis (paper uses 5 seeds)
    n_seeds: int = 5
    base_seed: int = 42
    
    # Epsilon values for Richardson extrapolation
    epsilon_values: List[float] = field(default_factory=lambda: [1e-3, 2e-3, 4e-3])
    
    # Evaluation batch configuration
    batch_size: int = 64
    seq_length: int = 32
    
    # Numerical precision
    dtype: str = "float64"
    
    # Output directory
    output_dir: str = "results/track1_multiscale"


def create_evaluation_batch(config: ReferenceConfig, batch_size: int, seq_length: int,
                            device: torch.device, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Create a fixed evaluation batch for consistent measurements."""
    torch.manual_seed(seed)
    return torch.randn(batch_size, seq_length, config.d_model, device=device, dtype=dtype)


def run_single_scale_seed(
    scale_config: ScaleConfig,
    campaign_config: CampaignConfig,
    seed: int,
    device: torch.device,
    dtype: torch.dtype
) -> SeedResult:
    """Run bound verification for a single scale and seed."""
    
    start_time = time.time()
    
    # Create reference config
    ref_config = scale_config.to_reference_config()
    
    # Create random parameters
    torch.manual_seed(seed)
    theta = random_mha_params(ref_config, device, dtype)
    
    # Create evaluation batch
    X = create_evaluation_batch(
        ref_config, 
        campaign_config.batch_size,
        campaign_config.seq_length,
        device, dtype, seed + 10000
    )
    
    # Setup bound verification config
    bound_config = BoundVerificationConfig(
        n_direction_pairs=campaign_config.n_direction_pairs,
        n_train_pairs=campaign_config.n_train_pairs,
        n_test_pairs=campaign_config.n_test_pairs,
        epsilon_values=campaign_config.epsilon_values
    )
    bispec_config = BispectrumConfig()
    
    # Run verification
    result = verify_bound(theta, X, ref_config, bound_config, bispec_config)
    
    elapsed = time.time() - start_time
    
    return SeedResult(
        seed=seed,
        c_ell_estimate=result.c_ell_estimate,
        train_correspondence_rate=result.train_correspondence_rate,
        test_correspondence_rate=result.test_correspondence_rate,
        overall_correspondence_rate=result.overall_correspondence_rate,
        pearson_correlation=result.pearson_correlation,
        spearman_correlation=result.spearman_correlation,
        curvature_mean=result.curvature_mean,
        curvature_std=result.curvature_std,
        energy_mean=result.energy_mean,
        energy_std=result.energy_std,
        mean_richardson_ratio=result.mean_richardson_ratio,
        computation_time_seconds=elapsed
    )


def run_scale_validation(
    scale_config: ScaleConfig,
    campaign_config: CampaignConfig,
    device: torch.device,
    dtype: torch.dtype
) -> ScaleResult:
    """Run full validation for a single model scale across all seeds."""
    
    print(f"\n{'='*70}")
    print(f"SCALE: h={scale_config.n_heads}, d_model={scale_config.d_model}")
    print(f"{'='*70}")
    
    seed_results = []
    
    for seed_idx in range(campaign_config.n_seeds):
        seed = campaign_config.base_seed + seed_idx * 100
        print(f"\n  Seed {seed_idx + 1}/{campaign_config.n_seeds} (seed={seed})...")
        
        try:
            result = run_single_scale_seed(
                scale_config, campaign_config, seed, device, dtype
            )
            seed_results.append(result)
            
            print(f"    c_ℓ = {result.c_ell_estimate:.4e}")
            print(f"    Correspondence: {result.overall_correspondence_rate*100:.1f}% "
                  f"(train: {result.train_correspondence_rate*100:.1f}%, "
                  f"test: {result.test_correspondence_rate*100:.1f}%)")
            print(f"    |ρ| = {abs(result.pearson_correlation):.4f}")
            print(f"    Time: {result.computation_time_seconds:.1f}s")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Create scale result and compute aggregates
    scale_result = ScaleResult(
        n_heads=scale_config.n_heads,
        d_model=scale_config.d_model,
        seed_results=seed_results
    )
    scale_result.compute_aggregates()
    
    print(f"\n  AGGREGATE RESULTS:")
    print(f"    c_ℓ = {scale_result.c_ell_mean:.4e} ± {scale_result.c_ell_std:.4e}")
    print(f"    Correspondence = {scale_result.correspondence_mean*100:.1f}% ± {scale_result.correspondence_std*100:.1f}%")
    print(f"    95% CI: [{scale_result.correspondence_ci_lower*100:.1f}%, {scale_result.correspondence_ci_upper*100:.1f}%]")
    print(f"    |ρ| = {scale_result.pearson_mean:.4f} ± {scale_result.pearson_std:.4f}")
    print(f"    Richardson ratio = {scale_result.richardson_mean:.4f}")
    print(f"    Total time: {scale_result.total_time_seconds:.1f}s")
    
    return scale_result


def run_full_campaign(campaign_config: CampaignConfig) -> Dict:
    """Run the complete multi-scale validation campaign."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64 if campaign_config.dtype == "float64" else torch.float32
    
    print("="*70)
    print("VALIDATION TRACK 1: MULTI-SCALE CORRESPONDENCE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Head counts: {campaign_config.head_counts}")
    print(f"  d_k = d_v = {campaign_config.d_k}")
    print(f"  Direction pairs: {campaign_config.n_direction_pairs} "
          f"({campaign_config.n_train_pairs} train / {campaign_config.n_test_pairs} test)")
    print(f"  Seeds per scale: {campaign_config.n_seeds}")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    campaign_start = time.time()
    scale_results = []
    
    for n_heads in campaign_config.head_counts:
        scale_config = ScaleConfig(
            n_heads=n_heads,
            d_k=campaign_config.d_k,
            d_v=campaign_config.d_v
        )
        
        result = run_scale_validation(scale_config, campaign_config, device, dtype)
        scale_results.append(result)
    
    campaign_elapsed = time.time() - campaign_start
    
    # Compile final results
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'dtype': campaign_config.dtype,
            'total_time_seconds': campaign_elapsed,
            'config': asdict(campaign_config)
        },
        'scale_results': [],
        'summary': {}
    }
    
    # Convert scale results to serializable format
    for sr in scale_results:
        sr_dict = {
            'n_heads': sr.n_heads,
            'd_model': sr.d_model,
            'c_ell_mean': sr.c_ell_mean,
            'c_ell_std': sr.c_ell_std,
            'correspondence_mean': sr.correspondence_mean,
            'correspondence_std': sr.correspondence_std,
            'test_correspondence_mean': sr.test_correspondence_mean,
            'test_correspondence_std': sr.test_correspondence_std,
            'pearson_mean': sr.pearson_mean,
            'pearson_std': sr.pearson_std,
            'richardson_mean': sr.richardson_mean,
            'correspondence_ci_lower': sr.correspondence_ci_lower,
            'correspondence_ci_upper': sr.correspondence_ci_upper,
            'total_time_seconds': sr.total_time_seconds,
            'seed_results': [asdict(seed_r) for seed_r in sr.seed_results]
        }
        results['scale_results'].append(sr_dict)
    
    # Compute overall summary
    all_corr = [sr.correspondence_mean for sr in scale_results]
    all_c_ell = [sr.c_ell_mean for sr in scale_results]
    all_pearson = [sr.pearson_mean for sr in scale_results]
    
    results['summary'] = {
        'mean_correspondence_rate': float(np.mean(all_corr)),
        'min_correspondence_rate': float(np.min(all_corr)),
        'max_correspondence_rate': float(np.max(all_corr)),
        'mean_c_ell': float(np.mean(all_c_ell)),
        'mean_pearson': float(np.mean(all_pearson)),
        'total_campaign_time_seconds': campaign_elapsed
    }
    
    # Print final summary
    print("\n" + "="*70)
    print("CAMPAIGN SUMMARY")
    print("="*70)
    print("\nResults by Scale (compare with paper Table 1):")
    print(f"{'Heads':>6} | {'c_ℓ':>12} | {'Corr Rate':>12} | {'Test Rate':>12} | {'|ρ|':>8} | {'Richardson':>10}")
    print("-" * 70)
    
    # Paper's reference values
    paper_c_ell = {4: 2.1e-3, 6: 1.8e-3, 8: 1.6e-3, 12: 1.2e-3, 16: 1.1e-3, 24: 0.9e-3}
    
    for sr in scale_results:
        paper_val = paper_c_ell.get(sr.n_heads, None)
        paper_str = f"(paper: {paper_val:.1e})" if paper_val else ""
        print(f"{sr.n_heads:>6} | {sr.c_ell_mean:>12.4e} | {sr.correspondence_mean*100:>11.1f}% | "
              f"{sr.test_correspondence_mean*100:>11.1f}% | {sr.pearson_mean:>8.4f} | {sr.richardson_mean:>10.4f}")
    
    print("-" * 70)
    print(f"{'MEAN':>6} | {results['summary']['mean_c_ell']:>12.4e} | "
          f"{results['summary']['mean_correspondence_rate']*100:>11.1f}% | "
          f"{'':>12} | {results['summary']['mean_pearson']:>8.4f} |")
    
    print(f"\nPaper claims: 98.9% mean correspondence, |ρ| < 0.35")
    print(f"Our results:  {results['summary']['mean_correspondence_rate']*100:.1f}% mean correspondence, "
          f"|ρ| = {results['summary']['mean_pearson']:.3f}")
    
    print(f"\nTotal campaign time: {campaign_elapsed/60:.1f} minutes")
    
    return results


def save_results(results: Dict, output_dir: str):
    """Save results to JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"track1_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def main():
    """Main entry point for Track 1 validation."""
    
    # Default configuration matching paper
    config = CampaignConfig()
    
    # For quick testing, can reduce parameters:
    # config.head_counts = [4, 8]
    # config.n_seeds = 2
    # config.n_direction_pairs = 10
    # config.n_train_pairs = 5
    # config.n_test_pairs = 5
    
    results = run_full_campaign(config)
    save_results(results, config.output_dir)
    
    return results


if __name__ == "__main__":
    main()
