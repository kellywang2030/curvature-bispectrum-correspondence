# Curvature-Bispectrum Correspondence Validation Framework

This repository contains the seven-module computational framework and three-track validation campaign used to empirically validate the theoretical correspondence between Fisher-Rao curvature and bispectral energy in multi-head attention mechanisms, as described in:

> **Curvature Meets Bispectrum: A Correspondence Theory for Transformer Gauge Invariants**  
> Hong Wang and Kelly Wang  
> NeurReps 2025 (Symmetry and Geometry in Neural Representations)

## Overview

The paper establishes that curvature on the parameter-to-function quotient provides a lower bound for permutation-bispectral energy in a linearized regime. This repository provides the implementation used to validate this theoretical result across three experimental tracks:

| Track | Description | Correspondence Rate |
|-------|-------------|---------------------|
| Track 1 | Multi-scale synthetic models (h ∈ {4,6,8,12,16,24}) | 96.3% |
| Track 2 | Training dynamics (10,000 steps) | 93-100% |
| Track 3 | Pretrained GPT-2 (124M-355M parameters) | 96.1% |

The validation demonstrates that geometric (curvature) and algebraic (bispectral) invariants capture complementary aspects of model structure, with mean correlation |ρ| ≈ 0.17.

## Repository Structure

```
.
├── README.md
├── run_validation_campaign.py      # Master orchestrator
│
├── # === Core Modules (1-7) ===
├── module1_gauge_algebra.py        # Gauge algebra structures
├── module2_fisher_rao_metric.py    # Fisher-Rao metric computation
├── module3_mechanical_connection.py # FR mechanical connection (reference)
├── module4_euclidean_curvature.py  # Euclidean connection implementation
├── module4_discrete_holonomy.py    # Holonomy-based curvature estimation
├── module4_diagnostic.py           # Richardson ratio diagnostics
├── module5_canonicalization.py     # Four-stage canonicalization
├── module6_bispectrum.py           # Full bispectrum and directional energy
├── module7_bound_verification.py   # Bound verification with train/test split
│
├── # === Validation Tracks ===
├── validation_track1_multiscale.py # Track 1: Multi-scale correspondence
├── validation_track2_training.py   # Track 2: Training dynamics
└── validation_track3_gpt2.py       # Track 3: Pretrained GPT-2
```

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (validated on H100 NVL)
- Minimum 8GB GPU memory for synthetic models
- Minimum 16GB GPU memory for GPT-2 Medium

### Software
```
python >= 3.10
torch >= 2.0
numpy
scipy
transformers  # For GPT-2 models
tqdm
```

### Installation
```bash
pip install torch numpy scipy transformers tqdm
```

## Seven-Module Framework

### Module 1: Gauge Algebra Structures
Implements the Lie algebra of the maximal gauge group G_max = ((GL(d_k))^h × (GL(d_v))^h) ⋊ S_h and the infinitesimal gauge action on MHA parameters.

**Key components:** `MHAParams`, `GaugeDirection`, `vertical_map`, `vertical_map_transpose`

### Module 2: Fisher-Rao Metric
Implements the empirical Fisher-Rao metric on parameter space via automatic differentiation.

**Key components:** `fisher_apply`, `fr_inner`, `fr_norm`, `apply_M_operator`

**Note:** Validation confirmed that the Fisher-Rao metric has the vertical subspace in its null space due to exact gauge invariance, motivating the Euclidean approach in Module 4.

### Module 3: Mechanical Connection (Fisher-Rao)
Implements the theoretical Fisher-Rao mechanical connection via normal equations.

**Note:** Included for completeness; all empirical results use Module 4 due to numerical singularity (condition numbers > 10^15).

### Module 4: Euclidean Mechanical Connection and Discrete Holonomy
Implements the canonical Euclidean connection (horizontal spaces as Euclidean orthogonal complements of vertical spaces) and discrete holonomy-based curvature estimation.

**Key components:** `EuclideanMechanicalConnection`, `DiscreteHolonomy`, `generate_horizontal_direction_pair`

**Curvature estimation:** Uses Richardson extrapolation with two step sizes to achieve accurate curvature estimates without explicit metric tensor construction.

### Module 5: Canonicalization
Removes continuous gauge freedom through deterministic normalization, reducing symmetry from G_max to discrete S_h.

**Four-stage algorithm:**
1. Query-Key balancing (Gram matrix whitening)
2. Value orthonormalization (QR with column pivoting)
3. Head sorting with stable tie-breaking
4. Permutation tracking

### Module 6: Bispectrum Computation
Computes the full S_h bispectrum and directional bispectral energy after canonicalization.

**Key components:** `extract_head_features`, `whiten_features`, `compute_triple_correlation`, `compute_bispectrum`, `compute_directional_bispectral_energy`

**Complexity:** O((h!)^2) for full bispectrum, limiting practical application to h ≤ 8.

### Module 7: Bound Verification
Integrates all modules to verify the theoretical bound ||Ω(u,v)||²_F ≥ c_ℓ E(u,v) with proper train/test methodology.

**Protocol:**
1. Generate n horizontal direction pairs
2. Split into n/2 training and n/2 test pairs
3. Estimate ĉ_ℓ from training pairs
4. Evaluate correspondence on test pairs
5. Compute diagnostics (correlations, Richardson ratios, bootstrap CIs)

## Running the Validation Campaign

### Full Campaign (All Three Tracks)
```bash
python run_validation_campaign.py --track all
```

**Expected runtime:** ~18.2 hours on NVIDIA H100 NVL
- Track 1: ~13.2 hours
- Track 2: ~2.7 hours  
- Track 3: ~2.3 hours

### Quick Validation (Reduced Parameters)
```bash
python run_validation_campaign.py --track all --quick
```

### Individual Tracks

**Track 1: Multi-Scale Correspondence**
```bash
python run_validation_campaign.py --track 1
```
Tests heads h ∈ {4,6,8,12,16,24} with 5 seeds × 30 direction pairs each.

**Track 2: Training Dynamics**
```bash
python run_validation_campaign.py --track 2
```
Trains 12-head model for 10,000 steps, validates at 7 checkpoints.

**Track 3: Pretrained GPT-2**
```bash
python run_validation_campaign.py --track 3
```
Validates on GPT-2 (124M) and GPT-2 Medium (355M) across early/middle/late layers.

### Running Individual Track Scripts Directly
```bash
# Track 1
python validation_track1_multiscale.py

# Track 2
python validation_track2_training.py

# Track 3
python validation_track3_gpt2.py
```

## Output

Results are saved to JSON files in the `results/` directory with timestamps:

```
results/
├── track1_multiscale_YYYYMMDD_HHMMSS.json
├── track2_training_YYYYMMDD_HHMMSS.json
├── track3_gpt2_YYYYMMDD_HHMMSS.json
└── full_campaign_YYYYMMDD_HHMMSS.log
```

### Output Format

Each JSON file contains:
- Per-configuration correspondence rates
- Train/test split statistics
- Estimated bound constants (ĉ_ℓ)
- Pearson and Spearman correlations
- Richardson ratios for linearization verification
- Timing information

## Key Results

### Track 1: Multi-Scale Validation

| Heads | Corr. Rate | Test Rate | ĉ_ℓ (mean) | \|ρ\| | Richardson |
|-------|------------|-----------|------------|-------|------------|
| 4 | 96.0% | 92.0% | 1.03×10⁻³ | 0.189 | 1.0000 |
| 6 | 95.3% | 90.7% | 8.40×10⁻⁴ | 0.150 | 1.0000 |
| 8 | 99.3% | 98.7% | 1.61×10⁻⁴ | 0.099 | 1.0000 |
| 12 | 91.3% | 82.7% | 1.39×10⁻³ | 0.171 | 1.0001 |
| 16 | 96.7% | 93.3% | 1.81×10⁻⁴ | 0.200 | 1.0000 |
| 24 | 99.3% | 98.7% | 1.06×10⁻⁵ | 0.217 | 1.0000 |
| **Mean** | **96.3%** | **92.7%** | 6.03×10⁻⁴ | **0.171** | 1.0000 |

### Track 2: Training Dynamics

Correspondence maintained at 93-100% throughout 10,000 training steps, with temporary dips during rapid loss reduction periods.

### Track 3: GPT-2 Validation

| Model | Layer | Corr. Rate | ĉ_ℓ | \|ρ\| |
|-------|-------|------------|-----|-------|
| GPT-2 124M | Early | 93.3% | 8.28×10⁻⁴ | 0.043 |
| GPT-2 124M | Middle | 100.0% | 9.04×10⁻⁶ | 0.074 |
| GPT-2 124M | Late | 96.7% | 1.40×10⁻⁶ | 0.294 |
| GPT-2 Medium | Early | 93.3% | 5.74×10⁻⁴ | 0.316 |
| GPT-2 Medium | Middle | 96.7% | 4.91×10⁻⁵ | 0.017 |
| GPT-2 Medium | Late | 96.7% | 1.56×10⁻¹⁵ | 0.202 |
| **Overall** | | **96.1%** | 2.44×10⁻⁴ | **0.158** |

## Numerical Considerations

### Fisher-Rao Degeneracy
The theoretical correspondence (Theorem 4 in the paper) is formulated in terms of Fisher-Rao curvature. However, exact gauge invariance renders the FR metric singular along vertical directions. The implementation uses Euclidean gauge-orthogonal curvature, which preserves the inequality form with empirically estimated ĉ_ℓ.

### Richardson Ratios
Values of exactly 1.0000 across all configurations confirm operation in the linearized regime predicted by theory.

### Precision
All computations use float64 precision with TF32 and cuDNN acceleration disabled for strict numerical compliance.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{wang2025curvature,
  title     = {Curvature Meets Bispectrum: A Correspondence Theory for Transformer Gauge Invariants},
  author    = {Wang, Hong and Wang, Kelly},
  booktitle = {Symmetry and Geometry in Neural Representations (NeurReps 2025), Proceedings Track},
  year      = {2025}
}
```

### Related Papers

```bibtex
@inproceedings{wang2025completegauge,
  title     = {Complete Characterization of Gauge Symmetries in Transformer Architectures},
  author    = {Wang, Hong and Wang, Kelly},
  booktitle = {Symmetry and Geometry in Neural Representations (NeurReps 2025), Proceedings Track},
  year      = {2025}
}

@inproceedings{wang2025fiberbundle,
  title     = {Gauge Fiber Bundle Geometry of Transformers},
  author    = {Wang, Hong and Wang, Kelly},
  booktitle = {Symmetry and Geometry in Neural Representations (NeurReps 2025), Proceedings Track},
  year      = {2025}
}
```

## License

Apache License 2.0

See [LICENSE](LICENSE) for the full license text.

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
