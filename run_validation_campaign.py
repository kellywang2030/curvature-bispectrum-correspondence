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

# --- run_validation_campaign.py ---
"""Validation Campaign Orchestrator

Master script to run the complete three-track validation campaign or
individual tracks.
"""

#!/usr/bin/env python3
"""
Master Validation Campaign Orchestrator

This script coordinates the complete empirical validation campaign for the
curvature-bispectrum correspondence paper. It runs three validation tracks:

Track 1: Multi-Scale Correspondence
    - Tests h ∈ {4, 6, 8, 12, 16, 24} with d_model = 64h
    - 30 direction pairs × 5 seeds per configuration
    - Reproduces Table 1 from the paper

Track 2: Training Dynamics
    - 12-head model trained for 10,000 steps
    - Checkpoints at steps 0, 100, 500, 1000, 2500, 5000, 10000
    - Reproduces Figure 1 from the paper

Track 3: GPT-2 Pretrained Models
    - Validates on gpt2, gpt2-medium, gpt2-large, gpt2-xl
    - Early, middle, late layers per model
    - Bridges synthetic to production-scale validation

Usage:
    python run_validation_campaign.py                    # Full campaign
    python run_validation_campaign.py --track 1         # Track 1 only
    python run_validation_campaign.py --track 2         # Track 2 only
    python run_validation_campaign.py --track 3         # Track 3 only
    python run_validation_campaign.py --quick           # Quick test mode

Estimated runtime (on H100 GPU):
    Track 1: ~2-3 hours
    Track 2: ~1-2 hours  
    Track 3: ~1-2 hours
    Full campaign: ~5-7 hours
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys


def run_track1(quick_mode: bool = False) -> Dict:
    """Run Track 1: Multi-Scale Correspondence Validation."""
    from validation_track1_multiscale import CampaignConfig, run_full_campaign, save_results
    
    config = CampaignConfig()
    
    if quick_mode:
        config.head_counts = [4, 8]
        config.n_seeds = 2
        config.n_direction_pairs = 10
        config.n_train_pairs = 5
        config.n_test_pairs = 5
    
    results = run_full_campaign(config)
    save_results(results, config.output_dir)
    
    return results


def run_track2(quick_mode: bool = False) -> Dict:
    """Run Track 2: Training Dynamics Validation."""
    from validation_track2_training import TrainingConfig, run_training_experiment, save_results
    
    config = TrainingConfig()
    
    if quick_mode:
        config.total_steps = 1000
        config.checkpoint_steps = [0, 100, 500, 1000]
        config.n_direction_pairs = 10
        config.n_train_pairs = 5
        config.n_test_pairs = 5
    
    result = run_training_experiment(config)
    save_results(result, config.output_dir)
    
    return {
        'config': result.config,
        'n_checkpoints': len(result.checkpoint_results),
        'final_correspondence': result.checkpoint_results[-1].overall_correspondence_rate if result.checkpoint_results else None,
        'total_time': result.total_training_time_seconds
    }


def run_track3(quick_mode: bool = False) -> Dict:
    """Run Track 3: GPT-2 Pretrained Model Validation."""
    from validation_track3_gpt2 import ValidationConfig, run_gpt2_validation, save_results
    
    config = ValidationConfig()
    
    if quick_mode:
        config.model_names = ['gpt2']
        config.n_direction_pairs = 10
        config.n_train_pairs = 5
        config.n_test_pairs = 5
    
    results = run_gpt2_validation(config)
    if results:
        save_results(results, config.output_dir)
    
    return results


def run_full_campaign(quick_mode: bool = False) -> Dict:
    """Run all three validation tracks."""
    
    print("="*70)
    print("CURVATURE-BISPECTRUM CORRESPONDENCE VALIDATION CAMPAIGN")
    print("="*70)
    print(f"\nStart time: {datetime.now().isoformat()}")
    print(f"Mode: {'QUICK TEST' if quick_mode else 'FULL VALIDATION'}")
    
    campaign_start = time.time()
    results = {
        'metadata': {
            'start_time': datetime.now().isoformat(),
            'quick_mode': quick_mode
        },
        'tracks': {}
    }
    
    # Track 1
    print("\n" + "="*70)
    print("TRACK 1: MULTI-SCALE CORRESPONDENCE")
    print("="*70)
    try:
        track1_start = time.time()
        results['tracks']['track1'] = run_track1(quick_mode)
        results['tracks']['track1']['elapsed_seconds'] = time.time() - track1_start
        print(f"\nTrack 1 completed in {results['tracks']['track1']['elapsed_seconds']/60:.1f} minutes")
    except Exception as e:
        print(f"\nTrack 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['tracks']['track1'] = {'error': str(e)}
    
    # Track 2
    print("\n" + "="*70)
    print("TRACK 2: TRAINING DYNAMICS")
    print("="*70)
    try:
        track2_start = time.time()
        results['tracks']['track2'] = run_track2(quick_mode)
        results['tracks']['track2']['elapsed_seconds'] = time.time() - track2_start
        print(f"\nTrack 2 completed in {results['tracks']['track2']['elapsed_seconds']/60:.1f} minutes")
    except Exception as e:
        print(f"\nTrack 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['tracks']['track2'] = {'error': str(e)}
    
    # Track 3
    print("\n" + "="*70)
    print("TRACK 3: GPT-2 PRETRAINED MODELS")
    print("="*70)
    try:
        track3_start = time.time()
        results['tracks']['track3'] = run_track3(quick_mode)
        results['tracks']['track3']['elapsed_seconds'] = time.time() - track3_start
        print(f"\nTrack 3 completed in {results['tracks']['track3']['elapsed_seconds']/60:.1f} minutes")
    except Exception as e:
        print(f"\nTrack 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['tracks']['track3'] = {'error': str(e)}
    
    campaign_elapsed = time.time() - campaign_start
    results['metadata']['end_time'] = datetime.now().isoformat()
    results['metadata']['total_elapsed_seconds'] = campaign_elapsed
    
    # Final summary
    print("\n" + "="*70)
    print("CAMPAIGN COMPLETE")
    print("="*70)
    
    print("\nTrack Summary:")
    for track_name, track_data in results['tracks'].items():
        if 'error' in track_data:
            print(f"  {track_name}: FAILED - {track_data['error']}")
        else:
            elapsed = track_data.get('elapsed_seconds', 0)
            print(f"  {track_name}: Completed in {elapsed/60:.1f} minutes")
    
    print(f"\nTotal campaign time: {campaign_elapsed/60:.1f} minutes ({campaign_elapsed/3600:.1f} hours)")
    
    # Save master results
    output_dir = Path("results/campaign")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"campaign_results_{timestamp}.json"
    
    # Convert non-serializable items
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            try:
                json.dumps(obj)
                return obj
            except:
                return str(obj)
    
    with open(output_file, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    
    print(f"\nMaster results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run curvature-bispectrum correspondence validation campaign',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_validation_campaign.py                    # Full campaign
    python run_validation_campaign.py --track 1         # Track 1 only
    python run_validation_campaign.py --quick           # Quick test mode
    python run_validation_campaign.py --track 1 --quick # Quick Track 1
        """
    )
    
    parser.add_argument(
        '--track',
        type=int,
        choices=[1, 2, 3],
        help='Run specific track only (1=multi-scale, 2=training, 3=GPT-2)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run in quick test mode with reduced parameters'
    )
    
    args = parser.parse_args()
    
    if args.track == 1:
        run_track1(args.quick)
    elif args.track == 2:
        run_track2(args.quick)
    elif args.track == 3:
        run_track3(args.quick)
    else:
        run_full_campaign(args.quick)


if __name__ == "__main__":
    main()
