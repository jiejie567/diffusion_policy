"""
Test script for dual arm depth dataset
"""

import sys
import os
import glob
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import torch
import h5py

from diffusion_policy.dataset.dual_arm_depth_dataset import DualArmDepthDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Quick check for DualArmDepthDataset")
    parser.add_argument("--dataset_path", type=str, default="data/dual_arm_demos",
                        help="Path or glob to HDF5 files.")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Optional: limit number of HDF5 files to load (e.g., 1 for a quick check).")
    return parser.parse_args()


def test_dataset(args):
    print("=" * 80)
    print("Testing DualArmDepthDataset")
    print("=" * 80)

    # Create dataset
    dataset_path = args.dataset_path
    max_files = args.max_files

    print(f"\nLoading dataset from: {dataset_path}")

    dataset = DualArmDepthDataset(
        dataset_path=dataset_path,
        horizon=16,
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.02,
        depth_as_3channel=True,  # Replicate to 3 channels for ResNet compatibility
        max_files=max_files
    )

    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Number of episodes: {dataset.replay_buffer.n_episodes}")
    print(f"  Total steps: {dataset.replay_buffer.n_steps}")
    print(f"  Dataset length (sequences): {len(dataset)}")

    # Get validation dataset
    val_dataset = dataset.get_validation_dataset()
    print(f"\n✓ Validation split:")
    print(f"  Train sequences: {len(dataset)}")
    print(f"  Val sequences: {len(val_dataset)}")

    # Test a sample
    print(f"\n{'=' * 80}")
    print("Testing sample extraction")
    print("=" * 80)

    sample = dataset[0]

    print(f"\n✓ Sample structure:")
    print(f"  Keys: {list(sample.keys())}")

    print(f"\n✓ Observation shapes:")
    for key, value in sample['obs'].items():
        # convert shape tuple to string before formatting to avoid TypeError
        print(f"  {key:20s}: {tuple(value.shape)} dtype={value.dtype}")

    print(f"\n✓ Action shape:")
    print(f"  action: {tuple(sample['action'].shape)} dtype={sample['action'].dtype}")

    # Check data ranges
    print(f"\n{'=' * 80}")
    print("Data statistics")
    print("=" * 80)

    for key, value in sample['obs'].items():
        print(f"\n  {key}:")
        print(f"    min: {value.min().item():.4f}")
        print(f"    max: {value.max().item():.4f}")
        print(f"    mean: {value.mean().item():.4f}")
        print(f"    std: {value.std().item():.4f}")

        # Check for any remaining invalid values
        has_nan = torch.isnan(value).any().item()
        has_inf = torch.isinf(value).any().item()
        if has_nan or has_inf:
            print(f"    ⚠️  WARNING: Contains NaN={has_nan}, Inf={has_inf}")
        else:
            print(f"    ✓ No invalid values")

    print(f"\n  action:")
    print(f"    min: {sample['action'].min().item():.4f}")
    print(f"    max: {sample['action'].max().item():.4f}")
    print(f"    mean: {sample['action'].mean().item():.4f}")
    print(f"    std: {sample['action'].std().item():.4f}")

    # Check raw data for inf/nan before processing
    print(f"\n{'=' * 80}")
    print("Checking raw HDF5 data for invalid values")
    print("=" * 80)

    # Resolve one sample file for raw check
    if os.path.isdir(dataset_path):
        sample_files = sorted(glob.glob(os.path.join(dataset_path, "*.hdf5")))
    else:
        sample_files = sorted(glob.glob(dataset_path))

    if len(sample_files) == 0:
        print("No HDF5 files found for raw check; skipping this section.")
    else:
        sample_file = sample_files[0]
        print(f"\nChecking first file: {sample_file}")

        with h5py.File(sample_file, 'r') as f:
            for depth_key in ['left_hand_depth', 'right_hand_depth', 'head_depth']:
                raw_depth = f['observations'][depth_key][:]
                n_inf = np.isinf(raw_depth).sum()
                n_nan = np.isnan(raw_depth).sum()
                n_total = raw_depth.size

                print(f"\n  {depth_key}:")
                print(f"    Total pixels: {n_total}")
                print(f"    Inf values: {n_inf} ({100*n_inf/n_total:.2f}%)")
                print(f"    NaN values: {n_nan} ({100*n_nan/n_total:.2f}%)")
                print(f"    Valid values: {n_total - n_inf - n_nan} ({100*(n_total-n_inf-n_nan)/n_total:.2f}%)")

    # Test normalizer
    print(f"\n{'=' * 80}")
    print("Testing normalizer")
    print("=" * 80)

    normalizer = dataset.get_normalizer()
    print(f"\n✓ Normalizer keys: {list(normalizer.params_dict.keys())}")

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)

if __name__ == "__main__":
    args = parse_args()
    test_dataset(args)
