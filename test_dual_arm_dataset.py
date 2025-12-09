"""
Test script for dual arm depth dataset
"""

import sys
import os
import glob

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import torch
import h5py

from diffusion_policy.dataset.dual_arm_depth_dataset import DualArmDepthDataset

def test_dataset():
    print("=" * 80)
    print("Testing DualArmDepthDataset")
    print("=" * 80)

    # Create dataset
    dataset_path = 'data/dual_arm_demos'  # Update this path

    print(f"\nLoading dataset from: {dataset_path}")

    dataset = DualArmDepthDataset(
        dataset_path=dataset_path,
        horizon=16,
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.02,
        depth_as_3channel=True  # Replicate to 3 channels for ResNet compatibility
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
        print(f"  {key:20s}: {tuple(value.shape):20s} dtype={value.dtype}")

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

    sample_file = sorted(glob.glob(f"{dataset_path}/*.hdf5"))[0]
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
    print(f"\n✓ Normalizer keys: {list(normalizer.keys())}")

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)

if __name__ == "__main__":
    test_dataset()
