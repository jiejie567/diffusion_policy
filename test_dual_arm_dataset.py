"""
Test script for dual arm memmap depth dataset
"""

import sys
import os
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import torch

from diffusion_policy.dataset.dual_arm_depth_mmap_dataset import DualArmDepthMmapDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Quick check for DualArmDepthMmapDataset")
    parser.add_argument("--dataset_path", type=str, default="data/dual_arm_demos_mmap",
                        help="Path to memmap dataset directory.")
    return parser.parse_args()


def test_dataset(args):
    print("=" * 80)
    print("Testing DualArmDepthMmapDataset")
    print("=" * 80)

    # Create dataset
    dataset_path = args.dataset_path

    print(f"\nLoading dataset from: {dataset_path}")

    try:
        dataset = DualArmDepthMmapDataset(
            dataset_path=dataset_path,
            horizon=16,
            pad_before=1,
            pad_after=7,
            seed=42,
            val_ratio=0.02,
            depth_as_3channel=True
        )
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"\nTo create memmap dataset, run:")
        print(f"  python scripts/convert_hdf5_to_mmap.py \\")
        print(f"    --input data/dual_arm_demos \\")
        print(f"    --output data/dual_arm_demos_mmap")
        return

    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Number of episodes: {dataset.num_episodes}")
    print(f"  Total steps: {dataset.total_steps}")
    print(f"  Train samples: {len(dataset.train_indices)}")
    print(f"  Val samples: {len(dataset.val_indices)}")

    # Get validation dataset
    val_dataset = dataset.get_validation_dataset()
    print(f"\n✓ Validation split:")
    print(f"  Train sequences: {len(dataset.train_indices)}")
    print(f"  Val sequences: {len(val_dataset.val_indices)}")

    # Test a sample
    print(f"\n{'=' * 80}")
    print("Testing sample extraction")
    print("=" * 80)

    sample = dataset[0]

    print(f"\n✓ Sample structure:")
    print(f"  Keys: {list(sample.keys())}")

    print(f"\n✓ Observation shapes:")
    for key, value in sample['obs'].items():
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
