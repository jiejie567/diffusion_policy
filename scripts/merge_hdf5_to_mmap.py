#!/usr/bin/env python3
"""
Merge multiple HDF5 directories into a single memmap dataset.
"""

import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json


def merge_hdf5_to_mmap(input_dirs: list, output_dir: str):
    """Merge multiple HDF5 directories into memmap format."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all HDF5 files from all input directories
    print("Collecting HDF5 files...")
    all_hdf5_files = []
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        hdf5_files = sorted(input_path.glob("*.hdf5"))
        print(f"  {input_dir}: {len(hdf5_files)} files")
        all_hdf5_files.extend(hdf5_files)
    
    print(f"Total: {len(all_hdf5_files)} HDF5 files")
    
    # First pass: count total steps and get shapes
    print("\nPass 1: Counting total steps...")
    total_steps = 0
    episode_ends = []
    sample_shapes = {}
    
    for hdf5_file in tqdm(all_hdf5_files, desc="Scanning"):
        try:
            with h5py.File(hdf5_file, 'r') as f:
                obs = f['observations']
                
                # Get shapes from first file
                if not sample_shapes:
                    sample_shapes = {
                        'left_hand_depth': obs['left_hand_depth'].shape[1:],
                        'right_hand_depth': obs['right_hand_depth'].shape[1:],
                        'head_depth': obs['head_depth'].shape[1:],
                        'left_arm_joint_pos': obs['left_arm_joint_pos'].shape[1:],
                        'left_gripper_joint_pos': obs['left_gripper_joint_pos'].shape[1:] if len(obs['left_gripper_joint_pos'].shape) > 1 else (),
                        'right_arm_joint_pos': obs['right_arm_joint_pos'].shape[1:],
                        'right_gripper_joint_pos': obs['right_gripper_joint_pos'].shape[1:] if len(obs['right_gripper_joint_pos'].shape) > 1 else (),
                    }
                
                episode_len = len(obs['left_arm_joint_pos'])
                total_steps += episode_len
                episode_ends.append(total_steps)
                
        except Exception as e:
            print(f"Error scanning {hdf5_file}: {e}")
            continue
    
    print(f"\nTotal steps: {total_steps}")
    print(f"Number of episodes: {len(episode_ends)}")
    print(f"Sample shapes: {sample_shapes}")
    
    # Create memmap files
    print("\nCreating memmap files...")
    mmap_files = {}
    
    # Depth images - store as float16 to save space
    for key in ['left_hand_depth', 'right_hand_depth', 'head_depth']:
        shape = (total_steps,) + sample_shapes[key]
        mmap_path = output_path / f"{key}.npy"
        print(f"  Creating {mmap_path} with shape {shape}")
        mmap_files[key] = np.memmap(mmap_path, dtype=np.float16, mode='w+', shape=shape)
    
    # Joint positions - float32
    for key in ['left_arm_joint_pos', 'right_arm_joint_pos']:
        shape = (total_steps,) + sample_shapes[key]
        mmap_path = output_path / f"{key}.npy"
        print(f"  Creating {mmap_path} with shape {shape}")
        mmap_files[key] = np.memmap(mmap_path, dtype=np.float32, mode='w+', shape=shape)
    
    # Gripper positions - float32
    for key in ['left_gripper_joint_pos', 'right_gripper_joint_pos']:
        if sample_shapes[key] == ():
            shape = (total_steps,)
        else:
            shape = (total_steps,) + sample_shapes[key]
        mmap_path = output_path / f"{key}.npy"
        print(f"  Creating {mmap_path} with shape {shape}")
        mmap_files[key] = np.memmap(mmap_path, dtype=np.float32, mode='w+', shape=shape)
    
    # Second pass: write data
    print("\nPass 2: Writing data...")
    current_idx = 0
    
    for hdf5_file in tqdm(all_hdf5_files, desc="Converting"):
        try:
            with h5py.File(hdf5_file, 'r') as f:
                obs = f['observations']
                episode_len = len(obs['left_arm_joint_pos'])
                end_idx = current_idx + episode_len
                
                # Depth images - normalize to 0-1, handle inf/nan
                for key in ['left_hand_depth', 'right_hand_depth', 'head_depth']:
                    raw_data = obs[key][:]  # Load raw data
                    data = raw_data.astype(np.float32)
                    
                    # Replace inf with 0
                    data = np.where(np.isinf(data), 0.0, data)
                    # Replace nan with 0
                    data = np.where(np.isnan(data), 0.0, data)
                    
                    # Normalize to 0-1
                    valid_mask = (data > 0) & (data < 100)  # Reasonable depth range
                    if valid_mask.sum() > 0:
                        valid_data = data[valid_mask]
                        depth_min = np.percentile(valid_data, 1)
                        depth_max = np.percentile(valid_data, 99)
                        if depth_max > depth_min:
                            data = np.clip(data, depth_min, depth_max)
                            data = (data - depth_min) / (depth_max - depth_min + 1e-8)
                        else:
                            data = np.zeros_like(data)
                    else:
                        data = np.zeros_like(data)
                    
                    mmap_files[key][current_idx:end_idx] = data.astype(np.float16)
                
                # Joint positions
                for key in ['left_arm_joint_pos', 'right_arm_joint_pos']:
                    data = obs[key][:].astype(np.float32)
                    mmap_files[key][current_idx:end_idx] = data
                
                # Gripper positions - handle both scalar and array
                for key in ['left_gripper_joint_pos', 'right_gripper_joint_pos']:
                    data = obs[key][:].astype(np.float32)
                    if data.ndim > 1:
                        data = data.squeeze()
                    mmap_files[key][current_idx:end_idx] = data
                
                current_idx = end_idx
                
        except Exception as e:
            print(f"Error converting {hdf5_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Flush all memmap files
    print("\nFlushing memmap files...")
    for key, mmap in mmap_files.items():
        mmap.flush()
        del mmap
    
    # Save metadata
    meta = {
        'total_steps': total_steps,
        'num_episodes': len(episode_ends),
        'episode_ends': episode_ends,
        'shapes': {k: list(v) for k, v in sample_shapes.items()},
        'dtypes': {
            'left_hand_depth': 'float16',
            'right_hand_depth': 'float16',
            'head_depth': 'float16',
            'left_arm_joint_pos': 'float32',
            'left_gripper_joint_pos': 'float32',
            'right_arm_joint_pos': 'float32',
            'right_gripper_joint_pos': 'float32',
        }
    }
    
    meta_path = output_path / "meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")
    
    # Calculate total size
    total_size = sum(os.path.getsize(output_path / f"{k}.npy") for k in mmap_files.keys())
    print(f"\nTotal output size: {total_size / (1024**3):.2f} GB")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input1", type=str, default="data/dual_arm_demos",
                        help="First input directory containing HDF5 files")
    parser.add_argument("--input2", type=str, default="",
                        help="Second input directory containing HDF5 files (optional)")
    parser.add_argument("--output", type=str, default="data/dual_arm_demos_mmap_merged",
                        help="Output directory for memmap files")
    args = parser.parse_args()
    
    input_dirs = [args.input1]
    if args.input2:
        input_dirs.append(args.input2)
    
    merge_hdf5_to_mmap(input_dirs, args.output)
