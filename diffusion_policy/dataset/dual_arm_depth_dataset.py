from typing import Dict
import torch
import numpy as np
import h5py
import copy
import glob
import re
from tqdm import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


def natural_sort_key(filename):
    """
    Sort filenames with embedded numbers naturally.
    Example: demo_1.hdf5, demo_2.hdf5, ..., demo_10.hdf5, demo_20.hdf5
    Instead of: demo_1.hdf5, demo_10.hdf5, demo_2.hdf5, demo_20.hdf5
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', filename)]


class DualArmDepthDataset(BaseImageDataset):
    """
    Dataset for dual-arm robot with depth image observations.

    Expected data structure:
    - Multiple HDF5 files, each containing one episode
    - Each HDF5 file structure:
        actions/
            left_arm_action      (N, 6)
            left_gripper_action  (N,)
            right_arm_action     (N, 6)
            right_gripper_action (N,)
        observations/
            left_hand_depth      (N, 240, 320)
            right_hand_depth     (N, 240, 320)
            head_depth           (N, 240, 320)
    """

    def __init__(self,
            dataset_path,  # Can be a file pattern like "data/*.hdf5" or a directory
            horizon=16,
            pad_before=1,
            pad_after=7,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            depth_as_3channel=False  # Set to True if you want to replicate to 3 channels for RGB encoders
            ):
        super().__init__()

        # Find all HDF5 files
        import os
        if os.path.isdir(dataset_path):
            hdf5_files = sorted(glob.glob(os.path.join(dataset_path, "*.hdf5")), key=natural_sort_key)
        else:
            hdf5_files = sorted(glob.glob(dataset_path), key=natural_sort_key)

        if len(hdf5_files) == 0:
            raise ValueError(f"No HDF5 files found at {dataset_path}")

        print(f"Found {len(hdf5_files)} HDF5 files")

        # Create replay buffer
        replay_buffer = ReplayBuffer.create_empty_numpy()

        # Load data from each HDF5 file
        for hdf5_path in tqdm(hdf5_files, desc="Loading episodes"):
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    # Validate file structure
                    if 'actions' not in f or 'observations' not in f:
                        raise ValueError(f"Missing required groups 'actions' or 'observations' in {hdf5_path}")

                    # Load actions
                    try:
                        left_arm_action = f['actions']['left_arm_action'][:]
                        left_gripper_action = f['actions']['left_gripper_action'][:]
                        right_arm_action = f['actions']['right_arm_action'][:]
                        right_gripper_action = f['actions']['right_gripper_action'][:]
                    except KeyError as e:
                        raise ValueError(f"Missing required action key in {hdf5_path}: {e}")

                    # Concatenate actions: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
                    # Reshape gripper actions to (N, 1)
                    left_gripper_action = left_gripper_action.reshape(-1, 1)
                    right_gripper_action = right_gripper_action.reshape(-1, 1)

                    actions = np.concatenate([
                        left_arm_action,
                        left_gripper_action,
                        right_arm_action,
                        right_gripper_action
                    ], axis=-1).astype(np.float32)  # (N, 14)

                    # Load depth images
                    try:
                        left_hand_depth = f['observations']['left_hand_depth'][:]
                        right_hand_depth = f['observations']['right_hand_depth'][:]
                        head_depth = f['observations']['head_depth'][:]
                    except KeyError as e:
                        raise ValueError(f"Missing required observation key in {hdf5_path}: {e}")

                    # Validate episode length
                    episode_len = len(actions)
                    if episode_len == 0:
                        print(f"Warning: Skipping empty episode in {hdf5_path}")
                        continue

                    # Validate that all arrays have the same length
                    if not (len(left_hand_depth) == len(right_hand_depth) == len(head_depth) == episode_len):
                        raise ValueError(
                            f"Data length mismatch in {hdf5_path}: "
                            f"actions={episode_len}, left_hand_depth={len(left_hand_depth)}, "
                            f"right_hand_depth={len(right_hand_depth)}, head_depth={len(head_depth)}"
                        )

                    # Validate action dimensions
                    if actions.shape[-1] != 14:
                        raise ValueError(
                            f"Expected action dimension 14, got {actions.shape[-1]} in {hdf5_path}"
                        )

                    # Add episode to replay buffer
                    episode = {
                        'left_hand_depth': left_hand_depth,
                        'right_hand_depth': right_hand_depth,
                        'head_depth': head_depth,
                        'action': actions
                    }
                    replay_buffer.add_episode(episode)

            except Exception as e:
                print(f"Error loading {hdf5_path}: {e}")
                raise

        print(f"Loaded {replay_buffer.n_episodes} episodes, {replay_buffer.n_steps} total steps")

        # Create train/val split
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        # Create sampler
        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)

        self.replay_buffer = replay_buffer
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.depth_as_3channel = depth_as_3channel

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # Normalize actions only
        data = {
            'action': self.replay_buffer['action']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        # Depth images will be normalized to [0, 1] or [-1, 1]
        normalizer['left_hand_depth'] = get_image_range_normalizer()
        normalizer['right_hand_depth'] = get_image_range_normalizer()
        normalizer['head_depth'] = get_image_range_normalizer()

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """
        Convert sampled data to the format expected by the policy.

        Args:
            sample: dict with keys ['left_hand_depth', 'right_hand_depth', 'head_depth', 'action']

        Returns:
            dict with 'obs' (dict of images) and 'action'
        """
        # Process depth images
        # Input shape: (T, H, W) - grayscale depth
        # Output shape: (T, C, H, W) where C=1 (single channel depth)

        def process_depth(depth):
            # depth shape: (T, 240, 320)
            depth = depth.astype(np.float32)

            # Handle invalid values (inf, -inf, nan)
            # Replace with 0 (or could use median of valid pixels)
            invalid_mask = ~np.isfinite(depth)  # True for inf, -inf, nan
            if invalid_mask.any():
                # Option 1: Replace with 0
                depth[invalid_mask] = 0.0

                # Option 2 (alternative): Replace with median of valid pixels
                # valid_depth = depth[~invalid_mask]
                # if len(valid_depth) > 0:
                #     median_val = np.median(valid_depth)
                #     depth[invalid_mask] = median_val
                # else:
                #     depth[invalid_mask] = 0.0

            # Normalize depth values to [0, 1]
            # Use percentile-based normalization for robustness
            # Only compute percentiles on valid (non-zero) values
            valid_depth = depth[depth > 0]
            if len(valid_depth) > 0:
                depth_min = np.percentile(valid_depth, 1)  # Handle outliers
                depth_max = np.percentile(valid_depth, 99)

                if depth_max > depth_min:
                    depth = np.clip(depth, depth_min, depth_max)
                    depth = (depth - depth_min) / (depth_max - depth_min)
                else:
                    # All valid values are the same, set to 0.5
                    depth = np.where(depth > 0, 0.5, 0.0)
            else:
                # No valid depth values, set everything to 0
                depth = np.zeros_like(depth)

            if self.depth_as_3channel:
                # Replicate to 3 channels: (T, 240, 320) -> (T, 3, 240, 320)
                depth = np.stack([depth, depth, depth], axis=1)
            else:
                # Add single channel dimension: (T, 240, 320) -> (T, 1, 240, 320)
                depth = depth[:, np.newaxis, :, :]

            return depth

        left_hand_depth = process_depth(sample['left_hand_depth'])
        right_hand_depth = process_depth(sample['right_hand_depth'])
        head_depth = process_depth(sample['head_depth'])

        data = {
            'obs': {
                'left_hand_depth': left_hand_depth,   # T, C, H, W
                'right_hand_depth': right_hand_depth, # T, C, H, W
                'head_depth': head_depth,             # T, C, H, W
            },
            'action': sample['action'].astype(np.float32)  # T, 14
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])
