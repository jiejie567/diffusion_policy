"""
Memory-mapped numpy dataset for dual-arm depth data.
Uses numpy memmap for memory-efficient fast training - data stays on disk.
"""

import numpy as np
import torch
from typing import Dict, Optional
from pathlib import Path
import json
import copy

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class DualArmDepthMmapDataset(BaseImageDataset):
    """
    Memory-mapped dataset using numpy memmap.
    Data stays on disk and is loaded on-demand - minimal RAM usage.
    Much faster than HDF5 random access.
    """
    
    def __init__(
        self,
        dataset_path: str,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: Optional[int] = None,
        depth_as_3channel: bool = True,
    ):
        super().__init__()
        
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.depth_as_3channel = depth_as_3channel
        
        # Load metadata
        dataset_dir = Path(dataset_path)
        meta_path = dataset_dir / "meta.json"
        
        if not meta_path.exists():
            raise FileNotFoundError(
                f"meta.json not found in {dataset_path}. "
                "Run scripts/convert_hdf5_to_mmap.py first."
            )
        
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        
        self.total_steps = self.meta['total_steps']
        self.num_episodes = self.meta['num_episodes']
        self.episode_ends = np.array(self.meta['episode_ends'])
        
        print(f"Loading memmap dataset from {dataset_path}...")
        print(f"Total steps: {self.total_steps}, Episodes: {self.num_episodes}")
        
        # Open memmap files in read-only mode
        self.data = {}
        for key in ['left_hand_depth', 'right_hand_depth', 'head_depth',
                    'left_arm_joint_pos', 'left_gripper_joint_pos',
                    'right_arm_joint_pos', 'right_gripper_joint_pos']:
            mmap_path = dataset_dir / f"{key}.npy"
            dtype = self.meta['dtypes'][key]
            
            # Determine shape
            if key in ['left_hand_depth', 'right_hand_depth', 'head_depth']:
                shape = (self.total_steps,) + tuple(self.meta['shapes'][key])
            elif key in ['left_arm_joint_pos', 'right_arm_joint_pos']:
                shape = (self.total_steps,) + tuple(self.meta['shapes'][key])
            else:  # gripper
                if self.meta['shapes'][key]:
                    shape = (self.total_steps,) + tuple(self.meta['shapes'][key])
                else:
                    shape = (self.total_steps,)
            
            self.data[key] = np.memmap(mmap_path, dtype=dtype, mode='r', shape=shape)
        
        # Train/val split by episodes
        np.random.seed(seed)
        episode_indices = np.arange(self.num_episodes)
        np.random.shuffle(episode_indices)
        
        n_val = max(1, int(self.num_episodes * val_ratio))
        val_episode_indices = set(episode_indices[:n_val])
        train_episode_indices = set(episode_indices[n_val:])
        
        if max_train_episodes is not None:
            train_episode_indices = set(list(train_episode_indices)[:max_train_episodes])
        
        # Build sample indices
        self.train_indices = []
        self.val_indices = []
        
        for ep_idx in range(self.num_episodes):
            ep_start = 0 if ep_idx == 0 else int(self.episode_ends[ep_idx - 1])
            ep_end = int(self.episode_ends[ep_idx])
            ep_len = ep_end - ep_start
            
            for local_idx in range(ep_len):
                if ep_idx in train_episode_indices:
                    self.train_indices.append((ep_idx, local_idx, ep_start, ep_len))
                elif ep_idx in val_episode_indices:
                    self.val_indices.append((ep_idx, local_idx, ep_start, ep_len))
        
        print(f"Train samples: {len(self.train_indices)}, Val samples: {len(self.val_indices)}")
        
        # Default to training mode
        self.indices = self.train_indices
    
    def set_train(self):
        self.indices = self.train_indices
    
    def set_val(self):
        self.indices = self.val_indices
    
    def get_train_dataset(self):
        self.set_train()
        return self
    
    def get_val_dataset(self):
        val_dataset = copy.copy(self)
        val_dataset.indices = self.val_indices
        return val_dataset
    
    def get_validation_dataset(self):
        return self.get_val_dataset()
    
    def get_normalizer(self, mode='limits', **kwargs) -> LinearNormalizer:
        """Get normalizer for actions."""
        left_arm = self.data['left_arm_joint_pos'][:]
        right_arm = self.data['right_arm_joint_pos'][:]
        left_gripper = self.data['left_gripper_joint_pos'][:]
        right_gripper = self.data['right_gripper_joint_pos'][:]
        
        # Ensure grippers are 2D
        if left_gripper.ndim == 1:
            left_gripper = left_gripper[:, np.newaxis]
        if right_gripper.ndim == 1:
            right_gripper = right_gripper[:, np.newaxis]
        
        actions = np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=-1)
        actions = torch.from_numpy(actions)
        
        normalizer = LinearNormalizer()
        normalizer.fit(data={'action': actions}, last_n_dims=1, mode=mode, **kwargs)
        
        # Add image normalizers (depth images normalized to [0, 1])
        normalizer['left_hand_depth'] = get_image_range_normalizer()
        normalizer['right_hand_depth'] = get_image_range_normalizer()
        normalizer['head_depth'] = get_image_range_normalizer()
        
        return normalizer
    
    def get_all_actions(self) -> torch.Tensor:
        left_arm = self.data['left_arm_joint_pos'][:]
        right_arm = self.data['right_arm_joint_pos'][:]
        left_gripper = self.data['left_gripper_joint_pos'][:]
        right_gripper = self.data['right_gripper_joint_pos'][:]
        
        if left_gripper.ndim == 1:
            left_gripper = left_gripper[:, np.newaxis]
        if right_gripper.ndim == 1:
            right_gripper = right_gripper[:, np.newaxis]
        
        return torch.from_numpy(
            np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=-1)
        )
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, local_idx, ep_start, ep_len = self.indices[idx]
        
        start_idx = local_idx - self.pad_before
        end_idx = start_idx + self.horizon
        
        obs_dict = {}
        action_list = []
        
        for t in range(start_idx, end_idx):
            valid_t = max(0, min(t, ep_len - 1))
            global_t = ep_start + valid_t
            
            # Depth images
            for key in ['left_hand_depth', 'right_hand_depth', 'head_depth']:
                img = self.data[key][global_t]  # (H, W) or (H, W, C), float16
                img = img.astype(np.float32)  # Convert to float32 for torch
                
                if img.ndim == 2:
                    img = img[np.newaxis, :, :]  # (1, H, W)
                elif img.ndim == 3 and img.shape[-1] in [1, 3]:
                    img = np.transpose(img, (2, 0, 1))  # (C, H, W)
                
                if self.depth_as_3channel and img.shape[0] == 1:
                    img = np.repeat(img, 3, axis=0)
                
                if key not in obs_dict:
                    obs_dict[key] = []
                obs_dict[key].append(torch.from_numpy(img))
            
            # Actions
            left_arm = self.data['left_arm_joint_pos'][global_t]
            left_gripper = self.data['left_gripper_joint_pos'][global_t]
            right_arm = self.data['right_arm_joint_pos'][global_t]
            right_gripper = self.data['right_gripper_joint_pos'][global_t]
            
            if np.ndim(left_gripper) == 0:
                left_gripper = np.array([left_gripper])
            if np.ndim(right_gripper) == 0:
                right_gripper = np.array([right_gripper])
            
            action = np.concatenate([left_arm, left_gripper, right_arm, right_gripper])
            action_list.append(torch.from_numpy(action.astype(np.float32)))
        
        # Stack
        for key in obs_dict:
            obs_dict[key] = torch.stack(obs_dict[key], dim=0)
        
        actions = torch.stack(action_list, dim=0)
        
        return {
            'obs': obs_dict,
            'action': actions,
        }
