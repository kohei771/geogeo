import os.path as osp
import random

import numpy as np
import torch.utils.data

from geotransformer.utils.common import load_pickle
from geotransformer.utils.pointcloud import (
    random_sample_rotation,
    get_transform_from_rotation_translation,
    get_rotation_translation_from_transform,
)
from geotransformer.utils.registration import get_correspondences
from geotransformer.datasets.registration.newmethod_intensity.dataset import NewMethodIntensityPairDataset


class IntensityOnlyDataset(NewMethodIntensityPairDataset):
    """
    Intensity-only dataset that inherits from NewMethodIntensityPairDataset
    but only uses intensity as features (input_dim=1), not (x,y,z,intensity).
    """
    
    def __getitem__(self, index):
        data_dict = {}

        metadata = self.metadata[index]
        data_dict['seq_id'] = metadata['seq_id']
        data_dict['ref_frame'] = metadata['frame0']
        data_dict['src_frame'] = metadata['frame1']

        ref_points = self._load_point_cloud(self._get_pcd_path(metadata['pcd0']))
        src_points = self._load_point_cloud(self._get_pcd_path(metadata['pcd1']))
        transform = metadata['transform']

        if self.use_augmentation:
            ref_points, src_points, transform = self._augment_point_cloud(ref_points, src_points, transform)

        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points[:, :3], src_points[:, :3], transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['ref_points'] = ref_points[:, :3].astype(np.float32)  # Only xyz coordinates
        data_dict['src_points'] = src_points[:, :3].astype(np.float32)  # Only xyz coordinates
        
        # Intensity-only features: use only intensity channel as feature (input_dim=1)
        if self.use_intensity and ref_points.shape[1] >= 4:
            data_dict['ref_feats'] = ref_points[:, 3:4].astype(np.float32)  # Only intensity
        else:
            # If no intensity, use dummy features
            data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
            
        if self.use_intensity and src_points.shape[1] >= 4:
            data_dict['src_feats'] = src_points[:, 3:4].astype(np.float32)  # Only intensity
        else:
            # If no intensity, use dummy features
            data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
            
        data_dict['transform'] = transform.astype(np.float32)
        return data_dict


def train_valid_data_loader(cfg, distributed):
    train_dataset = IntensityOnlyDataset(
        dataset_root=cfg.data.dataset_root,
        subset='train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_min_scale=cfg.train.augmentation_min_scale,
        augmentation_max_scale=cfg.train.augmentation_max_scale,
        augmentation_shift=cfg.train.augmentation_shift,
        augmentation_rotation=cfg.train.augmentation_rotation,
        use_intensity=cfg.train.use_intensity,
    )
    
    val_dataset = IntensityOnlyDataset(
        dataset_root=cfg.data.dataset_root,
        subset='val',
        point_limit=cfg.train.point_limit,
        use_augmentation=False,
        use_intensity=cfg.train.use_intensity,
    )

    if distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.train.num_workers,
        collate_fn=lambda x: x,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.test.num_workers,
        collate_fn=lambda x: x,
        pin_memory=True,
        drop_last=False,
    )

    # Estimate neighbor limits
    neighbor_limits = []
    for stage_id in range(cfg.backbone.num_stages):
        neighbor_limits.append(cfg.backbone.kernel_size)

    return train_loader, val_loader, neighbor_limits


def test_data_loader(cfg):
    test_dataset = IntensityOnlyDataset(
        dataset_root=cfg.data.dataset_root,
        subset='test',
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
        use_intensity=cfg.test.use_intensity,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=cfg.test.num_workers,
        collate_fn=lambda x: x,
        pin_memory=True,
        drop_last=False,
    )

    # Estimate neighbor limits
    neighbor_limits = []
    for stage_id in range(cfg.backbone.num_stages):
        neighbor_limits.append(cfg.backbone.kernel_size)

    return test_loader, neighbor_limits