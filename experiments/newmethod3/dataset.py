import numpy as np

from geotransformer.datasets.registration.newmethod_intensity.dataset import NewMethodIntensityPairDataset
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)
from geotransformer.utils.registration import get_correspondences


class NewMethod3PairDataset(NewMethodIntensityPairDataset):
    """
    Dataset for newmethod3: weighted enhancement using intensity to modulate coordinates.
    This dataset inherits from NewMethodIntensityPairDataset but modifies the coordinates
    by multiplying them with intensity values. The features are replaced with dummy ones (all ones).
    """

    def __getitem__(self, index):
        data_dict = {}

        metadata = self.metadata[index]
        data_dict['seq_id'] = metadata['seq_id']
        data_dict['ref_frame'] = metadata['frame0']
        data_dict['src_frame'] = metadata['frame1']

        # Load point cloud and cast to float32 early to save memory
        ref_points_with_intensity = self._load_point_cloud(self._get_pcd_path(metadata['pcd0'])).astype(np.float32)
        src_points_with_intensity = self._load_point_cloud(self._get_pcd_path(metadata['pcd1'])).astype(np.float32)
        transform = metadata['transform'].astype(np.float32)

        if self.use_augmentation:
            ref_points_with_intensity, src_points_with_intensity, transform = self._augment_point_cloud(
                ref_points_with_intensity, src_points_with_intensity, transform
            )

        if self.return_corr_indices:
            # Correspondences should be calculated on original coordinates before weighting
            ref_xyz_for_corr = ref_points_with_intensity[:, :3]
            src_xyz_for_corr = src_points_with_intensity[:, :3]
            corr_indices = get_correspondences(ref_xyz_for_corr, src_xyz_for_corr, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        # Separate XYZ and intensity
        ref_xyz = ref_points_with_intensity[:, :3]
        src_xyz = src_points_with_intensity[:, :3]
        
        # Ensure intensity exists, otherwise use 1.0
        ref_intensity = ref_points_with_intensity[:, 3:4] if ref_points_with_intensity.shape[1] >= 4 else np.ones_like(ref_points_with_intensity[:, :1], dtype=np.float32)
        src_intensity = src_points_with_intensity[:, 3:4] if src_points_with_intensity.shape[1] >= 4 else np.ones_like(src_points_with_intensity[:, :1], dtype=np.float32)

        # Weight coordinates by intensity (in-place to save memory)
        np.multiply(ref_xyz, ref_intensity, out=ref_xyz)
        np.multiply(src_xyz, src_intensity, out=src_xyz)

        # Set weighted coordinates as 'points'
        data_dict['ref_points'] = ref_xyz
        data_dict['src_points'] = src_xyz

        # Set dummy features
        data_dict['ref_feats'] = np.ones((ref_points_with_intensity.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points_with_intensity.shape[0], 1), dtype=np.float32)

        data_dict['transform'] = transform
        
        return data_dict


def train_valid_data_loader(cfg, distributed):
    train_dataset = NewMethod3PairDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_min_scale=cfg.train.augmentation_min_scale,
        augmentation_max_scale=cfg.train.augmentation_max_scale,
        augmentation_shift=cfg.train.augmentation_shift,
        augmentation_rotation=cfg.train.augmentation_rotation,
        use_intensity=True,  # Ensure intensity is loaded
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
    )

    valid_dataset = NewMethod3PairDataset(
        cfg.data.dataset_root,
        'val',
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
        use_intensity=True, # Ensure intensity is loaded
    )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg):
    # The test loader needs neighbor_limits, which is calibrated on the training set.
    train_dataset = NewMethod3PairDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=False, # No augmentation for calibration
        use_intensity=True,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    test_dataset = NewMethod3PairDataset(
        cfg.data.dataset_root,
        'test',
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
        use_intensity=True, # Ensure intensity is loaded
    )
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )

    return test_loader, neighbor_limits
