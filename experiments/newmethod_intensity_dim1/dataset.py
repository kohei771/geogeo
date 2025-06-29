import numpy as np

from geotransformer.datasets.registration.newmethod_intensity.dataset import NewMethodIntensityPairDataset
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)
from geotransformer.utils.registration import get_correspondences


class IntensityOnlyDataset(NewMethodIntensityPairDataset):
    """
    Intensity-only dataset that inherits from NewMethodIntensityPairDataset
    but only uses intensity as features (input_dim=1), not (x,y,z,intensity).
    """
    def __init__(self, *args, use_near=False, **kwargs):
        self.use_near = use_near
        super().__init__(*args, **kwargs)

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
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_min_scale=cfg.train.augmentation_min_scale,
        augmentation_max_scale=cfg.train.augmentation_max_scale,
        augmentation_shift=cfg.train.augmentation_shift,
        augmentation_rotation=cfg.train.augmentation_rotation,
        use_intensity=cfg.train.use_intensity,
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

    valid_dataset = IntensityOnlyDataset(
        cfg.data.dataset_root,
        'val',
        point_limit=cfg.train.point_limit,
        use_augmentation=False,
        use_intensity=cfg.train.use_intensity,
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
    train_dataset = IntensityOnlyDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_min_scale=cfg.train.augmentation_min_scale,
        augmentation_max_scale=cfg.train.augmentation_max_scale,
        augmentation_shift=cfg.train.augmentation_shift,
        augmentation_rotation=cfg.train.augmentation_rotation,
        use_intensity=cfg.train.use_intensity,
        use_near=getattr(cfg, 'use_near', False),
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    test_dataset = IntensityOnlyDataset(
        cfg.data.dataset_root,
        'test',
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
        use_intensity=cfg.test.use_intensity,
        use_near=getattr(cfg, 'use_near', False),
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
        distributed=False,
    )
    return test_loader, neighbor_limits


if __name__ == "__main__":
    from types import SimpleNamespace
    import sys

    # 仮の設定（必要に応じて修正してください）
    cfg = SimpleNamespace()
    cfg.data = SimpleNamespace(dataset_root="data/Kitti")  # ← 実際にpklが存在するパスに修正
    cfg.train = SimpleNamespace(
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.01,
        augmentation_min_scale=0.8,
        augmentation_max_scale=1.2,
        augmentation_shift=2.0,
        augmentation_rotation=1.0,
        use_intensity=True,
        batch_size=1,
        num_workers=0,
    )
    cfg.test = SimpleNamespace(
        point_limit=None,
        use_intensity=True,
        batch_size=1,
        num_workers=0,
    )
    cfg.backbone = SimpleNamespace(
        num_stages=5,
        init_voxel_size=0.3,
        init_radius=1.0,
    )

    dataset = IntensityOnlyDataset(
        cfg.data.dataset_root,
        "train",
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        use_intensity=cfg.train.use_intensity,
    )
    sample = dataset[0]
    print("ref_feats shape:", sample["ref_feats"].shape)
    print("ref_feats (first 10):", sample["ref_feats"][:10])
    print("src_feats shape:", sample["src_feats"].shape)
    print("src_feats (first 10):", sample["src_feats"][:10])
    
    # --- ここからスーパーポイント数のprintテスト（ref/src個別・正確版・修正版） ---
    from geotransformer.utils.data import registration_collate_fn_stack_mode
    neighbor_limits = [32] * cfg.backbone.num_stages
    collated = registration_collate_fn_stack_mode(
        [sample],
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        precompute_data=True,
    )
    # 各ステージごとのref/src点数
    lengths_pyramid = np.array(collated['lengths']).reshape(cfg.backbone.num_stages, 2)  # (num_stages, 2)
    points_pyramid = collated['points'][:cfg.backbone.num_stages]
    ref_points_pyramid = [p[:l[0]] for p, l in zip(points_pyramid, lengths_pyramid)]
    src_points_pyramid = [p[l[0]:l[0]+l[1]] for p, l in zip(points_pyramid, lengths_pyramid)]
    print("各ステージのref点数:", [p.shape[0] for p in ref_points_pyramid])
    print("各ステージのsrc点数:", [p.shape[0] for p in src_points_pyramid])
    print("最終refスーパーポイント数:", ref_points_pyramid[-1].shape[0])
    print("最終srcスーパーポイント数:", src_points_pyramid[-1].shape[0])
    # --- ここまで追加 ---