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


class OdometryKittiPairDataset(torch.utils.data.Dataset):
    ODOMETRY_KITTI_DATA_SPLIT = {
        'train': ['00', '01', '02', '03', '04', '05'],
        'val': ['06', '07'],
        'test': ['08', '09', '10'],
    }

    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_min_scale=0.8,
        augmentation_max_scale=1.2,
        augmentation_shift=2.0,
        augmentation_rotation=1.0,
        return_corr_indices=False,
        matching_radius=None,
        use_intensity=False,  # intensity拡張用
        use_near=False,       # 追加: nearデータ用
    ):
        super(OdometryKittiPairDataset, self).__init__()
        self.dataset_root = dataset_root
        self.subset = subset
        self.point_limit = point_limit
        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise
        self.augmentation_min_scale = augmentation_min_scale
        self.augmentation_max_scale = augmentation_max_scale
        self.augmentation_shift = augmentation_shift
        self.augmentation_rotation = augmentation_rotation
        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')
        # ここでuse_nearに応じてmetadataファイル名を切り替え
        if use_near:
            self.metadata = load_pickle(osp.join(self.dataset_root, 'metadata', f'{subset}_newmethod_near.pkl'))
        else:
            self.metadata = load_pickle(osp.join(self.dataset_root, 'metadata', f'{subset}_newmethod.pkl'))
        self.use_intensity = use_intensity
        self.use_near = use_near

    def _augment_point_cloud(self, ref_points, src_points, transform):
        rotation, translation = get_rotation_translation_from_transform(transform)
        # ...既存のaugmentation処理...
        return ref_points, src_points, transform

    def _get_pcd_path(self, rel_path):
        return osp.join(self.dataset_root, rel_path)

    def _load_point_cloud(self, path):
        return np.load(path)

    def __len__(self):
        return len(self.metadata)
