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
        use_intensity=False,
        use_near=False,  # 追加
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
        if use_near:
            self.metadata = load_pickle(osp.join(self.dataset_root, 'metadata', f'{subset}_newmethod_near.pkl'))
        else:
            self.metadata = load_pickle(osp.join(self.dataset_root, 'metadata', f'{subset}_newmethod.pkl'))
        self.use_intensity = use_intensity
        self.use_near = use_near

    def _augment_point_cloud(self, ref_points, src_points, transform):
        rotation, translation = get_rotation_translation_from_transform(transform)
        # xyz/intensity分離
        ref_xyz, ref_intensity = ref_points[:, :3], ref_points[:, 3:] if ref_points.shape[1] > 3 else None
        src_xyz, src_intensity = src_points[:, :3], src_points[:, 3:] if src_points.shape[1] > 3 else None
        # add gaussian noise (xyzのみ)
        ref_xyz = ref_xyz + (np.random.rand(ref_xyz.shape[0], 3) - 0.5) * self.augmentation_noise
        src_xyz = src_xyz + (np.random.rand(src_xyz.shape[0], 3) - 0.5) * self.augmentation_noise
        # random rotation
        aug_rotation = random_sample_rotation(self.augmentation_rotation)
        if random.random() > 0.5:
            ref_xyz = np.matmul(ref_xyz, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_xyz = np.matmul(src_xyz, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)
        # random scaling
        scale = random.random()
        scale = self.augmentation_min_scale + (self.augmentation_max_scale - self.augmentation_min_scale) * scale
        ref_xyz = ref_xyz * scale
        src_xyz = src_xyz * scale
        translation = translation * scale
        # random shift
        ref_shift = np.random.uniform(-self.augmentation_shift, self.augmentation_shift, 3)
        src_shift = np.random.uniform(-self.augmentation_shift, self.augmentation_shift, 3)
        ref_xyz = ref_xyz + ref_shift
        src_xyz = src_xyz + src_shift
        translation = -np.matmul(src_shift[None, :], rotation.T) + translation + ref_shift
        # compose transform from rotation and translation
        transform = get_transform_from_rotation_translation(rotation, translation)
        # intensityを戻す
        if ref_intensity is not None:
            ref_points = np.concatenate([ref_xyz, ref_intensity], axis=1)
        else:
            ref_points = ref_xyz
        if src_intensity is not None:
            src_points = np.concatenate([src_xyz, src_intensity], axis=1)
        else:
            src_points = src_xyz
        return ref_points, src_points, transform

    def _get_pcd_path(self, pcd_relpath):
        if pcd_relpath.startswith('newmethod/') or pcd_relpath.startswith('newmethod_near/'):
            return osp.join(self.dataset_root, pcd_relpath)
        return osp.join(self.dataset_root, 'newmethod', pcd_relpath)

    def _load_point_cloud(self, file_name):
        points = np.load(file_name)
        if points.shape[1] == 3 and self.use_intensity:
            # intensity列がなければダミーで1.0を追加
            points = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

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
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        # intensity-only: 点群shapeが(N, 4)ならintensity（4列目）のみを特徴量とする（sqrt変換）
        if self.use_intensity and ref_points.shape[1] >= 4:
            data_dict['ref_feats'] = np.sqrt(ref_points[:, 3:4].astype(np.float32))
        else:
            data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        if self.use_intensity and src_points.shape[1] >= 4:
            data_dict['src_feats'] = np.sqrt(src_points[:, 3:4].astype(np.float32))
        else:
            data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)
        return data_dict

    def __len__(self):
        return len(self.metadata)

# for compatibility with experiments code
NewMethodIntensityPairDataset = OdometryKittiPairDataset
