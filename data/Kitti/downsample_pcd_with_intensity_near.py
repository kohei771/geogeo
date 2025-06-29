import os
import os.path as osp
import open3d as o3d
import numpy as np
import glob
from tqdm import tqdm
from scipy.spatial import cKDTree

def main():
    input_root = 'sequences'
    output_root = 'newmethod_near'
    voxel_size = 0.3
    num_seq = 21  # 00-20
    distance_threshold = 20.0  # 例: 20m以内の点のみ残す
    for i in range(num_seq):
        seq_id = '{:02d}'.format(i)
        seq_in_dir = osp.join(input_root, seq_id, 'velodyne')
        seq_out_dir = osp.join(output_root, seq_id)
        os.makedirs(seq_out_dir, exist_ok=True)
        file_names = sorted(glob.glob(osp.join(seq_in_dir, '*.bin')))
        for file_name in tqdm(file_names, desc=f'Seq {seq_id}'):
            frame = osp.splitext(osp.basename(file_name))[0]
            new_file_name = osp.join(seq_out_dir, frame + '.npy')
            points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
            xyz = points[:, :3]
            intensity = points[:, 3:4]
            # 距離フィルタ: 原点からdistance_threshold未満の点のみ残す
            dists = np.linalg.norm(xyz, axis=1)
            mask = dists < distance_threshold
            xyz = xyz[mask]
            intensity = intensity[mask]
            if xyz.shape[0] == 0:
                continue  # skip empty
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd = pcd.voxel_down_sample(voxel_size)
            down_xyz = np.array(pcd.points).astype(np.float32)
            if down_xyz.shape[0] == 0:
                continue  # skip empty
            tree = cKDTree(xyz)
            _, idx = tree.query(down_xyz, k=1)
            down_intensity = intensity[idx]
            down_points = np.concatenate([down_xyz, down_intensity], axis=1)
            np.save(new_file_name, down_points)

if __name__ == '__main__':
    main()
