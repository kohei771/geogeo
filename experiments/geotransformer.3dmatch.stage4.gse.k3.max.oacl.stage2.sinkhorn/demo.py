import argparse
import time
import torch.profiler

import torch
import numpy as np

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
#from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, save_geometries_as_image
from geotransformer.utils.open3d import save_pointclouds_as_2d_image
from geotransformer.utils.registration import compute_registration_error

from config import make_cfg
from model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", required=True, help="src point cloud numpy file")
    parser.add_argument("--ref_file", required=True, help="src point cloud numpy file")
    parser.add_argument("--gt_file", required=True, help="ground-truth transformation file")
    parser.add_argument("--weights", required=True, help="model weights file")
    return parser


def load_data(args):
    src_points = np.load(args.src_file)
    ref_points = np.load(args.ref_file)
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }

    if args.gt_file is not None:
        transform = np.load(args.gt_file)
        data_dict["transform"] = transform.astype(np.float32)

    return data_dict

def safe_get_stage3(data_dict):
    # バッチ1個分を抽出
    if isinstance(data_dict, (list, tuple)):
        data_dict = data_dict[0]
    # 各stageの点群・特徴量を取得
    def get_last(x):
        if isinstance(x, (list, tuple)):
            return x[-1]
        return x
    ref_points = get_last(data_dict["ref_points"])
    src_points = get_last(data_dict["src_points"])
    ref_feats = get_last(data_dict["ref_feats"])
    src_feats = get_last(data_dict["src_feats"])
    transform = data_dict.get("transform", None)
    return ref_points, src_points, ref_feats, src_feats, transform

def sample_points(data_dict, n):
    """
    data_dictからref/src点群・特徴量をN個だけランダムサンプリングして新しいdata_dictを返す。
    """
    import torch
    # バッチ1個分を抽出
    if isinstance(data_dict, (list, tuple)):
        data_dict = data_dict[0]
    # 各stageの点群・特徴量を取得
    def get_last(x):
        if isinstance(x, (list, tuple)):
            return x[-1]
        return x
    ref_points = get_last(data_dict["ref_points"])
    src_points = get_last(data_dict["src_points"])
    ref_feats = get_last(data_dict["ref_feats"])
    src_feats = get_last(data_dict["src_feats"])
    # torch.Tensor化
    ref_points = torch.as_tensor(ref_points)
    src_points = torch.as_tensor(src_points)
    ref_feats = torch.as_tensor(ref_feats)
    src_feats = torch.as_tensor(src_feats)
    # サンプリング数Nが点数より多い場合は全点
    n_ref = min(n, ref_points.shape[0])
    n_src = min(n, src_points.shape[0])
    # ランダムインデックス
    ref_idx = torch.randperm(ref_points.shape[0])[:n_ref]
    src_idx = torch.randperm(src_points.shape[0])[:n_src]
    # サンプリング
    ref_points_sample = ref_points[ref_idx]
    src_points_sample = src_points[src_idx]
    ref_feats_sample = ref_feats[ref_idx]
    src_feats_sample = src_feats[src_idx]
    # 新しいdata_dictを作成
    data_dict_sample = {
        "ref_points": ref_points_sample,
        "src_points": src_points_sample,
        "ref_feats": ref_feats_sample,
        "src_feats": src_feats_sample,
    }
    if "transform" in data_dict:
        data_dict_sample["transform"] = torch.as_tensor(data_dict["transform"])
    return data_dict_sample


def main():
    parser = make_parser()
    args = parser.parse_args()
    cfg = make_cfg()
    # prepare data
    data_dict_raw = load_data(args)
    # neighbor_limitsの長さをnum_stagesに合わせる
    neighbor_limits = [38] * cfg.backbone.num_stages
    data_dict = registration_collate_fn_stack_mode(
        [data_dict_raw], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )
    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])
    # 1回目（ウォームアップ）
    torch.cuda.synchronize()
    start_time = time.time()
    data_dict1 = to_cuda(data_dict)
    output_dict1 = model(data_dict1)
    torch.cuda.synchronize()
    elapsed1 = time.time() - start_time
    data_dict1 = release_cuda(data_dict1)
    output_dict1 = release_cuda(output_dict1)
    ref_points1 = output_dict1["ref_points"]
    src_points1 = output_dict1["src_points"]
    estimated_transform1 = output_dict1["estimated_transform"]
    transform1 = data_dict["transform"]
    rre1, rte1 = compute_registration_error(transform1, estimated_transform1)
    print(f"[WARMUP] RRE(deg): {rre1:.3f}, RTE(m): {rte1:.3f}, Time(s): {elapsed1:.3f}")

    # 2回目（計測のみ、プロファイラなし）
    torch.cuda.synchronize()
    start_time = time.time()
    data_dict2 = to_cuda(data_dict)
    output_dict2 = model(data_dict2)
    torch.cuda.synchronize()
    elapsed2 = time.time() - start_time
    data_dict2 = release_cuda(data_dict2)
    output_dict2 = release_cuda(output_dict2)
    ref_points2 = output_dict2["ref_points"]
    src_points2 = output_dict2["src_points"]
    estimated_transform2 = output_dict2["estimated_transform"]
    transform2 = data_dict["transform"]
    rre2, rte2 = compute_registration_error(transform2, estimated_transform2)
    print(f"[RUN2][ALL] RRE(deg): {rre2:.3f}, RTE(m): {rte2:.3f}, Time(s): {elapsed2:.3f}")

    # --- サンプリングver推論（例: N=128） ---
    N = 128
    data_dict_sample = sample_points(data_dict_raw, N)
    data_dict_sample = registration_collate_fn_stack_mode(
        [data_dict_sample], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )
    torch.cuda.synchronize()
    start_time = time.time()
    data_dict_sample_cuda = to_cuda(data_dict_sample)
    output_dict_sample = model(data_dict_sample_cuda)
    torch.cuda.synchronize()
    elapsed_sample = time.time() - start_time
    data_dict_sample_cuda = release_cuda(data_dict_sample_cuda)
    output_dict_sample = release_cuda(output_dict_sample)
    ref_points_sample = output_dict_sample["ref_points"]
    src_points_sample = output_dict_sample["src_points"]
    estimated_transform_sample = output_dict_sample["estimated_transform"]
    transform_sample = data_dict_sample["transform"]
    rre_sample, rte_sample = compute_registration_error(transform_sample, estimated_transform_sample)
    print(f"[SAMPLED N={N}] RRE(deg): {rre_sample:.3f}, RTE(m): {rte_sample:.3f}, Time(s): {elapsed_sample:.3f}")
    # --- サンプリングverここまで ---

    # 3回目（プロファイラ有効）
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        torch.cuda.synchronize()
        start_time = time.time()
        data_dict3 = to_cuda(data_dict)
        output_dict3 = model(data_dict3)
        torch.cuda.synchronize()
        elapsed3 = time.time() - start_time
        data_dict3 = release_cuda(data_dict3)
        output_dict3 = release_cuda(output_dict3)
    ref_points3 = output_dict3["ref_points"]
    src_points3 = output_dict3["src_points"]
    estimated_transform3 = output_dict3["estimated_transform"]
    transform3 = data_dict["transform"]
    rre3, rte3 = compute_registration_error(transform3, estimated_transform3)
    print(f"[PROFILED] RRE(deg): {rre3:.3f}, RTE(m): {rte3:.3f}, Time(s): {elapsed3:.3f}")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    # visualization（3回目の結果を使用）
    ref_pcd = make_open3d_point_cloud(ref_points3)
    ref_pcd.estimate_normals()
    ref_pcd.paint_uniform_color(get_color("custom_yellow"))
    src_pcd = make_open3d_point_cloud(src_points3)
    src_pcd.estimate_normals()
    src_pcd.paint_uniform_color(get_color("custom_blue"))
    save_pointclouds_as_2d_image("before_registration_2d.png", ref_pcd, src_pcd)
    src_pcd = src_pcd.transform(estimated_transform3)
    save_pointclouds_as_2d_image("after_registration_2d.png", ref_pcd, src_pcd)
    # compute error
    rre, rte = compute_registration_error(transform3, estimated_transform3)
    print(f"RRE(deg): {rre:.3f}, RTE(m): {rte:.3f}, Time(s): {elapsed3:.3f}")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))  # プロファイラは必要に応じて


if __name__ == "__main__":
    main()
