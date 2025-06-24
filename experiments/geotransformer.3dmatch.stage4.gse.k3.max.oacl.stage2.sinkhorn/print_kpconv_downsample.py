import numpy as np
from geotransformer.utils.data import registration_collate_fn_stack_mode
from config import make_cfg
ｓimport os

if __name__ == "__main__":
    cfg = make_cfg()
    # デモ用データの絶対パス取得
    base_dir = os.path.join(cfg.root_dir, "data", "demo")
    src_file = os.path.join(base_dir, "src.npy")
    ref_file = os.path.join(base_dir, "ref.npy")
    src_points = np.load(src_file)
    ref_points = np.load(ref_file)
    src_feats = np.ones((src_points.shape[0], 1), dtype=np.float32)
    ref_feats = np.ones((ref_points.shape[0], 1), dtype=np.float32)
    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }
    neighbor_limits = [32] * cfg.backbone.num_stages
    collated = registration_collate_fn_stack_mode(
        [data_dict],
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        precompute_data=True,
    )
    lengths_pyramid = np.array(collated['lengths'])  # shape: (num_stages, 2)
    print("各ステージのref/src点数:")
    for i, l in enumerate(lengths_pyramid):
        print(f"stage {i}: ref={l[0]}, src={l[1]}")
