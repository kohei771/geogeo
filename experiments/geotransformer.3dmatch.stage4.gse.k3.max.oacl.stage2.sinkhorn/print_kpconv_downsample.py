import numpy as np
from geotransformer.utils.data import registration_collate_fn_stack_mode
from config import make_cfg
from dataset import ThreeDMatchPairDataset

if __name__ == "__main__":
    cfg = make_cfg()
    dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        'val',
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
    )
    sample = dataset[0]
    neighbor_limits = [32] * cfg.backbone.num_stages
    collated = registration_collate_fn_stack_mode(
        [sample],
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
