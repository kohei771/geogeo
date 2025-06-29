import argparse
import os.path as osp
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from geotransformer.engine import SingleTester
from geotransformer.utils.common import ensure_dir, get_log_string
from geotransformer.utils.torch import release_cuda

from config import make_cfg
from dataset import test_data_loader
from loss import Evaluator
from model import create_model


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.evaluator = Evaluator(cfg).cuda()

        # preparation
        self.output_dir = osp.join(cfg.feature_dir)
        ensure_dir(self.output_dir)

        self.visualized = False  # 可視化フラグ

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']
        message = f'seq_id: {seq_id}, id0: {ref_frame}, id1: {src_frame}'
        message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def plot_registration(self, src, ref, src_aligned=None, title=''):
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(src[:, 0], src[:, 1], src[:, 2], c='r', s=1, label='src')
        ax1.scatter(ref[:, 0], ref[:, 1], ref[:, 2], c='b', s=1, label='ref')
        ax1.set_title('Before Registration')
        ax1.legend()
        ax1.axis('off')

        ax2 = fig.add_subplot(122, projection='3d')
        if src_aligned is not None:
            ax2.scatter(src_aligned[:, 0], src_aligned[:, 1], src_aligned[:, 2], c='r', s=1, label='src_aligned')
        ax2.scatter(ref[:, 0], ref[:, 1], ref[:, 2], c='b', s=1, label='ref')
        ax2.set_title('After Registration')
        ax2.legend()
        ax2.axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']

        # 可視化: 最初のサンプルのみ
        if not self.visualized:
            src = output_dict['src_points']  # (N,3) torch.Tensor or np.ndarray
            ref = output_dict['ref_points']
            est = output_dict['estimated_transform']  # (4,4)
            if hasattr(src, 'cpu'):
                src = src.cpu().numpy()
            if hasattr(ref, 'cpu'):
                ref = ref.cpu().numpy()
            if hasattr(est, 'cpu'):
                est = est.cpu().numpy()
            # src点群に推定変換を適用
            src_h = np.concatenate([src, np.ones((src.shape[0], 1))], axis=1)  # (N,4)
            src_aligned = (est @ src_h.T).T[:, :3]
            self.plot_registration(src, ref, src_aligned, title=f'{seq_id} {src_frame}->{ref_frame}')
            self.visualized = True

        file_name = osp.join(self.output_dir, f'{seq_id}_{src_frame}_{ref_frame}.npz')
        np.savez_compressed(
            file_name,
            ref_points=release_cuda(output_dict['ref_points']),
            src_points=release_cuda(output_dict['src_points']),
            ref_points_f=release_cuda(output_dict['ref_points_f']),
            src_points_f=release_cuda(output_dict['src_points_f']),
            ref_points_c=release_cuda(output_dict['ref_points_c']),
            src_points_c=release_cuda(output_dict['src_points_c']),
            ref_feats_c=release_cuda(output_dict['ref_feats_c']),
            src_feats_c=release_cuda(output_dict['src_feats_c']),
            ref_node_corr_indices=release_cuda(output_dict['ref_node_corr_indices']),
            src_node_corr_indices=release_cuda(output_dict['src_node_corr_indices']),
            ref_corr_points=release_cuda(output_dict['ref_corr_points']),
            src_corr_points=release_cuda(output_dict['src_corr_points']),
            corr_scores=release_cuda(output_dict['corr_scores']),
            gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
            gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
            estimated_transform=release_cuda(output_dict['estimated_transform']),
            transform=release_cuda(data_dict['transform']),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=str, default=None, help='Path to pretrained weights')
    parser.add_argument('--near', action='store_true', help='use newmethod_near data/metadata')
    args, unknown = parser.parse_known_args()
    sys.argv = [arg for arg in sys.argv if arg != '--near']
    cfg = make_cfg(use_near=args.near)
    if args.snapshot is not None:
        cfg.snapshot = args.snapshot
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()
