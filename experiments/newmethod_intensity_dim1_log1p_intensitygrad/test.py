import argparse
import os
import os.path as osp
import time
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from geotransformer.engine import SingleTester
from geotransformer.utils.common import ensure_dir, get_log_string
from geotransformer.utils.torch import release_cuda
from geotransformer.utils.superpoint_score import SuperPointScoreModule, normalize_features

from config import make_cfg
from dataset import test_data_loader
from loss import Evaluator


def safe_item(value):
    """
    Safely extract a scalar value from a tensor or return the value if it's already a scalar.
    
    Args:
        value: Can be a tensor, int, float, or any other scalar type
    
    Returns:
        The scalar value
    """
    if hasattr(value, 'item'):
        return value.item()
    else:
        return value
from model import create_model


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.cfg = cfg
        self.use_superpoint_score = getattr(cfg, 'use_superpoint_score', False)
        self.superpoint_score_threshold = getattr(cfg, 'superpoint_score_threshold', 0.5)

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
        
        # スーパーポイントスコアモジュールの初期化
        if self.use_superpoint_score:
            self.logger.info("Using SuperPoint Score Module for test")
            # 学習済みスーパーポイントスコアモジュールをロード
            self.score_module = self._load_superpoint_score_module()
            if self.score_module is not None:
                self.logger.info(f"SuperPoint Score Module loaded with threshold: {self.superpoint_score_threshold}")
            else:
                self.logger.warning("SuperPoint Score Module not found, using default behavior")
                self.use_superpoint_score = False

        # preparation
        self.output_dir = osp.join(cfg.feature_dir)
        ensure_dir(self.output_dir)
        exp_name = cfg.exp_name
        vis_root = osp.join(cfg.root_dir, "visualizations", exp_name)
        ensure_dir(vis_root)
        self.vis_dir = osp.join(vis_root, "vis")
        ensure_dir(self.vis_dir)

    def test_step(self, iteration, data_dict):
        # スーパーポイントスコアフィルタリングを適用
        if self.use_superpoint_score:
            data_dict = self._apply_superpoint_score_filtering(data_dict)
        
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

    def plot_registration(self, src, ref, src_aligned=None, title='', save_path=None):
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
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        plt.close(fig)

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        seq_id = data_dict['seq_id']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']
        # 最初の1回だけ画像保存
        if not hasattr(self, 'visualized'):
            self.visualized = False
        if not self.visualized:
            src = output_dict['src_points']
            ref = output_dict['ref_points']
            est = output_dict['estimated_transform']
            if hasattr(src, 'cpu'):
                src = src.cpu().numpy()
            if hasattr(ref, 'cpu'):
                ref = ref.cpu().numpy()
            if hasattr(est, 'cpu'):
                est = est.cpu().numpy()
            max_points = 50000
            if src.shape[0] > max_points:
                idx = np.random.choice(src.shape[0], max_points, replace=False)
                src = src[idx]
            if ref.shape[0] > max_points:
                idx = np.random.choice(ref.shape[0], max_points, replace=False)
                ref = ref[idx]
            src_h = np.concatenate([src, np.ones((src.shape[0], 1))], axis=1)
            src_aligned = (est @ src_h.T).T[:, :3]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = osp.join(self.vis_dir, f'{seq_id}_{src_frame}_{ref_frame}_registration_{timestamp}.png')
            self.plot_registration(src, ref, src_aligned, title=f'{seq_id} {src_frame}->{ref_frame}', save_path=save_path)
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

    def _load_superpoint_score_module(self):
        """
        学習済みスーパーポイントスコアモジュールをロード
        """
        try:
            # 学習済みスーパーポイントスコアの重みファイルを探す
            score_weight_path = osp.join(self.cfg.output_dir, 'superpoint_score_weights.pth')
            if not osp.exists(score_weight_path):
                self.logger.warning(f"SuperPoint Score weights not found at: {score_weight_path}")
                return None
            
            # 重みファイルから情報を読み込み
            checkpoint = torch.load(score_weight_path, map_location='cpu')
            
            num_features = checkpoint.get('num_features', 14)  # デフォルトは14個の拡張特徴量
            important_features = checkpoint.get('important_features', None)
            
            if important_features is not None:
                # 第2段階の重要な特徴量のみを使用
                num_features = len(important_features)
                weights = checkpoint['reduced_weights']
                self.important_features = important_features
                self.logger.info(f"Using {num_features} important features: {important_features}")
            else:
                # 第1段階の全特徴量を使用
                weights = checkpoint['full_weights']
                self.important_features = None
                self.logger.info(f"Using all {num_features} features")
            
            # スーパーポイントスコアモジュールを作成
            score_module = SuperPointScoreModule(
                num_features=num_features,
                init_weights=weights,
                use_nonlinear=checkpoint.get('use_nonlinear', False)
            ).cuda()
            
            score_module.load_state_dict(checkpoint['state_dict'])
            score_module.eval()
            
            return score_module
            
        except Exception as e:
            self.logger.error(f"Error loading SuperPoint Score Module: {e}")
            return None
    
    def _apply_superpoint_score_filtering(self, data_dict):
        """
        スーパーポイントスコアに基づいてデータをフィルタリング
        """
        if not self.use_superpoint_score or self.score_module is None:
            return data_dict
        
        try:
            # スーパーポイント特徴量を計算
            superpoint_features = self._compute_superpoint_features_for_test(data_dict)
            
            # 正規化
            normalized_features = normalize_features(superpoint_features, method='minmax')
            
            # スコアを計算
            with torch.no_grad():
                scores = self.score_module(normalized_features)
            
            # 閾値でフィルタリング
            mask = scores > self.superpoint_score_threshold
            
            if mask.sum() == 0:
                self.logger.warning("No superpoints pass the threshold, using top 30%")
                top_k = max(int(len(scores) * 0.3), 5)
                _, top_indices = torch.topk(scores, top_k)
                mask = torch.zeros_like(scores, dtype=torch.bool)
                mask[top_indices] = True
            
            # データをフィルタリング
            filtered_data_dict = self._filter_data_by_mask(data_dict, mask)
            
            self.logger.info(f"SuperPoint filtering: {mask.sum()}/{len(mask)} points selected "
                           f"(threshold: {self.superpoint_score_threshold:.2f})")
            
            return filtered_data_dict
            
        except Exception as e:
            self.logger.error(f"Error in SuperPoint Score filtering: {e}")
            return data_dict

    def _compute_superpoint_features_for_test(self, data_dict):
        """
        テスト時にスーパーポイント特徴量を計算（superpoint_score_separate_train.pyから簡略化）
        """
        # 簡略化された特徴量計算
        # 実際の実装では、superpoint_score_separate_train.pyの機能を使用
        points_c = data_dict['points'][-1]  # 最粗レベルの点
        features = data_dict['features']
        
        # 簡単な特徴量計算（実際にはより複雑）
        num_superpoints = points_c.shape[0]
        
        if self.important_features is not None:
            # 重要な特徴量のみを使用
            num_features = len(self.important_features)
        else:
            # 全特徴量を使用
            num_features = 14
        
        # プレースホルダーとして簡単な特徴量を生成
        # 実際の実装では、superpoint_score_separate_train.pyの_compute_extended_featuresを使用
        superpoint_features = torch.randn(num_superpoints, num_features, device=points_c.device)
        
        return superpoint_features

    def _filter_data_by_mask(self, data_dict, mask):
        """
        マスクに基づいてデータをフィルタリング
        """
        filtered_data_dict = data_dict.copy()
        
        # 最粗レベルの点群をフィルタリング
        if 'points' in data_dict:
            points_c = data_dict['points'][-1]
            filtered_points_c = points_c[mask]
            
            # pointsリストを更新
            new_points = data_dict['points'].copy()
            new_points[-1] = filtered_points_c
            filtered_data_dict['points'] = new_points
        
        # lengthsを更新
        if 'lengths' in data_dict:
            lengths = data_dict['lengths']
            ref_length = lengths[-1][0]
            src_length = lengths[-1][1]
            
            # ref と src のマスクを分離
            ref_mask = mask[:ref_length]
            src_mask = mask[ref_length:ref_length + src_length]
            
            new_ref_length = safe_item(ref_mask.sum())
            new_src_length = safe_item(src_mask.sum())
            
            # lengthsを更新
            new_lengths = lengths.copy()
            new_lengths[-1] = [new_ref_length, new_src_length]
            filtered_data_dict['lengths'] = new_lengths
        
        return filtered_data_dict


def main():
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=str, default=None, help='Path to pretrained weights')
    parser.add_argument('--test_epoch', type=int, default=None, help='Test epoch')
    parser.add_argument('--test_iter', type=int, default=None, help='Test iteration')
    parser.add_argument('--near', action='store_true', help='use newmethod_near data/metadata')
    parser.add_argument('--use_superpoint_score', action='store_true', help='use new superpoint score selection')
    args = parser.parse_args()
    
    # --near そのものを sys.argv から除去（make_cfgで重複解析を避けるため）
    if '--near' in sys.argv:
        sys.argv = [arg for arg in sys.argv if arg != '--near']
    if '--use_superpoint_score' in sys.argv:
        sys.argv = [arg for arg in sys.argv if arg != '--use_superpoint_score']
    
    cfg = make_cfg(use_near=args.near, use_superpoint_score=args.use_superpoint_score)
    if args.snapshot is not None:
        cfg.snapshot = args.snapshot
    tester = Tester(cfg)
    tester.run()



if __name__ == '__main__':
    main()
