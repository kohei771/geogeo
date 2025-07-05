import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from geotransformer.utils.superpoint_score import SuperPointScoreModule, normalize_features
from loss import OverallLoss, Evaluator
from geotransformer.modules.ops import point_to_node_partition

class ScoreWeightTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        # モデル本体はfreeze
        self.model = create_model(cfg).cuda()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        # スーパーポイントスコア重みのみ学習
        # 特徴量: [密度, 強度分散, 強度平均, 強度勾配]
        self.score_module = SuperPointScoreModule(num_features=4).cuda()
        self.optimizer = torch.optim.Adam(self.score_module.parameters(), lr=1e-3)
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()
        self.train_loader, self.val_loader, _ = train_valid_data_loader(cfg, distributed=False)
        self.max_epoch = 10  # スコア学習は短期間で十分
        self.max_batches = 50  # バッチ数を増やして安定化
        self.score_threshold = getattr(cfg, 'superpoint_score_threshold', 0.5)

    def compute_superpoint_features(self, data_dict):
        """
        スーパーポイントの特徴量を計算する
        Returns:
            superpoint_features: (N_superpoints, 4) [密度, 強度分散, 強度平均, 強度勾配]
        """
        # データ準備
        device = next(self.model.parameters()).device
        
        # 基本データの取得と前処理
        feats = data_dict['features'].to(device)
        points_list = data_dict['points']
        lengths = data_dict['lengths']
        
        # 最粗レベル（スーパーポイント）と細かいレベルの点群を取得
        ref_length_c = lengths[-1][0].item()
        ref_length_f = lengths[1][0].item()
        
        points_c = points_list[-1].to(device)  # 最粗レベル（スーパーポイント）
        points_f = points_list[1].to(device)   # 細かいレベル
        
        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        
        # 特徴量（intensity）の取得
        ref_feats_f = feats[:ref_length_f]
        src_feats_f = feats[ref_length_f:]
        
        # ref側のスーパーポイント特徴量計算
        ref_superpoint_features = self._compute_single_superpoint_features(
            ref_points_c, ref_points_f, ref_feats_f
        )
        
        # src側のスーパーポイント特徴量計算
        src_superpoint_features = self._compute_single_superpoint_features(
            src_points_c, src_points_f, src_feats_f
        )
        
        # ref と src を結合
        all_superpoint_features = torch.cat([ref_superpoint_features, src_superpoint_features], dim=0)
        
        return all_superpoint_features, ref_superpoint_features, src_superpoint_features
    
    def _compute_single_superpoint_features(self, points_c, points_f, feats_f):
        """
        単一点群のスーパーポイント特徴量を計算
        """
        num_points_in_patch = self.cfg.model.num_points_in_patch
        
        # point_to_node_partitionでスーパーポイントと細かい点の対応を取得
        _, node_masks, node_knn_indices, node_knn_masks = point_to_node_partition(
            points_f, points_c, num_points_in_patch
        )
        
        superpoint_features = []
        
        for i in range(points_c.shape[0]):
            # 各スーパーポイントに対応する細かい点のインデックス
            knn_indices = node_knn_indices[i]
            knn_mask = node_knn_masks[i]
            
            # 有効な点のみ選択
            valid_indices = knn_indices[knn_mask]
            valid_indices = torch.clamp(valid_indices, 0, feats_f.shape[0] - 1)
            
            if len(valid_indices) == 0:
                # 有効な点がない場合はゼロ特徴量
                features = torch.zeros(4, device=feats_f.device)
            else:
                # 対応する点の特徴量（intensity）
                patch_feats = feats_f[valid_indices]
                patch_points = points_f[valid_indices]
                
                # 1. 密度（パッチ内の点数）
                density = torch.tensor(len(valid_indices), dtype=torch.float32, device=feats_f.device)
                
                # 2. 強度分散
                intensity_var = torch.var(patch_feats[:, 0]) if patch_feats.shape[0] > 1 else torch.tensor(0.0, device=feats_f.device)
                
                # 3. 強度平均
                intensity_mean = torch.mean(patch_feats[:, 0])
                
                # 4. 強度勾配（最大値-最小値）
                intensity_grad = torch.max(patch_feats[:, 0]) - torch.min(patch_feats[:, 0])
                
                features = torch.stack([density, intensity_var, intensity_mean, intensity_grad])
            
            superpoint_features.append(features)
        
        return torch.stack(superpoint_features)  # (N_superpoints, 4)
    
    def apply_soft_weighting(self, data_dict, superpoint_scores):
        """
        スーパーポイントスコアに基づいてソフト重み付けを適用
        """
        device = next(self.model.parameters()).device
        lengths = data_dict['lengths']
        
        ref_length_c = lengths[-1][0].item()
        
        # スコアを0-1にnormalize
        normalized_scores = torch.sigmoid(superpoint_scores)
        
        # ref と src に分割
        ref_scores = normalized_scores[:ref_length_c]
        src_scores = normalized_scores[ref_length_c:]
        
        # 各スーパーポイントに対応する細かい点に重みを適用
        feats = data_dict['features'].to(device)
        points_list = data_dict['points']
        
        ref_length_f = lengths[1][0].item()
        points_c = points_list[-1].to(device)
        points_f = points_list[1].to(device)
        
        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        
        # ref側の重み適用
        ref_weighted_feats = self._apply_weights_to_feats(
            ref_points_c, ref_points_f, feats[:ref_length_f], ref_scores
        )
        
        # src側の重み適用
        src_weighted_feats = self._apply_weights_to_feats(
            src_points_c, src_points_f, feats[ref_length_f:], src_scores
        )
        
        # 重み付けした特徴量で置き換え
        data_dict['features'] = torch.cat([ref_weighted_feats, src_weighted_feats], dim=0)
        
        return data_dict
    
    def _apply_weights_to_feats(self, points_c, points_f, feats_f, scores):
        """
        スーパーポイントスコアを細かい点の特徴量に適用
        """
        num_points_in_patch = self.cfg.model.num_points_in_patch
        
        _, node_masks, node_knn_indices, node_knn_masks = point_to_node_partition(
            points_f, points_c, num_points_in_patch
        )
        
        weighted_feats = feats_f.clone()
        
        for i in range(points_c.shape[0]):
            knn_indices = node_knn_indices[i]
            knn_mask = node_knn_masks[i]
            
            valid_indices = knn_indices[knn_mask]
            valid_indices = torch.clamp(valid_indices, 0, feats_f.shape[0] - 1)
            
            if len(valid_indices) > 0:
                # スーパーポイントのスコアを対応する細かい点に適用
                weight = scores[i]
                weighted_feats[valid_indices] *= weight.unsqueeze(0)
        
        return weighted_feats

    def train(self):
        """
        スーパーポイントスコア重みの学習
        """
        print("Starting SuperPoint Score Training...")
        
        for epoch in range(self.max_epoch):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_count, data_dict in enumerate(self.train_loader):
                if batch_count >= self.max_batches:
                    break
                
                try:
                    # デバイスを統一
                    device = next(self.model.parameters()).device
                    
                    # バッチ内すべてのテンソルをmodelのデバイスへ
                    for k, v in data_dict.items():
                        if isinstance(v, torch.Tensor):
                            data_dict[k] = v.to(device)
                        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                            data_dict[k] = [x.to(device) for x in v]
                    
                    # 1. スーパーポイント特徴量の計算
                    all_superpoint_features, ref_features, src_features = self.compute_superpoint_features(data_dict)
                    
                    # 2. 特徴量の正規化
                    normalized_features = normalize_features(all_superpoint_features, method='minmax')
                    
                    # 3. スコア計算
                    superpoint_scores = self.score_module(normalized_features)
                    
                    # 4. ソフト重み付けの適用
                    weighted_data_dict = self.apply_soft_weighting(data_dict.copy(), superpoint_scores)
                    
                    # 5. メインモデルでの推論
                    with torch.no_grad():
                        output_dict = self.model(weighted_data_dict)
                    
                    # 6. 損失計算（スコアに対する勾配を計算するため、require_gradをTrueに）
                    # スコアの分散を最小化し、高スコア点を選好するような損失を追加
                    score_loss = self._compute_score_loss(superpoint_scores, output_dict, weighted_data_dict)
                    
                    # 7. 逆伝播
                    self.optimizer.zero_grad()
                    score_loss.backward()
                    
                    # 勾配クリッピング
                    torch.nn.utils.clip_grad_norm_(self.score_module.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    epoch_loss += score_loss.item()
                    n_batches += 1
                    
                    if batch_count % 10 == 0:
                        print(f"[Epoch {epoch+1}/{self.max_epoch}] Batch {batch_count+1}/{self.max_batches} "
                              f"Loss: {score_loss.item():.4f} "
                              f"Weights: {self.score_module.weights.data.cpu().numpy()}")
                
                except Exception as e:
                    print(f"Error in batch {batch_count}: {e}")
                    continue
            
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"\nEpoch {epoch+1}: avg_loss={avg_loss:.4f}")
            print(f"Current weights: {self.score_module.weights.data.cpu().numpy()}")
        
        # 重みの保存
        torch.save(self.score_module.state_dict(), "score_weights.pth")
        print("score_weights.pth saved.")
        print(f"Final weights: {self.score_module.weights.data.cpu().numpy()}")
    
    def _compute_score_loss(self, superpoint_scores, output_dict, data_dict):
        """
        スーパーポイントスコアに対する損失を計算
        """
        # 基本的な損失: スコアの分散を抑制（極端な値を避ける）
        score_variance_loss = torch.var(superpoint_scores) * 0.1
        
        # 高スコア点を選好する損失
        # 上位20%の点のスコアを高く、下位20%の点のスコアを低くする
        num_points = superpoint_scores.shape[0]
        top_k = max(1, num_points // 5)
        bottom_k = max(1, num_points // 5)
        
        # スコアの上位・下位を取得
        top_scores, _ = torch.topk(superpoint_scores, top_k, largest=True)
        bottom_scores, _ = torch.topk(superpoint_scores, bottom_k, largest=False)
        
        # 上位スコアを高く、下位スコアを低くする損失
        preference_loss = -torch.mean(top_scores) + torch.mean(bottom_scores)
        
        # マッチング品質に基づく損失（可能であれば）
        matching_quality_loss = torch.tensor(0.0, device=superpoint_scores.device)
        if 'corr_scores' in output_dict:
            corr_scores = output_dict['corr_scores']
            if corr_scores is not None and len(corr_scores) > 0:
                # 対応点の品質が高い場合は、それに関連するスーパーポイントのスコアを高くする
                matching_quality_loss = -torch.mean(corr_scores) * 0.5
        
        total_loss = score_variance_loss + preference_loss + matching_quality_loss
        
        return total_loss
    
    def evaluate_scores(self, data_dict):
        """
        学習したスコアを評価
        """
        self.score_module.eval()
        with torch.no_grad():
            all_superpoint_features, _, _ = self.compute_superpoint_features(data_dict)
            normalized_features = normalize_features(all_superpoint_features, method='minmax')
            scores = self.score_module(normalized_features)
            return scores
    
    def apply_hard_masking(self, data_dict, threshold=None):
        """
        推論時のhard maskingを適用
        """
        if threshold is None:
            threshold = self.score_threshold
        
        scores = self.evaluate_scores(data_dict)
        mask = scores > threshold
        
        # マスクを適用してスーパーポイントを選択
        # 実装はモデルの具体的な構造に依存
        return mask, scores

if __name__ == "__main__":
    cfg = make_cfg()
    trainer = ScoreWeightTrainer(cfg)
    
    # 学習実行
    trainer.train()
    
    # 簡単なテスト
    print("\n=== Testing Score Module ===")
    trainer.score_module.eval()
    
    # テストデータで評価
    test_count = 0
    for data_dict in trainer.val_loader:
        if test_count >= 3:  # 3バッチだけテスト
            break
        
        try:
            device = next(trainer.model.parameters()).device
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    data_dict[k] = v.to(device)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    data_dict[k] = [x.to(device) for x in v]
            
            # スコア評価
            scores = trainer.evaluate_scores(data_dict)
            print(f"Test batch {test_count+1}: Score range [{scores.min():.3f}, {scores.max():.3f}], "
                  f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")
            
            # Hard masking テスト
            mask, _ = trainer.apply_hard_masking(data_dict, threshold=0.5)
            print(f"Hard mask (threshold=0.5): {mask.sum().item()}/{len(mask)} points selected")
            
            test_count += 1
            
        except Exception as e:
            print(f"Error in test batch {test_count}: {e}")
            test_count += 1
            continue
    
    print("SuperPoint Score Training completed!")
