import os
import torch
import torch.nn.functional as F
import numpy as np

from config import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from geotransformer.utils.superpoint_score import SuperPointScoreModule, normalize_features
from loss import OverallLoss, Evaluator
from geotransformer.modules.ops import point_to_node_partition

def safe_item(value):
    """安全に.item()を呼び出すヘルパー関数"""
    if hasattr(value, 'item'):
        return value.item()
    elif isinstance(value, (int, float)):
        return value
    else:
        try:
            return int(value)
        except:
            return float(value)

class SuperPointScoreTrainer:
    def __init__(self, cfg):
        """SuperPoint Score学習のための初期化"""
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # メインモデル（一部パラメータのみfreeze）
        self.model = create_model(cfg).to(self.device)
        # 特徴抽出部分のみfreeze、最終層は学習可能にする
        for name, param in self.model.named_parameters():
            if 'final' not in name and 'head' not in name:
                param.requires_grad = False
        self.model.train()  # 学習モードに設定
        
        # SuperPoint Score学習用の特徴量重み
        initial_weights = [
            1.0,   # 密度
            0.5,   # 強度分散
            0.8,   # 強度平均
            0.3,   # 強度勾配
        ]
        
        self.score_module = SuperPointScoreModule(
            num_features=len(initial_weights), 
            init_weights=initial_weights,
            use_nonlinear=False
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.score_module.parameters(), lr=1e-3)
        self.loss_func = OverallLoss(cfg).to(self.device)
        self.evaluator = Evaluator(cfg).to(self.device)
        self.train_loader, self.val_loader, _ = train_valid_data_loader(cfg, distributed=False)
        
        self.max_epoch = 30
        self.score_threshold = getattr(cfg, 'superpoint_score_threshold', 0.5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        # マスキング機構用のパラメータ
        self.temperature = 2.0  # ソフトマスキングの温度パラメータ
        self.training_threshold_ratio = 0.3  # 学習時は緩い閾値（70%を使用）
        self.inference_threshold_ratio = 0.7  # 推論時は厳しい閾値（30%を使用）

    def compute_superpoint_features(self, data_dict):
        """スーパーポイントの基本特徴量を計算"""
        device = self.device
        
        # データ準備 - 確実にdeviceに移動
        feats = data_dict['features'].to(device)
        points_list = data_dict['points']
        lengths = data_dict['lengths']
        
        # 最粗レベル（スーパーポイント）と細かいレベルの点群を取得
        ref_length_c = safe_item(lengths[-1][0])
        ref_length_f = safe_item(lengths[1][0])
        
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
        """単一点群のスーパーポイント特徴量を計算"""
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
            
            # マスクされた有効な点のみを使用
            valid_indices = knn_indices[knn_mask]
            
            if len(valid_indices) == 0:
                # 有効な点がない場合はゼロ特徴量
                features = torch.zeros(4, device=points_c.device)
                superpoint_features.append(features)
                continue
            
            # 対応する細かい点とその特徴量
            neighbor_points = points_f[valid_indices]
            neighbor_feats = feats_f[valid_indices]
            
            # 1. 密度（点の数）
            density = len(valid_indices) / num_points_in_patch
            
            # 2. 強度分散
            intensity_var = torch.var(neighbor_feats).clamp(min=1e-6)
            
            # 3. 強度平均
            intensity_mean = torch.mean(neighbor_feats)
            
            # 4. 強度勾配（空間的な変化）
            if len(valid_indices) > 1:
                # 空間距離と強度の関係から勾配を計算
                center_point = points_c[i]
                distances = torch.norm(neighbor_points - center_point, dim=1)
                # 距離と強度の相関を勾配として使用
                intensity_gradient = torch.abs(torch.corrcoef(torch.stack([distances, neighbor_feats.squeeze()]))[0, 1])
                if torch.isnan(intensity_gradient):
                    intensity_gradient = torch.tensor(0.0, device=points_c.device)
            else:
                intensity_gradient = torch.tensor(0.0, device=points_c.device)
            
            features = torch.stack([
                torch.tensor(density, device=points_c.device),
                intensity_var,
                intensity_mean,
                intensity_gradient
            ])
            
            superpoint_features.append(features)
        
        return torch.stack(superpoint_features)

    def train_epoch(self, epoch):
        """1エポック分のトレーニング"""
        self.score_module.train()
        
        total_loss_value = 0.0
        batch_count = 0
        max_batches = 100  # エラーが多発する場合のバッチ制限
        
        for batch_idx, data_dict in enumerate(self.train_loader):
            if batch_idx >= max_batches:
                print(f"Reached maximum batch limit ({max_batches})")
                break
                
            try:
                # データをdeviceに移動（より詳細な処理）
                data_dict = self._move_data_to_device(data_dict)
                
                # SuperPoint特徴量を計算
                superpoint_features, ref_features, src_features = self.compute_superpoint_features(data_dict)
                
                # 特徴量を確実にdeviceに移動
                superpoint_features = superpoint_features.to(self.device)
                
                # スコアを計算
                all_scores = self.score_module(superpoint_features)
                
                # 適応的閾値でソフトマスキング（方針D）
                masked_data_dict, soft_mask = self._apply_soft_masking(data_dict, all_scores, is_training=True)
                
                # メインモデルの推論（勾配を保持）
                output_dict = self.model(masked_data_dict)
                
                # 損失計算（選択されたスーパーポイントに対して）
                loss_dict = self.loss_func(output_dict, masked_data_dict)
                main_loss = loss_dict['loss']
                
                # スコア正規化損失を追加（スコアの分散を促進）
                score_reg_loss = self._compute_score_regularization_loss(all_scores)
                
                # 総損失
                total_loss = main_loss + 0.01 * score_reg_loss
                
                # バックプロパゲーション
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                total_loss_value += total_loss.item()
                batch_count += 1
                
                if batch_idx % 10 == 0:
                    mask_ratio = soft_mask.mean().item()
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.4f}, '
                          f'Main: {main_loss.item():.4f}, Reg: {score_reg_loss.item():.4f}, '
                          f'Mask Ratio: {mask_ratio:.3f}')
                    
            except Exception as e:
                print(f'Error in batch {batch_idx}: {str(e)}')
                # CUDAエラーの場合はクリーンアップ
                if 'cuda' in str(e).lower():
                    torch.cuda.empty_cache()
                continue
        
        avg_loss = total_loss_value / max(batch_count, 1)
        print(f'Epoch {epoch} finished. Average Loss: {avg_loss:.4f}')
        
        return avg_loss

    def _apply_soft_masking(self, data_dict, scores, is_training=True):
        """
        適応的閾値を使用したソフトマスキング（方針D）
        
        Args:
            data_dict: 入力データ
            scores: スーパーポイントスコア
            is_training: 学習時かどうか
            
        Returns:
            masked_data_dict: マスクされたデータ
            soft_mask: ソフトマスク値
        """
        # 適応的閾値の計算
        if is_training:
            # 学習時は緩い閾値（多くのスーパーポイントを残す）
            threshold_quantile = self.training_threshold_ratio
        else:
            # 推論時は厳しい閾値（重要なもののみ）
            threshold_quantile = self.inference_threshold_ratio
        
        adaptive_threshold = torch.quantile(scores, threshold_quantile)
        
        # ソフトマスクの計算（sigmoid を使用して微分可能）
        soft_mask = torch.sigmoid((scores - adaptive_threshold) / self.temperature)
        
        # 最低限のスーパーポイント数を保証
        min_points = max(int(len(scores) * 0.1), 5)  # 最低10%または5個
        if soft_mask.sum() < min_points:
            # 上位のスコアを強制的に選択
            _, top_indices = torch.topk(scores, min_points)
            soft_mask = torch.zeros_like(scores)
            soft_mask[top_indices] = 1.0
        
        # データにソフトマスクを適用
        masked_data_dict = self._apply_mask_to_data(data_dict, soft_mask)
        
        return masked_data_dict, soft_mask
    
    def _apply_mask_to_data(self, data_dict, soft_mask):
        """
        ソフトマスクをデータに適用
        
        Args:
            data_dict: 入力データ
            soft_mask: ソフトマスク値
            
        Returns:
            masked_data_dict: マスクされたデータ
        """
        masked_data_dict = {}
        
        for key, value in data_dict.items():
            if key == 'points' and isinstance(value, list):
                # points リストの処理（最粗レベルにマスクを適用）
                masked_points = []
                for i, points in enumerate(value):
                    if i == len(value) - 1:  # 最粗レベル（スーパーポイント）
                        points = points.to(self.device)
                        # ソフトマスクを座標に適用（重み付け）
                        # 完全に削除するのではなく、重みを下げる
                        mask_weights = soft_mask.unsqueeze(-1).expand_as(points)
                        masked_points_tensor = points * mask_weights
                        masked_points.append(masked_points_tensor)
                    else:
                        # 他のレベルはそのまま
                        masked_points.append(points.to(self.device))
                masked_data_dict[key] = masked_points
            elif key == 'lengths':
                # lengths の処理（最粗レベルを調整）
                masked_lengths = []
                for i, length in enumerate(value):
                    if i == len(value) - 1:  # 最粗レベル
                        # ソフトマスクの合計に基づいて長さを調整
                        ref_length = length[0]
                        src_length = length[1]
                        ref_mask = soft_mask[:ref_length]
                        src_mask = soft_mask[ref_length:ref_length + src_length]
                        
                        # 有効な長さを計算（ソフトマスクの合計）
                        effective_ref_length = max(int(ref_mask.sum().item()), 1)
                        effective_src_length = max(int(src_mask.sum().item()), 1)
                        
                        masked_lengths.append([effective_ref_length, effective_src_length])
                    else:
                        masked_lengths.append(length)
                masked_data_dict[key] = masked_lengths
            elif torch.is_tensor(value):
                # 通常のテンソルはdeviceに移動
                masked_data_dict[key] = value.to(self.device)
            elif isinstance(value, list):
                # リストの処理
                masked_list = []
                for item in value:
                    if torch.is_tensor(item):
                        masked_list.append(item.to(self.device))
                    else:
                        masked_list.append(item)
                masked_data_dict[key] = masked_list
            else:
                # その他はそのまま
                masked_data_dict[key] = value
        
        return masked_data_dict
    
    def _compute_score_regularization_loss(self, scores):
        """
        スコアの正規化損失を計算
        スコアの分散を促進し、全て同じ値にならないようにする
        
        Args:
            scores: スーパーポイントスコア
            
        Returns:
            regularization_loss: 正規化損失
        """
        # スコアの分散を促進
        score_variance = torch.var(scores)
        variance_loss = -score_variance  # 分散を大きくしたい
        
        # スコアの範囲を適切に保つ
        score_range = torch.max(scores) - torch.min(scores)
        range_loss = -score_range  # 範囲を大きくしたい
        
        # 平均を0.5付近に保つ
        score_mean = torch.mean(scores)
        mean_loss = torch.abs(score_mean - 0.5)
        
        regularization_loss = variance_loss + range_loss + mean_loss
        
        return regularization_loss

    def validate(self):
        """検証"""
        self.score_module.eval()
        
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, data_dict in enumerate(self.val_loader):
                if batch_idx >= 20:  # 検証は20バッチまで
                    break
                    
                try:
                    # データをdeviceに移動（より詳細な処理）
                    data_dict = self._move_data_to_device(data_dict)
                    
                    # SuperPoint特徴量を計算
                    superpoint_features, _, _ = self.compute_superpoint_features(data_dict)
                    
                    # 特徴量を確実にdeviceに移動
                    superpoint_features = superpoint_features.to(self.device)
                    
                    # スコアを計算
                    all_scores = self.score_module(superpoint_features)
                    
                    # 適応的閾値でソフトマスキング（検証時は推論設定）
                    masked_data_dict, soft_mask = self._apply_soft_masking(data_dict, all_scores, is_training=False)
                    
                    # メインモデルの推論
                    output_dict = self.model(masked_data_dict)
                    
                    # 損失計算
                    loss_dict = self.loss_func(output_dict, masked_data_dict)
                    loss = loss_dict['loss']
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    if batch_idx >= 10:  # 検証は少数のバッチで十分
                        break
                        
                except Exception as e:
                    print(f'Error in validation batch {batch_idx}: {str(e)}')
                    # CUDAエラーの場合はクリーンアップ
                    if 'cuda' in str(e).lower():
                        torch.cuda.empty_cache()
                    continue
        
        avg_loss = total_loss / max(batch_count, 1)
        print(f'Validation Loss: {avg_loss:.4f}')
        
        return avg_loss

    def train(self):
        """メインの学習ループ"""
        print("Starting SuperPoint Score Training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.max_epoch):
            print(f'\nEpoch {epoch + 1}/{self.max_epoch}')
            
            # 訓練
            train_loss = self.train_epoch(epoch)
            
            # 検証
            val_loss = self.validate()
            
            # 学習率更新
            self.scheduler.step()
            
            # ベストモデルの保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(epoch, val_loss)
                print(f'New best model saved at epoch {epoch + 1}')
        
        print("Training completed!")

    def save_model(self, epoch, val_loss):
        """学習済みモデルの保存"""
        checkpoint = {
            'epoch': epoch,
            'score_module_state_dict': self.score_module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'cfg': self.cfg
        }
        
        # 保存ディレクトリの作成
        save_dir = 'weights'
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存
        save_path = os.path.join(save_dir, 'superpoint_score_best.pth')
        torch.save(checkpoint, save_path)
        print(f'Model saved to {save_path}')

    def _move_data_to_device(self, data_dict):
        """データを安全にdeviceに移動"""
        moved_data_dict = {}
        
        for key, value in data_dict.items():
            if torch.is_tensor(value):
                # テンソルの場合はdeviceに移動
                moved_data_dict[key] = value.to(self.device)
            elif isinstance(value, list):
                # リストの場合は各要素をチェック
                moved_list = []
                for item in value:
                    if torch.is_tensor(item):
                        moved_list.append(item.to(self.device))
                    else:
                        moved_list.append(item)
                moved_data_dict[key] = moved_list
            else:
                # その他の場合はそのまま
                moved_data_dict[key] = value
        
        return moved_data_dict

def main():
    cfg = make_cfg()
    trainer = SuperPointScoreTrainer(cfg)
    trainer.train()

if __name__ == '__main__':
    main()
