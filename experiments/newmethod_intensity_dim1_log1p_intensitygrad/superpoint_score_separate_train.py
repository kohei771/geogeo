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
        
        # メインモデルはfreeze
        self.model = create_model(cfg).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
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
        
        total_loss = 0.0
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
                
                # 上位スコアのスーパーポイントを選択
                top_k = max(int(len(all_scores) * self.score_threshold), 1)  # 最低1個は選択
                _, top_indices = torch.topk(all_scores, top_k)
                
                # 選択されたスーパーポイントでモデルを実行
                selected_data_dict = self._create_masked_data_dict(data_dict, top_indices)
                
                # メインモデルの推論
                with torch.no_grad():
                    output_dict = self.model(selected_data_dict)
                
                # 損失計算（選択されたスーパーポイントに対して）
                loss_dict = self.loss_func(output_dict, selected_data_dict)
                loss = loss_dict['loss']
                
                # 損失がテンソルでない場合の処理
                if not torch.is_tensor(loss):
                    loss = torch.tensor(loss, device=self.device, requires_grad=True)
                
                # バックプロパゲーション
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                    
            except Exception as e:
                print(f'Error in batch {batch_idx}: {str(e)}')
                # CUDAエラーの場合はクリーンアップ
                if 'cuda' in str(e).lower():
                    torch.cuda.empty_cache()
                continue
        
        avg_loss = total_loss / max(batch_count, 1)
        print(f'Epoch {epoch} finished. Average Loss: {avg_loss:.4f}')
        
        return avg_loss

    def _create_masked_data_dict(self, data_dict, top_indices):
        """選択されたスーパーポイントに対応するデータを作成"""
        # 現在の実装では簡略化のため、元のデータをそのまま返す
        # 実際の実装では、top_indicesに基づいてスーパーポイントをフィルタリングする
        
        # データが確実にdeviceに移動されていることを確認
        masked_data_dict = {}
        for key, value in data_dict.items():
            if torch.is_tensor(value):
                masked_data_dict[key] = value.to(self.device)
            elif isinstance(value, list):
                masked_list = []
                for item in value:
                    if torch.is_tensor(item):
                        masked_list.append(item.to(self.device))
                    else:
                        masked_list.append(item)
                masked_data_dict[key] = masked_list
            else:
                masked_data_dict[key] = value
        
        return masked_data_dict

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
                    
                    # 上位スコアのスーパーポイントを選択
                    top_k = max(int(len(all_scores) * self.score_threshold), 1)  # 最低1個は選択
                    _, top_indices = torch.topk(all_scores, top_k)
                    
                    # 選択されたスーパーポイントでモデルを実行
                    selected_data_dict = self._create_masked_data_dict(data_dict, top_indices)
                    
                    # メインモデルの推論
                    output_dict = self.model(selected_data_dict)
                    
                    # 損失計算
                    loss_dict = self.loss_func(output_dict, selected_data_dict)
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
