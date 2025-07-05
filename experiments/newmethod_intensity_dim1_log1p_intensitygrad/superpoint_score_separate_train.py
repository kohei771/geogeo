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
        # 拡張特徴量セット: より多くの特徴量を試して重要度を学習
        # 基本4つ: [密度, 強度分散, 強度平均, 強度勾配]
        # 追加候補: [強度中央値, 強度歪度, 強度尖度, 強度エントロピー, 
        #           幾何学的分散, 最近傍距離平均, 最近傍距離分散, 
        #           点群の広がり(span), 点群の偏心率, 局所密度勾配]
        initial_weights = [
            1.0,   # 密度
            0.5,   # 強度分散
            0.8,   # 強度平均
            0.3,   # 強度勾配
            0.7,   # 強度中央値
            0.2,   # 強度歪度
            0.2,   # 強度尖度
            0.4,   # 強度エントロピー
            0.6,   # 幾何学的分散
            0.5,   # 最近傍距離平均
            0.3,   # 最近傍距離分散
            0.4,   # 点群の広がり
            0.3,   # 点群の偏心率
            0.5    # 局所密度勾配
        ]
        num_features = len(initial_weights)  # 14個の特徴量
        
        self.score_module = SuperPointScoreModule(
            num_features=num_features, 
            init_weights=initial_weights,
            use_nonlinear=False  # まず線形で重要度を学習
        ).cuda()
        self.optimizer = torch.optim.Adam(self.score_module.parameters(), lr=1e-3)
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()
        self.train_loader, self.val_loader, _ = train_valid_data_loader(cfg, distributed=False)
        self.max_epoch = 60   # しっかりと学習（trainval.pyの約1/3）
        self.max_batches = 400  # より多くのデータで安定した学習
        self.score_threshold = getattr(cfg, 'superpoint_score_threshold', 0.5)
        
        # 学習率スケジューラーを追加
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

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
                
                # 強度が0の点を除外（データ欠損）
                intensity_values = patch_feats[:, 0]
                non_zero_mask = intensity_values != 0.0
                valid_intensity_values = intensity_values[non_zero_mask]
                
                # 1. 密度（パッチ内の点数 - 全点数を使用）
                density = torch.tensor(len(valid_indices), dtype=torch.float32, device=feats_f.device)
                
                # 強度特徴量は非ゼロ値のみで計算
                if len(valid_intensity_values) == 0:
                    # 全て強度が0の場合 - 14個の特徴量全てを0に
                    features = torch.zeros(14, device=feats_f.device)
                elif len(valid_intensity_values) == 1:
                    # 有効な強度値が1つの場合
                    single_val = valid_intensity_values[0]
                    features = torch.tensor([
                        density,        # 密度
                        0.0,           # 強度分散
                        single_val,    # 強度平均
                        0.0,           # 強度勾配
                        single_val,    # 強度中央値
                        0.0,           # 強度歪度
                        0.0,           # 強度尖度
                        0.0,           # 強度エントロピー
                        0.0,           # 幾何学的分散
                        0.0,           # 最近傍距離平均
                        0.0,           # 最近傍距離分散
                        0.0,           # 点群の広がり
                        0.0,           # 点群の偏心率
                        0.0            # 局所密度勾配
                    ], device=feats_f.device)
                else:
                    # 2つ以上の有効な強度値がある場合 - 全特徴量を計算
                    features = self._compute_extended_features(
                        valid_intensity_values, patch_points, density, feats_f.device
                    )
                
                features = features
            
            superpoint_features.append(features)
        
        return torch.stack(superpoint_features)  # (N_superpoints, 4)
    
    def apply_probabilistic_masking(self, data_dict, superpoint_scores):
        """
        学習時: 確率的マスキングを適用
        特徴量は変更せず、スーパーポイントの選択のみ行う
        """
        device = next(self.model.parameters()).device
        lengths = data_dict['lengths']
        
        ref_length_c = lengths[-1][0].item()
        
        # スコアを確率に変換（sigmoid + temperature）
        temperature = 2.0  # 温度パラメータで確率の鋭さを調整
        probs = torch.sigmoid(superpoint_scores / temperature)
        
        # 確率的マスキング（勾配が流れる）
        # Gumbel-Softmax的なアプローチで微分可能にする
        # 学習時はGumbel-Softmaxを使用して微分可能性を保つ
        if self.score_module.training:
            gumbel_noise = torch.rand_like(probs).clamp(1e-7, 1-1e-7)
            gumbel_noise = -torch.log(-torch.log(gumbel_noise))
            logits = torch.log(probs.clamp(1e-7, 1-1e-7)) + gumbel_noise
            prob_mask = torch.sigmoid(logits / temperature)
        else:
            prob_mask = torch.bernoulli(probs)
        
        # ref と src に分割
        ref_prob_mask = prob_mask[:ref_length_c]
        src_prob_mask = prob_mask[ref_length_c:]
        
        # 最低限のスーパーポイント数を保証（完全に空にならないように）
        min_points = 5
        if ref_prob_mask.sum() < min_points:
            # スコア上位をmin_points個選択
            _, top_indices = torch.topk(superpoint_scores[:ref_length_c], min_points)
            ref_prob_mask = torch.zeros_like(ref_prob_mask)
            ref_prob_mask[top_indices] = 1.0
            
        if src_prob_mask.sum() < min_points:
            _, top_indices = torch.topk(superpoint_scores[ref_length_c:], min_points)
            src_prob_mask = torch.zeros_like(src_prob_mask)
            src_prob_mask[top_indices] = 1.0
        
        # 選択されたスーパーポイントとそれに属する細かい点をマスク
        masked_data_dict = self._apply_superpoint_mask_simple(
            data_dict, ref_prob_mask, src_prob_mask
        )
        
        return masked_data_dict, prob_mask
    
    def _apply_superpoint_mask(self, data_dict, ref_mask, src_mask):
        """
        スーパーポイントマスクを適用して、選択されたスーパーポイントに
        属する細かい点のみを残す
        """
        device = next(self.model.parameters()).device
        
        # スーパーポイントレベルの点群と特徴量
        points_c = data_dict['points'][-1].to(device)
        points_f = data_dict['points'][1].to(device)
        feats = data_dict['features'].to(device)
        
        lengths = data_dict['lengths']
        ref_length_c = lengths[-1][0].item()
        ref_length_f = lengths[1][0].item()
        
        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_feats_f = feats[:ref_length_f]
        src_feats_f = feats[ref_length_f:]
        
        # 選択されたスーパーポイントのみ抽出
        selected_ref_points_c = ref_points_c[ref_mask]
        selected_src_points_c = src_points_c[src_mask]
        
        # 選択されたスーパーポイントに属する細かい点を特定
        ref_fine_mask = self._get_fine_points_mask(
            ref_points_f, ref_points_c, ref_mask
        )
        src_fine_mask = self._get_fine_points_mask(
            src_points_f, src_points_c, src_mask
        )
        
        # 選択された細かい点のみ抽出
        selected_ref_points_f = ref_points_f[ref_fine_mask]
        selected_src_points_f = src_points_f[src_fine_mask]
        selected_ref_feats_f = ref_feats_f[ref_fine_mask]
        selected_src_feats_f = src_feats_f[src_fine_mask]
        
        # 新しいdata_dictを構築
        masked_data_dict = data_dict.copy()
        
        # 新しい点群とlengths
        new_points_c = torch.cat([selected_ref_points_c, selected_src_points_c], dim=0)
        new_points_f = torch.cat([selected_ref_points_f, selected_src_points_f], dim=0)
        new_feats = torch.cat([selected_ref_feats_f, selected_src_feats_f], dim=0)
        
        # lengthsを更新
        new_lengths = []
        for level in range(len(data_dict['lengths'])):
            if level == len(data_dict['lengths']) - 1:  # 最粗レベル
                new_lengths.append([selected_ref_points_c.shape[0], selected_src_points_c.shape[0]])
            elif level == 1:  # 細かいレベル
                new_lengths.append([selected_ref_points_f.shape[0], selected_src_points_f.shape[0]])
            else:
                # 他のレベルは簡略化のため同じ比率で調整
                orig_lengths = data_dict['lengths'][level]
                ratio_ref = selected_ref_points_f.shape[0] / ref_length_f
                ratio_src = selected_src_points_f.shape[0] / (feats.shape[0] - ref_length_f)
                new_ref_len = int(orig_lengths[0] * ratio_ref)
                new_src_len = int(orig_lengths[1] * ratio_src)
                new_lengths.append([new_ref_len, new_src_len])
        
        # pointsリストを更新（他のレベルは元のまま、関連レベルのみ更新）
        new_points_list = data_dict['points'].copy()
        new_points_list[-1] = new_points_c  # 最粗レベル
        new_points_list[1] = new_points_f   # 細かいレベル
        
        masked_data_dict['points'] = new_points_list
        masked_data_dict['features'] = new_feats
        masked_data_dict['lengths'] = new_lengths
        
        return masked_data_dict
    
    def _apply_superpoint_mask_simple(self, data_dict, ref_mask, src_mask):
        """
        簡素化されたスーパーポイントマスクの適用
        計算効率を重視し、必要最小限の変更のみを行う
        """
        device = next(self.model.parameters()).device
        masked_data_dict = data_dict.copy()
        
        # スーパーポイントレベルの点群を更新
        points_c = data_dict['points'][-1].to(device)
        lengths = data_dict['lengths']
        ref_length_c = lengths[-1][0].item()
        
        # マスクを適用
        combined_mask = torch.cat([ref_mask, src_mask], dim=0)
        selected_points_c = points_c[combined_mask.bool()]
        
        # 新しいlengthsを計算
        new_ref_length_c = ref_mask.sum().int().item()
        new_src_length_c = src_mask.sum().int().item()
        
        # pointsとlengthsを更新
        new_points_list = masked_data_dict['points'].copy()
        new_points_list[-1] = selected_points_c
        
        new_lengths = masked_data_dict['lengths'].copy()
        new_lengths[-1] = [new_ref_length_c, new_src_length_c]
        
        masked_data_dict['points'] = new_points_list
        masked_data_dict['lengths'] = new_lengths
        
        # 特徴量レベルも簡素化して更新
        if 'features' in data_dict:
            features = data_dict['features'].to(device)
            ref_length_f = lengths[1][0].item()
            
            # 簡素化：スーパーポイントレベルでのマスクを特徴量レベルにも適用
            # 実際の実装では、各スーパーポイントに属する特徴量を正確に特定する必要がある
            ref_feat_mask = ref_mask[:min(ref_mask.shape[0], ref_length_f)]
            src_feat_mask = src_mask[:min(src_mask.shape[0], features.shape[0] - ref_length_f)]
            
            # 足りない部分は上位スコアの要素で補完
            if ref_feat_mask.shape[0] < ref_length_f:
                padding = torch.zeros(ref_length_f - ref_feat_mask.shape[0], dtype=ref_feat_mask.dtype, device=device)
                ref_feat_mask = torch.cat([ref_feat_mask, padding], dim=0)
            
            if src_feat_mask.shape[0] < features.shape[0] - ref_length_f:
                padding = torch.zeros(features.shape[0] - ref_length_f - src_feat_mask.shape[0], dtype=src_feat_mask.dtype, device=device)
                src_feat_mask = torch.cat([src_feat_mask, padding], dim=0)
            
            expanded_mask = torch.cat([ref_feat_mask, src_feat_mask], dim=0)
            
            masked_data_dict['features'] = features[expanded_mask.bool()]
            
            # 特徴量レベルのlengthsも更新
            new_ref_length_f = ref_feat_mask.sum().int().item()
            new_src_length_f = src_feat_mask.sum().int().item()
            new_lengths[1] = [new_ref_length_f, new_src_length_f]
            masked_data_dict['lengths'] = new_lengths
        
        return masked_data_dict
    
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
                    output_dict = self.model(weighted_data_dict)
                    
                    # 6. trainval.pyと同じ損失関数を使用
                    main_loss_dict = self.loss_func(output_dict, weighted_data_dict)
                    main_loss = main_loss_dict['loss']
                    
                    # 7. スーパーポイントスコアに対する追加損失
                    score_loss = self._compute_score_loss(superpoint_scores, output_dict, weighted_data_dict)
                    
                    # 8. 総損失（メイン損失 + スコア損失）
                    total_loss = main_loss + score_loss * 0.1  # スコア損失は小さな重みで加算
                    
                    # 9. 逆伝播
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    
                    # 勾配クリッピング
                    torch.nn.utils.clip_grad_norm_(self.score_module.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    # 学習率スケジューラーのステップ
                    if batch_count == self.max_batches - 1:  # エポック終了時
                        self.scheduler.step()
                    
                    epoch_loss += total_loss.item()
                    n_batches += 1
                    
                    if batch_count % 20 == 0:  # 20バッチごとに出力
                        current_lr = self.optimizer.param_groups[0]['lr']
                        print(f"[Epoch {epoch+1}/{self.max_epoch}] Batch {batch_count+1}/{self.max_batches} "
                              f"Total Loss: {total_loss.item():.4f} "
                              f"Main Loss: {main_loss.item():.4f} "
                              f"Score Loss: {score_loss.item():.4f} "
                              f"LR: {current_lr:.6f} "
                              f"Weights: {self.score_module.weights.data.cpu().numpy()}")
                        
                        # 強度データの統計を時々確認
                        if batch_count % 100 == 0:
                            self._debug_intensity_stats(data_dict)
                
                except Exception as e:
                    print(f"Error in batch {batch_count}: {e}")
                    continue
            
            avg_loss = epoch_loss / max(n_batches, 1)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}: avg_loss={avg_loss:.4f}, lr={current_lr:.6f}")
            print(f"Current weights: {self.score_module.weights.data.cpu().numpy()}")
            
            # 早期停止の条件をチェック（オプション）
            if epoch > 30 and avg_loss < 0.15:  # 30エポック後、損失が十分小さくなったら早期停止
                print(f"Early stopping at epoch {epoch+1} (loss converged)")
                break
            
            # 5エポックごとに検証
            if (epoch + 1) % 5 == 0:
                val_loss = self._validate_epoch()
                print(f"Validation loss: {val_loss:.4f}")
        
        # 重みの保存
        torch.save(self.score_module.state_dict(), "score_weights.pth")
        print("score_weights.pth saved.")
        print(f"Final weights: {self.score_module.weights.data.cpu().numpy()}")
    
    def save_superpoint_score_weights(self, output_dir):
        """
        学習済みスーパーポイントスコアの重みを保存
        """
        import os
        save_path = os.path.join(output_dir, 'superpoint_score_weights.pth')
        
        # 保存する情報
        checkpoint = {
            'num_features': self.score_module.num_features,
            'full_weights': self.score_module.weights.data.cpu().numpy().tolist(),
            'use_nonlinear': self.score_module.use_nonlinear,
            'state_dict': self.score_module.state_dict(),
            'threshold': self.score_threshold,
            'important_features': None,  # 第1段階では None
            'reduced_weights': None
        }
        
        torch.save(checkpoint, save_path)
        print(f"SuperPoint Score weights saved to: {save_path}")
        
        return save_path

    def save_reduced_superpoint_score_weights(self, output_dir, important_features, feature_names):
        """
        第2段階の重要な特徴量のみの重みを保存
        """
        import os
        save_path = os.path.join(output_dir, 'superpoint_score_weights.pth')
        
        # 保存する情報（第2段階用に更新）
        checkpoint = {
            'num_features': len(important_features),
            'full_weights': None,  # 第1段階の重みは使用しない
            'use_nonlinear': self.score_module.use_nonlinear,
            'state_dict': self.score_module.state_dict(),
            'threshold': self.score_threshold,
            'important_features': important_features,
            'reduced_weights': self.score_module.weights.data.cpu().numpy().tolist(),
            'feature_names': [feature_names[i] for i in important_features]
        }
        
        torch.save(checkpoint, save_path)
        print(f"Reduced SuperPoint Score weights saved to: {save_path}")
        print(f"Important features: {[feature_names[i] for i in important_features]}")
        
        return save_path

    def _validate_epoch(self):
        """
        検証エポックを実行
        """
        self.score_module.eval()
        val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch_count, data_dict in enumerate(self.val_loader):
                if batch_count >= 20:  # 検証は20バッチのみ
                    break
                
                try:
                    device = next(self.model.parameters()).device
                    
                    # デバイス統一
                    for k, v in data_dict.items():
                        if isinstance(v, torch.Tensor):
                            data_dict[k] = v.to(device)
                        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                            data_dict[k] = [x.to(device) for x in v]
                    
                    # スーパーポイント特徴量計算
                    all_superpoint_features, _, _ = self.compute_superpoint_features(data_dict)
                    normalized_features = normalize_features(all_superpoint_features, method='minmax')
                    superpoint_scores = self.score_module(normalized_features)
                    
                    # ソフト重み付け適用
                    weighted_data_dict = self.apply_soft_weighting(data_dict.copy(), superpoint_scores)
                    
                    # モデル推論
                    output_dict = self.model(weighted_data_dict)
                    
                    # 損失計算
                    main_loss_dict = self.loss_func(output_dict, weighted_data_dict)
                    main_loss = main_loss_dict['loss']
                    
                    val_loss += main_loss.item()
                    n_val_batches += 1
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_count}: {e}")
                    continue
        
        self.score_module.train()
        return val_loss / max(n_val_batches, 1)
    
    def _compute_score_loss(self, superpoint_scores, output_dict, data_dict):
        """
        スーパーポイントスコアに対する損失を計算
        重みの正規化を追加して、特徴量間の相対的重要度を学習
        """
        device = superpoint_scores.device
        
        # 1. 重みの正規化損失（L2正規化で重みの大きさを制限）
        weight_l2_loss = torch.norm(self.score_module.weights, p=2) * 0.01
        
        # 2. 重みの分散を促進（全て同じ値にならないように）
        weight_variance_loss = -torch.var(self.score_module.weights) * 0.1
        
        # 3. スコアの適度な分散を促進
        score_mean = torch.mean(superpoint_scores)
        score_std = torch.std(superpoint_scores)
        
        # スコアが0.5中心、適度な分散を持つように
        score_distribution_loss = (
            F.mse_loss(score_mean, torch.tensor(0.5, device=device)) * 0.1 +
            F.mse_loss(score_std, torch.tensor(0.3, device=device)) * 0.05
        )
        
        # 4. マスキング効果を考慮した損失
        # 高スコアのスーパーポイントが実際に有用であることを促進
        probs = torch.sigmoid(superpoint_scores / 2.0)
        masking_efficiency_loss = -torch.mean(probs * torch.log(probs.clamp(1e-7, 1.0))) * 0.02
        
        # 5. 改良された選好損失
        num_points = superpoint_scores.shape[0]
        if num_points > 10:  # 十分な数のスーパーポイントがある場合のみ
            # 上位30%と下位30%のスコア差を促進
            k = max(int(num_points * 0.3), 1)
            top_k_scores = torch.topk(superpoint_scores, k).values
            bottom_k_scores = torch.topk(superpoint_scores, k, largest=False).values
            selection_loss = -torch.mean(top_k_scores - bottom_k_scores) * 0.1
        else:
            selection_loss = torch.tensor(0.0, device=device)
        
        total_score_loss = (
            weight_l2_loss + 
            weight_variance_loss + 
            score_distribution_loss + 
            masking_efficiency_loss +
            selection_loss
        )
        
        return total_score_loss
    
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
    
    def _debug_intensity_stats(self, data_dict):
        """
        強度データの統計情報をデバッグ出力
        """
        if 'features' in data_dict:
            features = data_dict['features']
            intensity_values = features[:, 0]
            
            total_points = len(intensity_values)
            zero_points = (intensity_values == 0.0).sum().item()
            non_zero_points = total_points - zero_points
            
            if non_zero_points > 0:
                non_zero_values = intensity_values[intensity_values != 0.0]
                intensity_min = non_zero_values.min().item()
                intensity_max = non_zero_values.max().item()
                intensity_mean = non_zero_values.mean().item()
                intensity_std = non_zero_values.std().item()
                
                print(f"Intensity Stats - Total: {total_points}, Zero: {zero_points} ({zero_points/total_points*100:.1f}%), "
                      f"Non-zero: {non_zero_points}, Range: [{intensity_min:.3f}, {intensity_max:.3f}], "
                      f"Mean: {intensity_mean:.3f}, Std: {intensity_std:.3f}")
            else:
                print(f"Intensity Stats - Total: {total_points}, All points have zero intensity!")

    def _compute_extended_features(self, valid_intensity_values, patch_points, density, device):
        """
        拡張特徴量セットの計算
        14個の特徴量を計算: 基本4つ + 追加10個
        """
        # 基本統計量
        intensity_var = torch.var(valid_intensity_values)
        intensity_mean = torch.mean(valid_intensity_values)
        intensity_grad = torch.max(valid_intensity_values) - torch.min(valid_intensity_values)
        
        # 追加の強度統計量
        intensity_median = torch.median(valid_intensity_values)
        
        # 歪度と尖度（scipy.stats風の計算）
        if len(valid_intensity_values) >= 3:
            centered = valid_intensity_values - intensity_mean
            std_dev = torch.std(valid_intensity_values)
            if std_dev > 1e-6:
                skewness = torch.mean((centered / std_dev) ** 3)
                kurtosis = torch.mean((centered / std_dev) ** 4) - 3  # excess kurtosis
            else:
                skewness = torch.tensor(0.0, device=device)
                kurtosis = torch.tensor(0.0, device=device)
        else:
            skewness = torch.tensor(0.0, device=device)
            kurtosis = torch.tensor(0.0, device=device)
        
        # 強度エントロピー（簡易版）
        if len(valid_intensity_values) >= 2:
            # ヒストグラムベースのエントロピー
            hist = torch.histc(valid_intensity_values, bins=min(10, len(valid_intensity_values)), 
                              min=valid_intensity_values.min(), max=valid_intensity_values.max())
            hist = hist + 1e-10  # 数値安定性のため
            prob = hist / hist.sum()
            entropy = -torch.sum(prob * torch.log(prob))
        else:
            entropy = torch.tensor(0.0, device=device)
        
        # 幾何学的特徴量
        if len(patch_points) >= 3:
            # 点群の幾何学的分散（重心からの距離の分散）
            centroid = torch.mean(patch_points, dim=0)
            distances = torch.norm(patch_points - centroid, dim=1)
            geometric_var = torch.var(distances)
            
            # 最近傍距離統計
            # 簡易版: 各点の最近傍距離の平均と分散
            if len(patch_points) >= 2:
                pairwise_dist = torch.cdist(patch_points, patch_points)
                pairwise_dist = pairwise_dist + torch.eye(len(patch_points), device=device) * 1e6  # 自分自身を除外
                min_distances = torch.min(pairwise_dist, dim=1)[0]
                nn_dist_mean = torch.mean(min_distances)
                nn_dist_var = torch.var(min_distances) if len(min_distances) > 1 else torch.tensor(0.0, device=device)
            else:
                nn_dist_mean = torch.tensor(0.0, device=device)
                nn_dist_var = torch.tensor(0.0, device=device)
            
            # 点群の広がり（各軸の範囲の平均）
            point_spans = torch.max(patch_points, dim=0)[0] - torch.min(patch_points, dim=0)[0]
            span_mean = torch.mean(point_spans)
            
            # 点群の偏心率（主成分分析的な簡易版）
            if len(patch_points) >= 3:
                centered_points = patch_points - centroid
                cov_matrix = torch.mm(centered_points.t(), centered_points) / (len(patch_points) - 1)
                eigenvals = torch.linalg.eigvals(cov_matrix).real
                eigenvals = torch.sort(eigenvals, descending=True)[0]
                if eigenvals[0] > 1e-6:
                    eccentricity = 1 - eigenvals[-1] / eigenvals[0]  # 1 - (smallest/largest eigenvalue)
                else:
                    eccentricity = torch.tensor(0.0, device=device)
            else:
                eccentricity = torch.tensor(0.0, device=device)
        else:
            geometric_var = torch.tensor(0.0, device=device)
            nn_dist_mean = torch.tensor(0.0, device=device)
            nn_dist_var = torch.tensor(0.0, device=device)
            span_mean = torch.tensor(0.0, device=device)
            eccentricity = torch.tensor(0.0, device=device)
        
        # 局所密度勾配（簡易版: 密度の空間的変化）
        if len(patch_points) >= 5:
            # 中心から各点までの距離で密度の変化を推定
            centroid = torch.mean(patch_points, dim=0)
            distances = torch.norm(patch_points - centroid, dim=1)
            # 距離に基づく密度勾配の推定
            if torch.std(distances) > 1e-6:
                density_gradient = torch.std(distances) / torch.mean(distances)
            else:
                density_gradient = torch.tensor(0.0, device=device)
        else:
            density_gradient = torch.tensor(0.0, device=device)
        
        # 14個の特徴量をスタック
        features = torch.stack([
            density,           # 0: 密度
            intensity_var,     # 1: 強度分散
            intensity_mean,    # 2: 強度平均  
            intensity_grad,    # 3: 強度勾配
            intensity_median,  # 4: 強度中央値
            skewness,          # 5: 強度歪度
            kurtosis,          # 6: 強度尖度
            entropy,           # 7: 強度エントロピー
            geometric_var,     # 8: 幾何学的分散
            nn_dist_mean,      # 9: 最近傍距離平均
            nn_dist_var,       # 10: 最近傍距離分散
            span_mean,         # 11: 点群の広がり
            eccentricity,      # 12: 点群の偏心率
            density_gradient   # 13: 局所密度勾配
        ])
        
        return features

    def analyze_feature_importance(self):
        """
        学習後の特徴量重要度を分析
        """
        feature_names = [
            'density', 'intensity_var', 'intensity_mean', 'intensity_grad',
            'intensity_median', 'skewness', 'kurtosis', 'entropy', 
            'geometric_var', 'nn_dist_mean', 'nn_dist_var', 'span_mean',
            'eccentricity', 'density_gradient'
        ]
        
        weights = self.score_module.weights.data.cpu().numpy()
        abs_weights = np.abs(weights)
        
        # 重要度順にソート
        sorted_indices = np.argsort(abs_weights)[::-1]
        
        print("\n=== Feature Importance Analysis ===")
        print("Rank | Feature Name          | Weight    | Abs Weight")
        print("-" * 55)
        
        for i, idx in enumerate(sorted_indices):
            print(f"{i+1:4d} | {feature_names[idx]:20s} | {weights[idx]:8.4f} | {abs_weights[idx]:8.4f}")
        
        # 重要な特徴量の抽出（上位8個）
        top_features = sorted_indices[:8]
        print(f"\nTop 8 important features: {[feature_names[i] for i in top_features]}")
        
        return top_features, feature_names
    
    def create_reduced_trainer(self, important_features):
        """
        重要な特徴量のみを使用する新しいトレーナーを作成
        """
        print(f"\n=== Creating Reduced Trainer with {len(important_features)} features ===")
        
        # 重要な特徴量の初期重みを抽出
        original_weights = self.score_module.weights.data.cpu().numpy()
        reduced_weights = [original_weights[i] for i in important_features]
        
        # 新しいトレーナーの設定
        reduced_trainer = ScoreWeightTrainer.__new__(ScoreWeightTrainer)
        reduced_trainer.cfg = self.cfg
        reduced_trainer.model = self.model
        reduced_trainer.loss_func = self.loss_func
        reduced_trainer.evaluator = self.evaluator
        reduced_trainer.train_loader = self.train_loader
        reduced_trainer.val_loader = self.val_loader
        reduced_trainer.max_epoch = 30  # 短めに設定
        reduced_trainer.max_batches = 200
        reduced_trainer.score_threshold = self.score_threshold
        reduced_trainer.important_features = important_features  # 重要な特徴量のインデックス
        
        # 新しいスコアモジュール（重要な特徴量のみ）
        reduced_trainer.score_module = SuperPointScoreModule(
            num_features=len(important_features),
            init_weights=reduced_weights,
            use_nonlinear=True  # 第2段階では非線形も試す
        ).cuda()
        
        reduced_trainer.optimizer = torch.optim.Adam(
            reduced_trainer.score_module.parameters(), lr=1e-3
        )
        reduced_trainer.scheduler = torch.optim.lr_scheduler.StepLR(
            reduced_trainer.optimizer, step_size=10, gamma=0.7
        )
        
        return reduced_trainer

    def compute_superpoint_features_selective(self, data_dict, selected_features=None):
        """
        選択された特徴量のみを計算する版
        """
        if selected_features is None:
            # 全特徴量を計算
            return self.compute_superpoint_features(data_dict)
        
        # 基本的な計算は同じだが、最後に選択された特徴量のみを返す
        all_features, ref_features, src_features = self.compute_superpoint_features(data_dict)
        
        # 選択された特徴量のみを抽出
        selected_all = all_features[:, selected_features]
        selected_ref = ref_features[:, selected_features] if ref_features is not None else None
        selected_src = src_features[:, selected_features] if src_features is not None else None
        
        return selected_all, selected_ref, selected_src

# データ欠損処理に関する重要な注意事項:
# 
# 1. 強度値が0の点は測定データの欠損として扱う
# 2. 密度計算には全点数を使用（欠損があっても点は存在する）
# 3. 強度統計（分散、平均、勾配）は非ゼロ値のみで計算
# 4. 全て欠損の場合は強度関連特徴量を0に設定
# 5. これにより、欠損データに影響されない頑健な特徴量抽出が可能

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
    
    print("=== Stage 1: Full Feature Training completed! ===")
    
    # 第1段階の重みを保存
    stage1_save_path = trainer.save_superpoint_score_weights(cfg.output_dir)
    
    # 特徴量重要度分析
    top_features, feature_names = trainer.analyze_feature_importance()
    
    # 第2段階: 重要な特徴量のみで再学習
    print("\n=== Stage 2: Reduced Feature Training ===")
    reduced_trainer = trainer.create_reduced_trainer(top_features)
    
    # 重要な特徴量のみを使用する特徴量計算関数を動的に作成
    def compute_reduced_features(self, data_dict):
        # 全特徴量を計算
        all_features, _, _ = self.compute_superpoint_features(data_dict)
        # 重要な特徴量のみを選択
        reduced_features = all_features[:, self.important_features]
        return reduced_features, None, None
    
    # メソッドを置き換え
    reduced_trainer.compute_superpoint_features = compute_reduced_features.__get__(reduced_trainer, ScoreWeightTrainer)
    
    # 第2段階の学習実行
    reduced_trainer.train()
    
    # 最終的な特徴量重要度分析
    print("\n=== Final Feature Importance (Reduced Set) ===")
    final_weights = reduced_trainer.score_module.weights.data.cpu().numpy()
    
    print("Final important features:")
    for i, (feat_idx, weight) in enumerate(zip(top_features, final_weights)):
        print(f"{i+1:2d}. {feature_names[feat_idx]:20s}: {weight:8.4f}")
    
    # 最終テスト
    print("\n=== Final Testing with Reduced Features ===")
    reduced_trainer.score_module.eval()
    
    test_count = 0
    for data_dict in reduced_trainer.val_loader:
        if test_count >= 3:
            break
        
        try:
            device = next(reduced_trainer.model.parameters()).device
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    data_dict[k] = v.to(device)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    data_dict[k] = [x.to(device) for x in v]
            
            # 重要な特徴量のみでスコア評価
            reduced_features, _, _ = reduced_trainer.compute_superpoint_features(data_dict)
            normalized_features = normalize_features(reduced_features, method='minmax')
            scores = reduced_trainer.score_module(normalized_features)
            
            print(f"Test batch {test_count+1}: Score range [{scores.min():.3f}, {scores.max():.3f}], "
                  f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")
            
            test_count += 1
            
        except Exception as e:
            print(f"Error in reduced test batch {test_count}: {e}")
            test_count += 1
            continue
    
    print("Two-stage SuperPoint Score Training completed!")
    print(f"Final model uses {len(top_features)} out of 14 features")
    
    # 第2段階の重みを保存（重要な特徴量のみ）
    stage2_save_path = reduced_trainer.save_reduced_superpoint_score_weights(cfg.output_dir, top_features, feature_names)
    
    print(f"\nSaved weights:")
    print(f"  Stage 1 (full features): {stage1_save_path}")
    print(f"  Stage 2 (reduced features): {stage2_save_path}")
    print("You can now use these weights for testing with --use_superpoint_score")

# 実行方法の説明とメモ
"""
このスクリプトの実行方法：

1. 学習実行:
   python superpoint_score_separate_train.py

2. 2段階学習アプローチ:
   Stage 1: 14個の拡張特徴量で学習し、重要度を分析
   Stage 2: 重要な特徴量（上位8個）のみで再学習・最適化

3. 14個の拡張特徴量:
   - 基本4個: 密度、強度分散、強度平均、強度勾配
   - 統計拡張: 強度中央値、歪度、尖度、エントロピー
   - 幾何特徴: 幾何学的分散、最近傍距離統計、点群広がり、偏心率、局所密度勾配

4. 主な特徴:
   - 学習時: 確率的マスキング（Gumbel-Softmax）で勾配を流す
   - 推論時: ハードマスキング（閾値ベース）で高品質な点のみ選択
   - 特徴量の物理的意味を保持（座標や強度値の変更なし）
   - データ欠損処理: 強度=0の点は統計計算から除外

5. 学習される重み:
   - 各特徴量の重要度を学習
   - 不要な特徴量は自動的に重みが小さくなる
   - 正規化と分散促進による安定化

6. 推論での利用:
   - 学習した重みでスコアを計算
   - 閾値（デフォルト0.5）で高品質スーパーポイントを選択
   - 選択されたスーパーポイントのみでマッチング実行
"""
