import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_superpoint_score(features, weights):
    """
    features: (N, F) torch.Tensor, 各スーパーポイントの特徴量（例: [密度, 分散, 平均, 勾配]）
    weights: (F,) torch.nn.Parameter, 学習可能な重みベクトル
    return: (N,) torch.Tensor, 各スーパーポイントのスコア
    """
    return (features * weights).sum(dim=1)

class SuperPointScoreModule(nn.Module):
    """
    スーパーポイント情報価値スコア計算用モジュール。
    特徴量を受け取り、学習可能な重みで線形結合してスコアを計算。
    """
    def __init__(self, num_features, init_weights=None, use_nonlinear=True):
        super().__init__()
        
        self.num_features = num_features
        self.use_nonlinear = use_nonlinear
        
        # 初期重みの設定
        if init_weights is not None:
            self.weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))
        else:
            # 各特徴量に対して均等な重みで初期化
            self.weights = nn.Parameter(torch.ones(num_features, dtype=torch.float32))
        
        # 非線形変換を使用する場合
        if self.use_nonlinear:
            self.mlp = nn.Sequential(
                nn.Linear(num_features, num_features * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(num_features * 2, num_features),
                nn.ReLU(),
                nn.Linear(num_features, 1)
            )
        
        # 重み初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重みの初期化"""
        if self.use_nonlinear:
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        features: (N, F) torch.Tensor
        return: (N,) torch.Tensor, スコア
        """
        if self.use_nonlinear:
            # 非線形変換を使用
            weighted_features = features * self.weights  # 重み付き特徴量
            scores = self.mlp(weighted_features).squeeze(-1)  # MLPでスコア計算
        else:
            # 線形結合のみ
            scores = compute_superpoint_score(features, self.weights)
        
        return scores
    
    def get_feature_importance(self):
        """特徴量の重要度を取得"""
        if self.use_nonlinear:
            # 非線形の場合は重みベクトルを返す
            return self.weights.data.cpu().numpy()
        else:
            # 線形の場合は正規化した重みを返す
            weights = self.weights.data.cpu().numpy()
            return weights / (np.sum(np.abs(weights)) + 1e-8)

# 正規化関数の改良
def normalize_features(features, method='minmax', eps=1e-8):
    """
    features: (N, F) torch.Tensor
    method: 'minmax', 'zscore', 'robust'
    return: (N, F) 正規化後の特徴量
    """
    if method == 'minmax':
        minv = features.min(dim=0, keepdim=True)[0]
        maxv = features.max(dim=0, keepdim=True)[0]
        normed = (features - minv) / (maxv - minv + eps)
    elif method == 'zscore':
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        normed = (features - mean) / (std + eps)
    elif method == 'robust':
        # ロバスト正規化（中央値とMAD使用）
        median = features.median(dim=0, keepdim=True)[0]
        mad = torch.median(torch.abs(features - median), dim=0, keepdim=True)[0]
        normed = (features - median) / (mad + eps)
    else:
        raise ValueError(f'Unknown normalization method: {method}')
    
    return normed

def adaptive_threshold(scores, method='otsu', percentile=0.7):
    """
    スコアに基づく適応的閾値計算
    """
    if method == 'otsu':
        # Otsu's method風の実装
        scores_sorted = torch.sort(scores)[0]
        n = len(scores_sorted)
        
        best_threshold = 0
        best_variance = 0
        
        for i in range(1, n):
            threshold = scores_sorted[i].item()
            
            # 2クラスに分割
            low_class = scores_sorted[:i]
            high_class = scores_sorted[i:]
            
            if len(low_class) == 0 or len(high_class) == 0:
                continue
            
            # クラス間分散を計算
            w1 = len(low_class) / n
            w2 = len(high_class) / n
            
            m1 = torch.mean(low_class)
            m2 = torch.mean(high_class)
            
            variance = w1 * w2 * (m1 - m2) ** 2
            
            if variance > best_variance:
                best_variance = variance
                best_threshold = threshold
        
        return best_threshold
    
    elif method == 'percentile':
        # パーセンタイル基準
        return torch.quantile(scores, percentile).item()
    
    else:
        raise ValueError(f'Unknown threshold method: {method}')

class SuperPointSelector:
    """
    スーパーポイント選択のためのユーティリティクラス
    """
    def __init__(self, score_module, threshold_method='otsu'):
        self.score_module = score_module
        self.threshold_method = threshold_method
    
    def select_superpoints(self, features, hard_threshold=None):
        """
        特徴量からスーパーポイントを選択
        """
        # スコア計算
        scores = self.score_module(features)
        
        # 閾値計算
        if hard_threshold is None:
            threshold = adaptive_threshold(scores, method=self.threshold_method)
        else:
            threshold = hard_threshold
        
        # 選択マスク
        mask = scores > threshold
        
        return mask, scores, threshold
