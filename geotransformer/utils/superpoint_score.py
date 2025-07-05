import torch
import torch.nn as nn

def compute_superpoint_score(features, weights):
    """
    features: (N, F) torch.Tensor, 各スーパーポイントの特徴量（例: [密度, 分散, 平均, ...]）
    weights: (F,) torch.nn.Parameter, 学習可能な重みベクトル
    return: (N,) torch.Tensor, 各スーパーポイントのスコア
    """
    return (features * weights).sum(dim=1)

class SuperPointScoreModule(nn.Module):
    """
    スーパーポイント情報価値スコア計算用モジュール。
    weights（alpha, beta, ...）は学習可能パラメータ。
    """
    def __init__(self, num_features, init_weights=None):
        super().__init__()
        if init_weights is not None:
            self.weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))
        else:
            self.weights = nn.Parameter(torch.ones(num_features, dtype=torch.float32))
    def forward(self, features):
        return compute_superpoint_score(features, self.weights)

# 例: 正規化関数

def normalize_features(features, method='minmax'):
    """
    features: (N, F)
    method: 'minmax' or 'zscore'
    return: (N, F) 正規化後
    """
    if method == 'minmax':
        minv = features.min(dim=0, keepdim=True)[0]
        maxv = features.max(dim=0, keepdim=True)[0]
        normed = (features - minv) / (maxv - minv + 1e-8)
    elif method == 'zscore':
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        normed = (features - mean) / (std + 1e-8)
    else:
        raise ValueError(f'Unknown method: {method}')
    return normed
