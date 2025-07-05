import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config import make_cfg
from dataset import train_valid_data_loader
from geotransformer.utils.superpoint_score import SuperPointScoreModule, normalize_features
from model import create_model

# 特徴量正規化・スコア計算

def prepare_features(density, intensity_var):
    features = torch.stack([density, intensity_var], dim=1)
    features = normalize_features(features, method='minmax')
    return features

# スーパーポイントスコア重みのみを学習する分離学習

def train_score_weight_with_matching(cfg, epochs=5, lr=1e-2, max_batches=20, score_threshold=0.2):
    # 1. モデル本体（マッチングモデル）を構築しfreeze
    model = create_model(cfg)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # 2. スーパーポイントスコア重みモジュール
    score_module = SuperPointScoreModule(num_features=2)
    optimizer = torch.optim.Adam(score_module.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()  # 仮: 2値分類

    # 3. データローダ
    train_loader, _, _ = train_valid_data_loader(cfg, distributed=False)

    for epoch in range(epochs):
        batch_count = 0
        for batch in train_loader:
            if batch_count >= max_batches:
                break
            # 例: ref側のみ
            ref_feats = batch['ref_feats']  # (N, 1)
            ref_points = batch['ref_points']  # (N, 3)
            # 仮: 密度・強度分散
            density = torch.ones(ref_feats.shape[0])
            intensity_var = torch.var(ref_feats, dim=1) if ref_feats.ndim > 1 else torch.zeros(ref_feats.shape[0])
            features = prepare_features(density, intensity_var)
            # スコア計算
            scores = score_module(features)
            # ハードマスク: 閾値未満は除外
            mask = (scores > score_threshold).squeeze()
            if mask.sum() == 0:
                continue
            masked_feats = ref_feats[mask]
            masked_points = ref_points[mask]
            # --- ここで特徴量にスコアを掛ける場合は以下を有効化 ---
            # weighted_feats = ref_feats * scores.unsqueeze(1)
            #
            # --- マッチング損失計算例（仮: masked_featsを使って損失計算） ---
            # 本来はsrc側も同様に処理し、マッチング損失を計算
            # ここでは仮にラベルを密度>0.5で作成
            labels = (density[mask] > 0.5).float()
            loss = loss_fn(scores[mask], labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count += 1
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")
    # 重み保存
    torch.save(score_module.state_dict(), "score_weights.pth")
    print("score_weights.pth saved.")

if __name__ == "__main__":
    cfg = make_cfg()
    train_score_weight_with_matching(cfg, epochs=5, lr=1e-2, max_batches=20, score_threshold=0.2)
