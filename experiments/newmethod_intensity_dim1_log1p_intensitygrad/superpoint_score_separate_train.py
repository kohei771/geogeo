import torch
import torch.nn as nn
from config import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from geotransformer.utils.superpoint_score import SuperPointScoreModule, normalize_features
from loss import OverallLoss, Evaluator

class ScoreWeightTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        # モデル本体はfreeze
        self.model = create_model(cfg)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        # スーパーポイントスコア重みのみ学習
        self.score_module = SuperPointScoreModule(num_features=2)
        self.optimizer = torch.optim.Adam(self.score_module.parameters(), lr=cfg.optim.lr)
        self.loss_func = OverallLoss(cfg)
        self.evaluator = Evaluator(cfg)
        self.train_loader, self.val_loader, _ = train_valid_data_loader(cfg, distributed=False)
        self.max_epoch = cfg.optim.max_epoch
        self.max_batches = 20  # 必要に応じてcfg化
        self.score_threshold = getattr(cfg, 'superpoint_score_threshold', 0.2)

    def train(self):
        for epoch in range(self.max_epoch):
            epoch_loss = 0.0
            n_batches = 0
            for batch_count, batch in enumerate(self.train_loader):
                if batch_count >= self.max_batches:
                    break
                # 特徴量抽出
                ref_feats = batch.get('ref_feats', None)
                if ref_feats is None:
                    ref_feats = batch.get('features', None)
                if ref_feats is None:
                    ref_feats = batch.get('src_feats', None)
                if ref_feats is None:
                    raise KeyError(f"No suitable feature key found in batch: {batch.keys()}")
                ref_points = batch.get('ref_points', None)
                if ref_points is None:
                    ref_points = batch.get('points', None)
                if ref_points is None:
                    raise KeyError(f"No suitable points key found in batch: {batch.keys()}")
                # 仮: 密度・強度分散
                density = torch.ones(ref_feats.shape[0])
                intensity_var = torch.var(ref_feats, dim=1) if ref_feats.ndim > 1 else torch.zeros(ref_feats.shape[0])
                features = normalize_features(torch.stack([density, intensity_var], dim=1), method='minmax')
                # スコア計算
                scores = self.score_module(features)
                mask = (scores > self.score_threshold).squeeze()
                print(f"mask.sum(): {mask.sum().item()}")
                print(f"scores[mask][:10]: {scores[mask][:10].detach().cpu().numpy()}")
                # データdictをマスク済みにしてforward
                if mask.sum() == 0:
                    continue
                batch['features'] = ref_feats[mask]
                batch['points'] = ref_points[mask]
                # モデルforward
                output_dict = self.model(batch)
                loss_dict = self.loss_func(output_dict, batch)
                loss = loss_dict['loss']
                print(f"loss: {loss.item()}")
                self.optimizer.zero_grad()
                loss.backward()
                print(f"score_module.weights.grad: {self.score_module.weights.grad}")
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
                print(f"[Epoch {epoch+1}/{self.max_epoch}] Batch {batch_count+1}/{self.max_batches} Loss: {loss.item():.4f}", end='\r')
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"\nEpoch {epoch+1}: avg_loss={avg_loss:.4f}")
        torch.save(self.score_module.state_dict(), "score_weights.pth")
        print("score_weights.pth saved.")

if __name__ == "__main__":
    cfg = make_cfg()
    trainer = ScoreWeightTrainer(cfg)
    trainer.train()
