# SuperPoint Score Module

このモジュールは、3D点群マッチングにおけるスーパーポイントの「情報価値」に基づく重み付けスコアを学習・適用するためのものです。

## 概要

SuperPoint Score Moduleは以下の機能を提供します：

1. **スーパーポイント特徴量の計算**
   - 密度（パッチ内の点数）
   - 強度分散（Intensity variance）
   - 強度平均（Intensity mean）
   - 強度勾配（Intensity gradient）

2. **学習可能な重み付けスコア**
   - 各特徴量に対する重要度を自動学習
   - 線形結合またはMLP（非線形）による柔軟なスコア計算

3. **適応的な閾値決定**
   - Otsu's method風の自動閾値設定
   - パーセンタイル基準の閾値設定

## 使用方法

### 1. スーパーポイントスコアの学習

```bash
# 学習の実行
bash train_superpoint_score.sh

# または直接実行
python superpoint_score_separate_train.py
```

### 2. メインモデルでの使用

```python
# 設定でスーパーポイントスコアを有効化
cfg = make_cfg(use_superpoint_score=True, superpoint_score_threshold=0.5)

# モデル作成時に自動的にスコアモジュールが組み込まれます
model = create_model(cfg)
```

### 3. 学習時と推論時の動作

- **学習時**: Soft weighting（全スーパーポイントを重み付きで使用）
- **推論時**: Hard masking（閾値以上のスーパーポイントのみ選択）

## 特徴量の詳細

### 1. 密度（Density）
各スーパーポイントに含まれる点の数。密度の高いスーパーポイントは構造的に重要な可能性が高い。

### 2. 強度分散（Intensity Variance）
スーパーポイント内の点の強度値の分散。分散が大きいほど多様な反射特性を持つ。

### 3. 強度平均（Intensity Mean）
スーパーポイント内の点の強度値の平均。材質や表面特性を表現。

### 4. 強度勾配（Intensity Gradient）
スーパーポイント内の強度値の最大値と最小値の差。エッジやテクスチャの変化を表現。

## パラメータ

- `num_features`: 特徴量の次元数（デフォルト: 4）
- `use_nonlinear`: 非線形MLP使用フラグ（デフォルト: True）
- `superpoint_score_threshold`: Hard masking用閾値（デフォルト: 0.5）

## 出力ファイル

- `score_weights.pth`: 学習済みスーパーポイントスコア重み

## 期待される効果

1. **マッチング精度の向上**: 有用なスーパーポイントを自動選択
2. **ロバスト性の向上**: ノイズや外れ値の影響を軽減
3. **計算効率の向上**: 不要なスーパーポイントを除外

## 注意事項

- メインモデルの重みは固定したまま、スコアモジュールのみを学習
- 学習は短時間（10エポック程度）で完了
- GPUメモリ使用量は比較的少ない
