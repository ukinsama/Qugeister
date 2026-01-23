# Qugeister - Quantum Geister AI Training

Geisterボードゲーム用のQNN（量子ニューラルネットワーク）色推定モデルの学習環境です。

## 概要

QugeisterはQuAicプラットフォームと連携して、QNN色推定モジュールを学習するためのツールキットです。

### ワークフロー

```
QuAic HNN Composer
    ↓ config.json エクスポート
Qugeister（このリポジトリ）
    ↓ 棋譜データで学習
    ↓ weights.pth エクスポート
QuAic に提出
    ↓ 競技参加
```

## クイックスタート

### Google Colabで学習（推奨）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quaic/Qugeister/blob/main/notebooks/geister_qnn_training.ipynb)

1. 上のバッジをクリックしてColabを開く
2. QuAicからエクスポートした`config.json`をアップロード
3. 学習を実行
4. `weights.pth`をダウンロードしてQuAicに提出

### ローカルで学習

```bash
# リポジトリをクローン
git clone https://github.com/quaic/Qugeister.git
cd Qugeister

# 依存関係をインストール
pip install -r requirements.txt

# 学習を実行
python scripts/train_qnn.py \
  --config configs/your_config.json \
  --trajectory trajectories/diverse_agents_3000.pkl \
  --epochs 50 \
  --output models/my_qnn.pth
```

## ディレクトリ構成

```
Qugeister/
├── notebooks/
│   └── geister_qnn_training.ipynb  # Colab学習ノートブック
├── scripts/
│   └── train_qnn.py                # ローカル学習スクリプト
├── configs/
│   └── example_config.json         # 設定ファイル例
├── trajectories/                   # 棋譜データ（別途ダウンロード）
├── models/                         # 学習済みモデル出力先
└── requirements.txt                # 依存関係
```

## 設定ファイル（config.json）

QuAic HNN Composerからエクスポートされる設定ファイルの形式：

```json
{
  "model_name": "my_qnn",
  "description": "QNN Color Estimator",
  "network": {
    "nodes": [...],
    "edges": [...]
  },
  "training": {
    "epochs": 50,
    "batch_size": 256,
    "learning_rate": 0.001
  }
}
```

## 棋譜データ

学習用の棋譜データはQuAicの「コンペティション > データ」からダウンロードできます。

- **diverse_agents_3000.pkl** - 推奨（3000試合、多様なエージェント）
- その他のデータセットも利用可能

## モデル仕様

### 入力

- 448次元（7チャネル × 8×8ボード）
- チャネル: Own Good, Own Bad, Enemy, Own Goals, Enemy Goals, Turn, Valid

### 出力

- [8, 2] 形状（8駒 × good/bad確率）

## ライセンス

MIT License

## 関連リンク

- [QuAic](https://quaic.up.railway.app) - 競技プラットフォーム
- [Geisterルール](https://quaic.up.railway.app/learn/tutorial) - ゲームルール解説
