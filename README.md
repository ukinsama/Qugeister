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

# 学習を実行（基本）
python scripts/train_qnn.py \
  --config configs/your_config.json \
  --trajectory trajectories/diverse_agents_3000.pkl

# 学習を実行（詳細設定）
python scripts/train_qnn.py \
  --config configs/your_config.json \
  --trajectory trajectories/diverse_agents_3000.pkl \
  --epochs 100 \
  --batch-size 256 \
  --use-focal-loss \
  --device cuda
```

## ディレクトリ構成

```
Qugeister/
├── notebooks/
│   └── geister_qnn_training.ipynb  # Colab学習ノートブック
├── scripts/
│   └── train_qnn.py                # ローカル学習スクリプト（最適化版）
├── src/
│   └── qugeister/
│       ├── models/
│       │   ├── qnn.py              # ExplicitColorEstimationQNN
│       │   └── hnn_config_loader.py # HNN Composerコンフィグローダー
│       └── quantum/
│           └── quantum_layer.py    # PennyLane量子レイヤー
├── configs/
│   └── example_config.json         # 設定ファイル例
├── trajectories/                   # 棋譜データ（別途ダウンロード）
├── experiments/                    # 学習結果出力先
└── requirements.txt                # 依存関係
```

## 学習スクリプトオプション

```
python scripts/train_qnn.py --help

オプション:
  --config          HNN Composerからエクスポートした設定ファイル
  --trajectory      棋譜データ (.pkl) のパス [必須]
  --n-qubits        量子ビット数 (--configがある場合は無視)
  --n-layers        量子レイヤー数 (--configがある場合は無視)
  --backend         量子バックエンド (backpropが最速)
  --epochs          エポック数 (デフォルト: 100)
  --batch-size      バッチサイズ (デフォルト: 256)
  --lr              学習率 (デフォルト: 0.001)
  --device          デバイス: cuda/cpu
  --patience        Early Stoppingのpatience (デフォルト: 15)
  --use-focal-loss  困難なサンプルに重み付けするFocal Lossを使用
  --resume          チェックポイントから学習を再開
  --save-dir        出力ディレクトリを指定
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

### 学習機能

- **CUDA/GPU対応**: 自動混合精度(AMP)による高速学習
- **Focal Loss**: 困難なサンプルへの重み付けで精度向上
- **Early Stopping**: 過学習を自動検出して停止
- **チェックポイント**: 学習の中断・再開が可能

## ライセンス

MIT License

## 関連リンク

- [QuAic](https://quaic.up.railway.app) - 競技プラットフォーム
- [Geisterルール](https://quaic.up.railway.app/learn/tutorial) - ゲームルール解説
