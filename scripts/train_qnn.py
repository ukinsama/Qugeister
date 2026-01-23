#!/usr/bin/env python3
"""
Qugeister - QNN色推定モデル学習スクリプト

Usage:
    python scripts/train_qnn.py \
        --config configs/example_config.json \
        --trajectory trajectories/diverse_agents_3000.pkl \
        --epochs 50 \
        --output models/my_qnn.pth
"""

import argparse
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
from tqdm import tqdm
from pathlib import Path


class ExplicitColorEstimationQNN(nn.Module):
    """QNN色推定モデル

    入力: 448次元 (7チャネル × 64)
    出力: [8, 2] (8駒 × good/bad確率)
    """

    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # 前処理層
        self.pre_net = nn.Sequential(
            nn.Linear(448, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits)
        )

        # 量子デバイス
        self.dev = qml.device('default.qubit', wires=n_qubits)

        # 量子回路の重み
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        # 後処理層
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # 8駒 × 2 (good/bad)
        )

        # 量子ノードを定義
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            # 入力埋め込み（AngleEmbedding）
            qml.AngleEmbedding(inputs, wires=range(n_qubits))

            # Ansatz（BasicEntanglerLayers）
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(weights[layer, i, 0],
                           weights[layer, i, 1],
                           weights[layer, i, 2],
                           wires=i)
                # エンタングル
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])

            # 期待値を返す
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x):
        batch_size = x.shape[0]

        # 前処理
        x = self.pre_net(x)

        # 量子回路を適用
        q_out = []
        for i in range(batch_size):
            result = self.circuit(x[i], self.q_weights)
            q_out.append(torch.stack(result))
        q_out = torch.stack(q_out)

        # 後処理
        out = self.post_net(q_out)

        # [batch, 8, 2]にリシェイプしてsoftmax
        out = out.view(batch_size, 8, 2)
        out = torch.softmax(out, dim=-1)

        return out


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込む"""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_trajectory(trajectory_path: str) -> list:
    """棋譜データを読み込む"""
    with open(trajectory_path, 'rb') as f:
        return pickle.load(f)


def prepare_data(trajectory_data: list, train_ratio: float = 0.8):
    """棋譜データから学習用データを準備"""
    X_list = []
    y_list = []

    for sample in trajectory_data:
        if 'observation' in sample and 'labels' in sample:
            obs = np.array(sample['observation']).flatten()
            labels = np.array(sample['labels'])

            if obs.shape[0] == 448 and labels.shape[0] == 8:
                X_list.append(obs)
                y_list.append(labels)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # シャッフル
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # 分割
    split_idx = int(len(X) * train_ratio)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_val, y_val


def train_epoch(model, loader, optimizer, criterion, device):
    """1エポックの学習"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)

        # 各駒の損失を計算
        loss = 0
        for i in range(8):
            loss += criterion(outputs[:, i, :], y_batch[:, i])
        loss = loss / 8

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 精度計算
        preds = outputs.argmax(dim=-1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.numel()

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    """検証"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            loss = 0
            for i in range(8):
                loss += criterion(outputs[:, i, :], y_batch[:, i])
            loss = loss / 8

            total_loss += loss.item()

            preds = outputs.argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.numel()

    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser(description='QNN色推定モデル学習')
    parser.add_argument('--config', type=str, required=True,
                        help='設定ファイルのパス')
    parser.add_argument('--trajectory', type=str, required=True,
                        help='棋譜データのパス')
    parser.add_argument('--epochs', type=int, default=None,
                        help='エポック数（設定ファイルを上書き）')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='バッチサイズ（設定ファイルを上書き）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学習率（設定ファイルを上書き）')
    parser.add_argument('--output', type=str, default='models/qnn_weights.pth',
                        help='出力ファイルパス')
    parser.add_argument('--device', type=str, default='auto',
                        help='デバイス (auto/cpu/cuda)')
    args = parser.parse_args()

    # 設定読み込み
    print(f'設定ファイル: {args.config}')
    config = load_config(args.config)

    # 学習パラメータ
    training_config = config.get('training', {})
    epochs = args.epochs or training_config.get('epochs', 50)
    batch_size = args.batch_size or training_config.get('batch_size', 32)
    learning_rate = args.lr or training_config.get('learning_rate', 0.001)

    # デバイス設定
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'デバイス: {device}')

    # 量子パラメータ
    network = config.get('network', {})
    quantum_nodes = [n for n in network.get('nodes', []) if n.get('type') == 'quantum']
    if quantum_nodes:
        n_qubits = quantum_nodes[0].get('data', {}).get('n_qubits', 4)
    else:
        n_qubits = 4
    print(f'量子ビット数: {n_qubits}')

    # データ読み込み
    print(f'棋譜データ: {args.trajectory}')
    trajectory_data = load_trajectory(args.trajectory)
    print(f'  サンプル数: {len(trajectory_data)}')

    X_train, y_train, X_val, y_val = prepare_data(trajectory_data)
    print(f'  学習データ: {len(X_train)}')
    print(f'  検証データ: {len(X_val)}')

    # DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # モデル
    model = ExplicitColorEstimationQNN(n_qubits=n_qubits).to(device)
    print(f'パラメータ数: {sum(p.numel() for p in model.parameters()):,}')

    # 学習設定
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f'\n学習開始 (epochs={epochs}, batch_size={batch_size}, lr={learning_rate})')
    print('-' * 60)

    best_val_acc = 0
    best_model_state = None

    for epoch in tqdm(range(epochs), desc='Training'):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            tqdm.write(f'Epoch {epoch+1}/{epochs} - '
                      f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | '
                      f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')

    print('-' * 60)
    print(f'学習完了! ベスト検証精度: {best_val_acc:.4f}')

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if best_model_state:
        torch.save(best_model_state, output_path)
    else:
        torch.save(model.state_dict(), output_path)

    print(f'モデルを保存しました: {output_path}')
    print('\n次のステップ:')
    print('1. QuAic (https://quaic.up.railway.app) にアクセス')
    print('2. 「コンペティション > モデル」に移動')
    print(f'3. {output_path.name} をアップロード')


if __name__ == '__main__':
    main()
