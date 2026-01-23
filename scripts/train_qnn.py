#!/usr/bin/env python3
"""
Qugeister - QNN色推定モデル学習スクリプト (最適化版)

HNN Composerからエクスポートした設定ファイルを使用してQNNを学習します。

Features:
- CUDA/GPU対応 (自動混合精度AMP)
- PyTorch DataLoaderによる高速データ読み込み
- Focal Loss (困難なサンプルへの重み付け)
- チェックポイント再開 (--resume)
- Early Stopping

Usage:
    # 基本的な使用法
    python scripts/train_qnn.py \
        --config configs/example_config.json \
        --trajectory trajectories/diverse_agents_3000.pkl

    # 詳細設定
    python scripts/train_qnn.py \
        --config configs/my_config.json \
        --trajectory trajectories/data.pkl \
        --epochs 100 \
        --batch-size 256 \
        --use-focal-loss \
        --device cuda

    # 学習再開
    python scripts/train_qnn.py \
        --config configs/my_config.json \
        --trajectory trajectories/data.pkl \
        --resume experiments/qnn/checkpoint.pth
"""

import argparse
import pickle
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to Python path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch

# Import from src package
from src.qugeister.models.qnn import ExplicitColorEstimationQNN
from src.qugeister.models.hnn_config_loader import (
    load_hnn_config,
    build_model_from_config,
    export_for_competition,
    HNNConfig,
    HNNColorEstimator,
    QuantumConfig,
)


def convert_to_quaic_format(state_dict: dict, use_hnn_model: bool = False) -> dict:
    """Convert model state_dict to QuAic internal format.

    QuAic expects:
    - preprocessing.* (448D -> n_qubits)
    - quantum_layer.quantum_weights [n_layers, n_qubits, 3]
    - color_head.* (n_qubits -> 16)

    ExplicitColorEstimationQNN has:
    - preprocessing.* (same)
    - quantum_layer.weights [n_layers, n_qubits, 2] (RY+RZ only)
    - color_head.* (same)

    HNNColorEstimator has:
    - pre_layers.* -> preprocessing.*
    - quantum_layer.weights -> quantum_layer.quantum_weights
    - post_layers.* -> color_head.*
    """
    quaic_state = {}

    for key, value in state_dict.items():
        # Skip internal buffers
        if key == '_device_tracker':
            continue

        # Skip gating_head (not needed for QuAic)
        if key.startswith('gating_head.'):
            continue

        # HNNColorEstimator: pre_layers -> preprocessing
        if use_hnn_model and key.startswith('pre_layers.'):
            new_key = key.replace('pre_layers.', 'preprocessing.')
            quaic_state[new_key] = value
            continue

        # HNNColorEstimator: post_layers -> color_head
        if use_hnn_model and key.startswith('post_layers.'):
            new_key = key.replace('post_layers.', 'color_head.')
            quaic_state[new_key] = value
            continue

        # Quantum layer: weights -> quantum_weights, expand [n_layers, n_qubits, 2] -> [n_layers, n_qubits, 3]
        if key == 'quantum_layer.weights':
            n_layers, n_qubits, _ = value.shape
            expanded = torch.zeros(n_layers, n_qubits, 3, dtype=value.dtype)
            # Original: [RY, RZ] -> QuAic: [RX, RY, RZ]
            expanded[:, :, 1:] = value  # RY, RZ in positions 1, 2
            quaic_state['quantum_layer.quantum_weights'] = expanded
            continue

        # Keep other keys as-is
        quaic_state[key] = value

    return quaic_state


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Class balance weight (default: None)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, gamma: float = 2.0, alpha: float = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(inputs, dim=-1)
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight

        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ColorEstimationDataset(Dataset):
    """PyTorch Dataset for efficient data loading"""

    def __init__(self, states, colors, turns=None):
        self.states = torch.FloatTensor(states)
        self.colors = torch.LongTensor(colors)
        if turns is not None:
            self.turns = torch.LongTensor(turns)
        else:
            self.turns = torch.full((len(states),), 100, dtype=torch.long)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.colors[idx], self.turns[idx]


def load_trajectories(trajectory_path):
    """Load trajectories from pickle file"""
    with open(trajectory_path, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories


def convert_to_dataset(trajectories):
    """Convert trajectories to (state, true_colors, turns) dataset"""
    states = []
    colors = []
    turns = []

    for traj in trajectories:
        # Player A
        states_A = traj['states_A']
        colors_A = traj['true_colors_A']
        states.extend(states_A)
        colors.extend(colors_A)
        turns.extend([i * 2 for i in range(len(states_A))])

        # Player B
        states_B = traj['states_B']
        colors_B = traj['true_colors_B']
        states.extend(states_B)
        colors.extend(colors_B)
        turns.extend([i * 2 + 1 for i in range(len(states_B))])

    return np.array(states), np.array(colors), np.array(turns)


def save_checkpoint(
    model, optimizer, scheduler, scaler, epoch,
    best_val_loss, best_val_acc, patience_counter,
    history, best_model_state, train_indices, val_indices, save_path
):
    """Save full training checkpoint for resume"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'patience_counter': patience_counter,
        'history': history,
        'best_model_state': best_model_state,
        'train_indices': train_indices,
        'val_indices': val_indices,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device):
    """Load training checkpoint for resume"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return (
        checkpoint['epoch'],
        checkpoint['best_val_loss'],
        checkpoint.get('best_val_acc', 0.0),
        checkpoint['patience_counter'],
        checkpoint['history'],
        checkpoint.get('best_model_state'),
        checkpoint.get('train_indices'),
        checkpoint.get('val_indices'),
    )


def train_qnn_optimized(
    qnn_model,
    train_loader,
    val_loader,
    n_epochs=100,
    lr=0.001,
    device='cuda',
    patience=15,
    min_delta=1e-4,
    save_dir='experiments/qnn',
    use_amp=True,
    gradient_accumulation_steps=1,
    use_focal_loss=False,
    focal_gamma=2.0,
    min_eval_turn=0,
    resume_path=None,
    train_indices=None,
    val_indices=None,
    use_hnn_model=False,
):
    """Train QNN with optimizations (GPU, AMP, DataLoader)"""
    training_start_time = time.time()

    # Optimizer
    optimizer = optim.Adam(qnn_model.parameters(), lr=lr)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Loss function
    if use_focal_loss:
        criterion = FocalLoss(gamma=focal_gamma)
        print(f"[Loss] Using Focal Loss (gamma={focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("[Loss] Using CrossEntropy Loss")

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if use_amp and device == 'cuda' else None

    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_loss': [], 'epoch_times': []}

    # Early stopping
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    start_epoch = 0
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    if resume_path and Path(resume_path).exists():
        print(f"\n学習を再開: {resume_path}")
        (start_epoch, best_val_loss, best_val_acc, patience_counter,
         history, best_model_state, _, _) = load_checkpoint(
            resume_path, qnn_model, optimizer, scheduler, scaler, device
        )
        start_epoch += 1
        print(f"  エポック {start_epoch} から再開, best_val_loss={best_val_loss:.4f}")

    print(f"\n学習開始 ({n_epochs} エポック)...")
    print(f"  最小評価ターン: {min_eval_turn}")

    for epoch in range(start_epoch, n_epochs):
        epoch_start_time = time.time()

        # Training
        qnn_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Train]")
        optimizer.zero_grad()

        for batch_idx, (batch_states, batch_colors, batch_turns) in enumerate(pbar):
            batch_states = batch_states.to(device, non_blocking=True)
            batch_colors = batch_colors.to(device, non_blocking=True)
            batch_turns = batch_turns.to(device, non_blocking=True)

            if use_amp and device == 'cuda':
                with torch.amp.autocast('cuda'):
                    output = qnn_model(batch_states)
                    color_logits = output if use_hnn_model else output['color_estimation']

                    valid_mask = (batch_colors != -1)
                    flat_logits = color_logits.view(-1, 2)
                    flat_targets = batch_colors.view(-1)
                    flat_mask = valid_mask.view(-1)

                    if flat_mask.sum() > 0:
                        valid_logits = flat_logits[flat_mask]
                        valid_targets = flat_targets[flat_mask]
                        loss = criterion(valid_logits, valid_targets)
                    else:
                        loss = torch.tensor(0.0, device=device)

                    loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                output = qnn_model(batch_states)
                color_logits = output if use_hnn_model else output['color_estimation']

                valid_mask = (batch_colors != -1)
                flat_logits = color_logits.view(-1, 2)
                flat_targets = batch_colors.view(-1)
                flat_mask = valid_mask.view(-1)

                if flat_mask.sum() > 0:
                    valid_logits = flat_logits[flat_mask]
                    valid_targets = flat_targets[flat_mask]
                    loss = criterion(valid_logits, valid_targets)
                else:
                    loss = torch.tensor(0.0, device=device)

                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # Compute accuracy
            with torch.no_grad():
                pred_colors = torch.argmax(color_logits, dim=-1)
                turn_mask = (batch_turns >= min_eval_turn).unsqueeze(1).expand_as(valid_mask)
                eval_mask = valid_mask & turn_mask
                correct_predictions = (pred_colors == batch_colors) & eval_mask
                train_correct += correct_predictions.sum().item()
                train_total += eval_mask.sum().item()
                train_loss += loss.item() * gradient_accumulation_steps

            pbar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0

        # Validation
        qnn_model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        n_val_batches = 0

        with torch.no_grad():
            for batch_states, batch_colors, batch_turns in tqdm(
                    val_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Val]"):
                batch_states = batch_states.to(device, non_blocking=True)
                batch_colors = batch_colors.to(device, non_blocking=True)
                batch_turns = batch_turns.to(device, non_blocking=True)

                valid_mask = (batch_colors != -1)

                if use_amp and device == 'cuda':
                    with torch.amp.autocast('cuda'):
                        output = qnn_model(batch_states)
                        color_logits = output if use_hnn_model else output['color_estimation']
                else:
                    output = qnn_model(batch_states)
                    color_logits = output if use_hnn_model else output['color_estimation']

                pred_colors = torch.argmax(color_logits, dim=-1)

                flat_logits = color_logits.view(-1, 2)
                flat_targets = batch_colors.view(-1)
                flat_mask = valid_mask.view(-1)

                if flat_mask.sum() > 0:
                    valid_logits = flat_logits[flat_mask]
                    valid_targets = flat_targets[flat_mask]
                    batch_loss = criterion(valid_logits, valid_targets)
                    val_loss += batch_loss.item()
                    n_val_batches += 1

                turn_mask = (batch_turns >= min_eval_turn).unsqueeze(1).expand_as(valid_mask)
                eval_mask = valid_mask & turn_mask
                correct_predictions = (pred_colors == batch_colors) & eval_mask
                val_correct += correct_predictions.sum().item()
                val_total += eval_mask.sum().item()

        val_loss = val_loss / n_val_batches if n_val_batches > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start_time

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)

        lr_current = optimizer.param_groups[0]['lr']
        elapsed = time.time() - training_start_time
        print(f"\nEpoch {epoch + 1:3d}/{n_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | "
              f"LR: {lr_current:.2e} | Time: {epoch_time:.1f}s")

        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in qnn_model.state_dict().items()}
            best_checkpoint_path = save_path / 'best_model_checkpoint.pth'
            if use_hnn_model:
                n_qubits = qnn_model.config.quantum.n_qubits
                n_layers = qnn_model.config.quantum.n_layers
            else:
                n_qubits = qnn_model.n_qubits
                n_layers = qnn_model.n_layers
            torch.save({
                'model_state_dict': best_model_state,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'n_qubits': n_qubits,
                'n_layers': n_layers
            }, best_checkpoint_path)
            print(f"  -> 改善! Val Loss: {val_loss:.4f} (保存済み)")
        else:
            patience_counter += 1
            print(f"  -> 改善なし ({patience_counter}/{patience})")

        # Save checkpoint
        checkpoint_path = save_path / 'checkpoint.pth'
        save_checkpoint(
            qnn_model, optimizer, scheduler, scaler, epoch,
            best_val_loss, best_val_acc, patience_counter,
            history, best_model_state, train_indices, val_indices,
            str(checkpoint_path)
        )

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}!")
            break

    # Load best model
    if best_model_state is not None:
        qnn_model.load_state_dict(best_model_state)
        qnn_model.to(device)

    total_time = time.time() - training_start_time
    history['total_time_seconds'] = total_time

    print("-" * 60)
    print("学習完了!")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Best Val Acc: {best_val_acc:.2%}")
    print(f"  合計時間: {total_time / 60:.1f} 分")

    return history


def main():
    parser = argparse.ArgumentParser(
        description='QNN色推定モデル学習 (最適化版)'
    )
    # Data
    parser.add_argument('--trajectory', type=str, required=True,
                        help='棋譜データ (.pkl) のパス')

    # HNN Config
    parser.add_argument('--config', type=str, default=None,
                        help='HNN Composerからエクスポートした設定ファイル')

    # Model architecture
    parser.add_argument('--n-qubits', type=int, default=4,
                        help='量子ビット数 (--configがある場合は無視)')
    parser.add_argument('--n-layers', type=int, default=2,
                        help='量子レイヤー数 (--configがある場合は無視)')
    parser.add_argument('--backend', type=str, default='backprop',
                        choices=['pennylane', 'auto', 'backprop'],
                        help='量子バックエンド (backpropが最速)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='エポック数')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stoppingのpatience')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学習率')
    parser.add_argument('--device', type=str, default='cuda',
                        help='デバイス (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoaderのワーカー数')
    parser.add_argument('--no-amp', action='store_true',
                        help='自動混合精度(AMP)を無効化')

    # Focal Loss
    parser.add_argument('--use-focal-loss', action='store_true',
                        help='Focal Lossを使用')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal Lossのgamma値')

    # Resume and output
    parser.add_argument('--resume', type=str, default=None,
                        help='再開するチェックポイントのパス')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='出力ディレクトリ')

    args = parser.parse_args()

    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA利用不可、CPUにフォールバック")
        args.device = 'cpu'
        args.no_amp = True

    # Load HNN config
    hnn_config = None
    if args.config:
        print(f"HNN設定を読み込み: {args.config}")
        hnn_config = load_hnn_config(args.config)
        args.n_qubits = hnn_config.quantum.n_qubits
        args.n_layers = hnn_config.quantum.n_layers
        # Override training params from config
        if args.lr == 0.001:
            args.lr = hnn_config.training.get('learning_rate',
                      hnn_config.training.get('recommended_lr', 0.001))
        if args.batch_size == 256:
            args.batch_size = hnn_config.training.get('batch_size',
                              hnn_config.training.get('recommended_batch_size', 256))
        if args.epochs == 100:
            args.epochs = hnn_config.training.get('epochs',
                          hnn_config.training.get('recommended_epochs', 100))
        print(f"  量子ビット: {args.n_qubits}, レイヤー: {args.n_layers}")

    # Setup save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = Path(f'experiments/qnn/qnn_{args.n_qubits}q{args.n_layers}l_{timestamp}')
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("QNN色推定モデル学習 (最適化版)")
    print("=" * 70)
    if args.config:
        print(f"設定ファイル: {args.config}")
    print(f"モデル: {args.n_qubits}q x {args.n_layers}L ({args.backend})")
    print(f"デバイス: {args.device.upper()}")
    print(f"バッチサイズ: {args.batch_size}")
    print(f"自動混合精度 (AMP): {not args.no_amp and args.device == 'cuda'}")
    print(f"出力先: {save_dir}")
    print()

    # Load trajectories
    print(f"棋譜データを読み込み: {args.trajectory}")
    trajectories = load_trajectories(args.trajectory)

    # Convert to dataset
    print("データセットに変換中...")
    states, true_colors, turns = convert_to_dataset(trajectories)
    print(f"サンプル数: {len(states)}")

    # Train/val split
    n_samples = len(states)
    n_val = int(n_samples * 0.2)
    indices = np.random.permutation(n_samples)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    # Create datasets
    train_dataset = ColorEstimationDataset(
        states[train_indices], true_colors[train_indices], turns[train_indices]
    )
    val_dataset = ColorEstimationDataset(
        states[val_indices], true_colors[val_indices], turns[val_indices]
    )

    print(f"学習データ: {len(train_dataset)}")
    print(f"検証データ: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
        persistent_workers=(args.num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
        persistent_workers=(args.num_workers > 0)
    )

    # Create model
    use_hnn_model = hnn_config is not None
    print(f"\nモデルを作成中 (qubits={args.n_qubits}, layers={args.n_layers})...")
    if use_hnn_model:
        print(f"  HNNColorEstimator (QuAic互換)")
        qnn_model = build_model_from_config(args.config, device=args.device, backend=args.backend)
    else:
        print(f"  ExplicitColorEstimationQNN")
        qnn_model = ExplicitColorEstimationQNN(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            backend=args.backend
        )
        qnn_model.to(args.device)

    # Train
    history = train_qnn_optimized(
        qnn_model,
        train_loader,
        val_loader,
        n_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        patience=args.patience,
        save_dir=str(save_dir),
        use_amp=(not args.no_amp and args.device == 'cuda'),
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        resume_path=args.resume,
        train_indices=train_indices,
        val_indices=val_indices,
        use_hnn_model=use_hnn_model,
    )

    # Save model
    model_path = save_dir / 'qnn_color_model.pth'
    torch.save(qnn_model.state_dict(), model_path)
    print(f"\nモデルを保存: {model_path}")

    # Save QuAic-compatible weights
    weights_path = save_dir / 'weights.pth'
    quaic_state_dict = convert_to_quaic_format(qnn_model.state_dict(), use_hnn_model)
    torch.save(quaic_state_dict, weights_path)
    print(f"QuAic用重みを保存: {weights_path}")

    # Save history
    history_path = save_dir / 'training_history.json'
    history_serializable = {}
    for k, v in history.items():
        if isinstance(v, list):
            history_serializable[k] = [float(x) if hasattr(x, 'item') else x for x in v]
        else:
            history_serializable[k] = float(v) if hasattr(v, 'item') else v
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)

    print("\n" + "=" * 70)
    print("学習完了!")
    print("=" * 70)
    print(f"結果は {save_dir} に保存されました")
    print()
    print("次のステップ:")
    print("1. QuAic (https://quaic.up.railway.app) にアクセス")
    print("2. 「コンペティション > モデル」に移動")
    print(f"3. {weights_path.name} をアップロード")


if __name__ == '__main__':
    main()
