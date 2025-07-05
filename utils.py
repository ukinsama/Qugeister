# utils.py


# -------------------------------------------------------------------------------
# セル 1: インポート
# -------------------------------------------------------------------------------
import os
import tqdm
import time
import random
from copy import deepcopy
import abc
import json # JSONモジュールをインポート

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pennylane as qml
from pennylane import numpy as pnp # PennyLaneのnumpy
from pennylane.optimize import NesterovMomentumOptimizer
# from qiskit_aer import AerSimulator # PennyLane-Qiskitを使う場合
from graphviz import Digraph
# -------------------------------------------------------------------------------
# セル 2: ユーティリティ関数
# -------------------------------------------------------------------------------
def set_seed(seed=42):
    """
    乱数シードを設定する関数。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # GPUがある場合
    qml.enable_torch_interface()  # PennyLaneのTorchインターフェースを有効化
def get_device():
    """
    デバイスを取得する関数。
    Returns:
        str: 'cuda' if GPU is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'
def get_device_type():
    """
    デバイスのタイプを取得する関数。
    Returns:
        str: 'cuda' if GPU is available, otherwise 'cpu'.
    """
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
def get_device_name():
    """
    デバイス名を取得する関数。
    Returns:
        str: デバイス名 ('cuda' or 'cpu').
    """
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
def get_device_properties():
    """
    デバイスのプロパティを取得する関数。
    Returns:
        dict: デバイスのプロパティ。
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        return {
            'name': torch.cuda.get_device_name(device),
            'capability': torch.cuda.get_device_capability(device),
            'total_memory': torch.cuda.get_device_properties(device).total_memory
        }
    else:
        return {'name': 'cpu', 'capability': None, 'total_memory': None}
def get_device_info():
    """
    デバイス情報を取得する関数。
    Returns:
        dict: デバイス情報。
    """
    device = get_device()
    device_name = get_device_name()
    device_properties = get_device_properties()
    
    return {
        'device': device,
        'device_name': device_name,
        'device_properties': device_properties
    }


def visualize_cqcnn_structure_with_clusters(cnn_layers=None, qnn_layers=None, title="CQ-CNN Hybrid Model"):
    """
    CNNとQNNをクラスタ(囲み枠)でグループ化して表示。
    """
    dot = Digraph(comment=title)
    dot.attr(rankdir='LR')  # 左から右方向

    dot.node("Input", "Input", shape="oval", style="filled", color="lightgray")

    # --- CNN クラスタ ---
    with dot.subgraph(name='cluster_cnn') as c:
        c.attr(style='filled', color='lightblue', label='CNN')
        prev = "Input"
        if cnn_layers is None:
            cnn_layers = ["CNN Feature Extractor"]
        for idx, layer in enumerate(cnn_layers):
            node_id = f"CNN{idx}"
            c.node(node_id, layer)
            c.edge(prev, node_id)
            prev = node_id
        c.node("CNN_out", "CNN Output")
        c.edge(prev, "CNN_out")

    # --- QNN クラスタ ---
    with dot.subgraph(name='cluster_qnn') as q:
        q.attr(style='filled', color='lightpink', label='QNN')
        prev = "CNN_out"
        q.node("QNN_input", "QNN Input")
        q.edge(prev, "QNN_input")
        prev = "QNN_input"
        if qnn_layers is None:
            qnn_layers = ["AngleEmbedding", "RealAmplitudes", "Expval"]
        for idx, layer in enumerate(qnn_layers):
            node_id = f"QNN{idx}"
            q.node(node_id, layer)
            q.edge(prev, node_id)
            prev = node_id

    # 出力
    dot.node("Output", "Output", shape="oval", style="filled", color="lightgray")
    dot.edge(prev, "Output")

    return dot

def save_agent_2(agent, dir_path: str, model_type: str = "CQCNN", name: str = "agent"):

    os.makedirs(dir_path, exist_ok=True)

    if model_type == "CQCNN":
        model = agent.HNN
    elif model_type == "CNN":
        model = agent.NN
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    torch.save(model.state_dict(), os.path.join(dir_path, f"{name}_weights.pth"))
    print(f"💾 Saved weights: {name}_weights.pth")

    if model_type == "CQCNN":
        config = {
            "dev_type": model.dev.name,
            "embedding_type": model.embedding_type,
            "ansatz_type": model.ansatz_type,
            "n_qubits_qnn": model.n_qubits,
            "exp_or_prob": model.exp_or_prob,
            "feature_map_reps": model.feature_map_reps,
            "ansatz_reps": model.ansatz_reps,
            "input_channels_cnn": model.input_channels_cnn,
            "board_size_cnn": model.board_size_cnn,
            "cnn_fc_out_features": model.cnn_fc_out_features,
            "qnn_fc_out_features": model.qnn_fc_out_features
        }
        with open(os.path.join(dir_path, f"{name}_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print(f"📝 Saved config: {name}_config.json")
