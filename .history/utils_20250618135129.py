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