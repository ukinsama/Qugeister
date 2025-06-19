# model_cqcnn.py

import pennylane as qml
from pennylane import numpy as pnp
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------
# PennyLaneデバイス設定 (グローバルに定義)
# -------------------------------------------------------------------------------
qubits_for_qnn_global = 4 # QNNで使用する量子ビット数。ガイスターの状態特徴量に応じて調整。
# dev_qnn_global = qml.device("default.qubit", wires=qubits_for_qnn_global)
# print(f"Global PennyLane device for QNN: {dev_qnn_global.name}")
# QNNを使用しない場合は上記はコメントアウトしてもよいが、CQCAgentの初期化でデバイスが必要になる。
# ダミーデバイスを渡すか、
# Agentを使わない場合はこのセクション自体不要。
# 今回はCQCAgentの定義を残すため、ダミーのdevを後でCNNAgentの学習前に定義します。
dev_qnn_global = qml.device("lightning.qubit", wires=qubits_for_qnn_global) # あとで初期化

# --- QNNComponent (元のコードから、一部修正の可能性あり) ---
class QNNComponent:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def TPE(self, theta, reps: int = 1): # Tensor Product Encoder
        for _ in range(reps):
            for qubit_index in range(self.n_qubits):
                # Ensure theta has enough elements
                if qubit_index < len(theta):
                    qml.RX(theta[qubit_index], wires=qubit_index)

    def HEE(self, theta, reps: int = 1): # Hardware Efficient Ansatz / Encoder
        # Ensure theta has enough elements for all rotations
        param_idx = 0
        for _ in range(reps):
            for qubit_index in range(self.n_qubits):
                if param_idx < len(theta):
                    qml.RY(theta[param_idx], wires=qubit_index) # Use RY for HEE generally
                    param_idx +=1
                else: break # Not enough params
            if param_idx >= len(theta) and self.n_qubits > 0 : break # Stop if out of params
            
            for qubit_index in range(self.n_qubits - 1):
                qml.CZ(wires=[qubit_index, qubit_index + 1]) # Use CZ for HEE
            if self.n_qubits > 1: # Circular entanglement for the last qubit
                 qml.CZ(wires=[self.n_qubits-1, 0])


    def ZFeatureMap(self, theta, reps: int = 1):
        for _ in range(reps):
            for i in range(self.n_qubits):
                if i < len(theta):
                    qml.Hadamard(wires=i)
                    qml.RZ(2.0 * theta[i], wires=i)

    def ZZFeatureMap(self, theta, reps: int = 1):
        pi_val = pnp.pi
        for _ in range(reps):
            for i in range(self.n_qubits):
                if i < len(theta):
                    qml.Hadamard(wires=i)
                    qml.RZ(2.0 * theta[i], wires=i)

            for i in range(self.n_qubits - 1):
                if i + 1 < len(theta):
                    qml.CNOT(wires=[i, i + 1])
                    angle = 2.0 * (pi_val - theta[i]) * (pi_val - theta[i + 1])
                    qml.RZ(angle, wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])

    def RealAmplitudes(self, theta, reps: int = 1):
        idx = 0
        for i in range(self.n_qubits):
            if idx < len(theta):
                qml.RY(theta[idx], wires=i)
                idx += 1

        for _ in range(reps):
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            for i in range(self.n_qubits):
                if idx < len(theta):
                    qml.RZ(theta[idx], wires=i)
                    idx += 1
            for i in range(self.n_qubits):
                if idx < len(theta):
                    qml.RY(theta[idx], wires=i)
                    idx += 1
            if idx >= len(theta): break


    def EfficientSU2(self, theta, reps: int = 1):
        # theta should have (reps+1) * 2 * n_qubits parameters
        idx = 0
        for i in range(self.n_qubits): # Initial layer
            if idx + 1 < len(theta):
                qml.RY(theta[idx], wires=i); idx += 1
                qml.RZ(theta[idx], wires=i); idx += 1
            else: break
        if idx >= len(theta) and reps > 0: return # Not enough params for even one rep

        for _ in range(reps):
            # Entanglement: all-to-all CZs (or CNOTs) or a specific pattern
            # Simplified to a chain for now, similar to HEE / RealAmplitudes
            for i in range(self.n_qubits -1 ): qml.CZ(wires=[i,i+1])
            if self.n_qubits > 1: qml.CZ(wires=[self.n_qubits-1,0]) # Circular

            for i in range(self.n_qubits): # Subsequent layers
                if idx + 1 < len(theta):
                    qml.RY(theta[idx], wires=i); idx += 1
                    qml.RZ(theta[idx], wires=i); idx += 1
                else: break
            if idx >= len(theta): break


    def make_circuit(self, embedding_type, ansatz_type, input_params, weight_params, exp_or_prob, feature_map_reps=1, ansatz_reps=1):
        # Feature map
        if embedding_type in ["ZFeatureMap", "ZZFeatureMap", "TPE", "HEE"]:
            if input_params.shape[0] != self.n_qubits:
                qml.AngleEmbedding(input_params, wires=range(self.n_qubits), rotation='X')
            else:
                getattr(self, embedding_type)(input_params, feature_map_reps)
        else:
            qml.AngleEmbedding(input_params, wires=range(self.n_qubits), rotation='X')

        # Ansatz
        if hasattr(self, ansatz_type):
            getattr(self, ansatz_type)(weight_params, ansatz_reps)
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")

        # Measurement
        if exp_or_prob == "exp":
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]
        elif exp_or_prob == "prob":
            return qml.probs(wires=range(self.n_qubits))
        else:
            raise ValueError(f"Unknown measurement type: {exp_or_prob}")

# --- CNN Model (CCNN2_Geister) ---
class CCNN2_Geister(nn.Module):
    def __init__(self, input_channels=6, board_size=BOARD_SIZE, num_outputs=BOARD_SIZE*BOARD_SIZE):
        super().__init__()
        self.board_size = board_size
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # Optional: Max Pooling
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 6x6 -> 3x3 if applied after conv2
        
        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.board_size, self.board_size)
            # x = self.pool1(self.relu2(self.conv2(self.relu1(self.conv1(dummy_input))))) # If pooling
            x = self.relu2(self.conv2(self.relu1(self.conv1(dummy_input)))) # No pooling
            self.flatten_size = x.numel() // x.shape[0] # numel gives total elements, divide by batch size

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.flatten_size, num_outputs) # No Tanh, raw Q-values

    def forward(self, state):
        x = self.relu1(self.conv1(state))
        x = self.relu2(self.conv2(x))
        # x = self.pool1(x) # If pooling
        x = self.flatten(x)
        x = self.linear(x)
        # Reshape to (batch_size, board_height, board_width) for Q-value map
        x = x.view(-1, self.board_size, self.board_size)
        return x

class CNN_QNN_CNN_Geister(nn.Module):
    def __init__(self, dev, embedding_type: str, ansatz_type: str, 
                 n_qubits_qnn=4, 
                 exp_or_prob="exp", 
                 feature_map_reps=1, 
                 ansatz_reps=1,
                 input_channels_cnn=6, 
                 board_size_cnn=6, 
                 cnn_fc_out_features=4,  # CNNからQNNに送る特徴数 = QNN入力数
                 qnn_fc_out_features=36):  # Q値出力 = 6x6マス
        super().__init__()
        self.dev = dev
        self.embedding_type = embedding_type
        self.ansatz_type = ansatz_type
        self.n_qubits = n_qubits_qnn  # ✅ ←ここを追加
        self.exp_or_prob = exp_or_prob
        self.feature_map_reps = feature_map_reps
        self.input_channels_cnn = input_channels_cnn
        self.ansatz_reps = ansatz_reps
        self.board_size_cnn = board_size_cnn
        self.cnn_fc_out_features = cnn_fc_out_features  # ✅ ←ここを追加
        self.qnn_fc_out_features = qnn_fc_out_features
        # CNN部分
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels_cnn, 16, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels_cnn, board_size_cnn, board_size_cnn)
            cnn_flat_dim = self.cnn_feature_extractor(dummy_input).shape[1]
        self.fc_to_qnn = nn.Linear(cnn_flat_dim, cnn_fc_out_features)

        # QNN部分の重み
        if ansatz_type == "RealAmplitudes":
            num_weights = (ansatz_reps + 1) * self.n_qubits
        elif ansatz_type == "EfficientSU2":
            num_weights = (ansatz_reps + 1) * 2 * self.n_qubits
        else:
            raise ValueError(f"Unsupported ansatz type: {ansatz_type}")
        self.q_weights = nn.Parameter(torch.randn(num_weights, requires_grad=True))

        # QNNコンポーネント
        self.qcomp = QNNComponent(n_qubits=self.n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def qnode(inputs, weights):
            return self.qcomp.make_circuit(
                embedding_type=self.embedding_type,
                ansatz_type=self.ansatz_type,
                input_params=inputs,
                weight_params=weights,
                exp_or_prob=self.exp_or_prob,
                feature_map_reps=self.feature_map_reps,
                ansatz_reps=self.ansatz_reps
            )

        self.qnode = qnode  # メンバに保存
        self.qnn_output_dim = self.n_qubits if self.exp_or_prob == "exp" else 2**self.n_qubits
        self.fc_from_qnn = nn.Linear(self.qnn_output_dim, qnn_fc_out_features)

    def forward(self, state):
        cnn_features = self.cnn_feature_extractor(state)  # shape: (B, N)
        qnn_input_features = self.fc_to_qnn(cnn_features)  # shape: (B, n_qubits)
        qnn_outputs = []
        for i in range(qnn_input_features.shape[0]):
            input_vector = qnn_input_features[i]
            qnn_output = self.qnode(input_vector, self.q_weights)
            # ↓ここを追加
            if isinstance(qnn_output, list):
                qnn_output = torch.tensor(qnn_output,dtype=torch.float32, device=self.fc_from_qnn.weight.device)
            qnn_outputs.append(qnn_output)

        qnn_outputs_tensor = torch.stack(qnn_outputs)  # shape: (B, qnn_output_dim)
        x = self.fc_from_qnn(qnn_outputs_tensor)
        return x.view(-1, self.board_size_cnn, self.board_size_cnn)
    
__all__ = ['QNNComponent', 'CNN_QNN_CNN_Geister','CCNN2_Geister']
# __all__に追加
# これにより、from model_cqcnn import * でQNNComponentとCNN_QNN_CNN_Geisterがインポートされる
# もしQNNComponentを使わない場合は、上記の__all__から削除してもよい
# -------------------------------------------------------------------------------
