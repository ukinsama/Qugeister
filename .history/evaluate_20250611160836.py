# evaluate.py
import torch
import torch.nn as nn
import pennylane as qml
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
                 n_qubits_qnn=4, exp_or_prob="exp", 
                 feature_map_reps=1, ansatz_reps=1,
                 input_channels_cnn=6, board_size_cnn=6, 
                 cnn_fc_out_features=4,  # CNNからQNNに送る特徴数 = QNN入力数
                 qnn_fc_out_features=36):  # Q値出力 = 6x6マス
        super().__init__()
        self.dev = dev
        self.embedding_type = embedding_type
        self.ansatz_type = ansatz_type
        self.exp_or_prob = exp_or_prob
        self.feature_map_reps = feature_map_reps
        self.ansatz_reps = ansatz_reps
        self.n_qubits = n_qubits_qnn
        self.board_size_cnn = board_size_cnn

        # CNN部分
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels_cnn, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
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

import torch
import os

# 保存先ディレクトリ
save_dir = "./models_geister_final/agentA/"
os.makedirs(save_dir, exist_ok=True)

# --- 保存処理 ---
if hasattr(agent_a_q, 'NN') and agent_a_q.NN is not None:
    cnn_path = os.path.join(save_dir, "agentA_CNN_final.pth")
    torch.save(agent_a_q.NN.state_dict(), cnn_path)
    print(f"✅ CNN part of Agent A saved to: {cnn_path}")

if hasattr(agent_a_q, 'QNN') and agent_a_q.QNN is not None:
    qnn_path = os.path.join(save_dir, "agentA_QNN_final.pth")
    torch.save(agent_a_q.QNN.state_dict(), qnn_path)
    print(f"✅ QNN part of Agent A saved to: {qnn_path}")

# --- 最終モデルの保存 ---
save_dir = "./models_geister_final/agentA/"
os.makedirs(save_dir, exist_ok=True)

if hasattr(agent_a_q, 'NN') and agent_a_q.NN is not None:
    torch.save(agent_a_q.NN.state_dict(), os.path.join(save_dir, "agentA_CNN_final.pth"))
    print("✅ CNN model saved")

if hasattr(agent_a_q, 'QNN') and agent_a_q.QNN is not None:
    torch.save(agent_a_q.QNN.state_dict(), os.path.join(save_dir, "agentA_QNN_final.pth"))
    print("✅ QNN model saved")