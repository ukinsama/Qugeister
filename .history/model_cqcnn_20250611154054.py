# model_cqcnn.py

import pennylane as qml
from pennylane import numpy as pnp


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
dev_qnn_global = qml.device("default.qubit", wires=qubits_for_qnn_global) # あとで初期化

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