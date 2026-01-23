"""
Explicit Color Estimation QNN (Quantum Neural Network)

Quantum neural network for opponent piece color estimation in Geister.
Implements Tomoda & Hasebe (2021) Phase 2 approach.

Architecture:
    Input [B, 448] (7-channel partial observation, 7x8x8 flattened)
    -> Preprocessing (448 -> 128 -> 64 -> n_qubits)
    -> QuantumLayer (variational quantum circuit)
    -> Color Head: [B, n_qubits] -> [B, 16] -> [B, 8, 2]

Note: This model also includes a gating head (legacy), but it is NOT used
      in the Tomoda & Hasebe (2021) approach. Only color estimation is used.
"""

import torch
import torch.nn as nn

# Import quantum layers (support multiple backends)
try:
    from ..quantum.quantum_layer import QuantumLayer
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

try:
    from ..quantum.qulacs_layer import QulacsQuantumLayer
    QULACS_AVAILABLE = True
except ImportError:
    QULACS_AVAILABLE = False


class ExplicitColorEstimationQNN(nn.Module):
    """Quantum Neural Network for Color Estimation (Tomoda & Hasebe 2021)

    Architecture:
        Input [B, 448] (7x8x8 flattened partial observation)
        -> Preprocessing (448 -> 128 -> 64 -> n_qubits)
        -> QuantumLayer (variational quantum circuit)
        -> Color Head: [B, n_qubits] -> [B, 16] -> [B, 8, 2]

    Primary Output:
        color_estimation: [B, 8, 2] logits for 8 pieces x 2 classes (good/bad)

    Legacy Output (NOT USED):
        gating_value: [B, 1] in range [0, 1] (kept for backward compatibility)

    Args:
        n_qubits: Number of qubits (default: 4)
        n_layers: Number of quantum layers (default: 2)
        embedding: Quantum embedding type (default: 'angle')
        entanglement: Quantum entanglement type (default: 'linear')
        backend: Quantum backend ('pennylane', 'qulacs', 'backprop', default: 'auto')
            - 'auto': Auto-select (PennyLane Lightning > Qulacs)
            - 'pennylane': PennyLane with auto-device selection
            - 'qulacs': Qulacs simulator
            - 'backprop': PennyLane default.qubit with backprop (FASTEST, 500x)
              Uses PyTorch autograd directly, no parameter-shift needed

    Example:
        >>> qnn = ExplicitColorEstimationQNN(n_qubits=4, n_layers=2, backend='auto')
        >>> state = torch.randn(2, 448)  # [batch, 448]
        >>> output = qnn(state)
        >>> print(output['color_estimation'].shape)  # [2, 8, 2]
        >>> print(output['gating_value'].shape)     # [2, 1]
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        embedding: str = 'angle',
        entanglement: str = 'linear',
        backend: str = 'auto'
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.embedding = embedding
        self.entanglement = entanglement
        self.backend = backend

        # Preprocessing: 448D -> 4D (quantum input)
        self.preprocessing = nn.Sequential(
            nn.Linear(448, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),

            nn.Linear(64, n_qubits),
            nn.Tanh()  # Constrain to [-1, 1] for quantum circuit
        )

        # Select quantum backend
        if backend == 'auto':
            # Prefer PennyLane Lightning (faster, better maintained)
            if PENNYLANE_AVAILABLE:
                backend = 'pennylane'
            elif QULACS_AVAILABLE:
                backend = 'qulacs'
            else:
                raise RuntimeError("No quantum backend available. Install PennyLane or Qulacs.")

        # Create quantum layer based on backend
        if backend == 'pennylane':
            if not PENNYLANE_AVAILABLE:
                raise RuntimeError(
                    "PennyLane not available. Install with: pip install pennylane pennylane-lightning")
            self.quantum_layer = QuantumLayer(
                n_qubits=n_qubits,
                n_layers=n_layers,
                embedding=embedding,
                entanglement=entanglement,
                device='auto'  # Auto-select lightning.gpu/lightning.qubit/default.qubit
            )
            self.backend_name = 'PennyLane'
        elif backend == 'backprop':
            # FASTEST: PennyLane with default.qubit and backprop differentiation
            # Uses PyTorch autograd directly - 500x faster than parameter-shift
            if not PENNYLANE_AVAILABLE:
                raise RuntimeError(
                    "PennyLane not available. Install with: pip install pennylane")
            self.quantum_layer = QuantumLayer(
                n_qubits=n_qubits,
                n_layers=n_layers,
                embedding=embedding,
                entanglement=entanglement,
                device='default.qubit.backprop'
            )
            self.backend_name = 'PennyLane-Backprop'
        elif backend == 'qulacs':
            if not QULACS_AVAILABLE:
                raise RuntimeError("Qulacs not available. Install with: pip install qulacs")
            self.quantum_layer = QulacsQuantumLayer(
                n_qubits=n_qubits,
                n_layers=n_layers,
                embedding=embedding,
                entanglement=entanglement
            )
            self.backend_name = 'Qulacs'
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Choose 'auto', 'pennylane', 'qulacs', or 'backprop'.")

        # Head 1: Color estimation (4 -> 16 -> [8, 2])
        self.color_head = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 16)  # 8 pieces × 2 classes
        )

        # Head 2: Gating (LEGACY - NOT USED, kept for backward compatibility)
        self.gating_head = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Constrain to [0, 1]
        )

        print(f"[INFO] ExplicitColorEstimationQNN initialized ({self.backend_name} backend)")
        print(f"[INFO] Input: 448D (7-channel partial obs) -> Quantum: {n_qubits} qubits")
        print("[INFO] Output: Color estimation [8, 2] (8 pieces x 2 classes)")

    def forward(self, state):
        """Forward pass through QNN

        Args:
            state: Input tensor [B, 448] or [448]

        Returns:
            Dict with keys:
            - 'color_estimation': [B, 8, 2] or [8, 2]
            - 'gating_value': [B, 1] or [1]
            - 'q_output': [B, 4] or [4] (raw quantum output)
        """
        # Handle single input (add batch dimension)
        is_single = state.dim() == 1
        if is_single:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]

        # Preprocessing: 448D -> 4D
        preprocessed = self.preprocessing(state)  # [B, 4]

        # DEBUG: Check preprocessed shape
        if preprocessed.shape[-1] != self.n_qubits:
            raise RuntimeError(
                f"Preprocessing output shape mismatch! "
                f"Expected [..., {self.n_qubits}], got {preprocessed.shape}. "
                f"Input state shape: {state.shape}"
            )

        # Quantum layer: 4D -> 4D
        q_output = self.quantum_layer(preprocessed)  # [B, 4]

        # Ensure q_output is on the same device as the rest of the model
        # (PennyLane quantum layers may return CPU tensors even with lightning.gpu)
        if q_output.device != preprocessed.device:
            q_output = q_output.to(preprocessed.device)

        # Head 1: Color estimation [B, 4] -> [B, 16] -> [B, 8, 2]
        color_logits = self.color_head(q_output)  # [B, 16]
        color_logits = color_logits.view(batch_size, 8, 2)  # [B, 8, 2]

        # Head 2: Gating [B, 4] -> [B, 1] in [0, 1]
        gating = self.gating_head(q_output)  # [B, 1]

        # Prepare output dict
        output = {
            'color_estimation': color_logits,
            'gating_value': gating,
            'q_output': q_output
        }

        # Remove batch dimension if input was single
        if is_single:
            output['color_estimation'] = output['color_estimation'].squeeze(0)
            output['gating_value'] = output['gating_value'].squeeze(0)
            output['q_output'] = output['q_output'].squeeze(0)

        return output

    def get_color_probabilities(self, state):
        """Get color probabilities (softmax over logits)

        Args:
            state: Input tensor [B, 448] or [448]

        Returns:
            Color probabilities [B, 8, 2] or [8, 2]
            probabilities[:, :, 0] = P(Good)
            probabilities[:, :, 1] = P(Bad)
        """
        output = self.forward(state)
        color_probs = torch.softmax(output['color_estimation'], dim=-1)
        return color_probs

    def predict_colors(self, state):
        """Predict enemy piece colors (argmax)

        Args:
            state: Input tensor [B, 448] or [448]

        Returns:
            Predicted colors [B, 8] or [8]
            0 = Good, 1 = Bad
        """
        output = self.forward(state)
        color_predictions = torch.argmax(output['color_estimation'], dim=-1)
        return color_predictions

    def get_num_parameters(self):
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_explicit_color_estimation_qnn():
    """Test function for ExplicitColorEstimationQNN"""
    print("=" * 60)
    print("Testing ExplicitColorEstimationQNN")
    print("=" * 60)

    # Create QNN
    qnn = ExplicitColorEstimationQNN(n_qubits=4, n_layers=2)

    # Test single input
    print("\nTest 1: Single input")
    state = torch.randn(448)
    output = qnn(state)
    print(f"Input shape: {state.shape}")
    print(f"Color estimation shape: {output['color_estimation'].shape}")
    print(f"Gating value shape: {output['gating_value'].shape}")
    print(f"Q output shape: {output['q_output'].shape}")
    print(f"Gating value: {output['gating_value'].item():.4f}")
    assert output['color_estimation'].shape == (8, 2), "Color estimation shape mismatch"
    assert output['gating_value'].shape == (1,), "Gating value shape mismatch"

    # Test batch input
    print("\nTest 2: Batch input")
    state_batch = torch.randn(4, 448)
    output_batch = qnn(state_batch)
    print(f"Input shape: {state_batch.shape}")
    print(f"Color estimation shape: {output_batch['color_estimation'].shape}")
    print(f"Gating value shape: {output_batch['gating_value'].shape}")
    assert output_batch['color_estimation'].shape == (
        4, 8, 2), "Batch color estimation shape mismatch"
    assert output_batch['gating_value'].shape == (4, 1), "Batch gating value shape mismatch"

    # Test color probabilities
    print("\nTest 3: Color probabilities")
    color_probs = qnn.get_color_probabilities(state)
    print(f"Color probabilities shape: {color_probs.shape}")
    print(f"Sum per piece: {color_probs.sum(dim=-1)}")
    assert torch.allclose(color_probs.sum(dim=-1), torch.ones(8)), "Probabilities do not sum to 1"

    # Test color predictions
    print("\nTest 4: Color predictions")
    color_preds = qnn.predict_colors(state)
    print(f"Color predictions shape: {color_preds.shape}")
    print(f"Predicted colors: {color_preds}")
    assert color_preds.shape == (8,), "Color predictions shape mismatch"
    assert torch.all((color_preds >= 0) & (color_preds <= 1)), "Invalid color predictions"

    # Test gradient computation
    print("\nTest 5: Gradient computation")
    state_grad = torch.randn(2, 448, requires_grad=True)
    output_grad = qnn(state_grad)
    loss = output_grad['color_estimation'].sum() + output_grad['gating_value'].sum()
    loss.backward()
    print(f"Input gradient computed: {state_grad.grad is not None}")
    assert state_grad.grad is not None, "Gradient not computed"

    # Parameter count
    print("\nTest 6: Parameter count")
    n_params = qnn.get_num_parameters()
    print(f"Total trainable parameters: {n_params:,}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_explicit_color_estimation_qnn()
