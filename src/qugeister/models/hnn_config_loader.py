"""
HNN Composer Configuration Loader

Loads QNN configurations exported from hnn-composer and builds
corresponding PyTorch models for training.

Author: Qugeister Team
Date: 2025-11-21
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Import quantum components
try:
    from src.qugeister.quantum.quantum_layer import QuantumLayer
except ImportError:
    from ..quantum.quantum_layer import QuantumLayer


@dataclass
class InputChannelSpec:
    """Input channel specification"""
    index: int
    name: str
    name_ja: str
    size: int
    description: str


@dataclass
class QuantumConfig:
    """Quantum layer configuration"""
    n_qubits: int
    n_layers: int
    embedding: str
    ansatz: str
    measurement: str
    diff_method: str


@dataclass
class HNNConfig:
    """Full HNN Composer configuration"""
    version: str
    task: str
    input_shape: List[int]
    output_shape: List[int]
    n_pieces: int  # Number of opponent pieces (8)
    classical_pre: List[Dict[str, Any]]
    quantum: QuantumConfig
    classical_post: List[Dict[str, Any]]
    training: Dict[str, Any]
    channel_specs: Optional[List[InputChannelSpec]] = None


def load_hnn_config(config_path: str) -> HNNConfig:
    """
    Load configuration from hnn-composer JSON export.

    Supports both:
    - AI Recipe format (module_03_neural_networks)
    - Direct format (architecture at top level)

    Args:
        config_path: Path to JSON configuration file

    Returns:
        HNNConfig dataclass with parsed configuration
    """
    with open(config_path, 'r') as f:
        data = json.load(f)

    # Handle AI Recipe format (module_02_neural_networks or module_03_neural_networks)
    # module_02 is used when learning algorithm tab is removed (current)
    # module_03 is used when learning algorithm tab exists (legacy)
    nn_data = None
    if 'module_02_neural_networks' in data:
        nn_data = data['module_02_neural_networks']
    elif 'module_03_neural_networks' in data:
        nn_data = data['module_03_neural_networks']

    if nn_data is not None:
        # AI Recipe format: parse from module data
        input_data = nn_data.get('input', {})
        output_data = nn_data.get('output', {})
        network_data = nn_data.get('network', {})

        # Get architecture from network (either graph_structure or architecture)
        if 'graph_structure' in network_data:
            # Extract from graph nodes
            arch_data = _extract_architecture_from_graph(network_data['graph_structure'])
        elif 'architecture' in network_data:
            arch_data = network_data['architecture']
        else:
            arch_data = {}
    else:
        # Direct format
        input_data = data.get('input', {})
        output_data = data.get('output', {})

        # Check for network with nodes/edges at top level (HNN Composer export format)
        if 'network' in data and 'nodes' in data['network']:
            # HNN Composer graph format at top level
            arch_data = _extract_architecture_from_graph(data['network'])
        else:
            # Traditional architecture dict format
            arch_data = data.get('architecture', {})

    # Parse quantum config
    q_data = arch_data.get('quantum', {})
    quantum_config = QuantumConfig(
        n_qubits=q_data.get('n_qubits', 4),
        n_layers=q_data.get('n_layers', 2),
        embedding=q_data.get('embedding', 'AngleEmbedding'),
        ansatz=q_data.get('ansatz', 'BasicEntanglerLayers'),
        measurement=q_data.get('measurement', 'expval_PauliZ'),
        diff_method=q_data.get('diff_method', 'best'),
    )

    # Parse channel specs if present
    channel_specs = None
    if 'channel_specs' in input_data:
        channel_specs = [
            InputChannelSpec(
                index=cs['index'],
                name=cs['name'],
                name_ja=cs.get('name_ja', ''),
                size=cs['size'],
                description=cs.get('description', ''),
            )
            for cs in input_data['channel_specs']
        ]

    # Get input/output shapes
    input_shape = input_data.get('shape', [448])
    output_shape = output_data.get('shape', [8, 2])

    # Get training config (different location in AI Recipe vs Direct format)
    if nn_data is not None:
        training_data = nn_data.get('training', {})
    else:
        training_data = data.get('training', {})

    return HNNConfig(
        version=data.get('version', data.get('schema_version', '1.0')),
        task=data.get('task', 'geister_8x8_color_estimation'),
        input_shape=input_shape,
        output_shape=output_shape,
        n_pieces=output_data.get('n_pieces', 8),
        classical_pre=arch_data.get('classical_pre', []),
        quantum=quantum_config,
        classical_post=arch_data.get('classical_post', []),
        training=training_data,
        channel_specs=channel_specs,
    )


def _extract_architecture_from_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract architecture configuration from HNN Composer graph structure.

    Traces edges to properly split layers between pre-quantum and post-quantum.

    Args:
        graph: Graph structure with nodes and edges

    Returns:
        Architecture dict with classical_pre, quantum, classical_post
    """
    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])

    # Build node lookup and edge maps
    node_map = {node['id']: node for node in nodes}
    outgoing = {}  # node_id -> list of target node_ids
    incoming = {}  # node_id -> list of source node_ids

    for edge in edges:
        src = edge.get('source')
        tgt = edge.get('target')
        if src and tgt:
            outgoing.setdefault(src, []).append(tgt)
            incoming.setdefault(tgt, []).append(src)

    # Find quantum node
    quantum_node_id = None
    for node in nodes:
        if node.get('type') == 'quantum':
            quantum_node_id = node['id']
            break

    # Build architecture
    arch = {
        'classical_pre': [],
        'quantum': {},
        'classical_post': []
    }

    # Extract quantum config
    if quantum_node_id:
        quantum_node = node_map[quantum_node_id]
        node_data = quantum_node.get('data', {})
        q_circuit = node_data.get('qCircuit', {})
        arch['quantum'] = {
            'n_qubits': node_data.get('n_qubits', 4),
            'n_layers': 2,  # Default
            'embedding': q_circuit.get('embeddingTemplateName', 'AngleEmbedding'),
            'ansatz': q_circuit.get('ansatzTemplateName', 'BasicEntanglerLayers'),
            'measurement': 'expval_PauliZ',
            'diff_method': 'best',
        }

        # Trace path from merge to quantum (pre-layers)
        # Find the dense node(s) between merge and quantum
        pre_dense_nodes = []
        visited = set()

        def find_pre_nodes(node_id):
            if node_id in visited or node_id == quantum_node_id:
                return
            visited.add(node_id)
            node = node_map.get(node_id)
            if node and node.get('type') == 'dense':
                pre_dense_nodes.append(node)
            # Continue backward
            for src in incoming.get(node_id, []):
                find_pre_nodes(src)

        # Start from quantum node and trace backward
        for src in incoming.get(quantum_node_id, []):
            find_pre_nodes(src)

        # Trace path from quantum to output (post-layers)
        post_dense_nodes = []
        visited = set()

        def find_post_nodes(node_id):
            if node_id in visited or node_id == quantum_node_id:
                return
            visited.add(node_id)
            node = node_map.get(node_id)
            if node and node.get('type') == 'dense':
                post_dense_nodes.append(node)
            # Continue forward
            for tgt in outgoing.get(node_id, []):
                find_post_nodes(tgt)

        # Start from quantum node and trace forward
        for tgt in outgoing.get(quantum_node_id, []):
            find_post_nodes(tgt)

        # Build pre-layers (only the last dense before quantum matters)
        if pre_dense_nodes:
            # Get the dense node directly connected to quantum
            dense = pre_dense_nodes[0]  # Closest to quantum
            node_data = dense.get('data', {})
            out_features = node_data.get('out_features', arch['quantum']['n_qubits'])
            arch['classical_pre'].append({
                'type': 'linear',
                'out_features': out_features
            })

        # Build post-layers
        if post_dense_nodes:
            dense = post_dense_nodes[0]  # First after quantum
            node_data = dense.get('data', {})
            out_features = node_data.get('out_features', 16)
            arch['classical_post'].append({
                'type': 'linear',
                'out_features': out_features
            })
        else:
            # Default post-processing for [8, 2] output
            arch['classical_post'] = [
                {'type': 'linear', 'out_features': 16}
            ]

    return arch


class HNNColorEstimator(nn.Module):
    """
    Color Estimator built from hnn-composer configuration.

    Architecture:
        Input -> Classical Pre -> Quantum -> Classical Post -> Output
    """

    def __init__(self, config: HNNConfig, device: str = 'cpu', backend: str = 'backprop'):
        """
        Args:
            config: HNN configuration
            device: PyTorch device ('cpu' or 'cuda')
            backend: Quantum backend ('auto', 'backprop')
                - 'backprop': PennyLane default.qubit with backprop (FASTEST, 500x)
                - 'auto': Auto-select (may be slow with parameter-shift)
        """
        super().__init__()
        self.config = config
        self.device = device
        self.backend = backend
        self.n_pieces = config.n_pieces  # 8 opponent pieces
        self._is_eval = False  # Cache eval state for estimate_* methods

        # Build classical pre-processing layers
        self.pre_layers = self._build_classical_layers(
            config.classical_pre,
            config.input_shape[0]
        )

        # Calculate quantum input size
        pre_out = self._get_output_size(config.classical_pre, config.input_shape[0])

        # Build quantum layer
        self.quantum_layer = self._build_quantum_layer(config.quantum, pre_out, backend)

        # Build classical post-processing layers
        # Output: n_pieces * 2 = 16 (8 pieces x 2 classes: good/bad)
        quantum_out = config.quantum.n_qubits
        self.post_layers = self._build_classical_layers(
            config.classical_post,
            quantum_out,
            final_output=config.n_pieces * 2  # [8, 2] flattened = 16
        )

        # Register buffer for target device tracking (avoids repeated device checks)
        self.register_buffer('_device_tracker', torch.zeros(1))

    def _build_classical_layers(
        self,
        layer_configs: List[Dict[str, Any]],
        input_size: int,
        final_output: Optional[int] = None
    ) -> nn.Sequential:
        """Build classical neural network layers"""
        layers = []
        current_size = input_size

        for i, layer_cfg in enumerate(layer_configs):
            layer_type = layer_cfg.get('type', 'linear')

            if layer_type == 'linear':
                out_features = layer_cfg.get('out_features')
                if out_features is None:
                    # Use final_output for last layer
                    out_features = final_output or current_size

                layers.append(nn.Linear(current_size, out_features))
                current_size = out_features

            elif layer_type == 'relu':
                layers.append(nn.ReLU())

            elif layer_type == 'dropout':
                p = layer_cfg.get('p', 0.1)
                layers.append(nn.Dropout(p))

            elif layer_type == 'batchnorm':
                layers.append(nn.BatchNorm1d(current_size))

            elif layer_type == 'softmax':
                layers.append(nn.Softmax(dim=-1))

        return nn.Sequential(*layers)

    def _get_output_size(self, layer_configs: List[Dict[str, Any]], input_size: int) -> int:
        """Calculate output size after classical layers"""
        current_size = input_size
        for layer_cfg in layer_configs:
            if layer_cfg.get('type') == 'linear':
                out_features = layer_cfg.get('out_features')
                if out_features:
                    current_size = out_features
        return current_size

    def _build_quantum_layer(self, q_config: QuantumConfig, input_size: int, backend: str = 'backprop') -> QuantumLayer:
        """Build quantum variational layer

        Note: QuantumLayer expects input dim = n_qubits.
        The classical pre-processing should reduce to n_qubits.

        Args:
            q_config: Quantum configuration
            input_size: Input size (should equal n_qubits)
            backend: 'backprop' (500x faster) or 'auto'
        """
        # Map embedding name to type
        embedding = 'angle'
        if 'amplitude' in q_config.embedding.lower():
            embedding = 'amplitude'

        # Select device based on backend
        if backend == 'backprop':
            qdevice = 'default.qubit.backprop'
        else:
            qdevice = 'auto'

        return QuantumLayer(
            n_qubits=q_config.n_qubits,
            n_layers=q_config.n_layers,
            embedding=embedding,
            device=qdevice,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, n_pieces, 2) = (B, 8, 2)
            Logits for each piece's color (good=0, bad=1)
        """
        # Classical pre-processing
        x = self.pre_layers(x)

        # Quantum layer
        x = self.quantum_layer(x)

        # Move to model's device if needed (uses registered buffer for fast device check)
        target_device = self._device_tracker.device
        if x.device != target_device:
            x = x.to(target_device)

        # Classical post-processing
        x = self.post_layers(x)  # [B, 16]

        # Reshape to [B, 8, 2] for per-piece classification
        batch_size = x.shape[0]
        x = x.view(batch_size, self.n_pieces, 2)

        return x

    def estimate_colors(self, observation: np.ndarray) -> np.ndarray:
        """
        Estimate opponent piece colors.

        Compatible with QuAic competition interface.

        Args:
            observation: Game observation array (448,) or batch (N, 448)

        Returns:
            Predicted colors for 8 pieces: [8] or [N, 8] with values 0=good, 1=bad
        """
        # Only call eval() if not already in eval mode (avoid overhead)
        if self.training:
            self.eval()

        with torch.no_grad():
            # Efficient numpy to tensor conversion
            if isinstance(observation, np.ndarray):
                x = torch.from_numpy(observation.astype(np.float32))
            else:
                x = torch.as_tensor(observation, dtype=torch.float32)

            # Move to device
            target_device = self._device_tracker.device
            x = x.to(target_device, non_blocking=True)

            is_single = x.dim() == 1
            if is_single:
                x = x.unsqueeze(0)

            logits = self.forward(x)  # [B, 8, 2]
            predictions = logits.argmax(dim=-1)  # [B, 8]

            result = predictions.cpu().numpy()
            return result.squeeze() if is_single else result

    def estimate_color_probs(self, observation: np.ndarray) -> np.ndarray:
        """
        Get color probabilities for each piece.

        Args:
            observation: Game observation array (448,) or batch (N, 448)

        Returns:
            Probability distribution for each piece: [8, 2] or [N, 8, 2]
            probs[i, 0] = P(piece i is good)
            probs[i, 1] = P(piece i is bad)
        """
        # Only call eval() if not already in eval mode
        if self.training:
            self.eval()

        with torch.no_grad():
            # Efficient numpy to tensor conversion
            if isinstance(observation, np.ndarray):
                x = torch.from_numpy(observation.astype(np.float32))
            else:
                x = torch.as_tensor(observation, dtype=torch.float32)

            # Move to device
            target_device = self._device_tracker.device
            x = x.to(target_device, non_blocking=True)

            is_single = x.dim() == 1
            if is_single:
                x = x.unsqueeze(0)

            logits = self.forward(x)  # [B, 8, 2]
            probs = torch.softmax(logits, dim=-1)  # [B, 8, 2]

            result = probs.cpu().numpy()
            return result.squeeze() if is_single else result

    def load_weights(self, path: str):
        """Load model weights from file"""
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.eval()

    def save_weights(self, path: str):
        """Save model weights to file"""
        torch.save(self.state_dict(), path)


def build_model_from_config(config_path: str, device: str = 'cpu', backend: str = 'backprop') -> HNNColorEstimator:
    """
    Build QNN model from hnn-composer configuration file.

    Args:
        config_path: Path to JSON configuration
        device: Device to use ('cpu' or 'cuda')
        backend: Quantum backend ('backprop' for 500x faster, 'auto' for default)

    Returns:
        HNNColorEstimator model ready for training
    """
    config = load_hnn_config(config_path)
    model = HNNColorEstimator(config, device=device, backend=backend)
    model.to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model built from: {config_path}")
    print(f"  Task: {config.task}")
    print(f"  Input: {config.input_shape}")
    print(f"  Output: {config.output_shape}")
    print(f"  Quantum: {config.quantum.n_qubits}q x {config.quantum.n_layers}L")
    print(f"  Total params: {total_params}")
    print(f"  Trainable params: {trainable_params}")

    return model


# Competition export helper
def export_for_competition(
    model: HNNColorEstimator,
    config: HNNConfig,
    output_path: str
):
    """
    Export trained model for QuAic competition submission.

    Creates a ZIP file with:
        - model.py: ColorEstimator class with PennyLane quantum layer
        - weights.pth: Model weights
        - config.json: Model configuration

    Args:
        model: Trained HNNColorEstimator
        config: Model configuration
        output_path: Output ZIP file path
    """
    import zipfile
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save weights
        weights_path = tmpdir / 'weights.pth'
        model.save_weights(str(weights_path))

        # Save config
        config_path = tmpdir / 'config.json'
        config_dict = {
            'version': config.version,
            'task': config.task,
            'input': {'shape': config.input_shape},
            'output': {'shape': config.output_shape, 'n_pieces': config.n_pieces},
            'architecture': {
                'classical_pre': config.classical_pre,
                'quantum': {
                    'n_qubits': config.quantum.n_qubits,
                    'n_layers': config.quantum.n_layers,
                    'embedding': config.quantum.embedding,
                    'ansatz': config.quantum.ansatz,
                },
                'classical_post': config.classical_post,
            },
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Create model.py with PennyLane quantum layer (batch-optimized)
        model_code = '''"""
Color Estimator for QuAic Competition
Uses PennyLane for quantum computation with batch processing.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# PennyLane quantum layer
import pennylane as qml

class QuantumLayer(nn.Module):
    """PennyLane variational quantum circuit layer with batch processing"""

    def __init__(self, n_qubits: int, n_layers: int, embedding: str = 'angle'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.embedding = embedding

        # Create quantum device
        self.dev = qml.device('default.qubit', wires=n_qubits)

        # Trainable quantum parameters
        self.quantum_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        # Create batch-compatible quantum circuit using parameter broadcasting
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit_batched(inputs, weights):
            # Encoding - use [..., i] for batch compatibility
            for i in range(n_qubits):
                qml.RY(inputs[..., i], wires=i)

            # Variational layers
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)

                # Entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])

            # Return tuple for broadcasting support
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))

        self.circuit = circuit_batched

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Batch-optimized forward pass using parameter broadcasting"""
        is_single = x.dim() == 1
        if is_single:
            x = x.unsqueeze(0)

        # Execute with parameter broadcasting (no loop needed)
        result_tuple = self.circuit(x, self.quantum_weights)
        y = torch.stack(result_tuple, dim=-1)

        # Convert to float32 if needed
        if y.dtype != torch.float32:
            y = y.float()

        if is_single:
            y = y.squeeze(0)

        return y


class ColorEstimator:
    """Color Estimator for QuAic competition - 8 piece binary classification"""

    def __init__(self):
        # Load config
        config_path = Path(__file__).parent / 'config.json'
        with open(config_path) as f:
            self.config = json.load(f)

        # Build model
        self.device = 'cpu'
        self.n_pieces = self.config['output'].get('n_pieces', 8)
        self.model = self._build_model()

    def _build_model(self):
        layers = []

        # Pre layers
        input_size = self.config['input']['shape'][0]
        for layer_cfg in self.config['architecture']['classical_pre']:
            layer_type = layer_cfg.get('type', 'linear')
            if layer_type == 'linear':
                out_size = layer_cfg.get('out_features', input_size)
                layers.append(nn.Linear(input_size, out_size))
                input_size = out_size
            elif layer_type == 'relu':
                layers.append(nn.ReLU())
            elif layer_type == 'batchnorm':
                layers.append(nn.BatchNorm1d(input_size))
            elif layer_type == 'dropout':
                p = layer_cfg.get('p', 0.1)
                layers.append(nn.Dropout(p))

        self.pre_layers = nn.Sequential(*layers)

        # Quantum layer (actual PennyLane)
        q_cfg = self.config['architecture']['quantum']
        embedding = 'angle'
        if 'amplitude' in q_cfg.get('embedding', '').lower():
            embedding = 'amplitude'

        self.quantum_layer = QuantumLayer(
            n_qubits=q_cfg['n_qubits'],
            n_layers=q_cfg['n_layers'],
            embedding=embedding
        )
        input_size = q_cfg['n_qubits']

        # Post layers
        post_layers = []
        output_size = self.n_pieces * 2  # 8 pieces x 2 classes
        for layer_cfg in self.config['architecture']['classical_post']:
            layer_type = layer_cfg.get('type', 'linear')
            if layer_type == 'linear':
                out_size = layer_cfg.get('out_features', output_size)
                post_layers.append(nn.Linear(input_size, out_size))
                input_size = out_size
            elif layer_type == 'relu':
                post_layers.append(nn.ReLU())
            elif layer_type == 'softmax':
                pass  # Skip softmax, we use argmax for predictions

        self.post_layers = nn.Sequential(*post_layers)

        return nn.ModuleList([self.pre_layers, self.quantum_layer, self.post_layers])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning [B, 8, 2] logits"""
        x = self.pre_layers(x)
        x = self.quantum_layer(x)
        x = self.post_layers(x)

        # Reshape to [B, 8, 2]
        batch_size = x.shape[0]
        x = x.view(batch_size, self.n_pieces, 2)
        return x

    def estimate_colors(self, observation: np.ndarray) -> np.ndarray:
        """
        Estimate opponent piece colors.

        Args:
            observation: Game observation array (448,) or batch (N, 448)

        Returns:
            Predicted colors for 8 pieces: [8] or [N, 8] with values 0=good, 1=bad
        """
        # Only call eval() if not already in eval mode
        if self.pre_layers.training:
            self.model.eval()

        with torch.no_grad():
            # Efficient numpy to tensor conversion
            if isinstance(observation, np.ndarray):
                x = torch.from_numpy(observation.astype(np.float32))
            else:
                x = torch.as_tensor(observation, dtype=torch.float32)

            is_single = x.dim() == 1
            if is_single:
                x = x.unsqueeze(0)

            logits = self.forward(x)  # [B, 8, 2]
            predictions = logits.argmax(dim=-1)  # [B, 8]

            result = predictions.numpy()
            return result.squeeze() if is_single else result

    def load_weights(self, path: str = None):
        """Load model weights from file"""
        if path is None:
            path = str(Path(__file__).parent / 'weights.pth')
        state_dict = torch.load(path, map_location='cpu', weights_only=True)

        # Load weights for each component
        pre_state = {k.replace('pre_layers.', ''): v
                     for k, v in state_dict.items() if k.startswith('pre_layers.')}
        quantum_state = {k.replace('quantum_layer.', ''): v
                        for k, v in state_dict.items() if k.startswith('quantum_layer.')}
        post_state = {k.replace('post_layers.', ''): v
                     for k, v in state_dict.items() if k.startswith('post_layers.')}

        if pre_state:
            self.pre_layers.load_state_dict(pre_state)
        if quantum_state:
            self.quantum_layer.load_state_dict(quantum_state)
        if post_state:
            self.post_layers.load_state_dict(post_state)

    def reset(self):
        """Reset internal state (not needed for stateless model)"""
        pass
'''

        model_path = tmpdir / 'model.py'
        with open(model_path, 'w') as f:
            f.write(model_code)

        # Create ZIP
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(weights_path, 'weights.pth')
            zf.write(config_path, 'config.json')
            zf.write(model_path, 'model.py')

    print(f"Submission exported to: {output_path}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hnn_config_loader.py <config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    model = build_model_from_config(config_path)
    print("\nModel structure:")
    print(model)
