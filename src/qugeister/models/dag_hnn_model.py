"""
DAG HNN Model - DAG構造のHNNモデルをconfigから構築・実行する

QuAicのhnn_loader.pyから移植。Colab学習スクリプトでDAG configを使って
学習するために、QuAicリポジトリに依存せず自己完結するモジュール。

Supports:
- 14種類のノードタイプ（input, output, dense, quantum, conv, pooling, etc.）
- split/mergeによるDAG分岐
- トポロジカルソートによるforward実行

Config Format (v2.0 DAG):
{
    "format_version": "2.0",
    "layers": [
        {"id": "inp", "type": "input", "in_features": 448},
        {"id": "d1", "type": "dense", "in_features": 448, "out_features": 128},
        ...
    ],
    "edges": [
        {"source": "inp", "target": "d1"},
        ...
    ]
}
"""

import json
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union

# QuantumLayer: Qugeister_clean版からimport
try:
    from src.qugeister.quantum.quantum_layer import QuantumLayer
    PENNYLANE_AVAILABLE = True
except ImportError:
    try:
        from ..quantum.quantum_layer import QuantumLayer
        PENNYLANE_AVAILABLE = True
    except ImportError:
        PENNYLANE_AVAILABLE = False


# =============================================================================
# ヘルパー関数・クラス
# =============================================================================

def create_activation_layer(activation_type: str) -> nn.Module:
    """活性化関数のファクトリ。文字列からnn.Moduleを生成する。"""
    activation_map = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
        'gelu': nn.GELU(),
        'selu': nn.SELU(),
        'elu': nn.ELU(),
    }
    activation_type = activation_type.lower()
    if activation_type not in activation_map:
        print(f"[WARNING] Unknown activation type '{activation_type}', using ReLU")
        return nn.ReLU()
    return activation_map[activation_type]


def create_pooling_layer(config: Dict[str, Any]) -> nn.Module:
    """プーリングレイヤーのファクトリ。configからMaxPool2d/AvgPool2dを生成する。"""
    pool_type = config.get('pool_type', 'max').lower()
    kernel_size = config.get('pool_kernel', 2)
    stride = config.get('pool_stride') or kernel_size
    padding = config.get('pool_padding', 0)

    if pool_type == 'max':
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    elif pool_type == 'avg':
        return nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    else:
        print(f"[WARNING] Unknown pooling type '{pool_type}', using MaxPool2d")
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)


class MergeLayer(nn.Module):
    """merge用ダミーレイヤー。実際の結合はDAGHNNModel._merge_inputsで行う。"""

    def __init__(self, merge_type: str = 'concat'):
        super().__init__()
        self.merge_type = merge_type.lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ReshapeLayer(nn.Module):
    """テンソルのshapeを変更するレイヤー。"""

    def __init__(self, target_shape: Union[str, List[int]]):
        super().__init__()
        if isinstance(target_shape, str):
            self.target_shape = tuple(int(x.strip()) for x in target_shape.split(','))
        else:
            self.target_shape = tuple(target_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return x.view(batch_size, *self.target_shape)


# =============================================================================
# レイヤーファクトリ（14ノードタイプ対応）
# =============================================================================

def build_layer_from_config(
    layer_config: Dict[str, Any],
    layer_idx: int,
    quantum_device: str = 'auto'
) -> Optional[nn.Module]:
    """configから1つのnn.Moduleレイヤーを構築する。

    Args:
        layer_config: レイヤーの設定辞書
        layer_idx: レイヤーのインデックス（ログ用）
        quantum_device: 量子レイヤーのデバイス
            - 'auto': 推論時（PennyLaneが自動選択）
            - 'default.qubit.backprop': 学習時（勾配計算可能、500x高速）

    Returns:
        nn.Module or None（input/outputノードの場合）
    """
    layer_type = layer_config.get('type', '').lower()

    if layer_type == 'input':
        return None

    elif layer_type == 'output':
        return None

    elif layer_type == 'dense':
        in_features = layer_config.get('in_features')
        out_features = layer_config.get('out_features')
        if in_features is None or out_features is None:
            raise ValueError(f"Dense layer requires 'in_features' and 'out_features': {layer_config}")
        return nn.Linear(in_features, out_features)

    elif layer_type == 'quantum':
        if not PENNYLANE_AVAILABLE:
            raise RuntimeError("PennyLane not available")

        n_qubits = layer_config.get('n_qubits', 4)
        n_layers = layer_config.get('n_layers', 2)
        embedding = layer_config.get('embedding', 'angle')
        entanglement = layer_config.get('entanglement', 'linear')

        return QuantumLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            embedding=embedding,
            entanglement=entanglement,
            device=quantum_device
        )

    elif layer_type == 'conv':
        return nn.Conv2d(
            in_channels=layer_config.get('channels_in', 1),
            out_channels=layer_config.get('channels_out', 8),
            kernel_size=layer_config.get('kernel_size', 3),
            stride=layer_config.get('stride', 1),
            padding=layer_config.get('padding', 0),
            dilation=layer_config.get('dilation', 1)
        )

    elif layer_type == 'pooling':
        return create_pooling_layer(layer_config)

    elif layer_type == 'batchnorm':
        num_features = layer_config.get('batchnorm_dims') or layer_config.get('num_features')
        if num_features is None:
            raise ValueError(f"BatchNorm requires 'batchnorm_dims' or 'num_features': {layer_config}")
        is_2d = layer_config.get('is_2d', False)
        if is_2d:
            return nn.BatchNorm2d(num_features)
        return nn.BatchNorm1d(num_features)

    elif layer_type == 'layernorm':
        num_features = layer_config.get('num_features') or layer_config.get('normalized_shape')
        if num_features is None:
            raise ValueError(f"LayerNorm requires 'num_features' or 'normalized_shape': {layer_config}")
        return nn.LayerNorm(num_features)

    elif layer_type == 'dropout':
        p = layer_config.get('dropout_p', layer_config.get('p', 0.1))
        return nn.Dropout(p=p)

    elif layer_type == 'activation':
        activation_type = layer_config.get('activation_type', layer_config.get('activation', 'relu'))
        if activation_type == 'activation':
            activation_type = layer_config.get('activation', 'relu')
        return create_activation_layer(activation_type)

    elif layer_type == 'split':
        return nn.Identity()

    elif layer_type == 'merge':
        return MergeLayer(layer_config.get('merge_type', 'concat'))

    elif layer_type == 'flatten':
        return nn.Flatten(start_dim=1)

    elif layer_type == 'reshape':
        return ReshapeLayer(layer_config.get('target_shape', '-1'))

    else:
        print(f"[WARNING] Unknown layer type '{layer_type}', skipping")
        return None


# =============================================================================
# DAGHNNModel
# =============================================================================

class DAGHNNModel(nn.Module):
    """DAG構造のHNNモデル。split/mergeノードを含む任意のDAGをforward実行する。

    ConfigBasedHNNModel（Sequential実行）と異なり、各ノードをトポロジカル順に実行し、
    vals辞書パターンで中間テンソルを管理する。

    出力形式: Dict[str, torch.Tensor] で {'color_estimation': [B, 8, 2]} を返す。
    これはExplicitColorEstimationQNNと同じインターフェースなので、
    学習スクリプトのuse_hnn_model=Falseパス（Dict出力）でそのまま動作する。
    """

    def __init__(
        self,
        layers: nn.ModuleDict,
        edges: List[Dict[str, str]],
        layer_configs: Dict[str, Dict[str, Any]],
        n_pieces: int = 8,
    ):
        super().__init__()
        self.layers = layers
        self.n_pieces = n_pieces
        self.layer_configs = layer_configs

        # 隣接リスト（親→子）と逆引き（子→親）を構築
        self.adj: Dict[str, List[str]] = {}
        self.parents: Dict[str, List[str]] = {}
        all_node_ids = set(layer_configs.keys())
        for nid in all_node_ids:
            self.adj[nid] = []
            self.parents[nid] = []
        for edge in edges:
            src, tgt = edge['source'], edge['target']
            self.adj[src].append(tgt)
            self.parents[tgt].append(src)

        # Kahnのアルゴリズムでトポロジカル順序を計算
        self.topo_order = self._compute_topo_order(all_node_ids)

        # input/outputノードを特定
        self.input_node = None
        self.output_node = None
        for nid, cfg in layer_configs.items():
            if cfg.get('type', '').lower() == 'input':
                self.input_node = nid
            elif cfg.get('type', '').lower() == 'output':
                self.output_node = nid

    def _compute_topo_order(self, node_ids: set) -> List[str]:
        """Kahnのアルゴリズムでトポロジカルソートを実行する。

        入次数0のノードから順に処理し、全ノードを依存関係順に並べる。
        サイクルがあればエラーを出す。
        """
        from collections import deque

        indeg = {nid: 0 for nid in node_ids}
        for nid in node_ids:
            for child in self.adj.get(nid, []):
                if child in node_ids:
                    indeg[child] = indeg.get(child, 0) + 1

        # 入次数0のノード（=依存なし）からスタート
        q = deque([nid for nid, d in indeg.items() if d == 0])
        topo: List[str] = []
        while q:
            u = q.popleft()
            topo.append(u)
            for v in self.adj.get(u, []):
                if v not in indeg:
                    continue
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        if len(topo) != len(node_ids):
            raise ValueError("DAGにサイクルが含まれています")
        return topo

    def _merge_inputs(self, node_id: str, parent_tensors: List[torch.Tensor]) -> torch.Tensor:
        """mergeノードの入力テンソルを結合する。merge_typeに応じた結合方式を適用。"""
        cfg = self.layer_configs[node_id]
        merge_type = str(cfg.get('merge_type', 'concat')).lower()

        # エイリアスを正規化
        merge_type = {
            'add': 'sum', 'average': 'mean', 'avg': 'mean',
            'mul': 'multiply', 'product': 'multiply', 'multiple': 'multiply',
        }.get(merge_type, merge_type)

        if merge_type == 'sum':
            result = parent_tensors[0]
            for t in parent_tensors[1:]:
                result = result + t
            return result
        elif merge_type == 'mean':
            result = parent_tensors[0]
            for t in parent_tensors[1:]:
                result = result + t
            return result / len(parent_tensors)
        elif merge_type == 'multiply':
            result = parent_tensors[0]
            for t in parent_tensors[1:]:
                result = result * t
            return result
        elif merge_type == 'max':
            result = parent_tensors[0]
            for t in parent_tensors[1:]:
                result = torch.max(result, t)
            return result
        elif merge_type == 'min':
            result = parent_tensors[0]
            for t in parent_tensors[1:]:
                result = torch.min(result, t)
            return result
        else:
            # concat（デフォルト）: dim=1でテンソルを連結
            return torch.cat(parent_tensors, dim=1)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """DAGのforward実行。vals辞書でノード出力を管理する。"""
        is_single = state.dim() == 1
        if is_single:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]

        # vals辞書: ノードID → 出力テンソル
        vals: Dict[str, torch.Tensor] = {}

        for nid in self.topo_order:
            cfg = self.layer_configs[nid]
            kind = cfg.get('type', '').lower()

            if kind == 'input':
                # 入力ノード: stateをそのまま格納
                vals[nid] = state
                continue

            # 親ノードのテンソルを収集
            parent_ids = self.parents.get(nid, [])
            parent_tensors = [vals[pid] for pid in parent_ids if pid in vals]

            if not parent_tensors:
                continue

            if kind == 'merge':
                # mergeノード: 複数入力を結合
                x = self._merge_inputs(nid, parent_tensors)
            elif kind == 'split':
                # splitノード: パススルー（同じテンソルが複数の子で参照される）
                x = parent_tensors[0]
            elif len(parent_tensors) == 1:
                x = parent_tensors[0]
            else:
                # merge以外で複数親がある場合はconcatをデフォルトに
                x = torch.cat(parent_tensors, dim=1)

            # レイヤーを実行（input/output/split/mergeは実レイヤーなし）
            if nid in self.layers and kind not in ('input', 'output', 'split', 'merge'):
                x = self.layers[nid](x)
                # 量子レイヤーの出力をfloat32に変換
                if PENNYLANE_AVAILABLE and isinstance(self.layers[nid], QuantumLayer) and x.dtype != torch.float32:
                    x = x.float()

            vals[nid] = x

        # 出力ノードのテンソルを取得
        if self.output_node and self.output_node in vals:
            x = vals[self.output_node]
        else:
            # 出力ノードがない場合はトポロジカル順の最後のノード
            x = vals[self.topo_order[-1]]

        # [B, 8, 2] にreshape
        x = x.view(batch_size, self.n_pieces, 2)

        result = {'color_estimation': x}
        if is_single:
            result['color_estimation'] = result['color_estimation'].squeeze(0)

        return result


# =============================================================================
# DAGモデル構築関数
# =============================================================================

def build_dag_model_from_config(
    config: Dict[str, Any],
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    device: str = 'cpu',
    quantum_device: str = 'auto'
) -> DAGHNNModel:
    """DAG構造のHNNモデルをconfigから構築する。

    各レイヤーにidが付いており、edgesで接続関係を指定する。
    build_layer_from_config()を再利用して個別レイヤーを構築する。

    Args:
        config: v2.0形式のDAGモデル設定（layers + edges）
        state_dict: モデルの重み（学習開始時はNone）
        device: PyTorchデバイス
        quantum_device: 量子レイヤーのデバイス
            - 'auto': 推論時
            - 'default.qubit.backprop': 学習時（勾配計算可能）
    """
    layers_config = config.get('layers', [])
    edges = config.get('edges', [])
    if not layers_config:
        raise ValueError("Config must have 'layers' list")
    if not edges:
        raise ValueError("DAG config must have 'edges' list")

    print(f"[INFO] Building DAG model from config (v{config.get('format_version', '2.0')})")

    # ノードID → nn.Moduleの辞書を構築
    module_dict = nn.ModuleDict()
    # ノードID → config辞書のマッピング
    layer_configs_map: Dict[str, Dict[str, Any]] = {}

    for i, layer_config in enumerate(layers_config):
        node_id = layer_config.get('id')
        if node_id is None:
            raise ValueError(f"DAG config: layers[{i}] に 'id' がありません")

        layer_configs_map[node_id] = layer_config

        # レイヤーを構築（input/output/split/mergeはNoneが返る or 特殊レイヤー）
        layer = build_layer_from_config(layer_config, i, quantum_device=quantum_device)
        if layer is not None:
            # nn.ModuleDictはドットを含むキーに対応しないので安全な名前に変換
            safe_key = node_id.replace('.', '_').replace('-', '_')
            module_dict[safe_key] = layer
            print(f"[INFO] Node '{node_id}': {layer_config.get('type')} -> {type(layer).__name__}")

    # 元のnode_idでアクセスできるようにModuleDictを再構築
    id_to_safe = {
        lc.get('id'): lc.get('id', '').replace('.', '_').replace('-', '_')
        for lc in layers_config if lc.get('id')
    }
    final_module_dict = nn.ModuleDict()
    for lc in layers_config:
        nid = lc.get('id')
        safe_key = id_to_safe.get(nid)
        if safe_key and safe_key in module_dict:
            final_module_dict[nid] = module_dict[safe_key]

    model = DAGHNNModel(final_module_dict, edges, layer_configs_map)

    if state_dict:
        model.load_state_dict(state_dict, strict=False)
    model.to(device)

    print(f"[INFO] Built DAGHNNModel with {len(final_module_dict)} layers, {len(edges)} edges")
    return model


# =============================================================================
# ユーティリティ関数
# =============================================================================

def is_dag_config(config: Dict[str, Any]) -> bool:
    """JSON configがDAG形式（v2.0 + edges）かどうかを判定する。

    DAG形式の条件:
    - format_version が '2.' で始まる
    - edgesフィールドが存在し、1つ以上のエッジがある
    - layersの各要素に 'id' フィールドがある
    """
    version = config.get('format_version', '')
    if not str(version).startswith('2.'):
        return False

    edges = config.get('edges', [])
    if not isinstance(edges, list) or len(edges) == 0:
        return False

    # layersの各要素にidがあることを確認
    layers = config.get('layers', [])
    if not layers:
        return False

    return all(lc.get('id') is not None for lc in layers)


def extract_quantum_params_from_dag_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """DAG configからquantumレイヤーのn_qubits/n_layersを抽出する。

    Returns:
        {'n_qubits': int, 'n_layers': int} or {'n_qubits': 4, 'n_layers': 2} (デフォルト)
    """
    layers = config.get('layers', [])
    for lc in layers:
        if lc.get('type', '').lower() == 'quantum':
            return {
                'n_qubits': lc.get('n_qubits', 4),
                'n_layers': lc.get('n_layers', 2),
            }
    # quantumノードが見つからない場合はデフォルト
    return {'n_qubits': 4, 'n_layers': 2}


def load_dag_config(config_path: str) -> Dict[str, Any]:
    """DAG configをJSONファイルから読み込む。

    Args:
        config_path: JSONファイルのパス

    Returns:
        config辞書
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
