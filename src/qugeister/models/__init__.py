"""
Models module - QNN and HNN model implementations
"""

from .qnn import ExplicitColorEstimationQNN
from .hnn_config_loader import (
    HNNConfig,
    HNNColorEstimator,
    QuantumConfig,
    load_hnn_config,
    build_model_from_config,
    export_for_competition,
)
from .dag_hnn_model import (
    DAGHNNModel,
    build_dag_model_from_config,
    is_dag_config,
    extract_quantum_params_from_dag_config,
)

__all__ = [
    "ExplicitColorEstimationQNN",
    "HNNConfig",
    "HNNColorEstimator",
    "QuantumConfig",
    "load_hnn_config",
    "build_model_from_config",
    "export_for_competition",
    "DAGHNNModel",
    "build_dag_model_from_config",
    "is_dag_config",
    "extract_quantum_params_from_dag_config",
]
