import json
import torch
import pennylane as qml
from geister.model_cqcnn import CNN_QNN_CNN_Geister
import os

def save_agent(agent, dir_path: str, model_type: str = "CQCNN"):
    import os
    os.makedirs(dir_path, exist_ok=True)

    if model_type == "CQCNN":
        model = agent.HNN
    elif model_type == "CNN":
        model = agent.NN
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    torch.save(model.state_dict(), os.path.join(dir_path, "weights.pth"))

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
        with open(os.path.join(dir_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

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

def load_cqc_agent(config_path, weight_path, game, player_id="A", AgentClass=None):
    with open(config_path, "r") as f:
        config = json.load(f)

    dev = qml.device(config["dev_type"], wires=config["n_qubits_qnn"])
    model = CNN_QNN_CNN_Geister(dev=dev, **config)
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()

    if AgentClass is None:
        from resource.old_train import CQCAgent_Geister
        AgentClass = CQCAgent_Geister

    agent = AgentClass(
        player_id=player_id,
        game=game,
        dev_qnn=dev,
        embedding_type=config["embedding_type"],
        ansatz_type=config["ansatz_type"],
        n_qubits_qnn=config["n_qubits_qnn"],
        exp_or_prob=config["exp_or_prob"],
        feature_map_reps=config["feature_map_reps"],
        ansatz_reps=config["ansatz_reps"],
        input_channels_cnn=config["input_channels_cnn"],
        board_size_cnn=config["board_size_cnn"],
        cnn_fc_out_features=config["cnn_fc_out_features"]
    )
    agent.HNN.load_state_dict(model.state_dict())
    agent.eval_mode_on()

    return agent
