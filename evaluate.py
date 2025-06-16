# evaluate.py
import torch.nn as nn
import pennylane as qml
from model_cqcnn import QNNComponent  # Assuming this is defined in qnn_component.py
from geister_game import BOARD_SIZE, GeisterGame # Assuming BOARD_SIZE is defined in geister_game.py
# --- CNN Model (CCNN2_Geister) ---
from geister_game import GeisterGame
from train import Env_Geister, RandomPolicy
import torch

def evaluate_model(agent, episodes=100, player_id="A", board_size=6):
    """指定したエージェントをランダムポリシーと対戦させて評価する関数"""
    print(f"--- Evaluating Agent {player_id} vs RandomPolicy ---")
    game = GeisterGame(board_size=board_size)

    agent.eval_mode_on()
    agent.epsilon = 0.0

    if player_id == "A":
        eval_env = Env_Geister(agent, RandomPolicy("B", game), game)
    else:
        eval_env = Env_Geister(RandomPolicy("A", game), agent, game)

    eval_env.start_training(
        episodes=episodes,
        visualize_interval=0,
        train_agents=False
    )

def run_evaluation_suite(agent_list, episodes=100):
    results = {}

    for i, agent_a in enumerate(agent_list):
        for j, agent_b in enumerate(agent_list):
            if i == j:
                continue  # 自己対戦はスキップ
            print(f"Evaluating {agent_a.__class__.__name__} vs {agent_b.__class__.__name__}")
            game = GeisterGame()
            env = Env_Geister(agent_a, agent_b, game)
            env.start_training(episodes, visualize_interval=0, train_agents=False)

            key = f"{agent_a.__class__.__name__} vs {agent_b.__class__.__name__}"
            results[key] = "完了"  # 勝敗記録するならカウント追加

    return results