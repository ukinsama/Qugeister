import multiprocessing as mp
from old_train import run_geister_cqcnn_training_until_epsilon_2
from geister_game import GeisterGame

def train_config(cfg):
    emb = cfg["embedding"]
    ans = cfg["ansatz"]
    name = f"CQCNN_{emb}_{ans}_4qbits_4feat_minigeister"

    game = GeisterGame(board_size=4, num_ghosts_per_player=2)

    run_geister_cqcnn_training_until_epsilon_2(
        epsilon_threshold=0.05,
        n_qbits=4,
        cnn_out_feat=4,
        game_instance=game,
        model_save_dir=f"./trained_models/threads/{name}",
        name_agent1=f"{name}_A",
        name_agent2=f"{name}_B",
        embedding_type=emb,
        ansatz_type=ans
    )

if __name__ == '__main__':
    configs = [
        {"embedding": "AngleEmbedding", "ansatz": "RealAmplitudes"},
        {"embedding": "ZFeatureMap",    "ansatz": "RealAmplitudes"},
        {"embedding": "AngleEmbedding", "ansatz": "EfficientSU2"},
        {"embedding": "ZZFeatureMap",   "ansatz": "EfficientSU2"},
    ]

    processes = []
    for cfg in configs:
        p = mp.Process(target=train_config, args=(cfg,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("✅ すべてのモデル構成の並列学習が完了しました。")
