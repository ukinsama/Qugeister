# train.py
import os
import abc
import random
import json
from geister_game import GeisterGame
import numpy as np
from model_cqcnn import CCNN2_Geister
from model_cqcnn import CNN_QNN_CNN_Geister
import torch
import torch.optim as optim
import torch.nn as nn
import pennylane as qml
dev_qnn_global = None # グローバル変数として量子デバイスを保持
# --- エージェントクラス (Agent, Human, RandomPolicy, CNNAgent_Geister, CQCAgent_Geister) ---
class Agent(abc.ABC):
    def __init__(self, player_id: str, game: GeisterGame):
        self.player_id = player_id
        self.game = game
        self.eval_mode = True # Start in eval mode

    def train_mode_on(self): # Changed name
        self.eval_mode = False

    def eval_mode_on(self): # Changed name
        self.eval_mode = True

    def check_state(self) -> torch.tensor:
        return self.game.get_state(self.player_id)

    def check_actions(self) -> list:
        return self.game.get_possible_actions(self.player_id)
    
    @abc.abstractmethod
    def action(self, possible_moves: list) -> tuple: # Returns ((fr,fc),(tr,tc)) or None
        pass

class Human(Agent):
    def __init__(self, player_id: str, game: GeisterGame):
        super().__init__(player_id, game)
        self.eval_mode_on()

    def action(self, possible_moves: list) -> tuple:
        if not possible_moves:
            print("選択可能な手がありません。")
            return None

        print(f"\n🎲 プレイヤー {self.player_id} の番です。選択可能な手:")
        for i, ((fr, fc), (tr, tc)) in enumerate(possible_moves):
            piece = self.game.board.get_piece(fr, fc)
            kind = self.game.get_kind_of_piece(piece) if piece != EMPTY else '?'
            print(f" {i}: ({fr},{fc}) の {kind}オバケ → ({tr},{tc})")

        print("💡 番号で手を選んでください。 q を入力すると中断できます。")

        while True:
            try:
                choice_str = input(f"選択 (0-{len(possible_moves)-1}, q=中断): ").strip()
                if choice_str.lower() == 'q':
                    print("中断が選ばれました。")
                    return None
                if not choice_str:
                    print("⚠ 入力が空です。番号を入力してください。")
                    continue
                choice = int(choice_str)
                if 0 <= choice < len(possible_moves):
                    return possible_moves[choice]
                else:
                    print(f"⚠ 無効な番号です。0～{len(possible_moves)-1} の間で入力してください。")
            except ValueError:
                print("⚠ 数字または q を入力してください。")
            except Exception as e:
                print(f"⚠ 予期せぬエラー: {e}")

class RandomPolicy(Agent):
    def __init__(self, player_id: str, game: GeisterGame):
        super().__init__(player_id, game)
        self.eval_mode_on()
        self.episode_count = 0  # ← これを追加

    def update(self, *args, **kwargs):
        return None  # 何もしない

    def action(self, possible_moves: list) -> tuple:
        if not possible_moves: return None
        return random.choice(possible_moves)

class CNNAgent_Geister(Agent):
    def __init__(self, player_id:str, game:GeisterGame, network_name="CCNN2_Geister", 
                 board_size=None, input_channels=6, elo=1500, epsilon=0.5, lr=0.001): # input_channelsを6に変更
        if board_size is None:
            board_size = game.board_size
        super().__init__(player_id, game)
        self.board_size = board_size
        self.discount = 0.99
        self.epsilon_start = epsilon
        self.epsilon_end = 0.01
        self.epsilon_decay = 5000 # エピソード数で減衰
        self.epsilon = epsilon
        self.lr = lr
        self.alpha = 0.0 # L2正則化なしに一旦変更
        self.input_channels = input_channels # 状態表現のチャネル数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.elo = elo
        self.episode_count = 0 # epsilon減衰用

        if network_name == "CCNN2_Geister":
            self.NN = CCNN2_Geister(input_channels=self.input_channels, board_size=self.board_size, num_outputs=self.board_size*self.board_size).to(self.device)
        else:
            raise ValueError(f"Unknown network name: {network_name}")
        
        self.optimizer = optim.Adam(self.NN.parameters(), lr=self.lr)
        self.eval_mode_on()

    def get_qvalues_map(self, state_tensor: torch.tensor) -> torch.tensor:
        state_tensor = state_tensor.unsqueeze(0).to(self.device)
        qvalues_map = self.NN(state_tensor)
        return qvalues_map.squeeze(0)

    def _get_the_best_action_and_qvalue(self, state_tensor: torch.tensor, possible_moves: list):
        if not possible_moves:
            return None, -float('inf')
        with torch.no_grad():
            qvalues_map = self.get_qvalues_map(state_tensor)
        best_move = None
        best_q_value = -float('inf')
        # 複数の手が同じ最大Q値を持つ場合、ランダムに選ぶ
        best_moves_candidates = []

        for move in possible_moves:
            _from_pos, to_pos = move
            to_r, to_c = to_pos
            q_value = qvalues_map[to_r, to_c].item()
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_moves_candidates = [move]
            elif q_value == best_q_value:
                best_moves_candidates.append(move)
        
        if best_moves_candidates:
            best_move = random.choice(best_moves_candidates)
        elif possible_moves : # 全てのQ値が-infの場合など（通常はありえない）
             best_move = random.choice(possible_moves)
             _from_pos, to_pos = best_move # best_q_valueは-infのまま
             best_q_value = qvalues_map[to_pos[0], to_pos[1]].item()


        return best_move, best_q_value

    def action(self, possible_moves: list) -> tuple:
        current_state_tensor = self.check_state()
        
        # epsilon decay
        if not self.eval_mode:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) *\
                           np.exp(-1. * self.episode_count / self.epsilon_decay)

        if not self.eval_mode and random.random() < self.epsilon:
            if not possible_moves: return None
            return random.choice(possible_moves)
        else:
            best_action, _ = self._get_the_best_action_and_qvalue(current_state_tensor, possible_moves)
            return best_action

    def update(self, state_tensor: torch.tensor, action: tuple, reward: float, next_state_tensor: torch.tensor, done: bool):
        if self.eval_mode: return None
        if action is None: return None # 行動がない場合は更新スキップ

        _from_pos, to_pos = action
        to_r, to_c = to_pos
        
        q_value_map_current = self.get_qvalues_map(state_tensor)
        old_qvalue = q_value_map_current[to_r, to_c]

        target_q_value = torch.tensor(reward, dtype=torch.float32, device=self.device)
        
        if not done:
            with torch.no_grad():
                # 次の状態 s' における次の手番プレイヤーの最適な行動のQ値 (max_a' Q(s',a'))
                # 相手の手番だとしても、自分のNNを使って相手の最善手を推定し、その負の値を使う
                next_player_id = self.game.current_player # Make_move後なので、相手の手番になっている
                next_possible_moves = self.game.get_possible_actions(next_player_id)
                
                if not next_possible_moves: # 相手が動けない -> 自分の勝ちが確定したようなもの
                    max_next_q_for_opponent = 0.0 # 相手はこれ以上Q値を改善できない
                else:
                    # 相手も同じ戦略(このNN)を使うと仮定。相手視点の状態を取得。
                    next_state_tensor_opponent_view = self.game.get_state(next_player_id)
                    _ , max_next_q_for_opponent = self._get_the_best_action_and_qvalue(next_state_tensor_opponent_view, next_possible_moves)
                    if max_next_q_for_opponent == -float('inf'): max_next_q_for_opponent = 0.0

                # 相手の最大Q値の符号を反転させたものが、自分にとっての次の状態の価値
                target_q_value += self.discount * (-max_next_q_for_opponent)
        
        loss = nn.functional.huber_loss(old_qvalue, target_q_value.detach())
        if self.alpha > 0:
            l2 = torch.tensor(0., requires_grad=True, device=self.device)
            for w in self.NN.parameters(): l2 = l2 + torch.norm(w)**2
            loss = loss + self.alpha * l2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

class CQCAgent_Geister(Agent):
    def __init__(self, player_id:str, game:GeisterGame, dev_qnn,
                 embedding_type: str, ansatz_type: str, n_qubits_qnn=4, 
                 exp_or_prob="exp", feature_map_reps=1, ansatz_reps=1,
                 input_channels_cnn=6, board_size_cnn=None, 
                 cnn_fc_out_features=4, # QNNへの入力特徴数 (n_qubits_qnnと一致させる想定)
                 elo=1500, epsilon=0.8, lr=0.001):
        if board_size is None:
            board_size = game.board_size
        super().__init__(player_id, game)
        self.discount = 0.99
        self.epsilon_start = epsilon
        self.epsilon_end = 0.01
        self.epsilon_decay = 5000
        self.epsilon = epsilon
        self.lr = lr
        self.board_size = board_size_cnn
        self.input_channels = input_channels_cnn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dev_qnn = dev_qnn
        self.elo = elo
        self.episode_count = 0

        self.HNN = CNN_QNN_CNN_Geister(
            dev=self.dev_qnn, embedding_type=embedding_type, ansatz_type=ansatz_type,
            n_qubits_qnn=n_qubits_qnn, exp_or_prob=exp_or_prob,
            feature_map_reps=feature_map_reps, ansatz_reps=ansatz_reps,
            input_channels_cnn=self.input_channels, board_size_cnn=self.board_size,
            cnn_fc_out_features=cnn_fc_out_features,
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.HNN.parameters(), lr=self.lr)
        self.eval_mode_on()

    def get_qvalues_map(self, state_tensor: torch.tensor) -> torch.tensor:
        state_tensor = state_tensor.unsqueeze(0).to(self.device)
        qvalues_map = self.HNN(state_tensor)
        return qvalues_map.squeeze(0)

    def _get_the_best_action_and_qvalue(self, state_tensor: torch.tensor, possible_moves: list):
        # (CNNAgent_Geister と同じロジック)
        if not possible_moves: return None, -float('inf')
        with torch.no_grad(): qvalues_map = self.get_qvalues_map(state_tensor)
        best_move = None; best_q_value = -float('inf'); best_moves_candidates = []
        for move in possible_moves:
            _from_pos, to_pos = move; to_r, to_c = to_pos
            q_value = qvalues_map[to_r, to_c].item()
            if q_value > best_q_value: best_q_value = q_value; best_moves_candidates = [move]
            elif q_value == best_q_value: best_moves_candidates.append(move)
        if best_moves_candidates: best_move = random.choice(best_moves_candidates)
        elif possible_moves:
             best_move = random.choice(possible_moves)
             _from_pos, to_pos = best_move
             best_q_value = qvalues_map[to_pos[0], to_pos[1]].item()
        return best_move, best_q_value
        
    def action(self, possible_moves: list) -> tuple:
        current_state_tensor = self.check_state()

        # ε の減衰（エピソード数に基づく）
        if not self.eval_mode:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        np.exp(-1. * self.episode_count / self.epsilon_decay)

        # ε-greedy に基づくランダム行動選択
        if not self.eval_mode and random.random() < self.epsilon:
            if not possible_moves:
                return None
            return random.choice(possible_moves)

        # Q値に基づく最良行動選択
        else:
            best_action, _ = self._get_the_best_action_and_qvalue(current_state_tensor, possible_moves)
            return best_action

    def update(self, state_tensor: torch.tensor, action: tuple, reward: float, next_state_tensor: torch.tensor, done: bool):
        # (CNNAgent_Geister と同じロジック)
        if self.eval_mode: return None
        if action is None: return None
        _from_pos, to_pos = action; to_r, to_c = to_pos
        q_value_map_current = self.get_qvalues_map(state_tensor)
        old_qvalue = q_value_map_current[to_r, to_c]
        target_q_value = torch.tensor(reward, dtype=torch.float32, device=self.device)
        if not done:
            with torch.no_grad():
                next_player_id = self.game.current_player 
                next_possible_moves = self.game.get_possible_actions(next_player_id)
                if not next_possible_moves: max_next_q_for_opponent = 0.0
                else:
                    next_state_tensor_opponent_view = self.game.get_state(next_player_id)
                    _ , max_next_q_for_opponent = self._get_the_best_action_and_qvalue(next_state_tensor_opponent_view, next_possible_moves)
                    if max_next_q_for_opponent == -float('inf'): max_next_q_for_opponent = 0.0
                target_q_value += self.discount * (-max_next_q_for_opponent) # Minimax Q-like update
        loss = nn.functional.huber_loss(old_qvalue, target_q_value.detach())
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return loss.item()
    
# --- 対戦環境クラス (Env_Geister) ---
class Env_Geister:
    def __init__(self, agent1, agent2, game):
        self.game = game
        self.agent1 = agent1
        self.agent2 = agent2
        self.agent1.player_id = PLAYER_A_ID
        self.agent1.game = self.game
        self.agent2.player_id = PLAYER_B_ID
        self.agent2.game = self.game
        self.max_turns_per_game = 500
        self.loss_history_agent1 = []
        self.loss_history_agent2 = []

    def play_one_game_with_log(self, visualize=False, train_agents=True):
        self.game.reset_board()

        if train_agents:
            self.agent1.train_mode_on()
            self.agent2.train_mode_on()
            self.agent1.episode_count += 1
            self.agent2.episode_count += 1
        else:
            self.agent1.eval_mode_on()
            self.agent2.eval_mode_on()

        last_info_agent1 = {"state": None, "action": None}
        last_info_agent2 = {"state": None, "action": None}
        moves_log = []

        for turn_count in range(self.max_turns_per_game):

            current_player_id = self.game.current_player
            active_agent = self.agent1 if current_player_id == PLAYER_A_ID else self.agent2
            opponent_agent = self.agent2 if current_player_id == PLAYER_A_ID else self.agent1

            current_state_tensor = active_agent.check_state()
            possible_moves = active_agent.check_actions()

            if not possible_moves:
                self.game.winner = opponent_agent.player_id
                self.game.game_over = True
                print(f"Game ended: {self.game.winner} wins by no moves left.")
                break

            action = active_agent.action(possible_moves)
            if action is None:
                action = random.choice(possible_moves)

            moves_log.append({
                "turn": turn_count + 1,
                "player": current_player_id,
                "action": action
            })

            if active_agent == self.agent1:
                last_info_agent1 = {"state": current_state_tensor.clone(), "action": action}
            else:
                last_info_agent2 = {"state": current_state_tensor.clone(), "action": action}

            self.game.make_move(action[0], action[1])

            reward_active = self.game.checkwinner_for_reward(active_agent.player_id)
            done_active = self.game.gameover()
            next_state_tensor_for_active = active_agent.check_state()

        if not self.game.game_over and turn_count >= self.max_turns_per_game - 1:
            self.game.winner = "Draw"
            self.game.game_over = True
            print("Game ended: Draw by turn limit.")

        final_board_state = self.game.get_board_state_as_list()
        return self.game.winner, moves_log, final_board_state


    def start_training(self, episodes, visualize_interval=0, train_agents=True, model_save_interval=100, model_dir_prefix="./models_geister"):
        wins_A = 0; wins_B = 0; draws = 0
        for i in range(episodes):
            winner_tuple = self.play_one_game_with_log(visualize=(visualize_interval > 0 and (i + 1) % visualize_interval == 0), train_agents=train_agents)
            winner = winner_tuple[0]
            if winner == PLAYER_A_ID: wins_A += 1
            elif winner == PLAYER_B_ID: wins_B += 1
            else: draws += 1

            if (i + 1) % 100 == 0:
                total_played = wins_A + wins_B + draws
                print(f"--- Episode {i + 1}/{episodes} ---")
                print(f"Agent A Wins: {wins_A} ({wins_A/total_played:.2%}), Agent B Wins: {wins_B} ({wins_B/total_played:.2%}), Draws: {draws} ({draws/total_played:.2%})")
                avg_loss_A = np.mean(self.loss_history_agent1[-500:]) if self.loss_history_agent1 else float('nan')
                avg_loss_B = np.mean(self.loss_history_agent2[-500:]) if self.loss_history_agent2 else float('nan')
                print(f"Avg Loss A (last 500): {avg_loss_A:.4f}, Avg Loss B (last 500): {avg_loss_B:.4f}")
                print(f"Agent A Epsilon: {self.agent1.epsilon if hasattr(self.agent1, 'epsilon') else 'N/A'}")
                print(f"Agent B Epsilon: {self.agent2.epsilon if hasattr(self.agent2, 'epsilon') else 'N/A'}")

            if train_agents and model_save_interval > 0 and (i + 1) % model_save_interval == 0:
                # ----- Agent A -----
                model_dir_A = f"{model_dir_prefix}_agentA/"
                os.makedirs(model_dir_A, exist_ok=True)
                #CNN
                if hasattr(self.agent1, 'NN') and self.agent1.NN is not None:
                    model_path = os.path.join(model_dir_A, f"agentA_eps{i+1}.pth")
                    torch.save(self.agent1.NN.state_dict(), model_path)
                    print(f"Agent A CNN model saved to {model_path}")
                # QNN
                if hasattr(self.agent1, 'QNN') and self.agent1.QNN is not None:
                    qmodel_path = os.path.join(model_dir_A, f"agentA_qnn_eps{i+1}.pth")
                    torch.save(self.agent1.QNN.state_dict(), qmodel_path)
                    print(f"Agent A QNN saved to {qmodel_path}")
                # CQCNN
                if hasattr(self.agent1, 'HNN') and self.agent1.HNN is not None:
                    hnn_path = os.path.join(model_dir_A, f"agentA_cqcnn_eps{i+1}.pth")
                    torch.save(self.agent1.HNN.state_dict(), hnn_path)
                    print(f"Agent A CQCNN model saved to {hnn_path}")
                    config_path = os.path.join(model_dir_A, f"agentA_cqcnn_config.json")
                    config = {
                        "dev_name": self.agent1.HNN.dev.name,
                        "n_qubits_qnn": self.agent1.HNN.n_qubits,
                        "exp_or_prob": self.agent1.HNN.exp_or_prob,
                        "embedding_type": self.agent1.HNN.embedding_type,
                        "ansatz_type": self.agent1.HNN.ansatz_type,
                        "feature_map_reps": self.agent1.HNN.feature_map_reps,
                        "ansatz_reps": self.agent1.HNN.ansatz_reps,
                        "input_channels_cnn": self.agent1.HNN.input_channels_cnn,
                        "board_size_cnn": self.agent1.HNN.board_size_cnn,
                        "cnn_fc_out_features": self.agent1.HNN.cnn_fc_out_features,
                        "qnn_fc_out_features": self.agent1.HNN.qnn_fc_out_features
                    }
                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=2)

                # ----- Agent B -----
                if self.agent2 != self.agent1:
                    model_dir_B = f"{model_dir_prefix}_agentB/"
                    os.makedirs(model_dir_B, exist_ok=True)
                    #CNN
                    if hasattr(self.agent2, 'NN') and self.agent2.NN is not None:
                        model_path = os.path.join(model_dir_B, f"agentB_eps{i+1}.pth")
                        torch.save(self.agent2.NN.state_dict(), model_path)
                        print(f"Agent B CNN model saved to {model_path}")
                    # QNN
                    if hasattr(self.agent2, 'QNN') and self.agent2.QNN is not None:
                        qmodel_path = os.path.join(model_dir_B, f"agentB_qnn_eps{i+1}.pth")
                        torch.save(self.agent2.QNN.state_dict(), qmodel_path)
                        print(f"Agent B QNN saved to {qmodel_path}")
                    # CQCNN
                    if hasattr(self.agent2, 'HNN') and self.agent2.HNN is not None:
                        hnn_path = os.path.join(model_dir_B, f"agentB_cqcnn_eps{i+1}.pth")
                        torch.save(self.agent2.HNN.state_dict(), hnn_path)
                        print(f"Agent B CQCNN model saved to {hnn_path}")
                            # ✅ 構成情報の保存

        print("Training finished.")
        final_total = wins_A + wins_B + draws
        if final_total > 0:
            print(f"Final Score: A Wins: {wins_A} ({wins_A/final_total:.2%}), B Wins: {wins_B} ({wins_B/final_total:.2%}), Draws: {draws} ({draws/final_total:.2%})")

# --- 学習・評価関数の定義 (簡略化) ---
def run_geister_cnn_training(episodes=1000):
    print("--- Training CNN Agent for Geister ---")
    game_instance = GeisterGame()
    # 状態表現のチャネル数 (GeisterGame.get_state の出力に合わせる)
    # 例: 自良(1),自悪(1),敵駒(1),自駒位置(1),自脱出口(1),敵脱出口(1) -> 6チャネル
    INPUT_CHANNELS_FOR_GEISTER = 6 

    agent_a = CNNAgent_Geister(PLAYER_A_ID, game_instance, input_channels=INPUT_CHANNELS_FOR_GEISTER, epsilon=0.5, lr=0.0005)
    agent_b = CNNAgent_Geister(PLAYER_B_ID, game_instance, input_channels=INPUT_CHANNELS_FOR_GEISTER, epsilon=0.5, lr=0.0005) # 自己対戦
    # agent_b = RandomPolicy(PLAYER_B_ID, game_instance) # 対ランダム

    # モデル保存先ディレクトリ
    save_dir = "saved_models/cnn"
    os.makedirs(save_dir, exist_ok=True)
    # エージェントのモデル保存
    agent_a.NN.model_save_dir = os.path.join(save_dir, "agentA")
    agent_b.NN.model_save_dir = os.path.join(save_dir, "agentB")
    if not os.path.exists(agent_a.NN.model_save_dir):
        os.makedirs(agent_a.NN.model_save_dir)
    if not os.path.exists(agent_b.NN.model_save_dir):
        os.makedirs(agent_b.NN.model_save_dir)
    env = Env_Geister(agent_a, agent_b, game_instance)
    env.start_training(episodes, visualize_interval=0, train_agents=True, model_save_interval=episodes // 5) # 5回保存

    print("\n--- Evaluating Trained CNN Agent A vs Random ---")
    agent_a.eval_mode_on() # 評価モード
    eval_env_A_vs_Random = Env_Geister(agent_a, RandomPolicy(PLAYER_B_ID, game_instance), game_instance)
    eval_env_A_vs_Random.start_training(episodes=100, visualize_interval=0, train_agents=False)

    print("\n--- Evaluating Trained CNN Agent B vs Random ---")
    agent_b.eval_mode_on() # 評価モード
    eval_env_B_vs_Random = Env_Geister(RandomPolicy(PLAYER_A_ID, game_instance), agent_b, game_instance)
    eval_env_B_vs_Random.start_training(episodes=100, visualize_interval=0, train_agents=False)

# --- CQCAgent の学習・評価関数 (注意：dev_qnn_global の初期化が必要) ---
def run_geister_cqcnn_training(episodes=100, n_qbits=4, cnn_out_feat=4):
    print(f"--- Training CQCNN Agent (Qubits: {n_qbits}, CNN->QNN Feat: {cnn_out_feat}) ---")
    global dev_qnn_global # グローバルデバイスを使う
    if dev_qnn_global is None or len(dev_qnn_global.wires) != n_qbits:
        dev_qnn_global = qml.device("lightning.qubit", wires=n_qbits)
        print(f"Initialized QNN device: {dev_qnn_global.name} with {n_qbits} qubits.")

    game_instance_q = GeisterGame()
    INPUT_CHANNELS_FOR_GEISTER = 6

    # cnn_fc_out_features は QNNComponent のエンコーディング層が期待する入力次元数
    # AngleEmbedding を使う場合は cnn_fc_out_features と n_qubits_qnn が一致しなくても良いが、
    # ZFeatureMapなどを使う場合は一致させる必要がある (またはスライス/パディング)
    # ここでは cnn_fc_out_features = n_qubits_qnn と仮定。
    
    agent_a_q = CQCAgent_Geister(
        PLAYER_A_ID, game_instance_q, dev_qnn_global,
        embedding_type="AngleEmbedding", ansatz_type="RealAmplitudes",
        n_qubits_qnn=n_qbits,
        input_channels_cnn=INPUT_CHANNELS_FOR_GEISTER,
        board_size_cnn=board_size,
        cnn_fc_out_features=n_qbits, # QNNへの入力特徴数 (n_qubitsと一致させる)
        epsilon=0.5, lr=0.0005
    )
    agent_b_q = CQCAgent_Geister(
        PLAYER_B_ID, game_instance_q, dev_qnn_global,
        embedding_type="AngleEmbedding", ansatz_type="RealAmplitudes",
        n_qubits_qnn=n_qbits,
        input_channels_cnn=INPUT_CHANNELS_FOR_GEISTER,
        board_size_cnn=BOARD_SIZE,
        cnn_fc_out_features=n_qbits, # QNNへの入力特徴数 (n_qubitsと一致させる)
        epsilon=0.5, lr=0.0005
    )

    """
    agent_b_q = RandomPolicy(PLAYER_B_ID, game_instance_q) # 対ランダム
    save_dir = "saved_models/cqcnn"
    os.makedirs(save_dir, exist_ok=True)
    """

    env_q = Env_Geister(agent_a_q, agent_b_q, game_instance_q)
    env_q.start_training(episodes, visualize_interval=0, train_agents=True, model_save_interval=20)
    
    print("\n--- Evaluating Trained CQCNN Agent A vs Random ---")
    agent_a_q.eval_mode_on()
    agent_a_q.epsilon = 0.0 # 評価時はランダム性なし
    eval_env_Q_vs_Random = Env_Geister(agent_a_q, RandomPolicy(PLAYER_B_ID, game_instance_q), game_instance_q)
    eval_env_Q_vs_Random.start_training(episodes=100, visualize_interval=0, train_agents=False)

class AgentFactory:
    @staticmethod
    def create_cqc_agent(player_id, game, config, weights_path):
        # デバイス構築
        dev_qnn = qml.device(config.get("dev_type", "lightning.qubit"), wires=config["n_qubits_qnn"])

        # モデル構築
        model = CNN_QNN_CNN_Geister(
            dev=dev_qnn,
            n_qubits_qnn=config["n_qubits_qnn"],
            exp_or_prob=config["exp_or_prob"],
            embedding_type=config["embedding_type"],
            ansatz_type=config["ansatz_type"],
            feature_map_reps=config["feature_map_reps"],
            ansatz_reps=config["ansatz_reps"],
            input_channels_cnn=config["input_channels_cnn"],
            board_size_cnn=config["board_size_cnn"],
            cnn_fc_out_features=config["cnn_fc_out_features"],
            qnn_fc_out_features=config["qnn_fc_out_features"]
        )
        model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
        model.eval()

        # エージェント構築
        agent = CQCAgent_Geister(
            player_id=player_id,
            game=game,
            dev_qnn=dev_qnn,
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

class MatchRunner:
    @staticmethod
    def run_match(agentA_cfg, agentA_w, agentB_cfg, agentB_w, episodes=100):
        game = GeisterGame()
        agentA = AgentFactory.create_agent(agentA_cfg["agent_type"], "A", game, agentA_cfg, agentA_w)
        agentB = AgentFactory.create_agent(agentB_cfg["agent_type"], "B", game, agentB_cfg, agentB_w)
        env = Env_Geister(agentA, agentB, game)

        results = []
        for _ in range(episodes):
            winner, moves, board = env.play_one_game_with_log()
            results.append({"winner": winner, "moves": moves, "board": board})
        return results