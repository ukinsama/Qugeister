# train.py
import os
import abc
import random
from geister_game import GeisterGame
from geister_game import (
    GeisterGame,
    BOARD_SIZE,
    PLAYER_A_ID,
    PLAYER_B_ID,
    GOOD_GHOST,
    BAD_GHOST,
    EMPTY,
    NUM_GHOSTS_PER_PLAYER
)
import numpy as np

import torch
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
        if not possible_moves: return None
        print(f"プレイヤー {self.player_id} の番です。選択可能な手:")
        for i, ((fr, fc), (tr, tc)) in enumerate(possible_moves):
            piece = self.game.board.get_piece(fr,fc)
            kind = self.game.get_kind_of_piece(piece) if piece != EMPTY else '?'
            print(f"{i}: ({fr},{fc}) の {kind}オバケ を ({tr},{tc}) へ")
        while True:
            try:
                choice_str = input(f"移動する駒の番号を選んでください (0-{len(possible_moves)-1}): ")
                if not choice_str: continue # 空入力を無視
                choice = int(choice_str)
                if 0 <= choice < len(possible_moves):
                    return possible_moves[choice]
                else:
                    print(f"無効な番号です。0から{len(possible_moves)-1}の間で入力してください。")
            except ValueError:
                print("数値を入力してください。")
            except Exception as e:
                print(f"予期せぬエラー: {e}")


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
                 board_size=BOARD_SIZE, input_channels=6, elo=1500, epsilon=0.1, lr=0.001): # input_channelsを6に変更
        super().__init__(player_id, game)
        self.discount = 0.99
        self.epsilon_start = epsilon
        self.epsilon_end = 0.01
        self.epsilon_decay = 5000 # エピソード数で減衰
        self.epsilon = epsilon
        self.lr = lr
        self.alpha = 0.0 # L2正則化なしに一旦変更
        self.board_size = board_size
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
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
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
                 input_channels_cnn=6, board_size_cnn=BOARD_SIZE, 
                 cnn_fc_out_features=4, # QNNへの入力特徴数 (n_qubits_qnnと一致させる想定)
                 elo=1500, epsilon=0.1, lr=0.001):
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
        # (CNNAgent_Geister と同じロジック)
        current_state_tensor = self.check_state()
        if not self.eval_mode:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                           np.exp(-1. * self.episode_count / self.epsilon_decay)
        if not self.eval_mode and random.random() < self.epsilon:
            if not possible_moves: return None
            return random.choice(possible_moves)
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
    def __init__(self, agent1: Agent, agent2: Agent, game: GeisterGame):
        self.game = game
        self.agent1 = agent1
        self.agent2 = agent2
        self.agent1.player_id = PLAYER_A_ID
        self.agent1.game = self.game
        self.agent2.player_id = PLAYER_B_ID
        self.agent2.game = self.game
        self.max_turns_per_game = 200
        self.loss_history_agent1 = []
        self.loss_history_agent2 = []

    def play_one_game(self, visualize=False, train_agents=True):
        self.game.reset_board()
        if train_agents:
            self.agent1.train_mode_on(); self.agent2.train_mode_on()
            self.agent1.episode_count +=1; self.agent2.episode_count +=1 # For epsilon decay
        else:
            self.agent1.eval_mode_on(); self.agent2.eval_mode_on()

        # (s, a, r, s_next, done) storage for each agent for a full episode or a step
        # For simplicity, we update after each agent's turn if it's not the first turn
        # (state_tensor, action_tuple)
        last_info_agent1 = {"state": None, "action": None, "reward_pending":0.0, "next_state_pending":None, "done_pending":False}
        last_info_agent2 = {"state": None, "action": None, "reward_pending":0.0, "next_state_pending":None, "done_pending":False}

        for turn_count in range(self.max_turns_per_game):
            if visualize: self.game.display_board(reveal_opponent_pieces=True) # Reveal for debugging/vis

            current_player_id = self.game.current_player
            active_agent = self.agent1 if current_player_id == PLAYER_A_ID else self.agent2
            opponent_agent = self.agent2 if current_player_id == PLAYER_A_ID else self.agent1
            
            current_state_tensor = active_agent.check_state()
            possible_moves = active_agent.check_actions()

            if not possible_moves: # No legal moves
                self.game.winner = opponent_agent.player_id
                self.game.game_over = True
                # Update the previous player (opponent) who made the winning move
                if train_agents:
                    if opponent_agent.player_id == PLAYER_A_ID and last_info_agent1["state"] is not None:
                        loss = self.agent1.update(last_info_agent1["state"], last_info_agent1["action"], 1.0, current_state_tensor, True)
                        if loss is not None: self.loss_history_agent1.append(loss)
                    elif opponent_agent.player_id == PLAYER_B_ID and last_info_agent2["state"] is not None:
                        loss = self.agent2.update(last_info_agent2["state"], last_info_agent2["action"], 1.0, current_state_tensor, True)
                        if loss is not None: self.loss_history_agent2.append(loss)
                break

            action = active_agent.action(possible_moves)
            if action is None: # Agent failed to choose an action
                if possible_moves: action = random.choice(possible_moves)
                else: # Should be caught above
                    self.game.winner = opponent_agent.player_id
                    self.game.game_over = True
                    break
            
            # Execute the action and get the outcome
            # The (s,a) is for 'active_agent'. 'r' and 's_next' are observed after this.
            # The 'opponent_agent' was the one who acted previously to lead to 'current_state_tensor' for 'active_agent'.
            # So, we update 'opponent_agent' first.
            
            # If opponent had a pending state/action, update opponent
            if train_agents:
                info_to_update = last_info_agent1 if active_agent == self.agent2 else last_info_agent2
                if info_to_update["state"] is not None:
                    # Reward for opponent is 0 if game not over, or -1 if active_agent (self) won on this turn
                    # This s' for opponent is current_state_tensor for active_agent
                    reward_for_opponent = 0.0 # Intermediate reward
                    done_for_opponent = False
                    # No, this update logic is complex. Let's simplify for now:
                    # Update happens for the agent *after* their action and *after* opponent's response (or game end).
                    pass


            # Store current s,a for active_agent
            if active_agent == self.agent1:
                last_info_agent1 = {"state": current_state_tensor.clone(), "action": action}
            else:
                last_info_agent2 = {"state": current_state_tensor.clone(), "action": action}

            self.game.make_move(action[0], action[1]) # Game state changes, player possibly switches
            
            # Now, the game state is s' (s_next) resulting from active_agent's action.
            # The reward r is for active_agent.
            # The game might have ended.
            reward_active = self.game.checkwinner_for_reward(active_agent.player_id)
            done_active = self.game.gameover()
            next_state_tensor_for_active = active_agent.check_state() # s' from active_agent's perspective
            
            if train_agents:
                if active_agent == self.agent1:
                    loss = self.agent1.update(last_info_agent1["state"], last_info_agent1["action"], reward_active, next_state_tensor_for_active, done_active)
                    if loss is not None: self.loss_history_agent1.append(loss)
                else: # active_agent == self.agent2
                    loss = self.agent2.update(last_info_agent2["state"], last_info_agent2["action"], reward_active, next_state_tensor_for_active, done_active)
                    if loss is not None: self.loss_history_agent2.append(loss)
            
            if done_active:
                break
        
        if not self.game.game_over and turn_count >= self.max_turns_per_game -1 :
            self.game.winner = "Draw" # Max turns reached
            self.game.game_over = True
            # Final update for both if pending
            if train_agents:
                reward_draw = 0.0
                final_state_A = self.game.get_state(PLAYER_A_ID)
                final_state_B = self.game.get_state(PLAYER_B_ID)
                if last_info_agent1["state"] is not None:
                    loss = self.agent1.update(last_info_agent1["state"], last_info_agent1["action"], reward_draw, final_state_A, True)
                    if loss is not None: self.loss_history_agent1.append(loss)
                if last_info_agent2["state"] is not None:
                    loss = self.agent2.update(last_info_agent2["state"], last_info_agent2["action"], reward_draw, final_state_B, True)
                    if loss is not None: self.loss_history_agent2.append(loss)
        
        return self.game.winner

    def start_training(self, episodes, visualize_interval=0, train_agents=True, model_save_interval=100, model_dir_prefix="./models_geister"):
        wins_A = 0; wins_B = 0; draws = 0
        for i in range(episodes):
            winner = self.play_one_game(visualize=(visualize_interval > 0 and (i + 1) % visualize_interval == 0), train_agents=train_agents)
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
                if not os.path.exists(model_dir_A): os.makedirs(model_dir_A)

                if hasattr(self.agent1, 'NN') and self.agent1.NN is not None:
                    model_path = os.path.join(model_dir_A, f"agentA_eps{i+1}.pth")
                    torch.save(self.agent1.NN.state_dict(), model_path)
                    print(f"Agent A model saved to {model_path}")
                if hasattr(self.agent1, 'QNN') and self.agent1.QNN is not None:
                    qmodel_path = os.path.join(model_dir_A, f"agentA_qnn_eps{i+1}.pth")
                    torch.save(self.agent1.QNN.state_dict(), qmodel_path)
                    print(f"Agent A QNN saved to {qmodel_path}")

                # ----- Agent B -----
                if self.agent2 != self.agent1:
                    model_dir_B = f"{model_dir_prefix}_agentB/"
                    if not os.path.exists(model_dir_B): os.makedirs(model_dir_B)

                    if hasattr(self.agent2, 'NN') and self.agent2.NN is not None:
                        model_path = os.path.join(model_dir_B, f"agentB_eps{i+1}.pth")
                        torch.save(self.agent2.NN.state_dict(), model_path)
                        print(f"Agent B model saved to {model_path}")
                    if hasattr(self.agent2, 'QNN') and self.agent2.QNN is not None:
                        qmodel_path = os.path.join(model_dir_B, f"agentB_qnn_eps{i+1}.pth")
                        torch.save(self.agent2.QNN.state_dict(), qmodel_path)
                        print(f"Agent B QNN saved to {qmodel_path}")

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
        dev_qnn_global = qml.device("default.qubit", wires=n_qbits)
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
        board_size_cnn=BOARD_SIZE,
        cnn_fc_out_features=n_qbits, # QNNへの入力特徴数 (n_qubitsと一致させる)
        epsilon=0.5, lr=0.0005
    )
    agent_b_q = RandomPolicy(PLAYER_B_ID, game_instance_q) # 対ランダム

    env_q = Env_Geister(agent_a_q, agent_b_q, game_instance_q)
    env_q.start_training(episodes, visualize_interval=0, train_agents=True, model_save_interval=episodes // 2)

    print("\n--- Evaluating Trained CQCNN Agent A vs Random ---")
    agent_a_q.eval_mode_on()
    agent_a_q.epsilon = 0.0 # 評価時はランダム性なし
    eval_env_Q_vs_Random = Env_Geister(agent_a_q, RandomPolicy(PLAYER_B_ID, game_instance_q), game_instance_q)
    eval_env_Q_vs_Random.start_training(episodes=100, visualize_interval=0, train_agents=False)

# -------------------------------------------------------------------------------
# メイン実行ブロック
# -------------------------------------------------------------------------------
if __name__ == '__main__': # Jupyter Notebookではこのブロックは直接実行されないが、.py化を考慮
    # 1. CNN Agentの学習と評価
    run_geister_cnn_training(episodes=3000) # エピソード数を調整
    # 2. CQCNN Agentの学習と評価 (必要であればコメントを外して実行)
    # 注意: QNNの学習は非常に時間がかかる可能性があります。
    #qubits_for_qnn_run = 4  # QNNで使う量子ビット数
    #cnn_to_qnn_features_run = 4 # CNNからの特徴量をQNNの入力にする数 (上記qubitsと合わせるか、AngleEmbedding等で調整)
    #run_geister_cqcnn_training(episodes=3000, n_qbits=qubits_for_qnn_run, cnn_out_feat = cnn_to_qnn_features_run)



    # 3. 人間 vs 学習済みAI の対戦 (例)
    # print("\n--- Human vs Trained CNN Agent ---")
    # game_vs_human = GeisterGame()
    # trained_cnn_path = "./models_geister_cnn_agentA/agentA_eps10000.pth" # 保存したモデルのパス
    # if os.path.exists(trained_cnn_path):
    #     human_player = Human(PLAYER_A_ID, game_vs_human)
    #     ai_opponent = CNNAgent_Geister(PLAYER_B_ID, game_vs_human, input_channels=6)
    #     ai_opponent.NN.load_state_dict(torch.load(trained_cnn_path, map_location=ai_opponent.device))
    #     ai_opponent.eval_mode_on()
        
    #     env_human_vs_ai = Env_Geister(human_player, ai_opponent, game_vs_human)
    #     env_human_vs_ai.play_one_game(visualize=True, train_agents=False)
    # else:
    #     print(f"Trained model not found at {trained_cnn_path}. Skipping Human vs AI game.")