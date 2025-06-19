# geister_game.py


import abc
import torch
import random
import time

# -------------------------------------------------------------------------------
# ガイスターゲームの定数
# -------------------------------------------------------------------------------
BOARD_SIZE = 6
NUM_GHOSTS_PER_PLAYER = 8
GOOD_GHOST = "G"
BAD_GHOST = "B"
EMPTY = "."
PLAYER_A_ID = "A"
PLAYER_B_ID = "B"

def get_piece_str(player_id, ghost_kind):
    return player_id + ghost_kind

PLAYER_A_GOOD = get_piece_str(PLAYER_A_ID, GOOD_GHOST)
PLAYER_A_BAD = get_piece_str(PLAYER_A_ID, BAD_GHOST)
PLAYER_B_GOOD = get_piece_str(PLAYER_B_ID, GOOD_GHOST)
PLAYER_B_BAD = get_piece_str(PLAYER_B_ID, BAD_GHOST)

PLAYER_A_EXITS = [(0, 0), (0, BOARD_SIZE - 1)]
PLAYER_B_EXITS = [(BOARD_SIZE - 1, 0), (BOARD_SIZE - 1, BOARD_SIZE - 1)]

# -------------------------------------------------------------------------------
# セル : ガイスターゲームのルール (geister.py のクラス群をここに定義)
# -------------------------------------------------------------------------------
class Board:
    def __init__(self, size=BOARD_SIZE):
        self.size = size
        self.grid = [[EMPTY for _ in range(size)] for _ in range(size)]

    def get_piece(self, r, c):
        if 0 <= r < self.size and 0 <= c < self.size:
            return self.grid[r][c]
        return None

    def set_piece(self, r, c, piece_str):
        if 0 <= r < self.size and 0 <= c < self.size:
            self.grid[r][c] = piece_str
        # else: # 学習中はエラー出力を抑制することが多い
            # print(f"エラー: 盤外({r},{c})に駒を置こうとしました。")

    def remove_piece(self, r, c):
        if 0 <= r < self.size and 0 <= c < self.size:
            removed_piece = self.grid[r][c]
            self.grid[r][c] = EMPTY
            return removed_piece
        return None
    
    def to_list(self):
        result = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                cell = self.grid[r][c]
                if cell == EMPTY:
                    row.append(EMPTY)
                else:
                    row.append(cell[1])  # 駒の種類だけ返す
            result.append(row)
        return result

    def display(self, current_player_id_for_display=None, reveal_opponent_pieces=False):
        header = "  " + " ".join(map(str, range(self.size)))
        print(header)
        for r_idx in range(self.size):
            row_display_list = [str(r_idx) + " "]
            for c_idx in range(self.size):
                piece = self.grid[r_idx][c_idx]
                if piece == EMPTY:
                    row_display_list.append(EMPTY)
                else:
                    player_of_piece = piece[0]
                    kind_of_piece = piece[1]
                    if (reveal_opponent_pieces or
                            current_player_id_for_display is None or
                            player_of_piece == current_player_id_for_display):
                        row_display_list.append(kind_of_piece)
                    else:
                        row_display_list.append("X")
            print(" ".join(row_display_list))
        print("-" * (self.size * 2 + 3)) # 表示調整

    def get_all_piece_positions(self, player_id=None):
        positions = []
        for r in range(self.size):
            for c in range(self.size):
                piece = self.grid[r][c]
                if piece != EMPTY:
                    if player_id is None or piece.startswith(player_id):
                        positions.append(((r, c), piece))
        return positions

class PieceSetupStrategy(abc.ABC):
    @abc.abstractmethod
    def setup_pieces(self, board: Board, player_id: str, pieces_to_place: list):
        pass

class RandomSetupStrategy(PieceSetupStrategy):
    def _get_valid_setup_positions_for_player(self, player_id: str, board_size: int) -> list:
        positions = []
        if player_id == PLAYER_A_ID: # Player A (盤面下側から見て手前2列の中央4マス)
            # row board_size - 2 (下から2行目)
            for c in range(1, board_size - 1): positions.append((board_size - 2, c))
            # row board_size - 1 (一番下の行)
            for c in range(1, board_size - 1): positions.append((board_size - 1, c))
        elif player_id == PLAYER_B_ID: # Player B (盤面上側から見て手前2列の中央4マス)
            # row 1 (上から2行目)
            for c in range(1, board_size - 1): positions.append((1, c))
            # row 0 (一番上の行)
            for c in range(1, board_size - 1): positions.append((0, c))
        return positions

    def setup_pieces(self, board: Board, player_id: str, pieces_to_place: list):
        if len(pieces_to_place) != NUM_GHOSTS_PER_PLAYER:
            raise ValueError(f"駒の数({len(pieces_to_place)})が期待値({NUM_GHOSTS_PER_PLAYER})と異なります。")
        
        valid_initial_positions = self._get_valid_setup_positions_for_player(player_id, board.size)
        if len(valid_initial_positions) < NUM_GHOSTS_PER_PLAYER:
             raise ValueError(f"配置可能マス({len(valid_initial_positions)})が駒の数({NUM_GHOSTS_PER_PLAYER})より少ないです。")

        shuffled_pieces = random.sample(pieces_to_place, len(pieces_to_place))
        # 配置可能なマスの中から、実際に配置する駒の数だけランダムに選ぶ
        shuffled_positions = random.sample(valid_initial_positions, NUM_GHOSTS_PER_PLAYER)

        for i in range(NUM_GHOSTS_PER_PLAYER):
            r, c = shuffled_positions[i]
            board.set_piece(r, c, shuffled_pieces[i])

class GeisterGame:
    def __init__(self, 
                 board_size=6,
                 num_ghosts_per_player=8,
                 exits_a=None,
                 exits_b=None,
                 setup_strategy_a: PieceSetupStrategy = None,
                 setup_strategy_b: PieceSetupStrategy = None):
        self.board_size = board_size
        self.num_ghosts_per_player = num_ghosts_per_player  # グローバル NUM_GHOSTS_PER_PLAYER を除去
        self.player_a_exits = exits_a if exits_a else [(0, 0), (0, board_size - 1)]  # PLAYER_A_EXITS 除去
        self.player_b_exits = exits_b if exits_b else [(board_size - 1, 0), (board_size - 1, board_size - 1)]  # PLAYER_B_EXITS 除去

        self.setup_strategy_player_a = setup_strategy_a if setup_strategy_a else RandomSetupStrategy()
        self.setup_strategy_player_b = setup_strategy_b if setup_strategy_b else RandomSetupStrategy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset_board()

    def reset_board(self):
        self.board = Board(self.board_size)
        self.current_player = PLAYER_A_ID
        self.player_stats = {
            PLAYER_A_ID: {"captured_good": 0, "captured_bad": 0, "escaped_good": 0, "pieces_left": self.num_ghosts_per_player},
            PLAYER_B_ID: {"captured_good": 0, "captured_bad": 0, "escaped_good": 0, "pieces_left": self.num_ghosts_per_player},
        }
        self.game_over = False
        self.winner = None
        self._initialize_game_pieces()

    def _initialize_game_pieces(self):
        half = self.num_ghosts_per_player // 2
        player_a_pieces = [PLAYER_A_GOOD] * half + [PLAYER_A_BAD] * (self.num_ghosts_per_player - half)
        self.setup_strategy_player_a.setup_pieces(self.board, PLAYER_A_ID, player_a_pieces, self.num_ghosts_per_player)
        player_b_pieces = [PLAYER_B_GOOD] * half + [PLAYER_B_BAD] * (self.num_ghosts_per_player - half)
        self.setup_strategy_player_b.setup_pieces(self.board, PLAYER_B_ID, player_b_pieces, self.num_ghosts_per_player)

    def get_player_of_piece(self, piece_str):
        if piece_str != EMPTY and len(piece_str) > 0: return piece_str[0]
        return None

    def get_kind_of_piece(self, piece_str):
        if piece_str != EMPTY and len(piece_str) > 1: return piece_str[1]
        return None

    def _is_valid_target(self, r, c, moving_player_id):
        if not (0 <= r < self.board.size and 0 <= c < self.board.size): return False
        target_piece = self.board.get_piece(r, c)
        if target_piece != EMPTY and self.get_player_of_piece(target_piece) == moving_player_id: return False
        return True

    def get_possible_actions(self, player_id=None):
        if player_id is None: player_id = self.current_player
        moves = []
        for (r, c), piece_str in self.board.get_all_piece_positions(player_id):
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if self._is_valid_target(nr, nc, player_id):
                    moves.append(((r, c), (nr, nc)))
        return moves
    
    def place(self, action, player_id): # For compatibility with OX game's Env
        if player_id != self.current_player: return False
        return self.make_move(action[0], action[1])

    def make_move(self, from_pos, to_pos):
        from_r, from_c = from_pos
        to_r, to_c = to_pos
        moving_piece_str = self.board.get_piece(from_r, from_c)
        if moving_piece_str == EMPTY or self.get_player_of_piece(moving_piece_str) != self.current_player:
            return False
        if not self._is_valid_target(to_r, to_c, self.current_player):
            return False

        opponent_id = PLAYER_B_ID if self.current_player == PLAYER_A_ID else PLAYER_A_ID
        captured_piece_str = self.board.get_piece(to_r, to_c)
        if captured_piece_str != EMPTY and self.get_player_of_piece(captured_piece_str) == opponent_id:
            captured_kind = self.get_kind_of_piece(captured_piece_str)
            if captured_kind == GOOD_GHOST:
                self.player_stats[self.current_player]["captured_good"] += 1
            elif captured_kind == BAD_GHOST:
                self.player_stats[self.current_player]["captured_bad"] += 1
            self.player_stats[opponent_id]["pieces_left"] -= 1

        self.board.set_piece(to_r, to_c, moving_piece_str)
        self.board.remove_piece(from_r, from_c)

        if self.get_kind_of_piece(moving_piece_str) == GOOD_GHOST and (to_r, to_c) in (self.player_a_exits if self.current_player == PLAYER_A_ID else self.player_b_exits):
            self.player_stats[self.current_player]["escaped_good"] += 1
            self.board.remove_piece(to_r, to_c)
            self.player_stats[self.current_player]["pieces_left"] -= 1

        self.check_win_condition()
        if not self.game_over:
            self.switch_player()
        return True
        
    def switch_player(self):
        self.current_player = PLAYER_B_ID if self.current_player == PLAYER_A_ID else PLAYER_A_ID

    def check_win_condition(self):
        stats_a = self.player_stats[PLAYER_A_ID]
        stats_b = self.player_stats[PLAYER_B_ID]
        if stats_a["captured_good"] >= 4: self.game_over = True; self.winner = PLAYER_A_ID
        elif stats_b["captured_good"] >= 4: self.game_over = True; self.winner = PLAYER_B_ID
        elif stats_a["captured_bad"] >= 4: self.game_over = True; self.winner = PLAYER_B_ID # B wins if A's bad are all captured
        elif stats_b["captured_bad"] >= 4: self.game_over = True; self.winner = PLAYER_A_ID # A wins if B's bad are all captured
        elif stats_a["escaped_good"] >= 1: self.game_over = True; self.winner = PLAYER_A_ID
        elif stats_b["escaped_good"] >= 1: self.game_over = True; self.winner = PLAYER_B_ID

    def checkwinner_for_reward(self, player_id):
        reward = 0.0
        opponent_id = PLAYER_B_ID if player_id == PLAYER_A_ID else PLAYER_A_ID

        # 良駒悪駒の捕獲報酬
        good_diff = self.player_stats[player_id]["captured_good"] - self.player_stats[opponent_id]["captured_good"]
        bad_diff = self.player_stats[player_id]["captured_bad"] - self.player_stats[opponent_id]["captured_bad"]
        reward += 0.1 * good_diff
        reward -= 0.05 * bad_diff

        # 赤駒の出口への接近
        good_positions = [(pos, piece) for pos, piece in self.board.get_all_piece_positions(player_id) if self.get_kind_of_piece(piece) == GOOD_GHOST]
        if good_positions:
            my_exits = PLAYER_A_EXITS if player_id == PLAYER_A_ID else PLAYER_B_EXITS
            min_distance = min(
                min(abs(pos[0] - ex[0]) + abs(pos[1] - ex[1]) for ex in my_exits)
                for pos, _ in good_positions
            )
            reward += 0.1 * (BOARD_SIZE - min_distance) / BOARD_SIZE

        # 勝敗報酬
        if self.game_over:
            if self.winner == player_id:
                reward += 5.0
            elif self.winner == "Draw":
                reward += 0.0
            else:
                reward -= 5.0

        return reward
        
    def gameover(self): # For OX game's Env compatibility
        return self.game_over

    def checkwinner(self): # For OX game's Env reward calculation
        if self.game_over: return self.winner
        return None

    def get_state(self, player_id_perspective: str) -> torch.tensor:
        
        num_channels = 6
        state = torch.zeros((num_channels, self.board_size, self.board_size), dtype=torch.float32, device=self.device)
        opponent_id = PLAYER_B_ID if player_id_perspective == PLAYER_A_ID else PLAYER_A_ID
        my_exits = self.player_a_exits if player_id_perspective == PLAYER_A_ID else self.player_b_exits
        opponent_exits = self.player_b_exits if player_id_perspective == PLAYER_A_ID else self.player_a_exits

        for r in range(self.board_size):
            for c in range(self.board_size):
                piece = self.board.get_piece(r, c)
                if piece != EMPTY:
                    owner = self.get_player_of_piece(piece)
                    kind = self.get_kind_of_piece(piece)
                    if owner == player_id_perspective:
                        state[3, r, c] = 1.0
                        if kind == GOOD_GHOST:
                            state[0, r, c] = 1.0
                        elif kind == BAD_GHOST:
                            state[1, r, c] = 1.0
                    elif owner == opponent_id:
                        state[2, r, c] = 1.0
                if (r, c) in my_exits:
                    state[4, r, c] = 1.0
                if (r, c) in opponent_exits:
                    state[5, r, c] = 1.0
        return state

    def display_board(self, sleep_seconds=0, reveal_opponent_pieces=False): # Renamed for clarity
        self.board.display(current_player_id_for_display=self.current_player, reveal_opponent_pieces=reveal_opponent_pieces)
        # print(f"Current player: {self.current_player}")
        # for p_id in [PLAYER_A_ID, PLAYER_B_ID]:
        #     stats = self.player_stats[p_id]
        #     print(f"Player {p_id}: Captured(G:{stats['captured_good']},B:{stats['captured_bad']}), Escaped(G:{stats['escaped_good']}), Pieces Left({stats['pieces_left']})")
        if sleep_seconds > 0: time.sleep(sleep_seconds)

    def get_board_state_as_list(self):
        return self.board.to_list()
