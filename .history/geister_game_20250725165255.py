# geister_game.py


import abc
import torch
import random
import time

# -------------------------------------------------------------------------------
# ガイスターゲームの定数
# -------------------------------------------------------------------------------
#なし

# -------------------------------------------------------------------------------
# セル : ガイスターゲームのルール (geister.py のクラス群をここに定義)
# -------------------------------------------------------------------------------
class Board:
    def __init__(self, size, empty_symbol="."):
        self.size = size
        self.empty = empty_symbol
        self.grid = [[self.empty for _ in range(size)] for _ in range(size)]

    def get_piece(self, r, c):
        if 0 <= r < self.size and 0 <= c < self.size:
            return self.grid[r][c]
        return None

    def set_piece(self, r, c, piece_str):
        if 0 <= r < self.size and 0 <= c < self.size:
            self.grid[r][c] = piece_str

    def remove_piece(self, r, c):
        if 0 <= r < self.size and 0 <= c < self.size:
            removed_piece = self.grid[r][c]
            self.grid[r][c] = self.empty
            return removed_piece
        return None
    
    def to_list(self):
        result = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                cell = self.grid[r][c]
                row.append(cell if cell == self.empty else cell[1])
            result.append(row)
        return result

    def display(self, current_player_id_for_display=None, reveal_opponent_pieces=False):
        header = "  " + " ".join(map(str, range(self.size)))
        print(header)
        for r_idx in range(self.size):
            row_display_list = [str(r_idx) + " "]
            for c_idx in range(self.size):
                piece = self.grid[r_idx][c_idx]
                if piece == self.empty:
                    row_display_list.append(self.empty)
                else:
                    player_of_piece = piece[0]
                    kind_of_piece = piece[1]
                    if (reveal_opponent_pieces or current_player_id_for_display is None or player_of_piece == current_player_id_for_display):
                        row_display_list.append(kind_of_piece)
                    else:
                        row_display_list.append("X")
            print(" ".join(row_display_list))
        print("-" * (self.size * 2 + 3))

    def get_all_piece_positions(self, player_id=None):
        positions = []
        for r in range(self.size):
            for c in range(self.size):
                piece = self.grid[r][c]
                if piece != self.empty:
                    if player_id is None or piece.startswith(player_id):
                        positions.append(((r, c), piece))
        return positions

class PieceSetupStrategy(abc.ABC):
    @abc.abstractmethod
    def setup_pieces(self, board: Board, player_id: str, pieces_to_place: list, num_ghosts: int):
        pass

class RandomSetupStrategy(PieceSetupStrategy):
    def __init__(self, player_a_id="A", player_b_id="B"):
        self.player_a_id = player_a_id
        self.player_b_id = player_b_id

    def _get_valid_setup_positions_for_player(self, player_id: str, board_size: int) -> list:
        positions = []
        if player_id == self.player_a_id:
            for c in range(1, board_size - 1):
                positions.append((board_size - 2, c))
                positions.append((board_size - 1, c))
        elif player_id == self.player_b_id:
            for c in range(1, board_size - 1):
                positions.append((1, c))
                positions.append((0, c))
        return positions

    def setup_pieces(self, board: Board, player_id: str, pieces_to_place: list, num_ghosts: int):
        if len(pieces_to_place) != num_ghosts:
            raise ValueError("駒の数が期待値と異なります。")
        valid_initial_positions = self._get_valid_setup_positions_for_player(player_id, board.size)
        if len(valid_initial_positions) < num_ghosts:
            raise ValueError("配置可能マスが駒の数より少ないです。")
        shuffled_pieces = random.sample(pieces_to_place, len(pieces_to_place))
        shuffled_positions = random.sample(valid_initial_positions, num_ghosts)
        for i in range(num_ghosts):
            r, c = shuffled_positions[i]
            board.set_piece(r, c, shuffled_pieces[i])

class GeisterGame:
    def __init__(self, board_size=6, num_ghosts_per_player=8, good_ghost="G", bad_ghost="B", empty_symbol=".", player_a_id="A", player_b_id="B", exits_a=None, exits_b=None, setup_strategy_a=None, setup_strategy_b=None):
        self.board_size = board_size
        self.num_ghosts_per_player = num_ghosts_per_player
        self.GOOD_GHOST_A = "G"
        self.BAD_GHOST_A = "B"
        self.GOOD_GHOST_B = "g"
        self.BAD_GHOST_B = "b"
        self.PLAYER_A_ID = player_a_id
        self.PLAYER_B_ID = player_b_id
        self.PLAYER_A_GOOD = self.get_piece_str(self.PLAYER_A_ID, self.GOOD_GHOST_A)
        self.PLAYER_A_BAD = self.get_piece_str(self.PLAYER_A_ID, self.BAD_GHOST_A)
        self.PLAYER_B_GOOD = self.get_piece_str(self.PLAYER_B_ID, self.GOOD_GHOST_B)
        self.PLAYER_B_BAD = self.get_piece_str(self.PLAYER_B_ID, self.BAD_GHOST_B)
        self.EMPTY = empty_symbol
        self.player_a_exits = exits_a if exits_a else [(0, 0), (0, board_size - 1)]
        self.player_b_exits = exits_b if exits_b else [(board_size - 1, 0), (board_size - 1, board_size - 1)]
        self.setup_strategy_player_a = setup_strategy_a if setup_strategy_a else RandomSetupStrategy(self.PLAYER_A_ID, self.PLAYER_B_ID)
        self.setup_strategy_player_b = setup_strategy_b if setup_strategy_b else RandomSetupStrategy(self.PLAYER_A_ID, self.PLAYER_B_ID)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.board = Board(self.board_size, self.EMPTY)
        self.reset_board()
        

    @staticmethod
    def get_piece_str(player_id, ghost_kind):
        return player_id + ghost_kind

    def reset_board(self):
        self.board.grid = [[self.EMPTY for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = self.PLAYER_A_ID
        self.player_stats = {
            self.PLAYER_A_ID: {"captured_good": 0, "captured_bad": 0, "escaped_good": 0, "pieces_left": self.num_ghosts_per_player},
            self.PLAYER_B_ID: {"captured_good": 0, "captured_bad": 0, "escaped_good": 0, "pieces_left": self.num_ghosts_per_player},
        }
        self.game_over = False
        self.winner = None
        self._initialize_game_pieces()

    def _initialize_game_pieces(self):
        half = self.num_ghosts_per_player // 2
        player_a_pieces = [self.PLAYER_A_GOOD] * half + [self.PLAYER_A_BAD] * (self.num_ghosts_per_player - half)
        self.setup_strategy_player_a.setup_pieces(self.board, self.PLAYER_A_ID, player_a_pieces, self.num_ghosts_per_player)
        player_b_pieces = [self.PLAYER_B_GOOD] * half + [self.PLAYER_B_BAD] * (self.num_ghosts_per_player - half)
        self.setup_strategy_player_b.setup_pieces(self.board, self.PLAYER_B_ID, player_b_pieces, self.num_ghosts_per_player)


    def get_player_of_piece(self, piece_str):
        if piece_str == self.EMPTY:
            return None
        return piece_str[0] if len(piece_str) > 0 else None

    def get_kind_of_piece(self, piece_str):
        if piece_str == self.EMPTY:
            return None
        return piece_str[1] if len(piece_str) > 1 else None

    def _is_valid_target(self, r, c, moving_player_id):
        if not (0 <= r < self.board.size and 0 <= c < self.board.size): return False
        target_piece = self.board.get_piece(r, c)
        if target_piece != self.EMPTY and self.get_player_of_piece(target_piece) == moving_player_id: return False
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
        if moving_piece_str == self.EMPTY or self.get_player_of_piece(moving_piece_str) != self.current_player:
            return False
        if not self._is_valid_target(to_r, to_c, self.current_player):
            return False

        opponent_id = self.PLAYER_B_ID if self.current_player == self.PLAYER_A_ID else self.PLAYER_A_ID
        captured_piece_str = self.board.get_piece(to_r, to_c)
        if captured_piece_str != self.EMPTY and self.get_player_of_piece(captured_piece_str) == opponent_id:
            captured_kind = self.get_kind_of_piece(captured_piece_str)
            if captured_kind == self.GOOD_GHOST:
                self.player_stats[self.current_player]["captured_good"] += 1
            elif captured_kind == self.BAD_GHOST:
                self.player_stats[self.current_player]["captured_bad"] += 1
            self.player_stats[opponent_id]["pieces_left"] -= 1

        self.board.set_piece(to_r, to_c, moving_piece_str)
        self.board.remove_piece(from_r, from_c)

        if self.get_kind_of_piece(moving_piece_str) == self.GOOD_GHOST and (to_r, to_c) in (self.player_a_exits if self.current_player == self.PLAYER_A_ID else self.player_b_exits):
            self.player_stats[self.current_player]["escaped_good"] += 1
            self.board.remove_piece(to_r, to_c)
            self.player_stats[self.current_player]["pieces_left"] -= 1

        self.check_win_condition()
        if not self.game_over:
            self.switch_player()
        return True
        
    def switch_player(self):
        self.current_player = self.PLAYER_B_ID if self.current_player == self.PLAYER_A_ID else self.PLAYER_A_ID

    def check_win_condition(self):
        stats_a = self.player_stats[self.PLAYER_A_ID]
        stats_b = self.player_stats[self.PLAYER_B_ID]
        win_threshold = self.num_ghosts_per_player // 2  # 動的閾値

        if stats_a["captured_good"] >= win_threshold:
            self.game_over = True
            self.winner = self.PLAYER_A_ID
        elif stats_b["captured_good"] >= win_threshold:
            self.game_over = True
            self.winner = self.PLAYER_B_ID
        elif stats_a["captured_bad"] >= win_threshold:
            self.game_over = True
            self.winner = self.PLAYER_B_ID
        elif stats_b["captured_bad"] >= win_threshold:
            self.game_over = True
            self.winner = self.PLAYER_A_ID
        elif stats_a["escaped_good"] >= 1:
            self.game_over = True
            self.winner = self.PLAYER_A_ID
        elif stats_b["escaped_good"] >= 1:
            self.game_over = True
            self.winner = self.PLAYER_B_ID


    def checkwinner_for_reward(self, player_id):
        reward = 0.0
        opponent_id = self.PLAYER_B_ID if player_id == self.PLAYER_A_ID else self.PLAYER_A_ID

        good_diff = self.player_stats[player_id]["captured_good"] - self.player_stats[opponent_id]["captured_good"]
        bad_diff = self.player_stats[player_id]["captured_bad"] - self.player_stats[opponent_id]["captured_bad"]
        reward += 0.05 * good_diff
        reward -= 0.05 * bad_diff

        good_positions = [(pos, piece) for pos, piece in self.board.get_all_piece_positions(player_id) if self.get_kind_of_piece(piece) == self.GOOD_GHOST]
        if good_positions:
            my_exits = self.player_a_exits if player_id == self.PLAYER_A_ID else self.player_b_exits
            min_distance = min(
                min(abs(pos[0] - ex[0]) + abs(pos[1] - ex[1]) for ex in my_exits)
                for pos, _ in good_positions
            )
            reward += 0.1 * (self.board_size - min_distance) / self.board_size  # 動的 board_size 使用

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
        opponent_id = self.PLAYER_B_ID if player_id_perspective == self.PLAYER_A_ID else self.PLAYER_A_ID
        my_exits = self.player_a_exits if player_id_perspective == self.PLAYER_A_ID else self.player_b_exits
        opponent_exits = self.player_b_exits if player_id_perspective == self.PLAYER_A_ID else self.player_a_exits

        for r in range(self.board_size):
            for c in range(self.board_size):
                piece = self.board.get_piece(r, c)
                if piece != self.EMPTY:
                    owner = self.get_player_of_piece(piece)
                    kind = self.get_kind_of_piece(piece)
                    if owner == player_id_perspective:
                        state[3, r, c] = 1.0
                        if kind == self.GOOD_GHOST:
                            state[0, r, c] = 1.0
                        elif kind == self.BAD_GHOST:
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
