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
