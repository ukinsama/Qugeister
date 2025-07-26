import sys
import types
from pathlib import Path

# Ensure the repository root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Provide a minimal torch stub so geister_game can be imported without real torch
if 'torch' not in sys.modules:
    torch_stub = types.SimpleNamespace(
        device=lambda x=None: x,
        float32='float32',
        zeros=lambda *args, **kwargs: None,
        tensor=lambda *args, **kwargs: None,
        manual_seed=lambda *args, **kwargs: None,
        cuda=types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda *a, **k: None)
    )
    sys.modules['torch'] = torch_stub

from geister_game import Board, GeisterGame, PieceSetupStrategy

class FixedSetupStrategy(PieceSetupStrategy):
    def __init__(self, positions):
        # positions: dict mapping player_id -> list[(r,c)]
        self.positions = positions

    def setup_pieces(self, board: Board, player_id: str, pieces_to_place: list, num_ghosts: int):
        for pos, piece in zip(self.positions[player_id], pieces_to_place):
            r, c = pos
            board.set_piece(r, c, piece)


def test_board_basic_operations():
    board = Board(3)
    board.set_piece(1, 1, 'AX')
    assert board.get_piece(1, 1) == 'AX'
    assert board.to_list()[1][1] == 'X'

    removed = board.remove_piece(1, 1)
    assert removed == 'AX'
    assert board.get_piece(1, 1) == board.empty

    # invalid positions
    assert board.get_piece(5, 5) is None
    assert board.remove_piece(5, 5) is None


def test_board_get_all_piece_positions():
    board = Board(3)
    board.set_piece(0, 0, 'AY')
    board.set_piece(2, 2, 'BZ')
    all_pos = board.get_all_piece_positions()
    assert ((0, 0), 'AY') in all_pos
    assert ((2, 2), 'BZ') in all_pos

    a_pos = board.get_all_piece_positions('A')
    assert ((0, 0), 'AY') in a_pos
    assert all(p[1].startswith('A') for p in a_pos)


def test_make_move_and_win_by_capture():
    positions = {
        'A': [(2, 2), (5, 0)],
        'B': [(2, 3), (0, 0)]
    }
    strat = FixedSetupStrategy(positions)
    game = GeisterGame(board_size=6, num_ghosts_per_player=2,
                       setup_strategy_a=strat, setup_strategy_b=strat)

    # ensure initial placement
    assert game.get_piece_at(2, 2).startswith('A')
    assert game.get_piece_at(2, 3).startswith('B')

    # valid move capturing B's good ghost
    assert game.make_move((2, 2), (2, 3)) is True
    assert game.game_over is True
    assert game.winner == game.PLAYER_A_ID

    # board updated
    assert game.get_piece_at(2, 3).startswith('A')
    assert game.get_piece_at(2, 2) == game.EMPTY
