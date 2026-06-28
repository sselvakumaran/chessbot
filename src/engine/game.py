import ctypes
import platform
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
import numpy as np

# class Piece(int):
#   NONE = 0
#   PLR_PAWN = 1
#   PLR_KNIGHT = 2
#   PLR_BISHOP = 3
#   PLR_ROOK = 4
#   PLR_QUEEN = 5
#   PLR_KING = 6
#   OPP_PAWN = 7
#   OPP_KNIGHT = 8
#   OPP_BISHOP = 9
#   OPP_ROOK = 10
#   OPP_QUEEN = 11
#   OPP_KING = 12

# class MoveType(int):
#   NORMAL = 0
#   CASTLE = 1
#   ENPASSANT = 2
#   PROMOTION_KNIGHT = 3
#   PROMOTION_BISHOP = 4
#   PROMOTION_ROOK = 5
#   PROMOTION_QUEEN = 6

# @dataclass(frozen=True)
# class BoardState:
#   board: list[Piece] # current player at the bottom
#   castling: int # 4 bits to represent who can castle (16)
#   en_passant_square: int # -1 or pos of pawn attacked
#   repetition_count: int # 0 - 2
#   halfmove_clock: int # 0 - 99

# @dataclass(frozen=True)
# class Move:
#   s_from: int
#   s_to: int
#   move_type: MoveType

_lib_name = "../../lib/libchess.dylib" if platform.system() == "Darwin" else "../../lib/libchess.so"
_lib_path = Path(__file__).resolve().parent / _lib_name
if not _lib_path.exists():
  raise FileNotFoundError(
    f"{_lib_path} not found — please run with make"
  )
_lib = ctypes.CDLL(str(_lib_path))

_lib.game_max_moves.restype = ctypes.c_int
MAX_MOVES = _lib.game_max_moves()

_lib.game_new.restype = ctypes.c_void_p
_lib.game_free.argtypes = [ctypes.c_void_p]
_lib.game_reset.argtypes = [ctypes.c_void_p]
_lib.game_result.argtypes = [ctypes.c_void_p]
_lib.game_result.restype = ctypes.c_int
_lib.game_num_moves.argtypes = [ctypes.c_void_p]
_lib.game_num_moves.restype = ctypes.c_int
_lib.game_to_move.argtypes = [ctypes.c_void_p]
_lib.game_to_move.restype = ctypes.c_int
_lib.game_play.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.game_play.restype = ctypes.c_int
_lib.game_undo.argtypes = [ctypes.c_void_p]
_lib.game_build_encoding.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 7
_lib.game_build_encoding.restype = ctypes.c_int
_lib.game_in_check.argtypes = [ctypes.c_void_p]
_lib.game_in_check.restype = ctypes.c_int
_lib.game_encode_position.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 5
_lib.game_encode_position.restype = None

class Color(IntEnum):
  WHITE = 0
  BLACK = 1

class GameResult(IntEnum):
  ONGOING = 0
  CHECKMATE = 1
  STALEMATE = 2
  DRAW_OOM = 3
  DRAW_REPETITION = 4

class MoveType(IntEnum):
  NORMAL = 0
  CASTLE = 1
  EN_PASSANT = 2
  PROMOTION_KNIGHT = 3
  PROMOTION_BISHOP = 4
  PROMOTION_ROOK = 5
  PROMOTION_QUEEN = 6

def _str_to_square(s: str) -> int:
  sq_file, sq_rank = ord(s[0]) - ord("a"), int(s[1].lower()) - 1
  return sq_rank * 8 + sq_file

def _square_to_str(i: int) -> str:
  return "abcdefgh"[i & 7] + str((i >> 3) + 1)

def _ptr(arr: np.ndarray, i: int) -> ctypes.c_void_p:
  return ctypes.c_void_p(arr.ctypes.data + int(i) * arr.strides[0])

@dataclass(frozen=True)
class Move:
  s_from: int
  s_to: int
  m_type: MoveType
  piece: int
  side: Color

  def __str__(self) -> str:
    s_from, s_to = self.s_from, self.s_to
    if self.side == Color.BLACK:
      s_from ^= 0b111000
      s_to ^= 0b111000
    out = f"{_square_to_str(s_from)}-{_square_to_str(s_to)}"
    if self.m_type != MoveType.NORMAL:
      out += " " + MoveType(self.m_type).name
    return out

class MoveList:
  def __init__(self, batch: "GameBatch", i: int):
    self._rows = batch.moves[i]
    self._n = int(batch.num_moves[i])
    self._side = _lib.game_to_move(batch.handles[i])
  
  def __len__(self) -> int: return self._n

  def __getitem__(self, i: int) -> Move:
    if not 0 <= i < self._n: raise IndexError(i)
    s_from, s_to, m_type, piece = (int(v) for v in self._rows[i])
    return Move(s_from, s_to, m_type, piece, self._side)

  def __iter__(self):
    return (self[i] for i in range(self._n))
  
  def get(self, m: Move) -> int:
    for i in range(self._n):
      s_from, s_to, m_type, _ = (int(v) for v in self._rows[i])
      if s_from == m.s_from and s_to == m.s_to and m_type == m.m_type:
        return i
    return -1

class Game:
  def __init__(self, batch: "GameBatch", i: int):
    self._b, self._i = batch, i
  
  @classmethod
  def standalone(cls) -> "Game":
    return GameBatch(1)[0]

  @property
  def _h(self):
    return self._b.handles[self._i]
  
  def reset(self):
    _lib.game_reset(self._h)
    self._b.stale_entries[self._i] = True
  
  def result(self) -> GameResult:
    return GameResult(_lib.game_result(self._h))
  
  def to_move(self) -> int:
    return _lib.game_to_move(self._h)
  
  def num_moves(self) -> int:
    return _lib.game_num_moves(self._h)
  
  def in_check(self) -> bool:
    return bool(_lib.game_in_check(self._h))
  
  def play(self, idx: int) -> GameResult:
    self._b.stale_entries[self._i] = True
    return GameResult(_lib.game_play(self._h, int(idx)))
  
  def undo(self):
    _lib.game_undo(self._h)
    self._b.stale_entries[self._i] = True 
  
  def moves(self) -> MoveList:
    self._b._refresh(self._i)
    return MoveList(self._b, self._i)
  
  def parse(self, move_str: str) -> Move:
    parts = move_str.strip().split()
    str_from, str_to = parts[0].split("-")
    s_from, s_to = _str_to_square(str_from), _str_to_square(str_to)
    if self.to_move() == Color.BLACK:
      s_from ^= 0b111000
      s_to ^= 0b111000
    tag = MoveType[parts[1].upper()] if len(parts) > 1 else None
    candidates = [
      move for move in self.moves()
      if move.s_from == s_from and move.s_to == s_to
        and (tag is None or move.m_type == tag)
    ]
    if not candidates: raise ValueError("bad move")
    if len(candidates) > 1:
      raise ValueError(f"{move_str!r} is ambiguous")
    return candidates[0]
  
  def play_str(self, move_str: str) -> GameResult:
    return self.play(self.moves().get(self.parse(move_str)))

  def encode_into(self, i: int, board, castling, ep, rep, clock, moves, num_moves) -> None:
    moves[i].fill(0)
    _lib.game_build_encoding(
      self._h,
      _ptr(board, i),
      _ptr(castling, i),
      _ptr(ep, i),
      _ptr(rep, i),
      _ptr(clock, i),
      _ptr(moves, i),
      _ptr(num_moves, i)
    )
  
  def encode_position_into(self, i: int, board, castling, ep, rep, clock) -> None:
    _lib.game_encode_position(
      self._h,
      _ptr(board, i),
      _ptr(castling, i),
      _ptr(ep, i),
      _ptr(rep, i),
      _ptr(clock, i)
    )

  def ascii(self) -> str:
    self._b._refresh(self._i)
    side = self.to_move()
    brd = self._b.board[self._i]
    lines = []
    for r in range(7, -1, -1):
      row = []
      for c in range(8):
        s = r * 8 + c                       # absolute square
        code = int(brd[s ^ 56 if side == Color.BLACK else s])
        if code == 0:
          row.append(".")
        else:
          t = (code - 1) % 6                # 0-5 = P N B R Q K
          is_white = (code <= 6) == (side == Color.WHITE)
          row.append("PNBRQK"[t] if is_white else "pnbrqk"[t])
      lines.append(f"{r + 1}  " + " ".join(row))
    lines.append("   a b c d e f g h")
    return "\n".join(lines)

class GameBatch:
  def __init__(self, batch_size: int):
    B, M = batch_size, MAX_MOVES
    self.handles = [_lib.game_new() for _ in range(B)]
    self.stale_entries = np.ones(B, dtype=bool)

    self.board = np.zeros((B, 64), dtype=np.uint8)
    self.castling = np.zeros((B,), dtype=np.uint8)
    self.ep = np.zeros((B,), dtype=np.int8)
    self.rep = np.zeros((B,), dtype=np.uint8)
    self.clock = np.zeros((B,), dtype=np.uint8)
    self.moves = np.zeros((B, M, 4), dtype=np.uint8)
    self.num_moves = np.zeros((B,), dtype=np.int32)
  
  def __del__(self):
    for h in getattr(self, "handles", []):
      _lib.game_free(h)
    self.handles = []
  
  def __len__(self) -> int: return len(self.handles)

  def __getitem__(self, i: int) -> Game:
    return Game(self, i)
  
  @property
  def active(self) -> np.ndarray:
    return np.array([_lib.game_result(h) == 0 for h in self.handles])

  def _refresh(self, i: int):
    if not self.stale_entries[i]: return
    self.moves[i].fill(0)
    _lib.game_build_encoding(
      self.handles[i],
      _ptr(self.board, i), 
      _ptr(self.castling, i),
      _ptr(self.ep, i), 
      _ptr(self.rep, i),
      _ptr(self.clock, i), 
      _ptr(self.moves, i),
      _ptr(self.num_moves, i),
    )
    self.stale_entries[i] = False
  
  def get_encoding(self) -> tuple:
    for i in np.nonzero(self.stale_entries)[0]:
      self._refresh(i)
    return (
      self.board, 
      self.castling, 
      self.ep, 
      self.rep, 
      self.clock, 
      self.moves, 
      self.num_moves
    )
  
  def results(self) -> np.ndarray:
    return np.fromiter(
      (_lib.game_result(h) for h in self.handles),
      dtype=np.int8, count=len(self.handles),
    )

  def to_moves(self) -> np.ndarray:
    return np.fromiter(
      (_lib.game_to_move(h) for h in self.handles),
      dtype=np.int8, count=len(self.handles),
    )
 
  def num_moves_all(self) -> np.ndarray:
    return np.fromiter(
      (_lib.game_num_moves(h) for h in self.handles),
      dtype=np.int32, count=len(self.handles),
    )

  def play_batch(self, actions: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    B = len(self.handles)
    out = np.empty(B, dtype=np.int8)
    for i in range(B):
      if mask is not None and not mask[i]:
        out[i] = _lib.game_result(self.handles[i])
        continue
      out[i] = _lib.game_play(self.handles[i], int(actions[i]))
      self.stale_entries[i] = True
    return out

  def reset_where(self, mask: np.ndarray) -> None:
    for i in np.nonzero(mask)[0]:
      _lib.game_reset(self.handles[i])
      self.stale_entries[i] = True