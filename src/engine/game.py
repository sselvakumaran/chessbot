import ctypes
import platform
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
import numpy as np

# ctypes layer over batching.cpp's GameManager. One manager owns all B lanes;
# every call crosses the FFI once for the whole batch. The tree API is fully
# LOCAL: a node is (state idx, action idx local to that state) per lane, so
# search stats live in numpy keyed [lane, state, action]. Root is state 0.

_lib_name = "../../lib/libchess.dylib" \
  if platform.system() == "Darwin" \
  else "../../lib/libchess.so"
_lib_path = Path(__file__).resolve().parent / _lib_name
if not _lib_path.exists():
  raise FileNotFoundError(
    f"{_lib_path} not found — please run with make"
  )
_lib = ctypes.CDLL(str(_lib_path))

_lib.game_max_moves_capacity.restype = ctypes.c_int
MAX_MOVES_CAPACITY = _lib.game_max_moves_capacity()

_lib.gamebatch_new.restype = ctypes.c_void_p
_lib.gamebatch_new.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.gamebatch_free.argtypes = [ctypes.c_void_p]
_lib.gamebatch_size.restype = ctypes.c_int
_lib.gamebatch_size.argtypes = [ctypes.c_void_p]
_lib.gamebatch_reset.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.gamebatch_get_roots.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 3
_lib.gamebatch_play.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 3
_lib.gamebatch_get_children.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 5
_lib.gamebatch_encode.argtypes = \
  [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int] \
  + [ctypes.c_void_p] * 7

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

def _ptr(arr: np.ndarray | None) -> ctypes.c_void_p | None:
  return None if arr is None else ctypes.c_void_p(arr.ctypes.data)

def _i32(a) -> np.ndarray:
  return np.ascontiguousarray(a, dtype=np.int32)

def _mask(m) -> np.ndarray | None:
  return None if m is None else np.ascontiguousarray(m, dtype=np.uint8)

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
  def __init__(self, rows: np.ndarray, n: int, side: Color):
    self._rows, self._n, self._side = rows, n, side

  def __len__(self) -> int: return self._n

  def __getitem__(self, i: int) -> Move:
    if not 0 <= i < self._n: raise IndexError(i)
    s_from, s_to, m_type, piece = (int(v) for v in self._rows[i])
    return Move(s_from, s_to, MoveType(m_type), piece, self._side)

  def __iter__(self):
    return (self[i] for i in range(self._n))

  def get(self, move: Move) -> int:
    for i in range(self._n):
      s_from, s_to, m_type, _ = (int(v) for v in self._rows[i])
      if s_from == move.s_from and s_to == move.s_to and m_type == move.m_type:
        return i
    return -1

class Game:
  """Per-lane view onto a GameBatch — convenience/debug only; search should
  use the batched GameBatch calls directly."""
  def __init__(self, batch: "GameBatch", i: int):
    self._board, self._i = batch, i

  @classmethod
  def standalone(cls) -> "Game":
    return GameBatch(1)[0]

  def reset(self):
    mask = np.zeros(len(self._board), np.uint8)
    mask[self._i] = 1
    self._board.reset_where(mask)

  def result(self) -> GameResult:
    return GameResult(int(self._board.results()[self._i]))

  def to_move(self) -> int:
    return int(self._board.to_moves()[self._i])

  def num_moves(self) -> int:
    return int(self._board.num_moves_all()[self._i])

  def play(self, idx: int) -> GameResult:
    mask = np.zeros(len(self._board), np.uint8)
    mask[self._i] = 1
    actions = np.zeros(len(self._board), np.int32)
    actions[self._i] = idx
    return GameResult(int(self._board.play_batch(actions, mask)[self._i]))

  def moves(self) -> MoveList:
    _, _, _, _, _, moves, num_moves = self._encode_root()
    return MoveList(moves[0], int(num_moves[0]), Color(self.to_move()))

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

  def ascii(self) -> str:
    board = self._encode_root()[0][0]
    side = self.to_move()
    lines = []
    for r in range(7, -1, -1):
      row = []
      for c in range(8):
        s = r * 8 + c                       # absolute square
        code = int(board[s ^ 56 if side == Color.BLACK else s])
        if code == 0:
          row.append(".")
        else:
          t = (code - 1) % 6                # 0-5 = P N B R Q K
          is_white = (code <= 6) == (side == Color.WHITE)
          row.append("PNBRQK"[t] if is_white else "pnbrqk"[t])
      lines.append(f"{r + 1}  " + " ".join(row))
    lines.append("   a b c d e f g h")
    return "\n".join(lines)

  def _encode_root(self) -> tuple:
    return self._board.encode(
      np.array([self._i], np.int32), np.zeros(1, np.int32)
    )


class GameBatch:
  def __init__(self, batch_size: int, sim_cap: int = 64):
    B = batch_size
    self._gamebatch = _lib.gamebatch_new(B, sim_cap)
    self.sim_cap = sim_cap
    self.states_cap = sim_cap + 1

    # root info cache
    self._root_num_actions = np.zeros(B, np.int32)
    self._root_result = np.zeros(B, np.int8)
    self._root_to_move = np.zeros(B, np.int8)
    self._roots_stale = True

    # encode buffers
    self.board = np.zeros((B, 64), np.uint8)
    self.castling = np.zeros((B,), np.uint8)
    self.ep = np.zeros((B,), np.int8)
    self.rep = np.zeros((B,), np.uint8)
    self.clock = np.zeros((B,), np.uint8)
    self.moves = np.zeros((B, MAX_MOVES_CAPACITY, 4), np.uint8)
    self.num_moves = np.zeros((B,), np.int32)

  def __del__(self):
    if not getattr(self, "_gamebatch", None):
      return
    _lib.gamebatch_free(self._gamebatch)
    self._gamebatch = None

  def __len__(self) -> int: return len(self.board)

  def __getitem__(self, i: int) -> Game:
    return Game(self, i)

  # ---- real game ----
  def _refresh_roots(self):
    if not self._roots_stale: return
    _lib.gamebatch_get_roots(
      self._gamebatch,
      _ptr(self._root_num_actions),
      _ptr(self._root_result),
      _ptr(self._root_to_move),
    )
    self._roots_stale = False

  def results(self) -> np.ndarray:
    self._refresh_roots()
    return self._root_result

  def to_moves(self) -> np.ndarray:
    self._refresh_roots()
    return self._root_to_move

  def num_moves_all(self) -> np.ndarray:
    self._refresh_roots()
    return self._root_num_actions

  @property
  def active(self) -> np.ndarray:
    return self.results() == GameResult.ONGOING

  def play_batch(self, actions: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    out = np.zeros(len(self), np.int8)
    _lib.gamebatch_play(self._gamebatch, _ptr(_i32(actions)), _ptr(_mask(mask)), _ptr(out))
    self._roots_stale = True
    return out

  def reset_where(self, mask: np.ndarray | None = None) -> None:
    """Resets lanes where mask (all lanes if None)."""
    _lib.gamebatch_reset(self._gamebatch, _ptr(_mask(mask)))
    self._roots_stale = True

  # ---- search tree (root = state 0; tree resets on every play/reset) ----
  def get_children(self, parent_states: np.ndarray, actions: np.ndarray) -> tuple:
    """One descent step: child of (parent_states[i], actions[i]) per lane,
    actions[i] < 0 skips lane i (state -1).
    Returns (states, num_actions, results), each (B,)."""
    s = np.full(len(self), -1, np.int32)
    n = np.zeros(len(self), np.int32)
    r = np.zeros(len(self), np.int8)
    _lib.gamebatch_get_children(
      self._gamebatch, _ptr(_i32(parent_states)), _ptr(_i32(actions)),
      _ptr(s), _ptr(n), _ptr(r),
    )
    return s, n, r

  def encode(self, lanes: np.ndarray | None = None,
             states: np.ndarray | None = None) -> tuple:
    """Encodes (lanes[i], states[i]) into row i of the shared buffers and
    returns (board, castling, ep, rep, clock, moves, num_moves) views of the
    first n rows. Defaults to every lane's root. Rows are overwritten by the
    next encode — copy anything you keep."""
    if lanes is None:
      lanes = np.arange(len(self), dtype=np.int32)
    if states is None:
      states = np.zeros(len(lanes), np.int32)
    lanes, states = _i32(lanes), _i32(states)
    n = len(lanes)
    _lib.gamebatch_encode(
      self._gamebatch, _ptr(lanes), _ptr(states), n,
      _ptr(self.board), _ptr(self.castling), _ptr(self.ep),
      _ptr(self.rep), _ptr(self.clock), _ptr(self.moves), _ptr(self.num_moves),
    )
    return (
      self.board[:n],
      self.castling[:n],
      self.ep[:n],
      self.rep[:n],
      self.clock[:n],
      self.moves[:n],
      self.num_moves[:n],
    )

  def get_encoding(self) -> tuple:
    """Root encoding of every lane (old game.py interface)."""
    return self.encode()
