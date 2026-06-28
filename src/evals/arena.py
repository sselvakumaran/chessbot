"""Shared arena for evaluating chessbot v1 checkpoints by playing matches.

Reuses the Gumbel-AlphaZero search from bots/overnight-gumbel-v1.py, but adds a
per-game ``play_mask`` so two different models can each move in their own subset
of a single GameBatch — the basis for model-vs-model and model-vs-random matches.

Imports resolve the src/ root (dir holding engine/, models/, helpers/) and add it
to sys.path; nothing chdir's, so this runs from anywhere. Checkpoint paths are
absolute; libchess is located relative to engine/game.py.

Model config MUST match training (overnight-gumbel-v1.py): width-192, 3 layers,
move budget 62 (override with MODEL_MOVES if a checkpoint was trained otherwise).
Param shapes don't depend on the budget — it only sets the evaluator's chunk size.
"""
import os
import sys
import math
from pathlib import Path
import numpy as np

# --- locate src root (dir containing engine/ and models/) ---
for _p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    if (_p / "engine").exists() and (_p / "models").exists():
        SRC_ROOT = _p
        break
else:
    raise RuntimeError("could not locate src root (dir with engine/ and models/)")
sys.path.insert(0, str(SRC_ROOT))

REPO_ROOT = SRC_ROOT.parent
CKPT_DIR = REPO_ROOT / "data" / "overnight-gumbel-v1"

from tinygrad import Device
from tinygrad.nn.state import load_state_dict, safe_load
from engine.game import GameBatch, GameResult
from models.v1 import Model, Config
from helpers.evaluatorv1 import Evaluator, Encoding

MODEL_MOVES = int(os.environ.get("MODEL_MOVES", 62))
MODEL_CONFIG = Config(d_hidden=192, n_heads=4, n_layers=3, max_moves=MODEL_MOVES)


def load_evaluator(step: int, batch_size: int) -> Evaluator:
    """Load checkpoint ``model_<step>.safetensors`` into a v1 Evaluator."""
    path = CKPT_DIR / f"model_{step:06d}.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    model = Model(MODEL_CONFIG)
    load_state_dict(model, safe_load(str(path)))
    return Evaluator(model, batch_size)


# ============================ Gumbel search ============================
# (copied from bots/overnight-gumbel-v1.py; only change is play_mask gating)
def _sigma(q, max_N):
    return (50.0 + max_N) * q

def _np_softmax(x: np.ndarray, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def _terminal_value(result):
    return -1.0 if result == GameResult.CHECKMATE else 0.0

class Node:
    __slots__ = ("n_moves", "terminal", "expanded", "children",
                 "value", "p_logits", "N", "W")

    def __init__(self, n_moves: int, terminal: bool = False, value: float = 0.0):
        self.n_moves = n_moves
        self.terminal = terminal
        self.value = value
        self.expanded = False
        self.p_logits = None
        self.N = None
        self.W = None
        self.children = None

    def expand(self, priors, value):
        self.p_logits = priors
        self.value = value
        self.N = np.zeros(self.n_moves, dtype=np.float32)
        self.W = np.zeros(self.n_moves, dtype=np.float32)
        self.children = [None for _ in range(self.n_moves)]
        self.expanded = True

    def Q(self, a):
        return self.W[a] / self.N[a]

    def v_mix(self):
        visited = self.N > 0
        if not visited.any():
            return self.value
        policy = _np_softmax(self.p_logits)
        q = self.W[visited] / self.N[visited]
        policy_mean_value = (policy[visited] * q).sum() / policy[visited].sum()
        node_N = self.N.sum()
        return (self.value + node_N * policy_mean_value) / (1 + node_N)

    def completed_Q(self):
        _v_mix = self.v_mix()
        return np.where(self.N > 0, self.W / np.maximum(self.N, 1.0), _v_mix)

    def improved_policy(self) -> np.ndarray:
        max_N = self.N.max() if self.expanded else 0
        return _np_softmax(self.p_logits + _sigma(self.completed_Q(), max_N))

    def select(self):
        _improved_policy = self.improved_policy()
        node_N = self.N.sum()
        return int((_improved_policy - self.N / (1 + node_N)).argmax())

def _make_child(game, parent: Node, a: int):
    game.play(a)
    child = parent.children[a]
    if child is None:
        result = game.result()
        term = result != GameResult.ONGOING
        child = Node(game.num_moves(), terminal=term,
                     value=_terminal_value(result) if term else 0.0)
        parent.children[a] = child
    return child

def _descend(game, root: Node, initial_a: int):
    path = [(root, initial_a)]
    node = _make_child(game, root, initial_a)
    while node.expanded:
        a = node.select()
        path.append((node, a))
        node = _make_child(game, node, a)
    return node, path

def search_moves(game_batch: GameBatch, evaluator: Evaluator,
                 rng: np.random.Generator, n_sims: int, k: int,
                 play_mask: np.ndarray) -> np.ndarray:
    """Gumbel search restricted to games where play_mask is True.

    Returns an int (B,) array; only play_mask entries are meaningful. The
    evaluator scores the whole batch each call (other games are discarded) — fine
    for the small eval batches here.
    """
    B = len(game_batch)
    moves = np.zeros(B, dtype=np.int32)
    if not play_mask.any():
        return moves

    root_p_logits, root_values = evaluator.eval_logits(Encoding(*game_batch.get_encoding()))
    n_moves = game_batch.num_moves_all()

    roots: list = [None] * B
    gumbel: list = [None] * B
    candidates: list = [None] * B

    for i in range(B):
        if not play_mask[i]:
            continue
        _n = int(n_moves[i])
        node = Node(_n, terminal=False, value=float(root_values[i]))
        node.expand(root_p_logits[i, :_n].astype(np.float32).copy(), float(root_values[i]))
        roots[i] = node
        g = rng.gumbel(size=_n).astype(np.float32)
        gumbel[i] = g
        candidates[i] = list(np.argsort(-(node.p_logits + g))[:min(k, _n)])

    num_phases = max(1, math.ceil(math.log2(k)))
    for _phase in range(num_phases):
        to_eval = [[] for _ in range(B)]
        max_eval_len = 0
        for i in range(B):
            if not play_mask[i] or len(candidates[i]) <= 1:
                continue
            sims_per_cand = max(1, n_sims // (num_phases * len(candidates[i])))
            to_eval[i] = [a for a in candidates[i] for _ in range(sims_per_cand)]
            max_eval_len = max(max_eval_len, len(to_eval[i]))

        for eval_run in range(max_eval_len):
            leaves = []
            for i in range(B):
                if eval_run >= len(to_eval[i]):
                    continue
                leaf, path = _descend(game_batch[i], roots[i], to_eval[i][eval_run])
                leaves.append((i, leaf, path))
            if not leaves:
                continue
            p_logits, values = evaluator.eval_logits(Encoding(*game_batch.get_encoding()))
            for i, leaf, path in leaves:
                if not leaf.terminal:
                    _n = leaf.n_moves
                    leaf.expand(p_logits[i, :_n].astype(np.float32).copy(), float(values[i]))
                v = leaf.value
                for node, a in reversed(path):
                    v = -v
                    node.N[a] += 1
                    node.W[a] += v
                for _ in range(len(path)):
                    game_batch[i].undo()

        for i in range(B):
            if not play_mask[i] or len(candidates[i]) <= 1:
                continue
            root, g = roots[i], gumbel[i]
            root_max_N = root.N.max()
            ranked = sorted(
                candidates[i],
                key=lambda a: g[a] + root.p_logits[a]
                + _sigma(root.Q(a) if root.N[a] > 0 else root.v_mix(), root_max_N),
                reverse=True,
            )
            candidates[i] = ranked[:max(1, len(ranked) // 2)]

    for i in range(B):
        if play_mask[i]:
            moves[i] = int(candidates[i][0])
    return moves


# ============================== players ===============================
class ModelPlayer:
    def __init__(self, evaluator: Evaluator, n_sims: int, k: int,
                 rng: np.random.Generator, name: str = "model"):
        self.evaluator, self.n_sims, self.k, self.rng, self.name = \
            evaluator, n_sims, k, rng, name

    def choose(self, batch: GameBatch, play_mask: np.ndarray) -> np.ndarray:
        return search_moves(batch, self.evaluator, self.rng,
                            self.n_sims, self.k, play_mask)

class RandomPlayer:
    def __init__(self, rng: np.random.Generator, name: str = "random"):
        self.rng, self.name = rng, name

    def choose(self, batch: GameBatch, play_mask: np.ndarray) -> np.ndarray:
        B = len(batch)
        moves = np.zeros(B, dtype=np.int32)
        if not play_mask.any():
            return moves
        n_moves = batch.num_moves_all()
        for i in range(B):
            if play_mask[i]:
                moves[i] = int(self.rng.integers(max(1, int(n_moves[i]))))
        return moves


# =============================== match ================================
def play_match(player_a, player_b, n_games: int, max_plies: int = 300,
               log_every: int = 25):
    """Play n_games between A and B. A is White in the first half, Black in the
    second (so each side splits evenly). All active games advance in lockstep,
    one ply per iteration. Returns (a_wins, b_wins, draws) from A's POV.
    Unfinished games at max_plies are scored as draws.
    """
    B = n_games
    half = B // 2
    a_color = np.array([0 if i < half else 1 for i in range(B)], dtype=np.int8)  # 0=White
    batch = GameBatch(B)
    # outcome: -1 ongoing, 0 = A win, 1 = B win, 2 = draw
    outcome = np.full(B, -1, dtype=np.int8)

    for ply in range(max_plies):
        active = batch.active
        if not active.any():
            break
        turn = batch.to_moves()                       # 0=White, 1=Black, per game
        a_to_move = active & (turn == a_color)
        b_to_move = active & (turn != a_color)

        moves_a = player_a.choose(batch, a_to_move)
        moves_b = player_b.choose(batch, b_to_move)
        moves = np.where(a_to_move, moves_a, moves_b)

        res = batch.play_batch(moves, mask=active)
        to_after = batch.to_moves()
        for i in range(B):
            if not active[i]:
                continue
            r = int(res[i])
            if r == GameResult.ONGOING:
                continue
            if r == GameResult.CHECKMATE:
                winner = 1 - int(to_after[i])         # side that just moved won
                outcome[i] = 0 if winner == a_color[i] else 1
            else:
                outcome[i] = 2                         # stalemate / 50-move / repetition

        if ply % log_every == 0:
            done = int((outcome != -1).sum())
            print(f"    ply {ply:3d}: {done}/{B} games finished", flush=True)

    outcome[outcome == -1] = 2                         # hit the ply cap -> draw
    return (int((outcome == 0).sum()),
            int((outcome == 1).sum()),
            int((outcome == 2).sum()))


def print_backend():
    print(f"backend: {Device.DEFAULT}  ckpt dir: {CKPT_DIR}  move budget: {MODEL_MOVES}",
          flush=True)
