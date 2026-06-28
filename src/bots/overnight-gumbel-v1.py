"""
Overnight Gumbel-AlphaZero self-play trainer — v1 (querier model + chunking evaluator).

Differences from overnight-gumbel.py:
  - uses models.v1 (move tokens don't attend to each other -> move-chunking is EXACT)
  - uses helpers.evaluatorv1 (slices the engine's wide move list to the model's per-forward
    budget MODEL_MOVES, chunking any position with > MODEL_MOVES legal moves)
  - MODEL_MOVES is the knob we ablate (default 62 -> 128 total tokens). search handles
    overflow exactly; train_step just slices to MODEL_MOVES (only ~0.45% of individual
    positions exceed it, so truncating their targets is negligible).
  - warm-starts the replay buffer from the v0 run so training runs from step 1, and writes
    checkpoints/buffer to its own dir.

Run from a terminal (BEAM is set before tinygrad imports):
    python bots/overnight-gumbel-v1.py 2>&1 | tee ../data/overnight-gumbel-v1/run.log
Ablate the budget:
    MODEL_MOVES=96 MAX_STEPS=1000 python bots/overnight-gumbel-v1.py
"""

import os
os.environ["BEAM"] = "2"            # MUST be set before any tinygrad import
os.environ.pop("DEBUG", None)

import sys, math, time
from pathlib import Path
import numpy as np

# --- locate project root (dir containing "bots") so imports resolve ---
_cur = Path.cwd().resolve()
for _p in [_cur, *_cur.parents]:
    if (_p / "bots").exists():
        sys.path.append(str(_p))
        os.chdir(_p)
        break

from tinygrad import Tensor, Device, TinyJit
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import (
    get_parameters, get_state_dict, load_state_dict, safe_save, safe_load,
)
from engine.game import GameBatch, Game, GameResult, MAX_MOVES
from models.v1 import Model, Config, init_weights
from helpers.evaluatorv1 import Evaluator, Encoding
from helpers.replay_buffer import ReplayBuffer

# ============================== config ==============================
DATA_DIR        = Path("../data")
CKPT_DIR        = DATA_DIR / "overnight-gumbel-v1"
BUFFER_PATH     = CKPT_DIR / "replay_buffer.npz"                 # save target (this run)
WARM_BUFFER     = DATA_DIR / "overnight-gumbel" / "replay_buffer.npz"  # warm-start from v0

GAME_BATCH_SIZE  = 256       # drop to 128 if you OOM at startup
MODEL_BATCH_SIZE = 256       # keep == GAME_BATCH_SIZE
MB               = 256       # train minibatch
N_SIMS, K        = 16, 8
MAX_STEPS        = int(os.environ.get("MAX_STEPS", 20_000))   # short runtime check by default
LR               = 3e-4
WEIGHT_DECAY     = 0.01
MIN_BUFFER       = 1_000
BUFFER_CAPACITY  = 200_000   # only used if no buffer file is found
CKPT_EVERY       = 500
BUFFER_EVERY     = 1_000
LOG_EVERY        = 50
SEED             = 42

# the move-token budget we ablate. total tokens fed to the transformer = N_FIXED(66) + this.
MODEL_MOVES      = int(os.environ.get("MODEL_MOVES", 62))     # 62 -> 128 total tokens
MODEL_CONFIG     = Config(d_hidden=192, n_heads=4, n_layers=3, max_moves=MODEL_MOVES)
# ====================================================================

# ===================== search code (pure defs) =====================
def sigma(q, max_N):
    return (50.0 + max_N) * q

def np_softmax(x: np.ndarray, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

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
        policy = np_softmax(self.p_logits)
        q = self.W[visited] / self.N[visited]
        policy_mean_value = (policy[visited] * q).sum() / policy[visited].sum()
        node_N = self.N.sum()
        return (self.value + node_N * policy_mean_value) / (1 + node_N)

    def completed_Q(self):
        _v_mix = self.v_mix()
        return np.where(self.N > 0, self.W / np.maximum(self.N, 1.0), _v_mix)

    def improved_policy(self) -> np.ndarray:
        max_N = self.N.max() if self.expanded else 0
        return np_softmax(self.p_logits + sigma(self.completed_Q(), max_N))

    def select(self):
        _improved_policy = self.improved_policy()
        node_N = self.N.sum()
        return int((_improved_policy - self.N / (1 + node_N)).argmax())

def make_child(game: Game, parent: Node, a: int):
    game.play(a)
    child = parent.children[a]
    if child is None:
        result = game.result()
        term = result != GameResult.ONGOING
        child = Node(game.num_moves(), terminal=term,
                     value=_terminal_value(result) if term else 0.0)
        parent.children[a] = child
    return child

def descend(game: Game, root: Node, initial_a: int):
    path = [(root, initial_a)]
    node = make_child(game, root, initial_a)
    while node.expanded:
        a = node.select()
        path.append((node, a))
        node = make_child(game, node, a)
    return node, path

def run_gumbel(game_batch: GameBatch, evaluator: Evaluator,
               rng: np.random.Generator, n_sims: int, k: int):
    B = len(game_batch)
    active = game_batch.active
    state = Encoding(*[val.copy() for val in game_batch.get_encoding()])

    root_p_logits, root_values = evaluator.eval_logits(Encoding(*game_batch.get_encoding()))
    n_moves = game_batch.num_moves_all()

    roots: list = [None for _ in range(B)]
    gumbel = [None for _ in range(B)]
    candidates = [None for _ in range(B)]

    for i in range(B):
        if not active[i]:
            continue
        _n_moves = int(n_moves[i])
        node = Node(_n_moves, terminal=False, value=float(root_values[i]))
        node.expand(root_p_logits[i, :_n_moves].astype(np.float32).copy(),
                    float(root_values[i]))
        roots[i] = node
        gumbel_noise = rng.gumbel(size=_n_moves).astype(np.float32)
        gumbel[i] = gumbel_noise
        n_selected_moves = min(k, _n_moves)
        candidates[i] = list(np.argsort(-(node.p_logits + gumbel_noise))[:n_selected_moves])

    num_phases = max(1, math.ceil(math.log2(k)))
    for phase in range(num_phases):
        to_eval = [[] for _ in range(B)]
        max_eval_len = 0
        for i in range(B):
            if not active[i] or len(candidates[i]) <= 1:
                continue
            sims_per_cand = max(1, n_sims // (num_phases * len(candidates[i])))
            to_eval[i] = [a for a in candidates[i] for _ in range(sims_per_cand)]
            max_eval_len = max(max_eval_len, len(to_eval[i]))

        for eval_run in range(max_eval_len):
            leaves = []
            for i in range(B):
                if eval_run >= len(to_eval[i]):
                    continue
                leaf, path = descend(game_batch[i], roots[i], to_eval[i][eval_run])
                leaves.append((i, leaf, path))
            if not leaves:
                continue
            p_logits, values = evaluator.eval_logits(Encoding(*game_batch.get_encoding()))
            for i, leaf, path in leaves:
                if not leaf.terminal:
                    _n_moves = leaf.n_moves
                    leaf.expand(p_logits[i, :_n_moves].astype(np.float32).copy(),
                                float(values[i]))
                v = leaf.value
                for node, a in reversed(path):
                    v = -v
                    node.N[a] += 1
                    node.W[a] += v
                for _ in range(len(path)):
                    game_batch[i].undo()

        for i in range(B):
            if not active[i] or len(candidates[i]) <= 1:
                continue
            root, gumbel_noise = roots[i], gumbel[i]
            root_max_N = root.N.max()
            ranked = sorted(
                candidates[i],
                key=lambda a: gumbel_noise[a] + root.p_logits[a]
                + sigma(root.Q(a) if root.N[a] > 0 else root.v_mix(), root_max_N),
                reverse=True,
            )
            candidates[i] = ranked[:max(1, len(ranked) // 2)]

    policy = np.zeros((B, MAX_MOVES), dtype=np.float32)
    moves = np.zeros(B, dtype=np.int32)
    for i in range(B):
        if not active[i]:
            continue
        moves[i] = int(candidates[i][0])
        _n_moves = roots[i].n_moves
        policy[i, :_n_moves] = roots[i].improved_policy()
    return state, policy, moves

# ===================== pure helpers (no global state) =====================
def _latest_model_ckpt():
    cks = sorted(CKPT_DIR.glob("model_*.safetensors"))
    return cks[-1] if cks else None

def _atomic_safe_save(state, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    safe_save(state, str(tmp))
    os.replace(tmp, path)


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("backend:", Device.DEFAULT, " ckpt dir:", CKPT_DIR.resolve(), flush=True)

    # ----------------------------- model / opt -----------------------------
    model = Model(MODEL_CONFIG)
    M = model.max_moves   # per-forward move-token budget

    decay_params   = [t for t in get_parameters(model) if t.ndim >= 2]
    nodecay_params = [t for t in get_parameters(model) if t.ndim < 2]
    opt_decay   = AdamW(decay_params,   lr=LR, weight_decay=WEIGHT_DECAY)
    opt_nodecay = AdamW(nodecay_params, lr=LR, weight_decay=0.0)
    _OPTS = [(opt_decay,   "opt_decay_latest.safetensors"),
             (opt_nodecay, "opt_nodecay_latest.safetensors")]
    print(f"weight decay {WEIGHT_DECAY} on {len(decay_params)} tensors; "
          f"0.0 on {len(nodecay_params)} (biases/norms)", flush=True)

    start_step = 0
    _ck = _latest_model_ckpt()
    if _ck is not None:
        start_step = int(_ck.stem.split("_")[1])
        load_state_dict(model, safe_load(str(_ck)))
        _resumed_opt = True
        for _o, _fn in _OPTS:
            _p = CKPT_DIR / _fn
            if _p.exists():
                try:
                    load_state_dict(_o, safe_load(str(_p)))
                except Exception as e:
                    _resumed_opt = False
                    print(f"opt resume failed for {_fn} ({e}); fresh moments", flush=True)
            else:
                _resumed_opt = False
        print(f"resumed model from step {start_step}"
              + (" + opt" if _resumed_opt else " (opt partially/not restored)"), flush=True)
    else:
        init_weights(model, MODEL_CONFIG)
        print("fresh v1 model", flush=True)

    n_params = sum(int(np.prod(t.shape)) for t in get_parameters(model))
    print(f"model params: {n_params/1e6:.2f}M  move budget {M} (total tokens {66 + M})  "
          f"config: {MODEL_CONFIG}", flush=True)

    evaluator = Evaluator(model, MODEL_BATCH_SIZE)

    # ----------------------------- replay buffer -----------------------------
    if BUFFER_PATH.exists():
        replay_buffer = ReplayBuffer.load(str(BUFFER_PATH))
        print(f"loaded replay buffer: size {replay_buffer.size} / cap {replay_buffer.capacity}", flush=True)
    elif WARM_BUFFER.exists():
        replay_buffer = ReplayBuffer.load(str(WARM_BUFFER))
        print(f"warm-started from v0 buffer: size {replay_buffer.size} / cap {replay_buffer.capacity}", flush=True)
    else:
        replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
        print(f"WARNING: no buffer found — starting EMPTY (slow warmup)", flush=True)

    batch = GameBatch(GAME_BATCH_SIZE)
    traj = [[] for _ in range(GAME_BATCH_SIZE)]

    @TinyJit
    def train_step(board, castling, ep, rep, clock, moves, num_moves, pi, z):
        p, v = model(board, castling, ep, rep, clock, moves, num_moves)
        policy_loss = -(pi * p.log_softmax(axis=-1)).sum(axis=-1).mean()
        value_loss = (pow(v.squeeze(-1) - z, 2)).mean()
        loss = policy_loss + value_loss
        opt_decay.zero_grad()
        opt_nodecay.zero_grad()
        loss.backward()
        opt_decay.step()
        opt_nodecay.step()
        return loss.realize()

    def save_ckpt(step: int):
        _atomic_safe_save(get_state_dict(model), CKPT_DIR / f"model_{step:06d}.safetensors")
        try:
            for _o, _fn in _OPTS:
                _atomic_safe_save(get_state_dict(_o), CKPT_DIR / _fn)
        except Exception as e:
            print(f"opt save failed at {step}: {e}", flush=True)

    def save_buffer():
        tmp = CKPT_DIR / "replay_buffer.tmp.npz"
        replay_buffer.save(str(tmp))
        os.replace(tmp, BUFFER_PATH)

    # ----------------------------- train loop -----------------------------
    step = start_step
    last = time.time()
    print(f"training from step {step} to {MAX_STEPS}", flush=True)
    try:
        while step < MAX_STEPS:
            Tensor.training = False
            states, pi, mv = run_gumbel(batch, evaluator, rng, n_sims=N_SIMS, k=K)
            side = batch.to_moves()
            for i in range(GAME_BATCH_SIZE):
                traj[i].append((states[i], pi[i].copy(), int(side[i])))

            results = batch.play_batch(mv)
            loser = batch.to_moves()
            for i in range(GAME_BATCH_SIZE):
                if results[i] == GameResult.ONGOING:
                    continue
                if results[i] == GameResult.CHECKMATE:
                    winner = 1 - int(loser[i])
                    zs = [1.0 if s == winner else -1.0 for (_, _, s) in traj[i]]
                else:
                    zs = [0.0] * len(traj[i])
                for (enc, p_, _), zz in zip(traj[i], zs):
                    replay_buffer.add(enc, p_, zz)
                traj[i] = []
                batch[i].reset()

            trained = False
            if replay_buffer.size >= MIN_BUFFER:
                Tensor.training = True
                s_enc, s_pi, s_z = replay_buffer.sample(MB, rng)
                # model takes only M move tokens; slice the buffer's wide moves/pi to M.
                # (chunking lives in the search evaluator; here a >M position is rare and
                #  truncating its target is negligible.)
                loss = train_step(
                    Tensor(s_enc.board), Tensor(s_enc.castling),
                    Tensor(s_enc.en_passant_square), Tensor(s_enc.repetition_count),
                    Tensor(s_enc.halfmove_clock),
                    Tensor(s_enc.moves[:, :M]),
                    Tensor(np.clip(s_enc.num_moves, 0, M).astype(np.int32)),
                    Tensor(s_pi[:, :M]), Tensor(s_z),
                )
                trained = True

            step += 1

            if step % LOG_EVERY == 0:
                dt = time.time() - last
                last = time.time()
                if trained:
                    print(f"step {step}  buffer {replay_buffer.size}  loss {loss.item():.3f}  ({dt:.1f}s/{LOG_EVERY})", flush=True)
                else:
                    print(f"warming buffer {replay_buffer.size}/{MIN_BUFFER}  ({dt:.1f}s/{LOG_EVERY})", flush=True)

            if step % CKPT_EVERY == 0:
                save_ckpt(step)
                print(f"  saved checkpoint {step}", flush=True)
            if step % BUFFER_EVERY == 0:
                save_buffer()
    finally:
        print(f"saving final state at step {step}...", flush=True)
        save_ckpt(step)
        save_buffer()
        print("done.", flush=True)


if __name__ == "__main__":
    main()