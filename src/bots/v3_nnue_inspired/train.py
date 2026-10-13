# shared trainer for v3 and v4 (v4 subclasses to swap the bot):
# self-play -> replay buffer -> minibatch grad steps, all in one process.
# targets are discounted FINAL OUTCOMES only:
#   g = discount ** plies_to_end;  decisive -> g*onehot(z) + (1-g)*draw
#   draws and ply-capped games   -> draw
# nothing search-derived is trained on. exploration = temperature over root
# values for the whole game (train.temperature; small).

from dataclasses import replace

import numpy as np
import torch

from engine.game import GameBatch, GameResult
from replay.v3_basic import ReplayBuffer
from models.v3_mlp_nnue import Model, get_optimizer
from run import Run
from trainer_base import TrainerBase

from .bot import Chessbot, Config

ONGOING = int(GameResult.ONGOING)
CHECKMATE = int(GameResult.CHECKMATE)
RESULT_NAMES = [
  ("mate", GameResult.CHECKMATE), ("stale", GameResult.STALEMATE),
  ("oom", GameResult.DRAW_OOM), ("rep", GameResult.DRAW_REPETITION),
  ("capped", GameResult.ONGOING),
]


class Trainer(TrainerBase[Config]):
  Bot = Chessbot  # v4 overrides

  def __init__(self, config: Config):
    super().__init__(config)
    torch.manual_seed(config.seed)
    self.rng = np.random.default_rng(config.seed)

    self.model = Model(config.model).to(config.device)
    self.optimizer = get_optimizer(
      self.model,
      config.train.weight_decay, config.train.lr,
      config.train.betas, config.train.eps,
    )
    # the self-play bot shares the model object: grad steps update it in
    # place, bot.refresh() drops anything the bot cached from old weights
    self.bot = self.Bot(self._search_config(config), self.model)
    self.bot.rng = self.bot.search.rng = self.rng
    self.batch = GameBatch(config.train.n_lanes, sim_cap=config.sim_cap)
    self.buffer = ReplayBuffer(config.train.buffer_capacity)

  def _search_config(self, config: Config) -> Config:
    "config the self-play bot runs with"
    return replace(config, temperature=config.train.temperature)

  def _play_games(self):
    """Self-play every lane to the end (or the ply cap); fill the buffer.
    Returns per-lane results, total plies, total net evals."""
    t_config = self.config.train
    B = len(self.batch)
    self.batch.reset_where()
    trajectories = [[] for _ in range(B)]  # (board, castling, ep) per ply
    results = np.full(B, ONGOING, np.int8)
    active = np.ones(B, bool)
    search_nodes = 0

    for _ in range(t_config.max_plies):
      if not active.any():
        break
      # snapshot roots BEFORE the search: it reuses the encode buffers
      board, castling, ep, *_ = self.batch.encode()
      board, castling, ep = board.copy(), castling.copy(), ep.copy()

      moves, info = self.bot.select_moves(self.batch, mask=active)
      search_nodes += int(info["nodes"][active].sum())
      for lane in np.flatnonzero(active):
        trajectories[lane].append((board[lane], castling[lane], ep[lane]))

      res = self.batch.play_batch(moves, mask=active)
      results = np.where(active, res, results)
      active &= res == ONGOING

    total_plies = 0
    for traj, result in zip(trajectories, results):
      n = len(traj)
      total_plies += n
      if n == 0:
        continue
      board, castling, ep = map(np.array, zip(*traj))
      distance = (n - 1) - np.arange(n)  # plies to the last recorded position
      # z from each position's mover's POV: on checkmate the mover of the last
      # recorded position delivered mate. draws and capped games -> 0
      if int(result) == CHECKMATE:
        z = np.where(distance % 2 == 0, 1.0, -1.0).astype(np.float32)
      else:
        z = np.zeros(n, np.float32)
      self.buffer.add(board, castling, ep, z, distance)
    return results, total_plies, search_nodes

  def _targets(self, z: np.ndarray, distance: np.ndarray) -> torch.Tensor:
    "Soft WDL targets: discount pulls far-from-the-end outcomes toward draw."
    g = (self.config.train.discount ** distance).astype(np.float32)[:, None]
    onehot = np.stack([z == 1.0, z == 0.0, z == -1.0], 1).astype(np.float32)
    draw = np.array([[0.0, 1.0, 0.0]], np.float32)
    return torch.from_numpy(g * onehot + (1.0 - g) * draw)

  def _train(self, run: Run) -> None:
    t_config = self.config.train
    device = self.config.device

    buffer_path = run.path / "buffer.npz"  # persists across pause/resume
    if buffer_path.is_file():
      self.buffer = ReplayBuffer.load(buffer_path)
      print(f"resumed buffer: {self.buffer.size} positions", flush=True)

    while self.step < t_config.max_steps:
      results, total_plies, search_nodes = self._play_games()

      loss_item = None
      if self.buffer.size >= t_config.min_buffer:
        self.model.train()
        losses = []
        for _ in range(t_config.grad_steps):
          board, castling, ep, z, distance = \
            self.buffer.sample(t_config.batch_size, self.rng)
          targets = self._targets(z, distance).to(device)
          logits, _ = self.model(
            torch.from_numpy(board).to(device),
            torch.from_numpy(castling).to(device),
            torch.from_numpy(ep).to(device),
          )
          loss = -(targets * logits.log_softmax(-1)).sum(dim=-1).mean()
          self.optimizer.zero_grad(set_to_none=True)
          loss.backward()
          self.optimizer.step()
          losses.append(loss.item())
        self.model.eval()
        self.bot.refresh()  # weights changed
        loss_item = float(np.mean(losses))

      self.step += 1
      counts = {name: int((results == int(r)).sum()) for name, r in RESULT_NAMES}
      run.log(self.step, loss=loss_item, buffer=self.buffer.size,
              plies=total_plies, search_nodes=search_nodes, **counts)

      loss_text = "-" if loss_item is None else f"{loss_item:.4f}"
      nodes_per_move = search_nodes / total_plies if total_plies else 0.0
      result_text = " ".join(f"{k}:{v}" for k, v in counts.items() if v)
      print(
        f"step {self.step:05d}/{t_config.max_steps:05d} | "
        f"loss {loss_text:>7} | buffer {self.buffer.size:>7} | "
        f"nodes/move {nodes_per_move:,.0f} | {result_text}",
        flush=True,
      )

      if self.step % t_config.checkpoint_interval == 0:
        self.save_checkpoint(run)
        temp = run.path / "buffer.tmp.npz"
        self.buffer.save(temp)
        temp.replace(buffer_path)

  def state_dict(self):
    return {
      "model": self.model.state_dict(),
      "optimizer": self.optimizer.state_dict(),
      "rng_state": self.rng.bit_generator.state,
    }

  def load_state_dict(self, state: dict) -> None:
    self.model.load_state_dict(state["model"])
    self.optimizer.load_state_dict(state["optimizer"])
    self.rng.bit_generator.state = state["rng_state"]
    self.bot.refresh()
