from dataclasses import replace
import numpy as np
import torch
from engine.game import GameBatch, GameResult
from models.v7_moveless import Model, get_optimizer
from trainer_base import TrainerBase
from .bot import Chessbot, Config
from .policy import forward_batch
from replay.v5_policy import ReplayBuffer

RESULT_NAMES = [
  ("mate", GameResult.CHECKMATE), ("stale", GameResult.STALEMATE),
  ("oom", GameResult.DRAW_OOM), ("rep", GameResult.DRAW_REPETITION),
  ("capped", GameResult.ONGOING),
]
TRAJECTORY_FIELDS = ('board', 'castling', 'ep', 'moves', 'num_moves', 'policy')

class Trainer(TrainerBase[Config]):
  # class attributes so subclasses (v5.1) can extend the result types and the
  # log line without copying _train
  RESULT_NAMES = RESULT_NAMES
  def _extra_log(self, metrics) -> str: return ""

  def __init__(self, config: Config):
    super().__init__(config)
    torch.manual_seed(config.seed)
    self.rng = np.random.default_rng(config.seed)
    self.model = Model(config.model).to(config.device)
    t_config = config.train
    self.optimizer = get_optimizer(self.model,
      t_config.weight_decay, t_config.lr, t_config.betas, t_config.eps
    )
    s_config = replace(config, 
      n_lanes=t_config.n_lanes, 
      n_sims=t_config.n_sims,
      c_puct=t_config.c_puct, 
      temperature=t_config.temperature,
      dirichlet_alpha=t_config.dirichlet_alpha,
      dirichlet_epsilon=t_config.dirichlet_epsilon,
    )
    self.bot = Chessbot(s_config, self.model)
    self.bot.rng = self.rng
    self.batch = GameBatch(t_config.n_lanes, sim_cap=config.node_cap)
    self.buffer = ReplayBuffer(t_config.buffer_capacity)

    # in-flight games. lanes persist across steps and are refilled the moment
    # a game ends, so every wave runs at full batch width. a game may span a
    # weight update: weights are frozen within each generation round and the
    # bot's cache is cleared after training, and the positions/targets stay
    # valid either way. in-flight games are dropped (not saved) on resume.
    self._trajectories = [[] for _ in range(t_config.n_lanes)]
    self._plies = np.zeros(t_config.n_lanes, np.int32)


  def _play_games(self):
    t_config = self.config.train
    self.model.eval()
    B = len(self.batch)
    metrics = dict(
      plies=0, nodes=0, cache_hits=0, nn_evals=0, model_calls=0,
      games=0, game_plies=0, **{name: 0 for name, _ in self.RESULT_NAMES},
    )

    for _ in range(t_config.waves_per_step):
      # every lane always holds a live game (finished ones were reset below)
      encoded = [x.copy() for x in self.batch.encode()]
      actions, info = self.bot.select_moves(self.batch)

      for lane in range(B):
        self._trajectories[lane].append((
          # board, castling, ep
          encoded[0][lane], encoded[1][lane], encoded[2][lane],
          # moves, num_moves, policy
          encoded[5][lane], encoded[6][lane], info['policy'][lane].copy()
        ))
      self._plies += 1
      metrics['plies'] += B
      metrics['nodes'] += int(info['nodes'].sum())
      for key in ('cache_hits', 'nn_evals', 'model_calls'):
        metrics[key] += info[key]

      results = self.batch.play_batch(actions)
      finished = (results != int(GameResult.ONGOING)) \
        | (self._plies >= t_config.max_plies)
      if not finished.any(): continue

      for lane in np.flatnonzero(finished):
        result = int(results[lane])  # ONGOING here means the game was capped
        self._flush(lane, result)
        metrics['games'] += 1
        metrics['game_plies'] += int(self._plies[lane])
        for name, r in self.RESULT_NAMES:
          if result == int(r): metrics[name] += 1
        self._trajectories[lane] = []
        self._plies[lane] = 0

      # new games in those lanes; per-lane prediction caches start clean
      self.batch.reset_where(finished)
      self.bot.reset_games(finished)
    return metrics


  def _flush(self, lane, result):
    """Moves one finished game's trajectory into the replay buffer."""
    trajectory = self._trajectories[lane]
    if not trajectory: return
    columns = list(zip(*trajectory))
    items = {name: np.asarray(column)
      for name, column in zip(TRAJECTORY_FIELDS, columns)
    }
    distance = np.arange(len(trajectory) - 1, -1, -1, dtype=np.int32)
    z = (np.where(distance % 2 == 0, 1.0, -1.0).astype(np.float32)
      if result == int(GameResult.CHECKMATE) else np.zeros(len(trajectory), np.float32)
    )
    self.buffer.add(**items, z=z, distance=distance)

  def _targets(self, z, distance: np.ndarray):
    g = np.pow(self.config.train.discount, distance).astype(np.float32)[:, None]
    one_hot = np.stack((z == 1, z == 0, z == -1), axis=1).astype(np.float32)
    draw = np.array([[0., 1., 0.]], np.float32)
    return torch.from_numpy(g * one_hot + (1 - g) * draw).to(self.config.device)


  def _loss(self, sample):
    board, castling, ep, moves, num_moves, policy, z, distance = sample
    device = self.config.device
    logits, wdl = forward_batch(
      self.model, board, castling, ep, moves, num_moves, device
    )
    num_moves = torch.from_numpy(num_moves).to(device)
    nonpadded_moves = torch.arange(logits.shape[1], device=device)[None, :] < num_moves[:, None]

    target_policy = torch.from_numpy(policy).to(device)
    log_policy = logits.log_softmax(-1).masked_fill(~nonpadded_moves, 0)
    policy_loss = -(target_policy * log_policy).sum(-1).mean()
    targets = self._targets(z, distance)
    value_loss = -(targets * wdl.log_softmax(-1)).sum(-1).mean()
    return policy_loss, value_loss


  def _train(self, run):
    t_config = self.config.train
    device = self.config.device

    buffer_path = run.path / "buffer.npz"
    if buffer_path.is_file():
      self.buffer = type(self.buffer).load(buffer_path)
      print(f"resumed buffer: {self.buffer.size} positions", flush=True)

    while self.step < t_config.max_steps:
      metrics = self._play_games()
      metrics.update(policy_loss=None, value_loss=None, grad_norm=None)
      if self.buffer.size >= t_config.min_buffer:
        self.model.train()
        losses = []
        for _ in range(t_config.grad_steps):
          sample = self.buffer.sample(t_config.batch_size, self.rng)

          policy_loss, value_loss = self._loss(sample)
          self.optimizer.zero_grad(set_to_none=True)
          (policy_loss + value_loss).backward()
          # returns the total norm BEFORE clipping: a useful health signal
          grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), t_config.grad_clip or float('inf')
          )
          self.optimizer.step()
          losses.append((policy_loss.item(), value_loss.item(), grad_norm.item()))

        if losses:
          p, v, g = np.mean(losses, axis=0)
          metrics.update(policy_loss=float(p), value_loss=float(v), grad_norm=float(g))

        self.model.eval()
        self.bot.refresh()
      self.step += 1
      metrics['buffer'] = self.buffer.size
      run.log(self.step, **metrics)

      p = metrics["policy_loss"]
      v = metrics["value_loss"]
      g = metrics["grad_norm"]
      policy_text = "-" if p is None else f"{p:.4f}"
      value_text = "-" if v is None else f"{v:.4f}"
      grad_text = "-" if g is None else f"{g:.2f}"
      nodes_per_move = metrics["nodes"] / metrics["plies"] if metrics["plies"] else 0
      game_length = metrics["game_plies"] / metrics["games"] if metrics["games"] else 0
      evaluations = metrics["cache_hits"] + metrics["nn_evals"]
      hit_rate = metrics["cache_hits"] / evaluations if evaluations else 0
      result_text = " ".join(
          f"{name}:{metrics[name]}"
          for name, _ in self.RESULT_NAMES
          if metrics[name]
      ) + self._extra_log(metrics)
      print(
        f"step {self.step:05d}/{t_config.max_steps:05d} | ",
        f"policy {policy_text:>7} value {value_text:>7} gnorm {grad_text:>5} | ",
        f"buffer {self.buffer.size:>7,} | ",
        f"games {metrics['games']:3d} len {game_length:3.0f} | ",
        f"nodes/move {nodes_per_move:,.0f} | ",
        f"cache {hit_rate:5.1%} | {result_text}",
        flush=True,
      )
      
      if self.step % self.config.train.checkpoint_interval == 0:
        self.save_checkpoint(run)


  def save_checkpoint(self, run):
    super().save_checkpoint(run)
    temp = run.path / 'buffer.tmp.npz'
    self.buffer.save(temp)
    temp.replace(run.path / 'buffer.npz')


  def state_dict(self):
    return {
      'model': self.model.state_dict(),
      'optimizer': self.optimizer.state_dict(),
      'rng_state': self.rng.bit_generator.state,
    }


  def load_state_dict(self, state):
    self.model.load_state_dict(state['model'])
    self.optimizer.load_state_dict(state['optimizer'])
    self.rng.bit_generator.state = state['rng_state']
    self.bot.refresh()