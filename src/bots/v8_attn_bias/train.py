from collections import namedtuple
from dataclasses import replace, asdict
import numpy as np
import torch
from engine.game import GameBatch, GameResult
from models.v8_attn_bias import Model, get_optimizer
from trainer_base import TrainerBase
from .bot import Chessbot, Config
from .policy import forward_batch
from replay.v5_policy import ReplayBuffer
import torch.nn.functional as F



RESULT_NAMES = [
  ("mate", GameResult.CHECKMATE), ("stale", GameResult.STALEMATE),
  ("oom", GameResult.DRAW_OOM), ("rep", GameResult.DRAW_REPETITION),
  ("capped", GameResult.ONGOING),
]
TRAJECTORY_FIELDS = ('board', 'castling', 'ep', 'moves', 'num_moves', 'policy')


Sample = namedtuple("Sample", "board castling ep moves num_moves policy z distance")

AUX_TARGETS = {
  "moves_left": lambda s, device: torch.log1p(
    torch.from_numpy(s.distance).to(device).float()
  ),
}
AUX_LOSSES = {
  "moves_left": lambda pred, target: F.huber_loss(pred.float(), target)
}



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
    # every trained loss by name -> weight. aux heads only exist when their weight > 0
    self.loss_weights = {
      "policy": 1.0, "value": t_config.value_weight,
      **{name: w for name, w in asdict(config.model.aux).items() if w > 0},
    }
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

      self.batch.reset_where(finished)
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


  def _loss(self, sample: Sample):
    device = self.config.device
    logits, _, wdl_logits, aux = forward_batch(
      self.model, sample.board, sample.castling, sample.ep, 
      sample.moves, sample.num_moves, 
      device, with_aux = True
    )

    num_moves = torch.from_numpy(sample.num_moves).to(device)
    nonpadded_moves = torch.arange(logits.shape[1], device=device)[None, :] < num_moves[:, None]

    target_policy = torch.from_numpy(sample.policy).to(device)
    log_policy = logits.log_softmax(-1).masked_fill(~nonpadded_moves, 0)
    policy_loss = -(target_policy * log_policy).sum(-1).mean()

    targets = self._targets(sample.z, sample.distance)

    # reduce value loss
    value_loss_each = -(targets * wdl_logits.log_softmax(-1)).sum(-1)
    if self.config.train.value_frac_trained < 1:
      value_train_mask = (
        torch.rand(len(value_loss_each), device=device) 
        < self.config.train.value_frac_trained
      ).float()
      value_loss = (
        (value_loss_each * value_train_mask).sum() /
        value_train_mask.sum().clamp(min=1)
      )
    else:
      value_loss = value_loss_each.mean()

    # every loss by name (unweighted); _train weights them with self.loss_weights
    losses = {"policy": policy_loss, "value": value_loss}
    for name, pred in aux.items():
      losses[name] = AUX_LOSSES[name](pred, AUX_TARGETS[name](sample, device))
    return losses


  def _train(self, run):
    t_config = self.config.train

    buffer_path = run.path / "buffer.npz"
    if buffer_path.is_file():
      self.buffer = type(self.buffer).load(buffer_path)
      print(f"resumed buffer: {self.buffer.size} positions", flush=True)

    while self.step < t_config.max_steps:
      metrics = self._play_games()

      if self.buffer.size >= t_config.min_buffer:
        self.model.train()
        sums = {}
        for _ in range(t_config.grad_steps):
          losses = self._loss(Sample(*self.buffer.sample(t_config.batch_size, self.rng)))
          self.optimizer.zero_grad(set_to_none=True)
          sum(self.loss_weights[k] * v for k, v in losses.items()).backward()
          # returns the total norm BEFORE clipping: a useful health signal.
          # added after backward, so it is only averaged/logged, never trained on
          losses["grad_norm"] = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), t_config.grad_clip or float('inf')
          )
          self.optimizer.step()
          for k, v in losses.items():
            sums[k] = sums.get(k, 0) + v.detach()  # stays on the GPU: one sync per RL step

        # logged as policy_loss / value_loss / <aux>_loss / grad_norm (same keys as v6/v7)
        metrics.update({
          (k if k == "grad_norm" else f"{k}_loss"): (v / t_config.grad_steps).item()
          for k, v in sums.items()
        })

        self.model.eval()
        self.bot.refresh()
      self.step += 1
      metrics['buffer'] = self.buffer.size
      run.log(self.step, **metrics)

      # keys are missing until the buffer reaches min_buffer -> printed as "-"
      fmt = lambda key, spec=".4f": "-" if metrics.get(key) is None else format(metrics[key], spec)
      loss_text = " ".join(f"{k} {fmt(k + '_loss'):>7}" for k in self.loss_weights)

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
        f"step {self.step:05d}/{t_config.max_steps:05d} |",
        loss_text,
        f"gnorm {fmt('grad_norm', '.2f'):>5} |",
        f"buffer {self.buffer.size:>7,} |",
        f"games {metrics['games']:3d} len {game_length:3.0f} |",
        f"nodes/move {nodes_per_move:,.0f} |",
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