from collections import namedtuple
from dataclasses import replace, asdict
import numpy as np
import torch
from engine.game import GameBatch, GameResult
from models.v11_muon_metal import Model, get_optimizer
from trainer_base import TrainerBase
from .bot import Chessbot, Config
from .policy import forward_batch
from replay.v9_gumbel import ReplayBuffer
import torch.nn.functional as F


RESULT_NAMES = [
  ("mate", GameResult.CHECKMATE), ("stale", GameResult.STALEMATE),
  ("oom", GameResult.DRAW_OOM), ("rep", GameResult.DRAW_REPETITION),
  ("capped", GameResult.ONGOING),
]
TRAJECTORY_FIELDS = ('board', 'castling', 'ep', 'moves', 'num_moves', 'policy')


Sample = namedtuple("Sample", "board castling ep moves num_moves policy z distance root_value")

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
  def _extra_log(self, metrics) -> str:
    if not (metrics.get("resign") or metrics.get("resign_checks")):
      return ""
    return f" resign:{metrics['resign']} (wrong {metrics['false_resigns']}/{metrics['resign_checks']} no-resign checks)"

  def __init__(self, config: Config):
    super().__init__(config)
    torch.manual_seed(config.seed)
    self.rng = np.random.default_rng(config.seed)
    self.model = Model(config.model).to(config.device)
    t_config = config.train
    self.optimizer = get_optimizer(self.model,
      t_config.weight_decay, t_config.lr, t_config.betas, t_config.eps,
      u_normuon=t_config.u_normuon
    )
    self.base_lrs = [g["lr"] for g in self.optimizer.param_groups]
    assert t_config.fast_sims <= t_config.n_sims
    # every trained loss by name -> weight. aux heads only exist when their weight > 0
    self.loss_weights = {
      "policy": 1.0, "value": t_config.value_weight,
      **{name: w for name, w in asdict(config.model.aux).items() if w > 0},
    }
    s_config = replace(config, 
      n_lanes=t_config.n_lanes, 
      n_sims=t_config.n_sims,
      c_puct=t_config.c_puct
    )
    self.bot = Chessbot(s_config, self.model)
    self.bot.rng = self.rng
    self.batch = GameBatch(t_config.n_lanes, sim_cap=config.node_cap)
    self.buffer = ReplayBuffer(t_config.buffer_capacity)

    self.search = self.bot.search_for(self.batch)
    assert t_config.n_lanes >= 2 * t_config.eval_batch, \
      "double buffering needs n_lanes >= 2 * eval_batch"
    # data reuse = positions trained per position recorded (Muon memorised the policy at reuse 4, not at reuse 1)
    recorded = t_config.pcr_p * t_config.n_lanes * t_config.waves_per_step
    print(f"reuse {t_config.grad_steps * t_config.batch_size / recorded:.2f}: {t_config.grad_steps} x {t_config.batch_size} "
          f"trained per ~{recorded:.0f} positions recorded each RL step", flush=True)

    self._trajectories = [[] for _ in range(t_config.n_lanes)]
    self._plies = np.zeros(t_config.n_lanes, np.int32)
    self._full = np.zeros(t_config.n_lanes, bool) # the lane's current search is a full (recorded) one
    self._roots = [None] * t_config.n_lanes # encoded root of the lane's current full search
    self._no_resign = self.rng.random(t_config.n_lanes) < t_config.no_resign_frac # this game never resigns
    self._would_resign = np.full(t_config.n_lanes, -1, np.int32) # no-resign game: first ply that would have resigned
    self._started = False


  def _set_lr(self):
    # warmup over the first warmup_steps RL steps of a run from scratch, then constant; with decay_length, a linear decay
    # over the last decay_length steps before max_steps (for a final branch, so the main run itself never decays).
    # note: the decay window follows max_steps, so raising max_steps on a decaying run moves it (the lr goes back up)
    t_config = self.config.train
    factor = min(1.0, (self.step + 1) / max(t_config.warmup_steps, 1))
    decay_start = t_config.max_steps - t_config.decay_length
    if t_config.decay_length and self.step >= decay_start:
      factor *= 1 - (self.step - decay_start) / t_config.decay_length
    for g, base in zip(self.optimizer.param_groups, self.base_lrs):
      g["lr"] = base * factor
    return factor


  def _begin_search(self, lanes):
    t_config = self.config.train
    full = self.rng.random(len(lanes)) < t_config.pcr_p

    K = int(self.batch.num_moves_all()[lanes].max())
    # noisy if full move or if reduced search only for first few opening plies (for exploration)
    noisy = full | (self._plies[lanes] < t_config.opening_noise_plies)
    gumbel = self.rng.gumbel(
      size=(len(lanes), K)
    ).astype(np.float32) * noisy[:, None]
    if full.any():
      board, castling, ep, _, _, moves, num_moves = self.batch.encode(lanes[full])
      for k, lane in enumerate(lanes[full]):
        self._roots[lane] = (
          board[k].copy(), 
          castling[k], ep[k], 
          moves[k].copy(), num_moves[k]
        )
    self._full[lanes] = full
    self.search.begin(
      lanes,
      np.where(full, t_config.n_sims, t_config.fast_sims),
      np.where(full, self.config.gumbel_m, t_config.fast_m),
      gumbel
    )


  def _finish_search(self, lanes, metrics):
    t_config = self.config.train
    B = len(self.batch)
    chosen, policy, root_value, nodes = self.search.results(lanes)
    metrics['plies'] += len(lanes)
    metrics['nodes'] += int(nodes.sum())
    for k, lane in enumerate(lanes):
      if self._full[lane]:
        self._trajectories[lane].append(
          (*self._roots[lane], policy[k], float(root_value[k]))
        )
        metrics['full_searches'] += 1
      else:
        self._trajectories[lane].append(None)

    # resign: the side to move gives up after a full search that sees the game as lost (its move is not played)
    resigned = np.zeros(B, bool)
    if t_config.resign_threshold > 0 and self.step >= t_config.resign_after:
      for lane in lanes[self._full[lanes] & (root_value < -t_config.resign_threshold)]:
        if not self._no_resign[lane]:
          resigned[lane] = True
        elif self._would_resign[lane] < 0:
          self._would_resign[lane] = self._plies[lane]
    played = lanes[~resigned[lanes]]

    mask = np.zeros(B, bool)
    mask[played] = True
    actions = np.zeros(B, np.int32)
    actions[lanes] = chosen
    results = self.batch.play_batch(actions, mask=mask)
    self._plies[played] += 1

    ended = np.concatenate([
      played[
        (results[played] != int(GameResult.ONGOING))
        | (self._plies[played] >= t_config.max_plies)
      ],
      lanes[resigned[lanes]]
    ])
    for lane in ended:
      result = int(results[lane])  # ONGOING here means the game was capped (or resigned)
      self._flush(lane, result, resigned=bool(resigned[lane]))
      metrics['games'] += 1
      metrics['game_plies'] += int(self._plies[lane])

      if resigned[lane]:
        metrics['resign'] += 1
      else:
        for name, r in self.RESULT_NAMES:
          if result == int(r):
            metrics[name] += 1
        if self._would_resign[lane] >= 0:
          # right iff the game ended in mate by the other side (the mating move is ply _plies - 1)
          p = int(self._would_resign[lane])
          right = result == int(GameResult.CHECKMATE) and (int(self._plies[lane]) - 1) % 2 != p % 2
          metrics['resign_checks'] += 1
          metrics['false_resigns'] += not right
      self._trajectories[lane] = []
      self._plies[lane] = 0
      self._no_resign[lane] = self.rng.random() < t_config.no_resign_frac
      self._would_resign[lane] = -1

    if len(ended):
      ended_mask = np.zeros(B, bool)
      ended_mask[ended] = True
      self.batch.reset_where(ended_mask)


  def _play_games(self):
    t_config = self.config.train
    self.model.eval()
    B = len(self.batch)

    metrics = dict(
      plies=0, nodes=0, cache_hits=0, nn_evals=0,
      games=0, game_plies=0, full_searches=0,
      resign=0, resign_checks=0, false_resigns=0,
      **{name: 0 for name, _ in self.RESULT_NAMES},
    )

    if not self._started:
      self._begin_search(np.arange(B, dtype=np.int32))
      self._started = True

    target = t_config.waves_per_step * B
    inflight = []

    while metrics['plies'] < target or inflight:
      # refresh finished searches
      done = self.search.take_finished()
      if len(done):
        self._finish_search(done, metrics)
        self._begin_search(done)

      # run all non-waiting lanes until blocked
      lanes = np.zeros(0, np.int32)
      if metrics['plies'] < target:
        lanes, states = self.search.run(t_config.eval_batch)

      # queue GPU
      if len(lanes):
        handle = self.bot.launch(
          self.batch, lanes, states, rows=t_config.eval_batch
        )
        inflight.append((lanes, states, handle))
        metrics['nn_evals'] += len(lanes)

      # wait for oldest batch to finish
      if inflight and (len(inflight) >= 2 or not len(lanes)):
        lanes_done, states_done, handle = inflight.pop(0)
        priors, values = self.bot.finish(handle)
        self.search.cache_evals(lanes_done, states_done, priors, values)
      elif not len(lanes) and not len(done):
        break

    metrics['cache_hits'] = self.search.stats()['cache_hits']
    return metrics


  def _flush(self, lane, result, resigned=False):
    # move finished game into replay buffer
    trajectory = self._trajectories[lane]
    if not trajectory:
      return
    n = len(trajectory)
    distance = np.arange(n - 1, -1, -1, dtype=np.int32)
    # mate: the last position's side to move (distance 0) mated, so won; resign: it resigned, so lost
    if result == int(GameResult.CHECKMATE) or resigned:
      sign = -1.0 if resigned else 1.0
      z = np.where(distance % 2 == 0, sign, -sign).astype(np.float32)
    else:
      z = np.zeros(n, np.float32)
    keep = [i for i, row in enumerate(trajectory) if row is not None]
    if not keep:
      return

    columns = list(zip(*[trajectory[i] for i in keep]))
    items = {name: np.asarray(column)
      for name, column in zip(TRAJECTORY_FIELDS, columns[:6])
    }
    self.buffer.add(**items, 
      z=z[keep], 
      distance=distance[keep],
      root_value=np.asarray(columns[6], np.float32)
    )

  def _targets(self, z, distance: np.ndarray):
    g = np.pow(self.config.train.discount, distance).astype(np.float32)[:, None]
    one_hot = np.stack((z == 1, z == 0, z == -1), axis=1).astype(np.float32)
    draw = np.array([[0., 1., 0.]], np.float32)
    return torch.from_numpy(g * one_hot + (1 - g) * draw).to(self.config.device)


  def _loss(self, sample: Sample):
    device = self.config.device
    with torch.autocast(device, dtype=torch.bfloat16, enabled=self.config.train.autocast):
      logits, _, wdl_logits, aux = forward_batch(
        self.model, sample.board, sample.castling, sample.ep,
        sample.moves, sample.num_moves,
        device, with_aux = True
      )
    # losses in fp32 (no-op without autocast)
    logits, wdl_logits = logits.float(), wdl_logits.float()
    aux = {name: pred.float() for name, pred in aux.items()}

    num_moves = torch.from_numpy(sample.num_moves).to(device)
    nonpadded_moves = torch.arange(logits.shape[1], device=device)[None, :] < num_moves[:, None]

    target_policy = torch.from_numpy(sample.policy).to(device)
    log_policy = logits.log_softmax(-1).masked_fill(~nonpadded_moves, 0)
    policy_loss = -(target_policy * log_policy).sum(-1).mean()

    targets = self._targets(sample.z, sample.distance)
    q = torch.from_numpy(sample.root_value).to(device).clamp(-1, 1)
    q_wdl = torch.stack([q.clamp(min=0), 1 - q.abs(), (-q).clamp(min=0)], 1)
    targets = (
      (1 - self.config.train.rootq_alpha) * targets
      + self.config.train.rootq_alpha * q_wdl
    )
    value_loss = -(targets * wdl_logits.log_softmax(-1)).sum(-1).mean()

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

        metrics["lr_factor"] = self._set_lr()   # warmup / constant / decay, see _set_lr

        sums = {}
        steps = t_config.grad_steps
        for _ in range(steps):
          idx = self.buffer.sample_indices(t_config.batch_size, self.rng)
          losses = self._loss(Sample(*self.buffer.gather(idx)))
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

        # logged as policy_loss / value_loss / <aux>_loss / grad_norm (+ lr_factor above)
        metrics.update({
          (k if k == "grad_norm" else f"{k}_loss"): (v / steps).item()
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
        f"cache {hit_rate:5.1%} | full {metrics['full_searches']}/{metrics['plies']} | {result_text}",
        flush=True,
      )
      
      if self.step % t_config.checkpoint_interval == 0:
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