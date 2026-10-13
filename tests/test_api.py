import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from arena import Arena
from bot_base import ChessbotBase, load_bot
from engine.game import GameBatch
from run import Run, RunStatus
from search.v1_alpha_beta import AlphaBeta
from search.v2_alpha_beta_temperature import AlphaBetaTemperature
from train import train_run


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


class RecordingBot(ChessbotBase):
  VERSION = "recording"

  def __init__(self) -> None:
    self.calls = []

  def _select_moves(self, batch, mask):
    self.calls.append((len(batch), mask.copy()))
    return np.zeros(len(batch), dtype=np.int32), {}


class ApiTests(unittest.TestCase):
  def setUp(self) -> None:
    self.temp = tempfile.TemporaryDirectory()
    self.runs = Path(self.temp.name) / "runs"

  def tearDown(self) -> None:
    self.temp.cleanup()

  def train(self, config: str) -> None:
    train_run(config, configs_dir=CONFIGS, runs_dir=self.runs)

  def test_run_creates_and_reopens_from_run_config(self) -> None:
    with Run("v0_random", configs_dir=CONFIGS, runs_dir=self.runs) as run:
      self.assertEqual(run.status, RunStatus.NOT_STARTED)
      self.assertIsNone(run.metrics_fp)
      config_path = run.path / Run.CONFIG_FILE

    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["seed"] = 7
    config_path.write_text(json.dumps(data), encoding="utf-8")

    with Run("v0_random", configs_dir=CONFIGS, runs_dir=self.runs) as run:
      from bots.v0_random.bot import Config
      self.assertEqual(run.config(Config).seed, 7)

  def test_train_and_load_bots(self) -> None:
    for config in ("v0_random", "v1.0_heuristic"):
      with self.subTest(config=config):
        self.train(config)
        with Run(config, configs_dir=CONFIGS, runs_dir=self.runs) as run:
          self.assertEqual(run.status, RunStatus.COMPLETED)
          self.assertEqual(run.latest_checkpoint(), "0")

        bot = load_bot(config, configs_dir=CONFIGS, runs_dir=self.runs)
        batch = GameBatch(2)
        mask = np.array([True, False])
        moves, info = bot.select_moves(batch, mask=mask)
        self.assertEqual(moves.shape, (2,))
        self.assertGreaterEqual(int(moves[0]), 0)
        self.assertLess(int(moves[0]), int(batch.num_moves_all()[0]))
        self.assertIsInstance(info, dict)

  def test_arena_uses_batched_bot_api(self) -> None:
    self.train("v0_random")
    self.train("v1.0_heuristic")
    random_bot = load_bot(
      "v0_random", configs_dir=CONFIGS, runs_dir=self.runs
    )
    heuristic_bot = load_bot(
      "v1.0_heuristic", configs_dir=CONFIGS, runs_dir=self.runs
    )

    records = Arena(
      white=random_bot,
      black=heuristic_bot,
      max_plies=2,
    ).play(n_games=1, verbose=False)

    self.assertEqual(len(records), 1)
    self.assertEqual(records[0]["plies"], 2)
    self.assertEqual(records[0]["result"], "CAPPED")

  def test_arena_plays_all_games_in_one_full_batch(self) -> None:
    white = RecordingBot()
    black = RecordingBot()

    records = Arena(
      white=white,
      black=black,
      max_plies=2,
    ).play(n_games=32, verbose=False)

    self.assertEqual(len(records), 32)
    self.assertTrue(all(record["plies"] == 2 for record in records))
    self.assertEqual(len(white.calls), 1)
    self.assertEqual(len(black.calls), 1)
    for batch_size, mask in (*white.calls, *black.calls):
      self.assertEqual(batch_size, 32)
      self.assertEqual(mask.shape, (32,))
      self.assertTrue(mask.all())

  def test_alpha_beta_finds_mate_in_one(self) -> None:
    batch = GameBatch(1)
    for move in (
      "e2-e4", "e7-e5", "d1-h5",
      "b8-c6", "f1-c4", "g8-f6",
    ):
      batch[0].play_str(move)
    mate = batch[0].moves().get(batch[0].parse("h5-f7"))

    search = AlphaBeta(
      evaluate=lambda _tree, _state: 0.0,
      max_depth=1,
      max_nodes=100,
      mate_value=1000.0,
      epsilon=0.0,
      rng=np.random.default_rng(42),
    )
    moves, info = search.select_moves(
      batch, np.array([True])
    )

    self.assertEqual(int(moves[0]), mate)
    self.assertGreater(float(info["value"][0]), 0.0)

  def test_alpha_beta_epsilon_one_chooses_from_rng(self) -> None:
    batch = GameBatch(1)
    expected_rng = np.random.default_rng(42)
    expected_rng.random()
    expected = int(expected_rng.integers(batch.num_moves_all()[0]))

    search = AlphaBeta(
      evaluate=lambda _tree, _state: 0.0,
      max_depth=1,
      max_nodes=100,
      mate_value=1000.0,
      epsilon=1.0,
      rng=np.random.default_rng(42),
    )
    moves, _ = search.select_moves(batch, np.array([True]))

    self.assertEqual(int(moves[0]), expected)

  def test_alpha_beta_temperature_samples_root_values(self) -> None:
    batch = GameBatch(1)
    n_moves = int(batch.num_moves_all()[0])
    probabilities = np.full(n_moves, 1.0 / n_moves)
    expected = int(np.random.default_rng(42).choice(
      n_moves, p=probabilities
    ))

    search = AlphaBetaTemperature(
      evaluate=lambda _tree, _state: 0.0,
      max_depth=1,
      max_nodes=100,
      mate_value=1000.0,
      temperature=0.5,
      rng=np.random.default_rng(42),
    )
    moves, info = search.select_moves(batch, np.array([True]))

    self.assertEqual(int(moves[0]), expected)
    self.assertEqual(int(info["nodes"][0]), n_moves)

  def test_heuristic_training_logs_and_checkpoints(self) -> None:
    configs = Path(self.temp.name) / "configs"
    configs.mkdir()
    (configs / "heuristic_test.json").write_text(json.dumps({
      "version": "v1_heuristic",
      "max_depth": 1,
      "pst_weight": 1.0,
      "seed": 42,
      "train": {
        "num_games": 1,
        "max_depth": 1,
        "max_plies": 2,
        "checkpoint_interval": 1,
      },
    }), encoding="utf-8")

    output = io.StringIO()
    with redirect_stdout(output):
      train_run("heuristic_test", configs_dir=configs, runs_dir=self.runs)
    self.assertIn(
      "game 0001/0001 | length 002, result ONGOING, winner none",
      output.getvalue(),
    )

    with Run(
      "heuristic_test", configs_dir=configs, runs_dir=self.runs
    ) as run:
      self.assertEqual(run.status, RunStatus.COMPLETED)
      self.assertEqual(run.latest_checkpoint(), "1")
      records = (run.path / Run.METRICS_FILE).read_text(
        encoding="utf-8"
      ).splitlines()
      self.assertEqual(len(records), 1)
      self.assertEqual(json.loads(records[0])["step"], 1)

    bot = load_bot(
      "heuristic_test", configs_dir=configs, runs_dir=self.runs
    )
    self.assertEqual(bot.VERSION, "v1_heuristic")


if __name__ == "__main__":
  unittest.main()
