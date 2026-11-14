from importlib import import_module
from pathlib import Path
from run import Run
import argparse
import signal

def train_run(
  config: str | Path,
  *,
  configs_dir: str | Path = "configs",
  runs_dir: str | Path = "runs",
  duty: float = 1.0
) -> None:
  with Run(
    config, 
    configs_dir=configs_dir, runs_dir=runs_dir, duty=duty
  ) as run:
    bot_module = import_module(
      f"bots.{run.version}.bot"
    )
    train_module = import_module(
      f"bots.{run.version}.train"
    )

    bot_config = run.config(bot_module.Config)
    trainer = train_module.Trainer(bot_config)
    trainer.train(run)

def main() -> None:
  parser = argparse.ArgumentParser(prog="python -m train")
  parser.add_argument("config")
  parser.add_argument("--configs-dir", default="configs")
  parser.add_argument("--runs-dir", default="runs")
  parser.add_argument("--duty", type=float, default=1.0, help="share of the time spent training, (0, 1]")
  args = parser.parse_args()
  signal.signal(signal.SIGHUP, signal.default_int_handler) # a closed terminal stops like ctrl-c: a checkpoint first
  train_run(args.config, configs_dir=args.configs_dir, runs_dir=args.runs_dir, duty=args.duty)

if __name__ == "__main__":
  main()
