from importlib import import_module
from pathlib import Path
from run import Run
import sys

def train_run(
  config: str | Path,
  *,
  configs_dir: str | Path = "configs",
  runs_dir: str | Path = "runs"
) -> None:
  with Run(
    config, 
    configs_dir=configs_dir, runs_dir=runs_dir
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
  args = sys.argv[1:]
  if len(args) != 1:
    raise SystemExit("usage: python -m train CONFIG")

  train_run(args[0])

if __name__ == "__main__":
  main()
