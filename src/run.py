from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TextIO, TypeVar, get_origin, get_type_hints
import json

import torch

S = TypeVar("S")

class RunStatus(int):
  NOT_STARTED = 0
  INCOMPLETE = 1
  COMPLETED = 2

def _config_from_dict(
  config_type: type[S],
  data: dict[str, Any],
  trace: str = "config"
) -> S:
  hints = get_type_hints(config_type)
  names = {f.name for f in fields(config_type)}

  unknown = set(data) - names
  if unknown: raise ValueError(f"unknown key(s) {sorted(unknown)} in {trace}")

  kwargs = {}
  for field in fields(config_type):
    if field.name not in data: continue
    value = data[field.name]
    field_type = hints[field.name]

    if is_dataclass(field_type) and isinstance(value, dict):
      value = _config_from_dict(
        field_type, value, f"{trace}.{field.name}"
      )
    elif get_origin(field_type) is tuple and isinstance(value, list):
      value = tuple(value)
    
    kwargs[field.name] = value
  return config_type(**kwargs)

class Run:
  "manager for all I/O (configs, runs, checkpoints)"

  CONFIG_FILE = "config.json"
  CHECKPOINTS_DIR = "checkpoints"
  METRICS_FILE = "metrics.jsonl"
  CHECKPOINT_STATE_FILE = "state.pt"
  COMPLETED_FILE = "completed"

  
  def __init__(
    self, 
    config_path: str | Path,
    *,
    configs_dir: str | Path = "configs",
    runs_dir: str | Path = "runs"
  ):
    config_path = Path(config_path)
    if config_path.parent == Path("."):
      config_path = Path(configs_dir) / config_path

    if not config_path.name.endswith(".json"):
      config_path = config_path.with_name(
        config_path.name + ".json"
      )

    self.name = config_path.name.removesuffix(".json")
    self.path = Path(runs_dir) / self.name

    run_config_path = self.path / self.CONFIG_FILE

    if self.path.exists():
      if not self.path.is_dir() or not run_config_path.is_file():
        raise FileNotFoundError(self.path)

      config_text = run_config_path.read_text(encoding="utf-8")

    else:
      if not config_path.is_file():
        raise FileNotFoundError(config_path)

      config_text = config_path.read_text(encoding="utf-8")
      _ = json.loads(config_text) # test config is fine

      self.path.mkdir(parents=True)
      (self.path / self.CHECKPOINTS_DIR).mkdir()
      run_config_path.write_text(config_text, encoding="utf-8")
      (self.path / self.METRICS_FILE).touch()

    self._config: dict = json.loads(config_text)

    checkpoint_dir = self.path / self.CHECKPOINTS_DIR
    if not checkpoint_dir.is_dir():
      raise FileNotFoundError(checkpoint_dir)

    self.metrics_fp: TextIO | None = None

  def close(self) -> None:
    if self.metrics_fp is not None and not self.metrics_fp.closed:
      self.metrics_fp.close()

  def __enter__(self):
    return self
  def __exit__(self, exc_type, exc, traceback):
    self.close()

  @property
  def version(self) -> str:
    return self._config["version"]

  @property
  def status(self) -> RunStatus:
    if (self.path / self.COMPLETED_FILE).is_file():
      return RunStatus.COMPLETED
    if self.latest_checkpoint() is not None:
      return RunStatus.INCOMPLETE
    return RunStatus.NOT_STARTED

  def config(
    self,
    config_type: type[S],
    overrides: dict[str, Any] | None = None
  ) -> S:
    """Build the run's config dataclass. overrides patches values for THIS
    process only (the stored config.json is untouched); keys may be dotted
    paths into nested configs, e.g. {"n_sims": 512, "train.lr": 1e-4}."""
    if not isinstance(config_type, type) or not is_dataclass(config_type):
      raise TypeError(f"{config_type} is not a dataclass")

    data = json.loads(json.dumps(self._config))  # deep copy
    data.pop("version")

    for key, value in (overrides or {}).items():
      *parents, leaf = key.split(".")
      node = data
      for parent in parents:
        node = node.setdefault(parent, {})
      node[leaf] = value

    return _config_from_dict(
      config_type,
      data
    )

  def latest_checkpoint(self) -> str | None:
    checkpoint_root = self.path / self.CHECKPOINTS_DIR

    steps = [
      int(path.name)
      for path in checkpoint_root.iterdir()
      if path.is_dir()
      and path.name.isdigit()
      and (path / self.CHECKPOINT_STATE_FILE).is_file()
    ]

    return str(max(steps)) if steps else None


  def save_checkpoint(self, step: str, state: dict[str, Any]) -> None:
    dir = self.path / self.CHECKPOINTS_DIR / step
    dir.mkdir(parents=True, exist_ok=True)

    temp = dir / f"{self.CHECKPOINT_STATE_FILE}.tmp"
    dest = dir / self.CHECKPOINT_STATE_FILE
    torch.save(state, temp)
    temp.replace(dest)


  def load_checkpoint(self, step: str | None = None) -> dict[str, Any]:
    step = step or self.latest_checkpoint()
    if step is None: raise FileNotFoundError(f"{self.path} has no checkpoints")

    path = self.path / self.CHECKPOINTS_DIR / step / self.CHECKPOINT_STATE_FILE

    if not path.is_file():
      raise FileNotFoundError(path)

    return torch.load(
      path,
      weights_only=False
    )

  def log(self, step: int, **metrics) -> None:
    if self.metrics_fp is None:
      self.metrics_fp = (self.path / self.METRICS_FILE).open(
        "a", encoding="utf-8", buffering=1
      )
    record = {"step": step, **metrics}
    self.metrics_fp.write(json.dumps(record) + "\n")

  def complete(self) -> None:
    if self.metrics_fp is not None:
      self.metrics_fp.flush()
    (self.path / self.COMPLETED_FILE).touch()
