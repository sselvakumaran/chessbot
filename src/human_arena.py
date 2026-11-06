"""Play against a chessbot from a browser on the local network.

    venv/bin/python src/human_arena.py v1.1_heuristic_pst [--port 8000]

The human-facing counterpart to arena.py: instead of pitting two bots, it
serves ui/play.html at http://<lan-ip>:<port>/ — open it from any device on
the network, pick a side (White / Black / Random), and play. Click a piece
to see its legal moves, click a target to move, promotions pop a piece
picker, R starts a new game.

The bot comes from the config/run abstraction — any name in configs/ with a
checkpoint in runs/ works (bot_base.load_bot).

One game per browser connection (WebSocket). Multiple people can play at
once — each connection gets its own 1-lane GameBatch; the single shared bot
is serialized behind a lock, so concurrent games just queue for "thinking".
When a browser closes, its WebSocket drops and the session is discarded.

All game state lives server-side; the engine (lib/libchess) stays the single
source of truth for legality and results, via the Snapshot view.
"""
import argparse
import asyncio
import json
import secrets
import socket
from pathlib import Path

import numpy as np
import tornado.web
import tornado.websocket

from bot_base import ChessbotBase, load_bot
from engine.game import Color, GameBatch, GameResult, MoveType

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_ROOT = REPO_ROOT / "ui"

WHITE = int(Color.WHITE)

# promotion picker rows top to bottom; (move type, piece kind for texture)
PROMO_ORDER = [
  (MoveType.PROMOTION_QUEEN, 5), (MoveType.PROMOTION_KNIGHT, 2),
  (MoveType.PROMOTION_ROOK, 4), (MoveType.PROMOTION_BISHOP, 3),
]


class Snapshot:
  """View of the current position in ABSOLUTE coordinates (a1=0 .. h8=63,
  white POV). The engine encodes board and moves relative to the side to
  move; this is the only place that unflips them. Cached so nothing reads
  the batch's shared encode buffers while the bot searches."""

  def __init__(self, batch: GameBatch):
    board, _, _, _, _, moves, num_moves = batch.get_encoding()
    self.to_move = int(batch.to_moves()[0])
    self.result = int(batch.results()[0])
    flip = 0 if self.to_move == WHITE else 56
    self.board = [0] * 64
    for s in range(64):
      code = int(board[0][s])
      if not code:
        continue
      kind = (code - 1) % 6 + 1                       # 1-6 = P N B R Q K
      is_white = (code <= 6) == (self.to_move == WHITE)
      self.board[s ^ flip] = kind + (0 if is_white else 6)
    self.moves = []                                   # (from, to, mtype, engine idx)
    for i in range(int(num_moves[0])):
      f, t, mt, _ = (int(v) for v in moves[0][i])
      self.moves.append((f ^ flip, t ^ flip, mt, i))


def pretty(move_str: str) -> str:
  for k, v in [(" PROMOTION_QUEEN", "=Q"), (" PROMOTION_ROOK", "=R"),
               (" PROMOTION_BISHOP", "=B"), (" PROMOTION_KNIGHT", "=N"),
               (" EN_PASSANT", " ep"), (" CASTLE", " O-O")]:
    move_str = move_str.replace(k, v)
  return move_str


class Session:
  "One game: its own GameBatch lane plus the per-game UI state."

  def __init__(self, color: str, sim_cap: int):
    if color == "white":
      self.human = WHITE
    elif color == "black":
      self.human = 1 - WHITE
    else:
      self.human = secrets.randbelow(2)
    self.batch = GameBatch(1, sim_cap=sim_cap)
    self.game = self.batch[0]
    self.snap = Snapshot(self.batch)
    self.history = []            # pretty move strings, ply order
    self.last_move = None        # [abs from, abs to]
    self.bot_eval = None         # bot's search value, bot POV
    self.thinking = False

  def play(self, engine_idx: int, abs_from: int, abs_to: int):
    self.history.append(pretty(str(self.game.moves()[engine_idx])))
    self.game.play(engine_idx)
    self.last_move = [abs_from, abs_to]
    self.snap = Snapshot(self.batch)

  def human_turn(self) -> bool:
    return (self.snap.result == int(GameResult.ONGOING)
            and self.snap.to_move == self.human and not self.thinking)


class HumanArena:
  """Serves games of humans vs one bot. Unlike arena.Arena, the bot's color
  varies per game — each browser picks a side when it starts a Session."""

  def __init__(self, bot: ChessbotBase, sim_cap: int = 4096):
    self.bot = bot
    self.sim_cap = sim_cap       # per-session batch; must fit the bot's
                                 # search if it runs on the caller's batch
    self.bot_lock = asyncio.Lock()

  def serve(self, host: str, port: int):
    app = tornado.web.Application([
      (r"/", PageHandler),
      (r"/ws", GameSocket, {"arena": self}),
      (r"/assets/(.*)", tornado.web.StaticFileHandler,
       {"path": str(UI_ROOT / "assets")}),
    ])
    app.listen(port, address=host)
    print(f"{self.bot.VERSION} serving on http://{lan_ip()}:{port}/  "
          "(Ctrl-C to stop)", flush=True)
    try:
      tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
      pass


class PageHandler(tornado.web.RequestHandler):
  def get(self):
    self.write((UI_ROOT / "play.html").read_text())


class GameSocket(tornado.websocket.WebSocketHandler):
  def initialize(self, arena: HumanArena):
    self.arena = arena

  def open(self):
    self.sess = None
    self.write_message(json.dumps({
      "type": "hello",
      "botName": self.arena.bot.VERSION,
      "results": {
        "ongoing": int(GameResult.ONGOING),
        "checkmate": int(GameResult.CHECKMATE),
        "stalemate": int(GameResult.STALEMATE),
        "draw50": int(GameResult.DRAW_OOM),
        "drawRep": int(GameResult.DRAW_REPETITION),
      },
      "enPassant": int(MoveType.EN_PASSANT),
      "promoOrder": [[int(mt), kind] for mt, kind in PROMO_ORDER],
    }))

  def push_state(self):
    if self.ws_connection is None or self.sess is None:
      return
    s = self.sess
    self.write_message(json.dumps({
      "type": "state",
      "board": s.snap.board,
      "toMove": s.snap.to_move,
      "result": s.snap.result,
      "human": s.human,
      "lastMove": s.last_move,
      "history": s.history,
      "eval": s.bot_eval,
      "thinking": s.thinking,
      # legal moves only when it's actually the human's turn
      "moves": [list(m) for m in s.snap.moves] if s.human_turn() else [],
    }))

  def on_message(self, raw):
    try:
      msg = json.loads(raw)
    except ValueError:
      return
    if msg.get("type") == "new":
      # don't yank the board mid-search; its batch is in the executor thread
      if self.sess is not None and self.sess.thinking:
        return
      self.sess = Session(msg.get("color", "random"), self.arena.sim_cap)
      self.push_state()
      self.maybe_think()
    elif msg.get("type") == "move" and self.sess is not None:
      sess = self.sess
      if not sess.human_turn():
        return
      idx = msg.get("idx")
      cand = next((m for m in sess.snap.moves if m[3] == idx), None)
      if cand is None:
        return
      sess.play(idx, cand[0], cand[1])
      self.push_state()
      self.maybe_think()

  def maybe_think(self):
    sess = self.sess
    if (sess.snap.result == int(GameResult.ONGOING)
        and sess.snap.to_move != sess.human):
      asyncio.ensure_future(self.think(sess))

  async def think(self, sess: Session):
    sess.thinking = True
    self.push_state()
    loop = asyncio.get_running_loop()
    try:
      async with self.arena.bot_lock:
        # the browser may have gone away while we queued behind another game
        if self.ws_connection is None or self.sess is not sess:
          return
        moves, info = await loop.run_in_executor(
          None, self.arena.bot.select_moves, sess.batch)
    finally:
      sess.thinking = False
    if self.ws_connection is None or self.sess is not sess:
      return
    idx = int(np.asarray(moves)[0])
    value = info.get("value")
    sess.bot_eval = float(np.asarray(value)[0]) if value is not None else None
    f, t, _, _ = next(m for m in sess.snap.moves if m[3] == idx)
    sess.play(idx, f, t)
    self.push_state()

  def on_close(self):
    self.sess = None             # session (and its GameBatch) dies with us
    print("client disconnected", flush=True)


def lan_ip() -> str:
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    s.connect(("8.8.8.8", 80))   # no packets sent; just picks the LAN iface
    return s.getsockname()[0]
  except OSError:
    return "127.0.0.1"
  finally:
    s.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("config",
                      help="config name (configs/<name>.json with a run)")
  parser.add_argument("--checkpoint", default=None,
                      help="checkpoint step (default: latest)")
  parser.add_argument("--port", type=int, default=8000)
  parser.add_argument("--host", default="0.0.0.0",
                      help="use 127.0.0.1 to keep it localhost-only")
  parser.add_argument("--sim-cap", type=int, default=None,
                      help="session batch sim cap; must fit searches that "
                           "run on the caller's batch (v3 needs roughly "
                           "n_sims * branching). Default: the bot config's "
                           "sim_cap if it declares one, else 4096")
  parser.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                      help="override a bot config value at inference, e.g. "
                           "--set n_sims=512 (repeatable, dotted paths ok; "
                           "the stored run config is untouched)")
  args = parser.parse_args()

  overrides = {}
  for item in args.set:
    key, sep, value = item.partition("=")
    if not sep:
      parser.error(f"--set {item!r} is not KEY=VALUE")
    try:
      value = json.loads(value)     # numbers / bools / lists; else string
    except ValueError:
      pass
    overrides[key] = value

  print("loading bot...", flush=True)
  bot = load_bot(
    args.config,
    args.checkpoint,
    configs_dir=REPO_ROOT / "configs",
    runs_dir=REPO_ROOT / "runs",
    overrides=overrides,
  )
  bot_config = getattr(bot, "config", None)  # v0 keeps no config around
  sim_cap = args.sim_cap or getattr(bot_config, "sim_cap", None) or 4096
  HumanArena(bot, sim_cap=sim_cap).serve(args.host, args.port)


if __name__ == "__main__":
  main()
