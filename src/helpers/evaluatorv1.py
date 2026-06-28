from dataclasses import dataclass, fields
import numpy as np
from tinygrad import Tensor, TinyJit
from models.v1 import Model
from engine.game import GameBatch

@dataclass(frozen=True)
class Encoding: # encoding of a whole GameBatch
	board: np.ndarray
	castling: np.ndarray
	en_passant_square: np.ndarray
	repetition_count: np.ndarray
	halfmove_clock: np.ndarray
	moves: np.ndarray
	num_moves: np.ndarray
	mask: np.ndarray | None = None # bool (b,), True = active game; None = all active

	def tensors(self):
		return tuple(
			Tensor(getattr(self, f.name)) for f in fields(self) if f.name != "mask"
		)

	def __len__(self): return self.board.shape[0]
	def __getitem__(self, i: int) -> 'Encoding':
		return Encoding(**{
			f.name: None if f.name == "mask" else getattr(self, f.name)[i].copy()
			for f in fields(self)
    })

def _to_batch(x: np.ndarray, model_batch_size: int):
	n = x.shape[0]
	if n < model_batch_size:
		x = np.concatenate([
			x,
			np.zeros((model_batch_size - n,) + x.shape[1:], x.dtype)
		], axis=0)
	return Tensor(x)

class Evaluator:
	def __init__(self, 
		model: Model, 
		batch_size: int = 64
	):
		self.model = model
		self.batch_size = batch_size
		self.max_moves = model.max_moves
		self._fwd_logits = TinyJit(self._forward_logits)

	def _forward_logits(self, 
		board, castling, en_passant_square, repetition_count, halfmove_clock, 
		moves, num_moves
	):
		p, v = self.model(
			board, castling, en_passant_square, repetition_count, halfmove_clock, 
			moves, num_moves
		)
		return p.realize(), v.squeeze(-1).realize()

	def eval(self, enc: Encoding):
		# b: positions to evaluate; B: model batch size
		p_logits, values = self.eval_logits(enc)
		game_batch_size, width = p_logits.shape
		valid = np.arange(width)[None, :] \
			< enc.num_moves.reshape(game_batch_size, 1)
		x = np.where(valid, p_logits, -np.inf)
		x = x - x.max(axis=1, keepdims=True)
		ex = np.where(valid, np.exp(x), 0.0)
		denom = ex.sum(axis=1, keepdims=True)
		return ex / np.where(denom == 0, 1.0, denom), values
	
	def eval_logits(self, enc: Encoding):
		Tensor.training = False

		game_batch_size = len(enc)
		max_moves, model_batch_size = self.max_moves, self.batch_size
		width = enc.moves.shape[1]		
		num_moves = enc.num_moves
		chunks = np.maximum(1, -(-num_moves // max_moves))
		num_chunks = int(chunks.max()) if game_batch_size else 1
		width_pad = max(width, num_chunks * max_moves)

		moves_padded = np.zeros((game_batch_size, width_pad, 4), enc.moves.dtype)
		moves_padded[:, :width] = enc.moves

		p_logits = np.full((game_batch_size, width_pad), -1e9, dtype=np.float32)
		values = np.zeros((game_batch_size,), dtype=np.float32)

		# index by game
		games_all = np.concatenate([
			np.where(chunks > c)[0] for c in range(num_chunks)
		])
		# index by chunk
		chunks_all = np.concatenate([
			np.full((chunks > c).sum(), c) for c in range(num_chunks)
		])

		for i in range(0, games_all.size, model_batch_size):
			game = games_all[i:i+model_batch_size]
			chunk = chunks_all[i:i+model_batch_size]
			n = game.size

			cols = (chunk * max_moves)[:, None] + np.arange(max_moves)[None, :]
			moves = moves_padded[game[:, None], cols]
			num_moves_clipped = np.clip(
				num_moves[game] - chunk * max_moves, 0, max_moves
			).astype(np.int32)
			inputs = (
				enc.board[game],
				enc.castling[game],
				enc.en_passant_square[game],
				enc.repetition_count[game],
				enc.halfmove_clock[game],
				moves,
				num_moves_clipped
			)
			_p_logits, _value = self._fwd_logits(*[
				_to_batch(x, model_batch_size) for x in inputs
			])
			_p_logits = _p_logits.numpy()[:n]
			_value = _value.numpy()[:n]
			p_logits[game[:, None], cols] = _p_logits
			values[game[chunk == 0]] = _value[chunk == 0]
		return p_logits[:, :width], values

	def eval_games(self, gb: GameBatch):
		return self.eval(Encoding(*gb.get_encoding()))