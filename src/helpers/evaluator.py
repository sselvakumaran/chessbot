from dataclasses import dataclass, fields
import numpy as np
from tinygrad import Tensor, TinyJit
from models.v0 import Model
from engine.game import GameBatch, MAX_MOVES

MAX_TOKENS = MAX_MOVES + 64 + 2 # 128

@dataclass(frozen=True)
class Encoding:
	board: np.ndarray
	castling: np.ndarray
	en_passant_square: np.ndarray
	repetition_count: np.ndarray
	halfmove_clock: np.ndarray
	moves: np.ndarray
	num_moves: np.ndarray
	
	def tensors(self):
		return tuple(Tensor(getattr(self, f.name)) for f in fields(self))

class Evaluator:
	def __init__(self, 
		model: Model, 
		batch_size: int = 64,
		output_logits: bool = False
	):
		self.model = model
		self.batch_size = batch_size
		self.M = model.max_moves
		self._fwd = TinyJit(
			self._forward_logits if output_logits else self._forward
		)
	
	def _forward(self, 
		board, castling, en_passant_square, repetition_count, halfmove_clock, 
		moves, num_moves
	):
		p, v = self.model(
			board, castling, en_passant_square, repetition_count, halfmove_clock, 
			moves, num_moves
		)
		return p.softmax(axis=-1).realize(), v.squeeze(-1).realize()
	def _forward_logits(self, 
		board, castling, en_passant_square, repetition_count, halfmove_clock, 
		moves, num_moves
	):
		p, v = self.model(
			board, castling, en_passant_square, repetition_count, halfmove_clock, 
			moves, num_moves
		)
		return p.realize(), v.squeeze(-1).realize()
	
	@staticmethod
	def _pad_batch(t: Tensor, B: int) -> Tensor:
		n = t.shape[0]
		if n == B: 
			return t
		return t.pad((0, B - n))

	def eval(self, enc: Encoding):
		# b: positions to evaluate; B: model batch size
		b, B = enc.board.shape[0], self.batch_size
		priors = np.zeros((b, self.M), dtype=np.float32)
		values = np.zeros((b,), dtype=np.float32)
		Tensor.training = False

		inputs = enc.tensors()

		for s in range(0, b, B):
			e = min(s + B, b)
			n = e - s
			chunk = tuple(self._pad_batch(t[s:e], B) for t in inputs)
			pr, v = self._fwd(*chunk)
			priors[s:e] = pr.numpy()[:n]
			values[s:e] = v.numpy()[:n]
		return priors, values

	def eval_games(self, gb: GameBatch):
		return self.eval(Encoding(*gb.get_encoding()))