from dataclasses import dataclass, fields
import numpy as np
from tinygrad import Tensor, TinyJit
from models.v0 import Model
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


def _pad_batch(t: Tensor, B: int) -> Tensor:
	n = t.shape[0]
	if n == B:
		return t
	# pad the batch (axis 0) up to B; other axes unchanged
	return t.pad(((0, B - n),) + ((0, 0),) * (t.ndim - 1))

class Evaluator:
	def __init__(self, 
		model: Model, 
		batch_size: int = 64
	):
		self.model = model
		self.batch_size = batch_size
		self.M = model.max_moves
		self._fwd = TinyJit(self._forward)
		self._fwd_logits = TinyJit(self._forward_logits)
	
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

	def eval(self, enc: Encoding):
		# b: positions to evaluate; B: model batch size
		b, B = enc.board.shape[0], self.batch_size
		priors = np.zeros((b, self.M), dtype=np.float32)
		values = np.zeros((b,), dtype=np.float32)
		Tensor.training = False

		inputs = enc.tensors()

		for batch_start in range(0, b, B):
			batch_end = min(batch_start + B, b)
			n = batch_end - batch_start
			chunk = tuple(_pad_batch(t[batch_start:batch_end], B) for t in inputs)
			pr, v = self._fwd(*chunk)
			priors[batch_start:batch_end] = pr.numpy()[:n]
			values[batch_start:batch_end] = v.numpy()[:n]
		return priors, values
	
	def eval_logits(self, enc: Encoding):
		
		b, B = enc.board.shape[0], self.batch_size
		p_logits = np.zeros((b, self.M), dtype=np.float32)
		values = np.zeros((b,), dtype=np.float32)
		Tensor.training = False

		inputs = enc.tensors()

		for batch_start in range(0, b, B):
			batch_end = min(batch_start + B, b)
			n = batch_end - batch_start
			chunk = tuple(_pad_batch(t[batch_start:batch_end], B) for t in inputs)
			pr, v = self._fwd_logits(*chunk)
			p_logits[batch_start:batch_end] = pr.numpy()[:n]
			values[batch_start:batch_end] = v.numpy()[:n]
		return p_logits, values

	def eval_games(self, gb: GameBatch):
		return self.eval(Encoding(*gb.get_encoding()))