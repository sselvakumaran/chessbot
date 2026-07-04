from contextlib import nullcontext
from dataclasses import dataclass, fields
import numpy as np
import torch
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
	active_mask: np.ndarray | None = None

	def tensors(self):
		return tuple(
			torch.from_numpy(getattr(self, f.name)) 
			for f in fields(self) 
			if f.name != "active_mask"
		)

	def __len__(self): return self.board.shape[0]
	def __getitem__(self, i: int) -> 'Encoding':
		return Encoding(**{
			f.name: None 
				if f.name == "active_mask" 
				else getattr(self, f.name)[i].copy()
			for f in fields(self)
    })

class Evaluator:
	def __init__(self,
		model: Model,
		B: int = 64, # rows per model forward (fixed shape: pad, never shrink)
		autocast_dtype: torch.dtype | None = None # e.g. torch.bfloat16
	):
		self.model = model
		self.B = B
		self.max_moves = model.max_moves
		self.device = next(model.parameters()).device
		self.autocast_dtype = autocast_dtype

		# persistent device-side inputs: every call copies into the same
		# storage, so compile(mode="reduce-overhead") / CUDA graphs can replay
		max_moves, device = model.max_moves, self.device
		self._in = [
			torch.zeros((B, 64), dtype=torch.uint8, device=device), # board
			torch.zeros((B,), dtype=torch.uint8, device=device), # castling
			torch.zeros((B,), dtype=torch.int8, device=device), # en_passant
			torch.zeros((B,), dtype=torch.uint8, device=device), # repetition
			torch.zeros((B,), dtype=torch.uint8, device=device), # halfmove
			torch.zeros((B, max_moves, 4), dtype=torch.uint8, device=device), # moves
			torch.zeros((B,), dtype=torch.int32, device=device), # num_moves
		]

	@torch.inference_mode()
	def _forward_logits(self, *inputs): # np arrays, num_samples <= B each
		num_samples = inputs[0].shape[0]
		for buf, x in zip(self._in, inputs):
			buf[:num_samples].copy_(torch.from_numpy(np.ascontiguousarray(x)))
			if num_samples < self.B:
				buf[num_samples:].zero_()
		ctx = torch.autocast(self.device.type, self.autocast_dtype) \
			if self.autocast_dtype is not None else nullcontext()
		with ctx:
			p_logits, values, _ = self.model(*self._in)
		return (
			p_logits.float().cpu().numpy(),
			values.float().squeeze(-1).cpu().numpy()
		)

	def eval(self, enc: Encoding):
		p_logits, values = self.eval_logits(enc)
		game_batch_size, moves_capacity = p_logits.shape
		valid = np.arange(moves_capacity)[None, :] \
			< enc.num_moves.reshape(game_batch_size, 1)
		x = np.where(valid, p_logits, -np.inf)
		x = x - x.max(axis=1, keepdims=True)
		ex = np.where(valid, np.exp(x), 0.0)
		denom = ex.sum(axis=1, keepdims=True)
		return ex / np.where(denom == 0, 1.0, denom), values

	def eval_logits(self, enc: Encoding):
		# converts between two shapes: (game_batch_size, moves_capacity)
		# outside, (B, max_moves) through the model. a SAMPLE is one
		# (game, chunk) pair: a game with > max_moves legal moves occupies one
		# sample per max_moves-slice of its move list
		game_batch_size = len(enc)
		max_moves, B = self.max_moves, self.B
		moves_capacity = enc.moves.shape[1] # engine slots per game, >= num_moves
		num_moves = enc.num_moves
		chunks_per_game = np.maximum(1, -(-num_moves // max_moves))
		max_chunks = int(chunks_per_game.max()) if game_batch_size else 1
		moves_capacity_pad = max(moves_capacity, max_chunks * max_moves)

		moves_padded = np.zeros((game_batch_size, moves_capacity_pad, 4), enc.moves.dtype)
		moves_padded[:, :moves_capacity] = enc.moves

		p_logits = np.full((game_batch_size, moves_capacity_pad), -1e9, dtype=np.float32)
		values = np.zeros((game_batch_size,), dtype=np.float32)

		# sample -> game index
		sample_games = np.concatenate([
			np.where(chunks_per_game > c)[0] for c in range(max_chunks)
		])
		# sample -> chunk index
		sample_chunks = np.concatenate([
			np.full((chunks_per_game > c).sum(), c) for c in range(max_chunks)
		])

		for i in range(0, sample_games.size, B):
			batch_games = sample_games[i:i+B]
			batch_chunks = sample_chunks[i:i+B]
			num_samples = batch_games.size

			cols = (batch_chunks * max_moves)[:, None] + np.arange(max_moves)[None, :]
			moves = moves_padded[batch_games[:, None], cols]
			num_moves_clipped = np.clip(
				num_moves[batch_games] - batch_chunks * max_moves, 0, max_moves
			).astype(np.int32)
			inputs = (
				enc.board[batch_games],
				enc.castling[batch_games],
				enc.en_passant_square[batch_games],
				enc.repetition_count[batch_games],
				enc.halfmove_clock[batch_games],
				moves,
				num_moves_clipped
			)
			batch_p_logits, batch_values = self._forward_logits(*inputs)
			batch_p_logits = batch_p_logits[:num_samples]
			batch_values = batch_values[:num_samples]
			p_logits[batch_games[:, None], cols] = batch_p_logits
			values[batch_games[batch_chunks == 0]] = batch_values[batch_chunks == 0]
		return p_logits[:, :moves_capacity], values

	def eval_states(self, gb: GameBatch, lanes=None, states=None):
		# evaluate arbitrary (lane, state) tree nodes straight off the manager;
		# defaults to every lane's root. gb.encode fills shared buffers that we
		# consume before returning, so no copy is needed
		enc = Encoding(*gb.encode(lanes, states))
		return self.eval_logits(enc)

	def eval_games(self, gb: GameBatch):
		return self.eval(Encoding(*gb.get_encoding()))
