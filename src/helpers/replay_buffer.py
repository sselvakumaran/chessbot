import numpy as np
from engine.game import MAX_MOVES
from helpers.evaluator import Encoding

# circular buffer
class ReplayBuffer:
	def __init__(self, capacity: int):
		self.capacity, self.size, self.ptr = capacity, 0, 0
		self.board = np.zeros((capacity, 64), np.uint8)
		self.castling = np.zeros((capacity, ), np.uint8)
		self.en_passant_square = np.zeros((capacity, ), np.int8)
		self.repetition_count = np.zeros((capacity, ), np.uint8)
		self.halfmove_clock = np.zeros((capacity, ), np.uint8)
		self.moves = np.zeros((capacity, MAX_MOVES, 4), np.uint8)
		self.num_moves = np.zeros((capacity, ), np.int32)

		# store pi, z
		self.pi = np.zeros((capacity, MAX_MOVES), np.float32)
		self.z = np.zeros((capacity, ), np.float32)
	
	def add(self, enc: Encoding, pi, z):
		i = self.ptr
		self.board[i] = enc.board
		self.castling[i] = enc.castling
		self.en_passant_square[i] = enc.en_passant_square
		self.repetition_count[i] = enc.repetition_count
		self.halfmove_clock[i] = enc.halfmove_clock
		self.moves[i] = enc.moves
		self.num_moves[i] = enc.num_moves

		self.pi = pi
		self.z = z

		self.ptr = (self.ptr + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)
	
	def sample(self, batch_size: int, rng: np.random.Generator):
		idx = rng.integers(0, self.size, size=batch_size)
		return (
			Encoding(
				board=self.board[idx],
				castling=self.castling[idx],
				en_passant_square=self.en_passant_square[idx],
				repetition_count=self.repetition_count[idx],
				halfmove_clock=self.halfmove_clock[idx],
				moves=self.moves[idx],
				num_moves=self.num_moves[idx]
			), 
			self.pi[idx], 
			self.z[idx]
		)
	
	def save(self, path):
		size = self.size
		np.savez_compressed(path, 
			board=self.board[:size],
			castling=self.castling[:size],
			en_passant_square=self.en_passant_square[:size],
			repetition_count=self.repetition_count[:size],
			halfmove_clock=self.halfmove_clock[:size],
			moves=self.moves[:size],
			num_moves=self.num_moves[:size],
			meta=np.array([self.capacity, size, self.ptr], dtype=np.uint64)
		)
	
	@classmethod
	def load(cls, path):
		with np.load(path) as data:
			capacity, size, ptr = (int(x) for x in data['meta'])
			out = cls(capacity)
			out.size = size
			out.ptr = ptr

			out.board=data['board']
			out.castling=data['castling']
			out.en_passant_square=data['en_passant_square']
			out.repetition_count=data['repetition_count']
			out.halfmove_clock=data['halfmove_clock']
			out.moves=data['moves']
			out.num_moves=data['num_moves']
		return out