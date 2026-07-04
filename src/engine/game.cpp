#include "engine.cpp"
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <cassert>

// maximum amount of moves is technically 218;
// but adds a lot of bloat to modeling.
// either limit to 128 or 256 for total tokens,
// then run multiple times if called for.
// note: since moves aren't positional they don't increase parameter counts.
// (but does increase compute)

constexpr int MAX_MOVES_CAPACITY = 256;

enum class GameResult: uint8_t {
  ONGOING = 0,
  CHECKMATE = 1,
  STALEMATE = 2,
  DRAW_OOM = 3,
  DRAW_REPETITION = 4
};

template <class T>
struct Arena {
  std::vector<T> buffer;
  int offset = 0;
  
  Arena(int capacity) : buffer(capacity) {}

  int alloc(int k = 1) {
    assert(offset + k <= (int) buffer.size());
    int i = offset;
    offset += k;
    return i;
  }
  void reset() {
    offset = 0;
  }
  T& operator[](int i) {
    return buffer[i];
  }
  const T& operator[](int i) const {
    return buffer[i];
  }
};

struct State {
  Board board;
  int32_t parent_state;
  int32_t actions_offset;
  int32_t num_actions;
  uint8_t repetition_count;
  uint8_t halfmove_clock;
  GameResult result;
};

struct Action {
  int32_t parent_offset;
  int32_t child_offset;
  Move move;
};

class ChessInterface {
public:
  ChessInterface(int sim_cap = 64) : 
    states_(sim_cap + 1), 
    actions_((sim_cap + 1) * MAX_MOVES_CAPACITY) 
  {
    reset();
  }

  void reset() {
    root_board_ = init_board();
    seen_.clear();
    seen_[root_board_.hash] = 1;
    halfmove_clock_ = 0;
    reset_tree();
  }

  GameResult play(int root_action) {
    Move move = actions_[states_[0].actions_offset + root_action].move;
    bool halfmove_reset = reset_halfmove_clock(root_board_, move);
    make_move(root_board_, move);
    halfmove_clock_ = halfmove_reset ? 0 : halfmove_clock_ + 1;
    seen_[root_board_.hash]++;
    reset_tree();
    return states_[0].result;
  }

  void reset_tree() {
    states_.reset();
    actions_.reset();
    add_state_node(
      root_board_,
      -1,
      halfmove_clock_,
      repetition_count_base(root_board_.hash)
    );
  }

  int get_child_idx(int action_idx) {
    Action& a = actions_[action_idx];
    if (a.child_offset >= 0)
      return a.child_offset;
    Board board = states_[a.parent_offset].board;
    make_move(board, a.move);
    uint8_t halfmove_clock = reset_halfmove_clock(
      states_[a.parent_offset].board, a.move
    ) ? 0 : states_[a.parent_offset].halfmove_clock + 1;
    a.child_offset = add_state_node(
      board,
      a.parent_offset,
      halfmove_clock,
      repetition_count_simulated(board.hash, a.parent_offset)
    );
    return a.child_offset;
  }

  int num_actions(int s) { return states_[s].num_actions; }
  int actions_offset(int s) { return states_[s].actions_offset; }
  int state_result(int s) { return (int) states_[s].result; }
  int to_move() const { return (int) root_board_.to_move; }

  int encode(
    int s_idx, 
    uint8_t* board_out,
    uint8_t* castling_out,
    int8_t* en_passant_square_out,
    uint8_t* repetition_count_out,
    uint8_t* halfmove_clock_out,
    uint8_t* moves_out,
    int32_t* num_moves_out
  ) const {
    const State& s = states_[s_idx];
    const Board& board = s.board;
    const bool flip = board.to_move == PieceColor::BLACK;
    const auto canon = [&](uint8_t sq) { return flip ? (sq ^ 0b111000) : sq; };

    for (int i = 0; i < 64; i++) {
      uint8_t c = board.mailbox[canon(i)];
      board_out[i] = (flip && c) ? (c <= 6 ? c + 6 : c - 6) : c;
    }

    const uint8_t castling = board.castling;
    castling_out[0] = flip 
      ? (((castling >> 2) & 0b11) | ((castling & 0b11) << 2)) 
      : castling;
    en_passant_square_out[0] = board.en_passant_square < 0 ? -1 : 
      canon(board.en_passant_square);
    const int prior_repetition_count = (int) s.repetition_count - 1;
    repetition_count_out[0] = prior_repetition_count > 2 
      ? 2 : prior_repetition_count;
    halfmove_clock_out[0] = s.halfmove_clock;

    const int a_offset = s.actions_offset, a_n = s.num_actions;
    for (int i = 0; i < a_n; i++) {
      const Move move = actions_[a_offset + i].move;
      uint8_t* row = moves_out + (i << 2);
      row[0] = canon(move.from());
      row[1] = canon(move.to());
      row[2] = move.flag() == MoveFlag::PROMOTION 
        ? 2 + move.promo() : move.flag();
      row[3] = 1 + (uint8_t) ((board.mailbox[move.from()] - 1) % 6);
    }
    num_moves_out[0] = a_n;
    return a_n;
  }

private:
  int add_state_node(
    const Board& board, 
    int parent, 
    uint8_t halfmove_clock, 
    uint8_t repetition_count
  ) {
    int s_idx = states_.alloc();
    State& s = states_[s_idx];
    s.board = board;
    s.parent_state = parent;
    s.halfmove_clock = halfmove_clock;
    s.repetition_count = repetition_count;

    MoveList movelist; 
    get_moves(board, movelist);
    s.num_actions = movelist.n;
    s.actions_offset = actions_.alloc(movelist.n);
    for (int i = 0; i < movelist.n; i++)
      actions_[s.actions_offset + i] = { s_idx, -1, movelist.moves[i] };
    if (movelist.n == 0) {
      s.result = ::in_check(board, board.to_move)
        ? GameResult::CHECKMATE : GameResult::STALEMATE;
    } else if (halfmove_clock >= 100)
      s.result = GameResult::DRAW_OOM;
    else if (repetition_count >= 3)
      s.result = GameResult::DRAW_REPETITION;
    else
      s.result = GameResult::ONGOING;
    return s_idx;
  }

  static bool reset_halfmove_clock(const Board& board, Move move) {
    bool pawn = ((board.mailbox[move.from()] - 1) % 6 == PieceType::PAWN);
    bool capture = (board.mailbox[move.to()] != 0) 
      || (move.flag() == MoveFlag::ENPASSANT);
    return pawn || capture;
  }

  uint8_t repetition_count_base(uint64_t hash) {
    auto res = seen_.find(hash);
    return res == seen_.end() ? 0 : res->second;
  }
  uint8_t repetition_count_simulated(uint64_t hash, int sim_state) {
    uint8_t count = repetition_count_base(hash) + 1;
    for (int s = sim_state; 
      s >= 0 && states_[s].parent_state >= 0;
      s = states_[s].parent_state
    )
      if (states_[s].board.hash == hash) count++;
    return count;
  }

  Board root_board_;
  std::unordered_map<uint64_t,int> seen_;
  int halfmove_clock_ = 0;
  Arena<State> states_;
  Arena<Action> actions_;
};
