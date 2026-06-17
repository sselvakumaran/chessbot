#include "engine.cpp"
#include <unordered_map>
#include <vector>
#include <cstdlib>

// maximum amount of moves is technically 218;
// but adds a lot of bloat to modeling.
// either limit to 128 or 256 for total tokens,
// then run multiple times if called for.
// note: since moves aren't positional they don't increase parameter counts.
// (but does increase compute)

constexpr int MAX_MOVES = 128 - 64 - 2;

enum class GameResult: int {
  ONGOING = 0,
  CHECKMATE = 1,
  STALEMATE = 2,
  DRAW_OOM = 3,
  DRAW_REPETITION = 4
};

struct HistoryEntry {
  Move move;
  Undo undo;
  PieceType captured;
  int capture_square;
  int prev_halfmove;
};

class ChessInterface {
public:
  ChessInterface() { reset(); }

  void reset() {
    board_ = init_board();
    history_.clear();
    seen_.clear();
    halfmove_clock_ = 0;
    seen_[board_.hash] = 1;
    refresh();
  }

  GameResult result() const { return result_; }
  int num_moves() const { return legal_.n; }
  int to_move() const { return board_.to_move; }
  bool in_check() const { return ::in_check(board_, board_.to_move); }

  const Board& position() const { return board_; }
  const MoveList& legal_moves() const { return legal_; }

  void encode_position(
    uint8_t* board_out,
    uint8_t* castling_out,
    int8_t* en_passant_square_out,
    uint8_t* repetition_count_out,
    uint8_t* halfmove_clock_out
  ) const {
    const bool flip = board_.to_move == PieceColor::BLACK;
    const auto canon = [&](uint8_t s) { return flip ? (s ^ 0b111000) : s; };
    for (int i = 0; i < 64; i++)
      board_out[i] = piece_code(canon(i));
    const uint8_t c = board_.castling;
    castling_out[0] = flip ? ((c >> 2) & 0b11) | ((c & 0b11) << 2) : c;
    en_passant_square_out[0] = board_.en_passant_square < 0 ? -1 : canon(board_.en_passant_square);
    const int prior = repetition_count() - 1;
    repetition_count_out[0] = prior > 2 ? 2 : prior;
    halfmove_clock_out[0] = halfmove_clock_;
  }

  int build_encoding(
    uint8_t* board_out,
    uint8_t* castling_out,
    int8_t* en_passant_square_out,
    uint8_t* repetition_count_out,
    uint8_t* halfmove_clock_out,
    uint8_t* moves_out,
    int32_t* num_moves_out
  ) const {

    const bool flip = board_.to_move == PieceColor::BLACK;
    const auto canon = [&](uint8_t s) { return flip ? (s ^ 0b111000) : s; };

    encode_position(
      board_out,
      castling_out,
      en_passant_square_out,
      repetition_count_out,
      halfmove_clock_out
    );

    if (legal_.n > MAX_MOVES) std::abort();

    for (int i = 0; i < legal_.n; i++) {
      const Move move = legal_.moves[i];
      uint8_t* row = moves_out + (i << 2);
      row[0] = canon(move.from());
      row[1] = canon(move.to());
      row[2] = move.flag() == MoveFlag::PROMOTION ?
        2 + move.promo() : move.flag();
      row[3] = 1 + piece_on(board_, move.from());
    }
    num_moves_out[0] = legal_.n;
    return legal_.n;
  }

  GameResult play(int idx) {
    if (idx < 0 || idx >= legal_.n) std::abort();
    Move m = legal_.moves[idx];

    HistoryEntry e;
    e.move = m;
    e.prev_halfmove = halfmove_clock_;
    if (m.flag() == MoveFlag::ENPASSANT) {
      e.captured = PieceType::PAWN;
      e.capture_square = (board_.to_move == PieceColor::WHITE)
        ? m.to() - 8 : m.to() + 8;
    } else {
      e.captured = piece_on(board_, m.to());
      e.capture_square = (e.captured == PieceType::NONE) ? -1 : m.to();
    }
    PieceType moved = piece_on(board_, m.from());

    make_move(board_, m, e.undo);
    history_.push_back(e);
    halfmove_clock_ = (moved == PieceType::PAWN
      || e.captured != PieceType::NONE)
      ? 0 : halfmove_clock_ + 1;
    seen_[board_.hash]++;
    refresh();
    return result_;
  }

  void undo() {
    if (history_.empty()) return;
    HistoryEntry e = history_.back();
    history_.pop_back();
    if (--seen_[board_.hash] == 0)
      seen_.erase(board_.hash);
    unmake_move(board_, e.move, e.undo);
    halfmove_clock_ = e.prev_halfmove;
    refresh();
  }

private:
  void refresh() {
    get_moves(board_, legal_);
    if (legal_.n == 0) result_ = ::in_check(board_, board_.to_move)
      ? GameResult::CHECKMATE : GameResult::STALEMATE;
    else if (halfmove_clock_ >= 100) result_ = GameResult::DRAW_OOM;
    else if (repetition_count() >= 3) result_ = GameResult::DRAW_REPETITION;
    else result_ = GameResult::ONGOING;
  }

  uint8_t piece_code(int s) const {
    const Bitboard mask = ONE << s;
    if (!(board_.occupancy & mask)) return 0;
    const bool plr = board_.player_occupancy[board_.to_move] & mask;
    return (plr ? 1 : 7) + piece_on(board_, s);
  }

  int repetition_count() const {
    auto it = seen_.find(board_.hash);
    return it == seen_.end() ? 0 : it->second;
  }

  Board board_;
  MoveList legal_;
  GameResult result_ = GameResult::ONGOING;
  int halfmove_clock_ = 0;
  std::vector<HistoryEntry> history_;
  std::unordered_map<uint64_t,int> seen_;
};
