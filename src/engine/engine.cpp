#include <array>
#include <vector>
#include <cstdint>
#include <bit>

using Bitboard = uint64_t;
enum PieceColor : uint8_t {WHITE, BLACK};
enum PieceType : uint8_t {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NONE};

struct Board {
  Bitboard pieces[2][6];
  Bitboard player_occupancy[2];
  Bitboard occupancy;
  PieceColor to_move;
  uint8_t castling; // flags for rights to castle
  int8_t en_passant_square; // what square is available to en passant
  uint64_t hash; // zobrist hash 
};
// MOVE:
// 0-5: from
// 6-11: to
// 12-13: promo data (0K 1B 2R 3Q)
// 14-15: flag
enum MoveFlag : uint16_t { NORMAL, CASTLE, ENPASSANT, PROMOTION };
struct Move {
  uint16_t d;
  uint8_t from() const { return d & 0b111111; }
  uint8_t to() const { return (d >> 6) & 0b111111; }
  uint8_t promo() const { return ((d >> 12) & 3) + KNIGHT; }
  MoveFlag flag() const { return MoveFlag(d >> 14); }
};
struct Undo {
  PieceType captured;
  uint8_t castling;
  int8_t en_passant_square;
  uint64_t hash;
};
enum : uint8_t { CR_WK = 1, CR_WQ = 2, CR_BK = 4, CR_BQ = 8 };

struct MoveList {
  Move moves[256];
  uint8_t n = 0;
  void add(Move m) { moves[n++] = m; }
};

// DEFINE BITBOARD CONSTANTS HERE
struct Magic {
  Bitboard mask, magic;
  Bitboard *attacks;
  uint8_t shift;
};

inline Bitboard knight_attacks[64], king_attacks[64], pawn_attacks[2][64];

Bitboard ONE = 1ULL;
constexpr int to_pos(int r, int c) { return (r << 3) | c; }

constexpr Bitboard FILE_A = 0x0101010101010101ULL, 
  FILE_H = 0x8080808080808080ULL;
constexpr Bitboard NOT_A = ~FILE_A, NOT_H = ~FILE_H;
constexpr Bitboard NOT_AB = ~(FILE_A | (FILE_A << 1));
constexpr Bitboard NOT_GH = ~(FILE_H | (FILE_H >> 1));
constexpr Bitboard RANK_3 = 0x0000000000FF0000ULL, 
  RANK_6 = 0x0000FF0000000000ULL;
void init_leapers() {
  for (int s = 0; s < 64; s++) {
    Bitboard b = ONE << s;
    knight_attacks[s] = 
      ((b << 17) & NOT_A) | ((b << 15) & NOT_H) |
      ((b << 10) & NOT_AB) | ((b << 6) & NOT_GH) |
      ((b >> 6)  & NOT_AB) | ((b >> 10) & NOT_GH) |
      ((b >> 15) & NOT_A)  | ((b >> 17) & NOT_H);
    king_attacks[s] = 
      ((b << 1) & NOT_A) | ((b >> 1) & NOT_H) | 
      (b << 8) | (b >> 8) |
      ((b << 9) & NOT_A) | ((b << 7) & NOT_H) |
      ((b >> 7) & NOT_A) | ((b >> 9) & NOT_H);
    pawn_attacks[PieceColor::WHITE][s] = 
      ((b << 9) & NOT_A) | ((b << 7) & NOT_H);
    pawn_attacks[PieceColor::BLACK][s] = 
      ((b >> 7) & NOT_A) | ((b >> 9) & NOT_H);
  }
}
Bitboard slider_attacks(int s, Bitboard occupancy, const int dirs[4][2]) {
  Bitboard b = 0;
  int s_r = s >> 3, s_c = s & 0b111;
  for (int d = 0; d < 4; d++) {
    int r = s_r + dirs[d][0], c = s_c + dirs[d][1];
    while (r >= 0 && r < 8 && c >= 0 && c < 8) {
      b |= ONE << to_pos(r, c);
      if (occupancy & (ONE << to_pos(r, c))) break;
      r += dirs[d][0]; c += dirs[d][1];
    }
  }
  return b;
}

Bitboard slider_mask(int s, const int dirs[4][2]) {
  Bitboard mask = 0;
  int s_r = s >> 3, s_c = s & 0b111;
  for (int d = 0; d < 4; d++) {
    int r = s_r + dirs[d][0], c = s_c + dirs[d][1];
    while (true) {
      int r2 = r + dirs[d][0], c2 = c + dirs[d][1];
      if (r2 < 0 || r2 >= 8 || c2 < 0 || c2 >= 8) break;
      mask |= ONE << to_pos(r, c);
      r = r2; c = c2;
    }
  }
  return mask;
}

// MAGIC
uint64_t rng = 0x2545F4914F6CDD1DULL;
uint64_t rand_u64() {
  rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27; 
  return rng * 0x2545F4914F6CDD1DULL;
}
uint64_t rand_magic() { return rand_u64() & rand_u64() & rand_u64(); }

inline Bitboard rook_table[64][4096], bishop_table[64][512];
const int ROOK_DIRS[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
const int BISHOP_DIRS[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
inline Magic rook_magic[64], bishop_magic[64];

uint64_t zobrist_piece[2][6][64], zobrist_castle[16], zobrist_ep[8], zobrist_side;

void init_zobrist() {
  for (int color=0; color < 2; color++)
    for (int type=0; type < 6; type++)
      for (int s = 0; s < 64; s++)
        zobrist_piece[color][type][s] = rand_u64();
  for (int i = 0; i < 16; i++)
    zobrist_castle[i] = rand_u64();
  for (int i = 0; i < 8; i++)
    zobrist_ep[i] = rand_u64();
  zobrist_side = rand_u64();
}

void init_magics(
  Magic table[64], Bitboard *storage, 
  int rowlen, const int dirs[4][2]
) {
  for (int s = 0; s < 64; s++) {
    Bitboard mask = slider_mask(s, dirs);
    int bits = std::popcount(mask), shift = 64 - bits, count = 1 << bits;

    std::vector<Bitboard> occs(count), atts(count);
    Bitboard sub = 0;
    for (int i = 0; i < count; i++) {
      occs[i] = sub;
      atts[i] = slider_attacks(s, sub, dirs);
      sub = (sub - mask) & mask;
    }

    Bitboard magic;
    std::vector<Bitboard> used(count);
    std::vector<uint8_t> filled(count);
    while (true) {
      magic = rand_magic();
      if (std::popcount((mask * magic) >> 56) < 6) continue;
      std::fill(filled.begin(), filled.end(), 0);
      bool ok = true;
      for (int i = 0; i < count && ok; i++) {
        uint64_t idx = (occs[i] * magic) >> shift;
        if (!filled[idx]) {
          filled[idx] = 1; 
          used[idx] = atts[i];
        }
        else if (used[idx] != atts[i]) 
          ok = false;
      }
      if (ok) break;
    }
    Bitboard* row = storage + s * rowlen;
    for (int i = 0; i < count; i++)
      row[(occs[i] * magic) >> shift] = atts[i];
    table[s] = { mask, magic, row, (uint8_t) shift };
  }
}

void init_attack_tables() {
  init_leapers();
  init_magics(rook_magic, &rook_table[0][0], 4096, ROOK_DIRS);
  init_magics(bishop_magic, &bishop_table[0][0], 512, BISHOP_DIRS);
  init_zobrist();
}

inline int lsb(Bitboard b) { return std::countr_zero(b); }
inline int pop_lsb(Bitboard &b) { 
  int s = lsb(b); 
  b &= b - 1; 
  return s;
}

uint64_t zobrist_hash(const Board &board) {
  uint64_t hash = 0;
  for (int color = 0; color < 2; color++)
    for (int type = 0; type < 6; type++) {
      Bitboard b = board.pieces[color][type];
      while (b)
        hash ^= zobrist_piece[color][type][pop_lsb(b)];
  }
  hash ^= zobrist_castle[board.castling];
  if (board.en_passant_square != -1) 
    hash ^= zobrist_ep[board.en_passant_square & 0b111];
  if (board.to_move == PieceColor::BLACK) hash ^= zobrist_side;
  return hash;
}

inline PieceColor opposite_color(PieceColor color) {
  return static_cast<PieceColor>(color ^ 1);
}

inline Bitboard bishop_attacks(uint8_t s, Bitboard occupancy) {
  const Magic &m = bishop_magic[s];
  return m.attacks[((occupancy & m.mask) * m.magic) >> m.shift];
}
inline Bitboard rook_attacks(uint8_t s, Bitboard occupancy) {
  const Magic &m = rook_magic[s];
  return m.attacks[((occupancy & m.mask) * m.magic) >> m.shift];
}
inline Bitboard queen_attacks(uint8_t s, Bitboard occupancy) {
  return rook_attacks(s, occupancy) | bishop_attacks(s, occupancy);
}

auto encode_move = [](int from, int to, MoveFlag flag, int promo) {
  return Move{ static_cast<uint16_t>(
    from | (to << 6) | (promo << 12) | (flag << 14)
  ) };
};

void gen_pawn_moves(const Board &board, MoveList &moves) {
  PieceColor color_p = board.to_move, color_o = opposite_color(color_p);
  Bitboard board_o = board.player_occupancy[color_o];
  Bitboard empty = ~board.occupancy;
  Bitboard pawns = board.pieces[color_p][PieceType::PAWN];

  auto add_pawn = [&](int from, int to, bool promo) {
    if (promo) for (int p = 0; p < 4; p++)
      moves.add(encode_move(from, to, PROMOTION, p));
    else moves.add(encode_move(from, to, NORMAL, 0));
  };

  if (color_p == PieceColor::WHITE) {
    Bitboard move_single = (pawns << 8) & empty;
    Bitboard move_double = ((move_single & RANK_3) << 8) & empty;
    Bitboard capture_l = (pawns << 7) & NOT_H & board_o;
    Bitboard capture_r = (pawns << 9) & NOT_A & board_o;
    for (Bitboard t=move_single; t;) {
      int to = pop_lsb(t); add_pawn(to-8, to, to >= 56);
    }
    for (Bitboard t=move_double; t;) {
      int to = pop_lsb(t); add_pawn(to-16, to, false);
    }
    for (Bitboard t=capture_l; t;) {
      int to = pop_lsb(t); add_pawn(to-7, to, to >= 56);
    }
    for (Bitboard t=capture_r; t;) {
      int to = pop_lsb(t); add_pawn(to-9, to, to >= 56);
    }
  } else {
    Bitboard move_single = (pawns >> 8) & empty;
    Bitboard move_double = ((move_single & RANK_6) >> 8) & empty;
    Bitboard capture_l = (pawns >> 9) & NOT_H & board_o;
    Bitboard capture_r = (pawns >> 7) & NOT_A & board_o;
    for (Bitboard t=move_single; t;) {
      int to = pop_lsb(t); add_pawn(to+8, to, to < 8);
    }
    for (Bitboard t=move_double; t;) {
      int to = pop_lsb(t); add_pawn(to+16, to, false);
    }
    for (Bitboard t=capture_l; t;) {
      int to = pop_lsb(t); add_pawn(to+9, to, to < 8);
    }
    for (Bitboard t=capture_r; t;) {
      int to = pop_lsb(t); add_pawn(to+7, to, to < 8);
    }
  }

  if (board.en_passant_square != -1) {
    int ep = board.en_passant_square;
    Bitboard attackers = pawn_attacks[color_o][ep] & pawns;
    while (attackers)
      moves.add(encode_move(pop_lsb(attackers), ep, MoveFlag::ENPASSANT, 0));
  }

}
// END MAGIC


constexpr Bitboard RANK_2 = 0x000000000000FF00ULL, RANK_7 = 0x00FF000000000000ULL;
Board init_board() {
  Board board = Board();

  constexpr PieceType back_rank[8] = {
    PieceType::ROOK, PieceType::KNIGHT, PieceType::BISHOP, PieceType::QUEEN,
    PieceType::KING, PieceType::BISHOP, PieceType::KNIGHT, PieceType::ROOK
  };

  auto place = [&](PieceColor color, PieceType type, int s) {
    board.pieces[static_cast<int>(color)][static_cast<int>(type)] |= ONE << s;
  };

  for (int c = 0; c < 8; c++) {
    place(PieceColor::BLACK, back_rank[c], to_pos(7, c));
    place(PieceColor::BLACK, PieceType::PAWN, to_pos(6, c));
    place(PieceColor::WHITE, PieceType::PAWN, to_pos(1, c));
    place(PieceColor::WHITE, back_rank[c], to_pos(0, c));
  }

  board.occupancy = 0;
  for (int color = 0; color < 2; color++) {
    for (int type = 0; type < 6; type++)
      board.player_occupancy[color] |= board.pieces[color][type];
    board.occupancy |= board.player_occupancy[color];
  }

  board.to_move = PieceColor::WHITE;
  board.castling = 0b1111;
  board.en_passant_square = -1;
  board.hash = zobrist_hash(board);

  return board;
}

PieceType piece_on(const Board &board, int s) {
  Bitboard mask = 1ULL << s;
  if ((board.occupancy & mask) == 0) return PieceType::NONE;
  PieceColor color = (board.player_occupancy[PieceColor::WHITE] & mask) 
    ? PieceColor::WHITE : PieceColor::BLACK;
  for (int type = 0; type < 6; type++) 
    if (board.pieces[color][type] & mask)
      return static_cast<PieceType>(type);
  return PieceType::NONE;
}

void get_pseudo_legal_moves(const Board &board, MoveList &moves) {
  moves.n = 0;
  PieceColor color = board.to_move;
  Bitboard player_occupancy = board.player_occupancy[color];
  Bitboard occupancy = board.occupancy;

  gen_pawn_moves(board, moves);

  auto gen = [&](PieceType type, auto attack) {
    Bitboard b = board.pieces[color][type];
    while (b) {
      int from = pop_lsb(b);
      Bitboard targets = attack(from) & ~player_occupancy;
      while (targets) {
        int to = pop_lsb(targets); 
        moves.add(encode_move(from, to, MoveFlag::NORMAL, 0));
      }
    }
  };

  gen(PieceType::KNIGHT, [](int s){ return knight_attacks[s]; });
  gen(PieceType::KING, [](int s){ return king_attacks[s]; });
  gen(PieceType::BISHOP, [&](int s){ return bishop_attacks(s, occupancy); });
  gen(PieceType::ROOK, [&](int s){ return rook_attacks(s, occupancy); });
  gen(PieceType::QUEEN, [&](int s){ return queen_attacks(s, occupancy); });

  if (color == WHITE) {
    if ((board.castling & CR_WK) 
      && !(occupancy & ((ONE<<5) | (ONE<<6)))
    ) moves.add(encode_move(4, 6, CASTLE, 0));
    if ((board.castling & CR_WQ) 
      && !(occupancy & ((ONE<<1) | (ONE<<2) | (ONE<<3)))
    ) moves.add(encode_move(4, 2, CASTLE, 0));
  } else {
    if ((board.castling & CR_BK) 
      && !(occupancy & ((ONE<<61) | (ONE<<62)))
    ) moves.add(encode_move(60, 62, CASTLE, 0));
    if ((board.castling & CR_BQ) 
      && !(occupancy & ((ONE<<57) | (ONE<<58) | (ONE<<59)))
    ) moves.add(encode_move(60, 58, CASTLE, 0));
  }
}

inline void recompute_occupancy(Board &b) {
  b.occupancy = 0;
  for (int color = 0; color < 2; color++) {
    b.player_occupancy[color] = 0;
    for (int type = 0; type < 6; type++)
      b.player_occupancy[color] |= b.pieces[color][type];
    b.occupancy |= b.player_occupancy[color];
  }
}

static const std::array<uint8_t, 64> CASTLE_MASK = []{
  std::array<uint8_t, 64> mask; mask.fill(0xF);
  mask[0] = 0b1101; mask[4] = 0b1100; mask[7] = 0b1110;
  mask[56] = 0b0111; mask[60] = 0b0011; mask[63] = 0b1011;
  return mask;
}();

static void castle_rook_squares(int king_to, int &rook_from, int& rook_to) {
  switch (king_to) {
    case 6: rook_from = 7; rook_to = 5; break;
    case 2: rook_from = 0; rook_to = 3; break;
    case 62: rook_from = 63; rook_to = 61; break;
    default: rook_from = 56; rook_to = 59; break; 
  }
}

bool square_attacked(const Board &board, int s, PieceColor color_p) {
  PieceColor color_o = opposite_color(color_p);
  Bitboard occupancy = board.occupancy;
  if (pawn_attacks[color_o][s] & board.pieces[color_p][PieceType::PAWN]) 
    return true;
  if (knight_attacks[s] & board.pieces[color_p][PieceType::KNIGHT]) 
    return true;
  if (king_attacks[s] & board.pieces[color_p][PieceType::KING])
    return true;
  if (bishop_attacks(s, occupancy) 
      & (board.pieces[color_p][PieceType::BISHOP] 
      | board.pieces[color_p][PieceType::QUEEN]))
    return true;
  if (rook_attacks(s, occupancy) 
      & (board.pieces[color_p][PieceType::ROOK] 
      | board.pieces[color_p][PieceType::QUEEN]))
    return true;
  return false;
}

bool in_check(const Board& board, PieceColor color) {
  int king_s = lsb(board.pieces[color][PieceType::KING]);
  return square_attacked(board, king_s, opposite_color(color));
}

void make_move(Board &board, Move move, Undo &undo) {
  PieceColor color_p = board.to_move, color_o = opposite_color(color_p);
  int from = move.from(), to = move.to();
  MoveFlag flag = move.flag();
  PieceType type = piece_on(board, from);
  Bitboard from_board = ONE << from, to_board = ONE << to;

  undo.captured = piece_on(board, to);
  undo.castling = board.castling;
  undo.en_passant_square = board.en_passant_square;
  undo.hash = board.hash;

  uint64_t hash = board.hash;
  board.pieces[color_p][type] ^= from_board;
  hash ^= zobrist_piece[color_p][type][from];

  if (undo.captured != PieceType::NONE) {
    board.pieces[color_o][undo.captured] ^= to_board;
    hash ^= zobrist_piece[color_o][undo.captured][to];
  }
  
  if (flag == MoveFlag::PROMOTION) {
    board.pieces[color_p][move.promo()] ^= to_board;
    hash ^= zobrist_piece[color_p][move.promo()][to];
  } else {
    board.pieces[color_p][type] ^= to_board;
    hash ^= zobrist_piece[color_p][type][to];
  }

  if (flag == MoveFlag::ENPASSANT) {
    int cap = (color_p == PieceColor::WHITE) ? to - 8 : to + 8;
    board.pieces[color_o][PieceType::PAWN] ^= (ONE << cap);
    hash ^= zobrist_piece[color_o][PieceType::PAWN][cap];
  }

  if (flag == MoveFlag::CASTLE) {
    int rook_from, rook_to;
    castle_rook_squares(to, rook_from, rook_to);
    board.pieces[color_p][PieceType::ROOK] ^= 
      (ONE << rook_from) | (ONE << rook_to);
    hash ^= zobrist_piece[color_p][PieceType::ROOK][rook_from] 
      ^ zobrist_piece[color_p][PieceType::ROOK][rook_to];
  }
  
  if (board.en_passant_square != -1)
    hash ^= zobrist_ep[board.en_passant_square & 7];
  int diff = to - from;
  board.en_passant_square = (type == PieceType::PAWN 
    && (diff == 16 || diff == -16)
    ) ? (from + to) / 2 : -1;
  if (board.en_passant_square != -1)
    hash ^= zobrist_ep[board.en_passant_square & 7];
  
  uint8_t old_castle = board.castling;
  board.castling &= CASTLE_MASK[from] & CASTLE_MASK[to];
  hash ^= zobrist_castle[old_castle] ^ zobrist_castle[board.castling];

  hash ^= zobrist_side;
  board.to_move = color_o;
  board.hash = hash;
  recompute_occupancy(board);
}

void unmake_move(Board &board, Move move, const Undo &undo) {
  PieceColor color_p = opposite_color(board.to_move), color_o = board.to_move;
  int from = move.from(), to = move.to();
  MoveFlag flag = move.flag();
  Bitboard from_board = ONE << from, to_board = ONE << to;

  if (flag == MoveFlag::PROMOTION) {
    board.pieces[color_p][move.promo()] ^= to_board;
    board.pieces[color_p][PieceType::PAWN] ^= from_board;
  } else {
    PieceType type = piece_on(board, to);
    board.pieces[color_p][type] ^= to_board;
    board.pieces[color_p][type] ^= from_board;
  }
  if (undo.captured != PieceType::NONE)
    board.pieces[color_o][undo.captured] ^= to_board;
  if (flag == MoveFlag::ENPASSANT) {
    int cap = (color_p == PieceColor::WHITE) ? to - 8 : to + 8;
    board.pieces[color_o][PieceType::PAWN] ^= (ONE << cap);
  }
  if (flag == MoveFlag::CASTLE) {
    int rook_from, rook_to;
    castle_rook_squares(to, rook_from, rook_to);
    board.pieces[color_p][PieceType::ROOK] ^= 
      (ONE << rook_from) | (ONE << rook_to);
  }

  board.castling = undo.castling;
  board.en_passant_square = undo.en_passant_square;
  board.to_move = color_p;
  board.hash = undo.hash;
  recompute_occupancy(board);
}

void get_moves(Board &board, MoveList &legal) {
  legal.n = 0;
  MoveList pseudo; get_pseudo_legal_moves(board, pseudo);
  PieceColor color_p = board.to_move, color_o = opposite_color(color_p);

  for (int i = 0; i < pseudo.n; i++) {
    Move m = pseudo.moves[i];
    if (m.flag() == CASTLE) {
      int from = m.from(), to = m.to(), mid = (from + to) / 2;
      if (
        !square_attacked(board, from, color_o)
        && !square_attacked(board, mid, color_o)
        && !square_attacked(board, to, color_o)
      ) legal.add(m);
    } else {
      Undo u; make_move(board, m, u);
      if (!in_check(board, color_p)) legal.add(m);
      unmake_move(board, m, u);
    }
  }
}