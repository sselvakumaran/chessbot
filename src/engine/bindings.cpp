#include "game.cpp"

extern "C" {
  int game_max_moves() {
    return MAX_MOVES;
  }
  void* game_new() {
    static bool initialized = false;
    if (!initialized) {
      init_attack_tables();
      initialized = true;
    }

    return new ChessInterface();
  }

  void game_free(void* g) {
    delete static_cast<ChessInterface*>(g);
  }
  void game_reset(void* g) {
    static_cast<ChessInterface*>(g)->reset();
  }
  int game_result(void* g) {
    return int(static_cast<ChessInterface*>(g)->result());
  }
  int game_num_moves(void* g) {
    return static_cast<ChessInterface*>(g)->num_moves();
  }
  int game_to_move(void* g) {
    return static_cast<ChessInterface*>(g)->to_move();
  }
  int game_play(void* g, int idx) {
    return int(static_cast<ChessInterface*>(g)->play(idx));
  }
  void game_undo(void* g) {
    static_cast<ChessInterface*>(g)->undo();
  }
  int game_in_check(void* g) {
    return static_cast<ChessInterface*>(g)->in_check() ? 1 : 0;
  }
  void game_encode_position(void* g,
    uint8_t* board_out,
    uint8_t* castling_out,
    int8_t* en_passant_square_out,
    uint8_t* repetition_count_out,
    uint8_t* halfmove_clock_out
  ) {
    static_cast<ChessInterface*>(g)->encode_position(
      board_out, castling_out, en_passant_square_out,
      repetition_count_out, halfmove_clock_out
    );
  }
  int game_build_encoding(void* g,
    uint8_t* board_out,
    uint8_t* castling_out,
    int8_t* en_passant_square_out,
    uint8_t* repetition_count_out,
    uint8_t* halfmove_clock_out,
    uint8_t* moves_out,
    int32_t* num_moves_out
  ) {
    return static_cast<ChessInterface*>(g)->build_encoding(
      board_out, castling_out, en_passant_square_out,
      repetition_count_out, halfmove_clock_out, moves_out, num_moves_out
    );
  }
}
