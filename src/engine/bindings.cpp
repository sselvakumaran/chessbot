#pragma once

#include "batching.cpp"

// Boundary is index + tensor: C++ owns the tree (States/Actions arenas), Python
// owns the search stats (numpy) keyed by the indices handed across. Root is
// always state 0. The batched gamebatch_* API is fully LOCAL — a node is (state,
// action-local-to-state) per lane, so stats key as [lane, state, action] and
// arena offsets stay inside C++. Only the per-game game_* API below still
// speaks GLOBAL action indices (actions_offset(s) + local_a).

static void ensure_attack_tables() {
  static bool initialized = false;
  if (!initialized) { init_attack_tables(); initialized = true; }
}

extern "C" {
  int game_max_moves_capacity() { return MAX_MOVES_CAPACITY; }

  void* game_new(int sim_cap) {
    ensure_attack_tables();
    return new ChessInterface(sim_cap);
  }
  void game_free(void* g) { delete static_cast<ChessInterface*>(g); }

  // ---- real game ----
  void game_reset(void* g) { static_cast<ChessInterface*>(g)->reset(); }
  int  game_play(void* g, int root_action) {
    return int(static_cast<ChessInterface*>(g)->play(root_action));
  }
  int game_to_move(void* g) {
    return static_cast<ChessInterface*>(g)->to_move();
  }

  // ---- search tree ----
  void game_reset_tree(void* g) {
    static_cast<ChessInterface*>(g)->reset_tree();
  }
  int  game_get_child(void* g, int action_idx) {
    return static_cast<ChessInterface*>(g)->get_child_idx(action_idx);
  }
  int  game_num_actions(void* g, int s) {
    return static_cast<ChessInterface*>(g)->num_actions(s);
  }
  int  game_actions_offset(void* g, int s) {
    return static_cast<ChessInterface*>(g)->actions_offset(s);
  }
  int  game_state_result(void* g, int s) {
    return static_cast<ChessInterface*>(g)->state_result(s);
  }

  // fills the NN-input tensors for state s; returns its number of legal moves
  int  game_encode(void* g, int s,
    uint8_t* board_out,
    uint8_t* castling_out,
    int8_t*  en_passant_square_out,
    uint8_t* repetition_count_out,
    uint8_t* halfmove_clock_out,
    uint8_t* moves_out,
    int32_t* num_moves_out
  ) {
    return static_cast<ChessInterface*>(g)->encode(
      s, board_out, castling_out, en_passant_square_out,
      repetition_count_out, halfmove_clock_out, moves_out, num_moves_out
    );
  }

  // ---- batched (GameManager): one call per search step for all lanes ----
  void* gamebatch_new(int n, int sim_cap) {
    ensure_attack_tables();
    return new GameManager(n, sim_cap);
  }
  void gamebatch_free(void* m) { delete static_cast<GameManager*>(m); }
  int  gamebatch_size(void* m) { return static_cast<GameManager*>(m)->size(); }

  void gamebatch_reset(void* m, const uint8_t* mask) {
    static_cast<GameManager*>(m)->reset(mask);
  }
  void gamebatch_copy_lane(void* dst, void* src, int dst_lane, int src_lane) {
    static_cast<GameManager*>(dst)->copy_lane_from(
      *static_cast<GameManager*>(src), src_lane, dst_lane
    );
  }
  void gamebatch_reset_trees(void* m, const uint8_t* mask) {
    static_cast<GameManager*>(m)->reset_trees(mask);
  }
  void gamebatch_get_roots(void* m,
    int32_t* num_actions, int8_t* result, int8_t* to_move
  ) {
    static_cast<GameManager*>(m)->get_roots(num_actions, result, to_move);
  }
  void gamebatch_play(void* m,
    const int32_t* root_action, const uint8_t* mask, int8_t* result
  ) {
    static_cast<GameManager*>(m)->play(root_action, mask, result);
  }
  void gamebatch_get_children(void* m,
    const int32_t* parent_state, const int32_t* action,
    int32_t* state, int32_t* num_actions, int8_t* result
  ) {
    static_cast<GameManager*>(m)->get_children(
      parent_state, action, state, num_actions, result
    );
  }
  void gamebatch_search_keys(void* m,
    const int32_t* lane, const int32_t* state, int n, uint64_t* out
  ) {
    auto* manager = static_cast<GameManager*>(m);
    for (int i = 0; i < n; i++)
      manager->games[lane[i]].search_key(state[i], out + 3 * i);
  }
  void gamebatch_in_check(void* m,
    const int32_t* lane, const int32_t* state, int n, uint8_t* out
  ) {
    auto* manager = static_cast<GameManager*>(m);
    for (int i = 0; i < n; i++)
      out[i] = manager->games[lane[i]].state_in_check(state[i]);
  }
  void gamebatch_hash(void* m,
    const int32_t* lane, const int32_t* state, int n, uint64_t* out
  ) {
    static_cast<GameManager*>(m)->hashes(lane, state, n, out);
  }
  void gamebatch_encode(void* m,
    const int32_t* lane, const int32_t* state, int n,
    uint8_t* board,
    uint8_t* castling,
    int8_t*  en_passant_square,
    uint8_t* repetition_count,
    uint8_t* halfmove_clock,
    uint8_t* moves,
    int32_t* num_moves
  ) {
    static_cast<GameManager*>(m)->encode(
      lane, state, n, board, castling, en_passant_square,
      repetition_count, halfmove_clock, moves, num_moves
    );
  }
}
