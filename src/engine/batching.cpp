#pragma once

#include <cstring>
#include <vector>
#include "game.cpp"
// code to batch simulations / games.
// the API is fully LOCAL: a node is addressed as (state idx, action idx local
// to that state) per lane, so Python can key search stats [lane, state, action];
// arena action offsets never cross this boundary.

struct GameManager {
  std::vector<ChessInterface> games;
  GameManager(int n, int sim_cap) : games(n, ChessInterface(sim_cap)) {}

  int size() const { return (int) games.size(); }

  // mask == nullptr resets every lane
  void reset(const uint8_t* mask) {
    for (int i = 0; i < size(); i++)
      if (!mask || mask[i])
        games[i].reset();
  }

  void copy_lane_from(const GameManager& src, int src_lane, int dst_lane) {
    assert(src_lane >= 0 && src_lane < src.size());
    assert(dst_lane >= 0 && dst_lane < size());
    games[dst_lane].copy_game_from(src.games[src_lane]);
  }

  // frees every lane's search arena (root re-materializes as state 0).
  // searches MUST call this first: the arena only otherwise resets on
  // play/reset, so back-to-back searches would accumulate until overflow
  void reset_trees(const uint8_t* mask) {
    for (int i = 0; i < size(); i++)
      if (!mask || mask[i])
        games[i].reset_tree();
  }

  // zobrist hash of (lane[i], state[i]) — pieces, castling, ep file, side
  void hashes(
    const int32_t* lane, const int32_t* state, int n, uint64_t* out
  ) {
    for (int i = 0; i < n; i++)
      out[i] = games[lane[i]].state_hash(state[i]);
  }

  void get_roots(
    int32_t* num_actions_out,
    int8_t* result_out,
    int8_t* to_move_out
  ) {
    for (int i = 0; i < size(); i++) {
      num_actions_out[i] = games[i].num_actions(0);
      result_out[i] = (int8_t) games[i].state_result(0);
      to_move_out[i] = (int8_t) games[i].to_move();
    }
  }

  void play(
    const int32_t* root_action, 
    const uint8_t* mask, 
    int8_t* result_out
  ) {
    for (int i = 0; i < size(); i++) {
      if (mask && !mask[i])
        result_out[i] = (int8_t) games[i].state_result(0);
      else
        result_out[i] = (int8_t) games[i].play(root_action[i]);
    }
  }

  // one descent step for every lane: child of (parent_state[i], action[i]),
  // action local to the parent. action[i] < 0 skips lane i
  // (state_out[i] = -1, rest untouched)
  void get_children(
    const int32_t* parent_state,
    const int32_t* action_idx,
    int32_t* state_out,
    int32_t* num_actions_out,
    int8_t* result_out
  ) {
    for (int i = 0; i < size(); i++) {
      if (action_idx[i] < 0) {
        state_out[i] = -1;
        continue;
      }
      assert(parent_state[i] >= 0
        && action_idx[i] < games[i].num_actions(parent_state[i]));
      int s = games[i].get_child_idx(
        games[i].actions_offset(parent_state[i]) + action_idx[i]
      );
      state_out[i] = s;
      num_actions_out[i] = games[i].num_actions(s);
      result_out[i] = (int8_t) games[i].state_result(s);
    }
  }

  // encodes n (lane, state) pairs into row i of each NN-input buffer;
  // moves rows are zeroed past num_moves
  void encode(
    const int32_t* game_idxs, 
    const int32_t* state, 
    int num_games,
    uint8_t* board_out,
    uint8_t* castling_out,
    int8_t* en_passant_square_out,
    uint8_t* repetition_count_out,
    uint8_t* halfmove_clock_out,
    uint8_t* moves_out,
    int32_t* num_moves_out
  ) {
    for (int i = 0; i < num_games; i++) {
      uint8_t* moves_row = moves_out + i * MAX_MOVES_CAPACITY * 4;
      memset(moves_row, 0, MAX_MOVES_CAPACITY * 4);
      games[game_idxs[i]].encode(
        state[i],
        board_out + i * 64,
        castling_out + i,
        en_passant_square_out + i,
        repetition_count_out + i,
        halfmove_clock_out + i,
        moves_row,
        num_moves_out + i
      );
    }
  }
};
