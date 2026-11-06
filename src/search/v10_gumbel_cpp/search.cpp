#pragma once

#include "engine/batching.cpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

// v10 search, v9 gumbel search in cpp
// each lane has its own state machine to perform async evals

// own namespace: every native module is compiled into one translation unit (unity build)
namespace v10_gumbel_cpp {


constexpr int MAXM = MAX_MOVES_CAPACITY;
constexpr int ONGOING = (int) GameResult::ONGOING;
constexpr int CHECKMATE = (int) GameResult::CHECKMATE;
constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

std::vector<int> halving_schedule(int m, int n_sims) {
  std::vector<int> seq;
  if (m <= 1) {
    for (int i = 0; i < n_sims; i++)
      seq.push_back(i);
    return seq;
  }
  std::vector<int> visits(m, 0);
  int num_phases = (int) std::ceil(std::log2((double) m));
  int num_candidates = m;
  while ((int) seq.size() < n_sims) {
    int repeats = std::max(1, (int) ((double) n_sims / (num_phases * num_candidates)));
    for (int r = 0; r < repeats; r++) {
      for (int i = 0; i < num_candidates; i++)
        seq.push_back(visits[i]);
      for (int i = 0; i < num_candidates; i++)
        visits[i]++;
    }
    num_candidates = std::max(2, num_candidates / 2);
  }
  seq.resize(n_sims);
  return seq;
}

struct EvalCache {
  uint64_t mask = 0;
  std::vector<uint64_t> keys;
  std::vector<uint32_t> epochs;
  std::vector<float> values;
  std::vector<float> priors;
  uint32_t epoch = 1;

  explicit EvalCache(int log2_slots) {
    if (log2_slots <= 0)
      return;
    size_t slots = size_t(1) << log2_slots;
    mask = slots - 1;
    keys.resize(slots);
    epochs.assign(slots, 0);
    values.resize(slots);
    priors.resize(slots * MAXM);
  }

  bool enabled() const {
    return !keys.empty();
  }

  const float *find(uint64_t hash, float& value) const {
    if (!enabled())
      return nullptr;
    size_t slot = hash & mask;
    if (epochs[slot] != epoch || keys[slot] != hash)
      return nullptr;
    value = values[slot];
    return &priors[slot * MAXM];
  }

  void put(uint64_t hash, const float *prior, int n, float value) {
    if (!enabled())
      return;
    size_t slot = hash & mask;
    keys[slot] = hash;
    epochs[slot] = epoch;
    values[slot] = value;
    std::copy(prior, prior + n, &priors[slot * MAXM]);
  }

  void clear() {
    if (++epoch == 0) {
      std::fill(epochs.begin(), epochs.end(), 0);
      epoch = 1;
    }
  }
};

enum class Phase : uint8_t { IDLE, ROOT, RUNNING, WAITING, DONE };

struct Lane {
  Phase phase = Phase::IDLE;
  int n_sims = 0, m = 0, sim = 0;
  int n_root = 0, n_considered = 0, root_offset = 0;
  int root_action = -1, leaf = -1;       // leaf < 0: the root's evaluation is the one pending
  float root_v = 0;
  const int *schedule = nullptr;

  std::vector<int> visits;
  std::vector<float> value_sum;
  std::vector<uint8_t> expanded;
  std::vector<int> child;
  std::vector<float> prior;

  std::vector<float> gumbel, logits, root_q;
  std::vector<int> root_n;
  std::vector<uint8_t> candidate;

  std::vector<int> path;

  int chosen = -1, nodes = 0;
  float root_value = 0;
  std::vector<float> policy;

  Lane(int states, int actions)
    : visits(states),
    value_sum(states),
    expanded(states),
    child(actions, -1),
    prior(actions),
    gumbel(MAXM),
    logits(MAXM),
    root_q(MAXM),
    root_n(MAXM),
    candidate(MAXM),
    policy(MAXM) {}
};

struct Search {
  GameManager *manager;
  double c_puct = 1.5, fpu = 0.4, c_visit = 50.0, c_scale = 1.0;
  bool proven_override = false, forcing = false;
  int forcing_cap = 32;
  EvalCache cache;
  std::vector<Lane> lanes;
  // (n_considered, n_sims) -> halving schedule
  std::map<std::pair<int, int>, std::vector<int>> schedules;
  std::vector<int> finished;
  int cursor = 0;
  int64_t cache_hits = 0, cache_misses = 0;

  Search(GameManager *manager, int sim_cap, int cache_log2)
      : manager(manager), cache(cache_log2) {
    int states = sim_cap + 1, actions = (sim_cap + 1) * MAXM;
    lanes.reserve(manager->size());
    for (int i = 0; i < manager->size(); i++)
      lanes.emplace_back(states, actions);
  }

  ChessInterface &game(int lane) {
    return manager->games[lane];
  }

  const int *schedule_row(int n_considered, int n_sims) {
    auto key = std::make_pair(n_considered, n_sims);
    auto it = schedules.find(key);
    if (it == schedules.end())
      it = schedules.emplace(key, halving_schedule(std::max(n_considered, 1), n_sims)).first;
    return it->second.data();
  }

  void init_state(Lane &lane, int s) {
    lane.visits[s] = 0;
    lane.value_sum[s] = 0;
    lane.expanded[s] = 0;
  }

  void expand(int i, int s, const float *prior) {
    Lane &lane = lanes[i];
    int offset = game(i).actions_offset(s), n = game(i).num_actions(s);
    lane.expanded[s] = 1;
    for (int a = 0; a < n; a++) {
      lane.prior[offset + a] = prior[a];
      lane.child[offset + a] = -1;
    }
  }

  int create_child(int i, int action) {
    Lane &lane = lanes[i];
    int child_idx = game(i).get_child_idx(action);
    lane.child[action] = child_idx;
    init_state(lane, child_idx);
    lane.nodes++;
    return child_idx;
  }

  void begin(int i, int n_sims, int m, const float *gumbel) {
    Lane &lane = lanes[i];
    game(i).reset_tree();
    init_state(lane, 0);
    lane.n_sims = n_sims;
    lane.m = m;
    lane.sim = 0;
    lane.leaf = -1;
    lane.nodes = 1;
    lane.chosen = -1;
    lane.root_value = 0;
    std::fill(lane.policy.begin(), lane.policy.end(), 0.0f);
    std::copy(gumbel, gumbel + MAXM, lane.gumbel.begin());
    if (game(i).state_result(0) != ONGOING) {
      lane.phase = Phase::DONE;
      finished.push_back(i);
      return;
    }
    lane.phase = Phase::ROOT;
  }

  void root_expanded(int i, const float *prior, float value) {
    Lane &lane = lanes[i];
    expand(i, 0, prior);

    lane.visits[0] = 1;
    lane.value_sum[0] = value;
    lane.root_v = value;
    lane.n_root = game(i).num_actions(0);
    lane.root_offset = game(i).actions_offset(0);

    for (int a = 0; a < lane.n_root; a++) {
      lane.logits[a] = std::log(std::max(prior[a], 1e-9f));
      lane.root_n[a] = 0;
      lane.root_q[a] = 0;
    }
    if (forcing)
      choose_candidates(i);
    else
      lane.n_considered = std::min(lane.n_root, lane.m);
    lane.schedule = schedule_row(lane.n_considered, lane.n_sims);
    lane.phase = Phase::RUNNING;
  }

  // forcing moves (checks first, then by prior; at most forcing_cap) are always candidates,
  // the rest of the m slots go to the best other moves by gumbel + logits
  void choose_candidates(int i) {
    Lane &lane = lanes[i];
    int n = lane.n_root;
    uint8_t flags[MAXM];
    game(i).action_flags(0, flags);
    std::vector<int> forced, nonforced;
    for (int a = 0; a < n; a++)
      (flags[a] ? forced : nonforced).push_back(a);
    auto rank = [&](int a) {
      return 100.0 * ((flags[a] & 4) != 0) + lane.logits[a];
    };
    std::stable_sort(
      forced.begin(), forced.end(),
      [&](int x, int y) { return rank(x) > rank(y); }
    );
    if ((int) forced.size() > forcing_cap)
      forced.resize(forcing_cap);
    int fill = std::max(std::min(lane.m, n) - (int) forced.size(), 0);
    std::stable_sort(
      nonforced.begin(),
      nonforced.end(),
      [&](int x, int y) {
        return lane.gumbel[x] + lane.logits[x] > lane.gumbel[y] + lane.logits[y];
    });
    nonforced.resize(std::min(fill, (int) nonforced.size()));
    std::fill(lane.candidate.begin(), lane.candidate.begin() + n, 0);
    for (int a : forced)
      lane.candidate[a] = 1;
    for (int a : nonforced)
      lane.candidate[a] = 1;
    lane.n_considered = (int) (forced.size() + nonforced.size());
  }

  // completed-Q sigma per root action (same numbers as GumbelMCTS._sigma)
  void sigma(const Lane &lane, double *out) {
    int n = lane.n_root, max_n = 0;
    int64_t sum_n = 0;
    float p_visited = 0, pq_visited = 0;
    for (int a = 0; a < n; a++) {
      max_n = std::max(max_n, lane.root_n[a]);
      sum_n += lane.root_n[a];
      if (lane.root_n[a] > 0) {
        float p = lane.prior[lane.root_offset + a];
        p_visited += p;
        pq_visited += p * lane.root_q[a];
      }
    }
    double v_mix = (
      (double) lane.root_v
      + (p_visited > 0 ? (double) sum_n / (double) std::max(p_visited, 1e-9f) * (double) pq_visited : 0.0)
    ) / (double) (1 + sum_n);
    double scale = (c_visit + max_n) * c_scale;
    for (int a = 0; a < n; a++) {
      double completed = lane.root_n[a] > 0 ? (double) lane.root_q[a] : v_mix;
      out[a] = scale * (completed + 1) / 2;
    }
  }

  // sequential halving: the best candidate whose visit count is this simulation's target
  int pick_root_action(const Lane &lane) {
    double sigmas[MAXM];
    sigma(lane, sigmas);
    return best_root_action(lane, sigmas, lane.schedule[lane.sim]);
  }

  // among root candidates with exactly `visits` visits, the one with the highest gumbel + logit + sigma
  int best_root_action(const Lane &lane, const double *sigmas, int visits) {
    double best_score = NEG_INF;
    int best_a = 0;
    for (int a = 0; a < lane.n_root; a++) {
      if (lane.root_n[a] != visits || (forcing && !lane.candidate[a]))
        continue;
      double score = (double) (lane.gumbel[a] + lane.logits[a]) + sigmas[a];
      if (score > best_score) {
        best_score = score;
        best_a = a;
      }
    }
    return best_a;
  }

  // PUCT + FPU pick among an expanded node's actions (same numbers as GumbelMCTS._select_below_root)
  int select_puct(const Lane &lane, ChessInterface &g, int node) {
    int n = g.num_actions(node), offset = g.actions_offset(node);
    int parent_visits = std::max(1, lane.visits[node]);
    float c_puct32 = (float) c_puct, fpu32 = (float) fpu;

    float visited_prior = 0;
    for (int a = 0; a < n; a++) {
      int child_idx = lane.child[offset + a];
      if (child_idx >= 0 && lane.visits[child_idx] > 0)
        visited_prior += lane.prior[offset + a];
    }

    double node_q = (double) (-lane.value_sum[node]) / (double) parent_visits;
    float fpu_q = (float) (node_q - (double) (fpu32 * std::sqrt(visited_prior)));
    double sqrt_parent = std::sqrt((double) parent_visits);

    double best_score = NEG_INF;
    int best_a = 0;
    for (int a = 0; a < n; a++) {
      int child_idx = lane.child[offset + a];
      int child_visits = child_idx >= 0 ? lane.visits[child_idx] : 0;
      float q = child_visits > 0 ? (float) (
        (double) lane.value_sum[child_idx] / (double) child_visits
      ) : fpu_q;
      double u = (double) (
        c_puct32 * lane.prior[offset + a]
      ) * sqrt_parent / (double) (1 + child_visits);
      double score = (double) q + u;
      if (score > best_score) {
        best_score = score;
        best_a = a;
      }
    }
    return offset + best_a;
  }

  // walks from the root (its move forced by the halving schedule) to a leaf: terminal, not expanded, or new
  void descend(int i, int root_action) {
    Lane &lane = lanes[i];
    ChessInterface &g = game(i);
    lane.path.assign(1, 0);
    int action = lane.root_offset + root_action;
    while (true) {
      int node = lane.child[action];
      if (node < 0)
        node = create_child(i, action);   // a new node is not expanded, so the walk stops there
      lane.path.push_back(node);
      if (g.state_result(node) != ONGOING || !lane.expanded[node]) {
        lane.leaf = node;
        return;
      }
      action = select_puct(lane, g, node);
    }
  }

  void backup(Lane &lane, float value) {
    int depth = (int) lane.path.size() - 1;
    for (int d = depth; d >= 1; d--) {
      int s = lane.path[d];
      float v = ((depth - d + 1) % 2 == 0) ? value : -value;
      lane.visits[s]++;
      lane.value_sum[s] += v;
    }
    // the root stores value from its own side to move (root_value); every other node from its mover's
    lane.visits[0]++;
    lane.value_sum[0] += (depth % 2 == 0) ? value : -value;
  }

  void finish_sim(int i, float value) {
    Lane &lane = lanes[i];
    backup(lane, value);
    int child_idx = lane.child[lane.root_offset + lane.root_action];
    lane.root_n[lane.root_action] = lane.visits[child_idx];
    lane.root_q[lane.root_action] = (float) (
      (double) lane.value_sum[child_idx] / (double) lane.visits[child_idx]
    );
    lane.sim++;
    lane.phase = Phase::RUNNING;
    if (lane.sim == lane.n_sims)
      finalize(i);
  }

  void finalize(int i) {
    Lane &lane = lanes[i];
    ChessInterface &g = game(i);
    int n = lane.n_root;

    lane.root_value = (float) (
      (double) lane.value_sum[0] / (double) lane.visits[0]
    );

    // improved policy: softmax(logits + sigma)
    double sigmas[MAXM], z[MAXM];
    sigma(lane, sigmas);
    double z_max = NEG_INF;
    for (int a = 0; a < n; a++) {
      z[a] = (double) lane.logits[a] + sigmas[a];
      z_max = std::max(z_max, z[a]);
    }
    double total = 0;
    for (int a = 0; a < n; a++) {
      z[a] = std::exp(z[a] - z_max);
      total += z[a];
    }
    for (int a = 0; a < n; a++)
      lane.policy[a] = (float) (z[a] / total);

    // move: the best of the most-visited candidates
    int max_n = *std::max_element(lane.root_n.begin(), lane.root_n.begin() + n);
    lane.chosen = best_root_action(lane, sigmas, max_n);

    if (proven_override) {
      // a visited child is checkmate: play it, target uniform over all visited mates, value 1
      uint8_t is_mate[MAXM];
      int mates = 0;
      for (int a = 0; a < n; a++) {
        int child_idx = lane.child[lane.root_offset + a];
        is_mate[a] = child_idx >= 0 && g.state_result(child_idx) == CHECKMATE;
        mates += is_mate[a];
      }
      if (mates) {
        for (int a = 0; a < n; a++)
          lane.policy[a] = is_mate[a] ? 1.0f / mates : 0.0f;
        lane.chosen = (int) (std::find(is_mate, is_mate + n, 1) - is_mate);
        lane.root_value = 1.0f;
      }
    }
    lane.phase = Phase::DONE;
    finished.push_back(i);
  }

  // an evaluation (cache hit or network) for lane i's pending state
  void apply(int i, const float *prior, float value) {
    Lane &lane = lanes[i];
    if (lane.leaf < 0) {
      root_expanded(i, prior, value);
      return;
    }
    expand(i, lane.leaf, prior);
    finish_sim(i, value);
  }

  bool try_cache(int i, int s) {
    float value;
    const float *prior = cache.find(game(i).state_hash(s), value);
    if (!prior) {
      cache_misses++;
      return false;
    }
    cache_hits++;
    apply(i, prior, value);
    return true;
  }

  // runs lanes until each waits for the network or finishes; returns the (lane, state) pairs to evaluate
  int run(int max_requests, int32_t *out_lanes, int32_t *out_states) {
    int n_req = 0, n_lanes = (int) lanes.size();
    int t = 0;
    for (; t < n_lanes && n_req < max_requests; t++) {
      int i = (cursor + t) % n_lanes;
      Lane &lane = lanes[i];
      while (true) {
        int s = 0;                       // the state to evaluate: the root, or a new leaf
        if (lane.phase == Phase::RUNNING) {
          lane.root_action = pick_root_action(lane);
          descend(i, lane.root_action);
          int result = game(i).state_result(lane.leaf);
          if (result != ONGOING) {
            finish_sim(i, result == CHECKMATE ? -1.0f : 0.0f);
            continue;
          }
          s = lane.leaf;
        } else if (lane.phase != Phase::ROOT)
          break;

        if (try_cache(i, s))
          continue;
        lane.phase = Phase::WAITING;
        out_lanes[n_req] = i;
        out_states[n_req++] = s;
        break;
      }
    }
    cursor = (cursor + t) % n_lanes;
    return n_req;
  }

  void cache_evals(int n,
    const int32_t *lanes_in, const int32_t *states,
    const float *priors, const float *values
  ) {
    for (int k = 0; k < n; k++) {
      int i = lanes_in[k];
      const float *prior = priors + (size_t) k * MAXM;
      cache.put(
        game(i).state_hash(states[k]),
        prior, game(i).num_actions(states[k]), values[k]
      );
      apply(i, prior, values[k]);
    }
  }
};

}  // namespace v10_gumbel_cpp


extern "C" {
  using v10_gumbel_cpp::Search;
  using v10_gumbel_cpp::MAXM;

  void *v10_gumbel_cpp_new(void *manager, int sim_cap, int cache_log2) {
    return new Search(static_cast<GameManager *>(manager), sim_cap, cache_log2);
  }
  void v10_gumbel_cpp_free(void *s) { delete static_cast<Search *>(s); }

  void v10_gumbel_cpp_set_params(void *s, double c_puct, double fpu, double c_visit, double c_scale,
                                 int proven_override, int forcing, int forcing_cap) {
    auto *search = static_cast<Search *>(s);
    search->c_puct = c_puct;
    search->fpu = fpu;
    search->c_visit = c_visit;
    search->c_scale = c_scale;
    search->proven_override = proven_override != 0;
    search->forcing = forcing != 0;
    search->forcing_cap = forcing_cap;
  }

  // n lanes start a search: per lane n_sims, m, and a gumbel row of MAXM floats (already scaled)
  void v10_gumbel_cpp_begin(void *s, const int32_t *lanes, int n, const int32_t *n_sims, const int32_t *m,
                            const float *gumbel) {
    auto *search = static_cast<Search *>(s);
    for (int k = 0; k < n; k++)
      search->begin(lanes[k], n_sims[k], m[k], gumbel + (size_t) k * MAXM);
  }

  int v10_gumbel_cpp_run(void *s, int max_requests, int32_t *out_lanes, int32_t *out_states) {
    return static_cast<Search *>(s)->run(max_requests, out_lanes, out_states);
  }

  // priors: n rows of MAXM floats (only the first num_actions of each row are read)
  void v10_gumbel_cpp_cache_evals(void *s, const int32_t *lanes, const int32_t *states, int n,
                                  const float *priors, const float *values) {
    static_cast<Search *>(s)->cache_evals(n, lanes, states, priors, values);
  }

  // lanes that finished since the last call
  int v10_gumbel_cpp_take_finished(void *s, int32_t *out_lanes) {
    auto *search = static_cast<Search *>(s);
    int n = (int) search->finished.size();
    std::copy(search->finished.begin(), search->finished.end(), out_lanes);
    search->finished.clear();
    return n;
  }

  void v10_gumbel_cpp_results(void *s, const int32_t *lanes, int n, int32_t *chosen, float *root_value,
                              float *policy, int32_t *nodes) {
    auto *search = static_cast<Search *>(s);
    for (int k = 0; k < n; k++) {
      const auto &lane = search->lanes[lanes[k]];
      chosen[k] = lane.chosen;
      root_value[k] = lane.root_value;
      nodes[k] = lane.nodes;
      std::copy(lane.policy.begin(), lane.policy.end(), policy + (size_t) k * MAXM);
    }
  }

  // per lane: flags of every legal root move (bit 0 capture, 1 promotion, 2 check); out: n rows of MAXM
  void v10_gumbel_cpp_root_flags(void *s, const int32_t *lanes, int n, uint8_t *out) {
    auto *search = static_cast<Search *>(s);
    for (int k = 0; k < n; k++)
      search->game(lanes[k]).action_flags(0, out + (size_t) k * MAXM);
  }

  void v10_gumbel_cpp_clear_cache(void *s) { static_cast<Search *>(s)->cache.clear(); }

  // cache hits / misses since the last call
  void v10_gumbel_cpp_stats(void *s, int64_t *out) {
    auto *search = static_cast<Search *>(s);
    out[0] = search->cache_hits;
    out[1] = search->cache_misses;
    search->cache_hits = search->cache_misses = 0;
  }
}
