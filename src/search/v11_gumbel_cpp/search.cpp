#pragma once

#include "engine/batching.cpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

// v11 search: the v10 Gumbel search (v10_gumbel_cpp) with several network evaluations in flight per lane, so one
// game's search fills a batch (single-game play was one leaf per call; a call costs about the same for 1 or 32 rows)
//   max_pending   leaves one lane may have waiting for the network at once (1 = the v10 search, same moves)
//   vl_mode       0: at most one leaf per root candidate in flight (the candidates' subtrees are disjoint, so the
//                 leaves of one halving sweep go out together; same moves as v10 up to float order)
//                 1 / 2: several leaves per candidate; each descent adds a virtual visit along its path so the next
//                 descent takes another way, valued -virtual_loss (1: a virtual loss) or the node's current mean -
//                 virtual_loss (2: Q nearly unchanged, mostly the exploration term shrinks; 0 = neutral)
//   sync_phases   with vl_mode 1 / 2: issue nothing past a halving step (where candidates are cut) while leaves are in
//                 flight, so the survivors are chosen on complete statistics (vl_mode 0 always waits)
// a lane issues simulation k (target visit count schedule[k]) against root_n + in flight, so with max_pending 1 the
// picks are exactly v10's

namespace v11_gumbel_cpp {


constexpr int MAXM = MAX_MOVES_CAPACITY;
constexpr int ONGOING = (int) GameResult::ONGOING;
constexpr int CHECKMATE = (int) GameResult::CHECKMATE;
constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

// target visit count per simulation; cut[k] = 1 where simulation k is the first after a halving step
struct Schedule {
  std::vector<int> seq;
  std::vector<uint8_t> cut;
};

Schedule halving_schedule(int m, int n_sims) {
  Schedule out;
  std::vector<int> &seq = out.seq;
  if (m <= 1) {
    for (int i = 0; i < n_sims; i++)
      seq.push_back(i);
    out.cut.assign(n_sims, 0);
    return out;
  }
  std::vector<int> visits(m, 0);
  int num_phases = (int) std::ceil(std::log2((double) m));
  int num_candidates = m;
  while ((int) seq.size() < n_sims) {
    if (!seq.empty() && (int) out.cut.size() < (int) seq.size() + 1) {
      out.cut.resize(seq.size(), 0);
      out.cut.push_back(1);
    }
    int repeats = std::max(1, (int) ((double) n_sims / (num_phases * num_candidates)));
    for (int r = 0; r < repeats; r++) {
      for (int i = 0; i < num_candidates; i++)
        seq.push_back(visits[i]);
      for (int i = 0; i < num_candidates; i++)
        visits[i]++;
    }
    int next = std::max(2, num_candidates / 2);
    if (next == num_candidates)
      out.cut.resize(seq.size() + 1, 0);    // 2 -> 2: more sweeps, no cut
    num_candidates = next;
  }
  seq.resize(n_sims);
  out.cut.resize(n_sims, 0);
  return out;
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

enum class Phase : uint8_t { IDLE, ROOT, RUNNING, DONE };

// one simulation waiting for the network
struct Pending {
  int root_action = -1, leaf = -1;
  std::vector<int> path;
  std::vector<float> virtual_value;   // what the virtual visit added to each node of the path (vl_mode 1 / 2)
};

struct Lane {
  Phase phase = Phase::IDLE;
  int n_sims = 0, m = 0, sim = 0, issued = 0;   // sim: finished simulations; issued: finished + in flight
  int n_root = 0, n_considered = 0, root_offset = 0;
  bool root_waiting = false;
  float root_v = 0;
  const int *schedule = nullptr;
  const uint8_t *cut = nullptr;

  std::vector<int> visits;
  std::vector<float> value_sum;
  std::vector<uint8_t> expanded, waiting;
  std::vector<int> child;
  std::vector<float> prior;

  std::vector<float> gumbel, logits, root_q, virtual_root;   // virtual_root: virtual value on each root child
  std::vector<int> root_n, in_flight;
  std::vector<uint8_t> candidate;

  std::vector<int> path;
  std::vector<Pending> pending;
  int n_pending = 0;

  int chosen = -1, nodes = 0;
  float root_value = 0;
  std::vector<float> policy;

  Lane(int states, int actions, int max_pending)
    : visits(states),
    value_sum(states),
    expanded(states),
    waiting(states),
    child(actions, -1),
    prior(actions),
    gumbel(MAXM),
    logits(MAXM),
    root_q(MAXM),
    virtual_root(MAXM),
    root_n(MAXM),
    in_flight(MAXM),
    candidate(MAXM),
    pending(max_pending),
    policy(MAXM) {}
};

struct Search {
  GameManager *manager;
  double c_puct = 1.5, fpu = 0.4, c_visit = 50.0, c_scale = 1.0;
  bool proven_override = false, forcing = false;
  int forcing_cap = 32;
  bool forcing_append = false;     // forcing moves on top of the top m (at most m / 2 extra), not in the m slots
  bool root_value_policy = false;  // root value = improved-policy-weighted completed Q, not the mean of all root visits
  int max_pending = 1, vl_mode = 0;
  bool sync_phases = false;
  float virtual_loss = 1;
  EvalCache cache;
  std::vector<Lane> lanes;
  // (n_considered, n_sims) -> halving schedule
  std::map<std::pair<int, int>, Schedule> schedules;
  std::vector<int> finished;
  int cursor = 0;
  int64_t cache_hits = 0, cache_misses = 0, collisions = 0;

  Search(GameManager *manager, int sim_cap, int cache_log2, int max_pending)
      : manager(manager), max_pending(std::max(max_pending, 1)), cache(cache_log2) {
    int states = sim_cap + 1, actions = (sim_cap + 1) * MAXM;
    lanes.reserve(manager->size());
    for (int i = 0; i < manager->size(); i++)
      lanes.emplace_back(states, actions, this->max_pending);
  }

  ChessInterface &game(int lane) {
    return manager->games[lane];
  }

  const Schedule &schedule_row(int n_considered, int n_sims) {
    auto key = std::make_pair(n_considered, n_sims);
    auto it = schedules.find(key);
    if (it == schedules.end())
      it = schedules.emplace(key, halving_schedule(std::max(n_considered, 1), n_sims)).first;
    return it->second;
  }

  void init_state(Lane &lane, int s) {
    lane.visits[s] = 0;
    lane.value_sum[s] = 0;
    lane.expanded[s] = 0;
    lane.waiting[s] = 0;
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
    lane.issued = 0;
    lane.n_pending = 0;
    lane.root_waiting = false;
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
      lane.virtual_root[a] = 0;
      lane.in_flight[a] = 0;
    }
    if (forcing)
      choose_candidates(i);
    else
      lane.n_considered = std::min(lane.n_root, lane.m);
    const Schedule &row = schedule_row(lane.n_considered, lane.n_sims);
    lane.schedule = row.seq.data();
    lane.cut = row.cut.data();
    lane.phase = Phase::RUNNING;
  }

  // forcing moves (checks first, then by prior; at most forcing_cap) are always candidates,
  // the rest of the m slots go to the best other moves by gumbel + logits.
  // forcing_append: the usual top m by gumbel + logits (forcing or not), plus up to min(forcing_cap, m / 2) forcing
  // moves outside it (checks first, then by prior); forcing moves past that cap are candidates only if in the top m
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
    if (forcing_append) {
      std::vector<int> all(n);
      for (int a = 0; a < n; a++)
        all[a] = a;
      std::stable_sort(all.begin(), all.end(), [&](int x, int y) {
        return lane.gumbel[x] + lane.logits[x] > lane.gumbel[y] + lane.logits[y];
      });
      std::fill(lane.candidate.begin(), lane.candidate.begin() + n, 0);
      int top = std::min(lane.m, n), extra = 0, cap = std::min(forcing_cap, std::max(lane.m / 2, 1));
      for (int k = 0; k < top; k++)
        lane.candidate[all[k]] = 1;
      for (int a : forced) {
        if (extra >= cap)
          break;
        if (!lane.candidate[a]) {
          lane.candidate[a] = 1;
          extra++;
        }
      }
      lane.n_considered = top + extra;
      return;
    }
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

  // completed Q per root action (root's side to move): visited -> its mean Q, unvisited -> v_mix. returns max visits
  int completed_q(const Lane &lane, double *out) {
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
    for (int a = 0; a < n; a++)
      out[a] = lane.root_n[a] > 0 ? (double) lane.root_q[a] : v_mix;
    return max_n;
  }

  // completed-Q sigma per root action (same numbers as GumbelMCTS._sigma)
  void sigma(const Lane &lane, double *out) {
    int max_n = completed_q(lane, out);
    double scale = (c_visit + max_n) * c_scale;
    for (int a = 0; a < lane.n_root; a++)
      out[a] = scale * (out[a] + 1) / 2;
  }

  // sequential halving: the best candidate whose visits + in flight is the next simulation's target. -1 if no
  // candidate can take it now (without virtual loss a candidate with a leaf in flight waits for it); with nothing in
  // flight that does not happen, and the fallback 0 is v10's
  int pick_root_action(const Lane &lane) {
    if (lane.issued >= lane.n_sims)
      return -1;
    if (sync_phases && lane.n_pending && lane.cut[lane.issued])
      return -1;
    double sigmas[MAXM];
    sigma(lane, sigmas);
    int visits = lane.schedule[lane.issued];
    double best_score = NEG_INF;
    int best_a = lane.n_pending ? -1 : 0;
    for (int a = 0; a < lane.n_root; a++) {
      if (lane.root_n[a] + lane.in_flight[a] != visits || (forcing && !lane.candidate[a]))
        continue;
      if (lane.in_flight[a] > 0 && !vl_mode)
        return -1;   // still on its way to `visits` (the previous sweep): v10 picks only once it has finished, and
                     // at a halving step picking without it could drop it
      double score = (double) (lane.gumbel[a] + lane.logits[a]) + sigmas[a];
      if (score > best_score) {
        best_score = score;
        best_a = a;
      }
    }
    return best_a;
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

  // walks from the root (its move forced by the halving schedule) to a leaf: terminal, not expanded, or new.
  // the walk goes into `path` (root first); returns the leaf
  int descend(int i, int root_action, std::vector<int> &path) {
    Lane &lane = lanes[i];
    ChessInterface &g = game(i);
    path.assign(1, 0);
    int action = lane.root_offset + root_action;
    while (true) {
      int node = lane.child[action];
      if (node < 0)
        node = create_child(i, action);   // a new node is not expanded, so the walk stops there
      path.push_back(node);
      if (g.state_result(node) != ONGOING || !lane.expanded[node])
        return node;
      action = select_puct(lane, g, node);
    }
  }

  // a virtual visit along the path (root excluded: its move is forced): value -virtual_loss (vl_mode 1) or the node's
  // current mean - virtual_loss (2), remembered in p.virtual_value so remove_virtual takes back exactly that
  void add_virtual(Lane &lane, Pending &p) {
    p.virtual_value.resize(p.path.size());
    for (size_t d = 1; d < p.path.size(); d++) {
      int s = p.path[d];
      float mean = lane.visits[s] > 0 ? lane.value_sum[s] / lane.visits[s] : 0.0f;
      float v = vl_mode == 1 ? -virtual_loss : mean - virtual_loss;
      p.virtual_value[d] = v;
      lane.visits[s]++;
      lane.value_sum[s] += v;
    }
    lane.virtual_root[p.root_action] += p.virtual_value[1];
  }

  void remove_virtual(Lane &lane, const Pending &p) {
    for (size_t d = 1; d < p.path.size(); d++) {
      lane.visits[p.path[d]]--;
      lane.value_sum[p.path[d]] -= p.virtual_value[d];
    }
    lane.virtual_root[p.root_action] -= p.virtual_value[1];
  }

  void backup(Lane &lane, const std::vector<int> &path, float value) {
    int depth = (int) path.size() - 1;
    for (int d = depth; d >= 1; d--) {
      int s = path[d];
      float v = ((depth - d + 1) % 2 == 0) ? value : -value;
      lane.visits[s]++;
      lane.value_sum[s] += v;
    }
    // the root stores value from its own side to move (root_value); every other node from its mover's
    lane.visits[0]++;
    lane.value_sum[0] += (depth % 2 == 0) ? value : -value;
  }

  void finish_sim(int i, int root_action, const std::vector<int> &path, float value) {
    Lane &lane = lanes[i];
    backup(lane, path, value);
    int child_idx = lane.child[lane.root_offset + root_action];
    // the root child still carries the virtual visits of this candidate's other leaves in flight
    int k = vl_mode ? lane.in_flight[root_action] : 0;
    if (k == 0)
      lane.virtual_root[root_action] = 0;    // no float residue once nothing is in flight
    lane.root_n[root_action] = lane.visits[child_idx] - k;
    lane.root_q[root_action] = (float) (
      (double) (lane.value_sum[child_idx] - (k ? lane.virtual_root[root_action] : 0.0f))
      / (double) lane.root_n[root_action]
    );
    lane.sim++;
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

    if (root_value_policy) {
      // the value under the improved policy: moves cut by sequential halving no longer pull it toward their Q
      double q[MAXM], v = 0;
      completed_q(lane, q);
      for (int a = 0; a < n; a++)
        v += (double) lane.policy[a] * q[a];
      lane.root_value = (float) v;
    }

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

  // a network evaluation for lane i's state s: the root, or the leaf of one pending simulation
  void apply(int i, int s, const float *prior, float value) {
    Lane &lane = lanes[i];
    if (lane.phase == Phase::ROOT) {
      lane.root_waiting = false;
      root_expanded(i, prior, value);
      return;
    }
    int k = 0;
    while (lane.pending[k].leaf != s)
      k++;
    Pending &p = lane.pending[k];
    lane.waiting[s] = 0;
    if (vl_mode)
      remove_virtual(lane, p);
    lane.in_flight[p.root_action]--;
    expand(i, s, prior);
    finish_sim(i, p.root_action, p.path, value);
    // keep pending[0, n_pending) packed: the last one moves into this slot
    std::swap(lane.pending[k], lane.pending[--lane.n_pending]);
  }

  const float *try_cache(int i, int s, float &value) {
    const float *prior = cache.find(game(i).state_hash(s), value);
    if (!prior) {
      cache_misses++;
      return nullptr;
    }
    cache_hits++;
    return prior;
  }

  // runs lanes until each waits for the network (up to max_pending leaves) or finishes; returns the (lane, state)
  // pairs to evaluate
  int run(int max_requests, int32_t *out_lanes, int32_t *out_states) {
    int n_req = 0, n_lanes = (int) lanes.size();
    int t = 0;
    for (; t < n_lanes && n_req < max_requests; t++) {
      int i = (cursor + t) % n_lanes;
      Lane &lane = lanes[i];
      const float *prior;
      float value;
      if (lane.phase == Phase::ROOT && !lane.root_waiting) {
        if ((prior = try_cache(i, 0, value)))
          root_expanded(i, prior, value);
        else {
          lane.root_waiting = true;
          out_lanes[n_req] = i;
          out_states[n_req++] = 0;
        }
      }
      while (lane.phase == Phase::RUNNING && lane.n_pending < max_pending && n_req < max_requests) {
        int a = pick_root_action(lane);
        if (a < 0)
          break;
        int leaf = descend(i, a, lane.path);
        if (lane.waiting[leaf]) {         // another descent's leaf (virtual loss only): wait for its evaluation
          collisions++;
          break;
        }
        lane.issued++;
        int result = game(i).state_result(leaf);
        if (result != ONGOING) {
          finish_sim(i, a, lane.path, result == CHECKMATE ? -1.0f : 0.0f);
          continue;
        }
        if ((prior = try_cache(i, leaf, value))) {
          expand(i, leaf, prior);
          finish_sim(i, a, lane.path, value);
          continue;
        }
        Pending &p = lane.pending[lane.n_pending++];
        p.root_action = a;
        p.leaf = leaf;
        p.path.swap(lane.path);
        lane.waiting[leaf] = 1;
        lane.in_flight[a]++;
        if (vl_mode)
          add_virtual(lane, p);
        out_lanes[n_req] = i;
        out_states[n_req++] = leaf;
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
      apply(i, states[k], prior, values[k]);
    }
  }
};

}  // namespace v11_gumbel_cpp


extern "C" {
  // unity build: v10's extern block already brought its own Search / MAXM into the global scope
  using V11Search = v11_gumbel_cpp::Search;
  constexpr int V11_MAXM = v11_gumbel_cpp::MAXM;

  void *v11_gumbel_cpp_new(void *manager, int sim_cap, int cache_log2, int max_pending) {
    return new V11Search(static_cast<GameManager *>(manager), sim_cap, cache_log2, max_pending);
  }
  void v11_gumbel_cpp_free(void *s) { delete static_cast<V11Search *>(s); }

  void v11_gumbel_cpp_set_params(void *s, double c_puct, double fpu, double c_visit, double c_scale,
                              int proven_override, int forcing, int forcing_cap, int vl_mode, double virtual_loss,
                              int sync_phases, int forcing_append, int root_value_policy) {
    auto *search = static_cast<V11Search *>(s);
    search->forcing_append = forcing_append != 0;
    search->root_value_policy = root_value_policy != 0;
    search->c_puct = c_puct;
    search->fpu = fpu;
    search->c_visit = c_visit;
    search->c_scale = c_scale;
    search->proven_override = proven_override != 0;
    search->forcing = forcing != 0;
    search->forcing_cap = forcing_cap;
    search->vl_mode = vl_mode;
    search->virtual_loss = (float) virtual_loss;
    search->sync_phases = sync_phases != 0;
  }

  // n lanes start a search: per lane n_sims, m, and a gumbel row of MAXM floats (already scaled)
  void v11_gumbel_cpp_begin(void *s, const int32_t *lanes, int n, const int32_t *n_sims, const int32_t *m,
                         const float *gumbel) {
    auto *search = static_cast<V11Search *>(s);
    for (int k = 0; k < n; k++)
      search->begin(lanes[k], n_sims[k], m[k], gumbel + (size_t) k * V11_MAXM);
  }

  int v11_gumbel_cpp_run(void *s, int max_requests, int32_t *out_lanes, int32_t *out_states) {
    return static_cast<V11Search *>(s)->run(max_requests, out_lanes, out_states);
  }

  // priors: n rows of MAXM floats (only the first num_actions of each row are read)
  void v11_gumbel_cpp_cache_evals(void *s, const int32_t *lanes, const int32_t *states, int n,
                               const float *priors, const float *values) {
    static_cast<V11Search *>(s)->cache_evals(n, lanes, states, priors, values);
  }

  // lanes that finished since the last call
  int v11_gumbel_cpp_take_finished(void *s, int32_t *out_lanes) {
    auto *search = static_cast<V11Search *>(s);
    int n = (int) search->finished.size();
    std::copy(search->finished.begin(), search->finished.end(), out_lanes);
    search->finished.clear();
    return n;
  }

  void v11_gumbel_cpp_results(void *s, const int32_t *lanes, int n, int32_t *chosen, float *root_value,
                           float *policy, int32_t *nodes) {
    auto *search = static_cast<V11Search *>(s);
    for (int k = 0; k < n; k++) {
      const auto &lane = search->lanes[lanes[k]];
      chosen[k] = lane.chosen;
      root_value[k] = lane.root_value;
      nodes[k] = lane.nodes;
      std::copy(lane.policy.begin(), lane.policy.end(), policy + (size_t) k * V11_MAXM);
    }
  }

  // per lane: flags of every legal root move (bit 0 capture, 1 promotion, 2 check); out: n rows of MAXM
  void v11_gumbel_cpp_root_flags(void *s, const int32_t *lanes, int n, uint8_t *out) {
    auto *search = static_cast<V11Search *>(s);
    for (int k = 0; k < n; k++)
      search->game(lanes[k]).action_flags(0, out + (size_t) k * V11_MAXM);
  }

  void v11_gumbel_cpp_clear_cache(void *s) { static_cast<V11Search *>(s)->cache.clear(); }

  // cache hits / misses / virtual-loss collisions since the last call
  void v11_gumbel_cpp_stats(void *s, int64_t *out) {
    auto *search = static_cast<V11Search *>(s);
    out[0] = search->cache_hits;
    out[1] = search->cache_misses;
    out[2] = search->collisions;
    search->cache_hits = search->cache_misses = search->collisions = 0;
  }
}
