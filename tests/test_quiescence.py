import unittest
import numpy as np
from engine.game import GameBatch, GameResult
from search.v4_alpha_beta_quiescence import AlphaBetaQuiescence

PV = np.array([0,1,3,3,5,9,20,1,3,3,5,9,20], np.float32)

def position(moves, lanes=1):
  b = GameBatch(lanes, sim_cap=128)
  for lane in range(lanes):
    for move in moves.split():
      b[lane].play_str(move)
  return b

def search(evaluate=lambda t,s: 0., **kwargs):
  args = dict(max_depth=3, q_depth=2, max_nodes=1000,
              mate_value=1000., tree_cap=1024, temperature=0.,
              rng=np.random.default_rng(42))
  args.update(kwargs)
  return AlphaBetaQuiescence(evaluate, PV, **args)

class QuiescenceTests(unittest.TestCase):
  def test_check_flags_on_roots_children_and_copied_history(self):
    b = position('e2-e4 d7-d5 f1-b5', lanes=2)
    self.assertEqual(b.in_check().tolist(), [True, True])
    a = b[0].moves().get(b[0].parse('c7-c6'))
    child, _, _ = b.get_children(np.array([0,0]), np.array([a,a]))
    self.assertEqual(b.in_check(states=child).tolist(), [False, False])
    c = GameBatch(1)
    c.copy_lane_from(b, 1)
    self.assertTrue(c.in_check()[0])
    white = position('f2-f3 e7-e5 a2-a3 d8-h4')
    self.assertTrue(white.in_check()[0])

  def test_no_stand_pat_and_quiet_evasions_even_at_q_depth_zero(self):
    b = position('e2-e4 d7-d5 f1-b5')
    for qdepth in (0, 1):
      q = search(lambda t,s: 100. if s == 0 else 7.)
      q.tree.copy_lane_from(b, 0)
      q._used, q._nodes, q._truncated = 1, 0, 0
      _, forcing = q._ordered(0)
      self.assertFalse(forcing.any())
      self.assertEqual(q._qsearch(0, 0, qdepth, -np.inf, 50.), -7.)
      self.assertGreater(q._used, 1)

  def test_tiny_budget_still_finds_late_root_mate_and_respects_mask(self):
    b = position('f2-f3 e7-e5 g2-g4', lanes=2)
    q = search(root_prefetch=True, max_nodes=1, tree_cap=2)
    moves, info = q.select_moves(b, np.array([False, True]))
    self.assertEqual(str(b[1].moves()[moves[1]]), 'd8-h4')
    self.assertEqual(info['root_evaluated'][1], b.num_moves_all()[1])
    self.assertEqual(info['root_evaluated'][0], 0)
    self.assertEqual(info['value'][1], 999.)
    self.assertEqual(q._root_deep.sum(), 0)
    b.play_batch(moves, np.array([False, True]))
    self.assertEqual(b.results()[1], int(GameResult.CHECKMATE))
    self.assertEqual(b.results()[0], int(GameResult.ONGOING))

  def test_root_alpha_shared_but_not_initialized_from_shallow_values(self):
    class WindowSearch(AlphaBetaQuiescence):
      def _search(self, state, result, depth, ply, alpha, beta):
        self.windows.append((alpha,beta))
        return -1.
    q = WindowSearch(lambda t,s: -100., PV, max_depth=2, q_depth=0,
                     max_nodes=1000, mate_value=1000., tree_cap=1024,
                     temperature=0., rng=np.random.default_rng(42), root_prefetch=True)
    q.windows = []
    b = GameBatch(1)
    q.select_moves(b, np.array([True]))
    self.assertEqual(q.windows[0], (-np.inf, np.inf))
    self.assertEqual(len(q.windows), 40)
    for offset in (0, 20):
      self.assertEqual(q.windows[offset], (-np.inf, np.inf))
      self.assertTrue(all(w == (-np.inf, -1.) for w in q.windows[offset+1:offset+20]))
    self.assertTrue(np.all(q._root_initial == 100.))
    self.assertTrue(np.all(q._root_values == 1.))

  def test_shallow_fallback_scores_all_moves_and_temperature_is_finite(self):
    b = GameBatch(1)
    q = search(root_prefetch=True, max_nodes=1, temperature=.2)
    moves, info = q.select_moves(b, np.array([True]))
    self.assertEqual(info['root_evaluated'][0], 20)
    self.assertEqual(info['root_deep'][0], 0)
    self.assertTrue(np.isfinite(q._root_values).all())
    self.assertIn(int(moves[0]), range(20))

  def test_immediate_repetition_is_zero_without_static_evaluation(self):
    b = position('g1-f3 g8-f6 f3-g1 f6-g8 g1-f3 g8-f6 f3-g1')
    q = search(lambda t,s: 7., root_prefetch=True, max_nodes=1)
    q.select_moves(b, np.array([True]))
    a = b[0].moves().get(b[0].parse('f6-g8'))
    i = np.flatnonzero(q._root_actions == a)[0]
    self.assertEqual(q._root_initial[i], 0.)


class RootSelectionTests(unittest.TestCase):
  def test_incomplete_first_iteration_keeps_entire_static_fallback(self):
    class Limited(AlphaBetaQuiescence):
      def _search(self, *args):
        self._nodes = self.max_nodes
        return -1.
    q = Limited(lambda t,s: -100., PV, max_depth=2, q_depth=0,
                max_nodes=1000, mate_value=1000., tree_cap=1024,
                temperature=0., rng=np.random.default_rng(42), root_prefetch=True)
    b = GameBatch(1)
    move, info = q.select_moves(b, np.array([True]))
    self.assertEqual(move[0], q._root_actions[0])
    self.assertEqual(info['value'][0], 100.)
    self.assertEqual(info['completed_depth'][0], 0)
    self.assertTrue(np.all(q._root_values[1:] == 100.))

  def test_partial_search_does_not_replace_greedy_depth_one_fallback(self):
    class Limited(AlphaBetaQuiescence):
      def _search(self, *args):
        self._truncated += 1
        return -500.
    q = Limited(lambda t,s: 2., PV, max_depth=2, q_depth=0,
                max_nodes=1000, mate_value=1000., tree_cap=1024,
                temperature=0., rng=np.random.default_rng(42), root_prefetch=True)
    _, info = q.select_moves(GameBatch(1), np.array([True]))
    self.assertEqual(info['value'][0], -2.)

class IterativeTests(unittest.TestCase):
  def test_interrupted_iteration_preserves_last_completed_scores_for_sampling(self):
    class Interrupted(AlphaBetaQuiescence):
      def _search(self, state, result, depth, ply, alpha, beta):
        self.depths.append(depth)
        if depth == 2:
          self._truncated += 1
          return -999.
        return -float(depth + 1)
      def _pick(self, actions, values):
        self.sample_scores = np.array(values)
        return super()._pick(actions, values)
    q = Interrupted(lambda t,s: 0., PV, max_depth=4, q_depth=2,
                    max_nodes=1000, mate_value=1000., tree_cap=1024,
                    temperature=.1, rng=np.random.default_rng(42), root_prefetch=True)
    q.depths = []
    _, info = q.select_moves(GameBatch(1), np.array([True]))
    self.assertEqual(q.depths, [0]*20 + [1]*20 + [2])
    self.assertEqual(info['completed_depth'][0], 2)
    self.assertEqual(info['attempted_depth'][0], 3)
    self.assertTrue(np.all(q.sample_scores == 2.))

  def test_real_search_completes_iterations_with_one_cumulative_budget(self):
    q = search(root_prefetch=True, max_depth=2, q_depth=0, max_nodes=10000)
    _, info = q.select_moves(GameBatch(1), np.array([True]))
    self.assertEqual(info['completed_depth'][0], 2)
    self.assertEqual(info['root_evaluated'][0], 20)
    self.assertEqual(info['truncated'][0], 0)
    self.assertGreater(info['nodes'][0], 20)

if __name__ == '__main__':
  unittest.main()
