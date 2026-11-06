import unittest
import numpy as np
from engine.game import GameBatch
from test_quiescence import position, search

class TranspositionTests(unittest.TestCase):
  def prepare(self, q, b=None):
    q.tree.copy_lane_from(b or GameBatch(1), 0)
    q.tree.reset_trees()
    q._state_keys.clear()
    q._used, q._nodes, q._truncated = 1, 0, 0

  def test_history_distinguishes_identical_boards_with_different_repetition_context(self):
    a = position('g1-f3 g8-f6 f3-g1 f6-g8')
    b = position('b1-c3 b8-c6 c3-b1 c6-b8')
    self.assertEqual(a.hashes()[0], b.hashes()[0])
    self.assertEqual(a.search_keys()[0,2], b.search_keys()[0,2])
    self.assertNotEqual(a.search_keys()[0,1], b.search_keys()[0,1])
    for batch in (a,b): batch[0].play_str('e2-e4')
    # Old reversible history cannot affect repetition after this pawn move.
    np.testing.assert_array_equal(a.search_keys(), b.search_keys())

  def test_simulated_keys_match_actual_play_and_survive_copy(self):
    actual = GameBatch(1)
    tree = GameBatch(1)
    for move in 'g1-f3 g8-f6 f3-g1 f6-g8 e2-e4 d7-d5 e4-d5'.split():
      tree.copy_lane_from(actual,0)
      action = actual[0].moves().get(actual[0].parse(move))
      child,_,_ = tree.get_children(np.array([0],np.int32),np.array([action],np.int32))
      actual[0].play_str(move)
      np.testing.assert_array_equal(actual.search_keys(), tree.search_keys(states=child))
    copied = GameBatch(1)
    copied.copy_lane_from(actual,0)
    np.testing.assert_array_equal(actual.search_keys(), copied.search_keys())

  def test_exact_and_bounds_only_cut_off_compatible_windows(self):
    q = search(tt_capacity=100)
    self.prepare(q)
    calls = []
    def evaluate(): calls.append(1); return 3.
    q._cached_search(0,2,0,-np.inf,2.,'normal',evaluate)  # lower bound
    self.assertEqual(q._cached_search(0,2,0,-np.inf,2.,'normal',evaluate),3.)
    self.assertEqual(len(calls),1)
    q._cached_search(0,2,0,-np.inf,np.inf,'normal',evaluate)
    self.assertEqual(len(calls),2)  # lower bound cannot answer a full window
    q._cached_search(0,2,0,-np.inf,np.inf,'normal',evaluate)
    self.assertEqual(len(calls),2)
    q._cached_search(0,3,0,-np.inf,np.inf,'normal',evaluate)
    q._cached_search(0,2,0,-np.inf,np.inf,'quiet',evaluate)
    self.assertEqual(len(calls),4)
    q.clear_cache()
    q._cached_search(0,2,0,4.,np.inf,'normal',evaluate)  # upper bound
    q._cached_search(0,2,0,4.,np.inf,'normal',evaluate)
    self.assertEqual(len(calls),5)
    q._cached_search(0,2,0,-np.inf,np.inf,'normal',evaluate)
    self.assertEqual(len(calls),6)

  def test_mate_distance_adjusts_to_new_root_and_table_is_bounded(self):
    q = search(tt_capacity=2)
    self.prepare(q)
    q._cached_search(0,1,5,-np.inf,np.inf,'normal',lambda:993.)
    self.assertEqual(q._cached_search(0,1,1,-np.inf,np.inf,'normal',lambda:0.),997.)
    q._cached_search(0,2,5,-np.inf,np.inf,'normal',lambda:-993.)
    self.assertEqual(q._cached_search(0,2,1,-np.inf,np.inf,'normal',lambda:0.),-997.)
    q._cached_search(0,3,0,-np.inf,np.inf,'normal',lambda:0.)
    self.assertEqual(len(q.tt),2)
    q.clear_cache()
    self.assertFalse(q.tt)

  def test_truncated_result_is_not_stored(self):
    q = search(tt_capacity=100)
    self.prepare(q)
    def interrupted(): q._truncated += 1; return 5.
    q._cached_search(0,2,0,-np.inf,np.inf,'normal',interrupted)
    self.assertFalse(q.tt)

  def test_completed_search_matches_without_table_and_reuses_results(self):
    def material(tree,state):
      board=tree.encode(states=np.array([state],np.int32))[0][0]
      return float(np.array([0,1,3,3,5,9,0,-1,-3,-3,-5,-9,0])[board].sum())
    b=position('e2-e4 d7-d5')
    plain=search(material,root_prefetch=True,max_depth=2,max_nodes=100000)
    cached=search(material,root_prefetch=True,max_depth=2,max_nodes=100000,tt_capacity=10000)
    a,ia=plain.select_moves(b,np.array([True]));c,ic=cached.select_moves(b,np.array([True]))
    np.testing.assert_array_equal(a,c)
    np.testing.assert_array_equal(ia['value'],ic['value'])
    self.assertEqual(ic['completed_depth'][0],2)
    _,warm=cached.select_moves(b,np.array([True]))
    self.assertGreater(warm['tt_cutoffs'][0],0)
    self.assertEqual(warm['nodes'][0],0)

if __name__ == '__main__': unittest.main()
