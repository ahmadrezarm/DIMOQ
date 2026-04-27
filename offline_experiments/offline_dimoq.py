import copy
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from algs.dimoq import DIMOQ
from utils.dist_metrics import get_best


class OfflineDIMOQ(DIMOQ):
    """
    Offline variant of DIMOQ.

    Follows the same protocol as the paper for the online setting:
      - Transition probabilities are estimated from data (warmup / behavior policy).
      - During planning, transitions are NOT updated — only non_dominated is iterated
        to a fixed point using the frozen model.

    Usage:
      1. Call copy_model_from(online_dimoq) to transfer the estimated transition
         counts and reward distributions from an online run.
      2. Call train_offline(num_iters) to run fixed-point planning.
    """

    def load_dataset(self, dataset):
        """Populate transitions and reward_dists from a list of (s, a, r, s') tuples.

        States must already be flattened integers (use DIMOQ._flatten_state at
        collection time).
        """
        for s, a, r, s_next in dataset:
            self.transitions[s, a, s_next] += 1
            self.reward_dists[s][a][s_next].update(r)

    def copy_model_from(self, online_dimoq):
        """Copy transitions and reward_dists from a completed online DIMOQ run.

        non_dominated is left at its zero-initialized state so planning starts fresh.
        """
        self.transitions = online_dimoq.transitions.copy()
        self.reward_dists = copy.deepcopy(online_dimoq.reward_dists)

    def train_offline(self, num_iters=50, log_every=10):
        """Fixed-point iteration over non_dominated using the frozen model.

        For every (s, a, s') triple observed in the data (transitions > 0), updates
        non_dominated[s][a][s'] from the current Q-sets at s'.  Repeats for num_iters
        sweeps.  Returns the final DDS at state 0.
        """
        pool = self.make_pool()

        for i in range(num_iters):
            if i % log_every == 0:
                dds = self.get_local_dds(pool, state=0, keep_best=False)
                print(f'  Offline iter {i:3d}: DDS size = {len(dds)}')

            for s in range(self.num_states):
                for a in range(self.num_actions):
                    for s_next in range(self.num_states):
                        if self.transitions[s, a, s_next] > 0:
                            new_nd = self.calc_non_dominated(s_next, pool)
                            self.non_dominated[s][a][s_next] = get_best(
                                new_nd, max_dists=self.max_dists, rng=self.rng
                            )

        final_dds = self.get_local_dds(pool, state=0)
        self.close_pool(pool)
        return final_dds
