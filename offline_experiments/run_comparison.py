"""
Online vs Offline DIMOQ comparison (Phase A).

Protocol
--------
For each seed:
  1. Run MODVI on the true environment model  →  ground-truth DDS.
  2. Run online DIMOQ (warmup fills transitions via random walks; training
     updates reward_dists and non_dominated interleaved but does NOT update
     transitions — exactly as described in the paper).
  3. Copy the learned model (transitions + reward_dists) into OfflineDIMOQ,
     reset non_dominated, and run fixed-point planning to convergence.
  4. Evaluate all three against MODVI ground truth.

Notes
-----
- DIMOQ uses augmented states (timestep, state_id) to eliminate self-loops in
  the non-stationary finite-horizon setting, matching the paper's approach.
- MODVI operates directly on the true model and does not need augmented states.
  A separate non-augmented env instance is created for MODVI.

Run from the repo root:
  python offline_experiments/run_comparison.py [--env small|medium] [--seeds 1 2 3 4 5]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from algs.dimoq import DIMOQ
from algs.modvi import MODVI
from envs.random_momdp import RandomMOMDP
from offline_experiments.offline_dimoq import OfflineDIMOQ
from offline_experiments.metrics import compute_metrics, print_metrics


# --------------------------------------------------------------------------- #
# Environment configs
# --------------------------------------------------------------------------- #

ENV_CONFIGS = {
    'small': dict(
        num_states=5, num_objectives=2, num_actions=2,
        min_next_states=1, max_next_states=2, num_terminal_states=1,
        reward_min=np.zeros(2, dtype=np.float32), reward_max=np.ones(2) * 5,
        reward_dist='discrete', start_state=0, max_timesteps=3,
    ),
    'medium': dict(
        num_states=10, num_objectives=2, num_actions=3,
        min_next_states=1, max_next_states=2, num_terminal_states=1,
        reward_min=np.zeros(2, dtype=np.float32), reward_max=np.ones(2) * 5,
        reward_dist='discrete', start_state=0, max_timesteps=5,
    ),
}

# Distribution params shared across all algorithms.
# v_maxs covers the maximum possible cumulative return:
#   small:  3 steps × 5 max reward = 15  →  20 gives headroom
#   medium: 5 steps × 5 max reward = 25  →  30 gives headroom
DIST_PARAMS = {
    'small': dict(
        ref_point=np.array([0., 0.]),
        num_atoms=(21, 21),
        v_mins=(0., 0.),
        v_maxs=(20., 20.),
        max_dists=10,
        gamma=1.0,
        initial_epsilon=1.0,
        epsilon_decay=0.9975,
        final_epsilon=0.1,
    ),
    'medium': dict(
        ref_point=np.array([0., 0.]),
        num_atoms=(21, 21),
        v_mins=(0., 0.),
        v_maxs=(30., 30.),
        max_dists=15,
        gamma=1.0,
        initial_epsilon=1.0,
        epsilon_decay=0.9975,
        final_epsilon=0.1,
    ),
}


def make_dimoq_env(size, seed):
    """Augmented-state env for DIMOQ (eliminates self-loops via timestep dimension)."""
    cfg = dict(ENV_CONFIGS[size], augment_state=True)
    return RandomMOMDP(seed=seed, **cfg)


def make_modvi_env(size, seed):
    """Non-augmented env for MODVI (operates on true model directly)."""
    cfg = dict(ENV_CONFIGS[size], augment_state=False)
    return RandomMOMDP(seed=seed, **cfg)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env', default='small', choices=list(ENV_CONFIGS),
                   help='Environment size (default: small)')
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                   help='Random seeds (default: 1 2 3 4 5)')
    p.add_argument('--num-episodes', type=int, default=3000,
                   help='Online training episodes (default: 3000)')
    p.add_argument('--warmup', type=int, default=10000,
                   help='Warmup episodes for transition estimation (default: 10000)')
    p.add_argument('--offline-iters', type=int, default=50,
                   help='Fixed-point planning iterations for offline DIMOQ (default: 50)')
    p.add_argument('--modvi-iters', type=int, default=20,
                   help='MODVI iterations for ground truth (default: 20)')
    p.add_argument('--log-every', type=int, default=1000,
                   help='Log interval for online training (default: 1000)')
    p.add_argument('--pr-threshold', type=float, default=1.0,
                   help='W2 threshold for precision/recall in atom-index units (default: 1.0)')
    p.add_argument('--out-dir', type=str,
                   default=os.path.join(os.path.dirname(__file__), 'results'),
                   help='Directory for JSON results')
    return p.parse_args()


def run_seed(seed, args):
    dp = DIST_PARAMS[args.env]

    # ------------------------------------------------------------------ #
    # 1. Ground truth via MODVI (uses non-augmented env + true model)
    # ------------------------------------------------------------------ #
    modvi_env = make_modvi_env(args.env, seed)
    print(f'  [MODVI] running {args.modvi_iters} iterations...')
    t0 = time.time()
    modvi = MODVI(modvi_env, dp['gamma'], dp['num_atoms'], dp['v_mins'], dp['v_maxs'])
    modvi_dds = modvi.get_dds(num_iters=args.modvi_iters)
    print(f'  [MODVI] done in {time.time()-t0:.1f}s  |  DDS size = {len(modvi_dds)}')

    # ------------------------------------------------------------------ #
    # 2. Online DIMOQ (augmented-state env matching paper protocol)
    # ------------------------------------------------------------------ #
    dimoq_env = make_dimoq_env(args.env, seed)
    print(f'  [Online] warmup={args.warmup}, episodes={args.num_episodes}...')
    t0 = time.time()
    online_dimoq = DIMOQ(
        dimoq_env, dp['ref_point'], dp['gamma'],
        initial_epsilon=dp['initial_epsilon'],
        epsilon_decay=dp['epsilon_decay'],
        final_epsilon=dp['final_epsilon'],
        num_atoms=dp['num_atoms'],
        v_mins=dp['v_mins'],
        v_maxs=dp['v_maxs'],
        max_dists=dp['max_dists'],
        seed=seed, log=False,
    )
    online_dds = online_dimoq.train(
        num_episodes=args.num_episodes,
        warmup_time=args.warmup,
        learn_model=False,          # paper: transitions frozen after warmup
        log_every=args.log_every,
        action_eval='linear',
    )
    online_time = time.time() - t0
    print(f'  [Online] done in {online_time:.1f}s  |  DDS size = {len(online_dds)}')

    # ------------------------------------------------------------------ #
    # 3. Offline DIMOQ (same model, planning from scratch)
    # ------------------------------------------------------------------ #
    dimoq_env.reset()
    print(f'  [Offline] planning {args.offline_iters} iters on copied model...')
    t0 = time.time()
    offline_dimoq = OfflineDIMOQ(
        dimoq_env, dp['ref_point'], dp['gamma'],
        num_atoms=dp['num_atoms'],
        v_mins=dp['v_mins'],
        v_maxs=dp['v_maxs'],
        max_dists=dp['max_dists'],
        seed=seed, log=False,
    )
    offline_dimoq.copy_model_from(online_dimoq)
    offline_dds = offline_dimoq.train_offline(
        num_iters=args.offline_iters, log_every=10,
    )
    offline_time = time.time() - t0
    print(f'  [Offline] done in {offline_time:.1f}s  |  DDS size = {len(offline_dds)}')

    # ------------------------------------------------------------------ #
    # 4. Metrics
    # ------------------------------------------------------------------ #
    rp = dp['ref_point']
    thr = args.pr_threshold
    return {
        'seed': seed,
        'online_time_s': round(online_time, 2),
        'offline_time_s': round(offline_time, 2),
        'modvi':   compute_metrics(modvi_dds,   modvi_dds, rp, thr),
        'online':  compute_metrics(online_dds,  modvi_dds, rp, thr),
        'offline': compute_metrics(offline_dds, modvi_dds, rp, thr),
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = []
    for seed in args.seeds:
        print(f'\n=== Seed {seed} ===')
        result = run_seed(seed, args)
        all_results.append(result)

        print(f'\n  Results (W2 threshold = {args.pr_threshold} atom-index units):')
        for alg in ('modvi', 'online', 'offline'):
            print_metrics(alg, result[alg])

    # Aggregate across seeds
    print('\n=== Aggregate (mean ± std across seeds) ===')
    for alg in ('modvi', 'online', 'offline'):
        for metric in ('size', 'hypervolume', 'hv_ratio', 'hausdorff_w2', 'precision', 'recall'):
            vals = [r[alg][metric] for r in all_results]
            print(f'  {alg:8s}  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')

    # Save results
    out_path = os.path.join(args.out_dir, f'comparison_{args.env}.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
