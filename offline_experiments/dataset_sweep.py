"""
Phase B: Dataset-quality sweep.

For each seed we train one online DIMOQ, then collect offline datasets from
five behavior policies at four dataset sizes and compare the resulting offline
DDS to the MODVI ground truth.

Behavior policies
-----------------
  random          Uniform random — maximum, unbiased coverage.
  eps_0.5         ε-greedy (ε=0.5) on trained DIMOQ's linear-utility scores.
  eps_0.1         ε-greedy (ε=0.1) — near-greedy, concentrated coverage.
  scal_0.1        Scalarized greedy: α=0.1 → strongly prefers objective 2.
  scal_0.9        Scalarized greedy: α=0.9 → strongly prefers objective 1.

Dataset sizes
-------------
  1 000, 5 000, 25 000, 100 000 transitions.

Run from the repo root:
  python offline_experiments/dataset_sweep.py [--env small|medium] [--seeds 1 2 3 4 5]
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
from offline_experiments.offline_dimoq import OfflineDIMOQ
from offline_experiments.collect import (
    collect_dataset, random_policy,
    make_score_table, make_scalarized_score_table,
    greedy_policy, epsilon_greedy_policy,
)
from offline_experiments.metrics import compute_metrics, print_metrics
from offline_experiments.run_comparison import (
    ENV_CONFIGS, DIST_PARAMS, make_dimoq_env, make_modvi_env,
)


DATASET_SIZES = [1_000, 5_000, 25_000, 100_000]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env', default='small', choices=list(ENV_CONFIGS))
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--num-episodes', type=int, default=3000,
                   help='Online training episodes (default: 3000)')
    p.add_argument('--warmup', type=int, default=50000,
                   help='Warmup episodes for online DIMOQ (default: 50000)')
    p.add_argument('--offline-iters', type=int, default=50)
    p.add_argument('--modvi-iters', type=int, default=20)
    p.add_argument('--pr-threshold', type=float, default=1.0,
                   help='W2 threshold for precision/recall (default: 1.0)')
    p.add_argument('--out-dir', type=str,
                   default=os.path.join(os.path.dirname(__file__), 'results'))
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Per-seed logic
# --------------------------------------------------------------------------- #

def run_seed(seed, args):
    dp = DIST_PARAMS[args.env]
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 1. Ground truth (MODVI)
    # ------------------------------------------------------------------ #
    modvi_env = make_modvi_env(args.env, seed)
    modvi = MODVI(modvi_env, dp['gamma'], dp['num_atoms'], dp['v_mins'], dp['v_maxs'])
    modvi_dds = modvi.get_dds(num_iters=args.modvi_iters)
    print(f'  [MODVI] DDS size = {len(modvi_dds)}')

    # ------------------------------------------------------------------ #
    # 2. Online DIMOQ baseline
    # ------------------------------------------------------------------ #
    dimoq_env = make_dimoq_env(args.env, seed)
    print(f'  [Online] warmup={args.warmup}, episodes={args.num_episodes}...')
    t0 = time.time()
    online_dimoq = DIMOQ(
        dimoq_env, dp['ref_point'], dp['gamma'],
        initial_epsilon=dp['initial_epsilon'],
        epsilon_decay=dp['epsilon_decay'],
        final_epsilon=dp['final_epsilon'],
        num_atoms=dp['num_atoms'], v_mins=dp['v_mins'], v_maxs=dp['v_maxs'],
        max_dists=dp['max_dists'], seed=seed, log=False,
    )
    online_dds = online_dimoq.train(
        num_episodes=args.num_episodes, warmup_time=args.warmup,
        learn_model=False, log_every=args.num_episodes + 1,
        action_eval='linear',
    )
    print(f'  [Online] {time.time()-t0:.1f}s  |  DDS size = {len(online_dds)}')

    # ------------------------------------------------------------------ #
    # 3. Build behavior policy score tables (pre-computed once per seed)
    # ------------------------------------------------------------------ #
    print('  [Sweep] pre-computing policy score tables...')
    linear_table   = make_score_table(online_dimoq, 'linear')
    scal01_table   = make_scalarized_score_table(online_dimoq, alpha=0.1)
    scal09_table   = make_scalarized_score_table(online_dimoq, alpha=0.9)

    policies = {
        'random':   random_policy(online_dimoq.num_actions, rng),
        'eps_0.5':  epsilon_greedy_policy(linear_table, online_dimoq.num_actions, 0.5, rng),
        'eps_0.1':  epsilon_greedy_policy(linear_table, online_dimoq.num_actions, 0.1, rng),
        'scal_0.1': greedy_policy(scal01_table, rng),
        'scal_0.9': greedy_policy(scal09_table, rng),
    }

    # flatten_fn converts augmented obs to flat int
    flatten_fn = online_dimoq._flatten_state

    # ------------------------------------------------------------------ #
    # 4. Sweep: collect dataset → offline DIMOQ → metrics
    # ------------------------------------------------------------------ #
    rp, thr = dp['ref_point'], args.pr_threshold
    seed_results = {
        'seed': seed,
        'modvi': compute_metrics(modvi_dds, modvi_dds, rp, thr),
        'online': compute_metrics(online_dds, modvi_dds, rp, thr),
        'sweep': {},
    }

    for policy_name, policy_fn in policies.items():
        seed_results['sweep'][policy_name] = {}

        for n_trans in DATASET_SIZES:
            print(f'  [{policy_name}  N={n_trans:7d}] collecting...', end=' ', flush=True)
            t0 = time.time()

            # Collect dataset
            dimoq_env.reset()
            dataset = collect_dataset(dimoq_env, n_trans, policy_fn, flatten_fn)

            # Build and run offline DIMOQ
            dimoq_env.reset()
            offline = OfflineDIMOQ(
                dimoq_env, dp['ref_point'], dp['gamma'],
                num_atoms=dp['num_atoms'], v_mins=dp['v_mins'], v_maxs=dp['v_maxs'],
                max_dists=dp['max_dists'], seed=seed, log=False,
            )
            offline.load_dataset(dataset)
            offline_dds = offline.train_offline(
                num_iters=args.offline_iters, log_every=args.offline_iters + 1,
            )

            m = compute_metrics(offline_dds, modvi_dds, rp, thr)
            elapsed = time.time() - t0
            print(f'done in {elapsed:.1f}s  |  size={m["size"]}  '
                  f'hv_ratio={m["hv_ratio"]:.3f}  R={m["recall"]:.3f}')
            seed_results['sweep'][policy_name][n_trans] = m

    return seed_results


# --------------------------------------------------------------------------- #
# Aggregate and print
# --------------------------------------------------------------------------- #

def aggregate(all_results, policy_name, n_trans, metric):
    vals = [r['sweep'][policy_name][n_trans][metric] for r in all_results]
    return float(np.mean(vals)), float(np.std(vals))


def print_summary(all_results):
    policies = list(all_results[0]['sweep'].keys())
    print('\n=== Aggregate recall (mean ± std) across seeds ===')
    header = f'{"policy":12s}' + ''.join(f'  N={n:>7d}' for n in DATASET_SIZES)
    print(header)
    for pol in policies:
        row = f'{pol:12s}'
        for n in DATASET_SIZES:
            mu, sd = aggregate(all_results, pol, n, 'recall')
            row += f'  {mu:.3f}±{sd:.3f}'
        print(row)

    print('\n=== Aggregate hv_ratio (mean ± std) across seeds ===')
    print(header)
    for pol in policies:
        row = f'{pol:12s}'
        for n in DATASET_SIZES:
            mu, sd = aggregate(all_results, pol, n, 'hv_ratio')
            row += f'  {mu:.3f}±{sd:.3f}'
        print(row)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = []
    for seed in args.seeds:
        print(f'\n=== Seed {seed} ===')
        result = run_seed(seed, args)
        all_results.append(result)

    print_summary(all_results)

    out_path = os.path.join(args.out_dir, f'dataset_sweep_{args.env}.json')
    # Convert numpy scalars for JSON serialisation
    def to_py(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): to_py(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_py(v) for v in obj]
        return obj

    with open(out_path, 'w') as f:
        json.dump(to_py(all_results), f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
