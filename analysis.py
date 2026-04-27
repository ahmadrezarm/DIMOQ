import os
import json
import argparse
import time
import pandas as pd
import numpy as np
from collections import defaultdict

output = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='logs', help='The directory of the saved logs.')
    parser.add_argument("--seed", type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help="The seeds used in the experiment.")
    parser.add_argument("--env", type=str, nargs='+', default=["small", "medium", "large"],
                        help="The environments of the experiment.")
    parser.add_argument("--alg", type=str, nargs='+', default=['DIMOQ'], help="The algorithms that were used.")
    args = parser.parse_args()
    return args


def get_subset_sizes(env_name, alg):
    subset_sizes = defaultdict(list)

    for seed in args.seed:
        dists_dir = os.path.join(args.log_dir, env_name, str(seed), alg)
        df = pd.read_csv(os.path.join(dists_dir, 'pruning_results.csv'))
        subset_sizes['dds'].append(len(df[df['dds'] == 1]))
        subset_sizes['cdds'].append(len(df[(df['dds'] == 1) & (df['cdds'] == 1)]))
        subset_sizes['pf'].append(len(df[(df['dds'] == 1) & (df['pf'] == 1)]))
        subset_sizes['ch'].append(len(df[(df['dds'] == 1) & (df['ch'] == 1)]))

    subset_sizes = {k: np.array(v) for k, v in subset_sizes.items()}
    return subset_sizes


def check_pf_subset(env_name, alg):
    counts = []
    for seed in args.seed:
        dists_dir = os.path.join(args.log_dir, env_name, str(seed), alg)
        df = pd.read_csv(os.path.join(dists_dir, 'pruning_results.csv'))
        pf_not_cdus = df[(df['dds'] == 1) & (df['cdds'] == 0) & (df['pf'] == 1)]
        counts.append(int(len(pf_not_cdus)))
        print(f"Number of policies in the PF and not in the CDUS: {len(pf_not_cdus)}")
    output[env_name][alg]['pf_not_in_cdds_per_seed'] = counts


def get_percentages():
    subset_sizes = get_subset_sizes(env_name, alg)
    print(f'Mean and std of start DDS: {np.mean(subset_sizes["dds"])} +- {np.std(subset_sizes["dds"])}')

    output[env_name][alg]['subset_sizes'] = {
        'dds_mean': float(np.mean(subset_sizes['dds'])),
        'dds_std': float(np.std(subset_sizes['dds'])),
    }
    output[env_name][alg]['fractions'] = {}
    for subset, sizes in subset_sizes.items():
        percentage = sizes / subset_sizes['dds'] * 100
        mean = np.mean(percentage)
        std = np.std(percentage)
        output[env_name][alg]['fractions'][subset] = {'mean': float(mean), 'std': float(std)}
        print(f'Fraction {subset}: {mean:.2f}% +- {std:.2f}%')


def get_alg_stats():
    alg_durations = []

    for seed in args.seed:
        dists_dir = os.path.join(args.log_dir, env_name, str(seed), alg)
        with open(os.path.join(dists_dir, f'{alg}.json'), 'r') as f:
            alg_duration = json.load(f)
            alg_durations.append(alg_duration['duration'])

    alg_mean = np.mean(alg_durations)
    alg_std = np.std(alg_durations)
    alg_min = np.min(alg_durations)
    alg_max = np.max(alg_durations)

    output[env_name][alg]['algorithm'] = {
        'mean': time.strftime("%H:%M:%S", time.gmtime(alg_mean)),
        'std': time.strftime("%H:%M:%S", time.gmtime(alg_std)),
        'min': time.strftime("%H:%M:%S", time.gmtime(alg_min)),
        'max': time.strftime("%H:%M:%S", time.gmtime(alg_max)),
        'seed_min': int(args.seed[np.argmin(alg_durations)]),
        'seed_max': int(args.seed[np.argmax(alg_durations)]),
    }

    print(f'Algorithm: {alg}')
    print(f'Minimum found for seed: {args.seed[np.argmin(alg_durations)]}')
    print(f'Maximum found for seed: {args.seed[np.argmax(alg_durations)]}')
    print(f'Algorithm results: mean = {output[env_name][alg]["algorithm"]["mean"]}, std = {output[env_name][alg]["algorithm"]["std"]}, min = {output[env_name][alg]["algorithm"]["min"]}, max = {output[env_name][alg]["algorithm"]["max"]}')


def get_prune_stats():
    prune_durations = defaultdict(list)

    for seed in args.seed:
        dists_dir = os.path.join(args.log_dir, env_name, str(seed), alg)
        with open(os.path.join(dists_dir, 'pruning_durations.json'), 'r') as f:
            prune_res = json.load(f)
            for prune, durations in prune_res.items():
                prune_durations[prune].append(durations)

    output[env_name][alg]['pruning_durations'] = {}
    for prune, durations in prune_durations.items():
        prune_mean = np.mean(durations)
        prune_std = np.std(durations)
        prune_min = np.min(durations)
        prune_max = np.max(durations)
        output[env_name][alg]['pruning_durations'][prune] = {
            'mean': prune_mean, 'std': prune_std, 'min': prune_min, 'max': prune_max,
        }
        print(
            f'Pruning results - {prune}: mean = {prune_mean}, std = {prune_std}, min = {prune_min}, max = {prune_max}')


if __name__ == "__main__":
    args = parse_args()

    print(f'Analysing results')
    for env_name in args.env:
        output[env_name] = {}
        for alg in args.alg:
            output[env_name][alg] = {}
            get_alg_stats()
            print('---------------------------')
            get_prune_stats()
            print('---------------------------')
            get_percentages()
            print('---------------------------')
            check_pf_subset(env_name, alg)
            print('--------------------------------------------------------------------------')

    out_path = os.path.join(args.log_dir, 'analysis_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Results saved to {out_path}')
    print(f'Finished analysing results')
