"""
Visualise DDS recovery across behavior policies.

Figure 1  – EV scatter (per policy, N=100k):
            Which MODVI solutions were recovered (green) or missed (red)?
Figure 2  – Distribution heatmaps:
            Actual return distributions for recovered vs missed solutions
            under the most biased policy (scal_0.1, N=100k).
Figure 3  – Data-budget effect (eps_0.1):
            EV scatter at N = 1k / 5k / 25k / 100k — shows that data
            starvation, unlike preference bias, heals with more data.

Run from repo root:
    python offline_experiments/visualize.py [--seed 1]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull

from algs.dimoq import DIMOQ
from algs.modvi import MODVI
from offline_experiments.offline_dimoq import OfflineDIMOQ
from offline_experiments.collect import (
    collect_dataset, random_policy,
    make_score_table, make_scalarized_score_table,
    greedy_policy, epsilon_greedy_policy,
)
from offline_experiments.run_comparison import (
    DIST_PARAMS, make_dimoq_env, make_modvi_env,
)


# --------------------------------------------------------------------------- #
# Matching helpers
# --------------------------------------------------------------------------- #

EV_MATCH_THRESHOLD = 1.5   # in return-value / atom-index units

def get_evs(dds):
    if not dds:
        return np.zeros((0, 2))
    return np.array([d.get_expected_value() for d in dds])


def match(offline_dds, modvi_dds):
    """
    For each MODVI distribution, check whether the offline DDS contains a
    distribution within EV_MATCH_THRESHOLD (Euclidean distance in EV space).

    Returns
    -------
    recovered : bool array, shape (len(modvi_dds),)
    phantom   : bool array, shape (len(offline_dds),)  — offline solutions
                with no nearby MODVI solution
    """
    m_evs = get_evs(modvi_dds)
    o_evs = get_evs(offline_dds)

    if len(o_evs) == 0:
        return np.zeros(len(m_evs), bool), np.zeros(0, bool)

    # For each MODVI point: is there a nearby offline point?
    diffs = m_evs[:, None, :] - o_evs[None, :, :]   # (M, O, 2)
    dists = np.linalg.norm(diffs, axis=-1)            # (M, O)
    recovered = dists.min(axis=1) < EV_MATCH_THRESHOLD

    # For each offline point: is there a nearby MODVI point?
    phantom = dists.min(axis=0) >= EV_MATCH_THRESHOLD

    return recovered, phantom


def pareto_front_evs(evs):
    """Return EV points on the Pareto front (non-dominated in both objectives)."""
    if len(evs) == 0:
        return evs
    dominated = np.zeros(len(evs), bool)
    for i, p in enumerate(evs):
        if dominated[i]:
            continue
        dominated |= np.all(evs >= p, axis=1) & np.any(evs > p, axis=1)
        dominated[i] = False
    pf = evs[~dominated]
    return pf[pf[:, 0].argsort()]   # sort by objective 0


# --------------------------------------------------------------------------- #
# Experiment runner
# --------------------------------------------------------------------------- #

def run_experiment(seed, env_size='small'):
    dp  = DIST_PARAMS[env_size]
    rng = np.random.default_rng(seed)

    # Ground truth
    modvi_env = make_modvi_env(env_size, seed)
    modvi     = MODVI(modvi_env, dp['gamma'], dp['num_atoms'],
                      dp['v_mins'], dp['v_maxs'])
    modvi_dds = modvi.get_dds(num_iters=20)
    print(f'MODVI DDS size: {len(modvi_dds)}')

    # Online DIMOQ
    dimoq_env    = make_dimoq_env(env_size, seed)
    online_dimoq = DIMOQ(
        dimoq_env, dp['ref_point'], dp['gamma'],
        initial_epsilon=dp['initial_epsilon'],
        epsilon_decay=dp['epsilon_decay'],
        final_epsilon=dp['final_epsilon'],
        num_atoms=dp['num_atoms'], v_mins=dp['v_mins'], v_maxs=dp['v_maxs'],
        max_dists=dp['max_dists'], seed=seed, log=False,
    )
    online_dds = online_dimoq.train(
        num_episodes=3000, warmup_time=50000,
        learn_model=False, log_every=99999, action_eval='linear',
    )
    print(f'Online DDS size: {len(online_dds)}')

    # Score tables for behavior policies
    linear_table = make_score_table(online_dimoq, 'linear')
    scal01_table = make_scalarized_score_table(online_dimoq, alpha=0.1)
    scal09_table = make_scalarized_score_table(online_dimoq, alpha=0.9)

    flatten_fn = online_dimoq._flatten_state

    policies = {
        'random':   random_policy(online_dimoq.num_actions, rng),
        'eps_0.5':  epsilon_greedy_policy(linear_table, online_dimoq.num_actions, 0.5, rng),
        'eps_0.1':  epsilon_greedy_policy(linear_table, online_dimoq.num_actions, 0.1, rng),
        'scal_0.1': greedy_policy(scal01_table, rng),
        'scal_0.9': greedy_policy(scal09_table, rng),
    }

    sizes = [1_000, 5_000, 25_000, 100_000]

    results = {}
    for pol_name, pol_fn in policies.items():
        results[pol_name] = {}
        for n in sizes:
            print(f'  {pol_name}  N={n:7d} ...', end=' ', flush=True)
            dimoq_env.reset()
            dataset = collect_dataset(dimoq_env, n, pol_fn, flatten_fn)

            dimoq_env.reset()
            offline = OfflineDIMOQ(
                dimoq_env, dp['ref_point'], dp['gamma'],
                num_atoms=dp['num_atoms'], v_mins=dp['v_mins'], v_maxs=dp['v_maxs'],
                max_dists=dp['max_dists'], seed=seed, log=False,
            )
            offline.load_dataset(dataset)
            dds = offline.train_offline(num_iters=50, log_every=999)
            results[pol_name][n] = dds
            rec, _ = match(dds, modvi_dds)
            print(f'size={len(dds)}  recall={rec.mean():.2f}')

    return modvi_dds, online_dds, results


# --------------------------------------------------------------------------- #
# Figure 1: EV scatter per policy (N = 100k)
# --------------------------------------------------------------------------- #

POLICY_LABELS = {
    'random':   'Random (ε=1.0)',
    'eps_0.5':  'ε-greedy (ε=0.5)',
    'eps_0.1':  'ε-greedy (ε=0.1)',
    'scal_0.1': 'Scalarized (α=0.1\nprefers obj 2)',
    'scal_0.9': 'Scalarized (α=0.9\nprefers obj 1)',
}


def plot_ev_scatter(modvi_dds, online_dds, results, out_path, n=100_000):
    policies = list(POLICY_LABELS.keys())
    fig, axes = plt.subplots(1, len(policies), figsize=(4 * len(policies), 4),
                             sharey=True, sharex=True)

    m_evs = get_evs(modvi_dds)
    pf    = pareto_front_evs(m_evs)

    for ax, pol_name in zip(axes, policies):
        offline_dds          = results[pol_name][n]
        recovered, phantom   = match(offline_dds, modvi_dds)
        o_evs                = get_evs(offline_dds)
        recall               = recovered.mean()

        # Pareto front reference line
        if len(pf) > 1:
            ax.plot(pf[:, 0], pf[:, 1], '--', color='silver',
                    linewidth=1.2, zorder=1, label='MODVI PF')

        # Missed MODVI solutions
        missed_evs = m_evs[~recovered]
        if len(missed_evs):
            ax.scatter(missed_evs[:, 0], missed_evs[:, 1],
                       marker='x', s=80, color='crimson', linewidths=1.8,
                       zorder=3, label='Missed')

        # Recovered (MODVI solutions found by offline)
        rec_evs = m_evs[recovered]
        if len(rec_evs):
            ax.scatter(rec_evs[:, 0], rec_evs[:, 1],
                       marker='o', s=60, color='seagreen', zorder=4,
                       label='Recovered')

        # Phantom solutions (offline DDS not matching any MODVI solution)
        if phantom.any():
            ax.scatter(o_evs[phantom, 0], o_evs[phantom, 1],
                       marker='^', s=50, color='darkorange', alpha=0.7,
                       zorder=2, label='Phantom')

        ax.set_title(f'{POLICY_LABELS[pol_name]}\nRecall = {recall:.2f}',
                     fontsize=9)
        ax.set_xlabel('E[Return obj 0]')
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('E[Return obj 1]')

    # Shared legend
    handles = [
        mpatches.Patch(color='silver',     label='MODVI Pareto front'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='seagreen',
                   markersize=8,  label='Recovered'),
        plt.Line2D([0], [0], marker='x', color='crimson',
                   markersize=8, markeredgewidth=2, label='Missed'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='darkorange',
                   markersize=8,  label='Phantom (offline only)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.08), fontsize=8)

    fig.suptitle(f'DDS Recovery by Behavior Policy  (N={n:,} transitions)',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# --------------------------------------------------------------------------- #
# Figure 2: Distribution heatmaps (recovered vs missed, scal_0.1)
# --------------------------------------------------------------------------- #

def plot_heatmaps(modvi_dds, results, out_path,
                  policy='scal_0.1', n=100_000, num_each=3):
    offline_dds       = results[policy][n]
    recovered_mask, _ = match(offline_dds, modvi_dds)

    recovered_modvi = [d for d, r in zip(modvi_dds, recovered_mask) if r]
    missed_modvi    = [d for d, r in zip(modvi_dds, recovered_mask) if not r]

    # Pick evenly spaced examples
    def pick(lst, k):
        if len(lst) <= k:
            return lst
        idx = np.round(np.linspace(0, len(lst) - 1, k)).astype(int)
        return [lst[i] for i in idx]

    rec_show  = pick(recovered_modvi, num_each)
    miss_show = pick(missed_modvi,    num_each)

    n_cols  = max(len(rec_show), len(miss_show), 1)
    fig, axes = plt.subplots(2, n_cols,
                             figsize=(3.2 * n_cols, 6.5),
                             squeeze=False)

    dp = DIST_PARAMS['small']
    extent = [dp['v_mins'][0], dp['v_maxs'][0],
              dp['v_mins'][1], dp['v_maxs'][1]]

    def draw_dist(ax, dist, title, frame_color):
        im = ax.imshow(dist.dist.T, origin='lower', extent=extent,
                       aspect='auto', cmap='Blues', vmin=0)
        ev = dist.get_expected_value()
        ax.scatter([ev[0]], [ev[1]], marker='+', s=120,
                   color='crimson', linewidths=2, zorder=5)
        ax.set_title(title, fontsize=8, color=frame_color)
        ax.set_xlabel('Obj 0 return', fontsize=7)
        ax.set_ylabel('Obj 1 return', fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(frame_color)
            spine.set_linewidth(2)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for col, dist in enumerate(rec_show):
        draw_dist(axes[0][col], dist,
                  f'Recovered #{col+1}\nEV=({dist.get_expected_value()[0]:.1f}, '
                  f'{dist.get_expected_value()[1]:.1f})',
                  frame_color='seagreen')

    for col, dist in enumerate(miss_show):
        draw_dist(axes[1][col], dist,
                  f'Missed #{col+1}\nEV=({dist.get_expected_value()[0]:.1f}, '
                  f'{dist.get_expected_value()[1]:.1f})',
                  frame_color='crimson')

    # Hide unused axes
    for col in range(len(rec_show),  n_cols): axes[0][col].set_visible(False)
    for col in range(len(miss_show), n_cols): axes[1][col].set_visible(False)

    fig.suptitle(
        f'Return distributions: recovered vs missed\n'
        f'Policy = {POLICY_LABELS[policy]}  |  N={n:,}',
        fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# --------------------------------------------------------------------------- #
# Figure 3: Data-budget effect (eps_0.1 at four N values)
# --------------------------------------------------------------------------- #

def plot_budget_effect(modvi_dds, results, out_path, policy='eps_0.1'):
    sizes  = [1_000, 5_000, 25_000, 100_000]
    m_evs  = get_evs(modvi_dds)
    pf     = pareto_front_evs(m_evs)

    fig, axes = plt.subplots(1, len(sizes), figsize=(4 * len(sizes), 4),
                             sharey=True, sharex=True)

    for ax, n in zip(axes, sizes):
        offline_dds        = results[policy][n]
        recovered, phantom = match(offline_dds, modvi_dds)
        recall             = recovered.mean()

        if len(pf) > 1:
            ax.plot(pf[:, 0], pf[:, 1], '--', color='silver', linewidth=1.2)

        missed_evs = m_evs[~recovered]
        if len(missed_evs):
            ax.scatter(missed_evs[:, 0], missed_evs[:, 1],
                       marker='x', s=80, color='crimson', linewidths=1.8)

        rec_evs = m_evs[recovered]
        if len(rec_evs):
            ax.scatter(rec_evs[:, 0], rec_evs[:, 1],
                       marker='o', s=60, color='seagreen')

        o_evs = get_evs(offline_dds)
        if phantom.any():
            ax.scatter(o_evs[phantom, 0], o_evs[phantom, 1],
                       marker='^', s=50, color='darkorange', alpha=0.7)

        ax.set_title(f'N = {n:,}\nRecall = {recall:.2f}', fontsize=9)
        ax.set_xlabel('E[Return obj 0]')
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('E[Return obj 1]')
    fig.suptitle(
        f'Data-budget effect: {POLICY_LABELS[policy]}\n'
        '(data starvation heals with more data; bias does not)',
        fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--env',  default='small')
    p.add_argument('--out-dir', default=os.path.join(
        os.path.dirname(__file__), 'results', 'figures'))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Running experiment for seed={args.seed} ...')
    modvi_dds, online_dds, results = run_experiment(args.seed, args.env)

    plot_ev_scatter(
        modvi_dds, online_dds, results,
        out_path=os.path.join(args.out_dir, 'fig1_ev_scatter.png'),
    )
    plot_heatmaps(
        modvi_dds, results,
        out_path=os.path.join(args.out_dir, 'fig2_heatmaps.png'),
    )
    plot_budget_effect(
        modvi_dds, results,
        out_path=os.path.join(args.out_dir, 'fig3_budget_effect.png'),
    )
    print(f'\nAll figures saved to {args.out_dir}')


if __name__ == '__main__':
    main()
