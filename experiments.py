import argparse
import os
import time

import numpy as np

from algs.dimoq import DIMOQ
from algs.modvi import MODVI
from envs.random_momdp import RandomMOMDP
from envs.space_traders import SpaceTraders
from utils.data import save_dists, save_momdp, save_alg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='logs', help='The directory to save the logs.')
    parser.add_argument("--seed", type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help="The seed for random number generation.")
    parser.add_argument("--env", type=str, nargs='+', default=["space-traders"],
                        help="The environments to run experiments on.")
    parser.add_argument("--augment-env", action='store_true', default=True, help="Whether to augment the environment.")
    parser.add_argument("--alg", type=str, default='DIMOQ', help="The algorithm to use.")
    parser.add_argument("--num-episodes", type=int, default=400, help="The number of episodes to run.")
    parser.add_argument('--gamma', type=float, default=1., help='The discount factor.')
    parser.add_argument('--num-threads', type=int, default=1, help='The number of threads to use.')
    parser.add_argument('--log', action='store_true', default=True, help='Whether to log the results.')
    parser.add_argument('--log-every', type=int, default=100, help='The number of episodes between logging.')
    parser.add_argument('--save', action='store_true', default=True, help='Whether to save the results.')
    parser.add_argument('--warmup', type=int, default=50000, help='The number of warmup episodes.')
    args = parser.parse_args()
    return args


def dst_params():
    return {
        'ref_point': np.array([0, -25]),
        'initial_epsilon': 1.0,
        'epsilon_decay': 0.9975,
        'final_epsilon': 0.2,
        'num_atoms': (125, 21),
        'v_mins': (0., -20.),
        'v_maxs': (124., 0.),
        'project_name': "Distributional-Dominance",
        'experiment_name': "DIMOQ-DST"
    }


def space_traders_params():
    return {
        'ref_point': np.array([0, -30]),
        'initial_epsilon': 1.0,
        'epsilon_decay': 0.9975,
        'final_epsilon': 0.1,
        'num_atoms': (21, 23),
        'v_mins': (0., -22.),
        'v_maxs': (1., 0.),
        'project_name': "Distributional-Dominance",
        'experiment_name': "DIMOQ-ST"
    }


def random_momdp_params(size):
    if size == 'small':
        max_dists = 10
    elif size == 'medium':
        max_dists = 15
    elif size == 'large':
        max_dists = 20
    else:
        raise ValueError(f'Invalid size: {size}')
    return {
        'ref_point': np.array([0, 0]),
        'initial_epsilon': 1.0,
        'epsilon_decay': 0.9975,
        'final_epsilon': 0.1,
        'num_atoms': (21, 21),
        'v_mins': (0., 0.),
        'v_maxs': (20., 20.),
        'max_dists': max_dists,
        'project_name': "Distributional-Dominance",
        'experiment_name': f"DIMOQ-{size}MOMDP"
    }


def create_small_momdp(args, seed):
    num_states = 5
    num_objectives = 2
    num_actions = 2
    min_next_states = 1
    max_next_states = 2
    num_terminal_states = 1
    reward_min = np.zeros(num_objectives, dtype=np.float32)
    reward_max = np.ones(num_objectives) * 5
    start_state = 0
    max_timesteps = 3
    env = RandomMOMDP(num_states, num_objectives, num_actions, min_next_states, max_next_states, num_terminal_states,
                      reward_min, reward_max, reward_dist='discrete', start_state=start_state,
                      max_timesteps=max_timesteps, seed=seed, augment_state=args.augment_env)
    return env


def create_medium_momdp(args, seed):
    num_states = 10
    num_objectives = 2
    num_actions = 3
    min_next_states = 1
    max_next_states = 2
    num_terminal_states = 1
    reward_min = np.zeros(num_objectives, dtype=np.float32)
    reward_max = np.ones(num_objectives) * 5
    start_state = 0
    max_timesteps = 5
    env = RandomMOMDP(num_states, num_objectives, num_actions, min_next_states, max_next_states, num_terminal_states,
                      reward_min, reward_max, reward_dist='discrete', start_state=start_state,
                      max_timesteps=max_timesteps, seed=seed, augment_state=args.augment_env)
    return env


def create_large_momdp(args, seed):
    num_states = 15
    num_objectives = 2
    num_actions = 4
    min_next_states = 1
    max_next_states = 2
    num_terminal_states = 1
    reward_min = np.zeros(num_objectives, dtype=np.float32)
    reward_max = np.ones(num_objectives) * 5
    start_state = 0
    max_timesteps = 7
    env = RandomMOMDP(num_states, num_objectives, num_actions, min_next_states, max_next_states, num_terminal_states,
                      reward_min, reward_max, reward_dist='discrete', start_state=start_state,
                      max_timesteps=max_timesteps, seed=seed, augment_state=args.augment_env)
    return env


def create_larger_momdp(args, seed):
    num_states = 15
    num_objectives = 2
    num_actions = 3
    min_next_states = 1
    max_next_states = 3
    num_terminal_states = 1
    reward_min = np.zeros(num_objectives, dtype=np.float32)
    reward_max = np.ones(num_objectives) * 5
    start_state = 0
    max_timesteps = 7
    env = RandomMOMDP(num_states, num_objectives, num_actions, min_next_states, max_next_states, num_terminal_states,
                      reward_min, reward_max, reward_dist='discrete', start_state=start_state,
                      max_timesteps=max_timesteps, seed=seed, augment_state=args.augment_env)
    return env


def create_largest_momdp(args, seed):
    num_states = 20
    num_objectives = 2
    num_actions = 4
    min_next_states = 1
    max_next_states = 4
    num_terminal_states = 2
    reward_min = np.zeros(num_objectives, dtype=np.float32)
    reward_max = np.ones(num_objectives) * 5
    start_state = 0
    max_timesteps = 10
    env = RandomMOMDP(num_states, num_objectives, num_actions, min_next_states, max_next_states, num_terminal_states,
                      reward_min, reward_max, reward_dist='discrete', start_state=start_state,
                      max_timesteps=max_timesteps, seed=seed, augment_state=args.augment_env)
    return env


def run_dimoq(env, args, params, seed):
    print(f'Running DIMOQ')
    dimoq = DIMOQ(env,
                  params['ref_point'],
                  args.gamma,
                  params['initial_epsilon'],
                  params['epsilon_decay'],
                  params['final_epsilon'],
                  params['num_atoms'],
                  params['v_mins'],
                  params['v_maxs'],
                  seed=seed,
                  project_name=params['project_name'],
                  experiment_name=params['experiment_name'],
                  log=args.log,
                  num_threads=args.num_threads)
    dds = dimoq.train(num_episodes=args.num_episodes, log_every=args.log_every, warmup_time=args.warmup)
    return dimoq, dds


def run_modvi(env, args, params):
    print(f'Running MODVI')
    modvi = MODVI(env,
                  args.gamma,
                  params['num_atoms'],
                  params['v_mins'],
                  params['v_maxs'])
    dds = modvi.get_dds(num_iters=args.num_episodes)
    return modvi, dds


if __name__ == "__main__":
    args = parse_args()

    for env_name in args.env:
        print(f'Running on {env_name}')

        for seed in args.seed:
            if "dst" == env_name:
                import mo_gym
                from mo_gym.deep_sea_treasure.deep_sea_treasure import CONCAVE_MAP
                env = mo_gym.make('deep-sea-treasure-v0', dst_map=CONCAVE_MAP)
                params = dst_params()
            elif "space-traders" == env_name:
                env = SpaceTraders(seed=seed)
                params = space_traders_params()
            elif "small" == env_name:
                env = create_small_momdp(args, seed)
                params = random_momdp_params(env_name)
            elif "medium" == env_name:
                env = create_medium_momdp(args, seed)
                params = random_momdp_params(env_name)
            elif "large" == env_name:
                env = create_large_momdp(args, seed)
                params = random_momdp_params(env_name)
            else:
                raise ValueError(f'Unknown environment {args.env}')

            start = time.time()
            if args.alg == 'DIMOQ':
                alg, dds = run_dimoq(env, args, params, seed)
            elif args.alg == 'MODVI':
                alg, dds = run_modvi(env, args, params)
            else:
                raise ValueError(f'Unknown algorithm {args.alg}')
            end = time.time()
            duration = end - start
            print(f'Finished in {duration} seconds')

            if args.save:
                print(f'Saving {args.alg} results for {env_name} to {args.log_dir}')
                env_dir = os.path.join(args.log_dir, env_name, str(seed))
                alg_dir = os.path.join(env_dir, args.alg)
                save_momdp(env, env_dir, env_name)
                save_alg(alg, len(dds), duration, alg_dir, args.alg)
                save_dists(dds, alg_dir)
