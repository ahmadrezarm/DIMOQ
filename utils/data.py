import json
import os

import numpy as np
import pandas as pd


def save_dists(dists, dir_path):
    """Save a list of distributions to a file.

    Args:
        dists (list): A list of distributions.
        dir_path (str): The name of the directory.
    """
    os.makedirs(dir_path, exist_ok=True)
    for i, dist in enumerate(dists):
        dist.save(dir_path, file_name=f'dist_{i}')


def load_dists(dir_path, dist_class):
    """Load a list of distributions from a file.

    Args:
        dir_path (str): The name of the directory.

    Returns:
        list: A list of distributions.
    """
    dists = []

    for file_name in os.listdir(dir_path):
        if file_name.startswith('dist_'):
            with open(os.path.join(dir_path, file_name), 'r') as f:
                dist_data = json.load(f)

            num_atoms = np.array(dist_data['num_atoms'])
            v_mins = np.array(dist_data['v_mins'])
            v_maxs = np.array(dist_data['v_maxs'])
            name = dist_data['name']
            vecs = dist_data['dist']['vecs']
            probs = dist_data['dist']['probs']
            dist = dist_class(num_atoms, v_mins, v_maxs, name=name, vecs=vecs, probs=probs)
            dists.append(dist)

    return dists


def save_momdp(momdp, dir_path, file_name):
    os.makedirs(dir_path, exist_ok=True)
    file_name += '.json'
    config = momdp.get_config()
    with open(os.path.join(dir_path, file_name), 'w') as f:
        json.dump(config, f)


def save_alg(alg, dds_size, duration, dir_path, file_name):
    os.makedirs(dir_path, exist_ok=True)
    file_name += '.json'
    config = alg.get_config()
    config['dds_size'] = dds_size
    config['duration'] = duration

    with open(os.path.join(dir_path, file_name), 'w') as f:
        json.dump(config, f)


def save_pruning_results(results, durations, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.join(dir_path, 'pruning_results.csv'), index=False)
    with open(os.path.join(dir_path, 'pruning_durations.json'), 'w') as f:
        json.dump(durations, f)
