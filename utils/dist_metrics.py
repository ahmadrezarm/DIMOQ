import numpy as np
from pymoo.indicators.hv import HV
from sklearn.cluster import AgglomerativeClustering


def dist_hypervolume(ref_point, dists):
    """Compute the hypervolume of the expected values from a list of distributions.

    Args:
        ref_point (List[float]): The reference point.
        dists (List[MultivariateCategoricalDistribution]): The distributions.

    Returns:
        float: The hypervolume.
    """
    points = [dist.get_expected_value() for dist in dists]
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)


def max_inter_distance(dist_lst):
    """Compute the maximum distance between distributions in a list.

    Args:
        dist_lst (List[MultivariateCategoricalDistribution]): The distributions.

    Returns:
        float: The maximum distance.
    """
    return np.max(compute_distance_matrix(dist_lst))


def linear_utility(dist_lst):
    """Compute the linear utility of a list of distributions.

    Args:
        dist_lst (List[MultivariateCategoricalDistribution]): The distributions.

    Returns:
        float: The linear utility.
    """
    return sum(sum(dist.get_expected_value()) for dist in dist_lst)


def compute_distance_matrix(distributions, distance_metric='wasserstein'):
    """Compute the Jensen-Shannon distances of a list of distributions.

    Args:
        dists (List[MultivariateCategoricalDistribution]): The distributions.

    Returns:
        ndarray: A matrix of distances. Conceptually this is a graph where each node is a distribution.
    """
    distance_matrix = np.zeros((len(distributions), len(distributions)))
    for i, dist1 in enumerate(distributions):
        for j, dist2 in enumerate(distributions[i + 1:], i + 1):
            if distance_metric == 'wasserstein':
                distance = dist1.wasserstein_distance(dist2)
            elif distance_metric == 'jensen-shannon':
                distance = dist1.js_distance(dist2)
            else:
                raise Exception('Distance metric not implemented')
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


def get_best(dist_lst, distance_metric='jensen-shannon', max_dists=10, rng=None):
    """Get the best distributions from a list of distributions.

    Args:
        dist_lst (List[MultivariateCategoricalDistribution]): The distributions.
        max_dists (int): The maximum number of distributions to return.

    Returns:
        List[MultivariateCategoricalDistribution]: The best distributions.
    """
    if len(dist_lst) <= max_dists:
        return dist_lst

    rng = rng if rng is not None else np.random.default_rng()
    dist_matrix = compute_distance_matrix(dist_lst, distance_metric=distance_metric)
    clustering = AgglomerativeClustering(n_clusters=max_dists, metric='precomputed', linkage='average')
    clustering.fit(dist_matrix)
    keep = []
    for cluster in range(max_dists):
        cluster_nodes = np.flatnonzero(clustering.labels_ == cluster)
        if len(cluster_nodes) == 1:
            keep.append(cluster_nodes[0])
        else:
            keep.append(rng.choice(cluster_nodes))

    return [dist_lst[i] for i in keep]
