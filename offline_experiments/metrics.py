import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils.dist_metrics import dist_hypervolume


def one_sided_hausdorff_w2(dds_a, dds_b):
    """Max over A of the min W2 distance to any distribution in B."""
    if not dds_a or not dds_b:
        return float('inf')
    return max(
        min(d_a.wasserstein_distance(d_b) for d_b in dds_b)
        for d_a in dds_a
    )


def hausdorff_w2(dds_a, dds_b):
    """Symmetric Hausdorff distance between two DDSs under W2."""
    return max(
        one_sided_hausdorff_w2(dds_a, dds_b),
        one_sided_hausdorff_w2(dds_b, dds_a),
    )


def precision_recall(learned, reference, threshold):
    """
    Precision: fraction of learned distributions that are within `threshold` W2
               of at least one distribution in reference.
    Recall:    fraction of reference distributions matched by something in learned.

    Note: W2 here is in atom-index units (not return-value units).
    """
    if not learned or not reference:
        return 0.0, 0.0

    matched_learned = sum(
        any(d_l.wasserstein_distance(d_r) <= threshold for d_r in reference)
        for d_l in learned
    )
    matched_ref = sum(
        any(d_r.wasserstein_distance(d_l) <= threshold for d_l in learned)
        for d_r in reference
    )
    return matched_learned / len(learned), matched_ref / len(reference)


def compute_metrics(dds, reference_dds, ref_point, pr_threshold=1.0):
    """Compute size, hypervolume, Hausdorff-W2, precision, and recall.

    Args:
        dds:            the DDS to evaluate.
        reference_dds:  the ground-truth DDS (MODVI output).
        ref_point:      reference point for hypervolume.
        pr_threshold:   W2 threshold for counting a distribution as "matched".
    """
    hv = dist_hypervolume(ref_point, dds) if dds else 0.0
    ref_hv = dist_hypervolume(ref_point, reference_dds) if reference_dds else 0.0
    h_dist = hausdorff_w2(dds, reference_dds)
    prec, rec = precision_recall(dds, reference_dds, pr_threshold)

    return {
        'size': len(dds),
        'hypervolume': float(hv),
        'hv_ratio': float(hv / ref_hv) if ref_hv > 0 else 0.0,
        'hausdorff_w2': float(h_dist),
        'precision': float(prec),
        'recall': float(rec),
    }


def print_metrics(label, m):
    print(
        f'  {label:8s}  size={m["size"]:3d}  '
        f'hv={m["hypervolume"]:.4f}  hv_ratio={m["hv_ratio"]:.3f}  '
        f'hausdorff_w2={m["hausdorff_w2"]:.4f}  '
        f'P={m["precision"]:.3f}  R={m["recall"]:.3f}'
    )
