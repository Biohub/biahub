"""Point-set registration: match and fit together, from positions alone (no edges).

The correspondences are found from where the points lie under the current transform and
re-fitted in a loop, so these start from an initial transform; they need no distinctive
local structure.
"""

from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from biahub.core.transform import Transform


def _densest_votes(points, ref_points, ref_tree, capture_radius, cluster_radius, min_votes):
    """Let every point vote for its displacement to each reference point within reach.

    Returns the densest `cluster_radius`-ball of votes: its votes, their (point index,
    point index) pairs and its size -- or None when fewer than `min_votes` were cast.
    """
    votes, pairs = [], []
    for i, p in enumerate(points):
        for j in ref_tree.query_ball_point(p, r=capture_radius):
            votes.append(ref_points[j] - p)
            pairs.append((i, j))
    if len(votes) < min_votes:
        return None
    votes = np.asarray(votes)
    vote_tree = cKDTree(votes)
    counts = np.array([len(vote_tree.query_ball_point(v, r=cluster_radius)) for v in votes])
    members = vote_tree.query_ball_point(votes[np.argmax(counts)], r=cluster_radius)
    return votes[members], np.asarray(pairs)[members], len(members)


def vote_icp_register(
    mov_points: ArrayLike,
    ref_points: ArrayLike,
    initial_transform: Transform,
    initial_capture_radius: float = 80.0,
    min_capture_radius: float = 10.0,
    radius_decay: float = 0.5,
    cluster_radius: float = 10.0,
    min_votes: int = 3,
    max_iterations: int = 20,
    convergence_translation: float = 0.5,
    transform_type: str = "affine",
) -> tuple[Transform | None, dict]:
    """Register two point clouds by iterated displacement voting.

    An ICP whose correspondence stage is a vote: each iteration maps the moving points
    into reference space with the current transform, lets every mapped point vote for
    its displacement to each reference point within the capture radius, keeps the densest
    `cluster_radius`-ball of votes as the inlier correspondences and re-fits on them (a
    `transform_type` fit with >= D + 3 pairs, a pure translation otherwise). The capture radius
    shrinks geometrically, floored at `min_capture_radius`, so early iterations have the
    reach to recover a large residual and late ones only see unambiguous neighbours.

    Unlike one-to-one matching the vote never commits a point to a single partner, so a
    split or fuzzy detection just casts extra votes outside the winning cluster.

    `initial_transform` and the returned transform are forward (moving points onto the
    reference), and so is each fit: the inliers' residuals are minimized in reference
    space, where the result is judged. Returns `(None, info)` when voting abstained before any fit; `info` holds
    iterations, converged, n_inliers and the final capture radius.
    """
    mov_points = np.asarray(mov_points, dtype=float)
    ref_points = np.asarray(ref_points, dtype=float)
    transform = initial_transform
    ref_tree = cKDTree(ref_points)
    ndim = mov_points.shape[1]
    mapped = transform.apply_points(mov_points)
    radius = initial_capture_radius
    info = {"iterations": 0, "converged": False, "n_inliers": 0, "capture_radius": radius}
    fitted_once = False
    for iteration in range(1, max_iterations + 1):
        cluster = _densest_votes(
            mapped, ref_points, ref_tree, radius, cluster_radius, min_votes
        )
        if cluster is None:
            break
        cluster_votes, pairs, n_inliers = cluster
        inlier_mov, inlier_ref = pairs[:, 0], pairs[:, 1]
        # A fit needs a margin over its degrees of freedom; with fewer inliers the
        # translation update is the only fit that cannot hallucinate.
        if n_inliers >= ndim + 3:
            new = Transform.fit(mov_points[inlier_mov], ref_points[inlier_ref], transform_type)
        else:
            # the mapped points move by the votes' mean displacement
            new = Transform.from_translation(cluster_votes.mean(axis=0)) @ transform
        new_mapped = new.apply_points(mov_points)
        step = float(np.mean(np.linalg.norm(new_mapped - mapped, axis=1)))
        transform, mapped, fitted_once = new, new_mapped, True
        info.update(iterations=iteration, n_inliers=n_inliers, capture_radius=radius)
        radius = max(min_capture_radius, radius * radius_decay)
        if step < convergence_translation:
            info["converged"] = True
            break
    if not fitted_once:
        return None, info
    return transform, info


def vote_drift(
    mov_points: ArrayLike,
    ref_points: ArrayLike,
    capture_radius: float,
    cluster_radius: float,
    min_votes: int,
) -> tuple[np.ndarray | None, list[tuple[np.ndarray, np.ndarray]], int]:
    """One round of displacement voting between already-aligned point clouds.

    Returns the mean displacement of the densest vote cluster, that cluster's
    (moving point, reference point) pairs, and the cluster size -- or `(None, [], 0)`
    when fewer than `min_votes` votes were cast.
    """
    mov_points = np.asarray(mov_points, dtype=float)
    ref_points = np.asarray(ref_points, dtype=float)
    cluster = _densest_votes(
        mov_points, ref_points, cKDTree(ref_points), capture_radius, cluster_radius, min_votes
    )
    if cluster is None:
        return None, [], 0
    cluster_votes, pairs, size = cluster
    return cluster_votes.mean(axis=0), [(mov_points[i], ref_points[j]) for i, j in pairs], size
