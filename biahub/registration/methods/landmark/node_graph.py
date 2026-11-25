"""Graph matching of two point sets, as an estimator: the landmark methods' shared part.

`NodeGraphEstimator` composes a `NodeDetector` per channel (bead peaks, nuclei; label
centroids later), the graph matcher (`match_nodes`: hungarian, spectral or descriptor
matching, then the match filters) and a fit; each pass re-detects on the moving volume
warped by the current transform. `bead_matching` and `nuclei_matching` build it with
their detectors.
"""

from __future__ import annotations

import click
import numpy as np

from numpy.typing import ArrayLike

from biahub.core.nodes.costs import MatchingTooLargeError
from biahub.core.nodes.graph import Graph
from biahub.core.nodes.matching import GraphMatcher
from biahub.core.transform import Transform
from biahub.registration.estimators import (
    EstimationError,
    EstimatorBase,
    NodeDetector,
    NodeMatcher,
    ScoreFn,
    fit_pairs,
)
from biahub.settings import (
    GraphMatchSettings,
)


def match_nodes(
    mov_peaks: ArrayLike,
    ref_peaks: ArrayLike,
    matching: GraphMatchSettings,
    verbose: bool = False,
) -> ArrayLike:
    """
    Find bead correspondences between moving and reference peak sets.

    Supports three matching algorithms:
    - "hungarian": Builds k-NN graphs for both peak sets, computes a cost matrix
      based on position distance and edge consistency, then solves the assignment
      problem with the Hungarian algorithm.
    - "match_descriptor": Uses scikit-image's descriptor matching on peak positions.
    - "spectral": Matches from the pairwise distances of the raw coordinates
      (`spectral_match_settings`).

    After matching, applies geometric consistency filters (distance quantiles,
    direction threshold, angle threshold) to remove outliers.

    Parameters
    ----------
    mov_peaks : ArrayLike
        (N, D) array of moving peak coordinates (D = 2 or 3).
    ref_peaks : ArrayLike
        (M, D) array of reference peak coordinates.
    matching : GraphMatchSettings
        Settings controlling the matching algorithm, graph construction,
        cost matrix weights, and post-match filtering.
    verbose : bool
        If True, prints matching settings and match count.

    Returns
    -------
    ArrayLike
        (K, 2) array of matched index pairs [mov_idx, ref_idx].
    """
    if verbose:
        click.echo(f"Matching nodes with settings: {matching}")

    if matching.algorithm == "match_descriptor":
        mov_graph = Graph.from_nodes(mov_peaks)
        ref_graph = Graph.from_nodes(ref_peaks)

        match_descriptor_settings = matching.match_descriptor_settings
        matcher = GraphMatcher(
            algorithm="descriptor",
            cross_check=match_descriptor_settings.cross_check,
            max_ratio=match_descriptor_settings.max_ratio,
            metric=match_descriptor_settings.distance_metric,
            verbose=verbose,
        )

        matches = matcher.match(mov_graph, ref_graph)

    elif matching.algorithm == "hungarian":
        hungarian_match_settings = matching.hungarian_match_settings
        mov_graph = Graph.from_nodes(
            mov_peaks, mode="knn", k=hungarian_match_settings.edge_graph_settings.k
        )
        ref_graph = Graph.from_nodes(
            ref_peaks, mode="knn", k=hungarian_match_settings.edge_graph_settings.k
        )

        matcher = GraphMatcher(
            algorithm="hungarian",
            weights=hungarian_match_settings.cost_matrix_settings.weights,
            cost_threshold=hungarian_match_settings.cost_threshold,
            cross_check=hungarian_match_settings.cross_check,
            max_ratio=hungarian_match_settings.max_ratio,
            verbose=verbose,
        )

        matches = matcher.match(mov_graph, ref_graph)

    elif matching.algorithm == "spectral":
        spectral_match_settings = matching.spectral_match_settings
        # Spectral matching works from pairwise distances of the raw coordinates; the
        # graph is only a node container here.
        mov_graph = Graph.from_nodes(mov_peaks)
        ref_graph = Graph.from_nodes(ref_peaks)
        matcher = GraphMatcher(
            algorithm="spectral",
            spectral_sigma=spectral_match_settings.sigma,
            spectral_rel_cut=spectral_match_settings.rel_cut,
            spectral_max_iter=spectral_match_settings.max_iter,
            verbose=verbose,
        )
        matches = matcher.match(mov_graph, ref_graph)
    else:
        raise ValueError(f"Unknown matching algorithm: {matching.algorithm}")

    matches = matcher.filter_matches(
        matches,
        mov_graph,
        ref_graph,
        angle_threshold=matching.filter_matches_settings.angle_threshold,
        min_distance_quantile=matching.filter_matches_settings.min_distance_quantile,
        max_distance_quantile=matching.filter_matches_settings.max_distance_quantile,
        direction_threshold=matching.filter_matches_settings.direction_threshold,
    )

    if verbose:
        click.echo(f"Total of matches: {len(matches)}")

    return matches


def transform_from_matches(
    matches: ArrayLike,
    mov_peaks: ArrayLike,
    ref_peaks: ArrayLike,
    transform_type: str,
    verbose: bool = False,
) -> tuple[Transform, Transform]:
    """Return the (forward, inverse) transform fitted to matched peaks (`Transform.fit`, skimage).

    `matches` are (moving, reference) index pairs into the (N, D) peaks; the transform
    type is `transform_type`. Raises ValueError when the fit fails.
    """
    if verbose:
        click.echo(f"Estimating a {transform_type} transform from {len(matches)} matches")
    fwd_transform = Transform.fit(
        np.asarray(mov_peaks)[matches[:, 0]],
        np.asarray(ref_peaks)[matches[:, 1]],
        transform_type,
        solver="skimage",
    )
    return fwd_transform, fwd_transform.invert()


class NodeGraphEstimator(EstimatorBase):
    """TransformEstimator over matched point correspondences.

    Composes a `NodeDetector` per channel (beads, nuclei; segmentation centroids later)
    with graph matching and a fit: the landmark-based methods.

    One pass warps `mov` by the current guess (`Transform.apply`), detects nodes in the
    warped frame, matches, fits the residual correction and composes it back
    (`correction @ initial_transform`: the initial transform first, then the
    correction). Nodes are detected in the warped frame, so each pass sees a
    better-aligned image than the last; `iterations > 1` with a `score_fn` repeats the
    pass from the previous result and returns the best-scoring transform (never a later,
    worse one). Without a `score_fn` the last pass is returned.

    As in the legacy pipeline, a pass keeps its input when the refinement scores lower
    (ties go to the refinement), and a score of exactly 1.0 ends the passes early.
    """

    accepts_competitor = True

    def __init__(
        self,
        mov_detector: NodeDetector,
        ref_detector: NodeDetector,
        matching: GraphMatchSettings,
        transform_type: str,
        iterations: int = 1,
        score_fn: ScoreFn | None = None,
        matcher: NodeMatcher | None = None,
        fit=None,
    ):
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        self.mov_detector = mov_detector
        self.ref_detector = ref_detector
        self.matching = matching
        self.transform_type = transform_type
        self.iterations = iterations
        self.score_fn = score_fn
        # (matches, mov_nodes, ref_nodes) -> the correction; default: the transform
        # type's fit of the matched pairs (fit_pairs), as beads use
        self.fit = fit
        self.matcher = matcher or (
            lambda mov_nodes, ref_nodes: match_nodes(mov_nodes, ref_nodes, matching)
        )

    def estimate(
        self,
        mov: ArrayLike,
        ref: ArrayLike,
        initial_transform: Transform | None = None,
        competitor: Transform | None = None,
    ) -> Transform:
        """Refine `initial_transform` over the configured passes; return the best-scoring one.

        `competitor`, when given, is refined too on the first pass and replaces the
        initial transform's result only if it scores strictly higher; later passes
        continue from the winner. This is the legacy `user_transform` competition, used
        by propagation to let the run's initial transform compete with the previous
        timepoint's result.
        """
        mov = np.asarray(mov)
        ref = np.asarray(ref)
        if self.score_fn is None:
            if competitor is not None:
                raise ValueError("a competitor initial transform needs a score_fn to compete")
            current = initial_transform
            for _ in range(self.iterations):
                current = self._single_pass(mov, ref, current)
            return current
        current, current_score = initial_transform, None
        best: Transform | None = None
        best_score = -np.inf
        for i in range(self.iterations):
            candidates, error = [], None
            try:
                candidates.append(self._pass_keeping_input(mov, ref, current, current_score))
            except EstimationError as e:
                error = e
            if i == 0 and competitor is not None:
                try:
                    candidates.append(self._pass_keeping_input(mov, ref, competitor, None))
                except EstimationError:
                    pass
            if not candidates:
                # A later pass that cannot match (e.g. the previous pass drifted) must not
                # throw away an earlier usable result.
                if best is not None:
                    break
                raise error
            current, current_score = candidates[0]
            for transform, score in candidates[1:]:
                if score > current_score:  # ties: the initial transform's result
                    current, current_score = transform, score
            if current_score > best_score:
                best, best_score = current, current_score
            if current_score == 1.0:  # legacy stops on exactly 1 (overlap can exceed 1)
                break
        if best is None or not np.isfinite(best_score):
            raise EstimationError(
                f"no finite score in {self.iterations} iteration(s): nodes not detectable "
                "after warping"
            )
        return best

    def _pass_keeping_input(
        self, mov, ref, start: Transform | None, start_score: float | None
    ) -> tuple[Transform, float]:
        """One pass from `start`; keep `start` when the refinement scores lower (ties: refinement)."""
        refined = self._single_pass(mov, ref, start)
        refined_score = self._finite(self.score_fn(refined, mov, ref))
        if start is None:
            start = Transform.identity(mov.ndim)
        if start_score is None:
            start_score = self._finite(self.score_fn(start, mov, ref))
        if refined_score >= start_score:
            return refined, refined_score
        return start, start_score

    @staticmethod
    def _finite(score: float | None) -> float:
        return float(score) if score is not None and np.isfinite(score) else -np.inf

    def _single_pass(
        self, mov: np.ndarray, ref: np.ndarray, initial_transform: Transform | None
    ) -> Transform:
        mov_for_detection = (
            initial_transform.apply(mov, reference=ref)
            if initial_transform is not None
            else mov
        )
        if initial_transform is not None and self.mov_detector.missing_as_nan:
            # the warp's padding is no data, not dark: for a detector that leaves NaN out
            mov_for_detection = np.asarray(mov_for_detection, dtype=np.float32).copy()
            mov_for_detection[~initial_transform.coverage(mov.shape, ref.shape)] = np.nan
        mov_nodes = np.asarray(self.mov_detector.detect(mov_for_detection))
        ref_nodes = np.asarray(self.ref_detector.detect(ref))
        if len(mov_nodes) < 3 or len(ref_nodes) < 3:
            raise EstimationError(
                f"too few nodes to fit a transform: {len(mov_nodes)} moving, "
                f"{len(ref_nodes)} reference (need >= 3 each)"
            )
        try:
            matches = np.asarray(self.matcher(mov_nodes, ref_nodes))
        except MatchingTooLargeError as e:  # this matcher can't run here; others may
            raise EstimationError(str(e)) from e
        if len(matches) < 3:
            raise EstimationError(
                f"too few matches to fit a transform: {len(matches)} from "
                f"{len(mov_nodes)} x {len(ref_nodes)} nodes (need >= 3)"
            )
        if self.fit is not None:
            correction = self.fit(matches, mov_nodes, ref_nodes)
        else:
            correction = fit_pairs(
                mov_nodes[matches[:, 0]],
                ref_nodes[matches[:, 1]],
                self.transform_type,
                solver="skimage",  # the beads method's fit (see Transform.fit)
            )
        if not np.all(np.isfinite(correction.matrix)):
            raise EstimationError(
                f"degenerate fit from {len(matches)} matches (non-finite matrix)"
            )
        return correction @ initial_transform if initial_transform is not None else correction
