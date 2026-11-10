"""
Beads-based registration module.

Provides functions for registering volumetric imaging data by detecting fluorescent
bead landmarks in moving and reference channels, matching them using graph-based
algorithms, and estimating affine transformations.

Pipeline overview
-----------------
1. **Peak detection** (`peaks_from_beads`): Detect bead positions in both channels.
2. **Matching, fit and refinement** (`node_graph`, shared with nuclei: `match_nodes`,
   `transform_from_matches`, `NodeGraphEstimator`), built on bead peaks by
   `bead_node_graph`: graph matching (Hungarian, spectral or descriptor) with geometric
   consistency filtering, a fit, and passes that re-detect on the warped moving volume.

Key conventions
---------------
- Coordinates are in ZYX order for 3D data.
- "mov" / "moving" refers to the source channel being aligned.
- "ref" / "reference" refers to the fixed target channel.
- Transforms map from moving space to reference space (forward direction).
"""

from __future__ import annotations

from pathlib import Path

import click
import dask.array as da
import numpy as np

from iohub import open_ome_zarr
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from biahub.characterize_psf import detect_peaks
from biahub.core.nodes.point_set import vote_drift, vote_icp_register
from biahub.core.transform import Transform
from biahub.registration.estimators import (
    ChainedEstimator,
    CompetingEstimator,
    EstimationError,
    EstimatorBase,
    NodeDetector,
    ScoreFn,
    TransformEstimator,
    fit_pairs,
)
from biahub.registration.methods.landmark.node_graph import (
    NodeGraphEstimator,
)
from biahub.registration.utils import Remembered, transform_key
from biahub.settings import (
    BeadsMatchSettings,
    DetectPeaksSettings,
    InitialTransformCorrectionSettings,
    VoteIcpSettings,
)


def overlap_score(
    mov_peaks: ArrayLike,
    ref_peaks: ArrayLike,
    radius: int = 6,
    verbose: bool = False,
) -> float:
    """
    Compute the overlap fraction between two sets of bead peaks.

    For each reference peak, checks whether any moving peak falls within a
    spherical neighborhood of the given radius (using a KDTree). The score is
    the fraction of reference peaks that have at least one nearby moving peak,
    normalized by the smaller peak set size.

    Parameters
    ----------
    mov_peaks : ArrayLike
        (N_mov, D) array of moving bead coordinates (z, y, x).
    ref_peaks : ArrayLike
        (N_ref, D) array of reference bead coordinates (z, y, x).
    radius : int
        Spherical neighborhood radius in voxels for overlap counting.
    verbose : bool
        If True, prints peak counts and overlap statistics.

    Returns
    -------
    float
        Reference peaks with a moving peak within `radius`, over the smaller peak count:
        usually in [0, 1], but above 1 when the reference has more peaks than the moving
        volume (e.g. 3.0 for 1 moving and 3 reference peaks). np.nan if either set is empty.
    """
    if len(mov_peaks) == 0 or len(ref_peaks) == 0:
        click.echo("No peaks found, returning nan metrics")
        return np.nan

    # ---- Overlap counting using KDTree ----
    mov_tree = cKDTree(mov_peaks)

    ref_peaks_mask = np.zeros(len(ref_peaks), dtype=bool)
    mov_peaks_mask = np.zeros(len(mov_peaks), dtype=bool)

    for i, p in enumerate(ref_peaks):
        idx = mov_tree.query_ball_point(p, r=radius)
        if idx:
            ref_peaks_mask[i] = True
            mov_peaks_mask[idx] = True

    peaks_overlap_count = int(ref_peaks_mask.sum())

    # ---- Overlap fraction ----
    peaks_overlap_fraction = peaks_overlap_count / max(min(len(mov_peaks), len(ref_peaks)), 1)

    if verbose:
        click.echo(f"Mov peaks: {len(mov_peaks)}")
        click.echo(f"Ref peaks: {len(ref_peaks)}")
        click.echo(f"Peaks overlap count: {peaks_overlap_count}")
        click.echo(f"Peaks overlap fraction: {peaks_overlap_fraction}")

    return peaks_overlap_fraction


def peaks_from_beads(
    mov: da.Array,
    ref: da.Array,
    mov_peaks_settings: DetectPeaksSettings,
    ref_peaks_settings: DetectPeaksSettings,
    verbose: bool = False,
    mask_path: Path = None,
) -> tuple[ArrayLike, ArrayLike] | None:
    """
    Detect peaks in moving and reference channels using the detect_peaks function.

    Parameters
    ----------
    mov : da.Array
        (Z, Y, X) array of the moving channel (Dask array).
    ref : da.Array
        (Z, Y, X) array of the reference channel (Dask array).
    mov_peaks_settings : DetectPeaksSettings
        Settings for the moving peaks.
    ref_peaks_settings : DetectPeaksSettings
        Settings for the reference peaks.
    verbose : bool
        If True, prints detailed logs during the process.
    mask_path : Path
        Path to the mask file.

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        Tuple of (mov_peaks, ref_peaks).
    """
    if verbose:
        click.echo("Detecting beads in moving dataset")
    # TODO: detecte peaks in the zyx space, use skimage.feature.peak_local_max for 2D
    mov_peaks = detect_peaks(
        mov,
        block_size=mov_peaks_settings.block_size,
        threshold_abs=mov_peaks_settings.threshold_abs,
        nms_distance=mov_peaks_settings.nms_distance,
        min_distance=mov_peaks_settings.min_distance,
        verbose=verbose,
    )
    if verbose:
        click.echo("Detecting beads in reference dataset")
    # TODO: detecte peaks in the zyx space, use skimage.feature.peak_local_max for 2D
    ref_peaks = detect_peaks(
        ref,
        block_size=ref_peaks_settings.block_size,
        threshold_abs=ref_peaks_settings.threshold_abs,
        nms_distance=ref_peaks_settings.nms_distance,
        min_distance=ref_peaks_settings.min_distance,
        verbose=verbose,
    )
    if verbose:
        click.echo(f"Total of peaks in moving dataset: {len(mov_peaks)}")
        click.echo(f"Total of peaks in reference dataset: {len(ref_peaks)}")

    if len(mov_peaks) < 2 or len(ref_peaks) < 2:
        click.echo("Not enough beads detected")
        return
    if mask_path is not None:
        click.echo("Filtering peaks with mask")
        with open_ome_zarr(mask_path) as mask_ds:
            mask_load = np.asarray(mask_ds.data[0, 0])

        # filter the peaks with the mask
        # Keep only peaks whose (y, x) column is clean across all Z slices
        ref_peaks_filtered = []
        for peak in ref_peaks:
            z, y, x = peak.astype(int)
            if (
                0 <= y < mask_load.shape[1]
                and 0 <= x < mask_load.shape[2]
                and not mask_load[:, y, x].any()  # True if all Z are clean at (y, x)
            ):
                ref_peaks_filtered.append(peak)
        ref_peaks = np.array(ref_peaks_filtered)
    return mov_peaks, ref_peaks


# How the evaluation pairs this method's landmarks: (radius in µm, axes it is measured on).
# Beads are points: a partner within 1 µm in 3D.
POINT_MATCHING = (1.0, "zyx")


def detect_beads(
    volume: ArrayLike,
    voxel_um: tuple,
    threshold_abs: float,
    block_size: tuple = (8, 8, 8),
    nms_distance: int = 16,
    refine_halfwidth: tuple = (2, 4, 4),
) -> np.ndarray:
    """Return bead centres (N x 3 voxels, z y x) to a fraction of a voxel: refined peaks.

    Peaks sit on whole voxels; each is refined to the intensity centroid of a small box
    around it (background-subtracted), which a residual of a voxel or less needs.
    """
    vol = np.asarray(volume, dtype=np.float32)
    peaks = np.asarray(
        detect_peaks(
            vol,
            block_size=block_size,
            threshold_abs=threshold_abs,
            nms_distance=nms_distance,
            min_distance=0,
        ),
        dtype=int,
    ).reshape(-1, 3)
    background = float(np.median(vol))
    refined = []
    for z, y, x in peaks:
        box = tuple(
            slice(max(c - h, 0), min(c + h + 1, n))
            for c, h, n in zip((z, y, x), refine_halfwidth, vol.shape, strict=True)
        )
        weights = np.clip(vol[box] - background, 0, None)
        if weights.sum() <= 0:
            refined.append([z, y, x])
            continue
        grids = np.meshgrid(*[np.arange(s.start, s.stop) for s in box], indexing="ij")
        refined.append([(g * weights).sum() / weights.sum() for g in grids])
    return np.array(refined, dtype=float).reshape(-1, 3)


# Peaks a job detects in the same volume more than once (the reference at every pass and
# score, the moving volume under the transform being scored), keyed on the volumes'
# identity -- checked through weak references, so a freed array is never mistaken for a
# new one at the same address -- and on the detection settings. Only peaks are kept.
_PEAKS = Remembered(16)


def _detect(volume, settings: DetectPeaksSettings) -> np.ndarray:
    peaks = np.asarray(
        detect_peaks(
            np.asarray(volume),
            block_size=settings.block_size,
            threshold_abs=settings.threshold_abs,
            nms_distance=settings.nms_distance,
            min_distance=settings.min_distance,
        )
    )
    peaks.flags.writeable = False  # shared: a caller must not change it
    return peaks


def bead_peaks(volume: np.ndarray, settings: DetectPeaksSettings) -> np.ndarray:
    """Return the bead peaks of `volume` (detected once per volume and settings)."""
    return _PEAKS((volume,), (settings.model_dump_json(),), lambda: _detect(volume, settings))


def warped_bead_peaks(
    transform: Transform, mov: np.ndarray, ref: np.ndarray, settings: DetectPeaksSettings
) -> np.ndarray:
    """Return the bead peaks of `mov` warped onto `ref`'s grid (once per transform)."""
    return _PEAKS(
        (mov, ref),
        (transform_key(transform), settings.model_dump_json()),
        lambda: _detect(transform.apply(mov, reference=ref), settings),
    )


class BeadNodeDetector:
    """Detects bead centroids as local-maxima peaks."""

    missing_as_nan = False  # the padding stays zeros: no peak is found there

    def __init__(self, settings: DetectPeaksSettings):
        self.settings = settings

    def detect(self, array: ArrayLike) -> ArrayLike:
        return bead_peaks(np.asarray(array), self.settings)


def bead_node_graph(
    beads_match_settings: BeadsMatchSettings,
    transform_type: str,
    iterations: int | None = None,
    score_fn: ScoreFn | None = None,
) -> NodeGraphEstimator:
    """Return the node-graph estimator on bead peaks, with the settings' matcher.

    `iterations` defaults to `beads_match_settings.qc_settings.iterations`. This is the
    single matcher; `build_beads_estimator` composes the vote-ICP mode, initial
    transform correction and the spectral candidate around it.
    """
    return NodeGraphEstimator(
        mov_detector=BeadNodeDetector(beads_match_settings.moving_peaks_settings),
        ref_detector=BeadNodeDetector(beads_match_settings.reference_peaks_settings),
        beads_match_settings=beads_match_settings,
        transform_type=transform_type,
        iterations=(
            beads_match_settings.qc_settings.iterations if iterations is None else iterations
        ),
        score_fn=score_fn,
    )


class VoteIcpEstimator(EstimatorBase):
    """TransformEstimator over peak clouds by iterated displacement voting.

    Reach stage: dense detection on the raw moving volume (`vote_peaks_settings`), voted
    ICP from the initial transform with a wide, shrinking capture radius. Precision
    stage: re-run at short range from that result with the pipeline's own moving
    detection, kept only on a strict score win. Raises `EstimationError` when voting
    abstains before any fit.
    """

    def __init__(
        self,
        dense_detector: NodeDetector,
        precise_detector: NodeDetector,
        ref_detector: NodeDetector,
        settings: VoteIcpSettings,
        score_fn: ScoreFn,
        transform_type: str = "affine",
    ):
        self.transform_type = transform_type
        self.dense_detector = dense_detector
        self.precise_detector = precise_detector
        self.ref_detector = ref_detector
        self.settings = settings
        self.score_fn = score_fn

    @classmethod
    def from_beads_settings(
        cls,
        beads_match_settings: BeadsMatchSettings,
        score_fn: ScoreFn,
        transform_type: str = "affine",
    ) -> VoteIcpEstimator:
        settings = beads_match_settings.vote_icp_settings
        return cls(
            transform_type=transform_type,
            dense_detector=BeadNodeDetector(settings.vote_peaks_settings),
            precise_detector=BeadNodeDetector(beads_match_settings.moving_peaks_settings),
            ref_detector=BeadNodeDetector(beads_match_settings.reference_peaks_settings),
            settings=settings,
            score_fn=score_fn,
        )

    def _register(self, mov_peaks, ref_peaks, initial_transform, initial_radius):
        s = self.settings
        return _vote_icp_register(
            mov_points=mov_peaks,
            ref_points=ref_peaks,
            initial_transform=initial_transform,
            initial_capture_radius=initial_radius,
            min_capture_radius=s.min_capture_radius,
            radius_decay=s.radius_decay,
            cluster_radius=s.cluster_radius,
            min_votes=s.min_votes,
            max_iterations=s.max_iterations,
            convergence_translation=s.convergence_translation,
            transform_type=self.transform_type,
        )

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, initial_transform: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)
        ndim = mov.ndim
        if initial_transform is None:
            initial_transform = Transform.identity(ndim)
        ref_peaks = np.asarray(self.ref_detector.detect(ref))
        dense_peaks = np.asarray(self.dense_detector.detect(mov))
        if min(len(dense_peaks), len(ref_peaks)) < self.settings.min_votes:
            raise EstimationError(
                f"vote_icp: {len(dense_peaks)} dense moving / {len(ref_peaks)} reference "
                f"peaks (need >= {self.settings.min_votes})"
            )
        best, info = self._register(
            dense_peaks, ref_peaks, initial_transform, self.settings.initial_capture_radius
        )
        if best is None:
            raise EstimationError(
                f"vote_icp: voting abstained after {info['iterations']} iteration(s)"
            )
        best_score = self.score_fn(best, mov, ref)

        precise_peaks = np.asarray(self.precise_detector.detect(mov))
        if len(precise_peaks) >= self.settings.min_votes:
            precise, _info = self._register(
                precise_peaks, ref_peaks, best, self.settings.min_capture_radius
            )
            if precise is not None:
                precise_score = self.score_fn(precise, mov, ref)
                # a finite refinement beats a NaN coarse score (a NaN compares false)
                if np.isfinite(precise_score) and (
                    not np.isfinite(best_score) or precise_score > best_score
                ):
                    best = precise
        return best


class VoteInitialTransformCorrection(EstimatorBase):
    """Correct an initial transform by bead displacement voting: a chain's first stage.

    Warps `mov` by the initial transform, detects beads densely and votes for the
    residual drift; "vote_translation" composes the densest cluster's mean displacement
    into the initial transform, "vote_fit" also fits the run's transform type on the
    cluster's pairs (a translation with 4-5 pairs; a degenerate fit is rejected). Every
    candidate competes against the unchanged initial transform on the
    median nearest-neighbour distance of the pipeline's own detection, so a bad vote
    leaves the initial transform unchanged. Returns the (possibly unchanged) initial
    transform; without an initial transform, the identity. `trace()["best_candidate"]`
    names the pick: "keep", "+drift", "-drift" or "fit".
    """

    def __init__(
        self,
        settings: InitialTransformCorrectionSettings,
        mov_detector: NodeDetector,
        ref_detector: NodeDetector,
        transform_type: str = "affine",
    ):
        self.settings = settings
        self.mov_detector = mov_detector
        self.ref_detector = ref_detector
        self.transform_type = transform_type  # what vote_fit fits
        self.dense_detector = BeadNodeDetector(settings.vote_peaks_settings)
        self.last_note: str = ""
        self._pick: str | None = None

    def trace(self) -> dict:
        return {"best_candidate": self._pick, "blocks": None}

    @classmethod
    def from_beads_settings(
        cls, beads_match_settings: BeadsMatchSettings, transform_type: str
    ) -> VoteInitialTransformCorrection:
        return cls(
            settings=beads_match_settings.initial_transform_correction_settings,
            mov_detector=BeadNodeDetector(beads_match_settings.moving_peaks_settings),
            ref_detector=BeadNodeDetector(beads_match_settings.reference_peaks_settings),
            transform_type=transform_type,
        )

    def estimate(
        self, mov: ArrayLike, ref: ArrayLike, initial_transform: Transform | None = None
    ) -> Transform:
        mov = np.asarray(mov, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)
        initial_transform = (
            initial_transform
            if initial_transform is not None
            else Transform.identity(mov.ndim)
        )
        settings = self.settings
        ref_peaks = np.asarray(self.ref_detector.detect(ref))
        if len(ref_peaks) < settings.min_votes:
            self.last_note = f"only {len(ref_peaks)} ref peaks; initial transform unchanged"
            return initial_transform
        ref_tree = cKDTree(ref_peaks)

        def nn_median(transform: Transform) -> float:
            peaks = np.asarray(self.mov_detector.detect(transform.apply(mov, reference=ref)))
            if len(peaks) == 0:
                return np.inf
            return float(np.median(ref_tree.query(peaks)[0]))

        warped = initial_transform.apply(mov, reference=ref)
        dense_peaks = np.asarray(self.dense_detector.detect(warped))
        drift, pairs, n_votes = vote_drift(
            dense_peaks,
            ref_peaks,
            settings.capture_radius,
            settings.cluster_radius,
            settings.min_votes,
        )
        if drift is None:
            self.last_note = (
                f"only {len(dense_peaks)} dense peaks / too few votes; initial transform "
                "unchanged"
            )
            return initial_transform
        # The vote measured a residual drift d (reference minus warped moving peak): a
        # translation by +d after the initial transform takes it out; -d, the other sign,
        # and the unchanged initial transform compete with it.
        candidates = {
            "keep": initial_transform,
            "+drift": Transform.from_translation(drift) @ initial_transform,
            "-drift": Transform.from_translation(-drift) @ initial_transform,
        }
        nn = {name: nn_median(t) for name, t in candidates.items()}
        best = min(nn, key=nn.get)
        corrected = candidates[best]
        note = f"drift={np.round(drift, 1).tolist()} votes={n_votes} pick={best}"
        if settings.mode == "vote_fit" and len(pairs) >= 4:
            # The cluster's votes are correspondences (warped moving peak p, reference
            # peak q) in the initial transform's warped frame: the forward correction
            # after the initial transform maps p onto q.
            p = np.asarray([a for a, _ in pairs], dtype=float)
            q = np.asarray([b for _, b in pairs], dtype=float)
            try:
                correction = (
                    fit_pairs(p, q, self.transform_type)
                    if len(pairs) >= 6
                    else Transform.from_translation((q - p).mean(axis=0))
                )
            except EstimationError as error:  # a degenerate cluster: the vote's pick stands
                note += f"; fit rejected n={len(pairs)} ({error})"
            else:
                fitted = correction @ initial_transform
                nn_fitted = nn_median(fitted)
                if np.isfinite(nn_fitted) and nn_fitted < nn[best]:
                    note += f"; fit accepted n={len(pairs)} nn {nn[best]:.1f}->{nn_fitted:.1f}"
                    corrected, best = fitted, "fit"
                else:
                    note += f"; fit rejected n={len(pairs)} nn {nn[best]:.1f}->{nn_fitted:.1f}"
        self.last_note = note
        self._pick = best
        return corrected


def _vote_icp_register(**kwargs):
    """`vote_icp_register`, with a fit that fails as a failed estimate."""
    try:
        return vote_icp_register(**kwargs)
    except ValueError as error:
        raise EstimationError(str(error)) from error


def build_beads_estimator(
    beads_match_settings: BeadsMatchSettings,
    transform_type: str,
    score_fn: ScoreFn | None = None,
    iterations: int | None = None,
) -> TransformEstimator:
    """Build the beads method: matcher, mode, initial-transform correction, spectral.

    `estimation_mode: vote_icp` swaps the graph matcher for `VoteIcpEstimator`; an
    `initial_transform_correction_settings.mode` other than `none` chains a
    `VoteInitialTransformCorrection` in front; `spectral` on returns a
    `CompetingEstimator` of the configured matcher and a spectral-acquire-then-refine
    cascade, the higher-scoring candidate wins.
    """
    # imported here: metrics imports this module (overlap_score, peaks_from_beads)
    from biahub.registration.metrics import metric_fn

    score_fn = score_fn or metric_fn("overlap", beads_match_settings)

    def node_graph(settings: BeadsMatchSettings) -> NodeGraphEstimator:
        return bead_node_graph(
            settings, transform_type, iterations=iterations, score_fn=score_fn
        )

    if beads_match_settings.estimation_mode == "vote_icp":
        configured: TransformEstimator = VoteIcpEstimator.from_beads_settings(
            beads_match_settings,
            score_fn=score_fn,
            transform_type=transform_type,
        )
    else:
        configured = node_graph(beads_match_settings)
    if beads_match_settings.initial_transform_correction_settings.mode != "none":
        configured = ChainedEstimator(
            [
                VoteInitialTransformCorrection.from_beads_settings(
                    beads_match_settings, transform_type
                ),
                configured,
            ],
            score_fn=score_fn,
        )
    if beads_match_settings.spectral == "off":
        return configured
    spectral_settings = beads_match_settings.model_copy(deep=True)
    spectral_settings.algorithm = "spectral"
    cascade = ChainedEstimator(
        [node_graph(spectral_settings), node_graph(beads_match_settings)], score_fn=score_fn
    )
    # named for what each candidate is, so a record says what won
    first = (
        "vote_icp"
        if beads_match_settings.estimation_mode == "vote_icp"
        else beads_match_settings.algorithm
    )
    return CompetingEstimator(
        {first: configured, f"spectral+{beads_match_settings.algorithm}": cascade},
        score_fn=score_fn,
        escalate_below=(
            beads_match_settings.qc_settings.score_threshold
            if beads_match_settings.spectral == "on_low_score"
            else None
        ),
    )
