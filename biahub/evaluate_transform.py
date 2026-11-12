"""`biahub evaluate-transform`: how well a transforms file registers, in µm.

Evaluates the file's transform -- and, when its run folder holds a stack's records, every
block's -- on the given positions with `biahub.registration.evaluate`, against the
unregistered pair, into one table (CSV). Independent of the score the estimate optimised.
"""

from __future__ import annotations

import csv
import functools

from pathlib import Path

import click

from biahub.cli.parsing import (
    cluster,
    config_filepath,
    moving_position_dirpaths,
    output_filepath,
    pair_reference_positions,
    position_key,
    reference_position_dirpaths,
    refusals_as_usage_errors,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.cli.slurm import wait_for_jobs_to_finish
from biahub.core.transform import Transform
from biahub.registration.benchmark import block_candidates, evaluate_candidates
from biahub.settings import load_transform_settings
from biahub.utils.cluster import apply_sbatch_overrides, get_submitit_cluster


def _file_candidate(transforms, position_keys: list[str], t: int) -> dict:
    """Return the file's forward transform at `t`, per position (a per-position file)."""
    if transforms.per_position:
        return {(key, t): transforms.transform_for(t, key) for key in position_keys}
    return {t: transforms.transform_for(t)}


def _run_dir(transforms_filepath: Path) -> Path:
    """Return the run folder of a transforms file: `<stem>/`, or the run folder it is in.

    The second is a pooled run's `per_position.yml`, written inside its run folder.
    """
    from biahub.registration.engine.disk import RUN_FILENAME

    beside = transforms_filepath.with_suffix("")
    if not beside.is_dir() and (transforms_filepath.parent / RUN_FILENAME).exists():
        return transforms_filepath.parent
    return beside


def _bead_detection(run_dir: Path, threshold: float | tuple | None) -> dict:
    """Return bead detection settings, `threshold` or else the estimate's own.

    `threshold` is one value for both channels, or (moving, reference).
    """
    if isinstance(threshold, tuple):  # fluorescence and phase need their own
        moving, reference = threshold
        return {"moving": {"threshold_abs": moving}, "reference": {"threshold_abs": reference}}
    if threshold is not None:
        return {"threshold_abs": threshold}
    from biahub.registration.engine.disk import ENGINE_SETTINGS_FILENAME, RunFolder
    from biahub.settings import load_estimate_transform_settings

    # every position of a run shares its settings: the first work folder that has them
    files = [w / ENGINE_SETTINGS_FILENAME for _, w in RunFolder(run_dir).work_dirs()]
    settings_file = next((f for f in files if f.exists()), None)
    blocks = (
        load_estimate_transform_settings(settings_file).stack_settings()
        if settings_file is not None
        else []
    )
    # the stack's beads block, wherever it is (the last, if several)
    beads = next((b for b in reversed(blocks) if b.bead_matching is not None), None)
    if beads is None:
        raise click.UsageError(
            "--landmarks beads needs --bead-threshold, or a run folder whose settings "
            "detected beads"
        )

    def peaks(p):  # the estimate's detection, so the same beads are found
        return {
            "threshold_abs": p.threshold_abs,
            "nms_distance": p.nms_distance,
            "block_size": tuple(p.block_size),
        }

    return {
        "moving": peaks(beads.bead_matching.moving_peaks_settings),
        "reference": peaks(beads.bead_matching.reference_peaks_settings),
    }


def evaluate_transform(
    moving_position_dirpaths: list[Path],
    reference_position_dirpaths: list[Path] | None,
    transforms_filepath: Path,
    output_filepath: Path,
    moving_channel: str | None = None,
    reference_channel: str | None = None,
    landmarks: str | None = None,
    t: int = 0,
    blocks: bool = True,
    figures: bool = False,
    tags: dict | None = None,
    bead_threshold: float | tuple | None = None,
    reference_t: int | None = None,
    compare: str = "edges",
    cluster: str = "debug",
    sbatch_filepath: str | None = None,
    **image_kwargs,
) -> list[dict]:
    """Evaluate a transforms file (and its stack's blocks) on positions; write the table.

    `tags` become constant columns, first (e.g. the case and pipeline of a benchmark), so
    tables from several runs can be joined; a tag named like a column is refused. -r
    positions are paired with the moving ones by position (one serves them all). A
    stabilization (no reference channel) compares `t` with `reference_t` -- by default
    its reference frame, the run's first timepoint with data (`neighbours` has none: pass
    a quiet frame) -- each candidate relative to that frame's own transform.
    """
    if bead_threshold is not None and landmarks != "beads":
        raise click.UsageError("--bead-threshold goes with --landmarks beads")
    transforms = load_transform_settings(transforms_filepath)
    keys = [position_key(p) for p in moving_position_dirpaths]
    if transforms.reference_channel is None and reference_t is None:
        reference_t = _reference_frame_t(transforms)
    if reference_position_dirpaths:
        reference_for = pair_reference_positions(keys, reference_position_dirpaths)
    elif transforms.reference_channel is not None:
        raise click.UsageError(
            f"a registration onto {transforms.reference_channel!r} needs the reference "
            "positions (-r)"
        )
    else:  # a stabilization's reference is the moving positions themselves
        reference_for = dict(zip(keys, moving_position_dirpaths, strict=True))
    # positions the file has no list for (held out from a per-position estimate): each
    # estimated position's transform is a candidate there, named after it
    held_out = transforms.per_position and not set(keys) <= set(transforms.positions)
    if held_out:
        candidates = {
            f"transforms ({pos})": {t: transforms.transform_for(t, pos)}
            for pos in transforms.positions
        }
    else:
        candidates = {"transforms": _file_candidate(transforms, keys, t)}
    run_dir = _run_dir(Path(transforms_filepath))
    # a per-position file on its own positions: each position's own blocks; otherwise
    # (held out, or a pooled file) a run's per-position blocks are each a candidate,
    # named after the position they were estimated on
    own_blocks = transforms.per_position and not held_out
    if blocks and run_dir.is_dir():
        for name, by_key in block_candidates(run_dir).items():
            if own_blocks or not any(isinstance(key, tuple) for key in by_key):
                candidates[name] = by_key
                continue
            for (position, at), transform in by_key.items():
                if at == t:
                    candidates[f"{name} ({position})"] = {t: transform}
    if transforms.reference_channel is None:
        candidates = _relative_to_reference_frame(candidates, transforms, reference_t)
    output_filepath = Path(output_filepath)
    detect_kwargs = _bead_detection(run_dir, bead_threshold) if landmarks == "beads" else None
    pairs = [
        (path, reference_for[key])
        for key, path in zip(keys, moving_position_dirpaths, strict=True)
    ]
    evaluate_on = functools.partial(
        evaluate_candidates,
        candidates,
        moving_channel=moving_channel or transforms.moving_channels[0],
        # a stabilization has no reference channel: its reference is the moving channel
        reference_channel=(
            reference_channel or transforms.reference_channel or transforms.moving_channels[0]
        ),
        t=t,
        landmarks=landmarks,
        compare=compare,
        detect_kwargs=detect_kwargs,
        reference_t=reference_t,
        figures_dir=output_filepath.with_suffix("") if figures else None,
        **image_kwargs,
    )
    rows = _run_per_position(
        evaluate_on, pairs, len(candidates), output_filepath, cluster, sbatch_filepath
    )
    clash = sorted(set(tags or {}) & set(rows[0]))
    if clash:
        raise click.UsageError(f"--tag {clash} would overwrite the table's own columns")
    rows = [{**(tags or {}), **row} for row in rows]
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _reference_frame_t(transforms) -> int:
    """Return the timepoint a stabilization file's transforms are measured against."""
    entries = transforms.transforms or next(iter((transforms.positions or {}).values()), [])
    if transforms.reference_frame not in ("first", "previous"):
        # neighbours aligns each timepoint onto the median around it: no frame is the
        # reference, and the first may be one that jumps
        raise click.UsageError(
            f"a stabilization with reference frame {transforms.reference_frame!r} needs "
            "--reference-t: a quiet frame to compare with (its transform near zero)"
        )
    # the run's first timepoint with data (an empty frame's entry is a stand-in): first
    # aligns onto it and previous chains onto it
    return min(e.t for e in entries if e.t is not None and e.filled_from is None)


def _relative_to_reference_frame(candidates: dict, transforms, reference_t: int) -> dict:
    """Return each candidate relative to the reference frame's own transform.

    A stabilization moves every frame, the reference frame too (`neighbours`): frame t
    stabilized is compared with the reference frame stabilized, which is frame t warped
    by `T_ref^-1 @ T_t` compared with the raw reference frame. For `first` and
    `previous` the reference frame's transform is the identity: nothing changes.
    """

    def reference(key):
        position = key[0] if isinstance(key, tuple) else None
        return transforms.transform_for(reference_t, position).invert()

    return {
        name: (
            reference(None) @ by_key
            if isinstance(by_key, Transform)
            else {key: reference(key) @ transform for key, transform in by_key.items()}
        )
        for name, by_key in candidates.items()
    }


# A job's size: one position, every candidate warped once (2048 x 2048 x ~60 stacks).
EVALUATE_CPUS = 4
EVALUATE_GB_PER_CPU = 4
EVALUATE_MINUTES_PER_CANDIDATE = 10


def _run_per_position(
    evaluate_on, pairs, n_candidates, output_filepath, cluster, sbatch_filepath
) -> list[dict]:
    """Return the rows of every position, evaluated in this process or one job each.

    `cluster` "debug" (the default for Python callers, and what a Nextflow task passes)
    runs here; "slurm" / "local" submit one job per position, sized for its candidates
    (an sbatch file's settings win), and join the rows in the positions' order.
    """
    cluster = get_submitit_cluster(cluster=cluster)
    if cluster == "debug":
        return evaluate_on(pairs)
    import submitit

    slurm_args = {
        "slurm_job_name": "evaluate_transform",
        "slurm_cpus_per_task": EVALUATE_CPUS,
        "slurm_mem_per_cpu": f"{EVALUATE_GB_PER_CPU}G",
        # the unregistered pair is a candidate too
        "slurm_time": EVALUATE_MINUTES_PER_CANDIDATE * (n_candidates + 1),
        "slurm_partition": "preempted",
        "slurm_use_srun": False,
    }
    if sbatch_filepath:
        slurm_args = apply_sbatch_overrides(slurm_args, sbatch_to_submitit(sbatch_filepath))
    folder = (
        Path(output_filepath).with_suffix("").parent / "slurm_output" / "evaluate_transform"
    )
    executor = submitit.AutoExecutor(folder=folder, cluster=cluster)
    executor.update_parameters(**slurm_args)
    click.echo(
        f"Submitting {len(pairs)} evaluation job(s) on cluster='{cluster}': {slurm_args}"
    )
    with submitit.helpers.clean_env(), executor.batch():
        jobs = [executor.submit(evaluate_on, [pair]) for pair in pairs]
    wait_for_jobs_to_finish(jobs)
    return [row for job in jobs for row in job.result()]


def _thresholds(value: str | None) -> float | tuple | None:
    """Parse --bead-threshold: one number, or 'moving,reference'."""
    if value is None:
        return None
    try:
        numbers = tuple(float(v) for v in str(value).split(","))
    except ValueError:
        raise click.BadParameter(
            f"a number or 'moving,reference', got {value!r}", param_hint="--bead-threshold"
        ) from None
    if len(numbers) not in (1, 2):
        raise click.BadParameter(
            f"a number or 'moving,reference', got {value!r}", param_hint="--bead-threshold"
        )
    return numbers[0] if len(numbers) == 1 else numbers


def _parse_tags(tags) -> dict:
    parsed = {}
    for tag in tags:
        name, sep, value = tag.partition("=")
        if not sep or not name:
            raise click.BadParameter(f"a tag is name=value; got {tag!r}", param_hint="--tag")
        parsed[name] = value
    return parsed


@click.command("evaluate-transform")
@moving_position_dirpaths()
@reference_position_dirpaths(required=False)
@config_filepath(
    help="The transforms file (estimate-transform's output, or one written by hand)."
)
@output_filepath()
@click.option(
    "--moving-channel", default=None, help="Default: the file's first moving channel."
)
@click.option(
    "--reference-channel", default=None, help="Default: the file's reference channel."
)
@click.option(
    "--landmarks",
    type=click.Choice(["nuclei", "beads", "none"]),
    default="none",
    show_default=True,
    help="Point-based residuals from detected nuclei or beads (z is read from points).",
)
@click.option(
    "--timepoint",
    "--t",
    "t",
    type=int,
    default=0,
    show_default=True,
    help="Timepoint to evaluate.",
)
@click.option(
    "--reference-t",
    type=int,
    default=None,
    help="Read the reference at this timepoint. Default: a stabilization's reference frame "
    "(the run's first timepoint with data); a registration's reference at --t.",
)
@click.option(
    "--tile-px",
    type=int,
    default=256,
    show_default=True,
    help="Tile size (pixels); smaller (128) on a compact sample that fills few tiles.",
)
@click.option(
    "--grid", type=int, default=None, help="Tiles per side of the grid (default: evaluate's)."
)
@click.option(
    "--compare",
    type=click.Choice(["edges", "intensity"]),
    default="edges",
    show_default=True,
    help="What the tiles and z crops compare: edges (cross-modal pairs) or intensity "
    "(the same kind of image, e.g. membranes onto virtual-stained membranes).",
)
@click.option(
    "--no-blocks", is_flag=True, help="Only the file's transform, not each block of its stack."
)
@click.option(
    "--figures",
    is_flag=True,
    help="Save a figure per candidate and position next to the table.",
)
@click.option(
    "--bead-threshold",
    type=str,
    default=None,
    callback=lambda ctx, param, value: _thresholds(value),
    help="With --landmarks beads: the detection threshold, one for both channels or "
    "'moving,reference' (fluorescence and phase differ). Default: the estimate's own.",
)
@click.option(
    "--tag",
    "tags",
    multiple=True,
    help="name=value, a constant column (e.g. --tag case=dragonfly); repeat for several.",
)
@cluster()
@sbatch_filepath()
@refusals_as_usage_errors
def evaluate_transform_cli(
    moving_position_dirpaths,
    reference_position_dirpaths,
    config_filepath,
    output_filepath,
    moving_channel,
    reference_channel,
    landmarks,
    t,
    reference_t,
    tile_px,
    grid,
    compare,
    no_blocks,
    figures,
    bead_threshold,
    tags,
    cluster,
    sbatch_filepath,
) -> None:
    """Evaluate a transforms file on positions: residuals in µm, against unregistered.

    One row per candidate (the file's transform, each block of its stack, unregistered)
    and position: tile and point medians, residual scale and rotation, z and where it
    came from. Pass held-out positions to see whether the transform carries. Each
    position is one SLURM job (`--cluster debug` to run here, as a Nextflow task does).

    \b
    >>> biahub evaluate-transform -m moving.zarr/A/1/0 -r reference.zarr/A/1/0 \\
        -c out/transforms.yml -o out/evaluation.csv --landmarks nuclei

    \b
    A stabilization (no reference channel) is evaluated against its own positions:
    >>> biahub evaluate-transform -m data.zarr/A/1/0 -c out/transforms.yml \\
        -o out/evaluation.csv --t 5 --reference-t 0
    """  # noqa: D301
    rows = evaluate_transform(
        moving_position_dirpaths,
        reference_position_dirpaths,
        config_filepath,
        output_filepath,
        moving_channel=moving_channel,
        reference_channel=reference_channel,
        landmarks=None if landmarks == "none" else landmarks,
        t=t,
        blocks=not no_blocks,
        figures=figures,
        tags=_parse_tags(tags),
        bead_threshold=bead_threshold,
        reference_t=reference_t,
        compare=compare,
        cluster=cluster,
        sbatch_filepath=sbatch_filepath,
        tile_px=tile_px,
        **({} if grid is None else {"grid": grid}),
    )
    for row in rows:
        if row["empty"]:
            click.echo(
                f"{row['position']}: WARNING the {row['empty']} image is blank, skipped"
            )
            continue
        tiles = row["tiles_median_um"]
        click.echo(
            f"{row['position']} {row['candidate']:>20}: tiles "
            + ("n/a" if tiles is None else f"{tiles:.2f} µm")
            + (
                f", z {row['z_um']:+.2f} µm ({row['z_from']})"
                if row["z_um"] is not None
                else ""
            )
        )
    click.echo(f"Evaluation saved to {Path(output_filepath).resolve()}")
