import os

from collections.abc import Sequence
from pathlib import Path

import click
import submitit

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate
from waveorder.cli.apply_inverse_transfer_function import (
    apply_inverse_transfer_function_single_position,
    get_reconstruction_output_metadata,
    get_reconstruction_provenance,
)
from waveorder.cli.utils import estimate_resources as wo_estimate_resources

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    cluster,
    config_filepath,
    init_only,
    input_position_dirpaths,
    monitor,
    output_dirpath,
    reconstruction_names,
    resume,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.utils.cluster import (
    apply_sbatch_overrides,
    echo_resources,
    estimate_resources,
    get_submitit_cluster,
)
from biahub.utils.ngff import PROVENANCE_METADATA_KEYS
from biahub.utils.reconstruction import (
    Reconstruction,
    estimate_transfer_function_memory,
    load_reconstructions,
    reconstruction_config,
    transfer_function_path,
)

# Wall-time per reconstructed volume, in minutes, on CPU: ~2x the worst rate
# observed over completed phase runs (A549 2026_06_16, 101 min for 67 volumes =
# 1.51 min/vol).
# NOTE: reconstruct is the one step where per-volume cost is not
# geometry-driven -- it spans 0.14-1.51 min/volume across runs (10x), and
# keying on voxels instead is no better (also ~10x), because the cost is
# dominated by the reconstruction algorithm settings (regularization, TV
# iterations) rather than data size. Sizing for the worst case therefore
# over-requests on many-volume datasets; see PR #273 discussion.
_CPU_MINUTES_PER_VOLUME = 3.0
# On a GPU, per 160 Richardson-Lucy iterations: the mantis V2 neuromast time-lapse
# (2138 volumes of 171 x 768 x 794) took 14.6 s of RL per volume on an A40, the
# slowest GPU of `preempted` it ran on, and ~15% more for reading, writing and
# loading the transfer function. 0.5 min leaves ~1.7x headroom on an A40.
_GPU_MINUTES_PER_VOLUME = 0.5
_GPU_REFERENCE_RL_ITERATIONS = 160
# A GPU reconstruction runs one timepoint at a time (one worker per GPU); the CPUs
# only read and write.
_GPU_NUM_CPUS = 4


def _init_output_plate(
    input_position_dirpaths: list[Path],
    output_dirpath: Path,
    reconstruction: Reconstruction,
) -> list[str]:
    """Add one reconstruction's channels to the output plate, creating it if needed.

    Output shape/scale/version/channel names are derived by waveorder from the
    reconstruction's config, and per-position metadata is copied from the input
    plate. create_empty_plate appends the channels of each further
    reconstruction to the plate, so this is called once per reconstruction.

    A plate that already holds these positions with these channels and this
    reconstruction's recorded settings (made by ``--init``) is left alone.
    create_empty_plate would rewrite every position's attributes,
    read-modify-write, and in the pipeline a reconstruction's per-position runs
    overlap virtual-stain preprocessing, which writes its normalization
    statistics into those same attributes.

    Returns the reconstruction's output channel names.
    """
    # Pixel sizes come from the config (the source of truth). waveorder's
    # get_reconstruction_output_metadata already emits a PixelSizeMismatchWarning
    # when the config pixel sizes disagree (>5%) with the input zarr scale, so we
    # do not warn again here.
    with reconstruction_config(reconstruction) as entry_config:
        output_metadata = get_reconstruction_output_metadata(
            input_position_dirpaths[0], entry_config
        )
    output_metadata.pop("plate_metadata", None)
    channel_names = output_metadata["channel_names"]
    position_keys = [Path(p).parts[-3:] for p in input_position_dirpaths]
    provenance = get_reconstruction_provenance(reconstruction.settings)
    if _plate_holds(output_dirpath, position_keys, channel_names, provenance):
        return channel_names

    input_plate = Path(input_position_dirpaths[0]).parents[2]
    create_empty_plate(
        store_path=output_dirpath,
        position_keys=position_keys,
        **output_metadata,
        metadata_sources=input_plate,
        metadata_keys=PROVENANCE_METADATA_KEYS,
        # The reconstruction's settings, under its own top-level key
        # (`waveorder-Phase3D`, `waveorder-GFP_Density3D`, ...) beside the
        # `biahub-<step>` keys, recorded when the plate is created as for every
        # other step: the next step's create_empty_plate inherits provenance at
        # plate-creation time, and the pipeline scaffolds every store before any
        # of them hold data. waveorder writes the same record when it creates a
        # plate itself and never per position, and one key per reconstruction
        # means each reconstruction appending its channels overwrites only its own.
        extra_metadata=provenance,
    )

    return channel_names


def _plate_holds(
    plate_dirpath: Path,
    position_keys: list[tuple[str, ...]],
    channel_names: list[str],
    provenance: dict,
) -> bool:
    """Whether the plate has every position, every channel and this provenance."""
    if not Path(plate_dirpath).exists():
        return False
    with open_ome_zarr(plate_dirpath, layout="hcs", mode="r") as plate:
        if not set(channel_names) <= set(plate.channel_names):
            return False
        positions = dict(plate.positions())
        for key in position_keys:
            position = positions.get("/".join(key))
            if position is None or any(
                position.zattrs.get(name) != value for name, value in provenance.items()
            ):
                return False
    return True


def _estimate_resources(
    reconstruction: Reconstruction, input_shape: tuple[int, int, int, int, int]
) -> tuple[int, int, int, int]:
    """(num_workers, num_cpus, mem_gb, time_minutes) for one position of one reconstruction."""
    T, C, Z, Y, X = input_shape
    settings = reconstruction.settings
    if reconstruction.uses_gpu:
        num_workers, minutes_per_volume = 1, _GPU_MINUTES_PER_VOLUME
        rl = settings.fluorescence.apply_inverse.rl if settings.fluorescence else None
        if rl is not None:
            minutes_per_volume *= max(1.0, rl.iterations / _GPU_REFERENCE_RL_ITERATIONS)
    else:
        # waveorder sizes its per-position worker pool from this cap. In CI/tests
        # run serially, as biahub's own estimate_resources does: the test data is
        # tiny, so spawning a pool (one re-import of the stack per worker) costs
        # far more than the work itself.
        num_workers = 1 if os.environ.get("CI") == "true" else 16
        minutes_per_volume = _CPU_MINUTES_PER_VOLUME
    num_workers, mem_per_worker = wo_estimate_resources(
        list(input_shape), settings, num_workers
    )
    mem_gb = int(num_workers * mem_per_worker)
    # CPUs/RAM come from waveorder above. Wall-time scales with the number of
    # volumes reconstructed: a reconstruction only processes its config's input
    # channels, so pass a shape with C = that count and take only the time.
    time_minutes, _, _ = estimate_resources(
        (T, len(settings.input_channel_names), Z, Y, X),
        time_multiplier=minutes_per_volume,
        max_num_cpus=num_workers,
    )
    num_cpus = _GPU_NUM_CPUS if reconstruction.uses_gpu else num_workers
    return int(num_workers), int(num_cpus), mem_gb, time_minutes


def _apply_single_position(
    reconstruction: Reconstruction,
    input_position_dirpath: Path,
    transfer_function_dirpath: Path,
    output_position_dirpath: Path,
    num_workers: int,
    output_channel_names: list[str],
    resume: bool,
) -> None:
    """One position of one reconstruction; the job body submitit runs.

    The reconstruction's waveorder config is written here, in the job, because a
    temporary file of the submitting host is not visible on a compute node.
    """
    with reconstruction_config(reconstruction) as entry_config:
        apply_inverse_transfer_function_single_position(
            input_position_dirpath,
            transfer_function_dirpath,
            entry_config,
            output_position_dirpath,
            num_workers,
            output_channel_names,
            resume=resume,
        )


def apply_inverse_transfer_function(
    input_position_dirpaths: list[Path],
    transfer_functions_dirpath: Path | None,
    config_filepath: Path,
    output_dirpath: Path,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = True,
    init_only: bool = False,
    resume: bool = False,
    reconstruction_names: Sequence[str] = (),
) -> None:
    """Apply the inverse transfer functions of a reconstruct config to a dataset.

    Parameters
    ----------
    input_position_dirpaths : list[Path]
        Paths to input positions.
    transfer_functions_dirpath : Path or None
        Directory holding one ``<name>.zarr`` transfer function per
        reconstruction, as ``compute-tf`` writes it (ignored in ``--init`` mode).
    config_filepath : Path
        Path to the YAML reconstruct config.
    output_dirpath : Path
        Path to output zarr.
    sbatch_filepath : str, optional
        SBATCH file with slurm parameter overrides.
    cluster : str
        Execution cluster: 'slurm', 'local', or 'debug'.
    monitor : bool
        Monitor submitted SLURM jobs.
    init_only : bool
        Only initialize the output store and exit.
    resume : bool
        Skip the timepoints this position already finished in an earlier,
        interrupted attempt. waveorder keys each finished timepoint on the
        reconstruction settings and the transfer function, so a changed config
        or a recomputed transfer function recomputes instead of being skipped.
    reconstruction_names : sequence of str
        Reconstructions of the config to run; empty runs all of them.
    """
    output_dirpath = Path(output_dirpath)
    slurm_out_path = output_dirpath.parent / "slurm_output"

    try:
        reconstructions = load_reconstructions(config_filepath, reconstruction_names)
    except ValueError as err:
        raise click.UsageError(str(err)) from err

    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as ds:
        input_shape = ds.data.shape

    # One RESOURCES line per reconstruction: the Nextflow pipeline runs each as
    # its own tasks (a GPU reconstruction on a GPU), sized by its own line, and
    # starts a downstream step once the reconstructions writing its channels are done.
    plans = []
    for reconstruction in reconstructions.values():
        channel_names = _init_output_plate(
            input_position_dirpaths, output_dirpath, reconstruction
        )
        num_workers, num_cpus, mem_gb, time_minutes = _estimate_resources(
            reconstruction, input_shape
        )
        echo_resources(
            num_cpus,
            mem_gb,
            time_minutes,
            reconstruction=reconstruction.name,
            channels=channel_names,
            gpus=int(reconstruction.uses_gpu),
            tf_mem_gb=estimate_transfer_function_memory(reconstruction, input_shape[-3:]),
        )
        plans.append(
            (reconstruction, channel_names, num_workers, num_cpus, mem_gb, time_minutes)
        )

    if init_only:
        num_channels = sum(len(plan[1]) for plan in plans)
        click.echo(
            f"Created {output_dirpath} ({len(input_position_dirpaths)} positions, "
            f"{num_channels} output channels from {len(plans)} reconstructions)"
        )
        return

    # Fan out one job per position and reconstruction via submitit. With
    # --cluster debug, submitit's DebugExecutor runs the work in-process (the
    # slurm_* parameters are ignored): Nextflow already handles the fan-out and
    # resource scheduling, so the CLI must NOT submit its own SLURM jobs. See also:
    # examples/submitit_debug_nextflow/2026-05-27-submitit-debug-nextflow-concerns.md
    resolved_cluster = get_submitit_cluster(cluster=cluster)
    jobs, job_labels = [], []
    for reconstruction, channel_names, num_workers, num_cpus, mem_gb, time_minutes in plans:
        slurm_args = {
            "slurm_job_name": f"apply-inverse-transfer-function-{reconstruction.name}",
            "slurm_mem": f"{mem_gb}G",
            "slurm_cpus_per_task": num_cpus,
            "slurm_array_parallelism": 100,  # process up to 100 positions at a time
            "slurm_time": time_minutes,
            "slurm_partition": "preempted",
        }
        if reconstruction.uses_gpu:
            slurm_args["slurm_gres"] = "gpu:1"
        if sbatch_filepath:
            slurm_args = apply_sbatch_overrides(
                slurm_args, sbatch_to_submitit(sbatch_filepath)
            )

        click.echo(
            f"Preparing '{reconstruction.name}' jobs on cluster='{resolved_cluster}': {slurm_args}"
        )
        executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
        executor.update_parameters(**slurm_args)
        tf_path = transfer_function_path(transfer_functions_dirpath, reconstruction.name)
        with submitit.helpers.clean_env(), executor.batch():
            for pos_path in input_position_dirpaths:
                jobs.append(
                    executor.submit(
                        _apply_single_position,
                        reconstruction,
                        pos_path,
                        tf_path,
                        output_dirpath / Path(*pos_path.parts[-3:]),
                        num_workers,
                        channel_names,
                        resume,
                    )
                )
                job_labels.append(f"{pos_path} ({reconstruction.name})")

    slurm_out_path.mkdir(exist_ok=True)
    log_path = slurm_out_path / "submitit_jobs_ids.log"
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job.job_id for job in jobs))

    # submitit's DebugExecutor is lazy: .submit() wraps the callable in a DebugJob
    # but execution only happens when .wait()/.done()/.result() is called. Run each
    # one in the foreground and stream progress; monitor's async polling UI is
    # pointless against synchronous in-process jobs.
    if resolved_cluster == "debug":
        for job, label in zip(jobs, job_labels, strict=True):
            job.wait()
            click.echo(f"Apply-inv-tf complete: {label}")
        return

    if monitor:
        monitor_jobs(jobs, job_labels)


@click.command("apply-inv-tf")
@input_position_dirpaths()
@click.option(
    "--transfer-function-dirpath",
    "-t",
    default=None,
    type=click.Path(),
    help=(
        "Directory of transfer functions, one <name>.zarr per reconstruction, as "
        "compute-tf writes it (not required for --init)."
    ),
)
@config_filepath()
@output_dirpath()
@reconstruction_names()
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
@resume()
def apply_inverse_transfer_function_cli(
    input_position_dirpaths: list[Path],
    transfer_function_dirpath: str | None,
    config_filepath: Path,
    output_dirpath: Path,
    reconstruction_names: tuple[str, ...] = (),
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    init_only: bool = False,
    resume: bool = False,
):
    r"""Apply the inverse transfer functions of a reconstruct config to a dataset.

    Runs every reconstruction under the config's `reconstructions:` (or those
    named with -r) on all positions in `input-position-dirpaths`, so all
    positions must have the same TCZYX shape, and writes their channels into one
    output plate. A reconstruction whose waveorder `device` is a GPU gets a GPU
    and one worker per position.

    \b
    Full SLURM fan-out:
    >>> biahub apply-inv-tf -i ./input.zarr/*/*/* -t ./transfer_functions -c ./reconstruct.yml -o ./output.zarr

    \b
    Initialize the output plate only (Nextflow init step):
    >>> biahub apply-inv-tf --init -i ./input.zarr/*/*/* -c ./reconstruct.yml -o ./output.zarr

    \b
    In-process run of one reconstruction of a single position (Nextflow per-position step):
    >>> biahub apply-inv-tf --cluster debug -i ./input.zarr/A/1/0 -t ./transfer_functions -c ./reconstruct.yml -o ./output.zarr -r phase
    """
    if not init_only and transfer_function_dirpath is None:
        raise click.UsageError(
            "--transfer-function-dirpath / -t is required unless using --init."
        )

    apply_inverse_transfer_function(
        input_position_dirpaths=input_position_dirpaths,
        transfer_functions_dirpath=Path(transfer_function_dirpath)
        if transfer_function_dirpath
        else None,
        config_filepath=config_filepath,
        output_dirpath=output_dirpath,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        init_only=init_only,
        resume=resume,
        reconstruction_names=reconstruction_names,
    )


if __name__ == "__main__":
    apply_inverse_transfer_function_cli()
