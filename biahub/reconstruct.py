from pathlib import Path

import click

from biahub.apply_inverse_transfer_function import apply_inverse_transfer_function
from biahub.cli.parsing import (
    cluster,
    config_filepath,
    input_position_dirpaths,
    monitor,
    output_dirpath,
    reconstruction_names,
    resume,
    sbatch_filepath,
)
from biahub.compute_transfer_function import compute_transfer_functions


@click.command("reconstruct")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@reconstruction_names()
@sbatch_filepath()
@cluster()
@monitor()
@resume()
def reconstruct_cli(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
    reconstruction_names: tuple[str, ...] = (),
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    resume: bool = False,
):
    """Reconstruct a dataset using a reconstruct config.

    This is a convenience function for a `compute-tf` call followed by an
    `apply-inv-tf` call, for every reconstruction under the config's
    `reconstructions:` (or those named with -r).

    Calculates the transfer functions from the shape of the first position in
    the list `input-position-dirpaths`, into `transfer_functions/` beside the
    output plate, then applies them to all positions in the list
    `input-position-dirpaths`, so all positions must have the same TCZYX shape.

    See https://github.com/Biohub/waveorder/tree/main/docs/examples/cli/configs for the
    waveorder settings each reconstruction takes.

    \b
    >>> biahub reconstruct -i ./input.zarr/*/*/* -c ./reconstruct.yml -o ./output.zarr
    """  # noqa: D301
    transfer_functions_dirpath = Path(output_dirpath).parent / "transfer_functions"
    compute_transfer_functions(
        input_position_dirpaths[0],
        config_filepath,
        transfer_functions_dirpath,
        reconstruction_names,
    )
    apply_inverse_transfer_function(
        input_position_dirpaths,
        transfer_functions_dirpath,
        config_filepath,
        output_dirpath,
        sbatch_filepath,
        cluster,
        monitor,
        resume=resume,
        reconstruction_names=reconstruction_names,
    )


if __name__ == "__main__":
    reconstruct_cli()
