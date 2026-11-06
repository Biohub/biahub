from collections.abc import Sequence
from pathlib import Path

import click

from waveorder.cli.compute_transfer_function import (
    compute_transfer_function_cli as compute_transfer_function,
)

from biahub.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    output_dirpath,
    reconstruction_names,
)
from biahub.utils.reconstruction import (
    load_reconstructions,
    reconstruction_config,
    transfer_function_path,
)


def compute_transfer_functions(
    input_position_dirpath: Path,
    config_filepath: Path,
    transfer_functions_dirpath: Path,
    reconstruction_names: Sequence[str] = (),
) -> None:
    """Compute one transfer function per reconstruction, into ``<dir>/<name>.zarr``."""
    try:
        reconstructions = load_reconstructions(config_filepath, reconstruction_names)
    except ValueError as err:
        raise click.UsageError(str(err)) from err
    Path(transfer_functions_dirpath).mkdir(parents=True, exist_ok=True)
    for reconstruction in reconstructions.values():
        tf_path = transfer_function_path(transfer_functions_dirpath, reconstruction.name)
        with reconstruction_config(reconstruction) as entry_config:
            compute_transfer_function(input_position_dirpath, entry_config, tf_path)
        click.echo(f"Transfer function '{reconstruction.name}' saved to {tf_path}.")


@click.command("compute-tf")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@reconstruction_names()
def compute_transfer_function_cli(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
    reconstruction_names: tuple[str, ...] = (),
) -> None:
    """Compute the transfer functions of a reconstruct config.

    Writes one transfer function per reconstruction under the config's
    `reconstructions:`, as `<output-dirpath>/<name>.zarr`, from the shape of the
    first position in `input-position-dirpaths`.

    See https://github.com/Biohub/waveorder/tree/main/docs/examples/cli/configs for the
    waveorder settings each reconstruction takes.

    >>> biahub compute-tf \
        -i ./input.zarr/0/0/0 \
        -c ./reconstruct.yml \
        -o ./transfer_functions \
        -r phase
    """
    compute_transfer_functions(
        input_position_dirpaths[0], config_filepath, output_dirpath, reconstruction_names
    )


if __name__ == "__main__":
    compute_transfer_function_cli()
