import glob

from pathlib import Path
from typing import Annotated, Literal

import typer

from iohub.cli import OptionEatAll
from natsort import natsorted
from typer.core import TyperOption


class OptionEatAllGroups(TyperOption):
    """Greedy list option that keeps each occurrence as one group.

    ``-i a b -i c`` gives ``[("a", "b"), ("c",)]``, where iohub's ``OptionEatAll``
    gives ``["a", "b", "c"]``. Declare the option as ``list[str]``; its callback
    converts the groups.
    """

    def add_to_parser(self, parser, ctx) -> None:
        super().add_to_parser(parser, ctx)
        for opt_name in self.opts:
            registered = parser._long_opt.get(opt_name) or parser._short_opt.get(opt_name)
            if registered is None:
                continue
            append_group = registered.process

            def eat_all(value, state, _append=append_group, _prefixes=registered.prefixes):
                group = [value]
                # Stop before the next option so the parser can handle it.
                while state.rargs and not any(state.rargs[0].startswith(p) for p in _prefixes):
                    group.append(state.rargs.pop(0))
                _append(tuple(group), state)

            registered.process = eat_all
            break

    def type_cast_value(self, ctx, value):
        if value is None:
            return ()
        return tuple(tuple(self.type(v, self, ctx) for v in group) for group in value)


def _eat_all(callback):
    """Mark options using this callback for greedy parsing."""
    callback.__biahub_eat_all__ = OptionEatAll
    return callback


def eat_all_groups(callback):
    """Mark options using this callback for greedy parsing, one group per occurrence."""
    callback.__biahub_eat_all__ = OptionEatAllGroups
    return callback


@_eat_all
def _validate_and_process_paths(value: list[Path]) -> list[Path]:
    """Sort input positions and reject plate roots."""
    from iohub.ngff import Plate, open_ome_zarr

    if not value:  # an optional path option that was not given
        return []
    input_paths = [path for path in natsorted(value) if path.is_dir()]
    if not input_paths:
        raise typer.BadParameter(f"no position directory found in {[str(p) for p in value]}")
    with open_ome_zarr(input_paths[0], mode="r") as dataset:
        if isinstance(dataset, Plate):
            raise ValueError(
                "Please supply a single position instead of an HCS plate. Likely fix: "
                "replace 'input.zarr' with 'input.zarr/0/0/0'"
            )
    return input_paths


def position_key(position_dirpath: Path) -> str:
    """'row/col/fov' of a position directory."""
    return "/".join(Path(position_dirpath).parts[-3:])


def pair_reference_positions(
    position_keys: list[str], reference_position_dirpaths: list[Path] | None
) -> dict[str, Path]:
    """Pair each moving position with the reference position it is registered onto.

    One reference position serves every moving position; several are paired by position
    key, and a moving position without its reference is an error.
    """
    if not reference_position_dirpaths:
        return {}
    if len(reference_position_dirpaths) == 1:
        return {key: Path(reference_position_dirpaths[0]) for key in position_keys}
    by_key = {position_key(p): Path(p) for p in reference_position_dirpaths}
    missing = sorted(set(position_keys) - set(by_key))
    if missing:
        raise typer.BadParameter(f"no reference position for moving positions {missing}")
    return {key: by_key[key] for key in position_keys}


@_eat_all
def _validate_and_process_config_paths(value: list[Path]) -> list[Path]:
    matched_paths = []
    for pattern in value:
        expanded = glob.glob(str(pattern))
        if not expanded:
            raise typer.BadParameter(f"No files matched pattern: {pattern}")
        matched_paths.extend(expanded)

    validated = []
    for p in natsorted(map(Path, matched_paths)):
        if not p.exists():
            raise typer.BadParameter(f"Path does not exist: {p}")
        if not p.is_file():
            raise typer.BadParameter(f"Expected a file, not a directory: {p}")
        if p.suffix.lower() not in [".yml", ".yaml"]:
            raise typer.BadParameter(f"Expected a .yml file, got: {p}")
        validated.append(p)
    return validated


InputPositionDirpaths = Annotated[
    list[Path],
    typer.Option(
        "--input-position-dirpaths",
        "-i",
        callback=_validate_and_process_paths,
        help=(
            'Paths to input positions, for example: "input.zarr/0/0/0", '
            '"input.zarr/0/0/[0-9]", or "input.zarr/*/*/*"'
        ),
    ),
]

MovingPositionDirpaths = Annotated[
    list[Path],
    typer.Option(
        "--moving-position-dirpaths",
        "-m",
        callback=_validate_and_process_paths,
        help=(
            'Positions of the moving store, for example: "moving.zarr/0/0/0" '
            'or "moving.zarr/*/*/*"'
        ),
    ),
]

# Optional: omitted, it parses to an empty list. Declare it keyword-only (after `*`).
ReferencePositionDirpaths = Annotated[
    list[Path],
    typer.Option(
        "--reference-position-dirpaths",
        "-r",
        default_factory=list,
        callback=_validate_and_process_paths,
        help=(
            'Positions of the reference store, for example: "reference.zarr/0/0/0" '
            'or "reference.zarr/*/*/*"'
        ),
    ),
]

ConfigFilepaths = Annotated[
    list[Path],
    typer.Option(
        "--config-filepaths",
        "-c",
        callback=_validate_and_process_config_paths,
        help="Paths to YAML configuration files. All must be existing files with .yml extension.",
    ),
]

ConfigFilepath = Annotated[
    Path,
    typer.Option(
        "--config-filepath",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to YAML configuration file.",
    ),
]

OutputDirpath = Annotated[
    Path,
    typer.Option(
        "--output-dirpath",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Path to output directory",
    ),
]

OutputFilepath = Annotated[
    Path,
    typer.Option(
        "--output-filepath",
        "-o",
        file_okay=True,
        dir_okay=False,
        help="Path to output file",
    ),
]

SbatchFilepath = Annotated[
    Path | None,
    typer.Option(
        "--sbatch-filepath",
        "-sb",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help=(
            "SBATCH filepath that contains slurm parameters to overwrite defaults. "
            "For example, '#SBATCH --mem-per-cpu=16G' will override the default memory per CPU."
        ),
    ),
]

SbatchFilepathPreprocess = Annotated[
    Path | None,
    typer.Option(
        "--sbatch-filepath-preprocess",
        "-sb-preprocess",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help=(
            "SBATCH filepath that contains slurm parameters to overwrite defaults. "
            "For example, '#SBATCH --mem-per-cpu=16G' will override the default memory per CPU."
        ),
    ),
]

SbatchFilepathPredict = Annotated[
    Path | None,
    typer.Option(
        "--sbatch-filepath-predict",
        "-sb-predict",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help=(
            "SBATCH filepath that contains slurm parameters to overwrite defaults. "
            "For example, '#SBATCH --mem-per-cpu=16G' will override the default memory per CPU."
        ),
    ),
]


def install_eat_all_options(command) -> None:
    """Install greedy parsing on explicitly marked list options."""
    for param in command.params:
        option_class = getattr(param.callback, "__biahub_eat_all__", None)
        if isinstance(param, TyperOption) and param.multiple and option_class is not None:
            param.__class__ = option_class


def sbatch_to_submitit(filepath: str) -> dict:
    """Read a text configuration file and return a dictionary of parameters.

    Which can be passed to the submitit executor. This file can contain parameters
    starting with #SBATCH to configure SLURM jobs or parameters starting with #LOCAL
    to configure local jobs. The submitit executor will only apply valid parameters
    and will, for example, ignore local parameters when running on SLURM.

    Parameters
    ----------
    value : Path
        Path to sbatch file

    Returns
    -------
    dict
        Dictionary of slurm parameters

    Example:

    --- sbatch_file.sh ---
    #SBATCH --mem-per-cpu=16G
    #SBATCH --time=1:00:00
    #LOCAL --cpus-per-task=1
    ---

    >>> dict = sbatch_to_submitit(Path("sbatch_file.sh"))
    >>> print(dict)
    {'slurm_mem_per_cpu': '16G', 'slurm_time': '1:00:00', 'cpus_per_task': 1}
    """
    with open(filepath) as f:
        sbatch_file = f.readlines()

    keywords = ["SBATCH", "LOCAL"]
    sbatch_dict = {}
    for line in sbatch_file:
        for keyword in keywords:
            if line.startswith(f"#{keyword} --"):
                line = line.strip(f"#{keyword} --").strip()
                key, value = line.split("=", 1)
                key = key.replace("-", "_").strip()
                try:
                    value = int(value.strip())
                except ValueError:
                    # If conversion to int fails, keep it as a string
                    value = value.strip()
                if keyword == "SBATCH":
                    sbatch_dict["slurm_" + key] = value
                elif keyword == "LOCAL":
                    sbatch_dict[key] = value

    return sbatch_dict


local = Annotated[
    bool,
    typer.Option(
        "--local",
        "-l",
        help="Run jobs locally instead of submitting to SLURM.",
    ),
]

cluster = Annotated[
    Literal["slurm", "local", "debug"],
    typer.Option(
        "--cluster",
        case_sensitive=False,
        show_default=True,
        help=(
            "Execution cluster: 'slurm' submits to a Slurm cluster, "
            "'local' runs jobs as subprocesses on this machine, "
            "'debug' runs jobs in-process in the foreground."
        ),
    ),
]

init_only = Annotated[
    bool,
    typer.Option(
        "--init",
        help="Only initialize the output store and exit; skip per-position processing.",
    ),
]

monitor = Annotated[
    bool,
    typer.Option(
        "--monitor",
        "-m",
        help="Monitor of submitted SLURM jobs.",
    ),
]

# `--monitor` without `-m`, for CLIs whose moving store takes `-m`.
monitor_long = Annotated[
    bool,
    typer.Option(
        "--monitor",
        help="Monitor of submitted SLURM jobs.",
    ),
]

resume = Annotated[
    bool,
    typer.Option(
        "--resume/--no-resume",
        show_default=True,
        help=(
            "Skip the (time, channel) units this position already finished in an "
            "earlier attempt instead of recomputing the whole position. For retrying "
            "a run that was interrupted, e.g. by Slurm preemption. Finished units are "
            "keyed by the step's validated settings, so a settings change recomputes "
            "them; a change to the input data does not, so use a fresh output store "
            "(or --no-resume) after an upstream step was rerun."
        ),
    ),
]

num_processes = Annotated[
    int,
    typer.Option(
        "--num-processes",
        "-j",
        help="Number of parallel processes",
    ),
]
