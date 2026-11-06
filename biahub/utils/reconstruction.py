"""The reconstruct step's named reconstructions, as compute-tf and apply-inv-tf run them.

A reconstruct config (:class:`biahub.settings.ReconstructSettings`) lists waveorder
configs by name. waveorder's entry points take a config *file*, so each
reconstruction is handed to them as a temporary YAML holding exactly the mapping the
user wrote for it (:func:`reconstruction_config`), written in the process that calls
waveorder: a temporary file made on the submitting host is not visible on a compute
node.
"""

import math
import tempfile

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml

from waveorder.cli.settings import ReconstructionSettings

from biahub.settings import ReconstructSettings


@dataclass(frozen=True)
class Reconstruction:
    """One entry of a reconstruct config.

    ``config`` is the entry's mapping as written in the YAML, which is what
    waveorder reads back; ``settings`` is its validated form.
    """

    name: str
    settings: ReconstructionSettings
    config: dict

    @property
    def uses_gpu(self) -> bool:
        """Whether apply-inv-tf runs this reconstruction on a GPU (waveorder's ``device``)."""
        return self.settings.device not in (None, "cpu")


def load_reconstructions(
    config_filepath: Path, names: Sequence[str] = ()
) -> dict[str, Reconstruction]:
    """Read a reconstruct config and return its reconstructions, in config order.

    ``names`` selects entries (in config order, not the order given); empty selects
    all. An unknown name raises ``ValueError`` listing the config's names.
    """
    raw = yaml.safe_load(Path(config_filepath).read_text()) or {}
    settings = ReconstructSettings(**raw)
    unknown = sorted(set(names) - set(settings.reconstructions))
    if unknown:
        raise ValueError(
            f"{config_filepath} has no reconstruction named {unknown}; "
            f"its reconstructions are {list(settings.reconstructions)}"
        )
    return {
        name: Reconstruction(name, entry, raw["reconstructions"][name])
        for name, entry in settings.reconstructions.items()
        if not names or name in names
    }


@contextmanager
def reconstruction_config(reconstruction: Reconstruction) -> Iterator[Path]:
    """Write one reconstruction's waveorder config to a temporary YAML; yield its path."""
    with tempfile.TemporaryDirectory(prefix="biahub-reconstruct-") as tmpdir:
        path = Path(tmpdir) / f"{reconstruction.name}.yml"
        path.write_text(yaml.safe_dump(reconstruction.config, sort_keys=False))
        yield path


def transfer_function_path(transfer_functions_dirpath: Path, name: str) -> Path:
    """Return the path of reconstruction ``name``'s transfer function in the directory."""
    return Path(transfer_functions_dirpath) / f"{name}.zarr"


# Peak RAM of compute-tf for a confocal (pinhole) fluorescence transfer function, in
# bytes per voxel of the z-padded input volume. Measured at ~910 B/voxel on the mantis
# V2 (171 x 768 x 794 + 2 x 16 z-padding: 113 GB) and V1 (167 x 1024 x 960 + 2 x 16:
# 179 GB) neuromast stacks; 1100 leaves ~20% headroom.
_PINHOLE_TF_BYTES_PER_VOXEL = 1100
# Every other transfer function (phase, birefringence, widefield fluorescence) fits in
# the 64 GB the pipeline has always given compute-tf.
_DEFAULT_TF_MEM_GB = 64


def estimate_transfer_function_memory(
    reconstruction: Reconstruction, zyx_shape: tuple[int, int, int]
) -> int:
    """GB of RAM compute-tf needs for this reconstruction's transfer function."""
    fluorescence = reconstruction.settings.fluorescence
    if (
        fluorescence is None
        or fluorescence.transfer_function.confocal_pinhole_diameter is None
    ):
        return _DEFAULT_TF_MEM_GB
    Z, Y, X = zyx_shape
    padded_voxels = (Z + 2 * fluorescence.transfer_function.z_padding) * Y * X
    return max(
        _DEFAULT_TF_MEM_GB, math.ceil(padded_voxels * _PINHOLE_TF_BYTES_PER_VOXEL / 1e9)
    )
