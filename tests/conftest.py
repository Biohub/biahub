import os
import sys
import types

from pathlib import Path

import numpy as np
import pytest
import yaml

from iohub.ngff import TransformationMeta, open_ome_zarr

# Use submitit debug executor (in-process, no forking) for fast tests
os.environ["CI"] = "true"

# Run ITK (ANTs) on one thread for the whole suite. ITK reads this once, at its
# first threaded operation, so it must be set before any test module imports
# ants. The test volumes are tiny, so threading buys nothing, and with one
# thread a seeded ANTs registration is bit-repeatable -- which is what
# test_ants_estimates_are_repeatable checks in-process instead of in a fresh
# interpreter.
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

# iohub's default zarr implementation is a *process-global*. biahub relies on
# the ``zarrs-python`` (Rust) pipeline for its sharded zarr-v3 writes
# (flat-field, concatenate, ...). Importing ``cytoland``/``viscy_data`` runs
# ``viscy_data._zarr_codec`` at import time, which flips that global to
# ``zarr-python`` via ``set_default_implementation``. zarr-python's pure-Python
# pipeline mishandles orthogonally-indexed sharded rank-5 writes, so once any
# test imports cytoland (the virtual-stain config test, which also imports it
# at module scope during collection) every later sharded write in the same
# process fails with a broadcast ``shape mismatch``. In production each CLI
# command runs in its own process so the two never collide, but pytest imports
# them all together. Snapshot iohub's pristine default here -- conftest is
# imported before any test module, i.e. before cytoland is imported -- and
# restore it around every test below.
from iohub.core import registry as _iohub_registry  # noqa: E402

_IOHUB_DEFAULT_IMPLEMENTATION = _iohub_registry._default


@pytest.fixture(autouse=True)
def _restore_iohub_default_implementation():
    """Pin iohub's default zarr implementation for the duration of each test.

    See the module-level note above: keeps cytoland's import-time override from
    leaking into biahub's sharded zarr writes.
    """
    _iohub_registry.set_default_implementation(_IOHUB_DEFAULT_IMPLEMENTATION)
    try:
        yield
    finally:
        _iohub_registry.set_default_implementation(_IOHUB_DEFAULT_IMPLEMENTATION)


# These fixtures return paired
# - paths for testing CLIs
# - objects for testing underlying functions


@pytest.fixture(scope="function")
def example_deskew_settings():
    settings_path = "./settings/example_deskew_settings.yml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture(scope="function")
def example_concatenate_settings():
    settings_path = "./settings/example_concatenate_settings.yml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture(scope="function")
def example_stitch_settings():
    settings_path = "./settings/example_stitch_settings.yml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture(scope="function")
def example_track_settings():
    settings_path = "./settings/example_track_settings.yml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture(scope="function")
def example_process_with_config_settings():
    settings_path = "./settings/example_process_with_config_settings.yml"
    with open(settings_path) as file:
        settings = yaml.safe_load(file)
    yield settings_path, settings


@pytest.fixture()
def sbatch_file(tmp_path):
    filepath = tmp_path / "sbatch.txt"
    with open(filepath, "w") as f:
        f.write("#SBATCH --cpus-per-task=1\n")
        f.write("#SBATCH --array-parallelism=2\n")
        f.write("#LOCAL --cpus-per-task=1\n")
        f.write("#LOCAL --timeout-min=1\n")
    yield filepath


@pytest.fixture(scope="function")
def example_plate(tmp_path):
    """
    Example HCS plate with 3 positions and 6 channels and float32 data
    """
    plate_path = tmp_path / "plate.zarr"

    position_list = (
        ("A", "1", "0"),
        ("B", "1", "0"),
        ("B", "2", "0"),
    )

    # Generate input dataset

    plate_dataset = open_ome_zarr(
        plate_path,
        layout="hcs",
        mode="w",
        channel_names=["GFP", "RFP", "Phase3D", "Orientation", "Retardance", "Birefringence"],
    )

    # Lateral pixel size matches example_deskew_settings.yml (pixel_size_um:
    # 0.116) so deskew doesn't warn about a config/metadata scale mismatch. The
    # T scale is a real one (30 min, in seconds), not the default 1, so a step
    # that drops it fails its test.
    scale = (1800.0, 1, 1.0, 0.116, 0.116)
    for row, col, fov in position_list:
        position = plate_dataset.create_position(row, col, fov)
        position.create_image(
            "0",
            np.random.uniform(0.0, 255.0, size=(3, 6, 4, 5, 6)).astype(np.float32),
            transform=[TransformationMeta(type="scale", scale=scale)],
        )

    yield plate_path, plate_dataset


@pytest.fixture(scope="function")
def example_plate_2(tmp_path):
    """
    Example HCS plate with 3 positions and 2 channels and uint16 data
    """
    plate_path = tmp_path / "plate_2.zarr"

    position_list = (
        ("A", "1", "0"),
        ("B", "1", "0"),
        ("B", "2", "0"),
    )

    # Generate input dataset
    plate_dataset = open_ome_zarr(
        plate_path,
        layout="hcs",
        mode="w",
        channel_names=["GFP", "RFP"],
    )

    for row, col, fov in position_list:
        position = plate_dataset.create_position(row, col, fov)
        position["0"] = np.random.randint(
            100, np.iinfo(np.uint16).max, size=(3, 2, 4, 5, 6), dtype=np.uint16
        )
    yield plate_path, plate_dataset


@pytest.fixture(scope="function")
def create_custom_plate():
    """
    Factory fixture that creates an HCS plate with customizable channel names

    Returns a function that creates a plate with the specified channel names
    """

    def _create_plate(
        tmp_path,
        position_list=(("A", "1", "0"), ("B", "1", "0"), ("B", "2", "0")),
        channel_names=("GFP", "RFP", "Phase3D"),
        time_points=3,
        z_size=4,
        y_size=5,
        x_size=6,
    ):
        """
        Create a plate with custom channel names

        Args:
            tmp_path: Temporary path for the plate
            channel_names: List of channel names
            time_points: Number of time points (default: 3)
            z_size: Size of Z dimension (default: 4)
            y_size: Size of Y dimension (default: 5)
            x_size: Size of X dimension (default: 6)

        Returns:
            Tuple of (plate_path, plate_dataset)
        """
        plate_path = tmp_path / f"plate_custom_{'-'.join(channel_names)}.zarr"

        # Generate input dataset
        plate_dataset = open_ome_zarr(
            plate_path,
            layout="hcs",
            mode="w",
            channel_names=channel_names,
        )

        for row, col, fov in position_list:
            position = plate_dataset.create_position(row, col, fov)
            position["0"] = np.random.randint(
                100,
                np.iinfo(np.uint16).max,
                size=(time_points, len(channel_names), z_size, y_size, x_size),
                dtype=np.uint16,
            )

        return plate_path, plate_dataset

    return _create_plate


# --- cellpose stand-in shared by segment/track tests (the test extra does not install cellpose)


class RecordingModel:
    """Stand-in for cellpose.models.CellposeModel that records how it is used."""

    built = []
    evaluated = []

    def __init__(self, gpu=False, pretrained_model="cpsam_v2", model_type=None, device=None):
        # Like cellpose 4: model_type is accepted and ignored.
        self.pretrained_model, self.device = pretrained_model, device
        RecordingModel.built.append(pretrained_model)

    def eval(
        self,
        x,
        channels=None,
        channel_axis=None,
        z_axis=None,
        diameter=None,
        cellprob_threshold=0.0,
        flow_threshold=0.4,
        do_3D=False,
        anisotropy=None,
        min_size=15,
        stitch_threshold=0.0,
    ):
        # Cellpose 4 refuses a z axis for 2D processing (cellpose/transforms.py:491).
        if z_axis is not None and not do_3D and not stitch_threshold:
            raise ValueError("2D image processing selected, but z_axis is not None.")
        RecordingModel.evaluated.append(np.array(x))
        # Like cellpose: masks drop the channel axis and come back squeezed.
        shape = x.shape[1:] if channel_axis == 0 else x.shape
        return np.ones(shape, dtype=np.uint16).squeeze(), None, None


@pytest.fixture()
def fake_cellpose(monkeypatch, tmp_path):
    models = types.ModuleType("cellpose.models")
    models.CellposeModel = RecordingModel
    models.MODEL_NAMES = ["cpsam_v2", "cpsam", "cpdino", "cpdino-vitb"]
    models.get_user_models = lambda: []
    package = types.ModuleType("cellpose")
    package.models = models
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    # The cpdino models need dinov3 (checked by name); stand in for it like cellpose.
    monkeypatch.setitem(sys.modules, "dinov3", types.ModuleType("dinov3"))
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", str(tmp_path / "no-weights"))
    RecordingModel.built, RecordingModel.evaluated = [], []
    from biahub import segment

    segment._MODEL_CACHE.clear()  # models are cached per process
    return RecordingModel


# --- micro-sam 2 stand-in for segment tests (the test extra does not install micro_sam)


class RecordingMicroSam:
    """Stand-in for micro_sam (2.0.1 API): records models built, planes segmented, fetches.

    The post-processing callables carry the real parameter names (biahub reads their
    signatures to check ``eval_args``), and the cache directory honours
    MICROSAM_CACHEDIR like micro_sam.util.get_cache_directory.
    """

    MODEL_NAMES = ["hvit_t_cells", "hvit_s_cells", "hvit_b_cells", "hvit_l_cells"]
    BACKBONES = ["hvit_t", "hvit_s", "hvit_b", "hvit_l"]
    built = []  # (model_type, checkpoint, segmentation_mode, MICROSAM_CACHEDIR)
    segmented = []  # (image, generate_kwargs)
    fetched = []  # names downloaded (registry fetch or backbone checkpoint)

    @classmethod
    def get_predictor_and_segmenter(
        cls,
        model_type="hvit_t_cells",
        checkpoint=None,
        device=None,
        segmentation_mode=None,
        **_,
    ):
        cls.built.append(
            (model_type, checkpoint, segmentation_mode, os.environ.get("MICROSAM_CACHEDIR"))
        )
        return object(), object()

    @classmethod
    def automatic_instance_segmentation(
        cls, predictor, segmenter, input_path, ndim=None, verbose=True, **generate_kwargs
    ):
        assert ndim == 2
        cls.segmented.append((np.array(input_path), generate_kwargs))
        # One distinct id per pixel, so the tests can check where each label lands.
        return np.arange(1, input_path.size + 1, dtype=np.uint32).reshape(input_path.shape)

    class _Registry:
        registry = {
            **{
                n: "xxh128:0"
                for n in ["hvit_t_cells", "hvit_s_cells", "hvit_b_cells", "hvit_l_cells"]
            },
            **{f"{n}_decoder": "xxh128:0" for n in ["hvit_t_cells", "hvit_b_cells"]},
        }

        def fetch(self, name, progressbar=True):
            RecordingMicroSam.fetched.append(name)
            return f"/cache/{name}"

    @staticmethod
    def get_checkpoint(model_type="hvit_t"):
        RecordingMicroSam.fetched.append(model_type)
        return f"/cache/{model_type}"


# micro_sam 2.0.1's post-processing signatures (data arguments first).
def _flow_instance_segmentation(
    foreground,
    directed_distances,
    model_type=None,
    foreground_threshold=None,
    n_iter=None,
    dt=None,
    sigma=None,
    spacing=None,
    density_threshold=None,
    min_size=None,
    foreground_weight=None,
    n_threads=8,
    boundary=None,
    boundary_weight=None,
    boundary_mask_threshold=None,
):
    pass


def _run_multicut(
    boundary_map,
    distances,
    model_type=None,
    beta=None,
    density_threshold=None,
    n_iter=None,
    dt=None,
    sigma=None,
    n_threads=8,
):
    pass


class _AutomaticPromptGenerator:
    def generate(
        self,
        candidate_threshold=None,
        foreground_threshold=None,
        n_iter=None,
        dt=None,
        sigma=None,
        spacing=None,
        min_candidate_size=None,
        score_threshold=None,
        max_overlap=None,
        min_size=None,
        max_size_factor=None,
        refinement=None,
        refinement_kwargs=None,
        multimasking=False,
        n_objects_per_pass=1,
        early_stop_patience=None,
        propagation_waves=1,
        batch_size=1,
        n_threads=1,
        verbose=False,
        pbar_init=None,
        pbar_update=None,
    ):
        pass


class _AutomaticMaskGenerationSegmenter:
    def generate(
        self,
        min_object_size=0,
        max_object_size=None,
        with_background=True,
        output_mode="instance_segmentation",
    ):
        pass


@pytest.fixture()
def fake_microsam(monkeypatch, tmp_path):
    """micro_sam stand-in; its shared cache is tmp_path/microsam-shared (no weights in it)."""
    automatic = types.ModuleType("micro_sam.v2.automatic_segmentation")
    automatic.get_predictor_and_segmenter = RecordingMicroSam.get_predictor_and_segmenter
    automatic.automatic_instance_segmentation = (
        RecordingMicroSam.automatic_instance_segmentation
    )
    util = types.ModuleType("micro_sam.v2.util")
    util.get_model_names = lambda: list(RecordingMicroSam.MODEL_NAMES)
    util.SUPPORTED_MODELS = list(RecordingMicroSam.BACKBONES)
    util.models = RecordingMicroSam._Registry
    util._get_checkpoint = RecordingMicroSam.get_checkpoint
    postprocessing = types.ModuleType("micro_sam.v2.postprocessing")
    postprocessing.flow_instance_segmentation = _flow_instance_segmentation
    postprocessing.run_multicut = _run_multicut
    apg = types.ModuleType("micro_sam.v2.automatic_prompt_generation")
    apg.AutomaticPromptGenerator = _AutomaticPromptGenerator
    instance = types.ModuleType("micro_sam.v2.instance_segmentation")
    instance.AutomaticMaskGenerationSegmenter = _AutomaticMaskGenerationSegmenter
    top_util = types.ModuleType("micro_sam.util")
    top_util.get_cache_directory = lambda: Path(os.environ["MICROSAM_CACHEDIR"])
    v2 = types.ModuleType("micro_sam.v2")
    v2.automatic_segmentation, v2.util = automatic, util
    package = types.ModuleType("micro_sam")
    package.v2, package.util = v2, top_util
    for name, module in {
        "micro_sam": package,
        "micro_sam.util": top_util,
        "micro_sam.v2": v2,
        "micro_sam.v2.automatic_segmentation": automatic,
        "micro_sam.v2.util": util,
        "micro_sam.v2.postprocessing": postprocessing,
        "micro_sam.v2.automatic_prompt_generation": apg,
        "micro_sam.v2.instance_segmentation": instance,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("MICROSAM_CACHEDIR", str(tmp_path / "microsam-shared"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    RecordingMicroSam.built, RecordingMicroSam.segmented, RecordingMicroSam.fetched = (
        [],
        [],
        [],
    )
    from biahub import segment

    segment._MODEL_CACHE.clear()
    return RecordingMicroSam


@pytest.fixture()
def no_microsam(monkeypatch):
    """micro_sam is not installed: importing it fails as it would without the extra."""
    monkeypatch.setitem(sys.modules, "micro_sam", None)
    from biahub import segment

    segment._MODEL_CACHE.clear()
