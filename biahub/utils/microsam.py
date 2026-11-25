"""micro-sam 2 (Segment Anything for Microscopy 2) for `biahub segment` (``backend: microsam``).

micro_sam is an opt-in extra (``biahub[microsam]``): it pulls napari / PyQt6 and caps
pydantic. Every import of it is local, so biahub imports without it.

Weights are downloaded by micro-sam (pooch) to its cache, ``$MICROSAM_CACHEDIR`` or
``~/.cache/micro_sam``; `biahub segment --init` fetches them once on the head node. A
worker copies them to node-local scratch and points micro-sam's cache there
(:func:`stage_microsam_weights`), so the jobs do not all read the weights over NFS and
the temporary embeddings micro-sam writes for tiled images stay on the node.
"""

import getpass
import inspect
import logging
import os

from pathlib import Path
from typing import Any

import click
import numpy as np

from biahub.utils.cellpose import _copy_weight

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "micro-sam models need the micro_sam package. Install biahub's opt-in extra: "
    "`uv sync --extra microsam` (or `pip install 'biahub[microsam]'`)."
)

# AIS post-processing modes (micro-sam's UniSAM2 `generate(mode=...)`).
_AIS_MODES = ("sparse", "dense")


def _import_microsam():
    try:
        from micro_sam.v2 import automatic_segmentation, util
    except ImportError as error:
        raise ImportError(_INSTALL_HINT) from error
    return automatic_segmentation, util


def _keyword_parameters(function, skip: tuple[str, ...] = ()) -> set[str]:
    """Return the parameters of ``function`` that have a default (tunable keywords)."""
    return {
        name
        for name, p in inspect.signature(function).parameters.items()
        if p.default is not inspect.Parameter.empty and name not in skip
    }


def engine_eval_args(engine: str, mode: str = "sparse") -> set[str]:
    """Return the ``eval_args`` keys micro-sam's ``engine`` accepts.

    Read from the callables that receive them, so they follow the installed micro_sam:
    AIS forwards them to ``flow_instance_segmentation`` (``mode: sparse``, the default)
    or ``run_multicut`` (``mode: dense``), APG to ``AutomaticPromptGenerator.generate``,
    AMG to ``AutomaticMaskGenerationSegmenter.generate``. Arguments micro-sam or biahub
    set themselves are excluded.
    """
    _import_microsam()
    if engine == "ais":
        from micro_sam.v2 import postprocessing

        function = (
            postprocessing.run_multicut
            if mode == "dense"
            else postprocessing.flow_instance_segmentation
        )
        # `boundary` is a prediction array micro-sam passes itself, not a setting.
        return {"mode"} | _keyword_parameters(function, skip=("model_type", "boundary"))
    if engine == "apg":
        from micro_sam.v2.automatic_prompt_generation import AutomaticPromptGenerator

        return _keyword_parameters(
            AutomaticPromptGenerator.generate, skip=("verbose", "pbar_init", "pbar_update")
        )
    from micro_sam.v2.instance_segmentation import AutomaticMaskGenerationSegmenter

    return _keyword_parameters(
        AutomaticMaskGenerationSegmenter.generate, skip=("output_mode",)
    )


def check_microsam_model(
    pretrained_model: str, engine: str, eval_args: dict[str, Any] | None = None
) -> None:
    """Raise if micro-sam cannot run ``pretrained_model`` with ``engine`` and ``eval_args``.

    A checkpoint file is accepted as is (micro-sam reads the model size from it). A name
    must be a fine-tuned micro-sam 2 model, or a plain SAM2 backbone with ``engine: amg``
    (the backbones have no decoder, which AIS and APG need). ``eval_args`` keys must be
    parameters of the engine's post-processing (:func:`engine_eval_args`); otherwise
    every job would fail with a TypeError in micro-sam.
    """
    _, util = _import_microsam()
    if not Path(pretrained_model).is_file():
        finetuned, backbones = list(util.get_model_names()), list(util.SUPPORTED_MODELS)
        if pretrained_model in backbones:
            if engine != "amg":
                raise ValueError(
                    f"micro-sam model {pretrained_model!r} is a plain SAM2 backbone without "
                    f"a decoder: it supports engine 'amg' only, not {engine!r}. Use a "
                    f"fine-tuned model ({finetuned}) for 'ais' / 'apg'."
                )
        elif pretrained_model not in finetuned:
            raise ValueError(
                f"Unknown micro-sam model {pretrained_model!r}: not a file and not one of "
                f"{finetuned + backbones}. The micro-sam 1 models (vit_b_lm, ...) are not "
                "supported."
            )

    eval_args = eval_args or {}
    mode = eval_args.get("mode", "sparse")
    if engine == "ais" and mode not in _AIS_MODES:
        raise ValueError(
            f"eval_args.mode must be one of {_AIS_MODES} for engine 'ais', got {mode!r}."
        )
    valid = engine_eval_args(engine, mode)
    unknown = sorted(set(eval_args) - valid)
    if unknown:
        what = f"engine {engine!r}" + (f" (mode {mode!r})" if engine == "ais" else "")
        raise ValueError(
            f"eval_args {unknown} are not parameters of micro-sam's {what}. "
            f"Allowed: {sorted(valid)}."
        )


def warm_microsam_weights(pretrained_model: str) -> None:
    """Download a micro-sam model (and its decoder) into micro-sam's shared cache, once.

    Called on the head node (`--init`) so that N workers do not race for the download.
    Checkpoint files are skipped.
    """
    if Path(pretrained_model).is_file():
        return
    _, util = _import_microsam()
    registry = util.models()
    if pretrained_model in registry.registry:
        for name in (pretrained_model, f"{pretrained_model}_decoder"):
            if name in registry.registry:
                registry.fetch(name, progressbar=False)
    elif pretrained_model in util.SUPPORTED_MODELS:
        # Plain SAM2 backbones are not in the registry; micro-sam fetches them with its
        # own (private) helper when a model is built.
        get_checkpoint = getattr(util, "_get_checkpoint", None)
        if get_checkpoint is None:
            logger.warning(
                "Cannot prefetch SAM2 backbone %s with this micro_sam.", pretrained_model
            )
            return
        get_checkpoint(pretrained_model)


def _weight_files(pretrained_model: str) -> list[str]:
    """Return a model's cache file names: its weights and, if fine-tuned, its decoder."""
    if Path(pretrained_model).is_file():
        return []
    return [pretrained_model, f"{pretrained_model}_decoder"]


def stage_microsam_weights(pretrained_models) -> Path | None:
    """Copy micro-sam weights to node-local scratch and point micro-sam's cache there.

    Mirrors ``stage_cellpose_weights``: every job reading the shared cache over NFS at
    once is the access pattern that made cellpose loads fail with stale file handles.
    The weights (``models/<name>`` and ``models/<name>_decoder``) are copied to
    ``$TMPDIR/micro_sam-<user>`` once per node and ``MICROSAM_CACHEDIR`` is set to it,
    which micro-sam reads on every call. Its temporary embeddings for tiled images
    (``tmp_embeddings/``) then go to scratch too. Weights that are not in the shared
    cache yet are downloaded by micro-sam into scratch; checkpoint files are read where
    they are.

    Returns the scratch cache directory, or ``None`` if staging failed (micro-sam then
    keeps its shared cache, as before).
    """
    from micro_sam.util import get_cache_directory

    dest = Path(os.environ.get("TMPDIR", "/tmp")) / f"micro_sam-{getpass.getuser()}"
    source = Path(get_cache_directory())
    if source == dest:
        return dest  # already staged in this process, e.g. a second position on one worker
    try:
        (dest / "models").mkdir(parents=True, exist_ok=True)
        copied = []
        for model in dict.fromkeys(pretrained_models):
            for name in _weight_files(model):
                shared = source / "models" / name
                if shared.is_file():
                    _copy_weight(shared, dest / "models" / name)
                    copied.append(name)
    except OSError as exc:
        logger.warning("Not staging micro-sam weights from %s: %r", source, exc)
        return None
    os.environ["MICROSAM_CACHEDIR"] = os.fspath(dest)
    click.echo(f"micro-sam cache on node-local scratch: {dest} (staged {copied or 'nothing'})")
    return dest


def load_microsam_model(pretrained_model: str, engine: str, device) -> tuple[Any, Any]:
    """Build micro-sam's (predictor, segmenter) for 2D automatic segmentation."""
    automatic_segmentation, _ = _import_microsam()
    if Path(pretrained_model).is_file():
        kwargs = {"checkpoint": pretrained_model}
    else:
        kwargs = {"model_type": pretrained_model}
    return automatic_segmentation.get_predictor_and_segmenter(
        device=device, segmentation_mode=engine, **kwargs
    )


def segment_plane(
    plane: np.ndarray,
    predictor: Any,
    segmenter: Any,
    downsample: int = 1,
    **generate_kwargs,
) -> np.ndarray:
    """Segment one (Y, X) plane, downsampled by striding, and return labels on its grid.

    ``generate_kwargs`` (the model's ``eval_args``) apply to the downsampled plane, so
    sizes such as ``min_size`` are in downsampled pixels.
    """
    automatic_segmentation, _ = _import_microsam()
    small = np.ascontiguousarray(plane[::downsample, ::downsample])
    labels = automatic_segmentation.automatic_instance_segmentation(
        predictor, segmenter, input_path=small, ndim=2, verbose=False, **generate_kwargs
    )
    labels = np.asarray(labels)
    if downsample > 1:
        # Nearest upsampling: each label pixel covers a downsample x downsample block.
        labels = np.repeat(np.repeat(labels, downsample, axis=0), downsample, axis=1)
    return labels[: plane.shape[0], : plane.shape[1]].astype(np.uint32)
