"""micro-sam 2 (Segment Anything for Microscopy 2) for `biahub segment` (``backend: microsam``).

micro_sam is an opt-in extra (``biahub[microsam]``): it pulls napari / PyQt6 and caps
pydantic. Every import of it is local, so biahub imports without it.

Weights are downloaded by micro-sam (pooch) to its cache, ``$MICROSAM_CACHEDIR`` or
``~/.cache/micro_sam``; `biahub segment --init` fetches them once on the head node.
"""

from pathlib import Path
from typing import Any

import numpy as np

_INSTALL_HINT = (
    "micro-sam models need the micro_sam package. Install biahub's opt-in extra: "
    "`uv sync --extra microsam` (or `pip install 'biahub[microsam]'`)."
)


def _import_microsam():
    try:
        from micro_sam.v2 import automatic_segmentation, util
    except ImportError as error:
        raise ImportError(_INSTALL_HINT) from error
    return automatic_segmentation, util


def check_microsam_model(pretrained_model: str, engine: str) -> None:
    """Raise if micro-sam cannot run ``pretrained_model`` with ``engine``.

    A checkpoint file is accepted as is (micro-sam reads the model size from it). A name
    must be a fine-tuned micro-sam 2 model, or a plain SAM2 backbone with ``engine: amg``
    (the backbones have no decoder, which AIS and APG need).
    """
    _, util = _import_microsam()
    if Path(pretrained_model).is_file():
        return
    finetuned, backbones = list(util.get_model_names()), list(util.SUPPORTED_MODELS)
    if pretrained_model in backbones:
        if engine != "amg":
            raise ValueError(
                f"micro-sam model {pretrained_model!r} is a plain SAM2 backbone without a "
                f"decoder: it supports engine 'amg' only, not {engine!r}. Use a fine-tuned "
                f"model ({finetuned}) for 'ais' / 'apg'."
            )
        return
    if pretrained_model not in finetuned:
        raise ValueError(
            f"Unknown micro-sam model {pretrained_model!r}: not a file and not one of "
            f"{finetuned + backbones}. The micro-sam 1 models (vit_b_lm, ...) are not "
            "supported."
        )


def warm_microsam_weights(pretrained_model: str) -> None:
    """Download a micro-sam model (and its decoder) into micro-sam's cache, once.

    Called on the head node (`--init`) so that N workers do not race for the download.
    Checkpoint files and plain SAM2 backbones (downloaded when first built) are skipped.
    """
    if Path(pretrained_model).is_file():
        return
    _, util = _import_microsam()
    registry = util.models()
    if pretrained_model not in registry.registry:
        return
    for name in (pretrained_model, f"{pretrained_model}_decoder"):
        if name in registry.registry:
            registry.fetch(name, progressbar=False)


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
    """Segment one (Y, X) plane, downsampled by striding, and return labels on its grid."""
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
