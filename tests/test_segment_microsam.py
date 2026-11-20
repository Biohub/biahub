"""micro-sam 2 as a `biahub segment` backend: config schema and what segment hands to it.

The test extra does not install micro_sam; ``fake_microsam`` (conftest) records the
models built and the images segmented.
"""

import numpy as np
import pytest

from pydantic import ValidationError


def _settings(**model):
    from biahub.settings import SegmentationSettings

    base = {"backend": "microsam", "pretrained_model": "hvit_b_cells", "channels": ["nuc"]}
    return SegmentationSettings(models={"nucleus": {**base, **model}})


def _segment(czyx, channels, **model):
    """Run _segment_czyx on ``czyx`` whose channels are named c0, c1, ..."""
    from biahub.segment import _segment_czyx, resolve_models
    from biahub.settings import SegmentationSettings

    names = [f"c{i}" for i in range(czyx.shape[0])]
    settings = SegmentationSettings(
        models={"nuc": {"channels": [names[c] for c in channels], **model}}
    )
    models = resolve_models(settings, names, scale=(1,) * 5, z_size=czyx.shape[1])
    return _segment_czyx(czyx, models, gpu=False)


# --- config


def test_microsam_model_defaults():
    m = _settings().models["nucleus"]
    assert m.backend == "microsam"
    assert m.engine == "ais"  # micro-sam's own default
    assert m.downsample == 1
    assert m.eval_args == {}
    assert not m.is_3d


def test_a_config_without_backend_is_still_cellpose(fake_cellpose):
    from biahub.settings import SegmentationSettings

    s = SegmentationSettings(
        models={"nuc": {"pretrained_model": "cpsam_v2", "channels": ["n"]}}
    )
    assert s.models["nuc"].backend == "cellpose"


def test_cellpose_provenance_does_not_record_the_backend(fake_cellpose):
    # Label images written before the backend field existed record no `backend`; the
    # recorded settings (and the resume token) of a cellpose model must not change.
    from biahub.settings import SegmentationSettings

    s = SegmentationSettings(
        models={"nuc": {"pretrained_model": "cpsam_v2", "channels": ["n"]}}
    )
    assert "backend" not in s.model_dump(mode="json")["models"]["nuc"]


def test_microsam_provenance_records_the_backend():
    assert _settings().model_dump(mode="json")["models"]["nucleus"]["backend"] == "microsam"


def test_microsam_takes_one_channel():
    with pytest.raises(ValidationError, match="one channel"):
        _settings(channels=["nuc", "mem"])


@pytest.mark.parametrize("downsample", [0, -2])
def test_downsample_must_be_positive(downsample):
    with pytest.raises(ValidationError):
        _settings(downsample=downsample)


def test_unknown_engine_is_refused():
    with pytest.raises(ValidationError):
        _settings(engine="cellpose")


def test_microsam_options_are_refused_on_a_cellpose_model(fake_cellpose):
    from biahub.settings import SegmentationSettings

    with pytest.raises(ValidationError, match="engine"):
        SegmentationSettings(
            models={
                "nuc": {"pretrained_model": "cpsam_v2", "channels": ["n"], "engine": "apg"}
            }
        )


def test_unknown_backend_is_refused():
    with pytest.raises(ValidationError, match="backend"):
        _settings(backend="stardist")


def test_cellpose_and_microsam_models_can_share_a_config(fake_cellpose):
    from biahub.settings import SegmentationSettings

    s = SegmentationSettings(
        models={
            "cp": {"pretrained_model": "cpsam_v2", "channels": ["n"]},
            "msam": {
                "backend": "microsam",
                "pretrained_model": "hvit_b_cells",
                "channels": ["n"],
            },
        }
    )
    assert [m.backend for m in s.models.values()] == ["cellpose", "microsam"]
    assert not s.is_3d


def test_microsam_models_are_2d_only():
    from biahub.segment import resolve_models

    with pytest.raises(ValueError, match="2D"):
        resolve_models(_settings(), ["nuc"], scale=(1,) * 5, z_size=4)


# --- what segment hands to micro-sam


def test_segment_builds_the_configured_model_and_engine(fake_microsam):
    _segment(
        np.zeros((1, 1, 8, 8), np.float32),
        channels=[0],
        backend="microsam",
        pretrained_model="hvit_b_cells",
        engine="apg",
    )
    assert fake_microsam.built == [("hvit_b_cells", None, "apg")]


def test_a_checkpoint_path_is_passed_as_the_checkpoint(fake_microsam, tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"")
    _segment(
        np.zeros((1, 1, 8, 8), np.float32),
        channels=[0],
        backend="microsam",
        pretrained_model=str(ckpt),
    )
    ((model_type, checkpoint, _),) = fake_microsam.built
    assert checkpoint == str(ckpt)


def test_segment_passes_the_configured_plane_downsampled(fake_microsam):
    czyx = np.stack([np.full((1, 9, 12), c, np.float32) for c in (10, 20)])
    czyx[1, 0, ::3, ::3] = 99  # only the strided samples carry 99

    _segment(
        czyx, channels=[1], backend="microsam", pretrained_model="hvit_b_cells", downsample=3
    )

    ((image, _),) = fake_microsam.segmented
    assert image.shape == (3, 4)  # 2D plane, every 3rd pixel
    assert np.all(image == 99)


def test_segment_returns_labels_on_the_input_grid(fake_microsam):
    czyx = np.zeros((1, 1, 10, 13), np.float32)  # not a multiple of the factor
    labels = _segment(
        czyx, channels=[0], backend="microsam", pretrained_model="hvit_b_cells", downsample=4
    )
    assert labels.shape == (1, 1, 10, 13)  # (models, Z=1, Y, X)
    assert np.all(labels == 1)


def test_eval_args_are_forwarded_to_the_postprocessing(fake_microsam):
    _segment(
        np.zeros((1, 1, 8, 8), np.float32),
        channels=[0],
        backend="microsam",
        pretrained_model="hvit_b_cells",
        eval_args={"min_size": 100},
    )
    ((_, kwargs),) = fake_microsam.segmented
    assert kwargs == {"min_size": 100}


def test_segment_loads_a_microsam_model_once_across_timepoints(fake_microsam):
    from biahub.segment import _segment_czyx, resolve_models
    from biahub.settings import SegmentationSettings

    settings = SegmentationSettings(
        models={
            "nuc": {
                "backend": "microsam",
                "pretrained_model": "hvit_b_cells",
                "channels": ["c0"],
            }
        }
    )
    models = resolve_models(settings, ["c0"], scale=(1,) * 5, z_size=1)
    for _ in range(3):
        _segment_czyx(np.zeros((1, 1, 8, 8), np.float32), models, gpu=False)
    assert len(fake_microsam.built) == 1
    assert len(fake_microsam.segmented) == 3


def test_a_mixed_config_runs_each_backend(fake_cellpose, fake_microsam):
    from biahub.segment import _segment_czyx, resolve_models
    from biahub.settings import SegmentationSettings

    settings = SegmentationSettings(
        models={
            "cp": {"pretrained_model": "cpsam_v2", "channels": ["c0"]},
            "msam": {
                "backend": "microsam",
                "pretrained_model": "hvit_b_cells",
                "channels": ["c0"],
            },
        }
    )
    models = resolve_models(settings, ["c0"], scale=(1,) * 5, z_size=1)
    labels = _segment_czyx(np.zeros((1, 1, 8, 8), np.float32), models, gpu=False)
    assert labels.shape == (2, 1, 8, 8)
    assert fake_cellpose.built == ["cpsam_v2"]
    assert [b[0] for b in fake_microsam.built] == ["hvit_b_cells"]


# --- model checks (`--init`)


def test_check_accepts_a_finetuned_model(fake_microsam):
    from biahub.utils.microsam import check_microsam_model

    check_microsam_model("hvit_b_cells", "apg")


def test_check_refuses_an_unknown_name(fake_microsam):
    from biahub.utils.microsam import check_microsam_model

    with pytest.raises(ValueError, match="Unknown micro-sam model 'vit_b_lm'"):
        check_microsam_model("vit_b_lm", "ais")


def test_check_refuses_a_backbone_without_decoder_for_ais(fake_microsam):
    # The plain SAM2 backbones have no decoder: only AMG can use them.
    from biahub.utils.microsam import check_microsam_model

    with pytest.raises(ValueError, match="amg"):
        check_microsam_model("hvit_b", "ais")
    check_microsam_model("hvit_b", "amg")


def test_check_says_how_to_install_micro_sam(no_microsam):
    from biahub.utils.microsam import check_microsam_model

    with pytest.raises(ImportError, match=r"biahub\[microsam\]"):
        check_microsam_model("hvit_b_cells", "ais")


def test_warm_fetches_the_model_and_its_decoder(fake_microsam):
    from biahub.utils.microsam import warm_microsam_weights

    warm_microsam_weights("hvit_b_cells")
    assert fake_microsam.fetched == ["hvit_b_cells", "hvit_b_cells_decoder"]


def test_warm_skips_a_checkpoint_file(fake_microsam, tmp_path):
    from biahub.utils.microsam import warm_microsam_weights

    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"")
    warm_microsam_weights(str(ckpt))
    assert fake_microsam.fetched == []
