"""biahub segment with a micro-sam 2 model: --init checks and a debug (in-process) run."""

import pytest
import torch
import yaml

from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from biahub.cli.main import cli


@pytest.fixture(autouse=True)
def run_in_tmp_path(monkeypatch, tmp_path):
    """The submitit folder is ./slurm_output: keep it out of the repo."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def plate_2d(create_custom_plate, tmp_path):
    """A projected (Z=1) plate: (T=3, C=3, Z=1, Y=5, X=6)."""
    return create_custom_plate(tmp_path, z_size=1)


@pytest.fixture
def cpu_device(monkeypatch):
    """Workers ask for the GPU; run the (fake) model on the CPU instead."""
    monkeypatch.setattr("biahub.segment.cellpose_device", lambda gpu: torch.device("cpu"))


@pytest.fixture
def microsam_config(tmp_path):
    def _write(pretrained_model="hvit_b_cells", engine="apg", downsample=2):
        path = tmp_path / f"segment_{pretrained_model.replace('/', '_')}_{engine}.yml"
        config = {
            "models": {
                "nuc": {
                    "backend": "microsam",
                    "pretrained_model": pretrained_model,
                    "channels": ["GFP"],
                    "engine": engine,
                    "downsample": downsample,
                }
            }
        }
        path.write_text(yaml.safe_dump(config))
        return path

    return _write


def _run(*args):
    return CliRunner().invoke(cli, ["segment", *map(str, args)])


def test_init_creates_the_label_image_and_fetches_the_weights(
    fake_microsam, plate_2d, microsam_config
):
    plate_path, _ = plate_2d
    position = plate_path / "A" / "1" / "0"

    result = _run("--init", "-i", position, "-c", microsam_config())

    assert result.exit_code == 0, result.output
    assert "Created labels/{nuc}" in result.output
    with open_ome_zarr(position, mode="r") as pos:
        assert pos.label_names() == ["nuc"]
    assert fake_microsam.fetched == ["hvit_b_cells", "hvit_b_cells_decoder"]
    assert fake_microsam.segmented == []  # --init segments nothing


def test_init_without_micro_sam_says_which_extra_to_install(
    no_microsam, plate_2d, microsam_config
):
    plate_path, _ = plate_2d
    position = plate_path / "A" / "1" / "0"

    result = _run("--init", "-i", position, "-c", microsam_config())

    assert result.exit_code != 0
    assert "biahub[microsam]" in str(result.exception) + result.output
    assert not (position / "labels").exists()


def test_init_rejects_an_unknown_microsam_model_before_creating_labels(
    fake_microsam, plate_2d, microsam_config
):
    plate_path, _ = plate_2d
    position = plate_path / "A" / "1" / "0"

    result = _run("--init", "-i", position, "-c", microsam_config(pretrained_model="vit_b_lm"))

    assert result.exit_code != 0
    assert "Unknown micro-sam model 'vit_b_lm'" in str(result.exception) + result.output
    assert not (position / "labels").exists()


def test_debug_run_fills_the_label_image(fake_microsam, cpu_device, plate_2d, microsam_config):
    plate_path, _ = plate_2d
    config = microsam_config()
    position = plate_path / "B" / "1" / "0"
    assert _run("--init", "-i", position, "-c", config).exit_code == 0

    result = _run("--cluster", "debug", "-i", position, "-c", config)

    assert result.exit_code == 0, result.output
    with open_ome_zarr(position, mode="r") as pos:
        label = pos.get_label("nuc")["0"][:]
        assert label.shape == (3, 1, 1, 5, 6)
        assert label.min() >= 1  # every pixel got a label from the stand-in, upsampled back
        attrs = dict(pos.get_label("nuc").zattrs)
    assert attrs["biahub-segment"]["models"]["nuc"]["backend"] == "microsam"
    assert len(fake_microsam.built) == 1  # one model load for the whole position
    assert len(fake_microsam.segmented) == 3  # one per timepoint
    assert fake_microsam.segmented[0][0].shape == (3, 3)  # 5 x 6 downsampled by 2


def test_bad_eval_args_stop_before_any_label_image_is_written(
    fake_microsam, cpu_device, plate_2d, tmp_path
):
    # Without --init (SLURM fan-out or a direct debug run), a bad key must still fail on
    # the head node, not in every GPU job.
    plate_path, _ = plate_2d
    position = plate_path / "A" / "1" / "0"
    config = tmp_path / "bad.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "nuc": {
                        "backend": "microsam",
                        "pretrained_model": "hvit_b_cells",
                        "channels": ["GFP"],
                        "engine": "amg",
                        "eval_args": {"min_size": 100},
                    }
                }
            }
        )
    )

    result = _run("--cluster", "debug", "-i", position, "-c", config)

    assert result.exit_code != 0
    assert "min_size" in str(result.exception) + result.output
    assert not (position / "labels").exists()
    assert fake_microsam.segmented == []


def test_init_refuses_a_downsample_larger_than_the_plane(
    fake_microsam, plate_2d, microsam_config
):
    plate_path, _ = plate_2d  # Y=5, X=6
    position = plate_path / "A" / "1" / "0"

    result = _run("--init", "-i", position, "-c", microsam_config(downsample=6))

    assert result.exit_code != 0
    assert "downsample" in str(result.exception) + result.output
    assert not (position / "labels").exists()
