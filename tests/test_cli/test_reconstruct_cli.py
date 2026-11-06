import json

import numpy as np
import pytest
import yaml

from click.testing import CliRunner
from iohub.ngff import TransformationMeta, open_ome_zarr

from biahub.cli.main import cli

PHASE = {
    "input_channel_names": ["Phase3D"],
    "time_indices": "all",
    "reconstruction_dimension": 3,
    "phase": {
        "transfer_function": {
            "wavelength_illumination": 0.450,
            "yx_pixel_size": 0.1,
            "z_pixel_size": 0.25,
            "z_padding": 0,
            "index_of_refraction_media": 1.3,
            "numerical_aperture_detection": 1.2,
            "numerical_aperture_illumination": 0.5,
            "invert_phase_contrast": False,
        },
        "apply_inverse": {
            "reconstruction_algorithm": "Tikhonov",
            "regularization_strength": 1e-3,
        },
    },
}

# Deconvolves the test plate's one channel, so it writes Phase3D_Density3D.
FLUORESCENCE = {
    "input_channel_names": ["Phase3D"],
    "time_indices": "all",
    "reconstruction_dimension": 3,
    "fluorescence": {
        "transfer_function": {
            "yx_pixel_size": 0.1,
            "z_pixel_size": 0.25,
            "wavelength_emission": 0.5,
            "confocal_pinhole_diameter": 4.0,
        },
        "apply_inverse": {"reconstruction_algorithm": "RL", "rl": {"iterations": 3}},
    },
}


def write_config(path, **reconstructions):
    path.write_text(yaml.dump({"reconstructions": reconstructions}, sort_keys=False))
    return path


def run(*args):
    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result


def resources_lines(output):
    return [
        json.loads(line.removeprefix("RESOURCES:"))
        for line in output.splitlines()
        if line.startswith("RESOURCES:")
    ]


@pytest.fixture()
def reconstruct_config(tmp_path):
    return write_config(tmp_path / "reconstruct.yml", phase=PHASE)


@pytest.fixture()
def reconstruct_plate(tmp_path):
    """Small plate with a Phase3D channel for reconstruction tests."""
    plate_path = tmp_path / "input.zarr"

    position_list = (("A", "1", "0"), ("B", "1", "0"))

    plate = open_ome_zarr(
        plate_path,
        layout="hcs",
        mode="w",
        channel_names=["Phase3D"],
    )

    for row, col, fov in position_list:
        position = plate.create_position(row, col, fov)
        # Match the pixel sizes declared in the configs so waveorder doesn't
        # emit a PixelSizeMismatchWarning during reconstruction.
        position.create_image(
            "0",
            np.random.uniform(1.0, 100.0, size=(1, 1, 5, 8, 8)).astype(np.float32),
            transform=[TransformationMeta(type="scale", scale=(1, 1, 0.25, 0.1, 0.1))],
        )

    plate.close()
    return plate_path


def test_apply_inv_tf_cli_init_only(tmp_path, reconstruct_plate, reconstruct_config):
    output_path = tmp_path / "output.zarr"

    result = run(
        "apply-inv-tf",
        "--init",
        "-i",
        f"{reconstruct_plate}/A/1/0",
        f"{reconstruct_plate}/B/1/0",
        "-c",
        reconstruct_config,
        "-o",
        output_path,
    )

    assert output_path.exists()
    assert "RESOURCES:" in result.output


def test_apply_inv_tf_init_sizes_each_reconstruction(tmp_path, reconstruct_plate):
    """--init writes every reconstruction's channels into one plate and emits one
    RESOURCES line per reconstruction, which the pipeline routes by name: a GPU
    reconstruction asks for a GPU and runs one worker."""
    config = write_config(
        tmp_path / "reconstruct.yml",
        phase=PHASE,
        gfp=FLUORESCENCE | {"device": "cuda"},
    )
    output_path = tmp_path / "output.zarr"

    result = run(
        "apply-inv-tf", "--init", "-i", f"{reconstruct_plate}/A/1/0",
        "-c", config, "-o", output_path,
    )  # fmt: skip

    with open_ome_zarr(output_path / "A" / "1" / "0") as reconstructed:
        assert reconstructed.channel_names == ["Phase3D", "Phase3D_Density3D"]
    phase, gfp = resources_lines(result.output)
    assert (phase["reconstruction"], phase["gpus"]) == ("phase", 0)
    assert (gfp["reconstruction"], gfp["gpus"]) == ("gfp", 1)
    assert gfp["tf_mem_gb"] >= 64


def test_apply_inv_tf_runs_only_the_named_reconstruction(tmp_path, reconstruct_plate):
    """-r runs one reconstruction of the config against its own transfer function
    and leaves the other's channels untouched."""
    config = write_config(tmp_path / "reconstruct.yml", phase=PHASE, gfp=FLUORESCENCE)
    position = f"{reconstruct_plate}/A/1/0"
    output_path = tmp_path / "output.zarr"
    tf_dir = tmp_path / "transfer_functions"

    run("apply-inv-tf", "--init", "-i", position, "-c", config, "-o", output_path)
    run("compute-tf", "-i", position, "-c", config, "-o", tf_dir, "-r", "gfp")
    assert sorted(p.name for p in tf_dir.iterdir()) == ["gfp.zarr"]
    run(
        "apply-inv-tf", "--cluster", "debug", "-i", position, "-t", tf_dir,
        "-c", config, "-o", output_path, "-r", "gfp",
    )  # fmt: skip

    with open_ome_zarr(output_path / "A" / "1" / "0") as reconstructed:
        phase, gfp = reconstructed["0"][0]
        assert np.all(phase == 0)
        assert np.any(gfp != 0)


def test_reconstruction_name_not_in_config_is_a_usage_error(
    tmp_path, reconstruct_plate, reconstruct_config
):
    result = CliRunner().invoke(
        cli,
        [
            "apply-inv-tf", "--init", "-i", f"{reconstruct_plate}/A/1/0",
            "-c", str(reconstruct_config), "-o", str(tmp_path / "output.zarr"), "-r", "gfp",
        ],
    )  # fmt: skip
    assert result.exit_code == 2
    assert "no reconstruction named ['gfp']" in result.output
    assert not (tmp_path / "output.zarr").exists()


def test_reconstruct_config_rejects_two_reconstructions_writing_one_channel():
    from biahub.settings import ReconstructSettings

    with pytest.raises(ValueError, match="both write channel 'Phase3D_Density3D'"):
        ReconstructSettings(reconstructions={"a": FLUORESCENCE, "b": FLUORESCENCE})


def test_reconstruct_config_points_a_bare_waveorder_config_at_reconstructions():
    from biahub.settings import ReconstructSettings

    with pytest.raises(ValueError, match="nest this waveorder config under a name"):
        ReconstructSettings(**PHASE)


def test_apply_inv_tf_cli_debug_single_position(
    tmp_path, reconstruct_plate, reconstruct_config
):
    """Exercise biahub's --cluster debug submitit fan-out path.

    The reconstruction math is covered upstream in waveorder; here we only
    assert that biahub's debug orchestration runs to completion against a real
    position.
    """
    position = f"{reconstruct_plate}/A/1/0"
    output_path = tmp_path / "output.zarr"
    tf_path = tmp_path / "transfer_functions"

    # Setup: init the output plate, then compute a TF to feed the debug run.
    run("apply-inv-tf", "--init", "-i", position, "-c", reconstruct_config, "-o", output_path)
    run("compute-tf", "-i", position, "-c", reconstruct_config, "-o", tf_path)

    # Apply inv TF in debug mode (the biahub-specific in-process fan-out).
    result = run(
        "apply-inv-tf", "--cluster", "debug", "-i", position, "-t", tf_path,
        "-c", reconstruct_config, "-o", output_path,
    )  # fmt: skip
    assert "Apply-inv-tf complete:" in result.output


@pytest.mark.parametrize(
    "flag, expected", [([], False), (["--resume"], True), (["--no-resume"], False)]
)
def test_apply_inv_tf_cli_passes_resume(
    tmp_path, reconstruct_plate, reconstruct_config, monkeypatch, flag, expected
):
    """--resume reaches waveorder's per-position call, which does the skipping."""
    calls = []
    monkeypatch.setattr(
        "biahub.apply_inverse_transfer_function.apply_inverse_transfer_function_single_position",
        lambda *args, **kwargs: calls.append(kwargs),
    )

    run(
        "apply-inv-tf", "--cluster", "debug", "-i", f"{reconstruct_plate}/A/1/0",
        "-t", tmp_path / "transfer_functions", "-c", reconstruct_config,
        "-o", tmp_path / "output.zarr", *flag,
    )  # fmt: skip

    assert [call["resume"] for call in calls] == [expected]


def test_apply_inv_tf_cli_resume_skips_finished_timepoints(tmp_path, reconstruct_config):
    """End to end through biahub's init and --cluster debug path: a --resume rerun
    keeps the timepoints an earlier run finished instead of recomputing them."""
    plate_path = tmp_path / "input.zarr"
    with open_ome_zarr(plate_path, layout="hcs", mode="w", channel_names=["Phase3D"]) as plate:
        plate.create_position("A", "1", "0").create_image(
            "0",
            np.random.default_rng(0)
            .uniform(1.0, 100.0, size=(3, 1, 5, 8, 8))
            .astype(np.float32),
            transform=[TransformationMeta(type="scale", scale=(1, 1, 0.25, 0.1, 0.1))],
        )
    position = str(plate_path) + "/A/1/0"
    output_path = tmp_path / "output.zarr"
    tf_path = tmp_path / "transfer_functions"

    run("apply-inv-tf", "--init", "-i", position, "-c", reconstruct_config, "-o", output_path)
    run("compute-tf", "-i", position, "-c", reconstruct_config, "-o", tf_path)
    apply = (
        "apply-inv-tf", "--cluster", "debug", "-i", position, "-t", tf_path,
        "-c", reconstruct_config, "-o", output_path,
    )  # fmt: skip
    run(*apply)
    with open_ome_zarr(output_path / "A" / "1" / "0", mode="r+") as result:
        first = result["0"][:]
        result["0"][0] = 123.0  # stands in for "already finished; must not be recomputed"

    run(*apply, "--resume")
    with open_ome_zarr(output_path / "A" / "1" / "0") as result:
        np.testing.assert_array_equal(result["0"][0], 123.0)
        np.testing.assert_array_equal(result["0"][1:], first[1:])

    run(*apply, "--no-resume")
    with open_ome_zarr(output_path / "A" / "1" / "0") as result:
        np.testing.assert_array_equal(result["0"][:], first)


def test_apply_inv_tf_init_keeps_one_provenance_key_per_reconstruction(
    tmp_path, reconstruct_plate
):
    """Each reconstruction of a config keeps its settings under its own top-level
    key, and a downstream plate carries all of them forward."""
    from iohub.ngff.utils import create_empty_plate

    from biahub.settings import ReconstructSettings
    from biahub.utils.ngff import PROVENANCE_METADATA_KEYS

    config = write_config(tmp_path / "reconstruct.yml", phase=PHASE, gfp=FLUORESCENCE)
    settings = ReconstructSettings(**yaml.safe_load(config.read_text())).reconstructions
    output_path = tmp_path / "output.zarr"

    run(
        "apply-inv-tf", "--init", "-i", f"{reconstruct_plate}/A/1/0",
        "-c", config, "-o", output_path,
    )  # fmt: skip

    with open_ome_zarr(output_path / "A" / "1" / "0") as reconstructed:
        assert reconstructed.zattrs["waveorder-Phase3D"] == settings["phase"].model_dump()
        assert (
            reconstructed.zattrs["waveorder-Phase3D_Density3D"] == settings["gfp"].model_dump()
        )
        assert "waveorder" not in reconstructed.zattrs

    downstream_path = tmp_path / "downstream.zarr"
    create_empty_plate(
        store_path=downstream_path,
        position_keys=[("A", "1", "0")],
        channel_names=["Phase3D"],
        shape=(1, 1, 5, 8, 8),
        metadata_sources=output_path,
        metadata_keys=PROVENANCE_METADATA_KEYS,
    )
    with open_ome_zarr(downstream_path / "A" / "1" / "0") as downstream:
        assert {"waveorder-Phase3D", "waveorder-Phase3D_Density3D"} <= set(downstream.zattrs)


def test_legacy_waveorder_provenance_still_carries_forward(tmp_path):
    """Stores written before the switch keep the nested ``waveorder`` key."""
    from iohub.ngff.utils import create_empty_plate

    from biahub.utils.ngff import PROVENANCE_METADATA_KEYS

    legacy_path = tmp_path / "legacy.zarr"
    legacy = {"Phase3D": {"input_channel_names": ["BF"]}}
    create_empty_plate(
        store_path=legacy_path,
        position_keys=[("A", "1", "0")],
        channel_names=["Phase3D"],
        shape=(1, 1, 2, 4, 4),
        extra_metadata={"waveorder": legacy},
    )
    downstream_path = tmp_path / "downstream.zarr"
    create_empty_plate(
        store_path=downstream_path,
        position_keys=[("A", "1", "0")],
        channel_names=["Phase3D"],
        shape=(1, 1, 2, 4, 4),
        metadata_sources=legacy_path,
        metadata_keys=PROVENANCE_METADATA_KEYS,
    )
    with open_ome_zarr(downstream_path / "A" / "1" / "0") as downstream:
        assert downstream.zattrs["waveorder"] == legacy
