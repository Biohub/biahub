"""Survey positions before registering: which are usable, and how they differ."""

import numpy as np

from iohub import open_ome_zarr

UM = (0.2, 0.1, 0.1)


def _plate(tmp_path):
    """Four positions: cells everywhere, cells in one corner, nothing written, all NaN."""
    from tests.test_registration_estimators import _synthetic_blob_volume

    rng = np.random.default_rng(0)
    spread = _synthetic_blob_volume(rng, (24, 256, 256), n_blobs=300, sigma=3.0) + 10
    corner = np.full_like(spread, 10.0)
    corner[:, :80, :80] = spread[:, :80, :80]
    path = tmp_path / "plate.zarr"
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=["DAPI", "VS"]) as plate:
        for fov, vol in (("0", spread), ("1", corner), ("3", np.full_like(spread, np.nan))):
            plate.create_position("A", "1", fov)["0"] = np.stack([vol, vol])[None]
        plate.create_position("A", "1", "2").create_zeros(
            "0", shape=(1, 2, *spread.shape), dtype=np.float32
        )
    return [path / "A" / "1" / fov for fov in "0123"]


def test_the_survey_names_unusable_positions_and_how_structure_is_spread(tmp_path):
    from biahub.registration.benchmark import survey_positions

    rows = {r["position"]: r for r in survey_positions(_plate(tmp_path), ["DAPI", "VS"], t=0)}
    assert rows["A/1/2"]["empty"] and rows["A/1/3"]["nan_fraction"] == 1.0
    assert not rows["A/1/0"]["empty"] and rows["A/1/0"]["nan_fraction"] == 0.0
    # structure spread over the field constrains scale and rotation; a corner does not
    assert rows["A/1/0"]["spread"] > 0.8 > 0.3 > rows["A/1/1"]["spread"]
    assert rows["A/1/0"]["foreground"] > rows["A/1/1"]["foreground"]
    # the depth of peak signal, per channel
    assert set(rows["A/1/0"]["peak_z"]) == {"DAPI", "VS"}


def test_the_survey_reads_the_reference_store_and_skips_positions_it_lacks(tmp_path):
    # moving and reference are usually two stores (DragonFly: 0-convert DAPI, 2-VS nuclei):
    # the survey read the moving one only, and picked positions where the VS was blank
    from biahub.registration.benchmark import select_positions, survey_positions
    from tests.test_registration_estimators import _synthetic_blob_volume

    rng = np.random.default_rng(1)
    cells = _synthetic_blob_volume(rng, (24, 256, 256), n_blobs=300, sigma=3.0) + 10
    moving, reference = tmp_path / "m.zarr", tmp_path / "r.zarr"
    with open_ome_zarr(moving, layout="hcs", mode="w", channel_names=["DAPI"]) as plate:
        for fov in "012":
            plate.create_position("A", str(int(fov) + 1), fov)["0"] = cells[None, None]
    with open_ome_zarr(reference, layout="hcs", mode="w", channel_names=["VS"]) as plate:
        for fov in "012":  # the VS of the best-looking well is blank
            volume = np.zeros_like(cells) if fov == "0" else cells
            plate.create_position("A", str(int(fov) + 1), fov)["0"] = volume[None, None]
    keys = ["A/1/0", "A/2/1", "A/3/2"]
    rows = survey_positions(
        [moving / k for k in keys], ["DAPI"],
        reference_positions=[reference / k for k in keys], reference_channel="VS",
    )  # fmt: skip
    blank = {r["position"]: r["reference_blank"] for r in rows}
    assert blank == {"A/1/0": True, "A/2/1": False, "A/3/2": False}
    estimate, held_out = select_positions(rows, n_held_out=2)
    assert "A/1/0" not in [estimate, *held_out]


def test_the_survey_counts_nuclei_when_asked(tmp_path):
    from iohub.ngff.models import TransformationMeta

    from biahub.registration.benchmark import survey_positions
    from tests.test_registration_evaluate import VOX3, _disk_nuclei

    path = tmp_path / "nuclei.zarr"
    volume = _disk_nuclei()
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=["DAPI"]) as plate:
        position = plate.create_position("A", "1", "0")
        scale = TransformationMeta(type="scale", scale=[1.0, 1.0, *VOX3])
        position.create_image("0", volume[None, None], transform=[scale])
    (row,) = survey_positions([path / "A" / "1" / "0"], ["DAPI"], nuclei=True)
    assert row["n_nuclei"] > 5
