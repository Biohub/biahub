"""The engine's results on five synthetic runs, against the ones recorded in tests/data.

A change to how results are computed (CONTRIBUTING: "Same results") shows here: a
refactor or a speed-up leaves registration_golden.json as it is. A change meant to move
results rewrites it, `BIAHUB_UPDATE_GOLDEN=1 pytest tests/test_registration_golden.py`,
and the diff of the file is part of the change's review.

  matrix     : an initial matrix near the truth
  shapes     : the initial transform from the stores' grids, on a turned moving volume
  propagate  : each timepoint from the previous one, on a drifting plate
  repair     : a timepoint too far off for the run's start, repaired from t-1
  blank      : a blank timepoint: nothing to repair it from, a stand-in
  file       : an earlier result as the start of three positions, residual score
"""

import json
import os

from pathlib import Path

import numpy as np
import pytest

from iohub import open_ome_zarr
from scipy.ndimage import shift as ndi_shift

from biahub.core.transform import Transform
from biahub.estimate_transform import estimate_transform
from biahub.settings import (
    EstimateTransformSettings,
    TransformEntry,
    TransformSettings,
    load_transform_settings,
)
from biahub.utils.config import model_to_yaml
from tests.test_estimate_transform import (
    APPLIED_SHIFT_ZYX,
    SHAPE,
    _synthetic_bead_volume,
    _write_plate,
)
from tests.test_initial_transform_file_and_pool import SHIFTS

GOLDEN = Path(__file__).parent / "data" / "registration_golden.json"
PEAKS = {"threshold_abs": 100, "nms_distance": 4, "min_distance": 0, "block_size": [8, 8, 8]}
BEADS = {"moving_peaks_settings": PEAKS, "reference_peaks_settings": PEAKS}
# results recorded on one machine may round differently on another (BLAS, FFT)
ATOL = 1e-6


@pytest.fixture(autouse=True)
def _in_process_jobs(monkeypatch):
    monkeypatch.setenv("CI", "true")  # jobs run in this process


def _config(**extra) -> dict:
    return {
        "moving": {"channel": "GFP"},
        "reference": {"frame": "cross", "channel": "Phase3D"},
        "method": "bead_matching",
        "bead_matching": BEADS,
        "fallback": {"repair": {}},
        **extra,
    }


def _run(tmp_path: Path, name: str, positions: list[Path], config: dict) -> dict:
    """Run one config on `positions`; return its entries and reports, rounded to 1e-9."""
    path = tmp_path / f"{name}.yml"
    model_to_yaml(EstimateTransformSettings(**config), path)
    output = tmp_path / name / "transforms.yml"
    estimate_transform(
        positions, path, output, reference_position_dirpaths=positions, cluster="debug"
    )
    model = load_transform_settings(output)
    lists = {None: model.transforms} if model.transforms is not None else model.positions
    entries = {
        f"{key}/{entry.t}": {
            "matrix": np.round(np.asarray(entry.matrix, float), 9).tolist(),
            "status": entry.status,
            "filled_from": entry.filled_from,
            "initialized_from": entry.initialized_from,
        }
        for key, items in lists.items()
        for entry in items
    }
    reports = {}
    for report in sorted(output.with_suffix("").rglob("estimate_transform_report.json")):
        content = json.loads(report.read_text())
        keys = ("scores", "errors", "flagged", "repairs", "filled_from", "repaired_from")
        reports[str(report.relative_to(output.with_suffix("")))] = {
            k: content[k] for k in keys if k in content
        }
    return {"entries": entries, "reports": reports}


def _results(tmp_path: Path) -> dict:
    rng = np.random.default_rng(11)
    ref = _synthetic_bead_volume(rng, SHAPE)
    mov = ndi_shift(ref, shift=APPLIED_SHIFT_ZYX, order=1, mode="constant", cval=0.0)
    beads = _write_plate(tmp_path / "beads.zarr", [(ref, mov), (ref, mov)])
    # the moving camera turned a quarter: shapes' rotate90 -1 turns it back
    turned = np.ascontiguousarray(np.rot90(mov, k=1, axes=(1, 2)))
    turned_plate = _write_plate(tmp_path / "turned.zarr", [(ref, turned), (ref, turned)])
    near = Transform.from_translation(np.add(APPLIED_SHIFT_ZYX, (0.5, 0.5, -0.5))).to_list()
    euclidean = {"type": "euclidean"}
    results = {
        "matrix": _run(
            tmp_path,
            "matrix",
            [beads],
            _config(
                transform={
                    **euclidean,
                    "initial": {"from": "matrix", "matrix": near, "direction": "inverse"},
                }
            ),
        ),
        "shapes": _run(
            tmp_path,
            "shapes",
            [turned_plate],
            _config(
                transform={
                    **euclidean,
                    "initial": {"from": "shapes", "rotate90": -1, "fliplr": False},
                }
            ),
        ),
    }
    drift = _write_plate(
        tmp_path / "drift.zarr",
        [
            (ref, ndi_shift(ref, shift=tuple(t * s for s in APPLIED_SHIFT_ZYX), order=1))
            for t in range(3)
        ],
    )
    qc = {"iterations": 2, "score_threshold": 0.4, "score_centroid_mask_radius": 1}
    results["propagate"] = _run(
        tmp_path,
        "propagate",
        [drift],
        _config(
            bead_matching={**BEADS, "qc_settings": qc},
            transform={
                **euclidean,
                "initial": {
                    "from": "matrix",
                    "matrix": np.eye(4).tolist(),
                    "direction": "inverse",
                    "propagate": True,
                },
            },
        ),
    )
    blank = rng.normal(0, 5.0, size=SHAPE).astype(np.float32)
    broken = _write_plate(tmp_path / "blank.zarr", [(ref, mov), (ref, mov), (ref, blank)])
    results["blank"] = _run(tmp_path, "blank", [broken], _config())
    # t=2 moved 7 voxels further in x: from the run's start too few beads match, from
    # t-1's result they do
    far = ndi_shift(ref, shift=(2.0, -3.0, 11.0), order=1, mode="constant", cval=0.0)
    jumped = _write_plate(tmp_path / "jump.zarr", [(ref, mov), (ref, mov), (ref, far)])
    results["repair"] = _run(
        tmp_path, "repair", [jumped], _config(transform={"type": "euclidean"})
    )

    three = tmp_path / "three.zarr"
    with open_ome_zarr(three, layout="hcs", mode="w", channel_names=["Phase3D", "GFP"]) as p:
        for fov, shift in SHIFTS.items():
            moved = ndi_shift(ref, shift=shift, order=1, mode="constant", cval=0.0)
            p.create_position("A", "1", fov)["0"] = np.stack([ref, moved])[None].astype(
                np.float32
            )
    plate_file = tmp_path / "plate_transforms.yml"
    model_to_yaml(
        TransformSettings(
            direction="inverse",
            moving_channels=["GFP"],
            reference_channel="Phase3D",
            transforms=[
                TransformEntry(matrix=Transform.from_translation((3.0, -3.0, 4.0)).to_list())
            ],
        ),
        plate_file,
    )
    results["file"] = _run(
        tmp_path,
        "file",
        [three / "A" / "1" / fov for fov in SHIFTS],
        _config(
            time_indices=0,
            score={"metric": "residual"},
            transform={**euclidean, "initial": {"from": "file", "file": str(plate_file)}},
        ),
    )
    return results


def _differences(expected, actual, where: str = "") -> list[str]:
    """Where `actual` differs from `expected`: numbers beyond ATOL, anything else at all."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = [f"{where}/{k}: missing" for k in expected.keys() - actual.keys()]
        out += [f"{where}/{k}: new" for k in actual.keys() - expected.keys()]
        for k in expected.keys() & actual.keys():
            out += _differences(expected[k], actual[k], f"{where}/{k}")
        return out
    if (
        isinstance(expected, list)
        and isinstance(actual, list)
        and len(expected) == len(actual)
    ):
        return [
            d
            for i, (e, a) in enumerate(zip(expected, actual, strict=True))
            for d in _differences(e, a, f"{where}[{i}]")
        ]
    numbers = (int, float)
    if isinstance(expected, numbers) and isinstance(actual, numbers):
        same = np.isclose(expected, actual, rtol=0, atol=ATOL, equal_nan=True)
        return [] if same else [f"{where}: {expected} -> {actual}"]
    return [] if expected == actual else [f"{where}: {expected!r} -> {actual!r}"]


def test_the_engine_gives_the_recorded_results(tmp_path):
    results = json.loads(json.dumps(_results(tmp_path)))  # as the file holds them
    if os.environ.get("BIAHUB_UPDATE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    differences = _differences(json.loads(GOLDEN.read_text()), results)
    assert not differences, "results changed:\n" + "\n".join(differences)
