"""The benchmark helpers the registration skill uses to choose and run a pipeline."""

import csv

import pytest


def _table(tmp_path, rows):
    """Rows of (case, pipeline, candidate, position, tiles, points[, points_n]); n=20 if absent."""
    path = tmp_path / "benchmark.csv"
    fields = ["case", "pipeline", "candidate", "position", "tiles_median_um"]
    fields += ["points_median_um", "points_n"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row = tuple(row) + ((20,) if len(row) == 6 else ())
            writer.writerow(dict(zip(fields, row, strict=True)))
    return path


def _benchmark(tmp_path, ants_points=(0.6, 0.5), tiles_ants_points=(0.5, 0.6)):
    rows = []
    for pipeline, points in (("ants", ants_points), ("tiles_ants", tiles_ants_points)):
        for k, position in enumerate(("A/1/0", "A/2/0")):
            rows.append(("c", pipeline, "unregistered", position, 3.0, 3.0))
            rows.append(("c", pipeline, "transforms", position, 0.5, points[k]))
            rows.append(
                ("c", pipeline, "1:block", position, 1.0, 2.0)
            )  # blocks are not choices
    return _table(tmp_path, rows)


def test_the_winner_is_the_lowest_median_on_held_out_positions(tmp_path):
    from biahub.registration.benchmark import select_pipeline

    choice = select_pipeline(_benchmark(tmp_path, ants_points=(1.0, 1.2)))
    assert choice["pipeline"] == "tiles_ants" and choice["metric"] == "points_median_um"
    assert choice["median_um"] == pytest.approx(0.55)
    assert "tiles_ants" in choice["reason"]


def test_a_tie_goes_to_the_simpler_pipeline(tmp_path):
    # equal within the tolerance: fewer blocks wins
    from biahub.registration.benchmark import select_pipeline

    choice = select_pipeline(_benchmark(tmp_path), blocks={"ants": 1, "tiles_ants": 2})
    assert choice["pipeline"] == "ants" and "simpler" in choice["reason"]


def test_tiles_decide_when_there_are_no_landmarks(tmp_path):
    from biahub.registration.benchmark import select_pipeline

    rows = [
        ("c", p, cand, "A/1/0", tiles, "")
        for p, tiles in (("ants", 0.9), ("tiles_ants", 0.4))
        for cand, tiles in (("unregistered", 3.0), ("transforms", tiles))
    ]
    choice = select_pipeline(_table(tmp_path, rows))
    assert (choice["pipeline"], choice["metric"]) == ("tiles_ants", "tiles_median_um")


def test_no_winner_when_nothing_clearly_beats_unregistered(tmp_path):
    from biahub.registration.benchmark import select_pipeline

    rows = [
        ("c", "ants", cand, "A/1/0", tiles, tiles)
        for cand, tiles in (("unregistered", 1.0), ("transforms", 0.9))
    ]
    choice = select_pipeline(_table(tmp_path, rows))
    assert choice["pipeline"] is None and "unregistered" in choice["reason"]


# --- choosing positions and writing the benchmark file ------------------------------------


def _survey():
    return [
        {"position": "A/1/0", "empty": False, "nan_fraction": 0.0, "spread": 0.9},
        {"position": "A/1/1", "empty": False, "nan_fraction": 0.0, "spread": 1.0},
        {"position": "A/2/0", "empty": False, "nan_fraction": 0.0, "spread": 0.8},
        {"position": "B/1/0", "empty": False, "nan_fraction": 0.0, "spread": 0.7},
        {"position": "B/1/1", "empty": True},
        {"position": "B/2/0", "empty": False, "nan_fraction": 0.6, "spread": 1.0},
    ]


def test_positions_are_chosen_from_the_survey_held_out_in_other_wells_first():
    from biahub.registration.benchmark import select_positions

    estimate, held_out = select_positions(_survey(), n_held_out=3)
    assert estimate == "A/1/1"  # the most spread out of the usable ones
    # other wells first (by spread), then the estimate's own well; never empty or mostly NaN
    assert held_out == ["A/2/0", "B/1/0", "A/1/0"]


def _store(tmp_path, name, channels):
    import numpy as np

    from iohub import open_ome_zarr

    path = tmp_path / name
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=channels) as plate:
        for well, fov in (("1", "0"), ("2", "0")):
            plate.create_position("A", well, fov)["0"] = np.zeros(
                (1, len(channels), 2, 4, 4), np.float32
            )
    return path


def test_the_benchmark_file_is_written_and_checked(tmp_path):
    import yaml

    from biahub.registration.benchmark import write_benchmark_config

    moving, reference = (
        _store(tmp_path, "m.zarr", ["DAPI"]),
        _store(tmp_path, "r.zarr", ["VS"]),
    )
    templates = (
        __import__("pathlib").Path(__file__).parents[1]
        / ".claude/skills/register-with-engine/templates/pipelines"
    )
    path = write_benchmark_config(
        tmp_path / "benchmark.yml",
        "case",
        moving,
        reference,
        "DAPI",
        "VS",
        "A/1/0",
        ["A/2/0"],
        {"local_pcc_ants": templates / "local_pcc_ants.yml", "ants": templates / "ants.yml"},
        landmarks="nuclei",
    )
    spec = yaml.safe_load(path.read_text())
    assert spec["cases"][0]["evaluate_positions"] == ["A/2/0"]
    assert [p["name"] for p in spec["pipelines"]] == ["local_pcc_ants", "ants"]

    with pytest.raises(ValueError, match="not in"):  # a position the store does not have
        write_benchmark_config(
            tmp_path / "b.yml",
            "case",
            moving,
            reference,
            "DAPI",
            "VS",
            "A/1/0",
            ["C/9/9"],
            {"ants": templates / "ants.yml"},
        )
    with pytest.raises(ValueError, match="held out"):  # estimated on is not held out
        write_benchmark_config(
            tmp_path / "b.yml",
            "case",
            moving,
            reference,
            "DAPI",
            "VS",
            "A/1/0",
            ["A/1/0"],
            {"ants": templates / "ants.yml"},
        )


def test_every_pipeline_template_loads():
    from pathlib import Path

    import yaml

    from biahub.settings import EstimateTransformSettings

    folder = (
        Path(__file__).parents[1] / ".claude/skills/register-with-engine/templates/pipelines"
    )
    names = sorted(p.stem for p in folder.glob("*.yml"))
    assert names == [
        "ants",
        "bead_matching",
        "local_pcc",
        "local_pcc_ants",
        "local_pcc_nuclei_matching",
    ]
    for path in folder.glob("*.yml"):
        EstimateTransformSettings(**yaml.safe_load(path.read_text()))


def test_points_decide_only_when_every_pipeline_matched_enough_nuclei(tmp_path):
    # A pipeline that matched 4 nuclei shows a small median from its few near pairs:
    # ranking then falls back to tiles, which every pipeline measured over the whole
    # field. Beating unregistered is judged on tiles too (its points match few nuclei).
    from biahub.registration.benchmark import select_pipeline

    rows = []
    for pipeline, tiles, points, n in (("good", 0.4, 0.6, 30), ("few", 2.0, 0.3, 4)):
        rows.append(("c", pipeline, "unregistered", "A/1/0", 3.0, 0.5, 3))
        rows.append(("c", pipeline, "transforms", "A/1/0", tiles, points, n))
    choice = select_pipeline(_table(tmp_path, rows))
    assert (choice["pipeline"], choice["metric"]) == ("good", "tiles_median_um")


# --- select_pipeline: the review's cases ----------------------------------------------------


def test_a_per_position_pipeline_is_scored_on_its_estimated_positions_transforms(tmp_path):
    # without pool, a held-out position is evaluated with each estimated position's
    # transform ("transforms (<position>)"): those rows were left out
    from biahub.registration.benchmark import select_pipeline

    rows = [("c", "ants", "unregistered", "A/3/0", 3.0, "")]
    rows += [
        ("c", "ants", f"transforms ({p})", "A/3/0", t, "")
        for p, t in (("A/1/0", 0.4), ("A/2/0", 0.6))
    ]
    choice = select_pipeline(_table(tmp_path, rows))
    assert choice["pipeline"] == "ants" and choice["median_um"] == pytest.approx(0.5)


def test_several_cases_are_chosen_for_one_at_a_time(tmp_path):
    from biahub.registration.benchmark import select_pipeline

    rows = [
        (case, "ants", cand, "A/1/0", tiles, "")
        for case, good in (("easy", 0.2), ("hard", 0.4))
        for cand, tiles in (("unregistered", 3.0), ("transforms", good))
    ]
    table = _table(tmp_path, rows)
    with pytest.raises(ValueError, match="easy.*hard"):
        select_pipeline(table)
    assert select_pipeline(table, case="hard")["median_um"] == pytest.approx(0.4)


def test_a_pipeline_that_failed_on_a_position_cannot_win(tmp_path):
    # b had no transform on the hard position: scored on the easy one only, it won
    from biahub.registration.benchmark import select_pipeline

    rows = [("c", p, "unregistered", pos, 3.0, "") for p in "ab" for pos in ("A/1/0", "A/2/0")]
    rows += [
        ("c", "a", "transforms", "A/1/0", 0.5, ""),
        ("c", "a", "transforms", "A/2/0", 0.9, ""),
    ]
    rows += [
        ("c", "b", "transforms", "A/1/0", 0.3, ""),
        ("c", "b", "transforms", "A/2/0", "", ""),
    ]
    choice = select_pipeline(_table(tmp_path, rows))
    assert (
        choice["pipeline"] == "a" and "b" in choice["reason"] and "A/2/0" in choice["reason"]
    )


def test_a_perfect_tile_median_wins_and_a_missing_one_is_said(tmp_path):
    from biahub.registration.benchmark import select_pipeline

    perfect = [
        ("c", "a", "unregistered", "A/1/0", 3.0, ""),
        ("c", "a", "transforms", "A/1/0", 0.0, ""),
    ]
    assert select_pipeline(_table(tmp_path, perfect))["pipeline"] == "a"
    # points decide, but the winner has no tile median to compare with unregistered
    no_tiles = [
        ("c", "a", "unregistered", "A/1/0", 3.0, "", 0),
        ("c", "a", "transforms", "A/1/0", "", 0.4, 20),
    ]
    choice = select_pipeline(_table(tmp_path, no_tiles))
    assert choice["pipeline"] is None and "no tile median" in choice["reason"]


def test_no_winner_without_an_unregistered_baseline(tmp_path):
    from biahub.registration.benchmark import select_pipeline

    rows = [("c", "a", "transforms", "A/1/0", 2.9, "")]
    choice = select_pipeline(_table(tmp_path, rows))
    assert choice["pipeline"] is None and "unregistered" in choice["reason"]


def test_the_reason_says_when_no_landmarks_were_measured(tmp_path):
    from biahub.registration.benchmark import select_pipeline

    rows = [
        ("c", "a", "unregistered", "A/1/0", 3.0, ""),
        ("c", "a", "transforms", "A/1/0", 0.4, ""),
    ]
    assert "no landmarks measured" in select_pipeline(_table(tmp_path, rows))["reason"]


def test_block_counts_come_from_the_table(tmp_path):
    # the tie rule needed counts no caller passed: a stack's blocks are in the table
    from biahub.registration.benchmark import select_pipeline

    rows = []
    for pipeline, tiles, blocks in (
        ("stack", 0.40, ("1:pcc", "2:ants")),
        ("single", 0.42, ()),
    ):
        rows += [
            ("c", pipeline, "unregistered", "A/1/0", 3.0, ""),
            ("c", pipeline, "transforms", "A/1/0", tiles, ""),
        ]
        rows += [("c", pipeline, block, "A/1/0", 1.0, "") for block in blocks]
    choice = select_pipeline(_table(tmp_path, rows))
    assert choice["pipeline"] == "single" and "simpler" in choice["reason"]


def test_the_benchmark_file_holds_what_the_workflow_needs(tmp_path):
    # it wrote stabilization cases the workflow refuses (no evaluate_t), a reference
    # without its channel, and landmarks unchecked
    import yaml

    from biahub.registration.benchmark import write_benchmark_config

    moving = _store(tmp_path, "m.zarr", ["DAPI"])
    reference = _store(tmp_path, "r.zarr", ["VS"])
    templates = (
        __import__("pathlib").Path(__file__).parents[1]
        / ".claude/skills/register-with-engine/templates/pipelines"
    )
    pipelines = {"ants": templates / "ants.yml"}

    def write(*, reference=None, reference_channel=None, **options):
        return write_benchmark_config(
            tmp_path / "b.yml",
            "case",
            moving,
            reference,
            "DAPI",
            reference_channel,
            "A/1/0",
            ["A/2/0"],
            pipelines,
            **options,
        )

    with pytest.raises(ValueError, match="evaluate_t"):
        write()
    with pytest.raises(ValueError, match="reference_channel"):
        write(reference=reference)
    with pytest.raises(ValueError, match="landmarks"):
        write(reference=reference, reference_channel="VS", landmarks="bead")
    case = yaml.safe_load(write(evaluate_t=10, reference_frame="previous").read_text())
    assert case["cases"][0]["evaluate_t"] == 10
    assert case["cases"][0]["reference_frame"] == "previous"
