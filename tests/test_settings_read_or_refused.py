"""Every setting a config accepts is read: one the run would ignore is refused, saying why."""

import pytest

from pydantic import ValidationError

from biahub.settings import EstimateTransformSettings


def _settings(method="phase_cross_corr", **kwargs):
    kwargs.setdefault("reference", {"frame": "cross", "channel": "Phase3D"})
    return EstimateTransformSettings(moving={"channel": "GFP"}, method=method, **kwargs)


def test_another_methods_block_is_refused():
    with pytest.raises(ValidationError, match="ants block.*method phase_cross_corr"):
        _settings(ants={})
    # a stack's blocks carry their own settings: no top-level block is read
    with pytest.raises(ValidationError, match="belong in the blocks"):
        _settings(method=[{"phase_cross_corr": {}}, {"ants": {}}], phase_cross_corr={})
    _settings(phase_cross_corr={})  # its own block is fine


def test_the_neighbour_window_is_given_with_neighbours_and_only_there():
    # window defaulted to 6 (a hidden choice: how many frames on each side), and was
    # accepted by every frame though only neighbours reads it
    with pytest.raises(ValidationError, match="'neighbours' needs its window"):
        _settings(reference={"frame": "neighbours"})
    with pytest.raises(ValidationError, match="window is read only by frame neighbours"):
        _settings(reference={"frame": "first", "window": 4})
    assert _settings(reference={"frame": "neighbours", "window": 4}).reference.window == 4


SHAPES = {"rotate90": 0, "fliplr": False}


def test_propagation_and_repair_are_refused_only_when_shapes_starts_the_run():
    # every method applies its start; shapes computes one, so a start passed to it is lost
    propagate = {"type": "affine", "initial": {"propagate": True}}
    with pytest.raises(ValidationError, match="shapes ignores where it starts"):
        _settings(method="shapes", shapes=SHAPES, transform=propagate)
    with pytest.raises(ValidationError, match="fallback.repair re-estimates"):
        _settings(
            method="shapes",
            shapes=SHAPES,
            transform={"type": "affine"},
            fallback={"repair": {}},
        )
    # the image methods start from it now (they were refused)
    _settings(
        method="focus_finding",
        reference={"frame": "first"},
        transform={"initial": {"propagate": True}},
    )
    _settings(method="phase_cross_corr", fallback={"repair": {}})
    _settings(method="ants", transform={"initial": {"propagate": True}})


@pytest.mark.parametrize(
    "method, block",
    [
        ("bead_matching", {}),
        ("ants", {}),
        ("phase_cross_corr", {}),
        ("phase_cross_corr", {"local": {}}),
        ("focus_finding", {}),
        ("nuclei_matching", {}),
        ("manual", {}),
        ("shapes", {"rotate90": 0, "fliplr": False}),
    ],
)
def test_the_settings_know_which_methods_ignore_where_they_start(method, block):
    # the settings cannot build an estimator; their record must match the estimators'
    from biahub.registration.engine.core import build_estimator

    settings = _settings(
        method=method,
        time_indices=0 if method == "manual" else "all",
        transform={"type": "affine" if method == "shapes" else "euclidean"},
        **{method: block},
    )
    estimator, _score = build_estimator(
        settings, (8, 64, 64), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)
    )
    assert settings.ignores_initial_transform == (not estimator.uses_initial_transform)


def test_a_transform_type_the_method_cannot_fit_is_refused():
    # the type labelled the file as something the method never fitted (focus finding
    # measures a translation; manual fits euclidean or similarity; shapes rescales)
    with pytest.raises(ValidationError, match="focus_finding fits euclidean only"):
        _settings(
            method="focus_finding", reference={"frame": "first"}, transform={"type": "affine"}
        )
    with pytest.raises(ValidationError, match="manual fits euclidean or similarity"):
        _settings(method="manual", time_indices=0, transform={"type": "affine"})
    with pytest.raises(ValidationError, match="shapes fits affine only"):
        _settings(method="shapes", shapes={"rotate90": 0, "fliplr": False})
    with pytest.raises(ValidationError, match="phase_cross_corr fits euclidean only"):
        _settings(transform={"type": "similarity"})
    _settings(
        phase_cross_corr={"local": {}}, transform={"type": "similarity"}
    )  # local fits it
    # each block of a stack with its own type
    with pytest.raises(ValidationError, match=r"block 1 \(focus_finding\)"):
        _settings(
            method=[{"focus_finding": {}}, {"ants": {}}],
            reference={"frame": "first"},
            transform={"type": "affine"},
        )


def test_the_bead_detections_are_named_for_the_volumes():
    peaks = {"threshold_abs": 50.0}
    beads = _settings(
        "bead_matching",
        bead_matching={"moving_peaks_settings": peaks, "reference_peaks_settings": peaks},
    ).bead_matching
    assert beads.moving_peaks_settings.threshold_abs == 50.0
    assert beads.reference_peaks_settings.threshold_abs == 50.0
    # the legacy pipeline's names: refused, naming the new key and the converter
    for old, new in [
        ("source_peaks_settings", "moving_peaks_settings"),
        ("target_peaks_settings", "reference_peaks_settings"),
    ]:
        with pytest.raises(ValidationError, match=f"{old} is now {new}.*convert-settings"):
            _settings("bead_matching", bead_matching={old: peaks})


def test_local_phase_correlation_refuses_the_whole_volume_options():
    # local tiles are one size and limit their own shift: these options were loaded and
    # never read
    for extra, name in (
        ({"maximum_shift": 1.5}, "maximum_shift"),
        ({"function_type": "custom_padding"}, "custom_padding"),
    ):
        with pytest.raises(ValidationError, match=f"{name}.*whole-volume"):
            _settings(phase_cross_corr={"local": {}, **extra})
    # each mode's own sub-pixel default: 1 for the whole volume, 10 for tiles
    assert _settings(phase_cross_corr={}).phase_cross_corr.effective_upsample_factor == 1
    assert (
        _settings(phase_cross_corr={"local": {}}).phase_cross_corr.effective_upsample_factor
        == 10
    )


def test_propagation_loads_for_phase_correlation_without_its_section():
    # the start check read phase_cross_corr.local before the section was filled in
    propagate = {"initial": {"propagate": True}}
    assert _settings(method="phase_cross_corr", transform=propagate).transform.propagates
    stack = [{"phase_cross_corr": {}}, {"ants": {}}]
    assert _settings(method=stack, transform=propagate).transform.propagates


def test_an_old_ants_sobel_filter_names_both_blocks_and_the_converter():
    # fixing it by hand from the message changed the score: the old score compared edges
    with pytest.raises(ValidationError) as error:
        _settings(method="ants", ants={"sobel_filter": True})
    message = str(error.value)
    assert "- sobel: {}" in message and "score:, preprocess: [{sobel: {}}]" in message
    assert "biahub convert-settings" in message


@pytest.mark.parametrize(
    "method, preprocess, message",
    [
        ("ants", [{"sobel": True}], "sobel's parameters are a mapping"),
        ([{"ants": True}], [], "block's settings are a mapping"),
        ([{"ants": {"transform": "affine"}}], [], "block's transform is a mapping"),
    ],
)
def test_a_malformed_block_is_refused_saying_how_to_write_it(method, preprocess, message):
    # these raised TypeError tracebacks, or listed the characters of 'affine'
    with pytest.raises(ValidationError, match=message):
        _settings(method=method, preprocess=preprocess)


def test_a_stacks_refusal_names_the_block_and_nothing_else():
    # pydantic's nested type names, an index and two links buried the one useful line
    with pytest.raises(ValidationError) as error:
        _settings(method=[{"beads": {}}])
    message = str(error.value)
    assert "block 1: method 'beads' is now 'bead_matching'" in message
    # pydantic's one wrapper line and link, nothing nested inside it
    assert "function-before" not in message and message.count("errors.pydantic.dev") == 1


def test_a_stack_block_that_cannot_fit_the_runs_type_is_told_to_set_its_own():
    # it was refused naming transform.type, which the block never wrote
    with pytest.raises(
        ValidationError, match=r"set `transform: \{type: euclidean\}` in this block"
    ):
        _settings(
            method=[{"phase_cross_corr": {}}, {"ants": {}}], transform={"type": "affine"}
        )


@pytest.mark.parametrize("text", ["", "- a\n- b\n", "just a string\n"])
def test_a_config_that_is_not_a_mapping_is_refused_saying_so(text, tmp_path):
    # it reached model(**data) and raised "argument after ** must be a mapping"
    from biahub.settings import load_estimate_transform_settings

    path = tmp_path / "config.yml"
    path.write_text(text)
    with pytest.raises(ValueError, match="must hold a mapping of settings"):
        load_estimate_transform_settings(path)
