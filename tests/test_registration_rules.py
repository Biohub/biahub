"""The rules of docs/registration/contributing.md that a test can check.

Each test is named in the guide next to its rule: a method, block or setting added without
everything the guide lists fails here, saying what is missing.
"""

import inspect
import io
import re
import textwrap
import tokenize
import typing

from pathlib import Path
from typing import get_args

from biahub.settings import (
    DEFAULT_SCORE_METRIC,
    METHOD_FAMILIES,
    METHOD_TRANSFORM_TYPES,
    EstimateTransformSettings,
    EstimationMethod,
)

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".claude" / "skills" / "register-with-engine"
METHODS = get_args(EstimationMethod)


def test_every_method_has_a_family_and_says_what_it_estimates():
    assert set(METHOD_FAMILIES) == set(METHODS)
    assert {family for family, _ in METHOD_FAMILIES.values()} == {
        "landmark-based",
        "intensity-based",
        "manual",
    }


def test_every_method_is_named_like_its_settings_section():
    # YAML values and keys follow one style: `method: ants` reads `ants:`
    sections = set(EstimateTransformSettings.model_fields)
    assert [m for m in METHODS if m not in sections] == []


def test_every_method_lives_in_its_familys_folder_under_its_own_name():
    # registration/methods/<family>/<method>.py: the family and the name a config uses
    import biahub.registration.methods as methods

    folders = {
        "landmark-based": "landmark",
        "intensity-based": "intensity",
        "manual": "manual",
    }
    root = Path(methods.__file__).parent
    for method, (family, _) in METHOD_FAMILIES.items():
        assert (root / folders[family] / f"{method}.py").is_file(), method


def test_every_method_has_a_default_score():
    assert set(DEFAULT_SCORE_METRIC) == set(METHODS)


# what a method needs besides its defaults: shapes its block, manual the timepoint to click
NEEDED = {
    "shapes": {"shapes": {"rotate90": 0, "fliplr": False}},
    "manual": {"time_indices": 0},
}


def test_every_method_builds_from_its_default_settings():
    from biahub.registration.engine import build_estimator
    from biahub.registration.estimators import EstimatorBase

    for method in METHODS:
        settings = EstimateTransformSettings(
            moving={"channel": "GFP"},
            reference={"frame": "cross", "channel": "Phase3D"},
            method=method,
            transform={"type": METHOD_TRANSFORM_TYPES.get(method, ("affine",))[0]},
            **NEEDED.get(method, {}),
        )
        estimator, _ = build_estimator(settings, (8, 32, 32), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
        assert isinstance(estimator, EstimatorBase), method


def test_the_skill_names_every_method():
    # a row in the "Which method" table and a template the agent copies
    table = (SKILL / "references" / "methods.md").read_text().split("## Which method")[1]
    table = table.split("\n## ")[0]
    for method in METHODS:
        template = f"templates/estimate-transform-{method}.yml"
        assert f"`{method}`" in table, f"no row for {method} in the skill's method table"
        assert template in table, f"the table does not point {method} to {template}"
        assert (SKILL / template).is_file(), f"no {template} in the skill"


# Role words other than moving / reference: `target` anywhere; `source` and `fixed`
# with a volume word (bare `source` is provenance; `fixed=` and `fixed_parameters` are
# ANTs' and ITK's own names). In identifiers, and in messages and comments.
VOLUME_WORDS = "channel|shape|volume|image|peaks|points|position|store|grid"
OTHER_ROLE_WORDS = re.compile(rf"(^|_)target(_|$)|(^|_)(source|fixed)_({VOLUME_WORDS})")
OTHER_ROLE_PROSE = re.compile(
    rf"\btargets?\b|\b(source|fixed) ({VOLUME_WORDS})s?\b", re.IGNORECASE
)


def _role_words(source: str, where: str, first_line: int = 1) -> list[str]:
    found = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        prose = token.type in (tokenize.STRING, tokenize.COMMENT)
        if (token.type == tokenize.NAME and OTHER_ROLE_WORDS.search(token.string)) or (
            prose and OTHER_ROLE_PROSE.search(token.string)
        ):
            found.append(f"{where}:{first_line + token.start[0] - 1} {token.string[:60]}")
    return found


def _registration_settings_classes() -> set[type]:
    """The settings classes an estimate-transform config is built from."""
    from pydantic import BaseModel

    seen, todo = set(), [EstimateTransformSettings]
    while todo:
        model = todo.pop()
        if model in seen:
            continue
        seen.add(model)
        for field in model.model_fields.values():
            for kind in _types_in(field.annotation):
                if isinstance(kind, type) and issubclass(kind, BaseModel):
                    todo.append(kind)
    return seen


def _types_in(annotation) -> list:
    args = typing.get_args(annotation)
    return [annotation] if not args else [t for a in args for t in _types_in(a)]


def test_the_two_volumes_are_moving_and_reference():
    found = []
    for folder in ("biahub/registration", "biahub/core"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            if "legacy" in path.parts:  # the legacy pipeline keeps its names
                continue
            found += _role_words(path.read_text(), str(path.relative_to(ROOT)))
    for path in (ROOT / "biahub" / "estimate_transform.py",):
        found += _role_words(path.read_text(), str(path.relative_to(ROOT)))
    for model in _registration_settings_classes():  # settings.py holds other commands' too
        lines, first_line = inspect.getsourcelines(model)
        found += _role_words(textwrap.dedent("".join(lines)), model.__name__, first_line)
    assert not found, "say moving / reference for the two volumes:\n" + "\n".join(found)


def test_every_method_but_shapes_starts_from_its_initial_transform():
    # repair, propagation and a stack's later blocks hand each method a start; a method
    # that ignored it would make them return the same answer
    from biahub.registration.engine import build_estimator

    for method in METHODS:
        settings = EstimateTransformSettings(
            moving={"channel": "GFP"},
            reference={"frame": "cross", "channel": "Phase3D"},
            method=method,
            transform={"type": METHOD_TRANSFORM_TYPES.get(method, ("affine",))[0]},
            **NEEDED.get(method, {}),
        )
        estimator, _ = build_estimator(settings, (8, 32, 32), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
        assert estimator.uses_initial_transform == (method != "shapes"), method


def test_the_score_metric_names_agree_everywhere():
    # the names live in settings.ScoreMetric, settings.INTENSITY_METRICS and metrics.py's
    # functions: a metric added to one place only is refused or misread as a bead metric
    from biahub.registration.metrics import INTENSITY_METRIC_FNS
    from biahub.settings import INTENSITY_METRICS, ScoreMetric

    assert set(INTENSITY_METRIC_FNS) == INTENSITY_METRICS
    assert set(get_args(ScoreMetric)) == INTENSITY_METRICS | {"overlap", "residual"}
