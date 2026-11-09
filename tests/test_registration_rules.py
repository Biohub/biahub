"""The rules of biahub/registration/CONTRIBUTING.md that a test can check.

Each test is named in the guide next to its rule: a method, block or setting added without
everything the guide lists fails here, saying what is missing.
"""

import io
import re
import tokenize

from pathlib import Path
from typing import get_args

from biahub.settings import (
    DEFAULT_SCORE_METRIC,
    METHOD_BLOCKS,
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
    # YAML values and keys follow one style: a method's name is its settings section
    assert {m: METHOD_BLOCKS[m] for m in METHODS} == {
        m: m for m in METHODS if m in METHOD_BLOCKS
    }


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


# Role words other than moving / reference, in an identifier: `target` anywhere; `source`
# and `fixed` with a volume word (bare `source` is provenance; `fixed=` and
# `fixed_parameters` are ANTs' and ITK's own names).
OTHER_ROLE_WORDS = re.compile(
    r"(^|_)target(_|$)|(^|_)(source|fixed)_(channel|shape|volume|image|peaks|points|position|store|grid)"
)


def test_the_two_volumes_are_moving_and_reference():
    found = []
    for folder in ("biahub/registration", "biahub/core"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            if "legacy" in path.parts:  # the legacy pipeline keeps its names
                continue
            for token in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
                if token.type == tokenize.NAME and OTHER_ROLE_WORDS.search(token.string):
                    found.append(f"{path.relative_to(ROOT)}:{token.start[0]} {token.string}")
    assert not found, "say moving / reference for the two volumes:\n" + "\n".join(found)
