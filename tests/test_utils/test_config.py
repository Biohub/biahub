import pytest

from biahub.settings import EvaluationSettings
from biahub.utils.config import yaml_to_model


@pytest.mark.parametrize(
    ("text", "says"),
    [("", "is empty"), ("# only a comment\n", "is empty"), ("- 1\n", "mapping")],
)
def test_a_settings_file_that_holds_no_mapping_is_named(tmp_path, text, says):
    # an empty file (e.g. `-e /dev/null`) failed with "argument after ** must be a
    # mapping, not NoneType", naming neither the file nor what was expected
    path = tmp_path / "evaluation.yml"
    path.write_text(text)
    with pytest.raises(ValueError, match=says) as error:
        yaml_to_model(path, EvaluationSettings)
    assert str(path) in str(error.value)


def test_a_settings_file_is_read_into_its_model(tmp_path):
    path = tmp_path / "evaluation.yml"
    path.write_text("tile_px: 128\n")
    assert yaml_to_model(path, EvaluationSettings).tile_px == 128
