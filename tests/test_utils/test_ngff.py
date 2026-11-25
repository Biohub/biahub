import pytest

from biahub.utils.ngff import output_time_scale


@pytest.mark.parametrize(
    "time_indices",
    [None, [3], [0, 1, 2, 3], range(2, 6)],
)
def test_output_time_scale_keeps_the_input_scale_for_consecutive_time_points(time_indices):
    assert output_time_scale(1800.0, time_indices) == 1800.0


@pytest.mark.parametrize(
    "time_indices, expected", [([0, 2, 4], 3600.0), (range(1, 10, 3), 5400.0)]
)
def test_output_time_scale_multiplies_by_an_even_step(time_indices, expected):
    assert output_time_scale(1800.0, time_indices) == expected


@pytest.mark.parametrize("time_indices", [[0, 5, 7], [4, 2, 0], [1, 1]])
def test_output_time_scale_warns_and_keeps_the_input_scale_without_one_step(time_indices):
    with pytest.warns(UserWarning, match="not evenly spaced"):
        assert output_time_scale(1800.0, time_indices) == 1800.0
