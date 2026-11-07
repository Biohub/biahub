"""Registration helpers: shape matching."""

import numpy as np

from biahub.registration.utils import match_shape, match_shape_origin


def test_match_shape_pads_and_crops_each_axis_on_its_own():
    # one axis smaller, two larger: tuples compare lexicographically, so the per-axis
    # decision once read (8, 20, 20) < (10, 10, 10) as "smaller" and failed
    img = np.zeros((10, 10, 10))
    img[3, 4, 5] = 1.0
    out = match_shape(img, (8, 20, 20))
    assert out.shape == (8, 20, 20)
    # the input voxel lands where match_shape_origin says (reflect padding copies it too)
    landed = np.array([3, 4, 5]) + match_shape_origin((10, 10, 10), (8, 20, 20))
    assert out[tuple(landed)] == 1.0
    assert match_shape(img, (10, 10, 10)) is img  # nothing to do, no copy


def test_the_centred_window_takes_whole_axes_when_the_crop_is_larger_and_keeps_odd_sizes():
    from biahub.registration.utils import center_window_yx

    # crop_xy is (X, Y), the legacy center_crop_xy convention; slices are (y, x)
    assert center_window_yx((100, 80), (40, 21)) == (slice(40, 61), slice(20, 60))
    assert center_window_yx((100, 80), (150, 150)) == (slice(0, 100), slice(0, 80))


def test_a_result_recomputed_over_a_freed_arrays_entry_is_the_newest():
    # a new array at a freed one's address finds that entry dead and recomputes it; the
    # entry kept its old place, so the next result evicted it first
    import weakref

    from biahub.registration.utils import Remembered

    memo = Remembered(2)
    x, y, z = np.zeros(1), np.zeros(1), np.zeros(1)
    calls = []

    def compute(name):
        return lambda: calls.append(name) or name

    memo((x,), (), compute("x"))
    memo((y,), (), compute("y"))
    freed = np.zeros(1)
    dead = weakref.ref(freed)
    del freed
    memo._results[((id(x),), ())] = ((dead,), "freed")  # x's entry, as a freed array left it
    memo((x,), (), compute("x"))  # recomputed
    memo((z,), (), compute("z"))  # evicts the oldest: y, not the x just computed
    assert memo((x,), (), compute("x")) == "x"
    assert calls == ["x", "y", "x", "z"]
