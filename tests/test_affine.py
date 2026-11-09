import ants
import numpy as np

from biahub.core.transform import Transform


def _affine(rng):
    """A rotation, per-axis scale, shear and shift: every part of an affine at once."""
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    matrix = np.eye(4)
    matrix[:3, :3] = q @ np.diag([0.9, 1.05, 1.1]) + 0.02 * rng.normal(size=(3, 3))
    matrix[:3, 3] = rng.uniform(-10, 10, 3)
    return Transform(matrix)


def test_a_transform_survives_the_round_trip_through_ants():
    rng = np.random.default_rng(0)
    for _ in range(20):
        transform = _affine(rng)
        np.testing.assert_allclose(
            Transform.from_ants(transform.to_ants()).matrix, transform.matrix, atol=1e-10
        )


def test_an_ants_transform_with_a_centre_maps_points_as_ants_does():
    # ANTs keeps a centre of rotation (its fixed parameters); from_ants folds it in.
    # ANTs holds its parameters in float32: agreement to its precision
    rng = np.random.default_rng(1)
    t_ants = ants.new_ants_transform(transform_type="AffineTransform", dimension=3)
    t_ants.set_parameters(
        np.concatenate([np.eye(3).ravel() + 0.05 * rng.normal(size=9), rng.uniform(-5, 5, 3)])
    )
    t_ants.set_fixed_parameters([10.0, -4.0, 7.0])
    transform = Transform.from_ants(t_ants)
    for point in rng.uniform(-20, 20, (10, 3)):
        np.testing.assert_allclose(
            transform.apply_points(point[None])[0], t_ants.apply_to_point(point), atol=1e-5
        )
