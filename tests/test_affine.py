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


def _ants_transform(kind, parameters):
    transform = ants.new_ants_transform(transform_type=kind, dimension=3)
    transform.set_parameters(np.asarray(parameters, dtype=float))
    return transform


def test_every_linear_ants_transform_is_read_with_its_type():
    # Euler3D has 6 parameters and was read as a 2D affine; Similarity3D (7) raised; and
    # every result was labelled affine
    rng = np.random.default_rng(2)
    cases = {
        "Euler3DTransform": ([0.1, -0.2, 0.15, 3.0, -1.0, 2.0], "euclidean"),
        "Similarity3DTransform": ([0.05, -0.03, 0.02, 1.0, -2.0, 0.5, 1.1], "similarity"),
        "AffineTransform": (
            np.concatenate([np.eye(3).ravel() + 0.03, [1.0, 2.0, 3.0]]),
            "affine",
        ),
    }
    for kind, (parameters, expected_type) in cases.items():
        t_ants = _ants_transform(kind, parameters)
        transform = Transform.from_ants(t_ants)
        assert transform.ndim == 3 and transform.transform_type == expected_type, kind
        for point in rng.uniform(-20, 20, (5, 3)):
            np.testing.assert_allclose(
                transform.apply_points(point[None])[0], t_ants.apply_to_point(point), atol=1e-5
            )


def test_an_ants_estimate_is_labelled_with_the_type_it_fitted():
    # ANTs writes every linear fit as an AffineTransform; the estimate says what it fitted
    from scipy.ndimage import gaussian_filter, shift

    from biahub.registration.methods.intensity.ants import AntsEstimator

    rng = np.random.default_rng(3)
    ref = gaussian_filter(rng.random((16, 32, 32)), 2).astype(np.float32)
    mov = shift(ref, (0.0, 1.0, -1.0), order=1).astype(np.float32)
    for transform_type in ("euclidean", "similarity", "affine"):
        estimator = AntsEstimator.from_settings(None, transform_type)
        assert (
            estimator.estimate(mov, ref, Transform.identity(3)).transform_type
            == transform_type
        )
