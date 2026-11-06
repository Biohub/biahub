"""core.nodes: graphs, costs and solvers, usable on their own (arrays in, pairs out)."""

import numpy as np

from scipy.spatial.distance import cdist

from biahub.core.nodes.costs import spectral_affinity, weighted_cost
from biahub.core.nodes.graph import Graph
from biahub.core.nodes.matching import GraphMatcher
from biahub.core.nodes.solvers import cross_checked, linear_assignment, spectral_assignment


def test_linear_assignment_pairs_each_row_with_its_cheapest_free_column():
    costs = np.array([[1.0, 9.0, 9.0], [9.0, 9.0, 2.0], [9.0, 3.0, 9.0], [9.0, 9.0, 9.0]])
    pairs = linear_assignment(costs, cost_threshold=0.5)
    assert pairs.tolist() == [[0, 0], [1, 2], [2, 1]]  # row 3 only had costs at the cut


def test_cross_checked_keeps_pairs_both_directions_chose():
    forward = np.array([[0, 1], [1, 0], [2, 2]])
    backward = np.array([[1, 0], [0, 2], [2, 2]])  # (j, i): agrees on (0, 1) and (2, 2)
    assert cross_checked(forward, backward).tolist() == [[0, 1], [2, 2]]


def _clouds():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0, 100, (30, 3))
    mov = ref[::-1] + np.array([1.0, 2.0, -1.0])  # reversed order, shifted
    return mov, ref


def test_the_matchers_are_a_cost_and_a_solver():
    mov, ref = _clouds()
    mg, rg = Graph.from_nodes(mov, mode="knn", k=5), Graph.from_nodes(ref, mode="knn", k=5)
    hungarian = GraphMatcher(algorithm="hungarian")
    cost = weighted_cost(mg, rg, hungarian.weights)
    np.testing.assert_array_equal(
        hungarian.match(mg, rg), linear_assignment(cost, hungarian.cost_threshold)
    )
    spectral = GraphMatcher(algorithm="spectral")
    affinity = spectral_affinity(cdist(mov, mov), cdist(ref, ref), spectral.spectral_sigma)
    np.testing.assert_array_equal(
        spectral.match(mg, rg), spectral_assignment(affinity, len(mov), len(ref))
    )
    # reversed order: node i pairs with node 29 - i
    pairs = spectral.match(mg, rg)
    assert len(pairs) >= 28 and all(i + j == 29 for i, j in pairs)


def test_the_medoid_is_the_node_closest_to_all_the_others():
    from biahub.core.nodes.solvers import medoid

    points = np.array([[0.0], [1.0], [2.0], [10.0]])
    assert medoid(cdist(points, points)) == 1  # 1 + 1 + 9 beats 1 + 2 + 10 and 2 + 1 + 8


def test_mutual_nearest_pairs_nodes_that_are_each_others_closest():
    from biahub.core.nodes.solvers import mutual_nearest

    a = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    b = np.array([[0.5, 0.0], [10.4, 0.0], [10.8, 0.0], [50.0, 0.0]])
    i, j = mutual_nearest(a, b, radius=1.0)
    assert list(zip(i.tolist(), j.tolist(), strict=True)) == [(0, 0), (1, 1)]  # 2: none near


def test_filter_matches_skips_a_filter_whose_threshold_is_none_or_zero():
    # documented "None skips"; None ran the filter and crashed on `angles <= None`
    mov, ref = _clouds()
    mg, rg = Graph.from_nodes(mov, mode="knn", k=5), Graph.from_nodes(ref, mode="knn", k=5)
    matcher = GraphMatcher(algorithm="hungarian")
    matches = matcher.match(mg, rg)
    for threshold in (None, 0):
        kept = matcher.filter_matches(
            matches, mg, rg, angle_threshold=threshold, direction_threshold=threshold
        )
        np.testing.assert_array_equal(
            kept,
            matcher.filter_matches(matches, mg, rg, angle_threshold=0, direction_threshold=0),
        )


def _edge_cost_by_definition(moving, reference, attrs_of, default_cost):
    """Each pair's cost, one edge pair at a time: the mean of the best assignment."""
    from scipy.optimize import linear_sum_assignment

    mov_attrs, ref_attrs = attrs_of(moving), attrs_of(reference)
    cost = np.full((moving.n_nodes, reference.n_nodes), default_cost, dtype=np.float32)
    for i, i_nb in moving.neighbor_map.items():
        for j, j_nb in reference.neighbor_map.items():
            local = np.full((len(i_nb), len(j_nb)), default_cost, dtype=np.float32)
            for ii, ni in enumerate(i_nb):
                for jj, nj in enumerate(j_nb):
                    if (i, ni) in mov_attrs and (j, nj) in ref_attrs:
                        local[ii, jj] = abs(mov_attrs[(i, ni)] - ref_attrs[(j, nj)])
            rows, cols = linear_sum_assignment(local)
            cost[i, j] = local[rows, cols].mean()
    return cost


def test_the_edge_cost_is_the_mean_of_each_pairs_best_edge_assignment():
    # integer positions tie many edge lengths; a radius graph leaves some nodes with
    # fewer neighbours than others, and an isolated node with none
    from biahub.core.nodes.costs import edge_consistency_cost

    rng = np.random.default_rng(4)
    for dim, attr, attrs_of in [
        (3, "distance", lambda g: g.edge_distances),
        (2, "angle", lambda g: g.edge_angles),
    ]:
        ref = rng.integers(0, 40, size=(30, dim)).astype(float)
        ref[0] = 500.0  # no neighbour within the radius
        mov = ref[rng.permutation(30)][:25] + rng.integers(-1, 2, size=(25, dim))
        moving = Graph.from_nodes(mov, mode="knn", k=4)
        reference = Graph.from_nodes(ref, mode="radius", radius=12.0)
        # an edge without a value takes the default, as one missing from the dict would
        del reference.edge_distances[reference.edges[0]]
        if attr == "angle":
            del reference.edge_angles[reference.edges[1]]
        expected = _edge_cost_by_definition(moving, reference, attrs_of, 7.0)
        actual = edge_consistency_cost(moving, reference, attr, 7.0)
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected)
