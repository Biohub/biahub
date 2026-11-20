import numpy as np
import pytest

from biahub.core.transform import Transform
from biahub.registration.engine.policies import (
    ConsensusInitialTransform,
    FixedInitialTransform,
    InitialTransformPolicy,
)


def test_fixed_initial_transform_returns_the_same_transform_for_every_t():
    transform = Transform.from_translation([1.0, 2.0, 3.0])
    policy = FixedInitialTransform(transform)
    for t in range(5):
        assert policy.initial_for(t) is transform


def test_consensus_initial_transform_raises_when_too_few_timepoints_score_well():
    from biahub.registration.estimators import CandidateUnavailableError

    history = {0: Transform.from_translation([1.0, 1.0, 1.0])}
    scores = {0: 0.9}
    policy = ConsensusInitialTransform(history=history, scores=scores, score_threshold=0.75)
    with pytest.raises(CandidateUnavailableError, match="only 1 timepoints"):
        policy.initial_for(3)


def test_consensus_initial_transform_is_the_median_transform_over_good_timepoints():
    history = {
        0: Transform.from_translation([1.0, 0.0, 0.0]),
        1: Transform.from_translation([2.0, 0.0, 0.0]),
        2: Transform.from_translation([3.0, 0.0, 0.0]),
        3: Transform.from_translation([100.0, 0.0, 0.0]),  # poorly scored, excluded
        4: Transform.from_translation([4.0, 0.0, 0.0]),
        5: Transform.from_translation([5.0, 0.0, 0.0]),
    }
    scores = {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.1, 4: 0.9, 5: 0.9}
    policy = ConsensusInitialTransform(
        history=history, scores=scores, score_threshold=0.75, min_good=5
    )

    initial_transform = policy.initial_for(t=3)

    np.testing.assert_allclose(initial_transform.matrix[:3, 3], [3.0, 0.0, 0.0])


def test_consensus_initial_transform_excludes_t_itself_even_if_it_scores_well():
    history = {
        0: Transform.from_translation([1.0, 0.0, 0.0]),
        1: Transform.from_translation([2.0, 0.0, 0.0]),
        2: Transform.from_translation([3.0, 0.0, 0.0]),
        3: Transform.from_translation([-1000.0, 0.0, 0.0]),
        4: Transform.from_translation([4.0, 0.0, 0.0]),
    }
    # t=3 has a (spuriously) good score but must never contaminate its own consensus
    # initial transform.
    scores = {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.99, 4: 0.9}
    policy = ConsensusInitialTransform(
        history=history, scores=scores, score_threshold=0.75, min_good=4
    )

    initial_transform = policy.initial_for(t=3)

    np.testing.assert_allclose(initial_transform.matrix[:3, 3], [2.5, 0.0, 0.0])


def test_a_file_initial_transform_gives_each_timepoints_and_asks_to_be_kept_unless_beaten():
    from biahub.registration.engine.policies import FileInitialTransform

    by_t = {
        0: Transform.from_translation([1.0, 2.0, 3.0]),
        4: Transform.from_translation([0.0, -1.0, 0.0]),
    }
    policy = FileInitialTransform(by_t)
    assert policy.initial_for(4) is by_t[4]
    assert (
        policy.keep_unless_beaten is True
    )  # an earlier result: the engine keeps it unless an estimate scores better
    # only an earlier result asks to be kept unless beaten
    for other in (FixedInitialTransform, ConsensusInitialTransform):
        assert other.keep_unless_beaten is False
    with pytest.raises(KeyError, match="no initial transform for t=2"):
        policy.initial_for(2)
    assert isinstance(policy, InitialTransformPolicy)
