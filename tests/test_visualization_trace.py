from nanotorch import manual_gradient, train_iter


def test_train_iter_trace_is_materializable(scenario_single_point):
    # Visualization tools often want a full trace they can scrub through.
    # This test ensures the iterator can be materialized safely.
    scenario = scenario_single_point
    rule = manual_gradient(scenario.grad)

    states = list(
        train_iter(
            scenario.data,
            scenario.params,
            scenario.predict,
            scenario.loss,
            rule,
            steps=3,
            lr=scenario.lr,
        )
    )

    assert len(states) == 3
    assert states[0].loss > states[-1].loss
