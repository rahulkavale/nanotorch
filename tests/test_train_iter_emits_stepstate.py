from nanotorch import manual_gradient, train_iter


def test_train_iter_emits_stepstate(scenario_single_point):
    # train_iter should expose internal learning state at each step so
    # visualization and debugging can be built without hacking the loop.
    scenario = scenario_single_point
    rule = manual_gradient(scenario.grad)

    states = list(
        train_iter(
            scenario.data,
            scenario.params,
            scenario.predict,
            scenario.loss,
            rule,
            steps=5,
            lr=scenario.lr,
        )
    )

    assert len(states) == 5
    assert states[0].loss > states[-1].loss
    # We expect the state to surface parameters for visualization.
    assert "w" in states[-1].params
    assert "b" in states[-1].params
