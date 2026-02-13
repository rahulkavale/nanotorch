from nanotorch import manual_gradient, train


def test_train_calls_observer_with_stepstate(scenario_single_point):
    # Observer callbacks are the simplest native visualization hook.
    # We assert that the training loop emits StepState per step.
    scenario = scenario_single_point
    rule = manual_gradient(scenario.grad)

    observed = []

    def observer(state):
        observed.append(state)

    train(
        scenario.data,
        scenario.params,
        scenario.predict,
        scenario.loss,
        rule,
        steps=4,
        lr=scenario.lr,
        observer=observer,
    )

    assert len(observed) == 4
    assert observed[0].loss > observed[-1].loss
    assert "w" in observed[-1].params
    assert "b" in observed[-1].params
