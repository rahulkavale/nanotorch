from nanotorch import manual_gradient, train


def test_constant_target_bias_dominates(scenario_constant_target):
    # Constant targets should push bias toward the constant and
    # dampen the slope toward zero.
    scenario = scenario_constant_target
    rule = manual_gradient(scenario.grad)

    history = train(
        scenario.data,
        scenario.params,
        scenario.predict,
        scenario.loss,
        rule,
        steps=scenario.steps,
        lr=scenario.lr,
    )

    assert history[0] > history[-1]
    # Bias should move upward toward 5.
    assert scenario.params["b"] > 1.0
    # Slope should move downward toward 0.
    assert scenario.params["w"] < 1.0
