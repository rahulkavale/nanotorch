from nanotorch import manual_gradient, train


def test_linear_fit_with_bias(scenario_with_bias):
    # Adding a bias term introduces a second parameter to update.
    # This test forces the training loop to handle multi-parameter gradients.
    scenario = scenario_with_bias
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
    # Both parameters should move toward their expected values (w≈2, b≈1).
    assert scenario.params["w"] > 0.5
    assert scenario.params["b"] > 0.2
