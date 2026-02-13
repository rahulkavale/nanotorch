from nanotorch import manual_gradient, train


def test_noisy_linear_data_decreases_loss(scenario_noisy_linear):
    # Noisy data shouldn't allow perfect fit, but the system should still
    # reduce loss from the starting point.
    scenario = scenario_noisy_linear
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

    # End-to-end decrease is the right acceptance bar for noisy data.
    assert history[0] > history[-1]
    # Loss should remain above zero because the data is imperfect.
    assert history[-1] > 0.0
