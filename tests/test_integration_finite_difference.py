from nanotorch import finite_difference, train


def test_finite_difference_rule_learns_without_gradients(scenario_single_point):
    # This test asserts we can learn without supplying analytic gradients.
    # It forces the system to estimate gradients numerically.
    scenario = scenario_single_point

    # eps trades off accuracy vs numeric noise; we keep it explicit to
    # signal that finite differences are a modeling choice, not magic.
    rule = finite_difference(eps=1e-3, predict=scenario.predict, loss=scenario.loss)

    history = train(
        scenario.data,
        scenario.params,
        scenario.predict,
        scenario.loss,
        rule,
        steps=12,
        lr=0.1,
    )

    assert history[0] > history[-1]
    assert scenario.params["w"] > 0.0
    assert scenario.params["b"] > 0.0
