from nanotorch import manual_gradient, train


def test_single_point_fit_reduces_loss(scenario_single_point):
    # Single data point is the smallest end-to-end learning scenario.
    # The scenario registry keeps tests and plots consistent.
    scenario = scenario_single_point
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

    # We only require end-to-end decrease here; monotonic decrease is a stricter
    # requirement that we will add later as a separate test.
    assert history[0] > history[-1]
    assert scenario.params["w"] > 0.0
    assert scenario.params["b"] > 0.0
