from nanotorch import train, manual_gradient


def test_single_point_fit_reduces_loss():
    # Single data point is the smallest end-to-end learning scenario.
    # We choose it to force a full training loop without hiding behind data size.
    data = [(2.0, 10.0)]
    params = {"w": 0.0, "b": 0.0}

    def predict(x, p):
        # Linear rule is the simplest learnable mapping (one slope + one bias).
        return p["w"] * x + p["b"]

    def loss(y_hat, y):
        # Squared error makes the learning signal large when predictions are bad.
        return (y_hat - y) ** 2

    def grad(x, y, y_hat, p):
        # Gradient of squared error for linear model.
        err = y_hat - y
        return {"w": 2 * err * x, "b": 2 * err}

    # Use manual gradients first: this isolates the training loop from any
    # automatic differentiation and keeps the math transparent.
    rule = manual_gradient(grad)

    history = train(
        data,
        params,
        predict,
        loss,
        rule,
        steps=10,
        lr=0.1,
    )

    # We only require end-to-end decrease here; monotonic decrease is a stricter
    # requirement that we will add later as a separate test.
    assert history[0] > history[-1]
    assert params["w"] > 0.0
    assert params["b"] > 0.0
