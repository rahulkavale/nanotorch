from nanotorch import train, manual_gradient


def test_multi_point_linear_fit_no_bias():
    # Multi-point case forces the training loop to aggregate loss/gradients
    # across samples, not just a single point.
    data = [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)]
    params = {"w": 0.0}

    def predict(x, p):
        # No bias term in this variant: the slope alone should learn.
        return p["w"] * x

    def loss(y_hat, y):
        return (y_hat - y) ** 2

    def grad(x, y, y_hat, p):
        # d/dw (y_hat - y)^2 where y_hat = w*x
        err = y_hat - y
        return {"w": 2 * err * x}

    rule = manual_gradient(grad)

    history = train(
        data,
        params,
        predict,
        loss,
        rule,
        steps=25,
        lr=0.05,
    )

    # End-to-end decrease confirms learning on multiple samples.
    assert history[0] > history[-1]
    # Slope should move toward the true value 2.
    assert params["w"] > 0.5
