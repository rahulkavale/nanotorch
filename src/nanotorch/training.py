from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Tuple


Scalar = float
# We model parameters as a simple name -> value map to keep the first slice
# explicit and inspectable. This avoids hiding learning behind objects.
Params = Dict[str, Scalar]
DataPoint = Tuple[Scalar, Scalar]
PredictFn = Callable[[Scalar, Params], Scalar]
LossFn = Callable[[Scalar, Scalar], Scalar]
# Gradient function signature is deliberately explicit to surface the learning
# mechanics: x, y, y_hat, and current params in; parameter gradients out.
GradFn = Callable[[Scalar, Scalar, Scalar, Params], Dict[str, Scalar]]
RuleFn = Callable[[Scalar, Scalar, Scalar, Params], Dict[str, Scalar]]


def manual_gradient(grad_fn: GradFn) -> RuleFn:
    """Wrap a user-supplied gradient function as a learning rule."""

    def rule(x: Scalar, y: Scalar, y_hat: Scalar, params: Params) -> Dict[str, Scalar]:
        # We keep this as a direct pass-through so the user's math is the
        # single source of truth for "how learning happens" in this rule.
        return grad_fn(x, y, y_hat, params)

    return rule


def train(
    data: Iterable[DataPoint],
    params: Params,
    predict: PredictFn,
    loss: LossFn,
    rule: RuleFn,
    *,
    steps: int,
    lr: float,
) -> List[Scalar]:
    history: List[Scalar] = []
    # Materialize once so we can iterate multiple steps without re-consuming
    # generators; this keeps behavior deterministic for tests.
    data_list = list(data)
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if len(data_list) == 0:
        # If there's no data, we can't compute loss. Returning zeros keeps the
        # contract "history length == steps" without inventing a loss value.
        return [0.0 for _ in range(steps)]

    for _ in range(steps):
        total_loss = 0.0
        # We accumulate gradients across the dataset and average them so the
        # learning rate is stable w.r.t. dataset size (simple batch gradient).
        grads_sum: Dict[str, Scalar] = {k: 0.0 for k in params}

        for x, y in data_list:
            y_hat = predict(x, params)
            # Loss is per-sample; we sum and later average to get a single
            # comparable scalar per step.
            total_loss += loss(y_hat, y)
            grads = rule(x, y, y_hat, params)
            for name, value in grads.items():
                grads_sum[name] = grads_sum.get(name, 0.0) + value

        n = float(len(data_list))
        for name in grads_sum:
            grads_sum[name] /= n
            # Plain gradient descent update; learning rules can change this
            # without modifying the training loop.
            params[name] -= lr * grads_sum[name]

        # We store mean loss per step to make "learning progress" observable
        # without requiring external logging in Sprint 0.
        history.append(total_loss / n)

    return history
