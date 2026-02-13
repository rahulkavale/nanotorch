from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

# Scenario registry exists to keep tests and visualizations in sync.
# If we change a dataset or model, we change it once here and both
# the tests and the plots stay truthful.

Scalar = float
Params = Dict[str, Scalar]
DataPoint = Tuple[Scalar, Scalar]
PredictFn = Callable[[Scalar, Params], Scalar]
LossFn = Callable[[Scalar, Scalar], Scalar]
GradFn = Callable[[Scalar, Scalar, Scalar, Params], Dict[str, Scalar]]


@dataclass(frozen=True)
class Scenario:
    name: str
    test_name: str
    description: str
    data: List[DataPoint]
    params: Params
    predict: PredictFn
    loss: LossFn
    grad: GradFn
    steps: int
    lr: float


def _single_point() -> Scenario:
    data = [(2.0, 10.0)]
    params = {"w": 0.0, "b": 0.0}

    def predict(x: Scalar, p: Params) -> Scalar:
        return p["w"] * x + p["b"]

    def loss(y_hat: Scalar, y: Scalar) -> Scalar:
        return (y_hat - y) ** 2

    def grad(x: Scalar, y: Scalar, y_hat: Scalar, p: Params) -> Dict[str, Scalar]:
        err = y_hat - y
        return {"w": 2 * err * x, "b": 2 * err}

    return Scenario(
        name="single_point",
        test_name="test_single_point_fit_reduces_loss",
        description="Single point fit: y = w*x + b should move toward (2, 10).",
        data=data,
        params=params,
        predict=predict,
        loss=loss,
        grad=grad,
        steps=10,
        lr=0.1,
    )


def _multi_point_no_bias() -> Scenario:
    data = [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)]
    params = {"w": 0.0}

    def predict(x: Scalar, p: Params) -> Scalar:
        return p["w"] * x

    def loss(y_hat: Scalar, y: Scalar) -> Scalar:
        return (y_hat - y) ** 2

    def grad(x: Scalar, y: Scalar, y_hat: Scalar, p: Params) -> Dict[str, Scalar]:
        err = y_hat - y
        return {"w": 2 * err * x}

    return Scenario(
        name="multi_point_no_bias",
        test_name="test_multi_point_linear_fit_no_bias",
        description="Multi-point fit without bias: learn slope ~2.",
        data=data,
        params=params,
        predict=predict,
        loss=loss,
        grad=grad,
        steps=25,
        lr=0.05,
    )


def _with_bias() -> Scenario:
    data = [(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)]
    params = {"w": 0.0, "b": 0.0}

    def predict(x: Scalar, p: Params) -> Scalar:
        return p["w"] * x + p["b"]

    def loss(y_hat: Scalar, y: Scalar) -> Scalar:
        return (y_hat - y) ** 2

    def grad(x: Scalar, y: Scalar, y_hat: Scalar, p: Params) -> Dict[str, Scalar]:
        err = y_hat - y
        return {"w": 2 * err * x, "b": 2 * err}

    return Scenario(
        name="with_bias",
        test_name="test_linear_fit_with_bias",
        description="Linear fit with bias: learn slope ~2 and intercept ~1.",
        data=data,
        params=params,
        predict=predict,
        loss=loss,
        grad=grad,
        steps=40,
        lr=0.05,
    )


def _constant_target() -> Scenario:
    data = [(0.0, 5.0), (1.0, 5.0), (2.0, 5.0)]
    params = {"w": 1.0, "b": 0.0}

    def predict(x: Scalar, p: Params) -> Scalar:
        return p["w"] * x + p["b"]

    def loss(y_hat: Scalar, y: Scalar) -> Scalar:
        return (y_hat - y) ** 2

    def grad(x: Scalar, y: Scalar, y_hat: Scalar, p: Params) -> Dict[str, Scalar]:
        err = y_hat - y
        return {"w": 2 * err * x, "b": 2 * err}

    return Scenario(
        name="constant_target",
        test_name="test_constant_target_bias_dominates",
        description="Constant target pushes bias toward constant and slope toward 0.",
        data=data,
        params=params,
        predict=predict,
        loss=loss,
        grad=grad,
        steps=40,
        lr=0.05,
    )


def _noisy_linear() -> Scenario:
    # Slightly noisy points around y = 2x + 1 to model realistic data.
    data = [(0.0, 1.1), (1.0, 2.9), (2.0, 5.2), (3.0, 7.0)]
    params = {"w": 0.0, "b": 0.0}

    def predict(x: Scalar, p: Params) -> Scalar:
        return p["w"] * x + p["b"]

    def loss(y_hat: Scalar, y: Scalar) -> Scalar:
        return (y_hat - y) ** 2

    def grad(x: Scalar, y: Scalar, y_hat: Scalar, p: Params) -> Dict[str, Scalar]:
        err = y_hat - y
        return {"w": 2 * err * x, "b": 2 * err}

    return Scenario(
        name="noisy_linear",
        test_name="test_noisy_linear_data_decreases_loss",
        description="Noisy linear data: loss should decrease but not reach zero.",
        data=data,
        params=params,
        predict=predict,
        loss=loss,
        grad=grad,
        steps=60,
        lr=0.03,
    )


_SCENARIOS = {
    "single_point": _single_point,
    "multi_point_no_bias": _multi_point_no_bias,
    "with_bias": _with_bias,
    "constant_target": _constant_target,
    "noisy_linear": _noisy_linear,
}


def list_scenarios() -> List[str]:
    return sorted(_SCENARIOS.keys())


def get_scenario(name: str) -> Scenario:
    try:
        return _SCENARIOS[name]()
    except KeyError as exc:
        available = ", ".join(list_scenarios())
        raise ValueError(f"Unknown scenario '{name}'. Available: {available}") from exc
