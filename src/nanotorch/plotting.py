from __future__ import annotations

# Plotting lives in the library so it can be exposed as a console script
# ("nanotorch-plot") and so the logic is reusable from other contexts.

from pathlib import Path

import matplotlib.pyplot as plt

from nanotorch import manual_gradient, train
from nanotorch.scenarios import get_scenario, list_scenarios


def _x_range(xs: list[float]) -> list[float]:
    # We pad the x-range slightly so the line doesn't touch the plot edges,
    # which makes the slope and intercept easier to read at a glance.
    lo = min(xs)
    hi = max(xs)
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    pad = 0.2 * (hi - lo)
    return [lo - pad, hi + pad]


def _line(predict, params, x_min: float, x_max: float, steps: int = 50) -> tuple[list[float], list[float]]:
    xs = [x_min + i * (x_max - x_min) / (steps - 1) for i in range(steps)]
    ys = [predict(x, params) for x in xs]
    return xs, ys


def plot_scenario(name: str, out_dir: Path) -> Path:
    scenario = get_scenario(name)

    xs = [x for x, _ in scenario.data]
    ys = [y for _, y in scenario.data]

    x_min, x_max = _x_range(xs)

    # Preserve the initial params for the "before" line.
    initial_params = dict(scenario.params)

    rule = manual_gradient(scenario.grad)
    train(
        scenario.data,
        scenario.params,
        scenario.predict,
        scenario.loss,
        rule,
        steps=scenario.steps,
        lr=scenario.lr,
    )

    x_line, y_init = _line(scenario.predict, initial_params, x_min, x_max)
    _, y_final = _line(scenario.predict, scenario.params, x_min, x_max)

    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, color="black", label="data")
    plt.plot(x_line, y_init, linestyle="--", label="initial")
    plt.plot(x_line, y_final, linestyle="-", label="trained")
    plt.title(f"{scenario.test_name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scenario.test_name}_chart.png"
    plt.savefig(out_path)
    plt.close()

    return out_path


def main() -> None:
    out_dir = Path("artifacts/plots")
    for name in list_scenarios():
        path = plot_scenario(name, out_dir)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
