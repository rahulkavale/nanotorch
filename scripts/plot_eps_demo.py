from __future__ import annotations

# This visualization explains why "eps" matters in finite differences.
# We show a function, a base point x0, a perturbed point x0+eps, and the
# secant slope that approximates the derivative. Multiple eps values show
# the accuracy vs stability tradeoff.

from pathlib import Path

import matplotlib.pyplot as plt


def f(x: float) -> float:
    # A simple smooth function with visible curvature.
    return (x - 1.5) ** 2 + 1.0


def f_prime(x: float) -> float:
    # Analytic derivative for reference.
    return 2.0 * (x - 1.5)


def plot_eps_demo(out_path: Path) -> None:
    x0 = 1.0
    eps_values = [0.5, 0.1, 0.01]

    xs = [x / 100.0 for x in range(-50, 351)]  # [-0.5, 3.5]
    ys = [f(x) for x in xs]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1: Function and base point
    ax = axes[0]
    ax.plot(xs, ys, color="black")
    ax.scatter([x0], [f(x0)], color="red", zorder=3)
    ax.set_title("Step 1: Choose a point x0")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

    # Panel 2: Add x0+eps and secant line
    eps = eps_values[0]
    x1 = x0 + eps
    y0 = f(x0)
    y1 = f(x1)
    ax = axes[1]
    ax.plot(xs, ys, color="black")
    ax.scatter([x0, x1], [y0, y1], color="red", zorder=3)
    ax.plot([x0, x1], [y0, y1], color="blue", linestyle="--")
    ax.set_title("Step 2: Finite difference (eps)")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.text(
        0.02,
        0.02,
        f"eps = {eps}\nsecant slope = {(y1 - y0) / eps:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )

    # Panel 3: Compare eps values vs true slope
    ax = axes[2]
    ax.plot(xs, ys, color="black")
    ax.scatter([x0], [f(x0)], color="red", zorder=3)
    true_slope = f_prime(x0)
    ax.set_title("Step 3: eps tradeoff")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

    for eps in eps_values:
        x1 = x0 + eps
        y1 = f(x1)
        ax.plot([x0, x1], [f(x0), y1], linestyle="--", label=f"eps={eps}")

    ax.legend(fontsize=8, loc="upper left")
    ax.text(
        0.02,
        0.02,
        f"true slope = {true_slope:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    plot_eps_demo(Path("artifacts/plots/finite_difference_eps_step_by_step.png"))
