Next logical step (in my view)

Add a finite‑difference learning rule (numeric gradients) and prove it with a new integration test.

Why this is the next step
- It removes the need for users to manually derive gradients.
- It keeps the system open/closed: new learning rules can plug in without changing `train`.
- It’s the smallest leap in capability after manual gradients, and it introduces the idea of gradient estimation without committing to full autodiff.

Concrete deliverable
- A new rule factory, e.g., `finite_difference(eps=...)`, that uses the user’s `predict` + `loss` to approximate gradients.
- A new integration test that uses the same scenario but does NOT provide a gradient function.
- Clear comments explaining numerical gradient tradeoffs (accuracy vs cost, sensitivity to eps).

TDD shape
1) Failing test: train on a simple scenario using finite‑difference rule; assert loss decreases end‑to‑end.
2) Minimal implementation: finite‑difference gradient estimation for each parameter.
3) Refactor: keep the rule pluggable and side‑effect free.

If you agree, I’ll draft the failing test first.
