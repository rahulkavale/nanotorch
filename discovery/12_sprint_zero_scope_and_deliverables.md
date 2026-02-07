Sprint 0 (decision): scope and deliverables

Sprint 0 goal (user‑visible)
- A user can provide a tiny dataset and a simple prediction rule, run training for a small number of steps, and see the error decrease.

Constraints for Sprint 0
- CPU only, in‑memory data only.
- Scalars or 1D lists only (no tensors/arrays yet).
- No CSV, batching, checkpointing, or GPU.
- No autograd or symbolic differentiation; learning can be implemented via a simple numeric update rule.

User story (plain language)
- “Given a small set of (input, target) pairs and a prediction rule with adjustable values, I can run training steps and see the error go down.”

Acceptance criteria (testable)
1. Training reduces error on a toy linear dataset.
2. Parameters change in the expected direction.
3. Training returns a loss history (or exposes loss per step).

Sprint 0 tasks (TDD‑first)
1. Write the first failing test that encodes the user story and acceptance criteria.
2. Define the minimal API required by the test (function signatures only).
3. Implement only the code needed to pass the test.
4. Add a second test that verifies a tiny edge case (e.g., single data point or zero steps).

Proposed minimal API for the first test
- A `train` function that takes:
  - data: list of (x, y)
  - params: a simple parameter container (e.g., w, b)
  - predict: function(x, params) -> y_hat
  - loss: function(y_hat, y) -> scalar error
  - steps: int
  - lr: float
- Returns: a list of loss values per step

Out of scope for Sprint 0
- CSV loading, batching, device abstraction, logging sinks, hooks, serialization, custom op registry.

Why this is the right first slice
- It delivers immediate user value (“see it learn”).
- It forces the smallest set of abstractions needed for learning without committing to a full framework design.
- It’s testable end‑to‑end in a single failing test.
