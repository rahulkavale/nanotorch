Yes. The next step is to support training **without** a user‑provided gradient function by adding a finite‑difference learning rule.

What changes conceptually
- Today: the user supplies `grad(...)` so the system only applies updates.
- Next: the system *estimates* gradients numerically using `predict` + `loss`, so the user only supplies those.

What stays the same
- Manual gradients remain supported (faster, more precise).
- The learning‑rule interface stays pluggable (open/closed).

Tradeoffs of finite‑difference
- Pros: works for any model/loss, no manual math.
- Cons: slower (multiple evaluations per parameter), sensitive to epsilon, noisy for large parameter counts.
