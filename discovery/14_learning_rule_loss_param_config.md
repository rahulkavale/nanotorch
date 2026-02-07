Decisions from your answers

1) Learning rule extensibility
- We’ll design `train` to accept a pluggable learning rule (strategy), so new update methods can be added without changing core training code.
- Sprint 0 will include at least one concrete learning rule; we’ll choose which ones to implement first (see open questions below).

2) Configurable “loss decreases” criteria
- Tests will use a configurable assertion (e.g., end-to-end decrease vs monotonic decrease).
- The training loop won’t hard-code this; it’s test-level policy to keep the core open.

3) Parameter factory
- We’ll create a parameter factory to control initial values deterministically for tests.
- This keeps tests stable and allows users to plug in different initialization strategies later.

Open choices to lock before writing Test #1
- Which learning rules do we implement in Sprint 0? (finite-difference, manual gradient via user-supplied grad function, simple heuristic update, etc.)
- Which integration scenarios should be the first two tests? (I suggest linear no-bias and linear with bias, but you can choose.)
- Which loss-decrease criterion for initial tests? (end-to-end decrease is simplest; monotonic is stricter.)
