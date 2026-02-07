Yes, we can do all of them—in increasing complexity—with a strict TDD ladder. Each test introduces exactly one new requirement, and we only add code needed to make that test pass. Here’s a proposed order that builds user‑visible value while keeping changes minimal.

Test ladder (integration-first, increasing complexity)
1) Single‑point fit (simplest learning)
- Scenario: one data point, model y = w*x + b, fixed initial params.
- Requirement introduced: training loop updates parameters so loss decreases end‑to‑end.
- New system needs: parameter container, learning rule interface, train loop, loss function.

2) Multi‑point linear fit (no bias)
- Scenario: (0,0), (1,2), (2,4) with model y = w*x.
- Requirement introduced: training can aggregate loss over multiple points.
- New system needs: data iteration, loss aggregation.

3) Add bias term
- Scenario: (0,1), (1,3), (2,5) with model y = w*x + b.
- Requirement introduced: multiple parameters updated together.
- New system needs: parameter factory for multi‑param init and update.

4) Constant target
- Scenario: (0,5), (1,5), (2,5).
- Requirement introduced: learning should move w toward 0 while b moves toward 5.
- New system needs: gradient or update rule correctness across parameters.

5) Noisy data
- Scenario: (0,1.1), (1,2.9), (2,5.2), (3,7.0).
- Requirement introduced: loss decreases but doesn’t need to reach zero; tolerances.
- New system needs: flexible loss‑decrease criteria.

Learning‑rule expansion (after the base loop works)
6) Manual‑gradient rule
- Requirement: user can supply explicit gradients; training uses them.

7) Finite‑difference rule
- Requirement: system can estimate gradients numerically for any model.

8) Rule selection (open/close)
- Requirement: training accepts a learning‑rule object without changing core code.

Observability/configuration
9) Loss‑decrease policy (configurable)
- Requirement: tests can choose monotonic vs end‑to‑end loss decrease.

10) Parameter initialization factory
- Requirement: deterministic init for tests; pluggable init for users.

Why this order works
- Each test is a new requirement, not a rewrite.
- Early tests require the smallest possible system surface.
- Later tests validate extensibility (open/closed) without changing core training logic.

If you approve this ladder, we’ll start with Test #1 (single‑point fit) and write it as a failing integration test.
