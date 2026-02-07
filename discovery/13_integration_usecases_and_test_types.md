Integration use cases for Sprint 0 (training on tiny datasets)

Goal
- Define 4–5 concrete training scenarios that act as integration and end‑to‑end tests.
- Each scenario is user‑visible: “I ran training and saw error decrease / parameters move.”

Proposed integration scenarios
1. Linear fit without bias
- Model: y_hat = w * x
- Data: (0,0), (1,2), (2,4)
- Expected: loss decreases; w increases toward 2.

2. Linear fit with bias
- Model: y_hat = w * x + b
- Data: (0,1), (1,3), (2,5)
- Expected: loss decreases; w increases toward 2; b increases toward 1.

3. Constant target (bias‑only learning)
- Model: y_hat = w * x + b
- Data: (0,5), (1,5), (2,5)
- Expected: loss decreases; b increases toward 5; w moves toward 0.

4. Single‑point overfit
- Model: y_hat = w * x + b
- Data: (2,10)
- Expected: loss decreases; prediction at x=2 moves closer to 10.

5. Noisy linear data (robustness)
- Model: y_hat = w * x + b
- Data: (0,1.1), (1,2.9), (2,5.2), (3,7.0)
- Expected: loss decreases but may not converge to zero.

Test layers (how they map to these scenarios)
- Unit tests: verify numeric operations and parameter update rules on a single step.
- Integration tests: run `train` for N steps on one of the scenarios above and assert loss decreases.
- End‑to‑end tests: run a user‑style script or function that defines data, model, loss, and trains, then outputs loss history.

Open questions to confirm before writing tests
- What learning rule are we using in Sprint 0 (manual gradient, finite‑difference, or simple heuristic update)?
- How strict should “loss decreases” be (e.g., strictly decreasing each step vs decrease from start to end)?
- Should we fix initial parameters to ensure deterministic outcomes?
