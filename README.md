# nanotorch

**nanotorch** is a from‑scratch, TDD‑driven learning system. The goal is to understand *why* deep learning frameworks look the way they do by building the core pieces step‑by‑step from first principles, without borrowing abstractions from existing frameworks.

This is a learning‑first project:
- We start from real user jobs (e.g., “fit a simple model and see it learn”).
- Each new capability is introduced by a failing test.
- Every change is small, observable, and justified by a user‑visible requirement.

## Project Vision
We want a small, transparent system that lets a user:
1. Define a simple predictive rule.
2. Train it on data.
3. Observe improvement.
4. Extend it with new operations or data sources without rewriting core logic.

The guiding principles are:
- **Clarity over cleverness**: prefer explicit mechanics over hidden magic.
- **Test‑first correctness**: each behavior is locked in with tests.
- **Open/closed design**: new learning rules or operations should plug in without rewriting core code.

## How We Work
- Discovery notes live in `discovery/` (user stories, constraints, task breakdowns).
- Process and coordination notes live in `instructions/`.
- We build in small slices, each with unit, integration, and end‑to‑end tests.

## Current Status
We are in Sprint 0 and have the first integration tests for training on tiny datasets.

## Quickstart (uv)
```bash
# Run tests
uv run pytest -q
```

## Generate Scenario Plots
We generate plots from the same scenario registry used by tests, so the visuals
always match the data and model definitions under test.

```bash
# Generate plots into artifacts/plots/
uv run python scripts/plot_scenarios.py
```

You can also run the console script:
```bash
uv run nanotorch-plot
```

## Repo Structure
- `src/nanotorch/` — core library code
- `tests/` — unit/integration/e2e tests
- `discovery/` — design discovery and planning notes
- `instructions/` — working agreements and process notes

## Contributing (for collaborators)
- We do **strict TDD**. No production code is written before a failing test exists.
- Keep code **comment‑rich** with “why/how” insights, not obvious statements.
- Prefer small, reversible changes over large refactors.

## License
TBD
