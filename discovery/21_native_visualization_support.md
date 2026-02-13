Native visualization support (design direction)

Goal
- Make “what’s happening under the hood” a first‑class feature, not a notebook‑only add‑on.
- Visualization should work in any environment (terminal, notebook, exported artifacts) and be driven by the same execution data as tests.

Core idea: Observable training via events
- The training loop should emit structured events (StepState) instead of hiding values.
- Visualization layers subscribe to these events and render them however they want.

Proposed minimal data model
- StepState:
  - step: int
  - loss: float
  - params: dict[str, float]
  - grads: dict[str, float]
  - y_hat (optional): float or list
  - y (optional): float or list
- This is small enough for Sprint 0 but already useful for visualization.

Execution API surface (native support)
1. train(...) stays simple (no forced visualization)
2. train_iter(...) yields StepState each step (step‑through debugger style)
3. train(..., observer=...) accepts a callback or list of callbacks

Visualization layer
- A Visualizer consumes StepState and renders:
  - loss curve
  - model line vs data
  - parameter values over time
- Implementations:
  - notebook (ipywidgets slider)
  - terminal (step‑through + ASCII chart)
  - script exporter (PNG/GIF/HTML)

Why this is “native”
- The training loop itself provides trace data; visualization is not a sidecar.
- Tests can assert on StepState just like visualization uses it.

Tradeoffs
- Extra overhead for tracing (time/memory)
- Must keep trace optional and lightweight
- Requires a stable StepState schema to avoid breaking tools

TDD plan
1) Failing test: train_iter yields N steps and exposes loss/params
2) Implement StepState + train_iter
3) Add observer support to train
4) Build one visualizer (notebook) that uses StepState

Outcome
- You get a step‑through, inspectable view of learning without needing an IDE debugger.
- Visualization becomes an extension of the core, not an afterthought.
