Implementation plan from first principles (pre‑PyTorch)

Guiding principles
- Start from user jobs, not framework concepts.
- Prefer transparency over magic (easy to inspect values, states, and errors).
- No premature abstractions; each interface must be justified by a user slice.
- Everything is built via TDD, one observable behavior at a time.

End goal (system perspective)
- A small learning system that can take data, define a simple predictive rule, measure error, and improve the rule over repeated steps.
- Extensible enough to add new math operations, new data sources, and different update rules without rewriting core logic.

Architecture, derived from use cases (concepts, not framework names)
- Numeric container: holds numbers and supports basic math with clear rules.
- Parameters: numbers that can be updated by a learning rule.
- Loss computation: a function that maps predictions + targets to a single error value.
- Learning rule: updates parameters based on error feedback.
- Execution loop: repeats predict → loss → update for N steps.
- Data input: gets data from file or in‑memory lists.
- Logging/visibility: prints or records loss and key values.

Delivery slices (value‑first)
Slice 1: “Fit a line and see error decrease.”
- Minimal numeric container for scalars and 1D lists.
- Elementwise add/mul and sum of a list.
- A parameter type that can be updated by a learning rule.
- A simple loss function (e.g., mean squared error).
- A tiny training loop that updates parameters and prints loss.

Slice 2: “Try a model variant quickly.”
- Simple function composition for computations.
- A one‑function entrypoint to run training repeatedly with minimal setup.
- Basic random initialization helpers.

Slice 3: “Load data from CSV.”
- CSV ingestion and conversion.
- Batch iterator.
- Clear errors on malformed data.

Slice 4: “Inspect and debug learning.”
- Access intermediate values.
- Minimal hooks for logging.

Slice 5: “Resume training.”
- Save/load of parameters and training state.

Implementation order (first two slices)
1. Define the very first user‑visible behavior: loss decreases on a toy dataset.
2. Implement the smallest numeric container that can compute the loss.
3. Add a parameter update rule that changes values predictably.
4. Add a training loop that makes loss decrease for a trivial case.
5. Add logging of loss per step.
6. Allow quick changes to the model function without additional setup.

TDD discipline
- Each step starts with a failing test that describes a user‑visible behavior.
- Only implement what is needed to pass that test.
- Refactor only after a green test and only when the next test demands it.
