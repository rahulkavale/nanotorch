High‑level product vision (pre‑PyTorch)

Product vision (one paragraph)
- A small, transparent learning system that lets a user describe a predictive rule, train it on data, observe improvement, and extend it with new operations or data sources without rewriting the core. It prioritizes immediate feedback, clear errors, and incremental growth over maximum performance.

Core abstractions (why they exist and what they enable)
1. NumericValue
- What: A thin container for scalar or 1D numeric data with strict arithmetic rules.
- Why: Prevents ambiguous list behavior and centralizes validation.
- Enables: Use cases 1, 2, 7, 16.

2. Parameter
- What: A NumericValue that is explicitly marked as “learnable.”
- Why: Distinguishes what should change during training from fixed inputs/targets.
- Enables: Use cases 1, 4, 7, 15.

3. Operation (Op)
- What: A named, user‑defined function that maps inputs to outputs.
- Why: Provides a stable way to extend math without touching core.
- Enables: Use cases 2, 6, 10.

4. Loss
- What: A function that reduces predictions and targets to a single error value.
- Why: Defines “what improvement means” and is testable.
- Enables: Use cases 1, 7, 16.

5. LearningRule
- What: A function that updates Parameters from feedback (e.g., gradient or heuristic).
- Why: Separates “how to update” from “what to predict.”
- Enables: Use cases 1, 2, 4.

6. Trainer (Execution Loop)
- What: A minimal loop that runs predict → loss → update for N steps.
- Why: Guarantees repeatable learning behavior and observable progress.
- Enables: Use cases 1, 2, 4, 7.

7. DataSource
- What: A simple iterator over input/target pairs, optionally batched.
- Why: Separates data ingestion from learning logic.
- Enables: Use cases 1, 3, 12.

8. Logger
- What: A hookable channel for metrics (loss, parameters, debug values).
- Why: Makes “see it learn” testable and inspectable.
- Enables: Use cases 1, 7, 9, 17.

Lifecycle (how things flow)
1. Define data source
- User provides in‑memory data or a CSV path.
2. Define model/predict function
- User writes a pure Python function using Ops and Parameters.
3. Define loss
- User selects or defines a loss function.
4. Choose learning rule
- User picks a built‑in rule or provides one.
5. Train
- Trainer iterates for N steps and logs loss/metrics.
6. Inspect
- User checks logs, parameters, and predictions.
7. Save / resume (later slice)
- Persist Parameters and training state.

Hooks (where users can attach behavior)
- BeforeStep: inspect inputs/parameters before each update.
- AfterStep: record loss or parameter values.
- OnError: surface numeric or shape errors with context.
- OnBatch: inspect a batch during data iteration.

Why this structure helps the use cases
- Use case 1 (fit a simple model): Trainer + Loss + Parameter + Logger provide a direct path to “see it learn.”
- Use case 2 (iterate quickly): Ops + dynamic Python functions allow fast edits without recompilation.
- Use case 3 (CSV data): DataSource keeps ingestion separate from learning code.
- Use case 4 (resume): Parameter + Trainer state provide a natural checkpoint boundary.
- Use case 6/10 (custom ops): Op abstraction allows extension without touching core.
- Use case 7/16 (debug/numerics): Logger + hooks + NumericValue validation give visibility and safety.
- Use case 9/17 (tracking): Logger becomes the foundation for experiment summaries.
- Use case 12 (batching): DataSource and Trainer provide insertion points for batching behavior.

Non‑goals for early slices
- GPU support, static graphs, distributed training, or advanced optimizers. These are deferred until the core learning loop is proven.
