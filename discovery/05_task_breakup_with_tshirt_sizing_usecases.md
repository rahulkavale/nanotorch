Task breakup by use case (system perspective). Sizes are provisional (1,2,3,5,8,∞). ∞ indicates known unknowns or missing architectural clarity.

Use case 1: Fit a simple model to data and see it learn.
- Minimal numeric container with basic arithmetic (add, mul) and scalar values. [size: 3]
- Training loop primitive: repeated parameter update on loss. [size: 3]
- Loss visibility: print or record loss per step. [size: 1]

Use case 2: Try model variants quickly.
- Dynamic composition of computations (no pre-compile step). [size: 3]
- Simple rerun mechanism with low setup overhead. [size: 2]
- Lightweight parameter initialization utilities. [size: 2]

Use case 3: Load data from CSV and train without boilerplate.
- CSV ingestion to in-memory arrays with type coercion. [size: 3]
- Basic batching iterator. [size: 2]
- Error messages for malformed data. [size: 2]

Use case 4: Resume training from a checkpoint.
- Serialize model parameters and training state. [size: 3]
- Restore parameters and state into a new run. [size: 3]
- Minimal versioning of saved artifacts. [size: 2]

Use case 5: Train on GPU if available, otherwise CPU.
- Device abstraction and data movement API. [size: 8]
- Operator implementations for CPU and GPU. [size: ∞]
- Fallback and device selection logic. [size: 3]

Use case 6: Prototype a custom layer or loss.
- Public interface to define a new operation in high-level code. [size: 5]
- Hook it into training and parameter update. [size: 3]
- Minimal extension registry for reuse. [size: 3]

Use case 7: Inspect gradients and activations for debugging.
- Expose intermediate values during execution. [size: 3]
- Structured logging of gradients/values. [size: 2]
- Minimal hooks/callbacks for inspection. [size: 3]

Use case 8: Change model structure dynamically during training.
- Execution model that respects normal control flow. [size: 5]
- Safe handling of variable-length data/loops. [size: 5]
- Validation of dynamic shapes. [size: 3]

Use case 9: Run quick ablations and track results.
- Experiment metadata capture (config + run ID). [size: 3]
- Metric logging per run. [size: 2]
- Compare runs with simple summary. [size: 3]

Use case 10: Add a custom CPU/GPU op for a new idea.
- Extension interface for new ops. [size: 8]
- Dispatch mechanism to select implementation. [size: 5]
- ABI or compatibility layer for binary plugins. [size: ∞]

Use case 11: Run a prebuilt model for inference.
- Model serialization format. [size: 5]
- Lightweight runtime to load and execute. [size: 5]
- Minimal API for inference input/output. [size: 2]

Use case 12: Batch inference for throughput.
- Batch-friendly operator behavior. [size: 5]
- Input collation and batching utilities. [size: 3]
- Memory reuse between batches. [size: 5]

Use case 13: Export a model to a stable format.
- Stable, portable serialization schema. [size: 8]
- Versioning and compatibility checks. [size: 5]
- Export validation tests. [size: 3]

Use case 14: Run inference in a service with memory/latency limits.
- Predictable memory allocation strategy. [size: 8]
- Basic performance monitoring hooks. [size: 3]
- Latency budgeting and alerts. [size: 5]

Use case 15: Version and reproduce a model run.
- Capture code/params/data fingerprint. [size: 5]
- Restore a run exactly from metadata. [size: 5]
- Deterministic execution toggle. [size: 5]

Use case 16: Numerical correctness and stability checks.
- Detect NaNs/infs during execution. [size: 3]
- Optional strict mode to stop on error. [size: 2]
- Diagnostics to identify source operation. [size: 5]

Use case 17: Metrics and logging without plumbing.
- Metrics API with standard outputs. [size: 2]
- Pluggable sinks (stdout, file). [size: 2]
- Simple aggregation per epoch/run. [size: 3]

Use case 18: Determinism when needed.
- Central RNG control and seeding. [size: 3]
- Capture and restore RNG state. [size: 3]
- Documented determinism caveats. [size: 2]

Use case 19: Profile performance and find bottlenecks.
- Timing instrumentation around ops. [size: 5]
- Summarized profiler output. [size: 3]
- Optional detailed trace export. [size: 5]

Use case 20: Mixed precision or low precision.
- Precision-aware numeric types. [size: 8]
- Loss scaling or stability helpers. [size: 8]
- Operator validation for precision safety. [size: ∞]
