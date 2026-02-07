Task breakup into size‑1 subtasks (all tasks with size >1 and not ∞). Each subtask is intended to be a clear, testable delivery.

Use case 1
Original task: Minimal numeric container with basic arithmetic (add, mul) and scalar values.
1. Define a numeric container that holds a scalar value.
2. Define a numeric container that holds a 1D list of values.
3. Implement elementwise add for two equal‑length 1D containers.
4. Implement elementwise multiply for two equal‑length 1D containers.
5. Expose length/size metadata for 1D containers.

Original task: Training loop primitive: repeated parameter update on loss.
1. Define a parameter container with a mutable value.
2. Define a simple error function that returns a scalar loss.
3. Define a step function that updates a parameter using a supplied gradient and learning rate.
4. Define a loop function that repeats updates for N steps.

Use case 2
Original task: Dynamic composition of computations (no pre‑compile step).
1. Execute computations immediately when called (no compile step).
2. Allow computations to be expressed as normal functions.
3. Support runtime branching (if/else) in computations.

Original task: Simple rerun mechanism with low setup overhead.
1. Provide a single entrypoint to run a user script or function.
2. Ensure a clean state reset between runs.

Original task: Lightweight parameter initialization utilities.
1. Provide helper to initialize parameters with zeros.
2. Provide helper to initialize parameters with random values.

Use case 3
Original task: CSV ingestion to in‑memory arrays with type coercion.
1. Read a CSV file into rows.
2. Convert numeric columns to numbers.
3. Split columns into features and targets by user selection.

Original task: Basic batching iterator.
1. Create an iterator that yields fixed‑size batches.
2. Define behavior for the last partial batch.

Original task: Error messages for malformed data.
1. Detect non‑numeric data in numeric columns and raise a clear error.
2. Detect inconsistent row lengths and raise a clear error.

Use case 4
Original task: Serialize model parameters and training state.
1. Define a serialization format for parameter values.
2. Save parameter values to a file.
3. Save training metadata (e.g., step count, learning rate).

Original task: Restore parameters and state into a new run.
1. Load parameter values from file.
2. Load training metadata from file.
3. Reconstruct model state from loaded data.

Original task: Minimal versioning of saved artifacts.
1. Add a version field to saved artifacts.
2. Warn on version mismatch when loading.

Use case 5
Original task: Device abstraction and data movement API.
1. Define device identifiers (e.g., cpu, gpu).
2. Add a device attribute to data containers.
3. Provide a helper to move data to a target device.
4. Define a data copy API between devices.
5. Enforce same‑device checks for operations.
6. Provide a clear error on device mismatch.
7. Allow configuration of a default device.
8. Document device behavior and limitations.

Original task: Fallback and device selection logic.
1. Detect available devices at runtime.
2. Default to GPU if available, else CPU.
3. Allow user override of device selection.

Use case 6
Original task: Public interface to define a new operation in high‑level code.
1. Define a registry for named operations.
2. Provide an API to register a Python function as an op.
3. Validate op signatures (inputs/outputs).
4. Provide a way to list available ops.
5. Add a simple test op to verify registration.

Original task: Hook it into training and parameter update.
1. Allow registered ops to be used in computations.
2. Allow ops to accept parameters as inputs.
3. Allow ops to be composed in a loss computation.

Original task: Minimal extension registry for reuse.
1. Support lookup of registered ops by name.
2. Prevent duplicate names or define an override policy.
3. Expose registry metadata for introspection.

Use case 7
Original task: Expose intermediate values during execution.
1. Provide a way to name intermediate values.
2. Provide a way to retrieve intermediates after a run.
3. Provide an optional tracing mode to capture intermediates.

Original task: Structured logging of gradients/values.
1. Define a log record structure (name, value, step).
2. Provide a log sink that records or prints entries.

Original task: Minimal hooks/callbacks for inspection.
1. Define a hook interface (callable contract).
2. Allow hook registration on a run or step.
3. Invoke hooks at defined points (before/after step).

Use case 8
Original task: Execution model that respects normal control flow.
1. Allow user code to use if/else in model functions.
2. Allow user code to use loops in model functions.
3. Execute operations immediately when called.
4. Ensure side effects are visible immediately.
5. Document supported control flow behaviors.

Original task: Safe handling of variable‑length data/loops.
1. Accept variable‑length input batches.
2. Allow loop counts to depend on data.
3. Validate variable lengths at runtime.
4. Define a policy for handling ragged data (pad/truncate).
5. Provide clear errors when policies are violated.

Original task: Validation of dynamic shapes.
1. Infer input shapes at runtime.
2. Validate shape compatibility for operations.
3. Emit clear errors on shape mismatch.

Use case 9
Original task: Experiment metadata capture (config + run ID).
1. Generate a unique run ID.
2. Capture config parameters into a record.
3. Save the record to disk.

Original task: Metric logging per run.
1. Provide an API to log metric values with a step index.
2. Persist metric history to a file.

Original task: Compare runs with simple summary.
1. Load multiple run records.
2. Compute summary stats (e.g., final/best loss).
3. Print or export a comparison table.

Use case 10
Original task: Extension interface for new ops.
1. Define an op interface contract (inputs/outputs).
2. Define a registration API for external ops.
3. Provide versioned metadata for ops.
4. Add tests for registering an external op.
5. Provide a stub CPU implementation path.
6. Validate input types for external ops.
7. Provide a clear error when an op is missing.
8. Document the extension API.

Original task: Dispatch mechanism to select implementation.
1. Define dispatch keys (e.g., device or dtype).
2. Register per‑key implementations for an op.
3. Select implementation based on input attributes.
4. Provide a default/fallback implementation.
5. Raise clear error when no implementation exists.

Use case 11
Original task: Model serialization format.
1. Define an in‑memory model representation.
2. Define a serialization schema for model structure.
3. Serialize parameters to a file.
4. Serialize metadata (e.g., version).
5. Deserialize into a model representation.

Original task: Lightweight runtime to load and execute.
1. Define a runtime entrypoint that loads a model.
2. Execute a forward computation with inputs.
3. Validate input types and shapes.
4. Return outputs in a standard format.
5. Provide simple, clear error reporting.

Original task: Minimal API for inference input/output.
1. Define a predict/run API signature.
2. Provide input conversion from basic Python types.

Use case 12
Original task: Batch‑friendly operator behavior.
1. Define a convention for the batch dimension.
2. Implement add/mul to respect batch dimension.
3. Ensure shape checks include batch dimension.
4. Add tests for batched operations.
5. Document batch behavior.

Original task: Input collation and batching utilities.
1. Convert a list of examples into a batch.
2. Handle varying sizes (pad or truncate).
3. Provide a batch size parameter.

Original task: Memory reuse between batches.
1. Identify allocations inside the batch loop.
2. Reuse buffers for fixed‑size batches.
3. Overwrite buffers safely on each batch.
4. Provide a toggle for buffer reuse.
5. Report memory usage during batching.

Use case 13
Original task: Stable, portable serialization schema.
1. Define schema versioning rules.
2. Specify model graph representation.
3. Specify operator list and attributes.
4. Specify parameter storage format.
5. Define metadata section.
6. Define compatibility rules for versions.
7. Provide a serialization writer.
8. Provide a serialization reader.

Original task: Versioning and compatibility checks.
1. Check schema version on load.
2. Warn on minor version mismatch.
3. Error on major version mismatch.
4. Provide a migration stub or hook.
5. Provide a compatibility table.

Original task: Export validation tests.
1. Write a round‑trip export/import test.
2. Compare predictions before/after export.
3. Validate required schema fields.

Use case 14
Original task: Predictable memory allocation strategy.
1. Track memory allocations for data containers.
2. Reuse buffers when possible.
3. Provide a max memory budget config.
4. Fail gracefully when budget is exceeded.
5. Report memory usage to the user.
6. Ensure deallocation after use.
7. Add tests for memory growth.
8. Document memory behavior.

Original task: Basic performance monitoring hooks.
1. Add timing around operations.
2. Collect per‑op timing stats.
3. Emit a simple timing report.

Original task: Latency budgeting and alerts.
1. Allow a latency target to be set.
2. Measure per‑inference latency.
3. Compare measured latency to the target.
4. Provide a summary of violations.
5. Provide an optional abort on violation.

Use case 15
Original task: Capture code/params/data fingerprint.
1. Hash model parameters.
2. Hash training data reference.
3. Capture a code version identifier.
4. Save fingerprints to the run record.
5. Provide lookup by fingerprint.

Original task: Restore a run exactly from metadata.
1. Load parameters by fingerprint.
2. Load data reference by fingerprint.
3. Load code version reference.
4. Validate environment compatibility.
5. Re‑run with loaded metadata.

Original task: Deterministic execution toggle.
1. Expose a determinism flag in config.
2. Seed all RNGs when flag is enabled.
3. Disable known nondeterministic ops.
4. Record determinism settings in metadata.
5. Warn if full determinism is not guaranteed.

Use case 16
Original task: Detect NaNs/infs during execution.
1. Add optional numeric checks after operations.
2. Raise an error on NaN/inf detection.
3. Include location/context in the error.

Original task: Optional strict mode to stop on error.
1. Add a config flag for strict numeric checks.
2. Enable checks only when the flag is true.

Original task: Diagnostics to identify source operation.
1. Track operation names in execution.
2. Attach op name to numeric errors.
3. Provide recent‑op history in errors.
4. Provide a minimal diagnostic report.
5. Add a test that verifies diagnostic output.

Use case 17
Original task: Metrics API with standard outputs.
1. Define a metric object with name and value.
2. Provide a function to log a metric.

Original task: Pluggable sinks (stdout, file).
1. Implement a stdout sink.
2. Implement a file sink.

Original task: Simple aggregation per epoch/run.
1. Group metrics by epoch or run.
2. Compute mean/min/max per group.
3. Emit an aggregated summary.

Use case 18
Original task: Central RNG control and seeding.
1. Define an RNG manager with a global seed.
2. Provide an API to set the seed.
3. Provide an API to get the current seed.

Original task: Capture and restore RNG state.
1. Serialize RNG state.
2. Restore RNG state from serialization.
3. Test that sequences repeat after restore.

Original task: Documented determinism caveats.
1. Write determinism caveats in documentation.
2. Link caveats from error messages or warnings.

Use case 19
Original task: Timing instrumentation around ops.
1. Measure start/end time per op.
2. Accumulate timing by op type.
3. Store timing in the run record.
4. Provide a toggle to enable/disable timing.
5. Add a test for timing data capture.

Original task: Summarized profiler output.
1. Sort ops by total time.
2. Print top‑N ops by time.
3. Export a summary to file.

Original task: Optional detailed trace export.
1. Record per‑op timeline events.
2. Store events in a trace format.
3. Export trace to a file.
4. Provide a trace viewer hint.
5. Add a test that verifies trace export.

Use case 20
Original task: Precision‑aware numeric types.
1. Define a dtype enum (e.g., fp32, fp16).
2. Attach dtype to data containers.
3. Implement dtype conversion.
4. Ensure basic ops respect dtype.
5. Add dtype promotion rules.
6. Add tests for dtype behavior.
7. Provide a default dtype config.
8. Document dtype support.

Original task: Loss scaling or stability helpers.
1. Define a loss‑scaling configuration.
2. Apply scaling during loss computation.
3. Unscale gradients before parameter update.
4. Detect overflow/underflow events.
5. Adjust scale dynamically (basic policy).
6. Add tests for scaling behavior.
7. Provide logging for scale changes.
8. Document stability behavior.
