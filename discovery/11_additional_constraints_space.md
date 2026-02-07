Additional constraints to explore the design space (pre‑PyTorch)

The goal here is to surface “whole‑space” constraints before we commit to implementation. Each constraint includes its impact and what it would force us to design differently.

A. Runtime / Environment Constraints
1. No external dependencies (stdlib‑only).
- Impact: we can’t rely on numpy/numba; we must implement array math, CSV parsing, and RNG ourselves.
- Forces: minimal numeric container and manual loops; slower performance.

2. Must run in a single file (e.g., classroom or interview setting).
- Impact: architecture must be compact and avoid deep module graphs.
- Forces: fewer abstractions or heavily simplified ones.

3. Must run in a Jupyter notebook with instant feedback.
- Impact: fast iteration, easy introspection, minimal setup.
- Forces: immediate execution, strong logging/visualization hooks.

4. Offline / air‑gapped execution.
- Impact: no external model downloads or telemetry.
- Forces: local artifacts only, clear file‑based storage.

B. Language / API Constraints
5. Pure functional core (no mutation).
- Impact: parameters must be updated by returning new versions.
- Forces: explicit state passing; checkpointing is simpler but memory usage higher.

6. Minimal API surface (single “train” entrypoint).
- Impact: simpler usage but less composability.
- Forces: hide internal abstractions or expose them only optionally.

7. “No classes” rule (functional programming style).
- Impact: no object wrappers like Parameter or NumericValue.
- Forces: use tuples/dicts and explicit signatures; less type safety.

C. Data / Scale Constraints
8. Dataset larger than memory (streaming required).
- Impact: can’t materialize full arrays.
- Forces: DataSource must be iterator‑based; training loop must be streaming.

9. Small data only (<=1k samples).
- Impact: performance is secondary; clarity is primary.
- Forces: keep code very simple; avoid complex optimization.

10. Variable‑length sequences (ragged data).
- Impact: operations must tolerate variable sizes.
- Forces: define padding/truncation policy and shape rules early.

D. Numerical / Correctness Constraints
11. Strict determinism required.
- Impact: RNG control must be centralized and no hidden randomness.
- Forces: explicit seeds, deterministic ops; limits some performance tricks.

12. Strict numeric checks (NaNs/inf are fatal).
- Impact: operations must validate outputs.
- Forces: per‑op checks or configurable strict mode; performance tradeoff.

13. Mixed precision prohibited (fp32 only).
- Impact: simpler numeric rules, slower performance.
- Forces: no dtype system in early slices.

E. Performance Constraints
14. Fixed memory budget (e.g., 256 MB).
- Impact: must reuse buffers and avoid large intermediates.
- Forces: memory tracking and reuse policies.

15. Real‑time latency constraints (inference under 10ms).
- Impact: training not primary; inference runtime needs to be lean.
- Forces: separate inference path; reduced overhead.

16. No compilation or JIT step allowed.
- Impact: dynamic execution only.
- Forces: interpretive path; less optimization.

F. Extensibility Constraints
17. Users must add custom ops without touching core.
- Impact: a stable extension interface is required early.
- Forces: op registry and dispatch mechanisms.

18. Versioned model artifacts must load for years.
- Impact: serialization format must be stable and versioned.
- Forces: explicit schema design and compatibility rules.

G. Observability / UX Constraints
19. Users must “see learning” in <10 lines of code.
- Impact: high‑level helper and good defaults required.
- Forces: simple train function and auto‑logging.

20. Users must be able to introspect intermediates.
- Impact: hooks or trace mechanism needed early.
- Forces: capture points in the execution loop.

H. Deployment Constraints
21. Must run on CPU only.
- Impact: GPU abstraction can be deferred.
- Forces: focus on correctness and clarity.

22. Must run on GPU when available.
- Impact: device abstraction needed early.
- Forces: device‑aware containers and ops (bigger complexity).

How to use this list
- Choose a constraint pack (e.g., A1 + C9 + D11 + G19) and we’ll derive a minimal architecture and first slice for that reality.
- We can explore multiple packs to understand the full space before choosing one for implementation.
