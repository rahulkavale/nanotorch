Exhaustive learning‑rule options (pre‑PyTorch view) and increasing‑complexity order

Scope
- “Learning rule” = how parameters get updated to reduce error.
- We separate: (A) how gradients/updates are obtained, (B) how they are applied.

A) Ways to obtain update signals (from simplest to most complex)
1. Closed‑form solution (no iterative learning)
- Examples: linear regression via normal equation.
- Complexity: low math, but narrow applicability.

2. Manual gradients (user‑supplied)
- User writes gradient formulas for the model + loss.
- Complexity: low system complexity, higher user burden.

3. Finite‑difference gradients (numeric)
- Forward difference, central difference, or secant style.
- Works for any differentiable function; slow and noisy.

4. Stochastic/approximate gradients
- SPSA (simultaneous perturbation), random directional derivatives.
- Reduces cost vs full finite‑difference for many params.

5. Automatic differentiation
- Reverse‑mode (backprop) for scalar loss with many params.
- Forward‑mode for fewer params or Jacobians.

6. Symbolic differentiation / algebraic simplification
- Build and simplify expressions; rarely practical for large models.

7. Implicit differentiation
- Used when parameters are defined implicitly (advanced, niche).

B) Ways to apply updates (optimization rules)
1. Vanilla gradient descent
- p := p - lr * grad

2. Batch vs minibatch vs online (stochastic)
- Same rule, different data subsets per step.

3. Momentum
- Velocity term to smooth updates.

4. Adaptive learning rates
- Adagrad, RMSProp, Adadelta.

5. Adam / AdamW
- Momentum + adaptive scaling.

6. Second‑order methods
- Newton, quasi‑Newton (BFGS/L‑BFGS), conjugate gradient.

7. Derivative‑free optimizers
- Nelder–Mead, CMA‑ES, genetic algorithms, simulated annealing.

C) Non‑gradient learning rules (special cases)
- Perceptron update (classification boundary)
- Hebbian learning (correlation‑based)
- EM‑style updates (latent variable models)

Recommended order for nanotorch (increasing complexity)
0. Closed‑form (optional demo for linear regression)
1. Manual gradients + vanilla gradient descent (already in place)
2. Finite‑difference gradients + vanilla gradient descent
3. Rule selection interface (strategy object / callable, open‑closed)
4. Minibatch support (same update rule, different data subsets)
5. Momentum
6. Adam (or RMSProp)
7. Reverse‑mode autodiff (true backprop)
8. Second‑order or derivative‑free (only if needed)

Why this order
- Each step adds one new system responsibility:
  - Numerical gradients (finite‑diff) adds function evaluation loops.
  - Rule selection adds a stable extension boundary.
  - Momentum/Adam add stateful updates.
  - Autodiff introduces a computation graph and gradient propagation.

If you want, we can lock a smaller subset as “Sprint 1 learning‑rules roadmap.”
