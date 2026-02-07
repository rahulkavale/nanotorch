Use case 1: Fit a simple model to data and see it learn.

What this means (user intent)
- The user has a small dataset (even 5–100 points) and wants to fit a simple function (e.g., y = w*x + b).
- They want immediate confirmation that learning is happening: the loss should go down and predictions should improve.
- They want to do this without committing to a heavy framework or boilerplate-heavy setup.

What the current problem feels like (pain points)
- Too many prerequisites: before you can “fit a line,” you’re asked to learn about tensors, modules, optimizers, datasets, and training loops.
- Boilerplate overhead: you must write (or copy) data loaders, model classes, loss functions, optimizer setup, and a loop, even for a trivial task.
- Rigid structure: some tools require a static graph or formal model definition, which is overkill for quick experimentation.
- Debug friction: when loss doesn’t decrease, it’s hard to see why (no easy inspection of gradients, parameters, or intermediate values).
- Environment friction: you may need GPU configuration or specific versions before even running a toy example.
- Feedback latency: long setup times and lack of simple “print and inspect” loops slow understanding.

Ideal outcome from the user’s perspective
- A few lines of code can:
  - Define a simple model.
  - Run training for a small number of steps.
  - Show a loss curve (even just printing loss each step).
- The user can easily inspect parameters, predictions, and gradients to understand learning.
- No need to set up datasets, dataloaders, or device plumbing just to test a simple idea.

Why this matters for our design
- The smallest usable slice should optimize for “immediate learning feedback.”
- The first abstraction we build must remove boilerplate and make it obvious that learning is occurring.
- The API should minimize ceremony and maximize transparency.
